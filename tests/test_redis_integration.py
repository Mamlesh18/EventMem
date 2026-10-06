"""Redis transport against a real Redis server.

Skipped automatically when no server is reachable, so the default suite still
runs with no services. Point it elsewhere with ``EVENTMEM_TEST_REDIS``.

These matter more than they look. Everything else in the suite exercises the
in-memory transport, where delivery is a function call and nothing is
serialised. The claims that only Redis can falsify are exactly the ones worth
testing: that the log survives a restart, that a consumer group gives
at-least-once delivery, that an unacked entry is recoverable, and that a timing
stamp crossing a process boundary is detected as no longer precise.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import uuid

import pytest

from eventmem import Agent, EventMemRuntime, EventType, Subscription
from eventmem.transport.redis import (
    RedisEventBus,
    RedisEventStore,
    RedisInbox,
    _aclose,
    connect,
)

REDIS_URL = os.getenv("EVENTMEM_TEST_REDIS", "redis://localhost:6379")


def _reachable(url: str) -> bool:
    try:
        import redis as sync_redis

        client = sync_redis.from_url(url, socket_connect_timeout=2)
        client.ping()
        client.close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _reachable(REDIS_URL),
    reason=f"no Redis at {REDIS_URL}; set EVENTMEM_TEST_REDIS to point elsewhere",
)


@pytest.fixture
async def redis_env():
    """A client plus a unique key namespace, cleaned up afterwards.

    Namespaced and explicitly deleted rather than flushed: this may be running
    against a Redis holding someone else's data.
    """
    client = await connect(REDIS_URL)
    namespace = f"emtest{uuid.uuid4().hex[:8]}"
    try:
        yield client, namespace
    finally:
        cursor = 0
        while True:
            cursor, keys = await client.scan(cursor, match=f"*{namespace}*", count=500)
            if keys:
                await client.delete(*keys)
            if cursor == 0:
                break
        await _aclose(client)


async def _drain(agents, timeout=6.0):
    """Wait until every agent's consumer group has nothing pending."""
    for _ in range(int(timeout / 0.02)):
        await asyncio.sleep(0.02)
        pending = 0
        for agent in agents:
            # An inbox whose group is not created yet has nothing pending.
            with contextlib.suppress(Exception):
                pending += await agent.inbox.pending()
        if pending == 0:
            await asyncio.sleep(0.05)
            return True
    return False


def _runtime(client, namespace, **kw):
    return EventMemRuntime(
        bus=RedisEventBus(client),
        event_store=RedisEventStore(client, stream=f"{namespace}:log"),
        **kw,
    )


# ------------------------------------------------------------------ delivery

async def test_event_is_delivered_over_redis_streams(redis_env):
    client, namespace = redis_env
    runtime = _runtime(client, namespace)

    writer = Agent(f"{namespace}-writer", runtime)
    reader = Agent(f"{namespace}-reader", runtime)
    await runtime.subscribe(Subscription(reader.id, attribute_filters={"topic": "x"}))
    await reader.start()

    await writer.remember("crossed a real transport", memory_id="m1",
                          attributes={"topic": "x"})
    assert await _drain([reader])

    assert [e.content for e in reader.received] == ["crossed a real transport"]
    await reader.stop()
    await runtime.close()


async def test_selective_routing_holds_over_redis(redis_env):
    client, namespace = redis_env
    runtime = _runtime(client, namespace)

    writer = Agent(f"{namespace}-w", runtime)
    wanted = Agent(f"{namespace}-wanted", runtime)
    unwanted = Agent(f"{namespace}-unwanted", runtime)

    await runtime.subscribe(Subscription(wanted.id, attribute_filters={"topic": "a"}))
    await runtime.subscribe(Subscription(unwanted.id, attribute_filters={"topic": "b"}))
    await wanted.start()
    await unwanted.start()

    await writer.remember("topic a only", memory_id="m1", attributes={"topic": "a"})
    assert await _drain([wanted, unwanted])

    assert len(wanted.received) == 1
    assert unwanted.received == []
    await wanted.stop()
    await unwanted.stop()
    await runtime.close()


async def test_attributes_and_provenance_survive_serialisation(redis_env):
    client, namespace = redis_env
    runtime = _runtime(client, namespace)

    writer = Agent(f"{namespace}-w", runtime)
    reader = Agent(f"{namespace}-r", runtime)
    await runtime.subscribe(Subscription(reader.id))
    await reader.start()

    await writer.remember(
        "payload with structure", memory_id="m1",
        attributes={"topic": "x", "count": 7, "nested": {"a": 1}},
        confidence=0.75, reason="because",
    )
    assert await _drain([reader])

    got = reader.received[0]
    assert got.attributes["count"] == 7
    assert got.attributes["nested"] == {"a": 1}
    assert got.provenance.confidence == pytest.approx(0.75)
    assert got.provenance.reason == "because"
    assert got.committed_version == 1
    await reader.stop()
    await runtime.close()


async def test_embedding_is_not_shipped_to_inboxes(redis_env):
    """Routing is done by delivery time, so the vector is dead weight on the
    wire. The durable log keeps it; the inbox copy should not."""
    client, namespace = redis_env
    runtime = _runtime(client, namespace)

    writer = Agent(f"{namespace}-w", runtime)
    reader = Agent(f"{namespace}-r", runtime)
    await runtime.subscribe(Subscription(reader.id))
    await reader.start()

    await writer.remember("vector stays home", memory_id="m1")
    assert await _drain([reader])

    assert reader.received[0].embedding is None
    logged = await runtime.event_store.replay()
    assert logged[0].embedding is not None
    await reader.stop()
    await runtime.close()


# -------------------------------------------------------------------- the log

async def test_the_log_is_durable_and_replays_into_a_fresh_runtime(redis_env):
    """The claim that makes "the log is the source of truth" real: a brand new
    runtime, sharing nothing but Redis, reconstructs the same records."""
    client, namespace = redis_env

    first = _runtime(client, namespace)
    writer = Agent(f"{namespace}-w", first)
    await writer.remember("one", memory_id="m1")
    await writer.remember("two", memory_id="m2")
    await writer.remember("one revised", memory_id="m1",
                          event_type=EventType.MEMORY_UPDATED)
    expected = {r.memory_id: (r.content, r.version) for r in first.records.all()}
    await first.close()

    # A separate runtime object with its own empty record store.
    second = _runtime(await connect(REDIS_URL), namespace)
    replayed = await second.rebuild_projection()
    rebuilt = {r.memory_id: (r.content, r.version) for r in second.records.all()}

    assert replayed == 3
    assert rebuilt == expected
    await second.close()


async def test_log_length_matches_what_was_published(redis_env):
    client, namespace = redis_env
    runtime = _runtime(client, namespace)
    writer = Agent(f"{namespace}-w", runtime)

    for i in range(5):
        await writer.remember(f"fact {i}", memory_id=f"m{i}")

    assert await runtime.event_store.length() == 5
    await runtime.close()


# ------------------------------------------------------- delivery semantics

async def test_consumer_group_does_not_redeliver_an_acked_entry(redis_env):
    client, namespace = redis_env
    inbox = RedisInbox(client, f"{namespace}-solo")
    await inbox.setup()
    bus = RedisEventBus(client)

    runtime = _runtime(client, namespace)
    writer = Agent(f"{namespace}-w", runtime)
    event = await writer.remember("once only", memory_id="m1")
    await bus.deliver(f"{namespace}-solo", event)

    first = await inbox.get(timeout=2.0)
    assert first is not None
    await inbox.ack(first)

    assert await inbox.get(timeout=0.5) is None
    await runtime.close()


async def test_an_unacked_entry_stays_pending_and_is_reclaimable(redis_env):
    """At-least-once, demonstrated: read without acking, and the entry is still
    recoverable. Acking at read time would have lost it."""
    client, namespace = redis_env
    agent_id = f"{namespace}-crasher"
    inbox = RedisInbox(client, agent_id)
    await inbox.setup()
    bus = RedisEventBus(client)

    runtime = _runtime(client, namespace)
    writer = Agent(f"{namespace}-w", runtime)
    event = await writer.remember("must not be lost", memory_id="m1")
    await bus.deliver(agent_id, event)

    taken = await inbox.get(timeout=2.0)
    assert taken is not None
    assert await inbox.pending() == 1          # read, deliberately not acked

    # A replacement consumer claims what the dead one left behind.
    reclaimed = await inbox.claim_stale(min_idle_ms=0, count=10)
    assert [e.event_id for e in reclaimed] == [event.event_id]
    assert reclaimed[0].content == "must not be lost"

    await inbox.ack(reclaimed[0])
    assert await inbox.pending() == 0
    await runtime.close()


async def test_a_failing_handler_leaves_the_entry_recoverable(redis_env):
    """The agent nacks on failure, so the work is not silently dropped."""
    client, namespace = redis_env
    runtime = _runtime(client, namespace)

    class Exploding(Agent):
        async def handle(self, event):
            raise RuntimeError("handler blew up")

    writer = Agent(f"{namespace}-w", runtime)
    victim = Exploding(f"{namespace}-victim", runtime)
    await runtime.subscribe(Subscription(victim.id))
    await victim.start()

    await writer.remember("will fail to handle", memory_id="m1")
    for _ in range(100):
        await asyncio.sleep(0.02)
        if victim.errors:
            break

    assert len(victim.errors) == 1
    # Never acknowledged, so Redis still holds it for a future consumer.
    assert await victim.inbox.pending() == 1
    await victim.stop()
    await runtime.close()


async def test_setup_is_idempotent(redis_env):
    client, namespace = redis_env
    inbox = RedisInbox(client, f"{namespace}-idem")
    await inbox.setup()
    await inbox.setup()          # BUSYGROUP must be tolerated
    assert await inbox.pending() == 0


# ------------------------------------------------------------------- timing

async def test_cross_process_stamps_are_flagged_as_imprecise(redis_env):
    """A perf_counter value from another process is meaningless here, and the
    runtime has to say so rather than report a nonsense latency."""
    client, namespace = redis_env
    runtime = _runtime(client, namespace)

    writer = Agent(f"{namespace}-w", runtime)
    reader = Agent(f"{namespace}-r", runtime)
    await runtime.subscribe(Subscription(reader.id))
    await reader.start()

    await writer.remember("same process", memory_id="m1")
    assert await _drain([reader])

    # Everything here ran in one process, so timing is precise...
    assert runtime.metrics.timing_is_precise()

    # ...and a stamp claiming a different origin must flip that.
    from eventmem.core.clock import Stamp

    delivery = runtime.metrics.deliveries[0]
    delivery.published_at = Stamp(
        wall=delivery.published_at.wall, perf=0.0, origin="another-process")
    assert not runtime.metrics.timing_is_precise()

    await reader.stop()
    await runtime.close()


async def test_latency_is_measured_and_positive(redis_env):
    client, namespace = redis_env
    runtime = _runtime(client, namespace)

    writer = Agent(f"{namespace}-w", runtime)
    reader = Agent(f"{namespace}-r", runtime)
    await runtime.subscribe(Subscription(reader.id))
    await reader.start()

    for i in range(5):
        await writer.remember(f"fact {i}", memory_id=f"m{i}")
    assert await _drain([reader])

    latency = runtime.metrics.mean_propagation_ms()
    assert latency > 0
    assert latency < 1000, f"a local Redis round trip should not take {latency}ms"
    await reader.stop()
    await runtime.close()


# ------------------------------------------------------------------ conflict

async def test_conflicts_resolve_over_redis(redis_env):
    client, namespace = redis_env
    runtime = _runtime(client, namespace)

    low = Agent(f"{namespace}-low", runtime)
    high = Agent(f"{namespace}-high", runtime)

    await low.remember("original", memory_id="m1", confidence=0.5)
    await low.remember("low confidence revision", memory_id="m1",
                       event_type=EventType.MEMORY_UPDATED,
                       base_version=1, confidence=0.6)
    await high.remember("high confidence revision", memory_id="m1",
                        event_type=EventType.MEMORY_UPDATED,
                        base_version=1, confidence=0.95)

    assert runtime.metrics.conflicts_detected == 1
    assert runtime.get("m1").content == "high confidence revision"
    # The conflict notice is in the durable log too.
    logged = await runtime.event_store.replay()
    assert any(e.event_type == EventType.CONFLICT_DETECTED for e in logged)
    await runtime.close()

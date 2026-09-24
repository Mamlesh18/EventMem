"""Runtime pipeline: routing, conflicts, cascade bounding, replay."""

from __future__ import annotations

import pytest

from eventmem import (
    Agent,
    EventMemRuntime,
    EventType,
    LastWriterWins,
    MemoryEvent,
    Provenance,
    SetUnionMerge,
    Subscription,
)


async def test_publish_projects_into_the_record_store():
    runtime = EventMemRuntime()
    agent = Agent("writer", runtime)

    event = await agent.remember("the sky is blue", memory_id="m1")

    record = runtime.get("m1")
    assert record is not None
    assert record.content == "the sky is blue"
    assert record.version == 1
    assert event.committed_version == 1


async def test_selective_routing_reaches_only_matching_agents():
    runtime = EventMemRuntime()
    writer = Agent("writer", runtime)
    backend = Agent("backend", runtime)
    frontend = Agent("frontend", runtime)

    await runtime.subscribe(
        Subscription("backend", attribute_filters={"component": "backend"})
    )
    await runtime.subscribe(
        Subscription("frontend", attribute_filters={"component": "frontend"})
    )
    await backend.start()
    await frontend.start()

    await writer.remember("db migration", attributes={"component": "backend"})
    await _settle(backend, frontend)

    assert [e.content for e in backend.received] == ["db migration"]
    assert frontend.received == []

    await backend.stop()
    await frontend.stop()


async def test_an_agent_never_receives_its_own_write():
    runtime = EventMemRuntime()
    agent = Agent("solo", runtime)
    await runtime.subscribe(Subscription("solo"))
    await agent.start()

    await agent.remember("a note")
    await _settle(agent)

    assert agent.received == []
    await agent.stop()


async def test_collision_is_detected_and_resolved_by_confidence():
    runtime = EventMemRuntime()
    low = Agent("low", runtime)
    high = Agent("high", runtime)

    await low.remember("original", memory_id="m1", confidence=0.5)
    # Both writers declare the version they read.
    await low.remember(
        "low-confidence revision", memory_id="m1",
        event_type=EventType.MEMORY_UPDATED, base_version=1, confidence=0.6,
    )
    await high.remember(
        "high-confidence revision", memory_id="m1",
        event_type=EventType.MEMORY_UPDATED, base_version=1, confidence=0.95,
    )

    record = runtime.get("m1")
    assert record.content == "high-confidence revision"
    assert runtime.metrics.conflicts_detected == 1
    assert runtime.metrics.conflicts_resolved == 1


async def test_no_base_version_means_no_collision_detection():
    """Opting out of optimistic concurrency must overwrite silently.

    This is a real footgun, so it is pinned by a test: a caller that forgets
    base_version gets last-write-wins, not a conflict.
    """
    runtime = EventMemRuntime()
    agent = Agent("a", runtime)

    await agent.remember("first", memory_id="m1")
    await agent.remember(
        "second", memory_id="m1", event_type=EventType.MEMORY_UPDATED,
        base_version=None, confidence=0.1,
    )

    assert runtime.metrics.conflicts_detected == 0
    assert runtime.get("m1").content == "second"


async def test_conflict_emits_a_notice_event():
    runtime = EventMemRuntime()
    writer = Agent("writer", runtime)
    watcher = Agent("watcher", runtime)

    await runtime.subscribe(
        Subscription("watcher", event_types={EventType.CONFLICT_DETECTED})
    )
    await watcher.start()

    await writer.remember("v1", memory_id="m1")
    await writer.remember(
        "v2", memory_id="m1", event_type=EventType.MEMORY_UPDATED, base_version=1,
    )
    await writer.remember(
        "v3", memory_id="m1", event_type=EventType.MEMORY_UPDATED, base_version=1,
    )
    await _settle(watcher)

    assert len(watcher.received) == 1
    assert watcher.received[0].event_type == EventType.CONFLICT_DETECTED
    await watcher.stop()


async def test_conflict_notice_does_not_overwrite_the_record():
    """A notice describes state; projecting it would destroy the state."""
    runtime = EventMemRuntime()
    agent = Agent("a", runtime)

    await agent.remember("real content", memory_id="m1")
    await agent.remember(
        "revision", memory_id="m1", event_type=EventType.MEMORY_UPDATED,
        base_version=1,
    )
    await agent.remember(
        "colliding revision", memory_id="m1",
        event_type=EventType.MEMORY_UPDATED, base_version=1,
    )

    assert "conflict on" not in runtime.get("m1").content


async def test_set_union_merge_keeps_both_writes():
    runtime = EventMemRuntime(conflict_policy=SetUnionMerge())
    agent = Agent("a", runtime)

    await agent.remember("fact one", memory_id="m1")
    await agent.remember(
        "fact two", memory_id="m1", event_type=EventType.MEMORY_UPDATED,
        base_version=1,
    )
    await agent.remember(
        "fact three", memory_id="m1", event_type=EventType.MEMORY_UPDATED,
        base_version=1,
    )

    content = runtime.get("m1").content
    assert "fact two" in content and "fact three" in content
    assert runtime.metrics.conflicts_merged == 1


async def test_cascade_is_bounded_by_the_depth_ceiling():
    runtime = EventMemRuntime(max_depth=3)
    Agent("a", runtime)  # registers an inbox for "a" to deliver into
    await runtime.subscribe(Subscription("a", include_own=True))

    deep = MemoryEvent(
        event_type=EventType.MEMORY_CREATED, source_agent="a", memory_id="m1",
        payload={"content": "too deep"}, depth=4,
    )
    await runtime.publish(deep)

    assert runtime.metrics.cascades_capped == 1
    # Capped means not *delivered*. The fact is still recorded, because losing
    # data is worse than delivering a deep event.
    assert runtime.get("m1") is not None
    assert runtime.metrics.total_deliveries == 0


async def test_replay_rebuilds_the_projection_exactly():
    runtime = EventMemRuntime()
    agent = Agent("a", runtime)

    await agent.remember("one", memory_id="m1")
    await agent.remember("two", memory_id="m2")
    await agent.remember(
        "one revised", memory_id="m1", event_type=EventType.MEMORY_UPDATED,
    )
    await agent.remember("three", memory_id="m3")
    await agent.remember(
        "", memory_id="m3", event_type=EventType.MEMORY_DELETED,
    )

    before = {r.memory_id: (r.content, r.version) for r in runtime.records.all()}

    runtime.records.clear()
    assert runtime.records.all() == []

    replayed = await runtime.rebuild_projection()

    after = {r.memory_id: (r.content, r.version) for r in runtime.records.all()}
    assert after == before
    assert replayed == runtime.metrics.events_in_log
    assert "m3" not in after


async def test_replay_does_not_redeliver():
    """Rebuilding state must not re-fire side effects."""
    runtime = EventMemRuntime()
    writer = Agent("writer", runtime)
    reader = Agent("reader", runtime)
    await runtime.subscribe(Subscription("reader"))
    await reader.start()

    await writer.remember("a fact")
    await _settle(reader)
    delivered_before = len(reader.received)

    await runtime.rebuild_projection()
    await _settle(reader)

    assert len(reader.received) == delivered_before
    await reader.stop()


async def test_provenance_lineage_traces_a_cascade():
    runtime = EventMemRuntime()
    a = Agent("a", runtime)

    root = await a.remember("root cause", memory_id="m1", reason="observed")
    child = MemoryEvent(
        event_type=EventType.MEMORY_CREATED, source_agent="b", memory_id="m2",
        payload={"content": "derived"}, caused_by=root.event_id, depth=1,
        provenance=Provenance(source_agent="b", reason="derived from m1"),
    )
    await runtime.publish(child)

    chain = runtime.provenance.lineage(child.event_id)
    assert [n.memory_id for n in chain] == ["m1", "m2"]
    assert runtime.provenance.root_cause(child.event_id).memory_id == "m1"
    assert "m2" in runtime.provenance.contaminated_memories(root.event_id)


async def test_trust_path_takes_the_weakest_link():
    runtime = EventMemRuntime()
    a = Agent("a", runtime)

    root = await a.remember("shaky", memory_id="m1", confidence=0.3)
    child = MemoryEvent(
        event_type=EventType.MEMORY_CREATED, source_agent="b", memory_id="m2",
        payload={"content": "confident conclusion from shaky premise"},
        caused_by=root.event_id,
        provenance=Provenance(source_agent="b", confidence=1.0),
    )
    await runtime.publish(child)

    # A confident conclusion drawn from a shaky premise is not confident.
    assert runtime.provenance.trust_path("m2") == pytest.approx(0.3)


async def test_agent_survives_a_failing_handler():
    runtime = EventMemRuntime()
    writer = Agent("writer", runtime)

    class Exploding(Agent):
        async def handle(self, event):
            raise RuntimeError("boom")

    victim = Exploding("victim", runtime)
    await runtime.subscribe(Subscription("victim"))
    await victim.start()

    await writer.remember("first")
    await writer.remember("second")
    await _settle(victim)

    assert len(victim.errors) == 2
    assert len(victim.received) == 2  # still consuming after the first failure
    await victim.stop()


async def test_last_writer_wins_policy_is_swappable():
    runtime = EventMemRuntime(conflict_policy=LastWriterWins())
    agent = Agent("a", runtime)

    await agent.remember("first", memory_id="m1", confidence=0.99)
    await agent.remember(
        "second", memory_id="m1", event_type=EventType.MEMORY_UPDATED,
        base_version=1, confidence=0.99,
    )
    await agent.remember(
        "third", memory_id="m1", event_type=EventType.MEMORY_UPDATED,
        base_version=1, confidence=0.01,
    )

    # Low confidence still wins, because the policy ignores confidence.
    assert runtime.get("m1").content == "third"


async def _settle(*agents: Agent) -> None:
    """Wait until every delivered event has actually been handled.

    Uses the bus's in-flight count rather than a fixed sleep or a queue-depth
    check. Queue depth is not a drain signal: asyncio.wait_for removes the item
    before the handler resumes, so depth hits zero while an event is still in
    flight.
    """
    if not agents:
        return
    drained = await agents[0].runtime.bus.drain(timeout=5.0)
    assert drained, "bus did not drain: a handler is stuck"
    for agent in agents:
        await agent.wait_idle()

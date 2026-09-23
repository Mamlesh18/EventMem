"""Metrics arithmetic, lifecycle transitions, serialisation, and the harness."""

from __future__ import annotations

import math

import pytest

from eventmem import (
    AgeAndUsePolicy,
    Agent,
    EventMemRuntime,
    EventType,
    LifecycleState,
    MemoryRecord,
    Provenance,
    Subscription,
    TTLPolicy,
)
from eventmem.core.clock import Stamp, elapsed, now
from eventmem.lifecycle import LifecycleManager, is_legal
from eventmem.metrics import Delivery, MetricSet, ReadObservation, RuntimeMetrics
from eventmem.transport.serde import event_from_json, event_to_json

# ------------------------------------------------------------------- clock

def test_same_process_stamps_use_the_precise_clock():
    seconds, precise = elapsed(now(), now())
    assert precise
    assert seconds >= 0


def test_cross_process_stamps_fall_back_to_the_wall_clock():
    """perf_counter epochs differ per process, so mixing them must be detected."""
    foreign = Stamp(wall=1000.0, perf=5.0, origin="other-process")
    seconds, precise = elapsed(foreign, Stamp(wall=1000.5, perf=99999.0))
    assert not precise
    assert seconds == pytest.approx(0.5)


# ----------------------------------------------------------------- metrics

def test_fan_out_efficiency_uses_n_minus_one_as_the_denominator():
    """An agent never receives its own event, so the broadcast baseline is
    N-1 per event. Using N would inflate the result."""
    metrics = RuntimeMetrics(agent_count=5)
    metrics.fan_out_sizes = [1, 2, 1, 1]  # 4 events, 5 deliveries
    for _ in range(5):
        metrics.deliveries.append(
            Delivery("e", "m", "a", now(), now())
        )

    # 1 - 5 / (4 * 4) = 0.6875
    assert metrics.fan_out_efficiency() == pytest.approx(0.6875)


def test_fan_out_efficiency_is_zero_for_a_broadcaster():
    metrics = RuntimeMetrics(agent_count=3)
    metrics.fan_out_sizes = [2, 2]
    for _ in range(4):
        metrics.deliveries.append(Delivery("e", "m", "a", now(), now()))

    assert metrics.fan_out_efficiency() == pytest.approx(0.0)


def test_freshness_counts_only_reads_of_the_current_version():
    metrics = MetricSet(system="test")
    metrics.reads = [
        ReadObservation("a", "m1", held_version=2, latest_version=2, at=0),
        ReadObservation("a", "m2", held_version=1, latest_version=3, at=0),
        ReadObservation("b", "m1", held_version=None, latest_version=2, at=0),
    ]

    assert metrics.freshness() == pytest.approx(1 / 3)
    assert metrics.staleness() == pytest.approx(2 / 3)


def test_consistency_measures_agreement_among_holders_only():
    """An agent that never saw a memory is a coverage problem, not a
    consistency one, and must not be counted as disagreeing."""
    metrics = MetricSet(system="test")
    metrics.authoritative_versions = {"m1": 3}
    metrics.held_versions = {
        "a": {"m1": 3},
        "b": {"m1": 3},
        "c": {"m1": 1},
        "d": {},  # never saw it at all
    }

    assert metrics.consistency() == pytest.approx(2 / 3)


def test_routing_precision_and_recall_against_ground_truth():
    metrics = MetricSet(system="test")
    metrics.required_deliveries = {("m1", "a"), ("m1", "b"), ("m2", "a")}
    metrics.actual_deliveries = {("m1", "a"), ("m1", "c")}

    assert metrics.routing_precision() == pytest.approx(0.5)
    assert metrics.routing_recall() == pytest.approx(1 / 3)
    assert metrics.wasted_deliveries() == 1


def test_duplicate_reduction_requires_an_explicit_baseline():
    metrics = MetricSet(system="test")
    metrics.tool_calls = 1
    metrics.tool_calls_avoided = 3

    assert metrics.duplicate_reduction(baseline_tool_calls=4) == pytest.approx(0.75)
    # Without a baseline it falls back to this run's own denominator.
    assert metrics.duplicate_reduction() == pytest.approx(0.75)


def test_metrics_return_nan_rather_than_a_flattering_zero():
    """A metric with no data must not report 0.0, which reads as a real
    measurement of 'nothing'."""
    metrics = MetricSet(system="test")
    assert math.isnan(metrics.freshness())
    assert math.isnan(metrics.consistency())
    assert math.isnan(metrics.routing_precision())


# --------------------------------------------------------------- lifecycle

def test_lifecycle_transitions_only_move_forward():
    assert is_legal(LifecycleState.CREATED, LifecycleState.VERIFIED)
    assert is_legal(LifecycleState.ARCHIVED, LifecycleState.EXPIRED)
    assert not is_legal(LifecycleState.EXPIRED, LifecycleState.CREATED)
    assert not is_legal(LifecycleState.ARCHIVED, LifecycleState.VERIFIED)


def test_ttl_policy_expires_an_old_record():
    from eventmem.store.memory import InMemoryRecordStore

    store = InMemoryRecordStore()
    record = MemoryRecord(memory_id="m1", content="old")
    record.updated_at = 1000.0
    store.upsert(record)
    record.updated_at = 1000.0  # upsert touches, so set it back

    manager = LifecycleManager(store, TTLPolicy(ttl_s=60))
    events = manager.sweep(now=1100.0)

    assert len(events) == 1
    assert events[0].event_type == EventType.LIFECYCLE_CHANGED
    assert store.get("m1").lifecycle_state == LifecycleState.EXPIRED


def test_age_and_use_policy_promotes_a_corroborated_record():
    from eventmem.store.memory import InMemoryRecordStore

    store = InMemoryRecordStore()
    store.upsert(MemoryRecord(memory_id="m1", content="trusted", confidence=0.95))

    manager = LifecycleManager(store, AgeAndUsePolicy())
    manager.sweep(now=store.get("m1").updated_at)

    assert store.get("m1").lifecycle_state == LifecycleState.VERIFIED


async def test_expired_records_drop_out_of_reads():
    runtime = EventMemRuntime(lifecycle_policy=TTLPolicy(ttl_s=0.0))
    agent = Agent("a", runtime)
    await agent.remember("transient knowledge", memory_id="m1")

    assert await runtime.recall("transient knowledge") != []

    runtime.sweep_lifecycle()

    assert await runtime.recall("transient knowledge") == []
    assert runtime.metrics.lifecycle_transitions == 1


async def test_lifecycle_change_is_published_to_subscribers():
    """Silently retiring a record that agents still believe is live is exactly
    the staleness the runtime exists to prevent."""
    runtime = EventMemRuntime(lifecycle_policy=TTLPolicy(ttl_s=0.0))
    writer = Agent("writer", runtime)
    watcher = Agent("watcher", runtime)

    await runtime.subscribe(
        Subscription("watcher", event_types={EventType.LIFECYCLE_CHANGED})
    )
    await watcher.start()

    await writer.remember("transient", memory_id="m1")
    await runtime.sweep_and_publish()

    import asyncio
    await asyncio.sleep(0.05)

    assert len(watcher.received) == 1
    assert watcher.received[0].payload["new_state"] == "expired"
    await watcher.stop()


# --------------------------------------------------------------------- serde

def test_event_survives_a_round_trip():
    from eventmem import MemoryEvent

    original = MemoryEvent(
        event_type=EventType.TASK_COMPLETED, source_agent="a", memory_id="m1",
        payload={"content": "done", "extra": 1}, attributes={"topic": "x"},
        provenance=Provenance("a", reason="because", confidence=0.7),
        base_version=3, depth=2, caused_by="parent-id", committed_version=4,
    )
    original.published_at = now()

    restored = event_from_json(event_to_json(original, include_embedding=True))

    assert restored.event_id == original.event_id
    assert restored.event_type == original.event_type
    assert restored.payload == original.payload
    assert restored.provenance.confidence == pytest.approx(0.7)
    assert restored.base_version == 3
    assert restored.committed_version == 4
    assert restored.published_at.origin == original.published_at.origin


def test_unknown_wire_fields_are_ignored():
    """A newer publisher must not break an older consumer."""
    import json

    from eventmem import MemoryEvent

    payload = json.loads(event_to_json(MemoryEvent(
        event_type=EventType.MEMORY_CREATED, source_agent="a", memory_id="m1",
    )))
    payload["a_field_from_the_future"] = {"nested": True}
    payload["provenance"] = {"source_agent": "a", "unknown_key": 1}

    restored = event_from_json(json.dumps(payload))
    assert restored.provenance.source_agent == "a"


def test_embedding_is_omitted_from_inbox_copies():
    """Routing is already done by delivery time; the vector is dead weight."""
    from eventmem import MemoryEvent

    event = MemoryEvent(
        event_type=EventType.MEMORY_CREATED, source_agent="a", memory_id="m1",
    )
    event.embedding = [0.1] * 1024

    assert event_from_json(event_to_json(event)).embedding is None
    assert event_from_json(
        event_to_json(event, include_embedding=True)
    ).embedding is not None

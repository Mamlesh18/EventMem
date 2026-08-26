"""
runtime.py

The runtime is the operating system for memory. It owns every service and runs
the pipeline that turns a published event into an update delivered to the right
agents. An agent talks only to the runtime. It never touches the store, the bus,
or the subscription manager directly.

The publish pipeline, in order:

    1. embed          give the event a vector so it can be routed by meaning
    2. conflict check for updates, compare against the current record version
    3. persist        append the immutable event to the log
    4. project        upsert or delete in the semantic store
    5. lifecycle      advance the record's stage if warranted
    6. fan out        ask the subscription manager who wants it, deliver to them

Metrics are collected as the pipeline runs. Propagation latency and fan out are
measured for real. The rest are exposed as counters the demo drives.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from bus import InMemoryEventBus
from conflict import ConfidenceThenRecency, ConflictPolicy
from embeddings import Embedder, HashingEmbedder
from event_store import InMemoryEventStore
from events import (
    EventType,
    LifecycleState,
    MemoryEvent,
    MemoryRecord,
    Provenance,
)
from semantic_memory import SemanticMemory
from subscriptions import Subscription, SubscriptionManager


@dataclass
class Metrics:
    events_published: int = 0
    total_deliveries: int = 0
    conflicts_detected: int = 0
    conflicts_resolved: int = 0
    duplicate_work_avoided: int = 0
    propagation_latencies: List[float] = field(default_factory=list)
    fan_out_sizes: List[int] = field(default_factory=list)

    def avg_latency_ms(self) -> float:
        if not self.propagation_latencies:
            return 0.0
        return 1000.0 * sum(self.propagation_latencies) / len(self.propagation_latencies)

    def avg_fan_out(self) -> float:
        if not self.fan_out_sizes:
            return 0.0
        return sum(self.fan_out_sizes) / len(self.fan_out_sizes)


class EventMemRuntime:
    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        conflict_policy: Optional[ConflictPolicy] = None,
    ) -> None:
        self.embedder: Embedder = embedder or HashingEmbedder()
        self.conflict_policy: ConflictPolicy = conflict_policy or ConfidenceThenRecency()
        self.event_store = InMemoryEventStore()
        self.memory = SemanticMemory()
        self.subscriptions = SubscriptionManager()
        self.bus = InMemoryEventBus()
        self.metrics = Metrics()
        self._agent_ids: List[str] = []

    # ------------------------------------------------------------------ agents

    def register_agent(self, agent_id: str):
        """Give an agent an inbox and return the queue it should read from."""
        self._agent_ids.append(agent_id)
        return self.bus.register_agent(agent_id)

    def subscribe(self, subscription: Subscription) -> None:
        """
        Register interest. If the subscription carries a semantic query, embed
        it once here so matching never has to embed on the hot path.
        """
        if subscription.semantic_query and subscription.semantic_embedding is None:
            subscription.semantic_embedding = self.embedder.embed(subscription.semantic_query)
        self.subscriptions.register(subscription)

    # ----------------------------------------------------------------- writing

    async def publish(self, event: MemoryEvent) -> MemoryEvent:
        """
        The one entry point for changing shared memory. Runs the full pipeline
        and returns the event that was actually recorded, which may differ from
        the input if a conflict was resolved.
        """
        self.metrics.events_published += 1

        # 1. embed
        event.embedding = self.embedder.embed(event.content_for_embedding())
        event.published_at = time.perf_counter()

        # 2. conflict check and 4. project, handled per event type
        recorded = await self._apply(event)

        # 3. persist the immutable event
        await self.event_store.append(recorded)

        # 6. fan out
        await self._fan_out(recorded)
        return recorded

    async def _apply(self, event: MemoryEvent) -> MemoryEvent:
        """
        Turn an event into a change in the projection. Create and update events
        write a record. Delete removes one. Signalling events such as
        TASK_COMPLETED carry no record change, they exist only to notify.
        """
        if event.event_type in (EventType.MEMORY_CREATED, EventType.PREFERENCE_CHANGED,
                                 EventType.OBSERVATION_ADDED):
            record = self._record_from_event(event)
            self.memory.upsert(record)
            return event

        if event.event_type == EventType.MEMORY_UPDATED:
            return await self._apply_update(event)

        if event.event_type == EventType.MEMORY_DELETED:
            self.memory.delete(event.memory_id)
            return event

        # Signalling events, nothing to project.
        return event

    async def _apply_update(self, event: MemoryEvent) -> MemoryEvent:
        current = self.memory.get(event.memory_id)

        # First time we see this id, treat the update as a create.
        if current is None:
            record = self._record_from_event(event)
            self.memory.upsert(record)
            return event

        incoming = self._record_from_event(event, base=current)

        # Concurrency check. If the writer read an older version than the store
        # now holds, two writes collided.
        collided = event.base_version is not None and event.base_version != current.version
        if not collided:
            incoming.version = current.version + 1
            self.memory.upsert(incoming)
            return event

        # Conflict path.
        self.metrics.conflicts_detected += 1
        resolution = self.conflict_policy.resolve(current, incoming)
        self.metrics.conflicts_resolved += 1

        winner = resolution.winner
        winner.version = current.version + 1
        self.memory.upsert(winner)

        # Announce the conflict so interested agents learn a collision happened
        # and how it was settled. This is itself an event, fanned out normally.
        notice = MemoryEvent(
            event_type=EventType.CONFLICT_DETECTED,
            source_agent="runtime",
            memory_id=event.memory_id,
            payload={"content": f"conflict on {event.memory_id} resolved by {resolution.reason}"},
            attributes=dict(event.attributes),
            provenance=Provenance(source_agent="runtime", reason=resolution.reason),
        )
        notice.embedding = self.embedder.embed(notice.content_for_embedding())
        notice.published_at = time.perf_counter()
        await self.event_store.append(notice)
        await self._fan_out(notice)

        return event

    def _record_from_event(
        self, event: MemoryEvent, base: Optional[MemoryRecord] = None
    ) -> MemoryRecord:
        provenance = event.provenance or Provenance(source_agent=event.source_agent)
        return MemoryRecord(
            memory_id=event.memory_id,
            content=event.payload.get("content", ""),
            attributes=dict(event.attributes),
            version=base.version if base else 1,
            confidence=provenance.confidence,
            provenance=provenance,
            embedding=event.embedding,
            lifecycle_state=LifecycleState.CREATED,
        )

    # ----------------------------------------------------------------- reading

    def recall(self, query: str, k: int = 3, **filters) -> List[MemoryRecord]:
        """
        Let an agent read shared memory on demand. This is the escape hatch for
        the cases where an agent genuinely needs to pull rather than wait for a
        push, for example when it starts cold and needs to catch up.
        """
        query_vec = self.embedder.embed(query)
        results = self.memory.search(query_vec, k=k, attribute_filters=filters)
        return [record for record, _score in results]

    # ------------------------------------------------------------- delivery

    async def _fan_out(self, event: MemoryEvent) -> None:
        recipients = self.subscriptions.match(event)
        self.metrics.fan_out_sizes.append(len(recipients))
        for agent_id in recipients:
            await self.bus.deliver(agent_id, event)
            self.metrics.total_deliveries += 1

    def record_delivery_latency(self, event: MemoryEvent) -> None:
        """Called by an agent the moment it pulls an event off its inbox."""
        if event.published_at is not None:
            self.metrics.propagation_latencies.append(time.perf_counter() - event.published_at)

    # ---------------------------------------------------------- housekeeping

    def run_lifecycle(self, max_age_seconds: float = 3600.0) -> int:
        """
        Advance old records toward archival. In a real system this runs on a
        timer. Here it is a plain method the demo can call so the behaviour is
        visible. Returns how many records changed state.
        """
        now = time.time()
        changed = 0
        for record in self.memory.all():
            age = now - record.updated_at
            if record.lifecycle_state == LifecycleState.CREATED and record.access_count > 0:
                record.lifecycle_state = LifecycleState.VERIFIED
                changed += 1
            elif age > max_age_seconds and record.lifecycle_state != LifecycleState.ARCHIVED:
                record.lifecycle_state = LifecycleState.ARCHIVED
                changed += 1
        return changed

"""
runtime.py

The operating system for memory. It owns every service and runs the pipeline
that turns a published event into an update delivered to the right agents.

This version accepts its transport and its event store from outside, so the
same runtime runs on an in memory bus for a quick check or on Redis for the real
thing. Embedding runs off the event loop when the embedder supports it, so a
slow model call never stalls delivery. A depth ceiling stops LLM agents from
reacting to each other forever. A tracer reports each stage so the push is
visible.

Publish pipeline:
    1. embed        give the event a vector for semantic routing
    2. conflict     for updates, compare declared version against current
    3. persist      append the immutable event to the log
    4. project      upsert or delete in the semantic store
    5. fan out      match subscriptions and deliver, unless depth is capped
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

from bus import InMemoryEventBus
from conflict import ConfidenceThenRecency, ConflictPolicy
from embeddings import HashingEmbedder
from event_store import InMemoryEventStore
from events import EventType, LifecycleState, MemoryEvent, MemoryRecord, Provenance
from semantic_memory import SemanticMemory
from subscriptions import Subscription, SubscriptionManager
from tracing import NullTracer, Tracer


@dataclass
class Metrics:
    events_published: int = 0
    events_in_log: int = 0
    total_deliveries: int = 0
    conflicts_detected: int = 0
    conflicts_resolved: int = 0
    duplicate_work_avoided: int = 0
    cascades_capped: int = 0
    llm_calls: int = 0
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
        embedder: Optional[Any] = None,
        conflict_policy: Optional[ConflictPolicy] = None,
        bus: Optional[Any] = None,
        event_store: Optional[Any] = None,
        max_depth: int = 6,
        tracer: Optional[Tracer] = None,
    ) -> None:
        self.embedder = embedder or HashingEmbedder()
        self.conflict_policy: ConflictPolicy = conflict_policy or ConfidenceThenRecency()
        self.event_store = event_store or InMemoryEventStore()
        self.bus = bus or InMemoryEventBus()
        self.memory = SemanticMemory()
        self.subscriptions = SubscriptionManager()
        self.metrics = Metrics()
        self.max_depth = max_depth
        self.tracer = tracer or NullTracer()
        self._agent_ids: List[str] = []

    async def _embed(self, text: str) -> List[float]:
        if hasattr(self.embedder, "embed_async"):
            return await self.embedder.embed_async(text)
        return self.embedder.embed(text)

    # ------------------------------------------------------------------ agents

    def register_agent(self, agent_id: str):
        self._agent_ids.append(agent_id)
        return self.bus.register_agent(agent_id)

    async def subscribe(self, subscription: Subscription) -> None:
        if subscription.semantic_query and subscription.semantic_embedding is None:
            subscription.semantic_embedding = await self._embed(subscription.semantic_query)
        self.subscriptions.register(subscription)

    # ----------------------------------------------------------------- writing

    async def publish(self, event: MemoryEvent) -> MemoryEvent:
        self.metrics.events_published += 1
        event.embedding = await self._embed(event.content_for_embedding())
        event.published_at = time.perf_counter()
        self.tracer.on_publish(event)

        recorded = await self._apply(event)

        await self.event_store.append(recorded)
        self.metrics.events_in_log += 1

        await self._fan_out(recorded)
        return recorded

    async def _apply(self, event: MemoryEvent) -> MemoryEvent:
        if event.event_type in (
            EventType.MEMORY_CREATED,
            EventType.PREFERENCE_CHANGED,
            EventType.OBSERVATION_ADDED,
        ):
            self.memory.upsert(self._record_from_event(event))
            return event
        if event.event_type == EventType.MEMORY_UPDATED:
            return await self._apply_update(event)
        if event.event_type == EventType.MEMORY_DELETED:
            self.memory.delete(event.memory_id)
            return event
        return event

    async def _apply_update(self, event: MemoryEvent) -> MemoryEvent:
        current = self.memory.get(event.memory_id)
        if current is None:
            self.memory.upsert(self._record_from_event(event))
            return event

        incoming = self._record_from_event(event, base=current)
        collided = event.base_version is not None and event.base_version != current.version
        if not collided:
            incoming.version = current.version + 1
            self.memory.upsert(incoming)
            return event

        self.metrics.conflicts_detected += 1
        resolution = self.conflict_policy.resolve(current, incoming)
        self.metrics.conflicts_resolved += 1
        winner = resolution.winner
        winner.version = current.version + 1
        self.memory.upsert(winner)

        notice = MemoryEvent(
            event_type=EventType.CONFLICT_DETECTED,
            source_agent="runtime",
            memory_id=event.memory_id,
            payload={"content": f"conflict on {event.memory_id} resolved by {resolution.reason}"},
            attributes=dict(event.attributes),
            provenance=Provenance(source_agent="runtime", reason=resolution.reason),
            depth=event.depth + 1,
            caused_by=event.event_id,
        )
        notice.embedding = await self._embed(notice.content_for_embedding())
        notice.published_at = time.perf_counter()
        self.tracer.on_publish(notice)
        await self.event_store.append(notice)
        self.metrics.events_in_log += 1
        await self._fan_out(notice)
        return event

    def _record_from_event(self, event: MemoryEvent, base: Optional[MemoryRecord] = None) -> MemoryRecord:
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

    async def recall(self, query: str, k: int = 3, **filters) -> List[MemoryRecord]:
        query_vec = await self._embed(query)
        results = self.memory.search(query_vec, k=k, attribute_filters=filters)
        return [record for record, _score in results]

    # ------------------------------------------------------------- delivery

    async def _fan_out(self, event: MemoryEvent) -> None:
        if event.depth > self.max_depth:
            self.metrics.cascades_capped += 1
            self.tracer.on_capped(event)
            return
        recipients = self.subscriptions.match(event)
        self.metrics.fan_out_sizes.append(len(recipients))
        self.tracer.on_route(event, recipients)
        for agent_id in recipients:
            await self.bus.deliver(agent_id, event)
            self.metrics.total_deliveries += 1
            self.tracer.on_deliver(agent_id, event)

    def record_delivery_latency(self, event: MemoryEvent) -> None:
        if event.published_at is not None:
            self.metrics.propagation_latencies.append(time.perf_counter() - event.published_at)

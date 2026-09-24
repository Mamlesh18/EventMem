"""The EventMem runtime: the publish pipeline and the services around it.

Every collaborator is injected and held by protocol, so the routing ideology,
the conflict rule, the transport, the log, the record store, the embedder and
the lifecycle policy are all constructor arguments.

Publish pipeline:

    1. embed      give the event a vector for semantic routing
    2. resolve    for updates, compare declared base version against current
    3. persist    append the immutable event to the log (source of truth)
    4. project    upsert or delete in the record store
    5. route      compute the recipient set, unless the depth ceiling is hit
    6. deliver    push to each recipient's inbox

Order matters. Persisting before delivery means a crash between the two loses
a *delivery*, which a consumer group recovers, rather than losing the *fact*,
which nothing recovers.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence

from .conflict.base import Resolution, detect_collision
from .conflict.policies import ConfidenceThenRecency
from .core.clock import now
from .core.events import (
    MUTATING_TYPES,
    EventType,
    LifecycleState,
    MemoryEvent,
    MemoryRecord,
    Provenance,
)
from .embeddings.backends import HashingEmbedder
from .lifecycle import LifecycleManager, NeverExpire
from .metrics import Delivery, RuntimeMetrics
from .provenance import ProvenanceTracker
from .routing.layered import LayeredRouter
from .store.memory import InMemoryEventStore, InMemoryRecordStore
from .tracing import NullTracer
from .transport.memory import InMemoryEventBus

#: Default cascade ceiling. Deep enough for realistic agent chains, shallow
#: enough that a runaway loop is capped in well under a second of model calls.
DEFAULT_MAX_DEPTH = 6


class EventMemRuntime:
    """Owns the services and runs the pipeline."""

    def __init__(
        self,
        *,
        embedder: Any = None,
        router: Any = None,
        conflict_policy: Any = None,
        bus: Any = None,
        event_store: Any = None,
        record_store: Any = None,
        lifecycle_policy: Any = None,
        tracer: Any = None,
        max_depth: int = DEFAULT_MAX_DEPTH,
        embed_attribute_keys: Optional[List[str]] = None,
        track_provenance: bool = True,
    ) -> None:
        self.embedder = embedder or HashingEmbedder()
        self.router = router or LayeredRouter()
        self.conflict_policy = conflict_policy or ConfidenceThenRecency()
        self.event_store = event_store or InMemoryEventStore()
        self.records = record_store or InMemoryRecordStore()
        self.bus = bus or InMemoryEventBus()
        self.tracer = tracer or NullTracer()
        self.max_depth = max_depth
        self.embed_attribute_keys = embed_attribute_keys

        self.provenance = ProvenanceTracker() if track_provenance else None
        self.lifecycle = LifecycleManager(
            self.records, lifecycle_policy or NeverExpire(), self.tracer
        )
        self.metrics = RuntimeMetrics()

        # The router must embed interest queries with the same model that
        # embeds events, or cosine similarity compares vectors from different
        # spaces and the numbers are meaningless.
        if hasattr(self.router, "bind_embedder"):
            self.router.bind_embedder(self.embedder)

        self._agent_ids: List[str] = []
        self._warned_non_semantic = False

    # ------------------------------------------------------------- lifecycle

    @property
    def agent_ids(self) -> List[str]:
        return list(self._agent_ids)

    def register_agent(self, agent_id: str):
        if agent_id not in self._agent_ids:
            self._agent_ids.append(agent_id)
            self.metrics.agent_count = len(self._agent_ids)
        return self.bus.register_agent(agent_id)

    async def subscribe(self, subscription: Any) -> None:
        """Register a subscription with the router.

        Warns once if a semantic subscription is registered against an embedder
        that declares itself non-semantic. That combination silently degrades
        layer-3 matching into token overlap, and it has produced published
        results that claim semantic routing while doing keyword matching.
        """
        if (
            getattr(subscription, "semantic_query", None)
            and not getattr(self.embedder, "semantic", True)
            and not self._warned_non_semantic
        ):
            self._warned_non_semantic = True
            warnings.warn(
                f"semantic subscription registered against "
                f"{getattr(self.embedder, 'name', type(self.embedder).__name__)}, "
                "which does not model meaning: layer-3 matching degrades to token "
                "overlap and scores paraphrases at 0. Use "
                "SentenceTransformerEmbedder or a hosted embedder for semantic "
                "routing, or call Subscription.calibrate() to see the real margin.",
                RuntimeWarning,
                stacklevel=2,
            )
        await self.router.register(subscription)

    async def close(self) -> None:
        await self.lifecycle.stop()
        await self.bus.close()

    # ---------------------------------------------------------------- writing

    async def publish(self, event: MemoryEvent) -> MemoryEvent:
        """Run one event through the full pipeline."""
        self.metrics.events_published += 1

        event.embedding = await self._embed(
            event.text_for_embedding(self.embed_attribute_keys)
        )
        event.published_at = now()
        self.tracer.on_publish(event)

        await self._apply(event)

        await self.event_store.append(event)
        self.metrics.events_in_log += 1
        if self.provenance is not None:
            self.provenance.record(event)

        await self._route_and_deliver(event)
        return event

    async def _embed(self, text: str) -> List[float]:
        self.metrics.embed_calls += 1
        if hasattr(self.embedder, "embed_async"):
            return await self.embedder.embed_async(text)
        return self.embedder.embed(text)

    async def _apply(self, event: MemoryEvent) -> None:
        """Project the event into the record store, resolving conflicts."""
        if event.event_type == EventType.MEMORY_DELETED:
            self.records.delete(event.memory_id)
            return
        if event.event_type not in MUTATING_TYPES:
            # conflict_detected and lifecycle_changed are notifications about
            # state, not assertions of it. Projecting them would overwrite the
            # record they describe with the text of the notice.
            return

        # count_access=False: the runtime's own read must not inflate the
        # statistics that lifecycle policies then act on.
        current = self.records.get(event.memory_id, count_access=False)

        if current is None:
            record = self._record_from(event, version=1)
            self.records.upsert(record)
            event.committed_version = record.version
            return

        incoming = self._record_from(event, version=current.version, base=current)

        if not detect_collision(event.base_version, current):
            incoming.version = current.version + 1
            incoming.created_at = current.created_at
            incoming.access_count = current.access_count
            self.records.upsert(incoming)
            event.committed_version = incoming.version
            return

        self.metrics.conflicts_detected += 1
        resolution: Resolution = self.conflict_policy.resolve(current, incoming)
        self.metrics.conflicts_resolved += 1
        if resolution.merged:
            self.metrics.conflicts_merged += 1
        self.tracer.on_conflict(event, resolution)

        winner = resolution.winner
        winner.version = current.version + 1
        winner.created_at = current.created_at
        winner.last_event_id = event.event_id
        self.records.upsert(winner)
        event.committed_version = winner.version

        await self._publish_notice(event, resolution)

    async def _publish_notice(self, cause: MemoryEvent, resolution: Resolution) -> None:
        """Tell subscribers a conflict was settled.

        Published through the same pipeline as anything else, so it is logged,
        routed and depth-bounded like a normal event. It carries the cause's
        attributes so agents filtering on that memory's topic hear about it.
        """
        notice = cause.child(
            event_type=EventType.CONFLICT_DETECTED,
            source_agent="runtime",
            payload={
                "content": (
                    f"conflict on {cause.memory_id} resolved: {resolution.reason}"
                ),
                "winner_agent": (
                    resolution.winner.provenance.source_agent
                    if resolution.winner.provenance
                    else ""
                ),
                "reason": resolution.reason,
                "merged": resolution.merged,
            },
            provenance=Provenance(source_agent="runtime", reason=resolution.reason),
        )
        await self.publish(notice)

    def _record_from(
        self,
        event: MemoryEvent,
        version: int,
        base: Optional[MemoryRecord] = None,
    ) -> MemoryRecord:
        provenance = event.provenance or Provenance(source_agent=event.source_agent)
        return MemoryRecord(
            memory_id=event.memory_id,
            content=event.content,
            attributes=dict(event.attributes),
            version=version,
            confidence=provenance.confidence,
            provenance=provenance,
            embedding=event.embedding,
            lifecycle_state=(
                base.lifecycle_state if base is not None else LifecycleState.CREATED
            ),
            last_event_id=event.event_id,
        )

    # --------------------------------------------------------------- delivery

    async def _route_and_deliver(self, event: MemoryEvent) -> None:
        if event.depth > self.max_depth:
            self.metrics.cascades_capped += 1
            self.tracer.on_capped(event)
            return

        recipients = self.router.route(event)
        self.metrics.fan_out_sizes.append(len(recipients))
        self.tracer.on_route(event, recipients)

        for agent_id in recipients:
            await self.bus.deliver(agent_id, event)
            # Stamped at enqueue, not at dequeue. Publish-to-enqueue is the
            # transport cost the runtime is accountable for; publish-to-dequeue
            # additionally includes however long the agent was busy, which
            # belongs to the agent's workload and would otherwise silently
            # inflate every reported latency.
            self.metrics.deliveries.append(
                Delivery(
                    event_id=event.event_id,
                    memory_id=event.memory_id,
                    recipient=agent_id,
                    published_at=event.published_at,
                    enqueued_at=now(),
                )
            )
            self.tracer.on_deliver(agent_id, event)

    def note_pickup(self, event: MemoryEvent, agent_id: str) -> None:
        """Called by an agent when it dequeues an event.

        Completes the pickup timing without conflating it with propagation.
        """
        at = now()
        for delivery in reversed(self.metrics.deliveries):
            if delivery.event_id == event.event_id and delivery.recipient == agent_id:
                delivery.dequeued_at = at
                return

    # ---------------------------------------------------------------- reading

    async def recall(
        self, query: str, k: int = 3, min_score: float = 0.0, **filters: Any
    ) -> List[MemoryRecord]:
        """Semantic read of the current projection."""
        vector = await self._embed(query)
        return [
            record
            for record, _ in self.records.search(
                vector, k=k, attribute_filters=filters, min_score=min_score
            )
        ]

    async def recall_scored(
        self, query: str, k: int = 3, min_score: float = 0.0, **filters: Any
    ):
        vector = await self._embed(query)
        return self.records.search(
            vector, k=k, attribute_filters=filters, min_score=min_score
        )

    def get(self, memory_id: str) -> Optional[MemoryRecord]:
        return self.records.get(memory_id)

    # --------------------------------------------------------------- recovery

    async def rebuild_projection(self) -> int:
        """Rebuild the record store by replaying the log from the beginning.

        This is what makes "the log is the source of truth" a property rather
        than a slogan. Replay is side-effect free: no routing, no delivery, no
        conflict notices, no metrics. It reconstructs state and nothing else.

        Returns the number of events replayed.
        """
        events: Sequence[MemoryEvent] = await self.event_store.replay()
        self.records.clear()
        if self.provenance is not None:
            self.provenance = ProvenanceTracker()

        for event in events:
            if self.provenance is not None:
                self.provenance.record(event)
            if event.event_type == EventType.MEMORY_DELETED:
                self.records.delete(event.memory_id)
                continue
            if event.event_type not in MUTATING_TYPES:
                continue

            current = self.records.get(event.memory_id, count_access=False)
            if current is None:
                self.records.upsert(self._record_from(event, version=1))
                continue
            incoming = self._record_from(event, version=current.version, base=current)
            if detect_collision(event.base_version, current):
                # Replay the same decision the live run made. The policy must
                # be deterministic for this to reproduce, which is why
                # ConfidenceThenRecency breaks final ties on event id.
                incoming = self.conflict_policy.resolve(current, incoming).winner
            incoming.version = current.version + 1
            incoming.created_at = current.created_at
            self.records.upsert(incoming)

        return len(events)

    def sweep_lifecycle(self, now: Optional[float] = None) -> List[MemoryEvent]:
        """Advance lifecycle states and return the events to publish."""
        events = self.lifecycle.sweep(now)
        self.metrics.lifecycle_transitions += len(events)
        return events

    async def sweep_and_publish(self, now: Optional[float] = None) -> int:
        events = self.sweep_lifecycle(now)
        for event in events:
            await self.publish(event)
        return len(events)

    # ------------------------------------------------------------ diagnostics

    def report(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "router": getattr(self.router, "name", "?"),
            "conflict_policy": getattr(self.conflict_policy, "name", "?"),
            "embedder": getattr(self.embedder, "name", type(self.embedder).__name__),
            "transport": getattr(self.bus, "name", "?"),
            "agents": len(self._agent_ids),
            **self.metrics.snapshot(),
        }
        if self.provenance is not None:
            out["provenance"] = self.provenance.stats()
        return out

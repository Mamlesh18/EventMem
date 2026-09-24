"""Every pluggable seam in the runtime, in one file.

This is the contract an extension implements. Nothing here has behaviour; the
point is that a contributor can read one short file and know exactly what they
have to write to swap out a piece of the architecture.

The runtime holds each of these by protocol, never by concrete type, so
replacing the routing ideology, the conflict rule, the transport, the store, or
the embedder is a constructor argument rather than a fork.

See ``docs/extending.md`` for worked examples of each.
"""

from __future__ import annotations

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

from .events import MemoryEvent, MemoryRecord


@runtime_checkable
class Embedder(Protocol):
    """Maps text to a vector for semantic routing and recall.

    Implement ``embed``. If your backend blocks (a network call), also
    implement ``embed_async``; the runtime prefers it and will otherwise call
    ``embed`` directly on the event loop.
    """

    #: Dimensionality of the returned vectors. Used for sanity checks.
    dim: int

    def embed(self, text: str) -> List[float]: ...


@runtime_checkable
class Router(Protocol):
    """Decides which agents receive an event. This is the routing ideology.

    A router owns its own subscription representation; the runtime only asks it
    to register something and to answer ``route``. That means an alternative
    ideology (topic trees, learned routing, rule engines, capability matching)
    can define a completely different subscription object and still drop in.

    ``route`` must be pure with respect to the event: it may read subscription
    state but must not mutate the event.
    """

    name: str

    async def register(self, subscription: Any) -> None:
        """Accept a subscription. May precompute (e.g. embed an interest query)."""
        ...

    def unregister(self, subscriber_id: str) -> None: ...

    def route(self, event: MemoryEvent) -> List[str]:
        """Return the ids of every agent that should receive this event."""
        ...

    def subscribers(self) -> List[str]:
        """Every agent id with at least one registered subscription."""
        ...


@runtime_checkable
class ConflictPolicy(Protocol):
    """Resolves two writes to the same memory that collided.

    Return the record that survives. Implementations may merge rather than
    pick, which is how a CRDT-style policy plugs in.
    """

    name: str

    def resolve(
        self, current: MemoryRecord, incoming: MemoryRecord
    ) -> "Resolution": ...


@runtime_checkable
class EventStore(Protocol):
    """Append-only log. The source of truth.

    ``replay`` must yield every event ever appended, in append order, so the
    read model can be rebuilt from scratch.
    """

    async def append(self, event: MemoryEvent) -> str: ...

    async def replay(self) -> Sequence[MemoryEvent]: ...

    async def length(self) -> int: ...


@runtime_checkable
class Inbox(Protocol):
    """One agent's delivery endpoint."""

    async def setup(self) -> None:
        """Prepare to receive. Must be idempotent and must complete before the
        first delivery, or early events are lost."""
        ...

    async def get(self, timeout: float = 1.0) -> Optional[MemoryEvent]:
        """Block until an event arrives or the timeout elapses.

        The timeout exists so an agent can observe a shutdown request. It is
        not a poll interval: the implementation must sleep until pushed, not
        wake repeatedly to check.
        """
        ...

    async def ack(self, event: MemoryEvent) -> None:
        """Acknowledge successful handling.

        Called *after* the agent has processed the event, which makes delivery
        at-least-once. Acking before handling would make it at-most-once and
        lose events on a crash.
        """
        ...

    async def nack(self, event: MemoryEvent) -> None:
        """Report that handling failed.

        A durable transport should leave the entry recoverable rather than
        acknowledge it. Optional: an inbox without it is treated as having no
        redelivery to manage.
        """
        ...


@runtime_checkable
class EventBus(Protocol):
    """Transport carrying events from the runtime to agent inboxes."""

    name: str

    def register_agent(self, agent_id: str) -> Inbox: ...

    async def deliver(self, agent_id: str, event: MemoryEvent) -> None: ...

    async def close(self) -> None: ...


@runtime_checkable
class RecordStore(Protocol):
    """The searchable projection of current records."""

    def upsert(self, record: MemoryRecord) -> None: ...

    def get(self, memory_id: str, *, count_access: bool = True) -> Optional[MemoryRecord]: ...

    def delete(self, memory_id: str) -> None: ...

    def all(self, *, include_dead: bool = False) -> List[MemoryRecord]: ...

    def search(
        self,
        query_embedding: List[float],
        k: int = 5,
        attribute_filters: Optional[Dict[str, Any]] = None,
        min_score: float = 0.0,
    ) -> List[Tuple[MemoryRecord, float]]: ...

    def clear(self) -> None: ...


@runtime_checkable
class LifecyclePolicy(Protocol):
    """Decides when a record advances to a new lifecycle state.

    Called by the LifecycleManager on a sweep. Return the new state, or None to
    leave the record alone.
    """

    name: str

    def next_state(self, record: MemoryRecord, now: float) -> Optional[Any]: ...


@runtime_checkable
class Tracer(Protocol):
    """Observability hooks. Every runtime stage calls one of these."""

    def on_publish(self, event: MemoryEvent) -> None: ...
    def on_route(self, event: MemoryEvent, recipients: List[str]) -> None: ...
    def on_deliver(self, agent_id: str, event: MemoryEvent) -> None: ...
    def on_capped(self, event: MemoryEvent) -> None: ...
    def on_conflict(self, event: MemoryEvent, resolution: "Resolution") -> None: ...
    def on_lifecycle(self, record: MemoryRecord, previous: Any) -> None: ...


# Imported last to avoid a cycle: Resolution names MemoryRecord.
from ..conflict.base import Resolution  # noqa: E402  isort:skip

__all__ = [
    "Embedder",
    "Router",
    "ConflictPolicy",
    "EventStore",
    "Inbox",
    "EventBus",
    "RecordStore",
    "LifecyclePolicy",
    "Tracer",
    "Resolution",
]

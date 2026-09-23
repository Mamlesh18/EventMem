"""Immutable event and mutable record types.

The event log is the source of truth; the record store is a projection built
from it. Keep these types free of behaviour so every other module can import
them without pulling in dependencies.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Dict, List, Optional

from .clock import Stamp


class EventType(str, Enum):
    """Event vocabulary.

    Values are the wire format and must stay stable across versions; the member
    names are what the paper calls K. Adding a member is backwards compatible,
    renaming a value is not.
    """

    MEMORY_CREATED = "memory_created"
    MEMORY_UPDATED = "memory_updated"
    MEMORY_DELETED = "memory_deleted"
    OBSERVATION_ADDED = "observation_added"
    TASK_COMPLETED = "task_completed"
    GOAL_UPDATED = "goal_updated"
    TOOL_RESULT = "tool_result"
    PREFERENCE_CHANGED = "preference_changed"
    CONFLICT_DETECTED = "conflict_detected"
    LIFECYCLE_CHANGED = "lifecycle_changed"


#: Event types that create or replace the content of a memory record.
MUTATING_TYPES = frozenset(
    {
        EventType.MEMORY_CREATED,
        EventType.MEMORY_UPDATED,
        EventType.OBSERVATION_ADDED,
        EventType.PREFERENCE_CHANGED,
        EventType.TASK_COMPLETED,
        EventType.GOAL_UPDATED,
        EventType.TOOL_RESULT,
    }
)


class LifecycleState(str, Enum):
    """Where a record sits in its life. Advanced by the LifecycleManager."""

    CREATED = "created"
    VERIFIED = "verified"
    SUMMARIZED = "summarized"
    ARCHIVED = "archived"
    EXPIRED = "expired"


#: States whose records are excluded from reads.
DEAD_STATES = frozenset({LifecycleState.ARCHIVED, LifecycleState.EXPIRED})


@dataclass(frozen=True)
class Provenance:
    """Who asserted this, why, and how much they trust it.

    Frozen because provenance describes a past assertion. Revising trust means
    writing a new event, not editing the old one.
    """

    source_agent: str
    reason: str = ""
    source: str = ""
    confidence: float = 1.0
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")


@dataclass
class MemoryEvent:
    """An immutable statement that something happened.

    ``base_version`` drives optimistic concurrency: a writer declares the
    version it read, and if the store has moved past it the writes collided.
    Leaving it ``None`` opts out of conflict detection, which is the right
    choice for append-style events but means blind overwrite for updates.

    ``depth`` bounds reasoning cascades. An agent reacting to a depth-n event
    stamps its result depth n+1, and the runtime refuses to route past a
    ceiling, so agents cannot loop forever by reacting to each other.
    """

    event_type: EventType
    source_agent: str
    memory_id: str
    payload: Dict[str, Any] = field(default_factory=dict)
    attributes: Dict[str, Any] = field(default_factory=dict)
    provenance: Optional[Provenance] = None
    base_version: Optional[int] = None

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)

    #: Taken at publish time on both the precise and the comparable clock.
    #: See core/clock.py: neither clock alone can measure both in-process
    #: delivery and cross-process delivery honestly.
    published_at: Optional["Stamp"] = None

    embedding: Optional[List[float]] = None
    depth: int = 0
    caused_by: Optional[str] = None

    #: Version the runtime committed for this event, stamped during apply. Lets
    #: a recipient know what version it now holds without a follow-up read.
    committed_version: Optional[int] = None

    @property
    def content(self) -> str:
        return self.payload.get("content", "")

    def text_for_embedding(self, attribute_keys: Optional[List[str]] = None) -> str:
        """Text handed to the embedder.

        Selected attributes are folded in so a subscription can match on
        meaning carried by metadata as well as by the body. Which keys get
        folded in is configurable because it is workload-specific.
        """
        keys = attribute_keys if attribute_keys is not None else _DEFAULT_EMBED_KEYS
        parts: List[str] = [self.content]
        for key in keys:
            value = self.attributes.get(key)
            if value is not None:
                parts.append(f"{key} {value}")
        return " ".join(p for p in parts if p).strip()

    def child(self, **overrides: Any) -> "MemoryEvent":
        """Derive a causally-linked successor event.

        Copies nothing that must be unique (id, timestamps, embedding) and
        advances depth, so cascade bounding stays correct by construction.
        """
        base: Dict[str, Any] = dict(  # noqa: C408
            event_type=self.event_type,
            source_agent=self.source_agent,
            memory_id=self.memory_id,
            payload=dict(self.payload),
            attributes=dict(self.attributes),
            provenance=self.provenance,
            base_version=None,
            depth=self.depth + 1,
            caused_by=self.event_id,
        )
        base.update(overrides)
        return MemoryEvent(**base)


_DEFAULT_EMBED_KEYS = ("task_id", "component", "topic", "stage", "customer_id")


@dataclass
class MemoryRecord:
    """Current state of one memory: the read model.

    Version is monotonic per memory_id and is what freshness and consistency
    are measured against.
    """

    memory_id: str
    content: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    version: int = 1
    confidence: float = 1.0
    provenance: Optional[Provenance] = None
    embedding: Optional[List[float]] = None
    lifecycle_state: LifecycleState = LifecycleState.CREATED
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    access_count: int = 0
    #: Event that produced this version, for lineage queries.
    last_event_id: Optional[str] = None

    @property
    def is_live(self) -> bool:
        return self.lifecycle_state not in DEAD_STATES

    def touch(self, when: Optional[float] = None) -> None:
        self.updated_at = when if when is not None else time.time()

    def copy(self, **overrides: Any) -> "MemoryRecord":
        return replace(self, **overrides)

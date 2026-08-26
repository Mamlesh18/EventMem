"""
events.py

The vocabulary of the runtime. Everything that flows through EventMem is a
MemoryEvent. Everything the semantic store holds is a MemoryRecord. These two
types are deliberately small so that every other module can depend on them
without pulling in behaviour.

Design note on storage:
    An event is immutable once created. It records that something happened.
    A record is mutable state, the current view of a single memory. The event
    log is the source of truth, the record store is a projection built from it.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class EventType(str, Enum):
    """
    The kinds of things worth telling other agents about. Keep this list small
    and meaningful. A large taxonomy makes subscriptions harder to reason about.
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


class LifecycleState(str, Enum):
    """
    A memory does not live forever. It moves through stages so the store does
    not grow without bound. The lifecycle manager advances records through
    these states based on age, use, and verification.
    """

    CREATED = "created"
    VERIFIED = "verified"
    SUMMARIZED = "summarized"
    ARCHIVED = "archived"
    EXPIRED = "expired"


@dataclass
class Provenance:
    """
    Who made this memory, when, and why. Provenance is what lets a downstream
    agent decide whether to trust a fact, and it is what makes conflict
    resolution defensible rather than arbitrary.
    """

    source_agent: str
    reason: str = ""
    source: str = ""
    confidence: float = 1.0
    created_at: float = field(default_factory=time.time)


@dataclass
class MemoryRecord:
    """
    The current state of one memory. This is what the semantic store holds and
    what agents read. The embedding is filled in by the runtime, so it stays
    None until the record has passed through the pipeline.
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

    def touch(self) -> None:
        self.updated_at = time.time()


@dataclass
class MemoryEvent:
    """
    An immutable statement that something happened. This is the unit the event
    bus carries and the event store keeps. The payload holds the memory content
    for create and update events, or just the target id for a delete.

    base_version is how we detect concurrent writes. A writer declares the
    version it read. If the store has moved past that version by the time the
    event lands, two agents collided and the conflict resolver steps in.
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
    # A high resolution stamp taken at publish time, used to measure how long
    # the event takes to reach a subscriber. Wall clock is too coarse for this.
    published_at: Optional[float] = None
    embedding: Optional[List[float]] = None

    def content_for_embedding(self) -> str:
        """
        The text we hand to the embedder. We fold in a few structured
        attributes so that a subscription can match on meaning that lives in
        the metadata as well as the body.
        """
        parts: List[str] = [self.payload.get("content", "")]
        for key in ("task_id", "component", "topic", "customer_id"):
            if key in self.attributes:
                parts.append(f"{key} {self.attributes[key]}")
        return " ".join(p for p in parts if p).strip()

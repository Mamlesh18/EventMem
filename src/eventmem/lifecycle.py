"""The Lifecycle Manager.

Memory that only grows is not memory, it is a log. Records advance through
states as they age, get used, get corroborated, or go quiet, and archived and
expired records drop out of reads.

This is a sweep, not a hook on read: the manager is asked to run, evaluates
every record against a policy, applies transitions, and emits a
``lifecycle_changed`` event so agents holding a copy learn the record retired.
That last part matters. Silently archiving a record while three agents still
believe it is live is exactly the staleness the runtime exists to prevent.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .core.events import (
    EventType,
    LifecycleState,
    MemoryEvent,
    MemoryRecord,
    Provenance,
)

#: Legal forward transitions. The lifecycle is a DAG and never runs backwards:
#: a record that expired does not become live again, a new assertion about the
#: same subject is a new event.
LEGAL_TRANSITIONS: Dict[LifecycleState, frozenset] = {
    LifecycleState.CREATED: frozenset(
        {LifecycleState.VERIFIED, LifecycleState.SUMMARIZED,
         LifecycleState.ARCHIVED, LifecycleState.EXPIRED}
    ),
    LifecycleState.VERIFIED: frozenset(
        {LifecycleState.SUMMARIZED, LifecycleState.ARCHIVED, LifecycleState.EXPIRED}
    ),
    LifecycleState.SUMMARIZED: frozenset(
        {LifecycleState.ARCHIVED, LifecycleState.EXPIRED}
    ),
    LifecycleState.ARCHIVED: frozenset({LifecycleState.EXPIRED}),
    LifecycleState.EXPIRED: frozenset(),
}


def is_legal(current: LifecycleState, proposed: LifecycleState) -> bool:
    return proposed in LEGAL_TRANSITIONS.get(current, frozenset())


@dataclass
class AgeAndUsePolicy:
    """Default policy: corroboration promotes, age and disuse retire.

    The thresholds are wall-clock seconds and are deliberately exposed rather
    than tuned, because the right values are entirely workload-dependent: a
    support-desk memory goes stale in hours, a schema decision does not.
    """

    name: str = "age_and_use"

    #: Confidence at or above which a record counts as corroborated.
    verify_confidence: float = 0.9
    #: Reads needed before a merely-plausible record is promoted.
    verify_access_count: int = 3
    #: Seconds of no update after which a live record is archived.
    archive_after_s: float = 3600.0
    #: Seconds after which an archived record expires.
    expire_after_s: float = 86400.0
    #: Records never read at all are archived this much sooner.
    unused_archive_factor: float = 0.25

    def next_state(self, record: MemoryRecord, now: float) -> Optional[LifecycleState]:
        age = now - record.updated_at
        state = record.lifecycle_state

        if state == LifecycleState.ARCHIVED:
            return LifecycleState.EXPIRED if age >= self.expire_after_s else None

        if state in (LifecycleState.CREATED, LifecycleState.VERIFIED,
                     LifecycleState.SUMMARIZED):
            cutoff = self.archive_after_s
            if record.access_count == 0:
                # Nothing has ever read this. Retire it sooner; an unread
                # record is pure carrying cost.
                cutoff *= self.unused_archive_factor
            if age >= cutoff:
                return LifecycleState.ARCHIVED

        if state == LifecycleState.CREATED:
            corroborated = (
                record.confidence >= self.verify_confidence
                or record.access_count >= self.verify_access_count
            )
            if corroborated:
                return LifecycleState.VERIFIED

        return None


@dataclass
class NeverExpire:
    """Null policy. Everything stays CREATED forever.

    The honest default for a benchmark run that should not have records
    retiring underneath it mid-measurement.
    """

    name: str = "never_expire"

    def next_state(self, record: MemoryRecord, now: float) -> Optional[LifecycleState]:
        return None


@dataclass
class TTLPolicy:
    """Hard time-to-live. Expire N seconds after last update, no intermediate
    states. Simplest useful policy and the one to copy."""

    ttl_s: float
    name: str = "ttl"

    def next_state(self, record: MemoryRecord, now: float) -> Optional[LifecycleState]:
        expired = record.lifecycle_state == LifecycleState.EXPIRED
        if not expired and now - record.updated_at >= self.ttl_s:
            return LifecycleState.EXPIRED
        return None


class LifecycleManager:
    """Applies a policy across the record store on demand.

    Not a background task by default. A sweep that fires on a timer inside a
    benchmark makes results depend on scheduler luck, so the caller decides
    when to run it. ``start_background`` is available for real deployments.
    """

    def __init__(self, store: Any, policy: Any = None, tracer: Any = None) -> None:
        self.store = store
        self.policy = policy or NeverExpire()
        self.tracer = tracer
        self.transitions: List[Tuple[str, LifecycleState, LifecycleState]] = []
        self._task = None
        self._running = False

    def sweep(self, now: Optional[float] = None) -> List[MemoryEvent]:
        """Evaluate every record; return the lifecycle events to publish.

        Returns events rather than publishing them so the caller controls
        ordering and the manager stays free of a runtime dependency (which
        would be circular: the runtime owns the manager).
        """
        moment = now if now is not None else time.time()
        events: List[MemoryEvent] = []

        for record in self.store.all(include_dead=True):
            proposed = self.policy.next_state(record, moment)
            if proposed is None or proposed == record.lifecycle_state:
                continue
            if not is_legal(record.lifecycle_state, proposed):
                # A policy asking for an illegal move is a bug in the policy,
                # not a reason to corrupt the record. Skip and keep going.
                continue

            previous = record.lifecycle_state
            record.lifecycle_state = proposed
            self.transitions.append((record.memory_id, previous, proposed))
            if self.tracer is not None:
                self.tracer.on_lifecycle(record, previous)

            events.append(
                MemoryEvent(
                    event_type=EventType.LIFECYCLE_CHANGED,
                    source_agent="lifecycle",
                    memory_id=record.memory_id,
                    payload={
                        "content": (
                            f"memory {record.memory_id} moved from "
                            f"{previous.value} to {proposed.value}"
                        ),
                        "previous_state": previous.value,
                        "new_state": proposed.value,
                    },
                    attributes=dict(record.attributes),
                    provenance=Provenance(
                        source_agent="lifecycle",
                        reason=f"{self.policy.name}: {previous.value} -> {proposed.value}",
                    ),
                )
            )
        return events

    async def start_background(self, runtime: Any, interval_s: float = 60.0) -> None:
        """Run a sweep every ``interval_s`` and publish what it produces.

        For real deployments. Benchmarks should call ``sweep`` explicitly so
        the measurement is not at the mercy of the scheduler.
        """
        import asyncio

        self._running = True

        async def loop() -> None:
            while self._running:
                await asyncio.sleep(interval_s)
                for event in self.sweep():
                    await runtime.publish(event)

        self._task = asyncio.create_task(loop())

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            # The task is being cancelled on purpose; whatever it raises on the
            # way out is not an error the caller can act on.
            with contextlib.suppress(Exception):
                await self._task
            self._task = None

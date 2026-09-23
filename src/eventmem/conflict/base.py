"""Conflict resolution seam.

Optimistic concurrency detects the collision; a policy decides the outcome.
Policies are ordinary objects with one method, so a domain rule ("the security
agent always wins"), a merge, or a human-in-the-loop escalation all fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..core.events import MemoryRecord


@dataclass
class Resolution:
    """Outcome of a collision.

    ``winner`` is what gets committed. ``merged`` is True when the winner is a
    new record built from both sides rather than one of the originals, which
    matters for provenance: a merge has two parents.
    """

    winner: MemoryRecord
    loser: MemoryRecord
    reason: str
    merged: bool = False


class ConflictError(Exception):
    """Raised by a policy that refuses to resolve automatically."""

    def __init__(self, current: MemoryRecord, incoming: MemoryRecord) -> None:
        super().__init__(
            f"unresolvable conflict on {current.memory_id} "
            f"(v{current.version} vs base {incoming.version})"
        )
        self.current = current
        self.incoming = incoming


def detect_collision(base_version: Optional[int], current: Optional[MemoryRecord]) -> bool:
    """Optimistic concurrency check.

    A collision needs three things: an existing record, a declared base
    version, and a mismatch between them. A writer that declares no base
    version has opted out of detection and will overwrite blindly.
    """
    if current is None or base_version is None:
        return False
    return base_version != current.version

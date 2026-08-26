"""
conflict.py

Two agents update the same memory at once. The runtime detects this with
optimistic concurrency: each writer declares the version it read, and if the
store has moved on, the writes collided. This module decides who wins.

The default policy prefers the higher confidence write, breaking ties on
recency. That is a defensible rule because provenance carries a confidence
score. The policy is a plain object with one method, so you can drop in a CRDT
style merge for set valued memories, or a domain rule that always trusts a
particular agent, without touching the runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from events import MemoryRecord


@dataclass
class Resolution:
    winner: MemoryRecord
    loser: MemoryRecord
    reason: str


class ConflictPolicy(Protocol):
    def resolve(self, current: MemoryRecord, incoming: MemoryRecord) -> Resolution:
        ...


class ConfidenceThenRecency:
    def resolve(self, current: MemoryRecord, incoming: MemoryRecord) -> Resolution:
        if incoming.confidence > current.confidence:
            return Resolution(incoming, current, "higher confidence")
        if incoming.confidence < current.confidence:
            return Resolution(current, incoming, "higher confidence")
        # Equal confidence, take the more recent write.
        if incoming.updated_at >= current.updated_at:
            return Resolution(incoming, current, "equal confidence, more recent")
        return Resolution(current, incoming, "equal confidence, more recent")

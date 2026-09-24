"""In-memory record store and event log.

The record store is a projection and is always rebuildable from the log, so
losing it is an availability problem, never a correctness one.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..core.events import DEAD_STATES, MemoryEvent, MemoryRecord
from ..embeddings.similarity import cosine


class InMemoryRecordStore:
    """Dict of records with a linear cosine scan for search.

    Fine to a few thousand records. Past that, swap in a vector database; the
    RecordStore protocol maps one-to-one onto Chroma, Qdrant and pgvector
    clients, so it is a constructor change.
    """

    def __init__(self) -> None:
        self._records: Dict[str, MemoryRecord] = {}

    def upsert(self, record: MemoryRecord) -> None:
        record.touch()
        self._records[record.memory_id] = record

    def get(self, memory_id: str, *, count_access: bool = True) -> Optional[MemoryRecord]:
        """Fetch one record.

        ``count_access=False`` is for the runtime's own internal reads (the
        conflict check), which must not inflate the access statistics that
        lifecycle policies then make decisions from.
        """
        record = self._records.get(memory_id)
        if record is not None and count_access:
            record.access_count += 1
        return record

    def delete(self, memory_id: str) -> None:
        self._records.pop(memory_id, None)

    def all(self, *, include_dead: bool = False) -> List[MemoryRecord]:
        records = list(self._records.values())
        if include_dead:
            return records
        return [r for r in records if r.is_live]

    def search(
        self,
        query_embedding: List[float],
        k: int = 5,
        attribute_filters: Optional[Dict[str, Any]] = None,
        min_score: float = 0.0,
    ) -> List[Tuple[MemoryRecord, float]]:
        """Rank live records by similarity after exact-match filtering.

        Archived and expired records are excluded so retired knowledge cannot
        resurface in a read.
        """
        filters = attribute_filters or {}
        scored: List[Tuple[MemoryRecord, float]] = []
        for record in self._records.values():
            if record.lifecycle_state in DEAD_STATES:
                continue
            if any(record.attributes.get(key) != val for key, val in filters.items()):
                continue
            if record.embedding is None:
                continue
            score = cosine(query_embedding, record.embedding)
            if score >= min_score:
                scored.append((record, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        for record, _ in scored[:k]:
            record.access_count += 1
        return scored[:k]

    def clear(self) -> None:
        self._records.clear()

    def __len__(self) -> int:
        return len(self._records)


class InMemoryEventStore:
    """Append-only list behind a lock.

    Returns a monotonic sequence number as the entry id, mirroring a Redis
    Stream entry id or a Kafka offset.
    """

    def __init__(self) -> None:
        self._events: List[MemoryEvent] = []
        self._lock = asyncio.Lock()

    async def append(self, event: MemoryEvent) -> str:
        async with self._lock:
            self._events.append(event)
            return str(len(self._events) - 1)

    async def replay(self) -> Sequence[MemoryEvent]:
        async with self._lock:
            return list(self._events)

    async def since(self, entry_id: str) -> Sequence[MemoryEvent]:
        async with self._lock:
            return list(self._events[int(entry_id) + 1 :])

    async def length(self) -> int:
        return len(self._events)

    def __len__(self) -> int:
        return len(self._events)

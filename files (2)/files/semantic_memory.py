"""
semantic_memory.py

The read model. This holds the current state of every live memory keyed by
memory_id, along with its embedding, so agents can look up what is known right
now without touching the event log. It is a projection, meaning it is derived
from the log and can always be rebuilt from it.

The prototype keeps records in a dict and does a linear cosine scan for search.
That is fine up to a few thousand records. For production, back this with
ChromaDB, Qdrant, or pgvector and let the vector index do the scan. The public
methods here map one to one onto those clients.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from embeddings import cosine
from events import LifecycleState, MemoryRecord


class SemanticMemory:
    def __init__(self) -> None:
        self._records: Dict[str, MemoryRecord] = {}

    def upsert(self, record: MemoryRecord) -> None:
        record.touch()
        self._records[record.memory_id] = record

    def get(self, memory_id: str) -> Optional[MemoryRecord]:
        record = self._records.get(memory_id)
        if record is not None:
            record.access_count += 1
        return record

    def delete(self, memory_id: str) -> None:
        self._records.pop(memory_id, None)

    def all(self) -> List[MemoryRecord]:
        return list(self._records.values())

    def search(
        self,
        query_embedding: List[float],
        k: int = 5,
        attribute_filters: Optional[Dict[str, Any]] = None,
        min_score: float = 0.0,
    ) -> List[Tuple[MemoryRecord, float]]:
        """
        Rank live records by similarity to a query vector, after applying any
        exact match filters. Expired and archived records are skipped so stale
        knowledge never resurfaces in a read.
        """
        filters = attribute_filters or {}
        scored: List[Tuple[MemoryRecord, float]] = []
        for record in self._records.values():
            if record.lifecycle_state in (LifecycleState.ARCHIVED, LifecycleState.EXPIRED):
                continue
            if not _matches(record.attributes, filters):
                continue
            if record.embedding is None:
                continue
            score = cosine(query_embedding, record.embedding)
            if score >= min_score:
                scored.append((record, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]


def _matches(attributes: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    for key, value in filters.items():
        if attributes.get(key) != value:
            return False
    return True

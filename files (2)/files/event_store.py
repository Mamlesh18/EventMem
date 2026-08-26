"""
event_store.py

The append only log. This is the source of truth for the whole runtime. Nothing
is ever mutated here. New facts arrive as new events. If the semantic projection
is ever lost, it is rebuilt by replaying this log from the start.

The in memory version below is for the prototype. The Redis and Postgres seams
are described in the docstrings so the swap is obvious.
"""

from __future__ import annotations

import asyncio
from typing import List, Protocol

from events import MemoryEvent


class EventStore(Protocol):
    async def append(self, event: MemoryEvent) -> int:
        ...

    async def since(self, index: int) -> List[MemoryEvent]:
        ...

    async def all(self) -> List[MemoryEvent]:
        ...


class InMemoryEventStore:
    """
    A list guarded by a lock. Returns the sequence number of each appended
    event so callers can track their read position, the same way a Redis Stream
    entry id or a Kafka offset works.

    Redis Streams mapping:
        append  -> XADD stream *
        since   -> XREAD from a stored id
        all     -> XRANGE - +

    Postgres mapping:
        a single events table with a bigserial primary key, insert only.
    """

    def __init__(self) -> None:
        self._events: List[MemoryEvent] = []
        self._lock = asyncio.Lock()

    async def append(self, event: MemoryEvent) -> int:
        async with self._lock:
            self._events.append(event)
            return len(self._events) - 1

    async def since(self, index: int) -> List[MemoryEvent]:
        async with self._lock:
            return list(self._events[index + 1 :])

    async def all(self) -> List[MemoryEvent]:
        async with self._lock:
            return list(self._events)

    def __len__(self) -> int:
        return len(self._events)

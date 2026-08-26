"""
bus.py

The transport that carries an event from the runtime to a subscribed agent.
Every agent reads through an Inbox. An Inbox has one method, get, which blocks
until an event arrives or a short timeout passes. The timeout exists only so an
agent can shut down cleanly. It is not polling. The read sleeps until something
is pushed, then returns at once.

This file holds the in memory transport. The Redis transport in redis_bus.py
implements the same two interfaces, so the runtime and the agents do not know
or care which one is in use.
"""

from __future__ import annotations

import asyncio
from typing import Dict, Optional, Protocol

from events import MemoryEvent


class Inbox(Protocol):
    async def get(self, timeout: float = 1.0) -> Optional[MemoryEvent]:
        ...


class EventBus(Protocol):
    def register_agent(self, agent_id: str) -> Inbox:
        ...

    async def deliver(self, agent_id: str, event: MemoryEvent) -> None:
        ...


class _InMemoryInbox:
    def __init__(self) -> None:
        self._queue: asyncio.Queue = asyncio.Queue()

    async def setup(self) -> None:
        return None

    async def put(self, event: MemoryEvent) -> None:
        await self._queue.put(event)

    async def get(self, timeout: float = 1.0) -> Optional[MemoryEvent]:
        try:
            return await asyncio.wait_for(self._queue.get(), timeout)
        except asyncio.TimeoutError:
            return None


class InMemoryEventBus:
    def __init__(self) -> None:
        self._inboxes: Dict[str, _InMemoryInbox] = {}

    def register_agent(self, agent_id: str) -> _InMemoryInbox:
        inbox = _InMemoryInbox()
        self._inboxes[agent_id] = inbox
        return inbox

    async def deliver(self, agent_id: str, event: MemoryEvent) -> None:
        inbox = self._inboxes.get(agent_id)
        if inbox is not None:
            await inbox.put(event)

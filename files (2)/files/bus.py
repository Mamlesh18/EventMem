"""
bus.py

The transport that carries an event from the runtime to a subscribed agent.
Each agent has an inbox. Delivery is a put into that inbox. Agents await their
inbox and process events as they arrive, so nobody polls.

The in memory bus below uses one asyncio.Queue per agent. The interface is kept
narrow so a Redis backed bus is a drop in replacement.

    publish to a subscriber   -> XADD to a per agent stream, or PUBLISH to a
                                 per agent channel
    inbox await               -> XREADGROUP BLOCK, or SUBSCRIBE
"""

from __future__ import annotations

import asyncio
from typing import Dict

from events import MemoryEvent


class InMemoryEventBus:
    def __init__(self) -> None:
        self._inboxes: Dict[str, asyncio.Queue] = {}

    def register_agent(self, agent_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self._inboxes[agent_id] = queue
        return queue

    def inbox(self, agent_id: str) -> asyncio.Queue:
        return self._inboxes[agent_id]

    async def deliver(self, agent_id: str, event: MemoryEvent) -> None:
        queue = self._inboxes.get(agent_id)
        if queue is not None:
            await queue.put(event)

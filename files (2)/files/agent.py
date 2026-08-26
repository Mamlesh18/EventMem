"""
agent.py

A thin base class so the demo can focus on behaviour. An agent does three
things. It publishes memory when it learns something. It reacts to events the
runtime pushes to it. It can pull from shared memory when it needs to.

The key point is the react loop. The agent awaits its inbox. It never asks the
store "is there anything new". The runtime decides what reaches it, and the
agent simply responds.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, Dict, List, Optional

from events import EventType, MemoryEvent, Provenance
from runtime import EventMemRuntime


class Agent:
    def __init__(self, agent_id: str, runtime: EventMemRuntime) -> None:
        self.id = agent_id
        self.runtime = runtime
        self.inbox: asyncio.Queue = runtime.register_agent(agent_id)
        self.received: List[MemoryEvent] = []
        self._running = False
        self._task: Optional[asyncio.Task] = None

    # -- writing -----------------------------------------------------------

    async def remember(
        self,
        content: str,
        event_type: EventType = EventType.MEMORY_CREATED,
        memory_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
        confidence: float = 1.0,
        reason: str = "",
        base_version: Optional[int] = None,
    ) -> MemoryEvent:
        event = MemoryEvent(
            event_type=event_type,
            source_agent=self.id,
            memory_id=memory_id or str(uuid.uuid4()),
            payload={"content": content},
            attributes=attributes or {},
            provenance=Provenance(source_agent=self.id, reason=reason, confidence=confidence),
            base_version=base_version,
        )
        return await self.runtime.publish(event)

    # -- reading -----------------------------------------------------------

    def recall(self, query: str, k: int = 3, **filters):
        return self.runtime.recall(query, k=k, **filters)

    # -- reacting ----------------------------------------------------------

    async def on_event(self, event: MemoryEvent) -> None:
        """
        Override in a subclass to give the agent behaviour. The base version
        just records what arrived, which is enough for the demo to prove that
        the right agents were reached without polling.
        """
        self.received.append(event)

    async def _loop(self) -> None:
        while self._running:
            event = await self.inbox.get()
            if event is None:  # shutdown sentinel
                break
            self.runtime.record_delivery_latency(event)
            await self.on_event(event)

    def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._running = False
        await self.inbox.put(None)
        if self._task is not None:
            await self._task

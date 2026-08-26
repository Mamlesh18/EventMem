"""
agent.py

Agents come in two kinds here. The base Agent reacts to pushed events and can
read and write memory. The LLMAgent adds a model behind the reaction, so when an
event arrives the agent actually reasons about it and can publish what it
concludes as a new memory. That published memory is itself pushed onward, which
is how one agent's finding becomes the next agent's input without any of them
polling a database.

The react loop reads from an inbox that blocks until something is pushed. The
timeout on the read is only there so the agent can notice a shutdown. It is not
polling.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, Dict, List, Optional

from events import EventType, MemoryEvent, Provenance
from llm import LLM
from runtime import EventMemRuntime


class Agent:
    def __init__(self, agent_id: str, runtime: EventMemRuntime) -> None:
        self.id = agent_id
        self.runtime = runtime
        self.inbox = runtime.register_agent(agent_id)
        self.received: List[MemoryEvent] = []
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def remember(
        self,
        content: str,
        event_type: EventType = EventType.MEMORY_CREATED,
        memory_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
        confidence: float = 1.0,
        reason: str = "",
        base_version: Optional[int] = None,
        depth: int = 0,
        caused_by: Optional[str] = None,
    ) -> MemoryEvent:
        event = MemoryEvent(
            event_type=event_type,
            source_agent=self.id,
            memory_id=memory_id or str(uuid.uuid4()),
            payload={"content": content},
            attributes=attributes or {},
            provenance=Provenance(source_agent=self.id, reason=reason, confidence=confidence),
            base_version=base_version,
            depth=depth,
            caused_by=caused_by,
        )
        return await self.runtime.publish(event)

    async def recall(self, query: str, k: int = 3, **filters):
        return await self.runtime.recall(query, k=k, **filters)

    async def on_event(self, event: MemoryEvent) -> None:
        self.received.append(event)

    async def _loop(self) -> None:
        while self._running:
            event = await self.inbox.get(timeout=0.5)
            if event is None:
                continue
            self.runtime.record_delivery_latency(event)
            await self.on_event(event)

    def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        # Prepare the inbox before entering the read loop. For Redis this
        # creates the consumer group so no early event is missed.
        await self.inbox.setup()
        await self._loop()

    async def ready(self) -> None:
        """Await this before publishing so every inbox is set up first."""
        await self.inbox.setup()

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            await self._task


class LLMAgent(Agent):
    """
    An agent with a model behind it. On each pushed event it builds a prompt,
    asks the model, prints the reasoning, and optionally publishes the answer as
    a new memory that flows on to whoever subscribed to it.
    """

    def __init__(
        self,
        agent_id: str,
        runtime: EventMemRuntime,
        llm: LLM,
        system_prompt: str,
        publishes_as: Optional[EventType] = None,
        publish_attributes: Optional[Dict[str, Any]] = None,
        publish_confidence: float = 0.9,
        max_tokens: int = 200,
        temperature: float = 0.3,
    ) -> None:
        super().__init__(agent_id, runtime)
        self.llm = llm
        self.system_prompt = system_prompt
        self.publishes_as = publishes_as
        self.publish_attributes = publish_attributes or {}
        self.publish_confidence = publish_confidence
        self.max_tokens = max_tokens
        self.temperature = temperature

    async def on_event(self, event: MemoryEvent) -> None:
        self.received.append(event)
        incoming = event.payload.get("content", "")
        prompt = f"An event of type {event.event_type.value} arrived from {event.source_agent}.\n" \
                 f"Content: {incoming}\n\nRespond with your part of the work."

        self.runtime.metrics.llm_calls += 1
        answer = await self.llm.chat(
            self.system_prompt, prompt,
            max_tokens=self.max_tokens, temperature=self.temperature,
        )
        print(f"    THINK   [{self.id}] {answer}")

        if self.publishes_as is not None:
            await self.remember(
                content=answer,
                event_type=self.publishes_as,
                attributes=dict(self.publish_attributes),
                confidence=self.publish_confidence,
                reason=f"produced by {self.id}",
                depth=event.depth + 1,
                caused_by=event.event_id,
            )

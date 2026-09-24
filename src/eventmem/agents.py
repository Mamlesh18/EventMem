"""Agents: the things that publish and react.

``Agent`` is the base: it owns an inbox, reacts to pushed events, and can read
and write shared memory. ``LLMAgent`` puts a model behind the reaction, so one
agent's conclusion becomes the next agent's input with nobody polling.

The react loop blocks on the inbox. The timeout on that read exists so the
agent can observe a shutdown request; it is not a poll interval.
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from typing import Any, Dict, List, Optional

from .core.events import EventType, MemoryEvent, MemoryRecord, Provenance


class Agent:
    """Reacts to pushed events. Subclass and override ``handle``."""

    def __init__(self, agent_id: str, runtime: Any) -> None:
        self.id = agent_id
        self.runtime = runtime
        self.inbox = runtime.register_agent(agent_id)
        self.received: List[MemoryEvent] = []
        #: Version of each memory this agent currently believes is current.
        #: Freshness is measured against this, so it must only advance when the
        #: agent actually learns something.
        self.held_versions: Dict[str, int] = {}
        self.errors: List[Exception] = []
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._idle = asyncio.Event()
        self._idle.set()

    # ---------------------------------------------------------------- writing

    async def remember(
        self,
        content: str,
        event_type: EventType = EventType.MEMORY_CREATED,
        memory_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
        confidence: float = 1.0,
        reason: str = "",
        source: str = "",
        base_version: Optional[int] = None,
        depth: int = 0,
        caused_by: Optional[str] = None,
        payload_extra: Optional[Dict[str, Any]] = None,
    ) -> MemoryEvent:
        payload: Dict[str, Any] = {"content": content}
        if payload_extra:
            payload.update(payload_extra)
        event = MemoryEvent(
            event_type=event_type,
            source_agent=self.id,
            memory_id=memory_id or str(uuid.uuid4()),
            payload=payload,
            attributes=attributes or {},
            provenance=Provenance(
                source_agent=self.id, reason=reason, source=source,
                confidence=confidence,
            ),
            base_version=base_version,
            depth=depth,
            caused_by=caused_by,
        )
        published = await self.runtime.publish(event)
        # The author knows what it wrote without waiting for a delivery it will
        # never receive (subscriptions exclude self by default).
        if published.committed_version is not None:
            self.held_versions[published.memory_id] = published.committed_version
        return published

    async def update(
        self, memory_id: str, content: str, *, optimistic: bool = True, **kwargs: Any
    ) -> MemoryEvent:
        """Update a memory, declaring the version this agent last saw.

        ``optimistic=True`` is what turns on conflict detection. Without a
        declared base version the write silently overwrites whatever is there,
        which is occasionally what you want and usually a bug.
        """
        base = self.held_versions.get(memory_id) if optimistic else None
        kwargs.setdefault("base_version", base)
        return await self.remember(
            content, event_type=EventType.MEMORY_UPDATED, memory_id=memory_id, **kwargs
        )

    # ---------------------------------------------------------------- reading

    async def recall(self, query: str, k: int = 3, **filters: Any) -> List[MemoryRecord]:
        records = await self.runtime.recall(query, k=k, **filters)
        for record in records:
            self._observe(record.memory_id, record.version)
        return records

    def read(self, memory_id: str) -> Optional[MemoryRecord]:
        record = self.runtime.get(memory_id)
        if record is not None:
            self._observe(memory_id, record.version)
        return record

    def _observe(self, memory_id: str, version: int) -> None:
        """Advance this agent's view, never backwards.

        Out-of-order delivery is possible on a real transport, and letting a
        late-arriving older version overwrite a newer one would manufacture
        staleness that never happened.
        """
        if version > self.held_versions.get(memory_id, 0):
            self.held_versions[memory_id] = version

    # --------------------------------------------------------------- reacting

    async def handle(self, event: MemoryEvent) -> None:
        """React to one pushed event. Override this."""
        return None

    async def _dispatch(self, event: MemoryEvent) -> None:
        self.received.append(event)
        self.runtime.note_pickup(event, self.id)
        if event.committed_version is not None:
            self._observe(event.memory_id, event.committed_version)
        await self.handle(event)

    async def _loop(self) -> None:
        while self._running:
            event = await self.inbox.get(timeout=0.25)
            if event is None:
                continue
            self._idle.clear()
            try:
                await self._dispatch(event)
                # Ack only after handling: at-least-once, so a crash mid-handle
                # redelivers rather than silently dropping the work.
                await self.inbox.ack(event)
            except Exception as exc:
                # One bad handler must not kill the agent and strand every
                # later event in its inbox.
                self.errors.append(exc)
                nack = getattr(self.inbox, "nack", None)
                if nack is not None:
                    await nack(event)
            finally:
                self._idle.set()

    async def start(self) -> None:
        """Set the inbox up, then begin reacting.

        Setup completes before this returns, so a caller that awaits start()
        for every agent before publishing cannot lose an early event.
        """
        await self.inbox.setup()
        self._running = True
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        """Stop reacting.

        Cancels the loop rather than waiting out the inbox read timeout:
        waiting would add that timeout to every agent's shutdown and show up in
        benchmark wall times as if it were the architecture's cost. Cancelling
        mid-handle is safe because the event is only acked after handling, so
        an at-least-once transport redelivers it.
        """
        self._running = False
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        # Cancelled deliberately; the CancelledError it raises on the way out
        # is the expected outcome, not a failure to report.
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def wait_idle(self) -> None:
        await self._idle.wait()


class LLMAgent(Agent):
    """An agent that reasons with a model and republishes its conclusion."""

    def __init__(
        self,
        agent_id: str,
        runtime: Any,
        llm: Any,
        system_prompt: str,
        *,
        publishes_as: Optional[EventType] = None,
        publish_attributes: Optional[Dict[str, Any]] = None,
        publish_confidence: float = 0.9,
        memory_id_for: Optional[Any] = None,
        max_tokens: int = 200,
        temperature: float = 0.3,
        verbose: bool = False,
    ) -> None:
        super().__init__(agent_id, runtime)
        self.llm = llm
        self.system_prompt = system_prompt
        self.publishes_as = publishes_as
        self.publish_attributes = publish_attributes or {}
        self.publish_confidence = publish_confidence
        self.memory_id_for = memory_id_for
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.verbose = verbose
        self.llm_calls = 0

    def build_prompt(self, event: MemoryEvent) -> str:
        """Override to change what the model sees."""
        return (
            f"A {event.event_type.value} event arrived from {event.source_agent}.\n"
            f"Content: {event.content}\n\n"
            "Respond with your part of the work."
        )

    async def handle(self, event: MemoryEvent) -> None:
        self.llm_calls += 1
        answer = await self.llm.chat(
            self.system_prompt,
            self.build_prompt(event),
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        if self.verbose:
            print(f"    THINK   [{self.id}] {answer}")

        if self.publishes_as is None:
            return

        memory_id = self.memory_id_for(event) if self.memory_id_for else None
        await self.remember(
            content=answer,
            event_type=self.publishes_as,
            memory_id=memory_id,
            attributes=dict(self.publish_attributes),
            confidence=self.publish_confidence,
            reason=f"produced by {self.id}",
            # Depth advances so the cascade is bounded by construction.
            depth=event.depth + 1,
            caused_by=event.event_id,
        )

"""In-memory transport.

Same interface as the Redis transport, so the runtime cannot tell them apart.
``get`` blocks on an asyncio queue: it sleeps until something is pushed and
returns immediately, and the timeout exists only so an agent can observe a
shutdown request.
"""

from __future__ import annotations

import asyncio
from typing import Dict, Optional

from ..core.events import MemoryEvent


class InMemoryInbox:
    """Queue-backed inbox with exact in-flight accounting.

    ``outstanding`` counts events delivered but not yet settled. Queue depth
    cannot serve that purpose: ``asyncio.wait_for`` removes the item from the
    queue before the waiting coroutine resumes, so a depth of zero can mean
    "drained" or "handed over but not yet handled", and a test or harness that
    waits on depth will proceed while an event is still in flight.
    """

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self._queue: asyncio.Queue = asyncio.Queue()
        self.outstanding = 0
        self.handled = 0
        self.failed = 0

    async def setup(self) -> None:
        return None

    async def put(self, event: MemoryEvent) -> None:
        self.outstanding += 1
        await self._queue.put(event)

    async def get(self, timeout: float = 1.0) -> Optional[MemoryEvent]:
        try:
            return await asyncio.wait_for(self._queue.get(), timeout)
        except asyncio.TimeoutError:
            return None

    async def ack(self, event: MemoryEvent) -> None:
        # No redelivery to suppress in-process; the count is the point.
        self.outstanding -= 1
        self.handled += 1

    async def nack(self, event: MemoryEvent) -> None:
        """Handling failed. In-process there is nothing to redeliver to, so
        the event is dropped, but it still stops being in flight."""
        self.outstanding -= 1
        self.failed += 1

    def pending(self) -> int:
        return self._queue.qsize()


class InMemoryEventBus:
    name = "memory"

    def __init__(self) -> None:
        self._inboxes: Dict[str, InMemoryInbox] = {}

    def register_agent(self, agent_id: str) -> InMemoryInbox:
        inbox = self._inboxes.get(agent_id)
        if inbox is None:
            inbox = InMemoryInbox(agent_id)
            self._inboxes[agent_id] = inbox
        return inbox

    async def deliver(self, agent_id: str, event: MemoryEvent) -> None:
        inbox = self._inboxes.get(agent_id)
        if inbox is not None:
            await inbox.put(event)

    def outstanding(self) -> int:
        """Events delivered but not yet handled, across every inbox."""
        return sum(i.outstanding for i in self._inboxes.values())

    async def drain(self, timeout: float = 5.0, poll_s: float = 0.001) -> bool:
        """Wait until every delivered event has been settled.

        Returns False on timeout rather than raising, so a caller can decide
        whether an undrained bus is a failure or just a slow handler. This is
        what tests and benchmarks should wait on: a fixed sleep is either a
        race or a waste, and queue depth is not a drain signal.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while self.outstanding() > 0:
            if loop.time() >= deadline:
                return False
            await asyncio.sleep(poll_s)
        # One more yield so a handler that re-published lands its own delivery.
        await asyncio.sleep(0)
        return self.outstanding() == 0

    async def close(self) -> None:
        return None

"""
tracing.py

A tracer lets you watch the push happen. The runtime calls these methods at
each stage of an event's journey. NullTracer does nothing. ConsoleTracer prints
a readable trail so you can see an event get published, routed to a set of
agents, and delivered to each one.
"""

from __future__ import annotations

import time
from typing import List, Protocol

from events import MemoryEvent


class Tracer(Protocol):
    def on_publish(self, event: MemoryEvent) -> None: ...
    def on_route(self, event: MemoryEvent, recipients: List[str]) -> None: ...
    def on_deliver(self, agent_id: str, event: MemoryEvent) -> None: ...
    def on_capped(self, event: MemoryEvent) -> None: ...


class NullTracer:
    def on_publish(self, event: MemoryEvent) -> None: ...
    def on_route(self, event: MemoryEvent, recipients: List[str]) -> None: ...
    def on_deliver(self, agent_id: str, event: MemoryEvent) -> None: ...
    def on_capped(self, event: MemoryEvent) -> None: ...


class ConsoleTracer:
    def __init__(self) -> None:
        self._t0 = time.perf_counter()

    def _stamp(self) -> str:
        return f"{(time.perf_counter() - self._t0) * 1000:7.1f} ms"

    def _short(self, text: str, width: int = 68) -> str:
        text = " ".join(text.split())
        return text if len(text) <= width else text[: width - 1] + "\u2026"

    def on_publish(self, event: MemoryEvent) -> None:
        content = self._short(event.payload.get("content", ""))
        print(f"[{self._stamp()}] PUBLISH  {event.source_agent} -> "
              f"{event.event_type.value} (depth {event.depth})")
        if content:
            print(f"                    \"{content}\"")

    def on_route(self, event: MemoryEvent, recipients: List[str]) -> None:
        if recipients:
            print(f"[{self._stamp()}] ROUTE    matched {len(recipients)}: "
                  f"{', '.join(recipients)}")
        else:
            print(f"[{self._stamp()}] ROUTE    matched nobody")

    def on_deliver(self, agent_id: str, event: MemoryEvent) -> None:
        print(f"[{self._stamp()}] DELIVER  -> {agent_id}")

    def on_capped(self, event: MemoryEvent) -> None:
        print(f"[{self._stamp()}] CAPPED   depth {event.depth} reached the "
              f"ceiling, cascade stops here")

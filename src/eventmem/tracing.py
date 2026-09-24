"""Tracers. Every runtime stage calls one of these hooks."""

from __future__ import annotations

import json
import time
from typing import Any, List

from .core.events import MemoryEvent, MemoryRecord


class NullTracer:
    """Does nothing. The default, so tracing costs a method call and no I/O."""

    def on_publish(self, event: MemoryEvent) -> None: ...
    def on_route(self, event: MemoryEvent, recipients: List[str]) -> None: ...
    def on_deliver(self, agent_id: str, event: MemoryEvent) -> None: ...
    def on_capped(self, event: MemoryEvent) -> None: ...
    def on_conflict(self, event: MemoryEvent, resolution: Any) -> None: ...
    def on_lifecycle(self, record: MemoryRecord, previous: Any) -> None: ...


class ConsoleTracer:
    """Prints a readable trail of an event's journey.

    Timestamps are relative to construction so a trace reads as elapsed time
    rather than as epoch seconds.
    """

    def __init__(self, width: int = 68) -> None:
        self._t0 = time.perf_counter()
        self._width = width

    def _stamp(self) -> str:
        return f"{(time.perf_counter() - self._t0) * 1000:8.1f} ms"

    def _short(self, text: str) -> str:
        text = " ".join((text or "").split())
        return text if len(text) <= self._width else text[: self._width - 1] + "…"

    def on_publish(self, event: MemoryEvent) -> None:
        print(
            f"[{self._stamp()}] PUBLISH  {event.source_agent} -> "
            f"{event.event_type.value} (depth {event.depth}) [{event.memory_id}]"
        )
        content = self._short(event.content)
        if content:
            print(f'{" " * 13}         "{content}"')

    def on_route(self, event: MemoryEvent, recipients: List[str]) -> None:
        who = ", ".join(recipients) if recipients else "nobody"
        print(f"[{self._stamp()}] ROUTE    matched {len(recipients)}: {who}")

    def on_deliver(self, agent_id: str, event: MemoryEvent) -> None:
        print(f"[{self._stamp()}] DELIVER  -> {agent_id}")

    def on_capped(self, event: MemoryEvent) -> None:
        print(
            f"[{self._stamp()}] CAPPED   depth {event.depth} exceeds the ceiling, "
            "cascade stops"
        )

    def on_conflict(self, event: MemoryEvent, resolution: Any) -> None:
        print(f"[{self._stamp()}] CONFLICT {event.memory_id}: {resolution.reason}")

    def on_lifecycle(self, record: MemoryRecord, previous: Any) -> None:
        print(
            f"[{self._stamp()}] LIFECYCLE {record.memory_id}: "
            f"{previous.value} -> {record.lifecycle_state.value}"
        )


class JSONLTracer:
    """Writes one JSON object per line.

    For post-hoc analysis: a run's trace can be loaded into pandas and the
    routing decisions audited without re-running anything.
    """

    def __init__(self, path: str) -> None:
        # Held open for the tracer's lifetime, closed by close(). A context
        # manager cannot express an object that owns a handle across calls.
        self._fh = open(path, "w", encoding="utf-8")  # noqa: SIM115
        self._t0 = time.time()

    def _write(self, kind: str, **fields: Any) -> None:
        self._fh.write(
            json.dumps({"t": time.time() - self._t0, "kind": kind, **fields}) + "\n"
        )

    def on_publish(self, event: MemoryEvent) -> None:
        self._write("publish", event_id=event.event_id, agent=event.source_agent,
                    type=event.event_type.value, memory_id=event.memory_id,
                    depth=event.depth, caused_by=event.caused_by)

    def on_route(self, event: MemoryEvent, recipients: List[str]) -> None:
        self._write("route", event_id=event.event_id, recipients=recipients,
                    fan_out=len(recipients))

    def on_deliver(self, agent_id: str, event: MemoryEvent) -> None:
        self._write("deliver", event_id=event.event_id, agent=agent_id)

    def on_capped(self, event: MemoryEvent) -> None:
        self._write("capped", event_id=event.event_id, depth=event.depth)

    def on_conflict(self, event: MemoryEvent, resolution: Any) -> None:
        self._write("conflict", event_id=event.event_id, memory_id=event.memory_id,
                    reason=resolution.reason, merged=resolution.merged)

    def on_lifecycle(self, record: MemoryRecord, previous: Any) -> None:
        self._write("lifecycle", memory_id=record.memory_id, previous=previous.value,
                    new=record.lifecycle_state.value)

    def close(self) -> None:
        self._fh.close()


class MultiTracer:
    """Fans out to several tracers, e.g. console plus a JSONL file."""

    def __init__(self, *tracers: Any) -> None:
        self._tracers = tracers

    def _fan(self, method: str, *args: Any) -> None:
        for tracer in self._tracers:
            getattr(tracer, method)(*args)

    def on_publish(self, e): self._fan("on_publish", e)
    def on_route(self, e, r): self._fan("on_route", e, r)
    def on_deliver(self, a, e): self._fan("on_deliver", a, e)
    def on_capped(self, e): self._fan("on_capped", e)
    def on_conflict(self, e, r): self._fan("on_conflict", e, r)
    def on_lifecycle(self, rec, p): self._fan("on_lifecycle", rec, p)

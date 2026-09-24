"""Redis Streams transport.

One stream per agent inbox, plus one durable log stream as the source of truth.

Delivery semantics, stated precisely because the distinction matters and is
easy to get wrong: each agent reads its inbox through a consumer group with
``XREADGROUP BLOCK``, which sleeps inside Redis until an entry arrives. The
entry is acknowledged **after** the agent has handled it, which makes delivery
**at-least-once**: a crash mid-handling leaves the entry pending and it is
redelivered on recovery. Acking at read time instead would be at-most-once and
would silently drop work on a crash.

Because it is at-least-once, handlers should be idempotent. ``claim_stale``
recovers entries left pending by a dead consumer.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from ..core.events import MemoryEvent
from .serde import event_from_json, event_to_json

LOG_STREAM = "eventmem:log"
GROUP = "eventmem"


def inbox_stream(agent_id: str) -> str:
    return f"eventmem:inbox:{agent_id}"


class RedisInbox:
    """One agent's consumer-group cursor over its inbox stream."""

    def __init__(self, client: Any, agent_id: str, start_id: str = "$") -> None:
        self.client = client
        self.agent_id = agent_id
        self.stream = inbox_stream(agent_id)
        self.consumer = agent_id
        #: "$" starts at the live tail; "0" replays everything already in the
        #: stream. A never-before-seen agent that should catch up on history
        #: passes "0".
        self.start_id = start_id
        self._ready = False
        self._unacked: dict = {}

    async def setup(self) -> None:
        """Create the consumer group. Idempotent, and must run before the
        first delivery or early entries land before the group exists."""
        if self._ready:
            return
        try:
            await self.client.xgroup_create(
                self.stream, GROUP, id=self.start_id, mkstream=True
            )
        except Exception as exc:
            if "BUSYGROUP" not in str(exc):
                raise
        self._ready = True

    async def get(self, timeout: float = 1.0) -> Optional[MemoryEvent]:
        if not self._ready:
            await self.setup()
        result = await self.client.xreadgroup(
            GROUP, self.consumer, {self.stream: ">"},
            count=1, block=int(timeout * 1000),
        )
        if not result:
            return None
        _stream, entries = result[0]
        if not entries:
            return None
        entry_id, fields = entries[0]
        event = event_from_json(fields["data"])
        # Held, not acked. The ack happens once the agent says it is done.
        self._unacked[event.event_id] = entry_id
        return event

    async def ack(self, event: MemoryEvent) -> None:
        entry_id = self._unacked.pop(event.event_id, None)
        if entry_id is not None:
            await self.client.xack(self.stream, GROUP, entry_id)

    async def nack(self, event: MemoryEvent) -> None:
        """Handling failed: deliberately do nothing.

        Leaving the entry unacknowledged is what makes it recoverable. It stays
        in the group's pending list and ``claim_stale`` hands it to a live
        consumer, which is the whole point of at-least-once.
        """
        self._unacked.pop(event.event_id, None)

    async def claim_stale(self, min_idle_ms: int = 60_000, count: int = 10
                          ) -> List[MemoryEvent]:
        """Take over entries a dead consumer left pending.

        Without this, at-least-once is only a promise: entries owned by a
        crashed consumer stay pending forever and are never reprocessed.
        """
        if not self._ready:
            await self.setup()
        result = await self.client.xautoclaim(
            self.stream, GROUP, self.consumer, min_idle_time=min_idle_ms,
            start_id="0-0", count=count,
        )
        # Redis 6.2 returns (next, entries); 7.0 adds a third deleted-ids element.
        entries = result[1] if len(result) >= 2 else []
        events: List[MemoryEvent] = []
        for entry_id, fields in entries:
            if not fields:
                continue
            event = event_from_json(fields["data"])
            self._unacked[event.event_id] = entry_id
            events.append(event)
        return events

    async def pending(self) -> int:
        info = await self.client.xpending(self.stream, GROUP)
        return info["pending"] if isinstance(info, dict) else (info[0] if info else 0)


class RedisEventBus:
    name = "redis"

    def __init__(self, client: Any, start_id: str = "$") -> None:
        self.client = client
        self.start_id = start_id
        self._inboxes: dict = {}

    def register_agent(self, agent_id: str) -> RedisInbox:
        inbox = self._inboxes.get(agent_id)
        if inbox is None:
            inbox = RedisInbox(self.client, agent_id, self.start_id)
            self._inboxes[agent_id] = inbox
        return inbox

    async def deliver(self, agent_id: str, event: MemoryEvent) -> None:
        await self.client.xadd(inbox_stream(agent_id), {"data": event_to_json(event)})

    async def close(self) -> None:
        await _aclose(self.client)


class RedisEventStore:
    """The durable log. XRANGE over it rebuilds the projection from scratch."""

    def __init__(self, client: Any, stream: str = LOG_STREAM) -> None:
        self.client = client
        self.stream = stream

    async def append(self, event: MemoryEvent) -> str:
        return await self.client.xadd(
            self.stream, {"data": event_to_json(event, include_embedding=True)}
        )

    async def replay(self) -> Sequence[MemoryEvent]:
        entries = await self.client.xrange(self.stream)
        return [event_from_json(fields["data"]) for _id, fields in entries]

    async def since(self, entry_id: str) -> Sequence[MemoryEvent]:
        entries = await self.client.xrange(self.stream, min=f"({entry_id}")
        return [event_from_json(fields["data"]) for _id, fields in entries]

    async def length(self) -> int:
        return await self.client.xlen(self.stream)

    async def trim(self, max_len: int) -> int:
        return await self.client.xtrim(self.stream, maxlen=max_len, approximate=True)


async def connect(url: str = "redis://localhost:6379", **kwargs: Any):
    """Open a Redis connection with string responses so serde can read fields."""
    try:
        import redis.asyncio as aioredis
    except ImportError as exc:  # pragma: no cover - depends on extras
        raise ImportError(
            "the Redis transport needs the 'redis' extra: pip install eventmem[redis]"
        ) from exc
    client = aioredis.from_url(url, decode_responses=True, **kwargs)
    await client.ping()
    return client


async def _aclose(client: Any) -> None:
    # redis-py renamed close() to aclose() in 5.0.1 and deprecated the old name.
    closer = getattr(client, "aclose", None) or getattr(client, "close", None)
    if closer is not None:
        await closer()

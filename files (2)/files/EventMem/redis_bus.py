"""
redis_bus.py

The Redis transport. Same two interfaces as the in memory version, so it drops
straight into the runtime.

Delivery uses one stream per agent, named inbox:{agent_id}. The runtime pushes
with XADD. The agent reads with XREAD BLOCK, which sleeps inside Redis until an
entry arrives, then returns immediately. Each agent tracks the id of the last
entry it read, so it never sees the same event twice and can resume after a
restart.

The event store uses a single stream, eventmem:log, as the append only source
of truth. Because it is a stream, the whole history can be replayed with XRANGE
to rebuild the semantic projection from scratch.
"""

from __future__ import annotations

from typing import List, Optional

# redis 4.x exposes an asyncio submodule as `redis.asyncio`. Some
# environments or older installs provide the standalone `aioredis`
# package. Try to import the canonical location first and fall back to
# the alternate package to avoid "could not be resolved" import errors
# in editors/environments.
# try:
import redis.asyncio as aioredis  # type: ignore
# except Exception:
# import aioredis  # type: ignore

from events import MemoryEvent
from serde import event_from_json, event_to_json

LOG_STREAM = "eventmem:log"


def _inbox_stream(agent_id: str) -> str:
    return f"inbox:{agent_id}"


class RedisInbox:
    """
    Reads an agent's stream through a consumer group. The group is created at
    setup time, before any event is delivered, and it starts at the end of the
    stream. From then on Redis hands this consumer every new entry exactly once
    and remembers the position, so nothing is missed and nothing is seen twice,
    even across a restart. Setup must run before the first delivery, which the
    agent guarantees by awaiting it before it starts publishing.
    """

    GROUP = "eventmem"

    def __init__(self, client: "aioredis.Redis", agent_id: str) -> None:
        self.client = client
        self.stream = _inbox_stream(agent_id)
        self.consumer = agent_id
        self._ready = False

    async def setup(self) -> None:
        try:
            await self.client.xgroup_create(self.stream, self.GROUP, id="$", mkstream=True)
        except Exception as exc:  # group already exists is fine
            if "BUSYGROUP" not in str(exc):
                raise
        self._ready = True

    async def get(self, timeout: float = 1.0) -> Optional[MemoryEvent]:
        if not self._ready:
            await self.setup()
        block_ms = int(timeout * 1000)
        result = await self.client.xreadgroup(
            self.GROUP, self.consumer, {self.stream: ">"}, count=1, block=block_ms
        )
        if not result:
            return None
        _stream_name, entries = result[0]
        if not entries:
            return None
        entry_id, fields = entries[0]
        await self.client.xack(self.stream, self.GROUP, entry_id)
        return event_from_json(fields["data"])


class RedisEventBus:
    def __init__(self, client: "aioredis.Redis") -> None:
        self.client = client

    def register_agent(self, agent_id: str) -> RedisInbox:
        return RedisInbox(self.client, agent_id)

    async def deliver(self, agent_id: str, event: MemoryEvent) -> None:
        await self.client.xadd(_inbox_stream(agent_id), {"data": event_to_json(event)})


class RedisEventStore:
    def __init__(self, client: "aioredis.Redis") -> None:
        self.client = client

    async def append(self, event: MemoryEvent) -> str:
        return await self.client.xadd(
            LOG_STREAM, {"data": event_to_json(event, include_embedding=True)}
        )

    async def all(self) -> List[MemoryEvent]:
        entries = await self.client.xrange(LOG_STREAM)
        return [event_from_json(fields["data"]) for _entry_id, fields in entries]

    async def since(self, entry_id: str) -> List[MemoryEvent]:
        entries = await self.client.xrange(LOG_STREAM, min=f"({entry_id}")
        return [event_from_json(fields["data"]) for _entry_id, fields in entries]

    async def length(self) -> int:
        return await self.client.xlen(LOG_STREAM)


async def connect(url: str) -> "aioredis.Redis":
    """Open a Redis connection with string responses so serde can read fields."""
    client = aioredis.from_url(url, decode_responses=True)
    await client.ping()
    return client

"""The two systems under test, behind one interface.

Fairness rules, which matter more than the code:

1. **Same embedding model on both sides.** Both run sentence-transformers
   ``all-MiniLM-L6-v2``. Without this the benchmark measures embedders, not
   architectures, and the result is meaningless.

2. **Same Redis.** mem0 stores vectors in Redis (RediSearch); EventMem keeps its
   event log and inbox streams in Redis. One server, one network path, no
   in-process advantage for either.

3. **Same corpus, same queries, same ground truth.** From ``corpus.py``.

4. **mem0 gets its fast path.** ``infer=False`` skips mem0's LLM fact-extraction
   step. That is a deliberate concession: mem0's default ``infer=True`` makes an
   LLM call per write to extract and deduplicate facts, which would add hundreds
   of milliseconds and a token bill to every write here. Charging it that would
   inflate EventMem's win on something this benchmark is not about. The cost is
   that mem0's headline feature - LLM-curated memory - is switched off, so these
   numbers say nothing about the quality of what mem0 would have extracted.

What this benchmark does **not** measure, and where mem0 is genuinely ahead:
LLM-based fact extraction, contradiction detection and memory updating, graph
memory, and multi-user memory scoping. EventMem has none of those.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple

from .corpus import AGENTS, Fact

EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384


@dataclass
class Costs:
    """Everything a system spends. Counted, not estimated."""

    embed_calls: int = 0
    #: Must stay 0 for mem0: infer=False means no model call. Counted rather
    #: than assumed, because "we turned the LLM off" is a fairness claim and
    #: fairness claims should be measurable.
    llm_calls: int = 0
    vector_searches: int = 0
    pushes: int = 0
    write_latencies_ms: List[float] = field(default_factory=list)
    search_latencies_ms: List[float] = field(default_factory=list)
    #: Redis command counts, read from INFO commandstats so they are the
    #: server's own accounting rather than our guess at it.
    redis_commands: Dict[str, int] = field(default_factory=dict)
    #: Raw totals at each end of the measurement. Kept so that a counter reset
    #: part-way through is detectable instead of silently producing a small
    #: wrong number: a reset makes every diff negative, the positive-only
    #: filter drops them, and what survives looks like a plausible low count.
    redis_total_before: int = 0
    redis_total_after: int = 0
    #: True when the server's counters moved backwards during the measurement.
    redis_counters_reset: bool = False

    @property
    def redis_total(self) -> int:
        return sum(self.redis_commands.values())


@dataclass
class SearchHit:
    memory_id: str
    text: str
    score: float


class MemorySystem:
    """What both systems must provide."""

    name: str = "unnamed"
    label: str = ""
    kind: str = ""  # "pull" or "push"

    async def setup(self) -> None: ...

    async def write(self, fact: Fact) -> float:
        """Store a fact. Returns write latency in ms."""
        raise NotImplementedError

    async def search(self, agent: str, query: str, k: int) -> Tuple[List[SearchHit], float]:
        """Retrieve for an agent. Returns (hits, latency_ms)."""
        raise NotImplementedError

    def known_to(self, agent: str) -> Set[str]:
        """Fact ids this agent currently holds without being asked to look.

        For a pull system this is whatever past searches happened to surface.
        For a push system it is whatever was delivered. This is the field the
        propagation phase scores.
        """
        raise NotImplementedError

    async def settle(self) -> None: ...

    async def teardown(self) -> None: ...

    @property
    def costs(self) -> Costs:
        raise NotImplementedError


# --------------------------------------------------------------------- shared

_EMBEDDER = None


def shared_embedder():
    """One embedder instance, shared by both systems.

    Loaded once and reused so neither side pays a model-load cost the other
    does not, and so both produce vectors in the same space.
    """
    global _EMBEDDER
    if _EMBEDDER is None:
        import os

        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        from sentence_transformers import SentenceTransformer

        _EMBEDDER = SentenceTransformer(EMBED_MODEL)
    return _EMBEDDER


class CountingEmbedder:
    """Wraps the shared model so embedding calls are attributable per system."""

    name = f"sentence_transformers({EMBED_MODEL})"
    semantic = True
    dim = EMBED_DIM

    def __init__(self, costs: Costs) -> None:
        self._model = shared_embedder()
        self._costs = costs

    def embed(self, text: str) -> List[float]:
        self._costs.embed_calls += 1
        return self._model.encode(text or " ", normalize_embeddings=True).tolist()

    async def embed_async(self, text: str) -> List[float]:
        import asyncio

        return await asyncio.to_thread(self.embed, text)


async def redis_command_stats(client) -> Dict[str, int]:
    """Per-command call counts straight from the server."""
    info = await client.info("commandstats")
    out: Dict[str, int] = {}
    for key, value in info.items():
        name = key.split("cmdstat_")[-1]
        calls = value.get("calls") if isinstance(value, dict) else None
        if calls:
            out[name] = int(calls)
    return out


def diff_stats(before: Dict[str, int], after: Dict[str, int]) -> Dict[str, int]:
    """Per-command deltas, positive only.

    Negative deltas mean the server's counters were reset mid-measurement.
    Callers must check for that with ``counters_were_reset`` -- filtering the
    negatives away on their own turns a broken measurement into a believable
    small one.
    """
    keys = set(before) | set(after)
    return {
        k: after.get(k, 0) - before.get(k, 0)
        for k in sorted(keys)
        if after.get(k, 0) - before.get(k, 0) > 0
    }


def counters_were_reset(before: Dict[str, int], after: Dict[str, int]) -> bool:
    """Whether any command's count went backwards."""
    return any(after.get(k, 0) < v for k, v in before.items())


# ------------------------------------------------------------------ mem0 side

class Mem0System(MemorySystem):
    """mem0 OSS, Redis vector store, retrieved the normal way.

    The passive-store architecture: ``add`` to write, ``search`` to read.
    Nothing is pushed, so an agent learns a fact only at the moment it asks a
    question whose results happen to contain it.
    """

    name = "mem0"
    label = "mem0 (pull: add + search)"
    kind = "pull"

    def __init__(self, redis_url: str, namespace: str, vector_store: str = "redis") -> None:
        self.redis_url = redis_url
        self.namespace = namespace
        self.vector_store = vector_store
        self._costs = Costs()
        self._memory = None
        self._id_map: Dict[str, str] = {}      # mem0 id -> fact id
        self._known: Dict[str, Set[str]] = {a: set() for a in AGENTS}
        self._stats_before: Dict[str, int] = {}
        self._admin = None

    @property
    def costs(self) -> Costs:
        return self._costs

    async def setup(self) -> None:
        from mem0 import Memory

        collection = f"{self.namespace}_mem0"
        if self.vector_store == "redis":
            store = {
                "provider": "redis",
                "config": {
                    "redis_url": self.redis_url,
                    "collection_name": collection,
                    "embedding_model_dims": EMBED_DIM,
                },
            }
        else:
            # faiss persists to disk, so a stale directory from a previous run
            # would be ingested on top of and silently double every count.
            path = pathlib.Path(tempfile.gettempdir()) / f"eventmem_bench_{collection}"
            shutil.rmtree(path, ignore_errors=True)
            store = {
                "provider": "faiss",
                "config": {
                    "collection_name": collection,
                    "embedding_model_dims": EMBED_DIM,
                    "path": str(path),
                },
            }

        config = {
            "embedder": {
                "provider": "huggingface",
                "config": {"model": EMBED_MODEL},
            },
            "vector_store": store,
            # mem0 builds its LLM client eagerly in __init__, even though
            # infer=False means it is never invoked. A placeholder key is
            # therefore required just to construct the object. Nothing is sent
            # anywhere -- and rather than ask you to take that on trust, the
            # client is wrapped below and its call count is reported.
            "llm": {
                "provider": "openai",
                "config": {
                    "api_key": os.getenv("OPENAI_API_KEY", "unused-infer-false"),
                    "model": "gpt-4o-mini",
                },
            },
        }
        self._memory = Memory.from_config(config)

        # Hook the embedder so its calls are counted. mem0 holds it on the
        # Memory instance, so wrapping it there is enough and needs no fork.
        inner = self._memory.embedding_model
        costs = self._costs

        class _Counted:
            def __init__(self, target):
                self._t = target

            def embed(self, text, memory_action=None):
                costs.embed_calls += 1
                try:
                    return self._t.embed(text, memory_action=memory_action)
                except TypeError:
                    return self._t.embed(text)

            def __getattr__(self, item):
                return getattr(self._t, item)

        self._memory.embedding_model = _Counted(inner)

        # Tripwire on the LLM. If mem0 ever reaches for the model despite
        # infer=False, the count makes it visible instead of silently adding a
        # network round-trip to mem0's measured write latency.
        llm = self._memory.llm

        class _CountedLLM:
            def __init__(self, target):
                self._t = target

            def generate_response(self, *a, **kw):
                costs.llm_calls += 1
                return self._t.generate_response(*a, **kw)

            def __getattr__(self, item):
                return getattr(self._t, item)

        self._memory.llm = _CountedLLM(llm)

        if self.vector_store == "redis":
            from eventmem.transport.redis import connect

            self._admin = await connect(self.redis_url)
            self._stats_before = await redis_command_stats(self._admin)

    async def write(self, fact: Fact) -> float:
        started = time.perf_counter()
        result = self._memory.add(
            fact.text,
            user_id=self.namespace,
            metadata={"fact_id": fact.id, "topic": fact.topic, "author": fact.author},
            # The deliberate concession documented at the top of this file.
            infer=False,
        )
        latency = (time.perf_counter() - started) * 1000.0
        self._costs.write_latencies_ms.append(latency)

        for item in (result or {}).get("results", []) or []:
            if item.get("id"):
                self._id_map[item["id"]] = fact.id
        # The author knows what it wrote.
        self._known[fact.author].add(fact.id)
        return latency

    async def search(self, agent: str, query: str, k: int) -> Tuple[List[SearchHit], float]:
        started = time.perf_counter()
        raw = self._memory.search(query, user_id=self.namespace, limit=k)
        latency = (time.perf_counter() - started) * 1000.0
        self._costs.search_latencies_ms.append(latency)
        self._costs.vector_searches += 1

        hits: List[SearchHit] = []
        for item in (raw or {}).get("results", []) or []:
            meta = item.get("metadata") or {}
            fact_id = meta.get("fact_id") or self._id_map.get(item.get("id", ""), "")
            hits.append(SearchHit(
                memory_id=fact_id,
                text=item.get("memory", ""),
                score=float(item.get("score") or 0.0),
            ))
            # Searching is how a pull system learns: whatever came back is now
            # known to this agent.
            if fact_id:
                self._known[agent].add(fact_id)
        return hits, latency

    def known_to(self, agent: str) -> Set[str]:
        return set(self._known.get(agent, ()))

    async def settle(self) -> None:
        return None

    async def teardown(self) -> None:
        if self._admin is not None:
            after = await redis_command_stats(self._admin)
            self._costs.redis_commands = diff_stats(self._stats_before, after)
            self._costs.redis_total_before = sum(self._stats_before.values())
            self._costs.redis_total_after = sum(after.values())
            self._costs.redis_counters_reset = counters_were_reset(
                self._stats_before, after)
            from eventmem.transport.redis import _aclose

            await _aclose(self._admin)


# -------------------------------------------------------------- EventMem side

class EventMemRedisSystem(MemorySystem):
    """EventMem over Redis Streams, with the same embedder and the same Redis.

    The active architecture: a write is routed to the agents whose subscriptions
    match and pushed to their inbox streams. An agent learns without asking.
    ``search`` still exists and is used for the retrieval-quality phase, so the
    comparison is not push-versus-pull on EventMem's terms only.
    """

    name = "eventmem"
    label = "EventMem (push: publish + route)"
    kind = "push"

    def __init__(self, redis_url: str, namespace: str) -> None:
        self.redis_url = redis_url
        self.namespace = namespace
        self._costs = Costs()
        self.runtime = None
        self.agents: Dict[str, Any] = {}
        self._client = None
        self._stats_before: Dict[str, int] = {}
        #: (fact_id, agent) pairs the runtime actually delivered, with timings.
        self.deliveries: List[Dict[str, Any]] = []

    @property
    def costs(self) -> Costs:
        return self._costs

    async def setup(self) -> None:
        from eventmem import Agent, EventMemRuntime, Subscription
        from eventmem.transport.redis import (
            RedisEventBus,
            RedisEventStore,
            connect,
        )

        self._client = await connect(self.redis_url)
        # Deliberately NOT config_resetstat(). Zeroing the server's counters
        # here made every one of mem0's deltas negative, the positive-only
        # filter dropped them, and mem0's Redis cost was reported as a single
        # command instead of several hundred. A diff needs no reset, and
        # resetting a counter another measurement depends on is never safe --
        # especially on a server that is not exclusively ours.
        self._stats_before = await redis_command_stats(self._client)

        bus = RedisEventBus(self._client)
        store = RedisEventStore(self._client, stream=f"{self.namespace}:log")

        self.runtime = EventMemRuntime(
            embedder=CountingEmbedder(self._costs),
            bus=bus,
            event_store=store,
        )

        system = self

        class Recording(Agent):
            """Records what arrived, and when, without asking for it."""

            def __init__(self, agent_id, runtime):
                super().__init__(agent_id, runtime)
                self.learned: Set[str] = set()

            async def handle(self, event):
                fact_id = event.attributes.get("fact_id")
                if not fact_id:
                    return
                self.learned.add(fact_id)
                stamp = event.published_at
                latency_ms = None
                if stamp is not None:
                    from eventmem.core.clock import elapsed

                    seconds, _precise = elapsed(stamp)
                    latency_ms = seconds * 1000.0
                system.deliveries.append({
                    "fact_id": fact_id,
                    "agent": self.id,
                    "latency_ms": latency_ms,
                })

        for agent_id, topics in AGENTS.items():
            agent = Recording(agent_id, self.runtime)
            self.agents[agent_id] = agent
            # One subscription per topic of interest. This is the same interest
            # statement the mem0 side gets; here it routes, there it filters.
            for topic in topics:
                await self.runtime.subscribe(Subscription(
                    subscriber_id=agent_id,
                    attribute_filters={"topic": topic},
                    label=f"{agent_id} <- {topic}",
                ))

        for agent in self.agents.values():
            await agent.start()

    async def write(self, fact: Fact) -> float:
        started = time.perf_counter()
        event = await self.agents[fact.author].remember(
            fact.text,
            memory_id=fact.id,
            attributes={"topic": fact.topic, "fact_id": fact.id, "author": fact.author},
            reason=f"recorded by {fact.author}",
        )
        latency = (time.perf_counter() - started) * 1000.0
        self._costs.write_latencies_ms.append(latency)
        self._costs.pushes = self.runtime.metrics.total_deliveries
        _ = event
        return latency

    async def search(self, agent: str, query: str, k: int) -> Tuple[List[SearchHit], float]:
        started = time.perf_counter()
        scored = await self.runtime.recall_scored(query, k=k)
        latency = (time.perf_counter() - started) * 1000.0
        self._costs.search_latencies_ms.append(latency)
        self._costs.vector_searches += 1

        hits = [
            SearchHit(
                memory_id=record.attributes.get("fact_id", record.memory_id),
                text=record.content,
                score=float(score),
            )
            for record, score in scored
        ]
        for hit in hits:
            if hit.memory_id:
                self.agents[agent].learned.add(hit.memory_id)
        return hits, latency

    def known_to(self, agent: str) -> Set[str]:
        return set(self.agents[agent].learned)

    async def settle(self) -> None:
        # Redis delivery is out-of-process, so wait for the inbox streams to be
        # consumed rather than assuming a fixed pause is enough.
        import asyncio

        for _ in range(400):
            await asyncio.sleep(0.01)
            pending = 0
            for agent in self.agents.values():
                inbox = agent.inbox
                if hasattr(inbox, "pending"):
                    # pending() is best-effort: a consumer group that has
                    # not been created yet simply has nothing outstanding.
                    with contextlib.suppress(Exception):
                        pending += await inbox.pending()
            if pending == 0:
                await asyncio.sleep(0.02)
                return

    async def teardown(self) -> None:
        for agent in self.agents.values():
            await agent.stop()
        if self._client is not None:
            after = await redis_command_stats(self._client)
            self._costs.redis_commands = diff_stats(self._stats_before, after)
            self._costs.redis_total_before = sum(self._stats_before.values())
            self._costs.redis_total_after = sum(after.values())
            self._costs.redis_counters_reset = counters_were_reset(
                self._stats_before, after)
        if self.runtime is not None:
            await self.runtime.close()


class EventMemMemorySystem(EventMemRedisSystem):
    """EventMem with the in-process transport and log.

    Same runtime, same router, same embedder, same corpus - only the transport
    and the event store differ. That makes it a clean ablation: the gap between
    this and the Redis variant is the price of durability and of being able to
    run agents in separate processes, and nothing else.

    It is not a competitor to mem0 on equal terms, because it keeps nothing
    across a restart. It is in the table to answer a different question: how
    much does Redis cost EventMem?
    """

    name = "eventmem_memory"
    label = "EventMem in-memory (push, no durability)"
    kind = "push"

    async def setup(self) -> None:
        from eventmem import Agent, EventMemRuntime, Subscription
        from eventmem.store.memory import InMemoryEventStore
        from eventmem.transport.memory import InMemoryEventBus

        self.runtime = EventMemRuntime(
            embedder=CountingEmbedder(self._costs),
            bus=InMemoryEventBus(),
            event_store=InMemoryEventStore(),
        )

        system = self

        class Recording(Agent):
            def __init__(self, agent_id, runtime):
                super().__init__(agent_id, runtime)
                self.learned: Set[str] = set()

            async def handle(self, event):
                fact_id = event.attributes.get("fact_id")
                if not fact_id:
                    return
                self.learned.add(fact_id)
                latency_ms = None
                if event.published_at is not None:
                    from eventmem.core.clock import elapsed

                    seconds, _precise = elapsed(event.published_at)
                    latency_ms = seconds * 1000.0
                system.deliveries.append({
                    "fact_id": fact_id, "agent": self.id, "latency_ms": latency_ms,
                })

        for agent_id, topics in AGENTS.items():
            agent = Recording(agent_id, self.runtime)
            self.agents[agent_id] = agent
            for topic in topics:
                await self.runtime.subscribe(Subscription(
                    subscriber_id=agent_id,
                    attribute_filters={"topic": topic},
                    label=f"{agent_id} <- {topic}",
                ))

        for agent in self.agents.values():
            await agent.start()

    async def settle(self) -> None:
        # Exact, not timed: the bus knows how many deliveries are still in
        # flight, so there is nothing to guess at here.
        await self.runtime.bus.drain(timeout=10.0)
        for agent in self.agents.values():
            await agent.wait_idle()

    async def teardown(self) -> None:
        for agent in self.agents.values():
            await agent.stop()
        if self.runtime is not None:
            await self.runtime.close()
        # No Redis involved, so the command count is a true zero rather than
        # an unmeasured blank.
        self._costs.redis_commands = {}


async def reset_namespace(redis_url: str, namespace: str) -> int:
    """Delete this run's keys so a re-run is not polluted by the last one.

    Scoped to the namespace prefix and never FLUSHALL: this runs against a
    Redis the user started themselves, which may hold other data.
    """
    from eventmem.transport.redis import _aclose, connect

    client = await connect(redis_url)
    deleted = 0
    try:
        patterns = [
            f"{namespace}:*",
            "eventmem:inbox:*",
            f"{namespace}_mem0*",
            f"mem0:{namespace}*",
        ]
        for pattern in patterns:
            cursor = 0
            while True:
                cursor, keys = await client.scan(cursor, match=pattern, count=500)
                if keys:
                    deleted += await client.delete(*keys)
                if cursor == 0:
                    break
        # Drop the mem0 RediSearch index if it survived a previous run.
        for index in (f"{namespace}_mem0", f"mem0_{namespace}"):
            # A missing index is the normal case on a first run.
            with contextlib.suppress(Exception):
                await client.execute_command("FT.DROPINDEX", index)
    finally:
        await _aclose(client)
    return deleted


def new_namespace() -> str:
    return f"bench{uuid.uuid4().hex[:8]}"


def preflight(redis_url: str) -> Dict[str, Any]:
    """Check Redis reachability and whether RediSearch is present.

    mem0's Redis vector store needs the RediSearch module, which plain
    ``redis:7-alpine`` does not ship. Detecting that here produces an
    actionable message instead of a stack trace sixty lines deep.
    """
    import redis as sync_redis

    out: Dict[str, Any] = {"reachable": False, "redisearch": False, "modules": []}
    try:
        client = sync_redis.from_url(redis_url, decode_responses=True,
                                     socket_connect_timeout=3)
        client.ping()
        out["reachable"] = True
        try:
            modules = client.execute_command("MODULE", "LIST") or []
            names = []
            for entry in modules:
                if isinstance(entry, dict):
                    names.append(str(entry.get("name", "")).lower())
                elif isinstance(entry, (list, tuple)):
                    pairs = {
                        str(entry[i]).lower(): entry[i + 1]
                        for i in range(0, len(entry) - 1, 2)
                    }
                    names.append(str(pairs.get("name", "")).lower())
            out["modules"] = [n for n in names if n]
            out["redisearch"] = any("search" in n for n in out["modules"])
        except Exception as exc:
            out["module_error"] = str(exc)
        client.close()
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out

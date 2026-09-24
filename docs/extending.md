# Extending EventMem

Every part of the architecture is a constructor argument held by protocol.
This page has a worked example for each seam. All of them are real, runnable
code — none require touching `runtime.py`.

The protocols themselves live in one short file:
[`src/eventmem/core/protocols.py`](../src/eventmem/core/protocols.py). Read it
first; it is the whole contract.

---

## 1. A new routing ideology

The most interesting seam. A `Router` decides who receives an event, and it
owns its own subscription representation — so an alternative ideology can use
a completely different subscription object and still drop in.

### The contract

```python
class Router(Protocol):
    name: str
    async def register(self, subscription: Any) -> None: ...
    def unregister(self, subscriber_id: str) -> None: ...
    def route(self, event: MemoryEvent) -> List[str]: ...
    def subscribers(self) -> List[str]: ...
```

`route` must not mutate the event, and must exclude the author unless the
subscription explicitly opts in — an agent already knows what it wrote, and
self-delivery is the most common cascade loop.

### Worked example: route by agent capability and current load

```python
from eventmem import MemoryEvent

class CapabilityRouter:
    """Route work to whichever capable agent is least busy.

    A predicate router cannot express this: the decision depends on runtime
    state, not on the event alone.
    """

    name = "capability"

    def __init__(self, max_in_flight: int = 3) -> None:
        self._skills: dict[str, set[str]] = {}
        self._load: dict[str, int] = {}
        self.max_in_flight = max_in_flight

    async def register(self, subscription) -> None:
        agent_id = subscription.subscriber_id
        self._skills[agent_id] = set(getattr(subscription, "skills", ()))
        self._load.setdefault(agent_id, 0)

    def unregister(self, subscriber_id: str) -> None:
        self._skills.pop(subscriber_id, None)
        self._load.pop(subscriber_id, None)

    def route(self, event: MemoryEvent) -> list[str]:
        needed = event.attributes.get("requires")
        if not needed:
            return []
        candidates = [
            a for a, skills in self._skills.items()
            if a != event.source_agent
            and needed in skills
            and self._load[a] < self.max_in_flight
        ]
        if not candidates:
            return []
        # Exactly one recipient: this is work assignment, not notification.
        chosen = min(candidates, key=lambda a: self._load[a])
        self._load[chosen] += 1
        return [chosen]

    def completed(self, agent_id: str) -> None:
        self._load[agent_id] = max(0, self._load[agent_id] - 1)


runtime = EventMemRuntime(router=CapabilityRouter())
```

If your router needs embeddings, give it a `bind_embedder(embedder)` method;
the runtime calls it at construction so your interest vectors are produced by
the same model that embeds events. Mismatched models make cosine similarity
compare vectors from different spaces, and the numbers become meaningless.

### Then benchmark it

A new ideology is a claim. Test it:

```python
from benchmarks.harness import compare
from benchmarks.systems import BroadcastSystem, EventMemSystem
from benchmarks.workloads import NoisyBroadcastWorkload

comparison = await compare(
    NoisyBroadcastWorkload(),
    [
        EventMemSystem(router=CapabilityRouter),   # yours
        EventMemSystem(),                          # the default
        BroadcastSystem(),                         # the null hypothesis
    ],
)
```

`EventMemSystem` takes a router factory, so your ideology joins the comparison
table without a new system class.

---

## 2. A new conflict policy

```python
class ConflictPolicy(Protocol):
    name: str
    def resolve(self, current: MemoryRecord, incoming: MemoryRecord) -> Resolution: ...
```

### Worked example: numeric maximum

```python
from eventmem import Resolution

class KeepHighest:
    """For a memory holding a running maximum, such as a best score or a
    high-water mark. Commutative and associative, so replicas converge."""

    name = "keep_highest"

    def resolve(self, current, incoming):
        try:
            if float(incoming.content) > float(current.content):
                return Resolution(incoming, current, "incoming is larger")
            return Resolution(current, incoming, "current is larger")
        except ValueError:
            # Non-numeric content: fall back rather than guess.
            return Resolution(current, incoming, "unparseable, holding current")
```

**Determinism matters.** `rebuild_projection()` replays the log and re-runs
your policy on the same collisions. A policy that depends on wall-clock time or
iteration order will reconstruct a different state than the live run produced.
This is why `ConfidenceThenRecency` breaks its final tie on `event_id` rather
than leaving it to chance.

Set `Resolution.merged=True` when the winner is built from both sides rather
than being one of them, so the merge shows up in the metrics.

---

## 3. A new embedder

```python
class Embedder(Protocol):
    dim: int
    def embed(self, text: str) -> List[float]: ...
```

Add `async def embed_async(self, text)` if your backend blocks — the runtime
prefers it, and without it a network call freezes delivery for every agent.

```python
class MyEmbedder:
    name = "my_model"
    #: Declare honestly. If False, the runtime warns when a semantic
    #: subscription is registered against you, because layer-3 matching will
    #: silently degrade to token overlap.
    semantic = True
    dim = 768

    def embed(self, text: str) -> list[float]:
        return my_model.encode(text or " ").tolist()

    async def embed_async(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed, text)
```

Wrap anything hosted in `CachingEmbedder` — benchmarks re-embed the same text
constantly.

---

## 4. A new transport

Two protocols, `EventBus` and `Inbox`. The critical detail is acknowledgement
ordering:

```python
async def ack(self, event) -> None:
    """Called AFTER the agent handled it. At-least-once."""

async def nack(self, event) -> None:
    """Handling failed. Leave the entry recoverable, do not acknowledge."""
```

Acking at read time gives at-most-once and silently loses work on a crash.
`get` must *block* until an event arrives — sleep inside the transport, do not
wake repeatedly to check. The timeout argument exists so an agent can notice a
shutdown request, not as a poll interval.

If your transport delivers in-process, implement `outstanding()` and `drain()`
as the in-memory bus does; tests and benchmarks use them to wait for exact
quiescence instead of guessing with a sleep.

---

## 5. A new record store

The `RecordStore` protocol maps one-to-one onto Chroma, Qdrant and pgvector
clients. The in-memory implementation does a linear cosine scan, which is fine
to a few thousand records.

```python
class QdrantRecordStore:
    def upsert(self, record): ...
    def get(self, memory_id, *, count_access=True): ...
    def delete(self, memory_id): ...
    def all(self, *, include_dead=False): ...
    def search(self, query_embedding, k=5, attribute_filters=None, min_score=0.0): ...
    def clear(self): ...

runtime = EventMemRuntime(record_store=QdrantRecordStore(client))
```

`count_access=False` exists because the runtime reads the store during its own
conflict check, and that read must not inflate the access statistics lifecycle
policies then act on. Honour it.

---

## 6. A new lifecycle policy

```python
class TopicTTL:
    """Different knowledge goes stale at different rates. A support ticket is
    worthless in a day; a schema decision is not."""

    name = "topic_ttl"

    TTLS = {"ticket": 3600.0, "schema": 30 * 86400.0}
    DEFAULT = 86400.0

    def next_state(self, record, now):
        ttl = self.TTLS.get(record.attributes.get("kind"), self.DEFAULT)
        if now - record.updated_at >= ttl:
            return LifecycleState.EXPIRED
        return None
```

Transitions are validated against `LEGAL_TRANSITIONS` before being applied: the
lifecycle is a DAG and never runs backwards. A policy proposing an illegal move
is skipped rather than allowed to corrupt the record.

Retiring a record **publishes a `lifecycle_changed` event**, so agents holding a
copy learn it retired. That is not optional bookkeeping — silently archiving a
record three agents believe is live is exactly the staleness the runtime
exists to prevent.

---

## 7. A new tracer

Six hooks, all called synchronously on the publish path, so keep them cheap.
`JSONLTracer` writes one object per line for post-hoc analysis; `MultiTracer`
fans out to several.

```python
runtime = EventMemRuntime(
    tracer=MultiTracer(ConsoleTracer(), JSONLTracer("run.jsonl"))
)
```

---

## Checklist for a contribution

- [ ] It implements the protocol and nothing in `runtime.py` changed.
- [ ] It has a docstring saying **when to use it and when not to**.
- [ ] It has a test.
- [ ] If it is a router or a policy, it is in the benchmark comparison.
- [ ] If it makes EventMem look better, you added the case where it looks worse.

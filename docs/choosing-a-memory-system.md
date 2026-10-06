# Choosing a memory system: mem0 vs EventMem (Redis / in-memory)

A decision guide backed by measurements, not preference. Three systems, one
corpus, one embedding model, one Redis.

Everything here was measured on this machine; **[Reproducing](#reproducing)**
has the commands. Raw data: [`mem0-results.json`](mem0-results.json).

---

## TL;DR — pick by what you actually need

| Your situation | Use | Why |
|---|---|---|
| One agent, or agents that never need each other's findings | **mem0** | You need a store, not a coordinator. mem0 does more per memory and you pay nothing for routing you won't use. |
| You want memories **extracted and curated by an LLM** from conversation | **mem0** | EventMem has no equivalent. It stores what you give it. |
| You need **contradiction detection / memory updating** | **mem0** | Its `ADD`/`UPDATE`/`DELETE` reconciliation has no EventMem counterpart. |
| You need **graph memory** (entities, relationships) | **mem0** | Not implemented in EventMem at all. |
| Agent B must **react** to what agent A just learned | **EventMem** | mem0 cannot notify. B only learns when it happens to search. |
| Several agents would otherwise **repeat the same tool call** | **EventMem** | A result pushed once is a result nobody pays for twice. |
| Multi-process / multi-container agents, or you need the log to survive a restart | **EventMem + Redis** | Redis Streams carry events between processes and the log is durable. |
| Single process, latency-sensitive, state is rebuildable | **EventMem in-memory** | 10-15x lower push latency and zero Redis ops for identical correctness. |
| You need both curation *and* reactivity | **Both** | They are not mutually exclusive. mem0 as the curated store, EventMem as the coordination layer. |

**The honest summary:** mem0 is a better *memory*. EventMem is a better
*coordinator*. On this corpus retrieval quality **ties**, so the choice is not
about retrieval — it is about whether anything needs to react.

---

## The three systems

| | mem0 | EventMem + Redis | EventMem in-memory |
|---|---|---|---|
| Model | passive store (`add` / `search`) | active push (`publish` → route → deliver) | same, in-process |
| Write path | Redis hash + RediSearch index | `XADD` to log + per-agent inbox streams | list append + asyncio queues |
| Read path | `FT.SEARCH` over Redis | in-process cosine scan | in-process cosine scan |
| Survives restart | **yes** (vectors in Redis) | **log yes**, projection rebuilt by replay | **no** |
| Cross-process agents | n/a (it is a library call) | **yes** | no |
| Can notify an agent | **no** | yes | yes |
| LLM curation of memories | **yes** (off in this benchmark) | no | no |

### One thing that surprises people

**EventMem's searchable projection is in-process in *both* variants.** Redis
carries the event log and the inbox streams; it does not serve `recall()`. That
is why the two EventMem rows have nearly identical search latency below, and it
is the reason EventMem's reads do not scale the way a vector database does —
`InMemoryRecordStore` does a linear cosine scan. Swap in Chroma/Qdrant/pgvector
via the `RecordStore` protocol if your corpus outgrows that.

---

## Benchmark results

Two full runs each. `a / b` = run 1 / run 2. Same Redis
(`redis/redis-stack-server`), same embedder (`all-MiniLM-L6-v2`, one shared
instance), 32 facts, 12 hand-labelled queries, 44 required deliveries.

### Retrieval quality — a three-way tie

| metric | mem0 | EventMem+Redis | EventMem in-mem |
|---|---|---|---|
| recall@5 | 0.833 | **0.847** | **0.847** |
| precision@5 | 0.417 | 0.417 | 0.417 |
| MRR | 1.000 | 1.000 | 1.000 |

Identical across both runs. All three embed with the same model, so this is the
control: **push costs nothing in retrieval quality.** Had EventMem lost here,
its reactivity would have been bought at the price of being a worse memory.

### Latency

| metric | mem0 | EventMem+Redis | EventMem in-mem |
|---|---|---|---|
| search, median | 39.8 / 45.6 ms | **20.1 / 19.8 ms** | **18.8 / 21.3 ms** |
| search, p95 | 46.4 ms | 24.9 ms | 23.6 ms |
| write, median | 61.1 / 67.5 ms | **19.6 / 22.4 ms** | **18.1 / 20.7 ms** |
| write, p95 | 69.3 ms | 38.0 ms | 20.3 ms |
| **push → delivered, mean** | **n/a — cannot push** | 6.0 / 6.9 ms | **0.39 / 0.60 ms** |
| push, p95 | n/a | 8.8 ms | 0.48 ms |
| ingest wall (32 facts) | 2.0 s | 0.76 s | **0.59 s** |

**Read this carefully:**

- mem0 is **~2× slower on search** and **~3× slower on write**. That is not an
  architectural verdict — mem0's path goes through `redisvl` and does more per
  operation (`hgetall` + `hset` per write, `FT.SEARCH` with a tag filter per
  read) than EventMem's in-process scan plus one `XADD`.
- The two EventMem variants are **the same on search and write** (within
  ~2 ms, inside run-to-run noise), because both search in-process and both
  spend most of a write on embedding.
- **The only place Redis actually costs EventMem is push latency:** 6.4 ms
  against 0.5 ms, roughly **10-15x**, or about **+5.9 ms absolute** per delivery.
  That is what durability and cross-process delivery cost. (15x in run 1,
  11x in run 2 - quote the absolute figures, not the multiple.)

### Propagation — the architectural difference

| metric | mem0 | EventMem+Redis | EventMem in-mem |
|---|---|---|---|
| required deliveries | 44 | 44 | 44 |
| reachable by routing (ceiling) | 28 | 28 | 28 |
| satisfied | 19 | **28** | **28** |
| coverage of **required** | 0.432 | **0.636** | **0.636** |
| coverage of **reachable** | 0.571 | **1.000** | **1.000** |
| retrievals spent to get there | 11 | **0** | **0** |
| facts learned without asking | **0** | **40** | **40** |

EventMem delivered **28 of 28 reachable** required deliveries having spent
**zero retrievals**. mem0 reached 19 of 28 and spent 11 searches — and those 11
only happened because the harness *told* it to go looking, one per declared
interest. In a real system nobody issues those searches on a schedule.

**40 facts reached an agent that never asked.** mem0 scores zero by
construction: a passive store has no way to notify. That is a capability
difference, not a performance gap.

### Cost

| metric | mem0 | EventMem+Redis | EventMem in-mem |
|---|---|---|---|
| embedding calls | 87 | **44** | **44** |
| vector searches | 23 | **12** | **12** |
| LLM calls | 0 | 0 | 0 |
| **Redis commands** (server-counted) | **115** | 205 / 208 | **0** |

**mem0 uses fewer Redis commands than EventMem+Redis** — 115 against ~206.
EventMem pays commands to push (74 `XADD`, 60 `XREADGROUP`, 41 `XACK`); mem0
pays them to search. Which is cheaper depends entirely on your read/write
ratio, and on this corpus mem0 wins that column.

mem0 also makes **2 embedding calls per write** (87 for 33 writes + 23
searches) where EventMem makes 1.

---

## The ceiling, stated plainly

**No system here can exceed 0.636 coverage of *required* deliveries on this
corpus.** 16 of the 44 required deliveries name an agent that never declared an
interest in that fact's topic — e.g. `f21` ("the production TLS certificate
expires on the 14th", topic `infra`) is needed by `security`, which subscribes
only to `security` and `api`. No interest-based router can deliver that.

That is the standing cost of selective routing, which is why two denominators
are reported:

- **coverage of required** — of everything an agent needed, how much arrived.
  Bounded by how well the interests were written.
- **coverage of reachable** — of what the declared interests made deliverable,
  how much arrived. Isolates routing and transport. EventMem: **1.000**.

Quoting only the first understates EventMem; quoting only the second hides a
real limitation. Both are in the table.

---

## Decision guide by scenario

### Support-desk triage chain
Sentiment → triage → draft, each stage needing the last one's output.
→ **EventMem**. With mem0 each stage must poll or search, and acts on stale
context in between. Measured: a 50 ms poll on this shape drops freshness to
0.250 (see [`benchmarking.md`](benchmarking.md)).

### Several agents needing the same expensive lookup
→ **EventMem**. Measured ρ_dup = 0.75 against a no-sharing floor: 4 calls
become 1.

### Personal assistant remembering a user's preferences over months
→ **mem0**. You need curation, contradiction handling and durable vectors.
Nothing needs to react within milliseconds.

### RAG pipeline that retrieves once per request
→ **mem0**, or neither. No propagation to measure; EventMem's push layer is
overhead you won't use.

### 12 agents, each event relevant to ~3 of them
→ **EventMem**. Measured on the main suite: 75% fewer deliveries than
broadcast at identical freshness and recall, with zero wasted wakeups against
360.

### Agents in separate containers
→ **EventMem + Redis**. The in-memory transport cannot cross a process
boundary; mem0 has no delivery at all.

### One process, low latency, state rebuildable from source
→ **EventMem in-memory**. Same correctness, 0.4 ms push instead of 6 ms, no
Redis to operate.

### You already run mem0 and want reactivity
→ **Both.** Keep mem0 as the curated store. Put EventMem in front as the
coordination layer and publish an event when a memory lands. They are not
competitors at the same layer.

---

## Evaluation methodology

### Fairness contract

Held constant across all three systems:

| | held constant |
|---|---|
| Embedding model | `all-MiniLM-L6-v2`, **one shared instance** |
| Corpus | identical 32 facts |
| Queries | identical 12 hand-labelled queries |
| Ground truth | identical `relevant` / `needed_by` labels |
| Interest statement | mem0 filters searches by topic; EventMem routes on the same topic |
| Warmup | 1 write + 1 search, discarded, per system |
| Redis | same server for mem0 and EventMem+Redis |

**Without the shared embedder this would measure embedding models, not
architectures.** That is the most common way a memory comparison goes wrong.

### The concession to mem0, and how it is verified

mem0 runs with **`infer=False`**, skipping its LLM fact-extraction. Its default
makes an LLM call per write; charging it a network round-trip and a token bill
on every write would inflate EventMem's advantage on something this benchmark
is not about.

That concession is **verified, not assumed**: mem0's LLM client is wrapped with
a counter, `llm_calls` is reported every run (**0** in all runs above), and a
non-zero count makes the report declare the write latencies invalid.

### Metric definitions

| metric | definition | needs |
|---|---|---|
| **recall@k** | distinct relevant ids in top-k ÷ `min(|relevant|, k)` | labelled queries |
| **precision@k** | distinct relevant ids in top-k ÷ |top-k| | labelled queries |
| **MRR** | mean of 1/rank of the first relevant hit | labelled queries |
| **search / write latency** | wall time around the call; **median** headline | warmup |
| **push latency** | publish → agent handled it | hybrid clock |
| **coverage of required** | satisfied ÷ all required deliveries | `needed_by` ground truth |
| **coverage of reachable** | satisfied ÷ deliveries the interests allow | ground truth + interests |
| **retrievals spent** | searches issued to reach that coverage | counted |
| **learned without asking** | deliveries the agent did not request | counted |
| **Redis commands** | `INFO commandstats` delta | server-side |

Recall's denominator is capped at `k` deliberately: a query with 3 relevant
facts asked at k=2 cannot reach 1.0, and penalising a system for the harness's
own cut-off would measure the wrong thing.

### Phase order, and why it matters

```
ingest → propagation → reactivity → retrieval
```

Propagation and reactivity are scored from what each agent knows **without
having asked**, so they must run before any query phase. The retrieval phase's
searches teach a pull system things, and crediting those to propagation
measures the harness rather than the system.

This ordering is not cosmetic. With the phases the other way round, EventMem
scored **0.727 coverage against an achievable ceiling of 0.636** — a score
above the maximum, which is how the leak was found.

### Measurement decisions that change the numbers

- **Median, not mean, for latency.** The first write pays one-time costs (index
  creation, lazy model load). Including it made mem0's mean write latency
  (218 ms) exceed its own p95 (130 ms) — impossible without an outlier. There
  is now a discarded warmup and the headline is a median.
- **Hybrid clock.** `perf_counter` is sub-microsecond but process-local;
  `time.time()` is comparable across processes but has 15.6 ms granularity on
  Windows. Every stamp carries both plus its origin process, and
  `timing_precise` reports which was used.
- **Duplicate hits deduped before scoring.** mem0 writes a fresh row per
  `add()`, so a re-ingested corpus returns the same memory repeatedly.
  Counting those separately produced recall **1.36**.
- **`NaN`, never `0`, for absent data.** A system that propagated nothing must
  not appear to have the fastest propagation.
- **No `CONFIG RESETSTAT`.** Zeroing the server's counters mid-run made all of
  mem0's deltas negative; the positive-only filter dropped them and mem0's
  Redis cost printed as **1 command** instead of 115.

---

## Core testing

**117 tests total** — 104 that need nothing, plus 13 Redis integration tests
that skip automatically when no server is reachable.

| file | tests | covers |
|---|---|---|
| `test_runtime.py` | 17 | publish pipeline, selective routing, conflict detection and resolution, cascade depth cap, log replay, provenance lineage, handler-failure isolation, idle agents performing zero reads |
| `test_routing.py` | 17 | each predicate layer, layer conjunction, fail-closed semantics, topic wildcards, broadcast/random/predicate/composite routers, threshold calibration, the non-semantic-embedder warning |
| `test_metrics_and_lifecycle.py` | 17 | fan-out efficiency arithmetic, freshness, consistency, routing precision/recall, lifecycle transition legality, TTL expiry, serialisation round-trip, forward compatibility |
| `test_benchmarks.py` | 14 | workload ground-truth validation, the broadcast ablation, the duplicate-call floor, polling freshness, **and the case where EventMem loses to broadcast** |
| `test_mem0_comparison.py` | 17 | corpus ground truth, the recall>1 dedupe bug, recall denominator, MRR, verdict logic including a loss verdict, the SSE envelope collision |
| `test_redis_integration.py` | 13 | **real Redis**: durable log replay into a fresh runtime, consumer-group at-least-once, `XAUTOCLAIM` recovery of an unacked entry, nack-on-failure, attribute/provenance round-trip, embeddings excluded from inbox copies, cross-process stamps flagged imprecise |

The Redis file matters disproportionately: everything else runs on the
in-memory transport, where delivery is a function call and nothing is
serialised. **These are the only tests that can falsify the durability and
delivery-semantics claims.**

### Methods of testing

**Unit tests** — pure logic with no I/O. Metric arithmetic, scoring, lifecycle
transition legality, routing predicates.

**Integration tests** — the real transport against a real Redis, including
failure paths (crashed consumer, exploding handler) that cannot be exercised
in-process.

**Ablation** — `broadcast`, `random` and `eventmem` share one implementation
and differ only in the router. If broadcast beats EventMem on a metric,
nothing but the routing decision can explain it. Same technique for
EventMem+Redis vs EventMem in-memory: only the transport differs.

**Controls** — a `random` router at matched fan-out. If selective routing
cannot beat random selection of the same size, the predicate is doing no work.
Measured F1 1.000 vs 0.130.

**Ground truth** — every workload declares which agents genuinely needed each
fact. `Workload.validate()` runs before any result is produced, so a scripting
error fails loudly instead of appearing as a system's poor recall.

**Negative tests** — `test_a_too_narrow_predicate_loses_to_broadcast` is a
*passing* test asserting EventMem loses. A suite containing only cases its
subject wins measures nothing.

**Self-checking metrics** — `counters_were_reset` invalidates the Redis figure;
a non-zero `llm_calls` invalidates mem0's write latencies; a recall above 1.0
raises rather than clamps.

**Stability runs** — repeated end to end. Every logical metric was identical
across runs; only latencies moved.

### Stability

| | run 1 | run 2 |
|---|---|---|
| recall@5 (all three) | 0.833 / 0.847 / 0.847 | **identical** |
| coverage of reachable | 0.571 / 1.000 / 1.000 | **identical** |
| satisfied, retrievals, learned-without-asking | — | **identical** |
| embedding calls, LLM calls | — | **identical** |
| Redis commands | 115 / 205 / 0 | 115 / 208 / 0 |
| mem0 search median | 39.8 ms | 45.6 ms |
| EventMem+Redis search median | 20.1 ms | 19.8 ms |
| EventMem push latency | 6.0 ms | 6.9 ms |

Ordering never changed, but the mem0-to-EventMem search ratio moved between
runs. **Do not quote a single-run speed multiple from this benchmark** — quote
the range, or run it yourself with more repeats. The logical results are
reproducible; the timings are indicative on one loaded laptop.

---

## What is not measured

- **mem0's actual headline features.** `infer=False` switched off LLM
  extraction, contradiction detection, memory updating. EventMem has no
  equivalent of any of them. Nothing here says anything about their quality.
- **Scale.** 32 facts. EventMem's linear cosine scan will lose to a real vector
  index well before mem0 does; `LayeredRouter` is linear in subscriptions.
- **Multi-node.** Single machine, single Redis. Ordering across shards and
  conflict resolution under real network delay are untested.
- **Real LLM agents.** A deterministic mock stands in, so wall times measure
  coordination, not reasoning. Real model calls dominate by two to three orders
  of magnitude — which is the argument for the coordination layer being cheap,
  not evidence the system is fast.
- **Durability of EventMem's projection.** The log is durable; the searchable
  projection is rebuilt by replay and was not tested under a hard kill.

---

## Reproducing

```bash
# Redis with the search module (mem0's vector store needs RediSearch)
docker run -d --name eventmem-redis-stack -p 6380:6379 redis/redis-stack-server:latest
pip install -e ".[mem0bench,dev]"

# All three systems, live dashboard on http://localhost:8077
python -m benchmarks.mem0_vs_eventmem.run \
    --redis-url redis://localhost:6380 --with-in-memory

# Terminal only, machine-readable output
python -m benchmarks.mem0_vs_eventmem.run \
    --redis-url redis://localhost:6380 --with-in-memory \
    --headless --json results.json

# Tests
pytest -q                                                  # 104, no services
EVENTMEM_TEST_REDIS=redis://localhost:6380 pytest -q       # + 13 Redis tests

# The dashboard alone, no Redis (synthetic numbers, labelled as such)
python -m benchmarks.mem0_vs_eventmem.run --selftest
```

Drop `--with-in-memory` for the two-way comparison. Use `--mem0-store faiss` if
your Redis has no search module — EventMem stays on Redis, mem0 goes
in-process, and the report marks the comparison as weaker.

---

## Related

- [`mem0-results.md`](mem0-results.md) — the two-way run in detail
- [`mem0-comparison.md`](mem0-comparison.md) — method and fairness contract
- [`benchmarking.md`](benchmarking.md) — EventMem vs polling / broadcast / no-sharing
- [`extending.md`](extending.md) — swapping the router, store or transport

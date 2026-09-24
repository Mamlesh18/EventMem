# EventMem

**An event-driven shared memory runtime for collaborative AI agents.**

Most multi-agent frameworks treat memory as a database agents query. Every
agent retrieves the same facts repeatedly, repeats each other's tool calls, and
reasons over context that went stale between polls.

EventMem treats memory as an active coordination service. Every mutation is an
event, and the runtime pushes it to exactly the agents whose subscriptions
match. Nobody polls.

```python
from eventmem import Agent, EventMemRuntime, Subscription

runtime = EventMemRuntime()
reviewer = Agent("reviewer", runtime)

await runtime.subscribe(Subscription(
    "reviewer", attribute_filters={"component": "backend"},
))
await reviewer.start()

# This reaches the reviewer immediately. No polling, no retrieval loop.
await coder.remember(
    "Login handler builds SQL from raw request input",
    attributes={"component": "backend"},
)
```

---

## Is it actually better?

That depends on what you compare it against, so the repository ships the
comparison rather than the claim. `python -m benchmarks.run` runs six
architectures through the same workloads and scores them on the same
ground truth.

From `noisy_broadcast` (13 agents, 40 events, each relevant to ~3 of them):

| system | transfers | wasted | precision | recall | fresh | lat ms |
|---|---|---|---|---|---|---|
| no_sharing | 0 | 0 | n/a | 0.000 | 0.000 | n/a |
| on_demand (RAG-style) | 160 | 120 | 0.250 | 0.333 | 1.000 | n/a |
| polling (10 ms) | 120 | 0 | 1.000 | 1.000 | 1.000 | 0.397 |
| polling (50 ms) | 120 | 0 | 1.000 | 1.000 | 0.875 | 57.654 |
| broadcast | 480 | 360 | 0.250 | 1.000 | 1.000 | 0.054 |
| random (control) | 80 | 67 | 0.163 | 0.108 | 0.000 | 0.037 |
| **eventmem** | **120** | **0** | **1.000** | **1.000** | **1.000** | **0.060** |

**What this actually shows, stated plainly:**

- **vs broadcast — a real win.** Same freshness, same recall, **75% fewer
  deliveries** and **zero wasted wakeups** against 360. Every wasted delivery
  is an agent woken, a context window spent, and often a model call made for
  nothing.
- **vs fast polling — a tie on quality, a win on cost.** A 10 ms poll keeps up,
  and pays 35 background retrievals to do it where push makes zero. It is also
  7x slower to propagate (0.397 ms vs 0.060 ms), because the poll interval sits
  inside the latency. `--poll-sweep` plots the whole curve.
- **vs slow polling — a win on latency and freshness**, 57.7 ms against 0.060 ms,
  and 0.875 freshness against 1.000 at a 50 ms interval. On `support_desk`, a
  50 ms poll drops to **0.250** freshness: agents act on stale context.
- **vs no sharing — the tool-call win.** On `tool_dedup`, 4 calls become 1:
  **ρ_dup = 0.75**, measured against an actual no-sharing run rather than an
  assumed baseline.
- **vs random at the same fan-out — the predicate is doing real work.**
  F1 1.000 against 0.130. Cutting delivery volume is easy; cutting it while
  keeping every needed update is the claim, and random is what separates the two.

**Where EventMem loses:** the `narrow_subscription` workload is in the suite
because selectivity has a cost. An over-tight predicate silently drops updates
an agent needed, and broadcast beats EventMem there on recall and freshness.
That case is a passing test (`test_a_too_narrow_predicate_loses_to_broadcast`),
not a footnote.

Full numbers: [`docs/benchmarking.md`](docs/benchmarking.md).

---

## Install

```bash
pip install -e .                 # core runtime, zero dependencies
pip install -e ".[local]"        # + real local embeddings for semantic routing
pip install -e ".[redis]"        # + Redis Streams transport
pip install -e ".[dev]"          # + pytest, ruff, mypy
```

The core has **no required dependencies**. The in-memory transport, hashing
embedder and mock model mean a fresh clone runs and tests without keys,
services or model downloads.

```bash
python examples/01_quickstart.py       # see the push happen
python -m benchmarks.run               # compare against the baselines
pytest                                 # 72 tests
```

---

## The architecture is the configuration

Every part of the design is a constructor argument, held by protocol and never
by concrete type. Changing the ideology does not mean forking the runtime.

```python
runtime = EventMemRuntime(
    router=TopicRouter(),                  # the routing ideology
    conflict_policy=SetUnionMerge(),       # how concurrent writes resolve
    embedder=SentenceTransformerEmbedder(),
    bus=RedisEventBus(client),             # transport
    event_store=RedisEventStore(client),   # the durable log
    lifecycle_policy=TTLPolicy(ttl_s=3600),
    tracer=ConsoleTracer(),
)
```

| seam | ships with | swap it when |
|---|---|---|
| `Router` | `LayeredRouter` (type → attributes → meaning), `TopicRouter`, `PredicateRouter`, `BroadcastRouter`, `RandomRouter`, `CompositeRouter` | your matching rule is not a conjunction of filters |
| `ConflictPolicy` | `ConfidenceThenRecency`, `LastWriterWins`, `FirstWriterWins`, `PreferAgents`, `SetUnionMerge`, `Manual` | your memory is additive, or a domain rule decides |
| `Embedder` | `HashingEmbedder`, `SentenceTransformerEmbedder`, `AzureEmbedder`, `OpenAIEmbedder`, `CachingEmbedder` | you need real semantics, or a specific model |
| `EventBus` | in-memory, Redis Streams | you need durability or multiple processes |
| `RecordStore` | in-memory | you outgrow a linear scan — maps onto Chroma/Qdrant/pgvector |
| `LifecyclePolicy` | `NeverExpire`, `TTLPolicy`, `AgeAndUsePolicy` | your knowledge goes stale on a different schedule |

[`docs/extending.md`](docs/extending.md) has a worked example for each.
`examples/02_swap_the_ideology.py` runs five routers over one workload.

---

## How routing works

A subscription is a conjunction evaluated cheapest-first, so the expensive
check only runs on events that already passed the cheap ones:

```
layer 1   event type    set membership
layer 2   attributes    dict comparison
layer 3   semantics     cosine similarity against an interest vector
```

An agent uses whichever layers it needs. Layer 3 alone means *"wake me on
anything that reads like a security concern"* with no keyword rule written
down.

### A warning about layer 3

Semantic routing is only as good as the embedder, and the failure mode is
silent. With the `HashingEmbedder` fallback, cosine similarity measures **token
overlap, not meaning**: paraphrases sharing no vocabulary score exactly `0.0`,
while unrelated text sharing one word can clear a low threshold.

EventMem makes that visible rather than hiding it:

- `HashingEmbedder.semantic` is `False`, and the runtime raises a
  `RuntimeWarning` if you register a semantic subscription against it.
- `Subscription.calibrate(embedder, should_match, should_not_match)` reports
  the real margin between your positive and negative examples and tells you
  when **no threshold can separate them**.
- The benchmark CLI prints which embedder produced every number.

```python
cal = sub.calibrate(embedder, should_match=[...], should_not_match=[...])
print(cal.report())
if not cal.separable:
    ...  # this embedder cannot express your distinction; no threshold will help
```

Run `python examples/03_semantic_routing.py` to see the contrast measured.

---

## What the runtime guarantees

**The log is the source of truth.** The record store is a projection.
`await runtime.rebuild_projection()` replays the log and reconstructs every
record — side-effect free: no routing, no delivery, no duplicate metrics.

**Conflicts are detected, not silently lost.** A writer declares the version it
read; if the store moved past it, the policy decides and a `conflict_detected`
event is published to subscribers. Opting out (`base_version=None`) means blind
overwrite, which is occasionally what you want and usually a bug — hence
`agent.update(...)`, which declares the version for you.

**Cascades are bounded.** An agent reacting to a depth-*n* event stamps its
result *n+1*; the runtime refuses to route past `max_depth`, so agents cannot
loop forever reacting to each other. A capped event is still recorded — losing
data is worse than not delivering it.

**Delivery is at-least-once** on Redis. Entries are acknowledged *after*
handling, so a crash mid-handle redelivers rather than silently dropping work;
`claim_stale()` recovers entries a dead consumer left pending. Handlers should
be idempotent.

**Memory retires.** The `LifecycleManager` advances records through
`created → verified → summarized → archived → expired`, drops dead records out
of reads, and **publishes a `lifecycle_changed` event** so agents still holding
a copy learn it retired. Silently archiving a record three agents believe is
live is exactly the staleness the runtime exists to prevent.

**Provenance is a graph, not a field.** `ProvenanceTracker` indexes
`caused_by` across every event, so you can ask:

```python
runtime.provenance.explain("mem.schema")              # derivation, readable
runtime.provenance.lineage(event_id)                  # root cause → here
runtime.provenance.contaminated_memories(event_id)    # blast radius of a retraction
runtime.provenance.trust_path("mem.schema")           # weakest link in the chain
```

---

## Measurement

All seven memory-centric metrics are computed, and the ones that **cannot** be
computed from runtime counters alone live somewhere else on purpose:

| metric | where | why |
|---|---|---|
| propagation latency, sync delay, fan-out efficiency, silent-event rate | `RuntimeMetrics` | the runtime can measure these about itself |
| freshness, staleness, consistency, coverage, routing precision/recall/F1 | `MetricSet` | need a **ground truth** of who should have known what |
| duplicate tool-call reduction | `MetricSet` | needs a **baseline run** for `C_base` |

A metric the runtime cannot honestly measure does not get a counter on the
runtime that looks authoritative. Metrics with no data return `NaN`, never a
flattering `0.0`.

Two measurement details that are easy to get wrong and are handled explicitly:

- **Latency is stamped at enqueue, not dequeue.** Publish→enqueue is the
  transport cost the runtime is accountable for. Publish→dequeue additionally
  includes however long the agent was busy, which belongs to the agent's
  workload — it is reported separately as `mean_pickup_ms`.
- **Timing uses both clocks.** `perf_counter` is sub-microsecond but its epoch
  is process-local, so it is meaningless across a real transport. `time.time()`
  is comparable across processes but has 15.6 ms granularity on Windows. Every
  stamp carries both plus its origin process, and `timing_precise` tells you
  which was used. See [`src/eventmem/core/clock.py`](src/eventmem/core/clock.py).

---

## Repository layout

```
src/eventmem/
  core/         events, records, the clock, and every protocol in one file
  routing/      layered predicate + alternative ideologies
  conflict/     collision detection and resolution policies
  embeddings/   hashing, local, hosted, caching
  transport/    in-memory and Redis Streams, plus serialisation
  store/        record store (projection) and event log
  llm/          model backends, including a deterministic mock
  runtime.py    the publish pipeline
  agents.py     Agent and LLMAgent
  lifecycle.py  the Lifecycle Manager
  provenance.py the Provenance Tracker
  metrics.py    all seven metrics
benchmarks/
  systems/      EventMem + broadcast, random, polling, on-demand, no-sharing
  workloads/    scenarios with ground truth
  harness.py    runs one workload against one system
  run.py        the CLI
```

---

## Contributing

New routing ideologies, conflict policies and workloads are the most valuable
contributions, and the repository is shaped to make them cheap:

- **A new router** is one class with four methods → [`docs/extending.md`](docs/extending.md)
- **A new workload** is `agents()` + `script()` with `needed_by` ground truth →
  [`docs/benchmarking.md`](docs/benchmarking.md)
- **A new baseline** implements `System` and joins the comparison table

See [`CONTRIBUTING.md`](CONTRIBUTING.md). The one hard rule: **a workload that
only contains cases EventMem wins is not a benchmark.** If your change makes
EventMem look better, add the case where it looks worse.

## Citation

```bibtex
@misc{eventmem2026,
  title  = {EventMem: An Event-Driven Shared Memory Runtime for Collaborative AI Agents},
  author = {Mamlesh VA},
  year   = {2026},
  url    = {https://github.com/Mamlesh18/EventMem}
}
```

## License

MIT — see [LICENSE](LICENSE).

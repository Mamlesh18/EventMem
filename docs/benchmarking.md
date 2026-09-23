# Benchmarking

```bash
python -m benchmarks.run                          # every workload, every system
python -m benchmarks.run --workload tool_dedup    # one workload
python -m benchmarks.run --poll-sweep             # freshness vs poll interval
python -m benchmarks.run --repeats 5 --json out.json
```

## What makes this a benchmark rather than a demo

Three properties, each enforced in code:

**Every system runs the identical script.** A workload emits abstract steps;
the harness executes them against whichever architecture is under test. No
system gets its own scenario.

**Every workload carries a ground truth.** For each published fact the workload
declares `needed_by` — the agents that genuinely require it. Without that you
can measure how many messages a system sent but not whether they were the right
ones, and "fewer deliveries" is indistinguishable from "dropped the update".
`Workload.validate()` runs before any result is produced, so a scripting
mistake fails loudly instead of appearing as a system's poor recall.

**Baselines are real, not strawmen.** `PollingSystem` is a genuine shared store
with a genuine retrieval loop — what AutoGen-style and RAG-memory frameworks
actually do. At a short enough interval it *should* match push on freshness,
and showing exactly where it does and what that costs is a stronger result than
showing it fail.

The harness will not compensate for a system's weakness. `PollingSystem.settle`
sleeps exactly as long as the workload asked, never "at least one poll
interval" — the interval is the variable under test.

---

## The systems

| system | what it is | the question it answers |
|---|---|---|
| `no_sharing` | isolated per-agent memory | is coordination worth anything? establishes `C_base` |
| `on_demand` | shared store read at point of use (RAG-style) | is background propagation worth anything? |
| `polling(Δt)` | shared store with a retrieval loop | is *push* worth anything, at what Δt? |
| `broadcast` | push, no filtering | is *selective* routing worth anything? |
| `random(k)` | push to a random subset of the same size | is the predicate doing real work, or is any fan-out this good? |
| `eventmem` | push + selective predicate routing | the system under test |

`broadcast`, `random` and `eventmem` share one implementation and differ only
in the router. That is the cleanest available ablation: if broadcast beats
EventMem on some metric, nothing but the routing decision can explain it.

## The workloads

| workload | isolates | expected outcome |
|---|---|---|
| `support_desk` | a sequential chain; each agent needs the previous one's output | push wins on freshness against slow polls |
| `tool_dedup` | four agents needing the same expensive lookup | ρ_dup against a measured floor |
| `research_pipeline` | concurrent edits to one artifact | conflict handling, consistency |
| `noisy_broadcast` | 13 agents, each event relevant to ~3 | **selective routing's best case** |
| `narrow_subscription` | a predicate too tight to catch what's needed | **selective routing's worst case** |

That last row is the point. A suite containing only cases its subject wins is
not measuring anything.

---

## Results

Hashing embedder, in-memory transport, single process, Python 3.11 on Windows.
Layer-3 semantic subscriptions are **not** exercised — these workloads route on
type and attributes only, so the fallback embedder does not affect them. See
[Honest caveats](#honest-caveats).

### `noisy_broadcast` — selective routing's best case

13 agents, 40 events, each relevant to ~3 of them.

| system | transfers | wasted | precision | recall | F1 | fresh | lat ms |
|---|---|---|---|---|---|---|---|
| no_sharing | 0 | 0 | n/a | 0.000 | n/a | 0.000 | n/a |
| on_demand | 160 | 120 | 0.250 | 0.333 | 0.286 | 1.000 | n/a |
| polling(10ms) | 120 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0.397 |
| polling(50ms) | 120 | 0 | 1.000 | 1.000 | 1.000 | 0.875 | 57.468 |
| polling(200ms) | 0 | 0 | n/a | 0.000 | n/a | 0.000 | n/a (never polled) |
| broadcast | 480 | 360 | 0.250 | 1.000 | 0.400 | 1.000 | 0.054 |
| random(k=2) | 80 | 67 | 0.163 | 0.108 | 0.130 | 0.000 | 0.037 |
| **eventmem** | **120** | **0** | **1.000** | **1.000** | **1.000** | **1.000** | **0.060** |

**vs broadcast: 75% fewer deliveries, 360 fewer wasted wakeups, identical
freshness and recall.** This is the central result. Every wasted delivery is an
agent woken, a context window spent, and often a model call made for nothing.

**vs random at the same fan-out: F1 1.000 against 0.130.** The predicate is
doing real work, not just reducing volume.

### `tool_dedup` — duplicate tool calls

| system | tool calls | ρ_dup |
|---|---|---|
| no_sharing | **4** (`C_base`) | 0.000 |
| polling(50ms) | 2 | 0.500 |
| polling(10ms) | 1 | 0.750 |
| broadcast | 1 | 0.750 |
| **eventmem** | **1** | **0.750** |

ρ_dup is computed against an **actual no-sharing run**, not an assumed
denominator. EventMem ties broadcast and fast polling here — the workload has
every agent interested in the same thing, so there is nothing for selectivity
to filter. Reporting the tie is the point.

### `narrow_subscription` — where EventMem loses

An `auditor` needs every compliance decision but subscribes only to the finance
team's.

| system | recall | fresh | F1 |
|---|---|---|---|
| broadcast | **1.000** | **1.000** | **1.000** |
| random(k=2) | 1.000 | 1.000 | 1.000 |
| on_demand | 1.000 | 1.000 | 1.000 |
| **eventmem** | **0.500** | **0.500** | **0.667** |

Selectivity has a cost: an over-tight predicate silently drops updates. The CLI
prints a warning when random beats EventMem's F1, and
`test_a_too_narrow_predicate_loses_to_broadcast` pins this as a passing test.
If it ever starts passing for EventMem, the suite has stopped exercising the
failure mode.

### `--poll-sweep` — what freshness costs a poller

On `support_desk`:

| system | fresh | recall | lat ms | background retrievals |
|---|---|---|---|---|
| polling(5ms) | 1.000 | 1.000 | 4.597 | 35 |
| polling(10ms) | 1.000 | 1.000 | 3.736 | 35 |
| polling(25ms) | 1.000 | 1.000 | 31.360 | 20 |
| polling(50ms) | 0.250 | 0.750 | 46.257 | 10 |
| polling(100ms) | 0.000 | 0.750 | 75.745 | 5 |
| polling(250ms) | 0.000 | 0.000 | n/a | 0 |
| **eventmem** | **1.000** | **1.000** | **0.119** | **0** |

Polling *can* match push on freshness — it just has to poll fast enough, and
the bill arrives as retrievals. Push gets the same freshness for none of them,
at ~30x lower propagation latency, because the interval is not sitting inside
the latency.

---

## Honest caveats

Read these before citing any number above.

1. **Single process, single machine, in-memory transport.** Multi-node
   delivery, ordering across shards, and conflict resolution under real network
   delay are not measured.
2. **Semantic routing is not exercised.** These workloads route on type and
   attributes. Layer 3 is evaluated separately by
   `examples/03_semantic_routing.py`, and the answer there is uncomfortable:
   the hashing fallback scores genuine security findings at **0.0000** while
   scoring a lexical decoy at **+0.27**, and even a real sentence-transformer
   model is defeated by the same decoy.
3. **Agents use a deterministic mock model** unless configured otherwise, so
   wall times measure coordination, not reasoning. Real model calls dominate by
   two to three orders of magnitude — which is the argument for the
   coordination layer being cheap, not evidence that the system is fast.
4. **Latency is publish→enqueue**, the transport cost. Publish→dequeue is
   reported separately as `mean_pickup_ms` because it includes agent busy time.
5. **Timing uses a hybrid clock.** `perf_counter` in-process (sub-microsecond),
   wall clock across processes (15.6 ms granularity on Windows).
   `timing_precise` says which was used; treat a sub-millisecond figure with
   `timing_precise=False` as unresolvable.
6. **A blank latency cell means nothing was delivered**, not that delivery was
   instant. Systems that transferred nothing report `n/a`, never `0.000`.

---

## Adding a workload

```python
from benchmarks.workloads import AgentSpec, Publish, Read, Workload

class MyWorkload(Workload):
    name = "my_workload"
    description = "one line for the results table"

    def agents(self):
        return [
            AgentSpec("writer", label="publishes"),
            AgentSpec("reader", attributes={"kind": "report"},
                      topic_pattern="report.*", label="reports"),
        ]

    def script(self):
        yield Publish(
            actor="writer", memory_id="m1", content="the finding",
            attributes={"kind": "report", "topic": "report.q3"},
            needed_by={"reader"},        # <- the ground truth
            settle_s=0.03,               # <- time passes, identically for all
        )
        yield Read(actor="reader", memory_id="m1")
```

Then register it in `benchmarks/workloads/__init__.py`.

**The two things to get right:**

- **`needed_by` must reflect the scenario's semantics**, never a particular
  router's behaviour. If you write it by observing what EventMem delivered, you
  have written a tautology.
- **`settle_s` must be non-zero.** With zero elapsed time a pull architecture
  gets no opportunity to retrieve at all, and the comparison becomes a foregone
  conclusion rather than a measurement.

Run `Workload.validate()` — it catches unknown agents, reads before writes, and
an author listed as needing its own write.

## Adding a baseline

Implement `benchmarks/systems/base.System`. The key abstraction is **held
state**: what each agent currently believes about each memory. Freshness,
consistency and ρ_dup all derive from it, so every system is measured the same
way rather than each reporting its own favourable definition.

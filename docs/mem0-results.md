# mem0 vs EventMem — measured results

Both systems on the **same Redis** (`redis/redis-stack-server`, RediSearch),
with the **same embedding model** (`all-MiniLM-L6-v2`, one shared instance),
over the same 32-fact corpus and 12 hand-labelled queries.

Raw data: [`mem0-results.json`](mem0-results.json). Method and fairness
contract: [`mem0-comparison.md`](mem0-comparison.md).

```
Python 3.11.9 · Windows · single machine · mem0ai 0.1.118 · redisvl 0.27.2
mem0 vector store: Redis (RediSearch)     EventMem: Redis Streams + event log
mem0 infer=False (LLM extraction off, verified: 0 LLM calls)
```

---

## Results

| | mem0 | EventMem | |
|---|---|---|---|
| **Phase B — retrieval quality** | | | |
| recall@5 | 0.833 | **0.847** | tie |
| precision@5 | 0.417 | 0.417 | tie |
| MRR | 1.000 | 1.000 | tie |
| search latency, median | 60.9 / 47.2 ms | **19.7 / 26.4 ms** | 1.8–3.1× faster |
| search latency, p95 | 70.2 ms | **47.3 ms** | |
| **Phase A — ingest** | | | |
| write latency, median | 87.7 / 100.8 ms | **26.6 / 30.3 ms** | 2.9–3.7× faster |
| write latency, p95 | 114.9 ms | **32.9 ms** | |
| ingest wall time | 2 848 ms | **904 ms** | |
| **Phase C — propagation** | | | |
| required deliveries | 44 | 44 | |
| reachable by routing | 28 | 28 | ceiling |
| satisfied | 19 | **28** | |
| coverage of required | 0.432 | **0.636** | |
| coverage of reachable | 0.571 | **1.000** | |
| retrievals spent | 11 | **0** | |
| **Phase D — reactivity** | | | |
| facts learned without asking | 0 | **40** | capability |
| push latency, mean | n/a | 7.1 ms | |
| **Cost** | | | |
| embedding calls | 87 | **44** | |
| vector searches | 23 | **12** | |
| LLM calls | 0 | 0 | both honest |
| Redis commands (server-counted) | **115** | 212 / 213 | mem0 cheaper |

Latency cells show both runs where they differ (`run 1 / run 2`); every other
figure was **identical across both runs**. See [Stability](#stability).

---

## What this actually shows

**Retrieval quality ties.** recall@5 0.833 vs 0.847, precision identical, MRR
identical at 1.000. This is the result worth having: both sides embed with the
same model, so a tie demonstrates that **adding push costs nothing in retrieval
quality**. If EventMem had lost here, the reactivity would have been bought at
the price of being a worse memory.

**EventMem is 2–3× faster on reads and ~3× on writes** across two runs
(search 20–26 ms vs 47–61 ms; write 27–30 ms vs 88–101 ms). Not an
architectural claim — mem0's Redis path goes
through `redisvl` and does more per operation (`hgetall` + `hset` per write,
`FT.SEARCH` with a tag filter per read) than EventMem's in-process cosine scan
over a projection plus one `XADD`. A tuned vector index would close much of the
read gap, and EventMem's linear scan will lose at corpus sizes this benchmark
does not reach.

**Propagation is where the architectures genuinely differ.** EventMem delivered
**28 of 28 reachable** required deliveries (coverage of reachable 1.000) having
spent **zero retrievals**. mem0 reached 19 of 28 and spent 11 searches doing it
— and those 11 searches only happened because the harness *told* it to go
looking, one per declared interest. In a real system nobody issues those.

**40 facts reached an agent that never asked for them.** mem0 scores zero here
by construction: a passive store has no way to notify. That is a capability
difference, not a performance gap, and it is the thing EventMem exists to
provide.

**mem0 uses fewer Redis commands** — 115 against 212. EventMem pays commands to
push (74 `XADD`, 60 `XREADGROUP`, 41 `XACK`); mem0 pays them to search. Which is
cheaper depends entirely on the read/write ratio of the workload, and on this
corpus mem0 wins on raw command count.

---

## Stability

Two full runs, same machine, same Redis. **Every logical metric was identical**
— recall, precision, MRR, coverage, satisfied deliveries, retrievals spent,
facts learned without asking, embedding calls, LLM calls. Redis commands
differed by one on the EventMem side (212 vs 213), from one extra `xpending`
during the settle loop.

Only latency moved, and it moved in both directions:

| | run 1 | run 2 |
|---|---|---|
| mem0 search median | 60.9 ms | 47.2 ms |
| EventMem search median | 19.7 ms | 26.4 ms |
| mem0 write median | 87.7 ms | 100.8 ms |
| EventMem write median | 26.6 ms | 30.3 ms |

The ordering never changed, but the search ratio moved from 3.1× to 1.8×. **Do
not quote a single-run speed multiple from this benchmark** — quote the range,
or run it yourself with more repeats. The logical results are reproducible; the
timings are indicative on one loaded laptop.

---

## The ceiling, stated plainly

**Neither system can exceed 0.636 coverage of required deliveries on this
corpus.** 16 of the 44 required deliveries name an agent that never declared an
interest in that fact's topic — for example `f21` ("the TLS certificate expires
on the 14th", topic `infra`) is needed by `security`, which subscribes only to
`security` and `api`. No interest-based router can deliver that.

That is the standing cost of selective routing, and it is why two denominators
are reported:

- **coverage of required** — of everything an agent needed, how much arrived.
  Bounded by how well the interests were written.
- **coverage of reachable** — of what the declared interests made deliverable,
  how much arrived. Isolates routing and transport. EventMem: **1.000**.

Quoting only the first number would understate EventMem; quoting only the
second would hide a real limitation. Both are in the table.

---

## What is not measured

mem0 ran with `infer=False`, so these are switched off and **EventMem has no
equivalent of any of them**:

- LLM-based fact extraction from conversation
- Contradiction detection and memory updating (`ADD`/`UPDATE`/`DELETE`)
- Graph memory (entities and relationships)
- Mature multi-user scoping

The concession is verified, not asserted: mem0's LLM client is wrapped with a
counter and the run reports **0 LLM calls**. Had it been non-zero, the report
would have declared the write latencies invalid.

Also not measured: multi-node delivery, behaviour at corpus sizes where a
linear cosine scan stops being viable, and anything involving a real LLM in the
agent loop.

---

## Three measurement bugs found while producing these numbers

Recorded because each one produced a confidently wrong number first, and the
numbers above are only trustworthy because they were caught.

**1. Phase B leaked into phase C.** The retrieval phase's searches were being
credited as propagation. The tell was EventMem scoring 0.727 coverage against
an achievable ceiling of 0.636 — a score above the maximum. Fixed by ordering
the phases ingest → propagation → reactivity → retrieval and resetting agent
knowledge after the warmup.

**2. `CONFIG RESETSTAT` corrupted the Redis cost metric.** EventMem's setup
zeroed the server's counters, which made every one of mem0's deltas negative;
the positive-only filter dropped them, and mem0's Redis cost was reported as
**1 command** instead of 115. Fixed by removing the reset — a diff needs no
reset, and zeroing a counter another measurement depends on is never safe.
`diff_stats` now has a companion `counters_were_reset`, the cost is reported as
`n/a` when it fires, and the verdicts call it out as invalid.

**3. No warmup made the latency numbers impossible.** mem0's mean write latency
(218 ms) exceeded its own p95 (130 ms) — arithmetically impossible without an
outlier, which was first-write index creation. Fixed with a discarded warmup
write and search, and a median headline figure.

A fourth, caught earlier: duplicate rows from a re-ingested corpus pushed
recall to 1.36. Hits are now deduped before scoring, with a guard that raises
rather than clamps.

---

## Reproducing

```bash
docker run -d --name eventmem-redis-stack -p 6380:6379 redis/redis-stack-server:latest
pip install -e ".[mem0bench]"
python -m benchmarks.mem0_vs_eventmem.run --redis-url redis://localhost:6380
```

Live dashboard on `http://localhost:8077`. Add `--headless --json out.json` for
terminal output, or `--selftest` to check the dashboard without Redis.

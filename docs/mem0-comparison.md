# mem0 vs EventMem

A head-to-head between **mem0's normal retrieval** and **EventMem's
event-driven retrieval**, both running over the same Redis with the same
embedding model, with a live dashboard.

```bash
# 1. Redis WITH the search module. mem0's Redis vector store needs RediSearch,
#    which plain redis:7-alpine does not ship.
docker run -d --name eventmem-redis -p 6379:6379 redis/redis-stack:latest

# 2. Dependencies
pip install -e ".[mem0bench]"

# 3. Run it. The browser opens and fills in as the benchmark runs.
python -m benchmarks.mem0_vs_eventmem.run
```

Other entry points:

```bash
# Check the dashboard without Redis (synthetic numbers, clearly labelled)
python -m benchmarks.mem0_vs_eventmem.run --selftest

# Terminal only
python -m benchmarks.mem0_vs_eventmem.run --headless --json results.json

# If your Redis has no search module: EventMem stays on Redis, mem0 uses faiss.
# A weaker comparison, and the report says so.
python -m benchmarks.mem0_vs_eventmem.run --mem0-store faiss
```

---

## The fairness contract

A comparison between two systems is only worth reading if it says what it held
constant. This one holds:

| | mem0 | EventMem |
|---|---|---|
| Embedding model | `all-MiniLM-L6-v2` | `all-MiniLM-L6-v2` (same loaded instance) |
| Storage | Redis (RediSearch vectors) | Redis (Streams + event log) |
| Corpus | 32 facts from `corpus.py` | identical |
| Queries | 12 labelled queries | identical |
| Ground truth | hand-labelled `relevant` / `needed_by` | identical |
| Interest statement | filters its searches by topic | routes on the same topic |
| Warmup | 1 write + 1 search, discarded | identical |

**Without the shared embedder this benchmark would measure embedding models,
not architectures.** That is the single most common way a memory comparison
goes wrong.

### The concession made to mem0

mem0 runs with **`infer=False`**. Its default `infer=True` makes an LLM call per
write to extract and deduplicate facts; charging it a network round-trip and a
token bill on every write would inflate EventMem's advantage on something this
benchmark is not about.

That concession is **verified, not assumed**: mem0's LLM client is wrapped with
a counter, `llm_calls` is reported on every run, and if it is ever non-zero the
report says in plain language that the write numbers are invalid.

### What this does *not* measure, and where mem0 is genuinely ahead

- **LLM-based fact extraction** — turning a conversation into atomic memories
- **Contradiction detection and memory updating** — mem0's `ADD/UPDATE/DELETE`
  reconciliation
- **Graph memory** — entity/relationship memory
- **Multi-user scoping** — mature `user_id` / `agent_id` / `run_id` isolation

EventMem has no equivalent of any of these. This benchmark covers **retrieval
quality and propagation only.**

---

## The four phases

Each isolates one question so a win or loss can be attributed.

| phase | question | who should win |
|---|---|---|
| **A — Ingest** | what does storing a fact cost? | — |
| **B — Retrieval** | precision/recall/MRR on labelled queries | should **tie** |
| **C — Propagation** | did the agent who needed a fact end up with it? | EventMem, structurally |
| **D — Reactivity** | how many facts arrived without being asked for? | mem0 scores 0 by construction |

### Why a tie in phase B is the result worth having

Both sides embed with the same model, so phase B is a check that **adding push
does not cost you retrieval quality**. If EventMem won here it would mean the
stores rank differently; if it lost badly it would mean the push architecture
came at the price of being a worse memory. A tie says you get the reactivity for
free.

### How phase C is kept fair

The pull system cannot know a fact arrived, so it is given an explicit chance to
catch up: **one search per topic each agent declares an interest in.** That is a
*generous* reading of how a pull architecture is used, because it assumes the
agent both knows to go looking and knows what to look for. It is still charged
a retrieval for each one, and those are reported as `retrievals_spent`.

### Why phase D is a capability, not a score

A passive store has no way to notify anybody. mem0 scoring 0 here is not a
failure at a task it attempted — it is the absence of a capability. The report
says so rather than presenting it as a performance gap.

---

## Reading the dashboard

The page streams over SSE and fills in while the run is in progress. A browser
that connects late is replayed the whole history, so nothing is missed.

- **Headline tiles** — recall@k, propagation coverage, retrievals spent, facts
  learned without asking
- **Side by side** — median search/write latency, embedding calls, Redis
  commands
- **Reading the numbers** — plain-language verdicts, tagged `win` / `tie` /
  `loss` / `capability` / `caveat`. The `loss` and `caveat` tags fire against
  EventMem when the numbers warrant it.
- **Full results** — every metric per phase; the better of the two is green
- **Live activity** — the event log as it happens

Colors are the dataviz reference palette, slots 1 (blue, mem0) and 2 (orange,
EventMem), validated in both light and dark mode (worst adjacent CVD ΔE 24.7
light / 26.8 dark).

---

## Measurement details that change the numbers

**Latency is a median, not a mean.** The first write into either system pays
one-time costs — index creation, lazy model load, first connection. Including it
made mem0's mean write latency (218 ms) exceed its own p95 (130 ms), which is
arithmetically impossible without an outlier and is the signature of exactly
that problem. There is now a warmup phase, discarded, and the headline figure is
the median. Mean and p95 are still reported so the three can be compared.

**Redis commands are counted by the server.** From `INFO commandstats`, diffed
across the run, not estimated from the code. EventMem pays extra commands to
push; mem0 pays them to search. Which is cheaper depends entirely on the
read/write ratio of your workload, and the report says that rather than claiming
a winner.

**Duplicate hits are deduped before scoring.** mem0 writes a fresh row per
`add()`, so a re-ingested corpus returns the same memory several times. Counting
those as separate hits produced a recall of 1.36 — a number recall cannot take.
Pinned by `test_duplicate_hits_cannot_push_recall_above_one`.

**Recall's denominator is `min(len(relevant), k)`.** A query with 3 relevant
facts asked at k=2 cannot reach recall 1.0, and penalising a system for the
harness's own cut-off would measure the wrong thing.

**Metrics with no data are `n/a`, never `0`.** A system that propagated nothing
must not appear to have the fastest propagation in the table.

---

## Troubleshooting

**"Redis is up but has no search module"** — you are running plain Redis.
mem0's vector store needs RediSearch:

```bash
docker rm -f eventmem-redis
docker run -d --name eventmem-redis -p 6379:6379 redis/redis-stack:latest
```

Or run with `--mem0-store faiss` to keep EventMem on Redis and put mem0's
vectors in-process. The dashboard will mark the comparison as weaker, because
the latency numbers stop being directly comparable.

**First run is slow** — `all-MiniLM-L6-v2` downloads once (~80 MB) and the
warmup phase pays index creation. Subsequent runs are much faster.

**The dashboard says "disconnected"** — the run finished and the server closed
the stream, or the process was stopped. Results stay on the page; `/results`
serves the full event history as JSON.

**A stale index from a previous run** — each run uses a fresh namespace and
`reset_namespace` clears matching keys and drops the mem0 index. It is scoped to
the namespace prefix and never issues `FLUSHALL`, because this runs against a
Redis you started and may hold other data.

---

## Adding a system

`MemorySystem` in `systems.py` is the interface: `setup`, `write`, `search`,
`known_to`, `settle`, `teardown`, plus a `Costs` object. Implement it, add it to
the list in `run.py`, and it joins the comparison.

The field that matters is **`known_to(agent)`** — the facts an agent holds
*without being asked to go and look*. Phase C and D are both scored from it, so
every system is measured the same way instead of each reporting its own
favourable definition of "knows".

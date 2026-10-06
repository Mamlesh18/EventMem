# Why EventMem

**Normal memory waits to be asked. EventMem tells you.**

That one difference is what this page is about. Every number below is measured,
and you can re-run all of it yourself in about three minutes.

---

## The problem

Most agent memory works like a filing cabinet. Agent A learns something and
files it. Agent B needs it — but B has no idea anything was filed.

So B has three options, and all three are bad:

- **Ask repeatedly.** Burns retrievals, and B is working with stale
  information in between.
- **Ask at the last second.** Only works if B already knows to ask, and knows
  what to ask for.
- **Don't ask.** B does the work again, or gets it wrong.

EventMem removes the question. When A files something, it arrives at B.

---

## What changes in practice

| | Filing cabinet | EventMem |
|---|---|---|
| B finds out when… | it remembers to ask | A writes it |
| Cost for B to find out | one search, every time | nothing |
| If B forgets to ask | B never knows | B still knows |
| B can react to A | no | yes |

---

## Five places this wins

### 1. A chain of agents

Sentiment → triage → draft a reply. Each step needs the last step's answer.

With a filing cabinet, each step has to poll. We measured it: polling every
50 ms, only **25%** of reads had the current version. The agents were working
from stale context three times out of four.

With EventMem: **100%**. The answer is already there when the next agent wakes
up.

### 2. Several agents needing the same expensive lookup

Four agents all need the same pricing table. One API call, or four?

| | API calls |
|---|---|
| No sharing | 4 |
| **EventMem** | **1** |

**75% fewer.** The first agent pays; the others are simply told.

### 3. A big team where most news is irrelevant

12 agents, 40 updates, each update relevant to about 3 of them.

| | Messages delivered | Agents woken for nothing |
|---|---|---|
| Send to everyone | 480 | 360 |
| **EventMem** | **120** | **0** |

**75% fewer messages, zero wasted wake-ups** — and every agent still got
everything it needed. Each wasted wake-up is a real cost: an agent woken, a
context window spent, often a model call made for nothing.

### 4. Making sure the right people actually know

We wrote down, by hand, which agents genuinely needed which facts. Then we
checked who ended up with them.

| | Facts delivered | Searches it took |
|---|---|---|
| mem0 | 19 of 28 | 11 |
| **EventMem** | **28 of 28** | **0** |

EventMem got **everything** to **everyone** who needed it, and spent **zero
searches** doing it.

And mem0 only managed 19 because we explicitly told it to go looking. In a
real system, nobody runs those 11 searches for you.

### 5. Reacting at all

Over one run, **40 facts arrived at an agent that never asked for them.**

A filing cabinet scores **0** here — not because it's slow, but because it has
no way to tell anyone anything. This is the capability, in one number.

---

## The numbers

Same corpus, same embedding model, same Redis. Two runs, and every result
below came out the same both times.

| | mem0 | EventMem |
|---|---|---|
| **Retrieval quality** (recall@5) | 0.833 | **0.847** |
| Got the right facts to the right agents | 19 / 28 | **28 / 28** |
| Searches needed to do that | 11 | **0** |
| Facts that arrived unasked | 0 | **40** |
| Search speed | 40–46 ms | **20 ms** |
| Write speed | 61–68 ms | **20–22 ms** |
| Embedding calls | 87 | **44** |

**Retrieval quality is a tie — and that's the important part.** Both use the
same embedding model, so this proves EventMem's reactivity is *free*. You
aren't trading away a good memory to get it. You keep the memory and gain the
coordination.

EventMem was also about **2× faster to search** and **3× faster to write**, and
used **half the embedding calls**.

> Speeds moved a bit between runs. Treat them as "roughly 2–3× faster", not as
> an exact multiple. Everything else was identical both times.

---

## Redis or in-memory?

Both work identically. Pick on one question: **do agents live in separate
processes, or does the log need to survive a restart?**

| | In-memory | Redis |
|---|---|---|
| Delivery speed | **0.4 ms** | 6 ms |
| Agents in separate processes | no | **yes** |
| Survives a restart | no | **yes** |
| Correctness | identical | identical |

Yes to either question → Redis. Otherwise in-memory, which delivers in
**0.4 ms instead of 6 ms** — more than 10× faster across both runs.

Nothing else changes. Search and write speeds are the same either way, and
both got the same 28 of 28.

---

## When mem0 is the better choice

Worth saying clearly, because it's true and because it'll save you time:

- **You want memories written *for* you.** mem0 can read a conversation and
  extract the facts worth keeping, using an LLM. EventMem stores what you hand
  it. If you want curation, use mem0.
- **You need contradictions sorted out.** mem0 can notice a new fact
  contradicts an old one and update it. EventMem has nothing like this.
- **You want a knowledge graph.** mem0 has one. EventMem doesn't.
- **Nothing in your system needs to react.** One agent, or agents that never
  need each other's findings? You need a store, not a coordinator. Use mem0.
- **Raw Redis command count matters to you.** mem0 used fewer (115 vs ~206).
  EventMem spends commands pushing; mem0 spends them searching.

**And you can use both.** Keep mem0 as the curated store. Put EventMem in front
to handle the coordination. They solve different problems and sit at different
layers.

---

## How to check any of this

Nothing here is on trust. Every number comes from code in this repository.

```bash
docker run -d --name redis-stack -p 6380:6379 redis/redis-stack-server:latest
pip install -e ".[mem0bench,dev]"

# The comparison, with a live dashboard
python -m benchmarks.mem0_vs_eventmem.run \
    --redis-url redis://localhost:6380 --with-in-memory

# The tests
pytest -q                                              # 104 tests, no setup
EVENTMEM_TEST_REDIS=redis://localhost:6380 pytest -q   # + 13 Redis tests
```

A few things we did so these numbers would hold up:

- **Both systems use the same embedding model.** Otherwise you're comparing
  models, not designs.
- **The "right answers" were written by hand, in advance** — before we saw
  what either system returned.
- **The test suite includes a case where EventMem loses**, on purpose. A
  benchmark that only contains wins isn't measuring anything.
- **One limit we hit:** 16 of the 44 facts were needed by an agent that hadn't
  subscribed to that topic, so nothing could deliver them. EventMem got
  **100% of what was reachable**. If an agent doesn't say what it cares about,
  no system can guess.

Longer versions: [`choosing-a-memory-system.md`](choosing-a-memory-system.md)
for the full decision guide and method, [`mem0-results.md`](mem0-results.md)
for the raw detail.

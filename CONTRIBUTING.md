# Contributing to EventMem

EventMem is a research runtime. The most valuable contributions are new
**routing ideologies**, **conflict policies**, and **workloads** — especially
workloads that show where the architecture fails.

## Setup

```bash
git clone https://github.com/Mamlesh18/EventMem
cd EventMem
pip install -e ".[dev]"
pytest                    # 72 tests, no keys or services needed
ruff check src benchmarks tests
```

The core runtime has **no required dependencies**. Keep it that way: anything
needing a package goes behind an optional extra with a clear `ImportError`
pointing at it.

---

## The one hard rule

**A benchmark that only contains cases EventMem wins is not a benchmark.**

If your change makes EventMem look better, add the case where it looks worse.
`narrow_subscription` exists because selective routing silently drops updates
when a predicate is too tight, and
`test_a_too_narrow_predicate_loses_to_broadcast` is a *passing* test asserting
that broadcast beats us there. That is the standard.

This applies to documentation too. Every performance claim in the README names
the baseline it beat and the cases where it ties or loses.

---

## What good contributions look like

### A new router

One class, four methods. See [`docs/extending.md`](docs/extending.md).

- Does not require changes to `runtime.py`.
- Docstring says **when to use it and when not to**.
- Joins the benchmark: `EventMemSystem(router=YourRouter)` needs no new system
  class.

### A new conflict policy

- **Deterministic.** `rebuild_projection()` replays the log and re-runs your
  policy on the same collisions; a policy depending on wall-clock time or
  iteration order will reconstruct a different state than the live run
  produced. `ConfidenceThenRecency` breaks its final tie on `event_id` for
  exactly this reason.
- Set `Resolution.merged=True` when the winner is built from both sides.

### A new workload

- `needed_by` reflects the **scenario's semantics**, not any router's
  behaviour. Writing it by observing what EventMem delivered produces a
  tautology.
- `settle_s` is non-zero, or pull architectures get no chance to retrieve and
  the result is predetermined.
- `validate()` returns `[]`.

---

## Measurement standards

These are not style preferences; getting them wrong produces published numbers
that are wrong.

**A metric with no data returns `NaN`, never `0.0`.** A system that propagated
nothing must not appear to have the fastest propagation in the table.

**A metric the runtime cannot honestly measure does not get a runtime
counter.** Freshness, consistency and ρ_dup need ground truth or a baseline
run, so they live on `MetricSet`, populated by the harness. Putting a
`duplicate_work_avoided` counter on the runtime and incrementing it from a demo
script makes a scripted constant look like a measurement.

**State which embedder produced any semantic-routing number.** A result from
the hashing fallback is not comparable to one from a real model. The runtime
raises a `RuntimeWarning` when a semantic subscription meets a non-semantic
embedder, and the benchmark CLI prints the embedder in its header.

**Latency is publish→enqueue.** Publish→dequeue additionally includes however
long the agent was busy; it is reported separately as `mean_pickup_ms`.
Conflating them makes a slow handler look like slow transport.

**Timing crosses process boundaries carefully.** See
[`core/clock.py`](src/eventmem/core/clock.py). If you add a transport, carry
the full `Stamp` (both clocks plus origin process) through serialisation.

---

## Testing

```bash
pytest                              # everything
pytest tests/test_routing.py -v     # one file
pytest --cov=eventmem               # coverage
```

Tests must pass with no keys, no Redis and no model downloads.

**Do not wait on a fixed sleep.** Use `runtime.bus.drain()`, which blocks until
every delivered event has been handled. Queue depth is *not* a drain signal:
`asyncio.wait_for` removes the item before the handler resumes, so depth hits
zero while an event is still in flight. That bug cost a debugging session; the
in-flight counter exists so it cannot recur.

Name tests after the behaviour, not the method:
`test_a_too_narrow_predicate_loses_to_broadcast`, not `test_router_2`.

---

## Code style

`ruff check` and `ruff format` must pass. Beyond that:

**Comment the *why*, never the *what*.** The codebase explains non-obvious
decisions and the reasoning behind tradeoffs. If a comment restates the line
below it, delete it.

```python
# No:
self.outstanding += 1   # increment outstanding

# Yes:
# Queue depth cannot serve this purpose: asyncio.wait_for removes the item
# before the waiting coroutine resumes, so a depth of zero can mean "drained"
# or "handed over but not yet handled".
self.outstanding += 1
```

**Docstrings say when *not* to use the thing.** `HashingEmbedder`'s docstring
states that it scores paraphrases at zero and must not back a published
semantic-routing claim. That is more useful than describing hashing.

---

## Reporting a bug

Include the EventMem version, which router / conflict policy / embedder /
transport you used, and a minimal reproduction. The in-memory defaults
reproduce most issues without any services.

For a benchmark result that looks wrong, attach the `--json` output. It carries
every metric, not just the table columns.

---

## Security

Never commit a `.env`. `.env.example` is tracked and must contain only
placeholders. If you find a committed credential in history, open an issue
without quoting the value.

## License

Contributions are made under the MIT License.

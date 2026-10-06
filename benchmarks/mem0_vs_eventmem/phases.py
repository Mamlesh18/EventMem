"""The four benchmark phases.

Each phase isolates one question, so a win or loss can be attributed instead of
being folded into a single number nobody can interrogate.

  A  INGEST        write the corpus. Cost and latency of storing a fact.
  B  RETRIEVAL     answer the labelled queries. Precision/recall/MRR at k.
                   Both sides use the same embedder, so this should roughly
                   TIE - and that is the result worth having: it shows push
                   does not cost you retrieval quality.
  C  PROPAGATION   agent A writes something agent B needs. Does B get it?
                   This is the architectural difference, and the phase mem0
                   cannot win by construction, which the report says plainly.
  D  REACTIVITY    how many facts did each agent learn without asking?

Every phase emits progress through a callback so the dashboard can render while
the run is still going.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Set

from .corpus import (
    AGENTS,
    FACTS,
    QUERIES,
    WARMUP_FACT,
    WARMUP_QUERY,
    reachable_deliveries,
    required_deliveries,
)
from .systems import MemorySystem

Emit = Callable[[str, Dict[str, Any]], Awaitable[None]]

DEFAULT_K = 5


@dataclass
class PhaseResult:
    phase: str
    system: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    detail: List[Dict[str, Any]] = field(default_factory=list)


def _mean(values: List[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def _median(values: List[float]) -> float:
    """The headline latency figure.

    Preferred over the mean because a single one-time cost (index creation, a
    lazy model load) distorts a mean over 32 samples badly, and the point of
    the warmup phase is that such costs should not be in the sample at all.
    The mean is still reported so the two can be compared.
    """
    return statistics.median(values) if values else float("nan")


def _p95(values: List[float]) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[min(int(0.95 * len(ordered)), len(ordered) - 1)]


# ------------------------------------------------------------------ phase A

async def phase_warmup(system: MemorySystem, emit: Emit,
                       k: int = DEFAULT_K) -> None:
    """One untimed write and one untimed search.

    Pays the one-time costs - index creation, lazy model load, first
    connection - before anything is measured, identically for both systems.
    """
    await emit("phase_start", {"phase": "warmup", "system": system.name, "total": 2})
    await system.write(WARMUP_FACT)
    await system.settle()
    await system.search(WARMUP_FACT.author, WARMUP_QUERY.text, k)

    # Discard the warmup samples and the counters they moved, so the measured
    # phases start from zero and the warmup cannot leak into any result.
    costs = system.costs
    costs.write_latencies_ms.clear()
    costs.search_latencies_ms.clear()
    costs.embed_calls = 0
    costs.vector_searches = 0
    costs.llm_calls = 0

    # The warmup search surfaced results, which a pull system counts as having
    # learned. Forget them, or the warmup shows up as propagation.
    _forget_everything(system)
    await emit("phase_done", {"phase": "warmup", "system": system.name,
                              "metrics": {"warmed": True}})


def _forget_everything(system: MemorySystem) -> None:
    """Reset what every agent believes it knows.

    Used between phases so one phase's reads cannot be credited to another.
    Without it, the searches in the retrieval phase counted as propagation and
    EventMem scored 0.727 coverage against a routing ceiling of 0.636 - a
    score above the achievable maximum, which is how the leak was spotted.
    """
    for attr in ("_known",):
        known = getattr(system, attr, None)
        if isinstance(known, dict):
            for value in known.values():
                value.clear()
    agents = getattr(system, "agents", None)
    if isinstance(agents, dict):
        for agent in agents.values():
            if hasattr(agent, "learned"):
                agent.learned.clear()
    deliveries = getattr(system, "deliveries", None)
    if isinstance(deliveries, list):
        deliveries.clear()


async def phase_ingest(system: MemorySystem, emit: Emit) -> PhaseResult:
    """Write every fact. Measures what storing costs."""
    await emit("phase_start", {"phase": "ingest", "system": system.name,
                               "total": len(FACTS)})
    latencies: List[float] = []
    started = time.perf_counter()

    for i, fact in enumerate(FACTS, 1):
        latency = await system.write(fact)
        latencies.append(latency)
        await emit("ingest_progress", {
            "system": system.name, "done": i, "total": len(FACTS),
            "fact_id": fact.id, "latency_ms": round(latency, 3),
        })

    await system.settle()
    wall = (time.perf_counter() - started) * 1000.0

    result = PhaseResult("ingest", system.name, {
        "facts_written": len(FACTS),
        "median_write_ms": round(_median(latencies), 3),
        "mean_write_ms": round(_mean(latencies), 3),
        "p95_write_ms": round(_p95(latencies), 3),
        "total_wall_ms": round(wall, 1),
        "embed_calls": system.costs.embed_calls,
        "pushes": system.costs.pushes,
    })
    await emit("phase_done", {"phase": "ingest", "system": system.name,
                              "metrics": result.metrics})
    return result


# ------------------------------------------------------------------ phase B

def dedupe(ids: List[str]) -> List[str]:
    """Collapse repeated memory ids, keeping rank order.

    A store can legitimately return the same underlying memory more than once -
    mem0 writes a fresh row per ``add``, so re-ingesting a corpus leaves
    duplicates behind. Counting those as separate hits produced a recall of
    1.36, which is not a number recall can take and is the signature of
    double-counting. Deduping before scoring is also the right semantics: a
    system that returns one answer three times has found one answer.
    """
    return list(dict.fromkeys(i for i in ids if i))


def _score_hits(hit_ids: List[str], relevant: Set[str], k: int) -> Dict[str, float]:
    """Precision, recall and reciprocal rank at k.

    Recall uses min(len(relevant), k) as the denominator: a query with 3
    relevant facts asked at k=2 cannot reach recall 1.0, and punishing the
    system for the harness's own k would be measuring the wrong thing.
    """
    top = dedupe(hit_ids)[:k]
    found = {fid for fid in top if fid in relevant}
    attainable = min(len(relevant), k) or 1
    rr = 0.0
    for rank, fid in enumerate(top, 1):
        if fid in relevant:
            rr = 1.0 / rank
            break
    recall = len(found) / attainable
    if recall > 1.0 + 1e-9:  # unreachable after dedupe; a guard, not a clamp
        raise AssertionError(
            f"recall {recall} > 1 for relevant={sorted(relevant)} top={top}"
        )
    return {
        "precision": len(found) / len(top) if top else 0.0,
        "recall": recall,
        "rr": rr,
        "hits": len(found),
    }


async def phase_retrieval(system: MemorySystem, emit: Emit,
                          k: int = DEFAULT_K) -> PhaseResult:
    """Answer the labelled queries. mem0's home turf."""
    await emit("phase_start", {"phase": "retrieval", "system": system.name,
                               "total": len(QUERIES)})
    precisions: List[float] = []
    recalls: List[float] = []
    rrs: List[float] = []
    latencies: List[float] = []
    detail: List[Dict[str, Any]] = []

    for i, query in enumerate(QUERIES, 1):
        hits, latency = await system.search(query.asked_by, query.text, k)
        hit_ids = dedupe([h.memory_id for h in hits])
        scored = _score_hits(hit_ids, query.relevant, k)

        precisions.append(scored["precision"])
        recalls.append(scored["recall"])
        rrs.append(scored["rr"])
        latencies.append(latency)

        row = {
            "query_id": query.id,
            "query": query.text,
            "relevant": sorted(query.relevant),
            "returned": hit_ids[:k],
            "latency_ms": round(latency, 3),
            **{key: round(val, 4) for key, val in scored.items()},
        }
        detail.append(row)
        await emit("retrieval_progress", {
            "system": system.name, "done": i, "total": len(QUERIES), **row,
        })

    result = PhaseResult("retrieval", system.name, {
        "queries": len(QUERIES),
        "k": k,
        f"precision_at_{k}": round(_mean(precisions), 4),
        f"recall_at_{k}": round(_mean(recalls), 4),
        "mrr": round(_mean(rrs), 4),
        "median_search_ms": round(_median(latencies), 3),
        "mean_search_ms": round(_mean(latencies), 3),
        "p95_search_ms": round(_p95(latencies), 3),
    }, detail)
    await emit("phase_done", {"phase": "retrieval", "system": system.name,
                              "metrics": result.metrics})
    return result


# ------------------------------------------------------------------ phase C

async def phase_propagation(system: MemorySystem, emit: Emit,
                            k: int = DEFAULT_K) -> PhaseResult:
    """Did the agents who needed a fact actually end up with it?

    Scored identically for both systems against the same ground truth, with one
    difference that is the entire point:

      push  the agent was delivered it, at write time, having asked nothing
      pull  the agent must issue a search to have any chance of holding it

    The pull system is therefore given a fair opportunity: each agent runs one
    search per topic it declares an interest in. That is a *generous* reading of
    how a pull architecture is used - it assumes the agent knows to go looking
    and knows what to look for. It still pays a retrieval for each one.
    """
    await emit("phase_start", {"phase": "propagation", "system": system.name,
                               "total": len(AGENTS)})

    searches_before = system.costs.vector_searches

    if system.kind == "pull":
        # Give the pull system its chance to catch up, and charge it the cost.
        for agent_id, topics in AGENTS.items():
            for topic in topics:
                await system.search(agent_id, f"everything about {topic}", k)
            await emit("propagation_progress", {
                "system": system.name, "agent": agent_id,
                "searches": len(topics),
            })
    else:
        await system.settle()
        for agent_id in AGENTS:
            await emit("propagation_progress", {
                "system": system.name, "agent": agent_id, "searches": 0,
            })

    searches_used = system.costs.vector_searches - searches_before

    required = required_deliveries()
    reachable = reachable_deliveries()
    actual: Set[tuple] = set()
    per_agent: List[Dict[str, Any]] = []

    for agent_id in AGENTS:
        known = system.known_to(agent_id)
        needed = {fid for fid, who in required if who == agent_id}
        got = needed & known
        actual |= {(fid, agent_id) for fid in known}
        per_agent.append({
            "agent": agent_id,
            "needed": len(needed),
            "held": len(got),
            "coverage": round(len(got) / len(needed), 4) if needed else float("nan"),
            "missing": sorted(needed - got),
        })

    covered = len(required & actual)
    coverage = covered / len(required) if required else float("nan")

    reachable_covered = len(reachable & actual)

    result = PhaseResult("propagation", system.name, {
        "required_deliveries": len(required),
        "satisfied": covered,
        "coverage": round(coverage, 4),
        # Two denominators, because they answer different questions.
        # "coverage" asks: of everything an agent needed, how much did it get?
        # An interest-based router cannot exceed the ceiling here no matter how
        # well it works. "coverage_of_reachable" asks: of what the declared
        # interests made deliverable, how much actually arrived? That one
        # isolates the transport and routing from the interest declarations.
        "reachable_deliveries": len(reachable),
        "coverage_of_reachable": round(
            reachable_covered / len(reachable), 4) if reachable else float("nan"),
        "retrievals_spent": searches_used,
        "retrievals_per_fact_learned": (
            round(searches_used / covered, 4) if covered else float("nan")
        ),
    }, per_agent)
    await emit("phase_done", {"phase": "propagation", "system": system.name,
                              "metrics": result.metrics, "detail": per_agent})
    return result


# ------------------------------------------------------------------ phase D

async def phase_reactivity(system: MemorySystem, emit: Emit) -> PhaseResult:
    """Facts learned without asking, and how fast they arrived.

    A pull architecture scores zero here by construction, not by failing at
    something. The number is reported because reacting to another agent's write
    is a capability, and a capability nobody has is worth naming.
    """
    await emit("phase_start", {"phase": "reactivity", "system": system.name,
                               "total": 1})

    pushed = getattr(system, "deliveries", [])
    latencies = [d["latency_ms"] for d in pushed if d.get("latency_ms") is not None]

    result = PhaseResult("reactivity", system.name, {
        "learned_without_asking": len(pushed),
        "mean_push_latency_ms": round(_mean(latencies), 3) if latencies else float("nan"),
        "p95_push_latency_ms": round(_p95(latencies), 3) if latencies else float("nan"),
        "can_react_to_another_agents_write": system.kind == "push",
    })
    await emit("phase_done", {"phase": "reactivity", "system": system.name,
                              "metrics": result.metrics})
    return result


# ------------------------------------------------------------------- runner

async def run_all(system: MemorySystem, emit: Emit,
                  k: int = DEFAULT_K) -> Dict[str, PhaseResult]:
    """Every phase, in order, against one system."""
    await emit("system_start", {
        # "mode", not "kind": the event envelope already owns "kind".
        "system": system.name, "label": system.label, "mode": system.kind,
    })
    await system.setup()
    try:
        await phase_warmup(system, emit, k)
        # Order matters. Propagation and reactivity are scored from what each
        # agent knows without having asked, so they must run before any query
        # phase: the retrieval phase's searches teach a pull system things, and
        # crediting those to propagation measures the harness, not the system.
        results = {
            "ingest": await phase_ingest(system, emit),
            "propagation": await phase_propagation(system, emit, k),
            "reactivity": await phase_reactivity(system, emit),
            "retrieval": await phase_retrieval(system, emit, k),
        }
    finally:
        await system.teardown()

    costs = system.costs
    await emit("system_done", {
        "system": system.name,
        "costs": {
            "embed_calls": costs.embed_calls,
            "vector_searches": costs.vector_searches,
            "llm_calls": costs.llm_calls,
            "pushes": costs.pushes,
            "redis_commands_total": costs.redis_total,
            "redis_commands": dict(list(costs.redis_commands.items())[:14]),
        },
        "metrics": {name: r.metrics for name, r in results.items()},
    })
    return results


def comparison_table(results: Dict[str, Dict[str, PhaseResult]],
                     costs: Dict[str, Any], k: int = DEFAULT_K) -> List[Dict[str, Any]]:
    """Flatten both systems' results into dashboard-ready rows."""
    rows: List[Dict[str, Any]] = []
    for system_name, phases in results.items():
        row: Dict[str, Any] = {"system": system_name}
        for phase in phases.values():
            row.update(phase.metrics)
        cost = costs.get(system_name)
        if cost is not None:
            row.update({
                "embed_calls": cost.embed_calls,
                "vector_searches": cost.vector_searches,
                "llm_calls": cost.llm_calls,
                "redis_commands": cost.redis_total,
            })
        rows.append(row)
    return rows


def verdicts(rows: List[Dict[str, Any]], k: int = DEFAULT_K) -> List[Dict[str, str]]:
    """Plain-language readings, including where EventMem does not win."""
    by = {r["system"]: r for r in rows}
    mem0, em = by.get("mem0"), by.get("eventmem")
    out: List[Dict[str, str]] = []
    if not mem0 or not em:
        return out

    def num(row, key):
        v = row.get(key)
        return v if isinstance(v, (int, float)) and v == v else None

    # Retrieval quality: expected to tie.
    mr, er = num(mem0, f"recall_at_{k}"), num(em, f"recall_at_{k}")
    if mr is not None and er is not None:
        delta = er - mr
        if abs(delta) < 0.05:
            out.append({
                "kind": "tie",
                "title": f"Retrieval quality ties (recall@{k} {mr:.2f} vs {er:.2f})",
                "body": "Both run the same embedding model over the same corpus, "
                        "so this is the expected result and the one worth having: "
                        "adding push does not cost retrieval quality.",
            })
        elif delta < 0:
            out.append({
                "kind": "loss",
                "title": f"mem0 retrieves better (recall@{k} {mr:.2f} vs {er:.2f})",
                "body": "mem0's vector store is ranking the labelled answers "
                        "higher than EventMem's linear cosine scan. EventMem's "
                        "record store is a prototype projection, not a tuned "
                        "index; this is a real gap.",
            })
        else:
            out.append({
                "kind": "win",
                "title": f"EventMem retrieves better (recall@{k} {er:.2f} vs {mr:.2f})",
                "body": "Same embedder both sides, so the difference is in how "
                        "each store filters and ranks, not in the model.",
            })

    # Propagation: the architectural difference.
    mc, ec = num(mem0, "coverage"), num(em, "coverage")
    mrt, ert = num(mem0, "retrievals_spent"), num(em, "retrievals_spent")
    if mc is not None and ec is not None:
        from .corpus import routing_ceiling

        ceiling = routing_ceiling()
        out.append({
            "kind": "win" if ec >= mc else "loss",
            "title": f"Propagation coverage {ec:.0%} (EventMem) vs {mc:.0%} (mem0)",
            "body": f"EventMem spent {ert:.0f} retrievals to get there; mem0 spent "
                    f"{mrt:.0f}. The pull system was given one search per declared "
                    "interest, which is a generous reading of how it is used, and "
                    "it still has to know to go looking.",
        })
        mcr = num(mem0, "coverage_of_reachable")
        ecr = num(em, "coverage_of_reachable")
        if ecr is not None:
            out.append({
                "kind": "caveat",
                "title": f"Neither can exceed {ceiling:.0%} on this corpus",
                "body": "16 of the 44 required deliveries name an agent that never "
                        "declared an interest in that fact's topic, so no "
                        "interest-based router can deliver them. Against only the "
                        f"28 reachable ones, EventMem got {ecr:.0%} and mem0 got "
                        f"{mcr:.0%} - that is the number that isolates routing and "
                        "transport from how well the interests were written.",
            })
        if ecr is not None and ecr < 0.999:
            out.append({
                "kind": "loss",
                "title": f"EventMem missed {(1 - ecr):.0%} of what it could have delivered",
                "body": "These are deliveries the subscriptions did match, so the "
                        "gap is in the runtime or the measurement, not in the "
                        "interest declarations. Worth investigating.",
            })

    # Reactivity: structural.
    learned = num(em, "learned_without_asking") or 0
    out.append({
        "kind": "capability",
        "title": f"{learned:.0f} facts reached an agent without it asking",
        "body": "mem0 scores zero here by construction, not by failing: a "
                "passive store has no way to notify. This is a capability "
                "difference, not a performance one.",
    })

    # Cost.
    mredis, eredis = num(mem0, "redis_commands"), num(em, "redis_commands")
    if mredis and eredis:
        out.append({
            "kind": "info",
            "title": f"Redis commands: {eredis:.0f} (EventMem) vs {mredis:.0f} (mem0)",
            "body": "Counted from the server's own INFO commandstats. EventMem "
                    "pays extra commands to push; mem0 pays them to search. "
                    "Which is cheaper depends entirely on the read/write ratio "
                    "of your workload.",
        })

    llm = num(mem0, "llm_calls")
    if llm:
        out.append({
            "kind": "loss",
            "title": f"mem0 made {llm:.0f} LLM calls despite infer=False",
            "body": "Its write latency therefore includes network round-trips "
                    "to a model, and is not comparable to EventMem's. Treat the "
                    "write numbers as invalid for this run.",
        })

    out.append({
        "kind": "caveat",
        "title": "What this does not measure",
        "body": "mem0 ran with infer=False, so its LLM fact-extraction, "
                "contradiction detection and memory-update logic were switched "
                "off. Those are its headline features and EventMem has no "
                "equivalent. This benchmark covers retrieval and propagation "
                "only.",
    })
    return out

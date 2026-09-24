"""Benchmark CLI.

    python -m benchmarks.run                        every workload, every system
    python -m benchmarks.run --workload tool_dedup  one workload
    python -m benchmarks.run --poll-sweep           freshness vs poll interval
    python -m benchmarks.run --json results.json    machine-readable output

The report prints the configuration it actually ran with, including whether
semantic routing was backed by a real model. A number produced on the hashing
fallback is not comparable to one produced on a hosted embedder, and the only
way to keep that straight is to make every result carry its provenance.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import sys
from typing import Any, Dict, List, Optional

from eventmem import HashingEmbedder, __version__

from .harness import Comparison, compare
from .systems import (
    BroadcastSystem,
    EventMemSystem,
    NoSharingSystem,
    OnDemandRetrievalSystem,
    PollingSystem,
    RandomRoutingSystem,
)
from .workloads import REGISTRY

#: Columns worth showing, in reading order. Everything else stays in the JSON.
COLUMNS = [
    ("system", "system", "s"),
    ("transfers", "transfers", "d"),
    ("pushes", "total_deliveries", "d"),
    ("wasted", "wasted_deliveries", "d"),
    ("precision", "routing_precision", ".3f"),
    ("recall", "routing_recall", ".3f"),
    ("F1", "routing_f1", ".3f"),
    ("fresh", "freshness", ".3f"),
    ("consist", "consistency", ".3f"),
    ("tool calls", "tool_calls", "d"),
    ("rho_dup", "duplicate_reduction", ".3f"),
    ("lat ms", "mean_propagation_ms", ".3f"),
    ("wall s", "wall_time_s", ".3f"),
]


def build_systems(embedder: Any, poll_intervals: List[float]) -> List[Any]:
    """The comparison set.

    Ordered floor-first so the table reads as a progression: no coordination,
    then pull at various intervals, then push, then push with filtering.
    """
    systems: List[Any] = [NoSharingSystem(), OnDemandRetrievalSystem()]
    systems += [PollingSystem(interval_s=i) for i in poll_intervals]
    systems += [
        BroadcastSystem(embedder=embedder),
        RandomRoutingSystem(k=2, embedder=embedder),
        EventMemSystem(embedder=embedder),
    ]
    return systems


def format_table(rows: List[Dict[str, Any]]) -> str:
    headers = [label for label, _, _ in COLUMNS]
    table: List[List[str]] = []
    for row in rows:
        line = []
        for _, key, fmt in COLUMNS:
            value = row.get(key)
            if value is None:
                line.append("-")
            elif isinstance(value, float) and value != value:  # NaN
                line.append("n/a")
            elif fmt == "s":
                line.append(str(value))
            else:
                line.append(format(value, fmt))
        table.append(line)

    widths = [
        max(len(headers[i]), max((len(r[i]) for r in table), default=0))
        for i in range(len(headers))
    ]
    out = ["  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))]
    out.append("  ".join("-" * w for w in widths))
    for line in table:
        out.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(line)))
    return "\n".join(out)


def interpret(comparison: Comparison) -> List[str]:
    """Say what the numbers mean, including where EventMem does not win.

    A benchmark that only narrates its own victories is marketing. This
    deliberately calls out ties and losses.
    """
    by_system = comparison.by_system()
    notes: List[str] = []

    em = by_system.get("eventmem")
    bc = by_system.get("broadcast")
    floor = by_system.get("no_sharing")
    if em is None:
        return notes

    base_calls = comparison.baseline_tool_calls
    em_sum = em.summary(base_calls)

    if bc is not None:
        bc_sum = bc.summary(base_calls)
        saved = bc_sum["transfers"] - em_sum["transfers"]
        if bc_sum["transfers"]:
            pct = 100.0 * saved / bc_sum["transfers"]
            notes.append(
                f"vs broadcast: {saved} fewer deliveries ({pct:.0f}%), "
                f"{bc_sum['wasted_deliveries'] - em_sum['wasted_deliveries']} "
                f"fewer wasted wakeups."
            )
        if _close(em_sum["freshness"], bc_sum["freshness"]):
            notes.append(
                "  freshness ties with broadcast, as it should: both push "
                "immediately. The saving is in volume, not in timing."
            )
        elif _lt(em_sum["freshness"], bc_sum["freshness"]):
            notes.append(
                "  freshness is LOWER than broadcast, which means the routing "
                "predicate is dropping updates an agent needed. Check recall."
            )

    polls = sorted(
        (s for s in by_system if s.startswith("polling")),
        key=lambda s: by_system[s].summary(base_calls)["mean_propagation_ms"],
    )
    for name in polls:
        p_sum = by_system[name].summary(base_calls)
        if p_sum["transfers"] == 0:
            notes.append(
                f"vs {name}: its timer never fired inside this run, so it "
                f"learned nothing at all (freshness {p_sum['freshness']:.3f}). "
                "A blank latency cell means no delivery, not a fast one."
            )
        elif _lt(p_sum["freshness"], em_sum["freshness"]):
            notes.append(
                f"vs {name}: freshness {p_sum['freshness']:.3f} -> "
                f"{em_sum['freshness']:.3f}, latency "
                f"{p_sum['mean_propagation_ms']:.2f}ms -> "
                f"{em_sum['mean_propagation_ms']:.3f}ms."
            )
        elif _close(p_sum["freshness"], em_sum["freshness"]):
            notes.append(
                f"vs {name}: freshness ties. A poll this fast keeps up, but see "
                f"its retrieval count for what that costs."
            )

    if floor is not None:
        f_sum = floor.summary(base_calls)
        if f_sum["tool_calls"] > em_sum["tool_calls"]:
            notes.append(
                f"duplicate tool calls: {f_sum['tool_calls']} with no sharing -> "
                f"{em_sum['tool_calls']} with EventMem "
                f"(rho_dup = {em_sum['duplicate_reduction']:.3f}, "
                f"C_base measured from the no_sharing run)."
            )
        elif f_sum["tool_calls"] == em_sum["tool_calls"]:
            notes.append(
                "duplicate tool calls: no saving on this workload. Nothing here "
                "needed the same tool twice."
            )

    rnd = next((s for s in by_system if s.startswith("random")), None)
    if rnd is not None:
        r_sum = by_system[rnd].summary(base_calls)
        if _lt(em_sum["routing_f1"], r_sum["routing_f1"]):
            notes.append(
                f"WARNING: {rnd} scores a higher routing F1 than EventMem. The "
                "predicate is doing no useful work on this workload."
            )

    return notes


def _close(a: Any, b: Any, tol: float = 1e-9) -> bool:
    return _num(a) and _num(b) and abs(a - b) <= tol


def _lt(a: Any, b: Any) -> bool:
    return _num(a) and _num(b) and a < b - 1e-9


def _num(x: Any) -> bool:
    return isinstance(x, (int, float)) and x == x


async def poll_sweep(workload_name: str, embedder: Any) -> Dict[str, Any]:
    """Freshness and cost as a function of poll interval.

    The honest framing of push vs pull: polling can match push on freshness if
    you poll fast enough, and this shows exactly what that costs in retrievals.
    """
    workload = REGISTRY[workload_name]()
    intervals = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25]
    systems = [PollingSystem(interval_s=i) for i in intervals]
    systems.append(EventMemSystem(embedder=embedder))
    comparison = await compare(workload, systems, baseline_system=None)

    print(f"\nPoll-interval sweep on '{workload_name}'")
    print("=" * 60)
    print(
        format_table(
            [
                {
                    **r.summary(),
                    "retrievals": r.extra.get("retrievals", 0),
                }
                for r in comparison.results
            ]
        )
    )
    print(
        "\n  Retrievals (the cost of polling): "
        + ", ".join(
            f"{r.system}={r.extra.get('retrievals', 0)}"
            for r in comparison.results
        )
    )
    print(
        "  Push does zero background retrievals. Polling buys freshness with "
        "them, and the bill grows as the interval shrinks."
    )
    return {"sweep": [r.summary() | r.extra for r in comparison.results]}


async def main_async(args: argparse.Namespace) -> int:
    embedder = HashingEmbedder()
    poll_intervals = [float(x) for x in args.poll_intervals.split(",")]

    print(f"EventMem benchmark  v{__version__}")
    print(f"  python     {platform.python_version()} on {platform.system()}")
    print(f"  embedder   {embedder.name} (semantic={embedder.semantic})")
    if not embedder.semantic:
        print(
            "  NOTE       layer-3 semantic subscriptions are NOT exercised here;\n"
            "             the hashing embedder matches token overlap, not meaning.\n"
            "             These workloads route on type and attributes only, so\n"
            "             the results are unaffected. Run examples/semantic_routing.py\n"
            "             with a real embedder to evaluate layer 3."
        )

    if args.poll_sweep:
        payload = await poll_sweep(args.workload or "support_desk", embedder)
        if args.json:
            _write_json(args.json, payload)
        return 0

    names = [args.workload] if args.workload else list(REGISTRY)
    all_payload: Dict[str, Any] = {"version": __version__, "workloads": {}}

    for name in names:
        workload = REGISTRY[name]()
        systems = build_systems(embedder, poll_intervals)
        comparison = await compare(workload, systems, repeats=args.repeats)

        print(f"\n\nWorkload: {name}")
        print(f"  {workload.description}")
        print(f"  {workload.summary()}")
        print("=" * 110)
        print(format_table(comparison.rows()))

        notes = interpret(comparison)
        if notes:
            print("\nReading the table:")
            for note in notes:
                print(f"  - {note}")

        all_payload["workloads"][name] = {
            "description": workload.description,
            "summary": workload.summary(),
            "rows": comparison.rows(),
            "baseline_tool_calls": comparison.baseline_tool_calls,
            "notes": notes,
        }

    print(
        "\n\nColumn meanings: precision = share of deliveries the recipient "
        "actually needed;\nrecall = share of needed deliveries that happened; "
        "fresh = share of reads returning\nthe current version; rho_dup = tool "
        "calls saved against the no_sharing floor."
    )

    if args.json:
        _write_json(args.json, all_payload)
    return 0


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nWrote {path}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Compare memory architectures.")
    parser.add_argument("--workload", choices=sorted(REGISTRY),
                        help="run one workload instead of all of them")
    parser.add_argument("--repeats", type=int, default=1,
                        help="repeat each run and report the timing spread")
    parser.add_argument("--poll-intervals", default="0.01,0.05,0.2",
                        help="comma-separated poll intervals in seconds")
    parser.add_argument("--poll-sweep", action="store_true",
                        help="sweep the poll interval to show the freshness/cost curve")
    parser.add_argument("--json", help="also write results to this path")
    args = parser.parse_args(argv)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())

"""The benchmark harness.

Executes one workload script against one system, collecting ground-truth
observations as it goes. Every system sees the identical sequence of steps, so
differences in the results come from the architecture and nothing else.

Reproducibility rules this harness follows:

  * The workload is validated before it runs, so a ground-truth mistake fails
    loudly instead of appearing as a system's poor recall.
  * ``C_base`` for duplicate-tool-call reduction comes from an actual
    no-sharing run, not from an assumption.
  * Every run reports which embedder and model it used, because a result from
    the hashing fallback is not comparable to one from a real model.
  * Repeats are supported and reported with a spread, because a single timing
    sample on a loaded machine is not a measurement.
"""

from __future__ import annotations

import asyncio
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from eventmem.metrics import MetricSet, ReadObservation

from .systems.base import System
from .workloads.base import Publish, Read, ToolCall, Wait, Workload


@dataclass
class RunResult:
    """One (system, workload) run."""

    system: str
    workload: str
    metrics: MetricSet
    extra: Dict[str, Any] = field(default_factory=dict)

    def summary(self, baseline_tool_calls: Optional[int] = None) -> Dict[str, Any]:
        out = self.metrics.summary(baseline_tool_calls)
        out["workload"] = self.workload
        out.update(self.extra)
        return out


class Harness:
    def __init__(self, workload: Workload, *, strict: bool = True) -> None:
        problems = workload.validate()
        if problems and strict:
            raise ValueError(
                "workload ground truth is inconsistent:\n  "
                + "\n  ".join(problems)
            )
        self.workload = workload
        self.problems = problems

    async def run(self, system: System) -> RunResult:
        """Execute the script against one system."""
        specs = self.workload.agents()
        metrics = MetricSet(system=system.name)
        started = time.time()

        await system.setup(specs)
        try:
            for step in self.workload.steps():
                await self._execute(step, system, metrics)
            # Stop the clock before teardown. Shutting agents down costs time
            # that belongs to the harness, not to the architecture, and folding
            # it in would make whichever system has the most agents look slowest.
            metrics.wall_time_s = time.time() - started
        finally:
            await system.teardown()
        metrics.held_versions = system.held_versions()
        metrics.authoritative_versions = system.authoritative_versions()
        metrics.actual_deliveries = system.delivered_pairs()
        metrics.required_deliveries = self.workload.required_deliveries()
        system.collect(metrics)

        extra = system.extra_stats() if hasattr(system, "extra_stats") else {}
        return RunResult(system.name, self.workload.name, metrics, dict(extra))

    async def _execute(self, step: Any, system: System, metrics: MetricSet) -> None:
        if isinstance(step, Publish):
            await system.publish(
                actor=step.actor, memory_id=step.memory_id, content=step.content,
                event_type=step.event_type, attributes=step.attributes,
                confidence=step.confidence, base_version=step.base_version,
                reason=step.reason,
            )
            await system.settle(step.settle_s)

        elif isinstance(step, ToolCall):
            # The honest dedup measurement: consult what this agent can
            # actually see right now, then either skip or pay for the call.
            if system.knows_tool_result(step.actor, step.tool_key):
                metrics.tool_calls_avoided += 1
            else:
                metrics.tool_calls += 1
                if step.cost_s:
                    await asyncio.sleep(step.cost_s)
                attributes = {**step.attributes, "tool_key": step.tool_key}
                await system.publish(
                    actor=step.actor, memory_id=step.memory_id, content=step.result,
                    event_type=Publish.__dataclass_fields__["event_type"].default,
                    attributes=attributes,
                )
            await system.settle(step.settle_s)

        elif isinstance(step, Read):
            held = system.holds(step.actor, step.memory_id)
            latest = system.authoritative_version(step.memory_id)
            metrics.reads.append(
                ReadObservation(
                    agent_id=step.actor, memory_id=step.memory_id,
                    held_version=held, latest_version=latest, at=time.time(),
                )
            )
            await system.settle(step.settle_s)

        elif isinstance(step, Wait):
            await system.settle(step.seconds)

        else:
            raise TypeError(f"unknown workload step: {type(step).__name__}")


@dataclass
class Comparison:
    """Results for every system on one workload, with a shared baseline."""

    workload: str
    results: List[RunResult]
    baseline_tool_calls: Optional[int] = None

    def rows(self) -> List[Dict[str, Any]]:
        return [r.summary(self.baseline_tool_calls) for r in self.results]

    def by_system(self) -> Dict[str, RunResult]:
        return {r.system: r for r in self.results}


async def compare(
    workload: Workload,
    systems: List[System],
    *,
    repeats: int = 1,
    baseline_system: Optional[str] = "no_sharing",
) -> Comparison:
    """Run every system on one workload.

    ``baseline_system`` names the run whose tool-call count becomes C_base for
    every other system's duplicate-reduction figure. Defaults to the
    no-sharing floor, which is the only denominator that makes rho_dup mean
    "calls saved by coordinating" rather than "calls this run happened to skip".
    """
    harness = Harness(workload)
    results: List[RunResult] = []

    for system in systems:
        runs = [await harness.run(system) for _ in range(repeats)]
        results.append(_merge(runs) if repeats > 1 else runs[0])

    baseline_calls = None
    for result in results:
        if result.system == baseline_system:
            baseline_calls = result.metrics.tool_calls
            break

    return Comparison(workload.name, results, baseline_calls)


def _merge(runs: List[RunResult]) -> RunResult:
    """Fold repeats into one result.

    Logical metrics (deliveries, versions, tool calls) are deterministic across
    repeats, so the first run is representative. Timings are not, so the median
    is taken and the spread is reported: a mean would let one scheduler hiccup
    dominate, and a single sample would hide it entirely.
    """
    first = runs[0]
    latencies = [r.metrics.runtime.mean_propagation_ms() for r in runs]
    wall_times = [r.metrics.wall_time_s for r in runs]
    first.extra.update(
        {
            "repeats": len(runs),
            "latency_ms_median": round(statistics.median(latencies), 4),
            "latency_ms_min": round(min(latencies), 4),
            "latency_ms_max": round(max(latencies), 4),
            "wall_time_s_median": round(statistics.median(wall_times), 4),
        }
    )
    return first

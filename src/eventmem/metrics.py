"""Memory-centric metrics, all of them actually computed.

Each metric below states its formula and, just as importantly, what it needs in
order to be computable. Several of them (freshness, consistency, duplicate tool
calls) cannot be derived from runtime counters alone: they need either a ground
truth about who should have received what, or a baseline run to compare
against. Those live in ``MetricSet`` and are populated by the benchmark
harness, not by the runtime.

The separation is deliberate. A metric the runtime cannot honestly measure by
itself should not have a counter on the runtime that looks authoritative.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .core.clock import Stamp, elapsed


@dataclass
class Delivery:
    """One (event, recipient) pair and its timings."""

    event_id: str
    memory_id: str
    recipient: str
    published_at: Stamp
    enqueued_at: Stamp
    dequeued_at: Optional[Stamp] = None

    @property
    def propagation(self) -> float:
        """Publish -> arrival in the recipient's inbox.

        The transport cost, which is what the runtime is accountable for. It
        deliberately excludes the time the agent then spends busy before it
        picks the event up.
        """
        return elapsed(self.published_at, self.enqueued_at)[0]

    @property
    def precise(self) -> bool:
        """Whether this sample came from the sub-microsecond clock."""
        return elapsed(self.published_at, self.enqueued_at)[1]

    @property
    def pickup(self) -> Optional[float]:
        """Publish -> the agent actually looked at it.

        Includes queueing behind whatever the agent was doing, so it belongs to
        the agent's workload rather than to the runtime. Reported separately
        precisely so the two are never conflated.
        """
        if self.dequeued_at is None:
            return None
        return elapsed(self.published_at, self.dequeued_at)[0]


@dataclass
class ReadObservation:
    """One agent read of one memory, against the ground truth at that moment.

    Freshness is unmeasurable without this: you need to know what the agent
    held *and* what the authoritative version was at the same instant.
    """

    agent_id: str
    memory_id: str
    held_version: Optional[int]
    latest_version: Optional[int]
    at: float

    @property
    def fresh(self) -> bool:
        return self.held_version is not None and self.held_version == self.latest_version


@dataclass
class RuntimeMetrics:
    """What the runtime can honestly measure about itself.

    No freshness, no consistency, no duplicate-tool-call reduction: those need
    ground truth or a baseline, and a counter here would imply otherwise.
    """

    events_published: int = 0
    events_in_log: int = 0
    conflicts_detected: int = 0
    conflicts_resolved: int = 0
    conflicts_merged: int = 0
    cascades_capped: int = 0
    lifecycle_transitions: int = 0
    embed_calls: int = 0

    deliveries: List[Delivery] = field(default_factory=list)
    #: Recipient-set size per published event, including zeros.
    fan_out_sizes: List[int] = field(default_factory=list)
    #: Number of agents registered, for the broadcast denominator.
    agent_count: int = 0

    # ---------------------------------------------------------------- latency

    def propagation_samples(self) -> List[float]:
        return [d.propagation for d in self.deliveries]

    def mean_propagation_ms(self) -> float:
        """L_prop: mean publish-to-arrival time over all deliveries.

        NaN when nothing was delivered. Reporting 0.0 would let a system that
        propagated nothing at all appear to have the fastest propagation in the
        table, which is the single most flattering misreading available here.
        """
        samples = self.propagation_samples()
        return 1000.0 * statistics.fmean(samples) if samples else float("nan")

    def timing_is_precise(self) -> bool:
        """True when every latency sample came from the sub-microsecond clock.

        False means some samples crossed a process boundary and fell back to
        the wall clock, whose resolution here is
        ``WALL_RESOLUTION_S``. A sub-millisecond mean built from those samples
        is below the noise floor and should not be reported as a measurement.
        """
        return all(d.precise for d in self.deliveries) if self.deliveries else True

    def p95_propagation_ms(self) -> float:
        samples = sorted(self.propagation_samples())
        if not samples:
            return float("nan")
        return 1000.0 * samples[min(int(0.95 * len(samples)), len(samples) - 1)]

    def mean_pickup_ms(self) -> float:
        """Publish-to-dequeue. Includes agent busy time; not a transport metric."""
        samples = [d.pickup for d in self.deliveries if d.pickup is not None]
        return 1000.0 * statistics.fmean(samples) if samples else float("nan")

    def sync_delay_ms(self) -> Dict[str, float]:
        """D_sync per event: time until the *last* recipient had it.

        max over recipients of (arrival - publish). An event delivered to
        nobody has no sync delay and is omitted rather than recorded as zero,
        which would flatter the average.
        """
        per_event: Dict[str, float] = {}
        for d in self.deliveries:
            per_event[d.event_id] = max(per_event.get(d.event_id, 0.0), d.propagation)
        return {k: 1000.0 * v for k, v in per_event.items()}

    def mean_sync_delay_ms(self) -> float:
        delays = list(self.sync_delay_ms().values())
        return statistics.fmean(delays) if delays else float("nan")

    # ---------------------------------------------------------------- routing

    @property
    def total_deliveries(self) -> int:
        return len(self.deliveries)

    def mean_fan_out(self) -> float:
        return statistics.fmean(self.fan_out_sizes) if self.fan_out_sizes else 0.0

    def fan_out_efficiency(self) -> float:
        """eta_fanout = 1 - (actual deliveries) / (broadcast deliveries).

        The broadcast denominator is (N-1) per published event: an agent never
        receives its own write, so N would overstate the baseline and inflate
        the result. Returns 0.0 when there is nothing to compare.
        """
        if not self.fan_out_sizes or self.agent_count < 2:
            return 0.0
        broadcast = len(self.fan_out_sizes) * (self.agent_count - 1)
        return 1.0 - (self.total_deliveries / broadcast) if broadcast else 0.0

    def silent_event_rate(self) -> float:
        """Share of published events that reached nobody.

        High is not automatically bad (some writes are genuinely private) but a
        sudden rise usually means a subscription predicate stopped matching.
        """
        if not self.fan_out_sizes:
            return 0.0
        return sum(1 for n in self.fan_out_sizes if n == 0) / len(self.fan_out_sizes)

    def snapshot(self) -> Dict[str, float]:
        return {
            "events_published": self.events_published,
            "events_in_log": self.events_in_log,
            "total_deliveries": self.total_deliveries,
            "mean_fan_out": round(self.mean_fan_out(), 4),
            "fan_out_efficiency": round(self.fan_out_efficiency(), 4),
            "silent_event_rate": round(self.silent_event_rate(), 4),
            "mean_propagation_ms": _round(self.mean_propagation_ms()),
            "timing_precise": self.timing_is_precise(),
            "p95_propagation_ms": _round(self.p95_propagation_ms()),
            "mean_sync_delay_ms": _round(self.mean_sync_delay_ms()),
            "mean_pickup_ms": _round(self.mean_pickup_ms()),
            "conflicts_detected": self.conflicts_detected,
            "conflicts_resolved": self.conflicts_resolved,
            "cascades_capped": self.cascades_capped,
            "lifecycle_transitions": self.lifecycle_transitions,
        }


@dataclass
class MetricSet:
    """Runtime metrics plus everything that needs ground truth or a baseline.

    Produced by the benchmark harness. This is the object a results table is
    built from.
    """

    system: str
    runtime: RuntimeMetrics = field(default_factory=RuntimeMetrics)

    #: Ground-truth reads, for freshness.
    reads: List[ReadObservation] = field(default_factory=list)

    #: (agent_id, memory_id, version) held by each agent at end of run.
    held_versions: Dict[str, Dict[str, int]] = field(default_factory=dict)
    #: Authoritative version per memory at end of run.
    authoritative_versions: Dict[str, int] = field(default_factory=dict)

    #: Tool calls actually executed, and how many were avoided by a cache hit.
    tool_calls: int = 0
    tool_calls_avoided: int = 0

    #: Ground truth: (event_id, agent_id) pairs that genuinely needed delivery.
    required_deliveries: set = field(default_factory=set)
    #: What was actually delivered.
    actual_deliveries: set = field(default_factory=set)

    llm_calls: int = 0
    wall_time_s: float = 0.0

    # -------------------------------------------------------------- freshness

    def freshness(self) -> float:
        """Fresh = fraction of reads that returned the current version."""
        if not self.reads:
            return float("nan")
        return sum(1 for r in self.reads if r.fresh) / len(self.reads)

    def staleness(self) -> float:
        """Stale = 1 - Fresh."""
        fresh = self.freshness()
        return float("nan") if fresh != fresh else 1.0 - fresh

    # ------------------------------------------------------------ consistency

    def consistency(self) -> float:
        """Cons: mean agreement across agents holding a copy of each memory.

        For each memory, the share of holders that agree on the majority
        version. Memories held by nobody are skipped. Agents that never saw a
        memory are not counted as disagreeing, because not knowing about a fact
        is a coverage problem, not a consistency problem.
        """
        scores: List[float] = []
        for memory_id in self.authoritative_versions:
            holders = [
                versions[memory_id]
                for versions in self.held_versions.values()
                if memory_id in versions
            ]
            if not holders:
                continue
            majority = max(set(holders), key=holders.count)
            scores.append(holders.count(majority) / len(holders))
        return statistics.fmean(scores) if scores else float("nan")

    def coverage(self) -> float:
        """Share of (agent, memory) pairs that *should* be known and are.

        The companion to consistency: perfect agreement among two agents who
        both missed the update is not a good outcome, and this is what exposes
        that.
        """
        required = self.required_deliveries
        if not required:
            return float("nan")
        return len(required & self.actual_deliveries) / len(required)

    # ------------------------------------------------- routing quality vs truth

    def routing_precision(self) -> float:
        """Share of deliveries that the recipient genuinely needed.

        This is what broadcast loses: every wasted delivery is an agent woken,
        a context window spent, and possibly a model call made for nothing.
        """
        if not self.actual_deliveries:
            return float("nan")
        return len(self.actual_deliveries & self.required_deliveries) / len(
            self.actual_deliveries
        )

    def routing_recall(self) -> float:
        """Share of needed deliveries that actually happened.

        This is what an over-tight predicate loses.
        """
        if not self.required_deliveries:
            return float("nan")
        return len(self.actual_deliveries & self.required_deliveries) / len(
            self.required_deliveries
        )

    def routing_f1(self) -> float:
        p, r = self.routing_precision(), self.routing_recall()
        if p != p or r != r or (p + r) == 0:
            return float("nan")
        return 2 * p * r / (p + r)

    def wasted_deliveries(self) -> int:
        """Deliveries to agents that did not need the event."""
        return len(self.actual_deliveries - self.required_deliveries)

    # ---------------------------------------------------- duplicate tool calls

    def duplicate_reduction(self, baseline_tool_calls: Optional[int] = None) -> float:
        """rho_dup = (C_base - C_event) / C_base.

        Requires a baseline. Passing None uses this run's own avoided count,
        i.e. C_base = calls + avoided, which answers "how many of the calls
        this workload wanted did sharing save?" Either way the denominator is
        explicit, because a reduction figure without a stated baseline is not a
        measurement.
        """
        base = (
            baseline_tool_calls
            if baseline_tool_calls is not None
            else self.tool_calls + self.tool_calls_avoided
        )
        if not base:
            return float("nan")
        return (base - self.tool_calls) / base

    # ------------------------------------------------------------------ output

    def summary(self, baseline_tool_calls: Optional[int] = None) -> Dict[str, float]:
        out: Dict[str, float] = {"system": self.system}
        out.update(self.runtime.snapshot())
        out.update(
            {
                # Knowledge transfers, counted the same way for push and pull
                # systems: one (memory, agent) pair that the agent learned
                # about. "total_deliveries" counts pushes only, which is zero
                # for a pull architecture and would make it look free.
                "transfers": len(self.actual_deliveries),
                "freshness": _round(self.freshness()),
                "staleness": _round(self.staleness()),
                "consistency": _round(self.consistency()),
                "coverage": _round(self.coverage()),
                "routing_precision": _round(self.routing_precision()),
                "routing_recall": _round(self.routing_recall()),
                "routing_f1": _round(self.routing_f1()),
                "wasted_deliveries": self.wasted_deliveries(),
                "tool_calls": self.tool_calls,
                "tool_calls_avoided": self.tool_calls_avoided,
                "duplicate_reduction": _round(
                    self.duplicate_reduction(baseline_tool_calls)
                ),
                "llm_calls": self.llm_calls,
                "wall_time_s": round(self.wall_time_s, 4),
            }
        )
        return out


def _round(value: float, places: int = 4) -> float:
    return value if value != value else round(value, places)

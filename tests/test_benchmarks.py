"""The benchmark harness itself.

A benchmark nobody tests is a benchmark nobody should believe. These pin the
properties that make the comparison fair: every system sees the same script,
ground truth is validated before it becomes a result, and the baselines behave
the way their descriptions claim.
"""

from __future__ import annotations

import pytest
from benchmarks.harness import Harness, compare
from benchmarks.systems import (
    BroadcastSystem,
    EventMemSystem,
    NoSharingSystem,
    OnDemandRetrievalSystem,
    PollingSystem,
)
from benchmarks.workloads import (
    AgentSpec,
    NoisyBroadcastWorkload,
    Publish,
    Read,
    ResearchPipelineWorkload,
    SupportDeskWorkload,
    ToolDedupWorkload,
    Workload,
)


class _BadWorkload(Workload):
    name = "bad"

    def agents(self):
        return [AgentSpec("a")]

    def script(self):
        yield Publish(actor="ghost", memory_id="m1", content="x", needed_by={"nobody"})


def test_workload_validation_catches_ground_truth_mistakes():
    """A scripting error must fail loudly, not show up as a system's poor recall."""
    problems = _BadWorkload().validate()
    assert any("unknown actor" in p for p in problems)
    assert any("unknown agent" in p for p in problems)

    with pytest.raises(ValueError, match="ground truth is inconsistent"):
        Harness(_BadWorkload())


def test_an_author_cannot_be_listed_as_needing_its_own_write():
    class SelfNeed(Workload):
        name = "self"

        def agents(self):
            return [AgentSpec("a")]

        def script(self):
            yield Publish(actor="a", memory_id="m1", content="x", needed_by={"a"})

    assert any("its own write" in p for p in SelfNeed().validate())


@pytest.mark.parametrize(
    "workload",
    [
        SupportDeskWorkload(),
        ToolDedupWorkload(),
        ResearchPipelineWorkload(),
        NoisyBroadcastWorkload(n_agents=4, n_events=8),
    ],
    ids=lambda w: w.name,
)
def test_shipped_workloads_have_consistent_ground_truth(workload):
    assert workload.validate() == []


async def test_eventmem_achieves_perfect_routing_on_a_filterable_workload():
    workload = NoisyBroadcastWorkload(n_agents=8, n_events=16, handle_cost_s=0)
    result = await Harness(workload).run(EventMemSystem())
    summary = result.summary()

    assert summary["routing_precision"] == pytest.approx(1.0)
    assert summary["routing_recall"] == pytest.approx(1.0)
    assert summary["freshness"] == pytest.approx(1.0)


async def test_broadcast_matches_recall_but_wastes_deliveries():
    """The core claim, stated as a test: selective routing keeps broadcast's
    coverage while cutting its volume."""
    workload = NoisyBroadcastWorkload(n_agents=8, n_events=16, handle_cost_s=0)
    harness = Harness(workload)

    em = (await harness.run(EventMemSystem())).summary()
    bc = (await harness.run(BroadcastSystem())).summary()

    assert bc["routing_recall"] == pytest.approx(em["routing_recall"])
    assert bc["freshness"] == pytest.approx(em["freshness"])
    assert bc["transfers"] > em["transfers"]
    assert bc["wasted_deliveries"] > em["wasted_deliveries"]
    assert em["wasted_deliveries"] == 0


async def test_no_sharing_establishes_the_duplicate_call_baseline():
    workload = ToolDedupWorkload(n_consumers=3, tool_cost_s=0)
    harness = Harness(workload)

    floor = (await harness.run(NoSharingSystem())).summary()
    em = (await harness.run(EventMemSystem())).summary()

    # Every agent pays with no sharing; only the first pays with EventMem.
    assert floor["tool_calls"] == 4
    assert em["tool_calls"] == 1
    assert em["tool_calls_avoided"] == 3


async def test_duplicate_reduction_is_computed_against_the_measured_floor():
    workload = ToolDedupWorkload(n_consumers=3, tool_cost_s=0)
    comparison = await compare(
        workload, [NoSharingSystem(), EventMemSystem()], baseline_system="no_sharing"
    )

    assert comparison.baseline_tool_calls == 4
    em = comparison.by_system()["eventmem"].summary(comparison.baseline_tool_calls)
    assert em["duplicate_reduction"] == pytest.approx(0.75)


async def test_slow_polling_is_measurably_staler_than_push():
    """The push-vs-pull claim, with a poll interval slow enough to lose."""
    workload = SupportDeskWorkload()
    harness = Harness(workload)

    slow = (await harness.run(PollingSystem(interval_s=0.5))).summary()
    em = (await harness.run(EventMemSystem())).summary()

    assert slow["freshness"] < em["freshness"]
    assert em["freshness"] == pytest.approx(1.0)


async def test_fast_polling_keeps_up_but_pays_in_retrievals():
    """Stated as a test because it is the honest limit of the push claim:
    polling fast enough does match on freshness."""
    workload = SupportDeskWorkload()
    result = await Harness(workload).run(PollingSystem(interval_s=0.005))

    assert result.summary()["freshness"] == pytest.approx(1.0)
    assert result.extra["retrievals"] > 0


async def test_on_demand_retrieval_is_fresh_at_read_time():
    """The RAG baseline's genuine strength must not be scored away."""
    workload = SupportDeskWorkload()
    result = await Harness(workload).run(OnDemandRetrievalSystem())

    assert result.summary()["freshness"] == pytest.approx(1.0)


async def test_conflict_workload_keeps_agents_consistent():
    workload = ResearchPipelineWorkload()
    result = await Harness(workload).run(EventMemSystem())
    summary = result.summary()

    assert summary["conflicts_detected"] >= 1
    assert summary["consistency"] == pytest.approx(1.0)


async def test_every_system_sees_the_identical_script():
    """Fairness check: the workload must not vary between runs."""
    workload = SupportDeskWorkload()
    first = [type(s).__name__ for s in workload.steps()]
    second = [type(s).__name__ for s in workload.steps()]
    assert first == second


async def test_read_steps_produce_freshness_observations():
    workload = SupportDeskWorkload()
    result = await Harness(workload).run(EventMemSystem())

    reads = sum(1 for s in workload.steps() if isinstance(s, Read))
    assert len(result.metrics.reads) == reads


async def test_a_too_narrow_predicate_loses_to_broadcast():
    """The cost of selectivity, pinned as a test.

    If this ever starts passing for EventMem, the workload has stopped
    exercising the failure mode and the suite has become self-congratulatory.
    """
    from benchmarks.workloads import NarrowSubscriptionWorkload

    workload = NarrowSubscriptionWorkload()
    harness = Harness(workload)

    em = (await harness.run(EventMemSystem())).summary()
    bc = (await harness.run(BroadcastSystem())).summary()

    assert em["routing_recall"] < 1.0
    assert em["freshness"] < 1.0
    assert bc["routing_recall"] == pytest.approx(1.0)
    assert bc["freshness"] > em["freshness"]

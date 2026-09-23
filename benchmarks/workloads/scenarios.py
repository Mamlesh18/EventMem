"""Built-in workloads.

Each one isolates a different claim, so a win or loss can be attributed:

    support_desk       a chain where each agent's output is the next one's
                       input. Tests whether knowledge arrives in time to be
                       used at all.
    tool_dedup         several agents need the same expensive tool result.
                       Tests duplicate-tool-call reduction with a real
                       denominator.
    research_pipeline  concurrent edits to a shared artifact. Tests conflict
                       handling and consistency.
    noisy_broadcast    many agents, most events irrelevant to most of them.
                       Tests whether selective routing is worth anything; this
                       is where broadcast should lose.
    narrow_subscription  a predicate too tight to catch what the agent needs.
                       The cost of selectivity; this is where broadcast wins,
                       and it is in the suite so the tradeoff is visible.
"""

from __future__ import annotations

from typing import Iterator, List

from eventmem.core.events import EventType

from .base import AgentSpec, Interest, Publish, Read, Step, ToolCall, Wait, Workload

#: Simulated time between one agent finishing and the next needing the result.
#: Applied identically to every system under test. It has to be non-zero or a
#: pull architecture is given no opportunity to retrieve at all, which would
#: make the comparison a foregone conclusion rather than a measurement. At
#: 30ms, a 10ms poll keeps up and a 200ms poll does not, which is the curve
#: worth showing.
STAGE_GAP_S = 0.03


class SupportDeskWorkload(Workload):
    """A ticket flows through sentiment, triage and drafting.

    Sequential dependency: each stage needs the previous stage's output. A
    system that delivers late produces a stale read at the next stage, which is
    what the freshness metric picks up.
    """

    name = "support_desk"
    description = "sequential agent chain over one customer ticket"

    def agents(self) -> List[AgentSpec]:
        return [
            AgentSpec("intake", publishes_only=True, label="publishes tickets"),
            AgentSpec("sentiment", attributes={"stage": "message"},
                      topic_pattern="ticket.message", label="incoming messages"),
            AgentSpec("triage", attributes={"stage": "sentiment"},
                      topic_pattern="ticket.sentiment", label="sentiment labels"),
            AgentSpec("responder", attributes={"stage": "decision"},
                      topic_pattern="ticket.decision", label="triage decisions"),
            AgentSpec("qa", attributes={"stage": "draft"},
                      topic_pattern="ticket.draft", label="drafts to check"),
        ]

    def script(self) -> Iterator[Step]:
        yield Publish(
            actor="intake", memory_id="ticket.1042",
            content=("This is the worst service ever. I was charged twice and "
                     "nobody will give me a refund. If this is not fixed today "
                     "I am done."),
            event_type=EventType.OBSERVATION_ADDED,
            attributes={"stage": "message", "topic": "ticket.message",
                        "channel": "support"},
            needed_by={"sentiment"}, reason="new ticket", settle_s=STAGE_GAP_S,
        )
        yield Read(actor="sentiment", memory_id="ticket.1042")
        yield Publish(
            actor="sentiment", memory_id="ticket.1042.sentiment",
            content="negative",
            attributes={"stage": "sentiment", "topic": "ticket.sentiment"},
            needed_by={"triage"}, confidence=0.9, reason="classified",
            settle_s=STAGE_GAP_S,
        )
        yield Read(actor="triage", memory_id="ticket.1042.sentiment")
        yield Publish(
            actor="triage", memory_id="ticket.1042.decision",
            content="Priority high. Route to a human agent within the hour.",
            attributes={"stage": "decision", "topic": "ticket.decision"},
            needed_by={"responder"}, confidence=0.9, reason="triaged",
            settle_s=STAGE_GAP_S,
        )
        yield Read(actor="responder", memory_id="ticket.1042.decision")
        yield Publish(
            actor="responder", memory_id="ticket.1042.draft",
            content=("Thank you for reaching out. A specialist will contact you "
                     "shortly to resolve the refund."),
            attributes={"stage": "draft", "topic": "ticket.draft"},
            needed_by={"qa"}, confidence=0.85, reason="drafted", settle_s=STAGE_GAP_S,
        )
        yield Read(actor="qa", memory_id="ticket.1042.draft")


class ToolDedupWorkload(Workload):
    """Four agents all need the same expensive lookup.

    The first agent pays for it and publishes the result. Whether the other
    three pay again depends entirely on whether they learned about it in time,
    which is the measurement. The no-sharing run gives C_base = 4.
    """

    name = "tool_dedup"
    description = "four agents need the same expensive tool result"

    def __init__(self, n_consumers: int = 3, tool_cost_s: float = 0.01) -> None:
        self.n_consumers = n_consumers
        self.tool_cost_s = tool_cost_s

    def agents(self) -> List[AgentSpec]:
        specs = [
            AgentSpec("lead", attributes={"component": "pricing"},
                      topic_pattern="pricing.#", label="pricing work")
        ]
        specs += [
            AgentSpec(
                f"worker{i}", attributes={"component": "pricing"},
                topic_pattern="pricing.#", label="pricing work",
            )
            for i in range(1, self.n_consumers + 1)
        ]
        return specs

    def script(self) -> Iterator[Step]:
        attributes = {"component": "pricing", "topic": "pricing.table"}
        consumers = {f"worker{i}" for i in range(1, self.n_consumers + 1)}

        # The lead pays for the lookup and shares it.
        yield ToolCall(
            actor="lead", tool_key="fetch_pricing_table",
            result="Enterprise tier is $499/seat/month with a 20% annual discount.",
            memory_id="tool.pricing_table", attributes=attributes,
            needed_by=consumers, cost_s=self.tool_cost_s, settle_s=STAGE_GAP_S,
        )
        # Everyone else needs the same thing. Each either sees it or pays again.
        for i in range(1, self.n_consumers + 1):
            yield ToolCall(
                actor=f"worker{i}", tool_key="fetch_pricing_table",
                result="Enterprise tier is $499/seat/month with a 20% annual discount.",
                memory_id="tool.pricing_table", attributes=attributes,
                cost_s=self.tool_cost_s, settle_s=STAGE_GAP_S,
            )
            yield Read(actor=f"worker{i}", memory_id="tool.pricing_table")


class ResearchPipelineWorkload(Workload):
    """Concurrent edits to one shared schema memory.

    Two agents read version 1 and both write back against it. A system with
    conflict detection settles this; one without silently loses a write, which
    shows up as disagreement in the consistency score.
    """

    name = "research_pipeline"
    description = "concurrent edits to a shared artifact"

    def agents(self) -> List[AgentSpec]:
        return [
            AgentSpec("planner", event_types={EventType.TASK_COMPLETED},
                      topic_pattern="build.#", label="completed tasks"),
            AgentSpec(
                "research", event_types={EventType.GOAL_UPDATED},
                topic_pattern="build.goal", label="new goals",
                # Without this second interest research never hears that its
                # own revision was overruled, and goes on believing a version
                # the store has moved past. That is a real failure mode, and
                # NarrowSubscriptionWorkload exists to measure it; here the
                # subject is conflict handling, so the agent is told properly.
                interests=[Interest(attributes={"component": "backend"},
                                    topic_pattern="build.#",
                                    label="backend work")],
            ),
            AgentSpec("coder", attributes={"component": "backend"},
                      topic_pattern="build.#", label="backend work"),
            AgentSpec("reviewer", attributes={"component": "backend"},
                      topic_pattern="build.#", label="backend work"),
        ]

    def script(self) -> Iterator[Step]:
        yield Publish(
            actor="planner", memory_id="goal.booking_api",
            content="Build a booking API for the clinic scheduling service",
            event_type=EventType.GOAL_UPDATED,
            attributes={"topic": "build.goal", "component": "planning"},
            needed_by={"research"}, reason="kick off",
        )
        yield Publish(
            actor="research", memory_id="task.schema_research",
            content="Schema needs slots, patients and clinicians",
            event_type=EventType.TASK_COMPLETED,
            attributes={"topic": "build.task", "component": "backend"},
            needed_by={"planner", "coder"}, reason="research done",
        )
        yield Publish(
            actor="research", memory_id="mem.schema",
            content="Booking schema: slots, patients, clinicians; FK on slot id",
            attributes={"component": "backend", "topic": "build.schema"},
            needed_by={"coder", "reviewer"}, confidence=0.8, reason="record schema",
        )
        yield Read(actor="coder", memory_id="mem.schema")
        yield Read(actor="reviewer", memory_id="mem.schema")

        # Both now believe they hold version 1 and write against it.
        yield Publish(
            actor="research", memory_id="mem.schema",
            content="Schema revision: add a status column to slots",
            event_type=EventType.MEMORY_UPDATED,
            attributes={"component": "backend", "topic": "build.schema"},
            needed_by={"coder", "reviewer"}, confidence=0.6, base_version=1,
            reason="research revision",
        )
        yield Publish(
            actor="reviewer", memory_id="mem.schema",
            content="Schema revision: unique constraint on clinician and time",
            event_type=EventType.MEMORY_UPDATED,
            attributes={"component": "backend", "topic": "build.schema"},
            needed_by={"coder", "research"}, confidence=0.95, base_version=1,
            reason="reviewer revision",
        )
        yield Wait(0.05)
        yield Read(actor="coder", memory_id="mem.schema")
        yield Read(actor="research", memory_id="mem.schema")
        yield Read(actor="reviewer", memory_id="mem.schema")


class NoisyBroadcastWorkload(Workload):
    """Many agents, most events irrelevant to most of them.

    The case selective routing exists for. Each of ``n_agents`` cares about one
    team's events; the script publishes across all teams. Broadcast wakes
    everybody for everything and pays ``handle_cost_s`` each time; a predicate
    router wakes only the team that needs it.

    This is where the architecture should win, and if it does not, the claim is
    wrong.
    """

    name = "noisy_broadcast"
    description = "many agents, low per-event relevance"

    def __init__(self, n_agents: int = 12, n_events: int = 40,
                 handle_cost_s: float = 0.001) -> None:
        self.n_agents = n_agents
        self.n_events = n_events
        self.handle_cost_s = handle_cost_s
        self._teams = ["backend", "frontend", "data", "infra"]

    def _team_of(self, i: int) -> str:
        return self._teams[i % len(self._teams)]

    def agents(self) -> List[AgentSpec]:
        specs = [AgentSpec("publisher", publishes_only=True,
                           label="publishes everything")]
        for i in range(self.n_agents):
            team = self._team_of(i)
            specs.append(
                AgentSpec(
                    f"agent{i}", attributes={"component": team},
                    topic_pattern=f"work.{team}.*", label=f"{team} work",
                    handle_cost_s=self.handle_cost_s,
                )
            )
        return specs

    def script(self) -> Iterator[Step]:
        for n in range(self.n_events):
            team = self._teams[n % len(self._teams)]
            interested = {
                f"agent{i}" for i in range(self.n_agents) if self._team_of(i) == team
            }
            yield Publish(
                actor="publisher", memory_id=f"mem.{n}",
                content=f"Update {n} for the {team} team: status changed.",
                attributes={"component": team, "topic": f"work.{team}.update"},
                needed_by=interested, reason=f"update {n}",
            )
        yield Wait(0.05)
        # Every interested agent then consults what it should have learned.
        for n in range(0, self.n_events, max(1, self.n_events // 8)):
            team = self._teams[n % len(self._teams)]
            for i in range(self.n_agents):
                if self._team_of(i) == team:
                    yield Read(actor=f"agent{i}", memory_id=f"mem.{n}")
                    break


class NarrowSubscriptionWorkload(Workload):
    """An agent whose predicate is too tight to hear what it needs.

    The failure mode selective routing introduces and broadcast cannot have.
    ``auditor`` needs every compliance decision, but subscribes only to the
    finance team's, so it silently misses half of them and then reads a version
    the store has moved past.

    This workload is expected to score *badly* for EventMem on recall and
    freshness, and that is the point: a benchmark that only contains cases its
    subject wins is not measuring anything. Broadcast should beat it here.
    """

    name = "narrow_subscription"
    description = "an over-tight predicate silently drops needed updates"

    def agents(self) -> List[AgentSpec]:
        return [
            AgentSpec("officer", publishes_only=True, label="publishes decisions"),
            AgentSpec(
                "auditor", attributes={"team": "finance"},
                topic_pattern="compliance.finance.*",
                label="finance decisions only (too narrow)",
            ),
        ]

    def script(self) -> Iterator[Step]:
        for n, team in enumerate(("finance", "legal", "finance", "operations")):
            yield Publish(
                actor="officer", memory_id=f"decision.{n}",
                content=f"Compliance decision {n} for the {team} team.",
                attributes={"team": team, "topic": f"compliance.{team}.decision"},
                # The auditor needs all of them. Its subscription says otherwise.
                needed_by={"auditor"}, reason=f"decision {n}",
                settle_s=STAGE_GAP_S,
            )
            yield Read(actor="auditor", memory_id=f"decision.{n}")


REGISTRY = {
    w.name: w
    for w in (
        SupportDeskWorkload,
        ToolDedupWorkload,
        ResearchPipelineWorkload,
        NoisyBroadcastWorkload,
        NarrowSubscriptionWorkload,
    )
}

"""The workload contract.

A workload is a scripted scenario *plus a ground truth*. The ground truth is
the part that makes the benchmark meaningful: for every published fact, the
workload declares which agents genuinely needed to know it. Without that, you
can measure how many messages a system sent but not whether it sent the right
ones, and "fewer deliveries" is indistinguishable from "dropped the update".

A workload is transport- and system-agnostic. It emits abstract steps; the
harness executes them against whichever system is under test. That is what
makes the comparison fair: every system runs the identical script.

To add one, subclass ``Workload``, implement ``agents()`` and ``script()``, and
register it in ``benchmarks/workloads/__init__.py``. See
``docs/benchmarking.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Set

from eventmem.core.events import EventType


@dataclass
class Interest:
    """One thing an agent wants to hear about.

    Stated four ways so every routing ideology can be driven from the same
    spec: predicate routers read ``event_types`` and ``attributes``, topic
    routers read ``topic_pattern``, semantic routers read ``semantic_query``.
    A workload fills in whichever apply.
    """

    event_types: Optional[Set[EventType]] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    semantic_query: Optional[str] = None
    semantic_threshold: Optional[float] = None
    topic_pattern: Optional[str] = None
    label: str = ""

    def matches_attributes(self, attributes: Dict[str, Any]) -> bool:
        """Whether a record satisfies this interest's attribute filter.

        Used by the pull baselines so they retrieve exactly what a push router
        would have routed. Comparing a system that can express an interest
        against one that cannot would measure expressiveness, not architecture.
        """
        return all(attributes.get(k) == v for k, v in self.attributes.items())


@dataclass
class AgentSpec:
    """One agent and everything it is interested in.

    Real agents care about more than one thing, and forcing a single predicate
    makes a workload understate what every architecture can express. The inline
    fields are the short form; ``interests`` holds any extras, and
    ``all_interests`` merges the two.
    """

    agent_id: str
    event_types: Optional[Set[EventType]] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    semantic_query: Optional[str] = None
    semantic_threshold: Optional[float] = None
    topic_pattern: Optional[str] = None
    label: str = ""
    #: Additional interests beyond the inline one.
    interests: List["Interest"] = field(default_factory=list)
    #: This agent only writes; it needs nothing from anyone. Set it explicitly
    #: rather than inferring it from empty filters, because "no filter" and
    #: "no interest" are different statements and guessing between them either
    #: silently drops an agent that wanted everything or hands a pure publisher
    #: every event in the system.
    publishes_only: bool = False
    #: Simulated seconds this agent spends handling one event. Models the cost
    #: of a wasted wakeup, which is what a broadcaster pays and a selective
    #: router avoids.
    handle_cost_s: float = 0.0

    def all_interests(self) -> List["Interest"]:
        """Every interest this agent holds, the inline one first.

        Empty for a publisher-only agent. Otherwise an agent with no filters
        set gets one empty interest, which matches everything -- the honest
        reading of "subscribe, but filter nothing".
        """
        if self.publishes_only:
            return []
        inline = Interest(
            event_types=self.event_types,
            attributes=dict(self.attributes),
            semantic_query=self.semantic_query,
            semantic_threshold=self.semantic_threshold,
            topic_pattern=self.topic_pattern,
            label=self.label,
        )
        return [inline, *self.interests]


@dataclass
class Publish:
    """An agent writes a fact.

    ``needed_by`` is the ground truth: the agents that genuinely require this
    fact to do their job. Routing precision and recall are scored against it,
    so it must reflect the scenario's semantics and not the behaviour of any
    particular router.
    """

    actor: str
    content: str
    memory_id: str
    event_type: EventType = EventType.MEMORY_CREATED
    attributes: Dict[str, Any] = field(default_factory=dict)
    needed_by: Set[str] = field(default_factory=set)
    confidence: float = 1.0
    base_version: Optional[int] = None
    reason: str = ""
    #: Wait this long (simulated seconds) after publishing before the next step.
    settle_s: float = 0.0


@dataclass
class ToolCall:
    """An agent needs the result of an external tool.

    The agent first checks what it already knows. If it holds a result for
    ``tool_key`` that another agent published, it skips the call; otherwise it
    executes and publishes the result. This is the honest way to measure
    duplicate-tool-call reduction: the saving emerges from what each system
    actually propagated, rather than being incremented by the scenario.
    """

    actor: str
    tool_key: str
    result: str
    memory_id: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    needed_by: Set[str] = field(default_factory=set)
    cost_s: float = 0.0
    settle_s: float = 0.0


@dataclass
class Read:
    """An agent consults a memory.

    Scored for freshness: what the agent held versus what was authoritative at
    that instant.
    """

    actor: str
    memory_id: str
    settle_s: float = 0.0


@dataclass
class Wait:
    """Let time pass. Gives pull-based systems a chance to poll."""

    seconds: float


Step = Any  # Publish | ToolCall | Read | Wait


class Workload:
    """Base class. Subclass and implement ``agents`` and ``script``."""

    name: str = "unnamed"
    description: str = ""

    def agents(self) -> List[AgentSpec]:
        raise NotImplementedError

    def script(self) -> Iterator[Step]:
        raise NotImplementedError

    def steps(self) -> List[Step]:
        return list(self.script())

    # ------------------------------------------------------------ ground truth

    def required_deliveries(self) -> Set[tuple]:
        """(memory_id, agent_id) pairs that ought to have been delivered.

        Keyed on memory_id rather than event_id because event ids are generated
        per run and per system, while the workload's ground truth has to be
        stable across every system under test.
        """
        out: Set[tuple] = set()
        for step in self.steps():
            for attr in ("needed_by",):
                needed = getattr(step, attr, None)
                if needed:
                    for agent_id in needed:
                        out.add((step.memory_id, agent_id))
        return out

    def summary(self) -> Dict[str, Any]:
        steps = self.steps()
        return {
            "name": self.name,
            "agents": len(self.agents()),
            "steps": len(steps),
            "publishes": sum(1 for s in steps if isinstance(s, Publish)),
            "tool_calls": sum(1 for s in steps if isinstance(s, ToolCall)),
            "reads": sum(1 for s in steps if isinstance(s, Read)),
            "required_deliveries": len(self.required_deliveries()),
        }

    def validate(self) -> List[str]:
        """Catch ground-truth mistakes before they become results.

        A workload that declares an agent needs a fact it never could have
        received, or that names an agent that does not exist, produces
        recall numbers that look like a system failure but are a script bug.
        """
        problems: List[str] = []
        known = {spec.agent_id for spec in self.agents()}
        seen_memories: Set[str] = set()

        for i, step in enumerate(self.steps()):
            actor = getattr(step, "actor", None)
            if actor is not None and actor not in known:
                problems.append(f"step {i}: unknown actor {actor!r}")
            for agent_id in getattr(step, "needed_by", ()) or ():
                if agent_id not in known:
                    problems.append(f"step {i}: needed_by names unknown agent {agent_id!r}")
                if agent_id == actor:
                    problems.append(
                        f"step {i}: {actor!r} listed as needing its own write; "
                        "authors already know what they wrote"
                    )
            if isinstance(step, (Publish, ToolCall)):
                seen_memories.add(step.memory_id)
            if isinstance(step, Read) and step.memory_id not in seen_memories:
                problems.append(
                    f"step {i}: reads {step.memory_id!r} before anything wrote it"
                )
        return problems

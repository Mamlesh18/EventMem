"""EventMem under test, and the broadcast/random ablations of it.

All three share one implementation and differ only in the router they are given,
which is the cleanest possible ablation: if broadcast beats EventMem on some
metric, nothing but the routing decision can explain it.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Set

from eventmem import (
    Agent,
    ConfidenceThenRecency,
    EventMemRuntime,
    HashingEmbedder,
    MemoryEvent,
    Subscription,
)
from eventmem.metrics import MetricSet
from eventmem.routing.alternatives import BroadcastRouter, RandomRouter
from eventmem.routing.layered import LayeredRouter

from ..workloads.base import AgentSpec
from .base import PublishResult, System


class _BenchAgent(Agent):
    """Agent that records deliveries and simulates handling cost.

    The handling cost is the point of the wasted-wakeup measurement: an agent
    woken for an event it did not need spends real time on it, and a system
    that wakes everybody pays that cost N times per event.
    """

    def __init__(self, agent_id: str, runtime: Any, handle_cost_s: float = 0.0) -> None:
        super().__init__(agent_id, runtime)
        self.handle_cost_s = handle_cost_s
        self.delivered: Set[str] = set()
        self.tool_keys: Set[str] = set()
        self.wakeups = 0

    async def handle(self, event: MemoryEvent) -> None:
        self.wakeups += 1
        self.delivered.add(event.memory_id)
        key = event.attributes.get("tool_key")
        if key:
            self.tool_keys.add(key)
        if self.handle_cost_s:
            await asyncio.sleep(self.handle_cost_s)


class EventMemSystem(System):
    """Push-based selective routing: the architecture under test."""

    name = "eventmem"
    description = "event-driven push, selective predicate routing"

    def __init__(
        self,
        router: Optional[Any] = None,
        embedder: Optional[Any] = None,
        conflict_policy: Optional[Any] = None,
        settle_grace_s: float = 0.02,
    ) -> None:
        self._router_factory = router
        self._embedder = embedder or HashingEmbedder()
        self._conflict_policy = conflict_policy or ConfidenceThenRecency()
        self._settle_grace_s = settle_grace_s
        self.runtime: Optional[EventMemRuntime] = None
        self.agents: Dict[str, _BenchAgent] = {}
        self._specs: Dict[str, AgentSpec] = {}

    def _make_router(self) -> Any:
        if self._router_factory is None:
            return LayeredRouter()
        return self._router_factory() if callable(self._router_factory) else self._router_factory

    async def setup(self, agents: List[AgentSpec]) -> None:
        self.runtime = EventMemRuntime(
            embedder=self._embedder,
            router=self._make_router(),
            conflict_policy=self._conflict_policy,
        )
        for spec in agents:
            self._specs[spec.agent_id] = spec
            agent = _BenchAgent(spec.agent_id, self.runtime, spec.handle_cost_s)
            self.agents[spec.agent_id] = agent

        for spec in agents:
            for subscription in self._subscriptions(spec):
                await self.runtime.subscribe(subscription)

        # Start every agent (which sets its inbox up) before anything publishes.
        for agent in self.agents.values():
            await agent.start()

    def _subscriptions(self, spec: AgentSpec) -> List[Subscription]:
        """One subscription per interest.

        The router de-duplicates recipients, so an agent matching on two of its
        interests is delivered to once and fan-out is not double-counted.
        """
        interests = spec.all_interests()
        if not interests:
            # A publisher-only agent. It still registers, with a subscription
            # whose empty type set matches nothing: the layered router will
            # never route to it, while a router that works from the agent
            # roster (broadcast, random) still knows it exists -- and delivering
            # to it is precisely the cost those routers should be charged.
            return [Subscription(spec.agent_id, event_types=frozenset(),
                                 label="publisher only")]

        out: List[Subscription] = []
        for interest in interests:
            sub = Subscription(
                subscriber_id=spec.agent_id,
                event_types=interest.event_types,
                attribute_filters=dict(interest.attributes),
                semantic_query=interest.semantic_query,
                label=interest.label,
            )
            if interest.semantic_threshold is not None:
                sub.semantic_threshold = interest.semantic_threshold
            # Carried for routers reading them instead of the predicate fields.
            sub.topic_pattern = interest.topic_pattern  # type: ignore[attr-defined]
            out.append(sub)
        return out

    async def publish(
        self, actor: str, memory_id: str, content: str, event_type: Any,
        attributes: Dict[str, Any], confidence: float = 1.0,
        base_version: Optional[int] = None, reason: str = "",
    ) -> PublishResult:
        agent = self.agents[actor]
        before = len(self.runtime.metrics.deliveries)
        event = await agent.remember(
            content, event_type=event_type, memory_id=memory_id,
            attributes=attributes, confidence=confidence,
            base_version=base_version, reason=reason,
        )
        delivered = [
            d.recipient for d in self.runtime.metrics.deliveries[before:]
        ]
        return PublishResult(memory_id, event.committed_version or 0, delivered)

    async def settle(self, seconds: float = 0.0) -> None:
        """Wait until every delivered event has been handled.

        Exact rather than timed: the bus tracks in-flight deliveries, so this
        returns as soon as the work is genuinely done. A fixed sleep would be
        either a race or a tax on every measured wall time.
        """
        if seconds:
            await asyncio.sleep(seconds)
        await self.runtime.bus.drain(timeout=10.0)
        for agent in self.agents.values():
            await agent.wait_idle()

    def holds(self, agent_id: str, memory_id: str) -> Optional[int]:
        return self.agents[agent_id].held_versions.get(memory_id)

    def authoritative_version(self, memory_id: str) -> Optional[int]:
        record = self.runtime.records.get(memory_id, count_access=False)
        return record.version if record else None

    def knows_tool_result(self, agent_id: str, tool_key: str) -> bool:
        return tool_key in self.agents[agent_id].tool_keys

    def delivered_pairs(self) -> Set[tuple]:
        return {
            (d.memory_id, d.recipient) for d in self.runtime.metrics.deliveries
        }

    def held_versions(self) -> Dict[str, Dict[str, int]]:
        return {a.id: dict(a.held_versions) for a in self.agents.values()}

    def authoritative_versions(self) -> Dict[str, int]:
        return {
            r.memory_id: r.version
            for r in self.runtime.records.all(include_dead=True)
        }

    def collect(self, metrics: MetricSet) -> None:
        metrics.runtime = self.runtime.metrics

    async def teardown(self) -> None:
        for agent in self.agents.values():
            await agent.stop()
        if self.runtime is not None:
            await self.runtime.close()


class BroadcastSystem(EventMemSystem):
    """Push everything to everyone. The null hypothesis for selective routing.

    Identical to EventMem except for the router, so any difference in the
    results is attributable to routing and nothing else. Expect it to tie on
    freshness and lose badly on delivery volume and wasted wakeups.
    """

    name = "broadcast"
    description = "event-driven push, no filtering (every event to every agent)"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(router=BroadcastRouter, **kwargs)


class RandomRoutingSystem(EventMemSystem):
    """Deliver to a random subset. The control.

    If selective routing cannot beat random selection at the same fan-out, the
    predicate is not doing any work and the routing claim is unsupported.
    """

    name = "random"
    description = "push to a random subset of the same size (control)"

    def __init__(self, k: int = 2, seed: int = 7, **kwargs: Any) -> None:
        super().__init__(router=lambda: RandomRouter(k=k, seed=seed), **kwargs)
        self.name = f"random(k={k})"

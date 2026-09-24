"""Alternative routing ideologies.

These exist so the central claim is falsifiable. If selective predicate routing
is better than broadcasting, that has to be measured against an actual
broadcaster running the same workload through the same runtime, not asserted.

Each class here is a complete Router. Copy the shortest one as a starting point
for your own ideology.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Optional, Set

from ..core.events import MemoryEvent


class BroadcastRouter:
    """Every event to every agent except its author.

    The null hypothesis. Maximum freshness by construction, maximum delivery
    volume and maximum wasted wakeups. Registering with this router needs only
    an agent id, so any subscription object is accepted and ignored.
    """

    name = "broadcast"

    def __init__(self) -> None:
        self._agents: List[str] = []

    async def register(self, subscription: Any) -> None:
        agent_id = getattr(subscription, "subscriber_id", subscription)
        if agent_id not in self._agents:
            self._agents.append(agent_id)

    def unregister(self, subscriber_id: str) -> None:
        self._agents = [a for a in self._agents if a != subscriber_id]

    def route(self, event: MemoryEvent) -> List[str]:
        return [a for a in self._agents if a != event.source_agent]

    def subscribers(self) -> List[str]:
        return list(self._agents)


class TopicRouter:
    """Hierarchical topic matching, in the style of a message broker.

    An agent subscribes to ``support.ticket.*`` or ``support.#``; an event
    carries a topic in its attributes. No embeddings, no vectors, fully
    deterministic and explainable, which is exactly what some deployments want
    instead of a similarity threshold.

        *  matches one segment
        #  matches one or more trailing segments
    """

    name = "topic"

    def __init__(self, attribute: str = "topic") -> None:
        self.attribute = attribute
        self._patterns: List[tuple] = []  # (agent_id, pattern segments)

    async def register(self, subscription: Any) -> None:
        agent_id = getattr(subscription, "subscriber_id", None)
        pattern = getattr(subscription, "topic_pattern", None)
        if agent_id is None or pattern is None:
            raise ValueError(
                "TopicRouter needs a subscription with .subscriber_id and .topic_pattern"
            )
        self._patterns.append((agent_id, tuple(pattern.split("."))))

    def unregister(self, subscriber_id: str) -> None:
        self._patterns = [p for p in self._patterns if p[0] != subscriber_id]

    def route(self, event: MemoryEvent) -> List[str]:
        topic = event.attributes.get(self.attribute)
        if not isinstance(topic, str):
            return []
        segments = tuple(topic.split("."))
        out: List[str] = []
        seen: Set[str] = set()
        for agent_id, pattern in self._patterns:
            if agent_id == event.source_agent or agent_id in seen:
                continue
            if _topic_match(segments, pattern):
                seen.add(agent_id)
                out.append(agent_id)
        return out

    def subscribers(self) -> List[str]:
        return list(dict.fromkeys(a for a, _ in self._patterns))


def _topic_match(segments: tuple, pattern: tuple) -> bool:
    """Standard broker wildcard semantics, resolved recursively."""
    if not pattern:
        return not segments
    head, rest = pattern[0], pattern[1:]
    if head == "#":
        # '#' must consume at least one segment, then anything after must match.
        return any(_topic_match(segments[i:], rest) for i in range(1, len(segments) + 1))
    if not segments:
        return False
    if head == "*" or head == segments[0]:
        return _topic_match(segments[1:], rest)
    return False


class PredicateRouter:
    """Routing by arbitrary Python callables.

    The escape hatch. When an ideology is easier to write as code than to
    express declaratively, register ``(agent_id, lambda event: bool)``. Useful
    for capability matching, load shedding, or anything stateful.
    """

    name = "predicate"

    def __init__(self) -> None:
        self._rules: List[tuple] = []

    async def register(self, subscription: Any) -> None:
        agent_id = getattr(subscription, "subscriber_id", None)
        fn = getattr(subscription, "predicate", None)
        if agent_id is None or not callable(fn):
            raise ValueError(
                "PredicateRouter needs a subscription with .subscriber_id and "
                "a callable .predicate"
            )
        self._rules.append((agent_id, fn))

    def unregister(self, subscriber_id: str) -> None:
        self._rules = [r for r in self._rules if r[0] != subscriber_id]

    def route(self, event: MemoryEvent) -> List[str]:
        out: List[str] = []
        seen: Set[str] = set()
        for agent_id, fn in self._rules:
            if agent_id == event.source_agent or agent_id in seen:
                continue
            if fn(event):
                seen.add(agent_id)
                out.append(agent_id)
        return out

    def subscribers(self) -> List[str]:
        return list(dict.fromkeys(a for a, _ in self._rules))


class RandomRouter:
    """Deliver to a random subset of a fixed size.

    A control. If a real router cannot beat random selection of the same fan-out
    on routing precision, its predicate is not doing any work. Seeded, so runs
    are reproducible.
    """

    name = "random"

    def __init__(self, k: int = 1, seed: int = 0) -> None:
        self.k = k
        self._rng = random.Random(seed)
        self._agents: List[str] = []

    async def register(self, subscription: Any) -> None:
        agent_id = getattr(subscription, "subscriber_id", subscription)
        if agent_id not in self._agents:
            self._agents.append(agent_id)

    def unregister(self, subscriber_id: str) -> None:
        self._agents = [a for a in self._agents if a != subscriber_id]

    def route(self, event: MemoryEvent) -> List[str]:
        pool = [a for a in self._agents if a != event.source_agent]
        return self._rng.sample(pool, min(self.k, len(pool)))

    def subscribers(self) -> List[str]:
        return list(self._agents)


class CompositeRouter:
    """Union of several routers.

    Lets a deployment run a declarative rule set alongside a semantic one, or
    migrate ideologies incrementally by running the new router beside the old
    and comparing their recipient sets.
    """

    name = "composite"

    def __init__(self, routers: List[Any], name: Optional[str] = None) -> None:
        self._routers = routers
        self.name = name or "+".join(r.name for r in routers)

    async def register(self, subscription: Any) -> None:
        """Offer the subscription to each router; those that cannot use it
        decline by raising, which is not an error here."""
        accepted = False
        for router in self._routers:
            try:
                await router.register(subscription)
                accepted = True
            except (ValueError, TypeError):
                continue
        if not accepted:
            raise ValueError("no sub-router accepted this subscription")

    def unregister(self, subscriber_id: str) -> None:
        for router in self._routers:
            router.unregister(subscriber_id)

    def route(self, event: MemoryEvent) -> List[str]:
        out: List[str] = []
        seen: Set[str] = set()
        for router in self._routers:
            for agent_id in router.route(event):
                if agent_id not in seen:
                    seen.add(agent_id)
                    out.append(agent_id)
        return out

    def subscribers(self) -> List[str]:
        return list(dict.fromkeys(a for r in self._routers for a in r.subscribers()))


#: Name -> factory. Routers needing constructor arguments are built directly.
REGISTRY: Dict[str, Callable[[], Any]] = {
    "broadcast": BroadcastRouter,
    "topic": TopicRouter,
    "predicate": PredicateRouter,
    "random": RandomRouter,
}

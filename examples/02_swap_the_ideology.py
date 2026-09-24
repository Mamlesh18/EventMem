"""Changing the architecture's ideology without changing the runtime.

The same five agents and the same workload, routed four different ways. Each
run differs by one constructor argument.

    python examples/02_swap_the_ideology.py

This is the file to read before writing your own router: it shows what the
seam actually costs you (about ten lines) and what it buys (a falsifiable
comparison instead of an assertion).
"""

import asyncio
from typing import List

from eventmem import (
    Agent,
    BroadcastRouter,
    EventMemRuntime,
    LayeredRouter,
    MemoryEvent,
    PredicateRouter,
    RandomRouter,
    Subscription,
    TopicRouter,
)

EVENTS = [
    ("Database migration applied to the bookings table", "work.backend.db", "backend"),
    ("Checkout button restyled for mobile", "work.frontend.ui", "frontend"),
    ("Nightly ETL job failed on the pricing feed", "work.data.etl", "data"),
    # Tagged backend, but infra needs to see it. A tag-based rule cannot
    # express that; a predicate can.
    ("URGENT: production certificate expires in two hours", "work.backend.certs",
     "backend"),
]

TEAMS = ["backend", "frontend", "data", "infra"]


class Counting(Agent):
    def __init__(self, agent_id, runtime):
        super().__init__(agent_id, runtime)
        self.woke_for: List[str] = []

    async def handle(self, event: MemoryEvent) -> None:
        self.woke_for.append(event.memory_id)


async def run(router, describe: str, subscribe) -> None:
    runtime = EventMemRuntime(router=router)
    publisher = Agent("publisher", runtime)
    agents = {team: Counting(team, runtime) for team in TEAMS}

    for team in TEAMS:
        await subscribe(runtime, team)
    for agent in agents.values():
        await agent.start()

    for i, (content, topic, team) in enumerate(EVENTS):
        await publisher.remember(
            content, memory_id=f"mem.{i}",
            attributes={"component": team, "topic": topic},
        )
    await runtime.bus.drain()

    m = runtime.metrics
    woke = {t: len(a.woke_for) for t, a in agents.items()}
    print(f"\n{describe}")
    print(f"  router            {router.name}")
    print(f"  deliveries        {m.total_deliveries}")
    print(f"  fan-out           {m.mean_fan_out():.2f} of {m.agent_count - 1} possible")
    print(f"  vs broadcast      {m.fan_out_efficiency():.0%} fewer deliveries")
    print(f"  wakeups per agent {woke}")

    for agent in agents.values():
        await agent.stop()
    await runtime.close()


async def main() -> None:
    print("Same agents, same events, four routing ideologies.")
    print("Only the router argument changes.")

    # 1. The default: a layered predicate on event type, attributes, meaning.
    await run(
        router=LayeredRouter(),
        describe="1. Layered predicate (the default)",
        subscribe=lambda rt, team: rt.subscribe(
            Subscription(team, attribute_filters={"component": team})
        ),
    )

    # 2. Broker-style topic trees. Deterministic, explainable, no vectors.
    async def topic_sub(rt, team):
        sub = Subscription(team)
        sub.topic_pattern = f"work.{team}.*"
        await rt.subscribe(sub)

    await run(
        router=TopicRouter(),
        describe="2. Hierarchical topics (work.<team>.*)",
        subscribe=topic_sub,
    )

    # 3. Arbitrary Python. Here: wake the infra team on anything urgent,
    #    regardless of which team the event was tagged for.
    async def predicate_sub(rt, team):
        sub = Subscription(team)
        if team == "infra":
            sub.predicate = lambda e: (
                e.attributes.get("component") == "infra"
                or "urgent" in e.content.lower()
            )
        else:
            sub.predicate = lambda e, t=team: e.attributes.get("component") == t
        await rt.subscribe(sub)

    await run(
        router=PredicateRouter(),
        describe="3. Arbitrary predicates (infra also wakes on anything urgent)",
        subscribe=predicate_sub,
    )

    # 4. Broadcast: the null hypothesis. Everyone hears everything.
    await run(
        router=BroadcastRouter(),
        describe="4. Broadcast (the null hypothesis)",
        subscribe=lambda rt, team: rt.subscribe(Subscription(team)),
    )

    # 5. Random at fixed fan-out: the control. If a real router cannot beat
    #    this, its predicate is not doing any work.
    await run(
        router=RandomRouter(k=1, seed=3),
        describe="5. Random at the same fan-out (the control)",
        subscribe=lambda rt, team: rt.subscribe(Subscription(team)),
    )

    print("""
Read the wakeup counts, not just the delivery totals. Broadcast reaches
everyone who needed an event, and also everyone who did not: four agents woken
per event instead of one. Random matches the selective fan-out and wakes the
wrong agents, which is why it belongs in the comparison. The predicate router
shows the case declarative filters handle badly, where the rule is a
disjunction over content rather than a tag.

To write your own: implement register/unregister/route/subscribers, then pass
it as router=. Nothing else in the runtime changes.""")


if __name__ == "__main__":
    asyncio.run(main())

"""
demo.py

A small story that exercises the whole runtime. Five agents work on one task.
Nobody polls. The runtime pushes each change to exactly the agents that asked
for it. Run this file directly to watch it happen.

    python demo.py

What the scenario shows, in order:
    1. selective push      a goal reaches only the research agent
    2. structured routing  a completed task reaches the code and planner agents
    3. semantic routing     a risky line of code reaches the reviewer, matched
                            by meaning, with no keyword filter set for it
    4. duplicate work saved the execution agent reads shared memory and skips a
                            redundant tool call because the artifact is ready
    5. conflict resolution  two agents edit the same memory at once and the
                            runtime settles it by confidence
"""

from __future__ import annotations

import asyncio

from agent import Agent
from events import EventType
from runtime import EventMemRuntime
from subscriptions import Subscription


class ReactiveAgent(Agent):
    """An agent that narrates what it reacts to, so the demo is readable."""

    async def on_event(self, event) -> None:
        self.received.append(event)
        content = event.payload.get("content", "")
        print(f"    -> [{self.id}] reacted to {event.event_type.value} "
              f"from {event.source_agent}: {content}")


def banner(text: str) -> None:
    print()
    print(text)
    print("-" * len(text))


async def main() -> None:
    runtime = EventMemRuntime()

    planner = ReactiveAgent("planner", runtime)
    research = ReactiveAgent("research", runtime)
    code = ReactiveAgent("code", runtime)
    reviewer = ReactiveAgent("reviewer", runtime)
    execution = ReactiveAgent("execution", runtime)

    # Subscriptions. This is where each agent declares what it cares about.
    # Planner wants to know when any task finishes.
    await runtime.subscribe(Subscription(
        subscriber_id="planner",
        event_types={EventType.TASK_COMPLETED},
        label="any completed task",
    ))
    # Research reacts to new goals.
    await runtime.subscribe(Subscription(
        subscriber_id="research",
        event_types={EventType.GOAL_UPDATED},
        label="new goals",
    ))
    # Code wants completed research on the api schema, and any backend memory.
    await runtime.subscribe(Subscription(
        subscriber_id="code",
        event_types={EventType.TASK_COMPLETED},
        attribute_filters={"topic": "api_schema"},
        label="finished api schema research",
    ))
    await runtime.subscribe(Subscription(
        subscriber_id="code",
        attribute_filters={"component": "backend"},
        label="anything in the backend",
    ))
    # Reviewer subscribes by meaning. No keyword rule. It wants anything that
    # reads like a security or safety concern.
    await runtime.subscribe(Subscription(
        subscriber_id="reviewer",
        semantic_query="security vulnerability unsafe user input sql injection risk",
        semantic_threshold=0.12,
        label="security concerns, matched semantically",
    ))
    # Execution deploys anything marked ready.
    await runtime.subscribe(Subscription(
        subscriber_id="execution",
        attribute_filters={"deployable": "true"},
        label="deployable artifacts",
    ))

    for agent in (planner, research, code, reviewer, execution):
        agent.start()

    # ----------------------------------------------------------- 1. a goal
    banner("Step 1  Planner sets a goal. Only Research subscribed to goals.")
    await planner.remember(
        "Build a booking API for the clinic scheduling service",
        event_type=EventType.GOAL_UPDATED,
        memory_id="goal.booking_api",
        attributes={"topic": "api_schema"},
        reason="kick off the task",
    )
    await asyncio.sleep(0.03)

    # -------------------------------------------- 2. research finishes work
    banner("Step 2  Research finishes and records the schema. Code and Planner "
           "hear it. Nobody polled.")
    await research.remember(
        "Investigated the schedule endpoints, the schema needs slots, patients, and clinicians",
        event_type=EventType.TASK_COMPLETED,
        memory_id="task.schema_research",
        attributes={"topic": "api_schema"},
        reason="research done",
    )
    await research.remember(
        "Booking schema, tables for slots patients clinicians with a foreign key on slot id",
        event_type=EventType.MEMORY_CREATED,
        memory_id="mem.schema",
        attributes={"component": "backend", "topic": "api_schema"},
        confidence=0.8,
        reason="record the schema",
    )
    await asyncio.sleep(0.03)

    # ------------------------------------ 3. code writes a risky line
    banner("Step 3  Code writes backend code with a risky line. Reviewer never "
           "set a keyword filter, yet the runtime routes it there by meaning.")
    await code.remember(
        "Booking handler builds the sql query by concatenating raw user input from the request",
        event_type=EventType.MEMORY_CREATED,
        memory_id="mem.booking_handler",
        attributes={"component": "backend", "deployable": "true"},
        confidence=0.9,
        reason="first cut of the handler",
    )
    await asyncio.sleep(0.03)

    # ------------------------------------ 4. execution avoids duplicate work
    banner("Step 4  Execution needs the handler before deploying. It reads "
           "shared memory first and finds it, so it skips a redundant build.")
    ready = await execution.recall("booking handler backend", component="backend")
    if any(r.memory_id == "mem.booking_handler" for r in ready):
        runtime.metrics.duplicate_work_avoided += 1
        print("    -> [execution] found the handler already in shared memory, "
              "skipped rebuilding it")
    else:
        print("    -> [execution] handler missing, would have to build it")
    await asyncio.sleep(0.03)

    # ------------------------------------ 5. concurrent edit and resolution
    banner("Step 5  Research and Reviewer edit the same schema memory at once. "
           "Both read version 1. The runtime detects the collision and settles it.")
    # Both declare base_version 1, the version they read. Reviewer writes with
    # higher confidence, so the confidence policy should pick the reviewer.
    await research.remember(
        "Schema revision, add a status column to slots",
        event_type=EventType.MEMORY_UPDATED,
        memory_id="mem.schema",
        attributes={"component": "backend"},
        confidence=0.6,
        base_version=1,
        reason="research revision",
    )
    await reviewer.remember(
        "Schema revision, slots need a unique constraint on clinician and time",
        event_type=EventType.MEMORY_UPDATED,
        memory_id="mem.schema",
        attributes={"component": "backend"},
        confidence=0.95,
        base_version=1,
        reason="reviewer revision",
    )
    await asyncio.sleep(0.03)

    winner = runtime.memory.get("mem.schema")
    print(f"    -> winning schema content: {winner.content}")
    print(f"    -> winning confidence: {winner.confidence}, version now {winner.version}")

    # ------------------------------------------------------------- metrics
    banner("Runtime metrics")
    m = runtime.metrics
    print(f"    events published        {m.events_published}")
    print(f"    total deliveries        {m.total_deliveries}")
    print(f"    average fan out         {m.avg_fan_out():.2f} agents per event "
          f"(out of {len(runtime._agent_ids)})")
    print(f"    average propagation     {m.avg_latency_ms():.3f} ms per delivery")
    print(f"    conflicts detected      {m.conflicts_detected}")
    print(f"    conflicts resolved      {m.conflicts_resolved}")
    print(f"    duplicate work avoided  {m.duplicate_work_avoided}")
    print(f"    events in the log       {len(runtime.event_store)}")

    for agent in (planner, research, code, reviewer, execution):
        await agent.stop()


if __name__ == "__main__":
    asyncio.run(main())

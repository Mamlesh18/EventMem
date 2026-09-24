"""The smallest useful EventMem program.

Three agents. Nobody polls. A write reaches exactly the agents whose
subscription matches it.

    python examples/01_quickstart.py
"""

import asyncio

from eventmem import Agent, ConsoleTracer, EventMemRuntime, EventType, Subscription


class Narrating(Agent):
    """Prints what it reacts to, so the push is visible."""

    async def handle(self, event):
        print(f"    -> [{self.id}] woke on {event.event_type.value}: {event.content}")


async def main() -> None:
    runtime = EventMemRuntime(tracer=ConsoleTracer())

    writer = Agent("writer", runtime)
    backend = Narrating("backend", runtime)
    frontend = Narrating("frontend", runtime)

    # Each agent declares what it cares about. This is the entire coordination
    # protocol: there is no message passing between agents anywhere below.
    await runtime.subscribe(Subscription(
        "backend", attribute_filters={"component": "backend"},
        label="backend work",
    ))
    await runtime.subscribe(Subscription(
        "frontend", attribute_filters={"component": "frontend"},
        label="frontend work",
    ))

    # start() sets each inbox up before returning, so no early event is lost.
    await backend.start()
    await frontend.start()

    print("\n1. A backend change. Only the backend agent should wake.\n")
    await writer.remember(
        "Added a unique index on bookings(clinician_id, starts_at)",
        memory_id="mem.index",
        attributes={"component": "backend"},
        reason="schema change",
    )
    await runtime.bus.drain()

    print("\n2. A frontend change. Only the frontend agent should wake.\n")
    await writer.remember(
        "Booking form now shows clinician availability inline",
        memory_id="mem.form",
        attributes={"component": "frontend"},
        reason="ui change",
    )
    await runtime.bus.drain()

    print("\n3. A goal nobody subscribed to. It is still recorded.\n")
    await writer.remember(
        "Ship the booking API by Friday",
        event_type=EventType.GOAL_UPDATED,
        memory_id="goal.ship",
        reason="planning",
    )
    await runtime.bus.drain()

    print("\nAnyone can still read what was written, on demand:")
    for record in await runtime.recall("booking index clinician", k=2):
        print(f"    {record.memory_id} v{record.version}: {record.content}")

    m = runtime.metrics
    print(f"""
Metrics
    events published    {m.events_published}
    deliveries          {m.total_deliveries}
    mean fan-out        {m.mean_fan_out():.2f} of {m.agent_count - 1} possible
    fan-out efficiency  {m.fan_out_efficiency():.1%} fewer than broadcast
    silent events       {m.silent_event_rate():.1%} reached nobody
    mean propagation    {m.mean_propagation_ms():.3f} ms""")

    await backend.stop()
    await frontend.stop()
    await runtime.close()


if __name__ == "__main__":
    asyncio.run(main())

"""Proof that EventMem is event-driven, not polling in disguise.

"Nobody polls" is easy to claim and easy to get wrong, so this measures it
instead. Three experiments:

  1. IDLE       Agents sit for 1.5s with nothing happening. Count how many
                times they query the memory store. A pull architecture would
                show one read per agent per interval.

  2. WAKE       Publish one event and measure how long each agent takes to
                react. Polling latency is bounded below by the poll interval;
                push latency is bounded by the transport.

  3. CAUSATION  Follow a three-hop cascade and show, from the recorded causal
                graph, that each agent ran *because* of the previous event —
                not because a timer fired.

Run:
    python examples/04_prove_event_driven.py
"""

import asyncio
import time

from eventmem import Agent, EventMemRuntime, EventType, Subscription
from eventmem.core.clock import elapsed, now
from eventmem.store.memory import InMemoryRecordStore


class CountingStore(InMemoryRecordStore):
    """Record store that counts every read.

    This is the instrument for experiment 1. If agents were pulling, these
    counters would climb while the system is idle.
    """

    def __init__(self) -> None:
        super().__init__()
        self.gets = 0
        self.searches = 0

    def get(self, memory_id, *, count_access=True):
        # The runtime's own conflict check passes count_access=False; that is
        # an internal read, not an agent querying memory, so exclude it or the
        # instrument measures itself.
        if count_access:
            self.gets += 1
        return super().get(memory_id, count_access=count_access)

    def search(self, *args, **kwargs):
        self.searches += 1
        return super().search(*args, **kwargs)

    @property
    def reads(self) -> int:
        return self.gets + self.searches


class Timed(Agent):
    """Agent that records when it woke, and how often its loop ticked."""

    def __init__(self, agent_id, runtime):
        super().__init__(agent_id, runtime)
        self.wake_stamps = []
        self.loop_ticks = 0

    async def handle(self, event):
        self.wake_stamps.append((event, now()))


class TickCountingInbox:
    """Wraps an inbox to count how often the blocking read timed out.

    Honesty check: the agent loop passes a timeout to `get` so it can notice a
    shutdown request. That timer is the one thing in the design that looks like
    a poll, so it gets counted and reported rather than glossed over.
    """

    def __init__(self, inner):
        self._inner = inner
        self.timeouts = 0
        self.deliveries = 0

    async def setup(self):
        return await self._inner.setup()

    async def get(self, timeout: float = 1.0):
        event = await self._inner.get(timeout)
        if event is None:
            self.timeouts += 1
        else:
            self.deliveries += 1
        return event

    async def ack(self, event):
        return await self._inner.ack(event)

    async def nack(self, event):
        return await self._inner.nack(event)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def banner(text):
    print(f"\n{text}\n{'=' * len(text)}")


async def experiment_1_idle(runtime, store, agents, seconds=1.5):
    banner(f"1. IDLE - {seconds}s with nothing published")

    reads_before = store.reads
    started = time.perf_counter()
    await asyncio.sleep(seconds)
    real = time.perf_counter() - started

    reads_after = store.reads
    ticks = sum(a.inbox.timeouts for a in agents)

    print(f"  agents idle for            {real:.2f}s")
    print(f"  memory-store reads         {reads_after - reads_before}")
    print(f"  events delivered           {sum(a.inbox.deliveries for a in agents)}")
    print(f"  blocking reads that timed out  {ticks}")
    print()
    if reads_after == reads_before:
        print("  -> No agent queried memory even once while idle. There is no")
        print("     retrieval loop: an agent that is not sent anything does")
        print("     nothing at all.")
    else:
        print(f"  -> {reads_after - reads_before} reads happened while idle. "
              "Something IS polling.")
    print()
    print(f"  Being precise about those {ticks} timeouts: each agent's blocking")
    print("  read takes a 0.25s timeout so it can observe a shutdown request.")
    print("  That timer touches no memory and asks no question about state --")
    print("  it is a liveness check on the agent's own stop flag. On Redis the")
    print("  equivalent is XREADGROUP BLOCK, which sleeps inside Redis.")


async def experiment_2_wake(runtime, store, writer, agents):
    banner("2. WAKE - publish one event, measure the reaction")

    reads_before = store.reads
    published = await writer.remember(
        "Payment webhook signature is never verified",
        memory_id="mem.webhook",
        attributes={"component": "backend"},
        reason="code review finding",
    )
    await runtime.bus.drain()

    print(f"  published at   t0 (event {published.event_id[:8]})")
    for agent in agents:
        if not agent.wake_stamps:
            print(f"  {agent.id:<10} did not wake (subscription did not match)")
            continue
        _, stamp = agent.wake_stamps[-1]
        seconds, precise = elapsed(published.published_at, stamp)
        clock = "precise clock" if precise else "wall clock"
        print(f"  {agent.id:<10} woke {seconds * 1000:8.3f} ms after publish  ({clock})")

    print()
    print(f"  memory-store reads caused by this delivery: {store.reads - reads_before}")
    print("  -> The agent received the content in the event itself. It did not")
    print("     have to go back and fetch anything to find out what changed.")


async def experiment_3_causation(runtime, writer):
    banner("3. CAUSATION - a three-hop cascade, read from the causal graph")

    class Relay(Agent):
        """Reacts by publishing, tagging the cause so the chain is recoverable."""

        def __init__(self, agent_id, rt, output_stage, message):
            super().__init__(agent_id, rt)
            self.output_stage = output_stage
            self.message = message

        async def handle(self, event):
            await self.remember(
                self.message,
                memory_id=f"mem.{self.output_stage}",
                attributes={"stage": self.output_stage},
                reason=f"reacted to {event.source_agent}",
                # These two lines are what make the cascade a *causal* chain
                # rather than a coincidence: depth bounds it, caused_by records
                # what triggered it.
                depth=event.depth + 1,
                caused_by=event.event_id,
            )

    triage = Relay("triage", runtime, "decision", "Priority high, page the on-call")
    responder = Relay("responder", runtime, "reply", "Customer notified, fix in progress")

    await runtime.subscribe(Subscription("triage", attribute_filters={"stage": "alert"}))
    await runtime.subscribe(Subscription("responder", attribute_filters={"stage": "decision"}))
    await triage.start()
    await responder.start()

    root = await writer.remember(
        "Webhook endpoint returning 500 for 12% of requests",
        event_type=EventType.OBSERVATION_ADDED,
        memory_id="mem.alert",
        attributes={"stage": "alert"},
        reason="monitoring alert",
    )
    await runtime.bus.drain()

    final = runtime.provenance.history("mem.reply")[-1]
    chain = runtime.provenance.lineage(final.event_id)

    print("  Derivation of the final reply, reconstructed from recorded causes:")
    for i, node in enumerate(chain):
        arrow = "" if i == 0 else "  " * i + "+- "
        print(f"    {arrow}{node.agent} ({node.event_type}, depth {node.depth})")
        print(f"    {'  ' * i}   {node.content[:60]}")

    print()
    print(f"  root cause: {runtime.provenance.root_cause(final.event_id).agent}")
    print(f"  blast radius if the root is retracted: "
          f"{sorted(runtime.provenance.contaminated_memories(root.event_id))}")
    print()
    print("  -> Every hop records the event that caused it. Nothing here was")
    print("     scheduled, and nothing checked whether the previous stage had")
    print("     finished: each agent ran because it was handed a specific event.")

    await triage.stop()
    await responder.stop()


async def main():
    store = CountingStore()
    runtime = EventMemRuntime(record_store=store)

    writer = Agent("monitor", runtime)
    backend = Timed("backend", runtime)
    security = Timed("security", runtime)
    frontend = Timed("frontend", runtime)
    watchers = [backend, security, frontend]

    # Wrap each inbox so loop timeouts are visible.
    for agent in watchers:
        agent.inbox = TickCountingInbox(agent.inbox)

    await runtime.subscribe(Subscription(
        "backend", attribute_filters={"component": "backend"}))
    await runtime.subscribe(Subscription(
        "security", attribute_filters={"component": "backend"}))
    await runtime.subscribe(Subscription(
        "frontend", attribute_filters={"component": "frontend"}))

    for agent in watchers:
        await agent.start()

    print("EventMem: is it really event-driven?")
    print("Three agents are running and subscribed. Nothing has happened yet.")

    await experiment_1_idle(runtime, store, watchers)
    await experiment_2_wake(runtime, store, writer, watchers)

    for agent in watchers:
        await agent.stop()

    await experiment_3_causation(runtime, writer)

    banner("Summary")
    m = runtime.metrics
    print(f"  events published        {m.events_published}")
    print(f"  events in the log       {m.events_in_log}")
    print(f"  deliveries              {m.total_deliveries}")
    print(f"  mean propagation        {m.mean_propagation_ms():.3f} ms")
    print(f"  timing from precise clock  {m.timing_is_precise()}")
    print(f"  total agent reads of memory  {store.reads}")
    print()
    print("  Reads stayed at zero except where an agent explicitly asked a")
    print("  question. Everything else arrived because it was pushed.")

    await runtime.close()


if __name__ == "__main__":
    asyncio.run(main())

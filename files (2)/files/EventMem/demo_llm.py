"""
demo_llm.py

A working proof with real parts. Set the Azure and Redis environment variables
and this runs on your models and your Redis. Leave them unset and it runs on a
mock model and an in memory bus, so the wiring is visible either way.

The scenario is a support desk. One customer message sets off a chain that no
agent drives by polling:

    intake      publishes the raw message
    sentiment   reads it, writes a sentiment label            (pushed to triage)
    triage      reads the label, decides an action            (pushed to responder)
    responder   recalls the original message, drafts a reply  (chain ends)
    escalation  watches by meaning, wakes only on an angry message

Run it:
    python demo_llm.py
"""

from __future__ import annotations

import asyncio

from agent import Agent, LLMAgent
from config import build_models, build_transport
from events import EventType, MemoryEvent
from runtime import EventMemRuntime
from subscriptions import Subscription
from tracing import ConsoleTracer

SENTIMENT_PROMPT = (
    "You are a sentiment classifier for customer support. Read the customer "
    "message and reply with exactly one word: positive, negative, or neutral."
)
TRIAGE_PROMPT = (
    "You are a support triage agent. Given a customer mood label, state the "
    "priority and the single next action in one short sentence."
)
RESPONDER_PROMPT = (
    "You draft a short, warm customer support reply. Two or three sentences. "
    "Acknowledge the problem and say what happens next."
)
ESCALATION_PROMPT = (
    "You are an escalation monitor. In one sentence, say whether this message "
    "shows an angry or at risk customer and why."
)


class ResponderAgent(LLMAgent):
    """Pulls the original message from shared memory, then drafts on top of the
    triage decision. Shows recall and push working together in one agent."""

    async def on_event(self, event: MemoryEvent) -> None:
        self.received.append(event)
        decision = event.payload.get("content", "")
        originals = await self.recall("customer support message refund", stage="message")
        original_text = originals[0].content if originals else ""
        prompt = (f"Original customer message: {original_text}\n"
                  f"Triage decision: {decision}\n\nWrite the reply.")
        self.runtime.metrics.llm_calls += 1
        answer = await self.llm.chat(self.system_prompt, prompt, max_tokens=self.max_tokens)
        print(f"    THINK   [responder] {answer}")
        await self.remember(
            answer, EventType.MEMORY_CREATED, attributes={"stage": "draft"},
            reason="drafted reply", depth=event.depth + 1, caused_by=event.event_id,
        )


async def wait_for_draft(runtime: EventMemRuntime, timeout: float = 40.0) -> None:
    """The demo waits here for the chain to finish. This is the harness waiting,
    not an agent polling."""
    steps = int(timeout / 0.1)
    for _ in range(steps):
        await asyncio.sleep(0.1)
        if any(r.attributes.get("stage") == "draft" for r in runtime.memory.all()):
            return


async def main() -> None:
    models = build_models()
    tracer = ConsoleTracer()
    bus, store, transport_label, cleanup = await build_transport(tracer)

    print("EventMem live run")
    print("=================")
    print(f"  model      {models.llm_name}")
    print(f"  embeddings {models.embedder_name}")
    print(f"  transport  {transport_label}")
    print()

    runtime = EventMemRuntime(
        embedder=models.embedder, bus=bus, event_store=store, tracer=tracer,
    )

    intake = Agent("intake", runtime)
    sentiment = LLMAgent(
        "sentiment", runtime, models.llm, SENTIMENT_PROMPT,
        publishes_as=EventType.MEMORY_CREATED, publish_attributes={"stage": "sentiment"},
        max_tokens=10,
    )
    triage = LLMAgent(
        "triage", runtime, models.llm, TRIAGE_PROMPT,
        publishes_as=EventType.MEMORY_CREATED, publish_attributes={"stage": "decision"},
        max_tokens=80,
    )
    responder = ResponderAgent(
        "responder", runtime, models.llm, RESPONDER_PROMPT, max_tokens=120,
    )
    escalation = LLMAgent(
        "escalation", runtime, models.llm, ESCALATION_PROMPT, max_tokens=60,
    )

    # Subscriptions. The first three route by stage, a plain attribute match.
    # The last routes by meaning, with no keyword rule at all.
    await runtime.subscribe(Subscription(
        "sentiment", event_types={EventType.OBSERVATION_ADDED},
        attribute_filters={"stage": "message"}, label="incoming messages"))
    await runtime.subscribe(Subscription(
        "triage", attribute_filters={"stage": "sentiment"}, label="sentiment labels"))
    await runtime.subscribe(Subscription(
        "responder", attribute_filters={"stage": "decision"}, label="triage decisions"))
    await runtime.subscribe(Subscription(
        "escalation",
        semantic_query="angry furious customer urgent complaint refund unacceptable worst churn",
        semantic_threshold=0.06, label="anything that reads like an angry customer"))

    agents = (intake, sentiment, triage, responder, escalation)
    for agent in agents:
        agent.start()
    # Make sure every inbox is set up before the first event is delivered.
    await asyncio.gather(*(agent.ready() for agent in agents))

    message = ("This is the worst service ever. I was charged twice and nobody "
               "will give me a refund. If this is not fixed today I am done.")
    print("Customer message arrives. Watch it flow.\n")
    await intake.remember(
        message, event_type=EventType.OBSERVATION_ADDED,
        memory_id="ticket.1042", attributes={"stage": "message", "channel": "support"},
        reason="new support ticket",
    )

    await wait_for_draft(runtime)

    print("\nRuntime metrics")
    print("---------------")
    m = runtime.metrics
    print(f"    events published        {m.events_published}")
    print(f"    llm calls               {m.llm_calls}")
    print(f"    total deliveries        {m.total_deliveries}")
    print(f"    average fan out         {m.avg_fan_out():.2f} agents per event "
          f"(out of {len(runtime._agent_ids)})")
    print(f"    average propagation     {m.avg_latency_ms():.2f} ms per delivery")
    print(f"    events in the log       {m.events_in_log}")

    for agent in agents:
        await agent.stop()
    await cleanup()


if __name__ == "__main__":
    asyncio.run(main())

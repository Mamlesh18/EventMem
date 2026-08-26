# EventMem

An event driven memory runtime for collaborative LLM agents. Memory is not a
database agents query. It is a runtime that pushes each change to the agents
that care about it, the moment it happens.

There are two ways to run it. One is a scripted in memory demo that shows the
routing, conflict handling, and lifecycle. The other is a live demo where real
agents reason with a model, real embeddings drive the semantic routing, and
Redis carries every event.

## Quick check, no setup

    python demo.py

Runs five agents through one task on an in memory bus. Good for seeing the
routing and the conflict resolution without any keys or services.

## Live run with real parts

1. Start Redis.

       docker compose up -d

2. Install the two client libraries.

       pip install -r requirements.txt

3. Set your Azure OpenAI details. Copy .env.example to .env and fill it in, or
   export the variables in your shell.

       AZURE_OPENAI_ENDPOINT
       AZURE_OPENAI_API_KEY
       AZURE_OPENAI_API_VERSION            default 2025-01-01-preview
       AZURE_OPENAI_CHAT_DEPLOYMENT
       AZURE_OPENAI_EMBEDDING_DEPLOYMENT
       REDIS_URL                           default redis://localhost:6379

4. Run it.

       python demo_llm.py

You will see a customer message flow through a chain of agents. Each hop is a
model call triggered by a push. The trace prints every publish, the agents it
routed to, and each delivery, with timing.

If Azure variables are missing, it falls back to a mock model. If Redis is not
reachable, it falls back to an in memory bus. The same file runs either way, so
you can prove the wiring first and add the real parts after.

## What the live demo shows

    intake      publishes the raw customer message
    sentiment   reads it, writes a sentiment label            pushed to triage
    triage      reads the label, decides an action            pushed to responder
    responder   recalls the original message, drafts a reply  chain ends
    escalation  watches by meaning, wakes only on an angry message

Nobody polls. Each agent reacts to an event that was pushed to it because its
subscription matched. The responder also reads shared memory on demand to pull
the original message, which shows pull and push working together.

## Files

    events.py           the shared vocabulary, MemoryEvent and MemoryRecord
    embeddings.py       hashing fallback and the Azure embedder
    llm.py              the Azure chat wrapper and a mock for keyless runs
    serde.py            turns events into JSON for Redis and back
    event_store.py      in memory append only log
    redis_bus.py        Redis transport, per agent streams and the durable log
    bus.py              in memory transport, same interface as Redis
    semantic_memory.py  the searchable projection built from the log
    subscriptions.py    the three layer matching engine, the core of routing
    conflict.py         optimistic concurrency resolution policy
    tracing.py          the console tracer that makes the push visible
    runtime.py          the orchestrator and the publish pipeline
    agent.py            base agent and the LLM backed agent
    config.py           picks real or fallback parts from the environment
    demo.py             the scripted in memory demo
    demo_llm.py         the live cascade with a model and Redis

## How an agent subscribes

A subscription is a rule matched in three layers, cheapest first.

    layer 1  event type    a set membership test
    layer 2  attributes    exact matches on metadata such as stage=message
    layer 3  semantics      cosine similarity against an interest vector

The escalation agent in the live demo uses only layer 3, so it wakes on
anything that reads like an angry customer without naming a single keyword.

## Why the reads block but do not poll

Each agent waits on its inbox. On Redis that is XREADGROUP with BLOCK, which
sleeps inside Redis until an event is added, then returns at once. The short
timeout on the read exists only so an agent can notice a shutdown. No agent ever
loops asking whether something changed.

## Notes on correctness

Redis delivery uses a consumer group per agent, created before any event is
delivered. That gives exactly once handling per agent and lets an agent resume
after a restart without missing or repeating events. The event log is a Redis
stream, so the whole history can be replayed to rebuild the projection.

A depth ceiling bounds reasoning cascades. An agent that reacts to a depth n
event and publishes a result stamps it depth n plus one. The runtime stops
fanning out past the ceiling, so LLM agents cannot loop forever by reacting to
each other.

# EventMem

An event driven memory runtime for collaborative LLM agents. Memory here is not
a database agents query. It is a runtime that pushes each change to the agents
that care about it, the moment it happens.

## Run it

    python demo.py

No keys, no external services. The demo runs five agents through one task and
prints what reaches whom.

## The idea in one paragraph

An agent that learns something publishes an event to the runtime. The runtime
embeds it, checks it against the current state for conflicts, appends it to an
immutable log, updates a searchable projection, then delivers it only to the
agents whose subscriptions match. Agents react to what arrives. They do not
poll.

## Files

    events.py           the shared vocabulary, MemoryEvent and MemoryRecord
    embeddings.py       pluggable vectoriser, dependency free default
    event_store.py      append only log, the source of truth
    semantic_memory.py  the searchable projection built from the log
    subscriptions.py    the three layer matching engine, the core of routing
    conflict.py         optimistic concurrency resolution policy
    bus.py              per agent delivery
    runtime.py          the orchestrator and the publish pipeline
    agent.py            base agent, publishes and reacts
    demo.py             a runnable five agent scenario

## How an agent subscribes

A subscription is a predicate matched in three layers, cheapest first.

    layer 1  event type    a set membership test
    layer 2  attributes    exact matches on metadata such as component=backend
    layer 3  semantics      cosine similarity against an interest vector

An agent can use any combination. The reviewer in the demo uses only layer 3,
so it hears about anything that reads like a security concern without naming a
single keyword.

## The publish pipeline

    1. embed       give the event a vector for semantic routing
    2. conflict    for updates, compare declared version against current
    3. persist     append the immutable event to the log
    4. project     upsert or delete in the semantic store
    5. lifecycle   advance the record's stage when warranted
    6. fan out     match subscriptions and deliver to interested agents

## What is real and what is a seam

Real in this prototype: the full pipeline, three layer subscription matching,
optimistic concurrency conflict detection and resolution, event sourcing with a
replayable log, lifecycle states, and live measurement of propagation latency
and fan out.

Swap points for production, each behind a small interface:

    embeddings   HashingEmbedder -> OpenAI or a sentence transformer
    event bus    InMemoryEventBus -> Redis Streams or Kafka
    event store  InMemoryEventStore -> Redis Streams plus Postgres
    projection   SemanticMemory -> ChromaDB, Qdrant, or pgvector
    conflict     ConfidenceThenRecency -> CRDT merges for set valued memory

## Metrics the runtime measures

Propagation latency and fan out are measured directly. The runtime also counts
conflicts detected and resolved, deliveries, and duplicate work avoided. These
map onto the memory centric metrics in the scope document, and give an
evaluation story that ordinary agent frameworks cannot report.

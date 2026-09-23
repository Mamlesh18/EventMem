"""EventMem: an event-driven shared memory runtime for collaborative AI agents.

Memory as an active coordination service rather than a store agents poll. Every
mutation is an event, and the runtime pushes it to exactly the agents whose
subscriptions match.

    from eventmem import Agent, EventMemRuntime, EventType, Subscription

    runtime = EventMemRuntime()
    reviewer = Agent("reviewer", runtime)
    await runtime.subscribe(Subscription("reviewer", event_types={EventType.TASK_COMPLETED}))
    await reviewer.start()

Every part of the architecture is a constructor argument: the router (the
routing ideology), the conflict policy, the embedder, the transport, the event
store, the record store and the lifecycle policy. See ``docs/extending.md``.
"""

from .agents import Agent, LLMAgent
from .conflict.base import ConflictError, Resolution
from .conflict.policies import (
    ConfidenceThenRecency,
    FirstWriterWins,
    LastWriterWins,
    PreferAgents,
    SetUnionMerge,
)
from .core.events import (
    EventType,
    LifecycleState,
    MemoryEvent,
    MemoryRecord,
    Provenance,
)
from .embeddings.backends import (
    AzureEmbedder,
    CachingEmbedder,
    HashingEmbedder,
    OpenAIEmbedder,
    SentenceTransformerEmbedder,
)
from .lifecycle import AgeAndUsePolicy, LifecycleManager, NeverExpire, TTLPolicy
from .llm.backends import AzureLLM, EchoLLM, MockLLM, OpenAILLM
from .metrics import MetricSet, RuntimeMetrics
from .provenance import ProvenanceTracker
from .routing.alternatives import (
    BroadcastRouter,
    CompositeRouter,
    PredicateRouter,
    RandomRouter,
    TopicRouter,
)
from .routing.layered import Calibration, LayeredRouter, Subscription
from .runtime import EventMemRuntime
from .store.memory import InMemoryEventStore, InMemoryRecordStore
from .tracing import ConsoleTracer, JSONLTracer, MultiTracer, NullTracer
from .transport.memory import InMemoryEventBus

__version__ = "0.2.0"

__all__ = [
    # core
    "EventMemRuntime", "Agent", "LLMAgent",
    "MemoryEvent", "MemoryRecord", "EventType", "LifecycleState", "Provenance",
    # routing
    "Subscription", "LayeredRouter", "Calibration",
    "BroadcastRouter", "TopicRouter", "PredicateRouter", "RandomRouter",
    "CompositeRouter",
    # conflict
    "ConfidenceThenRecency", "LastWriterWins", "FirstWriterWins",
    "PreferAgents", "SetUnionMerge", "Resolution", "ConflictError",
    # embeddings
    "HashingEmbedder", "SentenceTransformerEmbedder", "AzureEmbedder",
    "OpenAIEmbedder", "CachingEmbedder",
    # llm
    "MockLLM", "EchoLLM", "AzureLLM", "OpenAILLM",
    # services
    "LifecycleManager", "AgeAndUsePolicy", "NeverExpire", "TTLPolicy",
    "ProvenanceTracker",
    # storage & transport
    "InMemoryEventStore", "InMemoryRecordStore", "InMemoryEventBus",
    # observability
    "RuntimeMetrics", "MetricSet",
    "NullTracer", "ConsoleTracer", "JSONLTracer", "MultiTracer",
    "__version__",
]

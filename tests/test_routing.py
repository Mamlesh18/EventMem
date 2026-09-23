"""Routing: the layered predicate, alternatives, and threshold calibration."""

from __future__ import annotations

import warnings

import pytest

from eventmem import (
    Agent,
    BroadcastRouter,
    EventMemRuntime,
    EventType,
    HashingEmbedder,
    MemoryEvent,
    RandomRouter,
    Subscription,
    TopicRouter,
)
from eventmem.routing.alternatives import CompositeRouter, PredicateRouter, _topic_match
from eventmem.routing.layered import LayeredRouter


def _event(content="hello", **attributes) -> MemoryEvent:
    return MemoryEvent(
        event_type=EventType.MEMORY_CREATED, source_agent="src",
        memory_id="m1", payload={"content": content}, attributes=attributes,
    )


# --------------------------------------------------------------- layer tests

async def test_layer_one_filters_by_event_type():
    router = LayeredRouter()
    await router.register(
        Subscription("a", event_types={EventType.TASK_COMPLETED})
    )

    assert router.route(_event()) == []
    task = _event()
    task.event_type = EventType.TASK_COMPLETED
    assert router.route(task) == ["a"]


async def test_layer_two_requires_every_attribute_to_match():
    router = LayeredRouter()
    await router.register(
        Subscription("a", attribute_filters={"component": "backend", "env": "prod"})
    )

    assert router.route(_event(component="backend")) == []
    assert router.route(_event(component="backend", env="prod")) == ["a"]


async def test_layers_compose_as_a_conjunction():
    router = LayeredRouter()
    await router.register(
        Subscription(
            "a", event_types={EventType.TASK_COMPLETED},
            attribute_filters={"topic": "pricing"},
        )
    )

    wrong_type = _event(topic="pricing")
    assert router.route(wrong_type) == []

    right = _event(topic="pricing")
    right.event_type = EventType.TASK_COMPLETED
    assert router.route(right) == ["a"]


async def test_semantic_layer_fails_closed_without_an_embedding():
    """A subscription that asked for semantic matching and cannot do it must
    drop the event, never widen to match everything."""
    router = LayeredRouter(embedder=HashingEmbedder())
    await router.register(Subscription("a", semantic_query="security risk"))

    event = _event("sql injection in the login handler")
    event.embedding = None  # never went through the pipeline
    assert router.route(event) == []


async def test_an_agent_with_two_matching_subscriptions_is_delivered_to_once():
    router = LayeredRouter()
    await router.register(Subscription("a", attribute_filters={"x": "1"}))
    await router.register(Subscription("a", attribute_filters={"y": "2"}))

    assert router.route(_event(x="1", y="2")) == ["a"]


async def test_subscription_stats_explain_why_a_rule_never_fires():
    router = LayeredRouter()
    sub = Subscription("a", attribute_filters={"component": "backend"}, label="be")
    await router.register(sub)

    router.route(_event(component="frontend"))
    router.route(_event(component="frontend"))
    router.route(_event(component="backend"))

    assert sub.stats["matched"] == 1
    assert sub.stats["rejected_attribute"] == 2


# ----------------------------------------------------------- alternative ideologies

@pytest.mark.parametrize(
    ("topic", "pattern", "expected"),
    [
        ("work.backend.update", "work.backend.*", True),
        ("work.backend.update", "work.*.update", True),
        ("work.backend.update", "work.#", True),
        ("work.backend.update", "work.frontend.*", False),
        ("work.backend", "work.backend.*", False),
        ("work.backend.deep.update", "work.#", True),
        ("work", "work.#", False),  # '#' must consume at least one segment
    ],
)
def test_topic_wildcards(topic, pattern, expected):
    assert _topic_match(tuple(topic.split(".")), tuple(pattern.split("."))) is expected


async def test_topic_router_routes_by_pattern():
    router = TopicRouter()
    sub = Subscription("a")
    sub.topic_pattern = "work.backend.*"
    await router.register(sub)

    assert router.route(_event(topic="work.backend.update")) == ["a"]
    assert router.route(_event(topic="work.frontend.update")) == []


async def test_broadcast_router_reaches_everyone_but_the_author():
    router = BroadcastRouter()
    for agent_id in ("a", "b", "src"):
        await router.register(Subscription(agent_id))

    assert set(router.route(_event())) == {"a", "b"}


async def test_random_router_is_reproducible_for_a_seed():
    a = RandomRouter(k=2, seed=42)
    b = RandomRouter(k=2, seed=42)
    for router in (a, b):
        for agent_id in ("w", "x", "y", "z"):
            await router.register(Subscription(agent_id))

    assert a.route(_event()) == b.route(_event())


async def test_predicate_router_takes_arbitrary_callables():
    router = PredicateRouter()
    sub = Subscription("a")
    sub.predicate = lambda e: len(e.content) > 5
    await router.register(sub)

    assert router.route(_event("hi")) == []
    assert router.route(_event("a much longer message")) == ["a"]


async def test_composite_router_unions_two_ideologies():
    topic = TopicRouter()
    predicate = PredicateRouter()
    router = CompositeRouter([topic, predicate])

    topic_sub = Subscription("by_topic")
    topic_sub.topic_pattern = "work.#"
    await router.register(topic_sub)

    pred_sub = Subscription("by_predicate")
    pred_sub.predicate = lambda e: "urgent" in e.content
    await router.register(pred_sub)

    matched = router.route(_event("urgent problem", topic="work.backend"))
    assert set(matched) == {"by_topic", "by_predicate"}


async def test_a_router_is_a_constructor_argument():
    """Swapping the routing ideology must not require touching the runtime."""
    runtime = EventMemRuntime(router=BroadcastRouter())
    writer = Agent("writer", runtime)
    a, b = Agent("a", runtime), Agent("b", runtime)

    for agent in (a, b, writer):
        await runtime.subscribe(Subscription(agent.id))
    await a.start()
    await b.start()

    await writer.remember("everyone hears this")
    import asyncio
    await asyncio.sleep(0.05)

    assert len(a.received) == 1
    assert len(b.received) == 1
    await a.stop()
    await b.stop()


# ------------------------------------------------------------------ honesty

def test_calibration_reports_an_unseparable_embedder():
    """The hashing embedder cannot express paraphrase similarity.

    This is pinned as a test because the failure is silent: without it, a
    layer-3 subscription on the fallback embedder looks like it works.
    """
    sub = Subscription("a", semantic_query="angry furious customer complaint")
    calibration = sub.calibrate(
        HashingEmbedder(),
        should_match=["this client is livid and threatening to cancel"],
        should_not_match=["the deployment finished successfully"],
    )

    assert not calibration.separable
    assert "NOT SEPARABLE" in calibration.report()


def test_calibration_finds_a_threshold_when_tokens_do_overlap():
    sub = Subscription("a", semantic_query="sql injection security vulnerability")
    calibration = sub.calibrate(
        HashingEmbedder(),
        should_match=["sql injection found in the login handler"],
        should_not_match=["the button colour was updated"],
    )

    assert calibration.separable
    assert 0.0 < calibration.suggested_threshold < 1.0


async def test_runtime_warns_on_semantic_routing_with_a_non_semantic_embedder():
    runtime = EventMemRuntime(embedder=HashingEmbedder())

    with pytest.warns(RuntimeWarning, match="does not model meaning"):
        await runtime.subscribe(
            Subscription("a", semantic_query="anything that reads like a risk")
        )


async def test_the_warning_fires_only_once():
    runtime = EventMemRuntime(embedder=HashingEmbedder())
    with pytest.warns(RuntimeWarning):
        await runtime.subscribe(Subscription("a", semantic_query="first"))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        await runtime.subscribe(Subscription("b", semantic_query="second"))

"""The layered subscription predicate: EventMem's default routing ideology.

A subscription is a conjunction evaluated cheapest-first, so the expensive
semantic check only runs on events that already passed the cheap filters:

    layer 1  event type   set membership
    layer 2  attributes   dict comparison
    layer 3  semantics    cosine similarity against an interest vector

An agent combines whichever layers it needs. Layer 3 alone means "wake me on
anything that reads like X" with no keyword rule written down.

A note on layer 3 and honesty: semantic routing is only as good as the
embedder. With the hashing fallback, cosine similarity degenerates to weighted
token overlap and scores paraphrases at zero, so layer-3 subscriptions behave
like a soft keyword OR. ``Subscription.calibrate`` exists to make that visible
rather than to hide it: it reports the actual margin between examples you say
should match and examples you say should not, so a threshold is chosen from
evidence instead of by guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set

from ..core.events import EventType, MemoryEvent
from ..embeddings.similarity import cosine

#: Chosen so that an unset threshold is deliberately conservative: with a real
#: embedding model this admits clearly-related text and little else.
DEFAULT_THRESHOLD = 0.35


@dataclass
class Subscription:
    """A predicate over events, owned by one agent."""

    subscriber_id: str

    #: Layer 1. None means every type is acceptable.
    event_types: Optional[Set[EventType]] = None

    #: Layer 2. Every listed attribute must match exactly.
    attribute_filters: Dict[str, Any] = field(default_factory=dict)

    #: Layer 3. Natural-language statement of interest, embedded at register time.
    semantic_query: Optional[str] = None
    semantic_embedding: Optional[List[float]] = None
    semantic_threshold: float = DEFAULT_THRESHOLD

    #: Receive events you published yourself. Off by default: an agent already
    #: knows what it wrote, and self-delivery is a common cascade loop.
    include_own: bool = False

    #: Skip events deeper than this, independent of the runtime-wide ceiling.
    max_depth: Optional[int] = None

    label: str = ""

    #: Populated by the router: how many events reached each layer. Makes it
    #: possible to see that a subscription is never firing, and why.
    stats: Dict[str, int] = field(
        default_factory=lambda: {
            "seen": 0,
            "rejected_self": 0,
            "rejected_type": 0,
            "rejected_attribute": 0,
            "rejected_semantic": 0,
            "rejected_depth": 0,
            "matched": 0,
        }
    )

    def wants(self, event: MemoryEvent, *, record_stats: bool = True) -> bool:
        s = self.stats if record_stats else _SINK
        s["seen"] += 1

        if not self.include_own and event.source_agent == self.subscriber_id:
            s["rejected_self"] += 1
            return False

        if self.max_depth is not None and event.depth > self.max_depth:
            s["rejected_depth"] += 1
            return False

        if self.event_types is not None and event.event_type not in self.event_types:
            s["rejected_type"] += 1
            return False

        for key, value in self.attribute_filters.items():
            if event.attributes.get(key) != value:
                s["rejected_attribute"] += 1
                return False

        if self.semantic_query is not None:
            # Fail closed. A subscription that asked for semantic matching and
            # cannot perform it must not silently widen to "match everything".
            if self.semantic_embedding is None or event.embedding is None:
                s["rejected_semantic"] += 1
                return False
            if cosine(self.semantic_embedding, event.embedding) < self.semantic_threshold:
                s["rejected_semantic"] += 1
                return False

        s["matched"] += 1
        return True

    def calibrate(
        self,
        embedder,
        should_match: Sequence[str],
        should_not_match: Sequence[str],
    ) -> "Calibration":
        """Measure this subscription's separation on labelled examples.

        Returns the score of every example plus the suggested threshold, which
        is the midpoint of the gap between the worst positive and the best
        negative. A non-positive margin means this embedder cannot express the
        distinction at all, and no threshold will save it.
        """
        if self.semantic_query is None:
            raise ValueError("calibrate() needs a semantic_query")
        query = embedder.embed(self.semantic_query)
        pos = [(t, cosine(query, embedder.embed(t))) for t in should_match]
        neg = [(t, cosine(query, embedder.embed(t))) for t in should_not_match]
        worst_pos = min((s for _, s in pos), default=0.0)
        best_neg = max((s for _, s in neg), default=0.0)
        return Calibration(
            positives=pos,
            negatives=neg,
            worst_positive=worst_pos,
            best_negative=best_neg,
            margin=worst_pos - best_neg,
            suggested_threshold=(worst_pos + best_neg) / 2.0,
        )


@dataclass
class Calibration:
    """Evidence for a threshold choice. See ``Subscription.calibrate``."""

    positives: List[tuple]
    negatives: List[tuple]
    worst_positive: float
    best_negative: float
    margin: float
    suggested_threshold: float

    @property
    def separable(self) -> bool:
        """True when some threshold cleanly splits positives from negatives."""
        return self.margin > 0

    def report(self) -> str:
        lines = [
            f"  worst positive     {self.worst_positive:+.4f}",
            f"  best negative      {self.best_negative:+.4f}",
            f"  margin             {self.margin:+.4f}"
            f"{'' if self.separable else '   <- NOT SEPARABLE by any threshold'}",
        ]
        if self.separable:
            lines.append(f"  suggested threshold {self.suggested_threshold:.4f}")
        return "\n".join(lines)


_SINK: Dict[str, int] = dict.fromkeys(("seen", "rejected_self", "rejected_type", "rejected_attribute", "rejected_semantic", "rejected_depth", "matched"), 0)


class LayeredRouter:
    """Routes by evaluating every registered subscription against the event.

    Linear in the number of subscriptions. That is fine to a few thousand; past
    that, index layer 1 and 2 by type and attribute and only scan the layer-3
    candidates, or put the interest vectors in an ANN index. The protocol does
    not change, so that optimisation is a drop-in replacement for this class.
    """

    name = "layered"

    def __init__(self, embedder=None) -> None:
        self._embedder = embedder
        self._subscriptions: List[Subscription] = []

    def bind_embedder(self, embedder) -> None:
        """Called by the runtime so layer-3 queries embed with the same model
        that embeds events. Mismatched models make cosine meaningless."""
        self._embedder = embedder

    async def register(self, subscription: Subscription) -> None:
        if subscription.semantic_query and subscription.semantic_embedding is None:
            if self._embedder is None:
                raise RuntimeError(
                    "a semantic subscription needs an embedder; "
                    "pass one to the runtime or call bind_embedder()"
                )
            subscription.semantic_embedding = await _embed(
                self._embedder, subscription.semantic_query
            )
        self._subscriptions.append(subscription)

    def unregister(self, subscriber_id: str) -> None:
        self._subscriptions = [
            s for s in self._subscriptions if s.subscriber_id != subscriber_id
        ]

    def route(self, event: MemoryEvent) -> List[str]:
        """Ids of every agent wanting this event, de-duplicated, order stable.

        An agent holding several matching subscriptions is still delivered to
        once; delivering per-subscription would double-charge fan-out and
        double-wake the agent.
        """
        seen: Set[str] = set()
        matched: List[str] = []
        for sub in self._subscriptions:
            if sub.subscriber_id in seen:
                continue
            if sub.wants(event):
                seen.add(sub.subscriber_id)
                matched.append(sub.subscriber_id)
        return matched

    def subscribers(self) -> List[str]:
        return list(dict.fromkeys(s.subscriber_id for s in self._subscriptions))

    def subscriptions(self) -> List[Subscription]:
        return list(self._subscriptions)

    def diagnostics(self) -> List[Dict[str, Any]]:
        """Per-subscription match counts, for debugging a rule that never fires."""
        return [
            {"subscriber": s.subscriber_id, "label": s.label or "(unlabelled)", **s.stats}
            for s in self._subscriptions
        ]


async def _embed(embedder, text: str) -> List[float]:
    if hasattr(embedder, "embed_async"):
        return await embedder.embed_async(text)
    return embedder.embed(text)

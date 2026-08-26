"""
subscriptions.py

This is the answer to "how do agents subscribe". A subscription is a predicate
over events. When an event arrives, the manager asks every subscription whether
it wants this event, and returns the list of agents that do. Only those agents
get the update, which is what makes fan out selective instead of a broadcast.

Matching runs in three layers, cheapest first, so the expensive semantic check
only ever runs on events that already passed the cheap filters:

    layer 1  event type      set membership, almost free
    layer 2  attributes      dict comparison, cheap
    layer 3  semantics       cosine similarity, the costly one

An agent can combine layers. "Tell me about anything tagged component=backend"
is a layer 2 subscription. "Tell me about anything that reads like a security
concern" is a layer 3 subscription. "Tell me about completed research tasks
that read like they touch pricing" uses all three.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from embeddings import cosine
from events import EventType, MemoryEvent


@dataclass
class Subscription:
    subscriber_id: str

    # Layer 1. None means every type is acceptable.
    event_types: Optional[Set[EventType]] = None

    # Layer 2. Every listed attribute must match exactly. Empty means no filter.
    attribute_filters: Dict[str, Any] = field(default_factory=dict)

    # Layer 3. A natural language description of what the agent cares about. The
    # runtime embeds this once at registration and stores the vector here.
    semantic_query: Optional[str] = None
    semantic_embedding: Optional[List[float]] = None
    semantic_threshold: float = 0.35

    # Human readable label, useful in logs and metrics.
    label: str = ""

    def wants(self, event: MemoryEvent) -> bool:
        # An agent never needs to hear about its own writes.
        if event.source_agent == self.subscriber_id:
            return False

        # Layer 1.
        if self.event_types is not None and event.event_type not in self.event_types:
            return False

        # Layer 2.
        for key, value in self.attribute_filters.items():
            if event.attributes.get(key) != value:
                return False

        # Layer 3.
        if self.semantic_query is not None:
            if self.semantic_embedding is None or event.embedding is None:
                return False
            score = cosine(self.semantic_embedding, event.embedding)
            if score < self.semantic_threshold:
                return False

        return True


class SubscriptionManager:
    def __init__(self) -> None:
        self._subscriptions: List[Subscription] = []

    def register(self, subscription: Subscription) -> None:
        self._subscriptions.append(subscription)

    def unregister(self, subscriber_id: str) -> None:
        self._subscriptions = [
            s for s in self._subscriptions if s.subscriber_id != subscriber_id
        ]

    def match(self, event: MemoryEvent) -> List[str]:
        """
        Return the ids of every agent that wants this event. A single agent may
        hold several subscriptions, so we de duplicate while preserving order.
        """
        seen: Set[str] = set()
        matched: List[str] = []
        for subscription in self._subscriptions:
            if subscription.subscriber_id in seen:
                continue
            if subscription.wants(event):
                seen.add(subscription.subscriber_id)
                matched.append(subscription.subscriber_id)
        return matched

    def subscriptions(self) -> List[Subscription]:
        return list(self._subscriptions)

"""Built-in conflict policies.

Four rules covering the common shapes. Each is ~10 lines, which is the point:
the seam is cheap enough that a new domain rule is not a fork.
"""

from __future__ import annotations

from typing import Callable, Dict, Sequence

from ..core.events import MemoryRecord
from .base import ConflictError, Resolution


class ConfidenceThenRecency:
    """Prefer the higher-confidence write, break ties by recency.

    Defensible because provenance carries a confidence score, so the rule is
    grounded in something the writer asserted rather than in arrival order.
    """

    name = "confidence_then_recency"

    def resolve(self, current: MemoryRecord, incoming: MemoryRecord) -> Resolution:
        if incoming.confidence > current.confidence:
            return Resolution(incoming, current, "incoming has higher confidence")
        if incoming.confidence < current.confidence:
            return Resolution(current, incoming, "current has higher confidence")
        if incoming.updated_at > current.updated_at:
            return Resolution(incoming, current, "equal confidence, incoming is newer")
        if incoming.updated_at < current.updated_at:
            return Resolution(current, incoming, "equal confidence, current is newer")
        # Identical confidence and timestamp. Break the tie on a stable key so
        # two replicas resolving the same pair independently agree.
        if incoming.last_event_id and current.last_event_id:
            if incoming.last_event_id > current.last_event_id:
                return Resolution(incoming, current, "tie broken on event id")
            return Resolution(current, incoming, "tie broken on event id")
        return Resolution(current, incoming, "tie, holding current")


class LastWriterWins:
    """Arrival order decides. Cheapest rule, and the least defensible."""

    name = "last_writer_wins"

    def resolve(self, current: MemoryRecord, incoming: MemoryRecord) -> Resolution:
        return Resolution(incoming, current, "last writer wins")


class FirstWriterWins:
    """The established value holds. Useful when a memory is a claim that
    should not be silently revised."""

    name = "first_writer_wins"

    def resolve(self, current: MemoryRecord, incoming: MemoryRecord) -> Resolution:
        return Resolution(current, incoming, "first writer wins")


class PreferAgents:
    """Trust an explicit agent ranking, falling back to another policy.

    For workloads with a designated authority: a reviewer outranks a drafter,
    a verified tool result outranks a model's guess.
    """

    name = "prefer_agents"

    def __init__(self, order: Sequence[str], fallback=None) -> None:
        #: Lower index means higher authority.
        self._rank: Dict[str, int] = {agent: i for i, agent in enumerate(order)}
        self._fallback = fallback or ConfidenceThenRecency()
        self.name = f"prefer_agents({','.join(order)})"

    def _rank_of(self, record: MemoryRecord) -> int:
        agent = record.provenance.source_agent if record.provenance else ""
        return self._rank.get(agent, len(self._rank))

    def resolve(self, current: MemoryRecord, incoming: MemoryRecord) -> Resolution:
        c, i = self._rank_of(current), self._rank_of(incoming)
        if i < c:
            return Resolution(incoming, current, "incoming agent outranks current")
        if c < i:
            return Resolution(current, incoming, "current agent outranks incoming")
        return self._fallback.resolve(current, incoming)


class SetUnionMerge:
    """CRDT-style merge for set-valued memories.

    Content is treated as a delimited set of lines; the winner is the union, so
    the operation is commutative and associative and two agents adding
    different facts both keep theirs. This is the policy to copy when the
    memory is additive rather than a single value.
    """

    name = "set_union_merge"

    def __init__(self, separator: str = "\n") -> None:
        self.separator = separator

    def resolve(self, current: MemoryRecord, incoming: MemoryRecord) -> Resolution:
        def items(text: str) -> list:
            return [line.strip() for line in text.split(self.separator) if line.strip()]

        # dict.fromkeys preserves first-seen order, so the merge is stable.
        union = list(dict.fromkeys(items(current.content) + items(incoming.content)))
        winner = current.copy(
            content=self.separator.join(union),
            confidence=max(current.confidence, incoming.confidence),
            attributes={**current.attributes, **incoming.attributes},
        )
        return Resolution(winner, incoming, "set union of both writes", merged=True)


class Manual:
    """Refuse to resolve. Raises so the caller can escalate to a human."""

    name = "manual"

    def resolve(self, current: MemoryRecord, incoming: MemoryRecord) -> Resolution:
        raise ConflictError(current, incoming)


#: Name -> zero-argument factory, for config-driven construction.
REGISTRY: Dict[str, Callable[[], object]] = {
    ConfidenceThenRecency.name: ConfidenceThenRecency,
    LastWriterWins.name: LastWriterWins,
    FirstWriterWins.name: FirstWriterWins,
    SetUnionMerge.name: SetUnionMerge,
    Manual.name: Manual,
}


def get(name: str):
    """Build a policy by name. Raises with the valid options listed."""
    try:
        return REGISTRY[name]()
    except KeyError:
        raise KeyError(
            f"unknown conflict policy {name!r}; available: {sorted(REGISTRY)}"
        ) from None

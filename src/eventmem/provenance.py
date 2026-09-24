"""The Provenance Tracker.

Carrying a ``Provenance`` field on a record is not provenance tracking; it just
means the last writer is recorded. Tracking means you can answer questions
after the fact:

    who asserted this, and on what basis?
    what chain of events produced this conclusion?
    if this source turns out to be wrong, what else is contaminated?

That last one is the point. When a tool result turns out to be wrong, you need
the transitive closure of everything derived from it, and no amount of
per-record metadata gives you that. It needs the causal graph, which is what
``caused_by`` on every event provides and what this class indexes.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Set

from .core.events import MemoryEvent


@dataclass
class LineageNode:
    """One step in a derivation chain."""

    event_id: str
    memory_id: str
    agent: str
    event_type: str
    reason: str
    confidence: float
    depth: int
    content: str = ""

    def __str__(self) -> str:
        head = f"{self.agent} {self.event_type} (conf {self.confidence:.2f})"
        return f"{head}: {self.reason}" if self.reason else head


class ProvenanceTracker:
    """Indexes the causal graph as events are appended.

    Memory cost is one small node plus two dict entries per event, which is the
    right trade: the log already holds every event, and without this index the
    only way to answer a lineage question is a full scan.
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, LineageNode] = {}
        self._parent: Dict[str, Optional[str]] = {}
        self._children: Dict[str, List[str]] = defaultdict(list)
        self._by_memory: Dict[str, List[str]] = defaultdict(list)
        self._by_agent: Dict[str, List[str]] = defaultdict(list)

    def record(self, event: MemoryEvent) -> None:
        prov = event.provenance
        self._nodes[event.event_id] = LineageNode(
            event_id=event.event_id,
            memory_id=event.memory_id,
            agent=event.source_agent,
            event_type=event.event_type.value,
            reason=prov.reason if prov else "",
            confidence=prov.confidence if prov else 1.0,
            depth=event.depth,
            content=event.content,
        )
        self._parent[event.event_id] = event.caused_by
        if event.caused_by:
            self._children[event.caused_by].append(event.event_id)
        self._by_memory[event.memory_id].append(event.event_id)
        self._by_agent[event.source_agent].append(event.event_id)

    # ------------------------------------------------------------- ancestry

    def lineage(self, event_id: str) -> List[LineageNode]:
        """Chain from the root cause down to this event, oldest first.

        Cycle-guarded: ``caused_by`` should form a tree, but a malformed or
        replayed log could close a loop and this must not hang.
        """
        chain: List[LineageNode] = []
        seen: Set[str] = set()
        current: Optional[str] = event_id
        while current and current in self._nodes and current not in seen:
            seen.add(current)
            chain.append(self._nodes[current])
            current = self._parent.get(current)
        chain.reverse()
        return chain

    def root_cause(self, event_id: str) -> Optional[LineageNode]:
        chain = self.lineage(event_id)
        return chain[0] if chain else None

    # ------------------------------------------------------------ descendants

    def descendants(self, event_id: str) -> Iterator[LineageNode]:
        """Every event transitively derived from this one, breadth-first."""
        queue = list(self._children.get(event_id, ()))
        seen: Set[str] = set()
        while queue:
            current = queue.pop(0)
            if current in seen:
                continue
            seen.add(current)
            node = self._nodes.get(current)
            if node is not None:
                yield node
            queue.extend(self._children.get(current, ()))

    def contaminated_memories(self, event_id: str) -> Set[str]:
        """Memory ids touched by anything derived from this event.

        The blast radius when a source is retracted. Includes the event's own
        memory, because the retracted assertion is itself contaminated.
        """
        out = {self._nodes[event_id].memory_id} if event_id in self._nodes else set()
        out.update(node.memory_id for node in self.descendants(event_id))
        return out

    # ------------------------------------------------------------- queries

    def history(self, memory_id: str) -> List[LineageNode]:
        """Every event that ever touched this memory, in order."""
        return [self._nodes[e] for e in self._by_memory.get(memory_id, ()) if e in self._nodes]

    def authored_by(self, agent_id: str) -> List[LineageNode]:
        return [self._nodes[e] for e in self._by_agent.get(agent_id, ()) if e in self._nodes]

    def trust_path(self, memory_id: str) -> float:
        """Weakest confidence anywhere in the derivation of a memory's current value.

        A conclusion is no more trustworthy than the shakiest step that
        produced it, so this takes the minimum rather than a product or a mean.
        Returns 1.0 for a memory with no recorded history.
        """
        history = self.history(memory_id)
        if not history:
            return 1.0
        chain = self.lineage(history[-1].event_id)
        return min((node.confidence for node in chain), default=1.0)

    def explain(self, memory_id: str) -> str:
        """Human-readable derivation of a memory's current value."""
        history = self.history(memory_id)
        if not history:
            return f"{memory_id}: no recorded provenance"
        chain = self.lineage(history[-1].event_id)
        lines = [f"{memory_id} (trust {self.trust_path(memory_id):.2f}):"]
        for i, node in enumerate(chain):
            lines.append(f"  {'  ' * i}{'└─ ' if i else ''}{node}")
        return "\n".join(lines)

    def stats(self) -> Dict[str, int]:
        return {
            "events": len(self._nodes),
            "memories": len(self._by_memory),
            "agents": len(self._by_agent),
            "roots": sum(1 for p in self._parent.values() if p is None),
            "max_depth": max((n.depth for n in self._nodes.values()), default=0),
        }

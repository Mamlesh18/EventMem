"""Event serialisation for transports that move bytes.

Forward compatible in both directions: unknown fields on the wire are ignored
rather than raising, so a newer publisher does not break an older consumer.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from ..core.clock import Stamp
from ..core.events import EventType, MemoryEvent, Provenance

#: Bumped when the wire format changes incompatibly.
SCHEMA_VERSION = 1


def event_to_json(event: MemoryEvent, include_embedding: bool = False) -> str:
    """Serialise an event.

    The embedding is omitted from inbox copies by default: routing has already
    happened by delivery time, the recipient has no use for the vector, and it
    is by far the largest field. The durable log keeps it so that replay can
    rebuild the projection without re-embedding everything.
    """
    data: Dict[str, Any] = {
        "v": SCHEMA_VERSION,
        "event_type": event.event_type.value,
        "source_agent": event.source_agent,
        "memory_id": event.memory_id,
        "payload": event.payload,
        "attributes": event.attributes,
        "base_version": event.base_version,
        "event_id": event.event_id,
        "created_at": event.created_at,
        "published_at": _stamp_to_wire(event.published_at),
        "depth": event.depth,
        "caused_by": event.caused_by,
        "committed_version": event.committed_version,
    }
    if event.provenance is not None:
        p = event.provenance
        data["provenance"] = {
            "source_agent": p.source_agent,
            "reason": p.reason,
            "source": p.source,
            "confidence": p.confidence,
            "created_at": p.created_at,
        }
    if include_embedding and event.embedding is not None:
        data["embedding"] = event.embedding
    return json.dumps(data, separators=(",", ":"))


def event_from_json(raw: str) -> MemoryEvent:
    data = json.loads(raw)

    provenance: Optional[Provenance] = None
    raw_prov = data.get("provenance")
    if raw_prov:
        # Filter to known fields so a provenance record written by a newer
        # version does not blow up the constructor here.
        known = {"source_agent", "reason", "source", "confidence", "created_at"}
        provenance = Provenance(**{k: v for k, v in raw_prov.items() if k in known})

    return MemoryEvent(
        event_type=EventType(data["event_type"]),
        source_agent=data["source_agent"],
        memory_id=data["memory_id"],
        payload=data.get("payload", {}),
        attributes=data.get("attributes", {}),
        provenance=provenance,
        base_version=data.get("base_version"),
        event_id=data["event_id"],
        created_at=data.get("created_at", 0.0),
        published_at=_stamp_from_wire(data.get("published_at")),
        embedding=data.get("embedding"),
        depth=data.get("depth", 0),
        caused_by=data.get("caused_by"),
        committed_version=data.get("committed_version"),
    )


def _stamp_to_wire(stamp: Optional[Stamp]) -> Optional[Dict[str, Any]]:
    """Serialise a timing stamp.

    The origin process id travels with it. Without it, the receiver cannot tell
    whether the perf_counter value is comparable to its own (same process) or
    meaningless (different process), and would silently compute nonsense
    latencies across a real transport.
    """
    if stamp is None:
        return None
    return {"w": stamp.wall, "p": stamp.perf, "o": stamp.origin}


def _stamp_from_wire(data: Any) -> Optional[Stamp]:
    if not data:
        return None
    if isinstance(data, (int, float)):  # pre-0.2 wire format: bare wall clock
        return Stamp(wall=float(data), perf=0.0, origin="legacy")
    return Stamp(wall=data["w"], perf=data["p"], origin=data["o"])

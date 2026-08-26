"""
serde.py

Events live in memory as dataclasses. To travel through Redis they have to
become bytes. This turns a MemoryEvent into JSON and back without losing the
provenance or the enum types.

The embedding is dropped from the inbox copy on purpose. Routing already
happened by the time an event is delivered, so the receiving agent has no use
for the vector, and leaving it out keeps the payload small.
"""

from __future__ import annotations

import json
from typing import Optional

from events import EventType, MemoryEvent, Provenance


def event_to_json(event: MemoryEvent, include_embedding: bool = False) -> str:
    data = {
        "event_type": event.event_type.value,
        "source_agent": event.source_agent,
        "memory_id": event.memory_id,
        "payload": event.payload,
        "attributes": event.attributes,
        "base_version": event.base_version,
        "event_id": event.event_id,
        "created_at": event.created_at,
        "published_at": event.published_at,
        "depth": event.depth,
        "caused_by": event.caused_by,
    }
    if event.provenance is not None:
        data["provenance"] = {
            "source_agent": event.provenance.source_agent,
            "reason": event.provenance.reason,
            "source": event.provenance.source,
            "confidence": event.provenance.confidence,
            "created_at": event.provenance.created_at,
        }
    if include_embedding:
        data["embedding"] = event.embedding
    return json.dumps(data)


def event_from_json(raw: str) -> MemoryEvent:
    data = json.loads(raw)
    provenance: Optional[Provenance] = None
    if data.get("provenance"):
        provenance = Provenance(**data["provenance"])
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
        published_at=data.get("published_at"),
        embedding=data.get("embedding"),
        depth=data.get("depth", 0),
        caused_by=data.get("caused_by"),
    )

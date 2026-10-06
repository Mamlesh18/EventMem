"""The shared corpus and its ground truth.

Both systems ingest exactly this, and both answer exactly these queries, so any
difference in the results is a property of the architecture rather than of the
data or the questions.

The ground truth is hand-labelled: for each query, the set of memory ids a
competent system ought to return. That is what makes retrieval *measurable*
rather than a matter of eyeballing which answers look nicer.

Scenario: a software team's shared memory during one sprint. Five agents write
into it; each has a declared interest, which is what EventMem routes on and what
the mem0 baseline filters its searches by.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set


@dataclass(frozen=True)
class Fact:
    """One memory, with the agents that genuinely need to know it."""

    id: str
    text: str
    author: str
    topic: str
    #: Ground truth for propagation: who needs this to do their job.
    needed_by: Set[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class Query:
    """One retrieval question, with the memories that genuinely answer it."""

    id: str
    text: str
    asked_by: str
    #: Ground truth for retrieval quality: ids that are actually relevant.
    relevant: Set[str] = field(default_factory=frozenset)


#: Who writes, and what each agent cares about. The topic list is the interest
#: both systems are given: EventMem subscribes on it, mem0 filters on it.
AGENTS: Dict[str, List[str]] = {
    "architect": ["schema", "api"],
    "backend": ["schema", "api", "security"],
    "frontend": ["api", "ui"],
    "security": ["security", "api"],
    "sre": ["infra", "security"],
}


FACTS: List[Fact] = [
    # ---------------------------------------------------------------- schema
    Fact("f01", "The bookings table uses a composite primary key on (clinician_id, slot_start).",
         "architect", "schema", frozenset({"backend"})),
    Fact("f02", "Slot times are stored as UTC timestamps; the client converts to clinic local time.",
         "architect", "schema", frozenset({"backend", "frontend"})),
    Fact("f03", "Patient records are soft-deleted with a deleted_at column, never hard-deleted.",
         "architect", "schema", frozenset({"backend"})),
    Fact("f04", "A unique constraint on (clinician_id, slot_start) prevents double-booking at the database level.",
         "backend", "schema", frozenset({"architect"})),
    Fact("f05", "The clinicians table has a nullable license_expires_at used by the compliance report.",
         "backend", "schema", frozenset({"architect"})),
    Fact("f06", "Appointment history is partitioned by month to keep the index small.",
         "architect", "schema", frozenset({"backend", "sre"})),
    Fact("f07", "Foreign keys use ON DELETE RESTRICT so an orphaned booking is impossible.",
         "architect", "schema", frozenset({"backend"})),

    # ------------------------------------------------------------------- api
    Fact("f08", "POST /bookings returns 409 Conflict when the slot is already taken.",
         "backend", "api", frozenset({"frontend", "architect"})),
    Fact("f09", "All list endpoints are cursor-paginated; offset pagination was removed in v2.",
         "backend", "api", frozenset({"frontend"})),
    Fact("f10", "The API rejects requests without an Idempotency-Key header on POST /bookings.",
         "backend", "api", frozenset({"frontend", "security"})),
    Fact("f11", "Rate limiting is 100 requests per minute per API token, returning 429 with Retry-After.",
         "backend", "api", frozenset({"frontend", "sre", "security"})),
    Fact("f12", "GET /clinicians/{id}/availability accepts a date range of at most 90 days.",
         "backend", "api", frozenset({"frontend"})),
    Fact("f13", "Error responses follow RFC 7807 problem+json with a machine-readable type URI.",
         "architect", "api", frozenset({"frontend", "backend"})),
    Fact("f14", "Webhook payloads are signed with HMAC-SHA256 in the X-Signature header.",
         "backend", "api", frozenset({"security", "frontend"})),

    # -------------------------------------------------------------- security
    Fact("f15", "The booking handler builds its SQL by concatenating raw request input, allowing injection.",
         "security", "security", frozenset({"backend", "architect"})),
    Fact("f16", "Uploaded patient documents are passed to a shell command without escaping the filename.",
         "security", "security", frozenset({"backend", "sre"})),
    Fact("f17", "Changing the id in /patients/{id} returns another tenant's record; authorization is missing.",
         "security", "security", frozenset({"backend", "architect"})),
    Fact("f18", "Session tokens never expire server-side, so a leaked token is valid forever.",
         "security", "security", frozenset({"backend", "sre"})),
    Fact("f19", "The webhook endpoint does not verify the HMAC signature it receives.",
         "security", "security", frozenset({"backend"})),
    Fact("f20", "Audit logs record the actor but not the affected patient id, so breaches are untraceable.",
         "security", "security", frozenset({"backend", "sre"})),

    # ----------------------------------------------------------------- infra
    Fact("f21", "The production TLS certificate expires on the 14th and renewal is not automated.",
         "sre", "infra", frozenset({"security"})),
    Fact("f22", "Postgres runs a single primary with no read replica; a failover means downtime.",
         "sre", "infra", frozenset({"architect"})),
    Fact("f23", "The nightly backup completes but has never been restored in a drill.",
         "sre", "infra", frozenset({"architect", "security"})),
    Fact("f24", "Deploys are blue-green but the database migration step is not reversible.",
         "sre", "infra", frozenset({"backend", "architect"})),
    Fact("f25", "Container memory limit is 512MB and the booking service OOMs under load tests.",
         "sre", "infra", frozenset({"backend"})),
    Fact("f26", "Logs are retained for 7 days, below the 90 days the compliance policy requires.",
         "sre", "infra", frozenset({"security"})),

    # -------------------------------------------------------------------- ui
    Fact("f27", "The booking form shows clinician availability inline, refreshed every 30 seconds.",
         "frontend", "ui", frozenset({"backend"})),
    Fact("f28", "Time zones are rendered using the clinic's locale, not the browser's.",
         "frontend", "ui", frozenset({"architect"})),
    Fact("f29", "The confirmation screen is the only place the booking reference is shown.",
         "frontend", "ui", frozenset()),
    Fact("f30", "Form validation errors are announced to screen readers via aria-live.",
         "frontend", "ui", frozenset()),
    Fact("f31", "The availability calendar renders 400 DOM nodes per week and janks on mobile.",
         "frontend", "ui", frozenset()),
    Fact("f32", "A double-click on Confirm submits twice because the button is not disabled on submit.",
         "frontend", "ui", frozenset({"backend"})),
]


QUERIES: List[Query] = [
    Query("q01", "How does the system prevent two patients booking the same slot?",
          "frontend", frozenset({"f04", "f08", "f01"})),
    Query("q02", "What are the known SQL injection or unsafe input problems?",
          "backend", frozenset({"f15", "f16"})),
    Query("q03", "How is authentication and session expiry handled?",
          "backend", frozenset({"f18", "f10"})),
    Query("q04", "What do I need to know about pagination and rate limits when calling the API?",
          "frontend", frozenset({"f09", "f11", "f12"})),
    Query("q05", "How are webhooks secured?",
          "security", frozenset({"f14", "f19"})),
    Query("q06", "What time zone handling is in place?",
          "frontend", frozenset({"f02", "f28"})),
    Query("q07", "What are our disaster recovery and backup risks?",
          "sre", frozenset({"f22", "f23", "f24"})),
    Query("q08", "Which issues could let one tenant read another tenant's data?",
          "security", frozenset({"f17", "f20"})),
    Query("q09", "What compliance gaps do we have?",
          "security", frozenset({"f26", "f20", "f05"})),
    Query("q10", "What performance problems have been reported?",
          "backend", frozenset({"f25", "f31", "f06"})),
    Query("q11", "How should a client handle a duplicate booking submission?",
          "frontend", frozenset({"f10", "f08", "f32"})),
    Query("q12", "What is expiring or needs renewal soon?",
          "sre", frozenset({"f21", "f05"})),
]


#: A throwaway write used to warm both systems up before anything is timed.
#: The first write into either system pays one-time costs - index creation,
#: lazy model load, connection setup - that have nothing to do with steady-state
#: write cost. Including it made mem0's mean write latency (218ms) exceed its own
#: p95 (130ms), which is arithmetically impossible without an outlier and is the
#: signature of exactly this problem. This fact is written and discarded, and is
#: not part of FACTS, so it reaches no score.
WARMUP_FACT = Fact(
    "warmup", "Warmup record, not part of the corpus or any ground truth.",
    "architect", "schema", frozenset(),
)

WARMUP_QUERY = Query(
    "warmup", "warmup query, not scored", "architect", frozenset({"f01"}),
)


#: Facts whose propagation is scored. A fact nobody needs cannot measure
#: propagation, so it is ingested as corpus but excluded from the phase-C score.
PROPAGATION_FACTS = [f for f in FACTS if f.needed_by]


def required_deliveries() -> Set[tuple]:
    """(fact_id, agent_id) pairs that ought to reach their recipient."""
    return {(f.id, agent) for f in FACTS for agent in f.needed_by}


def reachable_deliveries() -> Set[tuple]:
    """Required deliveries that an interest-based router could actually make.

    A fact only reaches an agent if that agent declared an interest in its
    topic. 16 of the 44 required deliveries name an agent that did not, so no
    amount of push can satisfy them -- that is the standing cost of selective
    routing, and reporting coverage without this denominator makes an
    interest-declaration problem look like an architecture one.
    """
    return {
        (f.id, agent)
        for f in FACTS
        for agent in f.needed_by
        if f.topic in AGENTS[agent]
    }


def routing_ceiling() -> float:
    """Highest propagation coverage any interest-based router can reach here."""
    required = required_deliveries()
    return len(reachable_deliveries()) / len(required) if required else float("nan")


def fact_by_id() -> Dict[str, Fact]:
    return {f.id: f for f in FACTS}


def summary() -> Dict[str, int]:
    return {
        "agents": len(AGENTS),
        "facts": len(FACTS),
        "queries": len(QUERIES),
        "topics": len({f.topic for f in FACTS}),
        "required_deliveries": len(required_deliveries()),
        "reachable_deliveries": len(reachable_deliveries()),
        "routing_ceiling": round(routing_ceiling(), 4),
        "labelled_relevant_pairs": sum(len(q.relevant) for q in QUERIES),
    }


def validate() -> List[str]:
    """Catch ground-truth mistakes before they become results."""
    problems: List[str] = []
    ids = {f.id for f in FACTS}
    if len(ids) != len(FACTS):
        problems.append("duplicate fact ids")
    for f in FACTS:
        if f.author not in AGENTS:
            problems.append(f"{f.id}: unknown author {f.author!r}")
        for agent in f.needed_by:
            if agent not in AGENTS:
                problems.append(f"{f.id}: needed_by unknown agent {agent!r}")
            if agent == f.author:
                problems.append(f"{f.id}: author listed as needing its own write")
    for q in QUERIES:
        if q.asked_by not in AGENTS:
            problems.append(f"{q.id}: unknown asker {q.asked_by!r}")
        if not q.relevant:
            problems.append(f"{q.id}: no relevant facts labelled")
        for fid in q.relevant:
            if fid not in ids:
                problems.append(f"{q.id}: relevant id {fid!r} does not exist")
    return problems

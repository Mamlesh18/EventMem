"""Baseline architectures.

These are the systems EventMem claims to improve on, implemented properly
rather than strawmanned. In particular ``PollingSystem`` is a real shared
store with a real retrieval loop: it is what AutoGen-style and RAG-memory
frameworks actually do, and at a short enough poll interval it *should* match
EventMem on freshness. Showing where it does and what it costs to get there is
a far stronger result than showing it fail.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Set

from eventmem.core.clock import Stamp, now
from eventmem.core.events import MemoryRecord, Provenance
from eventmem.metrics import Delivery, MetricSet, RuntimeMetrics

from ..workloads.base import AgentSpec
from .base import PublishResult, System


class _SharedStore:
    """A passive versioned key-value store. The thing agents pull from."""

    def __init__(self) -> None:
        self.records: Dict[str, MemoryRecord] = {}
        self.reads = 0

    def put(self, memory_id: str, content: str, attributes: Dict[str, Any],
            agent: str, confidence: float) -> int:
        current = self.records.get(memory_id)
        version = (current.version + 1) if current else 1
        self.records[memory_id] = MemoryRecord(
            memory_id=memory_id, content=content, attributes=dict(attributes),
            version=version, confidence=confidence,
            provenance=Provenance(source_agent=agent, confidence=confidence),
        )
        return version

    def snapshot(self) -> Dict[str, MemoryRecord]:
        self.reads += 1
        return dict(self.records)

    def version(self, memory_id: str) -> Optional[int]:
        record = self.records.get(memory_id)
        return record.version if record else None


class PollingSystem(System):
    """Shared store that agents poll on a timer. The pull baseline.

    Each agent wakes every ``interval_s``, scans the store, and updates its
    held state for anything it cares about. The interval is the whole tradeoff:
    short means fresh but burns retrievals, long means cheap but stale. Running
    the sweep at several intervals is what turns that tradeoff into a curve
    instead of an assertion.

    Retrieval cost is counted honestly: every poll that scans the store is one
    retrieval, whether or not anything changed. That is the cost push avoids.
    """

    name = "polling"
    description = "passive shared store, agents poll on a fixed interval"

    def __init__(self, interval_s: float = 0.05, filter_by_interest: bool = True) -> None:
        self.interval_s = interval_s
        self.filter_by_interest = filter_by_interest
        self.name = f"polling({interval_s * 1000:.0f}ms)"
        self.store = _SharedStore()
        self.specs: Dict[str, AgentSpec] = {}
        self.held: Dict[str, Dict[str, int]] = {}
        self.tool_keys: Dict[str, Set[str]] = {}
        self.delivered: Set[tuple] = set()
        self.wakeups: Dict[str, int] = {}
        self.retrievals = 0
        self.records_scanned = 0
        self.metrics = RuntimeMetrics()
        self._publish_times: Dict[str, Stamp] = {}
        self._tasks: List[asyncio.Task] = []
        self._running = False

    async def setup(self, agents: List[AgentSpec]) -> None:
        for spec in agents:
            self.specs[spec.agent_id] = spec
            self.held[spec.agent_id] = {}
            self.tool_keys[spec.agent_id] = set()
            self.wakeups[spec.agent_id] = 0
        self.metrics.agent_count = len(agents)
        self._running = True
        for spec in agents:
            self._tasks.append(asyncio.create_task(self._poll_loop(spec)))

    def _interested(self, spec: AgentSpec, record: MemoryRecord) -> bool:
        """Whether this agent would pull this record.

        Applies the same interests the push router would, so the comparison is
        between *when* knowledge arrives, not between how well each
        architecture can express what it wants.
        """
        if not self.filter_by_interest:
            return True
        return any(
            i.matches_attributes(record.attributes) for i in spec.all_interests()
        )

    async def _poll_loop(self, spec: AgentSpec) -> None:
        while self._running:
            await asyncio.sleep(self.interval_s)
            if not self._running:
                return
            self._poll_once(spec)

    def _poll_once(self, spec: AgentSpec) -> None:
        self.retrievals += 1
        snapshot = self.store.snapshot()
        held = self.held[spec.agent_id]
        for memory_id, record in snapshot.items():
            self.records_scanned += 1
            if record.provenance and record.provenance.source_agent == spec.agent_id:
                continue
            if not self._interested(spec, record):
                continue
            if held.get(memory_id, 0) >= record.version:
                continue

            held[memory_id] = record.version
            self.wakeups[spec.agent_id] += 1
            self.delivered.add((memory_id, spec.agent_id))
            key = record.attributes.get("tool_key")
            if key:
                self.tool_keys[spec.agent_id].add(key)

            published_at = self._publish_times.get(f"{memory_id}:{record.version}")
            if published_at is not None:
                # Arrival time is when this poll observed it. The poll interval
                # is therefore inside the latency, which is exactly the cost
                # being measured.
                observed = now()
                self.metrics.deliveries.append(
                    Delivery(
                        event_id=f"{memory_id}:{record.version}",
                        memory_id=memory_id,
                        recipient=spec.agent_id,
                        published_at=published_at,
                        enqueued_at=observed,
                        dequeued_at=observed,
                    )
                )

    async def publish(
        self, actor: str, memory_id: str, content: str, event_type: Any,
        attributes: Dict[str, Any], confidence: float = 1.0,
        base_version: Optional[int] = None, reason: str = "",
    ) -> PublishResult:
        version = self.store.put(memory_id, content, attributes, actor, confidence)
        self._publish_times[f"{memory_id}:{version}"] = now()
        self.metrics.events_published += 1
        self.metrics.events_in_log += 1
        # A write pushes nothing, so fan-out is zero by definition. Recorded so
        # the denominator for fan-out efficiency is the same as every other
        # system's.
        self.metrics.fan_out_sizes.append(0)
        self.held[actor][memory_id] = version
        key = attributes.get("tool_key")
        if key:
            self.tool_keys[actor].add(key)
        return PublishResult(memory_id, version, [])

    async def settle(self, seconds: float = 0.0) -> None:
        """Let exactly as much time pass as the workload asked for.

        Deliberately *not* "at least one poll interval". Waiting for this
        system's own timer would hand it a free catch-up that push never gets,
        and would make every interval look equally fresh -- but the interval is
        the variable under test, so the harness must not compensate for it. How
        much time passes between steps is the workload's decision, applied
        identically to every system.
        """
        await asyncio.sleep(seconds)

    def holds(self, agent_id: str, memory_id: str) -> Optional[int]:
        return self.held[agent_id].get(memory_id)

    def authoritative_version(self, memory_id: str) -> Optional[int]:
        return self.store.version(memory_id)

    def knows_tool_result(self, agent_id: str, tool_key: str) -> bool:
        return tool_key in self.tool_keys[agent_id]

    def delivered_pairs(self) -> Set[tuple]:
        return set(self.delivered)

    def held_versions(self) -> Dict[str, Dict[str, int]]:
        return {a: dict(v) for a, v in self.held.items()}

    def authoritative_versions(self) -> Dict[str, int]:
        return {m: r.version for m, r in self.store.records.items()}

    def collect(self, metrics: MetricSet) -> None:
        metrics.runtime = self.metrics

    async def teardown(self) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    def extra_stats(self) -> Dict[str, int]:
        return {"retrievals": self.retrievals, "records_scanned": self.records_scanned}


class OnDemandRetrievalSystem(System):
    """Shared store read only at the moment of use. The RAG-memory baseline.

    No background polling at all: an agent's view updates when it explicitly
    retrieves, which is what a framework does when it stuffs a vector-store
    query into the prompt before each step. Maximum freshness *at read time*
    and zero background cost, but the agent learns nothing between reads, so it
    cannot react to another agent's finding and cannot skip a duplicate tool
    call unless it happens to retrieve first.
    """

    name = "on_demand"
    description = "shared store, retrieved only at the point of use (RAG-style)"

    def __init__(self, retrieve_before_tool_calls: bool = True) -> None:
        self.retrieve_before_tool_calls = retrieve_before_tool_calls
        self.store = _SharedStore()
        self.specs: Dict[str, AgentSpec] = {}
        self.held: Dict[str, Dict[str, int]] = {}
        self.tool_keys: Dict[str, Set[str]] = {}
        self.delivered: Set[tuple] = set()
        self.retrievals = 0
        self.metrics = RuntimeMetrics()

    async def setup(self, agents: List[AgentSpec]) -> None:
        for spec in agents:
            self.specs[spec.agent_id] = spec
            self.held[spec.agent_id] = {}
            self.tool_keys[spec.agent_id] = set()
        self.metrics.agent_count = len(agents)

    def _retrieve(self, agent_id: str) -> None:
        self.retrievals += 1
        for memory_id, record in self.store.snapshot().items():
            if record.provenance and record.provenance.source_agent == agent_id:
                continue
            if self.held[agent_id].get(memory_id, 0) >= record.version:
                continue
            self.held[agent_id][memory_id] = record.version
            self.delivered.add((memory_id, agent_id))
            key = record.attributes.get("tool_key")
            if key:
                self.tool_keys[agent_id].add(key)

    async def publish(
        self, actor: str, memory_id: str, content: str, event_type: Any,
        attributes: Dict[str, Any], confidence: float = 1.0,
        base_version: Optional[int] = None, reason: str = "",
    ) -> PublishResult:
        version = self.store.put(memory_id, content, attributes, actor, confidence)
        self.metrics.events_published += 1
        self.metrics.events_in_log += 1
        self.metrics.fan_out_sizes.append(0)
        self.held[actor][memory_id] = version
        key = attributes.get("tool_key")
        if key:
            self.tool_keys[actor].add(key)
        return PublishResult(memory_id, version, [])

    async def settle(self, seconds: float = 0.0) -> None:
        return None

    def holds(self, agent_id: str, memory_id: str) -> Optional[int]:
        # A read is a retrieval: this system is always fresh at the point of
        # use, which is its genuine strength and must not be scored away.
        self._retrieve(agent_id)
        return self.held[agent_id].get(memory_id)

    def authoritative_version(self, memory_id: str) -> Optional[int]:
        return self.store.version(memory_id)

    def knows_tool_result(self, agent_id: str, tool_key: str) -> bool:
        if self.retrieve_before_tool_calls:
            self._retrieve(agent_id)
        return tool_key in self.tool_keys[agent_id]

    def delivered_pairs(self) -> Set[tuple]:
        return set(self.delivered)

    def held_versions(self) -> Dict[str, Dict[str, int]]:
        return {a: dict(v) for a, v in self.held.items()}

    def authoritative_versions(self) -> Dict[str, int]:
        return {m: r.version for m, r in self.store.records.items()}

    def collect(self, metrics: MetricSet) -> None:
        metrics.runtime = self.metrics

    def extra_stats(self) -> Dict[str, int]:
        return {"retrievals": self.retrievals}


class NoSharingSystem(System):
    """Agents keep private memory and never see each other's work.

    The floor. Establishes C_base for duplicate-tool-call reduction: with no
    sharing, every agent that needs a tool result pays for it. Without this
    baseline, rho_dup has no denominator and is not a measurement.
    """

    name = "no_sharing"
    description = "isolated per-agent memory, no coordination (floor / C_base)"

    def __init__(self) -> None:
        self.private: Dict[str, Dict[str, int]] = {}
        self.tool_keys: Dict[str, Set[str]] = {}
        self.authoritative: Dict[str, int] = {}
        self.metrics = RuntimeMetrics()

    async def setup(self, agents: List[AgentSpec]) -> None:
        for spec in agents:
            self.private[spec.agent_id] = {}
            self.tool_keys[spec.agent_id] = set()
        self.metrics.agent_count = len(agents)

    async def publish(
        self, actor: str, memory_id: str, content: str, event_type: Any,
        attributes: Dict[str, Any], confidence: float = 1.0,
        base_version: Optional[int] = None, reason: str = "",
    ) -> PublishResult:
        version = self.authoritative.get(memory_id, 0) + 1
        self.authoritative[memory_id] = version
        self.private[actor][memory_id] = version
        self.metrics.events_published += 1
        self.metrics.fan_out_sizes.append(0)
        key = attributes.get("tool_key")
        if key:
            self.tool_keys[actor].add(key)
        return PublishResult(memory_id, version, [])

    async def settle(self, seconds: float = 0.0) -> None:
        return None

    def holds(self, agent_id: str, memory_id: str) -> Optional[int]:
        return self.private[agent_id].get(memory_id)

    def authoritative_version(self, memory_id: str) -> Optional[int]:
        return self.authoritative.get(memory_id)

    def knows_tool_result(self, agent_id: str, tool_key: str) -> bool:
        return tool_key in self.tool_keys[agent_id]

    def delivered_pairs(self) -> Set[tuple]:
        return set()

    def held_versions(self) -> Dict[str, Dict[str, int]]:
        return {a: dict(v) for a, v in self.private.items()}

    def authoritative_versions(self) -> Dict[str, int]:
        return dict(self.authoritative)

    def collect(self, metrics: MetricSet) -> None:
        metrics.runtime = self.metrics

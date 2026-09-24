"""The system-under-test contract.

Every memory architecture being compared implements this interface, and the
harness drives all of them with the identical workload script. That is what
makes the numbers comparable.

The key abstraction is *held state*: what each agent currently believes about
each memory. Every system maintains it differently, and that difference is the
whole experiment:

    EventMem    updated when a matching event is pushed
    Broadcast   updated when any event is pushed
    Polling     updated only when the agent's poll timer fires
    NoSharing   never updated from another agent

Freshness, consistency and duplicate-tool-call reduction all fall out of held
state, so they are measured identically across systems rather than each system
reporting its own favourable definition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from eventmem.metrics import MetricSet

from ..workloads.base import AgentSpec


@dataclass
class PublishResult:
    """What a system did with one publish."""

    memory_id: str
    version: int
    delivered_to: List[str]


class System:
    """Base class for a memory architecture under test."""

    name: str = "unnamed"
    #: One line for the results table, describing what this system is.
    description: str = ""

    async def setup(self, agents: List[AgentSpec]) -> None:
        """Create agents and register their interests."""
        raise NotImplementedError

    async def publish(
        self,
        actor: str,
        memory_id: str,
        content: str,
        event_type: Any,
        attributes: Dict[str, Any],
        confidence: float = 1.0,
        base_version: Optional[int] = None,
        reason: str = "",
    ) -> PublishResult:
        raise NotImplementedError

    async def settle(self, seconds: float = 0.0) -> None:
        """Let delivery complete and let pull-based systems poll."""
        raise NotImplementedError

    def holds(self, agent_id: str, memory_id: str) -> Optional[int]:
        """Version of ``memory_id`` that ``agent_id`` currently believes current."""
        raise NotImplementedError

    def authoritative_version(self, memory_id: str) -> Optional[int]:
        """The true current version, regardless of who knows it."""
        raise NotImplementedError

    def knows_tool_result(self, agent_id: str, tool_key: str) -> bool:
        """Whether this agent can already see a result for ``tool_key``.

        Drives duplicate-tool-call measurement. A system where knowledge does
        not propagate answers False and pays for the call again.
        """
        raise NotImplementedError

    def delivered_pairs(self) -> Set[tuple]:
        """(memory_id, agent_id) pairs this system actually delivered."""
        raise NotImplementedError

    def held_versions(self) -> Dict[str, Dict[str, int]]:
        raise NotImplementedError

    def authoritative_versions(self) -> Dict[str, int]:
        raise NotImplementedError

    def collect(self, metrics: MetricSet) -> None:
        """Fill in system-specific fields on the MetricSet."""
        return None

    async def teardown(self) -> None:
        return None

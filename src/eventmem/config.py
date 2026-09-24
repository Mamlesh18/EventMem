"""Assemble a runtime from the environment, with honest fallbacks.

Set credentials and you get real models, real embeddings and Redis. Leave them
unset and the same code runs on a mock model, a hashing embedder and an
in-memory bus.

``build()`` returns the runtime *and* a description of what it actually got, so
a script can print the configuration it is really running. That matters: a
result produced on the hashing fallback and a result produced on a hosted
embedder are not comparable, and the only way to keep them straight is to make
the runtime say which one it used.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, List, Optional

from .embeddings.backends import (
    AzureEmbedder,
    CachingEmbedder,
    HashingEmbedder,
    OpenAIEmbedder,
    SentenceTransformerEmbedder,
)
from .llm.backends import AzureLLM, MockLLM, OpenAILLM
from .runtime import EventMemRuntime


@dataclass
class Config:
    """What the runtime is actually made of, and what it fell back from."""

    embedder_name: str
    llm_name: str
    transport_name: str
    semantic_routing_is_real: bool
    warnings: List[str] = field(default_factory=list)

    def describe(self) -> str:
        lines = [
            f"  model       {self.llm_name}",
            f"  embeddings  {self.embedder_name}",
            f"  transport   {self.transport_name}",
        ]
        for warning in self.warnings:
            lines.append(f"  !  {warning}")
        return "\n".join(lines)


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def build_embedder(prefer: Optional[str] = None) -> tuple:
    """Pick the best available embedder.

    Order: explicit choice, then hosted, then a local model, then hashing. The
    returned flag says whether layer-3 routing will actually model meaning.
    """
    _load_dotenv()
    choice = (prefer or os.getenv("EVENTMEM_EMBEDDER") or "auto").lower()

    if choice in ("hashing", "hash"):
        return HashingEmbedder(), "hashing", False

    if choice in ("auto", "azure"):
        client = _azure_client()
        deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        if client is not None and deployment:
            return (
                CachingEmbedder(AzureEmbedder(client, deployment)),
                f"azure({deployment})",
                True,
            )

    if choice in ("auto", "openai"):
        client = _openai_client()
        if client is not None:
            model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
            return (
                CachingEmbedder(OpenAIEmbedder(client, model)),
                f"openai({model})",
                True,
            )

    if choice in ("auto", "local", "sentence_transformers"):
        try:
            model = os.getenv("EVENTMEM_LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            embedder = SentenceTransformerEmbedder(model)
            return CachingEmbedder(embedder), f"sentence_transformers({model})", True
        except ImportError:
            pass

    return HashingEmbedder(), "hashing (no model available)", False


def build_llm(prefer: Optional[str] = None) -> tuple:
    _load_dotenv()
    choice = (prefer or os.getenv("EVENTMEM_LLM") or "auto").lower()

    if choice == "mock":
        return MockLLM(), "mock (forced)"

    if choice in ("auto", "azure"):
        client = _azure_client()
        deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
        if client is not None and deployment:
            return AzureLLM(client, deployment), f"azure({deployment})"

    if choice in ("auto", "openai"):
        client = _openai_client()
        if client is not None:
            model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
            return OpenAILLM(client, model), f"openai({model})"

    return MockLLM(), "mock (no credentials found)"


async def build_transport(prefer: Optional[str] = None) -> tuple:
    """Redis when reachable, in-memory otherwise. Returns (bus, store, name)."""
    _load_dotenv()
    choice = (prefer or os.getenv("EVENTMEM_TRANSPORT") or "auto").lower()

    if choice in ("auto", "redis"):
        url = os.getenv("REDIS_URL", "redis://localhost:6379")
        try:
            from .transport.redis import RedisEventBus, RedisEventStore, connect

            client = await connect(url)
            return RedisEventBus(client), RedisEventStore(client), f"redis({url})"
        except Exception as exc:
            if choice == "redis":
                raise
            reason = f"in-memory (redis unavailable: {type(exc).__name__})"
    else:
        reason = "in-memory"

    from .store.memory import InMemoryEventStore
    from .transport.memory import InMemoryEventBus

    return InMemoryEventBus(), InMemoryEventStore(), reason


async def build(**runtime_kwargs: Any) -> tuple:
    """Build a runtime plus a Config describing what it really is."""
    embedder, embedder_name, is_semantic = build_embedder()
    llm, llm_name = build_llm()
    bus, store, transport_name = await build_transport()

    warnings: List[str] = []
    if not is_semantic:
        warnings.append(
            "semantic routing is running on the hashing fallback: layer-3 "
            "subscriptions match token overlap, not meaning. Do not report "
            "these as semantic-routing results."
        )
    if llm_name.startswith("mock"):
        warnings.append("agents are reasoning with a mock model, not a real one.")

    runtime = EventMemRuntime(
        embedder=embedder, bus=bus, event_store=store, **runtime_kwargs
    )
    config = Config(embedder_name, llm_name, transport_name, is_semantic, warnings)
    return runtime, llm, config


def _azure_client() -> Optional[Any]:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    key = os.getenv("AZURE_OPENAI_API_KEY")
    if not endpoint or not key:
        return None
    try:
        from openai import AzureOpenAI
    except ImportError:
        return None
    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=key,
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
    )


def _openai_client() -> Optional[Any]:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    try:
        from openai import OpenAI
    except ImportError:
        return None
    return OpenAI(api_key=key)

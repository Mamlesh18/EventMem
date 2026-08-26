"""
config.py

Reads the environment and assembles the runtime from real parts when they are
available, or from fallbacks when they are not. Set the Azure and Redis
variables and you get real embeddings, real model calls, and Redis carrying
every event. Leave them unset and the same code runs on a hashing embedder, a
mock model, and an in memory bus, so you can check the wiring with no key.

Environment variables:
    AZURE_OPENAI_ENDPOINT
    AZURE_OPENAI_API_KEY
    AZURE_OPENAI_API_VERSION            default 2025-01-01-preview
    AZURE_OPENAI_CHAT_DEPLOYMENT        your chat model deployment name
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT   your embedding model deployment name
    REDIS_URL                           default redis://localhost:6379
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from embeddings import AzureEmbedder, HashingEmbedder
from llm import AzureLLM, MockLLM
import os
from dotenv import load_dotenv

load_dotenv()
@dataclass
class Selected:
    embedder: Any
    llm: Any
    embedder_name: str
    llm_name: str


def _azure_client() -> Optional[Any]:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    key = os.getenv("AZURE_OPENAI_API_KEY")
    if not endpoint or not key:
        return None
    from openai import AzureOpenAI

    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=key,
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
    )


def build_models() -> Selected:
    client = _azure_client()
    chat_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
    embed_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

    if client is not None and chat_deployment:
        llm: Any = AzureLLM(client, chat_deployment)
        llm_name = f"Azure OpenAI ({chat_deployment})"
    else:
        llm = MockLLM()
        llm_name = "MockLLM (no Azure key found)"

    if client is not None and embed_deployment:
        embedder: Any = AzureEmbedder(client, embed_deployment)
        embedder_name = f"Azure embeddings ({embed_deployment})"
    else:
        embedder = HashingEmbedder()
        embedder_name = "HashingEmbedder (no Azure embedding deployment found)"

    return Selected(embedder, llm, embedder_name, llm_name)


async def build_transport(tracer=None) -> Tuple[Any, Any, str, Any]:
    """
    Returns bus, event_store, a label, and a cleanup coroutine factory. Uses
    Redis when REDIS_URL is reachable, otherwise in memory.
    """
    url = os.getenv("REDIS_URL", "redis://localhost:6379")
    try:
        from redis_bus import RedisEventBus, RedisEventStore, connect

        client = await connect(url)
        bus = RedisEventBus(client)
        store = RedisEventStore(client)

        async def cleanup() -> None:
            await client.aclose()

        return bus, store, f"Redis ({url})", cleanup
    except Exception as exc:
        from bus import InMemoryEventBus
        from event_store import InMemoryEventStore

        async def cleanup() -> None:
            return None

        return InMemoryEventBus(), InMemoryEventStore(), f"in memory (redis unavailable: {exc})", cleanup

"""Embedding backends.

Three tiers, and the differences between them matter enough to state plainly:

``HashingEmbedder``   No model, no network, no install. Signed-hash bag of
                      words. It measures *token overlap*, not meaning: two
                      paraphrases sharing no vocabulary score exactly 0.0.
                      Use it to test plumbing. Do not use it to make a claim
                      about semantic routing, and do not report benchmark
                      numbers from layer-3 subscriptions backed by it without
                      saying so.

``SentenceTransformerEmbedder``  A real local model. No API key, one pip extra.
                      This is the cheapest honest option for semantic routing.

``AzureEmbedder`` / ``OpenAIEmbedder``  Hosted models. Sharpest separation,
                      needs credentials, costs money per call.

All four satisfy the Embedder protocol, so they are interchangeable at the
constructor.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
from typing import Any, Dict, List, Optional

from .similarity import normalize

_TOKEN = re.compile(r"[a-z0-9]+")

#: Tokens carrying no topical signal. Excluded from the hashing embedder
#: because otherwise a shared "the" or "from" produces similarity between
#: unrelated texts, which is how a fallback embedder silently manufactures
#: false positives in semantic routing.
_STOPWORD_TEXT = """
a an and are as at be been but by for from has have he her his i if in is it
its me my no not of on or our she so that the their them then there these they
this to was we were what when which who will with you your
"""
_STOPWORDS = frozenset(_STOPWORD_TEXT.split())


class HashingEmbedder:
    """Signed-hash bag of words, L2 normalised.

    Deterministic across processes and runs because it uses hashlib rather than
    the salted built-in ``hash``. Sublinear term weighting (1 + log tf) so a
    word repeated ten times does not dominate, and stopwords dropped so shared
    function words do not create spurious similarity.

    Honest about what it is: ``semantic`` is False, and the runtime warns once
    if a layer-3 subscription is registered against an embedder that declares
    itself non-semantic.
    """

    name = "hashing"
    semantic = False

    def __init__(self, dim: int = 1024, drop_stopwords: bool = True) -> None:
        self.dim = dim
        self.drop_stopwords = drop_stopwords

    def embed(self, text: str) -> List[float]:
        counts: Dict[int, float] = {}
        signs: Dict[int, float] = {}
        for token in _TOKEN.findall((text or "").lower()):
            if self.drop_stopwords and token in _STOPWORDS:
                continue
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            counts[index] = counts.get(index, 0.0) + 1.0
            signs[index] = 1.0 if digest[4] & 1 else -1.0
        vec = [0.0] * self.dim
        for index, tf in counts.items():
            vec[index] = signs[index] * (1.0 + math.log(tf))
        return normalize(vec)


class SentenceTransformerEmbedder:
    """Local sentence-transformers model. Real semantics, no API key.

    Install with ``pip install eventmem[local]``. The default model is small
    (~80MB) and runs fine on CPU.
    """

    name = "sentence_transformers"
    semantic = True

    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise ImportError(
                "SentenceTransformerEmbedder needs the 'local' extra: "
                "pip install eventmem[local]"
            ) from exc
        self._model = SentenceTransformer(model)
        self.dim = self._model.get_sentence_embedding_dimension()
        self.name = f"sentence_transformers({model})"

    def embed(self, text: str) -> List[float]:
        return self._model.encode(text or " ", normalize_embeddings=True).tolist()

    async def embed_async(self, text: str) -> List[float]:
        # Encoding is CPU-bound and blocking; keep it off the event loop so a
        # batch of embeds does not stall delivery to other agents.
        return await asyncio.to_thread(self.embed, text)


class _HostedEmbedder:
    """Shared logic for hosted OpenAI-compatible embedding endpoints."""

    semantic = True

    def __init__(self, client: Any, deployment: str, dim: int) -> None:
        self.client = client
        self.deployment = deployment
        self.dim = dim

    def embed(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model=self.deployment, input=text or " "
        )
        return response.data[0].embedding

    async def embed_async(self, text: str) -> List[float]:
        # The SDK call is blocking. Running it inline would freeze the runtime
        # for every agent while one embed is in flight.
        return await asyncio.to_thread(self.embed, text)


class AzureEmbedder(_HostedEmbedder):
    """Azure OpenAI embeddings."""

    def __init__(self, client: Any, deployment: str, dim: int = 1536) -> None:
        super().__init__(client, deployment, dim)
        self.name = f"azure({deployment})"


class OpenAIEmbedder(_HostedEmbedder):
    """OpenAI embeddings."""

    def __init__(self, client: Any, model: str = "text-embedding-3-small",
                 dim: int = 1536) -> None:
        super().__init__(client, model, dim)
        self.name = f"openai({model})"


class CachingEmbedder:
    """Memoises another embedder.

    Worth wrapping around anything hosted: benchmark workloads re-embed the
    same subscription queries and repeated content constantly, and an unbounded
    dict is the right trade for a run that lasts seconds.
    """

    def __init__(self, inner: Any, max_entries: Optional[int] = 10_000) -> None:
        self._inner = inner
        self._cache: Dict[str, List[float]] = {}
        self._max = max_entries
        self.dim = getattr(inner, "dim", 0)
        self.semantic = getattr(inner, "semantic", True)
        self.name = f"cached({getattr(inner, 'name', type(inner).__name__)})"
        self.hits = 0
        self.misses = 0

    def embed(self, text: str) -> List[float]:
        if text in self._cache:
            self.hits += 1
            return self._cache[text]
        self.misses += 1
        vec = self._inner.embed(text)
        self._store(text, vec)
        return vec

    async def embed_async(self, text: str) -> List[float]:
        if text in self._cache:
            self.hits += 1
            return self._cache[text]
        self.misses += 1
        if hasattr(self._inner, "embed_async"):
            vec = await self._inner.embed_async(text)
        else:
            vec = self._inner.embed(text)
        self._store(text, vec)
        return vec

    def _store(self, text: str, vec: List[float]) -> None:
        if self._max is not None and len(self._cache) >= self._max:
            # Plain FIFO eviction. Access-ordered eviction is not worth the
            # bookkeeping for a cache this short-lived.
            self._cache.pop(next(iter(self._cache)))
        self._cache[text] = vec

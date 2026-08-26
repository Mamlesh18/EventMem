"""
embeddings.py

Semantic routing needs vectors. Real deployments should use OpenAI embeddings
or a sentence-transformer. For a prototype that must run anywhere with no keys
and no downloads, we ship a small hashing vectoriser. It is not as sharp as a
learned model, but it is deterministic and it captures word overlap well enough
to demonstrate that "database schema" routes closer to "backend storage" than
to "customer sentiment".

Swapping in a real model is a one line change: implement the Embedder protocol
and hand your instance to the runtime.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import List, Protocol

_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    def embed(self, text: str) -> List[float]:
        ...


class HashingEmbedder:
    """
    Bag of words hashed into a fixed number of dimensions, then L2 normalised.
    Deterministic across processes because it uses hashlib rather than the
    salted built in hash.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        for token in _TOKEN.findall(text.lower()):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[index] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return vec
        return [v / norm for v in vec]


def cosine(a: List[float], b: List[float]) -> float:
    """
    Similarity of two vectors. When both are already normalised this is just the
    dot product, but we guard against unnormalised input to stay honest.
    """
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)

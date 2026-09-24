"""Vector similarity helpers."""

from __future__ import annotations

import math
from typing import List, Sequence


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity, guarding against unnormalised or empty input.

    Returns 0.0 rather than raising on a zero vector, because a zero vector is
    what a bag-of-words embedder produces for text with no in-vocabulary
    tokens, and that is a legitimate "no signal" answer rather than an error.
    """
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


def normalize(vec: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return list(vec)
    return [v / norm for v in vec]

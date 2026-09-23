"""Semantic routing, and how to tell whether yours actually works.

Layer 3 lets an agent say "wake me on anything that reads like a security
concern" with no keyword rule. Whether that works depends entirely on the
embedder, and the failure is silent.

This example measures two separate questions rather than asserting either:

  1. Can the embedder tell security concerns from ordinary unrelated work?
     A bag-of-words fallback cannot; a real model can.

  2. Can it resist text that is topically harmless but shares vocabulary with
     the interest query? Run it and see. This is the harder question, and the
     answer is not the comforting one.

    python examples/03_semantic_routing.py
    pip install eventmem[local]     # to include a real local model
"""

import asyncio
import os
from typing import Sequence

# sentence-transformers may pull in a TensorFlow backend whose startup banner
# would bury the measurement this example exists to show.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from eventmem import HashingEmbedder, Subscription

QUERY = "security vulnerability, unsafe handling of untrusted user input"

SHOULD_MATCH = [
    "The login handler concatenates the request body straight into the SQL string",
    "Uploaded filenames are passed to the shell without escaping",
    "An attacker can read another tenant's records by changing the id in the URL",
]

# Ordinary unrelated engineering work. Different topic, different vocabulary.
ORDINARY_NEGATIVES = [
    "Renamed the primary button colour token to match the new palette",
    "The nightly ETL job finished in 12 minutes, down from 19",
    "Updated the onboarding copy for the pricing page",
]

# Topically harmless, but sharing vocabulary with the interest query. This is
# what actually breaks semantic routing in production, and it is the case most
# demos quietly leave out.
LEXICAL_TRAPS = [
    "The user input field on the contact form now has a character counter",
    "Security badge photos are due for renewal this quarter",
]


def measure(embedder, negatives: Sequence[str], label: str) -> None:
    sub = Subscription("reviewer", semantic_query=QUERY)
    cal = sub.calibrate(embedder, SHOULD_MATCH, negatives)

    print(f"\n  {label}")
    for text, score in cal.negatives:
        print(f"    {score:+.4f}  {text[:62]}")
    print(f"    ---- worst positive {cal.worst_positive:+.4f}, "
          f"best negative {cal.best_negative:+.4f}, margin {cal.margin:+.4f}")
    if cal.separable:
        print(f"    SEPARABLE at threshold {cal.suggested_threshold:.3f}"
              f"{' (narrow, fragile)' if cal.margin < 0.15 else ''}")
    else:
        print("    NOT SEPARABLE: no threshold works. A negative outranks a positive.")


def report(name: str, embedder) -> None:
    print(f"\n{name}")
    print("=" * len(name))

    sub = Subscription("reviewer", semantic_query=QUERY)
    cal = sub.calibrate(embedder, SHOULD_MATCH, ORDINARY_NEGATIVES)
    print("  genuine security concerns (should match):")
    for text, score in cal.positives:
        print(f"    {score:+.4f}  {text[:62]}")

    measure(embedder, ORDINARY_NEGATIVES, "Q1 vs ordinary unrelated work:")
    measure(embedder, list(ORDINARY_NEGATIVES) + LEXICAL_TRAPS,
            "Q2 vs the same, plus lexical traps:")


async def main() -> None:
    print("Interest query:")
    print(f"  {QUERY!r}")

    report("HashingEmbedder (the no-dependency fallback)", HashingEmbedder())

    try:
        from eventmem import SentenceTransformerEmbedder

        report("SentenceTransformerEmbedder (a real local model)",
               SentenceTransformerEmbedder())
    except ImportError:
        print("\n\n  (Install eventmem[local] to also measure a real model.)")

    print("""

What this run shows
-------------------
Q1  A real model separates security concerns from unrelated work; the hashing
    fallback does not, because it compares tokens rather than meaning.

Q2  Neither survives the lexical traps. "The user input field on the contact
    form has a character counter" shares almost every content word with the
    interest query while meaning something entirely different, and it outranks
    genuine findings on both embedders.

The useful conclusion is not "semantic routing does not work". It is that a
layer-3 threshold has to be chosen from labelled examples that include the
near-misses you actually expect, and re-checked when the embedder changes.
calibrate() exists so that is a measurement rather than a guess, and it
reports NOT SEPARABLE instead of handing back a threshold that only fits the
examples you happened to try.

When a distinction turns out not to be separable, the fix is usually not a
better threshold: add a layer-1 or layer-2 filter so the semantic check only
ever sees candidates that are already plausible.""")


if __name__ == "__main__":
    asyncio.run(main())

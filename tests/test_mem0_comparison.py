"""The mem0-vs-EventMem benchmark's own correctness.

Scoring bugs produce confident wrong numbers, which is worse than a crash. The
two pinned here were both real: duplicate hits pushed recall to 1.36, and an
event payload field named ``kind`` silently replaced the event envelope's kind
so the dashboard dropped those events.
"""

from __future__ import annotations

import math

import pytest
from benchmarks.mem0_vs_eventmem import corpus
from benchmarks.mem0_vs_eventmem.phases import _median, _score_hits, dedupe, verdicts

# ------------------------------------------------------------------- corpus

def test_corpus_ground_truth_is_consistent():
    assert corpus.validate() == []


def test_corpus_has_enough_to_measure():
    s = corpus.summary()
    assert s["facts"] >= 20
    assert s["queries"] >= 10
    assert s["required_deliveries"] >= 20
    assert s["labelled_relevant_pairs"] >= 20


def test_warmup_fact_is_not_part_of_the_corpus():
    """The warmup write must reach no score, or it contaminates every count."""
    assert corpus.WARMUP_FACT.id not in {f.id for f in corpus.FACTS}
    assert corpus.WARMUP_FACT.needed_by == frozenset()


# ------------------------------------------------------------------ scoring

def test_duplicate_hits_cannot_push_recall_above_one():
    """The bug that produced recall 1.36.

    mem0 writes a fresh row per add(), so a re-ingested corpus returns the same
    memory several times. Counting those separately is double-counting.
    """
    hits = ["f01", "f01", "f01", "f02", "f02"]
    scored = _score_hits(hits, {"f01", "f02"}, k=5)
    assert scored["recall"] == pytest.approx(1.0)
    assert scored["hits"] == 2


def test_recall_denominator_is_capped_at_k():
    """A query with 3 relevant facts asked at k=2 cannot reach recall 1.0 for
    reasons that belong to the harness, not the system."""
    scored = _score_hits(["f01", "f02"], {"f01", "f02", "f03"}, k=2)
    assert scored["recall"] == pytest.approx(1.0)


def test_precision_counts_distinct_relevant_hits():
    scored = _score_hits(["f01", "zz1", "zz2", "zz3"], {"f01"}, k=4)
    assert scored["precision"] == pytest.approx(0.25)


def test_reciprocal_rank_uses_the_first_relevant_position():
    assert _score_hits(["x", "f01"], {"f01"}, k=5)["rr"] == pytest.approx(0.5)
    assert _score_hits(["f01", "x"], {"f01"}, k=5)["rr"] == pytest.approx(1.0)
    assert _score_hits(["x", "y"], {"f01"}, k=5)["rr"] == pytest.approx(0.0)


def test_empty_result_scores_zero_not_nan():
    scored = _score_hits([], {"f01"}, k=5)
    assert scored["precision"] == 0.0
    assert scored["recall"] == 0.0


def test_dedupe_preserves_rank_order_and_drops_blanks():
    assert dedupe(["b", "a", "b", "", "c", "a"]) == ["b", "a", "c"]


def test_median_is_robust_to_a_single_outlier():
    """Why the headline latency is a median: one index-creation cost on the
    first write must not define the reported figure."""
    samples = [70.0, 71.0, 72.0, 73.0, 5000.0]
    assert _median(samples) == pytest.approx(72.0)


# ------------------------------------------------------------------ verdicts

def _row(system, **kw):
    base = {
        "system": system, "recall_at_5": 0.8, "coverage": 0.5,
        "retrievals_spent": 10, "learned_without_asking": 0, "llm_calls": 0,
    }
    base.update(kw)
    return base


def test_close_recall_reads_as_a_tie():
    out = verdicts([_row("mem0", recall_at_5=0.83), _row("eventmem", recall_at_5=0.82)])
    assert any(v["kind"] == "tie" and "Retrieval quality ties" in v["title"] for v in out)


def test_worse_retrieval_is_reported_as_a_loss():
    """The benchmark has to be able to say EventMem lost."""
    out = verdicts([_row("mem0", recall_at_5=0.90), _row("eventmem", recall_at_5=0.55)])
    assert any(v["kind"] == "loss" and "mem0 retrieves better" in v["title"] for v in out)


def test_an_llm_call_despite_infer_false_invalidates_the_write_numbers():
    out = verdicts([_row("mem0", llm_calls=32), _row("eventmem")])
    assert any(v["kind"] == "loss" and "LLM calls" in v["title"] for v in out)


def test_the_caveat_is_always_present():
    """mem0's switched-off features must be stated on every run, not only when
    the numbers happen to favour EventMem."""
    out = verdicts([_row("mem0"), _row("eventmem")])
    assert any(v["kind"] == "caveat" for v in out)


def test_verdicts_are_empty_without_both_systems():
    assert verdicts([_row("mem0")]) == []


# ------------------------------------------------------------------- plumbing

def test_event_payload_cannot_overwrite_the_envelope_kind():
    """The bug that made the dashboard drop system_start events."""
    from benchmarks.mem0_vs_eventmem import run as runner

    before = len(runner._HISTORY)
    runner.publish("system_start", {"system": "x", "mode": "pull"})
    message = runner._HISTORY[-1]
    assert message["kind"] == "system_start"
    assert message["mode"] == "pull"
    del runner._HISTORY[before:]


def test_nan_metrics_survive_formatting():
    """A metric with no data must stay NaN, so a system that did nothing cannot
    look like the fastest."""
    assert math.isnan(_median([]))

"""Unit tests for src.evaluation.significance."""

from __future__ import annotations

import pytest

from src.evaluation.significance import compare_predictions, paired_bootstrap_test


def test_paired_bootstrap_identical_scores_gives_zero_delta() -> None:
    scores = [1.0, 0.0, 1.0, 1.0, 0.0]
    result = paired_bootstrap_test(scores, scores, n_iterations=1000)
    assert result["observed_delta"] == 0.0
    assert not result["significant_at_0_05"]


def test_paired_bootstrap_detects_large_consistent_improvement() -> None:
    baseline = [0.0] * 50
    rag = [1.0] * 50
    result = paired_bootstrap_test(baseline, rag, n_iterations=2000)
    assert result["observed_delta"] == pytest.approx(1.0)
    assert result["significant_at_0_05"]
    assert result["ci_lower"] > 0  # entire CI is above zero


def test_paired_bootstrap_small_noisy_difference_not_significant() -> None:
    # Alternating small differences that don't consistently favor either side.
    baseline = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    rag = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]
    result = paired_bootstrap_test(baseline, rag, n_iterations=2000)
    assert abs(result["observed_delta"]) < 0.3


def test_paired_bootstrap_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        paired_bootstrap_test([1.0, 0.0], [1.0])


def test_paired_bootstrap_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        paired_bootstrap_test([], [])


def test_paired_bootstrap_is_reproducible_with_same_seed() -> None:
    baseline = [0.0, 1.0, 0.0, 1.0, 1.0]
    rag = [1.0, 1.0, 0.0, 0.0, 1.0]
    result_a = paired_bootstrap_test(baseline, rag, n_iterations=500, seed=7)
    result_b = paired_bootstrap_test(baseline, rag, n_iterations=500, seed=7)
    assert result_a == result_b


def _make_record(example_id, question, reference, prediction, answer_type):
    return {
        "example_id": example_id,
        "question": question,
        "reference": reference,
        "prediction": prediction,
        "answer_type": answer_type,
    }


def test_compare_predictions_splits_by_answer_type() -> None:
    baseline_records = [
        _make_record("e1", "Is there a fracture?", "no", "yes", "CLOSED"),
        _make_record("e2", "Is there a fracture?", "yes", "yes", "CLOSED"),
        _make_record("e3", "What is shown?", "pneumonia", "normal lung", "OPEN"),
    ]
    rag_records = [
        _make_record("e1", "Is there a fracture?", "no", "no", "CLOSED"),
        _make_record("e2", "Is there a fracture?", "yes", "yes", "CLOSED"),
        _make_record("e3", "What is shown?", "pneumonia", "pneumonia", "OPEN"),
    ]

    results = compare_predictions(baseline_records, rag_records)

    assert "closed_exact_match" in results
    assert "open_rouge_l" in results
    # RAG fixed e1 and matched e3 exactly -> both deltas should be positive.
    assert results["closed_exact_match"]["observed_delta"] > 0
    assert results["open_rouge_l"]["observed_delta"] > 0


def test_compare_predictions_ignores_examples_missing_from_either_side() -> None:
    baseline_records = [
        _make_record("e1", "Q1", "yes", "yes", "CLOSED"),
        _make_record("e2", "Q2", "no", "no", "CLOSED"),
    ]
    rag_records = [
        _make_record("e1", "Q1", "yes", "yes", "CLOSED"),
        # e2 missing from rag_records; e3 only in rag_records.
        _make_record("e3", "Q3", "yes", "yes", "CLOSED"),
    ]
    results = compare_predictions(baseline_records, rag_records)
    # Only e1 is shared -> 1 example in the comparison.
    assert results["closed_exact_match"]["n_examples"] == 1

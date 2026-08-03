from __future__ import annotations

import pytest

from src.evaluation.metrics import (
    compute_bleu,
    compute_exact_match,
    compute_metrics_batch,
    compute_rouge_l,
    normalize_answer,
)


def test_normalize_answer_strips_articles_punctuation_case() -> None:
    assert normalize_answer("  A Fracture.  ") == "fracture"
    assert normalize_answer("The lung is clear") == "lung is clear"


def test_exact_match_ignores_surface_variation() -> None:
    assert compute_exact_match("A fracture.", "fracture") == 1.0
    assert compute_exact_match("no", "yes") == 0.0


def test_compute_bleu_identical_strings_scores_high() -> None:
    score = compute_bleu("no acute abnormality", "no acute abnormality")
    assert score > 0.9


def test_compute_bleu_unrelated_strings_scores_low() -> None:
    score = compute_bleu("pneumothorax present", "normal chest x-ray")
    assert score < 0.3


def test_compute_rouge_l_identical_strings_scores_high() -> None:
    score = compute_rouge_l("left lower lobe consolidation", "left lower lobe consolidation")
    assert score > 0.9


def test_compute_metrics_batch_splits_by_answer_type() -> None:
    predictions = ["yes", "no", "pneumonia", "a small pleural effusion"]
    references = ["yes", "yes", "pneumonia", "small pleural effusion"]
    answer_types = ["CLOSED", "CLOSED", "OPEN", "OPEN"]

    summary = compute_metrics_batch(predictions, references, answer_types)

    assert summary["num_examples"] == 4
    assert summary["num_closed"] == 2
    assert summary["num_open"] == 2
    assert summary["closed_accuracy"] == 0.5
    assert 0.0 <= summary["open_bleu"] <= 1.0
    assert 0.0 <= summary["open_rouge_l"] <= 1.0
    assert summary["open_rouge_l"] > 0.5


def test_compute_metrics_batch_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        compute_metrics_batch(["a", "b"], ["a"], ["CLOSED", "CLOSED"])


def test_compute_metrics_batch_rejects_empty_batch() -> None:
    with pytest.raises(ValueError):
        compute_metrics_batch([], [], [])


def test_compute_metrics_batch_all_closed_no_open_scores() -> None:
    summary = compute_metrics_batch(["yes", "no"], ["yes", "no"], ["CLOSED", "CLOSED"])
    assert summary["num_open"] == 0
    assert summary["open_bleu"] == 0.0
    assert summary["open_rouge_l"] == 0.0
    assert summary["closed_accuracy"] == 1.0

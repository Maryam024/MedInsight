from __future__ import annotations

from src.evaluation.error_analysis import categorize_examples, summarize_retrieval_quality


def _make_record(example_id, question, reference, prediction, answer_type):
    return {
        "example_id": example_id,
        "question": question,
        "reference": reference,
        "prediction": prediction,
        "answer_type": answer_type,
    }


def test_categorize_examples_all_four_categories() -> None:
    baseline_records = [
        _make_record("improved_ex", "Q?", "yes", "no", "CLOSED"),
        _make_record("regressed_ex", "Q?", "yes", "yes", "CLOSED"),
        _make_record("both_correct_ex", "Q?", "yes", "yes", "CLOSED"),
        _make_record("both_wrong_ex", "Q?", "yes", "no", "CLOSED"),
    ]
    rag_records = [
        _make_record("improved_ex", "Q?", "yes", "yes", "CLOSED"),
        _make_record("regressed_ex", "Q?", "yes", "no", "CLOSED"),
        _make_record("both_correct_ex", "Q?", "yes", "yes", "CLOSED"),
        _make_record("both_wrong_ex", "Q?", "yes", "no", "CLOSED"),
    ]

    categorized, summary = categorize_examples(baseline_records, rag_records)

    by_id = {c["example_id"]: c["category"] for c in categorized}
    assert by_id["improved_ex"] == "improved"
    assert by_id["regressed_ex"] == "regressed"
    assert by_id["both_correct_ex"] == "both_correct"
    assert by_id["both_wrong_ex"] == "both_wrong"

    assert summary["counts"] == {
        "improved": 1,
        "regressed": 1,
        "both_correct": 1,
        "both_wrong": 1,
    }


def test_categorize_examples_respects_max_examples_cap() -> None:
    baseline_records = [
        _make_record(f"e{i}", "Q?", "yes", "no", "CLOSED") for i in range(20)
    ]
    rag_records = [
        _make_record(f"e{i}", "Q?", "yes", "yes", "CLOSED") for i in range(20)
    ]
    _, summary = categorize_examples(baseline_records, rag_records, max_examples_per_category=3)
    assert summary["counts"]["improved"] == 20
    assert len(summary["example_ids"]["improved"]) == 3


def test_categorize_examples_open_questions_use_rouge_threshold() -> None:
    baseline_records = [
        _make_record(
            "open1", "What is shown?", "pneumonia present", "totally unrelated text", "OPEN"
        ),
    ]
    rag_records = [
        _make_record("open1", "What is shown?", "pneumonia present", "pneumonia present", "OPEN"),
    ]
    categorized, summary = categorize_examples(baseline_records, rag_records)
    assert categorized[0]["category"] == "improved"


def test_summarize_retrieval_quality_computes_means() -> None:
    rag_records = [
        {
            "retrieved_evidence": [
                {"pair_id": "p1", "caption": "c1", "similarity_score": 0.9},
                {"pair_id": "p2", "caption": "c2", "similarity_score": 0.8},
            ]
        },
        {
            "retrieved_evidence": [
                {"pair_id": "p3", "caption": "c3", "similarity_score": 0.7},
            ]
        },
        {"retrieved_evidence": []},
    ]
    summary = summarize_retrieval_quality(rag_records)
    assert summary["n_examples"] == 3
    assert summary["mean_top1_similarity"] == (0.9 + 0.7) / 2
    assert summary["mean_num_retrieved"] == (2 + 1 + 0) / 3
    assert summary["pct_examples_with_no_evidence"] == 1 / 3


def test_summarize_retrieval_quality_handles_all_empty() -> None:
    rag_records = [{"retrieved_evidence": []}, {"retrieved_evidence": []}]
    summary = summarize_retrieval_quality(rag_records)
    assert summary["mean_top1_similarity"] is None
    assert summary["pct_examples_with_no_evidence"] == 1.0

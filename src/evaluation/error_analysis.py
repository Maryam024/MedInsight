from __future__ import annotations

from typing import TypedDict

from src.evaluation.metrics import compute_exact_match, compute_rouge_l
from src.utils.logger import get_logger

logger = get_logger(__name__)

Category = str


class CategorizedExample(TypedDict):
    example_id: str
    question: str
    reference: str
    baseline_prediction: str
    rag_prediction: str
    answer_type: str
    category: Category


class CategorySummary(TypedDict):
    counts: dict[str, int]
    example_ids: dict[str, list[str]]


def _is_correct(prediction: str, reference: str, answer_type: str) -> bool:
    if answer_type == "CLOSED":
        return compute_exact_match(prediction, reference) == 1.0
    return compute_rouge_l(prediction, reference) >= 0.5  # rough cutoff, not validated


def categorize_examples(
    baseline_records: list[dict],
    rag_records: list[dict],
    max_examples_per_category: int = 10,
) -> tuple[list[CategorizedExample], CategorySummary]:
    baseline_by_id = {r["example_id"]: r for r in baseline_records}
    rag_by_id = {r["example_id"]: r for r in rag_records}
    shared_ids = sorted(set(baseline_by_id) & set(rag_by_id))

    categorized: list[CategorizedExample] = []
    for example_id in shared_ids:
        baseline_rec = baseline_by_id[example_id]
        rag_rec = rag_by_id[example_id]
        reference = baseline_rec["reference"]
        answer_type = baseline_rec["answer_type"]

        baseline_correct = _is_correct(baseline_rec["prediction"], reference, answer_type)
        rag_correct = _is_correct(rag_rec["prediction"], reference, answer_type)

        if rag_correct and not baseline_correct:
            category: Category = "improved"
        elif baseline_correct and not rag_correct:
            category = "regressed"
        elif baseline_correct and rag_correct:
            category = "both_correct"
        else:
            category = "both_wrong"

        categorized.append(
            {
                "example_id": example_id,
                "question": baseline_rec["question"],
                "reference": reference,
                "baseline_prediction": baseline_rec["prediction"],
                "rag_prediction": rag_rec["prediction"],
                "answer_type": answer_type,
                "category": category,
            }
        )

    counts: dict[str, int] = {"improved": 0, "regressed": 0, "both_correct": 0, "both_wrong": 0}
    example_ids: dict[str, list[str]] = {k: [] for k in counts}
    for item in categorized:
        counts[item["category"]] += 1
        if len(example_ids[item["category"]]) < max_examples_per_category:
            example_ids[item["category"]].append(item["example_id"])

    summary: CategorySummary = {"counts": counts, "example_ids": example_ids}
    logger.info("Error analysis category counts: %s", counts)
    return categorized, summary


def summarize_retrieval_quality(rag_records: list[dict]) -> dict:
    top1_similarities = []
    num_retrieved_counts = []
    n_with_no_evidence = 0

    for record in rag_records:
        evidence = record.get("retrieved_evidence", [])
        num_retrieved_counts.append(len(evidence))
        if evidence:
            top1_similarities.append(evidence[0]["similarity_score"])
        else:
            n_with_no_evidence += 1

    n = len(rag_records)
    summary = {
        "n_examples": n,
        "mean_top1_similarity": (
            sum(top1_similarities) / len(top1_similarities) if top1_similarities else None
        ),
        "mean_num_retrieved": sum(num_retrieved_counts) / n if n else 0.0,
        "pct_examples_with_no_evidence": (n_with_no_evidence / n) if n else 0.0,
    }
    logger.info("Retrieval quality summary: %s", summary)
    return summary

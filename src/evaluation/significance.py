from __future__ import annotations

import random
from typing import TypedDict

from src.utils.logger import get_logger

logger = get_logger(__name__)


class BootstrapResult(TypedDict):
    observed_delta: float
    ci_lower: float
    ci_upper: float
    p_value: float
    n_examples: int
    n_iterations: int
    significant_at_0_05: bool


def paired_bootstrap_test(
    baseline_scores: list[float],
    rag_scores: list[float],
    n_iterations: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    if len(baseline_scores) != len(rag_scores):
        raise ValueError(
            f"baseline_scores and rag_scores must be the same length, "
            f"got {len(baseline_scores)} and {len(rag_scores)}"
        )
    if not baseline_scores:
        raise ValueError("Cannot run a significance test on an empty set of scores.")

    n = len(baseline_scores)
    diffs = [rag_scores[i] - baseline_scores[i] for i in range(n)]
    observed_delta = sum(diffs) / n

    rng = random.Random(seed)
    bootstrap_deltas: list[float] = []
    for _ in range(n_iterations):
        resample = [diffs[rng.randrange(n)] for _ in range(n)]
        bootstrap_deltas.append(sum(resample) / n)

    bootstrap_deltas.sort()
    alpha = 1.0 - confidence_level
    lower_idx = int((alpha / 2) * n_iterations)
    upper_idx = int((1 - alpha / 2) * n_iterations) - 1
    ci_lower = bootstrap_deltas[max(0, lower_idx)]
    ci_upper = bootstrap_deltas[min(n_iterations - 1, upper_idx)]

    if observed_delta >= 0:
        p_value = 2 * (sum(1 for d in bootstrap_deltas if d <= 0) / n_iterations)
    else:
        p_value = 2 * (sum(1 for d in bootstrap_deltas if d >= 0) / n_iterations)
    p_value = min(p_value, 1.0)

    result: BootstrapResult = {
        "observed_delta": observed_delta,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
        "n_examples": n,
        "n_iterations": n_iterations,
        "significant_at_0_05": p_value < 0.05,
    }
    logger.info("Paired bootstrap test result: %s", result)
    return result


def compare_predictions(
    baseline_records: list[dict],
    rag_records: list[dict],
) -> dict[str, BootstrapResult]:
    from src.evaluation.metrics import compute_exact_match, compute_rouge_l

    baseline_by_id = {r["example_id"]: r for r in baseline_records}
    rag_by_id = {r["example_id"]: r for r in rag_records}
    shared_ids = sorted(set(baseline_by_id) & set(rag_by_id))

    missing_from_baseline = set(rag_by_id) - set(baseline_by_id)
    missing_from_rag = set(baseline_by_id) - set(rag_by_id)
    if missing_from_baseline or missing_from_rag:
        logger.warning(
            "%d example(s) only in RAG predictions, %d only in baseline "
            "predictions — these are excluded from the paired comparison.",
            len(missing_from_baseline),
            len(missing_from_rag),
        )

    closed_baseline, closed_rag = [], []
    open_baseline, open_rag = [], []

    for example_id in shared_ids:
        baseline_rec = baseline_by_id[example_id]
        rag_rec = rag_by_id[example_id]
        reference = baseline_rec["reference"]
        answer_type = baseline_rec["answer_type"]

        if answer_type == "CLOSED":
            closed_baseline.append(compute_exact_match(baseline_rec["prediction"], reference))
            closed_rag.append(compute_exact_match(rag_rec["prediction"], reference))
        else:
            open_baseline.append(compute_rouge_l(baseline_rec["prediction"], reference))
            open_rag.append(compute_rouge_l(rag_rec["prediction"], reference))

    results: dict[str, BootstrapResult] = {}
    if closed_baseline:
        results["closed_exact_match"] = paired_bootstrap_test(closed_baseline, closed_rag)
    if open_baseline:
        results["open_rouge_l"] = paired_bootstrap_test(open_baseline, open_rag)

    return results

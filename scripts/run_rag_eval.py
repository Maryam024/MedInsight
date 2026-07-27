from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
from pathlib import Path

from src.data.dataset import VQARadDataset
from src.evaluation.metrics import compute_metrics_batch
from src.models.rag_vlm import RAGVLM
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

BASELINE_METRICS_PATH = Path("experiments/logs/baseline_metrics.json")


def run_rag_eval(limit: int | None = None) -> dict:
   
    config = load_config()

    manifest_path = Path("data/processed/vqa_rad/manifest.jsonl")
    image_root = Path("data/processed/vqa_rad")
    dataset = VQARadDataset(manifest_path, image_root, split="test", transform=None)

    if limit is not None:
        logger.info("Limiting evaluation to the first %d examples (smoke test).", limit)

    model = RAGVLM.from_config(config)

    predictions: list[str] = []
    references: list[str] = []
    answer_types: list[str] = []
    per_example_records: list[dict] = []

    n = len(dataset) if limit is None else min(limit, len(dataset))
    for i in range(n):
        item = dataset[i]
        result = model.generate(item["image"], item["question"])

        predictions.append(result["answer"])
        references.append(item["answer"])
        answer_types.append(item["answer_type"])
        per_example_records.append(
            {
                "example_id": item["example_id"],
                "question": item["question"],
                "reference": item["answer"],
                "prediction": result["answer"],
                "answer_type": item["answer_type"],
                "retrieved_evidence": result["retrieved_evidence"],
            }
        )

        if (i + 1) % 50 == 0:
            logger.info("Generated predictions for %d/%d examples", i + 1, n)

    out_dir = Path("experiments/logs")
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = out_dir / "rag_predictions.jsonl"
    with open(predictions_path, "w", encoding="utf-8") as f:
        for record in per_example_records:
            f.write(json.dumps(record) + "\n")
    logger.info("Wrote %d predictions to %s", len(per_example_records), predictions_path)

    metrics = compute_metrics_batch(predictions, references, answer_types)

    metrics_path = out_dir / "rag_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Wrote metrics summary to %s: %s", metrics_path, metrics)

    _print_comparison_if_available(metrics)

    return metrics


def _print_comparison_if_available(rag_metrics: dict) -> None:
   
    if not BASELINE_METRICS_PATH.is_file():
        logger.info(
            "No baseline metrics found at %s — run scripts/run_baseline.py "
            "to see a side-by-side comparison next time.",
            BASELINE_METRICS_PATH,
        )
        return

    with open(BASELINE_METRICS_PATH, "r", encoding="utf-8") as f:
        baseline_metrics = json.load(f)

    print("\n=== Baseline vs. Retrieval-Augmented (VQA-RAD test split) ===")
    for key in ("closed_accuracy", "open_bleu", "open_rouge_l", "overall_exact_match"):
        baseline_value = baseline_metrics.get(key, float("nan"))
        rag_value = rag_metrics.get(key, float("nan"))
        delta = rag_value - baseline_value
        sign = "+" if delta >= 0 else ""
        print(
            f"{key:22s}  baseline={baseline_value:.4f}  "
            f"rag={rag_value:.4f}  delta={sign}{delta:.4f}"
        )
    print("===============================================================\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N test examples (smoke test).",
    )
    args = parser.parse_args()
    run_rag_eval(limit=args.limit)


if __name__ == "__main__":
    main()

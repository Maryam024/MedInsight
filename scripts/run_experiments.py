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


def run_topk_sweep(top_k_values: list[int], limit: int | None = None) -> dict[int, dict]:
    config = load_config()

    manifest_path = Path("data/processed/vqa_rad/manifest.jsonl")
    image_root = Path("data/processed/vqa_rad")
    dataset = VQARadDataset(manifest_path, image_root, split="test", transform=None)
    n = len(dataset) if limit is None else min(limit, len(dataset))

    model = RAGVLM.from_config(config, top_k_override=max(top_k_values))

    results: dict[int, dict] = {}
    for top_k in sorted(top_k_values):
        model.top_k = top_k
        logger.info("Running RAG eval with top_k=%d over %d examples", top_k, n)

        predictions, references, answer_types = [], [], []
        for i in range(n):
            item = dataset[i]
            result = model.generate(item["image"], item["question"])
            predictions.append(result["answer"])
            references.append(item["answer"])
            answer_types.append(item["answer_type"])

        metrics = compute_metrics_batch(predictions, references, answer_types)
        results[top_k] = metrics
        logger.info("top_k=%d -> %s", top_k, metrics)

    out_dir = Path("experiments/logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = out_dir / "topk_sweep.json"
    with open(sweep_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    logger.info("Wrote top_k sweep results to %s", sweep_path)

    _print_sweep_table(results)
    return results


def _print_sweep_table(results: dict[int, dict]) -> None:
    print("\n=== top_k sweep (VQA-RAD test split) ===")
    print(f"{'top_k':>6}  {'closed_acc':>10}  {'open_bleu':>10}  {'open_rouge_l':>12}")
    for top_k in sorted(results):
        m = results[top_k]
        print(
            f"{top_k:>6}  {m['closed_accuracy']:>10.4f}  "
            f"{m['open_bleu']:>10.4f}  {m['open_rouge_l']:>12.4f}"
        )
    print("=========================================\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--top-k-values",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10],
        help="top_k values to sweep (default: 1 3 5 10).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N test examples per sweep point (smoke test).",
    )
    args = parser.parse_args()
    run_topk_sweep(top_k_values=args.top_k_values, limit=args.limit)


if __name__ == "__main__":
    main()

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  
import matplotlib.pyplot as plt
from PIL import Image

from src.data.schema import ImageCaptionPair, VQAExample
from src.utils.logger import get_logger

logger = get_logger(__name__)

EDA_DIR = Path("experiments/logs/eda")


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _plot_bar(counts: dict, title: str, xlabel: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = list(counts.keys())
    values = list(counts.values())
    ax.bar(labels, values, color="#4C72B0")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_hist(values: list[float], title: str, xlabel: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=30, color="#55A868")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def analyze_rocov2(
    manifest_path: Path, image_root: Path
) -> dict:
    raw = _read_jsonl(manifest_path)
    records = [ImageCaptionPair(**r) for r in raw]

    modality_counts = Counter(r.modality for r in records)
    caption_lengths = [len(r.caption.split()) for r in records]

    sample = records[:200]
    resolutions = []
    for r in sample:
        img_path = image_root / r.image_path
        if img_path.is_file():
            with Image.open(img_path) as img:
                resolutions.append(img.size)

    stats = {
        "num_pairs": len(records),
        "modality_distribution": dict(modality_counts),
        "caption_length_words": {
            "min": min(caption_lengths, default=0),
            "max": max(caption_lengths, default=0),
            "mean": sum(caption_lengths) / len(caption_lengths) if caption_lengths else 0,
            "pct_over_128_tokens": (
                sum(1 for length in caption_lengths if length > 128) / len(caption_lengths)
                if caption_lengths
                else 0
            ),
        },
        "sampled_image_resolutions": {
            "n_sampled": len(resolutions),
            "distinct_resolutions": len(set(resolutions)),
        },
    }

    EDA_DIR.mkdir(parents=True, exist_ok=True)
    if modality_counts:
        _plot_bar(
            dict(modality_counts),
            "ROCOv2: Modality Distribution",
            "Modality",
            EDA_DIR / "rocov2_modality_distribution.png",
        )
    if caption_lengths:
        _plot_hist(
            caption_lengths,
            "ROCOv2: Caption Length (words)",
            "Words per caption",
            EDA_DIR / "rocov2_caption_lengths.png",
        )

    with open(EDA_DIR / "rocov2_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.info("ROCOv2 EDA complete: %s", stats)
    return stats


def analyze_vqa_rad(manifest_path: Path) -> dict:
    raw = _read_jsonl(manifest_path)
    records = [VQAExample(**r) for r in raw]

    answer_type_counts = Counter(r.answer_type for r in records)
    split_counts = Counter(r.split for r in records)
    question_lengths = [len(r.question.split()) for r in records]
    answer_lengths = [len(r.answer.split()) for r in records]

    stats = {
        "num_examples": len(records),
        "answer_type_distribution": dict(answer_type_counts),
        "split_distribution": dict(split_counts),
        "question_length_words": {
            "mean": sum(question_lengths) / len(question_lengths) if question_lengths else 0,
            "max": max(question_lengths, default=0),
        },
        "answer_length_words": {
            "mean": sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0,
            "max": max(answer_lengths, default=0),
        },
    }

    EDA_DIR.mkdir(parents=True, exist_ok=True)
    if answer_type_counts:
        _plot_bar(
            dict(answer_type_counts),
            "VQA-RAD: Answer Type Distribution",
            "Answer Type",
            EDA_DIR / "vqa_rad_answer_type_distribution.png",
        )
    if split_counts:
        _plot_bar(
            dict(split_counts),
            "VQA-RAD: Split Distribution",
            "Split",
            EDA_DIR / "vqa_rad_split_distribution.png",
        )

    with open(EDA_DIR / "vqa_rad_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.info("VQA-RAD EDA complete: %s", stats)
    return stats


def write_findings_markdown(rocov2_stats: dict | None, vqa_rad_stats: dict | None) -> None:
    lines = [
        "# EDA Findings",
        "",
        "> Auto-generated by `scripts/run_eda.py`. Do not hand-edit — re-run the",
        "> script after any change to the manifests and commit the regenerated file.",
        "",
    ]

    if rocov2_stats:
        lines += [
            "## ROCOv2 (Retrieval Corpus)",
            "",
            f"- Total pairs: **{rocov2_stats['num_pairs']}**",
            f"- Modality distribution: {rocov2_stats['modality_distribution']}",
            f"- Caption length (words): mean={rocov2_stats['caption_length_words']['mean']:.1f}, "
            f"max={rocov2_stats['caption_length_words']['max']}",
            f"- Captions exceeding the 128-token cap: "
            f"{rocov2_stats['caption_length_words']['pct_over_128_tokens']:.1%}",
            "",
            "![Modality distribution](../experiments/logs/eda/rocov2_modality_distribution.png)",
            "![Caption lengths](../experiments/logs/eda/rocov2_caption_lengths.png)",
            "",
        ]
    else:
        lines += [
            "## ROCOv2 (Retrieval Corpus)",
            "",
            "_No manifest found — run `python scripts/download_data.py --dataset rocov2` first._",
            "",
        ]

    if vqa_rad_stats:
        lines += [
            "## VQA-RAD (Evaluation Benchmark)",
            "",
            f"- Total examples: **{vqa_rad_stats['num_examples']}**",
            f"- Answer type distribution: {vqa_rad_stats['answer_type_distribution']}",
            f"- Split distribution: {vqa_rad_stats['split_distribution']}",
            f"- Question length (words): mean={vqa_rad_stats['question_length_words']['mean']:.1f}",
            f"- Answer length (words): mean={vqa_rad_stats['answer_length_words']['mean']:.1f}",
            "",
            "![Answer type distribution](../experiments/logs/eda/vqa_rad_answer_type_distribution.png)",  # noqa: E501
            "![Split distribution](../experiments/logs/eda/vqa_rad_split_distribution.png)",
            "",
        ]
    else:
        lines += [
            "## VQA-RAD (Evaluation Benchmark)",
            "",
            "_No manifest found — run `python scripts/download_data.py --dataset vqa_rad` first._",  # noqa: E501
            "",
        ]

    Path("docs/eda_findings.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote docs/eda_findings.md")


def main() -> None:
    rocov2_manifest = Path("data/processed/rocov2/manifest.jsonl")
    vqa_rad_manifest = Path("data/processed/vqa_rad/manifest.jsonl")

    rocov2_stats = None
    vqa_rad_stats = None

    if rocov2_manifest.is_file():
        rocov2_stats = analyze_rocov2(rocov2_manifest, Path("data/processed/rocov2"))
    else:
        logger.warning("Skipping ROCOv2 EDA: %s not found", rocov2_manifest)

    if vqa_rad_manifest.is_file():
        vqa_rad_stats = analyze_vqa_rad(vqa_rad_manifest)
    else:
        logger.warning("Skipping VQA-RAD EDA: %s not found", vqa_rad_manifest)

    write_findings_markdown(rocov2_stats, vqa_rad_stats)


if __name__ == "__main__":
    main()

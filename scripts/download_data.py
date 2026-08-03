#!/usr/bin/env python

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from src.data.preprocessing import stratified_split, truncate_caption
from src.data.schema import ImageCaptionPair, VQAExample
from src.utils.config_loader import MedInsightConfig, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _write_manifest(records: list[ImageCaptionPair] | list[VQAExample], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(record.model_dump_json() + "\n")
    logger.info("Wrote %d records to %s", len(records), out_path)


def download_rocov2(config: MedInsightConfig) -> None:
    from datasets import load_dataset

    data_cfg = config.data["retrieval_corpus"]
    cache_dir = Path(data_cfg["local_path"]) / "hf_cache"
    images_dir = Path("data/processed/rocov2/images")
    images_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading ROCOv2 from %s ...", data_cfg["source_url"])
    dataset = load_dataset(
        "eltorio/ROCOv2-radiology", split="train", cache_dir=str(cache_dir)
    )

    max_caption_length = config.data["preprocessing"]["max_caption_length"]
    known_modalities = {"CT", "MRI", "X-ray", "Ultrasound"}

    records: list[ImageCaptionPair] = []
    for i, example in tqdm(
        enumerate(dataset), total=len(dataset), desc="Writing ROCOv2 images to disk"
    ):
        pair_id = f"rocov2_{i:06d}"
        image_filename = f"{pair_id}.jpg"
        example["image"].convert("RGB").save(images_dir / image_filename)

        raw_modality = str(example.get("modality", "")).strip()
        modality = raw_modality if raw_modality in known_modalities else "Other"

        records.append(
            ImageCaptionPair(
                pair_id=pair_id,
                image_path=f"images/{image_filename}",
                caption=truncate_caption(example["caption"], max_caption_length),
                modality=modality,
                source_id=str(example.get("id", pair_id)),
            )
        )

    _write_manifest(records, Path("data/processed/rocov2/manifest.jsonl"))


_YES_NO_ANSWERS = {"yes", "no"}


def _infer_answer_type(answer: str) -> str:
    normalized = answer.strip().lower()
    return "CLOSED" if normalized in _YES_NO_ANSWERS else "OPEN"


def download_vqa_rad(config: MedInsightConfig) -> None:
    from datasets import load_dataset

    data_cfg = config.data["retrieval_corpus"]
    eval_cfg = config.data["evaluation_set"]
    cache_dir = Path(eval_cfg["local_path"]) / "hf_cache"
    images_dir = Path("data/processed/vqa_rad/images")
    images_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading VQA-RAD from %s ...", eval_cfg["source_url"])
    hf_splits = {
        "train": load_dataset(
            "flaviagiammarino/vqa-rad", split="train", cache_dir=str(cache_dir)
        ),
        "test": load_dataset(
            "flaviagiammarino/vqa-rad", split="test", cache_dir=str(cache_dir)
        ),
    }

    def _build_examples(hf_split_name: str, dataset) -> list[dict]:
        examples = []
        for i, example in tqdm(
            enumerate(dataset),
            total=len(dataset),
            desc=f"Writing VQA-RAD [{hf_split_name}] images to disk",
        ):
            example_id = f"vqarad_{hf_split_name}_{i:06d}"
            image_filename = f"{example_id}.jpg"
            example["image"].convert("RGB").save(images_dir / image_filename)
            answer = str(example["answer"])
            examples.append(
                {
                    "example_id": example_id,
                    "image_path": f"images/{image_filename}",
                    "question": example["question"],
                    "answer": answer,
                    "answer_type": _infer_answer_type(answer),
                }
            )
        return examples

    train_pool = _build_examples("train", hf_splits["train"])
    held_out_test = _build_examples("test", hf_splits["test"])

    splits_cfg = config.data["splits"]
    train_val_total = splits_cfg["train_ratio"] + splits_cfg["val_ratio"]
    val_share_of_pool = splits_cfg["val_ratio"] / train_val_total

    train, val, _unused = stratified_split(
        train_pool,
        train_ratio=1.0 - val_share_of_pool,
        val_ratio=val_share_of_pool,
        test_ratio=0.0,
        stratify_key=lambda r: r["answer_type"],
        seed=config.project.seed,
    )

    records: list[VQAExample] = []
    for split_name, split_examples in (
        ("train", train),
        ("val", val),
        ("test", held_out_test),
    ):
        for ex in split_examples:
            records.append(VQAExample(**ex, split=split_name))

    n_closed_test = sum(1 for ex in held_out_test if ex["answer_type"] == "CLOSED")
    logger.info(
        "VQA-RAD official test split: %d examples (%d CLOSED, %d OPEN by "
        "recovered answer_type)",
        len(held_out_test),
        n_closed_test,
        len(held_out_test) - n_closed_test,
    )

    _write_manifest(records, Path("data/processed/vqa_rad/manifest.jsonl"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["rocov2", "vqa_rad", "all"],
        default="all",
        help="Which dataset to download (default: all).",
    )
    args = parser.parse_args()

    config = load_config()

    if args.dataset in ("rocov2", "all"):
        download_rocov2(config)
    if args.dataset in ("vqa_rad", "all"):
        download_vqa_rad(config)

    logger.info("Download complete.")


if __name__ == "__main__":
    main()

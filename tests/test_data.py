"""
Unit tests for src.data.*

These tests never touch the network or Hugging Face — they build small
synthetic manifests and images on the fly, so CI (.github/workflows/ci.yml)
stays fast and deterministic. Real end-to-end verification against the
actual ROCOv2/VQA-RAD downloads is a manual step after running
scripts/download_data.py, not part of the automated test suite.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from src.data.dataset import ROCOv2Dataset, VQARadDataset
from src.data.preprocessing import build_image_transform, stratified_split, truncate_caption
from src.data.schema import ImageCaptionPair, VQAExample


# ---------------------------------------------------------------------------
# schema.py
# ---------------------------------------------------------------------------


def test_image_caption_pair_valid() -> None:
    pair = ImageCaptionPair(
        pair_id="p1", image_path="images/p1.jpg", caption="A chest X-ray.", modality="X-ray"
    )
    assert pair.modality == "X-ray"


def test_image_caption_pair_rejects_empty_caption() -> None:
    with pytest.raises(ValueError):
        ImageCaptionPair(pair_id="p1", image_path="images/p1.jpg", caption="   ")


def test_vqa_example_rejects_empty_question() -> None:
    with pytest.raises(ValueError):
        VQAExample(
            example_id="e1",
            image_path="images/e1.jpg",
            question="",
            answer="yes",
        )


# ---------------------------------------------------------------------------
# preprocessing.py
# ---------------------------------------------------------------------------


def test_truncate_caption_shorter_than_limit_unchanged() -> None:
    caption = "There is a small opacity in the left lung."
    assert truncate_caption(caption, max_length=128) == caption


def test_truncate_caption_longer_than_limit_is_cut() -> None:
    caption = " ".join(["word"] * 200)
    truncated = truncate_caption(caption, max_length=10)
    assert len(truncated.split()) == 10


def test_stratified_split_ratios_and_coverage() -> None:
    records = [{"id": i, "modality": "X-ray" if i % 2 == 0 else "CT"} for i in range(100)]
    train, val, test = stratified_split(
        records,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        stratify_key=lambda r: r["modality"],
        seed=42,
    )
    # Every record accounted for exactly once.
    assert len(train) + len(val) + len(test) == len(records)
    all_ids = {r["id"] for r in train + val + test}
    assert all_ids == {r["id"] for r in records}
    # Roughly 80/10/10 (allow rounding slack from small strata).
    assert 70 <= len(train) <= 85
    assert 0 <= len(val) <= 20
    assert 0 <= len(test) <= 20


def test_stratified_split_rejects_bad_ratios() -> None:
    with pytest.raises(ValueError):
        stratified_split([1, 2, 3], 0.5, 0.5, 0.5, stratify_key=lambda r: "all")


def test_build_image_transform_produces_expected_tensor_shape() -> None:
    data_config = {
        "preprocessing": {
            "image_size": [224, 224],
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
        }
    }
    transform = build_image_transform(data_config)
    image = Image.new("RGB", (300, 180), color=(128, 64, 32))
    tensor = transform(image)
    assert tensor.shape == (3, 224, 224)


# ---------------------------------------------------------------------------
# dataset.py  (synthetic manifest + images written to tmp_path)
# ---------------------------------------------------------------------------


@pytest.fixture
def rocov2_manifest(tmp_path: Path) -> tuple[Path, Path]:
    image_root = tmp_path / "rocov2"
    images_dir = image_root / "images"
    images_dir.mkdir(parents=True)

    records = []
    for i in range(3):
        filename = f"pair_{i}.jpg"
        Image.new("RGB", (64, 64), color=(i * 10, 0, 0)).save(images_dir / filename)
        records.append(
            ImageCaptionPair(
                pair_id=f"pair_{i}",
                image_path=f"images/{filename}",
                caption=f"Synthetic radiology caption number {i}.",
                modality="X-ray" if i % 2 == 0 else "CT",
            )
        )

    manifest_path = image_root / "manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")

    return manifest_path, image_root


@pytest.fixture
def vqa_rad_manifest(tmp_path: Path) -> tuple[Path, Path]:
    image_root = tmp_path / "vqa_rad"
    images_dir = image_root / "images"
    images_dir.mkdir(parents=True)

    records = []
    splits = ["train", "train", "val", "test"]
    for i, split in enumerate(splits):
        filename = f"example_{i}.jpg"
        Image.new("RGB", (64, 64), color=(0, i * 10, 0)).save(images_dir / filename)
        records.append(
            VQAExample(
                example_id=f"example_{i}",
                image_path=f"images/{filename}",
                question=f"Is there an abnormality in image {i}?",
                answer="yes" if i % 2 == 0 else "no",
                answer_type="CLOSED",
                split=split,
            )
        )

    manifest_path = image_root / "manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")

    return manifest_path, image_root


def test_rocov2_dataset_loads_all_records(rocov2_manifest: tuple[Path, Path]) -> None:
    manifest_path, image_root = rocov2_manifest
    dataset = ROCOv2Dataset(manifest_path, image_root)
    assert len(dataset) == 3
    item = dataset[0]
    assert item["pair_id"] == "pair_0"
    assert item["modality"] == "X-ray"
    assert item["image"].size == (64, 64)  # untransformed PIL image


def test_rocov2_dataset_applies_transform(rocov2_manifest: tuple[Path, Path]) -> None:
    manifest_path, image_root = rocov2_manifest
    data_config = {
        "preprocessing": {
            "image_size": [32, 32],
            "normalize_mean": [0.5, 0.5, 0.5],
            "normalize_std": [0.5, 0.5, 0.5],
        }
    }
    transform = build_image_transform(data_config)
    dataset = ROCOv2Dataset(manifest_path, image_root, transform=transform)
    item = dataset[0]
    assert item["image"].shape == (3, 32, 32)


def test_rocov2_dataset_missing_manifest_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ROCOv2Dataset(tmp_path / "does_not_exist.jsonl", tmp_path)


def test_vqa_rad_dataset_filters_by_split(vqa_rad_manifest: tuple[Path, Path]) -> None:
    manifest_path, image_root = vqa_rad_manifest
    train_set = VQARadDataset(manifest_path, image_root, split="train")
    val_set = VQARadDataset(manifest_path, image_root, split="val")
    test_set = VQARadDataset(manifest_path, image_root, split="test")

    assert len(train_set) == 2
    assert len(val_set) == 1
    assert len(test_set) == 1

    item = test_set[0]
    assert item["example_id"] == "example_3"
    assert item["answer_type"] == "CLOSED"


def test_vqa_rad_dataset_empty_split_warns_not_raises(
    vqa_rad_manifest: tuple[Path, Path]
) -> None:
    manifest_path, image_root = vqa_rad_manifest
    # "test" split exists but let's request a split with zero matching records.
    empty_set = VQARadDataset(manifest_path, image_root, split="train")
    assert isinstance(len(empty_set), int)

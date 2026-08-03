from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset

from src.data.schema import ImageCaptionPair, VQAExample
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {path}. Run scripts/download_data.py first."
        )
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class ROCOv2Dataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        image_root: str | Path,
        transform: Callable | None = None,
    ) -> None:
        self.image_root = Path(image_root)
        self.transform = transform
        raw_records = _read_jsonl(Path(manifest_path))
        self.records: list[ImageCaptionPair] = [ImageCaptionPair(**r) for r in raw_records]
        logger.info("Loaded ROCOv2Dataset: %d pairs from %s", len(self.records), manifest_path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        image_path = self.image_root / record.image_path
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {
            "pair_id": record.pair_id,
            "image": image,
            "caption": record.caption,
            "modality": record.modality,
        }


class VQARadDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        image_root: str | Path,
        split: str = "test",
        transform: Callable | None = None,
    ) -> None:
        self.image_root = Path(image_root)
        self.transform = transform
        raw_records = _read_jsonl(Path(manifest_path))
        all_records = [VQAExample(**r) for r in raw_records]
        self.records = [r for r in all_records if r.split == split]
        if not self.records:
            logger.warning(
                "VQARadDataset: no records found for split='%s' (manifest had %d total)",
                split,
                len(all_records),
            )
        logger.info(
            "Loaded VQARadDataset: %d examples (split=%s) from %s",
            len(self.records),
            split,
            manifest_path,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        image_path = self.image_root / record.image_path
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {
            "example_id": record.example_id,
            "image": image,
            "question": record.question,
            "answer": record.answer,
            "answer_type": record.answer_type,
        }

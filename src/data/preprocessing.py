from __future__ import annotations

import random
from collections import defaultdict
from typing import Callable, Hashable, Sequence, TypeVar

from torchvision import transforms

from src.utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


def build_image_transform(data_config: dict) -> transforms.Compose:
    
    prep = data_config["preprocessing"]
    size = tuple(prep["image_size"])
    mean = prep["normalize_mean"]
    std = prep["normalize_std"]

    return transforms.Compose(
        [
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def truncate_caption(caption: str, max_length: int) -> str:
   
    tokens = caption.split()
    if len(tokens) <= max_length:
        return caption
    return " ".join(tokens[:max_length])


def stratified_split(
    records: Sequence[T],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    stratify_key: Callable[[T], Hashable],
    seed: int = 42,
) -> tuple[list[T], list[T], list[T]]:
    
    ratios_sum = train_ratio + val_ratio + test_ratio
    if abs(ratios_sum - 1.0) > 1e-6:
        raise ValueError(
            f"train/val/test ratios must sum to 1.0, got {ratios_sum} "
            f"({train_ratio=}, {val_ratio=}, {test_ratio=})"
        )

    rng = random.Random(seed)
    strata: dict[Hashable, list[T]] = defaultdict(list)
    for record in records:
        strata[stratify_key(record)].append(record)

    train: list[T] = []
    val: list[T] = []
    test: list[T] = []

    for stratum_label, stratum_records in strata.items():
        shuffled = list(stratum_records)
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_train = round(n * train_ratio)
        n_val = round(n * val_ratio)
        # Remainder goes to test so every record is accounted for exactly once,
        # even when rounding would otherwise drop or duplicate a record.
        n_test = n - n_train - n_val

        train.extend(shuffled[:n_train])
        val.extend(shuffled[n_train : n_train + n_val])
        test.extend(shuffled[n_train + n_val :])

        logger.debug(
            "Stratum '%s': %d total -> train=%d val=%d test=%d",
            stratum_label,
            n,
            n_train,
            n_val,
            n_test,
        )

    logger.info(
        "Stratified split complete: %d train / %d val / %d test (from %d total, %d strata)",
        len(train),
        len(val),
        len(test),
        len(records),
        len(strata),
    )
    return train, val, test

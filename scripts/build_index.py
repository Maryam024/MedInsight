#!/usr/bin/env python

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
from pathlib import Path

from src.data.dataset import ROCOv2Dataset
from src.retrieval.embedder import ImageEmbedder
from src.retrieval.index import RetrievalIndex
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

MANIFEST_PATH = Path("data/processed/rocov2/manifest.jsonl")
IMAGE_ROOT = Path("data/processed/rocov2")
INDEX_DIR = Path("data/processed/rocov2/index")


def build_index(limit: int | None = None, batch_size: int = 32) -> RetrievalIndex:
    config = load_config()
    dataset = ROCOv2Dataset(MANIFEST_PATH, IMAGE_ROOT, transform=None)

    n = len(dataset) if limit is None else min(limit, len(dataset))
    if limit is not None:
        logger.info("Limiting index build to the first %d corpus items (smoke test).", n)

    embedder = ImageEmbedder.from_config(config)

    all_embeddings = []
    all_ids = []
    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        images = [dataset[i]["image"] for i in range(batch_start, batch_end)]
        ids = [dataset[i]["pair_id"] for i in range(batch_start, batch_end)]

        embeddings = embedder.embed_images(images)
        all_embeddings.append(embeddings)
        all_ids.extend(ids)

        logger.info("Embedded %d/%d corpus items", batch_end, n)

    import numpy as np

    embeddings_matrix = np.concatenate(all_embeddings, axis=0)
    index_type = config.model["retriever"]["index_type"]
    index = RetrievalIndex.build(embeddings_matrix, all_ids, index_type=index_type)
    index.save(INDEX_DIR)

    logger.info("Index build complete: %d vectors saved to %s", index.index.ntotal, INDEX_DIR)
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Embed only the first N corpus items (smoke test).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Images per embedding forward pass (default: 32).",
    )
    args = parser.parse_args()
    build_index(limit=args.limit, batch_size=args.batch_size)


if __name__ == "__main__":
    main()

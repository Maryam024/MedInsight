from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import numpy as np

from src.retrieval.embedder import normalize_embeddings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class _FaissLikeIndex(Protocol):
    ntotal: int

    def add(self, vectors: np.ndarray) -> None: ...

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]: ...


class RetrievalIndex:
    def __init__(self, index: _FaissLikeIndex, id_map: list[str]) -> None:
        if index.ntotal != len(id_map):
            raise ValueError(
                f"Index/id_map size mismatch: index has {index.ntotal} vectors "
                f"but id_map has {len(id_map)} entries."
            )
        self.index = index
        self.id_map = id_map

    @classmethod
    def build(
        cls,
        embeddings: np.ndarray,
        ids: list[str],
        index_type: str = "flat_ip",
    ) -> "RetrievalIndex":
        if len(embeddings) != len(ids):
            raise ValueError(
                f"embeddings and ids must have the same length, "
                f"got {len(embeddings)} and {len(ids)}"
            )
        if index_type != "flat_ip":
            raise NotImplementedError(
                f"index_type={index_type!r} is not implemented yet; only 'flat_ip' is supported."
            )

        import faiss

        embeddings = normalize_embeddings(np.asarray(embeddings, dtype=np.float32))
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

        logger.info(
            "Built flat_ip FAISS index: %d vectors, dimension=%d", index.ntotal, dimension
        )
        return cls(index=index, id_map=list(ids))

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")

        query = np.asarray(query_embedding, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        query = normalize_embeddings(query)

        effective_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query, effective_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # faiss pads with -1 if fewer than top_k found
                continue
            results.append((self.id_map[idx], float(score)))
        return results

    def save(self, dir_path: str | Path) -> None:
        import faiss

        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(dir_path / "index.faiss"))
        with open(dir_path / "id_map.json", "w", encoding="utf-8") as f:
            json.dump(self.id_map, f)
        logger.info("Saved retrieval index to %s (%d vectors)", dir_path, self.index.ntotal)

    @classmethod
    def load(cls, dir_path: str | Path) -> "RetrievalIndex":
        import faiss

        dir_path = Path(dir_path)
        index = faiss.read_index(str(dir_path / "index.faiss"))
        with open(dir_path / "id_map.json", "r", encoding="utf-8") as f:
            id_map = json.load(f)
        logger.info("Loaded retrieval index from %s (%d vectors)", dir_path, index.ntotal)
        return cls(index=index, id_map=id_map)

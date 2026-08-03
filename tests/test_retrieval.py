from __future__ import annotations

import numpy as np
import pytest

from src.retrieval.embedder import normalize_embeddings
from src.retrieval.index import RetrievalIndex


class _FakeFaissIndex:
    def __init__(self) -> None:
        self._vectors: np.ndarray | None = None

    @property
    def ntotal(self) -> int:
        return 0 if self._vectors is None else len(self._vectors)

    def add(self, vectors: np.ndarray) -> None:
        self._vectors = vectors if self._vectors is None else np.vstack([self._vectors, vectors])

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        assert self._vectors is not None

        scores_matrix = queries @ self._vectors.T
        indices = np.argsort(-scores_matrix, axis=1)[:, :k]
        scores = np.take_along_axis(scores_matrix, indices, axis=1)
        return scores, indices


def test_normalize_embeddings_unit_length() -> None:
    vectors = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
    normalized = normalize_embeddings(vectors)
    norms = np.linalg.norm(normalized, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-6)


def test_normalize_embeddings_handles_zero_vector() -> None:
    vectors = np.array([[0.0, 0.0]], dtype=np.float32)
    normalized = normalize_embeddings(vectors)

    assert not np.isnan(normalized).any()


def _build_fake_index() -> RetrievalIndex:
    vectors = normalize_embeddings(
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.9, 0.1, 0.0],
            ],
            dtype=np.float32,
        )
    )
    fake_index = _FakeFaissIndex()
    fake_index.add(vectors)
    return RetrievalIndex(index=fake_index, id_map=["pair_a", "pair_b", "pair_c"])


def test_search_returns_most_similar_first() -> None:
    index = _build_fake_index()
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    results = index.search(query, top_k=2)

    assert results[0][0] == "pair_a"
    assert results[1][0] == "pair_c"
    assert results[0][1] > results[1][1]


def test_search_clamps_top_k_to_index_size() -> None:
    index = _build_fake_index()
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    results = index.search(query, top_k=100)
    assert len(results) == 3


def test_search_rejects_non_positive_top_k() -> None:
    index = _build_fake_index()
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    with pytest.raises(ValueError):
        index.search(query, top_k=0)


def test_init_rejects_id_map_size_mismatch() -> None:
    fake_index = _FakeFaissIndex()
    fake_index.add(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    with pytest.raises(ValueError):
        RetrievalIndex(index=fake_index, id_map=["only_one_id"])

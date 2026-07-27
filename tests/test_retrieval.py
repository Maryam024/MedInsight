"""
Unit tests for src.retrieval.embedder (normalize_embeddings) and
src.retrieval.index (RetrievalIndex).

`RetrievalIndex.build`/`save`/`load` call the real `faiss` package and are
not exercised here (no network/package access in this sandbox and no need
to duplicate FAISS's own test suite). Instead, `RetrievalIndex.__init__`'s
dependency-injection design (see its docstring) is used directly: these
tests inject `_FakeFaissIndex`, a brute-force numpy implementation of the
same `add`/`search` contract, to verify MedInsight's own logic — id mapping,
score ordering, top_k clamping, error handling — independent of FAISS itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.retrieval.embedder import normalize_embeddings
from src.retrieval.index import RetrievalIndex


class _FakeFaissIndex:
    """Brute-force inner-product search, matching `faiss.IndexFlatIP`'s API."""

    def __init__(self) -> None:
        self._vectors: np.ndarray | None = None

    @property
    def ntotal(self) -> int:
        return 0 if self._vectors is None else len(self._vectors)

    def add(self, vectors: np.ndarray) -> None:
        self._vectors = vectors if self._vectors is None else np.vstack([self._vectors, vectors])

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        assert self._vectors is not None
        # Inner product of each query against every indexed vector.
        scores_matrix = queries @ self._vectors.T  # (n_queries, n_indexed)
        indices = np.argsort(-scores_matrix, axis=1)[:, :k]
        scores = np.take_along_axis(scores_matrix, indices, axis=1)
        return scores, indices


# ---------------------------------------------------------------------------
# normalize_embeddings
# ---------------------------------------------------------------------------


def test_normalize_embeddings_unit_length() -> None:
    vectors = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
    normalized = normalize_embeddings(vectors)
    norms = np.linalg.norm(normalized, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-6)


def test_normalize_embeddings_handles_zero_vector() -> None:
    vectors = np.array([[0.0, 0.0]], dtype=np.float32)
    normalized = normalize_embeddings(vectors)
    # Should not divide by zero / produce NaN.
    assert not np.isnan(normalized).any()


# ---------------------------------------------------------------------------
# RetrievalIndex (with injected fake backend)
# ---------------------------------------------------------------------------


def _build_fake_index() -> RetrievalIndex:
    """3 orthogonal-ish unit vectors so nearest-neighbor results are unambiguous."""
    vectors = normalize_embeddings(
        np.array(
            [
                [1.0, 0.0, 0.0],   # pair_a
                [0.0, 1.0, 0.0],   # pair_b
                [0.9, 0.1, 0.0],   # pair_c: close to pair_a
            ],
            dtype=np.float32,
        )
    )
    fake_index = _FakeFaissIndex()
    fake_index.add(vectors)
    return RetrievalIndex(index=fake_index, id_map=["pair_a", "pair_b", "pair_c"])


def test_search_returns_most_similar_first() -> None:
    index = _build_fake_index()
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # identical to pair_a
    results = index.search(query, top_k=2)

    assert results[0][0] == "pair_a"
    assert results[1][0] == "pair_c"  # pair_c is closer to pair_a than pair_b is
    assert results[0][1] > results[1][1]  # scores are descending


def test_search_clamps_top_k_to_index_size() -> None:
    index = _build_fake_index()
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    results = index.search(query, top_k=100)  # index only has 3 vectors
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

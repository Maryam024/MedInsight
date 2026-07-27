"""
Tests for the MedInsight FastAPI app.

Same dependency-injection testing pattern as the rest of the test suite:
`deployment/api/dependencies.py`'s `get_baseline_vlm`/`get_rag_vlm` are
overridden via FastAPI's `app.dependency_overrides` with fakes, so these
tests never load a real checkpoint or FAISS index. `TestClient` drives the
app through actual HTTP request/response handling (routing, file upload
parsing, status codes, response validation) rather than calling route
functions directly, so this is closer to an integration test of the API
layer than a unit test of any single function.
"""

from __future__ import annotations

import io

from fastapi.testclient import TestClient
from PIL import Image

from deployment.api.dependencies import get_baseline_vlm, get_rag_vlm
from deployment.api.main import app


class _FakeBaselineVLM:
    def generate(self, image, question: str) -> str:
        return f"fake baseline answer to: {question}"


class _FakeRAGVLM:
    def generate(self, image, question: str) -> dict:
        return {
            "answer": f"fake rag answer to: {question}",
            "prompt": f"Reference findings:\n- Normal study.\n\nQuestion: {question}",
            "retrieved_evidence": [
                {"pair_id": "pair_1", "caption": "Normal study.", "similarity_score": 0.9},
            ],
        }


class _FakeRAGVLMMissingIndex:
    def generate(self, image, question: str) -> dict:
        raise FileNotFoundError("index.faiss not found")


def _make_test_image_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color=(100, 100, 100)).save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


def _client_with_fakes(rag_vlm=None) -> TestClient:
    app.dependency_overrides[get_baseline_vlm] = lambda: _FakeBaselineVLM()
    app.dependency_overrides[get_rag_vlm] = lambda: (rag_vlm or _FakeRAGVLM())
    return TestClient(app)


def test_health_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_baseline_returns_answer_and_empty_evidence() -> None:
    client = _client_with_fakes()
    image_bytes = _make_test_image_bytes()

    response = client.post(
        "/predict/baseline",
        params={"question": "Is there a fracture?"},
        files={"image": ("test.png", image_bytes, "image/png")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["question"] == "Is there a fracture?"
    assert "fake baseline answer" in body["answer"]
    assert body["retrieved_evidence"] == []
    app.dependency_overrides.clear()


def test_predict_rag_returns_answer_and_evidence() -> None:
    client = _client_with_fakes()
    image_bytes = _make_test_image_bytes()

    response = client.post(
        "/predict/rag",
        params={"question": "Is there consolidation?"},
        files={"image": ("test.png", image_bytes, "image/png")},
    )

    assert response.status_code == 200
    body = response.json()
    assert "fake rag answer" in body["answer"]
    assert len(body["retrieved_evidence"]) == 1
    assert body["retrieved_evidence"][0]["pair_id"] == "pair_1"
    app.dependency_overrides.clear()


def test_predict_rag_missing_index_returns_503() -> None:
    client = _client_with_fakes(rag_vlm=_FakeRAGVLMMissingIndex())
    image_bytes = _make_test_image_bytes()

    response = client.post(
        "/predict/rag",
        params={"question": "Is there a mass?"},
        files={"image": ("test.png", image_bytes, "image/png")},
    )

    assert response.status_code == 503
    assert "build_index.py" in response.json()["detail"]
    app.dependency_overrides.clear()


def test_predict_baseline_rejects_invalid_image() -> None:
    client = _client_with_fakes()

    response = client.post(
        "/predict/baseline",
        params={"question": "Is there a fracture?"},
        files={"image": ("not_an_image.txt", b"this is not image data", "text/plain")},
    )

    assert response.status_code == 400
    assert "Could not decode" in response.json()["detail"]
    app.dependency_overrides.clear()


def test_predict_baseline_missing_question_returns_422() -> None:
    client = _client_with_fakes()
    image_bytes = _make_test_image_bytes()

    # No "question" query param supplied.
    response = client.post(
        "/predict/baseline",
        files={"image": ("test.png", image_bytes, "image/png")},
    )

    assert response.status_code == 422  # FastAPI's validation error for a missing required param
    app.dependency_overrides.clear()

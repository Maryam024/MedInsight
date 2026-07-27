"""
Unit tests for src.models.rag_vlm.

Follows the same dependency-injection testing pattern as
tests/test_models.py and tests/test_retrieval.py: `RAGVLM.__init__` takes
already-constructed `BaselineVLM`/`ImageEmbedder`/`RetrievalIndex` instances,
so these tests inject minimal fakes for all three rather than downloading
CLIP/BLIP-2 weights or building a real FAISS index. `RAGVLM.from_config`
(which does real loading) is not exercised here — see the module docstring
in src/models/rag_vlm.py.
"""

from __future__ import annotations

from PIL import Image

from src.models.rag_vlm import RAGVLM, build_prompt


class _FakeEmbedder:
    """Returns a fixed vector regardless of input image."""

    def embed_image(self, image):
        return [1.0, 0.0, 0.0]


class _FakeIndex:
    """Returns a canned, ordered list of (pair_id, score) results."""

    def __init__(self, results: list[tuple[str, float]]) -> None:
        self.results = results
        self.last_top_k: int | None = None

    def search(self, query_embedding, top_k: int):
        self.last_top_k = top_k
        return self.results[:top_k]


class _FakeVLM:
    """Records the prompt it was called with and returns a canned answer."""

    def __init__(self, canned_answer: str = "no acute abnormality") -> None:
        self.canned_answer = canned_answer
        self.last_prompt: str | None = None

    def generate_from_prompt(self, image, prompt: str) -> str:
        self.last_prompt = prompt
        return self.canned_answer


def _build_rag_vlm(
    vlm=None,
    index_results: list[tuple[str, float]] | None = None,
    caption_lookup: dict[str, str] | None = None,
    top_k: int = 3,
) -> tuple[RAGVLM, _FakeVLM, _FakeIndex]:
    vlm = vlm or _FakeVLM()
    index_results = index_results if index_results is not None else [
        ("pair_1", 0.95),
        ("pair_2", 0.80),
        ("pair_3", 0.60),
    ]
    caption_lookup = caption_lookup if caption_lookup is not None else {
        "pair_1": "Left lower lobe consolidation consistent with pneumonia.",
        "pair_2": "Normal chest radiograph.",
        "pair_3": "Small right pleural effusion.",
    }
    fake_index = _FakeIndex(index_results)
    rag = RAGVLM(
        vlm=vlm,
        embedder=_FakeEmbedder(),
        index=fake_index,
        caption_lookup=caption_lookup,
        top_k=top_k,
        evidence_instruction="Reference findings from similar images:",
        max_caption_words_in_prompt=60,
    )
    return rag, vlm, fake_index


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------


def test_build_prompt_with_no_evidence_returns_bare_question() -> None:
    prompt = build_prompt(
        "Is there a fracture?",
        captions=[],
        evidence_instruction="Reference findings:",
        max_caption_words=60,
    )
    assert prompt == "Is there a fracture?"


def test_build_prompt_includes_instruction_and_captions() -> None:
    prompt = build_prompt(
        "Is there a fracture?",
        captions=["No fracture seen.", "Normal bone density."],
        evidence_instruction="Reference findings:",
        max_caption_words=60,
    )
    assert "Reference findings:" in prompt
    assert "No fracture seen." in prompt
    assert "Normal bone density." in prompt
    assert "Question: Is there a fracture?" in prompt


def test_build_prompt_truncates_long_captions() -> None:
    long_caption = " ".join(["word"] * 100)
    prompt = build_prompt(
        "What is shown?",
        captions=[long_caption],
        evidence_instruction="Reference findings:",
        max_caption_words=5,
    )
    # Only 5 words of the 100-word caption should appear.
    evidence_line = [line for line in prompt.split("\n") if line.startswith("- ")][0]
    assert len(evidence_line[2:].split()) == 5


# ---------------------------------------------------------------------------
# RAGVLM.retrieve
# ---------------------------------------------------------------------------


def test_retrieve_maps_ids_to_captions_in_order() -> None:
    rag, _, _ = _build_rag_vlm()
    image = Image.new("RGB", (32, 32))
    evidence = rag.retrieve(image)

    assert [e["pair_id"] for e in evidence] == ["pair_1", "pair_2", "pair_3"]
    assert evidence[0]["caption"] == "Left lower lobe consolidation consistent with pneumonia."
    assert evidence[0]["similarity_score"] == 0.95


def test_retrieve_skips_ids_missing_from_caption_lookup() -> None:
    rag, _, _ = _build_rag_vlm(
        index_results=[("pair_1", 0.9), ("unknown_pair", 0.8)],
        caption_lookup={"pair_1": "Normal study."},
    )
    image = Image.new("RGB", (32, 32))
    evidence = rag.retrieve(image)
    assert len(evidence) == 1
    assert evidence[0]["pair_id"] == "pair_1"


def test_retrieve_passes_configured_top_k_to_index() -> None:
    rag, _, fake_index = _build_rag_vlm(top_k=2)
    image = Image.new("RGB", (32, 32))
    rag.retrieve(image)
    assert fake_index.last_top_k == 2


# ---------------------------------------------------------------------------
# RAGVLM.generate (end to end, with fakes)
# ---------------------------------------------------------------------------


def test_generate_returns_answer_prompt_and_evidence() -> None:
    rag, fake_vlm, _ = _build_rag_vlm()
    image = Image.new("RGB", (32, 32))
    result = rag.generate(image, "Is there consolidation?")

    assert result["answer"] == "no acute abnormality"
    assert "Question: Is there consolidation?" in result["prompt"]
    assert len(result["retrieved_evidence"]) == 3
    # The prompt actually passed to the underlying VLM matches what's returned.
    assert fake_vlm.last_prompt == result["prompt"]


def test_generate_with_empty_index_falls_back_to_bare_question() -> None:
    rag, fake_vlm, _ = _build_rag_vlm(index_results=[])
    image = Image.new("RGB", (32, 32))
    result = rag.generate(image, "Is there a mass?")

    assert result["prompt"] == "Is there a mass?"
    assert result["retrieved_evidence"] == []
    assert fake_vlm.last_prompt == "Is there a mass?"

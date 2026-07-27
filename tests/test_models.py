"""
Unit tests for src.models.baseline_vlm.

These tests never load real model weights. `BaselineVLM.__init__` accepts
an already-constructed processor/model (see the class docstring for why),
so tests inject minimal fakes that implement the same call signature as a
Hugging Face processor/model. `BaselineVLM.from_config` (the code path that
*does* hit Hugging Face) is intentionally not exercised here — that's a
manual integration check, not part of the offline unit suite.
"""

from __future__ import annotations

from PIL import Image

from src.models.baseline_vlm import BaselineVLM


class _FakeProcessor:
    """Minimal stand-in for a Hugging Face processor."""

    def __init__(self) -> None:
        self.last_call_kwargs: dict = {}

    def __call__(self, images, text, return_tensors: str) -> dict:
        self.last_call_kwargs = {"images": images, "text": text, "return_tensors": return_tensors}
        # A real processor returns a dict of tensors; the fake model below
        # doesn't inspect these, so a placeholder dict is sufficient.
        return {"pixel_values": None, "input_ids": None}

    def batch_decode(self, token_ids, skip_special_tokens: bool) -> list[str]:
        # token_ids is whatever the fake model's generate() returned.
        return token_ids


class _FakeModel:
    """Minimal stand-in for a Hugging Face generative model."""

    def __init__(self, canned_response: str = "no acute abnormality") -> None:
        self.canned_response = canned_response
        self.generate_call_count = 0

    def to(self, device: str) -> "_FakeModel":
        self.device = device
        return self

    def generate(self, **kwargs) -> list[str]:
        self.generate_call_count += 1
        # Real models return token id tensors; batch_decode would turn those
        # into strings. The fake short-circuits by returning strings directly
        # and having _FakeProcessor.batch_decode pass them through unchanged.
        return [self.canned_response]


def _build_baseline(canned_response: str = "no acute abnormality") -> BaselineVLM:
    processor = _FakeProcessor()
    model = _FakeModel(canned_response=canned_response)
    return BaselineVLM(
        processor=processor,
        model=model,
        generation_kwargs={"max_new_tokens": 128, "do_sample": False},
        device="cpu",
    )


def test_generate_returns_stripped_string() -> None:
    baseline = _build_baseline(canned_response="  no acute abnormality  ")
    image = Image.new("RGB", (64, 64))
    answer = baseline.generate(image, "Is there an abnormality?")
    assert answer == "no acute abnormality"


def test_generate_passes_generation_kwargs_to_model() -> None:
    processor = _FakeProcessor()
    model = _FakeModel()
    baseline = BaselineVLM(
        processor=processor,
        model=model,
        generation_kwargs={"max_new_tokens": 64, "temperature": 0.2, "do_sample": False},
        device="cpu",
    )
    image = Image.new("RGB", (64, 64))
    baseline.generate(image, "What is shown?")
    assert model.generate_call_count == 1


def test_generate_batch_matches_length_and_order() -> None:
    baseline = _build_baseline(canned_response="fracture")
    images = [Image.new("RGB", (32, 32)) for _ in range(3)]
    questions = ["Q1?", "Q2?", "Q3?"]
    answers = baseline.generate_batch(images, questions)
    assert answers == ["fracture", "fracture", "fracture"]


def test_generate_batch_rejects_mismatched_lengths() -> None:
    baseline = _build_baseline()
    images = [Image.new("RGB", (32, 32))]
    questions = ["Q1?", "Q2?"]
    try:
        baseline.generate_batch(images, questions)
        assert False, "expected ValueError for mismatched lengths"
    except ValueError:
        pass

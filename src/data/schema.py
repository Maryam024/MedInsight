from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Modality = Literal["CT", "MRI", "X-ray", "Ultrasound", "Other"]
AnswerType = Literal["CLOSED", "OPEN"]


class ImageCaptionPair(BaseModel):

    pair_id: str
    image_path: str
    caption: str
    modality: Modality = "Other"
    source_id: str | None = Field(
        default=None,
        description="Original ROCOv2/PMC identifier, kept for citation and audit.",
    )

    @field_validator("caption")
    @classmethod
    def caption_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("caption must not be empty")
        return v.strip()

    def image_exists(self, base_dir: str | Path = ".") -> bool:
        """Check the referenced image file actually exists on disk."""
        return (Path(base_dir) / self.image_path).is_file()


class VQAExample(BaseModel):

    example_id: str
    image_path: str
    question: str
    answer: str
    answer_type: AnswerType = "OPEN"
    split: Literal["train", "val", "test"] = "test"

    @field_validator("question", "answer")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("question/answer must not be empty")
        return v.strip()

    def image_exists(self, base_dir: str | Path = ".") -> bool:
        return (Path(base_dir) / self.image_path).is_file()

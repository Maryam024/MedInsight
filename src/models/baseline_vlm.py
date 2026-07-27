from __future__ import annotations

from typing import Any, Protocol

from PIL import Image

from src.utils.config_loader import MedInsightConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class _Processor(Protocol):

    def __call__(self, images: Any, text: Any, return_tensors: str) -> dict: ...

    def batch_decode(self, token_ids: Any, skip_special_tokens: bool) -> list[str]: ...


class _GenerativeModel(Protocol):
    """Structural type for the Hugging Face model this class expects."""

    def generate(self, **kwargs: Any) -> Any: ...

    def to(self, device: str) -> "_GenerativeModel": ...


class BaselineVLM:

    def __init__(
        self,
        processor: _Processor,
        model: _GenerativeModel,
        generation_kwargs: dict[str, Any],
        device: str = "cpu",
    ) -> None:
        self.processor = processor
        self.model = model.to(device)
        self.generation_kwargs = generation_kwargs
        self.device = device
        logger.info(
            "BaselineVLM ready on device=%s with generation_kwargs=%s",
            device,
            generation_kwargs,
        )

    @classmethod
    def from_config(cls, config: MedInsightConfig) -> "BaselineVLM":
       
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        vlm_cfg = config.model["vision_language_model"]
        checkpoint = vlm_cfg["baseline_name"]

        logger.info("Loading baseline VLM checkpoint: %s", checkpoint)
        processor = AutoProcessor.from_pretrained(checkpoint)
        model = AutoModelForVision2Seq.from_pretrained(checkpoint, torch_dtype=torch.float16)

        if vlm_cfg.get("freeze_vision_encoder", True) and hasattr(model, "vision_model"):
            for param in model.vision_model.parameters():
                param.requires_grad = False

        return cls(
            processor=processor,
            model=model,
            generation_kwargs=dict(config.model["generation"]),
            device=config.device.resolved_device,
        )

    def generate(self, image: Image.Image, question: str) -> str:
        
        return self.generate_from_prompt(image, f"Question: {question} Answer:")

    def generate_from_prompt(self, image: Image.Image, prompt: str) -> str:
        
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        output_ids = self.model.generate(**inputs, **self.generation_kwargs)
        decoded = self.processor.batch_decode(output_ids, skip_special_tokens=True)
        answer = decoded[0].strip()
        return answer

    def generate_batch(self, images: list[Image.Image], questions: list[str]) -> list[str]:
        
        if len(images) != len(questions):
            raise ValueError(
                f"images and questions must be the same length, "
                f"got {len(images)} and {len(questions)}"
            )
        return [self.generate(img, q) for img, q in zip(images, questions)]

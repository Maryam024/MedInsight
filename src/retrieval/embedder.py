from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from PIL import Image

from src.utils.config_loader import MedInsightConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class _ImageProcessor(Protocol):
    def __call__(self, images: Any, return_tensors: str) -> dict: ...


class _ImageEncoderModel(Protocol):
    def get_image_features(self, **kwargs: Any) -> Any: ...

    def to(self, device: str) -> "_ImageEncoderModel": ...


def normalize_embeddings(vectors: np.ndarray) -> np.ndarray:
    
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)  # avoid divide-by-zero for a zero vector
    return vectors / norms


class ImageEmbedder:

    def __init__(
        self,
        processor: _ImageProcessor,
        model: _ImageEncoderModel,
        device: str = "cpu",
    ) -> None:
        self.processor = processor
        self.model = model.to(device)
        self.device = device

    @classmethod
    def from_config(cls, config: MedInsightConfig) -> "ImageEmbedder":
       
        from transformers import CLIPModel, CLIPProcessor

        checkpoint = config.model["retriever"]["image_encoder"]
        logger.info("Loading image encoder: %s", checkpoint)
        processor = CLIPProcessor.from_pretrained(checkpoint)
        model = CLIPModel.from_pretrained(checkpoint)
        return cls(processor=processor, model=model, device=config.device.resolved_device)

    def embed_image(self, image: Image.Image) -> np.ndarray:
        """Embed a single PIL image into a normalized 1-D vector."""
        return self.embed_images([image])[0]

    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
    
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        image_features = self.model.get_image_features(**inputs)
        vectors = np.asarray(image_features.detach().cpu().numpy(), dtype=np.float32)
        return normalize_embeddings(vectors)

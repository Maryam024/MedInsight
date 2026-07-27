"""FastAPI service exposing the baseline and RAG MedInsight models."""

from __future__ import annotations

import io

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError

from deployment.api.dependencies import get_baseline_vlm, get_rag_vlm
from deployment.api.schemas import HealthResponse, PredictionResponse
from src.models.baseline_vlm import BaselineVLM
from src.models.rag_vlm import RAGVLM
from src.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="MedInsight API",
    description=(
        "Retrieval-augmented and baseline vision-language model endpoints "
        "for medical image question answering. Research prototype - see "
        "README.md's Ethical Considerations note. Not a diagnostic tool."
    ),
    version="0.1.0",
)


def _load_uploaded_image(file: UploadFile) -> Image.Image:
    
    try:
        contents = file.file.read()
        return Image.open(io.BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Could not decode '{file.filename}' as an image. "
                "Supported formats: JPEG, PNG."
            ),
        ) from exc


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    
    return HealthResponse(status="ok")


@app.post("/predict/baseline", response_model=PredictionResponse, tags=["prediction"])
def predict_baseline(
    question: str,
    image: UploadFile = File(...),
    model: BaselineVLM = Depends(get_baseline_vlm),
) -> PredictionResponse:
    """Answer `question` about `image` with no retrieved evidence (control condition)."""
    pil_image = _load_uploaded_image(image)
    answer = model.generate(pil_image, question)
    return PredictionResponse(answer=answer, question=question, retrieved_evidence=[])


@app.post("/predict/rag", response_model=PredictionResponse, tags=["prediction"])
def predict_rag(
    question: str,
    image: UploadFile = File(...),
    model: RAGVLM = Depends(get_rag_vlm),
) -> PredictionResponse:
    
    pil_image = _load_uploaded_image(image)
    try:
        result = model.generate(pil_image, question)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Retrieval index not found. Run `python scripts/build_index.py` "
                "before using this endpoint."
            ),
        ) from exc
    return PredictionResponse(
        answer=result["answer"],
        question=question,
        retrieved_evidence=result["retrieved_evidence"],
    )

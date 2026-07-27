"""Lazy-loaded singleton models to avoid reloading checkpoints on every request."""

from __future__ import annotations

from functools import lru_cache

from src.models.baseline_vlm import BaselineVLM
from src.models.rag_vlm import RAGVLM
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_baseline_vlm() -> BaselineVLM:
    
    """Return the shared `BaselineVLM` instance, loading it on first use."""

    logger.info("Loading BaselineVLM for the API (first request only)...")
    config = load_config()
    return BaselineVLM.from_config(config)


@lru_cache(maxsize=1)
def get_rag_vlm() -> RAGVLM:
    
    """Return the shared `RAGVLM` instance, loading it on first use."""

    logger.info("Loading RAGVLM for the API (first request only)...")
    config = load_config()
    return RAGVLM.from_config(config)

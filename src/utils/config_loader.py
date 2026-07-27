"""
Configuration loading and validation for MedInsight.

Why this file exists
---------------------
We keep settings in YAML files (configs/*.yaml) instead of hardcoding them
in Python, so that:
  1. Non-code changes (batch size, dataset path, top_k for retrieval) don't
     require touching source code — important for running many experiments.
  2. We get one auditable, versioned record of exactly what settings produced
     a given result (a core requirement of reproducible research).
  3. We can validate settings *before* a multi-hour training run starts,
     rather than crashing halfway through with a typo in a YAML key.

We use Pydantic for validation because plain `dict`s from `yaml.safe_load()`
give no guarantees about types or required fields — a typo like
`batch_size: "8"` (string instead of int) would silently propagate bugs.
Pydantic catches this immediately, at load time, with a clear error message.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml
from pydantic import BaseModel, Field, field_validator

from src.utils.logger import get_logger

logger = get_logger(__name__)


class PathsConfig(BaseModel):
    """Filesystem locations used throughout the project."""

    data_raw: str
    data_processed: str
    data_external: str
    experiment_logs: str
    checkpoints: str


class LoggingConfig(BaseModel):
    """Controls for the logging system (see src/utils/logger.py)."""

    level: str = "INFO"
    log_to_file: bool = True
    log_dir: str = "experiments/logs"
    log_filename: str = "medinsight.log"

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f"logging.level must be one of {allowed}, got '{v}'")
        return v.upper()


class DeviceConfig(BaseModel):
    """Compute device preferences, resolved to an actual device at load time."""

    use_gpu_if_available: bool = True
    resolved_device: str = Field(default="cpu", exclude=False)


class WandbConfig(BaseModel):
    """Weights & Biases experiment tracking settings."""

    enabled: bool = False
    project_name: str = "medinsight"
    entity: str | None = None


class ProjectConfig(BaseModel):
    """Top-level project metadata."""

    name: str
    version: str
    seed: int
    description: str


class MedInsightConfig(BaseModel):
    """
    Root configuration object combining all sub-configs.

    This is the single object the rest of the codebase imports and passes
    around — every module that needs a path, a hyperparameter, or a logging
    setting reads it from here rather than re-parsing YAML itself.
    """

    project: ProjectConfig
    paths: PathsConfig
    logging: LoggingConfig
    device: DeviceConfig
    wandb: WandbConfig
    data: dict[str, Any] = Field(default_factory=dict)
    model: dict[str, Any] = Field(default_factory=dict)


def _load_yaml(path: Path) -> dict[str, Any]:
    """Read a single YAML file into a dict, raising a clear error if missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}. "
            f"Expected it relative to the project root."
        )
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(
    config_dir: str = "configs",
    config_filename: str = "config.yaml",
    data_config_filename: str = "data_config.yaml",
    model_config_filename: str = "model_config.yaml",
) -> MedInsightConfig:
    """
    Load, merge, and validate all MedInsight configuration files.

    Parameters
    ----------
    config_dir : str
        Directory containing the YAML config files (relative to project root).
    config_filename, data_config_filename, model_config_filename : str
        Filenames of the three config files described in Milestone 1.

    Returns
    -------
    MedInsightConfig
        A validated, typed configuration object. Accessing e.g.
        `config.model.get("retriever")` or `config.paths.data_raw` is now
        guaranteed to exist and be well-formed.

    Raises
    ------
    FileNotFoundError
        If any of the three expected YAML files is missing.
    pydantic.ValidationError
        If a required field is missing or has the wrong type.
    """
    base_dir = Path(config_dir)

    main_cfg = _load_yaml(base_dir / config_filename)
    data_cfg = _load_yaml(base_dir / data_config_filename)
    model_cfg = _load_yaml(base_dir / model_config_filename)

    merged = {
        **main_cfg,
        "data": data_cfg,
        "model": model_cfg,
    }

    config = MedInsightConfig(**merged)

    # Resolve the actual compute device once, at load time, rather than
    # scattering `torch.cuda.is_available()` checks throughout the codebase.
    if config.device.use_gpu_if_available and torch.cuda.is_available():
        config.device.resolved_device = "cuda"
    else:
        if config.device.use_gpu_if_available:
            logger.warning(
                "GPU requested (use_gpu_if_available=True) but no CUDA device "
                "was detected. Falling back to CPU. Training will be slow."
            )
        config.device.resolved_device = "cpu"

    logger.info(
        "Loaded config for project '%s' v%s | device=%s | seed=%d",
        config.project.name,
        config.project.version,
        config.device.resolved_device,
        config.project.seed,
    )

    return config

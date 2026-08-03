from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml
from pydantic import BaseModel, Field, field_validator

from src.utils.logger import get_logger

logger = get_logger(__name__)


class PathsConfig(BaseModel):
    data_raw: str
    data_processed: str
    data_external: str
    experiment_logs: str
    checkpoints: str


class LoggingConfig(BaseModel):
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
    use_gpu_if_available: bool = True
    resolved_device: str = Field(default="cpu", exclude=False)


class WandbConfig(BaseModel):
    enabled: bool = False
    project_name: str = "medinsight"
    entity: str | None = None


class ProjectConfig(BaseModel):
    name: str
    version: str
    seed: int
    description: str


class MedInsightConfig(BaseModel):
    project: ProjectConfig
    paths: PathsConfig
    logging: LoggingConfig
    device: DeviceConfig
    wandb: WandbConfig
    data: dict[str, Any] = Field(default_factory=dict)
    model: dict[str, Any] = Field(default_factory=dict)


def _load_yaml(path: Path) -> dict[str, Any]:
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

    if config.device.use_gpu_if_available and torch.cuda.is_available():  # else cpu
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

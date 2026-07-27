"""Unit tests for src.utils.config_loader."""

import pytest

from src.utils.config_loader import MedInsightConfig, load_config


def test_load_config_returns_valid_object() -> None:
    """The three real YAML files in configs/ should load and validate cleanly."""
    config = load_config()
    assert isinstance(config, MedInsightConfig)
    assert config.project.name == "MedInsight"
    assert config.device.resolved_device in {"cpu", "cuda"}


def test_missing_config_file_raises() -> None:
    """Pointing at a nonexistent directory should fail loudly, not silently."""
    with pytest.raises(FileNotFoundError):
        load_config(config_dir="nonexistent_dir")


def test_data_and_model_subconfigs_are_populated() -> None:
    """The merged config should expose the data/model YAML content as dicts."""
    config = load_config()
    assert "retrieval_corpus" in config.data
    assert "vision_language_model" in config.model

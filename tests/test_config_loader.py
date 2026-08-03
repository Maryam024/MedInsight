import pytest

from src.utils.config_loader import MedInsightConfig, load_config


def test_load_config_returns_valid_object() -> None:
    config = load_config()
    assert isinstance(config, MedInsightConfig)
    assert config.project.name == "MedInsight"
    assert config.device.resolved_device in {"cpu", "cuda"}


def test_missing_config_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_config(config_dir="nonexistent_dir")


def test_data_and_model_subconfigs_are_populated() -> None:
    config = load_config()
    assert "retrieval_corpus" in config.data
    assert "vision_language_model" in config.model

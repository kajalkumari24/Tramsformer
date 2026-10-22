import pytest

from model.configuration import ModelConfig, TrainingConfig, load_model_config, load_training_config


def test_head_dim():
    assert ModelConfig(embedding_dim=64, num_heads=2).head_dim == 32


def test_embedding_must_divide_by_heads():
    with pytest.raises(ValueError):
        ModelConfig(embedding_dim=65, num_heads=2)


def test_with_vocab_size_returns_copy():
    base = ModelConfig()
    filled = base.with_vocab_size(65)
    assert base.vocab_size is None and filled.vocab_size == 65


def test_learning_rate_string_is_converted():
    assert TrainingConfig(learning_rate="3e-4").learning_rate == pytest.approx(0.0003)


def test_project_yaml_files_load():
    assert load_model_config().num_heads >= 1
    assert load_training_config().batch_size >= 1

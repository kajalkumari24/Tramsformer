import pytest
import torch

from model.configuration import ModelConfig
from model.embeddings import InputEmbedding
from model.position import PositionalEmbedding


def make_config(**overrides):
    values = dict(vocab_size=20, context_length=16, embedding_dim=32,
                  num_heads=2, num_layers=2, dropout=0.0)
    values.update(overrides)
    return ModelConfig(**values)


def test_output_shape():
    layer = InputEmbedding(make_config())
    ids = torch.randint(0, 20, (4, 10))
    assert layer(ids).shape == (4, 10, 32)


def test_position_changes_the_vector():
    layer = InputEmbedding(make_config()).eval()
    ids = torch.full((1, 5), 7)             # the same token five times
    out = layer(ids)
    assert not torch.allclose(out[0, 0], out[0, 1])


def test_same_token_same_position_matches_across_batch():
    layer = InputEmbedding(make_config()).eval()
    ids = torch.tensor([[3, 4, 5], [3, 9, 9]])
    out = layer(ids)
    assert torch.allclose(out[0, 0], out[1, 0])


def test_sequence_longer_than_context_rejected():
    layer = InputEmbedding(make_config(context_length=8))
    with pytest.raises(ValueError):
        layer(torch.randint(0, 20, (1, 9)))


def test_positional_embedding_rejects_long_sequence():
    with pytest.raises(ValueError):
        PositionalEmbedding(8, 32)(9)


def test_missing_vocab_size_rejected():
    with pytest.raises(ValueError):
        InputEmbedding(make_config(vocab_size=None))


def test_wrong_input_shape_rejected():
    layer = InputEmbedding(make_config())
    with pytest.raises(ValueError):
        layer(torch.randint(0, 20, (10,)))


def test_token_id_out_of_range_rejected():
    layer = InputEmbedding(make_config())
    with pytest.raises(IndexError):
        layer(torch.tensor([[25]]))


def test_eval_mode_is_deterministic():
    layer = InputEmbedding(make_config(dropout=0.5)).eval()
    ids = torch.randint(0, 20, (2, 6))
    assert torch.equal(layer(ids), layer(ids))


def test_gradients_reach_both_embedding_tables():
    layer = InputEmbedding(make_config())
    layer(torch.randint(0, 20, (2, 6))).sum().backward()
    assert layer.token_embedding.embedding.weight.grad is not None
    assert layer.position_embedding.embedding.weight.grad is not None

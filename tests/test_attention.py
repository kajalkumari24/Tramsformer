import pytest
import torch
import torch.nn.functional as F

from model.attention import CausalSelfAttention
from model.configuration import ModelConfig


def make_config(**overrides):
    values = dict(vocab_size=20, context_length=16, embedding_dim=32,
                  num_heads=4, num_layers=2, dropout=0.0)
    values.update(overrides)
    return ModelConfig(**values)


def test_output_shape():
    layer = CausalSelfAttention(make_config())
    x = torch.randn(3, 10, 32)
    assert layer(x).shape == (3, 10, 32)


def test_shorter_than_context_works():
    layer = CausalSelfAttention(make_config())
    assert layer(torch.randn(2, 5, 32)).shape == (2, 5, 32)


def test_no_peeking_at_future_tokens():
    torch.manual_seed(0)
    layer = CausalSelfAttention(make_config()).eval()
    a = torch.randn(1, 12, 32)
    b = a.clone()
    b[:, 8:] = torch.randn(1, 4, 32)           # change only positions 8..11
    out_a, out_b = layer(a), layer(b)
    assert torch.allclose(out_a[:, :8], out_b[:, :8], atol=1e-6)
    assert not torch.allclose(out_a[:, 8:], out_b[:, 8:])


def test_attention_weights_are_valid():
    layer = CausalSelfAttention(make_config()).eval()
    _, w = layer(torch.randn(2, 9, 32), return_weights=True)
    assert w.shape == (2, 4, 9, 9)
    assert torch.allclose(w.sum(dim=-1), torch.ones(2, 4, 9), atol=1e-5)   # rows sum to 1
    assert torch.count_nonzero(w.triu(diagonal=1)) == 0                   # nothing above the diagonal
    assert torch.allclose(w[..., 0, 0], torch.ones(2, 4))                  # first token sees only itself


def test_matches_pytorch_reference():
    torch.manual_seed(0)
    layer = CausalSelfAttention(make_config()).eval()
    x = torch.randn(2, 11, 32)
    q, k, v = layer.qkv_proj(x).split(32, dim=2)
    q, k, v = (t.view(2, 11, 4, 8).transpose(1, 2) for t in (q, k, v))
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    ref = layer.out_proj(ref.transpose(1, 2).reshape(2, 11, 32))
    assert torch.allclose(layer(x), ref, atol=1e-5)


def test_sequence_longer_than_context_rejected():
    layer = CausalSelfAttention(make_config(context_length=8))
    with pytest.raises(ValueError):
        layer(torch.randn(1, 9, 32))


def test_wrong_shapes_rejected():
    layer = CausalSelfAttention(make_config())
    with pytest.raises(ValueError):
        layer(torch.randn(5, 32))
    with pytest.raises(ValueError):
        layer(torch.randn(1, 5, 16))


def test_gradients_flow():
    layer = CausalSelfAttention(make_config())
    layer(torch.randn(2, 6, 32)).sum().backward()
    assert layer.qkv_proj.weight.grad is not None
    assert layer.out_proj.weight.grad is not None


def test_eval_mode_is_deterministic():
    layer = CausalSelfAttention(make_config(dropout=0.5)).eval()
    x = torch.randn(2, 6, 32)
    assert torch.equal(layer(x), layer(x))

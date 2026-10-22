import math

import pytest
import torch

from model.configuration import ModelConfig
from model.gpt import GPT
from model.transformer_block import TransformerBlock


def make_config(**overrides):
    values = dict(vocab_size=20, context_length=16, embedding_dim=32,
                  num_heads=4, num_layers=2, dropout=0.0)
    values.update(overrides)
    return ModelConfig(**values)


def expected_params(cfg):
    C, V, T, L = cfg.embedding_dim, cfg.vocab_size, cfg.context_length, cfg.num_layers
    embeddings = V * C + T * C
    block = (4 * C                          # two LayerNorms (weight + bias)
             + (3 * C * C + 3 * C)          # qkv projection
             + (C * C + C)                  # attention output projection
             + (4 * C * C + 4 * C)          # feed-forward up
             + (4 * C * C + C))             # feed-forward down
    return embeddings + L * block + 2 * C   # + final LayerNorm


def test_block_keeps_shape():
    block = TransformerBlock(make_config())
    x = torch.randn(3, 10, 32)
    assert block(x).shape == (3, 10, 32)


def test_block_is_identity_when_sublayers_output_zero():
    block = TransformerBlock(make_config()).eval()
    with torch.no_grad():
        for layer in (block.attn.out_proj, block.ffn.net[2]):
            layer.weight.zero_()
            layer.bias.zero_()
    x = torch.randn(2, 6, 32)
    assert torch.allclose(block(x), x)      # proves the residual path is wired


def test_logits_shape_and_no_loss_without_targets():
    model = GPT(make_config())
    logits, loss = model(torch.randint(0, 20, (3, 10)))
    assert logits.shape == (3, 10, 20)
    assert loss is None


def test_initial_loss_is_close_to_random_guessing():
    torch.manual_seed(0)
    cfg = make_config()
    model = GPT(cfg).eval()
    x = torch.randint(0, 20, (8, 16))
    # Independent targets. Using targets == inputs would give a lower loss,
    # because the tied output layer slightly favours repeating the input token.
    y = torch.randint(0, 20, (8, 16))
    _, loss = model(x, y)

    assert abs(loss.item() - math.log(cfg.vocab_size)) < 0.3


def test_parameter_count_matches_formula():
    cfg = make_config()
    assert GPT(cfg).num_parameters() == expected_params(cfg)


def test_output_layer_shares_embedding_weights():
    model = GPT(make_config())
    assert model.lm_head.weight is model.input_embedding.token_embedding.embedding.weight


def test_whole_model_never_sees_the_future():
    torch.manual_seed(0)
    model = GPT(make_config()).eval()
    a = torch.randint(0, 20, (1, 12))
    b = a.clone()
    b[:, 8:] = (b[:, 8:] + 1) % 20          # change only positions 8..11
    la, _ = model(a)
    lb, _ = model(b)
    assert torch.allclose(la[:, :8], lb[:, :8], atol=1e-5)


def test_sequence_longer_than_context_rejected():
    model = GPT(make_config(context_length=8))
    with pytest.raises(ValueError):
        model(torch.randint(0, 20, (1, 9)))


def test_target_shape_mismatch_rejected():
    model = GPT(make_config())
    with pytest.raises(ValueError):
        model(torch.randint(0, 20, (2, 6)), torch.randint(0, 20, (2, 5)))


def test_missing_vocab_size_rejected():
    with pytest.raises(ValueError):
        GPT(make_config(vocab_size=None))


def test_every_parameter_receives_a_gradient():
    model = GPT(make_config())
    x = torch.randint(0, 20, (2, 8))
    _, loss = model(x, x)
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"No gradient for: {missing}"


def test_eval_mode_is_deterministic():
    model = GPT(make_config(dropout=0.5)).eval()
    x = torch.randint(0, 20, (2, 8))
    assert torch.equal(model(x)[0], model(x)[0])


def test_model_can_learn_a_simple_rule():
    torch.manual_seed(0)
    cfg = make_config()
    model = GPT(cfg).train()
    x = torch.randint(0, 20, (8, 12))
    y = (x + 1) % 20                          # rule: next token = current token + 1
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = model(x, y)[1].item()
    for _ in range(300):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.5, f"loss only went from {first:.3f} to {loss.item():.3f}"

"""Multi-head causal self-attention, written out step by step.

For every position the layer asks: "which earlier positions matter to me?"
  query (Q): what this position is looking for
  key   (K): what each position offers
  value (V): the information each position hands over
The score between two positions is Q . K. Softmax turns scores into weights
that add up to 1, and the output is the weighted sum of the values.
The causal mask stops a position from looking at later positions.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.embedding_dim = config.embedding_dim
        self.context_length = config.context_length

        # One linear layer produces Q, K and V together (3 x embedding_dim outputs).
        self.qkv_proj = nn.Linear(self.embedding_dim, 3 * self.embedding_dim)
        # Mixes the heads' results back into one vector per position.
        self.out_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # True where attention is ALLOWED: position t may see positions 0..t.
        mask = torch.tril(torch.ones(self.context_length, self.context_length, dtype=torch.bool))
        self.register_buffer(
            "causal_mask",
            mask.view(1, 1, self.context_length, self.context_length),
            persistent=False,
        )

    def forward(self, x, return_weights=False):
        """x: (batch, seq_len, embedding_dim) -> same shape.

        With return_weights=True also returns the attention weights,
        shape (batch, num_heads, seq_len, seq_len), for inspection.
        """
        if x.dim() != 3:
            raise ValueError(f"x must have shape (batch, seq_len, embedding_dim), got {tuple(x.shape)}")
        batch, seq_len, channels = x.shape
        if channels != self.embedding_dim:
            raise ValueError(f"Expected embedding_dim {self.embedding_dim}, got {channels}")
        if seq_len > self.context_length:
            raise ValueError(f"Sequence length {seq_len} is longer than context_length {self.context_length}")

        # 1. Queries, keys, values
        q, k, v = self.qkv_proj(x).split(self.embedding_dim, dim=2)

        # 2. Split into heads: (B, T, C) -> (B, num_heads, T, head_dim)
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Attention scores, scaled so softmax does not saturate
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)   # (B, heads, T, T)

        # 4. Causal mask: forbidden positions get -inf, so softmax gives them weight 0
        allowed = self.causal_mask[:, :, :seq_len, :seq_len]
        scores = scores.masked_fill(~allowed, float("-inf"))

        # 5. Softmax over the keys: each row sums to 1
        weights = F.softmax(scores, dim=-1)

        # 6. Weighted sum of values, then merge the heads back together
        out = self.attn_dropout(weights) @ v                             # (B, heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, channels)
        out = self.resid_dropout(self.out_proj(out))

        if return_weights:
            return out, weights
        return out

"""Positional embeddings: position numbers -> vectors.

Self-attention has no built-in sense of order, so we give every position
0, 1, 2, ... context_length-1 its own learnable vector.
"""
import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, context_length, embedding_dim):
        super().__init__()
        self.context_length = context_length
        self.embedding = nn.Embedding(context_length, embedding_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, seq_len, device=None):
        """Return position vectors of shape (seq_len, embedding_dim)."""
        if seq_len > self.context_length:
            raise ValueError(
                f"Sequence length {seq_len} is longer than "
                f"context_length {self.context_length}"
            )
        if device is None:
            device = self.embedding.weight.device
        positions = torch.arange(seq_len, device=device)
        return self.embedding(positions)

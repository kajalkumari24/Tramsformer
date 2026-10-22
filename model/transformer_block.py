"""One transformer block: attention, then a feed-forward network.

Each half is wrapped the same way:  x = x + sublayer(LayerNorm(x))
  - LayerNorm keeps the numbers well scaled before each sublayer.
  - The "x +" is a residual connection: the block only has to learn a
    correction, and gradients can flow straight through during training.
"""
import torch.nn as nn

from .attention import CausalSelfAttention


class FeedForward(nn.Module):
    """Expand to 4x the width, apply GELU, shrink back. Works on each position separately."""

    def __init__(self, config):
        super().__init__()
        dim = config.embedding_dim
        self.net = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embedding_dim)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.embedding_dim)
        self.ffn = FeedForward(config)

    def forward(self, x):                      # x: (batch, seq_len, embedding_dim)
        x = x + self.attn(self.ln1(x))         # positions share information
        x = x + self.ffn(self.ln2(x))          # each position thinks on its own
        return x

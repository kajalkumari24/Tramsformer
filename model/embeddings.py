"""Token embeddings, and the combined input layer (token + position)."""
import torch.nn as nn

from .position import PositionalEmbedding


class TokenEmbedding(nn.Module):
    """Token IDs -> vectors.  (batch, seq_len) -> (batch, seq_len, embedding_dim)."""

    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, token_ids):
        return self.embedding(token_ids)


class InputEmbedding(nn.Module):
    """Token vectors + position vectors, followed by dropout.

    This is the first layer of the GPT: it turns token IDs into the
    hidden vectors that the transformer blocks will work on.
    """

    def __init__(self, config):
        super().__init__()
        if config.vocab_size is None:
            raise ValueError(
                "config.vocab_size is not set. "
                "Use config.with_vocab_size(tokenizer.vocab_size)."
            )
        self.token_embedding = TokenEmbedding(config.vocab_size, config.embedding_dim)
        self.position_embedding = PositionalEmbedding(
            config.context_length, config.embedding_dim
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, token_ids):
        if token_ids.dim() != 2:
            raise ValueError(
                f"token_ids must have shape (batch, seq_len), got {tuple(token_ids.shape)}"
            )
        seq_len = token_ids.shape[1]
        tokens = self.token_embedding(token_ids)                          # (B, T, C)
        positions = self.position_embedding(seq_len, token_ids.device)    # (T, C)
        return self.dropout(tokens + positions)                           # broadcasts over B

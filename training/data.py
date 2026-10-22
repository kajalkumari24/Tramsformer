"""Turn a long list of token IDs into training batches."""
import torch


def split_data(token_ids, train_split):
    """Split into train and validation tensors. Order is kept (no shuffling),
    so validation text is genuinely unseen during training."""
    data = torch.tensor(token_ids, dtype=torch.long)
    n_train = int(len(data) * train_split)
    return data[:n_train], data[n_train:]


def get_batch(data, batch_size, context_length):
    """Random windows of the text.

    x = tokens[i : i + T]
    y = tokens[i + 1 : i + T + 1]   (the same window shifted one step ahead)
    At every position the model must predict the next token, so y is the answer key.
    """
    if len(data) <= context_length:
        raise ValueError(
            f"Data has {len(data)} tokens, but context_length is {context_length}. "
            "Need at least context_length + 1 tokens."
        )
    starts = torch.randint(len(data) - context_length, (batch_size,), device=data.device)
    offsets = torch.arange(context_length, device=data.device)
    index = starts[:, None] + offsets[None, :]          # (B, T)
    return data[index], data[index + 1]

"""Text generation: the model predicts one token, we append it, and repeat.

Three ways to choose the next token from the model's scores (logits):
  greedy       (temperature = 0): always the single most likely token. Repetitive.
  temperature  (> 0): divide the logits first. Below 1 = safer, above 1 = more random.
  top-k        keep only the k most likely tokens, then sample among them.
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def generate(model, token_ids, max_new_tokens, temperature=1.0, top_k=None):
    """token_ids: (batch, prompt_len) integers. Returns (batch, prompt_len + max_new_tokens)."""
    if token_ids.dim() != 2 or token_ids.size(1) < 1:
        raise ValueError("token_ids must have shape (batch, prompt_len) with prompt_len >= 1")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be 0 or more")
    if temperature < 0:
        raise ValueError("temperature must be 0 or more (0 means greedy)")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be at least 1 (or None to turn it off)")

    was_training = model.training
    model.eval()                                    # dropout off while generating
    context_length = model.config.context_length

    for _ in range(max_new_tokens):
        context = token_ids[:, -context_length:]    # the model only sees its last window
        logits, _ = model(context)
        logits = logits[:, -1, :]                   # scores for the NEXT token only

        if temperature == 0:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                kth_best = torch.topk(logits, k).values[:, -1, None]
                logits = logits.masked_fill(logits < kth_best, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        token_ids = torch.cat([token_ids, next_id], dim=1)

    model.train(was_training)
    return token_ids


def generate_text(model, tokenizer, prompt, max_new_tokens=200, temperature=0.8, top_k=40):
    """Prompt text in, prompt + continuation out.

    Characters the tokenizer never saw in training are skipped. If nothing
    usable is left, generation starts from a newline (or token 0).
    """
    device = next(model.parameters()).device
    ids = tokenizer.encode(prompt, errors="ignore")
    if not ids:
        ids = tokenizer.encode("\n", errors="ignore") or [0]
    start = torch.tensor([ids], dtype=torch.long, device=device)
    out = generate(model, start, max_new_tokens, temperature, top_k)
    return tokenizer.decode(out[0].tolist())

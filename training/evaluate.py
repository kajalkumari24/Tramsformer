"""Evaluation: full-dataset loss, perplexity, generation speed and memory."""

import math
import time

import torch
import torch.nn.functional as F

from model.generation import generate


@torch.no_grad()
def full_loss(model, data, context_length, batch_size=32):
    """Mean next-token loss over the WHOLE dataset (no random sampling).

    The text is cut into consecutive, non-overlapping windows of context_length
    tokens, so the answer is the same every time. A leftover piece shorter than
    one window is ignored. Early positions in each window have less context than
    later ones, so this is a slightly pessimistic estimate.
    """
    if len(data) <= context_length:
        raise ValueError("Data must have more than context_length tokens.")

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    data = data.to(device)

    starts = torch.arange(0, len(data) - context_length, context_length, device=device)
    offsets = torch.arange(context_length, device=device)
    total_loss, total_tokens = 0.0, 0

    try:
        for i in range(0, len(starts), batch_size):
            index = starts[i:i + batch_size][:, None] + offsets[None, :]
            x, y = data[index], data[index + 1]
            logits, _ = model(x)
            loss_sum = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum"
            )
            total_loss += loss_sum.item()
            total_tokens += y.numel()
    finally:
        model.train(was_training)

    return total_loss / total_tokens


def loss_metrics(loss):
    """Turn a loss (in nats) into easier-to-read numbers."""
    return {
        "loss": loss,
        "perplexity": math.exp(loss),          # "as unsure as choosing among N tokens"
        "bits_per_token": loss / math.log(2),  # bits per character for a char-level model
    }


@torch.no_grad()
def measure_generation_speed(model, context_length, device, new_tokens=200, runs=3):
    """Greedy-generate new_tokens tokens several times; report the fastest run."""
    device = torch.device(device)
    prompt = torch.zeros((1, 1), dtype=torch.long, device=device)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    generate(model, prompt, 5, temperature=0)                      # warm-up
    times = []
    for _ in range(runs):
        sync()
        start = time.perf_counter()
        generate(model, prompt, new_tokens, temperature=0)
        sync()
        times.append(time.perf_counter() - start)

    best = min(times)
    return {"tokens_per_second": new_tokens / best, "seconds_per_token": best / new_tokens}


def memory_report(model, device):
    """Parameter count and size; also GPU peak memory when running on CUDA."""
    device = torch.device(device)
    report = {
        "parameters": sum(p.numel() for p in model.parameters()),
        "parameter_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6,
    }
    if device.type == "cuda":
        report["gpu_peak_mb"] = torch.cuda.max_memory_allocated(device) / 1e6
    return report

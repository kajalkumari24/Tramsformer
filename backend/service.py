"""The model service: loads the trained GPT once and answers generation requests."""
import threading

import torch

from model.configuration import get_device
from model.generation import generate
from model.tokenizer import CharTokenizer
from training.checkpoint import load_model_from_checkpoint

from .prompt import build_chat_prompt, extract_reply


class ModelService:
    def __init__(self, model, tokenizer, device, checkpoint_info=None):
        if tokenizer.vocab_size != model.config.vocab_size:
            raise ValueError(
                f"Tokenizer has {tokenizer.vocab_size} characters but the model expects "
                f"{model.config.vocab_size}. They must come from the same training run."
            )
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.checkpoint_info = checkpoint_info or {}
        self._lock = threading.Lock()      # one generation at a time on the shared model

    @classmethod
    def from_checkpoint(cls, checkpoint_path, tokenizer_path, device="auto"):
        device = get_device(device)
        tokenizer = CharTokenizer.load(tokenizer_path)
        model, checkpoint = load_model_from_checkpoint(checkpoint_path, device)
        info = {
            "checkpoint_step": checkpoint["step"],
            "best_val_loss": checkpoint["best_val_loss"],
        }
        return cls(model, tokenizer, device, info)

    def info(self):
        cfg = self.model.config
        return {
            "parameters": self.model.num_parameters(),
            "vocab_size": cfg.vocab_size,
            "context_length": cfg.context_length,
            "embedding_dim": cfg.embedding_dim,
            "num_heads": cfg.num_heads,
            "num_layers": cfg.num_layers,
            "device": str(self.device),
            **self.checkpoint_info,
        }

    def _ignored(self, text):
        """Characters the tokenizer has never seen. They are skipped, not an error."""
        return sorted(set(text) - set(self.tokenizer.stoi))

    def _continue(self, prompt, max_new_tokens, temperature, top_k, seed):
        ids = self.tokenizer.encode(prompt, errors="ignore")
        if not ids:
            ids = self.tokenizer.encode("\n", errors="ignore") or [0]
        start = torch.tensor([ids], dtype=torch.long, device=self.device)
        with self._lock:
            if seed is not None:
                torch.manual_seed(seed)
            out = generate(self.model, start, max_new_tokens, temperature, top_k)
        return self.tokenizer.decode(out[0, len(ids):].tolist())

    def complete(self, prompt, max_new_tokens, temperature, top_k, seed=None):
        """Plain text continuation. Returns only the newly generated text."""
        text = self._continue(prompt, max_new_tokens, temperature, top_k, seed)
        return {"text": text, "ignored_characters": self._ignored(prompt)}

    def chat(self, messages, max_new_tokens, temperature, top_k, seed=None):
        """Write the history as a script and return BOT's next speech."""
        prompt = build_chat_prompt(messages)
        generated = self._continue(prompt, max_new_tokens, temperature, top_k, seed)
        everything = "".join(m["content"] for m in messages)
        return {"reply": extract_reply(generated), "ignored_characters": self._ignored(everything)}

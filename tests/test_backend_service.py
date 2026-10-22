import pytest
import torch

from backend.service import ModelService
from model.configuration import ModelConfig, TrainingConfig
from model.gpt import GPT
from model.tokenizer import CharTokenizer
from training.checkpoint import save_checkpoint

ALPHABET = "YOUBT:\n abcdefgh"


def make_service():
    tokenizer = CharTokenizer.from_text(ALPHABET)
    cfg = ModelConfig(vocab_size=tokenizer.vocab_size, context_length=32, embedding_dim=32,
                      num_heads=4, num_layers=2, dropout=0.0)
    return ModelService(GPT(cfg), tokenizer, torch.device("cpu"), {"checkpoint_step": 5})


def test_complete_returns_only_the_new_text():
    service = make_service()
    result = service.complete("abc", max_new_tokens=12, temperature=1.0, top_k=5)
    assert len(result["text"]) == 12
    assert result["ignored_characters"] == []


def test_unknown_characters_are_reported_not_fatal():
    service = make_service()
    result = service.complete("ab#é", max_new_tokens=3, temperature=1.0, top_k=None)
    assert result["ignored_characters"] == ["#", "é"]
    assert len(result["text"]) == 3


def test_same_seed_gives_the_same_text():
    service = make_service()
    a = service.complete("abc", 20, 1.0, None, seed=11)["text"]
    b = service.complete("abc", 20, 1.0, None, seed=11)["text"]
    assert a == b


def test_greedy_is_repeatable_without_a_seed():
    service = make_service()
    assert (service.complete("abc", 15, 0.0, None)["text"]
            == service.complete("abc", 15, 0.0, None)["text"])


def test_chat_reply_is_a_single_speech():
    service = make_service()
    messages = [{"role": "user", "content": "abc é"}]
    result = service.chat(messages, max_new_tokens=40, temperature=1.0, top_k=None, seed=1)
    assert isinstance(result["reply"], str)
    assert "\n\n" not in result["reply"]
    assert result["ignored_characters"] == ["é"]


def test_prompt_with_no_known_characters_still_generates():
    service = make_service()
    assert len(service.complete("###", 4, 1.0, None)["text"]) == 4


def test_info_reports_model_facts():
    service = make_service()
    info = service.info()
    assert info["parameters"] == service.model.num_parameters()
    assert info["context_length"] == 32 and info["checkpoint_step"] == 5


def test_mismatched_tokenizer_and_model_rejected():
    tokenizer = CharTokenizer.from_text("abc")
    cfg = ModelConfig(vocab_size=10, context_length=16, embedding_dim=32,
                      num_heads=4, num_layers=1, dropout=0.0)
    with pytest.raises(ValueError):
        ModelService(GPT(cfg), tokenizer, torch.device("cpu"))


def test_loading_from_a_real_checkpoint(tmp_path):
    tokenizer = CharTokenizer.from_text(ALPHABET)
    cfg = ModelConfig(vocab_size=tokenizer.vocab_size, context_length=16, embedding_dim=32,
                      num_heads=4, num_layers=1, dropout=0.0)
    model = GPT(cfg)
    optimizer = torch.optim.AdamW(model.parameters())
    tcfg = TrainingConfig(learning_rate=0.001, batch_size=2, max_steps=10, eval_interval=5,
                          random_seed=0, train_split=0.9)
    save_checkpoint(tmp_path / "best.pt", model, optimizer, 42, 1.25, cfg, tcfg, [])
    tokenizer.save(tmp_path / "tokenizer.json")

    service = ModelService.from_checkpoint(tmp_path / "best.pt", tmp_path / "tokenizer.json", "cpu")
    info = service.info()
    assert info["checkpoint_step"] == 42 and info["best_val_loss"] == 1.25
    assert info["parameters"] == model.num_parameters()

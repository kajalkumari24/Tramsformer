import pytest

from model.tokenizer import CharTokenizer


def test_vocab_is_sorted_and_sized():
    tok = CharTokenizer.from_text("banana")
    assert tok.vocab_size == 3
    assert list(tok.stoi) == ["a", "b", "n"]


def test_round_trip():
    text = "Plants need water"
    tok = CharTokenizer.from_text(text)
    assert tok.decode(tok.encode(text)) == text


def test_empty_text_rejected():
    with pytest.raises(ValueError):
        CharTokenizer().build_vocab("")


def test_encode_before_build_rejected():
    with pytest.raises(ValueError):
        CharTokenizer().encode("abc")


def test_unknown_character_strict_and_ignore():
    tok = CharTokenizer.from_text("abc")
    with pytest.raises(ValueError):
        tok.encode("abz")
    assert tok.decode(tok.encode("abz", errors="ignore")) == "ab"


def test_unknown_token_id_rejected():
    tok = CharTokenizer.from_text("abc")
    with pytest.raises(ValueError):
        tok.decode([99])


def test_save_and_load(tmp_path):
    tok = CharTokenizer.from_text("héllo wörld")
    path = tmp_path / "tok.json"
    tok.save(path)
    loaded = CharTokenizer.load(path)
    assert loaded.stoi == tok.stoi
    assert loaded.decode(loaded.encode("héllo")) == "héllo"

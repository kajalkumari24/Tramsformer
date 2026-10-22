"""Character-level tokenizer: text <-> token IDs, with save and load."""
import json
from pathlib import Path


class CharTokenizer:
    """Convert text to character IDs and character IDs back to text."""

    def __init__(self):
        self.stoi = {}
        self.itos = {}

    def build_vocab(self, text):
        """Build a vocabulary from the supplied training text."""
        if not text:
            raise ValueError("Training text cannot be empty.")

        characters = sorted(set(text))
        self.stoi = {character: index for index, character in enumerate(characters)}
        self.itos = {index: character for character, index in self.stoi.items()}

    @classmethod
    def from_text(cls, text):
        tokenizer = cls()
        tokenizer.build_vocab(text)
        return tokenizer

    @property
    def vocab_size(self):
        """Return the number of tokens in the vocabulary."""
        return len(self.stoi)

    def encode(self, text, errors="strict"):
        """Convert text into a list of token IDs.

        errors="strict" raises on characters the tokenizer has never seen.
        errors="ignore" skips them (useful for chatbot input).
        """
        if not self.stoi:
            raise ValueError("Build the vocabulary before encoding.")
        if errors not in ("strict", "ignore"):
            raise ValueError("errors must be 'strict' or 'ignore'")

        if errors == "ignore":
            return [self.stoi[c] for c in text if c in self.stoi]

        unknown = set(text) - set(self.stoi)
        if unknown:
            raise ValueError(f"Text contains unknown characters: {sorted(unknown)!r}")
        return [self.stoi[character] for character in text]

    def decode(self, token_ids):
        """Convert a list of token IDs back into text."""
        if not self.itos:
            raise ValueError("Build the vocabulary before decoding.")
        try:
            return "".join(self.itos[int(token_id)] for token_id in token_ids)
        except KeyError as error:
            raise ValueError(f"Unknown token ID: {error.args[0]}") from error

    def save(self, path):
        """Save the vocabulary as JSON so training and the chatbot share it."""
        if not self.stoi:
            raise ValueError("Build the vocabulary before saving.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "type": "char",
            "chars": [self.itos[i] for i in range(len(self.itos))],
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("type") != "char":
            raise ValueError("Not a character tokenizer file.")
        tokenizer = cls()
        tokenizer.stoi = {c: i for i, c in enumerate(data["chars"])}
        tokenizer.itos = {i: c for c, i in tokenizer.stoi.items()}
        return tokenizer

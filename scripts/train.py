"""Train the Mini GPT from a text file.

Example (run from the project folder):
    python scripts/train.py --data data/input.txt
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.configuration import get_device, load_model_config, load_training_config, set_seed
from model.gpt import GPT
from model.tokenizer import CharTokenizer
from training.data import split_data
from training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train the Mini GPT.")
    parser.add_argument("--data", default=str(ROOT / "data" / "input.txt"))
    parser.add_argument("--checkpoint_dir", default=str(ROOT / "checkpoints"))
    parser.add_argument("--resume", default=None, help="Path to a checkpoint to resume from.")
    args = parser.parse_args()

    train_cfg = load_training_config()
    model_cfg = load_model_config()
    set_seed(train_cfg.random_seed)
    device = get_device("auto")

    text = Path(args.data).read_text(encoding="utf-8")
    tokenizer = CharTokenizer.from_text(text)
    model_cfg = model_cfg.with_vocab_size(tokenizer.vocab_size)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(checkpoint_dir / "tokenizer.json")

    token_ids = tokenizer.encode(text)
    train_data, val_data = split_data(token_ids, train_cfg.train_split)

    model = GPT(model_cfg)
    trainer = Trainer(model, train_data, val_data, train_cfg, model_cfg, device, checkpoint_dir)
    trainer.train(resume_from=args.resume)


if __name__ == "__main__":
    main()

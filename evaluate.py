"""
Entry point for evaluating a trained discourse classification model.

Usage:
    python evaluate.py --config configs/bert_base.yaml \
                       --checkpoint checkpoints/best_model.pt \
                       --split test
"""

import argparse
import logging

import torch
from torch.utils.data import DataLoader

from discourse.config import DiscourseConfig
from discourse.data import build_datasets
from discourse.model import build_model
from discourse.evaluate import evaluate, print_results
from discourse.train import _collate_hde, _collate_token

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    args = parser.parse_args()

    config = DiscourseConfig.from_yaml(args.config)
    train_ds, dev_ds, test_ds, tokenizer = build_datasets(config)
    dataset = dev_ds if args.split == "dev" else test_ds

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, tokenizer=tokenizer)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.to(device)

    collate_fn = _collate_hde if config.model.use_hde else _collate_token
    loader = DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    results = evaluate(model, loader, device, config)
    print_results(results)


if __name__ == "__main__":
    main()

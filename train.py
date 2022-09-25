"""
Entry point for training discourse classification models.

Usage:
    python train.py --config configs/bert_base.yaml
    python train.py --config configs/hde_base.yaml
"""

import argparse
import logging

from discourse.config import DiscourseConfig
from discourse.data import build_datasets
from discourse.train import train

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = DiscourseConfig.from_yaml(args.config)
    train_ds, dev_ds, _, tokenizer = build_datasets(config)
    train(config, train_ds, dev_ds, tokenizer)


if __name__ == "__main__":
    main()

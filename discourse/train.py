import logging
import os
import random

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from .config import DiscourseConfig
from .evaluate import evaluate, print_results
from .model import build_model

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _collate_hde(batch):
    return {
        "segment_input_ids": torch.stack([b["segment_input_ids"] for b in batch]),
        "segment_attention_mask": torch.stack(
            [b["segment_attention_mask"] for b in batch]
        ),
        "sent_positions": torch.stack([b["sent_positions"] for b in batch]),
        "sentence_mask": torch.stack([b["sentence_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


def _collate_token(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


def train(
    config: DiscourseConfig,
    train_dataset,
    dev_dataset,
    tokenizer,
) -> None:
    """
    Full training loop for both the BERT/BigBIRD baseline and the HDE model.

    Checkpoints the best model (by overall dev F1) to `config.output_dir`.
    Early stops after `config.train.patience` epochs without improvement.
    """
    set_seed(config.train.seed)
    os.makedirs(config.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    use_hde = config.model.use_hde
    collate_fn = _collate_hde if use_hde else _collate_token

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    model = build_model(config, tokenizer=tokenizer)
    model.to(device)

    no_decay = {"bias", "LayerNorm.weight"}
    optimizer_params = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": config.train.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_params, lr=config.train.learning_rate)

    total_steps = len(train_loader) * config.train.n_epochs
    warmup_steps = int(total_steps * config.train.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_f1 = -1.0
    patience_count = 0
    ckpt_path = os.path.join(config.output_dir, "best_model.pt")

    for epoch in range(1, config.train.n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_steps = 0

        for step, batch in enumerate(train_loader, 1):
            if use_hde:
                loss, _ = model(
                    segment_input_ids=batch["segment_input_ids"].to(device),
                    segment_attention_mask=batch["segment_attention_mask"].to(device),
                    sent_positions=batch["sent_positions"].to(device),
                    sentence_mask=batch["sentence_mask"].to(device),
                    labels=batch["labels"].to(device),
                )
            else:
                loss, _ = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device),
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.train.max_grad_norm
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            n_steps += 1

            if step % 50 == 0:
                logger.info(
                    "Epoch %d | Step %d/%d | Loss %.4f",
                    epoch,
                    step,
                    len(train_loader),
                    epoch_loss / n_steps,
                )

        avg_loss = epoch_loss / n_steps
        logger.info("Epoch %d | Avg Loss %.4f", epoch, avg_loss)

        results = evaluate(model, dev_loader, device, config)
        print_results(results)
        dev_f1 = results["overall"]

        if dev_f1 > best_f1:
            best_f1 = dev_f1
            patience_count = 0
            torch.save(model.state_dict(), ckpt_path)
            logger.info("Epoch %d | New best dev F1 %.1f — checkpoint saved.", epoch, best_f1)
        else:
            patience_count += 1
            logger.info(
                "Epoch %d | Dev F1 %.1f (best %.1f) — patience %d/%d",
                epoch,
                dev_f1,
                best_f1,
                patience_count,
                config.train.patience,
            )
            if patience_count >= config.train.patience:
                logger.info("Early stopping.")
                break

    logger.info("Training complete. Best dev F1: %.1f", best_f1)
    logger.info("Checkpoint: %s", ckpt_path)

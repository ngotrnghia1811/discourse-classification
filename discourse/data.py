import os
import re
from typing import Dict, List, Optional, Tuple

import nltk
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from .config import (
    BIO2ID,
    BIO_LABELS,
    DISCOURSE_TYPES,
    LABEL2ID,
    DataConfig,
    DiscourseConfig,
    ModelConfig,
)

SENT_TOKEN = "[SENT]"


def ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)


def load_essay(essay_id: str, essay_dir: str) -> str:
    path = os.path.join(essay_dir, f"{essay_id}.txt")
    with open(path, encoding="utf-8") as f:
        return f.read()


def build_word_labels(essay_text: str, annotations: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Map discourse annotations to per-word BIO labels."""
    words = essay_text.split()
    labels = ["O"] * len(words)

    for _, row in annotations.iterrows():
        pred_str = str(row["predictionstring"])
        indices = [int(w) for w in pred_str.split() if w.isdigit()]
        dtype = row["discourse_type"]
        for pos, idx in enumerate(indices):
            if 0 <= idx < len(labels):
                labels[idx] = f"B-{dtype}" if pos == 0 else f"I-{dtype}"

    return words, labels


def _sentence_tokenize(text: str) -> List[str]:
    """Split text into sentences using NLTK."""
    ensure_nltk()
    return nltk.sent_tokenize(text)


def assign_sentence_label(
    sent_start_char: int,
    sent_end_char: int,
    essay_text: str,
    annotations: pd.DataFrame,
) -> str:
    """
    Assign a single discourse label to a sentence using majority overlap.
    Unannotated sentences return None.
    """
    sent_words = set(
        range(
            len(essay_text[:sent_start_char].split()),
            len(essay_text[:sent_end_char].split()),
        )
    )
    if not sent_words:
        return None

    votes: Dict[str, int] = {}
    for _, row in annotations.iterrows():
        pred_str = str(row["predictionstring"])
        ann_words = set(int(w) for w in pred_str.split() if w.isdigit())
        overlap = len(sent_words & ann_words)
        if overlap > 0:
            dtype = row["discourse_type"]
            votes[dtype] = votes.get(dtype, 0) + overlap

    if not votes:
        return None
    return max(votes, key=votes.__getitem__)


def build_sentence_labels(
    essay_text: str,
    annotations: pd.DataFrame,
) -> Tuple[List[str], List[str]]:
    """
    Segment essay into sentences and assign discourse labels.

    Label merging: unannotated sentences between two annotated ones inherit
    the label of the nearest annotated neighbor.
    Label imputing: leading unannotated sentences get Lead; trailing ones
    get Concluding Statement.

    Returns:
        sentences: list of sentence strings
        sent_labels: list of discourse type strings (one per sentence)
    """
    sentences = _sentence_tokenize(essay_text)

    char_pos = 0
    char_spans = []
    for sent in sentences:
        start = essay_text.find(sent, char_pos)
        end = start + len(sent)
        char_spans.append((start, end))
        char_pos = end

    raw_labels = []
    for start, end in char_spans:
        label = assign_sentence_label(start, end, essay_text, annotations)
        raw_labels.append(label)

    labels = _merge_labels(raw_labels)
    return sentences, labels


def _merge_labels(raw_labels: List[Optional[str]]) -> List[str]:
    """
    Merge/impute None labels in the sentence label sequence.
    - Impute leading Nones as Lead
    - Impute trailing Nones as Concluding Statement
    - Fill interior Nones by nearest neighbour
    """
    n = len(raw_labels)
    labels = list(raw_labels)

    # Impute leading
    first_known = next((i for i, l in enumerate(labels) if l is not None), None)
    if first_known is None:
        return ["Lead"] * n
    for i in range(first_known):
        labels[i] = "Lead"

    # Impute trailing
    last_known = next((i for i, l in enumerate(reversed(labels)) if l is not None), None)
    if last_known is not None:
        last_known_idx = n - 1 - last_known
        for i in range(last_known_idx + 1, n):
            labels[i] = "Concluding Statement"

    # Fill interior Nones (forward pass)
    for i in range(n):
        if labels[i] is None and i > 0 and labels[i - 1] is not None:
            labels[i] = labels[i - 1]

    # Backward pass for any remaining
    for i in range(n - 2, -1, -1):
        if labels[i] is None and labels[i + 1] is not None:
            labels[i] = labels[i + 1]

    # Final fallback
    labels = [l if l is not None else "Claim" for l in labels]
    return labels


class TokenLevelDataset(Dataset):
    """
    Dataset for token-level sequence labeling (BERT / BigBIRD baseline).
    Each example is a single essay truncated to max_seq_len tokens.
    Labels are BIO tags aligned to subword tokens.
    """

    def __init__(
        self,
        csv_file: str,
        essay_dir: str,
        tokenizer,
        max_len: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.examples: List[Dict] = []

        df = pd.read_csv(csv_file)
        for essay_id in df["id"].unique():
            try:
                essay_text = load_essay(essay_id, essay_dir)
            except FileNotFoundError:
                continue
            annotations = df[df["id"] == essay_id]
            words, word_labels = build_word_labels(essay_text, annotations)
            self.examples.append({"words": words, "labels": word_labels, "id": essay_id})

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        words = ex["words"]
        word_labels = ex["labels"]

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        word_ids = encoding.word_ids()
        label_ids = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != prev_word_id:
                label_ids.append(BIO2ID.get(word_labels[word_id], 0))
            else:
                label_ids.append(-100)
            prev_word_id = word_id

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label_ids, dtype=torch.long),
        }


class HDEDataset(Dataset):
    """
    Dataset for the Hierarchical Document Encoder (HDE).

    Preprocessing:
      1. Sentence segmentation (NLTK)
      2. Discourse label assignment per sentence
      3. Insert [SENT] token at the start of each sentence
      4. Split tokenized document into `num_segments` segments of `segment_len` tokens
      5. Record [SENT] token positions for extraction after BERT encoding

    Each example:
      - segment_input_ids:      [num_segments, segment_len]
      - segment_attention_mask: [num_segments, segment_len]
      - sent_positions:         [max_sentences, 2] — (segment_idx, token_idx) pairs
      - sentence_mask:          [max_sentences] — 1 for valid sentences
      - labels:                 [max_sentences] — integer class or BIO id
    """

    def __init__(
        self,
        csv_file: str,
        essay_dir: str,
        tokenizer,
        num_segments: int = 4,
        segment_len: int = 256,
        max_sentences: int = 100,
        label_scheme: str = "category",
    ):
        self.tokenizer = tokenizer
        self.num_segments = num_segments
        self.segment_len = segment_len
        self.max_sentences = max_sentences
        self.label_scheme = label_scheme
        self.sent_token_id = tokenizer.convert_tokens_to_ids(SENT_TOKEN)
        self.examples: List[Dict] = []

        df = pd.read_csv(csv_file)
        for essay_id in df["id"].unique():
            try:
                essay_text = load_essay(essay_id, essay_dir)
            except FileNotFoundError:
                continue
            annotations = df[df["id"] == essay_id]
            ex = self._process_essay(essay_text, annotations, essay_id)
            if ex is not None:
                self.examples.append(ex)

    def _process_essay(
        self, essay_text: str, annotations: pd.DataFrame, essay_id: str
    ) -> Optional[Dict]:
        sentences, sent_labels = build_sentence_labels(essay_text, annotations)

        if not sentences:
            return None

        total_tokens: List[int] = [self.tokenizer.cls_token_id]
        sent_flat_positions: List[int] = []
        sent_count = 0

        for sent in sentences:
            sent_toks = self.tokenizer.encode(sent, add_special_tokens=False)
            sent_flat_positions.append(len(total_tokens))
            total_tokens.append(self.sent_token_id)
            total_tokens.extend(sent_toks)
            sent_count += 1

        total_tokens.append(self.tokenizer.sep_token_id)

        max_flat_len = self.num_segments * self.segment_len
        if len(total_tokens) > max_flat_len:
            total_tokens = total_tokens[:max_flat_len - 1] + [self.tokenizer.sep_token_id]

        segment_ids = self._split_into_segments(total_tokens)
        sent_positions = self._resolve_sent_positions(sent_flat_positions)
        sentence_mask = torch.zeros(self.max_sentences, dtype=torch.long)
        n_valid = min(len(sentences), self.max_sentences)
        sentence_mask[:n_valid] = 1

        if self.label_scheme == "bio":
            label_ids = self._to_bio_labels(sent_labels)
        else:
            label_ids = [LABEL2ID.get(l, 0) for l in sent_labels]

        padded_labels = torch.full((self.max_sentences,), -100, dtype=torch.long)
        for i, lid in enumerate(label_ids[:n_valid]):
            padded_labels[i] = lid

        return {
            "segment_input_ids": segment_ids["input_ids"],
            "segment_attention_mask": segment_ids["attention_mask"],
            "sent_positions": sent_positions,
            "sentence_mask": sentence_mask,
            "labels": padded_labels,
            "id": essay_id,
        }

    def _split_into_segments(self, token_ids: List[int]) -> Dict[str, torch.Tensor]:
        """Split flat token sequence into `num_segments` segments of `segment_len`."""
        input_ids = torch.full(
            (self.num_segments, self.segment_len),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros(
            (self.num_segments, self.segment_len), dtype=torch.long
        )

        flat = torch.tensor(token_ids, dtype=torch.long)
        for seg_idx in range(self.num_segments):
            start = seg_idx * self.segment_len
            end = start + self.segment_len
            chunk = flat[start:end]
            chunk_len = len(chunk)
            input_ids[seg_idx, :chunk_len] = chunk
            attention_mask[seg_idx, :chunk_len] = 1

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def _resolve_sent_positions(
        self, flat_positions: List[int]
    ) -> torch.Tensor:
        """
        Convert flat token positions to (segment_idx, token_idx_in_segment) pairs.
        Returns tensor of shape [max_sentences, 2] padded with -1.
        """
        positions = torch.full((self.max_sentences, 2), -1, dtype=torch.long)
        for i, flat_pos in enumerate(flat_positions[: self.max_sentences]):
            seg_idx = flat_pos // self.segment_len
            tok_idx = flat_pos % self.segment_len
            if seg_idx < self.num_segments:
                positions[i, 0] = seg_idx
                positions[i, 1] = tok_idx
        return positions

    def _to_bio_labels(self, sent_labels: List[str]) -> List[int]:
        """Convert sentence category labels to BIO ids."""
        bio_ids = []
        prev = None
        for label in sent_labels:
            if label != prev:
                bio_ids.append(BIO2ID.get(f"B-{label}", 0))
            else:
                bio_ids.append(BIO2ID.get(f"I-{label}", 0))
            prev = label
        return bio_ids

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        return {
            k: v for k, v in ex.items() if k != "id"
        }


def get_tokenizer(model_name: str, add_sent_token: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if add_sent_token:
        tokenizer.add_special_tokens({"additional_special_tokens": [SENT_TOKEN]})
    return tokenizer


def build_datasets(config: DiscourseConfig):
    """
    Factory: build train, dev, test datasets depending on model type.

    Returns (train_ds, dev_ds, test_ds, tokenizer).
    """
    mc = config.model
    dc = config.data

    if mc.use_hde:
        tokenizer = get_tokenizer(mc.encoder, add_sent_token=True)
        train_ds = HDEDataset(
            dc.train_file, dc.essay_dir, tokenizer,
            num_segments=mc.num_segments, segment_len=mc.segment_len,
            max_sentences=mc.max_sentences, label_scheme=mc.label_scheme,
        )
        dev_ds = HDEDataset(
            dc.dev_file, dc.essay_dir, tokenizer,
            num_segments=mc.num_segments, segment_len=mc.segment_len,
            max_sentences=mc.max_sentences, label_scheme=mc.label_scheme,
        )
        test_ds = HDEDataset(
            dc.test_file, dc.essay_dir, tokenizer,
            num_segments=mc.num_segments, segment_len=mc.segment_len,
            max_sentences=mc.max_sentences, label_scheme=mc.label_scheme,
        )
    else:
        tokenizer = get_tokenizer(mc.encoder)
        train_ds = TokenLevelDataset(
            dc.train_file, dc.essay_dir, tokenizer, max_len=mc.max_seq_len
        )
        dev_ds = TokenLevelDataset(
            dc.dev_file, dc.essay_dir, tokenizer, max_len=mc.max_seq_len
        )
        test_ds = TokenLevelDataset(
            dc.test_file, dc.essay_dir, tokenizer, max_len=mc.max_seq_len
        )

    return train_ds, dev_ds, test_ds, tokenizer

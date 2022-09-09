"""
Evaluation for Argumentative Discourse Parsing.

F1 metric follows the Kaggle Feedback Prize 2021 scoring:
  A predicted span P (of class c) is a true positive if there exists a
  ground-truth span G of the same class c such that

      |P ∩ G| / max(|P|, |G|) > 0.5

  where spans are expressed as sets of *word indices*.

Per-class precision, recall, and F1 are computed; overall F1 is the
unweighted macro average across the 7 discourse types.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import (
    BIO2ID,
    BIO_LABELS,
    DISCOURSE_TYPES,
    ID2BIO,
    ID2LABEL,
    LABEL2ID,
    DiscourseConfig,
)


def bio_to_spans(
    word_labels: List[str],
) -> List[Tuple[str, int, int]]:
    """
    Convert a BIO label sequence to a list of (label, start, end) spans
    where start/end are *word* indices (inclusive, exclusive).
    """
    spans = []
    cur_label: Optional[str] = None
    cur_start: int = -1

    for i, tag in enumerate(word_labels):
        if tag.startswith("B-"):
            if cur_label is not None:
                spans.append((cur_label, cur_start, i))
            cur_label = tag[2:]
            cur_start = i
        elif tag.startswith("I-"):
            inner = tag[2:]
            if cur_label != inner:
                if cur_label is not None:
                    spans.append((cur_label, cur_start, i))
                cur_label = inner
                cur_start = i
        else:  # O
            if cur_label is not None:
                spans.append((cur_label, cur_start, i))
            cur_label = None
            cur_start = -1

    if cur_label is not None:
        spans.append((cur_label, cur_start, len(word_labels)))
    return spans


def category_to_spans(
    word_labels: List[str],
) -> List[Tuple[str, int, int]]:
    """
    Convert a category label sequence (no BIO) to spans by grouping
    consecutive identical labels. Each contiguous run becomes one span.
    """
    spans = []
    if not word_labels:
        return spans
    cur_label = word_labels[0]
    cur_start = 0
    for i in range(1, len(word_labels)):
        if word_labels[i] != cur_label:
            spans.append((cur_label, cur_start, i))
            cur_label = word_labels[i]
            cur_start = i
    spans.append((cur_label, cur_start, len(word_labels)))
    return spans


def _overlap_f1_for_class(
    pred_spans: List[Tuple[int, int]],
    gold_spans: List[Tuple[int, int]],
    threshold: float = 0.5,
) -> Tuple[float, float, float]:
    """Compute P/R/F1 for spans of a single class."""
    tp = 0
    matched_gold = set()

    for p_start, p_end in pred_spans:
        p_set = set(range(p_start, p_end))
        best_iou = 0.0
        best_g = -1
        for g_idx, (g_start, g_end) in enumerate(gold_spans):
            if g_idx in matched_gold:
                continue
            g_set = set(range(g_start, g_end))
            inter = len(p_set & g_set)
            if inter == 0:
                continue
            iou = inter / max(len(p_set), len(g_set))
            if iou > best_iou:
                best_iou = iou
                best_g = g_idx
        if best_iou > threshold and best_g >= 0:
            tp += 1
            matched_gold.add(best_g)

    precision = tp / len(pred_spans) if pred_spans else 0.0
    recall = tp / len(gold_spans) if gold_spans else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


def compute_overlap_f1(
    predictions: List[List[str]],
    ground_truths: List[List[str]],
    label_scheme: str = "bio",
) -> Dict[str, float]:
    """
    Compute per-class and overall overlap-based F1.

    Args:
        predictions:   list of word-label sequences (one per document)
        ground_truths: list of word-label sequences (one per document)
        label_scheme:  "bio" or "category"

    Returns:
        dict with per-class F1 and "overall" macro F1
    """
    span_fn = bio_to_spans if label_scheme == "bio" else category_to_spans

    class_pred_spans: Dict[str, List[Tuple[int, int]]] = {t: [] for t in DISCOURSE_TYPES}
    class_gold_spans: Dict[str, List[Tuple[int, int]]] = {t: [] for t in DISCOURSE_TYPES}

    offset = 0
    for pred_seq, gold_seq in zip(predictions, ground_truths):
        n = max(len(pred_seq), len(gold_seq))
        for label, s, e in span_fn(pred_seq):
            if label in class_pred_spans:
                class_pred_spans[label].append((offset + s, offset + e))
        for label, s, e in span_fn(gold_seq):
            if label in class_gold_spans:
                class_gold_spans[label].append((offset + s, offset + e))
        offset += n

    results: Dict[str, float] = {}
    f1s = []
    for dtype in DISCOURSE_TYPES:
        _, _, f1 = _overlap_f1_for_class(
            class_pred_spans[dtype], class_gold_spans[dtype]
        )
        results[dtype] = round(f1 * 100, 1)
        f1s.append(f1)

    results["overall"] = round(np.mean(f1s) * 100, 1)
    return results


def _predict_token_level(model, loader, device, config) -> List[List[str]]:
    """Run inference for BERT/BigBIRD baseline; returns word-label sequences."""
    model.eval()
    all_preds: List[List[str]] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            _, logits = model(input_ids=input_ids, attention_mask=attention_mask)

            if config.model.use_crf:
                pred_seqs = logits
                for pred_seq in pred_seqs:
                    all_preds.append([BIO_LABELS[i] for i in pred_seq])
            else:
                preds = logits.argmax(-1)
                for b in range(preds.shape[0]):
                    seq = []
                    for tok_idx in range(preds.shape[1]):
                        if attention_mask[b, tok_idx] == 1:
                            seq.append(BIO_LABELS[preds[b, tok_idx].item()])
                    all_preds.append(seq)

    return all_preds


def _predict_hde(model, loader, device, config) -> List[List[str]]:
    """Run inference for HDE; returns sentence-level label sequences."""
    model.eval()
    all_preds: List[List[str]] = []

    if config.model.label_scheme == "bio":
        id_to_label = {i: t for i, t in enumerate(BIO_LABELS)}
    else:
        id_to_label = ID2LABEL

    with torch.no_grad():
        for batch in loader:
            seg_ids = batch["segment_input_ids"].to(device)
            seg_mask = batch["segment_attention_mask"].to(device)
            sent_pos = batch["sent_positions"].to(device)
            sent_mask = batch["sentence_mask"].to(device)

            _, logits = model(
                segment_input_ids=seg_ids,
                segment_attention_mask=seg_mask,
                sent_positions=sent_pos,
                sentence_mask=sent_mask,
            )

            if config.model.use_crf:
                pred_seqs = logits
                for pred_seq, mask_row in zip(pred_seqs, sent_mask):
                    n = int(mask_row.sum().item())
                    all_preds.append([id_to_label[i] for i in pred_seq[:n]])
            else:
                preds = logits.argmax(-1)
                for b in range(preds.shape[0]):
                    n = int(sent_mask[b].sum().item())
                    seq = [id_to_label[preds[b, i].item()] for i in range(n)]
                    all_preds.append(seq)

    return all_preds


def evaluate(
    model,
    loader: DataLoader,
    device: torch.device,
    config: DiscourseConfig,
    ground_truth_labels: Optional[List[List[str]]] = None,
) -> Dict[str, float]:
    """
    Run inference and compute overlap F1.

    When ground_truth_labels is None, labels are extracted from the loader's
    dataset (assumes the dataset stores raw word/sentence label sequences).
    """
    if config.model.use_hde:
        predictions = _predict_hde(model, loader, device, config)
        label_scheme = config.model.label_scheme
    else:
        predictions = _predict_token_level(model, loader, device, config)
        label_scheme = "bio"

    if ground_truth_labels is None:
        ground_truth_labels = _extract_ground_truth(loader.dataset, config)

    return compute_overlap_f1(predictions, ground_truth_labels, label_scheme)


def _extract_ground_truth(
    dataset, config: DiscourseConfig
) -> List[List[str]]:
    """Extract word-level ground-truth label sequences from dataset."""
    truths = []
    for ex in dataset.examples:
        if config.model.use_hde:
            truths.append([])
        else:
            truths.append(ex["labels"])
    return truths


def print_results(results: Dict[str, float]) -> None:
    header = f"{'Class':<22} {'F1':>6}"
    print(header)
    print("-" * len(header))
    for dtype in DISCOURSE_TYPES:
        print(f"{dtype:<22} {results.get(dtype, 0.0):>6.1f}")
    print("-" * len(header))
    print(f"{'Overall':<22} {results.get('overall', 0.0):>6.1f}")

# Discourse Classification

This repository contains the code for:

**Segmenting and Classifying Discourse Elements in Students' Argumentative Essays** \
*Nghia Trung Ngo, Yongseok Soh, Hakyung Sung* \
University of Oregon, 2022

## Overview

We tackle **Argumentative Discourse Parsing (ADP)**: given a student essay, identify
and classify every discourse segment into one of seven types —
*Lead, Position, Claim, Counterclaim, Rebuttal, Evidence, Concluding Statement*.

The core limitation of standard BERT-based approaches is the 512-token input constraint,
which causes them to miss concluding statements that appear at the end of long essays.
We address this with a **Hierarchical Document Encoder (HDE)** that splits documents
into overlapping segments, encodes each with BERT, then classifies at the *sentence level*
using a lightweight transformer on top of extracted sentence embeddings.

![HDE architecture](docs/hde.png)

### Discourse element statistics (Kaggle Feedback Prize training set)

| Element | Token count | Proportion |
|---|---|---|
| Lead | 9,305 | 6.45% |
| Position | 15,419 | 10.69% |
| Claim | 50,208 | 34.80% |
| Counterclaim | 5,817 | 4.03% |
| Rebuttal | 4,337 | 3.01% |
| Evidence | 45,702 | 31.67% |
| Concluding Statement | 13,505 | 9.36% |

## Installation

```bash
conda create -n discourse python=3.9
conda activate discourse
pip install -r requirements.txt
```

## Data

Download the [Kaggle Feedback Prize 2021](https://www.kaggle.com/c/feedback-prize-2021/data)
dataset and place it in `data/`. See [`data/README.md`](data/README.md) for the expected
directory layout.

Split into train / dev / test (80 / 10 / 10):

```bash
python scripts/preprocess.py --csv data/train.csv --out_dir data/
```

## Training

### Baseline models (token-level sequence labeling)

```bash
# BERT
python train.py --config configs/bert_base.yaml

# BERT + CRF
python train.py --config configs/bert_crf.yaml

# BigBIRD
python train.py --config configs/bigbird_base.yaml

# BigBIRD + CRF
python train.py --config configs/bigbird_crf.yaml
```

Or run all baselines in sequence:

```bash
bash scripts/train_baseline.sh
```

### Proposed HDE models (sentence-level classification)

```bash
# HDE (category labels)
python train.py --config configs/hde_base.yaml

# HDE + CRF
python train.py --config configs/hde_crf.yaml

# HDE + Sentence Encoder
python train.py --config configs/hde_se.yaml

# HDE + Sentence Encoder + CRF
python train.py --config configs/hde_se_crf.yaml
```

Or run all HDE variants:

```bash
bash scripts/train_hde.sh
```

## Evaluation

```bash
python evaluate.py \
    --config configs/hde_se.yaml \
    --checkpoint checkpoints/hde_se/best_model.pt \
    --split test
```

### Baseline results

| Model | Lead | Position | Claim | Evidence | CClaim | Rebuttal | ConclStm | **Overall** |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| BERT | 76.7 | 61.6 | 45.4 | 55.8 | 33.7 | 23.9 | 54.4 | 50.2 |
| BERT+CRF | 77.3 | 60.4 | 46.6 | 58.1 | 32.6 | 23.6 | 57.6 | 50.9 |
| BigBIRD | 77.4 | 60.8 | 49.1 | 63.7 | 45.4 | 37.3 | 77.8 | 58.8 |
| BigBIRD+CRF | 77.3 | 63.3 | 51.0 | 65.5 | 46.1 | 40.4 | 80.2 | **60.5** |

### HDE results (sentence-level, category labels)

| Model | Lead | Position | Claim | Evidence | CClaim | Rebuttal | ConclStm | **Overall** |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| HDE | 85.4 | 73.1 | 61.4 | 87.3 | 46.2 | 34.1 | 62.9 | 80.0 |
| HDE+CRF | 84.7 | 74.0 | 58.4 | 87.5 | 39.7 | 29.2 | 60.6 | 80.0 |
| HDE+SE | **86.2** | **75.7** | 61.0 | **87.7** | **46.4** | **36.0** | **68.8** | **80.7** |
| HDE+SE+CRF | 85.9 | 75.4 | 59.7 | 87.6 | 40.7 | 27.3 | 66.6 | 80.5 |

F1 reported on the validation set using the Kaggle word-overlap criterion
(prediction counted correct when IoU with ground-truth span > 0.5).

## Architecture

### Baseline (token-level)

```
Essay tokens → BERT (512 or 1024 tokens) → Linear → [BIO labels]
                                          → CRF    → [BIO labels]
```

### HDE (sentence-level)

```
Essay
  └─ Sentence segmentation (NLTK)
  └─ Insert [SENT] marker at each sentence boundary
  └─ Split into 4 segments × 256 tokens
       ↓
  BERT (shared weights, per segment)
       ↓
  Gather [SENT] embeddings  →  sentence representations
       ↓  (optional)
  Sentence Encoder (2-layer transformer)
       ↓
  Linear → [category labels]   or   CRF → [category labels]
```

Key design choices:
- **4 segments × 256 tokens** covers up to 1,024 subword tokens per essay.
- Sentence-level labels (vs. BIO) are simpler for the hierarchical setting because
  ADP segments are always contiguous; no span separation ambiguity exists.
- CRF is beneficial for the *baseline* (longer prediction sequences) but not the HDE
  (max 100 sentence predictions per document).

## License

MIT

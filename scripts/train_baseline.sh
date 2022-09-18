#!/usr/bin/env bash
set -e

echo "=== Preprocessing ==="
python scripts/preprocess.py --csv data/train.csv --out_dir data/

echo "=== BERT baseline ==="
python train.py --config configs/bert_base.yaml

echo "=== BERT+CRF baseline ==="
python train.py --config configs/bert_crf.yaml

echo "=== BigBIRD baseline ==="
python train.py --config configs/bigbird_base.yaml

echo "=== BigBIRD+CRF baseline ==="
python train.py --config configs/bigbird_crf.yaml

echo "=== Evaluation ==="
for model in bert_base bert_crf bigbird_base bigbird_crf; do
    echo "--- $model ---"
    python evaluate.py \
        --config configs/${model}.yaml \
        --checkpoint checkpoints/${model}/best_model.pt \
        --split test
done

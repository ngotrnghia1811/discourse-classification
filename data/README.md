# Data

This project uses the **Kaggle Feedback Prize 2021** dataset, derived from the
[PERSUADE corpus](https://www.the-learning-agency-lab.com/the-feedback-prize/).

## Download

1. Accept the competition terms and download from the Kaggle page:
   https://www.kaggle.com/c/feedback-prize-2021/data

2. Place the downloaded files in this directory:

```
data/
├── train.csv          # Discourse annotations (id, predictionstring, discourse_type, …)
└── train/             # One .txt file per essay (named by essay id)
    ├── 0000D23A521A.txt
    ├── ...
```

## CSV schema

| Column | Description |
|---|---|
| `id` | Essay ID (matches filename in `train/`) |
| `discourse_id` | Unique ID for each annotated discourse span |
| `discourse_start` | Character start of span in essay |
| `discourse_end` | Character end of span in essay |
| `discourse_text` | Raw text of the span |
| `discourse_type` | One of: Lead, Position, Claim, Counterclaim, Rebuttal, Evidence, Concluding Statement |
| `predictionstring` | Space-separated word indices covered by the span |

## Preprocessing

Run the split script to create train/dev/test partitions (80/10/10):

```bash
python scripts/preprocess.py --csv data/train.csv --out_dir data/
```

This produces `data/train.csv`, `data/dev.csv`, and `data/test.csv` with the
same schema, filtered to the corresponding essay IDs.

## License

The dataset is provided under the Kaggle competition terms.
See https://www.kaggle.com/c/feedback-prize-2021/rules for details.

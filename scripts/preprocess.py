"""
Split the raw Kaggle train.csv into train/dev/test (80/10/10).

Usage:
    python scripts/preprocess.py --csv data/train.csv --out_dir data/
"""
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/train.csv")
    parser.add_argument("--out_dir", default="data/")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    essay_ids = df["id"].unique().tolist()

    train_ids, tmp_ids = train_test_split(essay_ids, test_size=0.2, random_state=args.seed)
    dev_ids, test_ids = train_test_split(tmp_ids, test_size=0.5, random_state=args.seed)

    for split_name, ids in [("train", train_ids), ("dev", dev_ids), ("test", test_ids)]:
        split_df = df[df["id"].isin(ids)]
        out_path = f"{args.out_dir}/{split_name}.csv"
        split_df.to_csv(out_path, index=False)
        print(f"{split_name}: {len(ids)} essays → {out_path}")

if __name__ == "__main__":
    main()

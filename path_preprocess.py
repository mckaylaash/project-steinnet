"""Preprocess path navigation labels for training.

This script:
1) Validates image paths from path_labels.csv
2) Keeps only known actions: forward/left/backward/right
3) Encodes actions to integer class IDs
4) Writes cleaned full/train/val CSVs for training
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split


INPUT_CSV = "path_labels.csv"
IMAGE_DIR = "path_training_data"
OUTPUT_DIR = "path_processed"
FULL_CSV = os.path.join(OUTPUT_DIR, "path_labels_full.csv")
TRAIN_CSV = os.path.join(OUTPUT_DIR, "path_labels_train.csv")
VAL_CSV = os.path.join(OUTPUT_DIR, "path_labels_val.csv")

ACTION_TO_ID = {
    "forward": 0,
    "left": 1,
    "backward": 2,
    "right": 3,
}


def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: missing {INPUT_CSV}. Run path_data_collection.py first.")
        return
    if not os.path.isdir(IMAGE_DIR):
        print(f"Error: missing directory {IMAGE_DIR}.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(INPUT_CSV)

    required_cols = {"filename", "action"}
    if not required_cols.issubset(df.columns):
        print(f"Error: {INPUT_CSV} must include columns: {sorted(required_cols)}")
        return

    df = df.copy()
    df["action"] = df["action"].astype(str).str.lower().str.strip()
    df = df[df["action"].isin(ACTION_TO_ID.keys())]
    df["image_path"] = df["filename"].astype(str).map(lambda name: os.path.join(IMAGE_DIR, name))
    df = df[df["image_path"].map(os.path.exists)]

    if df.empty:
        print("Error: no valid labeled images found after filtering.")
        return

    df["label_id"] = df["action"].map(ACTION_TO_ID).astype(int)
    df = df[["filename", "action", "label_id"]].reset_index(drop=True)

    # Keep class balance in split when possible.
    stratify = df["label_id"] if df["label_id"].nunique() > 1 else None
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=stratify)

    df.to_csv(FULL_CSV, index=False)
    train_df.to_csv(TRAIN_CSV, index=False)
    val_df.to_csv(VAL_CSV, index=False)

    print(f"Saved cleaned dataset: {FULL_CSV} ({len(df)} samples)")
    print(f"Saved train split: {TRAIN_CSV} ({len(train_df)} samples)")
    print(f"Saved val split: {VAL_CSV} ({len(val_df)} samples)")
    print("Class counts:")
    print(df["action"].value_counts().to_string())


if __name__ == "__main__":
    main()

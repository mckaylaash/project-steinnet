# This file preprocesses the images for the navigation/path finding task

import os
import pandas as pd
from sklearn.model_selection import train_test_split


input_csv = "path_labels.csv"
image_dir = "path_training_data"
output_dir = "path_processed"
full_csv = os.path.join(output_dir, "path_labels_full.csv")
train_csv = os.path.join(output_dir, "path_labels_train.csv")
val_csv = os.path.join(output_dir, "path_labels_val.csv")

actions_id = {
    "forward": 0,
    "left": 1,
    "backward": 2,
    "right": 3,
}


def main():
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_csv)

    required_cols = {"filename", "action"}

    df = df.copy()
    # strips action type from name
    df["action"] = df["action"].astype(str).str.lower().str.strip()
    df = df[df["action"].isin(actions_id.keys())]
    df["image_path"] = df["filename"].astype(str).map(lambda name: os.path.join(image_dir, name))
    df = df[df["image_path"].map(os.path.exists)]

    # strips label id from action and maps to integer for training
    df["label_id"] = df["action"].map(actions_id).astype(int)
    df = df[["filename", "action", "label_id"]].reset_index(drop=True)

    # keep class balance in split when possible.
    stratify = df["label_id"] if df["label_id"].nunique() > 1 else None
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=stratify)

    df.to_csv(full_csv, index=False)
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    print(f"Saved cleaned dataset: {full_csv} ({len(df)} samples)")
    print(f"Saved train split: {train_csv} ({len(train_df)} samples)")
    print(f"Saved val split: {val_csv} ({len(val_df)} samples)")
    print("Class counts:")
    print(df["action"].value_counts().to_string())


if __name__ == "__main__":
    main()

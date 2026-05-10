"""Evaluate SteinNet tree-click regression model."""

import math
import os

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms


LABELS_CSV = "labels.csv"
IMAGE_DIR = "processed_data"
WEIGHTS_PATH = "weights/stein_net_best.pth"
IMG_W = 1280.0
IMG_H = 720.0


def pixel_error(row):
    dx = (row["pred_x"] - row["x"]) * IMG_W
    dy = (row["pred_y"] - row["y"]) * IMG_H
    return math.sqrt(dx * dx + dy * dy)


def main():
    if not os.path.exists(LABELS_CSV):
        print(f"Error: missing {LABELS_CSV}.")
        return
    if not os.path.exists(WEIGHTS_PATH):
        print(f"Error: missing {WEIGHTS_PATH}.")
        return

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    full_df = pd.read_csv(LABELS_CSV)
    _, val_df = train_test_split(full_df, test_size=0.2, random_state=42)
    val_df = val_df.reset_index(drop=True)

    model = torch.load(WEIGHTS_PATH, map_location=device, weights_only=False)
    model.to(device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    preds = []
    with torch.no_grad():
        for _, row in val_df.iterrows():
            img_path = os.path.join(IMAGE_DIR, row["filename"])
            if not os.path.exists(img_path):
                continue
            image = Image.open(img_path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(device)
            output = model(input_tensor)[0].detach().cpu().numpy()
            pred_x = float(max(0.0, min(1.0, output[0])))
            pred_y = float(max(0.0, min(1.0, output[1])))
            preds.append((pred_x, pred_y))

    if not preds:
        print("Error: no validation predictions were produced.")
        return

    eval_df = val_df.iloc[: len(preds)].copy()
    eval_df["pred_x"] = [p[0] for p in preds]
    eval_df["pred_y"] = [p[1] for p in preds]
    eval_df["mse_xy"] = (
        (eval_df["pred_x"] - eval_df["x"]) ** 2 + (eval_df["pred_y"] - eval_df["y"]) ** 2
    ) / 2.0
    eval_df["pixel_error"] = eval_df.apply(pixel_error, axis=1)

    mean_px = float(eval_df["pixel_error"].mean())
    median_px = float(eval_df["pixel_error"].median())
    p90_px = float(eval_df["pixel_error"].quantile(0.90))
    p95_px = float(eval_df["pixel_error"].quantile(0.95))
    mse = float(eval_df["mse_xy"].mean())

    under_15 = float((eval_df["pixel_error"] < 15.0).mean() * 100.0)
    under_25 = float((eval_df["pixel_error"] < 25.0).mean() * 100.0)
    under_40 = float((eval_df["pixel_error"] < 40.0).mean() * 100.0)

    print("=== Tree Regression Evaluation ===")
    print(f"Validation samples: {len(eval_df)}")
    print(f"MSE (normalized xy): {mse:.6f}")
    print(f"Pixel Error Mean:   {mean_px:.2f}")
    print(f"Pixel Error Median: {median_px:.2f}")
    print(f"Pixel Error P90:    {p90_px:.2f}")
    print(f"Pixel Error P95:    {p95_px:.2f}")
    print(f"% under 15 px: {under_15:.2f}%")
    print(f"% under 25 px: {under_25:.2f}%")
    print(f"% under 40 px: {under_40:.2f}%")


if __name__ == "__main__":
    main()

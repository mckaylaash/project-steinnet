"""Evaluate path navigation classifier."""

import os

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


VAL_CSV = "path_processed/path_labels_val.csv"
IMAGE_DIR = "path_training_data"
WEIGHTS_PATH = "weights/path_nav_net_best.pth"


def safe_div(num, den):
    return float(num) / float(den) if den else 0.0


def main():
    if not os.path.exists(VAL_CSV):
        print(f"Error: missing {VAL_CSV}. Run path_preprocess.py first.")
        return
    if not os.path.exists(WEIGHTS_PATH):
        print(f"Error: missing {WEIGHTS_PATH}. Run path_train.py first.")
        return

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    payload = torch.load(WEIGHTS_PATH, map_location=device)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        print("Error: nav weights file format is invalid.")
        return

    id_to_action = payload.get("id_to_action", {0: "forward", 1: "left", 2: "backward", 3: "right"})
    id_to_action = {int(k): str(v) for k, v in id_to_action.items()}
    num_classes = int(payload.get("num_classes", len(id_to_action)))

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()

    val_df = pd.read_csv(VAL_CSV).reset_index(drop=True)
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    y_true = []
    y_pred = []

    with torch.no_grad():
        for _, row in val_df.iterrows():
            img_path = os.path.join(IMAGE_DIR, row["filename"])
            if not os.path.exists(img_path):
                continue
            image = Image.open(img_path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(device)
            logits = model(input_tensor)
            pred = int(torch.argmax(logits, dim=1).item())
            true = int(row["label_id"])
            y_true.append(true)
            y_pred.append(pred)

    if not y_true:
        print("Error: no validation predictions were produced.")
        return

    labels = sorted(set(y_true) | set(y_pred))
    confusion = {(t, p): 0 for t in labels for p in labels}
    for t, p in zip(y_true, y_pred):
        confusion[(t, p)] += 1

    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = safe_div(correct, len(y_true))

    print("=== Nav Classification Evaluation ===")
    print(f"Validation samples: {len(y_true)}")
    print(f"Accuracy: {accuracy:.4f}")
    print("")
    print("Per-class metrics:")
    for cls in labels:
        tp = confusion[(cls, cls)]
        fp = sum(confusion[(t, cls)] for t in labels if t != cls)
        fn = sum(confusion[(cls, p)] for p in labels if p != cls)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0
        action_name = id_to_action.get(cls, str(cls))
        print(
            f"- {action_name:9s} "
            f"precision={precision:.3f} recall={recall:.3f} f1={f1:.3f} support={tp + fn}"
        )

    print("")
    print("Confusion matrix (rows=true, cols=pred):")
    header = " " * 12 + " ".join([f"{id_to_action.get(c, c):>10s}" for c in labels])
    print(header)
    for t in labels:
        row_name = id_to_action.get(t, str(t))
        counts = " ".join([f"{confusion[(t, p)]:10d}" for p in labels])
        print(f"{row_name:>12s} {counts}")


if __name__ == "__main__":
    main()

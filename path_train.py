"""Train a path navigation classifier from path dataset splits."""

import os
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


TRAIN_CSV = "path_processed/path_labels_train.csv"
VAL_CSV = "path_processed/path_labels_val.csv"
IMAGE_DIR = "path_training_data"
WEIGHTS_PATH = "weights/path_nav_net_best.pth"

NUM_CLASSES = 4
BATCH_SIZE = 32
NUM_EPOCHS = 12
LR = 1e-3

ID_TO_ACTION = {
    0: "forward",
    1: "left",
    2: "backward",
    3: "right",
}


class PathDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["filename"])
        image = Image.open(img_path).convert("RGB")
        label = torch.tensor(int(row["label_id"]), dtype=torch.long)
        if self.transform:
            image = self.transform(image)
        return image, label


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / max(len(dataloader), 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def main():
    if not os.path.exists(TRAIN_CSV) or not os.path.exists(VAL_CSV):
        print("Error: missing path split CSVs.")
        print("Run: python path_preprocess.py")
        return

    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    if train_df.empty or val_df.empty:
        print("Error: train/val split is empty.")
        return

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_loader = DataLoader(
        PathDataset(train_df, IMAGE_DIR, transform),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        PathDataset(val_df, IMAGE_DIR, transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = models.resnet18(weights="IMAGENET1K_V1")
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"Training path nav model on {device}...")
    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")

    best_val_acc = -1.0
    os.makedirs("weights", exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / max(len(train_loader), 1)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} "
            f"- train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            payload = {
                "model_state_dict": model.state_dict(),
                "id_to_action": ID_TO_ACTION,
                "num_classes": NUM_CLASSES,
            }
            torch.save(payload, WEIGHTS_PATH)
            print(f"Saved new best model to {WEIGHTS_PATH} (val_acc={val_acc:.3f})")

    print(f"Done. Best validation accuracy: {best_val_acc:.3f}")


if __name__ == "__main__":
    main()

# This file is responsible for training the SteinNet model on the preprocessed dataset for the navigation task.
# It loads the images and labels from the CSV, applies necessary transformations,
# and trains a ResNet-18 regression model to predict the normalized coordinates for the navigation/path task.

import os
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


train_csv = "path_processed/path_labels_train.csv"
val_csv = "path_processed/path_labels_val.csv"
image_dir = "path_training_data"
weights_path = "weights/path_nav_net_best.pth"

num_classes = 4
batch_size = 32
num_epochs = 12
learning_rate= 1e-3

action_id = {
    0: "forward",
    1: "left",
    2: "backward",
    3: "right",
}


# custom dataset class to load images and labels for nav task
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

# evaluates the model on the validation set and returns average loss and accuracy 
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


# main training function that loads data and model and runs training loop
def main():

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

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
        PathDataset(train_df, image_dir, transform),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        PathDataset(val_df, image_dir, transform),
        batch_size=batch_size,
        shuffle=False,
    )

    # load in model
    model = models.resnet18(weights="IMAGENET1K_V1")
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model.to(device)

    # use cross-entropy loss since classification task and Adam optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_acc = -1.0
    os.makedirs("weights", exist_ok=True)

    for epoch in range(num_epochs):
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
            f"Epoch {epoch + 1}/{num_epochs} "
            f"- train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            payload = {
                "model_state_dict": model.state_dict(),
                "id_to_action": action_id,
                "num_classes": num_classes,
            }
            torch.save(payload, weights_path)

if __name__ == "__main__":
    main()

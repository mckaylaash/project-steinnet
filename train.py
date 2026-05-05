import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from PIL import Image  # FIX 1: Added this import
import pandas as pd
import os

# --- Configuration & Device Setup ---
# FIX 2: Define the device (MPS for Mac, CUDA for NVIDIA, or CPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 1. Dataset Class
class SteinDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.df = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        '''
        Loads an image and its corresponding normalized coordinates from the Master Label CSV.
        The image is transformed using the provided transformations (e.g., normalization for ResNet-18). The label is returned as a tensor of shape (2,) containing the normalized x and y coordinates
        '''
        img_name = os.path.join(self.img_dir, self.df.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        # Normalized coordinates from our Master Label CSV
        label = torch.tensor([self.df.iloc[idx, 1], self.df.iloc[idx, 2]], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)
        return image, label

# 2. Pixel Normalization (ImageNet standards for ResNet-18)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. Train/Validation Split (80/20)
if not os.path.exists('labels.csv'):
    print("Error: Run preprocess.py first to generate labels.csv!")
else:
    full_df = pd.read_csv('labels.csv')
    train_df, val_df = train_test_split(full_df, test_size=0.2, random_state=42)

    train_loader = DataLoader(SteinDataset(train_df, 'processed_data', transform), batch_size=32, shuffle=True)
    val_loader = DataLoader(SteinDataset(val_df, 'processed_data', transform), batch_size=32)

    # 4. Build the Model (ResNet-18 Regression)
    model = models.resnet18(weights='IMAGENET1K_V1')
    num_ftrs = model.fc.in_features
    # Replace the final classification layer with a Linear Regression Head
    model.fc = nn.Linear(num_ftrs, 2) 
    model.to(device)

    # 5. Loss and Optimizer[cite: 1]
    criterion = nn.MSELoss() # Defined in ANN Proposal.pdf[cite: 1]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(f"Ready! Training on {len(train_df)} samples, Validating on {len(val_df)}.")

    # --- Training Loop ---
    num_epochs = 10 
    print("Starting Training...")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels) # Calculate MSE[cite: 1]
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {running_loss/len(train_loader):.4f}")

    # --- Save the Weights ---
    os.makedirs('weights', exist_ok=True)
    torch.save(model, 'weights/stein_net_best.pth')
    print("Weights saved! Your agent now has a brain.")
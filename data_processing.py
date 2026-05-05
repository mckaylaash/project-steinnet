import os
import cv2
import numpy as np
from PIL import Image

# Directories
input_dir = 'training_data'
output_dir = 'processed_data'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Target size from your proposal
TARGET_SIZE = (224, 224)

print(f"Processing images from {input_dir}...")

for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # 1. Load the image
        img_path = os.path.join(input_dir, filename)
        img = Image.open(img_path).convert('RGB')
        
        # 2. Resize to 224x224
        img_resized = img.resize(TARGET_SIZE, Image.BILINEAR)
        
        # 3. Convert to array and Normalize (0 to 1 range)
        # This helps the Adam optimizer converge faster
        img_array = np.array(img_resized) / 255.0
        
        # 4. Save the processed version
        # Note: We save back as an image for visualization, 
        # but the /255 normalization usually happens live in the Training Loop.
        processed_filename = os.path.join(output_dir, f"proc_{filename}")
        img_resized.save(processed_filename)

print(f"Done! Processed images are in {output_dir}")
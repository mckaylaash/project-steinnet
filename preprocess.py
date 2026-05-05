import os
import pandas as pd
from PIL import Image

# Configuration from your proposal
RAW_DIR = 'training_data'
PROC_DIR = 'processed_data'
IMG_SIZE = (224, 224)
ORIG_W, ORIG_H = 1280, 720 # Steinworld raw resolution

if not os.path.exists(PROC_DIR):
    os.makedirs(PROC_DIR)

data_log = []

print("Starting normalization and resizing...")

for filename in os.listdir(RAW_DIR):
    if filename.endswith(".png"):
        # 1. Parse coordinates from filename: target_X_Y_uuid.png
        parts = filename.split('_')
        pixel_x = int(parts[1])
        pixel_y = int(parts[2])
        
        # 2. Coordinate Normalization (0.0 to 1.0)
        norm_x = pixel_x / ORIG_W
        norm_y = pixel_y / ORIG_H
        
        # 3. Image Resizing
        img_path = os.path.join(RAW_DIR, filename)
        img = Image.open(img_path).convert('RGB')
        img_resized = img.resize(IMG_SIZE, Image.BILINEAR)
        
        proc_filename = f"proc_{filename}"
        img_resized.save(os.path.join(PROC_DIR, proc_filename))
        
        # 4. Log for the Master Label file
        data_log.append({
            'filename': proc_filename,
            'x': norm_x,
            'y': norm_y
        })

# Save the Master Label file
df = pd.DataFrame(data_log)
df.to_csv('labels.csv', index=False)
print(f"Success! Created labels.csv and processed {len(data_log)} images.")
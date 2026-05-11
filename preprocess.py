# This file is responsible for preprocessing the raw screenshots and generating 
# the labels CSV (labels.csv) that contains the normalized coordinates for each image for 
# the chopping task. 

import os
import pandas as pd
from PIL import Image

raw_dataset = 'training_data'
processed_dataset = 'processed_data'
size = (224, 224)

if not os.path.exists(processed_dataset):
    os.makedirs(processed_dataset           )

data_log = []
skipped = 0

for filename in os.listdir(raw_dataset):
    
    if filename.endswith(".png"):
        # split based on naming convention
        parts = filename.split('_')
        pixel_x = int(parts[1])
        pixel_y = int(parts[2])
        
        # open source image and normalize
        img_path = os.path.join(raw_dataset, filename)
        img = Image.open(img_path).convert('RGB')
        img_w, img_h = img.size
        if img_w <= 0 or img_h <= 0:
            skipped += 1
            continue

        norm_x = pixel_x / img_w
        norm_y = pixel_y / img_h

        # resiize and save image
        img_resized = img.resize(size, Image.BILINEAR)
        
        proc_filename = f"proc_{filename}"
        img_resized.save(os.path.join(processed_dataset, proc_filename))
        
        # log into labels CSV
        data_log.append({
            'filename': proc_filename,
            'x': norm_x,
            'y': norm_y
        })

# Save the labels CSV
df = pd.DataFrame(data_log)
df.to_csv('labels.csv', index=False)
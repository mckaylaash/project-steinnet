import os

# 1. Get the directory where count.py lives (Project-SteinNet/scripts/)
script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Go up one level to the project root (Project-SteinNet/)
project_root = os.path.dirname(script_dir)

# 3. Now point to the data folder from the root
folder_path = os.path.join(project_root, 'data/training_data')

# List of common image extensions
image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

# Filter and count
image_count = len([f for f in os.listdir(folder_path) 
                   if f.lower().endswith(image_extensions)])

print(f"Number of images in {folder_path}: {image_count}")
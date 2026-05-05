import os

# Define your folder
folder_path = 'training_data'

# List of common image extensions
image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

# Filter and count
image_count = len([f for f in os.listdir(folder_path) 
                   if f.lower().endswith(image_extensions)])

print(f"Number of images in {folder_path}: {image_count}")
"""
This script processes a set of card images from specified source folders, 
removes their backgrounds, and saves the processed images into a destination folder.
It uses the rembg library to remove backgrounds and PIL for image handling.

Each card type (Ecke, Herz, Kreuz, Schaufel) is processed separately, and the output
is saved with a new filename that includes the card type and a sequence number.
"""

import os
import glob
from rembg import remove
from PIL import Image

# Paths of the source folders
folders = ['Data/Raw/original/Ecke', 'Data/Raw/original/Herz', 'Data/Raw/original/Kreuz', 'Data/Raw/original/Schaufel']

# Path of the destination folder
dest_folder = 'Data/Processed/no_background'

# Ensure the destination folder exists
if not os.path.exists(dest_folder):
    os.makedirs(dest_folder)

# Process each folder
for folder in folders:
    # Extract the card name from the folder name
    card_name = os.path.basename(folder)[0]
    
    # List all JPEG images in the current folder
    for file_num, file_path in enumerate(glob.glob(os.path.join(folder, '*.JPG')), start=1):
        # Read the image
        input_image = Image.open(file_path)

        # Remove the background
        output_image = remove(input_image)

        # Convert to 'RGB' to save as JPEG
        output_rgb = output_image.convert("RGB")

        # Construct the output file name
        output_file_name = f"{card_name}_{file_num-1}.jpg"
        output_file_path = os.path.join(dest_folder, output_file_name)

        # Save the image
        output_rgb.save(output_file_path)

        print(f"Processed {output_file_name}")

print("All images have been processed.")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File name: card_image_background_remover.py
Author: begep1 & rothl18
Date created: 02.12.2023
Date last modified: 08.01.2023
Python Version: 3.11.6

Description:
This script processes playing card images, removing backgrounds using 'rembg' and saving them 
in a specified folder. It supports different card types (Ecke, Herz, Kreuz, Schaufel), each 
processed in separate subfolders. The script is ideal for preparing card images for machine 
learning tasks or other image processing applications.

Usage:
Set the source ('Data/Raw/original/{Card Suit}') and destination ('Data/Processed/no_background') 
folder paths. The script handles JPEG image processing, background removal, and saving the 
output in an organized format.
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
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File name: image_transformation_processor.py
Author: begep1 & rothl18
Date created: 02.12.2023
Date last modified: 08.01.2023
Python Version: 3.11.6

Description:
This script applies various image transformation techniques to a set of images.
These transformations include rotation, brightness adjustment, random cropping,
noise addition, scaling, squeezing, color jittering, blurring, and shearing.
The transformed images are saved in a specified destination folder, which is created
if it does not exist. The script is designed to augment a dataset for machine learning
purposes, increasing the diversity of the dataset by generating altered versions of
the original images.

Usage:
Ensure the source folder path contains the images to be transformed and the destination
folder path is set for saving the processed images. The number of images generated from
each original image can be adjusted via the NUM_GENERATED_IMAGES constant.
"""

import os
import glob
import random
import numpy as np
import re
from PIL import Image, ImageEnhance, ImageFilter

# Constants for image transformation
NOISE_INTENSITY_LOW = 60
NOISE_INTENSITY_HIGH = 150
BRIGHTNESS_LOW = 3
BRIGHTNESS_HIGH = 5
NUM_GENERATED_IMAGES = 100

def crop_center_square(image: Image.Image) -> Image.Image:
    """
    Crops the largest possible square from the center of the image.
    
    Parameters:
    image (Image.Image): The original image to be cropped.

    Returns:
    Image.Image: The cropped square image.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("The provided input is not a PIL Image object.")

    width, height = image.size
    new_side = min(width, height)
    left = (width - new_side) // 2
    top = (height - new_side) // 2
    right = left + new_side
    bottom = top + new_side
    
    return image.crop((left, top, right, bottom))

def make_square(image: Image.Image, background_color: tuple = (255, 255, 255)) -> Image.Image:
    """
    Converts an image to a square by padding it with a specified background color.
    
    Parameters:
    image (Image.Image): The original image to be converted to square.
    background_color (tuple): The RGB color value for the background.

    Returns:
    Image.Image: The square image with the original image centered.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("The provided input is not a PIL Image object.")
    if not isinstance(background_color, tuple) or not all(isinstance(i, int) and 0 <= i <= 255 for i in background_color):
        raise ValueError("Background color must be a tuple with three integers between 0 and 255.")

    width, height = image.size
    new_size = max(width, height)
    new_image = Image.new('RGB', (new_size, new_size), background_color)
    paste_coords = ((new_size - width) // 2, (new_size - height) // 2)

    new_image.paste(image, paste_coords)
    return new_image

def random_rotation(image: Image.Image) -> Image.Image:
    """
    Rotates an image by a random angle between -180 and 180 degrees.

    Parameters:
    image (Image.Image): The original image to be rotated.

    Returns:
    Image.Image: The rotated image.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("The provided input is not a PIL Image object.")

    angle = random.randint(-180, 180)
    return image.rotate(angle, expand=True)

def random_brightness(image: Image.Image, low: float = 0.5, high: float = 1.8) -> Image.Image:
    """
    Adjusts the brightness of an image by a random factor.

    Parameters:
    image (Image.Image): The original image to be adjusted.
    low (float): The lower bound for the random brightness factor.
    high (float): The upper bound for the random brightness factor.

    Returns:
    Image.Image: The brightness-adjusted image.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("The provided input is not a PIL Image object.")

    factor = random.uniform(low, high)
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def random_crop(image: Image.Image) -> Image.Image:
    """
    Crops a random region of the image with a size between 90% to 100% of the original size.

    Parameters:
    image (Image.Image): The original image to be cropped.

    Returns:
    Image.Image: The randomly cropped image.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("The provided input is not a PIL Image object.")

    width, height = image.size
    crop_width, crop_height = int(random.uniform(0.9, 1.0) * width), int(random.uniform(0.9, 1.0) * height)
    left = random.randint(0, width - crop_width)
    top = random.randint(0, height - crop_height)
    return image.crop((left, top, left + crop_width, top + crop_height))

def add_random_noise(image: Image.Image, low: int = 0, high: int = 100) -> Image.Image:
    """
    Adds Gaussian noise to an image.

    Parameters:
    image (Image.Image): The original image to be noised.
    low (int): The lower bound for noise generation.
    high (int): The upper bound for noise generation.

    Returns:
    Image.Image: The noised image.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("The provided input is not a PIL Image object.")

    np_image = np.array(image)
    noise = np.random.normal(low, high, np_image.shape)
    np_image = np.clip(np_image + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(np_image)

def random_scaling(image: Image.Image) -> Image.Image:
    """
    Scales an image by a random factor between 0.5 and 1.5.

    Parameters:
    image (Image.Image): The original image to be scaled.

    Returns:
    Image.Image: The scaled image.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("The provided input is not a PIL Image object.")

    factor = random.uniform(0.5, 1.5)  # More appropriate scaling factor
    width, height = image.size
    return image.resize((int(width * factor), int(height * factor)), Image.Resampling.LANCZOS)

def color_jitter(image: Image.Image) -> Image.Image:
    """
    Applies random color jitter in terms of brightness, contrast, and saturation.

    Parameters:
    image (Image.Image): The original image to apply color jitter.

    Returns:
    Image.Image: The color-jittered image.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("The provided input is not a PIL Image object.")

    # Define transformation functions with more controlled randomness
    color_transforms = [
        lambda img: ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2)),
        lambda img: ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2)),
        lambda img: ImageEnhance.Color(img).enhance(random.uniform(0.8, 1.2))
    ]
    random.shuffle(color_transforms)
    for func in color_transforms:
        image = func(image)
    return image

def random_blur(image: Image.Image) -> Image.Image:
    """
    Applies a random Gaussian blur to the image.

    Parameters:
    image (Image.Image): The original image to be blurred.

    Returns:
    Image.Image: The blurred image.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("The provided input is not a PIL Image object.")

    blur_radius = random.uniform(0.5, 5)  # More appropriate range for blur radius
    return image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

def random_shear(image: Image.Image) -> Image.Image:
    """
    Applies a random shear transformation to the image.

    Parameters:
    image (Image.Image): The original image to be sheared.

    Returns:
    Image.Image: The sheared image.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("The provided input is not a PIL Image object.")

    shear_factor = random.uniform(-0.5, 0.5)  # Reduced shear factor for more natural transformation
    return image.transform(image.size, Image.AFFINE, (1, shear_factor, 0, 0, 1, shear_factor))

def random_squeeze(image: Image.Image) -> Image.Image:
    """
    Applies a random squeeze (scaling) either horizontally or vertically.

    Parameters:
    image (Image.Image): The original image to be squeezed.

    Returns:
    Image.Image: The squeezed image.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("The provided input is not a PIL Image object.")

    squeeze_horizontal = random.choice([True, False])
    factor = random.uniform(0.5, 1.5)  # Adjusted scale factor for more natural deformation

    width, height = image.size
    new_size = (int(width * factor), height) if squeeze_horizontal else (width, int(height * factor))

    return image.resize(new_size, Image.Resampling.LANCZOS)

def get_last_file_number(base_name: str, dest_folder: str) -> int:
    """
    Finds the highest file number for a given base name in the destination folder.

    Parameters:
    base_name (str): The base name of the files to search for.
    dest_folder (str): The folder where the files are located.

    Returns:
    int: The highest file number found, or -1 if no files are found.
    """
    if not os.path.exists(dest_folder):
        raise FileNotFoundError(f"The specified folder {dest_folder} does not exist.")

    pattern = re.compile(rf"{re.escape(base_name)}_(\d+).jpg")
    max_num = -1
    for file in os.listdir(dest_folder):
        match = pattern.match(file)
        if match:
            num = int(match.group(1))
            max_num = max(max_num, num)
    return max_num

def process_images(folder: str, dest_folder: str, num_images_to_generate: int = 1) -> None:
    """
    Processes images in the specified folder by applying random transformations 
    and saving the transformed images in the destination folder.

    Parameters:
    folder (str): The source folder containing images to process.
    dest_folder (str): The destination folder to save transformed images.
    num_images_to_generate (int): Number of transformed images to generate per original image.
    """
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Source folder '{folder}' does not exist.")
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    all_transformations = {
        'rotate': random_rotation,
        'brightness': random_brightness,
        'crop': random_crop,
        'noise': add_random_noise,
        'scaling': random_scaling,
        'squeeze': random_squeeze,
        'color_jitter': color_jitter,
        'blur': random_blur,
        'shear': random_shear
    }
    transformation_items = list(all_transformations.items())

    for file_path in glob.glob(os.path.join(folder, '*.jpg')):
        original_image = Image.open(file_path)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        file_num = get_last_file_number(base_name, dest_folder) + 1

        for _ in range(num_images_to_generate):
            transformed_image = apply_transformations(original_image, transformation_items)
            transformed_image = crop_center_square(transformed_image)

            output_file_name = f"{base_name}_{file_num}.jpg"
            output_file_path = os.path.join(dest_folder, output_file_name)
            transformed_image.save(output_file_path)

            file_num += 1
            print(f"Processed {output_file_name} for base image {base_name}")

def apply_transformations(image: Image.Image, transformation_items: list) -> Image.Image:
    """
    Applies a random set of transformations to an image.

    Parameters:
    image (Image.Image): The original image to transform.
    transformation_items (list): List of transformation functions and their names.

    Returns:
    Image.Image: The transformed image.
    """
    transformed_image = image.copy()
    for _, trans_func in random.sample(transformation_items, random.randint(1, len(transformation_items))):
        transformed_image = trans_func(transformed_image)
    return transformed_image

def main():
    """
    Main function to initiate the image processing workflow.
    Processes images from a source folder, applying transformations,
    and saving them in a destination folder.
    """
    source_folder = 'Data/Processed/no_background'
    dest_folder = 'Data/Processed/database'
    num_generated_images = 100  # Define the number of images to generate from each original image

    # Check if the source folder exists
    if not os.path.exists(source_folder):
        raise FileNotFoundError(f"Source folder '{source_folder}' does not exist. Please check the path.")

    # Create the destination folder if it doesn't exist
    os.makedirs(dest_folder, exist_ok=True)

    # Start the image processing
    process_images(source_folder, dest_folder, num_generated_images)
    print("Image processing completed.")

if __name__ == "__main__":
    main()
import os
import glob
import random
import numpy as np
import re
from PIL import Image, ImageEnhance, ImageOps, ImageFilter

def random_rotation(image):
    return image.rotate(random.randint(-180, 180))  # Rotates by up to ±30 degrees

def random_brightness(image):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(random.uniform(0.5, 1.8))  # Adjust brightness

def random_crop(image):
    width, height = image.size
    new_width, new_height = int(0.9 * width), int(0.9 * height)  # Crop size
    left = random.randint(0, width - new_width)
    top = random.randint(0, height - new_height)
    return image.crop((left, top, left + new_width, top + new_height))

def add_random_noise(image):
    np_image = np.array(image)
    noise = np.random.normal(0, 75, np_image.shape)  # Gaussian noise
    np_image = np.clip(np_image + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(np_image)

def random_scaling(image):
    factor = random.uniform(0.9, 1.1)  # Scale between 90% and 110%
    width, height = image.size
    return image.resize((int(width * factor), int(height * factor)))

def slight_rotation(image):
    return image.rotate(random.uniform(-5, 5))  # Rotate by up to ±5 degrees

def color_jitter(image):
    color_transforms = [
        lambda img: ImageEnhance.Brightness(img).enhance(random.uniform(0.9, 1.1)),
        lambda img: ImageEnhance.Contrast(img).enhance(random.uniform(0.9, 1.1)),
        lambda img: ImageEnhance.Color(img).enhance(random.uniform(0.9, 1.1))
    ]
    random.shuffle(color_transforms)
    for func in color_transforms:
        image = func(image)
    return image

def random_blur(image):
    return image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1.5)))


def random_shear(image):
    shear_factor = random.uniform(-2, 2)
    return image.transform(image.size, Image.AFFINE, (1, shear_factor, 0, 0, 1, 0))


def get_last_file_number(base_name, dest_folder):
    pattern = re.compile(rf"{re.escape(base_name)}_(\d+).jpg")
    max_num = -1
    for file in os.listdir(dest_folder):
        match = pattern.match(file)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num
    return max_num

def process_images(folder, dest_folder, num_images_to_generate=1):
    for file_path in glob.glob(os.path.join(folder, '*.jpg')):
        original_image = Image.open(file_path)
        base_name = os.path.splitext(os.path.basename(file_path))[0]

        # Start numbering from the last file number for this base image
        file_num = get_last_file_number(base_name, dest_folder) + 1

        # List of all possible transformations
        all_transformations = {
            'rotate': random_rotation,
            'brightness': random_brightness,
            'crop': random_crop,
            'noise': add_random_noise,
            'scaling': random_scaling,
            'slight_rotation': slight_rotation,
            'color_jitter': color_jitter,
            'blur': random_blur,
            'shear': random_shear
        }

        # Convert dictionary items to a list
        transformation_items = list(all_transformations.items())
        
        # Generate a specified number of images per original image
        for _ in range(num_images_to_generate):
            transformed_image = original_image.copy()  # Create a copy to apply transformations
            # Apply transformations randomly
            for trans_name, trans_func in random.sample(transformation_items, random.randint(1, len(transformation_items))):
                transformed_image = trans_func(transformed_image)

            # Save the transformed image with file number in the name
            output_file_name = f"{base_name}_{file_num}.jpg"
            transformed_image.save(os.path.join(dest_folder, output_file_name))

            file_num += 1  # Increment the file number for the next image

            print(f"Processed {output_file_name} for base image {base_name}")

# Paths of the source and destination folders
source_folder = 'Data/Processed/no_background'
dest_folder = 'Data/Processed/database'

# Ensure the destination folder exists
if not os.path.exists(dest_folder):
    os.makedirs(dest_folder)

# Process the images
num_generated_images_per_original = 300 
process_images(source_folder, dest_folder, num_generated_images_per_original)

print("Image processing completed.")
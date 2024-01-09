#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File name: helper.py
Authors: begep1 & rothl18
Date created: 05.12.2023
Date last modified: 08.01.2024
Python Version: 3.11.6

Description:
This utility script contains essential functions used in the real-time detection and 
classification of Jass playing cards. It includes functions for image preprocessing, 
postprocessing neural network outputs, and displaying the results on video frames. 
These functions are designed to work with PyTorch models and OpenCV for image manipulation and display.

Key Features:
- Function to convert images to square format
- Creation of a mapping for card classes
- Preprocessing of video frames for neural network input
- Postprocessing of network predictions to obtain class probabilities
- Display function to visualize prediction results on video frames

Usage:
Import the required functions in your main script using 'from card_detection_utils import *'. 
Ensure that the necessary libraries (PyTorch, OpenCV, PIL, etc.) are installed. 
Use these utility functions as part of your card detection and classification pipeline.
"""

import torch
from torchvision import transforms
import cv2
from PIL import Image


def make_square(image):
    """
    Converts a given PIL image into a square format by adding black padding.

    Args:
        image (PIL.Image): The input image.

    Returns:
        PIL.Image: The square image with black padding.
    """
    width, height = image.size

    # The new size is the maximum of the width and height to ensure the image remains square
    new_size = max(width, height)

    # Create a new black image to serve as a square background
    new_image = Image.new('RGB', (new_size, new_size), (0, 0, 0))

    # Calculate the position to center the original image on the black background
    paste_x = (new_size - width) // 2
    paste_y = (new_size - height) // 2

    # Paste the original image onto the black background
    new_image.paste(image, (paste_x, paste_y))

    return new_image

def create_card_mapping():
    """
    Creates a mapping from class IDs to card names based on suits and values.

    Returns:
        dict: A dictionary mapping class IDs to their corresponding card names.
    """
    suits = ['E', 'H', 'S', 'K']  # Suits: Ecke, Herz, Schaufel, Kreuz
    values = ['A', 'K', 'D', 'B', '10', '9', '8', '7', '6']  # Card values: Ace, King, Dame, Bauer, etc.

    mapping = {}
    class_id = 0
    for suit in suits:
        for value in values:
            mapping[class_id] = f'{suit}_{value}'
            class_id += 1
    return mapping

def preprocess(frame, input_size=(224, 224)):
    """
    Preprocesses a frame for input into a neural network.

    Args:
        frame (numpy.ndarray): The input frame.
        input_size (tuple): The target size for resizing the frame.

    Returns:
        torch.Tensor: The preprocessed frame as a PyTorch tensor.
    """
    # Define transformation sequence
    transform = transforms.Compose([
        transforms.Resize(input_size),  # Resize to the specified input size
        transforms.ToTensor(),          # Convert to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using standard values
    ])

    # Convert the frame to PIL image format and apply the transformations
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    processed_frame = transform(pil_image)

    # Add a batch dimension to the tensor
    processed_frame = processed_frame.unsqueeze(0)

    return processed_frame

def postprocess(predictions):
    """
    Postprocesses the predictions from a neural network to obtain the predicted class and probabilities.

    Args:
        predictions (torch.Tensor): The output from the neural network.

    Returns:
        tuple: The predicted class and the probabilities.
    """
    # Apply softmax to obtain probabilities
    probabilities = torch.nn.functional.softmax(predictions, dim=1)
    _, predicted_class = torch.max(probabilities, 1)  # Get the class with the highest probability

    return predicted_class, probabilities

def display(frame, card_info, class_mapping):
    """
    Displays the prediction results on the frame.

    Args:
        frame (numpy.ndarray): The original frame.
        card_info (tuple): Contains predicted class and probabilities.
        class_mapping (dict): Maps class IDs to class names.

    Returns:
        numpy.ndarray: The frame with prediction results displayed.
    """
    predicted_class, probabilities = card_info

    # Handle string keys in class_mapping
    if isinstance(next(iter(class_mapping)), str):
        class_mapping = {int(k.split('_')[1]): v for k, v in class_mapping.items()}

    # Loop through all predictions to display them on the frame
    for i, class_idx in enumerate(predicted_class):
        class_idx_item = class_idx.item()
        probability = probabilities[i, class_idx_item].item()  # Get probability of the predicted class

        # Check if the class index exists in the mapping
        if class_idx_item not in class_mapping:
            print(f"Warning: Class index {class_idx_item} not found in class_mapping.")
            continue

        class_name = class_mapping[class_idx_item]
        text = f"{class_name}: {probability:.2f}"
        cv2.putText(frame, text, (10, 25 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return frame
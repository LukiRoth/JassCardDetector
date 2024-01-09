#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File name: card_detection_v2.py
Author: begep1 & rothl18
Date created: 28.12.2023
Date last modified: 08.01.2023
Python Version: 3.11.6

Description:
This script is designed for real-time detection and classification of Jass playing cards 
using a pre-trained ResNet34 model. It utilizes the OpenCV library for video capture and processing, 
and PyTorch for running the inference with the neural network model. 
The script captures video frames from a camera, detects potential card contours, 
and classifies each detected card by applying image transformations and using the trained model. 
It includes features for debugging and visualization of the detection and classification process.

Key Features:
- Real-time video capture using OpenCV
- Contour detection for identifying card shapes
- Image preprocessing and transformations for neural network input
- Utilization of a pre-trained ResNet34 model for card classification
- Debugging options for visualizing edge detection and input to the CNN
- Display of the classification results overlaid on the video feed

Usage:
Before running the script, ensure that the required libraries (OpenCV, PyTorch, PIL, etc.) 
are installed and that the trained model file is correctly located in the specified path. 
Set the 'DEBUG_EDGE_DETECTION' and 'DEBUG_IMAGE_2_CNN' flags to True if you wish to visualize 
the intermediate steps of edge detection and CNN input processing. Run the script to start the 
real-time card detection and classification. 
Press 'q' to quit the video feed.
"""

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from Utils.helper import *
from Utils.model_definition import ResNet34

# Debug flags for visualizing intermediate steps
DEBUG_EDGE_DETECTION = False
DEBUG_IMAGE_2_CNN = False

# Thresholds for considering a contour as a potential card
MIN_WIDTH = 50  # Minimum width of the contour
MIN_HEIGHT = 70  # Minimum height of the contour

def main():
    # Load the card mapping dictionary
    card_mapping = create_card_mapping()

    # Load a pre-trained ResNet34 model for card classification
    model = torch.load('Models/TrainedModels/jass_card_classifier_model_v4.pth', map_location=torch.device('cpu'))
    model.eval()  # Set the model to evaluation mode

    # Define the transformations for preprocessing the images before inputting to the model
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224 as expected by ResNet34
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the tensor
    ])

    # Initialize video capture on the second camera (index 1)
    cap = cv2.VideoCapture(1)

    while True:
        # Read a frame from the video capture
        ret, frame = cap.read()
        if not ret:
            break  # Break the loop if no frame is captured

        # Convert the frame to grayscale for easier contour detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Debugging: Show intermediate grayscale and threshold images
        if DEBUG_EDGE_DETECTION:
            plt.imshow(gray, cmap='gray')
            plt.title("Grayscale Image")
            plt.show()

            plt.imshow(thresh, cmap='gray')
            plt.title("Threshold Image")
            plt.show()

        # Process each detected contour
        for contour in contours:
            # Approximate the contour shape to a polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            # Check if the polygon has 4 sides (indicative of a card)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)

                # Only consider contours that are large enough based on predefined minimum size
                if w >= MIN_WIDTH and h >= MIN_HEIGHT:
                    # Extract the region of interest (ROI) from the original frame
                    roi = frame[y:y+h, x:x+w]
                    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB if needed
                    roi_pil = Image.fromarray(roi)  # Convert the ROI to PIL format
                    roi_pil = make_square(roi_pil)  # Make the ROI square (assuming make_square is defined)
                    roi_tensor = transform(roi_pil).unsqueeze(0)  # Transform and add batch dimension

                    # Debugging: Visualize the preprocessed image input to the CNN
                    if DEBUG_IMAGE_2_CNN:
                        # Reverse the normalization for visualization
                        mean = np.array([0.485, 0.456, 0.406])
                        std = np.array([0.229, 0.224, 0.225])
                        roi_unnorm = torch.squeeze(roi_tensor)  # Remove batch dimension
                        roi_unnorm = roi_unnorm * torch.tensor(std[:, None, None]) + torch.tensor(mean[:, None, None])
                        roi_unnorm = roi_unnorm.numpy()
                        roi_unnorm = np.transpose(roi_unnorm, (1, 2, 0))  # Change dimension order for plotting
                        plt.imshow(roi_unnorm)
                        plt.title("ROI Tensor Visualized")
                        plt.show()
                    
                    # Make a prediction using the model
                    with torch.no_grad():
                        prediction = model(roi_tensor)
                    max_index = torch.argmax(prediction).item()  # Get the index of the maximum prediction

                    # Retrieve the corresponding card label from the mapping
                    card_label = card_mapping[max_index]

                    # Draw bounding box and label on the original frame
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, str(card_label), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the frame with detected cards
        cv2.imshow('Card Detection', frame)

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File name: card_detection_v1.py
Authors: begep1 & rothl18
Date created: 20.12.2023
Date last modified: 08.01.2024
Python Version: 3.11.6

Description:
This script is developed for real-time detection and classification of Jass playing cards 
using a pre-trained ResNet34 model. It captures video frames from a camera, processes each frame, 
and performs card detection and classification. The script makes use of the OpenCV library for 
video capture and display, and PyTorch for neural network operations. It includes preprocessing 
of video frames, inference using the trained model, and visualization of the detection results 
on the video feed.

Key Features:
- Real-time video capture for card detection
- Use of a pre-trained ResNet34 model for card classification
- Preprocessing of video frames for neural network compatibility
- Inference and post-processing to extract card information
- Visualization of detection results in the video feed

Usage:
Before running the script, ensure that the required libraries (OpenCV, PyTorch, etc.) are 
installed and the trained model file is available in the specified path. The script is 
designed to be executed directly and will start the video capture and card detection process. 
Press 'q' to exit the video feed.
"""

import cv2
import torch
from Utils.helper import *
from Utils.model_definition import ResNet34

def main():
    # Set the device to GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a mapping of card classes
    card_mapping = create_card_mapping()
    print(card_mapping)
    print("Number of classes:", len(card_mapping))

    # Number of classes derived from the card mapping
    num_classes = len(card_mapping)

    # Initialize the ResNet34 model with the number of classes and transfer it to the appropriate device
    model = ResNet34(num_classes).to(device)

    # Load the trained model
    model = torch.load('Models/TrainedModels/jass_card_classifier_model_v4.pth', map_location=torch.device('cpu'))
    model.eval()  # Set the model to evaluation mode

    # Initialize video capture on the second camera (index 1)
    cap = cv2.VideoCapture(1)

    while True:
        ret, frame = cap.read()  # Read a frame from the camera
        if not ret:
            break  # Exit the loop if no frame is captured

        # Preprocess the frame to be suitable for the model
        processed_frame = preprocess(frame)

        # Perform inference
        with torch.no_grad():
            predictions = model(processed_frame)

        # Post-process the predictions to extract card information
        card_info = postprocess(predictions)

        # Display the results on the frame
        display(frame, card_info, card_mapping)

        # Show the frame with card detection results
        cv2.imshow('Card Detection', frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
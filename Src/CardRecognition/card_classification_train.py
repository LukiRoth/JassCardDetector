#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File name: card_classification_train.py
Author: begep1 & rothl18
Date created: 05.12.2023
Date last modified: 08.01.2023
Python Version: 3.11.6

Description:
This script is developed for training a neural network model, specifically a ResNet34, 
to classify Jass playing cards. It includes steps for dataset downloading, preprocessing, model training, 
and validation. The script utilizes PyTorch for building and training the model. 
It is designed to work with a custom dataset of Jass card images, handling tasks such as dataset loading, 
image transformations, model training, and accuracy assessment.

Key Features:
- Dataset downloading from Kaggle
- Custom dataset class for image processing
- Implementation of ResNet34 with PyTorch
- Training and validation loops with progress tracking
- Model performance visualization (loss and accuracy)

Usage:
Set the 'dataset_path' and 'kaggle_path' variables to specify the location of your dataset. 
Ensure that the required libraries are installed. Run the script to train the model. 
The script will automatically handle dataset preparation, model training, and performance evaluation.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

NUM_EPOCHS = 70
DATASET_PATH = '../../Data/Processed/database'

def create_card_mapping_files():
    """
    Creates a mapping of card names to class IDs.

    This function generates a dictionary where each card, defined by its suit and value,
    is associated with a unique class ID. This is useful for classification tasks in machine learning,
    where each card needs to be distinctly identified.

    Returns:
        dict: A dictionary with keys as card names (in the format 'suit_value') and values as unique class IDs.
    """

    # Define the suits of the cards.
    # 'E' stands for Ecke, 'H' for Herz, 'S' for Schaufel, 'K' for Kreuz.
    suits = ['E', 'H', 'S', 'K']

    # Define the values of the cards.
    # The mapping follows: 0 = Ass, 1 = König, 2 = Dame, 3 = Bauer, 4 = 10, 5 = 9, 6 = 8, 7 = 7, 8 = 6.
    values = ['0', '1', '2', '3', '4', '5', '6', '7', '8']

    # Initialize an empty dictionary to hold the mapping.
    mapping = {}

    # Initialize a class ID starting from 0. This will be used to assign a unique ID to each card.
    class_id = 0

    # Iterate over each combination of suit and value.
    for suit in suits:
        for value in values:
            # Create a composite key in the format 'suit_value' and assign it the current class ID.
            mapping[f'{suit}_{value}'] = class_id
            # Increment the class ID for the next card.
            class_id += 1

    # Return the complete mapping dictionary.
    return mapping

# Custom dataset class
class JassCardDataset(Dataset):
    """
    A PyTorch Dataset class that handles loading of Jass card images for training a machine learning model.
    
    Attributes:
        directory (str): Directory where card images are stored.
        transform (callable, optional): Optional transform to be applied on a sample.
        images (list): List of image filenames in the directory.
        mapping (dict): Mapping from card IDs to numerical labels.
    """

    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.images = [img for img in os.listdir(directory) if img.endswith(('.png', '.jpg', '.jpeg'))]  # Filter for image files
        self.mapping = create_card_mapping_files()
        
        print(f"Total number of images found: {len(self.images)}")  # Report the total number of images loaded.

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.directory, self.images[idx])
        image = Image.open(img_name)
        label = self.extract_label(self.images[idx])

        if self.transform:
            image = self.transform(image)

        return image, label

    def extract_label(self, filename):
        # Attempt to extract the card ID from the filename
        card_id = filename.split('_')[0] + '_' + filename.split('_')[1]
        label = self.mapping.get(card_id, None)

        # Debugging print statements
        if label is None:
            print(f"Unmapped label for file: {filename}, Extracted card_id: {card_id}")

        return label

# ResNet34 class
class ResNet34(nn.Module):
    """
    ResNet34 neural network model for classification tasks.

    This class extends the PyTorch Module class, modifying the ResNet34 model
    (originally trained on ImageNet) for a custom number of output classes.

    Attributes:
        resnet (nn.Module): The ResNet34 model.

    Args:
        num_classes (int): The number of classes for the final output layer.
    """
    def __init__(self, num_classes):
        super(ResNet34, self).__init__()
        # Initialize the ResNet34 model with pretrained weights.
        # The weights parameter specifies using the model trained on ImageNet.
        self.resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

         # The feature size before the fully connected layer in ResNet34 is 512.
        self.resnet.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output of the ResNet34 model.
        """
        return self.resnet(x)

def main():
    # Set device to CUDA if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Print the current working directory
    print("Current Working Directory:", os.getcwd())
    # Change to the script's directory (if required)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Create a mapping of cards, and count the number of classes
    card_mapping = create_card_mapping_files()
    num_classes = len(card_mapping)

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize for ResNet input
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])

    # Create dataset and split into training and validation sets
    jass_dataset = JassCardDataset(directory=DATASET_PATH, transform=transform)
    validation_size = int(0.2 * len(jass_dataset))  # 20% for validation
    train_size = len(jass_dataset) - validation_size
    train_dataset, validation_dataset = random_split(jass_dataset, [train_size, validation_size])

    # Create DataLoaders for training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False)

    # Initialize the model
    model = ResNet34(num_classes).to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Lists to store loss and accuracy
    training_losses = []
    validation_losses = []
    accuracies = []

    # Training and validation loop
    try:
        for epoch in range(NUM_EPOCHS):
            epoch_start_time = time.time()
            # Training phase
            model.train()  # Set model to training mode
            for i, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Training Progress: {int((i+1)/len(train_loader)*100)}%", end='\r')
            # Validation phase
            model.eval()  # Set model to evaluation mode
            valid_loss = 0
            all_targets = []
            all_predictions = []
            with torch.no_grad():
                for i, (inputs, targets) in enumerate(validation_loader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    valid_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    all_targets.extend(targets.cpu())
                    all_predictions.extend(predicted.cpu())
                    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Validation Progress: {int((i+1)/len(validation_loader)*100)}%", end='\r')
            # Calculate and store metrics for the epoch
            avg_train_loss = loss.item()
            avg_valid_loss = valid_loss / len(validation_loader)
            epoch_accuracy = accuracy_score([t.numpy() for t in all_targets], [p.numpy() for p in all_predictions]) * 100
            training_losses.append(avg_train_loss)
            validation_losses.append(avg_valid_loss)
            accuracies.append(epoch_accuracy)
            # Print epoch summary
            epoch_end_time = time.time()
            print(f"\033[KEpoch {epoch+1}, Training Loss: {avg_train_loss:.3f}, Validation Loss: {avg_valid_loss:.3f}, Accuracy: {epoch_accuracy:.2f}%, Time: {int((epoch_end_time - epoch_start_time) // 60)}m {int((epoch_end_time - epoch_start_time) % 60)}s")
    except KeyboardInterrupt:
        print("Interrupted. Saving the current model...")
        torch.save(model, f'jass_card_classifier_model_interrupted_epoch_{epoch+1}.pth')
        print("Model saved. Exiting program.")
    
    # Save the model after training
    torch.save(model, 'jass_card_classifier_model.pth')
    print("Model saved.")

    # Generate confusion matrix
    cm = confusion_matrix([t.numpy() for t in all_targets], [p.numpy() for p in all_predictions])
    df_cm = pd.DataFrame(cm, index=[i for i in card_mapping.keys()], columns=[i for i in card_mapping.keys()])
    plt.figure(figsize=(10,7))
    sns.heatmap(df_cm, annot=True, cmap='Blues', fmt='g')

    plt.show()

    # Plot training loss, validation loss, and accuracy
    epochs = range(1, len(training_losses) + 1)

    plt.figure(figsize=(10, 3))
    plt.subplot(1, 3, 1)
    plt.plot(epochs, training_losses, 'r-', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, validation_losses, 'b-', label='Validation Loss')
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, accuracies, 'g-', label='Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()
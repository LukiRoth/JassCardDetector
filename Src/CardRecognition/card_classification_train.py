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
    suits = ['E', 'H', 'S', 'K']
    values = ['0', '1', '2', '3', '4', '5', '6', '7', '8']
    mapping = {}
    class_id = 0
    for suit in suits:
        for value in values:
            mapping[class_id] = f'{suit}_{value}'
            class_id += 1
    return mapping

# Custom dataset class
class JassCardDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.images = [img for img in os.listdir(directory) if img.endswith(('.png', '.jpg', '.jpeg'))]  # Filter for image files
        self.mapping = create_card_mapping_files()
        
        # Print the total number of images found
        print("Total number of images found:", len(self.images))

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
    
class ResNet34(nn.Module):
    def __init__(self, num_classes):
        super(ResNet34, self).__init__()
        #self.resnet = models.resnet34(pretrained=True)
        # Update the model initialization with the new 'weights' parameter
        self.resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

        # Replace the last fully connected layer
        # ResNet34 uses 512 for fc layers
        self.resnet.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.resnet(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Print the current working directory
print("Current Working Directory:", os.getcwd())
# Change the current working directory to the script's directory (if needed)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

card_mapping = create_card_mapping_files()
num_classes = len(card_mapping)


    
# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to the input size expected by ResNet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create dataset
jass_dataset = JassCardDataset(directory=DATASET_PATH, transform=transform)
# Define the size of the validation set

validation_size = int(0.2 * len(jass_dataset))  # 20% for validation
train_size = len(jass_dataset) - validation_size

# Split the dataset
train_dataset, validation_dataset = random_split(jass_dataset, [train_size, validation_size])

# Create DataLoaders for both training and validation sets
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False)

# After initializing your model
model = ResNet34(num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

training_losses = []
validation_losses = []
accuracies = []

try:
    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
        
        model.train()  # Set the model to training mode
        total_batches = len(train_loader)
        # Training loop
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # Print the progress
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Training Progress: {int((i+1)/total_batches*100)}%", end='\r')

        # Validation loop
        model.eval()
        valid_loss = 0
        all_targets = []
        all_predictions = []
        total_val_batches = len(validation_loader)

        # Inside your validation loop
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(validation_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                valid_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                all_targets.extend(targets.cpu())
                all_predictions.extend(predicted.cpu())

                # Print the validation progress
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Validation Progress: {int((i+1)/total_val_batches*100)}%", end='\r')

        # Convert to NumPy arrays before calculating accuracy
        all_targets_np = [t.numpy() for t in all_targets]
        all_predictions_np = [p.numpy() for p in all_predictions]
        accuracy = accuracy_score(all_targets_np, all_predictions_np)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        minutes = int(epoch_duration // 60)
        seconds = int(epoch_duration % 60)

        # Calculate and store the average training loss, validation loss, and accuracy for the epoch
        avg_train_loss = loss.item()
        avg_valid_loss = valid_loss / len(validation_loader)
        epoch_accuracy = accuracy_score(all_targets_np, all_predictions_np) * 100

        training_losses.append(avg_train_loss)
        validation_losses.append(avg_valid_loss)
        accuracies.append(epoch_accuracy)

        print(f"\033[KEpoch {epoch+1}, Training Loss: {avg_train_loss:.3f}, Validation Loss: {avg_valid_loss:.3f}, Accuracy: {epoch_accuracy:.2f}%, Time: {minutes}m {seconds}s")

    # Save the entire model
    torch.save(model, 'jass_card_classifier_model.pth')
    print("Model saved.")

except KeyboardInterrupt:
    print("Interrupted. Saving the current model...")
    torch.save(model, f'jass_card_classifier_model_interrupted_epoch_{epoch+1}.pth')
    print("Model saved. Exiting program.")

#Print confusion matrix
cm = confusion_matrix(all_targets_np, all_predictions_np)
df_cm = pd.DataFrame(cm, index = [i for i in card_mapping.keys()], columns = [i for i in card_mapping.keys()])
plt.figure(figsize = (10,7))
sns.heatmap(df_cm, annot=True, cmap='Blues', fmt='g')
plt.show()

epochs = range(1, len(training_losses)+1)

# Plot Training Loss
plt.figure(figsize=(10, 3))
plt.subplot(1, 3, 1)
plt.plot(epochs, training_losses, 'r-', label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot Validation Loss
plt.subplot(1, 3, 2)
plt.plot(epochs, validation_losses, 'b-', label='Validation Loss')
plt.title('Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot Accuracy
plt.subplot(1, 3, 3)
plt.plot(epochs, accuracies, 'g-', label='Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.show()
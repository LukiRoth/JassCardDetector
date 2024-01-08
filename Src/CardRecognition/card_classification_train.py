import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
#from Utils.helper import create_card_mapping_files

NUM_EPOCHS = 70

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

def train_model(model, train_loader, validation_loader, num_epochs, device):
    """
    Trains and validates the model.

    Parameters:
    model (nn.Module): The neural network model to be trained.
    train_loader (DataLoader): DataLoader for the training set.
    validation_loader (DataLoader): DataLoader for the validation set.
    num_epochs (int): Number of epochs to train the model.
    device (torch.device): Device to train the model on (CPU or GPU).
    """
    training_losses = []
    validation_losses = []
    accuracies = []

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        total_train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        training_losses.append(avg_train_loss)

        # Validation Phase
        model.eval()
        total_val_loss = 0
        all_targets = []
        all_predictions = []
        with torch.no_grad():
            for inputs, targets in validation_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        avg_val_loss = total_val_loss / len(validation_loader)
        validation_losses.append(avg_val_loss)
        accuracy = accuracy_score(all_targets, all_predictions) * 100
        accuracies.append(accuracy)

        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%")

    return training_losses, validation_losses, accuracies

def check_data_availability(dataset_path):
    """
    Checks if the dataset directory exists and contains at least one image file.
    
    Parameters:
    dataset_path (str): Path to the dataset directory.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset directory {dataset_path} not found.")
    
    image_files = [file for file in os.listdir(dataset_path) if file.endswith(('.png', '.jpg', '.jpeg'))]
    if len(image_files) == 0:
        raise FileNotFoundError(f"No image files found in {dataset_path}.")
    
def plot_metrics(training_losses, validation_losses, accuracies):
    """
    Plots the training loss, validation loss, and accuracy over epochs.

    Parameters:
    training_losses (list): List of training losses per epoch.
    validation_losses (list): List of validation losses per epoch.
    accuracies (list): List of accuracies per epoch.
    """
    epochs = range(1, len(training_losses) + 1)

    plt.figure(figsize=(15, 5))

    # Plot Training Loss
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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Set the current working directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("Current Working Directory:", os.getcwd())

    # Check if dataset is available
    dataset_path = '../../Data/Processed/database'
    check_data_availability(dataset_path)

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create dataset and DataLoaders
    jass_dataset = JassCardDataset(directory=dataset_path, transform=transform)
    validation_size = int(0.2 * len(jass_dataset))
    train_size = len(jass_dataset) - validation_size
    train_dataset, validation_dataset = random_split(jass_dataset, [train_size, validation_size])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False)

    # Initialize model and start training
    num_classes = len(jass_dataset.mapping)
    model = ResNet34(num_classes).to(device)

    training_losses, validation_losses, accuracies = train_model(model, train_loader, validation_loader, NUM_EPOCHS, device)

    # Plot the training metrics
    plot_metrics(training_losses, validation_losses, accuracies)

if __name__ == "__main__":
    main()

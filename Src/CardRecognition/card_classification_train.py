import os
import kaggle
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

# Constants
DATASET_PATH = '../../Data/Processed/database'
KAGGLE_PATH = 'pbegert/french-jass-cards'

# ----------------------------------------------------------------------------------------------------------------

def main():
    device = setup_device()
    download_dataset(DATASET_PATH, KAGGLE_PATH)
    card_mapping = create_card_mapping()
    train_loader, validation_loader = prepare_datasets(DATASET_PATH)
    model, criterion, optimizer = setup_model(len(card_mapping), device)
    training_losses, validation_losses, accuracies = train_and_validate(model, criterion, optimizer, train_loader, validation_loader, num_epochs=10, device=device)
    plot_results(training_losses, validation_losses, accuracies)

    model_filename = 'jass_card_classifier_model_test.pth'
    save_model(model, model_filename)
    print(f"Model saved to {model_filename}")


# ----------------------------------------------------------------------------------------------------------------

def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device

def download_dataset(dataset_path, kaggle_path):
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
        absolute_dataset_path = os.path.abspath(dataset_path)
        print("Absolute dataset path:", absolute_dataset_path)
        print("Downloading dataset...")
        kaggle.api.dataset_download_files(kaggle_path, path=dataset_path, unzip=True)
    else:
        print("Dataset already exists. Skipping download.")

def create_card_mapping():
    suits = ['E', 'H', 'S', 'K']  # Ecke, Herz, Schaufel, Kreuz
    values = ['0', '1', '2', '3', '4', '5', '6', '7', '8'] # 0 = Ass, 1 = König, 2 = Dame, 3 = Bauer, 4 = 10, 5 = 9, 6 = 8, 7 = 7, 8 = 6
    mapping = {}
    class_id = 0
    for suit in suits:
        for value in values:
            mapping[f'{suit}_{value}'] = class_id
            class_id += 1

    return mapping

# ----------------------------------------------------------------------------------------------------------------

class JassCardDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.images = [img for img in os.listdir(directory) if img.endswith(('.png', '.jpg', '.jpeg'))]  # Filter for image files
        self.mapping = create_card_mapping()
        
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

# ----------------------------------------------------------------------------------------------------------------

def prepare_datasets(dataset_path):
    """
    Prepares the datasets and dataloaders for training and validation.

    Args:
    dataset_path (str): Path to the dataset directory.

    Returns:
    tuple: Tuple containing the train DataLoader and validation DataLoader.
    """
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to the input size expected by ResNet
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create dataset
    jass_dataset = JassCardDataset(directory=dataset_path, transform=transform)

    # Define the size of the validation set
    validation_size = int(0.2 * len(jass_dataset))  # 20% for validation
    train_size = len(jass_dataset) - validation_size

    # Split the dataset
    train_dataset, validation_dataset = random_split(jass_dataset, [train_size, validation_size])

    # Create DataLoaders for both training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False)

    return train_loader, validation_loader


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

def setup_model(num_classes, device):
    """
    Initializes and returns the model, criterion, and optimizer.

    Args:
    num_classes (int): Number of classes in the dataset.
    device (torch.device): The device to run the model on (CPU or GPU).

    Returns:
    tuple: Model, criterion, and optimizer.
    """
    model = ResNet34(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return model, criterion, optimizer


def train_and_validate(model, criterion, optimizer, train_loader, validation_loader, num_epochs, device):
    """
    Trains and validates the model.

    Args:
    model (nn.Module): The neural network model.
    criterion: The loss function.
    optimizer: The optimization algorithm.
    train_loader (DataLoader): DataLoader for the training set.
    validation_loader (DataLoader): DataLoader for the validation set.
    num_epochs (int): Number of epochs to train.
    device (torch.device): The device to run the model on (CPU or GPU).

    Returns:
    tuple: Lists of training losses, validation losses, and accuracies per epoch.
    """
    training_losses = []
    validation_losses = []
    accuracies = []

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        # Training Phase
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        training_losses.append(avg_train_loss)

        # Validation Phase
        model.eval()
        valid_loss = 0.0
        all_targets = []
        all_predictions = []
        with torch.no_grad():
            for inputs, targets in validation_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                valid_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        avg_valid_loss = valid_loss / len(validation_loader)
        validation_losses.append(avg_valid_loss)
        epoch_accuracy = accuracy_score(all_targets, all_predictions) * 100
        accuracies.append(epoch_accuracy)

        # Time calculation for the epoch
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        minutes = int(epoch_duration // 60)
        seconds = int(epoch_duration % 60)

        # Print epoch statistics
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.3f}, Validation Loss: {avg_valid_loss:.3f}, Accuracy: {epoch_accuracy:.2f}%, Time: {minutes}m {seconds}s")

    return training_losses, validation_losses, accuracies



def save_model(model, filename):
    """
    Saves the model to a file.

    Args:
    model (nn.Module): The neural network model to save.
    filename (str): The filename to save the model to.
    """
    torch.save(model.state_dict(), filename)


def load_model(filename, device, num_classes):
    """
    Loads a model from a file.

    Args:
    filename (str): The filename to load the model from.
    device (torch.device): The device to run the model on (CPU or GPU).
    num_classes (int): Number of classes in the dataset (required to initialize the model structure).

    Returns:
    nn.Module: The loaded model.
    """
    model = ResNet34(num_classes).to(device)
    model.load_state_dict(torch.load(filename, map_location=device))
    return model


def plot_results(training_losses, validation_losses, accuracies, confusion_matrix=None, class_names=None):
    """
    Plots the training loss, validation loss, accuracies, and confusion matrix.

    Args:
    training_losses (list): List of training losses.
    validation_losses (list): List of validation losses.
    accuracies (list): List of accuracies.
    confusion_matrix (np.array, optional): Confusion matrix. Defaults to None.
    class_names (list, optional): List of class names for the confusion matrix. Defaults to None.
    """
    plt.figure(figsize=(12, 8))

    # Plot Training and Validation Loss
    plt.subplot(2, 2, 1)
    plt.plot(training_losses, label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(accuracies, label='Accuracy', color='green')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot Confusion Matrix if provided
    if confusion_matrix is not None and class_names is not None:
        plt.subplot(2, 2, 3)
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
import os
import kaggle
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import re

num_epochs = 100

# Function to download dataset
def download_dataset(dataset_path, kaggle_path):
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
        print("Downloading dataset...")
        kaggle.api.dataset_download_files(kaggle_path, path=dataset_path, unzip=True)
    else:
        print("Dataset already exists. Skipping download.")

# Download Jass Card Dataset
dataset_path = 'Data/Processed/database'
kaggle_path = 'pbegert/swiss-jass-cards'
download_dataset(dataset_path, kaggle_path)

# Custom dataset class
class JassCardDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.images = os.listdir(directory)

    def __len__(self):
        return len(self.images)
    
    @staticmethod
    def extract_label(filename):
        # Regex pattern to match filenames like 'E_2_256.jpg', capturing 'E' and '2'
        match = re.match(r"([a-zA-Z])_(\d+)_\d+\.jpg", filename)
        if match:
            group = match.group(1)
            value = match.group(2)
            print(f"Filename '{filename}' processed")
            return JassCardDataset.label_to_int(group), JassCardDataset.value_to_int(value)
        else:
            print(f"Warning: Filename '{filename}' does not match expected format", end='\r')
            return None, None

    @staticmethod
    def label_to_int(label_char):
        # Map label characters to integers
        mapping = {'H': 0, 'E': 1, 'S': 2, 'K': 3}  # Add other mappings as needed
        return mapping.get(label_char, -1)  # Returns -1 if label_char not in mapping

    @staticmethod
    def value_to_int(value_str):
        # Map value strings to integers
        mapping = {'6': 0, '7': 1, '8': 2, '9': 3, '10': 4, 'J': 5, 'Q': 6, 'K': 7, 'A': 8}
        return mapping.get(value_str, -1)  # Returns -1 if value_str not in mapping
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.directory, self.images[idx])
        image = Image.open(img_name)
        label = self.extract_label(self.images[idx])

        if label == (None, None):  # Skip files with unexpected format
            return None

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label)  # Ensure labels are tensors


# Define transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


# Create dataset
jass_dataset = JassCardDataset(directory=dataset_path, transform=transform)
train_loader = DataLoader(jass_dataset, batch_size=32, shuffle=True)

# Filter out None values
filtered_dataset = [data for data in jass_dataset if data is not None]
train_loader = DataLoader(filtered_dataset, batch_size=32, shuffle=True)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64 * 62 * 62, 128)
        self.fc2 = nn.Linear(128, 13)  # num_classes is the number of card types

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Step 3: Set Up Loss Function and Optimizer
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 4: Train the Model
for epoch in range(num_epochs):
    for data in train_loader:
        if data is None:  # Skip over None data items
            continue
        inputs, targets = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item()}")

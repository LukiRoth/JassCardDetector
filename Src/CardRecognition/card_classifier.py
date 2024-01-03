import os
import kaggle
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score

# Function to download dataset
def download_dataset(dataset_path, kaggle_path):
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
        print("Downloading dataset...")
        kaggle.api.dataset_download_files(kaggle_path, path=dataset_path, unzip=True)
    else:
        print("Dataset already exists. Skipping download.")

# Download Jass Card Dataset
dataset_path = '../../Data/Processed/database'
kaggle_path = 'pbegert/swiss-jass-cards'
download_dataset(dataset_path, kaggle_path)


def create_card_mapping():
    suits = ['E', 'H', 'S', 'K']  # Ecke, Herz, Schaufel, Kreuz
    values = ['0', '1', '2', '3', '4', '5', '6', '7', '8']
    mapping = {}
    class_id = 0
    for suit in suits:
        for value in values:
            mapping[f'{suit}_{value}'] = class_id
            class_id += 1
    print(mapping)
    return mapping

card_mapping = create_card_mapping()

# Custom dataset class
class JassCardDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.images = [img for img in os.listdir(directory) if img.endswith(('.png', '.jpg', '.jpeg'))]  # Filter for image files
        self.mapping = create_card_mapping()

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
    
# Define transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Create dataset
jass_dataset = JassCardDataset(directory=dataset_path, transform=transform)
print("Dataset size:", len(jass_dataset))
# Define the size of the validation set

validation_size = int(0.2 * len(jass_dataset))  # 20% for validation
train_size = len(jass_dataset) - validation_size

# Split the dataset
train_dataset, validation_dataset = random_split(jass_dataset, [train_size, validation_size])

# Create DataLoaders for both training and validation sets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(0.5)

        # After conv1 and pool: 64x64 -> 32x32
        # After conv2 and pool: 32x32 -> 16x16
        # After conv3 and pool: 16x16 -> 8x8
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, len(card_mapping))

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    # Training loop
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    # Validation loop
    model.eval()  # Set the model to evaluation mode
    valid_loss = 0
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        for inputs, targets in validation_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            valid_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            all_targets.extend(targets)
            all_predictions.extend(predicted)

    # Calculate validation accuracy
    accuracy = accuracy_score(all_targets, all_predictions)
    print(f"Epoch {epoch}, Training Loss: {loss.item()}, Validation Loss: {valid_loss / len(validation_loader)}, Accuracy: {accuracy}")


# Save the entire model
torch.save(model, 'jass_card_classifier_model.pth')
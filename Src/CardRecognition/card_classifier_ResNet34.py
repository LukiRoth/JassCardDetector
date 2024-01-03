import os
import kaggle
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score

# Print the current working directory
print("Current Working Directory:", os.getcwd())
# Change the current working directory to the script's directory (if needed)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ----------------------------------------------------------------------------------------------------------------

# Load Dataset
def download_dataset(dataset_path, kaggle_path):
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
        absolute_dataset_path = os.path.abspath(dataset_path)
        print("Absolute dataset path:", absolute_dataset_path)
        print("Downloading dataset...")
        kaggle.api.dataset_download_files(kaggle_path, path=dataset_path, unzip=True)
    else:
        print("Dataset already exists. Skipping download.")

# Download Jass Card Dataset
dataset_path = '../../Data/Processed/database'
kaggle_path = 'pbegert/swiss-jass-cards'
download_dataset(dataset_path, kaggle_path)

absolute_dataset_path = os.path.abspath(dataset_path)
print("Absolute dataset path:", absolute_dataset_path)

# ----------------------------------------------------------------------------------------------------------------

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

# ----------------------------------------------------------------------------------------------------------------

# Custom dataset class
class JassCardDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.images = [img for img in os.listdir(directory) if img.endswith(('.png', '.jpg', '.jpeg'))]  # Filter for image files
        self.mapping = create_card_mapping()
        print("Found images:", self.images[-1])

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
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)

# ----------------------------------------------------------------------------------------------------------------

class ResNet34(nn.Module):
    def __init__(self, num_classes):
        super(ResNet34, self).__init__()
        self.resnet = models.resnet34(pretrained=True)

        # Replace the last fully connected layer
        # ResNet34 uses 512 for fc layers
        self.resnet.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.resnet(x)

num_classes = len(card_mapping)  # Assuming 'card_mapping' is your class mapping
model = ResNet34(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ----------------------------------------------------------------------------------------------------------------

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

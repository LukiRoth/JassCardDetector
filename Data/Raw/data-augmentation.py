import torch
from torchvision import datasets, transforms
from PIL import Image
import os

# Definieren der Transformationen für die Daten-Augmentation
transform = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.5),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

# Pfad zu Ihren Bildern
image_folder_path = 'Data/Raw/original'

# Pfad zu Ihren Bildern
save_image_folder_path = 'Data/Raw/augmentat'

# Laden der Bilder
dataset = datasets.ImageFolder(root=image_folder_path, transform=transform)

# Bilder durchlaufen und speichern
for i, (image, label) in enumerate(dataset):
    # Konvertieren des Tensor-Images zurück in ein PIL-Image
    pil_image = transforms.ToPILImage()(image).convert("RGB")
    
    # Speichern des Bildes
    save_path = os.path.join(save_image_folder_path, f'augmented_image_{i}.jpg')
    pil_image.save(save_path)

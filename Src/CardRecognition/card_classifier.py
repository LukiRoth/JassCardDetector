import os
import kaggle

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



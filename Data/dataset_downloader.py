#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File name: dataset_downloader.py
Author: begep1 & rothl18
Date created: 08.01.2023
Date last modified: 08.01.2023
Python Version: 3.11.6

Description:
This script is responsible for downloading the card images dataset from Kaggle.
It uses the Kaggle API to check for the existence of the dataset in the specified
path and downloads it if not present. The dataset is essential for training a
machine learning model for card recognition.

Usage:
Set the 'dataset_path' and 'kaggle_path' variables appropriately before running the script.
Ensure that Kaggle API credentials are set up correctly in your environment.
"""

import os
import kaggle

def download_dataset(dataset_path, kaggle_path):
    """
    Downloads the dataset from Kaggle and extracts it into the specified path.

    Parameters:
    dataset_path (str): Local path where the dataset should be downloaded and extracted.
    kaggle_path (str): Kaggle dataset path.
    """
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
        absolute_dataset_path = os.path.abspath(dataset_path)
        print("Absolute dataset path:", absolute_dataset_path)
        print("Downloading dataset from Kaggle...")
        kaggle.api.dataset_download_files(kaggle_path, path=dataset_path, unzip=True)
    else:
        print("Dataset already exists. Skipping download.")

# Main execution
if __name__ == "__main__":
    # Download Jass Card Dataset
    dataset_path = 'Data/Processed/database'
    kaggle_path = 'pbegert/french-jass-cards'   # Update this path to the correct Kaggle dataset path
    download_dataset(dataset_path, kaggle_path)

    absolute_dataset_path = os.path.abspath(dataset_path)
    print("Absolute dataset path:", absolute_dataset_path)

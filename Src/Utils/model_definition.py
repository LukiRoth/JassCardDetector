#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File name: model_definition.py
Authors: begep1 & rothl18
Date created: 12.12.2023
Date last modified: 08.01.2024
Python Version: 3.11.6

Description:
This script defines the ResNet34 class, a modification of the standard ResNet34 neural network model, 
for classification tasks. It adapts the original ResNet34 architecture, which is pretrained on the 
ImageNet dataset, to a customizable number of output classes. This is achieved by replacing the final 
fully connected layer of the model. The class extends the PyTorch Module class, making it compatible 
with PyTorch workflows for model training and inference.

Key Features:
- Customizable number of output classes
- Initialization with pretrained ImageNet weights for improved performance
- Replacement of the final fully connected layer to adjust to the specific classification task
- Integration into the PyTorch framework for seamless use in training and inference pipelines

Usage:
Import the ResNet34 class from this script into your training or inference pipeline. 
Initialize the class with the desired number of output classes. The model can then be 
trained with your specific dataset or used for inference with pretrained weights.

"""

import torch.nn as nn
from torchvision import models

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
        # Pass the input through the ResNet34 model and return its output.
        return self.resnet(x)

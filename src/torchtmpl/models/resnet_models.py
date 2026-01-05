# coding: utf-8

# Standard imports
from functools import reduce
import operator

# External imports
import torch
import torch.nn as nn

#appeler un resnet 50 pretrained
def ResNet50(cfg, input_size, num_classes):
    """
    ResNet50 model from torchvision.models
    """
    from torchvision.models import resnet50, ResNet50_Weights

    # Load a pretrained ResNet50 model
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    # Modify the first convolutional layer if the input has a different number of channels
    if input_size[0] != 3:
        model.conv1 = nn.Conv2d(
            input_size[0],
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

    # Modify the final fully connected layer to match the number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
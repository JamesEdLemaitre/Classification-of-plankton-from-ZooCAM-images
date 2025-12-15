# coding: utf-8

# Standard imports
import logging
import random

# External imports
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
from torchvision import transforms
from torchvision.utils import make_grid
import torchvision.transforms.functional as F
import PIL

import matplotlib.pyplot as plt


def show_image(X):
    num_c = X.shape[0]
    plt.figure()
    plt.imshow(X[0] if num_c == 1 else X.permute(1, 2, 0))
    plt.show()

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def get_dataloaders(data_config, use_cuda):
    valid_ratio = data_config["valid_ratio"]
    batch_size = data_config["batch_size"]
    num_workers = data_config["num_workers"]
    resize = data_config["images_resize"]

    logging.info("  - Dataset creation")

    input_transform = transforms.Compose(
        [transforms.Grayscale(),
         transforms.Resize((resize, resize)),
         transforms.ToTensor()]
    )

    base_dataset = torchvision.datasets.ImageFolder(
        root=data_config["trainpath"], 
        transform=input_transform,
    )

    logging.info(f"  - I loaded {len(base_dataset)} samples")

    indices = list(range(len(base_dataset)))
    random.shuffle(indices)
    num_valid = int(valid_ratio * len(base_dataset))
    train_indices = indices[num_valid:]
    valid_indices = indices[:num_valid]

    train_dataset = torch.utils.data.Subset(base_dataset, train_indices)
    valid_dataset = torch.utils.data.Subset(base_dataset, valid_indices)

    # Build the dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )
    num_classes = len(base_dataset.classes)
    input_size = tuple(base_dataset[0][0].shape)

    return train_loader, valid_loader, input_size, num_classes


def test_dataloaders():
    print("_"*20)
    print("Test dataloaders:")
    
    data_config = {
        "trainpath": "./data/3-md-4040-2026-challenge/train",
        "valid_ratio": 0.2,
        "batch_size": 32,
        "num_workers": 0,
        "images_resize": 128
    }
    if torch.cuda.is_available():
        device = torch.device("cuda")
        use_cuda = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        use_cuda = True 
    else:
        use_cuda = False

    print(f"Use CUDA: {use_cuda} | Device: {device}")

    train_loader, valid_loader, input_size, num_classes = get_dataloaders(
        data_config, use_cuda
    )

    X, y = next(iter(train_loader))
    grid = make_grid(X, nrow=8)
    show(grid)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_dataloaders()
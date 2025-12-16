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

from torch.utils.data import WeightedRandomSampler

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

def show_class_distribution(dataset, results_save_path = "./data_analysis"):
    """Affiche la distribution des classes dans le dataset"""
    classes = dataset.classes
    counts = [0] * len(classes)
    
    # Compter les échantillons par classe
    for _, label in dataset:
        counts[label] += 1
    
    # Afficher
    plt.figure(figsize=(10, 5))
    plt.bar(classes, counts)
    plt.xlabel('Classes')
    plt.ylabel('Nombre d\'images')
    plt.title('Distribution des classes')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{results_save_path}/class_distribution.png")
    
    # Afficher aussi les chiffres
    for cls, count in zip(classes, counts):
        print(f"{cls}: {count}")


def get_dataloaders(data_config, use_cuda):
    valid_ratio = data_config["valid_ratio"]
    batch_size = data_config["batch_size"]
    num_workers = data_config["num_workers"]
    resize = data_config["images_resize"]

    logging.info("  - Dataset creation")

    input_transform = transforms.Compose(
        [transforms.Grayscale(),
         transforms.Resize((resize, resize)),
         transforms.RandomHorizontalFlip(p=0.5),
         transforms.RandomRotation(15),
         transforms.ToTensor()]
    )

    base_dataset = torchvision.datasets.ImageFolder(
        root=data_config["trainpath"], 
        transform=input_transform,
    )

    logging.info(f"  - I loaded {len(base_dataset)} samples")

    # Calculer les poids pour chaque classe
    targets = [label for _, label in base_dataset]
    class_counts = np.bincount(targets)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in targets]

    indices = list(range(len(base_dataset)))
    random.shuffle(indices)
    num_valid = int(valid_ratio * len(base_dataset))
    train_indices = indices[num_valid:]
    valid_indices = indices[:num_valid]

    train_dataset = torch.utils.data.Subset(base_dataset, train_indices)
    valid_dataset = torch.utils.data.Subset(base_dataset, valid_indices)

    # Build the dataloaders
    train_weights = [sample_weights[i] for i in train_indices]
    sampler = WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_weights)//5,
        replacement=True
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
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
        "trainpath": "./data/train",
        "valid_ratio": 0.2,
        "batch_size": 32,
        "num_workers": 0,
        "images_resize": 128
    }

    results_save_path = "./data_analysis"

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

    # Afficher la distribution
    base_dataset = torchvision.datasets.ImageFolder(
        root=data_config["trainpath"], 
        transform=transforms.ToTensor()
    )
    show_class_distribution(base_dataset, results_save_path = "./data_analysis")

    X, y = next(iter(train_loader))
    grid = make_grid(X, nrow=8)
    show(grid)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_dataloaders()
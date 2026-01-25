# coding: utf-8

# Standard imports
from functools import reduce
import operator

# External imports
import torch
import torch.nn as nn
import timm


def conv_relu_bn(cin, cout):
    return [
        nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(cout),
    ]


def conv_down(cin, cout):
    return [
        nn.Conv2d(cin, cout, kernel_size=2, stride=2, padding=0),
        nn.ReLU(),
        nn.BatchNorm2d(cout),
    ]


def VanillaCNN(cfg, input_size, num_classes):

    layers = []
    cin = input_size[0]
    cout = 16
    for i in range(cfg["num_layers"]):
        layers.extend(conv_relu_bn(cin, cout))
        layers.extend(conv_relu_bn(cout, cout))
        layers.extend(conv_down(cout, 2 * cout))
        cin = 2 * cout
        cout = 2 * cout
    conv_model = nn.Sequential(*layers)

    # Compute the output size of the convolutional part
    probing_tensor = torch.zeros((1,) + input_size)
    out_cnn = conv_model(probing_tensor)  # B, K, H, W
    num_features = reduce(operator.mul, out_cnn.shape[1:], 1)
    out_layers = [nn.Flatten(start_dim=1), nn.Linear(num_features, num_classes)]
    return nn.Sequential(conv_model, *out_layers)

class TimmModel(nn.Module):
    """
    source @gemini 3 pro
    """
    def __init__(self, cfg, input_size, num_classes):
        """
        Args:
            cfg (dict): Configuration du modèle (doit contenir 'backbone')
            input_size (tuple): (C, H, W) ex: (1, 224, 224) ou (3, 224, 224)
            num_classes (int): Nombre de classes de sortie
        """
        super().__init__()
        
        # 1. Récupération des paramètres
        # Le nom du modèle dans timm (ex: 'resnet50', 'efficientnet_b0', 'convnext_tiny')
        backbone_name = cfg.get("backbone", "resnet18") 
        pretrained = cfg.get("pretrained", True)
        
        # 2. Gestion des canaux d'entrée (Grayscale vs RGB)
        # timm est par défaut en RGB (3 canaux). Si input_size[0] == 1, 
        # timm va adapter la première couche de convolution automatiquement.
        in_chans = input_size[0]

        # 3. Création du modèle via timm factory
        # num_classes=num_classes force timm à remplacer la dernière couche (FC)
        # par une couche adaptée à ton dataset.
        self.model = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=in_chans
        )

        # 4. (Optionnel) Freezing du backbone
        # Si on veut geler les poids et n'entraîner que la nouvelle tête
        if cfg.get("freeze_backbone", False):
            for param in self.model.parameters():
                param.requires_grad = False
            # On débloque uniquement la dernière couche (souvent appelée 'fc' ou 'classifier' ou 'head')
            # timm fournit une méthode get_classifier() pour récupérer la tête génériquement
            for param in self.model.get_classifier().parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.model(x)

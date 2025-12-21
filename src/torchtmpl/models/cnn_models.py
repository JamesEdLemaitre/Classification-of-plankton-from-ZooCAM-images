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


def VanillaCNNV2(cfg, input_size, num_classes):
    """
    VanillaCNN with more options to configure it
    """
    layers = []
    cin = input_size[0]
    cout = cfg["layer1_size"]
    for i in range(cfg["num_layers"]):
        layers.extend(conv_relu_bn(cin, cout))
        layers.extend(conv_relu_bn(cout, cout))
        layers.extend(conv_down(cout, cout))
        cin = cout
        cout = 2 * cin
    conv_model = nn.Sequential(*layers)

    # Compute the output size of the convolutional part
    probing_tensor = torch.zeros((1,) + input_size)
    out_cnn = conv_model(probing_tensor)  # B, K, H, W
    num_features = reduce(operator.mul, out_cnn.shape[1:], 1)
    out_layers = [nn.Flatten(start_dim=1), nn.Linear(num_features, num_classes)]
    return nn.Sequential(conv_model, *out_layers)

def TimmClassifier(cfg, input_size, num_classes):
    """
    Classificateur basé sur n'importe quel modèle timm avec support du freeze/unfreeze
    """
    model_name = cfg.get("model_name", "resnext50_32x4d")
    pretrained = cfg.get("pretrained", True)
    freeze_backbone = cfg.get("freeze_backbone", False)
    head_hidden_dims = cfg.get("head_hidden_dims", [])
    dropout = cfg.get("dropout", 0.0)
    
    # Créer le backbone timm
    backbone = timm.create_model(
        model_name, 
        pretrained=pretrained,
        num_classes=0,
        global_pool='avg'
    )
    
    # Geler le backbone si demandé (Phase 1)
    if freeze_backbone:
        for param in backbone.parameters():
            param.requires_grad = False
    
    # Obtenir la dimension de sortie du backbone
    with torch.no_grad():
        dummy_input = torch.zeros((1,) + input_size)
        backbone_output = backbone(dummy_input)
        num_features = backbone_output.shape[1]
    
    # Construire la tête de classification
    head_layers = []
    
    if head_hidden_dims:
        in_features = num_features
        for hidden_dim in head_hidden_dims:
            head_layers.extend([
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(hidden_dim)
            ])
            if dropout > 0:
                head_layers.append(nn.Dropout(dropout))
            in_features = hidden_dim
        head_layers.append(nn.Linear(in_features, num_classes))
    else:
        if dropout > 0:
            head_layers.append(nn.Dropout(dropout))
        head_layers.append(nn.Linear(num_features, num_classes))
    
    head = nn.Sequential(*head_layers)
    
    # ✅ MODÈLE AVEC MÉTHODES FREEZE/UNFREEZE
    class TimmClassifierModel(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head
        
        def forward(self, x):
            features = self.backbone(x)
            return self.head(features)
        
        def freeze_backbone(self):
            """Gèle tous les paramètres du backbone"""
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        def unfreeze_backbone(self, num_layers=None):
            """
            Dégèle le backbone (totalement ou partiellement)
            
            Args:
                num_layers (int, optional): Nombre de couches à dégeler depuis la fin.
                                           Si None, dégèle tout le backbone.
            """
            if num_layers is None:
                # Dégeler tout le backbone
                for param in self.backbone.parameters():
                    param.requires_grad = True
            else:
                # Dégeler seulement les dernières couches
                # Note: ceci est un exemple simplifié, l'implémentation exacte
                # dépend de l'architecture du modèle timm
                all_params = list(self.backbone.parameters())
                total_layers = len(all_params)
                
                # Geler toutes les couches d'abord
                for param in all_params:
                    param.requires_grad = False
                
                # Dégeler les dernières num_layers
                start_idx = max(0, total_layers - num_layers)
                for param in all_params[start_idx:]:
                    param.requires_grad = True
    
    return TimmClassifierModel(backbone, head)

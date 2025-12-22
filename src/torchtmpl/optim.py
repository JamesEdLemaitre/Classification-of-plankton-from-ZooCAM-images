# coding: utf-8

# External imports
import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=0.5, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Poids par classe [num_classes]
        self.gamma = gamma  # Facteur de modulation
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # Calcul de la CrossEntropy standard (sans réduction)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Calcul des probabilités prédites: pt = exp(-ce_loss) = p(classe_correcte)
        # pt est proche de 1 pour les exemples faciles (bien classés)
        # pt est proche de 0 pour les exemples difficiles (mal classés)
        pt = torch.exp(-ce_loss)
        
        # Terme focal: (1 - pt)^gamma
        # - Si pt proche de 1 (facile): (1-pt)^gamma ≈ 0 → loss réduite
        # - Si pt proche de 0 (difficile): (1-pt)^gamma ≈ 1 → loss conservée
        focal_weight = (1 - pt) ** self.gamma
        
        # Focal loss = focal_weight * ce_loss
        focal_loss = focal_weight * ce_loss
        
        # Appliquer les poids par classe (alpha) si fournis
        if self.alpha is not None:
            # alpha doit être de shape [num_classes]
            if self.alpha.dim() == 1:
                # Sélectionner le poids alpha pour chaque exemple selon sa classe
                alpha_t = self.alpha[targets]
                focal_loss = alpha_t * focal_loss
            else:
                raise ValueError("alpha doit être un tensor 1D de shape [num_classes]")
        
        # Réduction finale
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


def get_loss(lossname, class_weights=None, device='cuda'):
    if lossname == "FocalLoss":
        print("USING FOCAL LOSS...")
        alpha = class_weights.to(device) if class_weights is not None else None
        return FocalLoss(alpha=alpha, gamma=2.0)
    else:
        return eval(f"nn.{lossname}()")


def get_optimizer(cfg, params):
    params_dict = cfg["params"]
    exec(f"global optim; optim = torch.optim.{cfg['algo']}(params, **params_dict)")
    return optim

import torch.optim as optim_torch

def get_scheduler(config, optimizer):
    """
    Build a learning rate scheduler.
    
    Args:
        config: dict with 'class' and parameters
        optimizer: PyTorch optimizer
    
    Returns:
        PyTorch scheduler
    """
    scheduler_class = config["class"]
    params = {k: v for k, v in config.items() if k != "class"}
    
    if scheduler_class == "StepLR":
        return optim_torch.lr_scheduler.StepLR(optimizer, **params)
    
    elif scheduler_class == "MultiStepLR":
        return optim_torch.lr_scheduler.MultiStepLR(optimizer, **params)
    
    elif scheduler_class == "ExponentialLR":
        return optim_torch.lr_scheduler.ExponentialLR(optimizer, **params)
    
    elif scheduler_class == "CosineAnnealingLR":
        return optim_torch.lr_scheduler.CosineAnnealingLR(optimizer, **params)
    
    elif scheduler_class == "ReduceLROnPlateau":
        return optim_torch.lr_scheduler.ReduceLROnPlateau(optimizer, **params)
    
    elif scheduler_class == "OneCycleLR":
        return optim_torch.lr_scheduler.OneCycleLR(optimizer, **params)
    
    elif scheduler_class == "CosineAnnealingWarmRestarts":
        return optim_torch.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **params)
    
    else:
        raise ValueError(f"Unknown scheduler class: {scheduler_class}")

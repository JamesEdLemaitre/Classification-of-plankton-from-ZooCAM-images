# coding: utf-8

# External imports
import torch
import torch.nn as nn


def get_loss(lossname):
    return eval(f"nn.{lossname}()")


def get_optimizer(cfg, params):
    params_dict = cfg["params"]
    exec(f"global optim; optim = torch.optim.{cfg['algo']}(params, **params_dict)")
    return optim


import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

def get_scheduler(config, optimizer):
    """
    Crée un scheduler dynamiquement basé sur la config.
    config exemple: {'class': 'StepLR', 'step_size': 7, 'gamma': 0.1}
    """
    if not config:
        return None
        
    config = config.copy() # Copie pour ne pas modifier le dictionnaire original
    name = config.pop("class") # On retire le nom pour ne garder que les arguments
    
    if name == "ReduceLROnPlateau":
        return lr_scheduler.ReduceLROnPlateau(optimizer, **config)
        
    elif name == "StepLR":
        return lr_scheduler.StepLR(optimizer, **config)
        
    elif name == "CosineAnnealingLR":
        return lr_scheduler.CosineAnnealingLR(optimizer, **config)
        
    elif name == "OneCycleLR":
        # OneCycle a besoin de steps_per_epoch, souvent calculé hors du yaml
        # Ici on suppose qu'ils sont passés dans la config si besoin, 
        # ou on gère les cas particuliers.
        return lr_scheduler.OneCycleLR(optimizer, **config)
    
    else:
        raise ValueError(f"Scheduler {name} not supported in optim.py")

# Standard imports
from functools import reduce
import operator

# External imports
import torch
import torch.nn as nn
import ssl
import torch
import timm 





def DINOv3Classifier(cfg, input_size, num_classes):
    """
    DINOv3 via timm Hugging Face Hub (contourne les problèmes de proxy!)
    
    Args:
        cfg: dict with 'model_size', 'freeze_backbone', 'head_hidden_dims', 'dropout'
        input_size: tuple (C, H, W)
        num_classes: int
    """
    
    model_size = cfg.get("model_size", "small")
    freeze_backbone = cfg.get("freeze_backbone", True)
    head_hidden_dims = cfg.get("head_hidden_dims", [])
    dropout = cfg.get("dropout", 0.0)
    
    # DINOv3 models from timm via HF Hub
    model_names = {
        'small': 'vit_small_patch16_dinov3.lvd1689m',
        'small_plus': 'vit_small_plus_patch16_dinov3.lvd1689m',
        'base': 'vit_base_patch16_dinov3.lvd1689m',
        'large': 'vit_large_patch16_dinov3.lvd1689m',
        'giant': 'vit_giant_patch16_dinov3.lvd1689m',
    }
    
    model_name = model_names.get(model_size, model_names['small'])
    
    print(f"Loading {model_name} from timm/HF Hub...")
    
    # Load backbone WITHOUT classification head
    backbone = timm.create_model(model_name, pretrained=True)

    
    embed_dim = backbone.num_features
    print(f"Embedding dimension: {embed_dim}")
    
    if freeze_backbone:
        for param in backbone.parameters():
            param.requires_grad = False
    
    # Build custom classification head
    layers = []
    in_dim = embed_dim
    
    for hidden_dim in head_hidden_dims:
        layers += [
            nn.Linear(in_dim, hidden_dim), 
            nn.ReLU(), 
            nn.BatchNorm1d(hidden_dim), 
            nn.Dropout(dropout)
        ]
        in_dim = hidden_dim
    
    layers.append(nn.Linear(in_dim, num_classes))
    head = nn.Sequential(*layers)
    
    # Wrapper model
    class DINOv3WithHead(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head
        
        def forward(self, x):
            features = self.backbone(x)  # [B, embed_dim]
            return self.head(features)    # [B, num_classes]
    
    model = DINOv3WithHead(backbone, head)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Frozen params: {total_params - trainable_params:,}")
    
    return model

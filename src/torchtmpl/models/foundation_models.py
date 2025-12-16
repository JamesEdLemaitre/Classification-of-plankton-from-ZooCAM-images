
# Standard imports
from functools import reduce
import operator

# External imports
import torch
import torch.nn as nn
import ssl
import torch





def DINOv3Classifier(cfg, input_size, num_classes):
    """
    DINOv3 (via torch.hub) with custom classification head

    Args:
        cfg: dict with 'model_size', 'freeze_backbone', 'head_hidden_dims', 'dropout'
        input_size: tuple (C, H, W)
        num_classes: int
    """
    import torch
    import torch.nn as nn

    # Config
    model_size = cfg.get("model_size", "small")
    freeze_backbone = cfg.get("freeze_backbone", True)
    head_hidden_dims = cfg.get("head_hidden_dims", [])
    dropout = cfg.get("dropout", 0.0)

    # DINOv3 model names from torch.hub
    model_names = {
        'tiny': 'dinov3_convnext_tiny',
        'small': 'dinov3_vits16',
        'small_plus': 'dinov3_vits16plus',
        'base': 'dinov3_vitb16',
        'large': 'dinov3_vitl16',
        'large_plus': 'dinov3_vitl16plus',
        'huge': 'dinov3_vith16plus',
        'giant': 'dinov3_vit7b16',  # 7B parameters!
    }
    # Désactiver la vérification SSL
    ssl._create_default_https_context = ssl._create_unverified_context
    
    model_name = model_names.get(model_size, 'dinov3_vits16')
    print(f"🔄 Loading {model_name} from torch.hub (facebookresearch/dinov3)...")
    
    backbone = torch.hub.load('facebookresearch/dinov3', model_name)
    
    # Get embedding dimension
    embed_dim = backbone.embed_dim
    print(f"📐 Embedding dimension: {embed_dim}")

    # Freeze if needed
    if freeze_backbone:
        print(f"❄️  Freezing backbone")
        for param in backbone.parameters():
            param.requires_grad = False
    else:
        print(f"🔥 Fine-tuning backbone")

    # Build classification head
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

    # Wrapper
    class DINOv3WithHead(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head

        def forward(self, x):
            features = self.backbone(x)
            return self.head(features)

    model = DINOv3WithHead(backbone, head)

    # Print info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Total params: {total_params:,}")
    print(f"🎯 Trainable params: {trainable_params:,}")
    print(f"❄️  Frozen params: {total_params - trainable_params:,}")

    return model

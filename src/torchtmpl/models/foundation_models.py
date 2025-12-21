
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

# def DINOv3Classifier_bystep(cfg, input_size, num_classes):
#     """
#     DINOv3 avec support pour unfreeze progressif
    
#     Args:
#         cfg: dict with 'model_size', 'freeze_backbone', 'head_hidden_dims', 
#              'dropout', 'unfreeze_after_epochs', 'unfreeze_layers'
#     """
#     model_size = cfg.get("model_size", "small")
#     head_hidden_dims = cfg.get("head_hidden_dims", [])
#     dropout = cfg.get("dropout", 0.0)
    
#     model_names = {
#         'small': 'vit_small_patch16_dinov3.lvd1689m',
#         'small_plus': 'vit_small_plus_patch16_dinov3.lvd1689m',
#         'base': 'vit_base_patch16_dinov3.lvd1689m',
#         'large': 'vit_large_patch16_dinov3.lvd1689m',
#         'giant': 'vit_giant_patch16_dinov3.lvd1689m',
#     }
    
#     model_name = model_names.get(model_size, model_names['small'])
#     print(f"Loading {model_name} from timm/HF Hub...")
    
#     backbone = timm.create_model(model_name, pretrained=True)
#     embed_dim = backbone.num_features
#     print(f"Embedding dimension: {embed_dim}")
    
#     # Freeze tout le backbone au départ
#     for param in backbone.parameters():
#         param.requires_grad = False
    
#     # Build custom head
#     layers = []
#     in_dim = embed_dim
    
#     for hidden_dim in head_hidden_dims:
#         layers += [
#             nn.Linear(in_dim, hidden_dim), 
#             nn.ReLU(), 
#             nn.BatchNorm1d(hidden_dim), 
#             nn.Dropout(dropout)
#         ]
#         in_dim = hidden_dim
    
#     layers.append(nn.Linear(in_dim, num_classes))
#     head = nn.Sequential(*layers)
    
#     class DINOv3WithProgressiveUnfreeze(nn.Module):
#         def __init__(self, backbone, head, cfg):
#             super().__init__()
#             self.backbone = backbone
#             self.head = head
#             self.cfg = cfg

#             if hasattr(self.backbone, 'head'):
#                 self.backbone.head = nn.Identity()  # ← CRUCIAL !
#             if hasattr(self.backbone, 'fc'):
#                 self.backbone.fc = nn.Identity()
                
#         def forward(self, x):
#             features = self.backbone.forward_features(x)

#             # DINOv3 retourne [B, num_patches, embed_dim]
#             # Prendre le [CLS] token (premier token)
#             if features.dim() == 3:  # [B, num_patches, embed_dim]
#                 features = features[:, 0] # [B, embed_dim]

#             return self.head(features)
        
#         def unfreeze_backbone(self, num_layers=None):
#             """
#             Unfreeze le backbone (tout ou partiellement)
            
#             Args:
#                 num_layers: None pour tout unfreeze, int pour les N derniers blocks
#             """
#             if num_layers is None or num_layers == "None":
#                 print("Unfreezing entire backbone")
#                 for param in self.backbone.parameters():
#                     param.requires_grad = True
#             else:
#                 print(f"Unfreezing last {num_layers} transformer blocks")
#                 # Pour ViT, unfreeze les derniers blocks
#                 if hasattr(self.backbone, 'blocks'):
#                     total_blocks = len(self.backbone.blocks)
#                     start_idx = max(0, total_blocks - num_layers)
#                     for i in range(start_idx, total_blocks):
#                         for param in self.backbone.blocks[i].parameters():
#                             param.requires_grad = True
#                     # Unfreeze aussi la norm finale
#                     if hasattr(self.backbone, 'norm'):
#                         for param in self.backbone.norm.parameters():
#                             param.requires_grad = True
    
#     model = DINOv3WithProgressiveUnfreeze(backbone, head, cfg)
    
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
#     print(f"Total params: {total_params:,}")
#     print(f"Trainable params (initial): {trainable_params:,}")
#     print(f"Frozen params: {total_params - trainable_params:,}")
    
#     return model

def DINOv3Classifier_bystep(cfg, input_size, num_classes):
    """
    DINOv3 avec support pour unfreeze progressif - VERSION OPTIMALE
    
    Args:
        cfg: dict with 'model_size', 'head_hidden_dims', 'dropout', etc.
    """
    model_size = cfg.get("model_size", "small")
    head_hidden_dims = cfg.get("head_hidden_dims", [])
    dropout = cfg.get("dropout", 0.0)
    use_layer_norm = cfg.get("use_layer_norm", True)  # LayerNorm > BatchNorm

    model_names = {
        'small': 'vit_small_patch16_dinov3.lvd1689m',
        'small_plus': 'vit_small_plus_patch16_dinov3.lvd1689m',
        'base': 'vit_base_patch16_dinov3.lvd1689m',
        'large': 'vit_large_patch16_dinov3.lvd1689m',
        'giant': 'vit_giant_patch16_dinov3.lvd1689m',
    }

    model_name = model_names.get(model_size, model_names['small'])
    print(f"Loading {model_name} from timm...")

    # 🔧 OPTIMISATION 1 : num_classes=0 (plus propre que Identity())
    backbone = timm.create_model(
        model_name, 
        pretrained=True,
        num_classes=0  # ← Pas de head interne
    )
    
    embed_dim = backbone.num_features
    print(f"Embedding dimension: {embed_dim}")

    # Freeze tout le backbone au départ
    for param in backbone.parameters():
        param.requires_grad = False

    # 🔧 OPTIMISATION 2 : Head moderne pour transformers
    layers = []
    in_dim = embed_dim

    # Normalisation initiale (stabilise le training)
    if len(head_hidden_dims) > 0 and use_layer_norm:
        layers.append(nn.LayerNorm(in_dim))

    for hidden_dim in head_hidden_dims:
        layers += [
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),  # ← GELU > ReLU pour transformers
            nn.Dropout(dropout)
        ]
        in_dim = hidden_dim

    layers.append(nn.Linear(in_dim, num_classes))
    head = nn.Sequential(*layers)

    # 🔧 OPTIMISATION 3 : Initialisation des poids
    def init_head_weights(m):
        if isinstance(m, nn.Linear):
            # Truncated normal (comme dans ViT)
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    head.apply(init_head_weights)

    class DINOv3WithProgressiveUnfreeze(nn.Module):
        def __init__(self, backbone, head, cfg):
            super().__init__()
            self.backbone = backbone
            self.head = head
            self.cfg = cfg
            
            # 🔧 FIX : Sécurité (au cas où num_classes=0 ne marche pas)
            if hasattr(self.backbone, 'head') and not isinstance(self.backbone.head, nn.Identity):
                self.backbone.head = nn.Identity()
            if hasattr(self.backbone, 'fc') and not isinstance(self.backbone.fc, nn.Identity):
                self.backbone.fc = nn.Identity()

        def forward(self, x):
            # Extract features
            features = self.backbone.forward_features(x)
            
            # Handle different output formats
            if features.dim() == 3:  # [B, num_patches, embed_dim]
                features = features[:, 0]  # Take [CLS] token
            elif features.dim() == 2:  # [B, embed_dim]
                pass  # Already in correct format
            else:
                raise ValueError(f"Unexpected feature shape: {features.shape}")
            
            return self.head(features)

        def unfreeze_backbone(self, num_layers=None):
            """
            Unfreeze le backbone (tout ou partiellement)
            
            Args:
                num_layers: None pour tout unfreeze, int pour les N derniers blocks
            """
            if num_layers is None or num_layers == "None":
                print("🔓 Unfreezing entire backbone")
                for param in self.backbone.parameters():
                    param.requires_grad = True
            else:
                num_layers = int(num_layers)
                print(f"🔓 Unfreezing last {num_layers} transformer blocks")
                
                if hasattr(self.backbone, 'blocks'):
                    total_blocks = len(self.backbone.blocks)
                    start_idx = max(0, total_blocks - num_layers)
                    
                    unfrozen_params = 0
                    for i in range(start_idx, total_blocks):
                        for param in self.backbone.blocks[i].parameters():
                            param.requires_grad = True
                            unfrozen_params += param.numel()
                    
                    # Unfreeze norm finale
                    if hasattr(self.backbone, 'norm'):
                        for param in self.backbone.norm.parameters():
                            param.requires_grad = True
                            unfrozen_params += param.numel()
                    
                    print(f"   → Unfrozen {unfrozen_params:,} parameters")
                else:
                    print("   ⚠️  Warning: backbone.blocks not found, unfreezing all")
                    for param in self.backbone.parameters():
                        param.requires_grad = True
        
        def get_lr_groups(self, head_lr, backbone_lr):
            """
            Retourne les parameter groups pour l'optimizer
            Permet d'avoir des learning rates différents
            """
            head_params = [p for p in self.head.parameters() if p.requires_grad]
            backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
            
            return [
                {'params': head_params, 'lr': head_lr},
                {'params': backbone_params, 'lr': backbone_lr}
            ]

    model = DINOv3WithProgressiveUnfreeze(backbone, head, cfg)

    # 🔧 OPTIMISATION 4 : Stats détaillées
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    head_params = sum(p.numel() for p in model.head.parameters())
    backbone_params = sum(p.numel() for p in model.backbone.parameters())

    print(f"\n📊 Model Statistics:")
    print(f"   Total params: {total_params:,}")
    print(f"   ├─ Backbone: {backbone_params:,} ({backbone_params/total_params*100:.1f}%)")
    print(f"   └─ Head: {head_params:,} ({head_params/total_params*100:.1f}%)")
    print(f"   Trainable (initial): {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    print(f"   Frozen: {total_params - trainable_params:,}")

    return model

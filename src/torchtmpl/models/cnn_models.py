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


class DINOv2Model(nn.Module):
    """
    DINOv2 ViT fine-tuning wrapper via timm.
    Supports 2-phase training with differential learning rates:
      - Phase 1: frozen backbone, only head trained
      - Phase 2: unfrozen backbone with layer-wise LR decay
    """
    def __init__(self, cfg, input_size, num_classes):
        super().__init__()

        backbone_name = cfg.get("backbone", "vit_base_patch14_dinov2")
        pretrained = cfg.get("pretrained", True)
        in_chans = input_size[0]
        drop_rate = cfg.get("drop_rate", 0.1)

        # Create DINOv2 model via timm
        # img_size must be passed to allow non-default resolutions
        img_size = cfg.get("img_size", input_size[1])  # H from (C, H, W)
        self.model = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=in_chans,
            drop_rate=drop_rate,
            img_size=img_size,
        )

        # Phase management
        phase = cfg.get("phase", 1)
        if phase == 1:
            # Phase 1: freeze entire backbone, train only head
            self._freeze_backbone()
        # Phase 2: all params unfrozen (handled by optimizer with differential LR)

    def _freeze_backbone(self):
        """Freeze all parameters except the classification head."""
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.get_classifier().parameters():
            param.requires_grad = True

    def _unfreeze_all(self):
        """Unfreeze all parameters for fine-tuning."""
        for param in self.model.parameters():
            param.requires_grad = True

    def get_param_groups(self, lr_backbone, lr_head, weight_decay=0.01, layer_decay=0.75):
        """
        Build parameter groups with layer-wise learning rate decay for the ViT backbone.
        Earlier layers get lower LR (layer_decay^depth), head gets lr_head.
        """
        # Separate head from backbone
        head_params = list(self.model.get_classifier().parameters())
        head_param_ids = {id(p) for p in head_params}

        # Get all ViT blocks
        if hasattr(self.model, 'blocks'):
            blocks = self.model.blocks
        else:
            blocks = []

        num_layers = len(blocks)

        # Build layer-wise groups
        param_groups = []

        # Embedding layers (patch_embed, cls_token, pos_embed) — lowest LR
        embed_params = []
        for name, param in self.model.named_parameters():
            if id(param) in head_param_ids:
                continue
            if any(k in name for k in ['patch_embed', 'cls_token', 'pos_embed']):
                embed_params.append(param)

        if embed_params:
            embed_lr = lr_backbone * (layer_decay ** num_layers)
            param_groups.append({
                'params': embed_params,
                'lr': embed_lr,
                'weight_decay': weight_decay,
                'name': 'embed'
            })

        # Block layers — increasing LR
        assigned_ids = {id(p) for p in embed_params} | head_param_ids
        for layer_idx, block in enumerate(blocks):
            block_lr = lr_backbone * (layer_decay ** (num_layers - layer_idx - 1))
            block_params = [p for p in block.parameters() if p.requires_grad and id(p) not in assigned_ids]
            if block_params:
                param_groups.append({
                    'params': block_params,
                    'lr': block_lr,
                    'weight_decay': weight_decay,
                    'name': f'block_{layer_idx}'
                })
                assigned_ids.update(id(p) for p in block_params)

        # Norm layer and other non-assigned backbone params
        remaining_params = [p for p in self.model.parameters()
                           if p.requires_grad and id(p) not in assigned_ids and id(p) not in head_param_ids]
        if remaining_params:
            param_groups.append({
                'params': remaining_params,
                'lr': lr_backbone,
                'weight_decay': weight_decay,
                'name': 'backbone_remaining'
            })

        # Head — highest LR
        trainable_head = [p for p in head_params if p.requires_grad]
        if trainable_head:
            param_groups.append({
                'params': trainable_head,
                'lr': lr_head,
                'weight_decay': 0.0,  # No weight decay on head bias typically
                'name': 'head'
            })

        return param_groups

    def forward(self, x):
        return self.model(x)


class DINOv3Model(nn.Module):
    """
    DINOv3 ViT fine-tuning wrapper via timm (patch16, RoPE, 256px native).
    Uses multi-block CLS token aggregation for richer feature extraction:
      - Registers forward hooks on the last N blocks
      - Concatenates CLS tokens from those blocks
      - Feeds the concatenated representation through a custom MLP head

    Supports 2-phase training:
      Phase 1: frozen backbone, only head trained
      Phase 2: unfrozen backbone with layer-wise LR decay
    """

    def __init__(self, cfg, input_size, num_classes):
        super().__init__()

        backbone_name = cfg.get("backbone", "vit_base_patch16_dinov3.lvd1689m")
        pretrained = cfg.get("pretrained", True)
        in_chans = input_size[0]
        img_size = cfg.get("img_size", input_size[1])

        # Multi-block feature extraction config
        self.n_last_blocks = cfg.get("n_last_blocks", 4)
        head_hidden_dim = cfg.get("head_hidden_dim", 1024)
        drop_rate = cfg.get("drop_rate", 0.1)

        # Create backbone WITHOUT classification head (num_classes=0)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,  # Remove timm's built-in head
            in_chans=in_chans,
            img_size=img_size,
        )

        # Determine embedding dimension from backbone
        self.embed_dim = self.backbone.embed_dim  # e.g. 768 for ViT-B
        num_blocks = len(self.backbone.blocks)
        assert self.n_last_blocks <= num_blocks, (
            f"n_last_blocks={self.n_last_blocks} > num_blocks={num_blocks}"
        )

        # Register forward hooks on the last N blocks to capture CLS tokens
        self._hook_features = {}
        self._hooks = []
        hook_block_indices = list(range(num_blocks - self.n_last_blocks, num_blocks))
        for idx in hook_block_indices:
            hook = self.backbone.blocks[idx].register_forward_hook(
                self._make_hook(idx)
            )
            self._hooks.append(hook)
        self._hook_block_indices = hook_block_indices

        # Custom MLP classification head
        # Input: concat of CLS tokens from last N blocks → embed_dim * n_last_blocks
        concat_dim = self.embed_dim * self.n_last_blocks
        self.head = nn.Sequential(
            nn.LayerNorm(concat_dim),
            nn.Linear(concat_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(head_hidden_dim, num_classes),
        )

        # Phase management
        phase = cfg.get("phase", 1)
        if phase == 1:
            self._freeze_backbone()

    def _make_hook(self, block_idx):
        """Create a forward hook that captures the CLS token output of a block."""
        def hook_fn(module, input, output):
            # output shape: [B, num_tokens, embed_dim]
            # CLS token is at position 0
            self._hook_features[block_idx] = output[:, 0]  # [B, embed_dim]
        return hook_fn

    def _freeze_backbone(self):
        """Freeze all backbone parameters. Head remains trainable."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _unfreeze_all(self):
        """Unfreeze all parameters (backbone + head) for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        for param in self.head.parameters():
            param.requires_grad = True

    def get_param_groups(self, lr_backbone, lr_head, weight_decay=0.01, layer_decay=0.75):
        """
        Build parameter groups with layer-wise LR decay for the ViT backbone.
        Earlier layers get lower LR (layer_decay^depth), head gets lr_head.
        """
        head_params = list(self.head.parameters())
        head_param_ids = {id(p) for p in head_params}

        blocks = self.backbone.blocks
        num_layers = len(blocks)

        param_groups = []

        # Embedding layers (patch_embed, cls_token, reg_token — DINOv3 uses RoPE,
        # so pos_embed may or may not exist, but we include it defensively)
        embed_params = []
        for name, param in self.backbone.named_parameters():
            if any(k in name for k in ['patch_embed', 'cls_token', 'pos_embed', 'reg_token']):
                embed_params.append(param)

        if embed_params:
            embed_lr = lr_backbone * (layer_decay ** num_layers)
            param_groups.append({
                'params': embed_params,
                'lr': embed_lr,
                'weight_decay': weight_decay,
                'name': 'embed'
            })

        # Block layers — increasing LR from early to late
        assigned_ids = {id(p) for p in embed_params} | head_param_ids
        for layer_idx, block in enumerate(blocks):
            block_lr = lr_backbone * (layer_decay ** (num_layers - layer_idx - 1))
            block_params = [p for p in block.parameters()
                           if p.requires_grad and id(p) not in assigned_ids]
            if block_params:
                param_groups.append({
                    'params': block_params,
                    'lr': block_lr,
                    'weight_decay': weight_decay,
                    'name': f'block_{layer_idx}'
                })
                assigned_ids.update(id(p) for p in block_params)

        # Norm layer and other non-assigned backbone params (e.g. final norm)
        remaining_params = [p for p in self.backbone.parameters()
                           if p.requires_grad and id(p) not in assigned_ids
                           and id(p) not in head_param_ids]
        if remaining_params:
            param_groups.append({
                'params': remaining_params,
                'lr': lr_backbone,
                'weight_decay': weight_decay,
                'name': 'backbone_remaining'
            })

        # Head — highest LR, no weight decay on bias
        trainable_head = [p for p in head_params if p.requires_grad]
        if trainable_head:
            param_groups.append({
                'params': trainable_head,
                'lr': lr_head,
                'weight_decay': 0.0,
                'name': 'head'
            })

        return param_groups

    def forward(self, x):
        # Clear hook buffer
        self._hook_features.clear()

        # Run backbone (hooks will capture CLS tokens from last N blocks)
        _ = self.backbone(x)  # output is pooled features; we don't use it

        # Concatenate CLS tokens from hooked blocks in order
        cls_tokens = [self._hook_features[idx] for idx in self._hook_block_indices]
        concat_features = torch.cat(cls_tokens, dim=1)  # [B, embed_dim * n_last_blocks]

        # Pass through custom head
        return self.head(concat_features)

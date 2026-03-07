import torch
import torch.nn as nn
from timm.loss import LabelSmoothingCrossEntropy

class FocalCrossEntropyMix(nn.Module):
    def __init__(self, weight=None, gamma=2.0, lambda_ce=1.0, lambda_focal=0.5):
        """
        Args:
            weight: Poids des classes (à laisser None si tu utilises un WeightedSampler)
            gamma: Paramètre de focus de la Focal Loss (plus il est haut, plus on focus sur les durs)
            lambda_ce: Coefficient de la Cross Entropy classique
            lambda_focal: Coefficient de la Focal Loss
        """
        super().__init__()
        self.gamma = gamma
        self.lambda_ce = lambda_ce
        self.lambda_focal = lambda_focal
        
        # On garde reduction='none' pour pouvoir manipuler la loss pixel par pixel avant la moyenne
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none')

    def forward(self, inputs, targets):
        # 1. Calcul de la Cross Entropy standard pour chaque exemple
        ce_loss = self.ce(inputs, targets)
        
        # 2. Calcul de la probabilité pt (exp(-loss))
        pt = torch.exp(-ce_loss)
        
        # 3. Calcul de la composante Focal
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        # 4. Mix pondéré : 1.0 * CE + 0.5 * Focal
        loss = (self.lambda_ce * ce_loss) + (self.lambda_focal * focal_loss)
        
        # Retourne la moyenne sur le batch
        return loss.mean()


class SoftTargetCrossEntropy(nn.Module):
    """
    Cross-entropy loss that accepts soft (mixed) targets from Mixup/CutMix.
    Expects inputs as raw logits and targets as probabilities (one-hot mixed).
    """
    def forward(self, inputs, targets):
        log_probs = torch.nn.functional.log_softmax(inputs, dim=-1)
        loss = -(targets * log_probs).sum(dim=-1)
        return loss.mean()


class PolyFocalLoss(nn.Module):
    """
    Poly-1 modification of Focal Loss (PolyLoss, ICLR 2022).
    Adds a polynomial term that pushes the loss higher for misclassified samples.
    loss = FocalLoss + epsilon * (1 - pt)^(gamma+1)
    """
    def __init__(self, gamma=2.0, epsilon=1.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.epsilon = epsilon
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal = ((1 - pt) ** self.gamma) * ce_loss
        poly_term = self.epsilon * (1 - pt) ** (self.gamma + 1)
        return (focal + poly_term).mean()


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-class classification (adapted from ASL, ICCV 2021).
    Applies asymmetric focusing: harder down-weighting of easy negatives than
    easy positives. gamma_neg > gamma_pos shifts focus to hard examples.
    """
    def __init__(self, gamma_pos=0.0, gamma_neg=4.0, clip=0.05, weight=None):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.weight = weight

    def forward(self, inputs, targets):
        num_classes = inputs.shape[1]
        # One-hot encode targets
        targets_oh = torch.zeros_like(inputs).scatter_(1, targets.unsqueeze(1), 1.0)

        probs = torch.softmax(inputs, dim=-1)
        probs = torch.clamp(probs, min=1e-7, max=1.0)

        # Positive and negative probabilities
        pos_probs = probs * targets_oh
        neg_probs = probs * (1 - targets_oh)

        # Asymmetric clipping on negative probs (probability margin)
        if self.clip > 0:
            neg_probs = (neg_probs + self.clip).clamp(max=1.0)

        # Focal-like modulation
        pos_loss = -targets_oh * ((1 - pos_probs) ** self.gamma_pos) * torch.log(probs.clamp(min=1e-7))
        neg_loss = -(1 - targets_oh) * (neg_probs ** self.gamma_neg) * torch.log((1 - probs).clamp(min=1e-7))

        loss = pos_loss + neg_loss
        if self.weight is not None:
            w = self.weight[targets].unsqueeze(1)
            loss = loss * w
        return loss.sum(dim=-1).mean()


class DiceCELoss(nn.Module):
    """
    Dice Loss + Cross Entropy for multi-class classification.
    Dice component is computed per-class and averaged (macro).
    Good for heavily imbalanced datasets.
    """
    def __init__(self, lambda_dice=1.0, lambda_ce=1.0, smooth=1.0, weight=None):
        super().__init__()
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce
        self.smooth = smooth
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)

        # Dice per class
        num_classes = inputs.shape[1]
        probs = torch.softmax(inputs, dim=-1)  # [B, C]
        targets_oh = torch.zeros_like(probs).scatter_(1, targets.unsqueeze(1), 1.0)  # [B, C]

        intersection = (probs * targets_oh).sum(dim=0)  # [C]
        cardinality = probs.sum(dim=0) + targets_oh.sum(dim=0)  # [C]
        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1.0 - dice_per_class.mean()

        return self.lambda_ce * ce_loss + self.lambda_dice * dice_loss


class GHMCLoss(nn.Module):
    """
    Gradient Harmonizing Mechanism for Classification (AAAI 2019).
    Bins gradient norms and re-weights samples to balance easy/hard contributions.
    Prevents both easy-sample dominance and outlier/noisy-sample dominance.
    """
    def __init__(self, bins=10, momentum=0.75, weight=None):
        super().__init__()
        self.bins = bins
        self.momentum = momentum
        self.weight = weight
        edges = torch.arange(bins + 1).float() / bins
        self.register_buffer('edges', edges)
        self.register_buffer('acc_sum', torch.zeros(bins))

    def forward(self, inputs, targets):
        num_classes = inputs.shape[1]
        targets_oh = torch.zeros_like(inputs).scatter_(1, targets.unsqueeze(1), 1.0)

        probs = torch.softmax(inputs, dim=-1)
        # Gradient norm = |p - target| (gradient of CE w.r.t. logit ≈ p - y)
        g = torch.abs(probs - targets_oh).detach()

        # Bin the gradients
        tot = inputs.numel()
        weights = torch.zeros_like(inputs)
        for i in range(self.bins):
            inds = (g >= self.edges[i]) & (g < self.edges[i + 1])
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if self.momentum > 0 and self.acc_sum[i] > 0:
                    num_in_bin = self.momentum * self.acc_sum[i] + (1 - self.momentum) * num_in_bin
                self.acc_sum[i] = num_in_bin
                weights[inds] = tot / num_in_bin

        # Weighted CE
        log_probs = torch.log_softmax(inputs, dim=-1)
        loss = -(targets_oh * log_probs * weights)
        return loss.sum() / max(weights.sum(), 1.0)


def get_loss(loss_config, device='cuda', class_weights=None):
    if class_weights is not None:
        class_weights = class_weights.to(device)

    # Accept both string and dict config
    if isinstance(loss_config, str):
        loss_name = loss_config
        loss_params = {}
    else:
        loss_name = loss_config.get("class", loss_config.get("name"))
        loss_params = loss_config
    
    if loss_name in ("MixedLoss", "FocalCrossEntropyMix"):
        return FocalCrossEntropyMix(weight=class_weights)
    elif loss_name == "SoftTargetCrossEntropy":
        return SoftTargetCrossEntropy()
    elif loss_name == "LabelSmoothingCE":
        smoothing = loss_params.get("smoothing", 0.1)
        return LabelSmoothingCrossEntropy(smoothing=smoothing)
    elif loss_name == "PolyFocalLoss":
        gamma = loss_params.get("gamma", 2.0)
        epsilon = loss_params.get("epsilon", 1.0)
        return PolyFocalLoss(gamma=gamma, epsilon=epsilon, weight=class_weights)
    elif loss_name == "AsymmetricLoss":
        gamma_pos = loss_params.get("gamma_pos", 0.0)
        gamma_neg = loss_params.get("gamma_neg", 4.0)
        clip = loss_params.get("clip", 0.05)
        return AsymmetricLoss(gamma_pos=gamma_pos, gamma_neg=gamma_neg, clip=clip, weight=class_weights)
    elif loss_name == "DiceCELoss":
        lambda_dice = loss_params.get("lambda_dice", 1.0)
        lambda_ce = loss_params.get("lambda_ce", 1.0)
        smooth = loss_params.get("smooth", 1.0)
        return DiceCELoss(lambda_dice=lambda_dice, lambda_ce=lambda_ce, smooth=smooth, weight=class_weights)
    elif loss_name == "GHMCLoss":
        bins = loss_params.get("bins", 10)
        momentum = loss_params.get("momentum", 0.75)
        l = GHMCLoss(bins=bins, momentum=momentum, weight=class_weights)
        l.to(device)
        return l
    else:
        return nn.CrossEntropyLoss(weight=class_weights)


def get_mixup_fn(mixup_config, num_classes):
    """
    Build a timm Mixup function from config.
    Config example:
        mixup:
          mixup_alpha: 0.8
          cutmix_alpha: 1.0
          prob: 1.0
          switch_prob: 0.5
          label_smoothing: 0.1
    """
    try:
        from timm.data import Mixup
    except ImportError:
        raise ImportError("timm is required for Mixup/CutMix. Install with: pip install timm")

    mixup_fn = Mixup(
        mixup_alpha=mixup_config.get("mixup_alpha", 0.8),
        cutmix_alpha=mixup_config.get("cutmix_alpha", 1.0),
        cutmix_minmax=mixup_config.get("cutmix_minmax", None),
        prob=mixup_config.get("prob", 1.0),
        switch_prob=mixup_config.get("switch_prob", 0.5),
        mode=mixup_config.get("mode", "batch"),
        label_smoothing=mixup_config.get("label_smoothing", 0.1),
        num_classes=num_classes,
    )
    return mixup_fn



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
        
    elif name == "CosineAnnealingWarmRestarts":
        return lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **config)

    elif name == "OneCycleLR":
        # OneCycle a besoin de steps_per_epoch, souvent calculé hors du yaml
        # Ici on suppose qu'ils sont passés dans la config si besoin, 
        # ou on gère les cas particuliers.
        return lr_scheduler.OneCycleLR(optimizer, **config)
    
    else:
        raise ValueError(f"Scheduler {name} not supported in optim.py")
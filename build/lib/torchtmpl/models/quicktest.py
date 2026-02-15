import torch
# Liste tous les modèles disponibles
available = torch.hub.list('facebookresearch/dinov3')
print("\n=== Modèles torch.hub facebookresearch/dinov3 ===")
for m in available:
    print(m)

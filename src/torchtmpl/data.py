# coding: utf-8

# Standard imports
import logging
import random
import os

# External imports
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
from torchvision import transforms
from torchvision.utils import make_grid
import torchvision.transforms.functional as F
import PIL
from PIL import Image

import matplotlib.pyplot as plt

from torch.utils.data import WeightedRandomSampler

def show_image(X):
    num_c = X.shape[0]
    plt.figure()
    plt.imshow(X[0] if num_c == 1 else X.permute(1, 2, 0))
    plt.show()

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

def show_class_distribution(dataset, results_save_path = "./data_analysis"):
    """Affiche la distribution des classes dans le dataset"""
    classes = dataset.classes
    counts = [0] * len(classes)
    
    # Compter les échantillons par classe
    for _, label in dataset:
        counts[label] += 1
    
    # Afficher
    plt.figure(figsize=(10, 5))
    plt.bar(classes, counts)
    plt.xlabel('Classes')
    plt.ylabel('Nombre d\'images')
    plt.title('Distribution des classes')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{results_save_path}/class_distribution.png")
    
    # Afficher aussi les chiffres
    for cls, count in zip(classes, counts):
        print(f"{cls}: {count}")

def show_images_information(dataset, results_save_path="./data_analysis"):
    """
    Analyse les dimensions des images du dataset et génère des statistiques
    par classe et globales. Sauvegarde tout dans results_save_path.
    """
    import os
    from collections import defaultdict
    from PIL import Image
    
    os.makedirs(results_save_path, exist_ok=True)
    
    print("📊 Analyzing image dimensions...")
    
    # Structures pour stocker les infos
    all_widths = []
    all_heights = []
    all_ratios = []
    
    class_stats = defaultdict(lambda: {
        'widths': [],
        'heights': [],
        'ratios': [],
        'count': 0
    })
    
    # Parcourir toutes les images
    for idx, (img_tensor, label) in enumerate(dataset):
        # Récupérer l'image PIL originale (avant transform)
        img_path, _ = dataset.imgs[idx]
        img = Image.open(img_path)
        width, height = img.size
        ratio = width / height
        
        # Stats globales
        all_widths.append(width)
        all_heights.append(height)
        all_ratios.append(ratio)
        
        # Stats par classe
        class_name = dataset.classes[label]
        class_stats[class_name]['widths'].append(width)
        class_stats[class_name]['heights'].append(height)
        class_stats[class_name]['ratios'].append(ratio)
        class_stats[class_name]['count'] += 1
        
        if (idx + 1) % 10000 == 0:
            print(f"  Processed {idx + 1}/{len(dataset)} images...")
    
    print(f"✅ Analyzed {len(dataset)} images\n")
    
    # ═══════════════════════════════════════════════════════
    # 1. STATISTIQUES GLOBALES
    # ═══════════════════════════════════════════════════════
    
    with open(f"{results_save_path}/images_stats.txt", 'w') as f:
        f.write("="*60 + "\n")
        f.write("GLOBAL IMAGE STATISTICS\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Total images: {len(dataset)}\n\n")
        
        f.write("WIDTH:\n")
        f.write(f"  Min:    {min(all_widths)} px\n")
        f.write(f"  Max:    {max(all_widths)} px\n")
        f.write(f"  Mean:   {np.mean(all_widths):.1f} px\n")
        f.write(f"  Median: {np.median(all_widths):.1f} px\n")
        f.write(f"  Std:    {np.std(all_widths):.1f} px\n\n")
        
        f.write("HEIGHT:\n")
        f.write(f"  Min:    {min(all_heights)} px\n")
        f.write(f"  Max:    {max(all_heights)} px\n")
        f.write(f"  Mean:   {np.mean(all_heights):.1f} px\n")
        f.write(f"  Median: {np.median(all_heights):.1f} px\n")
        f.write(f"  Std:    {np.std(all_heights):.1f} px\n\n")
        
        f.write("ASPECT RATIO (width/height):\n")
        f.write(f"  Min:    {min(all_ratios):.3f}\n")
        f.write(f"  Max:    {max(all_ratios):.3f}\n")
        f.write(f"  Mean:   {np.mean(all_ratios):.3f}\n")
        f.write(f"  Median: {np.median(all_ratios):.3f}\n")
        f.write(f"  Std:    {np.std(all_ratios):.3f}\n\n")
        
        # Catégorisation des ratios
        portrait = sum(1 for r in all_ratios if r < 0.95)
        square = sum(1 for r in all_ratios if 0.95 <= r <= 1.05)
        landscape = sum(1 for r in all_ratios if r > 1.05)
        
        f.write("ORIENTATION:\n")
        f.write(f"  Portrait (ratio < 0.95):  {portrait:6d} ({100*portrait/len(all_ratios):5.2f}%)\n")
        f.write(f"  Square (0.95 ≤ ratio ≤ 1.05): {square:6d} ({100*square/len(all_ratios):5.2f}%)\n")
        f.write(f"  Landscape (ratio > 1.05): {landscape:6d} ({100*landscape/len(all_ratios):5.2f}%)\n\n")
        
        # ═══════════════════════════════════════════════════════
        # 2. STATISTIQUES PAR CLASSE
        # ═══════════════════════════════════════════════════════
        
        f.write("\n" + "="*60 + "\n")
        f.write("STATISTICS BY CLASS\n")
        f.write("="*60 + "\n\n")
        
        for class_name in sorted(class_stats.keys()):
            stats = class_stats[class_name]
            f.write(f"\n{'─'*60}\n")
            f.write(f"CLASS: {class_name}\n")
            f.write(f"{'─'*60}\n")
            f.write(f"Count: {stats['count']}\n\n")
            
            f.write("Width:\n")
            f.write(f"  Min: {min(stats['widths'])} | Max: {max(stats['widths'])} | ")
            f.write(f"Mean: {np.mean(stats['widths']):.1f} | Std: {np.std(stats['widths']):.1f}\n")
            
            f.write("Height:\n")
            f.write(f"  Min: {min(stats['heights'])} | Max: {max(stats['heights'])} | ")
            f.write(f"Mean: {np.mean(stats['heights']):.1f} | Std: {np.std(stats['heights']):.1f}\n")
            
            f.write("Aspect Ratio:\n")
            f.write(f"  Min: {min(stats['ratios']):.3f} | Max: {max(stats['ratios']):.3f} | ")
            f.write(f"Mean: {np.mean(stats['ratios']):.3f} | Std: {np.std(stats['ratios']):.3f}\n")
    
    print(f"📄 Saved text stats to: {results_save_path}/images_stats.txt")
    
    # ═══════════════════════════════════════════════════════
    # 3. VISUALISATIONS GLOBALES
    # ═══════════════════════════════════════════════════════
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Histogramme des largeurs
    axes[0, 0].hist(all_widths, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Width Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Width (pixels)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(np.mean(all_widths), color='r', linestyle='--', label=f'Mean: {np.mean(all_widths):.0f}')
    axes[0, 0].legend()
    
    # Histogramme des hauteurs
    axes[0, 1].hist(all_heights, bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[0, 1].set_title('Height Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Height (pixels)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(np.mean(all_heights), color='r', linestyle='--', label=f'Mean: {np.mean(all_heights):.0f}')
    axes[0, 1].legend()
    
    # Histogramme des ratios
    axes[0, 2].hist(all_ratios, bins=50, edgecolor='black', alpha=0.7, color='green')
    axes[0, 2].set_title('Aspect Ratio Distribution', fontsize=14, fontweight='bold')
    axes[0, 2].set_xlabel('Aspect Ratio (W/H)')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].axvline(1.0, color='r', linestyle='--', label='Square (1.0)')
    axes[0, 2].axvline(np.mean(all_ratios), color='purple', linestyle='--', label=f'Mean: {np.mean(all_ratios):.2f}')
    axes[0, 2].legend()
    
    # Scatter plot Width vs Height
    axes[1, 0].scatter(all_widths, all_heights, alpha=0.3, s=1)
    axes[1, 0].set_title('Width vs Height', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Width (pixels)')
    axes[1, 0].set_ylabel('Height (pixels)')
    axes[1, 0].plot([0, max(all_widths)], [0, max(all_widths)], 'r--', label='Square (W=H)')
    axes[1, 0].legend()
    
    # Box plot des dimensions
    axes[1, 1].boxplot([all_widths, all_heights], labels=['Width', 'Height'])
    axes[1, 1].set_title('Dimensions Box Plot', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Pixels')
    
    # Pie chart des orientations
    orientations = [portrait, square, landscape]
    labels = [f'Portrait\n({portrait})', f'Square\n({square})', f'Landscape\n({landscape})']
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    axes[1, 2].pie(orientations, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[1, 2].set_title('Image Orientations', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{results_save_path}/images_dimensions_global.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved global plots to: {results_save_path}/images_dimensions_global.png")
    
    # ═══════════════════════════════════════════════════════
    # 4. VISUALISATIONS PAR CLASSE (aspect ratios)
    # ═══════════════════════════════════════════════════════
    
    num_classes = len(class_stats)
    fig, axes = plt.subplots(num_classes, 1, figsize=(12, 4 * num_classes))
    
    if num_classes == 1:
        axes = [axes]
    
    for idx, class_name in enumerate(sorted(class_stats.keys())):
        stats = class_stats[class_name]
        
        axes[idx].hist(stats['ratios'], bins=30, edgecolor='black', alpha=0.7)
        axes[idx].set_title(f"Class: {class_name} (n={stats['count']})", fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Aspect Ratio (W/H)')
        axes[idx].set_ylabel('Frequency')
        axes[idx].axvline(np.mean(stats['ratios']), color='r', linestyle='--', 
                         label=f"Mean: {np.mean(stats['ratios']):.2f}")
        axes[idx].axvline(1.0, color='green', linestyle=':', label='Square (1.0)')
        axes[idx].legend()
        axes[idx].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{results_save_path}/images_ratios_by_class.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved class plots to: {results_save_path}/images_ratios_by_class.png")
    
    # ═══════════════════════════════════════════════════════
    # 5. HEATMAP DES DIMENSIONS PAR CLASSE
    # ═══════════════════════════════════════════════════════
    
    class_names_sorted = sorted(class_stats.keys())
    mean_widths = [np.mean(class_stats[c]['widths']) for c in class_names_sorted]
    mean_heights = [np.mean(class_stats[c]['heights']) for c in class_names_sorted]
    mean_ratios = [np.mean(class_stats[c]['ratios']) for c in class_names_sorted]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, max(6, num_classes * 0.4)))
    
    # Mean Width
    axes[0].barh(class_names_sorted, mean_widths, color='steelblue')
    axes[0].set_xlabel('Mean Width (pixels)')
    axes[0].set_title('Average Width by Class', fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)
    
    # Mean Height
    axes[1].barh(class_names_sorted, mean_heights, color='coral')
    axes[1].set_xlabel('Mean Height (pixels)')
    axes[1].set_title('Average Height by Class', fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)
    
    # Mean Ratio
    axes[2].barh(class_names_sorted, mean_ratios, color='seagreen')
    axes[2].set_xlabel('Mean Aspect Ratio (W/H)')
    axes[2].set_title('Average Aspect Ratio by Class', fontweight='bold')
    axes[2].axvline(1.0, color='red', linestyle='--', label='Square (1.0)')
    axes[2].legend()
    axes[2].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{results_save_path}/images_dimensions_by_class.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved class comparison to: {results_save_path}/images_dimensions_by_class.png")
    
    print(f"\n✅ All analysis saved in: {results_save_path}/")
    print(f"   - images_stats.txt (detailed statistics)")
    print(f"   - images_dimensions_global.png (global distributions)")
    print(f"   - images_ratios_by_class.png (ratios per class)")
    print(f"   - images_dimensions_by_class.png (comparison by class)")

class SpecialistDataset(torch.utils.data.Dataset):
    """
    Dataset personnalisé pour le "Niveau 2" (Spécialistes).
    Lit un dossier plat contenant des images nommées 'NomClasse_NomFichier.jpg'.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        # Récupération des fichiers
        file_list = [f for f in os.listdir(root_dir) if f.lower().endswith(('jpg', 'png', 'jpeg', 'tif', 'bmp'))]
        
        # Dictionnaire pour mapper les noms de classes (str) -> indices (int)
        self.class_to_idx = {}
        unique_classes = set()
        
        # PREMIÈRE PASSE : Identifier toutes les classes uniques
        for fname in file_list:
            # --- LOGIQUE DE DÉCOUPAGE DU NOM DE CLASSE ---
            # Par défaut : on prend tout ce qui est avant le PREMIER underscore
            # Ex: "030_Copepoda_img1.jpg" -> classe "030"
            # Si tes classes sont complexes (ex: "049_01"), change le [0] en [:2] et join
            label_str = fname.split('_')[0] 
            unique_classes.add(label_str)
            
        # Création du mapping trié pour que les ID soient constants
        sorted_classes = sorted(list(unique_classes))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(sorted_classes)}
        self.classes = sorted_classes # Pour compatibilité avec ImageFolder
        
        # DEUXIÈME PASSE : Créer la liste des samples (path, class_idx)
        for fname in file_list:
            label_str = fname.split('_')[0] # DOIT MATCHER LA LOGIQUE CI-DESSUS
            class_idx = self.class_to_idx[label_str]
            path = os.path.join(self.root_dir, fname)
            self.samples.append((path, class_idx))
            
        # Pour compatibilité avec le calcul des poids (targets accessible directement)
        self.targets = [s[1] for s in self.samples]
        
        # Créer un alias 'imgs' qui pointe vers 'samples'
        # comme le fait torchvision.datasets.ImageFolder
        self.imgs = self.samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        # Chargement image (RGB par defaut pour PIL)
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
            
        if self.transform:
            img = self.transform(img)
            
        return img, target

def get_dataloaders(data_config, use_cuda):
    valid_ratio = data_config["valid_ratio"]
    batch_size = data_config["batch_size"]
    num_workers = data_config["num_workers"]
    resize = data_config["images_resize"]
    use_rgb = data_config.get("rgb", False)
    augmentations_v2 = data_config.get("augmentations_v2", False)
    dataset_size_divider = data_config.get("dataset_size_divider", 6)

    # Récupération du mode (standard par défaut)
    dataset_mode = data_config.get("dataset_mode", "standard") 

    logging.info("  - Dataset creation")

    if use_rgb:
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]
    else:
        MEAN = [0.485]
        STD = [0.229]

    transform_list = []

    if not augmentations_v2:
    
        if not use_rgb:
            transform_list.append(transforms.Grayscale())  # 1 canal
        # Sinon, reste en RGB natif (3 canaux)
        
        transform_list.extend([
            transforms.Resize((resize, resize)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD) 
        ])

        input_transform = transforms.Compose(transform_list)

    else:
        print("Using data augmentations v2")
        def square_pad(img):
            """Ajoute du padding pour rendre l'image carrée"""
            c, h, w = img.shape

            if h > w:
                padding_left = (h - w) // 2
                padding_right = h - w - padding_left
                padding_top = 0
                padding_bottom = 0
            else:
                padding_left = 0 
                padding_right = 0
                padding_top = (w - h)//2
                padding_bottom = w - h - padding_top

            return F.pad(
                img,
                (padding_left, padding_top, padding_right, padding_bottom),
                0,
                'constant',
                        
            )

        if not use_rgb:
            transform_list.append(transforms.Grayscale())

        transform_list.extend([
            transforms.ToTensor(),
            transforms.Lambda(square_pad),
            transforms.Resize((resize, resize)),
            transforms.RandomRotation(180),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(mean=MEAN, std=STD),
        ])

        input_transform = transforms.Compose(transform_list)
    
    # --- CHARGEMENT DU DATASET SELON LE MODE ---
    if dataset_mode == "specialist":
        # Mode Spécialiste : Lit les préfixes des fichiers dans un dossier plat
        base_dataset = SpecialistDataset(
            root_dir=data_config["trainpath"], # Doit pointer vers le sous-dossier (ex: .../1_Copepoda)
            transform=input_transform
        )
    else:
        # Mode Standard (Général) : Structure ImageFolder classique
        base_dataset = torchvision.datasets.ImageFolder(
            root=data_config["trainpath"], 
            transform=input_transform,
        )

    logging.info(f"  - I loaded {len(base_dataset)} samples")

    # ══════════════════════════════════════════════════════════════
    # CALCUL RAPIDE ET UNIQUE DES CLASS WEIGHTS
    # ══════════════════════════════════════════════════════════════
    logging.info("  - Computing class weights")
    
    # Accès direct aux targets (RAPIDE: 2-3 secondes)
    targets = np.array(base_dataset.targets)  # ← Pas d'itération !
    
    # Compter les occurrences
    unique_classes, class_counts = np.unique(targets, return_counts=True)
    
    # Formule "balanced" (comme sklearn)
    total_samples = len(targets)
    num_classes = len(unique_classes)
    class_weights_array = total_samples / (num_classes * class_counts)
    
    # Convertir en torch pour la loss
    class_weights_torch = torch.FloatTensor(class_weights_array)
    
    # Créer les sample weights pour le sampler
    sample_weights = class_weights_array[targets]
    
    # Logs informatifs
    logging.info(f"    Classes: {num_classes}, Samples: {total_samples:,}")
    logging.info(f"    Weights range: [{class_weights_array.min():.4f}, {class_weights_array.max():.4f}]")

    random.seed(42)
    indices = list(range(len(base_dataset)))
    random.shuffle(indices)
    num_valid = int(valid_ratio * len(base_dataset))
    train_indices = indices[num_valid:]
    valid_indices = indices[:num_valid]

    train_dataset = torch.utils.data.Subset(base_dataset, train_indices)
    valid_dataset = torch.utils.data.Subset(base_dataset, valid_indices)

    # Build the dataloaders
    train_weights = [sample_weights[i] for i in train_indices]
    sampler = WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_weights)//dataset_size_divider,
        replacement=True
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )
    num_classes = len(base_dataset.classes)
    input_size = tuple(base_dataset[0][0].shape)

    return train_loader, valid_loader, input_size, num_classes, class_weights_torch


def test_dataloaders():
    print("_"*20)
    print("Test dataloaders:")
    
    data_config = {
        "trainpath": "./data/train",
        "valid_ratio": 0.2,
        "batch_size": 32,
        "num_workers": 0,
        "images_resize": 128
    }

    results_save_path = "./data_analysis"

    if torch.cuda.is_available():
        device = torch.device("cuda")
        use_cuda = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        use_cuda = True 
    else:
        use_cuda = False

    print(f"Use CUDA: {use_cuda} | Device: {device}")

    train_loader, valid_loader, input_size, num_classes, _ = get_dataloaders(
        data_config, use_cuda
    )

    # Afficher la distribution
    base_dataset = torchvision.datasets.ImageFolder(
        root=data_config["trainpath"], 
        transform=transforms.ToTensor()
    )
    show_class_distribution(base_dataset, results_save_path = "./data_analysis")

    show_images_information(base_dataset, results_save_path="./data_analysis")

    X, y = next(iter(train_loader))
    grid = make_grid(X, nrow=8)
    show(grid)
    plt.tight_layout()
    plt.savefig(f"{results_save_path}/sample_batch.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_dataloaders()
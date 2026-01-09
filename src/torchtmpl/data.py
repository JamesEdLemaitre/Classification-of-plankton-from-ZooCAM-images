# coding: utf-8

# Standard imports
import logging
import random
import os
import numpy as np

# External imports
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.utils import make_grid
from PIL import Image
from torch.utils.data import WeightedRandomSampler, Dataset, DataLoader
from sklearn.model_selection import train_test_split

import pandas as pd
import matplotlib.pyplot as plt


def show_image(X):
    num_c = X.shape[0]
    plt.figure()
    plt.imshow(X[0] if num_c == 1 else X.permute(1, 2, 0))
    plt.show()

class SpecialistDataset(torch.utils.data.Dataset):
    """
    Source: Claude Sonnet 4.5

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

class FoundationDataset(Dataset):
    """
    Dataset pour le mode Foundation (SSL).
    Lit un CSV contenant 'filepath'.
    """
    def __init__(self, csv_file, root_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # On récupère le chemin relatif
        row = self.df.iloc[idx]
        rel_path = row['filepath']
        img_path = os.path.join(self.root_dir, rel_path)
        
        # Chargement image
        # .convert('RGB') est crucial pour les ViT qui attendent 3 canaux
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # On renvoie 0 comme label "dummy" car le dataloader l'attend souvent
        return image, 0

def collate_fn(batch):
    # 'batch' est une liste de ce que retourne ton dataset.__getitem__
    # Si ton dataset retourne (image, label) -> on prend que image (index 0)
    # Si ton dataset retourne juste image -> on prend tout
    
    images = []
    for item in batch:
        if isinstance(item, tuple) or isinstance(item, list):
            images.append(item[0]) # On jette le label
        else:
            images.append(item)    # C'est déjà juste l'image
            
    # On empile les images en un seul gros Tensor (Batch, Channels, Height, Width)
    pixel_values = torch.stack(images)
    
    # On retourne le format exact attendu par le modèle Hugging Face
    return {"pixel_values": pixel_values}


def get_transforms(version, train, use_rgb, resize):
    transform_list = []

    if not use_rgb:
            transform_list.append(transforms.Grayscale())
            MEAN = [0.485]
            STD = [0.229]
    else:
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]
        
    if version == "basic":
        # Very basic transformations

        transform_list.append(transforms.Resize((resize, resize)))

        if train:
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15)
            ])

        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD) 
        ]) 

    elif version == "preserve_aspect_ratio":
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
                'constant')
        
        if train:
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(180),
            ])

        transform_list.extend([
            transforms.ToTensor(),
            transforms.Lambda(square_pad),     # Rend carré
            transforms.Resize((resize, resize)), # Redimensionne le carré
            transforms.Normalize(mean=MEAN, std=STD),
        ])
    
    elif version == "foundation_basic":
        # --- Le Standard ImageNet pour les ViT ---
        
        # Note : Les ViT préfèrent souvent l'interpolation BICUBIC (plus précise que bilinear)
        from torchvision.transforms import InterpolationMode
        
        if train:
            transform_list.extend([
                # RandomResizedCrop est L'ARME SECRÈTE du training ImageNet.
                # Il sélectionne une zone aléatoire de l'image (entre 20% et 100% de la surface)
                # et la redimensionne à la taille cible.
                # Ça force le modèle à apprendre les détails de texture ET la forme globale.
                transforms.RandomResizedCrop(
                    resize, 
                    scale=(0.2, 1.0), 
                    interpolation=InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ])
        else:
            # Pour le test/valid, on ne veut pas d'aléatoire.
            # On applique la règle du "ratio 256/224" classique d'ImageNet :
            # On redimensionne l'image un peu plus grand que nécessaire, puis on coupe le centre.
            # Cela évite de déformer l'image et garde la zone la plus importante.
            
            # Calcul de la taille de resize "un peu plus grande"
            # Si resize=224, ça donne 256.
            resize_img = int(resize * (256 / 224))
            
            transform_list.extend([
                transforms.Resize(resize_img, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(resize),
            ])

        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])
    
    else:
        raise ValueError(f"Unknown transformation version: {version}")

    return transforms.Compose(transform_list)

def get_dataset(root, transform, training_style="standard", csv_file=None):
    
    dataset = None
    if training_style == "standard":
        dataset = torchvision.datasets.ImageFolder(
            root=root, 
            transform=transform)
        
    elif training_style == "specialist":
        dataset = SpecialistDataset(
            root_dir=root, 
            transform=transform)

    elif training_style == "foundation":
        if csv_file is None:
            raise ValueError("Foundation style requires a 'csv_file' path.")
        dataset = FoundationDataset(
            csv_file=csv_file,
            root_dir=root,
            transform=transform
        )
        
    else:
        print(f"Error: unknown training style {training_style} for dataset creation")

    return dataset

def get_dataloaders(data_config, training_style="standard"):
    # Read the data configuration for this training:
    valid_ratio = data_config.get("valid_ratio", None)
    batch_size = data_config["batch_size"]
    num_workers = data_config["num_workers"]
    resize = data_config["images_resize"]
    use_rgb = data_config.get("use_rgb", False)
    use_cuda = data_config.get("use_cuda", torch.cuda.is_available())
    transformations_version = data_config.get("transformations_version", "basic")
    random_seed = data_config.get("random_seed", 42)
    dataset_size_divider = data_config.get("dataset_size_divider", 1) 

    logging.info("  - Dataset creation")
    logging.info(f"    Style: {training_style} | Transform: {transformations_version}")

    train_transform = get_transforms(transformations_version, train=True, use_rgb=use_rgb, resize=resize)
    test_transform = get_transforms(transformations_version, train=False, use_rgb=use_rgb, resize=resize)

    # ==========================================
    # BRANCHE 1 : FOUNDATION (Simplifiée)
    # ==========================================
    if training_style == "foundation":
        # Dans ce mode, on attend que data_config contienne les chemins vers les CSV
        # ex: data_config["manifest_train"] et "manifest_test"
        
        train_csv = data_config.get("manifest_train", "foundation_train.csv")
        test_csv = data_config.get("manifest_test", "foundation_test.csv")
        root_dir = data_config["trainpath"] # Dossier racine des images

        logging.info(f"    Loading Foundation CSVs: {train_csv} / {test_csv}")

        train_dataset = get_dataset(root=root_dir, transform=train_transform, 
                                    training_style="foundation", csv_file=train_csv)
        valid_dataset = get_dataset(root=root_dir, transform=test_transform, 
                                    training_style="foundation", csv_file=test_csv)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                  num_workers=num_workers, pin_memory=use_cuda, collate_fn=collate_fn)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, 
                                  num_workers=num_workers, pin_memory=use_cuda, collate_fn=collate_fn)

        # Valeurs "vides" pour respecter la signature de la fonction
        input_size = (3, resize, resize)
        num_classes = 0
        class_weights_torch = None
        classes_names = []
        
        return train_loader, valid_loader, input_size, num_classes, class_weights_torch, classes_names
    
    else:

        # On charge le dataset "principal" (pour compter les classes et avoir les indices)
        # Peu importe la transfo ici, on ne s'en servira que pour la structure
        full_dataset_structure = get_dataset(root=data_config["trainpath"], 
                                            transform=train_transform, # Temporaire
                                            training_style=training_style)

        logging.info(f"  - I loaded {len(full_dataset_structure)} samples")

        logging.info("  - Computing class weights")


        targets = np.array(full_dataset_structure.targets)
        unique_classes, class_counts = np.unique(targets, return_counts=True)
        total_samples = len(targets)
        num_classes = len(unique_classes)
        class_weights_array = total_samples / (num_classes * class_counts)
        # Convert to torch
        class_weights_torch = torch.FloatTensor(class_weights_array)

        sample_weights_full = class_weights_array[targets]

        logging.info(f"    Classes: {num_classes}, Samples: {total_samples:,}")
        logging.info(f"    Weights range: [{class_weights_array.min():.4f}, {class_weights_array.max():.4f}]")

        # Splitting dataset for train and test
        random.seed(random_seed)
        targets = np.array(full_dataset_structure.targets)
        indices = np.arange(len(targets))

        # Stratify assure que la proportion des classes est respectée dans train et valid
        train_indices, test_indices = train_test_split(
            indices, 
            test_size=valid_ratio, 
            stratify=targets, 
            random_state=random_seed
        )

        if training_style == "foundation":
            # Cas spécial : on passe le CHEMIN DU CSV comme premier argument
            dataset_train_obj = get_dataset(data_config["manifest_train"], train_transform, training_style)
            dataset_valid_obj = get_dataset(data_config["manifest_test"], test_transform, training_style)
            # Mais attention, il faut que get_dataset puisse instancier FoundationDataset avec le root_dir
            # Il vaut mieux modifier FoundationDataset pour qu'il prenne le root_dir en dur ou via modif de get_dataset.
        else:
            dataset_train_obj = get_dataset(data_config["trainpath"], train_transform, training_style)
            dataset_valid_obj = get_dataset(data_config["trainpath"], test_transform, training_style)


        train_dataset = torch.utils.data.Subset(dataset_train_obj, train_indices)
        valid_dataset = torch.utils.data.Subset(dataset_valid_obj, test_indices)

        # On extrait les poids correspondants uniquement aux indices du train
        train_weights = [sample_weights_full[i] for i in train_indices]

        sampler = WeightedRandomSampler(
            weights=train_weights,
            num_samples=len(train_weights) // dataset_size_divider,
            replacement=True
        )

        # Build the dataloaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler, 
            shuffle=False, # Le sampler se cahrge de l'aléatoire.  
            num_workers=num_workers,
            pin_memory=use_cuda,
        )


        test_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_cuda,
        )

        if hasattr(full_dataset_structure, 'classes'):
            classes_names = full_dataset_structure.classes
        elif hasattr(full_dataset_structure, 'categories'):
            classes_names = full_dataset_structure.categories
        else:
            classes_names = [str(i) for i in range(num_classes)]

        num_classes = len(classes_names)
        input_size = tuple(full_dataset_structure[0][0].shape)

        return train_loader, test_loader, input_size, num_classes, class_weights_torch, classes_names

def test_dataloaders_and_analyse_dataset():
    print("_"*70)
    print("Testing dataloaders and analysing dataset:")

    data_config = {
        "trainpath": "./data/original_dataset/3-md-4040-2026-challenge/train",
        "valid_ratio": 0.2,
        "batch_size": 32,
        "num_workers": 0,
        "images_resize": 128
    }

    results_save_path = "./results/dataset_analysis"

    train_loader, test_loader, input_size, num_classes, class_weights_torch = get_dataloaders(
        data_config=data_config,
        training_style="standard"
    )

    print("Loaders created successfully. Testing and analysing the dataset...")

    #######################################
    #### Useful functions for analysis ####
    #######################################
    # Author: Claude Sonnet 4.5

    def show(imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
        for i, img in enumerate(imgs):
            img = img.detach()
            img = F.to_pil_image(img)
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    dataset = torchvision.datasets.ImageFolder(
        root=data_config["trainpath"], 
        transform=transforms.ToTensor()
    )

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

    #######################################
    #######################################
    # Make a grid and save it:
    X, y = next(iter(train_loader))
    grid = make_grid(X, nrow=8)
    show(grid)
    plt.tight_layout()
    plt.savefig(f"{results_save_path}/sample_batch.png", dpi=150)

    # Show images information:
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

    show_class_distribution(dataset, results_save_path = results_save_path)
    show_images_information(dataset, results_save_path = results_save_path)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_dataloaders_and_analyse_dataset()
# coding: utf-8

# Standard imports
import logging
import random
import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt

# External imports
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import WeightedRandomSampler
import torchvision
from torchvision import transforms
from sklearn.model_selection import train_test_split

from PIL import Image


def show_image(X):
    num_c = X.shape[0]
    plt.figure()
    plt.imshow(X[0] if num_c == 1 else X.permute(1, 2, 0))
    plt.show()

class PlanktonMultisourceDataset(torch.utils.data.Dataset):
    def __init__(self, sources_config, resize=224, is_validation=False):
        """
        Args:
            sources_config (dict): Dictionnary from the .yaml config file
        """
        self.sources_config = sources_config
        self.samples = []
        self.resize = resize

        ## 1. build class_to_idx (map class names to integer indices)
        # We need the original dataset to build the mapping
        if not "original_dataset" in sources_config:
            raise ValueError("sources_config must contain 'original_dataset' key")

        # Then we can check the dataset folder and build class_to_idx
        original_path = sources_config["original_dataset"]["path"]
        self.classes = self._find_classes(original_path)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        logging.info(f"Dataset initialized with {len(self.classes)} classes from the original dataset")

        ## 2. Load additional sources
        for source_name, config in sources_config.items():
            # Get the information needed in the .yaml
            root_path = config["path"]
            aug_version = config["augmentations_version"]
            weights_path = config["classes_weights_file"]

            # Get the mapping file only for non-original datasets
            if source_name != "original_dataset":
                mapping_path = config["classes_mapping_file"]
            else:
                mapping_path = None

            # Get transforms
            transforms_func = self._get_transforms(aug_version, train=(not is_validation))

            # load weights (= number of image to take in each class)
            weights = self._load_jsonc(weights_path)

            # load mapping for non-original datasets
            if mapping_path is not None:
                class_mapping = self._load_jsonc(mapping_path)
            else:
                class_mapping = {}

            count_added = 0

            for folder_name, count_limit in weights.items():
                folder_path = os.path.join(root_path, folder_name)

                if not os.path.exists(folder_path):
                    raise ValueError(f"Folder path {folder_path} does not exist")

                if source_name == "original_dataset":
                    target_class_name = folder_name
                else:
                    # Map to original class name
                    if folder_name in class_mapping:
                        target_class_name = class_mapping[folder_name]
                    else:
                        raise ValueError(f"Class {folder_name} not found in mapping for source {source_name}. Skipping.")
                        continue

                label_idx = self.class_to_idx[target_class_name]

                # List images in the folder
                images = glob.glob(os.path.join(folder_path, "*"))

                images.sort()

                images = [x for x in images if x.lower().endswith(('.png', '.jpg', '.jpeg'))]

                # Apply count limit if specified
                if count_limit != "all":
                    limit = int(count_limit)
                    random.seed(42)
                    if len(images) > limit:
                        random.shuffle(images)
                        images = images[:limit]

                # Add samples to the dataset
                for img_path in images:
                    self.samples.append((img_path, label_idx, transforms_func, source_name))
                    count_added += 1

            logging.info(f"Source [{source_name}]: {count_added} images added (Aug: {aug_version})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, transform_func, _ = self.samples[idx]
        # Conversion to RGB
        image = Image.open(path).convert("RGB") 
        
        if transform_func:
            image = transform_func(image)
        
        return image, label


    def _get_transforms(self, version_name, train=True):
        transform_list = []
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]

        if version_name == "basic":
            # Very basic transformations
            transform_list.append(transforms.Resize((self.resize, self.resize)))

            if train:
                transform_list.extend([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(180)
                ])

            transform_list.extend([
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD) 
            ]) 
        elif version_name == "pelgas_basic":
            transform_list.append(transforms.Resize((self.resize, self.resize)))

            if train:
                transform_list.extend([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomRotation(180),
                ])

                transform_list.extend([
                    transforms.RandomAffine(
                        degrees=0, 
                        translate=(0.1, 0.1), 
                        scale=(0.8, 1.2), 
                        shear=10 
                    ),
                    
                    transforms.RandomApply([
                        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0))
                    ], p=0.3),
                    
                    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
                    
                    transforms.ColorJitter(
                        brightness=0.3, 
                        contrast=0.3, 
                        saturation=0.2, 
                        hue=0.05
                    ),
                ])

            # 4. Conversion Tensor
            transform_list.append(transforms.ToTensor())
            # 6. Normalisation finale
            transform_list.append(transforms.Normalize(mean=MEAN, std=STD))
            
        return transforms.Compose(transform_list)
    
    def _find_classes(self, dir_path):
        """Scan original folder to find class names."""
        classes = [d.name for d in os.scandir(dir_path) if d.is_dir()]
        classes.sort()
        return classes
    
    def _load_jsonc(self, filepath):
        """
        Source @gemini
        """
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            clean_lines = [line.split('//')[0] for line in lines]
            clean_content = '\n'.join(clean_lines)
            return json.loads(clean_content)
        except Exception as e:
            logging.error(f"Error while reading JSONC for {filepath} : {e}")
            return {}

def get_dataloaders(data_config, use_cuda):
    valid_ratio = data_config["valid_ratio"]
    batch_size = data_config["batch_size"]
    num_workers = data_config["num_workers"]
    resize = data_config.get("resize", 224)

    logging.info("  - Dataset creation")

    train_ds_full = PlanktonMultisourceDataset(
        sources_config=data_config["sources"],
        resize=resize,
        is_validation=False 
    )

    valid_ds_full = PlanktonMultisourceDataset(
        sources_config=data_config["sources"],
        resize=resize,
        is_validation=True 
    )

    logging.info(f"  - I loaded {len(train_ds_full)} samples.")

    original_indices = [i for i, x in enumerate(train_ds_full.samples) if x[3] == "original_dataset"]
    other_indices = [i for i, x in enumerate(train_ds_full.samples) if x[3] != "original_dataset"]

    logging.info(f"  - Split strategy: keeping validation strictly on 'original_dataset'.")
    logging.info(f"    Original samples: {len(original_indices)} | External samples (Pelgas, etc): {len(other_indices)}")

    original_labels = [train_ds_full.samples[i][1] for i in original_indices]

    # startified split on original dataset only
    train_idx_strat, valid_idx_strat = train_test_split(
        original_indices,
        test_size=valid_ratio,
        stratify=original_labels, 
        random_state=42,
        shuffle=True
    )

    indices = list(range(len(train_ds_full)))
    random.seed(42)
    random.shuffle(original_indices)

    valid_indices = valid_idx_strat
    
    train_indices = train_idx_strat + other_indices

    train_dataset = torch.utils.data.Subset(train_ds_full, train_indices)
    valid_dataset = torch.utils.data.Subset(valid_ds_full, valid_indices)

    # Weighted Sampler
    logging.info("  - Computing class weights for Balanced Sampling...")
    
    # Compute the amount of images by class in the training subset
    train_labels = [train_ds_full.samples[i][1] for i in train_indices]
    class_counts = np.bincount(train_labels)

    class_weights = 1. / (class_counts + 1e-6)

    sample_weights = [class_weights[label] for label in train_labels]

    # Create the sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True # Allows to retake the same image multiple times in one batch
    )


    # Build the dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False, # Disabled because we use a sampler
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False, # No shuffle for validation, neither a sampler
        num_workers=num_workers,
        pin_memory=use_cuda,
    )

    num_classes = len(train_ds_full.classes)
    input_size = tuple(train_ds_full[0][0].shape)

    return train_loader, valid_loader, input_size, num_classes


def run_dataset_verification(sources_config, output_dir="data_output_test"):
    """
    Génère un rapport visuel complet :
    1. Grille Train (Augmentée)
    2. Grille Valid (Clean)
    3. Graphique de distribution
    4. Comparatif par classe (Original vs External) pour vérifier la cohérence.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import shutil
    import random
    from PIL import Image

    # 1. Préparation
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("DataVerifier")
    logger.info(f"=== Lancement de la vérification dans '{output_dir}' ===")

    # Config temporaire
    data_config = {
        "sources": sources_config,
        "valid_ratio": 0.2,
        "batch_size": 16, 
        "num_workers": 0, 
        "resize": 224
    }

    # 2. Création des Dataloaders
    try:
        # On force use_cuda=False pour le debug
        train_loader, valid_loader, input_size, num_classes = get_dataloaders(data_config, use_cuda=False)
    except Exception as e:
        logger.error(f"Erreur critique lors de la création des dataloaders : {e}")
        return

    # On récupère le dataset TRAIN complet (celui qui a les augmentations activées)
    full_dataset = train_loader.dataset.dataset 
    class_names = full_dataset.classes

    # --- UTILITAIRES ---
    def unnormalize(tensor):
        """Annule la normalisation ImageNet pour l'affichage"""
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = tensor.numpy().transpose((1, 2, 0)) # C, H, W -> H, W, C
        img = std * img + mean
        img = np.clip(img, 0, 1)
        return img

    def save_batch_grid(loader, title, filename, is_train):
        try:
            images, labels = next(iter(loader))
        except StopIteration:
            return

        aug_text = "Augmentations visibles" if is_train else "Pas d'augmentation"
        plt.figure(figsize=(12, 6))
        plt.suptitle(f"{title}\n({aug_text})", fontsize=14)

        count = min(len(images), 8)
        for i in range(count):
            ax = plt.subplot(2, 4, i + 1)
            img = unnormalize(images[i])
            label_name = class_names[labels[i]]
            plt.imshow(img)
            plt.title(f"{label_name}", fontsize=10)
            plt.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

    # 3. Génération des visuels globaux
    save_batch_grid(train_loader, "TRAIN BATCH", "1_visualization_train.png", is_train=True)
    save_batch_grid(valid_loader, "VALID BATCH", "2_visualization_valid.png", is_train=False)

    # 4. Organisation des données pour analyse
    logger.info("Analyse de la distribution et cohérence...")
    original_root = sources_config["original_dataset"]["path"]
    
    # Structure : data_map[source][class_name] = [list_of_indices]
    data_map = {}
    sources_found = set()

    # On parcourt le dataset. 
    # ATTENTION : full_dataset.samples contient maintenant (path, label, transform, source)
    for idx, sample in enumerate(full_dataset.samples):
        # Déballage robuste (compatible 3 ou 4 éléments selon ta version)
        if len(sample) == 4:
            path, label_idx, transform_func, source_name = sample
        else:
            path, label_idx, transform_func = sample
            # Fallback si l'étape précédente n'est pas encore appliquée
            source_name = "original_dataset" if path.startswith(original_root) else "external"

        c_name = class_names[label_idx]
        sources_found.add(source_name)

        if source_name not in data_map:
            data_map[source_name] = {}
        if c_name not in data_map[source_name]:
            data_map[source_name][c_name] = []
        
        data_map[source_name][c_name].append(idx)

    # 5. Graphique de distribution
    classes_sorted = sorted(class_names)
    x = np.arange(len(classes_sorted))

    plt.figure(figsize=(24, 12))

    # On récupère les comptes pour l'original
    y_orig = [len(data_map.get("original_dataset", {}).get(c, [])) for c in classes_sorted]

    # On plot l'original en bas (VERT FIXE)
    p1 = plt.bar(x, y_orig, color='#2ca02c', label='Original Dataset', alpha=0.8)
    bottom_y = np.array(y_orig)

    # Palette de couleurs distinctes pour les ajouts (Rouge, Bleu, Orange, Violet, Cyan)
    # Cela garantit qu'on ne réutilise pas le vert par accident
    ext_colors = ['#d62728', '#1f77b4', '#ff7f0e', '#9467bd', '#17becf']
    color_idx = 0

    # On empile les autres sources
    for source in sources_found:
        if source == "original_dataset": continue

        y_src = [len(data_map.get(source, {}).get(c, [])) for c in classes_sorted]
        
        # Sélection de la couleur dans la liste
        c_code = ext_colors[color_idx % len(ext_colors)]
        color_idx += 1

        plt.bar(x, y_src, bottom=bottom_y, label=f'Added: {source}', color=c_code, alpha=0.8)
        bottom_y += np.array(y_src)

    plt.ylabel('Nombre d\'images')
    plt.title('Distribution des classes par Source', fontsize=18)
    plt.xticks(x, classes_sorted, rotation=90, fontsize=9)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "3_distribution_analysis.png"))
    plt.close()

    # 6. Vérification de la COHÉRENCE par classe (Side-by-Side)
    logger.info("Génération des planches de cohérence (Original vs External)...")

    # Pour chaque source externe ajoutée
    for source in sources_found:
        if source == "original_dataset": continue
        
        logger.info(f" -> Création planche pour : {source}")
        
        # Trouver les classes communes
        common_classes = [c for c in classes_sorted 
                          if c in data_map["original_dataset"] and c in data_map[source]]
        
        if not common_classes:
            logger.warning(f"Aucune classe commune entre original et {source}")
            continue

        # Création d'une image géante : 1 ligne par classe
        # Colonnes : 3 Original | 3 External
        n_rows = len(common_classes)
        fig, axes = plt.subplots(n_rows, 6, figsize=(18, 3 * n_rows))
        
        # Titre global
        fig.suptitle(f"Cohérence : Original (Gauche) vs {source} (Droite)\n(Avec Augmentations Train)", fontsize=20, y=1.005)

        # Si une seule classe, axes n'est pas un tableau 2D, on corrige
        if n_rows == 1: axes = np.array([axes])

        for row_idx, cls_name in enumerate(common_classes):
            # Indices disponibles
            idxs_orig = data_map["original_dataset"][cls_name]
            idxs_ext = data_map[source][cls_name]

            # Sélection aléatoire de 3 images max
            sel_orig = random.sample(idxs_orig, min(3, len(idxs_orig)))
            sel_ext = random.sample(idxs_ext, min(3, len(idxs_ext)))

            # Remplissage pour avoir toujours 3 items (répétition si besoin ou vide)
            while len(sel_orig) < 3 and len(sel_orig) > 0: sel_orig.append(sel_orig[0])
            while len(sel_ext) < 3 and len(sel_ext) > 0: sel_ext.append(sel_ext[0])

            # Affichage
            all_samples_row = sel_orig + sel_ext # [Orig, Orig, Orig, Ext, Ext, Ext]
            
            for col_idx, sample_idx in enumerate(all_samples_row):
                ax = axes[row_idx, col_idx]
                
                # Récupération et transformation
                path, _, transform_func, _ = full_dataset.samples[sample_idx]
                
                try:
                    img = Image.open(path).convert("RGB")
                    # On applique la transfo d'augmentation stockée
                    if transform_func:
                        img_tensor = transform_func(img)
                        img_disp = unnormalize(img_tensor)
                    else:
                        img_disp = np.array(img)
                    
                    ax.imshow(img_disp)
                except Exception as e:
                    ax.text(0.5, 0.5, "Error", ha='center')
                
                # Cosmétique
                ax.axis("off")
                
                # Titres de colonnes (seulement première ligne)
                if row_idx == 0:
                    if col_idx == 1: ax.set_title("ORIGINAL DATASET", fontsize=16, color='green', fontweight='bold')
                    if col_idx == 4: ax.set_title(f"SOURCE: {source}", fontsize=16, color='red', fontweight='bold')

            # Label de la classe à gauche de la ligne
            axes[row_idx, 0].text(-0.5, 0.5, cls_name, transform=axes[row_idx, 0].transAxes, 
                                  fontsize=14, va='center', ha='right', fontweight='bold')

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"4_coherence_{source}.jpg") # JPG pour compresser si c'est gros
        plt.savefig(save_path, dpi=100) # DPI modéré pour éviter fichier énorme
        plt.close()
        logger.info(f"Saved: {save_path}")

    logger.info("=== Vérification terminée ===")



if __name__ == "__main__":
    my_sources_config = {
        "original_dataset": {
                "path": "/home/lmalerba/projects/kaggle_challenge_sdi/kaggle_challenge_sdi/data/original_dataset_optimized_512px/train",
                "augmentations_version": "basic",
                "classes_weights_file": "/home/lmalerba/projects/kaggle_challenge_sdi/kaggle_challenge_sdi/classes_weights/original_dataset_max20000.jsonc"
            },
        "pelgas_dataset": {
                "path": "/home/lmalerba/projects/kaggle_challenge_sdi/kaggle_challenge_sdi/data/pelgas_dataset_cropped_optimized_512px/101141/individual_images/",
                "augmentations_version": "pelgas_basic",
                "classes_weights_file": "/home/lmalerba/projects/kaggle_challenge_sdi/kaggle_challenge_sdi/classes_weights/pelgas_dataset_0.jsonc",
                "classes_mapping_file": "/home/lmalerba/projects/kaggle_challenge_sdi/kaggle_challenge_sdi/datasets_mappers/pelgas_to_original.jsonc"
            },
        "whoi_dataset": {
                "path": "/home/lmalerba/projects/kaggle_challenge_sdi/kaggle_challenge_sdi/data/whoi_dataset_optimized_512px/",
                "augmentations_version": "pelgas_basic",
                "classes_weights_file": "/home/lmalerba/projects/kaggle_challenge_sdi/kaggle_challenge_sdi/classes_weights/whoi_dataset_0.jsonc",
                "classes_mapping_file": "/home/lmalerba/projects/kaggle_challenge_sdi/kaggle_challenge_sdi/datasets_mappers/whoi_to_original.jsonc"
            },
        "ndsb_dataset": {
                "path": "/home/lmalerba/projects/kaggle_challenge_sdi/kaggle_challenge_sdi/data/ndsb_dataset_optimized_512px/train",
                "augmentations_version": "pelgas_basic",
                "classes_weights_file": "/home/lmalerba/projects/kaggle_challenge_sdi/kaggle_challenge_sdi/classes_weights/ndsb_dataset_0.jsonc",
                "classes_mapping_file": "/home/lmalerba/projects/kaggle_challenge_sdi/kaggle_challenge_sdi/datasets_mappers/ndsb_to_original.jsonc"
            }
    }

    run_dataset_verification(my_sources_config, output_dir="data_output_test_original_max20000_full_pelgas_augmented")

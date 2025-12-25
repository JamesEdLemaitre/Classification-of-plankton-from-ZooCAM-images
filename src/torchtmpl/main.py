# coding: utf-8

# Standard imports
import logging
import sys
import os
import pathlib
from tqdm import tqdm
import numpy as np
import pandas as pd

# External imports
import yaml
import wandb
import torch
import torchinfo.torchinfo as torchinfo
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import random

# Local imports
from . import data
from . import models
from . import optim
from . import utils


def train(config):
    # Get configuration variables properly, with default values to make sure eveything
    # will work constantly:
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    logging.info(f"Training on device: {device}")
    training_style = config["training"].get("style", "standard")
    phases = config.get("phases", [{"epochs": config.get("nepochs", 1), "unfreeze": "none"}])
    
    if "wandb" in config["logging"]:
        wandb_config = config["logging"]["wandb"]
        wandb.init(project=wandb_config["project"], entity=wandb_config["entity"])
        wandb_log = wandb.log
        wandb_log(config)
        logging.info(f"Will be recording in wandb run name : {wandb.run.name}")
    else:
        wandb_log = None

    # Build the dataloaders
    logging.info("= Building the dataloaders")
    data_config = config["data"]

    print(training_style)
    train_loader, valid_loader, input_size, num_classes, class_weights = data.get_dataloaders(
        data_config, training_style=training_style
    )

    # Build the model
    logging.info("= Model")
    model_config = config["model"]
    model = models.build_model(model_config, input_size, num_classes)
    
    # Send both model and class_weights to the device:
    model = model.to(device)
    class_weights = class_weights.to(device)

    # Build the loss
    logging.info("= Loss")
    loss = optim.get_loss(config["loss"], class_weights=class_weights, device=device)

    # Build the optimizer
    logging.info("= Optimizer")
    optim_config = config["optim"]
    optimizer = optim.get_optimizer(optim_config, model.parameters())

    # Build the callbacks
    logging_config = config["logging"]
    # Let us use as base logname the class name of the modek
    logname = model_config["class"]
    logdir = utils.generate_unique_logpath(logging_config["logdir"], logname)
    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    logging.info(f"Will be logging into {logdir}")

    # Copy the config file into the logdir
    logdir = pathlib.Path(logdir)
    with open(logdir / "config.yaml", "w") as file:
        yaml.dump(config, file)

    # Make a summary script of the experiment
    input_size = next(iter(train_loader))[0].shape
    summary_text = (
        f"Logdir : {logdir}\n"
        + "## Command \n"
        + " ".join(sys.argv)
        + "\n\n"
        + f" Config : {config} \n\n"
        + (f" Wandb run name : {wandb.run.name}\n\n" if wandb_log is not None else "")
        + "## Summary of the model architecture\n"
        + f"{torchinfo.summary(model, input_size=input_size)}\n\n"
        + "## Loss\n\n"
        + f"{loss}\n\n"
        + "## Datasets : \n"
        + f"Train : {train_loader.dataset.dataset}\n"
        + f"Validation : {valid_loader.dataset.dataset}"
    )

    with open(logdir / "summary.txt", "w", encoding='utf-8') as f:
        f.write(summary_text)
    logging.info(summary_text)
    if wandb_log is not None:
        wandb.log({"summary": summary_text})

    # Define the early stopping callback
    model_checkpoint = utils.ModelCheckpoint(
        model, str(logdir / "best_model.pt"), min_is_best=False
    )

    # Build the LR scheduler
    logging.info("= Learning Rate Scheduler")
    if "scheduler" in config:
        scheduler_config = config["scheduler"]
        scheduler = optim.get_scheduler(scheduler_config, optimizer)
        logging.info(f"  Using scheduler: {scheduler_config['class']}")
    else:
        scheduler = None
        logging.info("  No scheduler specified")

    global_epoch = 0
    for phase_idx, phase in enumerate(phases):
        logging.info(f"\n{'='*60}")
        logging.info(f"PHASE {phase_idx+1}: {phase['name']} ({phase['epochs']} epochs, lr={phase['lr']})")
        logging.info(f"{'='*60}\n")

         # Logique de dégel
        if phase.get("unfreeze") == "all":
            # Phase 3 : On dégèle tout
            for param in model.parameters(): param.requires_grad = True

        elif phase.get("unfreeze"):
            # Phase 2 : On dégèle partiellement (si supporté)
            if hasattr(model, 'unfreeze_backbone'):
                # Note: Assure-toi que 'unfreeze_layers' est bien dans le YAML pour la phase 2
                model.unfreeze_backbone(num_layers=phase.get("unfreeze_layers"))
            else:
                logging.warning("No unfreeze_backbone method")
        
        # Setup lr for the phase
        for group in optimizer.param_groups:
            group['lr'] = phase['lr']

        # Progress bar for this phase
        epoch_pbar = tqdm(range(phase["epochs"]), 
            desc="Training", 
            unit="epoch",
            position=0,
            leave=True)

        # Training loop
        for e in epoch_pbar:
            # Train 1 epoch
            train_loss = utils.train(model, train_loader, loss, optimizer, device)

            # Test
            test_metrics = utils.test(model, valid_loader, loss, device)

            # Scheduler step
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(test_metrics["loss"])
                else:
                    scheduler.step()

            # Update checkpoint
            updated = model_checkpoint.update(test_metrics["f1"])
            logging.info(
                f"[Phase {phase_idx+1}-{e+1}/{phase['epochs']}] loss: {test_metrics['loss']:.3f} "
                f"{'[>> BETTER <<]' if updated else ''} | {test_metrics}"
            )

            # Update the dashboard
            metrics = {"train_CE": train_loss, **test_metrics, "learning_rate": optimizer.param_groups[0]['lr']}
            if wandb_log is not None:
                logging.info("Logging on wandb")
                wandb_log(metrics)
                
            global_epoch += 1

def analyse_results(config, dataset, predictions, true_labels, class_names, model_path):
    """
    Analyse détaillée des prédictions avec matrices de confusion et analyses par groupes

    Args:
        config: Configuration du modèle
        dataset: Dataset (validation set avec labels)
        predictions: Liste des prédictions
        true_labels: Liste des vraies labels
        class_names: Noms des classes
        model_path: Chemin du modèle (pour sauvegarder les résultats à côté)
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, classification_report
    from PIL import Image
    from collections import defaultdict

    # Créer le dossier d'analyse à côté du modèle
    model_dir = pathlib.Path(model_path).parent
    analysis_dir = model_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    logging.info(f"= Saving analysis results to {analysis_dir}")

    # Convertir en numpy arrays
    true_labels = np.array(true_labels)
    predictions = np.array(predictions)

    # 1. RÉCUPÉRER LES MÉTADONNÉES DES IMAGES
    image_sizes = []
    filenames = []
    filepaths = []

    # Accéder aux indices du subset
    if hasattr(dataset, 'indices'):
        # C'est un Subset
        indices = dataset.indices
        base_dataset = dataset.dataset
        for idx in indices:
            filepath, _ = base_dataset.imgs[idx]
            filepaths.append(filepath)
            filenames.append(os.path.basename(filepath))

            # Récupérer la taille originale de l'image
            img = Image.open(filepath)
            width, height = img.size
            surface = width * height
            image_sizes.append(surface)
    else:
        # C'est un dataset normal
        for idx in range(len(dataset)):
            filepath, _ = dataset.imgs[idx]
            filepaths.append(filepath)
            filenames.append(os.path.basename(filepath))

            img = Image.open(filepath)
            width, height = img.size
            surface = width * height
            image_sizes.append(surface)

    image_sizes = np.array(image_sizes)
    filepaths = np.array(filepaths)

    # 2. MATRICE DE CONFUSION GLOBALE
    logging.info("= Generating confusion matrix")

    cm = confusion_matrix(true_labels, predictions)

    # MATRICE COMPACTE SANS LABELS (plus lisible)
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',  # annot=False !
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix (Absolute Counts)', fontsize=20, pad=20)
    plt.ylabel('True Label (class index)', fontsize=14)
    plt.xlabel('Predicted Label (class index)', fontsize=14)

    # Afficher seulement quelques labels (tous les 5 par exemple)
    tick_marks = np.arange(len(class_names))
    tick_labels = [f"{i}" for i in range(len(class_names))]
    plt.xticks(tick_marks[::5], tick_labels[::5], rotation=0)
    plt.yticks(tick_marks[::5], tick_labels[::5], rotation=0)

    plt.tight_layout()
    plt.savefig(analysis_dir / "confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()

    # MATRICE NORMALISÉE (encore plus lisible)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(20, 18))
    sns.heatmap(cm_normalized, annot=False, fmt='.2f', cmap='RdYlGn',  # Rouge=mauvais, Vert=bon
                vmin=0, vmax=1,
                cbar_kws={'label': 'Accuracy per class'})
    plt.title('Normalized Confusion Matrix (row-wise accuracy)', fontsize=20, pad=20)
    plt.ylabel('True Label (class index)', fontsize=14)
    plt.xlabel('Predicted Label (class index)', fontsize=14)

    plt.xticks(tick_marks[::5], tick_labels[::5], rotation=0)
    plt.yticks(tick_marks[::5], tick_labels[::5], rotation=0)

    plt.tight_layout()
    plt.savefig(analysis_dir / "confusion_matrix_normalized.png", dpi=150, bbox_inches='tight')
    plt.close()


    # 3. RAPPORT DE CLASSIFICATION
    logging.info("= Generating classification report")

    report = classification_report(true_labels, predictions, 
                                   target_names=class_names, 
                                   output_dict=True,
                                   zero_division=0)

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(analysis_dir / "classification_report.csv")

    # 4. ANALYSE PAR TAILLE D'IMAGE (10% - 40% - 40% - 10%)
    logging.info("= Analyzing by image size")

    # Créer des bins de taille avec les percentiles demandés
    size_percentiles = np.percentile(image_sizes, [0, 10, 50, 90, 100])
    size_bins = pd.cut(image_sizes, bins=size_percentiles, 
                       labels=['Small (0-10%)', 'Medium- (10-50%)', 'Medium+ (50-90%)', 'Large (90-100%)'],
                       include_lowest=True, duplicates='drop')

    size_analysis = []
    for size_group in size_bins.categories:
        mask = size_bins == size_group
        if mask.sum() == 0:
            continue

        group_true = true_labels[mask]
        group_pred = predictions[mask]

        accuracy = (group_true == group_pred).mean()
        count = mask.sum()

        # Calcul de la surface moyenne et range
        avg_surface = image_sizes[mask].mean()
        min_surface = image_sizes[mask].min()
        max_surface = image_sizes[mask].max()

        size_analysis.append({
            'size_group': size_group,
            'count': count,
            'percentage': 100 * count / len(predictions),
            'accuracy': accuracy,
            'avg_surface_pixels': avg_surface,
            'min_surface': min_surface,
            'max_surface': max_surface
        })

    size_df = pd.DataFrame(size_analysis)
    size_df.to_csv(analysis_dir / "analysis_by_size.csv", index=False)

    # Plot accuracy by size
    plt.figure(figsize=(10, 6))
    plt.bar(size_df['size_group'], size_df['accuracy'], color='skyblue', edgecolor='black')
    plt.axhline(y=(true_labels == predictions).mean(), color='red', linestyle='--', 
                label=f'Overall Accuracy: {(true_labels == predictions).mean():.3f}')
    plt.xlabel('Image Size Group', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Accuracy by Image Size', fontsize=14)
    plt.ylim(0, 1)
    plt.legend()
    plt.xticks(rotation=15)
    
    # Ajouter le nombre d'échantillons sur les barres
    for i, row in size_df.iterrows():
        plt.text(i, row['accuracy'] + 0.02, f"n={row['count']}\n({row['percentage']:.1f}%)", 
                ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(analysis_dir / "accuracy_by_size.png", dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"  Size analysis:")
    for _, row in size_df.iterrows():
        logging.info(f"    {row['size_group']}: {row['count']} images ({row['percentage']:.1f}%), "
                    f"accuracy={row['accuracy']:.3f}, avg_surface={row['avg_surface_pixels']:.0f}px")

    # 5. ANALYSE PAR TAILLE DE CLASSE (déséquilibre)
    logging.info("= Analyzing by class size")

    class_sizes = defaultdict(int)
    for label in true_labels:
        class_sizes[label] += 1

    class_analysis = []
    for class_idx in range(len(class_names)):
        mask = true_labels == class_idx
        if mask.sum() == 0:
            continue

        group_true = true_labels[mask]
        group_pred = predictions[mask]

        accuracy = (group_true == group_pred).mean()
        count = mask.sum()
        
        # Récupérer les métriques du rapport de classification
        class_report = report[class_names[class_idx]]

        class_analysis.append({
            'class': class_names[class_idx],
            'class_idx': class_idx,
            'training_size': count,
            'accuracy': accuracy,
            'precision': class_report['precision'],
            'recall': class_report['recall'],
            'f1_score': class_report['f1-score']
        })

    class_df = pd.DataFrame(class_analysis)
    class_df = class_df.sort_values('training_size')
    class_df.to_csv(analysis_dir / "analysis_by_class.csv", index=False)

    # Plot F1-score vs training size
    plt.figure(figsize=(12, 6))
    plt.scatter(class_df['training_size'], class_df['f1_score'], alpha=0.6, s=100)
    plt.xlabel('Training Size (number of samples)', fontsize=12)
    plt.ylabel('F1-Score', fontsize=12)
    plt.title('F1-Score vs Training Set Size', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Ajouter une ligne de tendance
    z = np.polyfit(class_df['training_size'], class_df['f1_score'], 1)
    p = np.poly1d(z)
    plt.plot(class_df['training_size'], p(class_df['training_size']), 
             "r--", alpha=0.5, label=f'Trend: y={z[0]:.4f}x+{z[1]:.2f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(analysis_dir / "f1_vs_training_size.png", dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"  Class imbalance:")
    logging.info(f"    Min class size: {class_df['training_size'].min()}")
    logging.info(f"    Max class size: {class_df['training_size'].max()}")
    logging.info(f"    Imbalance ratio: {class_df['training_size'].max() / class_df['training_size'].min():.2f}x")

    # 6. ANALYSE COMBINÉE (taille image × taille classe)
    logging.info("= Combined analysis (image size × class size)")

    combined_analysis = []
    for size_group in size_bins.categories:
        for class_idx in range(len(class_names)):
            size_mask = size_bins == size_group
            class_mask = true_labels == class_idx
            combined_mask = size_mask & class_mask

            if combined_mask.sum() == 0:
                continue

            group_true = true_labels[combined_mask]
            group_pred = predictions[combined_mask]
            accuracy = (group_true == group_pred).mean()

            combined_analysis.append({
                'size_group': size_group,
                'class': class_names[class_idx],
                'count': combined_mask.sum(),
                'accuracy': accuracy
            })

    combined_df = pd.DataFrame(combined_analysis)
    combined_df.to_csv(analysis_dir / "combined_analysis.csv", index=False)

    # Heatmap croisé
    pivot = combined_df.pivot_table(
        values='accuracy', 
        index='class', 
        columns='size_group', 
        aggfunc='mean'
    )

    plt.figure(figsize=(12, max(8, len(class_names) // 3)))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', 
                vmin=0, vmax=1, cbar_kws={'label': 'Accuracy'})
    plt.title('Accuracy: Class × Image Size', fontsize=14)
    plt.xlabel('Image Size Group', fontsize=12)
    plt.ylabel('Class', fontsize=12)
    plt.tight_layout()
    plt.savefig(analysis_dir / "combined_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 7. VISUALISATION DES PRÉDICTIONS PAR CLASSE
    logging.info("= Generating per-class prediction visualizations")
    
    vis_dir = analysis_dir / "class_predictions"
    vis_dir.mkdir(exist_ok=True)
    
    np.random.seed(42)  # Pour reproductibilité
    n_samples_per_class = 12  # 3x4 grid
    
    for class_idx, class_name in enumerate(class_names):
        # Trouver toutes les images de cette classe
        class_mask = true_labels == class_idx
        class_indices = np.where(class_mask)[0]
        
        if len(class_indices) == 0:
            continue
        
        # Sélectionner aléatoirement jusqu'à n_samples_per_class images
        n_samples = min(n_samples_per_class, len(class_indices))
        selected_indices = np.random.choice(class_indices, size=n_samples, replace=False)
        
        # Créer la figure
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        fig.suptitle(f'Class: {class_name} (True Label = {class_idx})', 
                    fontsize=16, fontweight='bold')
        axes = axes.flatten()
        
        for i, idx in enumerate(selected_indices):
            if i >= len(axes):
                break
                
            # Charger et afficher l'image
            img_path = filepaths[idx]
            img = Image.open(img_path)
            axes[i].imshow(img)
            axes[i].axis('off')
            
            # Ajouter le titre avec prédiction vs vérité
            pred_class = predictions[idx]
            pred_name = class_names[pred_class]
            
            # Couleur : vert si correct, rouge si incorrect
            color = 'green' if pred_class == class_idx else 'red'
            is_correct = '✓' if pred_class == class_idx else '✗'
            
            title = f'{is_correct} Pred: {pred_name} ({pred_class})\nTrue: {class_name} ({class_idx})'
            axes[i].set_title(title, fontsize=10, color=color, fontweight='bold')
            
            # Ajouter la taille de l'image
            width, height = img.size
            axes[i].text(0.5, -0.1, f'{width}×{height}px', 
                        transform=axes[i].transAxes, 
                        ha='center', fontsize=8, color='gray')
        
        # Cacher les axes non utilisés
        for i in range(n_samples, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(vis_dir / f"class_{class_idx}_{class_name}.png", 
                   dpi=200, bbox_inches='tight')
        plt.close()
        
        logging.info(f"  Generated visualization for class {class_name} ({n_samples} samples)")

    # ============================================
    # 8. VISUALISATION DES ÉCHECS PAR CLASSE (Section Ajoutée)
    # ============================================
    logging.info("= Generating per-class FAILURE visualizations")

    fail_dir = analysis_dir / "class_failures"
    fail_dir.mkdir(exist_ok=True)

    # On utilise 16 images pour faire une grille 4x4
    n_fails_per_class = 16 

    for class_idx, class_name in enumerate(class_names):
        # Trouver les indices où la VRAIE classe est celle-ci, mais la prédiction est FAUSSE
        # (True == Classe X) AND (Pred != Classe X)
        fail_mask = (true_labels == class_idx) & (predictions != class_idx)
        fail_indices = np.where(fail_mask)[0]

        # S'il n'y a aucune erreur pour cette classe, on passe (Bravo !)
        if len(fail_indices) == 0:
            continue

        # Sélectionner jusqu'à 16 erreurs aléatoirement
        n_samples = min(n_fails_per_class, len(fail_indices))
        selected_indices = np.random.choice(fail_indices, size=n_samples, replace=False)

        # Créer la figure (4x4)
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        fig.suptitle(f'FAILURES for Class: {class_name}\n(True Label = {class_idx})', 
                    fontsize=16, fontweight='bold', color='red')
        
        # Aplatir les axes pour itérer facilement
        axes_flat = axes.flatten()

        for i in range(len(axes_flat)):
            ax = axes_flat[i]
            
            if i < len(selected_indices):
                # On a une image à afficher
                idx = selected_indices[i]
                
                try:
                    img = Image.open(filepaths[idx])
                    ax.imshow(img)
                except Exception as e:
                    # En cas d'erreur de chargement
                    ax.text(0.5, 0.5, "Error loading img", ha='center')
                
                pred_label = predictions[idx]
                pred_name = class_names[pred_label] if pred_label < len(class_names) else str(pred_label)
                
                # Titre : Ce qu'on a prédit à tort
                ax.set_title(f"Wrongly Pred: {pred_name}\n({pred_label})", 
                            fontsize=10, color='red', fontweight='bold')
                ax.axis('off')
            else:
                # Case vide (si moins de 16 erreurs)
                ax.axis('off')

        plt.tight_layout()
        # Nom de fichier propre
        safe_classname = "".join([c if c.isalnum() else "_" for c in class_name])
        plt.savefig(fail_dir / f"fail_{class_idx}_{safe_classname}.png", dpi=150)
        plt.close()

    logging.info(f"✓ Analysis complete. Results saved in: {analysis_dir}")

    # 8. RÉSUMÉ GLOBAL
    logging.info("= Generating summary")

    overall_accuracy = (true_labels == predictions).mean()

    summary = {
        'Overall Accuracy': f"{overall_accuracy:.4f}",
        'Total Samples': len(predictions),
        'Number of Classes': len(class_names),
        'Macro Avg F1': f"{report['macro avg']['f1-score']:.4f}",
        'Weighted Avg F1': f"{report['weighted avg']['f1-score']:.4f}",
        'Class Imbalance Ratio': f"{class_df['training_size'].max() / class_df['training_size'].min():.2f}x",
        'Best Performing Size Group': size_df.loc[size_df['accuracy'].idxmax(), 'size_group'],
        'Worst Performing Size Group': size_df.loc[size_df['accuracy'].idxmin(), 'size_group'],
    }

    with open(analysis_dir / "summary.txt", 'w') as f:
        f.write("="*60 + "\n")
        f.write("ANALYSIS SUMMARY\n")
        f.write("="*60 + "\n\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("SIZE GROUP BREAKDOWN\n")
        f.write("="*60 + "\n\n")
        for _, row in size_df.iterrows():
            f.write(f"{row['size_group']}:\n")
            f.write(f"  Samples: {row['count']} ({row['percentage']:.1f}%)\n")
            f.write(f"  Accuracy: {row['accuracy']:.4f}\n")
            f.write(f"  Surface range: {row['min_surface']:.0f} - {row['max_surface']:.0f} pixels\n\n")

    logging.info(f"✓ Analysis complete! Results saved in {analysis_dir}")
    logging.info(f"  - Confusion matrices: confusion_matrix*.png")
    logging.info(f"  - Classification report: classification_report.csv")
    logging.info(f"  - Size analysis: analysis_by_size.csv, accuracy_by_size.png")
    logging.info(f"  - Class analysis: analysis_by_class.csv, f1_vs_training_size.png")
    logging.info(f"  - Combined analysis: combined_analysis.csv, combined_heatmap.png")
    logging.info(f"  - Per-class predictions: class_predictions/ ({len(class_names)} images)")
    logging.info(f"  - Summary: summary.txt")

    return analysis_dir


def test(config):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    print(f"Evaluating on device: {device}")

    # Charger la configuration
    data_config = config["data"]
    model_config = config["model"]
    test_path = data_config["testpath"]
    train_path = data_config["trainpath"]
    results_path = config["logging"].get("results_path", "./results")
    use_rgb = data_config.get("rgb", False)
    augmentations_v2 = data_config.get("augmentations_v2", False)
    dataset_size_divider = data_config.get("dataset_size_divider", 6)
    valid_ratio = data_config["valid_ratio"]

    # Récupération du mode (standard par défaut)
    dataset_mode = data_config.get("dataset_mode", "standard") 

    if use_rgb:
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]
    else:
        MEAN = [0.485]
        STD = [0.229]

    os.makedirs(results_path, exist_ok=True)

    logging.info("= Loading test data")

    # Transformation identique au train (SANS augmentation)
    resize = data_config["images_resize"]
    transform_list = []

    if not augmentations_v2:
        if not use_rgb:
            transform_list.append(transforms.Grayscale())

        transform_list.extend([
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD) 
        ])

        test_transform = transforms.Compose(transform_list)

    else:
        print("Using data augmentations v2")
        def square_pad(img):
            c, h, w = img.shape
            if h > w:
                padding_left = (h - w) // 2
                padding_right = h - w - padding_left
                padding_top = 0
                padding_bottom = 0
            else:
                padding_left = 0 
                padding_right = 0
                padding_top = (w - h) // 2
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
            transforms.Normalize(mean=MEAN, std=STD),
        ])

        test_transform = transforms.Compose(transform_list)

    logging.info("= Getting number of classes from training set")

    # Charger le dataset complet pour avoir les classes
    # --- CHARGEMENT DU DATASET SELON LE MODE ---
    if dataset_mode == "specialist":
        # Mode Spécialiste : Lit les préfixes des fichiers dans un dossier plat
        full_dataset = data.SpecialistDataset(
            root_dir=data_config["trainpath"], # Doit pointer vers le sous-dossier (ex: .../1_Copepoda)
            transform=test_transform
        )
    else:
        # Mode Standard (Général) : Structure ImageFolder classique
        full_dataset = torchvision.datasets.ImageFolder(
            root=data_config["trainpath"], 
            transform=test_transform,
        )

    targets = np.array(full_dataset.targets) 
    unique_classes, class_counts = np.unique(targets, return_counts=True)
    total_samples = len(targets)
    num_classes = len(unique_classes)
    class_weights_array = total_samples / (num_classes * class_counts)


    num_classes = len(full_dataset.classes)
    class_names = full_dataset.classes
    logging.info(f"  - Number of classes: {num_classes}")
    logging.info(f"  - Classes: {class_names}")

    # ✅ SPLIT TRAIN/VALID POUR L'ANALYSE
    valid_ratio = data_config.get("valid_ratio", 0.1)
    train_size = int((1 - valid_ratio) * len(full_dataset))
    valid_size = len(full_dataset) - train_size

    random.seed(42) 
    
    indices = list(range(len(full_dataset)))
    random.shuffle(indices)
    num_valid = int(valid_ratio * len(full_dataset))
    train_indices = indices[num_valid:]
    valid_indices = indices[:num_valid]
    valid_dataset = torch.utils.data.Subset(full_dataset, valid_indices)
   
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=data_config["batch_size"],
        shuffle=False, 
        num_workers=data_config["num_workers"],
        pin_memory=use_cuda
    )

    
    logging.info(f"  - Loaded {len(valid_dataset)} validation samples for analysis")

    # Charger les images de test Kaggle (SANS labels)
    if dataset_mode == "standard":
        logging.info("= Loading test dataset (standard mode)")
        test_dataset = torchvision.datasets.ImageFolder(
            root=test_path,
            transform=test_transform
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=data_config["batch_size"],
            shuffle=False, 
            num_workers=data_config["num_workers"],
            pin_memory=use_cuda
        )

        logging.info(f"  - Loaded {len(test_dataset)} test samples for Kaggle submission")

    # Charger le meilleur modèle
    logging.info("= Loading best model")

    if "model_path" in config["logging"]:
        model_path = pathlib.Path(config["logging"]["model_path"])
        logging.info(f"  - Using specified model: {model_path}")
    else:
        logdir = config["logging"]["logdir"]
        logdirs = sorted(pathlib.Path(logdir).glob("*"))
        if not logdirs:
            raise ValueError(f"No model found in {logdir}")
        latest_logdir = logdirs[-1]
        model_path = latest_logdir / "best_model.pt"
        logging.info(f"  - Using latest model: {model_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    logging.info(f"  - Loading model from {model_path}")

    # Reconstruire le modèle
    input_size = tuple(full_dataset[0][0].shape)
    model = models.build_model(model_config, input_size, num_classes)

    # Charger les poids
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    # ============================================
    # 1️⃣ ANALYSE SUR VALIDATION SET (avec labels)
    # ============================================
    logging.info("\n" + "="*50)
    logging.info("VALIDATION SET ANALYSIS (with true labels)")
    logging.info("="*50)

    valid_predictions = []
    valid_true_labels = []

    with torch.no_grad():
        for images, labels in tqdm(valid_loader, desc="Validation Inference", unit="batch"):
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            valid_predictions.extend(preds.cpu().numpy())
            valid_true_labels.extend(labels.numpy())

    # Appeler analyse_results avec le validation set
    analysis_dir = analyse_results(
        config=config,
        dataset=valid_dataset,
        predictions=valid_predictions,
        true_labels=valid_true_labels,
        class_names=class_names,
        model_path=model_path
    )

    logging.info(f"\n✓ Full analysis available in: {analysis_dir}")

    if dataset_mode == "standard":
        # ============================================
        # 2️⃣ INFÉRENCE SUR TEST SET (pour Kaggle)
        # ============================================
        logging.info("\n" + "="*50)
        logging.info("TEST SET INFERENCE (for Kaggle submission)")
        logging.info("="*50)

        test_predictions = []
        test_filenames = []

        with torch.no_grad():
            for images, _ in tqdm(test_loader, desc="Test Inference", unit="batch"):
                images = images.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                test_predictions.extend(preds.cpu().numpy())

        # Récupérer les noms de fichiers dans l'ordre
        for idx in range(len(test_dataset)):
            filepath = test_dataset.imgs[idx][0]
            filename = os.path.basename(filepath)
            test_filenames.append(filename)

        # Créer le DataFrame pour Kaggle
        import pandas as pd

        results_df = pd.DataFrame({
            'imgname': test_filenames,
            'label': test_predictions
        })

        # Sauvegarder
        output_file = os.path.join(results_path, "submission.csv")
        results_df.to_csv(output_file, index=False)

        logging.info(f"Kaggle submission saved to {output_file}")
        logging.info(f"  - Total predictions: {len(test_predictions)}")
        logging.info(f"  - Unique classes predicted: {len(set(test_predictions))}")

        # Afficher la distribution des prédictions
        logging.info("\nTest Prediction distribution:")
        for class_idx in sorted(set(test_predictions)):
            count = test_predictions.count(class_idx)
            percentage = 100 * count / len(test_predictions)
            class_name = class_names[class_idx] if class_idx < len(class_names) else f"Class_{class_idx}"
            logging.info(f"  {class_name}: {count} ({percentage:.2f}%)")

    else:
        output_file = None

    return output_file, analysis_dir

def validate(config):
    # 1. Setup : Device et Config
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logging.info(f"Evaluating on device: {device}")
    
    data_config = config["data"]
    model_config = config["model"]
    results_path = config["logging"].get("results_path", "./results")
    training_style = config["training"].get("style", "standard")
    
    os.makedirs(results_path, exist_ok=True)

    # 2. VALIDATION
    logging.info("= Rebuilding dataloaders for specific validation split...")
    
    _, valid_loader, input_size, num_classes, _ = data.get_dataloaders(
        data_config, training_style=training_style
    )
    
    subset = valid_loader.dataset
    if hasattr(subset, 'dataset'):
        source_dataset = subset.dataset
    else:
        source_dataset = subset # Cas peu probable avec ton code, mais sécurité
        
    # Récupération des noms de classes
    if hasattr(source_dataset, 'classes'):
        class_names = source_dataset.classes
    elif hasattr(source_dataset, 'categories'):
        class_names = source_dataset.categories
    else:
        class_names = [str(i) for i in range(num_classes)]
        
    logging.info(f"  - Classes found: {len(class_names)}")

    # --- CHARGEMENT DU MODÈLE ---
    logging.info("= Loading Model")
    if "model_path" in config["logging"]:
        model_path = pathlib.Path(config["logging"]["model_path"])
    else:
        # Fallback automatique sur le dernier modèle entrainé
        logdir = pathlib.Path(config["logging"]["logdir"])
        all_subdirs = [d for d in logdir.iterdir() if d.is_dir()]
        all_subdirs.sort()
        if not all_subdirs:
             raise ValueError(f"No log folder found in {logdir}")
        model_path = all_subdirs[-1] / "best_model.pt"
        
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    logging.info(f"  - Loading weights from: {model_path}")

    model = models.build_model(model_config, input_size, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # --- INFÉRENCE VALIDATION ---
    logging.info("\n" + "="*50)
    logging.info("VALIDATION SET ANALYSIS (With labels)")
    logging.info("="*50)

    valid_predictions = []
    valid_true_labels = []

    with torch.no_grad():
        for images, labels in tqdm(valid_loader, desc="Validation Inference"):
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            valid_predictions.extend(preds.cpu().numpy())
            valid_true_labels.extend(labels.numpy())

    # Génération des rapports (Matrice de confusion, etc.)
    analysis_dir = analyse_results(
        config=config,
        dataset=valid_loader.dataset, # On passe le subset (qui contient les indices corrects)
        predictions=valid_predictions,
        true_labels=valid_true_labels,
        class_names=class_names,
        model_path=model_path
    )
    logging.info(f"✓ Analysis saved to: {analysis_dir}")

    output_file = None

    # ============================================
    # 3. TEST KAGGLE (Uniquement si Standard)
    # ============================================
    if training_style == "standard":
        logging.info("\n" + "="*50)
        logging.info("TEST SET INFERENCE (Generate submission.csv)")
        logging.info("="*50)
        
        test_path = data_config["testpath"]
        
        # Astuce critique : On récupère la transformation EXACTE utilisée par get_dataset pour la validation
        # source_dataset est l'objet créé par : get_dataset(..., test_transform, ...)
        kaggle_transform = source_dataset.transform
        
        # Chargement du Test Set
        # Utilisation de ImageFolder. Attention : structure attendue = root/dossier/image.jpg
        test_dataset = torchvision.datasets.ImageFolder(
            root=test_path,
            transform=kaggle_transform
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=data_config["batch_size"],
            shuffle=False,
            num_workers=data_config["num_workers"],
            pin_memory=use_cuda
        )
        
        logging.info(f"  - Loaded {len(test_dataset)} test images.")

        test_predictions = []
        test_filenames = []

        with torch.no_grad():
            for images, _ in tqdm(test_loader, desc="Test Inference"):
                images = images.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                test_predictions.extend(preds.cpu().numpy())

        # Récupération des noms de fichiers
        for idx in range(len(test_dataset)):
            full_path = test_dataset.imgs[idx][0]
            filename = os.path.basename(full_path) 
            test_filenames.append(filename)

        # Création du DataFrame
        results_df = pd.DataFrame({
            'imgname': test_filenames,
            'label': test_predictions
        })

        output_file = os.path.join(results_path, "submission.csv")
        results_df.to_csv(output_file, index=False)

        logging.info(f"✅ Kaggle submission saved to: {output_file}")
        
        # Stats basiques
        unique, counts = np.unique(test_predictions, return_counts=True)
        logging.info("  - Prediction stats:")
        for cls_idx, count in zip(unique, counts):
            cls_name = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx)
            logging.info(f"    {cls_name}: {count} ({count/len(test_predictions)*100:.1f}%)")
            
    else:
        logging.info("Skipping Kaggle submission generation (Mode is not 'standard').")

    return output_file, analysis_dir




if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    if len(sys.argv) != 3:
        logging.error(f"Usage : {sys.argv[0]} config.yaml <train|test>")
        sys.exit(-1)

    logging.info("Loading {}".format(sys.argv[1]))
    config = yaml.safe_load(open(sys.argv[1], "r"))

    command = sys.argv[2]
    eval(f"{command}(config)")

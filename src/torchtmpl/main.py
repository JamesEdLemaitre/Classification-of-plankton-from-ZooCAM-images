# coding: utf-8

# Standard imports
import logging
import sys
import os
import pathlib

# External imports
import yaml
import wandb
import torch
import torchinfo.torchinfo as torchinfo
from tqdm import tqdm
from torchvision import transforms
import torchvision

# Local imports
from . import data
from . import models
from . import optim
from . import utils
import matplotlib.pyplot as plt
import numpy as np

def get_device(gpu_id=None):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        use_cuda = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        use_cuda = True  # On garde le nom pour compatibilité
    else:
        device = torch.device("cpu")
        use_cuda = False

    return device, use_cuda

def train(config):
    device, use_cuda = get_device()
    print(f"Training on device: {device}")

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

    train_loader, valid_loader, input_size, num_classes, class_weights = data.get_dataloaders(
        data_config, use_cuda
    )
    class_weights = class_weights.to(device)

    # Afficher la distribution des classes du validation_loader
    class_counts = [0] * num_classes
    for X, y in valid_loader:
        for label in y:
            class_counts[label.item()] += 1

    total = 0
    for i, count in enumerate(class_counts):
        total += count
    
    print("\n📊 Validation Set Distribution:")
    for i, count in enumerate(class_counts):
        print(f"  Class {i}: {count/total}")
    
    # Calculer les class weights pour gérer le déséquilibre
    # logging.info("= Computing class weights")
    # from sklearn.utils.class_weight import compute_class_weight
    # import numpy as np
    
    # train_labels = [label for _, label in train_loader.dataset]
    # class_weights = compute_class_weight(
    #     'balanced',
    #     classes=np.unique(train_labels),
    #     y=train_labels
    # )
    # class_weights = torch.FloatTensor(class_weights)
    # logging.info(f"  Class weights: {class_weights.numpy()}")


    # Build the model
    logging.info("= Model")
    model_config = config["model"]
    model = models.build_model(model_config, input_size, num_classes)

    if "pretrained_model_path" in config["logging"]:
        pretrained_path = config["logging"]["pretrained_model_path"]
        if os.path.exists(pretrained_path):
            logging.info(f"= Loading pretrained model from {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location=device)
            model.load_state_dict(checkpoint)
            logging.info(f"  Pretrained weights loaded")
        else:
            logging.warning(f"  Pretrained model not found at {pretrained_path}")

    # Build the loss
    logging.info("= Loss")
    loss = optim.get_loss(config["loss"], class_weights=class_weights, device=device)

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
        + f"Validation : {valid_loader.dataset.dataset}"#Va afficher le meme nombre de sample pour les deux mais c'est normale, c'est parce qu'on parle de leur subset
        + f"Vrai taille du train : {len(train_loader)}"
    )
    with open(logdir / "summary.txt", "w") as f:
        f.write(summary_text)
    logging.info(summary_text)
    if wandb_log is not None:
        wandb.log({"summary": summary_text})

    # Put model on device:
    model = model.to(device)

    # Build the optimizer
    logging.info("= Optimizer")
    optim_config = config["optim"]
    optimizer = optim.get_optimizer(optim_config, model.parameters())

    logging.info("= Learning Rate Scheduler")
    num_steps = config["nepochs"] * len(train_loader)  #formule
    if "scheduler" in config:
        scheduler_config = config["scheduler"]
        scheduler = optim.get_scheduler(scheduler_config, optimizer, num_steps)
        logging.info(f"  Using scheduler: {scheduler_config['class']}")
    else:
        scheduler = None
        logging.info("  No scheduler specified")


    # Define the early stopping callback
    model_checkpoint = utils.ModelCheckpoint(
        model, str(logdir / "best_model.pt"), min_is_best=True
    )

    # Listes pour stocker les losses
    train_losses = []
    valid_losses = []

    epoch_pbar = tqdm(
        range(config["nepochs"]), 
        desc="Training", 
        unit="epoch",
        position=0,
        leave=True
    )
    

    for e in epoch_pbar:
        # Train 1 epoch
        train_loss = utils.train(model, train_loader, loss, optimizer, device, scheduler=scheduler)
        train_losses.append(train_loss)

        # Test
        test_loss, test_metrics = utils.test(model, valid_loader, loss, device)
        valid_losses.append(test_loss)

        if scheduler is not None:
            # Différent selon le type de scheduler
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(test_loss)  
            '''else:
                scheduler.step()  '''
            
            # Log le learning rate actuel
            current_lr = optimizer.param_groups[0]['lr']
            logging.info(f"  Current LR: {current_lr:.6f}")

        updated = model_checkpoint.update(test_loss)
        
        #inutile il y a la focal loss j'ai pas besoin de faire une f1 loss
        #updated = model_checkpoint.update(metrics['f1'])
        #

        logging.info(
            "[%d/%d] Train loss: %.3f | Validation loss : %.3f %s | metrics: %s"
            % (
                e,
                config["nepochs"],
                train_loss,
                test_loss,
                "[>> BETTER <<]" if updated else "",
                test_metrics,
            )
        )

        # Update the dashboard
        metrics = {"train_CE": train_loss, "validation_CE": test_loss, **test_metrics, "learning_rate": optimizer.param_groups[0]['lr']}
        if wandb_log is not None:
            logging.info("Logging on wandb")
            wandb_log(metrics)

    # Créer et sauvegarder le graphique des losses
    plt.figure(figsize=(10, 6))
    epochs = np.arange(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs, valid_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Sauvegarder le graphique
    plot_path = logdir / "loss_curves.png"
    plt.savefig(plot_path, dpi=300)
    logging.info(f"Loss curves saved to {plot_path}")
    
    # Afficher le graphique (si vous êtes en mode interactif)
    # plt.show()
    plt.close()
    
    # Training terminé
    logging.info("=" * 60)
    logging.info("✅ TRAINING COMPLETED!")
    logging.info(f"Best model saved at: {logdir / 'best_model.pt'}")
    logging.info(f"Best validation loss: {model_checkpoint.best_score}")
    logging.info("=" * 60)

    #A ajouter pour éviter une evaluation en prenant le mauvais modèle, ça écrit dans le config le path du modele entrainer pour le test apres qui s'enchaine
    '''config["logging"]["model_path"] = str(logdir / "best_model.pt")
    with open(logdir / "config.yaml", "w") as file:
        yaml.dump(config, file)'''
    
    if wandb_log is not None:
        wandb.finish()


def test(config):
    device, use_cuda = get_device()
    print(f"Evaluating on device: {device}")

    # Charger la configuration
    data_config = config["data"]
    model_config = config["model"]
    test_path = data_config["testpath"]
    train_path = data_config["trainpath"]
    results_path = config["logging"].get("results_path", "./results")
    use_rgb = data_config.get("rgb", False)

    # Créer le dossier de résultats
    os.makedirs(results_path, exist_ok=True)

    logging.info("= Loading test data")
    
    # Transformation identique au train (SANS augmentation)
    resize = data_config["images_resize"]
    transform_list = []
    
    if not use_rgb:
        transform_list.append(transforms.Grayscale())  # 1 canal
    # Sinon, reste en RGB natif (3 canaux)
    
    transform_list.extend([
        transforms.Resize((resize, resize)),
        transforms.ToTensor()
    ])

    transform_list = transforms.Compose(transform_list)

    logging.info("= Getting number of classes from training set")
    train_dataset = torchvision.datasets.ImageFolder(
        root=train_path,
        transform=transform_list
    )
    num_classes = len(train_dataset.classes)
    class_names = train_dataset.classes
    logging.info(f"  - Number of classes: {num_classes}")
    logging.info(f"  - Classes: {class_names}")

    # Charger les images de test (sans labels)
    test_dataset = torchvision.datasets.ImageFolder(
        root=test_path,
        transform=transform_list
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=data_config["batch_size"],
        shuffle=False, 
        num_workers=data_config["num_workers"],
        pin_memory=use_cuda
    )

    logging.info(f"  - Loaded {len(test_dataset)} test samples")
    
    # Charger le meilleur modèle
    logging.info("= Loading best model")

    if "model_path" in config["logging"]:
        model_path = pathlib.Path(config["logging"]["model_path"])
        logging.info(f"  - Using specified model: {model_path}")
    else:
        # Sinon, chercher le dernier dossier
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
    input_size = tuple(test_dataset[0][0].shape)
    model = models.build_model(model_config, input_size, num_classes)
    
    # Charger les poids
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    logging.info("= Running inference")
    
    predictions = []
    filenames = []
    
    with torch.no_grad():
        for images, _ in tqdm(test_loader, desc="Inference", unit="batch"):
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
    
    # Récupérer les noms de fichiers dans l'ordre
    for idx in range(len(test_dataset)):
        filepath = test_dataset.imgs[idx][0]
        filename = os.path.basename(filepath)
        filenames.append(filename)
    
    # Créer le DataFrame pour Kaggle
    import pandas as pd
    
    results_df = pd.DataFrame({
        'imgname': filenames,
        'label': predictions
    })
    
    # Sauvegarder
    output_file = os.path.join(results_path, "submission.csv")
    results_df.to_csv(output_file, index=False)
    
    logging.info(f"Results saved to {output_file}")
    logging.info(f"  - Total predictions: {len(predictions)}")
    logging.info(f"  - Unique classes predicted: {len(set(predictions))}")
    
    # Afficher la distribution des prédictions
    logging.info("\n Prediction distribution:")
    for class_idx in sorted(set(predictions)):
        count = predictions.count(class_idx)
        percentage = 100 * count / len(predictions)
        class_name = test_dataset.classes[class_idx] if class_idx < len(test_dataset.classes) else f"Class_{class_idx}"
        logging.info(f"  {class_name}: {count} ({percentage:.2f}%)")
    
    return output_file
    

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    if len(sys.argv) != 3:
        logging.error(f"Usage : {sys.argv[0]} config.yaml <train|test>")
        sys.exit(-1)

    logging.info("Loading {}".format(sys.argv[1]))
    config = yaml.safe_load(open(sys.argv[1], "r"))

    command = sys.argv[2]
    eval(f"{command}(config)")

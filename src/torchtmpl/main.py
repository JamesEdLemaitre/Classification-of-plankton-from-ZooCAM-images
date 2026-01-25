# coding: utf-8

# Standard imports
import logging
import sys
import os
import pathlib
from tqdm import tqdm

# External imports
import yaml
import wandb
import torch
import torchinfo.torchinfo as torchinfo

# Local imports
from . import data
from . import models
from . import optim
from . import utils


def train(config):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

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

    train_loader, valid_loader, input_size, num_classes = data.get_dataloaders(
        data_config, use_cuda
    )

    # Build the model
    logging.info("= Model")
    model_config = config["model"]
    model = models.build_model(model_config, input_size, num_classes)
    model.to(device)

    # Build the loss
    logging.info("= Loss")
    loss = optim.get_loss(config["loss"]["class"])

    # Build the optimizer
    logging.info("= Optimizer")
    optim_config = config["optim"]
    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.get_optimizer(optim_config, params_to_update)

    # Build the lr-scheduler
    logging.info("= Learning Rate Scheduler")
    scheduler = None
    if "scheduler" in config:
        # On appelle la factory
        scheduler = optim.get_scheduler(config["scheduler"], optimizer)
        logging.info(f"  Using scheduler: {config['scheduler']['class']}")
    else:
        logging.info("  No scheduler specified")

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
    with open(logdir / "summary.txt", "w") as f:
        f.write(summary_text)
    logging.info(summary_text)
    if wandb_log is not None:
        wandb.log({"summary": summary_text})

    # Define the early stopping callback
    model_checkpoint = utils.ModelCheckpoint(
        model, str(logdir / "best_model.pt"), min_is_best=False
    )

    epoch_pbar = tqdm(range(config["nepochs"]), desc="Epochs")  
    for e in epoch_pbar:
        # Train 1 epoch
        train_loss = utils.train(model, train_loader, loss, optimizer, device)

        # Test
        test_loss, metrics = utils.test(model, valid_loader, loss, device)

        if scheduler is not None:
            # ReduceLROnPlateau prend un argument (la loss ou metrics)
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(test_loss) 
            else:
                # Les autres (StepLR, Cosine, etc.) n'en prennent pas
                scheduler.step()

            # Log du LR dans WandB
            current_lr = optimizer.param_groups[0]['lr']
            if wandb_log:
                wandb_log({"learning_rate": current_lr}, commit=False)

        updated = model_checkpoint.update(metrics["f1"])

        epoch_pbar.set_description(f"Epoch {e+1} | F1: {metrics['f1']:.3f} | Loss: {test_loss:.3f}")

        logging.info(
            "[%d/%d] Train Loss: %.3f | Test Loss: %.3f | Metrics: %s %s"
            % (
                e + 1,
                config["nepochs"],
                train_loss,
                test_loss,
                metrics,
                "[>> BEST <<]" if updated else "",
            )
        )

        # Update the dashboard
        metrics = {"train_CE": train_loss, "test_CE": test_loss}
        if wandb_log is not None:
            logging.info("Logging on wandb")
            wandb_log(metrics)


def test(config):
    raise NotImplementedError

from torch.utils.data import Dataset, DataLoader
import glob
from PIL import Image
import pandas as pd
from torchvision import transforms

# class InferenceDataset(Dataset):
#     """
#     Dataset pour charger des images depuis un dossier (plat),
#     insensible à la casse des extensions (jpg, JPG, png...).
#     """
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
        
#         # Extensions supportées
#         valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        
#         self.files = []
#         # On liste tous les fichiers et on filtre par extension (en minuscule)
#         if os.path.isdir(root_dir):
#             for f in os.listdir(root_dir):
#                 if f.lower().endswith(valid_extensions):
#                     self.files.append(os.path.join(root_dir, f))
        
#         self.files = sorted(self.files)
        
#         # Debug info
#         if len(self.files) == 0:
#             logging.warning(f"⚠️  WARNING: No images found in {root_dir}")
#             logging.warning(f"    Content of directory: {os.listdir(root_dir)[:10]} ...")
#         else:
#             logging.info(f"✅ Found {len(self.files)} images in {root_dir}")

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, idx):
#         path = self.files[idx]
#         filename = os.path.basename(path)
#         # .convert("RGB") est crucial pour éviter les erreurs sur les images N&B ou avec Alpha
#         image = Image.open(path).convert("RGB")
        
#         if self.transform:
#             image = self.transform(image)
            
#         return image, filename


# def generate_csv(config):
#     """
#     source @gemini 3 pro
    
#     """
    
#     print("/////////////////////////////////////////////////////////////")
#     print("/////////////////////////////////////////////////////////////")
#     print("/////////////////////////////////////////////////////////////")
#     print("/////////////////////////////////////////////////////////////")
#     print("/////////////////////////////////////////////////////////////")
#     print("SCRIPT TEMPROAIRE NON VERIFIE QUI FAIT DES TRANSFORMATIONS PAS VERIFIEES ET FIXES")
#     print("/////////////////////////////////////////////////////////////")
#     print("/////////////////////////////////////////////////////////////")
#     print("/////////////////////////////////////////////////////////////")
#     print("/////////////////////////////////////////////////////////////")

#     """
#     Génère le fichier de soumission CSV à partir d'un modèle entraîné.
#     """
#     use_cuda = torch.cuda.is_available()
#     device = torch.device("cuda") if use_cuda else torch.device("cpu")
#     logging.info(f"Inference on device: {device}")

#     # 1. Récupérer les métadonnées (taille d'entrée, nombre de classes)
#     logging.info("Loading dataset metadata...")
#     data_config = config["data"]
#     # On ignore les loaders ici, on veut juste input_size et num_classes
#     _, _, input_size, num_classes = data.get_dataloaders(data_config, use_cuda)

#     # 2. Créer la transformation de validation manuellement
#     # On ne peut pas la récupérer depuis le dataset, donc on la recrée.
#     # On suppose une entrée carrée et une normalisation ImageNet standard.
    
#     # Extraction de la dimension spatiale (H, W) depuis input_size
#     # input_size est souvent un tuple (C, H, W) ex: (3, 224, 224) ou juste un int
#     if isinstance(input_size, int):
#         img_size = input_size
#     elif isinstance(input_size, (tuple, list)):
#         # Si (C, H, W) on prend H, sinon on prend le premier élément
#         img_size = input_size[1] if len(input_size) == 3 else input_size[0]
#     else:
#         img_size = 224 # Valeur par défaut de sécurité
        
#     logging.info(f"Building inference transform with size: {img_size}x{img_size}")
    
#     val_transform = transforms.Compose([
#         transforms.Resize((img_size, img_size)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

#     # 3. Charger le modèle
#     logging.info("Building model...")
#     model_config = config["model"]
#     model = models.build_model(model_config, input_size, num_classes)
    
#     # 4. Charger les poids
#     if "model_path" not in model_config:
#         # Fallback : on cherche dans le dossier de log si non spécifié, ou on lève une erreur
#         raise ValueError("You must specify 'model_path' in the 'model' section of config.yaml")
    
#     weights_path = model_config["model_path"]
#     logging.info(f"Loading weights from {weights_path}")
    
#     # map_location est important si on a entraîné sur GPU et qu'on infère sur CPU (ou vice versa)
#     state_dict = torch.load(weights_path, map_location=device)
    
#     # Parfois le state_dict contient une clé 'state_dict' ou 'model_state_dict'
#     if "model_state_dict" in state_dict:
#         state_dict = state_dict["model_state_dict"]
        
#     model.load_state_dict(state_dict)
#     model.to(device)
#     model.eval()

#     # 5. Créer le Dataset et Loader de Test
#     test_path = data_config.get("test_path")
#     if not test_path:
#         # Essai de deviner le chemin si non spécifié
#         test_path = os.path.join(data_config["path"], "test")
    
#     logging.info(f"Loading test images from : {test_path}")
#     if not os.path.exists(test_path):
#         raise FileNotFoundError(f"Test directory not found: {test_path}")

#     test_dataset = InferenceDataset(root_dir=test_path, transform=val_transform)
#     test_loader = DataLoader(test_dataset, batch_size=config["data"]["batch_size"], shuffle=False, num_workers=4)

#     # 6. Inférence
#     test_filenames = []
#     test_predictions = []

#     logging.info("Starting inference...")
#     with torch.no_grad():
#         for images, filenames in tqdm(test_loader, desc="Inference"):
#             images = images.to(device)
            
#             outputs = model(images)
#             _, preds = torch.max(outputs, 1)
            
#             test_filenames.extend(filenames)
#             test_predictions.extend(preds.cpu().tolist())

#     # 7. Sauvegarde CSV
#     results_path = "results"
#     if not os.path.exists(results_path):
#         os.makedirs(results_path)

#     results_df = pd.DataFrame({
#         'imgname': test_filenames,
#         'label': test_predictions
#     })

#     output_file = os.path.join(results_path, "submission.csv")
#     results_df.to_csv(output_file, index=False)
    
#     logging.info(f"Submission saved to : {output_file}")
#     print(results_df.head())

import torch.nn.functional as F # Nécessaire pour le softmax

class InferenceDataset(Dataset):
    """
    Dataset pour charger des images depuis un dossier (plat),
    insensible à la casse des extensions (jpg, JPG, png...).
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Extensions supportées
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        
        self.files = []
        if os.path.isdir(root_dir):
            for f in os.listdir(root_dir):
                if f.lower().endswith(valid_extensions):
                    self.files.append(os.path.join(root_dir, f))
        
        self.files = sorted(self.files)

        if len(self.files) == 0:
            logging.warning(f"⚠️  WARNING: No images found in {root_dir}")
        else:
            logging.info(f"✅ Found {len(self.files)} images in {root_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        filename = os.path.basename(path)
        try:
            image = Image.open(path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            logging.error(f"Error loading {path}: {e}")
            # Retourne une image noire en cas d'erreur pour ne pas planter le batch
            image = torch.zeros((3, 224, 224)) 

        return image, filename

def generate_csv(config):
    """
    Génère le fichier de soumission CSV avec Test Time Augmentation (TTA).
    5 vues par image : Original, Flip H, Flip V, Rot90, Rot270.
    """
    
    print("="*60)
    print("🚀 LANCEMENT INFERENCE AVEC TTA (Test Time Augmentation)")
    print("   Stratégie : Ensembling de 5 transformations géométriques")
    print("   (Original + Flip H + Flip V + Rot90 + Rot270)")
    print("="*60)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    logging.info(f"Inference on device: {device}")

    # 1. Métadonnées
    data_config = config["data"]
    # On récupère input_size et num_classes via ton loader existant
    # Note: on met use_cuda=False ici juste pour charger la config sans créer de workers GPU
    try:
        _, _, input_size, num_classes = data.get_dataloaders(data_config, use_cuda=False)
    except:
        # Fallback si get_dataloaders échoue sans les données train
        logging.warning("Impossible de charger les métadonnées via get_dataloaders. Valeurs par défaut utilisées.")
        input_size = (3, 224, 224)
        # IL FAUT CONNAITRE LE NOMBRE DE CLASSES EXACT DU MODELE SAUVEGARDÉ
        # Si tu as une erreur de dimension plus bas, vérifie ce nombre.
        num_classes = config.get("model", {}).get("num_classes", 86) 

    # Gestion de la taille d'image
    if isinstance(input_size, int):
        img_h, img_w = input_size, input_size
    elif isinstance(input_size, (tuple, list)) and len(input_size) == 3:
        img_h, img_w = input_size[1], input_size[2]
    else:
        img_h, img_w = 224, 224

    logging.info(f"Building inference transform: {img_h}x{img_w}")

    # Transformation de base (Normalisation ImageNet standard)
    val_transform = transforms.Compose([
        transforms.Resize((img_h, img_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Modèle
    logging.info("Building model...")
    model_config = config["model"]
    model = models.build_model(model_config, input_size, num_classes)

    weights_path = model_config["model_path"]
    logging.info(f"Loading weights from {weights_path}")
    
    state_dict = torch.load(weights_path, map_location=device)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    
    # Gestion des erreurs de taille de couche finale si le nb de classes a changé
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        logging.error("❌ ERREUR DE CHARGEMENT DES POIDS : Les dimensions ne correspondent pas.")
        logging.error("Vérifie que 'num_classes' est correct par rapport au modèle entraîné.")
        raise e

    model.to(device)
    model.eval()

    # 3. Dataset Test
    test_path = data_config.get("test_path")
    if not test_path:
        test_path = os.path.join(data_config["path"], "test")

    if not os.path.exists(test_path):
        logging.error(f"❌ Test directory not found: {test_path}")
        return

    test_dataset = InferenceDataset(root_dir=test_path, transform=val_transform)
    
    if len(test_dataset) == 0:
        logging.error("❌ Dataset vide. Arrêt.")
        return

    test_loader = DataLoader(test_dataset, batch_size=data_config["batch_size"], shuffle=False, num_workers=4)

    # 4. Inférence avec TTA
    test_filenames = []
    test_predictions = []

    logging.info(f"Starting TTA Inference on {len(test_dataset)} images...")
    
    with torch.no_grad():
        for images, filenames in tqdm(test_loader, desc="TTA Prediction"):
            images = images.to(device) # Shape: [Batch, 3, H, W]

            # --- TTA : 5 augmentations ---
            # NOTE : Les rotations/flips fonctionnent bien sur les tenseurs normalisés 
            # car la géométrie ne change pas la distribution des valeurs de pixel.

            # 1. Original
            out1 = model(images)
            prob1 = F.softmax(out1, dim=1)

            # 2. Flip Horizontal
            # dims=(3) correspond à la largeur (W) pour un tenseur (B, C, H, W)
            out2 = model(torch.flip(images, dims=[3]))
            prob2 = F.softmax(out2, dim=1)

            # 3. Flip Vertical
            # dims=(2) correspond à la hauteur (H)
            out3 = model(torch.flip(images, dims=[2]))
            prob3 = F.softmax(out3, dim=1)

            # 4. Rotation 90° (Anti-horaire)
            # dims=[2, 3] sont H et W
            out4 = model(torch.rot90(images, k=1, dims=[2, 3]))
            prob4 = F.softmax(out4, dim=1)

            # 5. Rotation 270° ( = -90°)
            out5 = model(torch.rot90(images, k=3, dims=[2, 3]))
            prob5 = F.softmax(out5, dim=1)

            # --- Moyenne des probabilités ---
            avg_probs = (prob1 + prob2 + prob3 + prob4 + prob5) / 5.0

            # Argmax final
            _, preds = torch.max(avg_probs, 1)

            test_filenames.extend(filenames)
            test_predictions.extend(preds.cpu().tolist())

    # 5. Sauvegarde
    results_path = "results"
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    results_df = pd.DataFrame({
        'imgname': test_filenames,
        'label': test_predictions
    })

    # Timestamp pour ne pas écraser les essais précédents
    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_file = os.path.join(results_path, f"submission_tta_{timestamp}.csv")
    
    results_df.to_csv(output_file, index=False)

    logging.info(f"✅ Submission saved to : {output_file}")
    logging.info(f"   (Head of CSV):\n{results_df.head()}")



if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    if len(sys.argv) != 3:
        logging.error(f"Usage : {sys.argv[0]} config.yaml <train|test>")
        sys.exit(-1)

    logging.info("Loading {}".format(sys.argv[1]))
    config = yaml.safe_load(open(sys.argv[1], "r"))

    command = sys.argv[2]
    eval(f"{command}(config)")

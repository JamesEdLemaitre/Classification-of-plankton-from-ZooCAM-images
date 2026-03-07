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
    loss = optim.get_loss(config["loss"], device=device)

    # Build Mixup/CutMix if configured
    mixup_fn = None
    mixup_loss_fn = None
    if "mixup" in config:
        logging.info("= Mixup/CutMix")
        mixup_fn = optim.get_mixup_fn(config["mixup"], num_classes)
        mixup_loss_fn = optim.SoftTargetCrossEntropy()
        logging.info(f"  Mixup alpha={config['mixup'].get('mixup_alpha', 0.8)}, "
                     f"CutMix alpha={config['mixup'].get('cutmix_alpha', 1.0)}, "
                     f"switch_prob={config['mixup'].get('switch_prob', 0.5)}, "
                     f"label_smoothing={config['mixup'].get('label_smoothing', 0.1)}")

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
    train_ds_info = str(getattr(train_loader.dataset, 'dataset', train_loader.dataset))
    if valid_loader is not None:
        valid_ds_info = str(getattr(valid_loader.dataset, 'dataset', valid_loader.dataset))
    else:
        valid_ds_info = "None (valid_ratio=0, training on 100% data)"
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
        + f"Train : {train_ds_info}\n"
        + f"Validation : {valid_ds_info}"
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

    checkpoint_interval = config.get("checkpoint_interval", 10)

    # History for training curves
    history = {
        "train_loss": [], "valid_loss": [],
        "train_f1": [], "valid_f1": [], "valid_f1_tta": [],
    }

    epoch_pbar = tqdm(range(config["nepochs"]), desc="Epochs")  
    for e in epoch_pbar:
        # Train 1 epoch
        train_loss, train_f1 = utils.train(model, train_loader, loss, optimizer, device,
                                           mixup_fn=mixup_fn, mixup_loss_fn=mixup_loss_fn)

        if scheduler is not None:
            # ReduceLROnPlateau needs a metric; others don't
            if valid_loader is not None and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # Will step after we compute test_loss below
                pass
            else:
                scheduler.step()

            current_lr = optimizer.param_groups[0]['lr']
            if wandb_log:
                wandb_log({"learning_rate": current_lr}, commit=False)

        if valid_loader is not None:
            # Validation available
            test_loss, metrics = utils.test(model, valid_loader, loss, device)

            if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(test_loss)

            updated = model_checkpoint.update(metrics["f1"])
            epoch_pbar.set_description(f"Epoch {e+1} | F1: {metrics['f1']:.3f} | Loss: {test_loss:.3f}")

            logging.info(
                "[%d/%d] Train Loss: %.3f | Test Loss: %.3f | Train F1: %.3f | Metrics: %s %s"
                % (
                    e + 1,
                    config["nepochs"],
                    train_loss,
                    test_loss,
                    train_f1,
                    metrics,
                    "[>> BEST <<]" if updated else "",
                )
            )

            # Update history & plot curves
            history["train_loss"].append(train_loss)
            history["valid_loss"].append(test_loss)
            history["train_f1"].append(train_f1)
            history["valid_f1"].append(metrics["f1"])
            history["valid_f1_tta"].append(metrics["f1_tta"])
            utils.plot_training_curves(history, logdir)

            wb_metrics = {"train_CE": train_loss, "test_CE": test_loss}
            if wandb_log is not None:
                wandb_log(wb_metrics)

            if (e + 1) % checkpoint_interval == 0:
                ckpt_path = logdir / f"epoch_{e+1}.pt"
                torch.save(model.state_dict(), ckpt_path)
                logging.info(f"  [Checkpoint] Saved {ckpt_path}")
        else:
            # No validation (valid_ratio=0) — log train only + periodic checkpoints
            epoch_pbar.set_description(f"Epoch {e+1} | Train Loss: {train_loss:.3f} | Train F1: {train_f1:.3f}")
            logging.info(
                "[%d/%d] Train Loss: %.3f | Train F1: %.3f"
                % (e + 1, config["nepochs"], train_loss, train_f1)
            )
            if wandb_log is not None:
                wandb_log({"train_CE": train_loss})

            # Update history & plot curves (train only)
            history["train_loss"].append(train_loss)
            history["train_f1"].append(train_f1)
            utils.plot_training_curves(history, logdir)

            if (e + 1) % checkpoint_interval == 0:
                ckpt_path = logdir / f"epoch_{e+1}.pt"
                torch.save(model.state_dict(), ckpt_path)
                logging.info(f"  [Checkpoint] Saved {ckpt_path}")

    # Always save final model when no validation (no best_model selection possible)
    if valid_loader is None:
        final_path = logdir / "final_model.pt"
        torch.save(model.state_dict(), final_path)
        logging.info(f"Training complete. Final model saved to {final_path}")


def test(config):
    raise NotImplementedError


def train_dinov2(config):
    """
    Two-phase DINOv2 fine-tuning with best practices:
      Phase 1: Frozen backbone → train classification head only (warmup)
      Phase 2: Unfreeze backbone → fine-tune with layer-wise LR decay + CosineAnnealing
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    if "wandb" in config["logging"]:
        wandb_config = config["logging"]["wandb"]
        wandb.init(project=wandb_config["project"], entity=wandb_config["entity"])
        wandb_log = wandb.log
        wandb_log(config)
    else:
        wandb_log = None

    # Build the dataloaders
    logging.info("= Building the dataloaders")
    data_config = config["data"]
    train_loader, valid_loader, input_size, num_classes = data.get_dataloaders(
        data_config, use_cuda
    )

    # Build the model (Phase 1: frozen backbone)
    logging.info("= Model (DINOv2)")
    model_config = config["model"].copy()
    model_config["phase"] = 1  # Start with frozen backbone
    model = models.build_model(model_config, input_size, num_classes)
    model.to(device)

    # Build the loss
    logging.info("= Loss")
    loss = optim.get_loss(config["loss"], device=device)

    # Logging setup
    logging_config = config["logging"]
    logname = model_config["class"]
    logdir = utils.generate_unique_logpath(logging_config["logdir"], logname)
    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    logging.info(f"Will be logging into {logdir}")
    logdir = pathlib.Path(logdir)
    with open(logdir / "config.yaml", "w") as file:
        yaml.dump(config, file)

    model_checkpoint = utils.ModelCheckpoint(
        model, str(logdir / "best_model.pt"), min_is_best=False
    )

    # ============================================================
    # PHASE 1: Head-only training (backbone frozen)
    # ============================================================
    phase1_config = config.get("phase1", {})
    phase1_epochs = phase1_config.get("nepochs", 5)
    phase1_lr = phase1_config.get("lr", 1e-3)

    logging.info(f"=== PHASE 1: Head-only training ({phase1_epochs} epochs, lr={phase1_lr}) ===")

    head_params = [p for p in model.parameters() if p.requires_grad]
    logging.info(f"  Trainable params (head only): {sum(p.numel() for p in head_params):,}")

    optimizer_p1 = torch.optim.AdamW(head_params, lr=phase1_lr, weight_decay=0.01)
    scheduler_p1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_p1, T_max=phase1_epochs)

    for e in tqdm(range(phase1_epochs), desc="Phase 1 (Head)"):
        train_loss, _ = utils.train(model, train_loader, loss, optimizer_p1, device)
        scheduler_p1.step()

        if valid_loader is not None:
            test_loss, metrics = utils.test(model, valid_loader, loss, device)
            updated = model_checkpoint.update(metrics["f1"])
            logging.info(
                "[P1 %d/%d] Train: %.3f | Test: %.3f | F1: %.4f | F1_TTA: %.4f %s"
                % (e+1, phase1_epochs, train_loss, test_loss,
                   metrics["f1"], metrics["f1_tta"], "[BEST]" if updated else "")
            )
            if wandb_log:
                wandb_log({"phase": 1, "train_loss": train_loss, "test_loss": test_loss,
                           "f1": metrics["f1"], "f1_tta": metrics["f1_tta"],
                           "lr_head": optimizer_p1.param_groups[0]['lr']})
        else:
            logging.info(
                "[P1 %d/%d] Train: %.3f (no validation)"
                % (e+1, phase1_epochs, train_loss)
            )
            if wandb_log:
                wandb_log({"phase": 1, "train_loss": train_loss,
                           "lr_head": optimizer_p1.param_groups[0]['lr']})

    # ============================================================
    # PHASE 2: Full fine-tuning with layer-wise LR decay
    # ============================================================
    phase2_config = config.get("phase2", {})
    phase2_epochs = phase2_config.get("nepochs", 25)
    lr_backbone = phase2_config.get("lr_backbone", 1e-5)
    lr_head = phase2_config.get("lr_head", 5e-4)
    layer_decay = phase2_config.get("layer_decay", 0.75)
    weight_decay = phase2_config.get("weight_decay", 0.05)

    logging.info(f"=== PHASE 2: Full fine-tuning ({phase2_epochs} epochs) ===")
    logging.info(f"  lr_backbone={lr_backbone}, lr_head={lr_head}, layer_decay={layer_decay}")

    # Unfreeze everything
    model._unfreeze_all()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"  Trainable params (all): {total_params:,}")

    # Build differential param groups
    param_groups = model.get_param_groups(
        lr_backbone=lr_backbone,
        lr_head=lr_head,
        weight_decay=weight_decay,
        layer_decay=layer_decay
    )

    optimizer_p2 = torch.optim.AdamW(param_groups)
    scheduler_p2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_p2, T_0=phase2_epochs, T_mult=1, eta_min=1e-7
    )

    for e in tqdm(range(phase2_epochs), desc="Phase 2 (Full)"):
        train_loss, _ = utils.train(model, train_loader, loss, optimizer_p2, device)
        scheduler_p2.step()

        # Log per-group LR
        lr_info = {pg.get('name', f'g{i}'): pg['lr'] for i, pg in enumerate(optimizer_p2.param_groups)}

        if valid_loader is not None:
            test_loss, metrics = utils.test(model, valid_loader, loss, device)
            updated = model_checkpoint.update(metrics["f1"])
            logging.info(
                "[P2 %d/%d] Train: %.3f | Test: %.3f | F1: %.4f | F1_TTA: %.4f %s"
                % (e+1, phase2_epochs, train_loss, test_loss,
                   metrics["f1"], metrics["f1_tta"], "[BEST]" if updated else "")
            )
            if wandb_log:
                wandb_log({"phase": 2, "train_loss": train_loss, "test_loss": test_loss,
                           "f1": metrics["f1"], "f1_tta": metrics["f1_tta"], **lr_info})
        else:
            logging.info(
                "[P2 %d/%d] Train: %.3f (no validation)"
                % (e+1, phase2_epochs, train_loss)
            )
            if wandb_log:
                wandb_log({"phase": 2, "train_loss": train_loss, **lr_info})

    # If no validation was used, best_model.pt was never saved → save final weights
    if valid_loader is None:
        torch.save(model.state_dict(), str(logdir / "best_model.pt"))
        logging.info(f"Training complete (no val). Final model saved to {logdir / 'best_model.pt'}")
    else:
        logging.info(f"Training complete. Best model saved to {logdir / 'best_model.pt'}")


def train_dinov3(config):
    """
    Two-phase DINOv3 fine-tuning — identical logic to train_dinov2.
    The model class (DINOv3Model vs DINOv2Model) is determined by config['model']['class'].
    """
    return train_dinov2(config)


def train_dinov3_phase3(config):
    """
    Phase 3 DINOv3 fine-tuning: identical pipeline to train_dinov2/train_dinov3
    (head warmup + full fine-tune from pretrained backbone), but:
      - Uses 100% of the data (valid_ratio=0, valid_loader=None)
      - Saves a checkpoint every `checkpoint_interval` epochs during phase 2
      - Always saves a final_model.pt at the end
    Launch: python -m torchtmpl.main config_phase3_dinov3.yaml train_dinov3_phase3
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    if "wandb" in config["logging"]:
        wandb_config = config["logging"]["wandb"]
        wandb.init(project=wandb_config["project"], entity=wandb_config["entity"])
        wandb_log = wandb.log
        wandb_log(config)
    else:
        wandb_log = None

    checkpoint_interval = config.get("checkpoint_interval", 10)

    # Build the dataloaders (valid_loader will be None when valid_ratio=0)
    logging.info("= Building the dataloaders")
    data_config = config["data"]
    train_loader, valid_loader, input_size, num_classes = data.get_dataloaders(
        data_config, use_cuda
    )

    # Build the model (Phase 1: frozen backbone)
    logging.info("= Model (DINOv3 – Phase 3)")
    model_config = config["model"].copy()
    model_config["phase"] = 1
    model = models.build_model(model_config, input_size, num_classes)
    model.to(device)

    # Build the loss
    logging.info("= Loss")
    loss = optim.get_loss(config["loss"], device=device)

    # Logging setup
    logging_config = config["logging"]
    logname = model_config["class"]
    logdir = utils.generate_unique_logpath(logging_config["logdir"], logname)
    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    logging.info(f"Will be logging into {logdir}")
    logdir = pathlib.Path(logdir)
    with open(logdir / "config.yaml", "w") as file:
        yaml.dump(config, file)

    model_checkpoint = utils.ModelCheckpoint(
        model, str(logdir / "best_model.pt"), min_is_best=False
    )

    # ============================================================
    # PHASE 1: Head-only training (backbone frozen)
    # ============================================================
    phase1_config = config.get("phase1", {})
    phase1_epochs = phase1_config.get("nepochs", 5)
    phase1_lr = phase1_config.get("lr", 1e-3)

    logging.info(f"=== PHASE 1: Head-only training ({phase1_epochs} epochs, lr={phase1_lr}) ===")

    head_params = [p for p in model.parameters() if p.requires_grad]
    logging.info(f"  Trainable params (head only): {sum(p.numel() for p in head_params):,}")

    optimizer_p1 = torch.optim.AdamW(head_params, lr=phase1_lr, weight_decay=0.01)
    scheduler_p1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_p1, T_max=phase1_epochs)

    for e in tqdm(range(phase1_epochs), desc="Phase 1 (Head)"):
        train_loss, _ = utils.train(model, train_loader, loss, optimizer_p1, device)
        scheduler_p1.step()
        if valid_loader is not None:
            test_loss, metrics = utils.test(model, valid_loader, loss, device)
            updated = model_checkpoint.update(metrics["f1"])
            logging.info(
                "[P1 %d/%d] Train: %.3f | Test: %.3f | F1: %.4f | F1_TTA: %.4f %s"
                % (e+1, phase1_epochs, train_loss, test_loss,
                   metrics["f1"], metrics["f1_tta"], "[BEST]" if updated else "")
            )
            if wandb_log:
                wandb_log({"phase": 1, "train_loss": train_loss, "test_loss": test_loss,
                           "f1": metrics["f1"], "f1_tta": metrics["f1_tta"],
                           "lr_head": optimizer_p1.param_groups[0]['lr']})
        else:
            logging.info(
                "[P1 %d/%d] Train: %.3f"
                % (e+1, phase1_epochs, train_loss)
            )
            if wandb_log:
                wandb_log({"phase": 1, "train_loss": train_loss,
                           "lr_head": optimizer_p1.param_groups[0]['lr']})

    # ============================================================
    # PHASE 2: Full fine-tuning with layer-wise LR decay
    # ============================================================
    phase2_config = config.get("phase2", {})
    phase2_epochs = phase2_config.get("nepochs", 25)
    lr_backbone = phase2_config.get("lr_backbone", 1e-5)
    lr_head = phase2_config.get("lr_head", 5e-4)
    layer_decay = phase2_config.get("layer_decay", 0.75)
    weight_decay = phase2_config.get("weight_decay", 0.05)

    logging.info(f"=== PHASE 2: Full fine-tuning ({phase2_epochs} epochs) ===")
    logging.info(f"  lr_backbone={lr_backbone}, lr_head={lr_head}, layer_decay={layer_decay}")

    model._unfreeze_all()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"  Trainable params (all): {total_params:,}")

    param_groups = model.get_param_groups(
        lr_backbone=lr_backbone,
        lr_head=lr_head,
        weight_decay=weight_decay,
        layer_decay=layer_decay
    )

    optimizer_p2 = torch.optim.AdamW(param_groups)
    scheduler_p2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_p2, T_0=phase2_epochs, T_mult=1, eta_min=1e-7
    )

    for e in tqdm(range(phase2_epochs), desc="Phase 2 (Full)"):
        train_loss, _ = utils.train(model, train_loader, loss, optimizer_p2, device)
        scheduler_p2.step()
        lr_info = {pg.get('name', f'g{i}'): pg['lr'] for i, pg in enumerate(optimizer_p2.param_groups)}

        if valid_loader is not None:
            test_loss, metrics = utils.test(model, valid_loader, loss, device)
            updated = model_checkpoint.update(metrics["f1"])
            logging.info(
                "[P2 %d/%d] Train: %.3f | Test: %.3f | F1: %.4f | F1_TTA: %.4f %s"
                % (e+1, phase2_epochs, train_loss, test_loss,
                   metrics["f1"], metrics["f1_tta"], "[BEST]" if updated else "")
            )
            if wandb_log:
                wandb_log({"phase": 2, "train_loss": train_loss, "test_loss": test_loss,
                           "f1": metrics["f1"], "f1_tta": metrics["f1_tta"], **lr_info})
        else:
            logging.info(
                "[P2 %d/%d] Train: %.3f"
                % (e+1, phase2_epochs, train_loss)
            )
            if wandb_log:
                wandb_log({"phase": 2, "train_loss": train_loss, **lr_info})

        # Periodic checkpoint (phase 3 only)
        if (e + 1) % checkpoint_interval == 0:
            ckpt_path = logdir / f"epoch_{e+1}.pt"
            torch.save(model.state_dict(), ckpt_path)
            logging.info(f"  [Checkpoint] Saved {ckpt_path}")

    # Always save the final model at the end
    final_path = logdir / "final_model.pt"
    torch.save(model.state_dict(), final_path)
    logging.info(f"Training complete. Final model saved to {final_path}")


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
#         transforms.Normalize(mean=[0.879857, 0.879857, 0.879857], std=[0.190819, 0.190819, 0.190819])
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

def generate_csv_tta8(config):
    """
    Generate submission CSV with 8-transform TTA.

    Transforms:
      1. Original
      2. Flip Horizontal
      3. Flip Vertical
      4. Rotation 90°
      5. Rotation 180°
      6. Rotation 270°
      7. Flip H + Flip V (= Rot180 + identity, but via flip path)
      8. Rotation 90° + Flip H

    Usage:
        python -m torchtmpl.main config.yaml generate_csv_tta8
    """
    import time as _time

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    logging.info(f"[TTA-8] Inference on device: {device}")

    data_config = config["data"]
    try:
        _, _, input_size, num_classes = data.get_dataloaders(data_config, use_cuda=False)
    except Exception:
        input_size = (3, 224, 224)
        num_classes = config.get("model", {}).get("num_classes", 86)

    if isinstance(input_size, int):
        img_h, img_w = input_size, input_size
    elif isinstance(input_size, (tuple, list)) and len(input_size) == 3:
        img_h, img_w = input_size[1], input_size[2]
    else:
        img_h, img_w = 224, 224

    val_transform = transforms.Compose([
        transforms.Resize((img_h, img_w)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.879857, 0.879857, 0.879857],
            std=[0.190819, 0.190819, 0.190819],
        ),
    ])

    # Model
    model_config = config["model"]
    model = models.build_model(model_config, input_size, num_classes)
    weights_path = model_config["model_path"]
    logging.info(f"[TTA-8] Loading weights from {weights_path}")
    state_dict = torch.load(weights_path, map_location=device)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Test dataset
    test_path = data_config.get("test_path")
    if not test_path:
        test_path = os.path.join(data_config["path"], "test")
    test_dataset = InferenceDataset(root_dir=test_path, transform=val_transform)
    if len(test_dataset) == 0:
        logging.error("[TTA-8] Empty test dataset, aborting.")
        return
    test_loader = DataLoader(
        test_dataset,
        batch_size=data_config.get("batch_size", 64),
        shuffle=False, num_workers=4,
    )

    # 8 TTA transforms (applied to already-normalised tensors)
    tta_transforms = [
        ("original",    lambda x: x),
        ("flip_h",      lambda x: torch.flip(x, dims=[3])),
        ("flip_v",      lambda x: torch.flip(x, dims=[2])),
        ("rot90",       lambda x: torch.rot90(x, k=1, dims=[2, 3])),
        ("rot180",      lambda x: torch.rot90(x, k=2, dims=[2, 3])),
        ("rot270",      lambda x: torch.rot90(x, k=3, dims=[2, 3])),
        ("flip_hv",     lambda x: torch.flip(x, dims=[2, 3])),
        ("rot90_fliph", lambda x: torch.flip(torch.rot90(x, k=1, dims=[2, 3]), dims=[3])),
    ]
    n_tta = len(tta_transforms)
    logging.info(f"[TTA-8] {n_tta} transforms, {len(test_dataset)} images")

    test_filenames = []
    test_predictions = []

    with torch.no_grad():
        for images, filenames in tqdm(test_loader, desc="TTA-8 Inference"):
            images = images.to(device)
            avg_probs = None

            for _, tfn in tta_transforms:
                out = model(tfn(images))
                probs = F.softmax(out, dim=1)
                if avg_probs is None:
                    avg_probs = probs
                else:
                    avg_probs = avg_probs + probs

            avg_probs = avg_probs / n_tta
            _, preds = torch.max(avg_probs, 1)
            test_filenames.extend(filenames)
            test_predictions.extend(preds.cpu().tolist())

    # Save
    results_path = "results"
    os.makedirs(results_path, exist_ok=True)
    timestamp = _time.strftime("%Y%m%d-%H%M%S")

    model_tag = os.path.basename(os.path.dirname(weights_path))
    output_file = os.path.join(
        results_path, f"submission_tta8_{model_tag}_{timestamp}.csv"
    )
    results_df = pd.DataFrame({
        "imgname": test_filenames,
        "label": test_predictions,
    })
    results_df.to_csv(output_file, index=False)
    logging.info(f"[TTA-8] Saved: {output_file}")


def cache_tta_logits(config):
    """
    Run TTA inference and save raw logits (before softmax) to disk.

    For each test image, stores the per-TTA-view logits so that ensembling
    strategies can be tried offline without re-running inference.

    Saved files (in ``logits_cache_dir``):
      <model_tag>_logits.npy      — (N_test, n_tta, num_classes) float32
      <model_tag>_filenames.npy   — (N_test,) str   test image names
      <model_tag>_meta.yaml       — model tag, backbone, n_tta, num_classes

    TTA views (8):
      Original, FlipH, FlipV, Rot90, Rot180, Rot270, FlipH+V, Rot90+FlipH

    Config keys:
      model.model_path   — path to best_model.pt  (required)
      logits_cache_dir   — output dir (default: results/logits_cache)
      model_tag          — override auto-detected tag from model_path

    Usage:
        python -m torchtmpl.main config.yaml cache_tta_logits
    """
    import numpy as np

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    logging.info(f"[cache_tta_logits] device: {device}")

    # ── Model ──
    data_config = config["data"]
    try:
        _, _, input_size, num_classes = data.get_dataloaders(data_config, use_cuda=False)
    except Exception:
        input_size = (3, 224, 224)
        num_classes = config.get("model", {}).get("num_classes", 86)

    model_config = config["model"]
    model = models.build_model(model_config, input_size, num_classes)
    weights_path = model_config["model_path"]
    logging.info(f"[cache_tta_logits] Loading {weights_path}")
    state_dict = torch.load(weights_path, map_location=device)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # ── Test transform (respects AR-preserving configs) ──
    test_transform = _build_test_transform(config)

    test_path = data_config.get("test_path")
    if not test_path:
        test_path = os.path.join(data_config["path"], "test")
    test_dataset = InferenceDataset(root_dir=test_path, transform=test_transform)
    if len(test_dataset) == 0:
        logging.error("[cache_tta_logits] Empty test dataset, aborting.")
        return
    test_loader = DataLoader(
        test_dataset,
        batch_size=data_config.get("batch_size", 64),
        shuffle=False, num_workers=4,
    )

    # ── TTA views ──
    tta_transforms = [
        ("original",    lambda x: x),
        ("flip_h",      lambda x: torch.flip(x, dims=[3])),
        ("flip_v",      lambda x: torch.flip(x, dims=[2])),
        ("rot90",       lambda x: torch.rot90(x, k=1, dims=[2, 3])),
        ("rot180",      lambda x: torch.rot90(x, k=2, dims=[2, 3])),
        ("rot270",      lambda x: torch.rot90(x, k=3, dims=[2, 3])),
        ("flip_hv",     lambda x: torch.flip(x, dims=[2, 3])),
        ("rot90_fliph", lambda x: torch.flip(torch.rot90(x, k=1, dims=[2, 3]), dims=[3])),
    ]
    n_tta = len(tta_transforms)

    logging.info(f"[cache_tta_logits] {n_tta} TTA views, {len(test_dataset)} test images")

    # ── Inference ──
    all_logits = []   # list of (B, n_tta, C) arrays
    all_fnames = []

    with torch.no_grad():
        for images, filenames in tqdm(test_loader, desc="cache_tta_logits"):
            images = images.to(device)
            batch_logits = []
            for _, tfn in tta_transforms:
                logits = model(tfn(images))           # (B, C)
                batch_logits.append(logits.cpu())
            # Stack → (B, n_tta, C)
            batch_logits = torch.stack(batch_logits, dim=1).numpy()
            all_logits.append(batch_logits)
            all_fnames.extend(filenames)

    all_logits = np.concatenate(all_logits, axis=0).astype(np.float32)  # (N, n_tta, C)
    logging.info(f"[cache_tta_logits] Logits shape: {all_logits.shape}")

    # ── Save ──
    cache_dir = config.get("logits_cache_dir", os.path.join("results", "logits_cache"))
    os.makedirs(cache_dir, exist_ok=True)

    model_tag = config.get("model_tag",
                           os.path.basename(os.path.dirname(weights_path)))
    logits_path = os.path.join(cache_dir, f"{model_tag}_logits.npy")
    fnames_path = os.path.join(cache_dir, f"{model_tag}_filenames.npy")
    meta_path = os.path.join(cache_dir, f"{model_tag}_meta.yaml")

    np.save(logits_path, all_logits)
    np.save(fnames_path, np.array(all_fnames))
    with open(meta_path, "w") as f:
        yaml.dump({
            "model_tag": model_tag,
            "backbone": model_config.get("backbone", "unknown"),
            "model_class": model_config.get("class", "unknown"),
            "weights_path": weights_path,
            "n_tta": n_tta,
            "tta_names": [name for name, _ in tta_transforms],
            "num_classes": int(all_logits.shape[2]),
            "n_test": int(all_logits.shape[0]),
        }, f)

    logging.info(f"[cache_tta_logits] Saved:")
    logging.info(f"  logits : {logits_path}  ({all_logits.nbytes / 1e6:.1f} MB)")
    logging.info(f"  fnames : {fnames_path}")
    logging.info(f"  meta   : {meta_path}")


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
        transforms.Normalize(mean=[0.879857, 0.879857, 0.879857], std=[0.190819, 0.190819, 0.190819])
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


def ensemble_csv(configs):
    """
    Ensembling de plusieurs modèles avec TTA sur l'ensemble de test.
    Supporte à la fois les modèles standard (RGB) et multicanal (Original+CLAHE+Mask).
    Chaque modèle peut avoir sa propre résolution d'entrée et son propre backbone.
    Les probabilités softmax (5 vues TTA × N modèles) sont moyennées avant argmax.

    Args:
        configs: liste de dicts de configuration (un par modèle)
    """
    import time
    from PIL import Image

    num_models = len(configs)
    print("=" * 60)
    print(f"🚀 ENSEMBLE INFERENCE AVEC TTA ({num_models} modèles)")
    print("   Stratégie : Moyenne des softmax sur 5 vues TTA × N modèles")
    print("   (Original + Flip H + Flip V + Rot90 + Rot270)")
    print("=" * 60)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    logging.info(f"Inference on device: {device}")

    # --- 1. Récupérer num_classes via le premier config standard (non multicanal) ---
    # On cherche un config standard pour obtenir num_classes via get_dataloaders
    std_cfg = next((c for c in configs if c["data"].get("training_type", "multisource") != "multicanal"), None)
    if std_cfg is not None:
        data_config_ref = std_cfg["data"]
        try:
            _, _, _, num_classes = data.get_dataloaders(data_config_ref, use_cuda=False)
        except Exception:
            logging.warning("Impossible de charger les métadonnées via get_dataloaders. Fallback num_classes=86.")
            num_classes = 86
    else:
        # Tous les configs sont multicanal — on utilise get_dataloaders du premier
        try:
            _, _, _, num_classes = data.get_dataloaders(configs[0]["data"], use_cuda=False)
        except Exception:
            num_classes = 86

    # --- 2. Construire les modèles, transforms et datasets ---
    model_entries = []  # list of (model, is_multicanal, resize_index)

    # Déterminer le test_path standard (partagé par tous les modèles)
    test_path = None
    for cfg in configs:
        d_cfg = cfg["data"]
        if d_cfg.get("training_type", "multisource") != "multicanal":
            test_path = d_cfg.get("test_path")
            if test_path:
                break
    # Fallback : dériver depuis un chemin multicanal
    if not test_path:
        for cfg in configs:
            d_cfg = cfg["data"]
            if "original_dataset" in d_cfg:
                # train path → test path
                orig_train = d_cfg["original_dataset"]
                test_path = orig_train.replace("/train", "/test/imgs")
                break
    if not test_path or not os.path.exists(test_path):
        raise FileNotFoundError(f"Test directory not found: {test_path}")

    # Chemins multicanal pour le test (dérivés depuis les chemins train)
    clahe_test_path = None
    masks_test_path = None
    for cfg in configs:
        d_cfg = cfg["data"]
        if d_cfg.get("training_type", "multisource") == "multicanal":
            clahe_test_path = d_cfg["clahe_dataset"].replace("/train", "/test/imgs")
            masks_test_path = d_cfg["masks_dataset"].replace("/train", "/test/imgs")
            break

    batch_size = min(cfg["data"]["batch_size"] for cfg in configs)
    logging.info(f"Batch size commun : {batch_size}")

    loaders = []

    for i, cfg in enumerate(configs):
        d_cfg = cfg["data"]
        m_cfg = cfg["model"]
        is_multicanal = d_cfg.get("training_type", "multisource") == "multicanal"

        resize = d_cfg.get("resize", 224)
        img_h, img_w = resize, resize

        val_transform = transforms.Compose([
            transforms.Resize((img_h, img_w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.879857, 0.879857, 0.879857], std=[0.190819, 0.190819, 0.190819])
        ])

        input_size = (3, img_h, img_w)
        model = models.build_model(m_cfg, input_size, num_classes)

        weights_path = m_cfg["model_path"]
        logging.info(f"[Model {i}] Loading {m_cfg.get('backbone', '?')} from {weights_path}")

        state_dict = torch.load(weights_path, map_location=device)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        mode_str = "multicanal (Orig+CLAHE+Mask)" if is_multicanal else "standard (RGB)"
        model_entries.append((model, is_multicanal, i))
        logging.info(f"[Model {i}] {m_cfg.get('backbone', '?')} chargé (resize={resize}, mode={mode_str})")

        # Créer le DataLoader adapté à ce modèle
        if is_multicanal:
            ds = data.PlanktonMultiChannelDataset(
                root_original=test_path,
                root_clahe=clahe_test_path,
                root_masks=masks_test_path,
                transform=val_transform,
                is_test=True,
                mask_ext='.png'
            )
        else:
            ds = InferenceDataset(root_dir=test_path, transform=val_transform)

        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
        loaders.append(loader)

    num_test = len(loaders[0].dataset)
    logging.info(f"Nombre d'images de test : {num_test}")

    # Vérifier que tous les datasets ont la même taille
    for i, loader in enumerate(loaders):
        n = len(loader.dataset)
        if n != num_test:
            raise ValueError(f"Dataset du modèle {i} a {n} images, attendu {num_test}")

    # --- 3. Inférence TTA ensemblée ---
    all_filenames = []
    all_avg_probs = []  # liste de tenseurs [batch, num_classes]

    num_tta_views = 5
    total_weight = num_models * num_tta_views

    logging.info(f"Démarrage de l'inférence ensemblée ({num_models} modèles × {num_tta_views} vues TTA = {total_weight} prédictions par image)...")

    with torch.no_grad():
        iterators = [iter(loader) for loader in loaders]

        num_batches = len(loaders[0])
        for batch_idx in tqdm(range(num_batches), desc="Ensemble TTA"):
            batch_data = [next(it) for it in iterators]

            # Les filenames sont les mêmes pour tous les loaders (même dataset, même ordre)
            filenames = batch_data[0][1]
            if batch_idx == 0 or len(all_filenames) < num_test:
                all_filenames.extend(filenames)

            # Accumulateur pour ce batch
            batch_probs_sum = None

            for model_idx, (model, _, _) in enumerate(model_entries):
                images = batch_data[model_idx][0].to(device)  # [B, 3, H, W]

                # --- 5 vues TTA ---
                views = [
                    images,                                     # Original
                    torch.flip(images, dims=[3]),                # Flip Horizontal
                    torch.flip(images, dims=[2]),                # Flip Vertical
                    torch.rot90(images, k=1, dims=[2, 3]),      # Rot 90°
                    torch.rot90(images, k=3, dims=[2, 3]),      # Rot 270°
                ]

                for view in views:
                    out = model(view)
                    prob = F.softmax(out, dim=1)  # [B, num_classes]
                    if batch_probs_sum is None:
                        batch_probs_sum = prob
                    else:
                        batch_probs_sum = batch_probs_sum + prob

            # Moyenne sur (num_models × num_tta_views)
            batch_avg_probs = batch_probs_sum / total_weight
            all_avg_probs.append(batch_avg_probs.cpu())

    # --- 4. Argmax global ---
    all_avg_probs = torch.cat(all_avg_probs, dim=0)  # [N, num_classes]
    _, all_preds = torch.max(all_avg_probs, dim=1)
    all_preds = all_preds.tolist()

    # --- 5. Sauvegarde CSV ---
    results_path = "results"
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    results_df = pd.DataFrame({
        'imgname': all_filenames,
        'label': all_preds
    })

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_file = os.path.join(results_path, f"submission_ensemble_tta_{timestamp}.csv")
    results_df.to_csv(output_file, index=False)

    logging.info(f"✅ Submission ensemblée sauvegardée : {output_file}")
    logging.info(f"   {num_test} images, {num_models} modèles, {num_tta_views} vues TTA")
    logging.info(f"   Distribution des labels :\n{results_df['label'].value_counts().head(10)}")
    logging.info(f"   (Head of CSV):\n{results_df.head()}")


def ensemble_csv_meta_learner(configs):
    """
    Stacking / meta-learner ensembling :
      1. Charge chaque modèle et extrait les probas (softmax + TTA) sur val et test.
      2. Extrait les features morphologiques en temps réel depuis les masques de segmentation.
      3. Entraîne un XGBoost sur le val set (probas concaténées + morph features).
      4. Prédit sur le test set → CSV.

    Usage:
        python -m torchtmpl.main cfg1.yaml cfg2.yaml ... cfgN.yaml ensemble_csv_meta_learner

    Masques de segmentation :
        Les masques doivent exister dans un dossier miroir du dataset original :
          - Train : .../original_dataset_optimized_512px_masks/train/<class>/<img>.png
          - Test  : .../original_dataset_optimized_512px_masks/test/imgs/<img>.png
    """
    import time
    import numpy as np
    from PIL import Image
    from sklearn.metrics import f1_score
    from . import morph_features as mf

    # Lazy import — XGBoost est optionnel pour le reste du code
    try:
        from xgboost import XGBClassifier
    except ImportError:
        logging.error("❌ xgboost n'est pas installé. Installe-le avec : pip install xgboost")
        return

    num_models = len(configs)
    print("=" * 60)
    print(f"🚀 META-LEARNER ENSEMBLE ({num_models} modèles)")
    print("   Étape 1 : Extraction des probas (TTA) sur val + test")
    print("   Étape 2 : Entraînement XGBoost meta-learner sur val")
    print("   Étape 3 : Prédiction sur test → CSV")
    print("=" * 60)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logging.info(f"Device : {device}")

    # ── 1. Récupérer num_classes et le val split ──────────────────────
    # On utilise le premier config standard pour déterminer num_classes
    # et le split train/val (random_state=42 → déterministe)
    std_cfg = next((c for c in configs if c["data"].get("training_type", "multisource") != "multicanal"), configs[0])
    try:
        _, val_loader_ref, input_size_ref, num_classes = data.get_dataloaders(std_cfg["data"], use_cuda=use_cuda)
    except Exception:
        logging.warning("Fallback num_classes=86")
        num_classes = 86
        val_loader_ref = None

    # Récupérer le mapping idx → class_name depuis le val dataset
    idx_to_class = None
    if val_loader_ref is not None:
        ds = val_loader_ref.dataset
        if hasattr(ds, 'dataset'):
            ds = ds.dataset  # unwrap Subset
        if hasattr(ds, 'classes'):
            idx_to_class = {i: c for i, c in enumerate(ds.classes)}
        elif hasattr(ds, 'class_to_idx'):
            idx_to_class = {v: k for k, v in ds.class_to_idx.items()}
    if idx_to_class is None:
        idx_to_class = {i: str(i) for i in range(num_classes)}

    # test_path commun
    test_path = None
    for cfg in configs:
        d_cfg = cfg["data"]
        if d_cfg.get("training_type", "multisource") != "multicanal":
            test_path = d_cfg.get("test_path")
            if test_path:
                break
    if not test_path or not os.path.exists(test_path):
        raise FileNotFoundError(f"Test directory not found: {test_path}")

    # Multicanal paths (si nécessaire)
    clahe_test_path = None
    masks_test_path = None
    for cfg in configs:
        d_cfg = cfg["data"]
        if d_cfg.get("training_type", "multisource") == "multicanal":
            clahe_test_path = d_cfg["clahe_dataset"].replace("/train", "/test/imgs")
            masks_test_path = d_cfg["masks_dataset"].replace("/train", "/test/imgs")
            break

    batch_size = min(cfg["data"]["batch_size"] for cfg in configs)

    # ── 2. Pour chaque modèle : extraire probas val + test ────────────
    all_val_probas = []
    all_test_probas = []
    val_labels = None
    test_filenames = None

    for model_idx, cfg in enumerate(configs):
        d_cfg = cfg["data"]
        m_cfg = cfg["model"]
        is_multicanal = d_cfg.get("training_type", "multisource") == "multicanal"
        resize = d_cfg.get("resize", 224)

        logging.info(f"\n{'='*60}")
        logging.info(f"[Model {model_idx}] {m_cfg.get('backbone', m_cfg['class'])} (resize={resize})")
        logging.info(f"{'='*60}")

        # ── Build model ──
        input_size = (3, resize, resize)
        model = models.build_model(m_cfg, input_size, num_classes)

        weights_path = m_cfg.get("model_path")
        if not weights_path:
            raise ValueError(f"[Model {model_idx}] Pas de model_path dans la config !")

        logging.info(f"[Model {model_idx}] Loading from {weights_path}")
        state_dict = torch.load(weights_path, map_location=device)
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        # ── Build val loader for this model ──
        val_transform = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.879857, 0.879857, 0.879857], std=[0.190819, 0.190819, 0.190819])
        ])

        # Re-create dataloaders to get val split with THIS model's resize
        try:
            _, val_loader_model, _, _ = data.get_dataloaders(cfg["data"], use_cuda=use_cuda)
        except Exception as e:
            logging.error(f"[Model {model_idx}] Cannot build val loader: {e}")
            raise

        # ── Extract val probas ──
        model_val_probas = []
        model_val_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader_model, desc=f"[Model {model_idx}] Val"):
                images = batch[0].to(device)
                labels_batch = batch[1]

                # TTA : 5 vues
                views = [
                    images,
                    torch.flip(images, dims=[3]),
                    torch.flip(images, dims=[2]),
                    torch.rot90(images, k=1, dims=[2, 3]),
                    torch.rot90(images, k=3, dims=[2, 3]),
                ]
                avg_proba = torch.zeros(images.size(0), num_classes, device=device)
                for view in views:
                    avg_proba += F.softmax(model(view), dim=1)
                avg_proba /= len(views)

                model_val_probas.append(avg_proba.cpu().numpy())
                model_val_labels.append(labels_batch.numpy())

        model_val_probas = np.concatenate(model_val_probas, axis=0)
        model_val_labels = np.concatenate(model_val_labels, axis=0)

        if val_labels is None:
            val_labels = model_val_labels
        all_val_probas.append(model_val_probas)
        logging.info(f"[Model {model_idx}] Val probas: {model_val_probas.shape}")

        # ── Build test loader for this model ──
        if is_multicanal:
            ds_test = data.PlanktonMultiChannelDataset(
                root_original=test_path,
                root_clahe=clahe_test_path,
                root_masks=masks_test_path,
                transform=val_transform,
                is_test=True,
                mask_ext='.png'
            )
        else:
            ds_test = InferenceDataset(root_dir=test_path, transform=val_transform)

        test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=4)

        # ── Extract test probas ──
        model_test_probas = []
        model_test_fnames = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"[Model {model_idx}] Test"):
                images = batch[0].to(device)
                fnames = batch[1]

                views = [
                    images,
                    torch.flip(images, dims=[3]),
                    torch.flip(images, dims=[2]),
                    torch.rot90(images, k=1, dims=[2, 3]),
                    torch.rot90(images, k=3, dims=[2, 3]),
                ]
                avg_proba = torch.zeros(images.size(0), num_classes, device=device)
                for view in views:
                    avg_proba += F.softmax(model(view), dim=1)
                avg_proba /= len(views)

                model_test_probas.append(avg_proba.cpu().numpy())
                if isinstance(fnames, (list, tuple)):
                    model_test_fnames.extend(fnames)
                else:
                    model_test_fnames.extend(list(fnames))

        model_test_probas = np.concatenate(model_test_probas, axis=0)
        all_test_probas.append(model_test_probas)

        if test_filenames is None:
            test_filenames = model_test_fnames
        logging.info(f"[Model {model_idx}] Test probas: {model_test_probas.shape}")

        # Free GPU
        del model
        torch.cuda.empty_cache()

    # ── 3. Build feature matrices ─────────────────────────────────────
    X_val = np.concatenate(all_val_probas, axis=1)    # (N_val, 86 * N_models)
    X_test = np.concatenate(all_test_probas, axis=1)  # (N_test, 86 * N_models)
    logging.info(f"\nProbas concaténées : val={X_val.shape}, test={X_test.shape}")

    # ── 3b. Extraction des features morphologiques en temps réel ──────
    # Dérive le chemin des masques depuis le chemin du dataset original
    # original_dataset_optimized_512px → original_dataset_optimized_512px_masks
    original_train_path = None
    for cfg in configs:
        d_cfg = cfg["data"]
        if d_cfg.get("training_type", "multisource") != "multicanal":
            sources = d_cfg.get("sources", {})
            if "original_dataset" in sources:
                original_train_path = sources["original_dataset"]["path"]
                break

    masks_base = None
    if original_train_path:
        # Derive masks path: replace the dataset folder name
        # e.g. .../original_dataset_optimized_512px/train → .../original_dataset_optimized_512px_masks/train
        parts = original_train_path.rstrip("/").split("/")
        for i in range(len(parts) - 1, -1, -1):
            if "original_dataset_optimized" in parts[i] and "mask" not in parts[i]:
                parts[i] = parts[i] + "_masks"
                break
        masks_base = "/".join(parts)

    if masks_base and os.path.isdir(masks_base):
        logging.info(f"\n{'='*60}")
        logging.info(f"Extraction des features morphologiques...")
        logging.info(f"  Masks train dir : {masks_base}")
        logging.info(f"{'='*60}")

        # ── Val morph features ──
        # Récupérer les chemins des images du val set
        # Le val set est un Subset du PlanktonMultisourceDataset
        # On reconstruit le même split (random_state=42) pour obtenir les indices
        _, val_loader_for_paths, _, _ = data.get_dataloaders(std_cfg["data"], use_cuda=False)
        val_subset = val_loader_for_paths.dataset  # Subset
        val_ds = val_subset.dataset  # PlanktonMultisourceDataset
        val_indices = val_subset.indices

        val_img_paths = []
        val_mask_paths = []
        for idx in val_indices:
            img_path = val_ds.samples[idx][0]  # original image path
            val_img_paths.append(img_path)
            # Derive mask path: same relative structure, different root
            # img_path: .../original_dataset_optimized_512px/train/<class>/<file>.jpg
            # mask_path: .../original_dataset_optimized_512px_masks/train/<class>/<file>.png
            rel_parts = img_path.split("/train/", 1)
            if len(rel_parts) == 2:
                mask_path = os.path.join(masks_base, rel_parts[1])
                # Change extension to .png
                mask_path = os.path.splitext(mask_path)[0] + ".png"
            else:
                mask_path = ""
            val_mask_paths.append(mask_path)

        logging.info(f"  Extracting morph features for {len(val_img_paths)} val images...")
        val_morph_rows = []
        all_morph_keys = None
        for img_p, mask_p in tqdm(zip(val_img_paths, val_mask_paths), total=len(val_img_paths), desc="Val morph"):
            if os.path.exists(mask_p) and os.path.exists(img_p):
                row = mf.extract_features(mask_p, img_p)
            else:
                row = None
            val_morph_rows.append(row)
            if row is not None and all_morph_keys is None:
                all_morph_keys = set(row.keys())
            elif row is not None:
                all_morph_keys.update(row.keys())

        # ── Test morph features ──
        masks_test_dir = os.path.join(os.path.dirname(masks_base), "test", "imgs")
        if not os.path.isdir(masks_test_dir):
            # Try alternative: masks_base/../test/imgs
            masks_test_dir_alt = masks_base.replace("/train", "/test/imgs")
            if os.path.isdir(masks_test_dir_alt):
                masks_test_dir = masks_test_dir_alt

        logging.info(f"  Masks test dir  : {masks_test_dir}")
        logging.info(f"  Extracting morph features for {len(test_filenames)} test images...")

        test_morph_rows = []
        for fname in tqdm(test_filenames, desc="Test morph"):
            stem = os.path.splitext(fname)[0]
            mask_p = os.path.join(masks_test_dir, stem + ".png")
            img_p = os.path.join(test_path, fname)
            if os.path.exists(mask_p) and os.path.exists(img_p):
                row = mf.extract_features(mask_p, img_p)
            else:
                row = None
            test_morph_rows.append(row)
            if row is not None and all_morph_keys is None:
                all_morph_keys = set(row.keys())
            elif row is not None:
                all_morph_keys.update(row.keys())

        # ── Build morph arrays ──
        if all_morph_keys is not None and len(all_morph_keys) > 0:
            morph_keys = sorted(all_morph_keys)
            n_morph = len(morph_keys)

            morph_val = np.full((len(val_morph_rows), n_morph), np.nan, dtype=np.float64)
            for i, row in enumerate(val_morph_rows):
                if row is not None:
                    for j, key in enumerate(morph_keys):
                        morph_val[i, j] = row.get(key, np.nan)

            morph_test = np.full((len(test_morph_rows), n_morph), np.nan, dtype=np.float64)
            for i, row in enumerate(test_morph_rows):
                if row is not None:
                    for j, key in enumerate(morph_keys):
                        morph_test[i, j] = row.get(key, np.nan)

            # Imputer les NaN avec la médiane de chaque colonne
            from numpy import nanmedian
            for col in range(n_morph):
                all_vals = np.concatenate([morph_val[:, col], morph_test[:, col]])
                median_val = nanmedian(all_vals)
                if np.isnan(median_val):
                    median_val = 0.0
                morph_val[np.isnan(morph_val[:, col]), col] = median_val
                morph_test[np.isnan(morph_test[:, col]), col] = median_val

            val_ok = np.sum(~np.isnan(morph_val[:, 0]) if n_morph > 0 else 0)
            test_ok = np.sum(~np.isnan(morph_test[:, 0]) if n_morph > 0 else 0)
            logging.info(f"  ✅ Morph features : {n_morph} features extraites")

            X_val = np.concatenate([X_val, morph_val], axis=1)
            X_test = np.concatenate([X_test, morph_test], axis=1)
            logging.info(f"  Features totales : val={X_val.shape}, test={X_test.shape}")
        else:
            logging.warning("  ⚠️ Aucune feature morphologique n'a pu être extraite.")
    else:
        logging.info(f"ℹ️  Masques non trouvés ({masks_base}) → meta-learner sur probas uniquement.")

    logging.info(f"Features finales : val={X_val.shape}, test={X_test.shape}")

    # ── 4. Entraîner XGBoost meta-learner ─────────────────────────────
    logging.info(f"\n{'='*60}")
    logging.info("Entraînement XGBoost meta-learner (5-fold CV)...")
    logging.info(f"{'='*60}")

    # Sample weights: inverse class frequency → F1-macro alignment
    class_counts = np.bincount(val_labels.astype(int), minlength=num_classes)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * num_classes
    sample_weights_val = class_weights[val_labels.astype(int)]
    logging.info(f"  Sample weights: min={sample_weights_val.min():.4f}, max={sample_weights_val.max():.4f}, "
                 f"ratio={sample_weights_val.max()/sample_weights_val.min():.1f}x")

    xgb_params = dict(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=5.0,
        objective="multi:softprob",
        num_class=num_classes,
        tree_method="hist",
        eval_metric="mlogloss",
        n_jobs=8,
        random_state=42,
    )

    # Cross-validation interne pour estimer le score
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_accs = []
    cv_f1s = []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X_val, val_labels)):
        xgb_fold = XGBClassifier(**xgb_params)
        xgb_fold.fit(X_val[tr_idx], val_labels[tr_idx],
                      sample_weight=sample_weights_val[tr_idx])
        preds_fold = xgb_fold.predict(X_val[te_idx])
        acc = (preds_fold == val_labels[te_idx]).mean()
        f1 = f1_score(val_labels[te_idx], preds_fold, average="macro")
        cv_accs.append(acc)
        cv_f1s.append(f1)
        logging.info(f"  Fold {fold}: acc={acc:.4f}, F1_macro={f1:.4f}")

    logging.info(f"  CV mean: acc={np.mean(cv_accs):.4f}±{np.std(cv_accs):.4f}, F1={np.mean(cv_f1s):.4f}±{np.std(cv_f1s):.4f}")

    # Comparaison avec la simple moyenne (baseline)
    avg_val_probas = np.mean(all_val_probas, axis=0)  # (N_val, 86)
    avg_preds = np.argmax(avg_val_probas, axis=1)
    avg_f1 = f1_score(val_labels, avg_preds, average="macro")
    avg_acc = (avg_preds == val_labels).mean()
    logging.info(f"\n  📊 Baseline (moyenne simple) : acc={avg_acc:.4f}, F1_macro={avg_f1:.4f}")
    logging.info(f"  📊 Meta-learner (XGBoost CV)  : acc={np.mean(cv_accs):.4f}, F1_macro={np.mean(cv_f1s):.4f}")

    # Entraîner le modèle final sur tout le val set
    xgb_final = XGBClassifier(**xgb_params)
    xgb_final.fit(X_val, val_labels, sample_weight=sample_weights_val)

    # ── 5. Prédire sur test → CSV ────────────────────────────────────
    test_preds = xgb_final.predict(X_test)

    results_path = "results"
    os.makedirs(results_path, exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_file = os.path.join(results_path, f"submission_stacking_{timestamp}.csv")

    results_df = pd.DataFrame({
        'imgname': test_filenames,
        'label': [int(p) for p in test_preds]
    })
    results_df.to_csv(output_file, index=False)

    logging.info(f"\n✅ Submission meta-learner sauvegardée : {output_file}")
    logging.info(f"   {len(test_preds)} prédictions, {num_models} modèles")
    logging.info(f"   Distribution :\n{results_df['label'].value_counts().head(10)}")
    logging.info(f"   Head :\n{results_df.head()}")

    # Sauvegarder aussi les probas brutes pour analyse future
    probas_dir = os.path.join(results_path, f"probas_{timestamp}")
    os.makedirs(probas_dir, exist_ok=True)
    np.save(os.path.join(probas_dir, "val_probas_concat.npy"), X_val)
    np.save(os.path.join(probas_dir, "val_labels.npy"), val_labels)
    np.save(os.path.join(probas_dir, "test_probas_concat.npy"), X_test)
    np.save(os.path.join(probas_dir, "test_filenames.npy"), np.array(test_filenames, dtype=object))
    logging.info(f"   Probas brutes sauvegardées dans {probas_dir}/")


def _build_test_transform(cfg):
    """
    Build the correct validation/test transform for a given config.
    Handles AR-preserving preprocessing (heavy_preserve_ar) vs standard square resize.
    """
    d_cfg = cfg["data"]
    resize = d_cfg.get("resize", 224)

    # Detect augmentation version from the original_dataset source
    aug_version = None
    sources = d_cfg.get("sources", {})
    if "original_dataset" in sources:
        aug_version = sources["original_dataset"].get("augmentations_version")

    if aug_version == "heavy_preserve_ar":
        # AR-preserving: canonical rotation + LongestMaxSize + white padding
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        album_pipeline = A.Compose([
            A.ToGray(p=1.0),
            A.Normalize(mean=(0.879857, 0.879857, 0.879857), std=(0.190819, 0.190819, 0.190819)),
            ToTensorV2()
        ])
        return data.AlbumentationsPreserveARWrapper(album_pipeline, target_size=resize, pad_value=255)
    else:
        # Standard square resize
        return transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.879857, 0.879857, 0.879857], std=[0.190819, 0.190819, 0.190819])
        ])


def ensemble_csv_stacking(configs):
    """
    Stacking meta-learner (XGBoost) on concatenated TTA softmax probabilities.
    No morphological features — pure softmax stacking for maximum simplicity.

    Pipeline:
      1. For each model: extract TTA-averaged softmax probas on val + test.
      2. Concatenate all model probas → (N, 86 × num_models) feature matrix.
      3. Train XGBoost with 5-fold stratified CV on val set.
      4. Predict on test set → CSV.
      5. Also save baseline (simple average) CSV for comparison.

    Usage:
        python -m torchtmpl.main cfg1.yaml cfg2.yaml ... cfgN.yaml ensemble_csv_stacking
    """
    import time
    import numpy as np
    from PIL import Image
    from sklearn.metrics import f1_score

    try:
        from xgboost import XGBClassifier
    except ImportError:
        logging.error("xgboost not installed. Run: pip install xgboost")
        return

    num_models = len(configs)
    print("=" * 60)
    print(f"STACKING ENSEMBLE ({num_models} models)")
    print("   Step 1: Extract TTA probas on val + test per model")
    print("   Step 2: Train XGBoost meta-learner on val (5-fold CV)")
    print("   Step 3: Predict on test -> CSV")
    print("=" * 60)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logging.info(f"Device: {device}")

    # ── 1. Determine num_classes and idx_to_class ─────────────────────
    std_cfg = next(
        (c for c in configs if c["data"].get("training_type", "multisource") != "multicanal"),
        configs[0]
    )
    try:
        _, val_loader_ref, _, num_classes = data.get_dataloaders(std_cfg["data"], use_cuda=use_cuda)
    except Exception:
        logging.warning("Fallback num_classes=86")
        num_classes = 86
        val_loader_ref = None

    idx_to_class = None
    if val_loader_ref is not None:
        ds = val_loader_ref.dataset
        if hasattr(ds, 'dataset'):
            ds = ds.dataset  # unwrap Subset
        if hasattr(ds, 'classes'):
            idx_to_class = {i: c for i, c in enumerate(ds.classes)}
        elif hasattr(ds, 'class_to_idx'):
            idx_to_class = {v: k for k, v in ds.class_to_idx.items()}
    if idx_to_class is None:
        idx_to_class = {i: str(i) for i in range(num_classes)}

    # ── test_path ─────────────────────────────────────────────────────
    test_path = None
    for cfg in configs:
        d_cfg = cfg["data"]
        if d_cfg.get("training_type", "multisource") != "multicanal":
            test_path = d_cfg.get("test_path")
            if test_path:
                break
    if not test_path or not os.path.exists(test_path):
        raise FileNotFoundError(f"Test directory not found: {test_path}")

    batch_size = min(cfg["data"]["batch_size"] for cfg in configs)

    # ── 2. Per-model proba extraction ─────────────────────────────────
    all_val_probas = []
    all_test_probas = []
    val_labels = None
    test_filenames = None

    for model_idx, cfg in enumerate(configs):
        d_cfg = cfg["data"]
        m_cfg = cfg["model"]
        resize = d_cfg.get("resize", 224)

        logging.info(f"\n{'='*60}")
        logging.info(f"[Model {model_idx}] {m_cfg.get('backbone', m_cfg.get('class', '?'))} (resize={resize})")
        logging.info(f"{'='*60}")

        # Build model
        input_size = (3, resize, resize)
        model = models.build_model(m_cfg, input_size, num_classes)

        weights_path = m_cfg.get("model_path")
        if not weights_path:
            raise ValueError(f"[Model {model_idx}] No model_path in config!")

        logging.info(f"[Model {model_idx}] Loading from {weights_path}")
        state_dict = torch.load(weights_path, map_location=device)
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        # ── Val probas (use get_dataloaders for correct transform + split) ──
        try:
            _, val_loader_model, _, _ = data.get_dataloaders(cfg["data"], use_cuda=use_cuda)
        except Exception as e:
            logging.error(f"[Model {model_idx}] Cannot build val loader: {e}")
            raise

        model_val_probas = []
        model_val_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader_model, desc=f"[Model {model_idx}] Val TTA"):
                images = batch[0].to(device)
                labels_batch = batch[1]

                views = [
                    images,
                    torch.flip(images, dims=[3]),
                    torch.flip(images, dims=[2]),
                    torch.rot90(images, k=1, dims=[2, 3]),
                    torch.rot90(images, k=3, dims=[2, 3]),
                ]
                avg_proba = torch.zeros(images.size(0), num_classes, device=device)
                for view in views:
                    avg_proba += F.softmax(model(view), dim=1)
                avg_proba /= len(views)

                model_val_probas.append(avg_proba.cpu().numpy())
                model_val_labels.append(labels_batch.numpy())

        model_val_probas = np.concatenate(model_val_probas, axis=0)
        model_val_labels = np.concatenate(model_val_labels, axis=0)

        if val_labels is None:
            val_labels = model_val_labels
        all_val_probas.append(model_val_probas)
        logging.info(f"[Model {model_idx}] Val probas shape: {model_val_probas.shape}")

        # ── Test probas (use _build_test_transform for correct AR handling) ──
        test_transform = _build_test_transform(cfg)
        ds_test = InferenceDataset(root_dir=test_path, transform=test_transform)
        test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=4)

        model_test_probas = []
        model_test_fnames = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"[Model {model_idx}] Test TTA"):
                images = batch[0].to(device)
                fnames = batch[1]

                views = [
                    images,
                    torch.flip(images, dims=[3]),
                    torch.flip(images, dims=[2]),
                    torch.rot90(images, k=1, dims=[2, 3]),
                    torch.rot90(images, k=3, dims=[2, 3]),
                ]
                avg_proba = torch.zeros(images.size(0), num_classes, device=device)
                for view in views:
                    avg_proba += F.softmax(model(view), dim=1)
                avg_proba /= len(views)

                model_test_probas.append(avg_proba.cpu().numpy())
                if isinstance(fnames, (list, tuple)):
                    model_test_fnames.extend(fnames)
                else:
                    model_test_fnames.extend(list(fnames))

        model_test_probas = np.concatenate(model_test_probas, axis=0)
        all_test_probas.append(model_test_probas)

        if test_filenames is None:
            test_filenames = model_test_fnames
        logging.info(f"[Model {model_idx}] Test probas shape: {model_test_probas.shape}")

        # Free GPU
        del model
        torch.cuda.empty_cache()

    # ── 3. Build feature matrices ─────────────────────────────────────
    X_val = np.concatenate(all_val_probas, axis=1)    # (N_val, 86 * N_models)
    X_test = np.concatenate(all_test_probas, axis=1)  # (N_test, 86 * N_models)
    logging.info(f"\nConcatenated probas: val={X_val.shape}, test={X_test.shape}")

    # ── 3b. Save probas immediately (reusable cache) ────────────────
    probas_cache_dir = os.path.join("results", "probas_cache")
    os.makedirs(probas_cache_dir, exist_ok=True)
    np.save(os.path.join(probas_cache_dir, "val_probas_concat.npy"), X_val)
    np.save(os.path.join(probas_cache_dir, "val_labels.npy"), val_labels)
    np.save(os.path.join(probas_cache_dir, "test_probas_concat.npy"), X_test)
    np.save(os.path.join(probas_cache_dir, "test_filenames.npy"), np.array(test_filenames))
    # Save per-model probas (needed for baseline averaging)
    for i, vp in enumerate(all_val_probas):
        np.save(os.path.join(probas_cache_dir, f"val_probas_model{i}.npy"), vp)
    for i, tp in enumerate(all_test_probas):
        np.save(os.path.join(probas_cache_dir, f"test_probas_model{i}.npy"), tp)
    logging.info(f"Probas cached to {probas_cache_dir}/ (reusable)")

    # ── 4. XGBoost meta-learner ───────────────────────────────────────
    logging.info(f"\n{'='*60}")
    logging.info("Training XGBoost meta-learner (5-fold CV)...")
    logging.info(f"{'='*60}")

    # Compute sample weights: inverse class frequency → F1-macro alignment
    class_counts = np.bincount(val_labels.astype(int), minlength=num_classes)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * num_classes  # normalize
    sample_weights_val = class_weights[val_labels.astype(int)]
    logging.info(f"  Sample weights: min={sample_weights_val.min():.4f}, max={sample_weights_val.max():.4f}, "
                 f"ratio={sample_weights_val.max()/sample_weights_val.min():.1f}x")

    xgb_params = dict(
        n_estimators=200,       # lighter (was 500) — less overfitting
        max_depth=4,            # shallower (was 6) — less overfitting
        learning_rate=0.08,     # slightly higher to compensate fewer trees
        min_child_weight=5,     # prevents splits on tiny leaf sets
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,          # L1 regularization
        reg_lambda=5.0,         # L2 regularization
        objective="multi:softprob",
        num_class=num_classes,
        tree_method="hist",
        eval_metric="mlogloss",
        n_jobs=8,
        random_state=42,
    )

    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_accs = []
    cv_f1s = []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X_val, val_labels)):
        xgb_fold = XGBClassifier(**xgb_params)
        xgb_fold.fit(X_val[tr_idx], val_labels[tr_idx],
                      sample_weight=sample_weights_val[tr_idx])
        preds_fold = xgb_fold.predict(X_val[te_idx])
        acc = (preds_fold == val_labels[te_idx]).mean()
        f1 = f1_score(val_labels[te_idx], preds_fold, average="macro")
        cv_accs.append(acc)
        cv_f1s.append(f1)
        logging.info(f"  Fold {fold}: acc={acc:.4f}, F1_macro={f1:.4f}")

    logging.info(f"  CV mean: acc={np.mean(cv_accs):.4f}+/-{np.std(cv_accs):.4f}, F1={np.mean(cv_f1s):.4f}+/-{np.std(cv_f1s):.4f}")

    # Baseline: simple averaging
    avg_val_probas = np.mean(all_val_probas, axis=0)  # (N_val, 86)
    avg_preds = np.argmax(avg_val_probas, axis=1)
    avg_f1 = f1_score(val_labels, avg_preds, average="macro")
    avg_acc = (avg_preds == val_labels).mean()
    logging.info(f"\n  Baseline (simple avg): acc={avg_acc:.4f}, F1_macro={avg_f1:.4f}")
    logging.info(f"  Meta-learner (XGB CV): acc={np.mean(cv_accs):.4f}, F1_macro={np.mean(cv_f1s):.4f}")

    # Train final model on full val set
    xgb_final = XGBClassifier(**xgb_params)
    xgb_final.fit(X_val, val_labels, sample_weight=sample_weights_val)

    # ── 5. Predict on test -> CSV ────────────────────────────────────
    test_preds_stacking = xgb_final.predict(X_test)

    # Baseline averaging predictions on test
    avg_test_probas = np.mean(all_test_probas, axis=0)
    test_preds_avg = np.argmax(avg_test_probas, axis=1)

    results_path = "results"
    os.makedirs(results_path, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # --- Stacking CSV ---
    stacking_df = pd.DataFrame({
        'imgname': test_filenames,
        'label': [int(p) for p in test_preds_stacking]
    })
    stacking_file = os.path.join(results_path, f"submission_stacking_{timestamp}.csv")
    stacking_df.to_csv(stacking_file, index=False)
    logging.info(f"\nStacking submission saved: {stacking_file}")
    logging.info(f"  {len(test_preds_stacking)} predictions, {num_models} models")
    logging.info(f"  Distribution:\n{stacking_df['label'].value_counts().head(10)}")

    # --- Baseline averaging CSV ---
    avg_df = pd.DataFrame({
        'imgname': test_filenames,
        'label': [int(p) for p in test_preds_avg]
    })
    avg_file = os.path.join(results_path, f"submission_avg_baseline_{timestamp}.csv")
    avg_df.to_csv(avg_file, index=False)
    logging.info(f"Baseline avg submission saved: {avg_file}")

    # --- Save raw probas for debugging ---
    probas_dir = os.path.join(results_path, f"probas_stacking_{timestamp}")
    os.makedirs(probas_dir, exist_ok=True)
    np.save(os.path.join(probas_dir, "val_probas_concat.npy"), X_val)
    np.save(os.path.join(probas_dir, "val_labels.npy"), val_labels)
    np.save(os.path.join(probas_dir, "test_probas_concat.npy"), X_test)
    logging.info(f"Raw probas saved to {probas_dir}/")


def retrain_meta_learner_from_cache(config):
    """
    Reload cached probas (val + test + morph features already concatenated)
    and retrain the XGBoost meta-learner without running GPU inference.

    Usage:
        python -m torchtmpl.main config.yaml retrain_meta_learner_from_cache

    By default loads from results/probas_cache_meta/.
    Override with config["probas_cache_dir"], e.g. results/probas_20260225-141304
    """
    import time
    import numpy as np
    from sklearn.metrics import f1_score

    try:
        from xgboost import XGBClassifier
    except ImportError:
        logging.error("xgboost not installed. Run: pip install xgboost")
        return

    # ── Locate the probas cache directory ──
    cache_dir = config.get("probas_cache_dir",
                           os.path.join("results", "probas_cache_meta"))
    if not os.path.isdir(cache_dir):
        logging.error(f"Cache not found: {cache_dir}/ — run ensemble_csv_meta_learner first ")
        return

    logging.info(f"Loading cached probas from {cache_dir}/")
    X_val = np.load(os.path.join(cache_dir, "val_probas_concat.npy"))
    val_labels = np.load(os.path.join(cache_dir, "val_labels.npy"))
    X_test = np.load(os.path.join(cache_dir, "test_probas_concat.npy"))

    # ── Recover test filenames ──
    test_filenames_path = os.path.join(cache_dir, "test_filenames.npy")
    if os.path.exists(test_filenames_path):
        test_filenames = np.load(test_filenames_path, allow_pickle=True).tolist()
    else:
        # Fallback: reconstruct from test_path (sorted, same order as InferenceDataset)
        test_path = config["data"]["test_path"]
        valid_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        test_filenames = sorted(
            f for f in os.listdir(test_path) if f.lower().endswith(valid_ext)
        )
        logging.info(f"  test_filenames.npy absent → reconstruit depuis {test_path} ({len(test_filenames)} fichiers)")

    assert len(test_filenames) == X_test.shape[0], \
        f"Mismatch: {len(test_filenames)} filenames vs {X_test.shape[0]} test rows"

    num_classes = int(val_labels.max()) + 1
    logging.info(f"Loaded: val={X_val.shape}, test={X_test.shape}, {num_classes} classes")

    # ── Sample weights ──
    class_counts = np.bincount(val_labels.astype(int), minlength=num_classes)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * num_classes
    sample_weights_val = class_weights[val_labels.astype(int)]
    logging.info(f"  Sample weights: min={sample_weights_val.min():.4f}, max={sample_weights_val.max():.4f}, "
                 f"ratio={sample_weights_val.max()/sample_weights_val.min():.1f}x")

    # ── XGBoost hyperparams (same as ensemble_csv_meta_learner) ──
    xgb_params = dict(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=5.0,
        objective="multi:softprob",
        num_class=num_classes,
        tree_method="hist",
        eval_metric="mlogloss",
        n_jobs=8,
        random_state=42,
    )

    # Override xgb params from config if provided
    if "xgb_params" in config:
        xgb_params.update(config["xgb_params"])
        logging.info(f"  XGBoost params overridden from config: {config['xgb_params']}")

    # ── 5-fold CV ──
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_accs = []
    cv_f1s = []

    logging.info("\nTraining XGBoost meta-learner (5-fold CV)...")
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X_val, val_labels)):
        xgb_fold = XGBClassifier(**xgb_params)
        xgb_fold.fit(X_val[tr_idx], val_labels[tr_idx],
                      sample_weight=sample_weights_val[tr_idx])
        preds_fold = xgb_fold.predict(X_val[te_idx])
        acc = (preds_fold == val_labels[te_idx]).mean()
        f1 = f1_score(val_labels[te_idx], preds_fold, average="macro")
        cv_accs.append(acc)
        cv_f1s.append(f1)
        logging.info(f"  Fold {fold}: acc={acc:.4f}, F1_macro={f1:.4f}")

    logging.info(f"  CV mean: acc={np.mean(cv_accs):.4f}+/-{np.std(cv_accs):.4f}, "
                 f"F1={np.mean(cv_f1s):.4f}+/-{np.std(cv_f1s):.4f}")

    # ── Final model on full val set ──
    xgb_final = XGBClassifier(**xgb_params)
    xgb_final.fit(X_val, val_labels, sample_weight=sample_weights_val)

    test_preds = xgb_final.predict(X_test)

    results_path = "results"
    os.makedirs(results_path, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    output_file = os.path.join(results_path, f"submission_meta_cached_{timestamp}.csv")
    results_df = pd.DataFrame({
        'imgname': test_filenames,
        'label': [int(p) for p in test_preds]
    })
    results_df.to_csv(output_file, index=False)

    logging.info(f"\n✅ Submission meta-learner (from cache) : {output_file}")
    logging.info(f"   {len(test_preds)} prédictions")
    logging.info(f"   CV F1_macro={np.mean(cv_f1s):.4f}")
    logging.info(f"   Distribution :\n{results_df['label'].value_counts().head(10)}")
    logging.info(f"   Head :\n{results_df.head()}")


def retrain_stacking_from_cache(config):
    """
    Reload cached probas from results/probas_cache/ and retrain XGBoost
    with potentially different hyperparameters — no GPU inference needed.

    Usage:
        python -m torchtmpl.main dummy.yaml retrain_stacking_from_cache
    (config is unused, just pass any yaml)
    """
    import time
    import numpy as np
    from sklearn.metrics import f1_score

    try:
        from xgboost import XGBClassifier
    except ImportError:
        logging.error("xgboost not installed. Run: pip install xgboost")
        return

    cache_dir = os.path.join("results", "probas_cache")
    if not os.path.isdir(cache_dir):
        logging.error(f"Cache not found: {cache_dir}/ — run ensemble_csv_stacking first.")
        return

    logging.info(f"Loading cached probas from {cache_dir}/")
    X_val = np.load(os.path.join(cache_dir, "val_probas_concat.npy"))
    val_labels = np.load(os.path.join(cache_dir, "val_labels.npy"))
    X_test = np.load(os.path.join(cache_dir, "test_probas_concat.npy"))
    test_filenames = np.load(os.path.join(cache_dir, "test_filenames.npy"), allow_pickle=True).tolist()

    num_classes = X_val.shape[1] // (len([f for f in os.listdir(cache_dir) if f.startswith("val_probas_model")]) or 1)
    num_models = len([f for f in os.listdir(cache_dir) if f.startswith("val_probas_model")])
    logging.info(f"Loaded: val={X_val.shape}, test={X_test.shape}, {num_models} models, {num_classes} classes")

    # Load per-model probas for baseline averaging
    all_val_probas = [np.load(os.path.join(cache_dir, f"val_probas_model{i}.npy")) for i in range(num_models)]
    all_test_probas = [np.load(os.path.join(cache_dir, f"test_probas_model{i}.npy")) for i in range(num_models)]

    # Sample weights: inverse class frequency
    class_counts = np.bincount(val_labels.astype(int), minlength=num_classes)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * num_classes
    sample_weights_val = class_weights[val_labels.astype(int)]
    logging.info(f"  Sample weights: min={sample_weights_val.min():.4f}, max={sample_weights_val.max():.4f}, "
                 f"ratio={sample_weights_val.max()/sample_weights_val.min():.1f}x")

    xgb_params = dict(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        objective="multi:softprob",
        num_class=num_classes,
        tree_method="hist",
        eval_metric="mlogloss",
        n_jobs=8,
        random_state=42,
    )

    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_accs = []
    cv_f1s = []

    logging.info("\nTraining XGBoost (5-fold CV) with sample_weight...")
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X_val, val_labels)):
        xgb_fold = XGBClassifier(**xgb_params)
        xgb_fold.fit(X_val[tr_idx], val_labels[tr_idx],
                      sample_weight=sample_weights_val[tr_idx])
        preds_fold = xgb_fold.predict(X_val[te_idx])
        acc = (preds_fold == val_labels[te_idx]).mean()
        f1 = f1_score(val_labels[te_idx], preds_fold, average="macro")
        cv_accs.append(acc)
        cv_f1s.append(f1)
        logging.info(f"  Fold {fold}: acc={acc:.4f}, F1_macro={f1:.4f}")

    logging.info(f"  CV mean: acc={np.mean(cv_accs):.4f}+/-{np.std(cv_accs):.4f}, F1={np.mean(cv_f1s):.4f}+/-{np.std(cv_f1s):.4f}")

    # Baseline
    avg_val_probas = np.mean(all_val_probas, axis=0)
    avg_preds = np.argmax(avg_val_probas, axis=1)
    avg_f1 = f1_score(val_labels, avg_preds, average="macro")
    avg_acc = (avg_preds == val_labels).mean()
    logging.info(f"\n  Baseline (simple avg): acc={avg_acc:.4f}, F1_macro={avg_f1:.4f}")
    logging.info(f"  Meta-learner (XGB CV): acc={np.mean(cv_accs):.4f}, F1_macro={np.mean(cv_f1s):.4f}")

    # Final model
    xgb_final = XGBClassifier(**xgb_params)
    xgb_final.fit(X_val, val_labels, sample_weight=sample_weights_val)

    test_preds_stacking = xgb_final.predict(X_test)
    avg_test_probas = np.mean(all_test_probas, axis=0)
    test_preds_avg = np.argmax(avg_test_probas, axis=1)

    results_path = "results"
    os.makedirs(results_path, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    stacking_df = pd.DataFrame({
        'imgname': test_filenames,
        'label': [int(p) for p in test_preds_stacking]
    })
    stacking_file = os.path.join(results_path, f"submission_stacking_{timestamp}.csv")
    stacking_df.to_csv(stacking_file, index=False)
    logging.info(f"\nStacking submission saved: {stacking_file}")

    avg_df = pd.DataFrame({
        'imgname': test_filenames,
        'label': [int(p) for p in test_preds_avg]
    })
    avg_file = os.path.join(results_path, f"submission_avg_baseline_{timestamp}.csv")
    avg_df.to_csv(avg_file, index=False)
    logging.info(f"Baseline avg submission saved: {avg_file}")


def _load_cached_features(config, features_mode="all"):
    """Load features from cached .npy files (probas ± morph)."""
    import numpy as np

    cache_dir = config.get("probas_cache_dir",
                           os.path.join("results", "probas_cache_meta"))
    if not os.path.isdir(cache_dir):
        cache_dir = os.path.join("results", "probas_cache")
    if not os.path.isdir(cache_dir):
        logging.error(f"Cache not found in results/probas_cache_meta/ nor results/probas_cache/")
        return None, None, None, None, None

    logging.info(f"Loading cached features from {cache_dir}/")
    X_val = np.load(os.path.join(cache_dir, "val_probas_concat.npy"))
    val_labels = np.load(os.path.join(cache_dir, "val_labels.npy")).astype(int)
    X_test = np.load(os.path.join(cache_dir, "test_probas_concat.npy"))

    test_filenames_path = os.path.join(cache_dir, "test_filenames.npy")
    if os.path.exists(test_filenames_path):
        test_filenames = np.load(test_filenames_path, allow_pickle=True).tolist()
    else:
        test_path = config["data"]["test_path"]
        valid_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        test_filenames = sorted(
            f for f in os.listdir(test_path) if f.lower().endswith(valid_ext)
        )

    num_classes = int(val_labels.max()) + 1

    if features_mode == "probas_only":
        # Strip morph features: keep first N_models × num_classes columns
        num_models = len([f for f in os.listdir(cache_dir)
                          if f.startswith("val_probas_model")])
        if num_models > 0:
            n_proba_cols = num_models * num_classes
            X_val = X_val[:, :n_proba_cols]
            X_test = X_test[:, :n_proba_cols]
            logging.info(f"  probas_only: kept {n_proba_cols} proba cols, "
                         f"stripped {X_val.shape[1] - n_proba_cols} morph cols")

    return X_val, val_labels, X_test, test_filenames, num_classes


def _load_morph_only_features(config):
    """
    Extract morphological features from mask/image pairs.

    Scans masks_train_dir/<class>/*.png directly.
    Applies per-class subsampling controlled by ``max_per_class`` (config key):
      - 0 or absent : keep ALL images (no cap)
      - N > 0       : randomly subsample min(N, class_count) per class

    Returns X_train, y_train, X_test, test_filenames, num_classes.
    The caller (search_meta_learner_from_cache) does its own CV.
    """
    import numpy as np
    import glob as _glob
    from numpy import nanmedian
    from . import morph_features as mf

    masks_train_dir = config.get("masks_train_dir")
    masks_test_dir = config.get("masks_test_dir")
    test_path = config["data"]["test_path"]
    original_train_dir = config.get("original_train_dir")
    max_per_class = config.get("max_per_class", 0)

    if not masks_train_dir or not os.path.isdir(masks_train_dir):
        logging.error(f"masks_train_dir not found: {masks_train_dir}")
        return None, None, None, None, None
    if not masks_test_dir or not os.path.isdir(masks_test_dir):
        logging.error(f"masks_test_dir not found: {masks_test_dir}")
        return None, None, None, None, None

    # Auto-derive original_train_dir if not provided
    if not original_train_dir:
        original_train_dir = masks_train_dir.replace("_masks", "")
        logging.info(f"  original_train_dir auto-derived: {original_train_dir}")

    if not os.path.isdir(original_train_dir):
        logging.error(f"original_train_dir not found: {original_train_dir}")
        return None, None, None, None, None

    # ── Check for cached morph features ──
    morph_cache_dir = config.get("morph_cache_dir",
                                  os.path.join("results", "morph_cache"))
    cache_tag = f"mpc{max_per_class}" if max_per_class > 0 else "mpc0"
    cache_prefix = os.path.join(morph_cache_dir, cache_tag)

    cache_files = {
        "X_train": f"{cache_prefix}_X_train.npy",
        "labels":  f"{cache_prefix}_train_labels.npy",
        "X_test":  f"{cache_prefix}_X_test.npy",
        "fnames":  f"{cache_prefix}_test_filenames.npy",
        "keys":    f"{cache_prefix}_morph_keys.npy",
    }
    all_cached = all(os.path.isfile(p) for p in cache_files.values())

    if all_cached:
        logging.info(f"  Loading cached morph features from {morph_cache_dir}/ (tag={cache_tag})")
        X_train = np.load(cache_files["X_train"])
        train_labels = np.load(cache_files["labels"]).astype(int)
        X_test = np.load(cache_files["X_test"])
        test_filenames = np.load(cache_files["fnames"], allow_pickle=True).tolist()
        morph_keys = np.load(cache_files["keys"], allow_pickle=True).tolist()
        num_classes_cached = int(train_labels.max()) + 1
        logging.info(f"  Cached: train={X_train.shape}, test={X_test.shape}, "
                     f"{num_classes_cached} classes, {len(morph_keys)} features")
        return X_train, train_labels, X_test, test_filenames, num_classes_cached

    logging.info(f"  No morph cache found (looked for tag={cache_tag}), extracting from scratch...")

    # ── Discover classes from mask folder names (sorted = class index) ──
    class_folders = sorted([
        d for d in os.listdir(masks_train_dir)
        if os.path.isdir(os.path.join(masks_train_dir, d))
    ])
    num_classes = len(class_folders)
    class_to_idx = {name: i for i, name in enumerate(class_folders)}

    logging.info(f"  {num_classes} classes found in masks_train_dir")

    # ── Collect train mask/image pairs per class, then subsample ──
    rng = np.random.RandomState(42)
    train_entries = []  # (mask_path, img_path, label)
    total_before = 0

    for cls_name in class_folders:
        cls_idx = class_to_idx[cls_name]
        mask_cls_dir = os.path.join(masks_train_dir, cls_name)
        img_cls_dir = os.path.join(original_train_dir, cls_name)

        cls_entries = []
        mask_files = sorted(_glob.glob(os.path.join(mask_cls_dir, "*.png")))
        for mask_p in mask_files:
            stem = os.path.splitext(os.path.basename(mask_p))[0]
            img_p = None
            for ext in (".jpg", ".jpeg", ".png"):
                candidate = os.path.join(img_cls_dir, stem + ext)
                if os.path.exists(candidate):
                    img_p = candidate
                    break
            if img_p is not None:
                cls_entries.append((mask_p, img_p, cls_idx))

        total_before += len(cls_entries)

        # Subsample if max_per_class is set and class exceeds it
        if max_per_class > 0 and len(cls_entries) > max_per_class:
            indices = rng.choice(len(cls_entries), size=max_per_class, replace=False)
            indices.sort()
            cls_entries = [cls_entries[i] for i in indices]

        train_entries.extend(cls_entries)

    logging.info(f"  {total_before} total mask/image pairs found")
    if max_per_class > 0:
        logging.info(f"  max_per_class={max_per_class} → {len(train_entries)} selected")
    else:
        logging.info(f"  max_per_class=0 (no cap) → {len(train_entries)} selected")

    # Class distribution after subsampling
    train_labels_arr = np.array([e[2] for e in train_entries], dtype=int)
    counts = np.bincount(train_labels_arr, minlength=num_classes)
    logging.info(f"  Class distribution: min={counts[counts > 0].min()}, "
                 f"max={counts.max()}, median={int(np.median(counts[counts > 0]))}")

    # ── Test: list images and derive mask paths ──
    valid_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    test_filenames = sorted(
        f for f in os.listdir(test_path) if f.lower().endswith(valid_ext)
    )
    logging.info(f"  {len(test_filenames)} test images found")

    # ── Extract morph features (parallel or sequential) ──
    n_extract_jobs = config.get("n_extract_jobs", 1)

    def _extract_test_one(fname):
        stem = os.path.splitext(fname)[0]
        mask_p = os.path.join(masks_test_dir, stem + ".png")
        img_p = os.path.join(test_path, fname)
        if os.path.exists(mask_p) and os.path.exists(img_p):
            return mf.extract_features(mask_p, img_p)
        return None

    if n_extract_jobs > 1:
        from joblib import Parallel, delayed

        logging.info(f"Extracting morph features for {len(train_entries)} train images "
                     f"({n_extract_jobs} workers)...")
        train_morph_rows = Parallel(n_jobs=n_extract_jobs, verbose=5)(
            delayed(mf.extract_features)(mask_p, img_p)
            for mask_p, img_p, _ in train_entries
        )

        logging.info(f"Extracting morph features for {len(test_filenames)} test images "
                     f"({n_extract_jobs} workers)...")
        test_morph_rows = Parallel(n_jobs=n_extract_jobs, verbose=5)(
            delayed(_extract_test_one)(fname)
            for fname in test_filenames
        )
    else:
        logging.info(f"Extracting morph features for {len(train_entries)} train images...")
        train_morph_rows = []
        for mask_p, img_p, _ in tqdm(train_entries, desc="Train morph"):
            train_morph_rows.append(mf.extract_features(mask_p, img_p))

        logging.info(f"Extracting morph features for {len(test_filenames)} test images...")
        test_morph_rows = []
        for fname in tqdm(test_filenames, desc="Test morph"):
            test_morph_rows.append(_extract_test_one(fname))

    # Collect all morph keys from results
    all_morph_keys = None
    for row in train_morph_rows + test_morph_rows:
        if row is not None:
            if all_morph_keys is None:
                all_morph_keys = set(row.keys())
            else:
                all_morph_keys.update(row.keys())

    if all_morph_keys is None or len(all_morph_keys) == 0:
        logging.error("No morph features could be extracted. Check mask paths.")
        return None, None, None, None, None

    # ── Build feature arrays ──
    morph_keys = sorted(all_morph_keys)
    n_morph = len(morph_keys)
    logging.info(f"  {n_morph} morph features: {morph_keys[:10]}{'...' if n_morph > 10 else ''}")

    X_train = np.full((len(train_morph_rows), n_morph), np.nan, dtype=np.float64)
    train_labels = np.array([e[2] for e in train_entries], dtype=int)
    for i, row in enumerate(train_morph_rows):
        if row is not None:
            for j, key in enumerate(morph_keys):
                X_train[i, j] = row.get(key, np.nan)

    X_test = np.full((len(test_morph_rows), n_morph), np.nan, dtype=np.float64)
    for i, row in enumerate(test_morph_rows):
        if row is not None:
            for j, key in enumerate(morph_keys):
                X_test[i, j] = row.get(key, np.nan)

    # Drop rows where extraction failed entirely (all NaN)
    valid_train = ~np.all(np.isnan(X_train), axis=1)
    n_dropped = (~valid_train).sum()
    if n_dropped > 0:
        logging.info(f"  Dropped {n_dropped}/{len(X_train)} train rows (extraction failed)")
        X_train = X_train[valid_train]
        train_labels = train_labels[valid_train]

    # Impute remaining NaN with column median (computed on train only)
    for col in range(n_morph):
        med = nanmedian(X_train[:, col])
        if np.isnan(med):
            med = 0.0
        X_train[np.isnan(X_train[:, col]), col] = med
        X_test[np.isnan(X_test[:, col]), col] = med

    train_ok = valid_train.sum()
    test_ok = np.sum(np.array([r is not None for r in test_morph_rows]))
    logging.info(f"  Final: train={X_train.shape}, test={X_test.shape}")
    logging.info(f"  Morph extracted: train={train_ok}/{len(train_morph_rows)}, "
                 f"test={test_ok}/{len(test_morph_rows)}")

    # ── Save to cache ──
    os.makedirs(morph_cache_dir, exist_ok=True)
    np.save(cache_files["X_train"], X_train)
    np.save(cache_files["labels"], train_labels)
    np.save(cache_files["X_test"], X_test)
    np.save(cache_files["fnames"], np.array(test_filenames, dtype=object))
    np.save(cache_files["keys"], np.array(morph_keys, dtype=object))
    logging.info(f"  ✅ Morph features cached to {morph_cache_dir}/ (tag={cache_tag})")

    return X_train, train_labels, X_test, test_filenames, num_classes


def _speed_tier(name):
    """Return speed priority (lower = faster, run first)."""
    _TIERS = [
        (("NearestCentroid", "GaussianNB"), 0),
        (("Perceptron", "Ridge_", "SGD_", "PassiveAgg"), 1),
        (("LDA_",), 2),
        (("LogReg_L2",), 3),
        (("LogReg_L1", "LogReg_EN"), 4),
        (("ExtraTrees", "RF_", "BagStumps"), 5),
        (("LinearSVC",), 6),
        (("AdaBoost",), 7),
        (("LGBM", "XGB"), 8),
    ]
    for prefixes, tier in _TIERS:
        for prefix in prefixes:
            if name.startswith(prefix):
                return tier
    return 5


def _evaluate_single_candidate(name, model, needs_scaling, supports_sw,
                                X_raw, X_scaled, y_all, sample_weights, folds):
    """Evaluate one candidate with k-fold CV. Designed for joblib.Parallel."""
    import time
    import warnings
    import numpy as np
    from sklearn.metrics import f1_score
    from sklearn.base import clone

    X = X_scaled if needs_scaling else X_raw
    fold_f1s = []
    fold_accs = []
    t0 = time.time()

    try:
        for tr_idx, te_idx in folds:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model_clone = clone(model)
                if supports_sw:
                    model_clone.fit(X[tr_idx], y_all[tr_idx],
                                    sample_weight=sample_weights[tr_idx])
                else:
                    model_clone.fit(X[tr_idx], y_all[tr_idx])

            preds = model_clone.predict(X[te_idx])
            f1 = f1_score(y_all[te_idx], preds, average="macro")
            acc = (preds == y_all[te_idx]).mean()
            fold_f1s.append(f1)
            fold_accs.append(acc)

        elapsed = time.time() - t0
        return {
            "name": name,
            "f1_mean": float(np.mean(fold_f1s)),
            "f1_std": float(np.std(fold_f1s)),
            "acc_mean": float(np.mean(fold_accs)),
            "time_s": elapsed,
            "needs_scaling": needs_scaling,
            "supports_sw": supports_sw,
            "error": None,
        }
    except Exception as e:
        return {
            "name": name, "f1_mean": 0.0, "f1_std": 0.0,
            "acc_mean": 0.0, "time_s": time.time() - t0,
            "needs_scaling": needs_scaling, "supports_sw": supports_sw,
            "error": str(e),
        }


def _reduce_model_njobs(model):
    """Set n_jobs=1 on model to avoid CPU oversubscription in parallel mode."""
    from sklearn.base import clone
    try:
        params = model.get_params()
        updates = {k: 1 for k in params if k == 'n_jobs' or k.endswith('__n_jobs')}
        if updates:
            model = clone(model)
            model.set_params(**updates)
    except Exception:
        pass
    return model


def search_meta_learner_from_cache(config):
    """
    Exhaustive search of simple meta-learners on cached features.

    Supports three feature modes via config key `features_mode`:
      - "all"         : probas + morph features from cache (default)
      - "probas_only" : strip morph cols, keep only model probas
      - "morph_only"  : extract morph features from masks (ignores probas)

    Config keys:
      features_mode: "morph_only"   # "all" | "probas_only" | "morph_only"
      probas_cache_dir: "results/probas_cache"   # for labels & probas
      masks_train_dir: ".../original_dataset_optimized_512px_masks/train"
      masks_test_dir:  ".../original_dataset_optimized_512px_masks/test/imgs"
      top_k_submit: 5
      n_cv_folds: 5

    Usage:
        python -m torchtmpl.main config.yaml search_meta_learner_from_cache
    """
    import time
    import warnings
    import numpy as np
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    features_mode = config.get("features_mode", "all")
    n_cv_folds = config.get("n_cv_folds", 5)
    top_k_submit = config.get("top_k_submit", 5)

    if features_mode == "morph_only":
        X_all, y_all, X_test, test_filenames, num_classes = \
            _load_morph_only_features(config)
    else:
        X_all, y_all, X_test, test_filenames, num_classes = \
            _load_cached_features(config, features_mode)

    if X_all is None:
        return

    logging.info(f"  Features mode: {features_mode}")
    logging.info(f"  train={X_all.shape}, test={X_test.shape}, "
                 f"{num_classes} classes, {n_cv_folds}-fold CV")

    # ── Sample weights (inverse class freq) ──
    class_counts = np.bincount(y_all, minlength=num_classes)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * num_classes
    sample_weights = class_weights[y_all]

    # ── Feature selection ──
    feature_select_k = config.get("feature_select_k", 0)  # 0 = disabled
    if feature_select_k > 0 and feature_select_k < X_all.shape[1]:
        from sklearn.feature_selection import mutual_info_classif, SelectKBest

        logging.info(f"  Feature selection: keeping top {feature_select_k}/{X_all.shape[1]} "
                     f"features (mutual information)...")
        selector = SelectKBest(
            score_func=mutual_info_classif, k=feature_select_k
        )
        X_all = selector.fit_transform(X_all, y_all)
        X_test = selector.transform(X_test)

        mask_sel = selector.get_support()
        scores = selector.scores_
        selected_indices = np.where(mask_sel)[0]
        top_feats = sorted(zip(selected_indices, scores[mask_sel]),
                           key=lambda x: x[1], reverse=True)
        logging.info(f"  Selected {X_all.shape[1]} features. "
                     f"Top-5 MI scores: {[(int(i), f'{s:.4f}') for i, s in top_feats[:5]]}")
        logging.info(f"  After selection: train={X_all.shape}, test={X_test.shape}")
    elif feature_select_k > 0:
        logging.info(f"  feature_select_k={feature_select_k} >= n_features={X_all.shape[1]}, "
                     f"skipping selection.")

    # ── Standardize features (important for linear models) ──
    scaler = StandardScaler()
    X_all_sc = scaler.fit_transform(X_all)
    X_test_sc = scaler.transform(X_test)

    # ── Build model zoo ──
    from sklearn.linear_model import (
        LogisticRegression, RidgeClassifier, SGDClassifier,
        PassiveAggressiveClassifier, Perceptron,
    )
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import NearestCentroid
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import (
        ExtraTreesClassifier, RandomForestClassifier,
        BaggingClassifier, AdaBoostClassifier,
    )
    from sklearn.tree import DecisionTreeClassifier

    candidates = []

    # ─── 1. Logistic Regression (L2, various C) ───
    for C in [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 7.0, 10.0]:
        candidates.append((
            f"LogReg_L2_C{C}",
            LogisticRegression(C=C, solver="lbfgs", max_iter=2000,
                               n_jobs=8, random_state=42),
            True,   # needs_scaling
            True,   # supports sample_weight
        ))

    # ─── 2. Logistic Regression (L1, various C) ───
    for C in [0.01, 0.1, 1.0]:
        candidates.append((
            f"LogReg_L1_C{C}",
            LogisticRegression(C=C, penalty="l1", solver="saga", max_iter=2000,
                               n_jobs=8, random_state=42),
            True, True,
        ))

    # ─── 3. Logistic Regression (ElasticNet) ───
    for C in [0.01, 0.1, 1.0]:
        for l1_ratio in [0.3, 0.5, 0.7]:
            candidates.append((
                f"LogReg_EN_C{C}_l1r{l1_ratio}",
                LogisticRegression(C=C, penalty="elasticnet", solver="saga",
                                   l1_ratio=l1_ratio, max_iter=2000,
                                   n_jobs=8, random_state=42),
                True, True,
            ))

    # ─── 4. Ridge Classifier ───
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        candidates.append((
            f"Ridge_a{alpha}",
            RidgeClassifier(alpha=alpha),
            True, True,
        ))

    # ─── 5. SGD Classifier (various losses) ───
    for loss in ["hinge", "modified_huber", "log_loss"]:
        for alpha in [1e-3, 1e-4, 1e-5]:
            candidates.append((
                f"SGD_{loss}_a{alpha}",
                SGDClassifier(loss=loss, alpha=alpha, max_iter=2000,
                              n_jobs=8, random_state=42),
                True, True,
            ))

    # ─── 6. Passive Aggressive ───
    for C in [0.01, 0.1, 1.0]:
        candidates.append((
            f"PassiveAgg_C{C}",
            PassiveAggressiveClassifier(C=C, max_iter=2000,
                                        random_state=42),
            True, True,
        ))

    # ─── 7. Perceptron ───
    for alpha in [1e-3, 1e-4]:
        candidates.append((
            f"Perceptron_a{alpha}",
            Perceptron(alpha=alpha, max_iter=2000,
                       n_jobs=8, random_state=42),
            True, True,
        ))

    # ─── 8. Linear Discriminant Analysis ───
    for shrinkage in ["auto"]:
        candidates.append((
            f"LDA_shrink={shrinkage}",
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage=shrinkage),
            True, False,
        ))

    # ─── 9. Gaussian Naive Bayes ───
    for var_smoothing in [1e-9, 1e-7, 1e-5, 1e-3]:
        candidates.append((
            f"GaussianNB_vs{var_smoothing}",
            GaussianNB(var_smoothing=var_smoothing),
            True, False,
        ))

    # ─── 10. Nearest Centroid ───
    for metric in ["euclidean", "manhattan"]:
        candidates.append((
            f"NearestCentroid_{metric}",
            NearestCentroid(metric=metric),
            True, False,
        ))

    # ─── 11. Linear SVC (calibrated) ───
    for C in [0.01, 0.1, 1.0]:
        candidates.append((
            f"LinearSVC_C{C}",
            CalibratedClassifierCV(
                LinearSVC(C=C, max_iter=5000,
                          dual="auto", random_state=42),
                cv=3
            ),
            True, True,
        ))

    # ─── 12. Extra Trees (very constrained = low overfitting) ───
    for n_est in [50, 100]:
        for max_depth in [3, 5, 8]:
            for max_features in ["sqrt", "log2"]:
                candidates.append((
                    f"ExtraTrees_n{n_est}_d{max_depth}_{max_features}",
                    ExtraTreesClassifier(
                        n_estimators=n_est, max_depth=max_depth,
                        max_features=max_features, min_samples_leaf=5,
                        n_jobs=8, random_state=42),
                    False, True,
                ))

    # ─── 13. Random Forest (very constrained) ───
    for n_est in [50, 100]:
        for max_depth in [3, 5]:
            candidates.append((
                f"RF_n{n_est}_d{max_depth}",
                RandomForestClassifier(
                    n_estimators=n_est, max_depth=max_depth,
                    min_samples_leaf=10, max_features="sqrt",
                    n_jobs=8, random_state=42),
                False, True,
            ))

    # ─── 14. Bagging of shallow decision stumps ───
    for n_est in [50, 100, 200]:
        candidates.append((
            f"BagStumps_n{n_est}",
            BaggingClassifier(
                estimator=DecisionTreeClassifier(max_depth=2, random_state=42),
                n_estimators=n_est, max_samples=0.7, max_features=0.7,
                n_jobs=8, random_state=42),
            False, False,
        ))

    # ─── 15. AdaBoost with shallow trees ───
    for n_est in [50, 100, 200]:
        for lr in [0.05, 0.1, 0.5]:
            candidates.append((
                f"AdaBoost_n{n_est}_lr{lr}",
                AdaBoostClassifier(
                    estimator=DecisionTreeClassifier(max_depth=1),
                    n_estimators=n_est, learning_rate=lr,
                    random_state=42),
                False, True,
            ))

    # ─── 16. LightGBM (if installed) — very regularized ───
    try:
        from lightgbm import LGBMClassifier
        for n_est in [50, 100, 200]:
            for max_depth in [2, 3, 4]:
                for lr in [0.03, 0.1]:
                    for reg_lambda in [1.0, 10.0]:
                        candidates.append((
                            f"LGBM_n{n_est}_d{max_depth}_lr{lr}_l{reg_lambda}",
                            LGBMClassifier(
                                n_estimators=n_est, max_depth=max_depth,
                                learning_rate=lr, num_leaves=2**max_depth - 1,
                                min_child_samples=20, subsample=0.8,
                                colsample_bytree=0.8, reg_alpha=0.5,
                                reg_lambda=reg_lambda,
                                n_jobs=8, random_state=42, verbose=-1),
                            False, True,
                        ))
    except ImportError:
        logging.info("LightGBM not installed — skipping LGBM candidates.")

    # ─── 17. XGBoost (very regularized) ───
    try:
        from xgboost import XGBClassifier
        for n_est in [50, 100, 200]:
            for max_depth in [2, 3, 4]:
                for lr in [0.03, 0.1]:
                    for reg_lambda in [5.0, 20.0]:
                        candidates.append((
                            f"XGB_n{n_est}_d{max_depth}_lr{lr}_l{reg_lambda}",
                            XGBClassifier(
                                n_estimators=n_est, max_depth=max_depth,
                                learning_rate=lr, min_child_weight=10,
                                subsample=0.7, colsample_bytree=0.7,
                                reg_alpha=1.0, reg_lambda=reg_lambda,
                                objective="multi:softprob",
                                num_class=num_classes, tree_method="hist",
                                eval_metric="mlogloss",
                                n_jobs=8, random_state=42),
                            False, True,
                        ))
    except ImportError:
        logging.info("XGBoost not installed — skipping XGB candidates.")

    # ── Sort candidates: fastest models first ──
    candidates.sort(key=lambda c: _speed_tier(c[0]))

    # ── Parallel config ──
    n_parallel = config.get("n_parallel_models", 5)
    if n_parallel > 1:
        logging.info(f"  Parallel mode: {n_parallel} workers — reducing internal n_jobs to 1")
        candidates = [
            (name, _reduce_model_njobs(model), ns, sw)
            for name, model, ns, sw in candidates
        ]

    n_candidates = len(candidates)
    logging.info(f"\n{'='*70}")
    logging.info(f"  META-LEARNER SEARCH: {n_candidates} model configs × {n_cv_folds}-fold CV")
    logging.info(f"{'='*70}\n")

    # ── CV evaluation ──
    skf = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=42)
    folds = list(skf.split(X_all, y_all))

    # Probas-avg baseline (only relevant when probas are available)
    baseline_f1 = 0.0
    cache_dir = config.get("probas_cache_dir", "")
    if cache_dir and os.path.isdir(cache_dir):
        num_models_per = len([f for f in os.listdir(cache_dir)
                              if f.startswith("val_probas_model")])
        if num_models_per > 0:
            avg_probas = np.mean([
                np.load(os.path.join(cache_dir, f"val_probas_model{i}.npy"))
                for i in range(num_models_per)
            ], axis=0)
            avg_preds = np.argmax(avg_probas, axis=1)
            baseline_f1 = f1_score(y_all, avg_preds, average="macro")
            baseline_acc = (avg_preds == y_all).mean()
            logging.info(f"  Baseline (simple avg): acc={baseline_acc:.4f}, F1_macro={baseline_f1:.4f}\n")

    if n_parallel > 1:
        # ── Parallel evaluation via joblib ──
        from joblib import Parallel, delayed

        logging.info(f"  Evaluating {n_candidates} candidates with {n_parallel} parallel workers...")
        results = Parallel(n_jobs=n_parallel, verbose=10)(
            delayed(_evaluate_single_candidate)(
                name, model, needs_scaling, supports_sw,
                X_all, X_all_sc, y_all, sample_weights, folds
            )
            for name, model, needs_scaling, supports_sw in candidates
        )

        # Log summary for each candidate
        for idx, r in enumerate(results):
            if r["error"]:
                logging.warning(
                    f"  [{idx+1:3d}/{n_candidates}] {r['name']:45s}  FAILED: {r['error']}")
            else:
                marker = " **" if r["f1_mean"] > baseline_f1 else ""
                logging.info(
                    f"  [{idx+1:3d}/{n_candidates}] {r['name']:45s}  "
                    f"F1={r['f1_mean']:.4f}±{r['f1_std']:.4f}  acc={r['acc_mean']:.4f}  "
                    f"({r['time_s']:5.1f}s){marker}"
                )

    else:
        # ── Sequential evaluation with real-time logging ──
        results = []

        for idx, (name, model, needs_scaling, supports_sw) in enumerate(candidates):
            X = X_all_sc if needs_scaling else X_all

            fold_f1s = []
            fold_accs = []
            t0 = time.time()

            try:
                for fold_idx, (tr_idx, te_idx) in enumerate(folds):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        if supports_sw:
                            model_clone = _clone_model(model)
                            model_clone.fit(X[tr_idx], y_all[tr_idx],
                                            sample_weight=sample_weights[tr_idx])
                        else:
                            model_clone = _clone_model(model)
                            model_clone.fit(X[tr_idx], y_all[tr_idx])

                    preds = model_clone.predict(X[te_idx])
                    f1 = f1_score(y_all[te_idx], preds, average="macro")
                    acc = (preds == y_all[te_idx]).mean()
                    fold_f1s.append(f1)
                    fold_accs.append(acc)

                elapsed = time.time() - t0
                mean_f1 = np.mean(fold_f1s)
                std_f1 = np.std(fold_f1s)
                mean_acc = np.mean(fold_accs)

                results.append({
                    "name": name,
                    "f1_mean": mean_f1,
                    "f1_std": std_f1,
                    "acc_mean": mean_acc,
                    "time_s": elapsed,
                    "needs_scaling": needs_scaling,
                    "supports_sw": supports_sw,
                    "error": None,
                })

                marker = " **" if mean_f1 > baseline_f1 else ""
                logging.info(
                    f"  [{idx+1:3d}/{n_candidates}] {name:45s}  "
                    f"F1={mean_f1:.4f}±{std_f1:.4f}  acc={mean_acc:.4f}  "
                    f"({elapsed:5.1f}s){marker}"
                )

            except Exception as e:
                logging.warning(f"  [{idx+1:3d}/{n_candidates}] {name:45s}  FAILED: {e}")
                results.append({
                    "name": name, "f1_mean": 0.0, "f1_std": 0.0,
                    "acc_mean": 0.0, "time_s": 0.0,
                    "needs_scaling": needs_scaling, "supports_sw": supports_sw,
                    "error": str(e),
                })

    # ── Leaderboard ──
    results.sort(key=lambda r: r["f1_mean"], reverse=True)

    logging.info(f"\n{'='*70}")
    logging.info(f"  LEADERBOARD (top 30 / {n_candidates})")
    logging.info(f"{'='*70}")
    logging.info(f"  {'Rank':>4s}  {'Model':45s}  {'F1_macro':>10s}  {'±std':>8s}  {'Acc':>7s}  {'Time':>6s}")
    logging.info(f"  {'-'*4}  {'-'*45}  {'-'*10}  {'-'*8}  {'-'*7}  {'-'*6}")

    for rank, r in enumerate(results[:30]):
        beats = " ✓" if r["f1_mean"] > baseline_f1 else ""
        logging.info(
            f"  {rank+1:4d}  {r['name']:45s}  {r['f1_mean']:10.4f}  "
            f"±{r['f1_std']:.4f}  {r['acc_mean']:.4f}  {r['time_s']:5.1f}s{beats}"
        )

    logging.info(f"\n  Baseline (simple avg): F1_macro={baseline_f1:.4f}")

    # ── Generate test CSVs for top-K models ──
    if top_k_submit > 0:
        logging.info(f"\n{'='*70}")
        logging.info(f"  Generating test predictions for top-{top_k_submit} models")
        logging.info(f"{'='*70}")

        results_path = "results"
        os.makedirs(results_path, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")

        for rank, r in enumerate(results[:top_k_submit]):
            name = r["name"]
            needs_sc = r["needs_scaling"]
            supports_sw = r["supports_sw"]

            X = X_all_sc if needs_sc else X_all
            X_t = X_test_sc if needs_sc else X_test

            # Retrain on ALL training data
            # Find original model object
            orig_model = None
            for cname, cmodel, csc, csw in candidates:
                if cname == name:
                    orig_model = cmodel
                    break

            if orig_model is None:
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                final_model = _clone_model(orig_model)
                if supports_sw:
                    final_model.fit(X, y_all, sample_weight=sample_weights)
                else:
                    final_model.fit(X, y_all)

            test_preds = final_model.predict(X_t)

            safe_name = name.replace(" ", "_").replace("/", "_")
            output_file = os.path.join(
                results_path,
                f"sub_search_{rank+1}_{safe_name}_{timestamp}.csv"
            )
            df = pd.DataFrame({
                'imgname': test_filenames,
                'label': [int(p) for p in test_preds]
            })
            df.to_csv(output_file, index=False)
            logging.info(f"  [{rank+1}] {name} → {output_file}")

    logging.info(f"\n✅ Search complete. {n_candidates} models evaluated.")


def _clone_model(model):
    """Deep-clone a sklearn estimator (avoids import of sklearn.base at module level)."""
    from sklearn.base import clone
    return clone(model)


def logreg_morph_search(config):
    """
    Focused search: LogReg with multiple feature engineering pipelines.

    Pipelines tested (all combined with LogReg at various C):
      1. Raw scaled features
      2. PCA at various n_components
      3. Tree-based feature importance selection (ExtraTrees)
      4. Mutual information selection
      5. PCA + tree-selected features combined
      6. Variance threshold + PCA

    Config keys (same as search_meta_learner_from_cache, plus):
      pca_components: [10, 20, 30, 50, 0.90, 0.95, 0.99]  # ints or floats for variance ratio
      logreg_C_values: [0.001, 0.01, ..., 100.0]
      tree_select_k: [15, 30, 50]                           # top-k from tree importance

    Usage:
        python -m torchtmpl.main config_search_meta_learner.yaml logreg_morph_search
    """
    import time
    import warnings
    import numpy as np
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.feature_selection import (
        SelectFromModel, SelectKBest, mutual_info_classif,
        VarianceThreshold,
    )
    from sklearn.pipeline import Pipeline

    features_mode = config.get("features_mode", "morph_only")
    n_cv_folds = config.get("n_cv_folds", 5)
    top_k_submit = config.get("top_k_submit", 5)

    if features_mode == "morph_only":
        X_all, y_all, X_test, test_filenames, num_classes = \
            _load_morph_only_features(config)
    else:
        X_all, y_all, X_test, test_filenames, num_classes = \
            _load_cached_features(config, features_mode)

    if X_all is None:
        return

    n_samples, n_features = X_all.shape
    logging.info(f"  Features mode: {features_mode}")
    logging.info(f"  train={X_all.shape}, test={X_test.shape}, "
                 f"{num_classes} classes, {n_cv_folds}-fold CV")

    # ── Sample weights ──
    class_counts = np.bincount(y_all, minlength=num_classes)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * num_classes
    sample_weights = class_weights[y_all]

    # ── Config for search grid ──
    C_values = config.get("logreg_C_values",
                          [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0])
    pca_components = config.get("pca_components", [10, 20, 30, 50, 0.90, 0.95, 0.99])
    tree_select_ks = config.get("tree_select_k", [15, 30, 50])

    # Clamp PCA & tree-k to available features
    pca_components = [p for p in pca_components
                      if (isinstance(p, float) and 0 < p < 1) or
                         (isinstance(p, int) and 0 < p < n_features)]
    tree_select_ks = [k for k in tree_select_ks if 0 < k < n_features]

    # ── Tree-based importance (fit once on full data to rank features) ──
    logging.info("  Fitting ExtraTrees for feature importance ranking...")
    et_selector = ExtraTreesClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=5,
        n_jobs=8, random_state=42,
    )
    et_selector.fit(X_all, y_all, sample_weight=sample_weights)
    importances = et_selector.feature_importances_
    feat_ranking = np.argsort(importances)[::-1]

    logging.info(f"  Top-10 features by tree importance: "
                 f"{feat_ranking[:10].tolist()} "
                 f"(importances: {importances[feat_ranking[:10]].round(4).tolist()})")

    # ── Build feature pipelines ──
    # Each pipeline: (name, transformer_or_None)
    #   transformer=None → just StandardScaler (raw)
    #   transformer=object → StandardScaler → transformer
    pipelines = []

    # 1. PCA at various components
    for n_comp in pca_components:
        tag = f"pca{n_comp}" if isinstance(n_comp, int) else f"pca{n_comp:.0%}var"
        pipelines.append((tag, PCA(n_components=n_comp, random_state=42)))

    # 4. Tree-based top-K selection
    for k in tree_select_ks:
        idx_topk = feat_ranking[:k]
        pipelines.append((f"tree_top{k}", ("index_select", idx_topk)))

    # 5. MI-based top-K selection
    for k in tree_select_ks:
        pipelines.append((f"mi_top{k}",
                          SelectKBest(score_func=mutual_info_classif, k=k)))

    # 6. Variance threshold + PCA
    for n_comp in [p for p in pca_components if isinstance(p, int) and p <= 50]:
        pipelines.append((
            f"vt_pca{n_comp}",
            Pipeline([
                ("vt", VarianceThreshold(threshold=1e-4)),
                ("pca", PCA(n_components=n_comp, random_state=42)),
            ])
        ))

    # 7. Tree top-K + PCA
    for k in tree_select_ks:
        for n_comp in [p for p in pca_components
                       if isinstance(p, int) and p < k]:
            idx_topk = feat_ranking[:k]
            pipelines.append((
                f"tree{k}_pca{n_comp}",
                ("index_then_pca", idx_topk, n_comp),
            ))

    # ── Build all (pipeline × C) candidates ──
    candidates = []
    for pipe_name, transformer in pipelines:
        for C in C_values:
            candidates.append((pipe_name, transformer, C))

    n_candidates = len(candidates)
    logging.info(f"\n{'='*70}")
    logging.info(f"  LOGREG MORPH SEARCH: {len(pipelines)} pipelines × "
                 f"{len(C_values)} C values = {n_candidates} candidates, "
                 f"{n_cv_folds}-fold CV")
    logging.info(f"{'='*70}\n")

    # ── CV evaluation ──
    skf = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=42)
    folds = list(skf.split(X_all, y_all))

    results = []
    prev_pipe = None

    for idx, (pipe_name, transformer, C) in enumerate(candidates):
        # Log when entering a new pipeline group
        if pipe_name != prev_pipe:
            n_Cs_for_pipe = sum(1 for p, _, _ in candidates if p == pipe_name)
            # Describe the transformer
            if transformer is None:
                desc = "raw scaled features"
            elif isinstance(transformer, tuple) and transformer[0] == "index_select":
                desc = f"top-{len(transformer[1])} features by tree importance"
            elif isinstance(transformer, tuple) and transformer[0] == "index_then_pca":
                desc = f"top-{len(transformer[1])} tree features → PCA({transformer[2]})"
            elif isinstance(transformer, VarianceThreshold):
                desc = "variance threshold filter"
            elif isinstance(transformer, PCA):
                nc = transformer.n_components
                desc = f"PCA({nc})" if isinstance(nc, int) else f"PCA({nc:.0%} variance)"
            elif isinstance(transformer, SelectKBest):
                desc = f"mutual information top-{transformer.k}"
            elif isinstance(transformer, Pipeline):
                desc = " → ".join(type(s).__name__ for _, s in transformer.steps)
            else:
                desc = type(transformer).__name__
            logging.info(f"\n  ── Pipeline: {pipe_name} ({desc}) — {n_Cs_for_pipe} C values ──")
            prev_pipe = pipe_name

        fold_f1s = []
        fold_accs = []
        t0 = time.time()
        name = f"{pipe_name}_LogReg_C{C}"

        try:
            for fold_i, (tr_idx, te_idx) in enumerate(folds):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    X_tr, X_te = X_all[tr_idx], X_all[te_idx]
                    sw_tr = sample_weights[tr_idx]

                    # Apply scaling (always)
                    scaler = StandardScaler()
                    X_tr = scaler.fit_transform(X_tr)
                    X_te = scaler.transform(X_te)

                    # Apply feature transform
                    if transformer is None:
                        pass  # raw scaled
                    elif isinstance(transformer, tuple) and transformer[0] == "index_select":
                        idx_sel = transformer[1]
                        X_tr = X_tr[:, idx_sel]
                        X_te = X_te[:, idx_sel]
                    elif isinstance(transformer, tuple) and transformer[0] == "index_then_pca":
                        idx_sel = transformer[1]
                        n_comp = transformer[2]
                        X_tr = X_tr[:, idx_sel]
                        X_te = X_te[:, idx_sel]
                        pca = PCA(n_components=n_comp, random_state=42)
                        X_tr = pca.fit_transform(X_tr)
                        X_te = pca.transform(X_te)
                    else:
                        # sklearn transformer (PCA, SelectKBest, Pipeline, VT)
                        from sklearn.base import clone as sk_clone
                        tf = sk_clone(transformer)
                        if hasattr(tf, 'fit_transform'):
                            X_tr = tf.fit_transform(X_tr, y_all[tr_idx])
                            X_te = tf.transform(X_te)

                    model = LogisticRegression(
                        C=C, solver="saga", max_iter=2000,
                        tol=1e-3, n_jobs=4, random_state=42,
                    )
                    model.fit(X_tr, y_all[tr_idx], sample_weight=sw_tr)

                    # Log dimensionality on first C of each pipeline, first fold
                    if fold_i == 0 and (idx == 0 or candidates[idx-1][0] != pipe_name):
                        logging.info(f"    → {X_tr.shape[1]} features after transform, "
                                     f"{X_tr.shape[0]} train samples")

                preds = model.predict(X_te)
                f1 = f1_score(y_all[te_idx], preds, average="macro")
                acc = (preds == y_all[te_idx]).mean()
                fold_f1s.append(f1)
                fold_accs.append(acc)

            elapsed = time.time() - t0
            mean_f1 = np.mean(fold_f1s)
            std_f1 = np.std(fold_f1s)
            mean_acc = np.mean(fold_accs)

            results.append({
                "name": name,
                "pipe": pipe_name,
                "C": C,
                "f1_mean": mean_f1,
                "f1_std": std_f1,
                "acc_mean": mean_acc,
                "time_s": elapsed,
                "transformer": transformer,
            })

            logging.info(
                f"  [{idx+1:3d}/{n_candidates}] {name:50s}  "
                f"F1={mean_f1:.4f}±{std_f1:.4f}  acc={mean_acc:.4f}  "
                f"({elapsed:5.1f}s)"
            )

        except Exception as e:
            logging.warning(f"  [{idx+1:3d}/{n_candidates}] {name:50s}  FAILED: {e}")
            results.append({
                "name": name, "pipe": pipe_name, "C": C,
                "f1_mean": 0.0, "f1_std": 0.0,
                "acc_mean": 0.0, "time_s": 0.0,
                "transformer": transformer,
            })

    # ── Leaderboard ──
    results.sort(key=lambda r: r["f1_mean"], reverse=True)

    logging.info(f"\n{'='*70}")
    logging.info(f"  LEADERBOARD (top 30 / {n_candidates})")
    logging.info(f"{'='*70}")
    logging.info(f"  {'Rank':>4s}  {'Model':50s}  {'F1_macro':>10s}  {'±std':>8s}  {'Acc':>7s}  {'Time':>6s}")
    logging.info(f"  {'-'*4}  {'-'*50}  {'-'*10}  {'-'*8}  {'-'*7}  {'-'*6}")

    for rank, r in enumerate(results[:30]):
        logging.info(
            f"  {rank+1:4d}  {r['name']:50s}  {r['f1_mean']:10.4f}  "
            f"±{r['f1_std']:.4f}  {r['acc_mean']:.4f}  {r['time_s']:5.1f}s"
        )

    # ── Best per pipeline ──
    logging.info(f"\n{'='*70}")
    logging.info(f"  BEST PER PIPELINE")
    logging.info(f"{'='*70}")
    seen_pipes = set()
    for r in results:
        if r["pipe"] not in seen_pipes and r["f1_mean"] > 0:
            seen_pipes.add(r["pipe"])
            logging.info(f"  {r['pipe']:25s}  best_C={r['C']:<8}  "
                         f"F1={r['f1_mean']:.4f}±{r['f1_std']:.4f}")

    # ── Generate submissions for top-K ──
    if top_k_submit > 0:
        logging.info(f"\n{'='*70}")
        logging.info(f"  Generating submissions for top-{top_k_submit}")
        logging.info(f"{'='*70}")

        results_path = "results"
        os.makedirs(results_path, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")

        for rank, r in enumerate(results[:top_k_submit]):
            name = r["name"]
            C = r["C"]
            transformer = r["transformer"]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_all)
                X_te = scaler.transform(X_test)

                if transformer is None:
                    pass
                elif isinstance(transformer, tuple) and transformer[0] == "index_select":
                    idx_sel = transformer[1]
                    X_tr = X_tr[:, idx_sel]
                    X_te = X_te[:, idx_sel]
                elif isinstance(transformer, tuple) and transformer[0] == "index_then_pca":
                    idx_sel = transformer[1]
                    n_comp = transformer[2]
                    X_tr = X_tr[:, idx_sel]
                    X_te = X_te[:, idx_sel]
                    pca = PCA(n_components=n_comp, random_state=42)
                    X_tr = pca.fit_transform(X_tr)
                    X_te = pca.transform(X_te)
                else:
                    from sklearn.base import clone as sk_clone
                    tf = sk_clone(transformer)
                    X_tr = tf.fit_transform(X_tr, y_all)
                    X_te = tf.transform(X_te)

                final_model = LogisticRegression(
                    C=C, solver="saga", max_iter=2000,
                    tol=1e-3, n_jobs=8, random_state=42,
                )
                final_model.fit(X_tr, y_all, sample_weight=sample_weights)

            test_preds = final_model.predict(X_te)

            safe_name = name.replace(" ", "_").replace("/", "_")
            output_file = os.path.join(
                results_path,
                f"sub_logreg_{rank+1}_{safe_name}_{timestamp}.csv"
            )
            df = pd.DataFrame({
                'imgname': test_filenames,
                'label': [int(p) for p in test_preds]
            })
            df.to_csv(output_file, index=False)
            logging.info(f"  [{rank+1}] {name} → {output_file}")

    logging.info(f"\n✅ LogReg morph search complete. {n_candidates} candidates evaluated.")
    """
    Weighted ensemble with automatic Kaggle submission.
    Loads cached probas, optimizes weights via Nelder-Mead on val F1-macro,
    generates multiple candidate submissions (optimal, perturbations, rank avg,
    geometric mean, uniform), and submits them all to Kaggle.

    Configurable via the YAML config (all optional with sane defaults):
      kaggle_competition: "3-md-4040-2026-challenge"
      probas_cache_dir: "results/probas_cache"
      n_submissions: 15        # total candidates to generate & submit
      perturbation_sigma: 0.15 # std of Gaussian perturbations around optimum
      dry_run: false           # if true, generate CSVs but don't submit

    Usage:
        python -m torchtmpl.main config.yaml weighted_ensemble_submit
    """
    import time
    import numpy as np
    from scipy.optimize import minimize
    from scipy.special import entr, softmax as scipy_softmax
    from scipy.stats import rankdata
    from sklearn.metrics import f1_score

    # ── Config defaults ──
    competition = config.get("kaggle_competition", "3-md-4040-2026-challenge")
    cache_dir = config.get("probas_cache_dir", os.path.join("results", "probas_cache"))
    n_submissions = config.get("n_submissions", 15)
    sigma = config.get("perturbation_sigma", 0.15)
    dry_run = config.get("dry_run", False)

    # ── Kaggle API ──
    kaggle_api = None
    if not dry_run:
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            kaggle_api = KaggleApi()
            kaggle_api.authenticate()
            logging.info(f"Kaggle API authenticated. Competition: {competition}")
        except Exception as e:
            logging.error(f"Kaggle API init failed: {e}")
            logging.error("Install kaggle (pip install kaggle) and set up ~/.kaggle/kaggle.json")
            logging.error("Falling back to dry_run mode (CSV only, no submit).")
            dry_run = True

    # ── Load cached probas ──
    if not os.path.isdir(cache_dir):
        logging.error(f"Cache not found: {cache_dir}/ — run ensemble_csv_stacking first.")
        return

    val_labels = np.load(os.path.join(cache_dir, "val_labels.npy"))
    test_filenames = np.load(os.path.join(cache_dir, "test_filenames.npy"), allow_pickle=True).tolist()

    num_models = len([f for f in os.listdir(cache_dir) if f.startswith("val_probas_model")])
    all_val_probas = [np.load(os.path.join(cache_dir, f"val_probas_model{i}.npy")) for i in range(num_models)]
    all_test_probas = [np.load(os.path.join(cache_dir, f"test_probas_model{i}.npy")) for i in range(num_models)]

    num_classes = all_val_probas[0].shape[1]
    N_val = len(val_labels)
    N_test = len(test_filenames)
    logging.info(f"Loaded {num_models} models, {num_classes} classes, {N_val} val / {N_test} test samples")

    # ══════════════════════════════════════════════════════════════════════
    # Helper: generate submission CSV + optionally submit to Kaggle
    # ══════════════════════════════════════════════════════════════════════
    results_path = "results"
    os.makedirs(results_path, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    submissions_log = []  # track all submissions for final summary

    def _make_and_submit(test_preds, method_name, description, val_f1, weights_str=""):
        """Save CSV and submit to Kaggle. Returns the CSV path."""
        fname = f"sub_{method_name}_{timestamp}.csv"
        csv_path = os.path.join(results_path, fname)

        df = pd.DataFrame({
            'imgname': test_filenames,
            'label': [int(p) for p in test_preds]
        })
        df.to_csv(csv_path, index=False)

        msg = f"{method_name} | val_F1={val_f1:.4f}"
        if weights_str:
            msg += f" | w=[{weights_str}]"

        submissions_log.append({
            'method': method_name,
            'val_f1': val_f1,
            'weights': weights_str,
            'csv': csv_path,
            'kaggle_msg': msg,
        })

        logging.info(f"  [{len(submissions_log):2d}] {method_name:30s} val_F1={val_f1:.4f}  → {fname}")

        if not dry_run and kaggle_api is not None:
            try:
                kaggle_api.competition_submit(
                    file_name=csv_path,
                    message=msg,
                    competition=competition,
                    quiet=True
                )
                time.sleep(3)  # rate-limit safety
            except Exception as e:
                logging.warning(f"    ⚠ Kaggle submit failed: {e}")

        return csv_path

    # ══════════════════════════════════════════════════════════════════════
    # Helper: weighted average probas → preds + val F1
    # ══════════════════════════════════════════════════════════════════════
    val_stack = np.stack(all_val_probas, axis=0)   # (M, N_val, C)
    test_stack = np.stack(all_test_probas, axis=0) # (M, N_test, C)

    def _weighted_avg(weights):
        """Return (val_preds, test_preds, val_f1) for given per-model weights."""
        w = np.asarray(weights)
        val_avg = np.tensordot(w, val_stack, axes=([0], [0]))   # (N_val, C)
        test_avg = np.tensordot(w, test_stack, axes=([0], [0])) # (N_test, C)
        val_preds = np.argmax(val_avg, axis=1)
        test_preds = np.argmax(test_avg, axis=1)
        f1 = f1_score(val_labels, val_preds, average="macro")
        return val_preds, test_preds, f1

    def _softmax_weights(raw):
        """Convert raw reals to positive weights that sum to 1."""
        e = np.exp(raw - np.max(raw))
        return e / e.sum()

    def _weights_str(w):
        return ", ".join(f"{x:.3f}" for x in w)

    # ══════════════════════════════════════════════════════════════════════
    # Phase 1: Nelder-Mead optimisation of global weights
    # ══════════════════════════════════════════════════════════════════════
    logging.info("\n" + "=" * 60)
    logging.info("Phase 1: Optimizing global weights (Nelder-Mead, 20 restarts)")
    logging.info("=" * 60)

    # Fast F1-macro via numpy (avoids sklearn overhead per call)
    val_labels_int = val_labels.astype(np.int64)
    _unique_classes = np.unique(val_labels_int)

    def _fast_f1_macro(preds, labels=val_labels_int, classes=_unique_classes):
        """Compute F1-macro using vectorized numpy — much faster than sklearn."""
        f1_sum = 0.0
        for c in classes:
            tp = np.sum((preds == c) & (labels == c))
            fp = np.sum((preds == c) & (labels != c))
            fn = np.sum((preds != c) & (labels == c))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_sum += (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        return f1_sum / len(classes)

    def _neg_f1(raw_weights):
        w = _softmax_weights(raw_weights)
        avg = np.tensordot(w, val_stack, axes=([0], [0]))
        preds = np.argmax(avg, axis=1)
        return -_fast_f1_macro(preds)

    best_result = None
    best_f1_optim = -1
    for trial in range(20):
        x0 = np.zeros(num_models) if trial == 0 else np.random.randn(num_models) * 0.5
        res = minimize(_neg_f1, x0, method="Nelder-Mead",
                       options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-6})
        f1_val = -res.fun
        if f1_val > best_f1_optim:
            best_f1_optim = f1_val
            best_result = res
        logging.info(f"  Restart {trial+1}/20: F1={f1_val:.4f} (best so far: {best_f1_optim:.4f})")

    optimal_w = _softmax_weights(best_result.x)
    logging.info(f"Optimal weights: [{_weights_str(optimal_w)}]")
    logging.info(f"Optimal val F1-macro: {best_f1_optim:.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # Phase 2: Generate all candidate submissions
    # ══════════════════════════════════════════════════════════════════════
    logging.info("\n" + "=" * 60)
    logging.info(f"Phase 2: Generating {n_submissions} candidate submissions")
    logging.info("=" * 60)

    candidates_generated = 0

    # ── Candidate 0: Uniform average (baseline) ──
    uniform_w = np.ones(num_models) / num_models
    _, test_preds, val_f1 = _weighted_avg(uniform_w)
    _make_and_submit(test_preds, "uniform_avg", "Uniform average baseline", val_f1, _weights_str(uniform_w))
    candidates_generated += 1

    # ── Candidate 1: Nelder-Mead optimum ──
    _, test_preds, val_f1 = _weighted_avg(optimal_w)
    _make_and_submit(test_preds, "nelder_mead_opt", "Nelder-Mead optimal weights", val_f1, _weights_str(optimal_w))
    candidates_generated += 1

    # ── Candidate 2: Rank averaging ──
    def _rank_avg(probas_list):
        ranked = []
        for probas in probas_list:
            r = np.zeros_like(probas)
            for c in range(probas.shape[1]):
                r[:, c] = rankdata(probas[:, c], method='average')
            r = r / r.max(axis=0, keepdims=True)
            ranked.append(r)
        return np.mean(ranked, axis=0)

    rank_val = _rank_avg(all_val_probas)
    rank_test = _rank_avg(all_test_probas)
    rank_f1 = f1_score(val_labels, np.argmax(rank_val, axis=1), average="macro")
    _make_and_submit(np.argmax(rank_test, axis=1), "rank_avg", "Rank averaging", rank_f1)
    candidates_generated += 1

    # ── Candidate 3: Geometric mean ──
    log_val = [np.log(p + 1e-12) for p in all_val_probas]
    log_test = [np.log(p + 1e-12) for p in all_test_probas]
    geo_val = np.exp(np.mean(log_val, axis=0))
    geo_test = np.exp(np.mean(log_test, axis=0))
    geo_f1 = f1_score(val_labels, np.argmax(geo_val, axis=1), average="macro")
    _make_and_submit(np.argmax(geo_test, axis=1), "geometric_mean", "Geometric mean", geo_f1)
    candidates_generated += 1

    # ── Candidate 4: Confidence-weighted (inverse entropy, temp=1) ──
    def _confidence_weighted(probas_list, temperature=1.0):
        M = len(probas_list)
        N = probas_list[0].shape[0]
        entropies = np.zeros((M, N))
        for m, p in enumerate(probas_list):
            entropies[m] = entr(p + 1e-12).sum(axis=1)
        conf = (1.0 / (entropies + 1e-8)) ** temperature
        weights = conf / conf.sum(axis=0, keepdims=True)
        result = np.zeros_like(probas_list[0])
        for m in range(M):
            result += weights[m, :, np.newaxis] * probas_list[m]
        return result

    conf_val = _confidence_weighted(all_val_probas, temperature=1.0)
    conf_test = _confidence_weighted(all_test_probas, temperature=1.0)
    conf_f1 = f1_score(val_labels, np.argmax(conf_val, axis=1), average="macro")
    _make_and_submit(np.argmax(conf_test, axis=1), "confidence_t1.0", "Confidence-weighted temp=1.0", conf_f1)
    candidates_generated += 1

    # ── Candidate 5: Confidence-weighted (temp=2) ──
    conf_val2 = _confidence_weighted(all_val_probas, temperature=2.0)
    conf_test2 = _confidence_weighted(all_test_probas, temperature=2.0)
    conf_f1_2 = f1_score(val_labels, np.argmax(conf_val2, axis=1), average="macro")
    _make_and_submit(np.argmax(conf_test2, axis=1), "confidence_t2.0", "Confidence-weighted temp=2.0", conf_f1_2)
    candidates_generated += 1

    # ── Candidates 6+: Perturbations around the Nelder-Mead optimum ──
    n_perturb = max(0, n_submissions - candidates_generated)
    logging.info(f"\nGenerating {n_perturb} perturbation candidates (sigma={sigma})")

    # Sort perturbations by val F1 and keep only the top n_perturb
    perturb_candidates = []
    n_trials = max(n_perturb * 5, 50)  # generate more than needed, keep best
    for i in range(n_trials):
        noise = np.random.randn(num_models) * sigma
        raw_perturbed = best_result.x + noise
        w = _softmax_weights(raw_perturbed)
        _, test_preds, val_f1 = _weighted_avg(w)
        perturb_candidates.append((val_f1, w, test_preds))

    # Sort by val F1 descending, pick top n_perturb that are diverse enough
    perturb_candidates.sort(key=lambda x: x[0], reverse=True)

    submitted_weights = [optimal_w, uniform_w]  # already submitted
    for val_f1, w, test_preds in perturb_candidates:
        if candidates_generated >= n_submissions:
            break
        # Skip if too similar to an already submitted weight vector
        too_similar = any(np.max(np.abs(w - prev)) < 0.02 for prev in submitted_weights)
        if too_similar:
            continue
        _make_and_submit(test_preds, f"perturb_{candidates_generated - 5}",
                         f"Perturbation around optimum (sigma={sigma})",
                         val_f1, _weights_str(w))
        submitted_weights.append(w)
        candidates_generated += 1

    # ══════════════════════════════════════════════════════════════════════
    # Phase 3: Summary
    # ══════════════════════════════════════════════════════════════════════
    logging.info("\n" + "=" * 60)
    logging.info("SUMMARY")
    logging.info("=" * 60)
    logging.info(f"{'#':>3s}  {'Method':30s}  {'Val F1':>8s}  {'Weights'}")
    logging.info("-" * 80)
    for i, s in enumerate(submissions_log):
        logging.info(f"{i+1:3d}  {s['method']:30s}  {s['val_f1']:8.4f}  {s['weights']}")

    if dry_run:
        logging.info(f"\n⚠ DRY RUN — {len(submissions_log)} CSVs generated but NOT submitted to Kaggle.")
    else:
        logging.info(f"\n✅ {len(submissions_log)} submissions sent to Kaggle ({competition}).")
        logging.info("   Check https://www.kaggle.com/competitions/" + competition + "/submissions")

    logging.info(f"\nCSVs saved in: {os.path.abspath(results_path)}/")
    return submissions_log


def train_logreg_morph(config):
    """
    Train a single LogisticRegression on ALL morphological features.

    Steps:
      1. Load morph features (respects max_per_class from config)
      2. 5-fold stratified CV to evaluate
      3. Retrain on full data
      4. Generate submission CSV

    Config keys:
      max_per_class: 2000       # cap per-class images
      logreg_C: 1.0             # regularisation strength
      n_cv_folds: 5             # CV folds

    Usage:
        python -m torchtmpl.main config_train_logreg_morph.yaml train_logreg_morph
    """
    import time
    import warnings
    import numpy as np
    from sklearn.metrics import f1_score, classification_report
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    n_cv_folds = config.get("n_cv_folds", 5)
    C = config.get("logreg_C", 1.0)

    # ── Load features ──
    X_all, y_all, X_test, test_filenames, num_classes = \
        _load_morph_only_features(config)
    if X_all is None:
        return

    n_samples, n_features = X_all.shape
    logging.info(f"  train={X_all.shape}, test={X_test.shape}, "
                 f"{num_classes} classes, {n_features} features")

    # ── Sample weights (inverse class frequency) ──
    class_counts = np.bincount(y_all, minlength=num_classes)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * num_classes
    sample_weights = class_weights[y_all]

    # ── Cross-validation ──
    logging.info(f"\n  LogReg C={C}, {n_cv_folds}-fold CV on {n_features} features")
    skf = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=42)

    fold_f1s = []
    fold_accs = []
    t0 = time.time()

    for fold_i, (tr_idx, te_idx) in enumerate(skf.split(X_all, y_all)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_all[tr_idx])
            X_te = scaler.transform(X_all[te_idx])
            sw_tr = sample_weights[tr_idx]

            model = LogisticRegression(
                C=C, solver="saga", max_iter=2000,
                tol=1e-3, n_jobs=8, random_state=42,
            )
            model.fit(X_tr, y_all[tr_idx], sample_weight=sw_tr)

        preds = model.predict(X_te)
        f1 = f1_score(y_all[te_idx], preds, average="macro")
        acc = (preds == y_all[te_idx]).mean()
        fold_f1s.append(f1)
        fold_accs.append(acc)
        logging.info(f"    Fold {fold_i+1}/{n_cv_folds}: F1={f1:.4f}  acc={acc:.4f}")

    elapsed = time.time() - t0
    mean_f1 = np.mean(fold_f1s)
    std_f1 = np.std(fold_f1s)
    mean_acc = np.mean(fold_accs)
    logging.info(f"\n  CV result: F1={mean_f1:.4f}±{std_f1:.4f}, "
                 f"acc={mean_acc:.4f} ({elapsed:.1f}s)")

    # ── Final model on all data ──
    logging.info(f"\n  Retraining on full data ({n_samples} samples)...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scaler = StandardScaler()
        X_tr_full = scaler.fit_transform(X_all)
        X_te_full = scaler.transform(X_test)

        final_model = LogisticRegression(
            C=C, solver="saga", max_iter=2000,
            tol=1e-3, n_jobs=8, random_state=42,
        )
        final_model.fit(X_tr_full, y_all, sample_weight=sample_weights)

    test_preds = final_model.predict(X_te_full)

    # ── Save submission ──
    results_path = "results"
    os.makedirs(results_path, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    mpc = config.get("max_per_class", 0)
    output_file = os.path.join(
        results_path,
        f"sub_logreg_C{C}_mpc{mpc}_{timestamp}.csv"
    )
    df = pd.DataFrame({
        'imgname': test_filenames,
        'label': [int(p) for p in test_preds]
    })
    df.to_csv(output_file, index=False)
    logging.info(f"\n  Submission saved: {output_file}")
    logging.info(f"  ✅ train_logreg_morph complete. CV F1={mean_f1:.4f}")


def progressive_finetune(config):
    """
    Progressive fine-tuning: loads a checkpoint and runs a series of training
    stages with increasingly aggressive augmentations. After each stage,
    generates a TTA submission CSV and auto-submits to Kaggle.

    Each stage:
      1. Rebuilds dataloaders with the stage's augmentations_version
      2. Creates a fresh optimizer + CosineAnnealingLR scheduler
      3. Trains for nepochs (default 10)
      4. Saves stage checkpoint
      5. Runs TTA inference (5 views) on the test set → CSV
      6. Submits the CSV to Kaggle

    Config keys (under root):
      progressive_stages: list of {augmentations_version, nepochs, lr (optional)}
      kaggle_competition: str (default "3-md-4040-2026-challenge")
      dry_run: bool (default false)

    Usage:
        python -m torchtmpl.main config_progressive_resnext50_ar.yaml progressive_finetune
    """
    import time
    import copy

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    logging.info(f"Device: {device}")

    # ── Kaggle API ──
    competition = config.get("kaggle_competition", "3-md-4040-2026-challenge")
    dry_run = config.get("dry_run", False)
    kaggle_api = None
    if not dry_run:
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            kaggle_api = KaggleApi()
            kaggle_api.authenticate()
            logging.info(f"Kaggle API authenticated. Competition: {competition}")
        except Exception as e:
            logging.error(f"Kaggle API init failed: {e}. Falling back to dry_run.")
            dry_run = True

    # ── Wandb ──
    if "wandb" in config.get("logging", {}):
        wandb_config = config["logging"]["wandb"]
        wandb.init(project=wandb_config["project"], entity=wandb_config["entity"])
        wandb_log = wandb.log
        wandb_log(config)
    else:
        wandb_log = None

    # ── Build model + load checkpoint ──
    data_config = config["data"]
    # We need a first pass to get input_size and num_classes
    logging.info("= Building initial dataloaders (to get metadata)")
    train_loader, valid_loader, input_size, num_classes = data.get_dataloaders(data_config, use_cuda)

    logging.info("= Building model")
    model_config = config["model"]
    model = models.build_model(model_config, input_size, num_classes)

    checkpoint_path = model_config["model_path"]
    logging.info(f"= Loading checkpoint from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.to(device)
    logging.info("  Checkpoint loaded successfully.")

    # ── Loss ──
    loss_fn = optim.get_loss(config["loss"], device=device)

    # ── Mixup/CutMix ──
    mixup_fn = None
    mixup_loss_fn = None
    if "mixup" in config:
        mixup_fn = optim.get_mixup_fn(config["mixup"], num_classes)
        mixup_loss_fn = optim.SoftTargetCrossEntropy()
        logging.info(f"  Mixup enabled (alpha={config['mixup'].get('mixup_alpha', 0.8)})")

    # ── Logging / logdir ──
    logging_config = config.get("logging", {})
    logdir_base = logging_config.get("logdir", "./logs")
    logname = model_config.get("class", "ProgressiveFT")
    logdir = utils.generate_unique_logpath(logdir_base, logname + "_progressive")
    os.makedirs(logdir, exist_ok=True)
    logdir = pathlib.Path(logdir)
    logging.info(f"Logging into {logdir}")

    # Save config
    with open(logdir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # ── Test dataset / loader for TTA inference ──
    test_path = data_config.get("test_path")
    if not test_path:
        raise ValueError("data.test_path is required for progressive_finetune (Kaggle submission)")

    resize = data_config.get("resize", 224)
    val_transform = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.879857, 0.879857, 0.879857], std=[0.190819, 0.190819, 0.190819])
    ])
    test_dataset = InferenceDataset(root_dir=test_path, transform=val_transform)
    test_loader_inference = DataLoader(
        test_dataset, batch_size=data_config["batch_size"],
        shuffle=False, num_workers=data_config.get("num_workers", 4)
    )
    logging.info(f"  Test dataset: {len(test_dataset)} images from {test_path}")

    # ── Progressive stages ──
    stages = config.get("progressive_stages", [])
    if not stages:
        logging.error("No 'progressive_stages' defined in config. Nothing to do.")
        return

    logging.info(f"\n{'='*60}")
    logging.info(f"  PROGRESSIVE FINE-TUNING: {len(stages)} stages")
    logging.info(f"{'='*60}")
    for i, st in enumerate(stages):
        logging.info(f"  Stage {i}: aug={st['augmentations_version']}, "
                     f"epochs={st.get('nepochs', 10)}, lr={st.get('lr', config['optim']['params']['lr'])}")
    logging.info(f"{'='*60}\n")

    results_path = "results"
    os.makedirs(results_path, exist_ok=True)

    # Model checkpoint tracker (across all stages, best overall)
    model_checkpoint = utils.ModelCheckpoint(
        model, str(logdir / "best_model.pt"), min_is_best=False
    )

    global_epoch = 0  # running epoch counter for wandb/logging

    for stage_idx, stage_config in enumerate(stages):
        aug_version = stage_config["augmentations_version"]
        stage_epochs = stage_config.get("nepochs", 10)
        stage_lr = stage_config.get("lr", config["optim"]["params"]["lr"])
        stage_wd = stage_config.get("weight_decay", config["optim"]["params"].get("weight_decay", 0.01))

        logging.info(f"\n{'='*60}")
        logging.info(f"  STAGE {stage_idx}/{len(stages)-1}: {aug_version}")
        logging.info(f"  epochs={stage_epochs}, lr={stage_lr}, wd={stage_wd}")
        logging.info(f"{'='*60}")

        # ── Rebuild dataloaders with this stage's augmentations ──
        stage_data_config = copy.deepcopy(data_config)
        for source_name in stage_data_config["sources"]:
            stage_data_config["sources"][source_name]["augmentations_version"] = aug_version

        train_loader, valid_loader, _, _ = data.get_dataloaders(stage_data_config, use_cuda)
        logging.info(f"  Dataloaders rebuilt with augmentations_version={aug_version}")

        # ── Fresh optimizer + scheduler for this stage ──
        params_to_update = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params_to_update, lr=stage_lr, weight_decay=stage_wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=stage_epochs, eta_min=stage_lr * 0.01
        )

        # ── Training loop ──
        history_stage = {"train_loss": [], "valid_loss": [], "train_f1": [], "valid_f1": []}

        for e in tqdm(range(stage_epochs), desc=f"Stage {stage_idx} ({aug_version})"):
            train_loss, train_f1 = utils.train(
                model, train_loader, loss_fn, optimizer, device,
                mixup_fn=mixup_fn, mixup_loss_fn=mixup_loss_fn
            )
            scheduler.step()

            if valid_loader is not None:
                test_loss, metrics = utils.test(model, valid_loader, loss_fn, device)
                updated = model_checkpoint.update(metrics["f1"])

                logging.info(
                    "[S%d E%d/%d] Train: %.3f | Val: %.3f | F1: %.4f | F1_TTA: %.4f %s"
                    % (stage_idx, e+1, stage_epochs, train_loss, test_loss,
                       metrics["f1"], metrics.get("f1_tta", 0.0),
                       "[>> BEST <<]" if updated else "")
                )

                history_stage["train_loss"].append(train_loss)
                history_stage["valid_loss"].append(test_loss)
                history_stage["train_f1"].append(train_f1)
                history_stage["valid_f1"].append(metrics["f1"])

                if wandb_log:
                    wandb_log({
                        "stage": stage_idx,
                        "global_epoch": global_epoch,
                        "train_loss": train_loss,
                        "val_loss": test_loss,
                        "val_f1": metrics["f1"],
                        "val_f1_tta": metrics.get("f1_tta", 0.0),
                        "lr": optimizer.param_groups[0]["lr"],
                        "aug_version": aug_version,
                    })
            else:
                logging.info(
                    "[S%d E%d/%d] Train: %.3f | F1: %.3f"
                    % (stage_idx, e+1, stage_epochs, train_loss, train_f1)
                )
                history_stage["train_loss"].append(train_loss)
                history_stage["train_f1"].append(train_f1)

                if wandb_log:
                    wandb_log({
                        "stage": stage_idx,
                        "global_epoch": global_epoch,
                        "train_loss": train_loss,
                        "train_f1": train_f1,
                        "lr": optimizer.param_groups[0]["lr"],
                        "aug_version": aug_version,
                    })

            global_epoch += 1

        # ── Save stage checkpoint ──
        stage_ckpt_path = logdir / f"stage{stage_idx}_final.pt"
        torch.save(model.state_dict(), stage_ckpt_path)
        logging.info(f"  Stage {stage_idx} checkpoint saved: {stage_ckpt_path}")

        # ── TTA Inference + Kaggle submission ──
        logging.info(f"  Running TTA inference for stage {stage_idx}...")
        model.eval()

        test_filenames = []
        test_predictions = []

        with torch.no_grad():
            for images, filenames in tqdm(test_loader_inference, desc=f"TTA S{stage_idx}"):
                images = images.to(device)

                # 5-view TTA: Original, Flip H, Flip V, Rot90, Rot270
                prob = F.softmax(model(images), dim=1)
                prob += F.softmax(model(torch.flip(images, dims=[3])), dim=1)
                prob += F.softmax(model(torch.flip(images, dims=[2])), dim=1)
                prob += F.softmax(model(torch.rot90(images, k=1, dims=[2, 3])), dim=1)
                prob += F.softmax(model(torch.rot90(images, k=3, dims=[2, 3])), dim=1)
                avg_probs = prob / 5.0

                _, preds = torch.max(avg_probs, 1)
                test_filenames.extend(filenames)
                test_predictions.extend(preds.cpu().tolist())

        # Save CSV
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        val_f1_str = f"{history_stage['valid_f1'][-1]:.4f}" if history_stage["valid_f1"] else "NA"
        csv_name = f"sub_progressive_s{stage_idx}_{aug_version}_{timestamp}.csv"
        csv_path = os.path.join(results_path, csv_name)

        results_df = pd.DataFrame({
            'imgname': test_filenames,
            'label': test_predictions
        })
        results_df.to_csv(csv_path, index=False)
        logging.info(f"  Submission CSV saved: {csv_path}")

        # Submit to Kaggle
        kaggle_msg = (
            f"progressive_ft stage{stage_idx}/{len(stages)-1} "
            f"| aug={aug_version} | {stage_epochs}ep | val_F1={val_f1_str}"
        )
        if not dry_run and kaggle_api is not None:
            try:
                kaggle_api.competition_submit(
                    file_name=csv_path,
                    message=kaggle_msg,
                    competition=competition,
                    quiet=True
                )
                logging.info(f"  ✅ Submitted to Kaggle: {kaggle_msg}")
                time.sleep(5)  # rate-limit safety
            except Exception as e:
                logging.warning(f"  ⚠ Kaggle submit failed: {e}")
        else:
            logging.info(f"  [DRY RUN] Would submit: {kaggle_msg}")

        # Back to train mode for next stage
        model.train()

    # ── Final summary ──
    logging.info(f"\n{'='*60}")
    logging.info(f"  PROGRESSIVE FINE-TUNING COMPLETE")
    logging.info(f"  {len(stages)} stages, {global_epoch} total epochs")
    logging.info(f"  Logdir: {logdir}")
    logging.info(f"  Best model (by val F1): {logdir / 'best_model.pt'}")
    logging.info(f"{'='*60}")


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    if len(sys.argv) < 3:
        logging.error(f"Usage : {sys.argv[0]} config.yaml <train|test|generate_csv>")
        logging.error(f"        {sys.argv[0]} config1.yaml config2.yaml [... configN.yaml] ensemble_csv")
        logging.error(f"        {sys.argv[0]} config1.yaml config2.yaml [... configN.yaml] ensemble_csv_stacking")
        logging.error(f"        {sys.argv[0]} dummy.yaml retrain_stacking_from_cache")
        logging.error(f"        {sys.argv[0]} config1.yaml config2.yaml [... configN.yaml] ensemble_csv_meta_learner")
        logging.error(f"        {sys.argv[0]} config.yaml retrain_meta_learner_from_cache")
        logging.error(f"        {sys.argv[0]} config.yaml search_meta_learner_from_cache")
        logging.error(f"        {sys.argv[0]} config.yaml weighted_ensemble_submit")
        logging.error(f"        {sys.argv[0]} config.yaml progressive_finetune")
        sys.exit(-1)

    command = sys.argv[-1]
    config_files = sys.argv[1:-1]

    if command in ("ensemble_csv", "ensemble_csv_meta_learner", "ensemble_csv_stacking"):
        if command in ("ensemble_csv", "ensemble_csv_stacking") and len(config_files) < 2:
            logging.error(f"{command} nécessite au moins 2 fichiers de configuration.")
            sys.exit(-1)
        configs = []
        for cf in config_files:
            logging.info(f"Loading {cf}")
            configs.append(yaml.safe_load(open(cf, "r")))
        if command == "ensemble_csv":
            ensemble_csv(configs)
        elif command == "ensemble_csv_stacking":
            ensemble_csv_stacking(configs)
        else:
            ensemble_csv_meta_learner(configs)
    else:
        if len(config_files) != 1:
            logging.error(f"Usage : {sys.argv[0]} config.yaml <train|test|generate_csv>")
            sys.exit(-1)
        logging.info("Loading {}".format(config_files[0]))
        config = yaml.safe_load(open(config_files[0], "r"))
        eval(f"{command}(config)")

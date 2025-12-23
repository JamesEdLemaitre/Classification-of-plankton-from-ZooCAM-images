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


def test(config):
    raise NotImplementedError


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    if len(sys.argv) != 3:
        logging.error(f"Usage : {sys.argv[0]} config.yaml <train|test>")
        sys.exit(-1)

    logging.info("Loading {}".format(sys.argv[1]))
    config = yaml.safe_load(open(sys.argv[1], "r"))

    command = sys.argv[2]
    eval(f"{command}(config)")

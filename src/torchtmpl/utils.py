# coding: utf-8

# Standard imports
import os

# External imports
import torch
import torch.nn
from tqdm import tqdm

from sklearn.metrics import f1_score
import numpy as np

import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server/cluster
import matplotlib.pyplot as plt

try:
    from timm.data import Mixup
except ImportError:
    Mixup = None


def generate_unique_logpath(logdir, raw_run_name):
    """
    Generate a unique directory name
    Argument:
        logdir: the prefix directory
        raw_run_name(str): the base name
    Returns:
        log_path: a non-existent path like logdir/raw_run_name_xxxx
                  where xxxx is an int
    """
    i = 0
    while True:
        run_name = raw_run_name + "_" + str(i)
        log_path = os.path.join(logdir, run_name)
        if not os.path.isdir(log_path):
            return log_path
        i = i + 1


class ModelCheckpoint(object):
    """
    Early stopping callback
    """

    def __init__(
        self,
        model: torch.nn.Module,
        savepath,
        min_is_best: bool = True,
    ) -> None:
        self.model = model
        self.savepath = savepath
        self.best_score = None
        if min_is_best:
            self.is_better = self.lower_is_better
        else:
            self.is_better = self.higher_is_better

    def lower_is_better(self, score):
        return self.best_score is None or score < self.best_score

    def higher_is_better(self, score):
        return self.best_score is None or score > self.best_score

    def update(self, score):
        if self.is_better(score):
            torch.save(self.model.state_dict(), self.savepath)
            self.best_score = score
            return True
        return False


def train(model, loader, f_loss, optimizer, device, dynamic_display=True,
          mixup_fn=None, mixup_loss_fn=None):
    """
    Train a model for one epoch, iterating over the loader
    using the f_loss to compute the loss and the optimizer
    to update the parameters of the model.
    Arguments :
    model          -- A torch.nn.Module object
    loader         -- A torch.utils.data.DataLoader
    f_loss         -- The loss function, i.e. a loss Module
    optimizer      -- A torch.optim.Optimzer object
    device         -- A torch.device
    mixup_fn       -- (optional) timm.data.Mixup instance for CutMix/Mixup
    mixup_loss_fn  -- (optional) loss compatible with soft targets (used when mixup is active)
    Returns :
    (avg_loss, train_f1) — train_f1 is macro F1 on original hard targets
    """

    # We enter train mode.
    # This is important for layers such as dropout, batchnorm, ...
    model.train()

    total_loss = 0
    num_samples = 0
    all_preds = []
    all_targets = []
    with tqdm(loader, desc="  Train", leave=False) as pbar:
        for inputs, targets in pbar:

            inputs, targets = inputs.to(device), targets.to(device)

            # Keep original hard targets for F1 computation
            hard_targets = targets

            # Apply Mixup/CutMix at the batch level if configured
            if mixup_fn is not None:
                inputs, targets_mixed = mixup_fn(inputs, targets)
                outputs = model(inputs)
                loss = mixup_loss_fn(outputs, targets_mixed)
            else:
                outputs = model(inputs)
                loss = f_loss(outputs, targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update the metrics
            # We here consider the loss is batch normalized
            total_loss += inputs.shape[0] * loss.item()
            num_samples += inputs.shape[0]

            # Collect predictions vs original hard targets for F1
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_targets.extend(hard_targets.cpu().numpy())

            pbar.set_description(f"  Train | Loss: {total_loss/num_samples:.4f}")

    avg_loss = total_loss / num_samples
    train_f1 = f1_score(all_targets, all_preds, average='macro')
    return avg_loss, train_f1


def test(model, loader, f_loss, device):
    model.eval()  

    total_loss = 0
    num_samples = 0
    all_preds_std = [] # std = standard

    all_preds_tta = []

    all_targets = []

    with torch.no_grad(), tqdm(loader, desc="  Validation (Std + TTA)", leave=False) as pbar:
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.shape[0]

            outputs_std = model(inputs)
            loss = f_loss(outputs_std, targets)

            total_loss += batch_size * loss.item()
            num_samples += batch_size
 
            _, preds = torch.max(outputs_std, 1)
            all_preds_std.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            ## TTA:
            probs_sum = F.softmax(outputs_std, dim=1)

            for k in [1, 2, 3]: # make rotations
                inputs_rot = torch.rot90(inputs, k=k, dims=[2, 3])
                outputs_rot = model(inputs_rot)
                probs_sum += F.softmax(outputs_rot, dim=1)

            # make horizontal flip
            inputs_flip = torch.flip(inputs, dims=[3])
            outputs_flip = model(inputs_flip)
            probs_sum += F.softmax(outputs_flip, dim=1)

            # tta:
            _, preds_tta = torch.max(probs_sum, 1) # standard + std
            all_preds_tta.extend(preds_tta.cpu().numpy())


            pbar.set_description(f"  Valid | Loss: {total_loss/num_samples:.4f}")

    avg_loss = total_loss / num_samples
    
    # standard score
    f1_std = f1_score(all_targets, all_preds_std, average='macro')

    # tta score
    f1_tta = f1_score(all_targets, all_preds_tta, average='macro')

    metrics = {
        "f1": f1_std,
        "f1_tta": f1_tta
    }

    return avg_loss, metrics


def plot_training_curves(history, logdir):
    """
    Save / update loss and F1 curves (train vs. validation) as PNG files.
    Called at the end of every epoch so the plots stay up-to-date.

    Args:
        history (dict): keys → lists of floats, expected keys:
            'train_loss', 'valid_loss', 'train_f1', 'valid_f1', 'valid_f1_tta'
        logdir (str or Path): directory where PNGs are written.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    # ── Loss curve ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history["train_loss"], label="Train loss", linewidth=1.5)
    if history.get("valid_loss"):
        ax.plot(epochs, history["valid_loss"], label="Valid loss", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss — Train vs. Validation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(str(logdir), "curve_loss.png"), dpi=150)
    plt.close(fig)

    # ── F1 curve ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history["train_f1"], label="Train F1", linewidth=1.5)
    if history.get("valid_f1"):
        ax.plot(epochs, history["valid_f1"], label="Valid F1", linewidth=1.5)
    if history.get("valid_f1_tta"):
        ax.plot(epochs, history["valid_f1_tta"], label="Valid F1 (TTA)",
                linewidth=1.5, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Macro F1-score")
    ax.set_title("F1-score — Train vs. Validation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(str(logdir), "curve_f1.png"), dpi=150)
    plt.close(fig)

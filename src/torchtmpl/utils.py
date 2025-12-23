# coding: utf-8

# Standard imports
import os

# External imports
import torch
import torch.nn
from tqdm import tqdm
from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
import numpy as np
import torch.nn.functional as F
from sklearn.preprocessing import label_binarize

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


def train(model, loader, f_loss, optimizer, device, dynamic_display=True):
    """
    Train a model for one epoch, iterating over the loader
    using the f_loss to compute the loss and the optimizer
    to update the parameters of the model.
    Arguments :
    model     -- A torch.nn.Module object
    loader    -- A torch.utils.data.DataLoader
    f_loss    -- The loss function, i.e. a loss Module
    optimizer -- A torch.optim.Optimzer object
    device    -- A torch.device
    Returns :
    The averaged train metrics computed over a sliding window
    """

    # We enter train mode.
    # This is important for layers such as dropout, batchnorm, ...
    model.train()

    total_loss = 0
    num_samples = 0

    pbar = tqdm(loader, desc="Training", leave=True)
    
    for i, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)

        # Compute the forward propagation
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
        pbar.set_description(f"Train loss : {total_loss/num_samples:.2f}")

    return total_loss / num_samples


def test(model, loader, f_loss, device):
    model.eval()

    total_loss = 0
    num_samples = 0
    all_targets = []
    all_preds = []
    all_probs = [] # Pour stocker les probabilités (softmax)

    pbar = tqdm(loader, desc="Validation", unit="batch", leave=False, position=1)

    with torch.no_grad():
        for (inputs, targets) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward
            outputs = model(inputs)
            loss = f_loss(outputs, targets)

            # Metrics basics
            total_loss += inputs.shape[0] * loss.item()
            num_samples += inputs.shape[0]

            # Prédictions (Classe la plus probable)
            _, preds = torch.max(outputs, 1)

            # Probabilités (Pour AUC-ROC)
            # On transforme les logits bruts en probabilités [0, 1]
            probs = F.softmax(outputs, dim=1)

            # Stockage
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calcul moyen de la loss
    # (Utiliser num_samples est plus sûr que len(loader.dataset) à cause des subsets)
    avg_loss = total_loss / num_samples

    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy_score(all_targets, all_preds),
        'f1': f1_score(all_targets, all_preds, average='macro'),
        'precision': precision_score(all_targets, all_preds, average='macro', zero_division=0),
        'recall': recall_score(all_targets, all_preds, average='macro', zero_division=0),
    }

    # --- Gestion Robuste de l'AUC-ROC ---
    try:
        # On détermine le nombre de classes réel attendu par le modèle
        # (et pas juste celles présentes dans le batch)
        num_classes_model = np.array(all_probs).shape[1]
        
        # Binariser les targets pour matcher la forme des probabilités (N, Num_Classes)
        targets_onehot = label_binarize(all_targets, classes=range(num_classes_model))
        
        # Cas spécial : Si on n'a que 2 classes (Binary Classification)
        if num_classes_model == 2:
            # Pour le binaire, roc_auc_score attend souvent array 1D pour y_score
            # On prend la probabilité de la classe positive (index 1)
            metrics['auc_roc'] = roc_auc_score(all_targets, np.array(all_probs)[:, 1])
        else:
            # Cas Multi-classe (One vs Rest)
            metrics['auc_roc'] = roc_auc_score(
                targets_onehot,
                np.array(all_probs),
                multi_class='ovr',
                average='macro'
            )
            
    except Exception as e:
        # En cas de problème (ex: dataset trop petit, manque de classes), on ne crashe pas
        # print(f"Warning: Could not compute AUC-ROC: {e}") 
        metrics['auc_roc'] = 0.0

    return metrics


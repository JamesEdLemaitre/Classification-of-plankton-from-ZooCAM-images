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
            print(f"Best model saved at: {self.savepath}")
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
    pbar = tqdm(loader, desc="Training", leave=False)
    
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
    """
    Test a model over the loader
    using the f_loss as metrics
    Arguments :
    model     -- A torch.nn.Module object
    loader    -- A torch.utils.data.DataLoader
    f_loss    -- The loss function, i.e. a loss Module
    device    -- A torch.device
    Returns :
    """

    # We enter eval mode.
    # This is important for layers such as dropout, batchnorm, ...
    model.eval()

    total_loss = 0
    num_samples = 0
    all_targets = []
    all_preds = []
    all_outputs = []

    pbar = tqdm(
        loader,
        desc="Validation",
        unit="batch",
        leave=False,  # ← Important : ne pas laisser la barre après
        position=1     # ← Position 1 pour être sous la barre epoch
    )

    with torch.no_grad():
        for (inputs, targets) in pbar:

            inputs, targets = inputs.to(device), targets.to(device)

            # Compute the forward propagation
            outputs = model(inputs)

            loss = f_loss(outputs, targets)

            # Update the metrics
            # We here consider the loss is batch normalized
            total_loss += inputs.shape[0] * loss.item()
            num_samples += inputs.shape[0]

            _, preds = torch.max(outputs, 1)

            # Stocke les résultats pour les métriques globales
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())  # Pour AUC-ROC

    # Calcul des métriques
    metrics = {
        'loss': total_loss / len(loader.dataset),
        'accuracy': accuracy_score(all_targets, all_preds),
        'f1': f1_score(all_targets, all_preds, average='macro'),
        'precision': precision_score(all_targets, all_preds, average='macro'),
        'recall': recall_score(all_targets, all_preds, average='macro'),
    }

    if len(set(all_targets)) > 2:  # Multi-classe
        from sklearn.preprocessing import label_binarize
        targets_onehot = label_binarize(all_targets, classes=sorted(set(all_targets)))
        metrics['auc_roc'] = roc_auc_score(
            targets_onehot,
            np.array(all_outputs),
            multi_class='ovr'
        )

    return total_loss / num_samples, metrics

"""
rice_model.py

ResNet50 + training engine for the cross-dataset study.

Freeze modes:
    "full"    - everything trainable        (full fine-tuning)
    "encoder" - backbone trainable, fc frozen   (your "encoder only" idea)
    "head"    - backbone frozen, fc trainable   (linear probe, the standard baseline)
"""

import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score)
from torchvision.models import ResNet50_Weights, resnet50

from rice_data import CLASSES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------

def build_resnet50(num_classes=len(CLASSES), pretrained=True):
    m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m.to(DEVICE)


def set_trainable(model, mode):
    if mode == "full":
        for p in model.parameters():
            p.requires_grad = True
    elif mode == "head":
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True
    elif mode == "encoder":
        for p in model.parameters():
            p.requires_grad = True
        for p in model.fc.parameters():
            p.requires_grad = False
    else:
        raise ValueError(f"unknown mode: {mode}")
    return model


def n_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        with torch.autocast("cuda", enabled=torch.cuda.is_available()):
            out = model(x)
        ps.append(out.float().argmax(1).cpu().numpy())
        ys.append(y.numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)

    return {
        "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
        "bal_acc": float(balanced_accuracy_score(y, p)),
        "acc": float(accuracy_score(y, p)),
        "per_class_f1": {c: float(v) for c, v in zip(
            CLASSES, f1_score(y, p, average=None, labels=range(len(CLASSES)), zero_division=0))},
        "confusion": confusion_matrix(y, p, labels=range(len(CLASSES))).tolist(),
        "n": int(len(y)),
    }


# ----------------------------------------------------------------------------
# train
# ----------------------------------------------------------------------------

def fit(model, train_loader, val_loader, epochs=30, lr=1e-3, wd=1e-4,
        weights=None, patience=7, verbose=False):
    """
    Trains, early-stops on val macro-F1, restores the best weights.
    Only parameters with requires_grad=True are optimized.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise SystemExit("fit() called but nothing is trainable")

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(
        weight=None if weights is None else weights.to(DEVICE), label_smoothing=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    best_f1, best_state, bad = -1.0, copy.deepcopy(model.state_dict()), 0

    for ep in range(epochs):
        model.train()
        tot = 0.0
        for x, y in train_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=torch.cuda.is_available()):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * x.size(0)
        sched.step()

        vm = evaluate(model, val_loader)
        if verbose:
            print(f"    ep {ep+1:02d}/{epochs} loss={tot/len(train_loader.dataset):.4f} "
                  f"val_f1={vm['macro_f1']:.4f}")

        if vm["macro_f1"] > best_f1:
            best_f1, bad = vm["macro_f1"], 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"    early stop at ep {ep+1}")
                break

    model.load_state_dict(best_state)
    return model, best_f1


# ----------------------------------------------------------------------------
# AdaBN  - free domain adaptation, no labels needed
# ----------------------------------------------------------------------------

@torch.no_grad()
def adabn(model, target_loader, max_batches=100):
    """
    Recompute BatchNorm running stats on the TARGET images.
    Uses no labels. Often recovers a chunk of the domain gap for free.
    """
    model = copy.deepcopy(model)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None          # cumulative moving average
    model.train()
    for i, (x, _) in enumerate(target_loader):
        model(x.to(DEVICE, non_blocking=True))
        if i + 1 >= max_batches:
            break
    model.eval()
    return model

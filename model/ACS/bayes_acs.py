#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Attention-based Compound-specific (ACS) Neural Network Model
=============================================================

This script implements a multi-task neural network with:
- Shared MLP backbone for feature extraction
- Task-specific heads for predictions
- Optuna-based hyperparameter optimization
- Support for regression and classification tasks

Architecture:
    Input -> Backbone (MLP) -> Task Heads -> Output

Features:
- Automatic detection of task type (regression/classification)
- Support for multi-task learning
- Class weighting for imbalanced classification
- Early stopping with validation loss
- Scaffold-based train/valid/test splitting

Usage:
    # Regression task
    python bayes_acs.py --dataset esol --descriptor esol_nsc500_sl20.pkl --task-type regression

    # Classification task
    python bayes_acs.py --dataset bace --descriptor bace_nsc500_sl20.pkl --task-type classification

    # Multi-task classification
    python bayes_acs.py --dataset tox21 --descriptor tox21_nsc500_sl20.pkl --task-type classification
"""

import os
import sys
import json
import copy
import pickle
import random
import warnings
import argparse
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, average_precision_score
)

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


# =============================================================================
# Model Definitions
# =============================================================================

class SingleTaskDataset(Dataset):
    """Dataset for single-task or multi-task learning."""
    
    def __init__(self, X: np.ndarray, Y: np.ndarray, mask: Optional[np.ndarray] = None):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(np.nan_to_num(Y, nan=0.0)).float()
        if mask is None:
            self.mask = torch.ones_like(self.Y)
        else:
            self.mask = torch.from_numpy(mask.astype(np.float32))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.mask[idx]


class MLPBackbone(nn.Module):
    """Shared backbone network for feature extraction."""
    
    def __init__(self, in_dim: int, hidden_dims: List[int], dropout: float = 0.3):
        super().__init__()
        layers = []
        d_in = in_dim
        for d_out in hidden_dims:
            layers.extend([nn.Linear(d_in, d_out), nn.ReLU(), nn.Dropout(dropout)])
            d_in = d_out
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TaskHead(nn.Module):
    """Task-specific prediction head."""
    
    def __init__(self, in_dim: int, hidden_dims: List[int], dropout: float = 0.3):
        super().__init__()
        layers = []
        d_in = in_dim
        for d_out in hidden_dims:
            layers.extend([nn.Linear(d_in, d_out), nn.ReLU(), nn.Dropout(dropout)])
            d_in = d_out
        layers.append(nn.Linear(d_in, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, h):
        return self.net(h).squeeze(-1)


class ACSModel(nn.Module):
    """
    Attention-based Compound-specific Model.
    
    Architecture: Input -> Backbone -> Task Heads -> Output
    """
    
    def __init__(self, in_dim: int, num_tasks: int,
                 backbone_dims: List[int], head_dims: List[int], dropout: float):
        super().__init__()
        self.backbone = MLPBackbone(in_dim, backbone_dims, dropout)
        self.heads = nn.ModuleList([
            TaskHead(backbone_dims[-1], head_dims, dropout) for _ in range(num_tasks)
        ])

    def forward(self, x):
        h = self.backbone(x)
        logits = [head(h) for head in self.heads]
        return logits, h


# =============================================================================
# Loss Functions
# =============================================================================

def regression_loss(preds, targets, mask=None):
    """MSE loss for regression."""
    loss = F.mse_loss(preds, targets, reduction="none")
    if mask is not None:
        loss = loss * mask
        return loss.sum() / mask.sum().clamp(min=1.0)
    return loss.mean()


def bce_with_logits_weighted(logits, targets, mask, pos_weight=None):
    """BCE loss with class weighting for classification."""
    if pos_weight is not None:
        loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none",
            pos_weight=torch.tensor(pos_weight, device=logits.device)
        )
    else:
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    loss = loss * mask
    return loss.sum() / mask.sum().clamp(min=1.0)


def compute_pos_weights(Y: np.ndarray, mask: np.ndarray) -> List[float]:
    """Compute positive class weights for imbalanced classification."""
    num_tasks = Y.shape[1]
    pos_weights = []
    for t in range(num_tasks):
        y = Y[:, t]
        m = mask[:, t]
        pos = ((y == 1) & (m == 1)).sum()
        neg = ((y == 0) & (m == 1)).sum()
        pw = float(neg) / float(pos) if pos > 0 else 1.0
        pos_weights.append(max(pw, 1.0))
    return pos_weights


# =============================================================================
# Scaffold Splitting
# =============================================================================

def get_ring_count(smiles: str, max_rings: int = 5) -> int:
    """Get ring count for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    n_rings = mol.GetRingInfo().NumRings()
    return min(n_rings, max_rings)


def get_scaffold(smiles: str) -> str:
    """Get Murcko scaffold for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)


def scaffold_split(df_smiles: pd.DataFrame, frac_train: float, frac_valid: float, 
                   frac_test: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform scaffold-based train/valid/test split."""
    rng = random.Random(seed)
    ring_groups = {}
    scaffold_groups = {}

    for i, smi in enumerate(df_smiles["smiles"].values):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        n_rings = get_ring_count(smi)
        scaffold = get_scaffold(smi)
        if scaffold is None:
            continue
        ring_groups.setdefault(n_rings, []).append(i)
        scaffold_groups.setdefault(n_rings, {}).setdefault(scaffold, []).append(i)

    print("Ring count distribution:")
    for n_rings, idxs in sorted(ring_groups.items()):
        label = f">={max(5, n_rings)} rings" if n_rings >= 5 else f"{n_rings} rings"
        print(f"  {label}: {len(idxs)} molecules")

    train_idx, valid_idx, test_idx = [], [], []
    random_split_layers = {}

    for n_rings in sorted(scaffold_groups.keys(), reverse=True):
        scf_dict = scaffold_groups[n_rings]
        scf_list = list(scf_dict.keys())

        if len(scf_list) <= 3:
            idxs = ring_groups[n_rings]
            random_split_layers[n_rings] = idxs
            rng.shuffle(idxs)
            n_total = len(idxs)
            n_test = round(frac_test * n_total)
            n_valid = round(frac_valid * n_total)
            test_idx.extend(idxs[:n_test])
            valid_idx.extend(idxs[n_test:n_test + n_valid])
            train_idx.extend(idxs[n_test + n_valid:])
            continue

        rng.shuffle(scf_list)
        n_total = len(scf_list)
        n_test = round(frac_test * n_total)
        n_valid = round(frac_valid * n_total)

        test_scf = scf_list[:n_test]
        valid_scf = scf_list[n_test:n_test + n_valid]
        train_scf = scf_list[n_test + n_valid:]

        train_mols, valid_mols, test_mols = [], [], []
        for s in train_scf:
            train_mols.extend(scf_dict[s])
        for s in valid_scf:
            valid_mols.extend(scf_dict[s])
        for s in test_scf:
            test_mols.extend(scf_dict[s])

        print(f"Layer {n_rings}: {len(train_scf)} scaffolds ({len(train_mols)} mols) train, "
              f"{len(valid_scf)} ({len(valid_mols)}) valid, {len(test_scf)} ({len(test_mols)}) test")

        train_idx.extend(train_mols)
        valid_idx.extend(valid_mols)
        test_idx.extend(test_mols)

    # Adjust ratios if deviation > 2%
    target_frac = np.array([frac_train, frac_valid, frac_test])
    current_counts = np.array([len(train_idx), len(valid_idx), len(test_idx)])
    current_frac = current_counts / current_counts.sum()

    deviation = target_frac - current_frac
    if np.any(np.abs(deviation) > 0.02):
        print("\nAdjusting using random-split layers to fix ratio...")
        all_random_idxs = []
        for idxs in random_split_layers.values():
            all_random_idxs.extend(idxs)
        rng.shuffle(all_random_idxs)

        total = len(train_idx) + len(valid_idx) + len(test_idx)
        target_counts = (target_frac * total).astype(int)

        n_train, n_valid, n_test = len(train_idx), len(valid_idx), len(test_idx)

        while n_train > target_counts[0] and all_random_idxs:
            i = all_random_idxs.pop()
            if i in train_idx:
                train_idx.remove(i)
                if n_valid < target_counts[1]:
                    valid_idx.append(i)
                    n_valid += 1
                else:
                    test_idx.append(i)
                    n_test += 1
                n_train -= 1

        while n_train < target_counts[0] and all_random_idxs:
            i = all_random_idxs.pop()
            if i not in train_idx and i not in valid_idx and i not in test_idx:
                train_idx.append(i)
                n_train += 1

    total = len(train_idx) + len(valid_idx) + len(test_idx)
    print(f"\nSplit: Train {len(train_idx)/total:.2%}, Valid {len(valid_idx)/total:.2%}, Test {len(test_idx)/total:.2%}")

    return np.array(train_idx), np.array(valid_idx), np.array(test_idx)


# =============================================================================
# Training Functions
# =============================================================================

def train_model(
    model: ACSModel,
    loaders: Dict[str, DataLoader],
    pos_weights: Optional[List[float]],
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    task_names: List[str],
    task_type: str,
    print_every: int = 10,
    save_dir: str = None,
    trial: optuna.Trial = None
) -> Tuple[List[Dict], List[int], List[float]]:
    """Train ACS model with early stopping."""
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    num_tasks = len(model.heads)
    best_val_loss = [float("inf")] * num_tasks
    best_ckpts = [None] * num_tasks
    best_epochs = [-1] * num_tasks
    no_improve_epochs = 0

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        total_loss, n_batches = 0.0, 0
        for xb, yb, mb in loaders["train"]:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            optimizer.zero_grad()
            logits, _ = model(xb)
            
            loss_all = 0.0
            for t in range(num_tasks):
                if task_type == "classification":
                    pw = pos_weights[t] if pos_weights else None
                    loss_all += bce_with_logits_weighted(logits[t], yb[:, t], mb[:, t], pw)
                else:
                    loss_all += regression_loss(logits[t], yb[:, t], mb[:, t])
            
            loss_all.backward()
            optimizer.step()
            total_loss += loss_all.item()
            n_batches += 1

        # Validation
        model.eval()
        val_losses = [0.0] * num_tasks
        counts = [0.0] * num_tasks
        with torch.no_grad():
            for xb, yb, mb in loaders["val"]:
                xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
                logits, _ = model(xb)
                for t in range(num_tasks):
                    if task_type == "classification":
                        pw = pos_weights[t] if pos_weights else None
                        loss_t = bce_with_logits_weighted(logits[t], yb[:, t], mb[:, t], pw)
                    else:
                        loss_t = regression_loss(logits[t], yb[:, t], mb[:, t])
                    val_losses[t] += loss_t.item()
                    counts[t] += 1

        val_losses = [vl / max(c, 1) for vl, c in zip(val_losses, counts)]
        mean_val_loss = float(np.mean(val_losses))

        # Optuna pruning
        if trial is not None:
            trial.report(-mean_val_loss if task_type == "classification" else mean_val_loss, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned(f"Pruned at epoch {epoch}")

        # Update best checkpoints
        updated_any = False
        for t in range(num_tasks):
            if val_losses[t] < best_val_loss[t]:
                best_val_loss[t], best_epochs[t] = val_losses[t], epoch
                ckpt = {
                    "epoch": epoch,
                    "backbone": copy.deepcopy(model.backbone.state_dict()),
                    "head": copy.deepcopy(model.heads[t].state_dict()),
                    "val_loss": val_losses[t],
                }
                best_ckpts[t] = ckpt
                if save_dir:
                    torch.save(ckpt, os.path.join(save_dir, f"best_{task_names[t]}.pt"))
                updated_any = True

        if not updated_any:
            no_improve_epochs += 1
        else:
            no_improve_epochs = 0

        if epoch % print_every == 0 or epoch == 1:
            avg_train = total_loss / max(n_batches, 1)
            print(f"[Epoch {epoch:03d}] train_loss={avg_train:.4f} | "
                  + " | ".join([f"val[{task_names[t]}]={val_losses[t]:.4f}" for t in range(num_tasks)])
                  + f" | mean_val_loss={mean_val_loss:.4f}")

        if no_improve_epochs >= patience:
            print(f"Early stopping after {no_improve_epochs} epochs without improvement.")
            break

    return best_ckpts, best_epochs, best_val_loss


# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_regression(model: ACSModel, loader: DataLoader, device: torch.device,
                        ckpts: List[Dict], task_names: List[str]) -> pd.DataFrame:
    """Evaluate regression model."""
    rows = []
    base_state = {
        "backbone": copy.deepcopy(model.backbone.state_dict()),
        "heads": [copy.deepcopy(h.state_dict()) for h in model.heads],
    }

    for t, ckpt in enumerate(ckpts):
        name = task_names[t] if t < len(task_names) else f"task_{t}"
        if ckpt is not None:
            model.backbone.load_state_dict(ckpt["backbone"])
            model.heads[t].load_state_dict(ckpt["head"])
        else:
            warnings.warn(f"[{name}] No best checkpoint found")

        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for xb, yb, mb in loader:
                xb = xb.to(device)
                logits, _ = model(xb)
                mask = mb[:, t].cpu().numpy() == 1
                ys.append(yb[:, t].cpu().numpy()[mask])
                ps.append(logits[t].cpu().numpy()[mask])
        
        y = np.concatenate(ys) if ys else np.array([])
        p = np.concatenate(ps) if ps else np.array([])

        if len(y) > 0:
            rmse = math.sqrt(mean_squared_error(y, p))
            mae = mean_absolute_error(y, p)
            r2 = r2_score(y, p)
        else:
            rmse, mae, r2 = np.nan, np.nan, np.nan

        rows.append({"task": name, "rmse": rmse, "mae": mae, "r2": r2})

        # Restore state
        model.backbone.load_state_dict(base_state["backbone"])
        model.heads[t].load_state_dict(base_state["heads"][t])

    return pd.DataFrame(rows)


def evaluate_classification(model: ACSModel, loader: DataLoader, device: torch.device,
                            ckpts: List[Dict], task_names: List[str]) -> pd.DataFrame:
    """Evaluate classification model."""
    rows = []
    base_state = {
        "backbone": copy.deepcopy(model.backbone.state_dict()),
        "heads": [copy.deepcopy(h.state_dict()) for h in model.heads],
    }

    for t, ckpt in enumerate(ckpts):
        name = task_names[t] if t < len(task_names) else f"task_{t}"
        if ckpt is not None:
            model.backbone.load_state_dict(ckpt["backbone"])
            model.heads[t].load_state_dict(ckpt["head"])
        else:
            warnings.warn(f"[{name}] No best checkpoint found")

        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for xb, yb, mb in loader:
                xb = xb.to(device)
                logits, _ = model(xb)
                prob = torch.sigmoid(logits[t]).cpu().numpy()
                mask = mb[:, t].cpu().numpy() == 1
                ys.append(yb[:, t].cpu().numpy()[mask])
                ps.append(prob[mask])

        y = np.concatenate(ys) if ys else np.array([])
        p = np.concatenate(ps) if ps else np.array([])

        if len(np.unique(y)) >= 2:
            roc = roc_auc_score(y, p)
            pr = average_precision_score(y, p)
        else:
            roc, pr = np.nan, np.nan

        rows.append({"task": name, "roc_auc": roc, "pr_auc": pr})

        # Restore state
        model.backbone.load_state_dict(base_state["backbone"])
        model.heads[t].load_state_dict(base_state["heads"][t])

    return pd.DataFrame(rows)


# =============================================================================
# Optuna Objective
# =============================================================================

def create_objective(X_tr, Y_tr, mask_tr, X_va, Y_va, mask_va, task_names, task_type, device, args):
    """Create Optuna objective function."""
    
    input_dim = X_tr.shape[1]
    num_tasks = len(task_names)

    def suggest_dims(trial, name_prefix: str, max_layers: int = 3):
        n_layers = trial.suggest_int(f"{name_prefix}_n", 1, max_layers)
        choices = [128, 256, 512, 768, 1024]
        dims = []
        for i in range(n_layers):
            dims.append(trial.suggest_categorical(f"{name_prefix}_{i}", choices))
        return dims

    def objective(trial: optuna.Trial) -> float:
        backbone_dims = suggest_dims(trial, "backbone_dims", 3)
        head_dims = suggest_dims(trial, "head_dims", 2)
        dropout = trial.suggest_float("dropout", 0.1, 0.6)
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_va_s = scaler.transform(X_va)

        train_loader = DataLoader(SingleTaskDataset(X_tr_s, Y_tr, mask_tr), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(SingleTaskDataset(X_va_s, Y_va, mask_va), batch_size=batch_size, shuffle=False)

        pos_weights = compute_pos_weights(Y_tr, mask_tr) if task_type == "classification" else None

        model = ACSModel(input_dim, num_tasks, backbone_dims, head_dims, dropout)

        ckpts, _, _ = train_model(
            model, {"train": train_loader, "val": val_loader}, pos_weights, device,
            epochs=args.max_epochs_tune, lr=lr, weight_decay=weight_decay,
            patience=args.patience_tune, task_names=task_names, task_type=task_type,
            print_every=args.print_every, trial=trial
        )

        if task_type == "regression":
            metrics_df = evaluate_regression(model, val_loader, device, ckpts, task_names)
            return metrics_df["rmse"].mean()
        else:
            metrics_df = evaluate_classification(model, val_loader, device, ckpts, task_names)
            return metrics_df["roc_auc"].dropna().mean()

    return objective


# =============================================================================
# Task Detection
# =============================================================================

def detect_task_type(labels: np.ndarray) -> str:
    """Auto-detect task type."""
    unique_vals = np.unique(labels[~np.isnan(labels)])
    if set(unique_vals).issubset({0.0, 1.0, 0, 1}):
        return "classification"
    if len(unique_vals) < 10:
        return "classification"
    return "regression"


def infer_label_cols(df: pd.DataFrame) -> List[str]:
    """Infer label columns from DataFrame."""
    drop_like = {"smiles", "SMILES", "drug", "Drug", "name", "Name", "id", "ID"}
    cand = [c for c in df.columns if c not in drop_like]
    return [c for c in cand if pd.api.types.is_numeric_dtype(df[c])]


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='ACS Neural Network with Optuna Optimization')
    
    # Data arguments
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--descriptor', type=str, required=True, help='Descriptor file name')
    parser.add_argument('--task-type', type=str, choices=['auto', 'classification', 'regression'],
                        default='auto', help='Task type (default: auto-detect)')
    
    # Split arguments
    parser.add_argument('--frac-train', type=float, default=0.8, help='Train fraction')
    parser.add_argument('--frac-valid', type=float, default=0.1, help='Validation fraction')
    parser.add_argument('--frac-test', type=float, default=0.1, help='Test fraction')
    parser.add_argument('--seed', type=int, default=5, help='Random seed for scaffold split')
    
    # Training arguments
    parser.add_argument('--max-epochs-tune', type=int, default=100, help='Max epochs for tuning')
    parser.add_argument('--max-epochs-final', type=int, default=200, help='Max epochs for final training')
    parser.add_argument('--patience-tune', type=int, default=20, help='Early stopping patience for tuning')
    parser.add_argument('--patience-final', type=int, default=30, help='Early stopping patience for final')
    parser.add_argument('--print-every', type=int, default=10, help='Print every N epochs')
    
    # Optuna arguments
    parser.add_argument('--n-trials', type=int, default=100, help='Number of Optuna trials')
    parser.add_argument('--n-jobs', type=int, default=1, help='Number of parallel jobs')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device')
    
    # Path arguments
    parser.add_argument('--descriptor-dir', type=str, default='../calculate_descriptors/descriptors')
    parser.add_argument('--data-dir', type=str, default='../data')
    parser.add_argument('--output-dir', type=str, default='acs_results')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set paths
    descriptor_dir = Path(args.descriptor_dir)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load descriptors
    descriptor_path = descriptor_dir / args.descriptor
    print(f"Loading descriptors: {descriptor_path}")
    with open(descriptor_path, "rb") as f:
        X = pickle.load(f)
    X = np.asarray(X, dtype=np.float32)
    print(f"  Shape: {X.shape}")
    
    # Load labels
    label_path = data_dir / f"{args.dataset}.csv"
    print(f"Loading labels: {label_path}")
    df_labels = pd.read_csv(label_path)
    if len(df_labels) != X.shape[0]:
        raise ValueError(f"Sample mismatch: features={X.shape[0]}, labels={len(df_labels)}")
    
    # Get SMILES and labels
    if 'smiles' not in df_labels.columns:
        raise ValueError("Dataset must have 'smiles' column")
    df_smiles = df_labels[['smiles']].copy()
    
    task_names = infer_label_cols(df_labels)
    print(f"  Tasks: {task_names}")
    
    Y = df_labels[task_names].to_numpy(dtype=np.float32)
    mask = (~np.isnan(Y)).astype(np.float32)
    
    # Determine task type
    if args.task_type == 'auto':
        if len(task_names) == 1:
            task_type = detect_task_type(Y[:, 0])
        else:
            task_type = "classification"
    else:
        task_type = args.task_type
    print(f"Task type: {task_type}")
    
    # Scaffold split
    print("\nPerforming scaffold split...")
    train_idx, valid_idx, test_idx = scaffold_split(
        df_smiles, args.frac_train, args.frac_valid, args.frac_test, args.seed
    )
    
    X_tr, X_va, X_te = X[train_idx], X[valid_idx], X[test_idx]
    Y_tr, Y_va, Y_te = Y[train_idx], Y[valid_idx], Y[test_idx]
    mask_tr, mask_va, mask_te = mask[train_idx], mask[valid_idx], mask[test_idx]
    
    # Optuna optimization
    print(f"\n[Optuna] Starting Bayesian optimization ({args.n_trials} trials)...")
    study = optuna.create_study(
        direction="minimize" if task_type == "regression" else "maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=8, n_warmup_steps=10, interval_steps=5),
    )
    
    objective = create_objective(
        X_tr, Y_tr, mask_tr, X_va, Y_va, mask_va, task_names, task_type, device, args
    )
    study.optimize(objective, n_trials=args.n_trials, n_jobs=args.n_jobs)
    
    print(f"\n[Optuna] Best value: {study.best_value:.4f}")
    print(f"[Optuna] Best params: {study.best_params}")
    
    # Parse best config
    best = study.best_params
    backbone_dims = [best[k] for k in sorted([k for k in best if k.startswith("backbone_dims_") and k != "backbone_dims_n"],
                                               key=lambda s: int(s.split("_")[-1]))][:best["backbone_dims_n"]]
    head_dims = [best[k] for k in sorted([k for k in best if k.startswith("head_dims_") and k != "head_dims_n"],
                                           key=lambda s: int(s.split("_")[-1]))][:best["head_dims_n"]]
    
    best_config = {
        "backbone_dims": backbone_dims,
        "head_dims": head_dims,
        "dropout": best["dropout"],
        "lr": best["lr"],
        "weight_decay": best["weight_decay"],
        "batch_size": best["batch_size"],
    }
    
    # Final training
    print("\n[Training] Final model with best parameters...")
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)
    
    train_loader = DataLoader(SingleTaskDataset(X_tr_s, Y_tr, mask_tr), batch_size=best_config["batch_size"], shuffle=True)
    val_loader = DataLoader(SingleTaskDataset(X_va_s, Y_va, mask_va), batch_size=best_config["batch_size"], shuffle=False)
    test_loader = DataLoader(SingleTaskDataset(X_te_s, Y_te, mask_te), batch_size=best_config["batch_size"], shuffle=False)
    
    pos_weights = compute_pos_weights(Y_tr, mask_tr) if task_type == "classification" else None
    
    model = ACSModel(X.shape[1], len(task_names), backbone_dims, head_dims, best_config["dropout"])
    
    ckpts, _, _ = train_model(
        model, {"train": train_loader, "val": val_loader}, pos_weights, device,
        epochs=args.max_epochs_final, lr=best_config["lr"], weight_decay=best_config["weight_decay"],
        patience=args.patience_final, task_names=task_names, task_type=task_type,
        print_every=args.print_every, save_dir=str(output_dir)
    )
    
    # Evaluation
    if task_type == "regression":
        metrics_df = evaluate_regression(model, test_loader, device, ckpts, task_names)
        mean_metric = metrics_df["rmse"].mean()
        metric_name = "rmse"
    else:
        metrics_df = evaluate_classification(model, test_loader, device, ckpts, task_names)
        mean_metric = metrics_df["roc_auc"].dropna().mean()
        metric_name = "roc_auc"
    
    print(f"\n[Test Results]")
    print(metrics_df)
    print(f"\nMean {metric_name}: {mean_metric:.4f}")
    
    # Save results
    metrics_df.to_csv(output_dir / f"test_metrics_{args.dataset}.csv", index=False)
    
    with open(output_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    with open(output_dir / "best_config.json", "w") as f:
        json.dump({
            "config": best_config,
            f"mean_test_{metric_name}": float(mean_metric),
            "tasks": task_names,
            "task_type": task_type,
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
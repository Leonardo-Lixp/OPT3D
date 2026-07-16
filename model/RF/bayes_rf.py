#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Random Forest Hyperparameter Optimization with Optuna
======================================================

This script performs Bayesian hyperparameter optimization for Random Forest
models on molecular datasets using scaffold splitting.

Features:
- Automatic detection of regression vs classification tasks
- Support for multi-task datasets (e.g., tox21, sider)
- Scaffold-based train/test splitting
- Optuna-based hyperparameter search
- Cross-validation for robust evaluation

Usage:
    # Regression task
    python bayes_rf.py --dataset esol --descriptor esol_nsc500_sl20.pkl --n-trials 100

    # Classification task (multi-task)
    python bayes_rf.py --dataset tox21 --descriptor tox21_nsc500_sl20.pkl --n-trials 200

    # Specify task type manually
    python bayes_rf.py --dataset lip --descriptor lip_nsc500_sl20.pkl --task-type regression
"""

import os
import sys
import json
import math
import pickle
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

import optuna
from optuna.samplers import TPESampler

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, mean_squared_error,
    mean_absolute_error, r2_score
)


# =============================================================================
# Configuration
# =============================================================================

# CPU configuration for parallel processing
CPUS_PER_TASK = 40

# Set environment variables for thread control
for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[k] = str(CPUS_PER_TASK)
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")


# =============================================================================
# Scaffold Splitting Functions
# =============================================================================

def get_ring_count(smiles: str) -> int:
    """Get ring count for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    n_rings = mol.GetRingInfo().NumRings()
    return min(n_rings, 5)


def get_scaffold(smiles: str) -> str:
    """Get Murcko scaffold for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)


def scaffold_split(df_smiles: pd.DataFrame, frac_train: float, frac_test: float, seed: int):
    """
    Perform scaffold-based train/test split.
    
    Parameters
    ----------
    df_smiles : pd.DataFrame
        DataFrame with 'smiles' column
    frac_train : float
        Fraction of data for training
    frac_test : float
        Fraction of data for testing
    seed : int
        Random seed
        
    Returns
    -------
    train_idx, test_idx : np.ndarray
        Indices for train and test sets
    """
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
        label = ">=5 rings" if n_rings == 5 else f"{n_rings} rings"
        print(f"  {label}: {len(idxs)} molecules")

    train_idx, test_idx = [], []
    random_split_layers = {}

    for n_rings, scf_dict in reversed(list(scaffold_groups.items())):
        scf_list = list(scf_dict.keys())

        if len(scf_list) <= 3:
            print(f"Layer {n_rings}: {len(scf_list)} scaffolds too few, using random split.")
            idxs = ring_groups[n_rings]
            random_split_layers[n_rings] = idxs
            rng.shuffle(idxs)
            n_total = len(idxs)
            n_test = round(frac_test * n_total)
            test_idx.extend(idxs[:n_test])
            train_idx.extend(idxs[n_test:])
            continue

        rng.shuffle(scf_list)
        n_total = len(scf_list)
        n_test = round(frac_test * n_total)
        test_scf = scf_list[:n_test]
        train_scf = scf_list[n_test:]

        train_mols, test_mols = [], []
        for s in train_scf:
            train_mols.extend(scf_dict[s])
        for s in test_scf:
            test_mols.extend(scf_dict[s])

        print(f"Layer {n_rings}: {len(train_scf)} scaffolds ({len(train_mols)} mols) train, "
              f"{len(test_scf)} ({len(test_mols)}) test")

        train_idx.extend(train_mols)
        test_idx.extend(test_mols)

    n_train, n_test = len(train_idx), len(test_idx)
    n_total = n_train + n_test
    print(f"\nTotal molecules: {n_total}")
    print(f"Train: {n_train} ({n_train/n_total:.2%})")
    print(f"Test : {n_test} ({n_test/n_total:.2%})")

    # Adjust ratios if deviation > 1%
    target_frac = np.array([frac_train, frac_test])
    current_counts = np.array([len(train_idx), len(test_idx)])
    current_frac = current_counts / current_counts.sum()

    deviation = target_frac - current_frac
    if np.any(np.abs(deviation) > 0.01):
        print("\nAdjusting using random-split layers to fix ratio...")
        all_random_idxs = []
        for idxs in random_split_layers.values():
            all_random_idxs.extend(idxs)
        rng.shuffle(all_random_idxs)

        total = len(train_idx) + len(test_idx)
        target_counts = (target_frac * total).astype(int)

        while len(train_idx) > target_counts[0] and all_random_idxs:
            i = all_random_idxs.pop()
            if i in train_idx:
                train_idx.remove(i)
                test_idx.append(i)

        while len(train_idx) < target_counts[0] and all_random_idxs:
            i = all_random_idxs.pop()
            if i not in train_idx and i not in test_idx:
                train_idx.append(i)

        print(f"After adjustment: Train {len(train_idx)/total:.2%}, Test {len(test_idx)/total:.2%}")

    return np.array(train_idx), np.array(test_idx)


# =============================================================================
# Cross-validation Functions
# =============================================================================

def _safe_n_splits(y: np.ndarray, desired: int = 5) -> int:
    """Determine safe number of CV splits based on class distribution."""
    y = np.asarray(y).astype(int)
    cnt = np.bincount(y)
    min_cls = cnt.min() if cnt.size > 0 else 0
    return int(max(2, min(desired, min_cls)))


def cv_auc_classification(model, X, y, n_splits=5, seed=42) -> float:
    """Cross-validation AUC for classification."""
    y = np.asarray(y).astype(int)
    n_splits = _safe_n_splits(y, desired=n_splits)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []
    for tr, va in skf.split(X, y):
        model.fit(X[tr], y[tr])
        proba = model.predict_proba(X[va])[:, 1]
        aucs.append(roc_auc_score(y[va], proba))
    return float(np.mean(aucs))


def cv_rmse_regression(model, X, y, n_splits=5, seed=42) -> float:
    """Cross-validation RMSE for regression."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rmses = []
    for tr, va in kf.split(X):
        model.fit(X[tr], y[tr])
        pred = model.predict(X[va])
        rmses.append(math.sqrt(mean_squared_error(y[va], pred)))
    return float(np.mean(rmses))


# =============================================================================
# Hyperparameter Optimization
# =============================================================================

def bayes_optimize_rf_classification(X_train, y_train, n_trials=50, seed=42):
    """Optuna optimization for RandomForestClassifier."""
    
    def objective(trial: optuna.trial.Trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1200, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
            "random_state": seed,
            "n_jobs": CPUS_PER_TASK,
        }
        model = RandomForestClassifier(**params)
        return cv_auc_classification(model, X_train, y_train, n_splits=5, seed=seed)

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False)

    print(f"\n[Optuna] Best AUC (CV): {study.best_value:.5f}")
    print("[Optuna] Best Params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    best_params = study.best_params.copy()
    best_params.update({"random_state": seed, "n_jobs": 1})
    return RandomForestClassifier(**best_params), study


def bayes_optimize_rf_regression(X_train, y_train, n_trials=50, seed=42):
    """Optuna optimization for RandomForestRegressor."""
    
    def objective(trial: optuna.trial.Trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1200, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "random_state": seed,
            "n_jobs": CPUS_PER_TASK,
        }
        model = RandomForestRegressor(**params)
        return cv_rmse_regression(model, X_train, y_train, n_splits=5, seed=seed)

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False)

    print(f"\n[Optuna] Best RMSE (CV): {study.best_value:.5f}")
    print("[Optuna] Best Params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    best_params = study.best_params.copy()
    best_params.update({"random_state": seed, "n_jobs": 1})
    return RandomForestRegressor(**best_params), study


# =============================================================================
# Task Detection and Evaluation
# =============================================================================

def detect_task_type(labels: np.ndarray) -> str:
    """
    Automatically detect if task is classification or regression.
    
    Classification: labels are 0/1 or have few unique values (<10% of samples)
    Regression: continuous values
    """
    unique_vals = np.unique(labels[~np.isnan(labels)])
    
    # Check if binary classification (0/1)
    if set(unique_vals).issubset({0.0, 1.0, 0, 1}):
        return "classification"
    
    # Check if few unique values (<10)
    if len(unique_vals) < 10:
        return "classification"
    
    return "regression"


def evaluate_classification(model, X_test, y_test):
    """Evaluate classification model."""
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    return {"roc_auc": auc, "pr_auc": ap}


def evaluate_regression(model, X_test, y_test):
    """Evaluate regression model."""
    y_pred = model.predict(X_test)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {"rmse": rmse, "mae": mae, "r2": r2}


# =============================================================================
# Main Processing Functions
# =============================================================================

def process_single_task(X_all, y_all, train_idx, test_idx, task_name, 
                        task_type, n_trials, seed, output_dir):
    """Process a single task (regression or classification)."""
    
    # Filter valid samples
    if task_type == "classification":
        valid_mask = np.isin(y_all, [0, 1])
    else:
        valid_mask = ~np.isnan(y_all)
    
    tr_mask = np.zeros_like(valid_mask, dtype=bool)
    te_mask = np.zeros_like(valid_mask, dtype=bool)
    tr_mask[train_idx] = True
    te_mask[test_idx] = True
    
    tr_keep = tr_mask & valid_mask
    te_keep = te_mask & valid_mask
    
    X_tr, y_tr = X_all[tr_keep], y_all[tr_keep]
    X_te, y_te = X_all[te_keep], y_all[te_keep]
    
    # Skip if too few samples
    if X_tr.shape[0] < 20:
        print(f"[Skip] Task '{task_name}': too few training samples ({X_tr.shape[0]})")
        return None
    
    if task_type == "classification":
        if np.unique(y_tr).size < 2:
            print(f"[Skip] Task '{task_name}': single class in training set")
            return None
        if np.unique(y_te).size < 2:
            print(f"[Skip] Task '{task_name}': single class in test set")
            return None
    
    print(f"\n{'='*60}")
    print(f"Task: {task_name} | Type: {task_type}")
    print(f"Train: {X_tr.shape[0]} | Test: {X_te.shape[0]}")
    if task_type == "classification":
        print(f"Pos rate (train): {y_tr.mean():.3f} | Pos rate (test): {y_te.mean():.3f}")
    
    # Hyperparameter optimization
    if task_type == "classification":
        best_model, study = bayes_optimize_rf_classification(
            X_tr, y_tr, n_trials=n_trials, seed=seed
        )
    else:
        best_model, study = bayes_optimize_rf_regression(
            X_tr, y_tr, n_trials=n_trials, seed=seed
        )
    
    # Train and evaluate
    best_model.fit(X_tr, y_tr)
    
    if task_type == "classification":
        metrics = evaluate_classification(best_model, X_te, y_te)
        print(f"[{task_name}] ROC-AUC={metrics['roc_auc']:.5f} | PR-AUC={metrics['pr_auc']:.5f}")
    else:
        metrics = evaluate_regression(best_model, X_te, y_te)
        print(f"[{task_name}] RMSE={metrics['rmse']:.5f} | MAE={metrics['mae']:.5f} | R2={metrics['r2']:.5f}")
    
    # Save best params
    param_file = output_dir / f"params_{task_name}.json"
    with open(param_file, "w", encoding="utf-8") as f:
        json.dump(study.best_params, f, ensure_ascii=False, indent=2)
    
    return {
        "task": task_name,
        "task_type": task_type,
        "n_train": int(X_tr.shape[0]),
        "n_test": int(X_te.shape[0]),
        **metrics,
        "best_params": study.best_params
    }


def main():
    parser = argparse.ArgumentParser(
        description='Random Forest Hyperparameter Optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Regression task (auto-detect)
  python bayes_rf.py --dataset esol --descriptor esol_nsc500_sl20.pkl
  
  # Classification task
  python bayes_rf.py --dataset bace --descriptor bace_nsc500_sl20.pkl
  
  # Multi-task classification
  python bayes_rf.py --dataset tox21 --descriptor tox21_nsc500_sl20.pkl --n-trials 200
  
  # Force task type
  python bayes_rf.py --dataset lip --descriptor lip_nsc500_sl20.pkl --task-type regression
"""
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Dataset name (e.g., esol, bace, tox21)'
    )
    
    parser.add_argument(
        '--descriptor',
        type=str,
        required=True,
        help='Descriptor file name (e.g., esol_nsc500_sl20.pkl)'
    )
    
    parser.add_argument(
        '--task-type',
        type=str,
        choices=['auto', 'classification', 'regression'],
        default='auto',
        help='Task type (default: auto-detect)'
    )
    
    parser.add_argument(
        '--n-trials',
        type=int,
        default=100,
        help='Number of Optuna trials (default: 100)'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test set fraction (default: 0.2)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=5,
        help='Random seed (default: 5)'
    )
    
    parser.add_argument(
        '--descriptor-dir',
        type=str,
        default='../calculate_descriptors/descriptors',
        help='Directory containing descriptor files'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='../data',
        help='Directory containing label and SMILES files'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Set paths
    descriptor_dir = Path(args.descriptor_dir)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load descriptors
    descriptor_path = descriptor_dir / args.descriptor
    print(f"Loading descriptors: {descriptor_path}")
    with open(descriptor_path, "rb") as f:
        X_all = pickle.load(f)
    print(f"  Shape: {X_all.shape}")
    
    # Load labels
    label_path = data_dir / f"{args.dataset}.csv"
    print(f"Loading labels: {label_path}")
    df_labels = pd.read_csv(label_path)
    if len(df_labels) != X_all.shape[0]:
        raise ValueError(f"Sample mismatch: features={X_all.shape[0]}, labels={len(df_labels)}")
    print(f"  Tasks: {list(df_labels.columns)}")
    
    # Load SMILES for scaffold split
    smiles_path = data_dir / f"{args.dataset}.csv"
    df_smiles = pd.read_csv(smiles_path)
    if 'smiles' not in df_smiles.columns:
        raise ValueError("Dataset must have 'smiles' column")
    
    # Scaffold split
    print(f"\nPerforming scaffold split (seed={args.seed})...")
    train_idx, test_idx = scaffold_split(
        df_smiles, 
        frac_train=1-args.test_size, 
        frac_test=args.test_size, 
        seed=args.seed
    )
    
    # Process each task
    task_cols = list(df_labels.columns)
    # Remove 'smiles' column if present
    if 'smiles' in task_cols:
        task_cols.remove('smiles')
    
    results = []
    
    for task in task_cols:
        y_all = df_labels[task].to_numpy()
        
        # Detect or use specified task type
        if args.task_type == 'auto':
            task_type = detect_task_type(y_all)
        else:
            task_type = args.task_type
        
        result = process_single_task(
            X_all, y_all, train_idx, test_idx, task,
            task_type, args.n_trials, args.seed, output_dir
        )
        
        if result is not None:
            results.append(result)
    
    # Save summary
    if results:
        df_results = pd.DataFrame(results)
        
        # Flatten best_params
        params_df = pd.json_normalize(df_results["best_params"])
        df_flat = pd.concat([df_results.drop(columns=["best_params"]), params_df], axis=1)
        
        # Sort by primary metric
        if "roc_auc" in df_flat.columns:
            df_flat.sort_values("roc_auc", ascending=False, inplace=True)
        elif "rmse" in df_flat.columns:
            df_flat.sort_values("rmse", ascending=True, inplace=True)
        
        output_file = output_dir / f"results_{args.dataset}.csv"
        df_flat.to_csv(output_file, index=False)
        
        print(f"\n{'='*60}")
        print(f"Results saved to: {output_file}")
        print(f"\nTop 5 tasks:")
        print(df_flat.head().to_string(index=False))
    else:
        print("\n[Warning] No tasks were successfully processed")


if __name__ == "__main__":
    main()
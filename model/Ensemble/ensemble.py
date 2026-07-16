#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fixed-Weight Ensemble Model with AutoGluon
===========================================

This script trains an ensemble model using fixed weights for three base learners:
- NeuralNetTorch (Neural Network)
- RandomForestEntr (Random Forest with entropy criterion)
- ExtraTreesEntr (Extra Trees with entropy criterion)

Features:
- Support for regression (RMSE) and classification (AUC) tasks
- Support for single-task and multi-task datasets
- Scaffold-based train/valid/test splitting
- Multiple random seeds for robust evaluation
- Fixed-weight ensemble prediction

Usage:
    # Regression task
    python ensemble.py --dataset esol --descriptor esol_nsc500_sl20.pkl --task-type regression

    # Classification task
    python ensemble.py --dataset bace --descriptor bace_nsc500_sl20.pkl --task-type classification

    # Multi-task classification
    python ensemble.py --dataset tox21 --descriptor tox21_nsc500_sl20.pkl --task-type classification

    # Custom weights
    python ensemble.py --dataset lip --descriptor lip_nsc500_sl20.pkl --w-nn 0.5 --w-rf 0.4 --w-xt 0.1

Note:
    This script requires AutoGluon to be installed in a separate conda environment.
    Activate it first: conda activate autogluon
"""

import os
import sys
import json
import math
import pickle
import random
import shutil
import tempfile
import argparse
import logging
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

# AutoGluon (requires separate conda environment)
try:
    import autogluon.tabular as ag
except ImportError:
    print("Error: AutoGluon not installed. Please activate the autogluon conda environment:")
    print("  conda activate autogluon")
    sys.exit(1)

# Scikit-learn metrics
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, average_precision_score
)


# =============================================================================
# Logging Configuration
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# =============================================================================
# Scaffold Splitting Functions
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
    """
    Perform scaffold-based train/valid/test split.
    
    Parameters
    ----------
    df_smiles : pd.DataFrame
        DataFrame with 'smiles' column
    frac_train : float
        Fraction of data for training
    frac_valid : float
        Fraction of data for validation
    frac_test : float
        Fraction of data for testing
    seed : int
        Random seed
        
    Returns
    -------
    train_idx, valid_idx, test_idx : np.ndarray
        Indices for train, valid, and test sets
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
        label = f">=5 rings" if n_rings == 5 else f"{n_rings} rings"
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

    n_train, n_valid, n_test = len(train_idx), len(valid_idx), len(test_idx)
    n_total = n_train + n_valid + n_test
    print(f"\nTotal molecules: {n_total}")
    print(f"Train: {n_train} ({n_train/n_total:.2%})")
    print(f"Valid: {n_valid} ({n_valid/n_total:.2%})")
    print(f"Test : {n_test} ({n_test/n_total:.2%})")

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

        print(f"After adjustment: Train {len(train_idx)/total:.2%}, "
              f"Valid {len(valid_idx)/total:.2%}, Test {len(test_idx)/total:.2%}")

    return np.array(train_idx), np.array(valid_idx), np.array(test_idx)


# =============================================================================
# Utility Functions
# =============================================================================

def find_model_by_keyword(model_names: List[str], keyword: str) -> str:
    """Find model name containing keyword."""
    matches = [m for m in model_names if keyword in str(m)]
    if not matches:
        raise RuntimeError(f"Cannot find model containing keyword='{keyword}'. Available models: {model_names}")
    return sorted(matches, key=len)[0]


def get_pos_col(proba_df: pd.DataFrame):
    """Get positive class column from probability DataFrame."""
    if 1 in proba_df.columns:
        return 1
    if "1" in proba_df.columns:
        return "1"
    return proba_df.columns[-1]


def infer_label_cols(df: pd.DataFrame) -> List[str]:
    """Infer label columns from DataFrame."""
    drop_like = {"smiles", "SMILES", "drug", "Drug", "name", "Name", "id", "ID"}
    cand = [c for c in df.columns if c not in drop_like]
    return [c for c in cand if pd.api.types.is_numeric_dtype(df[c])]


def detect_task_type(labels: np.ndarray) -> str:
    """Auto-detect if task is classification or regression."""
    unique_vals = np.unique(labels[~np.isnan(labels)])
    if set(unique_vals).issubset({0.0, 1.0, 0, 1}):
        return "classification"
    if len(unique_vals) < 10:
        return "classification"
    return "regression"


def clean_data(features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Remove rows with NaN values."""
    mask = ~np.isnan(features).any(axis=1) & ~np.isnan(labels)
    return features[mask], labels[mask]


# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate regression metrics."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred))
    }


def evaluate_classification(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Calculate classification metrics."""
    y_true = np.asarray(y_true).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob))
    }


# =============================================================================
# Model Training Functions
# =============================================================================

def train_single_task_ensemble(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str,
    task_type: str,
    fixed_weights: Dict[str, float],
    model_path: str,
    presets: str = "medium_quality"
):
    """
    Train fixed-weight ensemble for a single task.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training data
    val_df : pd.DataFrame
        Validation data
    test_df : pd.DataFrame
        Test data
    label_col : str
        Label column name
    task_type : str
        'classification' or 'regression'
    fixed_weights : dict
        Fixed weights for each model
    model_path : str
        Path to save AutoGluon model
    presets : str
        AutoGluon presets
        
    Returns
    -------
    dict
        Evaluation metrics and model info
    """
    # Define hyperparameters based on task type
    if task_type == "classification":
        criterion = "entropy"
        eval_metric = "roc_auc"
    else:
        criterion = "squared_error"
        eval_metric = "root_mean_squared_error"

    hyperparameters = {
        "RF": [{"criterion": criterion, "ag_args": {"name_suffix": "Entr"}}],
        "XT": [{"criterion": criterion, "ag_args": {"name_suffix": "Entr"}}],
        "NN_TORCH": {},
    }

    # Train AutoGluon model
    predictor = ag.TabularPredictor(
        label=label_col,
        eval_metric=eval_metric,
        path=model_path,
        problem_type=task_type,
    ).fit(
        train_data=train_df,
        tuning_data=val_df,
        presets=presets,
        hyperparameters=hyperparameters,
        num_stack_levels=0,
        num_bag_folds=0,
        verbosity=0,
    )

    # Get model names
    model_names = predictor.model_names()
    rf_name = find_model_by_keyword(model_names, "RandomForestEntr")
    xt_name = find_model_by_keyword(model_names, "ExtraTreesEntr")
    nn_name = find_model_by_keyword(model_names, "NeuralNetTorch")

    # Prepare test data
    y_test = test_df[label_col]
    X_test = test_df.drop(columns=[label_col])

    # Evaluate single models
    single_metrics = {}
    
    if task_type == "classification":
        # Classification: weighted probability average
        proba_rf = predictor.predict_proba(X_test, model=rf_name)
        proba_xt = predictor.predict_proba(X_test, model=xt_name)
        proba_nn = predictor.predict_proba(X_test, model=nn_name)
        
        pos_col = get_pos_col(proba_rf)
        
        # Single model evaluations
        for name, proba in [(rf_name, proba_rf), (xt_name, proba_xt), (nn_name, proba_nn)]:
            single_metrics[name] = evaluate_classification(y_test, proba[pos_col].values)
        
        # Fixed-weight ensemble
        ens_prob = (
            fixed_weights["RandomForestEntr"] * proba_rf[pos_col].values +
            fixed_weights["ExtraTreesEntr"] * proba_xt[pos_col].values +
            fixed_weights["NeuralNetTorch"] * proba_nn[pos_col].values
        )
        ens_metrics = evaluate_classification(y_test, ens_prob)
        
    else:
        # Regression: weighted prediction average
        pred_rf = predictor.predict(X_test, model=rf_name).to_numpy()
        pred_xt = predictor.predict(X_test, model=xt_name).to_numpy()
        pred_nn = predictor.predict(X_test, model=nn_name).to_numpy()
        
        # Single model evaluations
        for name, pred in [(rf_name, pred_rf), (xt_name, pred_xt), (nn_name, pred_nn)]:
            single_metrics[name] = evaluate_regression(y_test, pred)
        
        # Fixed-weight ensemble
        ens_pred = (
            fixed_weights["RandomForestEntr"] * pred_rf +
            fixed_weights["ExtraTreesEntr"] * pred_xt +
            fixed_weights["NeuralNetTorch"] * pred_nn
        )
        ens_metrics = evaluate_regression(y_test, ens_pred)

    return {
        "ensemble_metrics": ens_metrics,
        "single_metrics": single_metrics,
        "models": {"rf": rf_name, "xt": xt_name, "nn": nn_name},
        "predictor": predictor
    }


# =============================================================================
# Main Processing
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Fixed-Weight Ensemble Model with AutoGluon',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Regression task
  python ensemble.py --dataset esol --descriptor esol_nsc500_sl20.pkl --task-type regression
  
  # Classification task
  python ensemble.py --dataset bace --descriptor bace_nsc500_sl20.pkl --task-type classification
  
  # Multi-task classification
  python ensemble.py --dataset tox21 --descriptor tox21_nsc500_sl20.pkl --task-type classification
  
  # Custom weights
  python ensemble.py --dataset lip --descriptor lip_nsc500_sl20.pkl --w-nn 0.5 --w-rf 0.4 --w-xt 0.1

Note:
  This script requires AutoGluon. Activate it first: conda activate autogluon
"""
    )
    
    # Data arguments
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--descriptor', type=str, required=True, help='Descriptor file name')
    parser.add_argument('--task-type', type=str, choices=['auto', 'classification', 'regression'],
                        default='auto', help='Task type (default: auto-detect)')
    
    # Split arguments
    parser.add_argument('--frac-train', type=float, default=0.8, help='Train fraction')
    parser.add_argument('--frac-valid', type=float, default=0.1, help='Validation fraction')
    parser.add_argument('--frac-test', type=float, default=0.1, help='Test fraction')
    
    # Evaluation arguments
    parser.add_argument('--n-accept', type=int, default=10, help='Number of accepted seeds')
    parser.add_argument('--max-seed-tries', type=int, default=100, help='Maximum seeds to try')
    parser.add_argument('--min-tasks', type=int, default=5, help='Minimum tasks for multi-task')
    
    # Ensemble weights
    parser.add_argument('--w-nn', type=float, default=0.562, help='Weight for NeuralNetTorch')
    parser.add_argument('--w-rf', type=float, default=0.375, help='Weight for RandomForestEntr')
    parser.add_argument('--w-xt', type=float, default=0.062, help='Weight for ExtraTreesEntr')
    
    # AutoGluon arguments
    parser.add_argument('--presets', type=str, default='medium_quality', help='AutoGluon presets')
    parser.add_argument('--tmp-dir', type=str, default=None, help='Temporary directory for AutoGluon')
    
    # Path arguments
    parser.add_argument('--descriptor-dir', type=str, default='../calculate_descriptors/descriptors',
                        help='Descriptor directory')
    parser.add_argument('--data-dir', type=str, default='../data', help='Data directory')
    parser.add_argument('--output-dir', type=str, default='ensemble_results',
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Validate weights
    weight_sum = args.w_nn + args.w_rf + args.w_xt
    if abs(weight_sum - 1.0) > 0.001:
        logger.warning(f"Weights sum to {weight_sum:.3f}, normalizing to 1.0")
        args.w_nn /= weight_sum
        args.w_rf /= weight_sum
        args.w_xt /= weight_sum
    
    fixed_weights = {
        "NeuralNetTorch": args.w_nn,
        "RandomForestEntr": args.w_rf,
        "ExtraTreesEntr": args.w_xt,
    }
    
    # Set paths
    descriptor_dir = Path(args.descriptor_dir)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load descriptors
    descriptor_path = descriptor_dir / args.descriptor
    logger.info(f"Loading descriptors: {descriptor_path}")
    with open(descriptor_path, "rb") as f:
        X_all = pickle.load(f)
    logger.info(f"  Shape: {X_all.shape}")
    
    # Load labels
    label_path = data_dir / f"{args.dataset}.csv"
    logger.info(f"Loading labels: {label_path}")
    df_labels = pd.read_csv(label_path)
    
    # Get SMILES
    if 'smiles' not in df_labels.columns:
        raise ValueError("Dataset must have 'smiles' column")
    df_smiles = df_labels[['smiles']].copy()
    
    # Get label columns
    label_cols = infer_label_cols(df_labels)
    logger.info(f"  Label columns: {label_cols}")
    
    # Check alignment
    if len(df_labels) != X_all.shape[0]:
        raise ValueError(f"Sample mismatch: features={X_all.shape[0]}, labels={len(df_labels)}")
    
    # Build feature DataFrame
    X_df = pd.DataFrame(X_all)
    X_df.columns = [f"c{i}" for i in range(X_df.shape[1])]
    
    # Determine task type
    if args.task_type == 'auto':
        if len(label_cols) == 1:
            task_type = detect_task_type(df_labels[label_cols[0]].values)
        else:
            task_type = "classification"  # Default for multi-task
    else:
        task_type = args.task_type
    
    logger.info(f"Task type: {task_type}")
    
    # Process single or multi-task
    is_multi_task = len(label_cols) > 1
    
    # Multiple seed evaluation
    metrics_list = []
    best_metric = float('inf') if task_type == "regression" else -1.0
    best_predictor = None
    best_seed = None
    best_detail = None
    
    accepted = 0
    seed_cursor = 0
    
    while accepted < args.n_accept and seed_cursor < args.max_seed_tries:
        logger.info(f"\n{'='*70}\nTry seed={seed_cursor} (accepted {accepted}/{args.n_accept})")
        
        try:
            train_idx, val_idx, test_idx = scaffold_split(
                df_smiles, 
                frac_train=args.frac_train,
                frac_valid=args.frac_valid,
                frac_test=args.frac_test,
                seed=seed_cursor
            )
        except Exception as e:
            logger.error(f"Scaffold split failed: {e}")
            seed_cursor += 1
            continue
        
        # Check split ratio
        total = len(train_idx) + len(val_idx) + len(test_idx)
        train_frac = len(train_idx) / total if total > 0 else 0.0
        test_frac = len(test_idx) / total if total > 0 else 0.0
        
        if abs(train_frac - args.frac_train) > 0.02 or abs(test_frac - args.frac_test) > 0.02:
            logger.info(f"Skipped (train={train_frac:.2%}, test={test_frac:.2%})")
            seed_cursor += 1
            continue
        
        # For multi-task, evaluate all tasks
        if is_multi_task:
            task_metrics = []
            valid_tasks = 0
            
            for label_col in label_cols:
                # Prepare data for this task
                df_all = X_df.copy()
                df_all[label_col] = df_labels[label_col].values
                
                train_df = df_all.iloc[train_idx].dropna(subset=[label_col])
                val_df = df_all.iloc[val_idx].dropna(subset=[label_col])
                test_df = df_all.iloc[test_idx].dropna(subset=[label_col])
                
                # Skip if too few samples
                if len(train_df) < 20 or len(val_df) < 10 or len(test_df) < 10:
                    continue
                if train_df[label_col].nunique() < 2 or test_df[label_col].nunique() < 2:
                    continue
                
                # Create temporary model path
                safe_label = str(label_col).replace("/", "_").replace("\\", "_")
                model_path = tempfile.mkdtemp(prefix=f"ag_seed{seed_cursor}_{safe_label}_", dir=args.tmp_dir)
                
                try:
                    result = train_single_task_ensemble(
                        train_df, val_df, test_df, label_col, task_type,
                        fixed_weights, model_path, args.presets
                    )
                    
                    metric = result["ensemble_metrics"]
                    primary_metric = metric["rmse"] if task_type == "regression" else metric["roc_auc"]
                    task_metrics.append(primary_metric)
                    valid_tasks += 1
                    
                except Exception as e:
                    logger.error(f"Task {label_col} failed: {e}")
                finally:
                    shutil.rmtree(model_path, ignore_errors=True)
            
            if valid_tasks < args.min_tasks:
                logger.info(f"Seed={seed_cursor}: only {valid_tasks} valid tasks, skip")
                seed_cursor += 1
                continue
            
            # Average across tasks
            avg_metric = float(np.mean(task_metrics))
            metrics_list.append(avg_metric)
            accepted += 1
            
            logger.info(f"Seed={seed_cursor}: {valid_tasks} tasks, avg metric={avg_metric:.4f}")
            
        else:
            # Single task
            label_col = label_cols[0]
            df_all = X_df.copy()
            df_all[label_col] = df_labels[label_col].values
            
            # Remove NaN
            df_all = df_all.dropna(subset=[label_col])
            
            # Re-index split indices
            valid_indices = set(df_all.index)
            train_idx_f = [i for i in train_idx if i in valid_indices]
            val_idx_f = [i for i in val_idx if i in valid_indices]
            test_idx_f = [i for i in test_idx if i in valid_indices]
            
            train_df = df_all.iloc[train_idx_f]
            val_df = df_all.iloc[val_idx_f]
            test_df = df_all.iloc[test_idx_f]
            
            if len(train_df) < 20 or len(test_df) < 10:
                logger.info(f"Skipped: too few samples")
                seed_cursor += 1
                continue
            
            model_path = output_dir / f"ag_seed{seed_cursor}"
            
            try:
                result = train_single_task_ensemble(
                    train_df, val_df, test_df, label_col, task_type,
                    fixed_weights, str(model_path), args.presets
                )
                
                metric = result["ensemble_metrics"]
                primary_metric = metric["rmse"] if task_type == "regression" else metric["roc_auc"]
                metrics_list.append(primary_metric)
                accepted += 1
                
                logger.info(f"Seed={seed_cursor}: metric={primary_metric:.4f}")
                
                # Track best model
                if (task_type == "regression" and primary_metric < best_metric) or \
                   (task_type == "classification" and primary_metric > best_metric):
                    best_metric = primary_metric
                    best_predictor = result["predictor"]
                    best_seed = seed_cursor
                    best_detail = {
                        "seed": seed_cursor,
                        "ensemble_metrics": metric,
                        "single_metrics": result["single_metrics"],
                        "fixed_weights": fixed_weights,
                    }
                
            except Exception as e:
                logger.error(f"Training failed: {e}")
        
        seed_cursor += 1
    
    # Summary
    if metrics_list:
        mean_metric = float(np.mean(metrics_list))
        std_metric = float(np.std(metrics_list, ddof=1)) if len(metrics_list) > 1 else 0.0
        se_metric = std_metric / math.sqrt(len(metrics_list))
        
        logger.info(f"\n{'='*70}")
        logger.info(f"SUMMARY")
        logger.info(f"{'='*70}")
        logger.info(f"Accepted seeds: {accepted}")
        logger.info(f"Mean metric: {mean_metric:.4f} ± {std_metric:.4f} (SE={se_metric:.4f})")
        logger.info(f"All metrics: {[f'{m:.4f}' for m in metrics_list]}")
        
        # Save best model (single task)
        if best_predictor is not None and not is_multi_task:
            save_path = output_dir / "best_predictor"
            best_predictor.save(str(save_path))
            logger.info(f"Best model saved to: {save_path}")
            
            # Save detail
            with open(output_dir / "best_model_info.txt", "w") as f:
                f.write(f"best_seed\t{best_seed}\n")
                f.write(f"best_metric\t{best_metric:.6f}\n")
                f.write(f"fixed_weights\t{fixed_weights}\n")
                f.write(f"mean_metric\t{mean_metric:.6f}\n")
                f.write(f"std_metric\t{std_metric:.6f}\n")
                f.write(f"metrics_list\t{','.join([f'{m:.6f}' for m in metrics_list])}\n")
        
        # Save results
        results_df = pd.DataFrame({
            "seed": list(range(len(metrics_list))),
            "metric": metrics_list
        })
        results_df.to_csv(output_dir / f"results_{args.dataset}.csv", index=False)
        
    else:
        logger.warning("No valid evaluations completed")


if __name__ == "__main__":
    main()
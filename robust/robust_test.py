#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Robustness Testing Script
==========================

This script evaluates model performance across different noise levels in features.
It tests the robustness of machine learning models when features are corrupted
with increasing levels of random noise.

Features:
- Support for regression (RMSE) and classification (AUC) tasks
- Support for single-task and multi-task datasets
- Multiple random train/test splits
- Automatic detection of task type
- Summary statistics (mean, std, SE) for each noise level

Usage:
    # Regression task
    python robust_test.py --dataset esol --task-type regression --pkl-files pct0.pkl pct10.pkl pct20.pkl pct40.pkl

    # Classification task
    python robust_test.py --dataset bace --task-type classification --pkl-files pct0.pkl pct10.pkl pct20.pkl pct40.pkl

    # Multi-task classification
    python robust_test.py --dataset tox21 --task-type classification --pkl-files pct0.pkl pct10.pkl pct20.pkl pct40.pkl
"""

import os
import re
import sys
import json
import pickle
import argparse
import math
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, roc_auc_score


# =============================================================================
# Utility Functions
# =============================================================================

def load_pkl_as_numpy(pkl_path: str) -> np.ndarray:
    """
    Load a pickle file and extract numpy array.
    
    Parameters
    ----------
    pkl_path : str
        Path to pickle file
        
    Returns
    -------
    np.ndarray
        Numpy array from pickle file
    """
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, np.ndarray):
        return obj

    if isinstance(obj, (list, tuple)) and len(obj) > 0 and isinstance(obj[0], np.ndarray):
        return obj[0]

    raise TypeError(
        f"Failed to parse a numpy.ndarray from the .pkl file; the actual type is: {type(obj)}"
    )


def extract_pct_from_name(fname: str) -> float:
    """
    Extract noise percentage from filename.
    
    Parameters
    ----------
    fname : str
        Filename (e.g., 'pct10.pkl')
        
    Returns
    -------
    float
        Noise percentage (e.g., 10.0)
    """
    base = os.path.basename(fname)
    m = re.search(r"pct(\d+(?:\.\d+)?)\.pkl$", base)
    if not m:
        return float("nan")
    return float(m.group(1))


def infer_label_columns(df: pd.DataFrame) -> List[str]:
    """
    Infer label columns from DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
        
    Returns
    -------
    List[str]
        List of label column names
    """
    exclude = {
        "smiles", "SMILES", "mol", "molecule", "name", "Name",
        "id", "ID", "mol_id", "MolID", "index", "Index",
        "split", "fold", "Fold"
    }
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    label_cols = [c for c in num_cols if c not in exclude]
    if len(label_cols) == 0:
        raise ValueError(
            "Cannot infer label columns automatically. "
            "Please specify --label-cols manually."
        )
    return label_cols


def detect_task_type(labels: np.ndarray) -> str:
    """Auto-detect if task is classification or regression."""
    unique_vals = np.unique(labels[~np.isnan(labels)])
    if set(unique_vals).issubset({0.0, 1.0, 0, 1}):
        return "classification"
    if len(unique_vals) < 10:
        return "classification"
    return "regression"


def valid_binary_mask(y: np.ndarray) -> np.ndarray:
    """Check if values are valid binary labels (0 or 1)."""
    return np.isfinite(y) & np.isin(y, [0, 1])


# =============================================================================
# Model Training Functions
# =============================================================================

def fit_predict_regression(X_train, X_test, y_train, y_test, model_params: dict = None):
    """
    Train and evaluate regression model.
    
    Parameters
    ----------
    X_train, X_test : np.ndarray
        Training and test features
    y_train, y_test : np.ndarray
        Training and test labels
    model_params : dict, optional
        RandomForest parameters
        
    Returns
    -------
    float
        RMSE on test set
    """
    if model_params is None:
        model_params = {
            "n_estimators": 500,
            "random_state": 42,
            "n_jobs": -1,
            "max_features": "sqrt",
        }

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestRegressor(**model_params)),
    ])

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    return rmse


def fit_predict_classification_multitask(
    X_train, X_test, Y_train, Y_test, model_params: dict = None
) -> tuple:
    """
    Train and evaluate classification model for multi-task data.
    
    Parameters
    ----------
    X_train, X_test : np.ndarray
        Training and test features
    Y_train, Y_test : np.ndarray
        Training and test labels (n_samples, n_tasks)
    model_params : dict, optional
        RandomForest parameters
        
    Returns
    -------
    tuple
        (macro_auc, weighted_auc, valid_tasks_count)
    """
    n_tasks = Y_train.shape[1]
    task_aucs = np.full(n_tasks, np.nan, dtype=np.float64)
    task_counts = np.zeros(n_tasks, dtype=np.int64)

    if model_params is None:
        model_params = {
            "n_estimators": 500,
            "random_state": 42,
            "n_jobs": -1,
            "max_features": "sqrt",
            "class_weight": "balanced",
        }

    for t in range(n_tasks):
        ytr = Y_train[:, t]
        yte = Y_test[:, t]

        mtr = valid_binary_mask(ytr)
        mte = valid_binary_mask(yte)

        if mtr.sum() < 5 or mte.sum() < 5:
            continue

        ytr_valid = ytr[mtr].astype(int)
        yte_valid = yte[mte].astype(int)

        if len(np.unique(yte_valid)) < 2:
            continue

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("rf", RandomForestClassifier(**model_params)),
        ])

        model.fit(X_train[mtr], ytr_valid)
        proba = model.predict_proba(X_test[mte])[:, 1]
        task_aucs[t] = float(roc_auc_score(yte_valid, proba))
        task_counts[t] = int(mte.sum())

    # Calculate macro and weighted AUC
    macro_auc = float(np.nanmean(task_aucs))
    if np.nansum(task_counts) > 0:
        weighted_auc = float(np.nansum(task_aucs * task_counts) / np.nansum(task_counts))
    else:
        weighted_auc = float("nan")
    
    valid_tasks = int(np.isfinite(task_aucs).sum())

    return macro_auc, weighted_auc, valid_tasks


# =============================================================================
# Main Processing
# =============================================================================

def run_robustness_test(
    pkl_files: List[str],
    Y: np.ndarray,
    task_type: str,
    n_tasks: int,
    n_repeats: int,
    test_size: float,
    base_seed: int,
    model_params: dict,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run robustness test across multiple noise levels.
    
    Parameters
    ----------
    pkl_files : List[str]
        List of pickle files with different noise levels
    Y : np.ndarray
        Labels
    task_type : str
        'classification' or 'regression'
    n_tasks : int
        Number of tasks
    n_repeats : int
        Number of random splits
    test_size : float
        Test set fraction
    base_seed : int
        Base random seed
    model_params : dict
        Model parameters
    verbose : bool
        Print progress
        
    Returns
    -------
    pd.DataFrame
        Summary results
    """
    n_samples = Y.shape[0]
    
    # Check all files exist and have correct shape
    for pkl_path in pkl_files:
        X = load_pkl_as_numpy(pkl_path).astype(np.float64, copy=False)
        if X.shape[0] != n_samples:
            raise ValueError(
                f"Inconsistent sample sizes for {pkl_path}: X={X.shape[0]}, Y={n_samples}"
            )
    
    # Storage for results
    if task_type == "regression":
        metrics_dict = {fp: [] for fp in pkl_files}
    else:
        auc_dict = {fp: [] for fp in pkl_files}
        wauc_dict = {fp: [] for fp in pkl_files}
        valid_tasks_dict = {fp: [] for fp in pkl_files}

    idx = np.arange(n_samples)

    if verbose:
        print(f"[INFO] Running {n_repeats} random splits...")

    for rep in range(n_repeats):
        rs = base_seed + rep
        idx_train, idx_test = train_test_split(
            idx, test_size=test_size, random_state=rs, shuffle=True
        )
        
        if task_type == "regression":
            Y_train = Y[idx_train]
            Y_test = Y[idx_test]
        else:
            Y_train = Y[idx_train]
            Y_test = Y[idx_test]

        for pkl_path in pkl_files:
            X = load_pkl_as_numpy(pkl_path).astype(np.float64, copy=False)
            X_train, X_test = X[idx_train], X[idx_test]

            if task_type == "regression":
                if n_tasks == 1:
                    y_train = Y_train.flatten()
                    y_test = Y_test.flatten()
                else:
                    y_train = Y_train[:, 0]
                    y_test = Y_test[:, 0]
                
                rmse = fit_predict_regression(X_train, X_test, y_train, y_test, model_params)
                metrics_dict[pkl_path].append(rmse)
            else:
                macro_auc, weighted_auc, valid_tasks = fit_predict_classification_multitask(
                    X_train, X_test, Y_train, Y_test, model_params
                )
                auc_dict[pkl_path].append(macro_auc)
                wauc_dict[pkl_path].append(weighted_auc)
                valid_tasks_dict[pkl_path].append(valid_tasks)

        if verbose and (rep + 1) % 1 == 0:
            print(f"  Completed repeat {rep + 1}/{n_repeats}")

    # Aggregate results
    rows = []
    for fp in sorted(pkl_files, key=extract_pct_from_name):
        if task_type == "regression":
            metrics = np.array(metrics_dict[fp], dtype=float)
            mean_metric = float(np.nanmean(metrics))
            std_metric = float(np.nanstd(metrics, ddof=1)) if n_repeats > 1 else 0.0
            se_metric = float(std_metric / np.sqrt(n_repeats)) if n_repeats > 1 else 0.0

            rows.append({
                "file": os.path.basename(fp),
                "pct": extract_pct_from_name(fp),
                "rmse_mean": mean_metric,
                "rmse_std": std_metric,
                "rmse_se": se_metric,
                "n_repeats": n_repeats,
            })
        else:
            aucs = np.array(auc_dict[fp], dtype=float)
            waucs = np.array(wauc_dict[fp], dtype=float)
            vts = np.array(valid_tasks_dict[fp], dtype=int)

            mean_auc = float(np.nanmean(aucs))
            sd_auc = float(np.nanstd(aucs, ddof=1)) if n_repeats > 1 else 0.0

            mean_wauc = float(np.nanmean(waucs))
            sd_wauc = float(np.nanstd(waucs, ddof=1)) if n_repeats > 1 else 0.0

            rows.append({
                "file": os.path.basename(fp),
                "pct": extract_pct_from_name(fp),
                "macro_auc_mean": mean_auc,
                "macro_auc_sd": sd_auc,
                "weighted_auc_mean": mean_wauc,
                "weighted_auc_sd": sd_wauc,
                "valid_tasks_mean": float(vts.mean()),
                "n_repeats": n_repeats,
            })

    return pd.DataFrame(rows).sort_values("pct").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(
        description='Robustness Testing Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Regression task
  python robust_test.py --dataset esol --task-type regression --pkl-files pct0.pkl pct10.pkl pct20.pkl pct40.pkl
  
  # Classification task
  python robust_test.py --dataset bace --task-type classification --pkl-files pct0.pkl pct10.pkl pct20.pkl pct40.pkl
  
  # Multi-task with custom label columns
  python robust_test.py --dataset tox21 --task-type classification --pkl-files pct0.pkl pct10.pkl pct20.pkl pct40.pkl --n-repeats 10

Note:
  This script expects pickle files containing numpy arrays with shape (n_samples, n_features).
  The file naming convention should be: pct{noise_level}.pkl (e.g., pct0.pkl, pct10.pkl)
"""
    )

    # Data arguments
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--pkl-files', type=str, nargs='+', required=True,
                        help='Pickle files with features (e.g., pct0.pkl pct10.pkl)')
    parser.add_argument('--task-type', type=str, choices=['auto', 'classification', 'regression'],
                        default='auto', help='Task type (default: auto-detect)')
    parser.add_argument('--label-cols', type=str, nargs='+', default=None,
                        help='Label column names (default: auto-detect)')

    # Evaluation arguments
    parser.add_argument('--n-repeats', type=int, default=5, help='Number of random splits')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set fraction')
    parser.add_argument('--base-seed', type=int, default=42, help='Base random seed')

    # Model arguments
    parser.add_argument('--n-estimators', type=int, default=500, help='Number of trees')
    parser.add_argument('--max-features', type=str, default='sqrt', help='Max features')
    parser.add_argument('--class-weight', type=str, default='balanced', help='Class weight')

    # Path arguments
    parser.add_argument('--data-dir', type=str, default='../data', help='Data directory')
    parser.add_argument('--pkl-dir', type=str, default='.', help='Directory containing pickle files')
    parser.add_argument('--output-dir', type=str, default='.', help='Output directory')

    args = parser.parse_args()

    # Set paths
    data_dir = Path(args.data_dir)
    pkl_dir = Path(args.pkl_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load labels
    csv_path = data_dir / f"{args.dataset}.csv"
    print(f"[INFO] Loading labels from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Get label columns
    if args.label_cols:
        label_cols = args.label_cols
    else:
        label_cols = infer_label_columns(df)

    print(f"[INFO] Label columns: {label_cols[:10]}{'...' if len(label_cols) > 10 else ''}")

    # Extract labels
    Y = df[label_cols].to_numpy(dtype=np.float64)
    n_samples, n_tasks = Y.shape
    print(f"[INFO] n_samples={n_samples}, n_tasks={n_tasks}")

    # Determine task type
    if args.task_type == 'auto':
        if n_tasks == 1:
            task_type = detect_task_type(Y[:, 0])
        else:
            task_type = "classification"
    else:
        task_type = args.task_type
    print(f"[INFO] Task type: {task_type}")

    # Prepare pickle file paths
    pkl_files = [pkl_dir / f for f in args.pkl_files]
    for fp in pkl_files:
        if not fp.exists():
            raise FileNotFoundError(f"Feature file not found: {fp}")

    # Model parameters
    if task_type == "regression":
        model_params = {
            "n_estimators": args.n_estimators,
            "random_state": args.base_seed,
            "n_jobs": -1,
            "max_features": args.max_features,
        }
    else:
        model_params = {
            "n_estimators": args.n_estimators,
            "random_state": args.base_seed,
            "n_jobs": -1,
            "max_features": args.max_features,
            "class_weight": args.class_weight,
        }

    print(f"[INFO] Model params: {model_params}")

    # Run robustness test
    results_df = run_robustness_test(
        pkl_files=[str(fp) for fp in pkl_files],
        Y=Y,
        task_type=task_type,
        n_tasks=n_tasks,
        n_repeats=args.n_repeats,
        test_size=args.test_size,
        base_seed=args.base_seed,
        model_params=model_params,
        verbose=True
    )

    # Print results
    print(f"\n{'='*70}")
    print(f"ROBUSTNESS TEST RESULTS")
    print(f"{'='*70}")
    print(f"Dataset: {args.dataset}")
    print(f"Task type: {task_type}")
    print(f"Number of repeats: {args.n_repeats}")
    print()

    if task_type == "regression":
        print(f"{'File':<15} {'Pct':>6} {'RMSE Mean':>12} {'RMSE SE':>12}")
        print("-" * 50)
        for _, r in results_df.iterrows():
            print(f"{r['file']:<15} {r['pct']:>5.0f}% {r['rmse_mean']:>12.6f} {r['rmse_se']:>12.6f}")

        output_csv = output_dir / f"rmse_summary_{args.dataset}.csv"
    else:
        print(f"{'File':<15} {'Pct':>6} {'Macro-AUC':>12} {'Weighted-AUC':>14} {'Valid Tasks':>12}")
        print("-" * 70)
        for _, r in results_df.iterrows():
            print(f"{r['file']:<15} {r['pct']:>5.0f}% "
                  f"{r['macro_auc_mean']:>12.6f} {r['weighted_auc_mean']:>14.6f} "
                  f"{int(r['valid_tasks_mean']):>12}")

        output_csv = output_dir / f"auc_summary_{args.dataset}.csv"

    # Save results
    results_df.to_csv(output_csv, index=False)
    print(f"\n[OK] Results saved to: {output_csv}")

    # Save detailed results
    detailed_results = {
        "dataset": args.dataset,
        "task_type": task_type,
        "n_samples": n_samples,
        "n_tasks": n_tasks,
        "n_repeats": args.n_repeats,
        "test_size": args.test_size,
        "model_params": model_params,
        "results": results_df.to_dict(orient="records"),
    }

    detailed_json = output_dir / f"robustness_detailed_{args.dataset}.json"
    with open(detailed_json, "w") as f:
        json.dump(detailed_results, f, indent=2)
    print(f"[OK] Detailed results saved to: {detailed_json}")


if __name__ == "__main__":
    main()
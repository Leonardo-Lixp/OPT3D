# Molecular Property Prediction with 3D Descriptors

This repository contains a comprehensive toolkit for molecular property prediction using 3D molecular descriptors and machine learning models.

## Overview

The toolkit provides:
- **3D molecular descriptors**: Distance-weighted atom pair descriptors based on quantum chemistry calculations
- **Coulomb Matrix**: Global molecular representation encoding atomic numbers and interatomic distances
- **Machine Learning Models**: Random Forest, Ensemble, and ACS Neural Network
- **Robustness Testing**: Evaluate model performance under feature noise

## Repository Structure

```
GitHub/
├── data/                                    # Dataset files (8 datasets)
│   ├── esol.csv                             # Regression: aqueous solubility
│   ├── lip.csv                              # Regression: lipophilicity  
│   ├── freesolv.csv                         # Regression: free solvation energy
│   ├── bace.csv                             # Classification: BACE-1 binding
│   ├── bbbp.csv                             # Classification: BBB permeability
│   ├── sider.csv                            # Classification: side effects (27 tasks)
│   ├── tox21.csv                            # Classification: toxicity (12 tasks)
│   └── toxcast.csv                          # Classification: toxicity (617 tasks)
│
├── calculate_descriptors/                   # 3D descriptor calculation
│   ├── calculate_descriptors.py             # Main calculation script
│   └── src/
│       └── molecular_descriptors.py         # Core descriptor module
│
├── Coulomb_Matrix/                          # Coulomb Matrix descriptors
│   └── compute_coulomb_matrix.py            # Coulomb Matrix calculator
│
├── model/                                   # Machine learning models
│   ├── RF/                                  # Random Forest
│   │   ├── bayes_rf.py                      # Hyperparameter optimization
│   │   └── evaluate_rf.py                   # Evaluation with fixed params
│   ├── ACS/                                 # ACS Neural Network
│   │   ├── bayes_acs.py                     # Hyperparameter optimization
│   │   └── evaluate_acs.py                  # Evaluation with fixed params
│   └── Ensemble/                            # Fixed-weight ensemble
│       └── ensemble.py                      # AutoGluon-based ensemble
│
└── robust/                                  # Robustness testing
    └── robust_test.py                       # Noise robustness evaluation
```

## Installation

### Requirements

```bash
# Core dependencies
pip install numpy pandas scikit-learn rdkit-pypi optuna torch

# For OpenBabel (Coulomb Matrix)
conda install -c conda-forge openbabel

# For Ensemble model (separate conda environment)
conda create -n autogluon python=3.10
conda activate autogluon
pip install autogluon.tabular
```

### Python Version
- Python >= 3.8
- Recommended: Python 3.10

## Quick Start

### 1. Calculate 3D Molecular Descriptors

```bash
cd calculate_descriptors

# Calculate for a single dataset
python calculate_descriptors.py --dataset esol --sl 0.2 --nsc 500

# Output: descriptors/esol_nsc500_sl20.pkl
```

**Parameters:**
- `--dataset`: Dataset name (esol, lip, freesolv, bace, bbbp, sider, tox21, toxcast)
- `--sl`: Scaling factor(s) for distance (default: 0.2)
- `--nsc`: Number of descriptor components (default: 500)

### 2. Train Machine Learning Models

#### Random Forest

```bash
cd model/RF

# Step 1: Hyperparameter optimization
python bayes_rf.py --dataset esol --descriptor esol_nsc500_sl20.pkl --n-trials 100

# Step 2: Evaluate with best parameters (multiple splits)
python evaluate_rf.py --dataset esol --descriptor esol_nsc500_sl20.pkl --n-accept 10
```

#### ACS Neural Network

```bash
cd model/ACS

# Step 1: Hyperparameter optimization
python bayes_acs.py --dataset esol --descriptor esol_nsc500_sl20.pkl --n-trials 100

# Step 2: Evaluate with best parameters
python evaluate_acs.py --dataset esol --descriptor esol_nsc500_sl20.pkl --n-accept 10
```

#### Ensemble Model

```bash
conda activate autogluon
cd model/Ensemble

python ensemble.py --dataset esol --descriptor esol_nsc500_sl20.pkl
```

### 3. Robustness Testing

```bash
cd robust

# Test model robustness across noise levels
python robust_test.py --dataset esol --pkl-files pct0.pkl pct10.pkl pct20.pkl pct40.pkl
```

### 4. Coulomb Matrix Descriptors

```bash
cd Coulomb_Matrix

# Compute for a dataset
python compute_coulomb_matrix.py --input ../data/esol.csv --output esol_coulomb.pkl

# Batch processing
python compute_coulomb_matrix.py --input-dir ../data --output-dir coulomb_descriptors
```

## Descriptor Details

### 3D Distance-Weighted Atom Pair Descriptors

The 3D molecular descriptors are computed using quantum chemistry calculations:

1. **3D Structure Generation**: 
   - PySCF for geometry optimization (RHF/def2-SVP)
   - OpenBabel and RDKit as fallback methods

2. **Atomic Properties** (9 properties per atom):
   - Molecular weight
   - van der Waals volume
   - Electronegativity
   - Polarizability
   - Gasteiger charge
   - Mulliken charge (from PySCF)
   - Number of hydrogen bond donors
   - Number of hydrogen bond acceptors
   - Topological polar surface area contribution

3. **Descriptor Calculation**:
   ```
   D_ij = sum_k(exp(-sl * d_ij) * w_ik * w_jk)
   ```
   - `d_ij`: Distance between atoms i and j
   - `sl`: Scaling factor (user-defined)
   - `w_ik`: Atomic property k for atom i

4. **Output**: Vector of length `9 × nsc` (default: 4500)

### Coulomb Matrix

Global molecular representation:
```
C_ij = 0.5 * Z_i^2.4           (diagonal)
C_ij = Z_i * Z_j / |R_i - R_j| (off-diagonal)
```

## Model Architectures

### Random Forest
- Bootstrap aggregation with decision trees
- Hyperparameters: n_estimators, max_depth, min_samples_split, etc.
- Supports both regression and classification

### ACS Neural Network
- Multi-task architecture with shared backbone
- Task-specific prediction heads
- Early stopping with validation loss
- Class weighting for imbalanced data

### Ensemble Model
- Fixed-weight combination of three learners:
  - NeuralNetTorch (weight: 0.562)
  - RandomForestEntr (weight: 0.375)
  - ExtraTreesEntr (weight: 0.062)
- AutoGluon-based training

## Data Splitting

All models use **scaffold-based splitting**:
- Groups molecules by Murcko scaffold
- Ensures train/test split based on molecular structure
- Avoids data leakage between similar molecules
- Maintains target train/test ratio (e.g., 8:2)

## Evaluation Metrics

### Regression
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of Determination

### Classification
- **ROC-AUC**: Area Under ROC Curve
- **PR-AUC**: Area Under Precision-Recall Curve

## Output Files

### Descriptor Calculation
- `{dataset}_nsc{nsc}_sl{sl}.pkl`: Descriptor arrays (numpy format)

### Model Training
- `best_config.json`: Optimal hyperparameters
- `params_{task}.json`: Task-specific parameters
- `test_metrics_{dataset}.csv`: Test set metrics

### Model Evaluation
- `evaluation_{dataset}.csv`: Multi-seed evaluation results
- `evaluation_details_{dataset}.json`: Detailed statistics (mean, std, SE)

## Datasets

| Dataset | Task Type | Size | Tasks |
|---------|-----------|------|-------|
| ESOL | Regression | 1,128 | 1 (solubility) |
| Lipophilicity | Regression | 4,200 | 1 (logP) |
| FreeSolv | Regression | 642 | 1 (hydration free energy) |
| BACE | Classification | 1,513 | 1 (BACE-1 binding) |
| BBBP | Classification | 2,050 | 1 (BBB permeability) |
| SIDER | Classification | 1,427 | 27 (side effects) |
| Tox21 | Classification | 7,831 | 12 (toxicity assays) |
| ToxCast | Classification | 8,597 | 617 (toxicity assays) |

## Citation

If you use this code, please cite:

```bibtex
@misc{molecular-descriptors,
  author = {Your Name},
  title = {Molecular Property Prediction with 3D Descriptors},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/molecular-descriptors}
}
```

## License

MIT License

## Contact

For questions or issues, please open a GitHub issue or contact [your email].
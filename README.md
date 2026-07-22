# Molecular Property Prediction with 3D Descriptors

This repository contains a comprehensive toolkit for molecular property prediction using 3D molecular descriptors and machine learning models.

## Overview

The toolkit provides:
- **OPT3D descriptors**: Distance-weighted atom-pair descriptors derived from three-dimensional molecular conformations and multiple atomic-property weighting schemes
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
├── processed smiles.xlsx                    # SMILES transformation log
│                                            # (original vs transformed with treatment info)
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

# Open Babel for molecular conformation generation and Coulomb Matrix calculation
conda install -c conda-forge openbabel

# For Ensemble model (separate conda environment)
conda create -n autogluon python=3.10
conda activate autogluon
pip install autogluon.tabular
```

### Python Version
- Main environment: Python 3.11.8
- AutoGluon environment: Python 3.10

### Experimental Environment

The main experiments were performed using Python 3.11.8, NumPy 1.26.4, pandas 2.3.3, scikit-learn 1.7.2, RDKit 2024.03.2, Open Babel 3.1.0, PySCF 2.7.0, Optuna 4.4.0, and PyTorch 2.6.0 with CUDA 12.4. AutoGluon experiments were performed using Python 3.10 and AutoGluon 1.3.0.

## Quick Start

### 1. Calculate 3D Molecular Descriptors

```bash
cd calculate_descriptors

# Calculate for a single dataset
python calculate_descriptors.py --dataset esol --sl 0.1 --nsc 500

# Output: descriptors/esol_nsc500_sl10.pkl
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
python bayes_rf.py --dataset esol --descriptor esol_nsc500_sl10.pkl --n-trials 100

# Step 2: Evaluate with best parameters (multiple splits)
python evaluate_rf.py --dataset esol --descriptor esol_nsc500_sl10.pkl --n-accept 10
```

#### ACS Neural Network

```bash
cd model/ACS

# Step 1: Hyperparameter optimization
python bayes_acs.py --dataset esol --descriptor esol_nsc500_sl10.pkl --n-trials 100

# Step 2: Evaluate with best parameters
python evaluate_acs.py --dataset esol --descriptor esol_nsc500_sl10.pkl --n-accept 10
```

#### Ensemble Model

```bash
conda activate autogluon
cd model/Ensemble

python ensemble.py --dataset esol --descriptor esol_nsc500_sl10.pkl
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

The 3D molecular descriptors are computed using molecular mechanics optimization:

1. **3D Structure Generation**:
   - OpenBabel for 3D conformation generation
   - UFF and MMFF optimization in RDKit
   - PySCF for quantum chemistry charge calculation (when Gasteiger charges fail)

2. **Atomic Properties** (9 properties per atom):
   - Unit weight (uniform weighting)
   - Molecular weight
   - van der Waals volume
   - Polarizability
   - Electronegativity
   - Atomic charge (Gasteiger or Mulliken)
   - Ionization potential
   - Electrotopological state (E-state)
   - Covalent radius

3. **Descriptor Calculation**:
   ```
   D_ij(s) = sin(s * sl * d_ij) / (s * sl * d_ij)
   ```
   - `d_ij`: Distance between atoms i and j
   - `sl`: Scaling factor (user-defined)
   - `s`: Component index (0 to nsc-1)

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

All models use **scaffold-based data splitting** based on Bemis–Murcko scaffolds. RF models use training and test sets, whereas ACS models use separate training, validation, and test sets. Independently generated scaffold partitions and corresponding random seeds are used for repeated evaluation.

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

| Dataset | Task Type | Original Size | Processed Size | Tasks |
|---------|-----------|---------------|----------------|-------|
| ESOL | Regression | 1,128 | 1,128 | 1 (solubility) |
| Lipophilicity | Regression | 4,200 | 4,200 | 1 (logP) |
| FreeSolv | Regression | 642 | 642 | 1 (hydration free energy) |
| BACE | Classification | 1,513 | 1,513 | 1 (BACE-1 binding) |
| BBBP | Classification | 2,050 | 2,050 | 1 (BBB permeability) |
| SIDER | Classification | 1,427 | 1,219 | 27 (side effects) |
| Tox21 | Classification | 7,831 | 7,581 | 12 (toxicity assays) |
| ToxCast | Classification | 8,597 | 6,841 | 617 (toxicity assays) |

## Citation

If you use this code, please cite:

```bibtex
@misc{molecular-descriptors,
  author = {Li Xiaopeng},
  title = {Molecular Property Prediction with 3D Descriptors},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/Leonardo-Lixp/OPT3D}
}
```

## License

MIT License

## Contact

For questions or issues, please open a GitHub issue or contact lixp39@mail2.sysu.edu.cn.

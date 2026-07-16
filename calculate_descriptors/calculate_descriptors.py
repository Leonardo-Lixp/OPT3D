#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Molecular Descriptor Calculator
===============================

This script calculates 3D molecular descriptors for datasets.

Usage (run from GitHub/calculate_descriptors directory):
    python calculate_descriptors.py --dataset esol --sl 0.2 --nsc 500
    python calculate_descriptors.py --dataset tox21 --sl 0.1 0.2 0.5 --nsc 500
    python calculate_descriptors.py --dataset all --sl 0.2 --nsc 500

Parameters:
    --dataset: Dataset name (esol, lip, freesolv, bace, bbbp, sider, tox21, toxcast, or all)
    --sl: Scaling factor(s) for distance (default: 0.2)
    --nsc: Number of descriptor components (default: 500)
    --smiles-col: Column name for SMILES in the dataset (default: smiles)
    --output-dir: Output directory for descriptor files (default: descriptors)
    --data-dir: Directory containing dataset files (default: ../data)
"""

import os
import sys
import argparse
import pickle
from pathlib import Path

import pandas as pd
import numpy as np

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from molecular_descriptors import calculate_smiles_descriptors, set_debug_mode


# Available datasets
AVAILABLE_DATASETS = [
    'esol',      # Regression: solubility
    'lip',       # Regression: lipophilicity
    'freesolv',  # Regression: free solvation energy
    'bace',      # Classification: BACE-1 binding
    'bbbp',      # Classification: BBB permeability
    'sider',     # Classification: side effects (27 tasks)
    'tox21',     # Classification: toxicity (12 tasks)
    'toxcast',   # Classification: toxicity (617 tasks)
]


def load_dataset(data_dir, dataset_name, smiles_col='smiles'):
    """
    Load dataset from CSV file.
    
    Parameters
    ----------
    data_dir : str or Path
        Directory containing dataset files
    dataset_name : str
        Name of the dataset
    smiles_col : str
        Column name for SMILES
        
    Returns
    -------
    pandas.Series
        Series of SMILES strings
    """
    csv_path = Path(data_dir) / f'{dataset_name}.csv'
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    if smiles_col not in df.columns:
        raise ValueError(f"SMILES column '{smiles_col}' not found in {csv_path}. "
                        f"Available columns: {list(df.columns)}")
    
    print(f"Loaded dataset: {dataset_name}")
    print(f"  Total molecules: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    
    return df[smiles_col]


def calculate_and_save(smiles_series, dataset_name, sl_list, nsc, output_dir, start_idx=0, end_idx=None):
    """
    Calculate descriptors and save to pickle file.
    
    Parameters
    ----------
    smiles_series : pandas.Series
        Series of SMILES strings
    dataset_name : str
        Name of the dataset
    sl_list : list of float
        Scaling factors
    nsc : int
        Number of descriptor components
    output_dir : str or Path
        Output directory
    start_idx : int
        Starting index for processing (for partial runs)
    end_idx : int
        Ending index for processing (for partial runs)
    """
    # Slice the data if needed
    if end_idx is not None:
        smiles_series = smiles_series.iloc[start_idx:end_idx]
    elif start_idx > 0:
        smiles_series = smiles_series.iloc[start_idx:]
    
    print(f"\nCalculating descriptors for {len(smiles_series)} molecules...")
    print(f"  SL values: {sl_list}")
    print(f"  NSC: {nsc}")
    
    # Calculate descriptors
    descriptors_df = calculate_smiles_descriptors(smiles_series, sl_list, nsc)
    
    if descriptors_df is None:
        print("Error: Descriptor calculation failed")
        return
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save descriptors for each SL value
    for sl in sl_list:
        col_name = f's{int(sl * 100)}'
        
        if col_name in descriptors_df.columns:
            # Extract descriptor array
            desc_array = np.array(descriptors_df[col_name])
            desc_array = desc_array.reshape((len(smiles_series), -1))
            
            # Generate output filename
            output_file = output_dir / f'{dataset_name}_nsc{nsc}_sl{int(sl*100)}.pkl'
            
            # Save to pickle file
            with open(output_file, 'wb') as f:
                pickle.dump(desc_array, f)
            
            print(f"  Saved: {output_file}")
            print(f"    Shape: {desc_array.shape}")
    
    print(f"\nDescriptor calculation complete!")


def main():
    parser = argparse.ArgumentParser(
        description='Calculate 3D molecular descriptors for datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Calculate descriptors for ESOL dataset with default parameters
  python calculate_descriptors.py --dataset esol
  
  # Calculate for multiple SL values
  python calculate_descriptors.py --dataset tox21 --sl 0.1 0.2 0.5
  
  # Calculate for all datasets
  python calculate_descriptors.py --dataset all --sl 0.2 --nsc 500
  
Available datasets: {', '.join(AVAILABLE_DATASETS)}
"""
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=AVAILABLE_DATASETS + ['all'],
        help='Dataset name (or "all" to process all datasets)'
    )
    
    parser.add_argument(
        '--sl',
        type=float,
        nargs='+',
        default=[0.2],
        help='Scaling factor(s) for distance (default: 0.2)'
    )
    
    parser.add_argument(
        '--nsc',
        type=int,
        default=500,
        help='Number of descriptor components (default: 500)'
    )
    
    parser.add_argument(
        '--smiles-col',
        type=str,
        default='smiles',
        help='Column name for SMILES (default: smiles)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='descriptors',
        help='Output directory for descriptor files (default: descriptors)'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='../data',
        help='Directory containing dataset files (default: ../data)'
    )
    
    parser.add_argument(
        '--start',
        type=int,
        default=0,
        help='Starting index for processing (default: 0)'
    )
    
    parser.add_argument(
        '--end',
        type=int,
        default=None,
        help='Ending index for processing (default: None, process all)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    args = parser.parse_args()
    
    # Set debug mode
    set_debug_mode(args.debug)
    
    # Determine which datasets to process
    if args.dataset == 'all':
        datasets = AVAILABLE_DATASETS
    else:
        datasets = [args.dataset]
    
    # Process each dataset
    for dataset_name in datasets:
        try:
            # Load dataset
            smiles_series = load_dataset(args.data_dir, dataset_name, args.smiles_col)
            
            # Calculate and save descriptors
            calculate_and_save(
                smiles_series,
                dataset_name,
                args.sl,
                args.nsc,
                args.output_dir,
                args.start,
                args.end
            )
            
        except FileNotFoundError as e:
            print(f"Error: {e}")
            continue
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            raise


if __name__ == '__main__':
    main()
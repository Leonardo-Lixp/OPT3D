#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Coulomb Matrix Descriptor Calculator
=====================================

This script computes Coulomb Matrix descriptors for molecules from SMILES strings.
The Coulomb Matrix is a global molecular representation that encodes atomic numbers
and interatomic distances in a matrix format.

Features:
- Support for multiple input formats (CSV file with SMILES, single SMILES)
- 3D structure generation using OpenBabel (with RDKit fallback)
- Molecular mechanics optimization (UFF and MMFF)
- Configurable maximum atom count
- Batch processing with progress display

Coulomb Matrix Definition:
    C_ij = 0.5 * Z_i^2.4                    (diagonal, i = j)
    C_ij = Z_i * Z_j / |R_i - R_j|          (off-diagonal, i != j)

Where:
    Z_i: atomic number of atom i
    R_i: 3D coordinates of atom i

Usage:
    # Compute for dataset
    python compute_coulomb_matrix.py --input smiles.csv --output coulomb_matrices.pkl --smiles-col smiles

    # Compute for all SMILES files in a directory
    python compute_coulomb_matrix.py --input-dir ../data --output-dir descriptors

    # Compute for single SMILES (debug)
    python compute_coulomb_matrix.py --smiles "CCO" --max-atoms 20

Requirements:
    - rdkit
    - openbabel (pybel)
    - numpy
    - pandas
"""

import os
import sys
import pickle
import argparse
import warnings
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem

try:
    from openbabel import pybel
    HAS_PYBEL = True
except ImportError:
    HAS_PYBEL = False
    warnings.warn("OpenBabel not installed. Will use RDKit for 3D generation.")


# =============================================================================
# Coulomb Matrix Computation
# =============================================================================

def compute_coulomb_matrix(mol, smiles: str, max_atoms: Optional[int] = None) -> np.ndarray:
    """
    Compute Coulomb Matrix for a single molecule.
    
    Parameters
    ----------
    mol : rdkit.Chem.Mol
        RDKit molecule object (without 3D coordinates)
    smiles : str
        SMILES string (used for OpenBabel 3D generation)
    max_atoms : int, optional
        Maximum number of atoms for padding. If None, use actual atom count.
        
    Returns
    -------
    np.ndarray
        Coulomb matrix (max_atoms x max_atoms)
    """
    # Generate 3D structure
    mol = generate_3d_structure(mol, smiles)
    if mol is None:
        raise RuntimeError(f"Failed to generate 3D structure for {smiles}")
    
    # Get atom information
    n_atoms = mol.GetNumAtoms()
    
    # Get atomic coordinates and numbers
    coords = np.array([mol.GetConformer().GetAtomPosition(i) for i in range(n_atoms)])
    atomic_numbers = np.array([mol.GetAtomWithIdx(i).GetAtomicNum() for i in range(n_atoms)])
    
    # Determine matrix size
    matrix_size = max_atoms if max_atoms is not None else n_atoms
    
    # Initialize matrix
    coulomb_matrix = np.zeros((matrix_size, matrix_size), dtype=np.float32)
    
    # Compute Coulomb Matrix elements
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i == j:
                # Diagonal: 0.5 * Z_i^2.4
                coulomb_matrix[i, i] = 0.5 * (atomic_numbers[i] ** 2.4)
            else:
                # Off-diagonal: Z_i * Z_j / R_ij
                rij = np.linalg.norm(coords[i] - coords[j])
                if rij > 1e-10:
                    coulomb_matrix[i, j] = (atomic_numbers[i] * atomic_numbers[j]) / rij
    
    return coulomb_matrix


def generate_3d_structure(mol, smiles: str, optimize: bool = True) -> Optional[Chem.Mol]:
    """
    Generate 3D structure for a molecule.
    
    Priority:
    1. OpenBabel (pybel) - more robust 3D generation
    2. RDKit - fallback method
    
    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Input molecule (without 3D)
    smiles : str
        SMILES string
    optimize : bool
        Whether to optimize geometry with UFF and MMFF
        
    Returns
    -------
    rdkit.Chem.Mol or None
        Molecule with 3D coordinates
    """
    # Try OpenBabel first
    if HAS_PYBEL:
        try:
            pybel_mol = pybel.readstring("smi", smiles)
            pybel_mol.addh()
            pybel_mol.make3D()
            
            # Convert back to RDKit
            output_str = pybel_mol.write("mol")
            mol = Chem.MolFromMolBlock(output_str, removeHs=False)
            
            if optimize:
                try:
                    AllChem.UFFOptimizeMolecule(mol)
                except Exception:
                    pass
                try:
                    AllChem.MMFFOptimizeMolecule(mol)
                except Exception:
                    pass
            
            return mol
            
        except Exception as e:
            pass  # Fall back to RDKit
    
    # Use RDKit
    try:
        mol = Chem.AddHs(mol)
        result = AllChem.EmbedMolecule(mol, randomSeed=0xf00d)
        
        if result < 0:
            # Try random coordinates
            result = AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42)
        
        if result < 0:
            # Try ETKDG
            result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3(), randomSeed=42)
        
        if result < 0:
            return None
        
        if optimize:
            try:
                AllChem.UFFOptimizeMolecule(mol)
            except Exception:
                pass
            try:
                AllChem.MMFFOptimizeMolecule(mol)
            except Exception:
                pass
        
        return mol
        
    except Exception as e:
        return None


def smiles_to_coulomb_matrix(smiles: str, max_atoms: Optional[int] = None) -> Optional[np.ndarray]:
    """
    Convert SMILES to Coulomb Matrix.
    
    Parameters
    ----------
    smiles : str
        SMILES string
    max_atoms : int, optional
        Maximum number of atoms
        
    Returns
    -------
    np.ndarray or None
        Coulomb matrix, or None if conversion fails
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    try:
        return compute_coulomb_matrix(mol, smiles, max_atoms)
    except Exception as e:
        return None


# =============================================================================
# Batch Processing
# =============================================================================

def compute_max_atoms(smiles_list: List[str]) -> Tuple[int, List[int]]:
    """
    Determine maximum atom count across all molecules.
    
    Parameters
    ----------
    smiles_list : List[str]
        List of SMILES strings
        
    Returns
    -------
    tuple
        (max_atoms, valid_indices)
    """
    max_atoms = 0
    valid_indices = []
    
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mol = Chem.AddHs(mol)
            n_atoms = mol.GetNumAtoms()
            if n_atoms > max_atoms:
                max_atoms = n_atoms
            valid_indices.append(i)
    
    return max_atoms, valid_indices


def process_dataset(
    smiles_list: List[str],
    max_atoms: Optional[int] = None,
    verbose: bool = True
) -> Tuple[np.ndarray, List[int]]:
    """
    Process a dataset of SMILES strings.
    
    Parameters
    ----------
    smiles_list : List[str]
        List of SMILES strings
    max_atoms : int, optional
        Maximum atom count. If None, auto-detect.
    verbose : bool
        Print progress
        
    Returns
    -------
    tuple
        (coulomb_matrices, failed_indices)
    """
    n_samples = len(smiles_list)
    
    # Auto-detect max_atoms if needed
    if max_atoms is None:
        if verbose:
            print("Analyzing molecule sizes...")
        max_atoms, valid_indices = compute_max_atoms(smiles_list)
        if verbose:
            print(f"  Max atoms: {max_atoms}")
            print(f"  Valid molecules: {len(valid_indices)}/{n_samples}")
    else:
        valid_indices = list(range(n_samples))
    
    # Initialize output array
    feature_length = max_atoms * max_atoms
    coulomb_matrices = np.zeros((n_samples, feature_length), dtype=np.float32)
    failed_indices = []
    
    # Process each molecule
    if verbose:
        print(f"Computing Coulomb matrices ({n_samples} molecules)...")
    
    for i, idx in enumerate(valid_indices):
        smiles = smiles_list[idx]
        cm = smiles_to_coulomb_matrix(smiles, max_atoms=max_atoms)
        
        if cm is not None:
            coulomb_matrices[idx] = cm.flatten()
        else:
            failed_indices.append(idx)
        
        # Progress
        if verbose and ((i + 1) % 100 == 0 or (i + 1) == len(valid_indices)):
            print(f"  Progress: {i+1}/{len(valid_indices)} ({(i+1)/len(valid_indices)*100:.1f}%)")
    
    return coulomb_matrices, failed_indices


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Compute Coulomb Matrix descriptors from SMILES',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compute for a single dataset
  python compute_coulomb_matrix.py --input smiles.csv --output coulomb_matrices.pkl
  
  # Specify SMILES column
  python compute_coulomb_matrix.py --input data.csv --smiles-col SMILES --output descriptors.pkl
  
  # Process all CSV files in a directory
  python compute_coulomb_matrix.py --input-dir ../data --output-dir coulomb_descriptors
  
  # Compute for a single SMILES (debug)
  python compute_coulomb_matrix.py --smiles "CCO" --max-atoms 20
"""
    )
    
    # Input options
    parser.add_argument('--input', type=str, help='Input CSV file with SMILES')
    parser.add_argument('--input-dir', type=str, help='Input directory with CSV files')
    parser.add_argument('--smiles', type=str, help='Single SMILES string (for testing)')
    parser.add_argument('--smiles-col', type=str, default='smiles', help='SMILES column name')
    
    # Output options
    parser.add_argument('--output', type=str, help='Output pickle file')
    parser.add_argument('--output-dir', type=str, default='.', help='Output directory')
    
    # Processing options
    parser.add_argument('--max-atoms', type=int, default=None, help='Maximum atom count (auto if not set)')
    parser.add_argument('--no-optimize', action='store_true', help='Skip geometry optimization')
    
    args = parser.parse_args()
    
    # Single SMILES mode
    if args.smiles:
        print(f"Computing Coulomb matrix for: {args.smiles}")
        max_atoms = args.max_atoms or 20
        cm = smiles_to_coulomb_matrix(args.smiles, max_atoms=max_atoms)
        
        if cm is not None:
            print(f"  Shape: {cm.shape}")
            print(f"  Flattened: {cm.flatten().shape}")
            print(f"  Diagonal (first 5): {np.diag(cm)[:5]}")
        else:
            print("  Failed to compute")
        return
    
    # Batch mode
    if args.input:
        # Single file
        input_file = Path(args.input)
        output_file = Path(args.output) if args.output else input_file.with_suffix('.pkl')
        
        print(f"Loading: {input_file}")
        df = pd.read_csv(input_file)
        
        if args.smiles_col not in df.columns:
            raise ValueError(f"Column '{args.smiles_col}' not found in {input_file}")
        
        smiles_list = df[args.smiles_col].tolist()
        
        coulomb_matrices, failed_indices = process_dataset(
            smiles_list, max_atoms=args.max_atoms, verbose=True
        )
        
        # Save
        output_file = Path(args.output_dir) / output_file.name
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'wb') as f:
            pickle.dump(coulomb_matrices, f)
        
        print(f"\nResults:")
        print(f"  Output: {output_file}")
        print(f"  Shape: {coulomb_matrices.shape}")
        print(f"  Failed: {len(failed_indices)}/{len(smiles_list)}")
        
    elif args.input_dir:
        # Directory mode
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        csv_files = list(input_dir.glob('*.csv'))
        print(f"Found {len(csv_files)} CSV files in {input_dir}")
        
        for csv_file in csv_files:
            print(f"\n{'='*60}")
            print(f"Processing: {csv_file.name}")
            print(f"{'='*60}")
            
            df = pd.read_csv(csv_file)
            
            if args.smiles_col not in df.columns:
                print(f"  Warning: Column '{args.smiles_col}' not found, skipping")
                continue
            
            smiles_list = df[args.smiles_col].tolist()
            
            coulomb_matrices, failed_indices = process_dataset(
                smiles_list, max_atoms=args.max_atoms, verbose=True
            )
            
            # Save
            output_file = output_dir / csv_file.name.replace('.csv', '_coulomb.pkl')
            with open(output_file, 'wb') as f:
                pickle.dump(coulomb_matrices, f)
            
            print(f"\n  Output: {output_file}")
            print(f"  Shape: {coulomb_matrices.shape}")
            print(f"  Failed: {len(failed_indices)}/{len(smiles_list)}")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
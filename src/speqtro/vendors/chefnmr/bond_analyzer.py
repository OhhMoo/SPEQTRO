# Vendored from chefnmr/src/evaluation/bond_analyzer.py — inference-only.
# Converts predicted atom coordinates + types to SMILES via RDKit bond determination.

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple, List, Optional, Dict, Any

import numpy as np
import torch
from rdkit import RDLogger, Chem
from rdkit.Chem import rdDetermineBonds

from speqtro.vendors.chefnmr import data_utils


def get_smiles_from_mol(
    mol: Chem.Mol, remove_stereo: bool
) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract canonical SMILES and largest-fragment SMILES from an RDKit molecule.
    Returns (full_smiles, largest_fragment_smiles) or (None, None) on error.
    """
    try:
        mol_no_h = Chem.RemoveHs(mol)
        if remove_stereo:
            Chem.RemoveStereochemistry(mol_no_h)

        full_smiles = Chem.MolToSmiles(mol_no_h)
        full_smiles = Chem.CanonSmiles(full_smiles)

        fragments = Chem.rdmolops.GetMolFrags(
            mol_no_h, asMols=True, sanitizeFrags=True
        )
        if len(fragments) > 1:
            largest_fragment = max(fragments, key=lambda m: m.GetNumAtoms())
            Chem.SanitizeMol(largest_fragment)
            largest_fragment_smiles = Chem.MolToSmiles(largest_fragment)
            largest_fragment_smiles = Chem.CanonSmiles(largest_fragment_smiles)
            return (None, largest_fragment_smiles)
        else:
            return (full_smiles, full_smiles)
    except Exception:
        return (None, None)


def _atom_features_to_smiles_one(
    args: Tuple,
) -> Tuple[int, Optional[str], Optional[str]]:
    """
    Worker function: converts one molecule's atom features to SMILES.
    Uses RDKit DetermineBonds on the 3D coordinates.
    """
    (
        molecule_index,
        atom_coords,
        atom_one_hot,
        atom_mask,
        dataset_name,
        remove_h,
        remove_stereo,
        atom_decoder,
        timeout_seconds,
    ) = args

    molecule = data_utils.Molecule(
        atom_coords=atom_coords,
        atom_one_hot=atom_one_hot,
        atom_mask=atom_mask,
        atom_decoder=atom_decoder,
        remove_h=remove_h,
        collapse=True,
    )

    if remove_h:
        raise NotImplementedError(
            "remove_h=True is not supported for SMILES reconstruction."
        )

    try:
        rdkit_mol = molecule.to_rdkit_molecule()
        rdDetermineBonds.DetermineBonds(rdkit_mol)
        reconstructed_smiles, largest_fragment_smiles = get_smiles_from_mol(
            rdkit_mol, remove_stereo
        )
    except Exception:
        reconstructed_smiles = None
        largest_fragment_smiles = None

    return molecule_index, reconstructed_smiles, largest_fragment_smiles


def atom_features_to_smiles(
    dataset_name: str,
    atom_features: Dict[str, Any],
    multiplicity: int,
    remove_h: bool,
    remove_stereo: bool,
    atom_decoder: List[str],
    timeout_seconds: int = 0,
    num_workers: Optional[int] = None,
) -> Tuple[List[Optional[str]], List[Optional[str]]]:
    """
    Convert a batch of atom features into SMILES strings (parallel).

    Args:
        dataset_name: Dataset name (for logging).
        atom_features: Dict with 'atom_coords', 'atom_one_hot', 'atom_mask'.
        multiplicity: Number of samples per molecule.
        remove_h: Whether H atoms were removed.
        remove_stereo: Whether to strip stereochemistry from output.
        atom_decoder: Index-to-atom-symbol mapping.
        timeout_seconds: Unused (kept for API compat).
        num_workers: Parallel workers (default: cpu_count).

    Returns:
        (full_smiles_list, largest_fragment_smiles_list)
    """

    def to_numpy(data):
        if isinstance(data, torch.Tensor):
            return data.cpu().detach().numpy()
        return np.array(data)

    atom_coords_list = to_numpy(atom_features["atom_coords"])
    atom_one_hot_list = to_numpy(atom_features["atom_one_hot"])
    atom_mask_list = to_numpy(atom_features["atom_mask"])

    atom_one_hot_list = np.repeat(atom_one_hot_list, multiplicity, axis=0)
    atom_mask_list = np.repeat(atom_mask_list, multiplicity, axis=0)

    RDLogger.DisableLog("rdApp.*")

    args_list = [
        (
            i,
            coords,
            one_hot,
            mask,
            dataset_name,
            remove_h,
            remove_stereo,
            atom_decoder,
            timeout_seconds,
        )
        for i, (coords, one_hot, mask) in enumerate(
            zip(atom_coords_list, atom_one_hot_list, atom_mask_list)
        )
    ]

    if num_workers is None:
        num_workers = os.cpu_count() or 1

    reconstructed_smiles_list = [None] * len(args_list)
    reconstructed_largest_fragment_smiles_list = [None] * len(args_list)

    try:
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            chunksize = max(1, len(args_list) // (num_workers * 4))
            results = list(
                pool.map(_atom_features_to_smiles_one, args_list, chunksize=chunksize)
            )
            for i, reconstructed_smiles, largest_frag_smiles in results:
                reconstructed_smiles_list[i] = reconstructed_smiles
                reconstructed_largest_fragment_smiles_list[i] = largest_frag_smiles
    except Exception:
        raise

    return reconstructed_smiles_list, reconstructed_largest_fragment_smiles_list

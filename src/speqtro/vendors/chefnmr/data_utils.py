# Vendored from chefnmr/src/data/utils.py — inference-only.
# Molecule class + helper functions for SMILES handling.

import numpy as np
from rdkit import Chem
import torch
from typing import Union, Dict, List, Optional


def n_atoms_in_smiles(smiles: str, remove_h: bool = False) -> int:
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return 0
    if not remove_h:
        m = Chem.AddHs(m)
    return m.GetNumAtoms()


def canonicalize(
    smiles: str,
    remove_stereo: bool = False,
) -> Optional[str]:
    """Canonicalize a SMILES string using RDKit."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        isomeric_smiles = not remove_stereo
        smiles_out = Chem.MolToSmiles(mol, isomericSmiles=isomeric_smiles)
        if smiles_out is None:
            return None
        return Chem.CanonSmiles(smiles_out)
    except Exception:
        return None


class Molecule:
    """
    A molecule in 3D space: atom coordinates, types (one-hot), and mask.
    Handles conversion to RDKit mol objects.
    """

    def __init__(
        self,
        atom_coords: Union[np.ndarray, torch.Tensor],
        atom_one_hot: Union[np.ndarray, torch.Tensor],
        atom_mask: Union[np.ndarray, torch.Tensor],
        atom_decoder: Union[Dict[int, str], List[str]],
        remove_h: bool,
        collapse: bool = False,
    ):
        if isinstance(atom_coords, np.ndarray):
            self.atom_coords = atom_coords.astype(np.float64)
        elif isinstance(atom_coords, torch.Tensor):
            self.atom_coords = atom_coords.to(dtype=torch.float64)
        else:
            raise TypeError("atom_coords must be numpy.ndarray or torch.Tensor")

        self.atom_coords = self.atom_coords.reshape(-1, 3)

        self.atom_one_hot = atom_one_hot
        if isinstance(atom_one_hot, torch.Tensor):
            self.atom_types = (
                torch.argmax(atom_one_hot, dim=1).detach().cpu().numpy().astype(int)
            )
        else:
            self.atom_types = np.argmax(atom_one_hot, axis=1).astype(int)

        self.atom_decoder = list(atom_decoder)

        if isinstance(atom_mask, np.ndarray):
            self.atom_mask = atom_mask.astype(bool)
        elif isinstance(atom_mask, torch.Tensor):
            self.atom_mask = atom_mask.to(dtype=torch.bool)
        else:
            raise TypeError("atom_mask must be numpy.ndarray or torch.Tensor")

        self.remove_h = remove_h

        if not self.remove_h:
            assert (
                "H" in self.atom_decoder
            ), "remove_h=False - H atom type must be present in the atom decoder."

        if collapse:
            self.collapse()

    def collapse(self):
        """Remove masked (invalid) atoms in-place."""
        if isinstance(self.atom_mask, torch.Tensor):
            n_atoms = self.atom_mask.sum().item()
        else:
            n_atoms = np.sum(self.atom_mask)

        self.atom_coords = (
            self.atom_coords[self.atom_mask]
            if self.atom_coords is not None
            else None
        )
        self.atom_one_hot = (
            self.atom_one_hot[self.atom_mask]
            if self.atom_one_hot is not None
            else None
        )

        mask_np = (
            self.atom_mask.detach().cpu().numpy()
            if isinstance(self.atom_mask, torch.Tensor)
            else self.atom_mask
        )
        self.atom_types = (
            self.atom_types[mask_np].astype(int)
            if self.atom_types is not None
            else None
        )

        self.atom_mask = np.ones(int(n_atoms), dtype=bool)

    def to_rdkit_molecule(self) -> Chem.Mol:
        """Convert to an RDKit molecule (atoms + 3D coords, no bonds)."""
        mol = Chem.RWMol()

        for atom_idx in self.atom_types:
            atom_symbol = self.atom_decoder[atom_idx]
            a = Chem.Atom(atom_symbol)
            mol.AddAtom(a)

        coords = self.atom_coords
        if isinstance(coords, torch.Tensor):
            coords = coords.detach().cpu().numpy()

        conf = Chem.Conformer(len(coords))
        for i, pos in enumerate(coords):
            conf.SetAtomPosition(i, pos.astype(np.float64))

        mol.AddConformer(conf)
        return mol

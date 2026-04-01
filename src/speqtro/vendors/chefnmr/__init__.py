# Vendored ChefNMR — inference-only subset.
# Original: Xiong Z, Zhang Y, Alauddin F, et al.
# "Atomic Diffusion Models for Small Molecule Structure Elucidation from NMR Spectra"
# NeurIPS 2025  |  https://arxiv.org/abs/2512.03127
#
# All imports are lazy — heavy dependencies (torch, lightning, omegaconf)
# are only loaded when actually accessed.


def __getattr__(name):
    if name == "NMRTo3DStructureElucidation":
        from speqtro.vendors.chefnmr.model import NMRTo3DStructureElucidation
        return NMRTo3DStructureElucidation
    if name in ("atom_features_to_smiles", "get_smiles_from_mol"):
        from speqtro.vendors.chefnmr import bond_analyzer
        return getattr(bond_analyzer, name)
    if name in ("Molecule", "canonicalize", "n_atoms_in_smiles"):
        from speqtro.vendors.chefnmr import data_utils
        return getattr(data_utils, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "NMRTo3DStructureElucidation",
    "atom_features_to_smiles",
    "get_smiles_from_mol",
    "Molecule",
    "canonicalize",
    "n_atoms_in_smiles",
]

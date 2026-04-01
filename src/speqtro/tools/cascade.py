"""
CASCADE 2.0 wrapper for ¹³C NMR shift prediction.

CASCADE 2.0 (Paton Lab, Colorado State University) is a 3D-GNN that predicts
¹³C chemical shifts from molecular geometry with ~0.73 ppm MAE.

Architecture: PAiNN (Physics-Aware Equivariant Interaction Networks)
Framework: TensorFlow 2.11 + KGCNN 2.2.1 + nfp (vendored)
Input: SMILES → RDKit 3D conformer (MMFF94 optimized)
Output: per-carbon atom ¹³C shifts in ppm

Configuration:
  Set cascade.model_dir in speqtro config, or set SPEQ_CASCADE_DIR env var,
  pointing to the cascade-2.0/models/Predict_SMILES_FF directory.
"""

import logging
import os
from pathlib import Path
from typing import Optional

from speqtro.tools import registry

logger = logging.getLogger("speqtro.tools.cascade")

# Lazy-cached model and preprocessor
_cascade_model = None
_cascade_preprocessor = None
_cascade_model_dir: Optional[Path] = None


# ── Config helpers ──────────────────────────────────────────────────────────

def _get_model_dir() -> Optional[Path]:
    """Find the CASCADE model directory from config or environment."""
    try:
        from speqtro.agent.config import Config
        cfg = Config.load()
        d = cfg.get("cascade.model_dir")
        if d:
            return Path(d)
    except Exception:
        pass

    env_dir = os.environ.get("SPEQ_CASCADE_DIR")
    if env_dir:
        return Path(env_dir)

    # Auto-detect: ~/.speqtro/models/cascade2/ (populated by `speqtro fetch-weights`)
    home_dir = Path.home() / ".speqtro" / "models" / "cascade2"
    if home_dir.exists() and (home_dir / "best_model.h5").exists():
        return home_dir

    repo_root = Path(__file__).resolve().parents[4]
    candidates = [
        repo_root.parent / "cascade-2.0" / "models" / "Predict_SMILES_FF",
        repo_root.parent / "CASCADE-2.0" / "models" / "Predict_SMILES_FF",
    ]
    for c in candidates:
        if c.exists() and (c / "best_model.h5").exists():
            return c

    return None


def _get_model_and_preprocessor(model_dir: Path):
    """Lazy-load CASCADE model and preprocessor, caching for reuse."""
    global _cascade_model, _cascade_preprocessor, _cascade_model_dir

    if _cascade_model is not None and _cascade_model_dir == model_dir:
        return _cascade_model, _cascade_preprocessor

    import pickle

    # Force TF to CPU — CASCADE doesn't need GPU
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")

    # Install pickle aliases so the old module paths resolve
    from speqtro.vendors.cascade import install_pickle_aliases
    install_pickle_aliases()

    # Load preprocessor
    preprocessor_path = model_dir / "preprocessor_orig.p"
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"CASCADE preprocessor not found: {preprocessor_path}")
    with open(preprocessor_path, "rb") as f:
        input_data = pickle.load(f)
    preprocessor = input_data["preprocessor"]

    # Import kgcnn custom objects (must be imported before load_model)
    from kgcnn.layers.casting import ChangeTensorType  # noqa: F401
    from kgcnn.layers.conv.painn_conv import PAiNNUpdate, EquivariantInitialize, PAiNNconv  # noqa: F401
    from kgcnn.layers.geom import (  # noqa: F401
        NodeDistanceEuclidean, EdgeDirectionNormalized,
        CosCutOffEnvelope, NodePosition, ShiftPeriodicLattice,
    )
    from kgcnn.layers.modules import LazyAdd, OptionalInputEmbedding  # noqa: F401
    from kgcnn.layers.mlp import GraphMLP, MLP  # noqa: F401
    from kgcnn.layers.norm import GraphLayerNormalization, GraphBatchNormalization  # noqa: F401

    from keras.models import load_model
    model_path = model_dir / "best_model.h5"
    if not model_path.exists():
        raise FileNotFoundError(f"CASCADE model weights not found: {model_path}")

    model = load_model(str(model_path))

    _cascade_model = model
    _cascade_preprocessor = preprocessor
    _cascade_model_dir = model_dir

    return model, preprocessor


# ── Direct inference ─────────────────────────────────────────────────────────

def _predict_direct(smiles_list: list[str], model_dir: Path) -> list[dict]:
    """
    Run CASCADE inference in-process using vendored code.
    Returns list of dicts: [{"smiles": str, "atom_id": int, "shift_ppm": float}, ...]
    """
    import numpy as np
    import tensorflow as tf
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from speqtro.vendors.cascade.nfp.preprocessing.sequence import GraphSequence

    model, preprocessor = _get_model_and_preprocessor(model_dir)

    # Build 3D conformers
    def _get_carbon_indices(mol):
        return [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6]

    mols, valid_smiles, atom_indices = [], [], []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            logger.warning("Invalid SMILES, skipping: %s", smi)
            continue
        mol = Chem.AddHs(mol)
        result = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        if result != 0:
            AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol)
        c_idx = _get_carbon_indices(mol)
        if not c_idx:
            continue
        mols.append(mol)
        valid_smiles.append(smi)
        atom_indices.append(c_idx)

    if not mols:
        return []

    import pandas as pd
    inp_df = pd.DataFrame({"Mol": mols, "atom_index": atom_indices})

    # Preprocessing
    def _compute_stacked_offsets(sizes, repeats):
        return np.repeat(np.cumsum(np.hstack([0, sizes[:-1]])), repeats)

    def ragged_const(inp_arr):
        return tf.ragged.constant(np.expand_dims(inp_arr, axis=0), ragged_rank=1)

    class RBFSequence(GraphSequence):
        def process_data(self, batch_data):
            offset = _compute_stacked_offsets(
                batch_data["n_pro"], batch_data["n_atom"])
            offset = np.where(batch_data["atom_index"] >= 0, offset, 0)
            batch_data["atom_index"] += offset
            for feature in ["node_attributes", "node_coordinates", "edge_indices",
                             "atom_index", "n_pro"]:
                batch_data[feature] = ragged_const(batch_data[feature])
            for key in ["n_atom", "n_bond", "distance", "bond", "node_graph_indices"]:
                del batch_data[key]
            return batch_data

    def mol_iter(df):
        for _, row in df.iterrows():
            yield row["Mol"], row["atom_index"]

    inputs_test = preprocessor.predict(mol_iter(inp_df))
    test_sequence = RBFSequence(inputs_test, batch_size=32)

    raw_preds = []
    for x in test_sequence:
        raw_preds.extend(model(x).numpy().flatten())

    # Rescale: prediction * 50.484337 + 99.798111 → ppm
    ppm_values = [round(v * 50.484337 + 99.798111, 2) for v in raw_preds]

    results = []
    pred_idx = 0
    for smi, c_indices in zip(valid_smiles, atom_indices):
        for atom_id in c_indices:
            if pred_idx < len(ppm_values):
                results.append({
                    "smiles": smi,
                    "atom_id": int(atom_id),
                    "shift_ppm": ppm_values[pred_idx],
                })
                pred_idx += 1

    return results


# ── speQ tool registration ───────────────────────────────────────────────────

@registry.register(
    name="nmr.predict_c13_cascade",
    description=(
        "Predict ¹³C NMR chemical shifts using CASCADE 2.0 (Paton Lab). "
        "Input: SMILES string(s). Output: per-carbon atom shift predictions in ppm. "
        "Uses 3D-GNN on MMFF-optimized geometry. ~0.73 ppm MAE on experimental data."
    ),
    category="nmr",
    parameters={
        "smiles": {
            "type": "string",
            "description": "SMILES string of the molecule (or comma-separated list)",
        },
    },
)
def predict_c13_cascade(smiles: str) -> dict:
    """
    Predict ¹³C shifts using CASCADE 2.0.

    Args:
        smiles: SMILES string, or comma-separated list of SMILES.

    Returns:
        Dict with 'predictions' (list of per-atom shifts), 'model', 'citations'.
    """
    smiles_list = [s.strip() for s in smiles.split(",") if s.strip()]
    if not smiles_list:
        return {"error": "No valid SMILES provided."}

    model_dir = _get_model_dir()
    if model_dir is None:
        return {
            "error": (
                "CASCADE model directory not found. "
                "Set config key 'cascade.model_dir' to the path of "
                "cascade-2.0/models/Predict_SMILES_FF, "
                "or set the SPEQ_CASCADE_DIR environment variable."
            )
        }

    if not (model_dir / "best_model.h5").exists():
        return {
            "error": (
                f"CASCADE model weights not found at {model_dir / 'best_model.h5'}. "
                "Ensure the pretrained weights are present in the model directory."
            )
        }

    try:
        predictions = _predict_direct(smiles_list, model_dir)
    except Exception as e:
        logger.exception("CASCADE prediction failed")
        return {"error": f"CASCADE inference error: {e}"}

    # Group by SMILES for readability
    grouped: dict[str, list] = {}
    for p in predictions:
        grouped.setdefault(p["smiles"], []).append({
            "atom_id": p["atom_id"],
            "shift_ppm": p["shift_ppm"],
        })

    return {
        "predictions": grouped,
        "model": "CASCADE 2.0 (PAiNN, MMFF geometry)",
        "citations": [
            {
                "title": "CASCADE 2.0: Real-time Prediction of 13C NMR Shifts with sub-ppm Accuracy",
                "authors": "Bhadauria A, Feng Z, Popescu M, Paton R",
                "url": "https://nova.chem.colostate.edu/v2/cascade/",
                "note": "Model weights: pretrained, MIT license",
            }
        ],
        "method": "3D-GNN (PAiNN) on MMFF94-optimized molecular geometry",
        "units": "ppm (CDCl3 reference implied)",
    }

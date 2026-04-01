"""
CASCADE 1.0 wrapper for ¹H NMR shift prediction.

CASCADE 1.0 (Paton Lab) is a message-passing GNN that predicts per-hydrogen
¹H chemical shifts from MMFF94s conformer ensembles with Boltzmann weighting.

Architecture: Message-passing GNN with EdgeNetwork (old TF1/Keras style, ported to TF2)
Framework: TensorFlow 2 + vendored nfp layers
Input: SMILES → RDKit MMFF94s conformer ensemble (nc=50, efilter=10.0, rmspost=0.5)
Output: per-hydrogen atom ¹H shifts in ppm, Boltzmann-averaged over conformers

Configuration:
  Set cascade1.model_dir in speqtro config, or set SPEQ_CASCADE1_DIR env var,
  pointing to the folder containing best_model.hdf5 and preprocessor.p.
"""

import logging
import math
import os
from pathlib import Path
from typing import Optional

from speqtro.tools import registry

logger = logging.getLogger("speqtro.tools.cascade1")

# Lazy-cached model and preprocessor
_cascade1_model = None
_cascade1_preprocessor = None
_cascade1_model_dir: Optional[Path] = None


# ── Config helpers ──────────────────────────────────────────────────────────

def _get_model_dir() -> Optional[Path]:
    """Find the CASCADE 1.0 model directory from config or environment."""
    try:
        from speqtro.agent.config import Config
        cfg = Config.load()
        d = cfg.get("cascade1.model_dir")
        if d:
            return Path(d)
    except Exception:
        pass

    env_dir = os.environ.get("SPEQ_CASCADE1_DIR")
    if env_dir:
        return Path(env_dir)

    # Auto-detect: ~/.speqtro/models/cascade1/ (populated by `speqtro fetch-weights`)
    home_dir = Path.home() / ".speqtro" / "models" / "cascade1"
    if home_dir.exists() and (home_dir / "best_model.hdf5").exists():
        return home_dir

    # Auto-detect: look relative to the repo root
    repo_root = Path(__file__).resolve().parents[4]
    candidates = [
        repo_root.parent / "CASCADE" / "cascade-Jupyternotebook-SMILES" / "models" / "cascade" / "trained_model",
        repo_root.parent / "cascade" / "cascade-Jupyternotebook-SMILES" / "models" / "cascade" / "trained_model",
    ]
    for c in candidates:
        if c.exists() and (c / "best_model.hdf5").exists():
            return c

    return None


def _get_model_and_preprocessor(model_dir: Path):
    """Lazy-load CASCADE 1.0 model and preprocessor, caching for reuse."""
    global _cascade1_model, _cascade1_preprocessor, _cascade1_model_dir

    if _cascade1_model is not None and _cascade1_model_dir == model_dir:
        return _cascade1_model, _cascade1_preprocessor

    import pickle

    # Force TF to CPU — CASCADE 1.0 doesn't need GPU
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")

    # Install pickle aliases so the old nfp.* module paths resolve
    from speqtro.vendors.cascade1 import install_pickle_aliases
    install_pickle_aliases()

    # preprocessor.p sits one level above the trained_model/ subfolder
    preprocessor_path = model_dir.parent / "preprocessor.p"
    if not preprocessor_path.exists():
        preprocessor_path = model_dir / "preprocessor.p"
    if not preprocessor_path.exists():
        raise FileNotFoundError(
            f"CASCADE 1.0 preprocessor not found: {preprocessor_path}"
        )
    with open(preprocessor_path, "rb") as f:
        input_data = pickle.load(f)
    preprocessor = input_data["preprocessor"]

    # Load model with CASCADE 1.0 custom layer objects
    from speqtro.vendors.cascade1.nfp import custom_layers
    from tensorflow.keras.models import load_model

    model_path = model_dir / "best_model.hdf5"
    if not model_path.exists():
        raise FileNotFoundError(
            f"CASCADE 1.0 model weights not found: {model_path}"
        )

    model = load_model(str(model_path), custom_objects=custom_layers)

    _cascade1_model = model
    _cascade1_preprocessor = preprocessor
    _cascade1_model_dir = model_dir

    return model, preprocessor


# ── Inference logic ──────────────────────────────────────────────────────────

def _predict_direct(smiles: str, model_dir: Path) -> dict:
    """
    Run CASCADE 1.0 ¹H NMR inference for a single SMILES string.

    Returns a dict with keys:
      'predictions': list of {'atom_id': int, 'shift_ppm': float}
      or 'error': str on failure
    """
    import numpy as np
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from speqtro.vendors.cascade1.genConf import genConf
    from speqtro.vendors.cascade1.nfp.preprocessing.sequence import GraphSequence

    # Boltzmann constants
    R = 0.001987  # kcal / (mol * K)
    T = 298.15    # K

    # ── Build initial mol with embedded conformer ────────────────────────────
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": f"Invalid SMILES: {smiles}"}

    AllChem.EmbedMolecule(mol, useRandomCoords=True)
    mol = Chem.AddHs(mol, addCoords=True)

    # ── Generate MMFF94s conformer ensemble ─────────────────────────────────
    try:
        mol_conf, ids, _nr = genConf(mol, nc=50, rms=-1, efilter=10.0, rmspost=0.5)
    except Exception as exc:
        logger.warning("CASCADE 1.0 genConf failed for SMILES %s: %s", smiles, exc)
        return {"error": f"Conformer generation failed: {exc}"}

    if not ids:
        return {"error": "No conformers generated for the given molecule."}

    # ── Build per-conformer DataFrame for preprocessing ──────────────────────
    rows = []
    for energy, conf_id in ids:
        m_i = Chem.RWMol(mol_conf)
        # Propagate the specific conformer geometry
        for ia in range(mol_conf.GetNumAtoms()):
            m_i.GetConformer().SetAtomPosition(
                ia, mol_conf.GetConformer(conf_id).GetAtomPosition(ia)
            )
        m_i.SetProp("E", str(energy))
        m_i.SetProp("_Name", f"0_{conf_id}")

        # Hydrogen atom indices
        h_indices = np.array(
            [a.GetIdx() for a in m_i.GetAtoms() if a.GetAtomicNum() == 1],
            dtype=int,
        )
        if len(h_indices) == 0:
            continue

        rows.append({
            "mol_id": 0,
            "Mol": m_i,
            "n_atoms": m_i.GetNumAtoms(),
            "atom_index": h_indices,
            "relative_E": energy,
            "cf_id": conf_id,
        })

    if not rows:
        return {"error": "No hydrogen atoms found in molecule."}

    df = pd.DataFrame(rows)

    # ── Load model and preprocessor ─────────────────────────────────────────
    model, preprocessor = _get_model_and_preprocessor(model_dir)

    # ── Preprocess ──────────────────────────────────────────────────────────
    def mol_iter(dataframe):
        for _, row in dataframe.iterrows():
            yield row["Mol"], row["atom_index"]

    inputs_list = preprocessor.predict(mol_iter(df))

    # ── RBFSequence for batched inference ────────────────────────────────────
    def _compute_stacked_offsets(sizes, repeats):
        return np.repeat(np.cumsum(np.hstack([0, sizes[:-1]])), repeats)

    class RBFSequence(GraphSequence):
        def process_data(self, batch_data):
            batch_data["distance_rbf"] = self._rbf_expansion(batch_data["distance"])

            offset = _compute_stacked_offsets(
                batch_data["n_pro"], batch_data["n_atom"]
            )
            offset = np.where(batch_data["atom_index"] >= 0, offset, 0)
            batch_data["atom_index"] += offset

            del batch_data["n_atom"]
            del batch_data["n_bond"]
            del batch_data["distance"]
            return batch_data

        @staticmethod
        def _rbf_expansion(distances, mu=0, delta=0.1, kmax=256):
            k = np.arange(0, kmax)
            logits = -(np.atleast_2d(distances).T - (-mu + delta * k)) ** 2 / delta
            return np.exp(logits)

    evaluate_sequence = RBFSequence(inputs_list, batch_size=32)

    predicted = []
    for x in evaluate_sequence:
        out = model.predict_on_batch(x)
        out = np.concatenate(out)
        predicted.extend(out.tolist())

    # ── Boltzmann averaging ──────────────────────────────────────────────────
    # Build a flat DataFrame: one row per (mol_id, atom_index, cf_id)
    spread_rows = []
    pred_cursor = 0
    for _, row in df.iterrows():
        for h_idx in row["atom_index"]:
            if pred_cursor >= len(predicted):
                break
            spread_rows.append({
                "mol_id": row["mol_id"],
                "atom_index": int(h_idx),
                "relative_E": float(row["relative_E"]),
                "cf_id": int(row["cf_id"]),
                "predicted": predicted[pred_cursor],
            })
            pred_cursor += 1

    spread_df = pd.DataFrame(spread_rows)
    spread_df["b_weight"] = spread_df["relative_E"].apply(
        lambda x: math.exp(-x / (R * T))
    )

    final_predictions = []
    grouped = spread_df.groupby("atom_index")
    for atom_idx, group in grouped:
        total_weight = group["b_weight"].sum()
        weighted_shift = (group["b_weight"] * group["predicted"]).sum() / total_weight
        final_predictions.append({
            "atom_id": int(atom_idx),
            "shift_ppm": round(float(weighted_shift), 2),
        })

    # Sort by atom_id for consistent output
    final_predictions.sort(key=lambda x: x["atom_id"])

    return {"predictions": final_predictions}


# ── speQ tool registration ───────────────────────────────────────────────────

@registry.register(
    name="nmr.predict_h1_cascade",
    description=(
        "Predict \u00b9H NMR chemical shifts using CASCADE 1.0 (Paton Lab). "
        "Input: SMILES string. Output: per-hydrogen atom shift predictions in ppm, "
        "Boltzmann-averaged over MMFF94s conformer ensemble. "
        "Requires CASCADE 1.0 model weights (best_model.hdf5 + preprocessor.p)."
    ),
    category="nmr",
    parameters={
        "smiles": {
            "type": "string",
            "description": "SMILES string of the molecule",
        },
    },
)
def predict_h1_cascade(smiles: str) -> dict:
    """
    Predict ¹H shifts using CASCADE 1.0.

    Args:
        smiles: SMILES string of the molecule.

    Returns:
        Dict with 'predictions' (list of per-hydrogen atom shifts), 'model',
        'citations'.
    """
    smiles = smiles.strip()
    if not smiles:
        return {"error": "No SMILES provided."}

    model_dir = _get_model_dir()
    if model_dir is None:
        return {
            "error": (
                "CASCADE 1.0 model directory not found. "
                "Set config key 'cascade1.model_dir' to the path of the folder "
                "containing best_model.hdf5 and preprocessor.p, "
                "or set the SPEQ_CASCADE1_DIR environment variable."
            )
        }

    if not (model_dir / "best_model.hdf5").exists():
        return {
            "error": (
                f"CASCADE 1.0 model weights not found at "
                f"{model_dir / 'best_model.hdf5'}. "
                "Ensure the pretrained weights are present in the model directory."
            )
        }

    try:
        result = _predict_direct(smiles, model_dir)
    except Exception as e:
        logger.exception("CASCADE 1.0 prediction failed")
        return {"error": f"CASCADE 1.0 inference error: {e}"}

    if "error" in result:
        return result

    return {
        "predictions": result["predictions"],
        "smiles": smiles,
        "model": "CASCADE 1.0 (Message-passing GNN, MMFF94s conformer ensemble)",
        "citations": [
            {
                "title": (
                    "Rapid Prediction of 1H and 13C NMR Chemical Shifts from "
                    "Molecular Structure Using Convolutional Neural Networks"
                ),
                "authors": "Jonas E, Kuhn S",
                "journal": "J Cheminform",
                "year": "2019",
                "url": "https://nova.chem.colostate.edu/cascade/",
                "note": "CASCADE 1.0 model weights: pretrained, Paton Lab",
            }
        ],
        "method": (
            "Message-passing GNN on MMFF94s conformer ensemble "
            "(nc=50, efilter=10.0 kcal/mol, rmspost=0.5 Å), "
            "Boltzmann-averaged at 298.15 K"
        ),
        "units": "ppm",
        "n_predictions": len(result["predictions"]),
    }

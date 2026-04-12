"""
ms-pred wrapper: MS/MS spectrum prediction using ICEBERG.

ICEBERG (Iterative Combinatorial Enumeration of subgraph fragmentation with
Biologically Evaluated Ranking using Graphs) predicts tandem mass spectra (MS/MS)
from molecular structures. It learns a fragmentation DAG then scores fragment
intensities with a contrastive-trained GNN.

Architecture: Two-stage — (1) DAG fragment generator (GNN), (2) Intensity predictor
Framework: PyTorch Lightning + PyG (PyTorch Geometric)
Input: SMILES + collision energy (eV) + precursor m/z + adduct type
Output: Predicted MS/MS spectrum (m/z vs. normalized intensity)

Configuration:
  Set mspred.gen_checkpoint / mspred.inten_checkpoint in speqtro config,
  or set SPEQ_MSPRED_GEN_CKPT / SPEQ_MSPRED_INTEN_CKPT env vars.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from speqtro.tools import registry

logger = logging.getLogger("speqtro.tools.mspred")

# Common adduct types (for validation)
_VALID_ADDUCTS = {
    "[M+H]+", "[M+Na]+", "[M+K]+", "[M+NH4]+",
    "[M-H]-", "[M+Cl]-", "[M+HCOO]-",
    "[M+2H]2+", "[M+H+Na]2+",
}

# Lazy-cached model
_iceberg_model = None
_iceberg_ckpt_key: Optional[str] = None


# ── Config helpers ────────────────────────────────────────────────────────────

def _get_config() -> dict:
    """Return {'gen_checkpoint', 'inten_checkpoint'}."""
    gen_ckpt = inten_ckpt = repo_dir = None

    try:
        from speqtro.agent.config import Config
        cfg = Config.load()
        if cfg.get("mspred.repo_dir"):
            repo_dir = Path(cfg.get("mspred.repo_dir"))
        if cfg.get("mspred.gen_checkpoint"):
            gen_ckpt = Path(cfg.get("mspred.gen_checkpoint"))
        if cfg.get("mspred.inten_checkpoint"):
            inten_ckpt = Path(cfg.get("mspred.inten_checkpoint"))
    except Exception:
        pass

    if repo_dir is None:
        env_dir = os.environ.get("SPEQ_MSPRED_DIR")
        if env_dir:
            repo_dir = Path(env_dir)
        else:
            guess = Path(__file__).resolve().parents[4].parent / "ms-pred"
            if guess.exists():
                repo_dir = guess

    if gen_ckpt is None and os.environ.get("SPEQ_MSPRED_GEN_CKPT"):
        gen_ckpt = Path(os.environ["SPEQ_MSPRED_GEN_CKPT"])
    if inten_ckpt is None and os.environ.get("SPEQ_MSPRED_INTEN_CKPT"):
        inten_ckpt = Path(os.environ["SPEQ_MSPRED_INTEN_CKPT"])

    # Auto-discover checkpoints in repo
    if repo_dir and (gen_ckpt is None or inten_ckpt is None):
        results_dir = repo_dir / "results" / "iceberg"
        if results_dir.exists():
            if gen_ckpt is None:
                candidates = list(results_dir.rglob("*gen*.ckpt")) + list(results_dir.rglob("*dag_gen*.ckpt"))
                if candidates:
                    gen_ckpt = sorted(candidates)[-1]
            if inten_ckpt is None:
                candidates = list(results_dir.rglob("*inten*.ckpt")) + list(results_dir.rglob("*dag_inten*.ckpt"))
                if candidates:
                    inten_ckpt = sorted(candidates)[-1]

    # Fallback 1: vendored checkpoints alongside this package
    _vendor_dir = Path(__file__).resolve().parent.parent / "vendors" / "mspred"
    if gen_ckpt is None:
        p = _vendor_dir / "iceberg_dag_gen_msg_best.ckpt"
        if p.exists():
            gen_ckpt = p
    if inten_ckpt is None:
        p = _vendor_dir / "iceberg_dag_inten_msg_best.ckpt"
        if p.exists():
            inten_ckpt = p

    # Fallback 2: ~/.speqtro/models/iceberg/ (fetch-weights destination)
    _home_dir = Path.home() / ".speqtro" / "models" / "iceberg"
    if gen_ckpt is None:
        p = _home_dir / "iceberg_dag_gen_msg_best.ckpt"
        if p.exists():
            gen_ckpt = p
    if inten_ckpt is None:
        p = _home_dir / "iceberg_dag_inten_msg_best.ckpt"
        if p.exists():
            inten_ckpt = p

    return {
        "gen_checkpoint": gen_ckpt,
        "inten_checkpoint": inten_ckpt,
    }


# ── Model loading ────────────────────────────────────────────────────────────

def _get_model(gen_ckpt: Path, inten_ckpt: Path):
    """Lazy-load and cache ICEBERG JointModel."""
    global _iceberg_model, _iceberg_ckpt_key

    cache_key = f"{gen_ckpt}|{inten_ckpt}"
    if _iceberg_model is not None and _iceberg_ckpt_key == cache_key:
        return _iceberg_model

    import torch
    from speqtro.vendors.mspred.joint_model import JointModel

    logger.info("Loading ICEBERG checkpoints: gen=%s inten=%s", gen_ckpt, inten_ckpt)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = JointModel.from_checkpoints(
        gen_checkpoint=str(gen_ckpt),
        inten_checkpoint=str(inten_ckpt),
    )
    model = model.to(device)
    model.eval()

    _iceberg_model = model
    _iceberg_ckpt_key = cache_key
    return model


# ── Direct inference ─────────────────────────────────────────────────────────

def _predict_direct(
    smiles: str,
    collision_energy: float,
    precursor_mz: float,
    adduct: str,
    max_peaks: int,
) -> dict:
    """Run ICEBERG inference in-process. Returns dict with 'peaks' and 'error'."""
    import numpy as np
    import torch

    cfg = _get_config()
    gen_ckpt = cfg["gen_checkpoint"]
    inten_ckpt = cfg["inten_checkpoint"]

    if gen_ckpt is None or not gen_ckpt.exists():
        return {"peaks": [], "error": "ICEBERG gen checkpoint not found"}
    if inten_ckpt is None or not inten_ckpt.exists():
        return {"peaks": [], "error": "ICEBERG inten checkpoint not found"}

    model = _get_model(gen_ckpt, inten_ckpt)
    device = next(model.parameters()).device

    with torch.no_grad():
        output = model.predict_mol(
            smi=smiles,
            collision_eng=collision_energy,
            precursor_mz=precursor_mz,
            adduct=adduct,
            device=device,
            binned_out=False,
            threshold=0.0,
            max_nodes=300,
            adduct_shift=True,
        )

    # Extract peaks from output
    peaks = []
    if isinstance(output, dict):
        spec = output.get("spec", [])
        if spec and len(spec) > 0:
            spec_arr = spec[0] if isinstance(spec, list) else spec
            spec_arr = np.array(spec_arr)
            if spec_arr.ndim == 2 and spec_arr.shape[1] == 2:
                mask = spec_arr[:, 1] > 0.001
                spec_arr = spec_arr[mask]
                order = np.argsort(spec_arr[:, 1])[::-1][:max_peaks]
                for idx in order:
                    peaks.append({
                        "mz": round(float(spec_arr[idx, 0]), 4),
                        "intensity": round(float(spec_arr[idx, 1]), 4),
                    })
            elif spec_arr.ndim == 1:
                mz_bins = np.linspace(0, 1500, len(spec_arr))
                mask = spec_arr > 0.001
                mz_nonzero = mz_bins[mask]
                int_nonzero = spec_arr[mask]
                order = np.argsort(int_nonzero)[::-1][:max_peaks]
                for idx in order:
                    peaks.append({
                        "mz": round(float(mz_nonzero[idx]), 1),
                        "intensity": round(float(int_nonzero[idx]), 4),
                    })

    peaks.sort(key=lambda p: p["mz"])
    return {"peaks": peaks, "error": None}


# ── Cosine similarity scoring ─────────────────────────────────────────────────

def cosine_similarity_spectra(
    predicted: list[dict],
    experimental: list[dict],
    mz_tolerance: float = 0.02,
    bin_width: float = 1.0,
) -> float:
    """
    Compute cosine similarity between two MS/MS spectra.
    Both spectra are lists of {'mz': float, 'intensity': float}.
    Uses binning at bin_width Da resolution.
    Returns cosine similarity in [0, 1].
    """
    try:
        import numpy as np
    except ImportError:
        return 0.0

    if not predicted or not experimental:
        return 0.0

    max_mz = max(
        max(p["mz"] for p in predicted),
        max(p["mz"] for p in experimental),
    ) + bin_width

    n_bins = int(max_mz / bin_width) + 1
    pred_vec = np.zeros(n_bins)
    exp_vec = np.zeros(n_bins)

    for p in predicted:
        idx = int(p["mz"] / bin_width)
        if idx < n_bins:
            pred_vec[idx] = max(pred_vec[idx], float(p["intensity"]))

    for p in experimental:
        idx = int(p["mz"] / bin_width)
        if idx < n_bins:
            exp_vec[idx] = max(exp_vec[idx], float(p["intensity"]))

    p_norm = np.linalg.norm(pred_vec)
    e_norm = np.linalg.norm(exp_vec)
    if p_norm == 0 or e_norm == 0:
        return 0.0

    return float(np.dot(pred_vec / p_norm, exp_vec / e_norm))


# ── speQ tool registration ────────────────────────────────────────────────────

@registry.register(
    name="ms.predict_msms_iceberg",
    description=(
        "Predict tandem MS/MS fragmentation spectrum using ICEBERG (ms-pred). "
        "Input: SMILES + collision energy (eV) + precursor m/z + adduct type. "
        "Output: predicted (m/z, intensity) peak list. "
        "Requires downloaded ICEBERG checkpoint files (gen.ckpt + inten.ckpt). "
        "See: ICEBERG paper / MassSpecGym for open weights."
    ),
    category="ms",
    parameters={
        "smiles": {
            "type": "string",
            "description": "SMILES string of the molecule.",
        },
        "collision_energy": {
            "type": "number",
            "description": "Collision energy in eV (e.g. 20, 40, 60). Default: 20.",
        },
        "precursor_mz": {
            "type": "number",
            "description": "Precursor ion m/z value.",
        },
        "adduct": {
            "type": "string",
            "description": "Ionization adduct type, e.g. '[M+H]+', '[M-H]-'. Default: '[M+H]+'.",
        },
        "experimental_peaks": {
            "type": "array",
            "description": (
                "Optional experimental MS/MS spectrum for scoring. "
                "List of {'mz': float, 'intensity': float} dicts. "
                "If provided, cosine similarity is computed."
            ),
        },
        "max_peaks": {
            "type": "integer",
            "description": "Maximum number of peaks in output. Default: 50.",
        },
    },
)
def predict_msms_iceberg(
    smiles: str,
    collision_energy: float = 20.0,
    precursor_mz: float = 0.0,
    adduct: str = "[M+H]+",
    experimental_peaks: Optional[list] = None,
    max_peaks: int = 50,
) -> dict:
    """
    Predict MS/MS spectrum using ICEBERG.

    Args:
        smiles: SMILES of the molecule.
        collision_energy: Collision energy in eV.
        precursor_mz: Precursor m/z (auto-computed if 0).
        adduct: Ionization adduct type.
        experimental_peaks: Optional experimental spectrum for cosine scoring.
        max_peaks: Max peaks in output.

    Returns:
        Dict with predicted peaks, optional cosine similarity vs experimental.
    """
    if not smiles:
        return {"error": "No SMILES provided."}

    if adduct not in _VALID_ADDUCTS:
        logger.warning("Unusual adduct type: %s. Proceeding anyway.", adduct)

    # Auto-compute precursor_mz if not provided
    if precursor_mz == 0.0:
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                mw = Descriptors.ExactMolWt(mol)
                if adduct == "[M+H]+":
                    precursor_mz = mw + 1.00728
                elif adduct == "[M-H]-":
                    precursor_mz = mw - 1.00728
                elif adduct == "[M+Na]+":
                    precursor_mz = mw + 22.98922
                elif adduct == "[M+NH4]+":
                    precursor_mz = mw + 18.03437
                else:
                    precursor_mz = mw + 1.00728
        except Exception:
            precursor_mz = 100.0

    try:
        output = _predict_direct(
            smiles=smiles,
            collision_energy=collision_energy,
            precursor_mz=precursor_mz,
            adduct=adduct,
            max_peaks=max_peaks,
        )
    except Exception as e:
        logger.exception("ICEBERG inference failed")
        return {"error": f"ICEBERG inference error: {e}"}

    if output.get("error"):
        return {"error": output["error"]}

    predicted_peaks = output.get("peaks", [])
    result = {
        "smiles": smiles,
        "predicted_peaks": predicted_peaks,
        "n_peaks": len(predicted_peaks),
        "collision_energy_ev": collision_energy,
        "precursor_mz": round(precursor_mz, 4),
        "adduct": adduct,
        "model": "ICEBERG (ms-pred, DAG fragment + intensity GNN)",
        "citations": [
            {
                "title": (
                    "Generating molecular fragmentation graphs with "
                    "autoregressive neural networks"
                ),
                "authors": "Goldman S, Li J, Coley CW",
                "journal": "Analytical Chemistry, 2024",
                "repo": "https://github.com/samgoldman97/ms-pred",
            }
        ],
    }

    if experimental_peaks and predicted_peaks:
        cos_sim = cosine_similarity_spectra(predicted_peaks, experimental_peaks)
        result["cosine_similarity_vs_experimental"] = round(cos_sim, 4)
        result["score_interpretation"] = (
            "cosine >= 0.7 = good match, 0.4-0.7 = partial match, < 0.4 = poor match"
        )

    return result


# ── Utility: spectrum scoring exposed for pipeline ───────────────────────────

def score_candidate_vs_experimental(
    candidate_smiles: str,
    experimental_peaks: list[dict],
    collision_energy: float = 20.0,
    adduct: str = "[M+H]+",
) -> dict:
    """
    Predict MS/MS for a candidate SMILES and score against experimental spectrum.
    Returns dict with cosine_similarity and peak info.
    Used by pipeline.full_elucidation.
    """
    result = predict_msms_iceberg(
        smiles=candidate_smiles,
        collision_energy=collision_energy,
        adduct=adduct,
        experimental_peaks=experimental_peaks,
    )
    if "error" in result:
        return {"cosine_similarity": None, "error": result["error"]}
    return {
        "cosine_similarity": result.get("cosine_similarity_vs_experimental"),
        "n_predicted_peaks": result.get("n_peaks", 0),
        "model": "ICEBERG",
    }

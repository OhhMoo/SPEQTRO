"""
SSIN wrapper for IR functional group detection.

SSIN (Substructure-directed Spectrum Interpreter Network) is a deep learning model
that identifies functional groups in IR spectra using cross-attention between
an input spectrum and a reference database of known IR spectra.

Architecture: Cross-attention over reference DB spectra + 1D CNN encoder
Framework: PyTorch
Input: IR spectrum as JDX file OR (wavenumber, intensity) arrays
       Wavenumber range: 550-3801 cm-1, normalized intensity [0, 1]
Output: Per-functional-group detection (Detected/Not detected) + important wavenumbers

Integration strategy:
  Direct import from vendored speqtro.vendors.ssin package.
  Models are lazily loaded and cached in a global dict keyed by functional group name.

Configuration:
  Set ssin.repo_dir in speqtro config, or SPEQ_SSIN_DIR environment variable,
  pointing to the SSIN repo directory (needed for pretrained weights and ref_dataset.pt).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from speqtro.tools import registry
from speqtro.vendors.ssin import SSIN, read_jdx_file, predict_from_jdx, identify_impt_peaks
from speqtro.vendors.ssin.data import IRSpectrum, interpol_absorbance

logger = logging.getLogger("speqtro.tools.ssin")

# Functional groups with available pretrained models
_AVAILABLE_FGS = ("alcohol", "aldehyde")

# Wavenumber grid parameters (must match SSIN training)
_WMIN, _WMAX = 550, 3801

# Lazy model cache: {functional_group_name: (SSIN_model, ref_dataset)}
_ssin_models: dict[str, SSIN] = {}

# Cached reference dataset (shared across all functional groups)
_ref_dataset = None


# -- Config helpers ------------------------------------------------------------

def _get_repo_dir() -> Optional[Path]:
    """Find SSIN repo directory from config or environment."""
    try:
        from speqtro.agent.config import Config
        cfg = Config.load()
        d = cfg.get("ssin.repo_dir")
        if d:
            return Path(d)
    except Exception:
        pass

    env_dir = os.environ.get("SPEQ_SSIN_DIR")
    if env_dir:
        return Path(env_dir)

    # Auto-detect: ~/.speqtro/models/ssin/ (populated by `speqtro fetch-weights`)
    home_dir = Path.home() / ".speqtro" / "models" / "ssin"
    if home_dir.exists() and (home_dir / "save" / "model").exists():
        return home_dir

    # Well-known relative location
    guess = Path(__file__).resolve().parents[4].parent / "SSIN"
    if guess.exists() and (guess / "method").exists():
        return guess

    return None


def _find_model_weights(repo_dir: Path, functional_group: str) -> list[Path]:
    """Return available model weight paths for a functional group (fold 0 preferred)."""
    model_dir = repo_dir / "save" / "model" / functional_group
    if not model_dir.exists():
        return []
    weights = sorted(model_dir.glob("model_*.pt"))
    return weights


def _load_ref_dataset(repo_dir: Path):
    """Load and cache the reference dataset required for SSIN model construction."""
    global _ref_dataset
    if _ref_dataset is not None:
        return _ref_dataset

    ref_path = repo_dir / "res" / "ref_dataset.pt"
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference dataset not found: {ref_path}")

    _ref_dataset = torch.load(str(ref_path), map_location="cpu", weights_only=False)
    return _ref_dataset


def _get_model(repo_dir: Path, functional_group: str) -> SSIN:
    """Load an SSIN model for the given functional group, caching it globally."""
    if functional_group in _ssin_models:
        return _ssin_models[functional_group]

    ref_db = _load_ref_dataset(repo_dir)
    len_spect = ref_db.len_spect

    weights_files = _find_model_weights(repo_dir, functional_group)
    if not weights_files:
        raise FileNotFoundError(
            f"SSIN model weights not found for '{functional_group}' at "
            f"{repo_dir / 'save' / 'model' / functional_group}"
        )

    # Use fold 0 weights by default
    weights_path = weights_files[0]

    model = SSIN(dim_emb=128, len_spect=len_spect, ref_db=ref_db)
    state_dict = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()

    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()

    _ssin_models[functional_group] = model
    logger.info("Loaded SSIN model for '%s' from %s", functional_group, weights_path)
    return model


def _build_ir_spectrum_from_arrays(
    wavenumbers: list,
    intensities: list,
) -> IRSpectrum:
    """Construct an IRSpectrum object from raw wavenumber/intensity arrays."""
    wn = np.array(wavenumbers, dtype=np.float64)
    ab = np.array(intensities, dtype=np.float64)

    # Normalize
    max_ab = ab.max()
    if max_ab > 0:
        ab = ab / max_ab

    # Interpolate to SSIN grid
    wn_interp, ab_interp = interpol_absorbance(wn, ab, _WMIN, _WMAX)

    # Re-normalize after interpolation
    max_ab2 = ab_interp.max()
    if max_ab2 > 0:
        ab_interp = ab_interp / max_ab2

    return IRSpectrum(
        data_id="user_input",
        compound_name="unknown",
        state="none",
        wavenumber=wn_interp,
        absorbance=ab_interp,
    )


def _run_inference(
    repo_dir: Path,
    functional_groups: list[str],
    irs: IRSpectrum,
) -> dict:
    """Run SSIN inference on a single IR spectrum for the requested functional groups."""
    results = {}

    for fg in functional_groups:
        try:
            model = _get_model(repo_dir, fg)
            label, attns = predict_from_jdx(model, irs)

            # Compute confidence from raw logit
            model.eval()
            with torch.no_grad():
                absorbance_t = torch.tensor(irs.absorbance_savgol, dtype=torch.float).unsqueeze(0)
                if torch.cuda.is_available():
                    absorbance_t = absorbance_t.cuda()
                pred, _, _ = model(absorbance_t, model.pos_refs.cuda() if torch.cuda.is_available() else model.pos_refs)
                prob = torch.sigmoid(pred).item()

            # Identify important peaks via attention
            try:
                impt_peaks = identify_impt_peaks(irs, attns, fg)
                important_wns = [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in impt_peaks]
            except Exception:
                # Fallback: top-10 individual wavenumbers
                grid = irs.wavenumber
                top_k_idx = np.argsort(attns)[::-1][:10]
                important_wns = [round(float(grid[i]), 1) for i in sorted(top_k_idx)]

            results[fg] = {
                "label": label,
                "confidence": round(float(prob), 4),
                "important_wavenumber_ranges_cm1": important_wns,
            }
        except Exception as e:
            logger.exception("SSIN inference failed for '%s'", fg)
            results[fg] = {"label": "error", "error": str(e)}

    return results


# -- speQ tool registration ---------------------------------------------------

@registry.register(
    name="ir.detect_functional_groups_ssin",
    description=(
        "Detect functional groups in an IR spectrum using SSIN (Substructure-directed "
        "Spectrum Interpreter Network). Takes a JDX file path or wavenumber/intensity arrays. "
        "Returns per-group detection label, confidence score, and the most diagnostic "
        "wavenumbers (cm-1). Pretrained models available for: alcohol, aldehyde. "
        "Wavenumber range: 550-3801 cm-1."
    ),
    category="ir",
    parameters={
        "jdx_file": {
            "type": "string",
            "description": "Absolute path to JCAMP-DX (.jdx) IR spectrum file.",
        },
        "wavenumbers": {
            "type": "array",
            "description": (
                "List of wavenumber values (cm-1) -- alternative to jdx_file. "
                "Provide together with 'intensities'."
            ),
        },
        "intensities": {
            "type": "array",
            "description": "List of IR intensity/absorbance values matching 'wavenumbers'.",
        },
        "functional_groups": {
            "type": "array",
            "description": (
                "Functional groups to test. Available: ['alcohol', 'aldehyde']. "
                "Defaults to all available."
            ),
        },
    },
)
def detect_functional_groups_ssin(
    jdx_file: Optional[str] = None,
    wavenumbers: Optional[list] = None,
    intensities: Optional[list] = None,
    functional_groups: Optional[list[str]] = None,
) -> dict:
    """
    Detect functional groups in an IR spectrum using SSIN.

    Args:
        jdx_file: Path to JCAMP-DX file.
        wavenumbers: List of wavenumber values (cm-1).
        intensities: List of intensity values (paired with wavenumbers).
        functional_groups: Subset of functional groups to test. Defaults to all.

    Returns:
        Dict with per-group detection results, confidence scores, important peaks.
    """
    if not jdx_file and (not wavenumbers or not intensities):
        return {"error": "Provide either 'jdx_file' or both 'wavenumbers' and 'intensities'."}

    repo_dir = _get_repo_dir()
    if repo_dir is None:
        return {
            "error": (
                "SSIN repository not found. "
                "Set config key 'ssin.repo_dir' to the SSIN directory, "
                "or set SPEQ_SSIN_DIR environment variable."
            )
        }

    # Validate requested functional groups
    fgs = functional_groups if functional_groups else list(_AVAILABLE_FGS)
    invalid = [fg for fg in fgs if fg not in _AVAILABLE_FGS]
    if invalid:
        return {
            "error": f"No pretrained models for: {invalid}. Available: {list(_AVAILABLE_FGS)}"
        }

    # Check weights exist
    for fg in fgs:
        weights = _find_model_weights(repo_dir, fg)
        if not weights:
            return {
                "error": (
                    f"SSIN model weights not found for '{fg}' at "
                    f"{repo_dir / 'save' / 'model' / fg}. "
                    "Ensure SSIN pretrained weights are present."
                )
            }

    # Build IRSpectrum from input
    irs = None
    if jdx_file:
        irs = read_jdx_file(jdx_file, norm_y=True, wmin=_WMIN, wmax=_WMAX)
        if irs is None:
            return {"error": f"Failed to parse JDX file: {jdx_file}"}
    elif wavenumbers and intensities:
        if len(wavenumbers) != len(intensities):
            return {"error": "wavenumbers and intensities must have the same length."}
        irs = _build_ir_spectrum_from_arrays(wavenumbers, intensities)

    try:
        results = _run_inference(repo_dir, fgs, irs)
    except Exception as e:
        logger.exception("SSIN inference failed")
        return {"error": f"SSIN inference error: {e}"}

    # Build summary
    detected = [fg for fg, r in results.items() if r.get("label") == "Detected"]
    not_detected = [fg for fg, r in results.items() if r.get("label") == "Not detected"]

    return {
        "results": results,
        "detected_functional_groups": detected,
        "not_detected": not_detected,
        "model": "SSIN (Cross-attention IR functional group detector)",
        "available_groups": list(_AVAILABLE_FGS),
        "wavenumber_range_cm1": f"{_WMIN}-{_WMAX}",
        "citations": [
            {
                "title": (
                    "SSIN: Substructure-directed Spectrum Interpreter Network "
                    "for IR Spectroscopy"
                ),
                "note": "Pretrained on NIST IR database, gas-phase spectra",
                "repo": str(repo_dir),
            }
        ],
        "method": (
            "Cross-attention between input IR spectrum and reference DB spectra. "
            "Important wavenumbers identified via attention weight analysis."
        ),
    }

"""
ChefNMR wrapper for NMR-to-structure elucidation.

ChefNMR (Princeton / MIT, NeurIPS 2025) is an atomic diffusion model that takes
1H and/or 13C NMR spectra as input and generates candidate 3D molecular structures
(returned as SMILES strings with confidence scores).

Architecture: EDM-based diffusion transformer (DiT) over atom coordinates + types
Framework: PyTorch Lightning + Hydra
Input: NMR spectra as discretized 1D vectors (10k-point grids)
Output: Top-k SMILES candidates

Configuration:
  Set chefnmr.checkpoint in speqtro config, or set SPEQ_CHEFNMR_CKPT env var.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

from speqtro.tools import registry

logger = logging.getLogger("speqtro.tools.chefnmr")

# Lazy-cached model
_chefnmr_model = None
_chefnmr_ckpt_path: Optional[Path] = None


# ── Grid constants (must match ChefNMR meta/grids/) ─────────────────────────

H_GRID_MIN, H_GRID_MAX, H_GRID_NPTS = -2.0, 10.0, 10000
C_GRID_MIN, C_GRID_MAX, C_GRID_NPTS_80 = 3.42, 231.3, 80
C10K_GRID_MIN, C10K_GRID_MAX, C10K_GRID_NPTS = -20.0, 230.0, 10000

# Model variants and the atom decoders they support
_MODEL_ATOM_DECODERS = {
    "NP": ["C", "H", "O", "N", "S", "P", "F", "Cl", "Br", "I"],
    "SB": ["C", "H", "O", "N"],
    "US": ["C", "H", "O", "N", "S", "P", "F", "Cl", "Br", "I"],
}


def _make_grid(min_ppm: float, max_ppm: float, n_pts: int) -> np.ndarray:
    return np.linspace(min_ppm, max_ppm, n_pts)


def peaks_to_spectrum(
    peaks: list[dict],
    grid: np.ndarray,
    sigma_ppm: float = 0.05,
) -> np.ndarray:
    """
    Convert a list of NMR peaks to a discretized 1D spectrum vector
    by Gaussian broadening onto a uniform ppm grid.

    Args:
        peaks: List of {"shift": float, ...} dicts.
        grid: 1D numpy array of ppm values (the target grid).
        sigma_ppm: Gaussian broadening width in ppm.

    Returns:
        1D float32 array of shape (len(grid),), L2-normalized.
    """
    spectrum = np.zeros(len(grid), dtype=np.float32)
    for peak in peaks:
        shift = peak.get("shift", peak.get("ppm", None))
        if shift is None:
            continue
        weight = float(peak.get("integral") or 1.0)
        spectrum += weight * np.exp(-0.5 * ((grid - float(shift)) / sigma_ppm) ** 2)

    norm = np.linalg.norm(spectrum)
    if norm > 0:
        spectrum /= norm

    return spectrum


# ── Config helpers ────────────────────────────────────────────────────────────

def _get_config() -> dict:
    """Return {'checkpoint': Path|None, 'all_checkpoints': list[Path]}."""
    checkpoint = None

    try:
        from speqtro.agent.config import Config
        cfg = Config.load()
        c = cfg.get("chefnmr.checkpoint")
        if c:
            checkpoint = Path(c)
    except Exception:
        pass

    if checkpoint is None:
        env_ckpt = os.environ.get("SPEQ_CHEFNMR_CKPT")
        if env_ckpt:
            checkpoint = Path(env_ckpt)

    # Search dirs in priority order
    search_dirs = [
        Path.home() / "Desktop" / "chefnmr",          # user Desktop (where weights currently live)
        Path.home() / ".speqtro" / "models" / "chefnmr",
        Path(__file__).resolve().parents[4].parent / "chefnmr" / "checkpoints",
    ]
    env_dir = os.environ.get("SPEQ_CHEFNMR_DIR")
    if env_dir:
        search_dirs.insert(0, Path(env_dir))

    all_checkpoints: list[Path] = []
    for d in search_dirs:
        if d.exists():
            all_checkpoints.extend(sorted(d.glob("*.ckpt")))

    if checkpoint is None and all_checkpoints:
        # Default: prefer NP-H10kC10k-S128 (fast, broad atom set)
        preferred = [p for p in all_checkpoints if "NP" in p.name and "S128" in p.name]
        checkpoint = preferred[0] if preferred else all_checkpoints[0]

    return {"checkpoint": checkpoint, "all_checkpoints": all_checkpoints}


def _select_checkpoint(
    model_variant: Optional[str],
    all_checkpoints: list[Path],
    has_heteroatoms: bool,
) -> Optional[Path]:
    """
    Pick the best checkpoint given a model variant hint and whether the formula
    has atoms beyond C/H/O/N (heteroatoms that require NP or US model).
    """
    if not all_checkpoints:
        return None

    if model_variant:
        # Exact or prefix match (e.g. "NP" matches "NP-H10kC10k-S128-...")
        matches = [p for p in all_checkpoints if model_variant.upper() in p.name.upper()]
        if matches:
            return matches[0]

    # Auto-select: heteroatoms need NP or US (SB only covers CHON)
    if has_heteroatoms:
        preferred = [
            p for p in all_checkpoints
            if ("NP" in p.name or "US" in p.name) and "S128" in p.name
        ]
    else:
        preferred = [p for p in all_checkpoints if "S128" in p.name]

    return preferred[0] if preferred else all_checkpoints[0]


def _formula_to_atoms(formula: str, atom_decoder: list[str]) -> list[str]:
    """
    Parse a molecular formula string and return an ordered atom list
    following atom_decoder ordering (same convention as ChefNMR training data).

    Example: "C6H12O" with decoder ['C','H','O','N'] → ['C']*6 + ['H']*12 + ['O']*1
    """
    import re
    counts: dict[str, int] = {}
    for elem, cnt in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        if elem:
            counts[elem] = counts.get(elem, 0) + (int(cnt) if cnt else 1)
    # Return atoms in decoder order
    atoms = []
    for sym in atom_decoder:
        atoms.extend([sym] * counts.get(sym, 0))
    return atoms


# ── Model loading ────────────────────────────────────────────────────────────

def _get_model(checkpoint: Path):
    """Lazy-load and cache ChefNMR model."""
    global _chefnmr_model, _chefnmr_ckpt_path

    if _chefnmr_model is not None and _chefnmr_ckpt_path == checkpoint:
        return _chefnmr_model

    import torch
    from speqtro.vendors.chefnmr.model import NMRTo3DStructureElucidation

    logger.info("Loading ChefNMR checkpoint: %s", checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = NMRTo3DStructureElucidation.load_from_checkpoint(
        str(checkpoint), map_location=device
    )
    model = model.to(device)
    model.eval()

    _chefnmr_model = model
    _chefnmr_ckpt_path = checkpoint
    return model


# ── Direct inference ─────────────────────────────────────────────────────────

def _predict_direct(
    h1_spectrum: Optional[np.ndarray],
    c13_spectrum: Optional[np.ndarray],
    checkpoint: Path,
    n_samples: int,
    n_steps: int,
    atom_list: Optional[list[str]] = None,
) -> list[str]:
    """
    Run ChefNMR inference in-process.

    Args:
        h1_spectrum: H1 spectrum array (h1_dim floats), or None.
        c13_spectrum: C13 spectrum array (c13_dim floats), or None.
        checkpoint: Path to .ckpt file.
        n_samples: Number of candidate structures to generate.
        n_steps: Diffusion denoising steps.
        atom_list: Ordered atom symbol list from molecular formula (decoder-ordered).
                   If None, atom_one_hot is all-zero (formula-agnostic mode).

    Returns:
        List of unique valid SMILES strings.
    """
    import torch
    from speqtro.vendors.chefnmr.bond_analyzer import atom_features_to_smiles

    model = _get_model(checkpoint)
    device = next(model.parameters()).device

    input_generator = str(model.dataset_args.input_generator)
    atom_decoder = list(model.dataset_args.atom_decoder)
    max_n_atoms = int(model.dataset_args.max_n_atoms)

    # ── Atom one-hot & mask ───────────────────────────────────────────────────
    n_types = len(atom_decoder)
    one_hot_np = np.zeros((max_n_atoms, n_types), dtype=np.float32)
    mask_np = np.zeros(max_n_atoms, dtype=np.float32)

    if atom_list:
        decoder_map = {sym: i for i, sym in enumerate(atom_decoder)}
        for pos, sym in enumerate(atom_list[:max_n_atoms]):
            idx = decoder_map.get(sym, -1)
            if idx >= 0:
                one_hot_np[pos, idx] = 1.0
                mask_np[pos] = 1.0
    else:
        # Formula-agnostic: let model decide; set all positions active
        mask_np[:] = 1.0

    atom_one_hot = torch.tensor(one_hot_np, dtype=torch.float32, device=device).unsqueeze(0)
    atom_mask = torch.tensor(mask_np, dtype=torch.float32, device=device).unsqueeze(0)

    # ── NMR condition tensor ─────────────────────────────────────────────────
    # The model expects a single concatenated tensor [B, h1_dim + c13_dim].
    h1_dim = int(model.dataset_args.input_generator_addn_args["h1nmr"]["input_dim"])
    c13_dim = int(model.dataset_args.input_generator_addn_args["c13nmr"]["input_dim"])

    if input_generator == "H1NMRSpectrum":
        if h1_spectrum is None:
            raise ValueError("Model expects H1 spectrum but none provided.")
        _h1 = _resize_spectrum(h1_spectrum, h1_dim)
        cond = torch.tensor(_h1, dtype=torch.float32, device=device).unsqueeze(0)

    elif input_generator == "C13NMRSpectrum":
        if c13_spectrum is None:
            raise ValueError("Model expects C13 spectrum but none provided.")
        _c13 = _resize_spectrum(c13_spectrum, c13_dim)
        cond = torch.tensor(_c13, dtype=torch.float32, device=device).unsqueeze(0)

    elif input_generator == "H1C13NMRSpectrum":
        _h1 = _resize_spectrum(
            h1_spectrum if h1_spectrum is not None else np.zeros(h1_dim, dtype=np.float32),
            h1_dim,
        )
        _c13 = _resize_spectrum(
            c13_spectrum if c13_spectrum is not None else np.zeros(c13_dim, dtype=np.float32),
            c13_dim,
        )
        # Concatenate into single [B, h1_dim + c13_dim] tensor — required by NMRSpectraEmbedder
        cond = torch.tensor(
            np.concatenate([_h1, _c13]), dtype=torch.float32, device=device
        ).unsqueeze(0)

    else:
        raise ValueError(f"Unknown condition type: {input_generator}")

    model_inputs = {
        "atom_one_hot": atom_one_hot,
        "atom_mask": atom_mask,
        "condition": cond,
    }

    result = model.sample_from_inputs(
        model_inputs=model_inputs,
        n_samples=n_samples,
        num_sampling_steps=n_steps,
    )

    # ── SMILES reconstruction ─────────────────────────────────────────────────
    atom_features = {
        "atom_coords": result["atom_coords"],    # (n_samples, max_n_atoms, 3)
        "atom_one_hot": result["atom_one_hot"],  # (1, max_n_atoms, n_types)
        "atom_mask": result["atom_mask"],         # (1, max_n_atoms)
    }

    try:
        smiles_list, _ = atom_features_to_smiles(
            dataset_name=str(model.dataset_args.name),
            atom_features=atom_features,
            multiplicity=n_samples,
            remove_h=bool(model.dataset_args.remove_h),
            remove_stereo=bool(model.dataset_args.remove_stereo),
            atom_decoder=atom_decoder,
            timeout_seconds=0,
            num_workers=1,
        )
    except Exception as e:
        logger.warning("SMILES reconstruction failed: %s", e)
        smiles_list = [None] * n_samples

    # Deduplicate valid SMILES while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for s in smiles_list:
        if s and s not in seen:
            seen.add(s)
            unique.append(s)

    return unique


def _resize_spectrum(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Resample arr to target_len via linear interpolation if sizes differ."""
    arr = np.asarray(arr, dtype=np.float32)
    if len(arr) == target_len:
        return arr
    return np.interp(
        np.linspace(0, 1, target_len),
        np.linspace(0, 1, len(arr)),
        arr,
    ).astype(np.float32)


# ── speQ tool registration ────────────────────────────────────────────────────

@registry.register(
    name="nmr.elucidation_chefnmr",
    description=(
        "Structure elucidation from NMR spectra using ChefNMR (NeurIPS 2025). "
        "Takes ¹H and/or ¹³C peak lists and returns top SMILES candidates. "
        "Uses atomic diffusion model conditioned on molecular formula + spectra. "
        "Requires downloaded checkpoint weights."
    ),
    category="nmr",
    parameters={
        "molecular_formula": {
            "type": "string",
            "description": (
                "Molecular formula, e.g. 'C10H12O3'. Strongly recommended — the model "
                "conditions on atom types as well as spectra. Derive from HRMS if available."
            ),
        },
        "h1_peaks": {
            "type": "array",
            "description": (
                "¹H NMR peaks as list of dicts with 'shift' (ppm) and optional "
                "'integral'. Example: [{'shift': 7.26, 'integral': 5}]"
            ),
        },
        "c13_peaks": {
            "type": "array",
            "description": (
                "¹³C NMR peaks as list of dicts with 'shift' (ppm). "
                "Provide either h1_peaks or c13_peaks or both."
            ),
        },
        "model_variant": {
            "type": "string",
            "description": (
                "Checkpoint variant to use. Options: 'NP' (natural products, 10 atom types, up to 274 atoms), "
                "'SB' (drug-like CHON only, up to 59 atoms), 'US' (USPTO synthetic, 10 atom types, up to 101 atoms). "
                "Default: auto-selected based on formula. Suffix '-L64' or '-S128' selects model size."
            ),
        },
        "n_candidates": {
            "type": "integer",
            "description": "Number of candidate structures to generate (default: 10).",
        },
        "n_diffusion_steps": {
            "type": "integer",
            "description": "Diffusion sampling steps (default: 50; more = better quality but slower).",
        },
    },
)
def elucidation_chefnmr(
    molecular_formula: Optional[str] = None,
    h1_peaks: list = None,
    c13_peaks: list = None,
    model_variant: Optional[str] = None,
    n_candidates: int = 10,
    n_diffusion_steps: int = 50,
) -> dict:
    """
    Run ChefNMR structure elucidation from NMR peak lists.

    Args:
        molecular_formula: Molecular formula string (e.g. 'C10H12O3'). Recommended.
        h1_peaks: List of ¹H peaks (dicts with 'shift').
        c13_peaks: List of ¹³C peaks (dicts with 'shift').
        model_variant: Checkpoint variant ('NP', 'SB', 'US', or with suffix like 'NP-S128').
        n_candidates: Number of candidate structures to generate.
        n_diffusion_steps: Diffusion sampling steps (50–200).

    Returns:
        Dict with 'smiles_candidates', 'n_candidates', 'citations'.
    """
    if not h1_peaks and not c13_peaks:
        return {"error": "Provide at least one of h1_peaks or c13_peaks."}

    cfg = _get_config()
    all_checkpoints = cfg["all_checkpoints"]

    # Determine if formula has heteroatoms (S, P, F, Cl, Br, I) to guide model selection
    has_heteroatoms = False
    atom_list: Optional[list[str]] = None
    if molecular_formula:
        import re
        hetero_elems = {"S", "P", "F", "Cl", "Br", "I"}
        found = {e for e, _ in re.findall(r"([A-Z][a-z]?)(\d*)", molecular_formula)}
        has_heteroatoms = bool(found & hetero_elems)

    checkpoint = cfg["checkpoint"]
    if model_variant or (checkpoint is None and all_checkpoints):
        checkpoint = _select_checkpoint(model_variant, all_checkpoints, has_heteroatoms)

    if checkpoint is None or not checkpoint.exists():
        return {
            "error": (
                "ChefNMR checkpoint not found. "
                "Run: speqtro fetch-weights --model chefnmr  "
                "or place .ckpt files in ~/.speqtro/models/chefnmr/. "
                "Set config key 'chefnmr.checkpoint' or SPEQ_CHEFNMR_CKPT env var."
            )
        }

    # Build atom list from formula once we know which checkpoint (and its decoder) to use
    if molecular_formula:
        # Load the model briefly to get the decoder, or infer from checkpoint name
        prefix = checkpoint.name.split("-")[0]  # "NP", "SB", or "US"
        decoder = _MODEL_ATOM_DECODERS.get(prefix, _MODEL_ATOM_DECODERS["NP"])
        atom_list = _formula_to_atoms(molecular_formula, decoder)
        if not atom_list:
            return {"error": f"No recognized atoms in formula '{molecular_formula}' for model '{prefix}'."}

    # Convert peak lists to spectrum vectors
    h1_spectrum = None
    if h1_peaks:
        h_grid = _make_grid(H_GRID_MIN, H_GRID_MAX, H_GRID_NPTS)
        h1_spectrum = peaks_to_spectrum(h1_peaks, h_grid, sigma_ppm=0.05)

    c13_spectrum = None
    if c13_peaks:
        c_grid = _make_grid(C_GRID_MIN, C_GRID_MAX, C_GRID_NPTS_80)
        c13_spectrum = peaks_to_spectrum(c13_peaks, c_grid, sigma_ppm=1.5)

    try:
        smiles_candidates = _predict_direct(
            h1_spectrum=h1_spectrum,
            c13_spectrum=c13_spectrum,
            checkpoint=checkpoint,
            n_samples=n_candidates,
            n_steps=n_diffusion_steps,
            atom_list=atom_list,
        )
    except Exception as e:
        logger.exception("ChefNMR inference failed")
        return {"error": f"ChefNMR inference error: {e}"}

    return {
        "smiles_candidates": smiles_candidates,
        "n_candidates": len(smiles_candidates),
        "requested_candidates": n_candidates,
        "molecular_formula": molecular_formula,
        "checkpoint_used": checkpoint.name,
        "model": "ChefNMR (Atomic Diffusion Transformer, NeurIPS 2025)",
        "citations": [
            {
                "title": (
                    "Atomic Diffusion Models for Small Molecule Structure "
                    "Elucidation from NMR Spectra"
                ),
                "authors": "Xiong Z, Zhang Y, Alauddin F, et al.",
                "venue": "NeurIPS 2025",
                "arxiv": "https://arxiv.org/abs/2512.03127",
                "zenodo": "https://zenodo.org/records/17766755",
            }
        ],
        "method": (
            "EDM-based atomic diffusion model. "
            "Inputs: Gaussian-broadened ¹H (10k-dim, -2–10 ppm) + "
            "¹³C (80-dim, 3.42–231.3 ppm) spectra."
        ),
    }


# ── Utility: spectrum conversion exposed for tools/verify ───────────────────

def h1_peaks_to_vector(peaks: list[dict]) -> np.ndarray:
    """Convert ¹H peak list to 10k-dim spectrum vector (for direct model input)."""
    return peaks_to_spectrum(peaks, _make_grid(H_GRID_MIN, H_GRID_MAX, H_GRID_NPTS))


def c13_peaks_to_vector_80(peaks: list[dict]) -> np.ndarray:
    """Convert ¹³C peak list to 80-dim spectrum vector (for ChefNMR C80 input)."""
    return peaks_to_spectrum(peaks, _make_grid(C_GRID_MIN, C_GRID_MAX, C_GRID_NPTS_80),
                              sigma_ppm=1.5)


def c13_peaks_to_vector_10k(peaks: list[dict]) -> np.ndarray:
    """Convert ¹³C peak list to 10k-dim spectrum vector (for ChefNMR C10k input)."""
    return peaks_to_spectrum(peaks, _make_grid(C10K_GRID_MIN, C10K_GRID_MAX, C10K_GRID_NPTS),
                              sigma_ppm=1.5)

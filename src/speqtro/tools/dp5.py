"""
DP5 / DP4 statistical scoring wrapper.

DP5 (Goodman group, Cambridge) computes Bayesian probabilities for NMR-based
structural assignment. It compares experimental ¹H/¹³C NMR shifts against
DFT-predicted shifts and returns a probability that each candidate structure
is the correct one.

DP4 probability: relative (assumes one candidate is correct)
DP5 probability: absolute (each candidate evaluated independently)

Both use kernel density estimation (KDE) over scaled error distributions
trained on a database of known molecular structures with DFT-computed shifts.

Two modes:
  1. DFT mode (full DP5): requires Gaussian/NWChem computed shifts. Slow.
  2. Heuristic mode: uses CASCADE 2.0 predictions as DFT proxies.
     Accuracy lower than true DP5, but no DFT required.

Configuration:
  Set dp5.repo_dir in speqtro config or SPEQ_DP5_DIR env var.

Pretrained statistical models (in DP5 repo):
  c_w_kde_mean_s_0.025.p     — KDE for ¹³C weighted mean
  i_w_kde_mean_s_0.025.p     — KDE for ¹H weighted mean
  folded_scaled_errors.p     — folded scaled error distributions
  atomic_reps.gz             — FCHL atomic representations
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from speqtro.tools import registry

logger = logging.getLogger("speqtro.tools.dp5")


# ── Config helpers ────────────────────────────────────────────────────────────

def _get_repo_dir() -> Optional[Path]:
    try:
        from speqtro.agent.config import Config
        cfg = Config.load()
        d = cfg.get("dp5.repo_dir")
        if d:
            return Path(d)
    except Exception:
        pass

    env_dir = os.environ.get("SPEQ_DP5_DIR")
    if env_dir:
        return Path(env_dir)

    # Auto-detect: ~/.speqtro/models/dp5/ (populated by `speqtro fetch-weights`)
    home_dir = Path.home() / ".speqtro" / "models" / "dp5"
    if home_dir.exists() and (home_dir / "c_w_kde_mean_s_0.025.p").exists():
        return home_dir

    guess = Path(__file__).resolve().parents[4].parent / "DP5"
    if guess.exists() and (guess / "DP5.py").exists():
        return guess

    return None


def _check_dp5_data(repo_dir: Path) -> list[str]:
    """Check which DP5 data files are present. Returns list of missing files."""
    required = [
        "c_w_kde_mean_s_0.025.p",
        "i_w_kde_mean_s_0.025.p",
        "folded_scaled_errors.p",
    ]
    missing = []
    for fname in required:
        if not (repo_dir / fname).exists():
            missing.append(fname)
    return missing


# ── Heuristic scaling parameters ─────────────────────────────────────────────
# These scale factors map CASCADE-predicted shifts to approximate DFT/B3LYP-level
# shifts, mimicking the DP4 scaling applied to Gaussian outputs.
# SOURCE: Calibrated empirically; NOT the same as official DP4 scaling.
# Use for candidate ranking only — not publication-grade probabilities.

_CASCADE_SCALE_C13 = {
    "slope": 1.0,      # CASCADE is already well-calibrated; minimal scaling
    "intercept": 0.0,
    "sigma": 2.5,       # Estimated error std for CASCADE vs. experimental (ppm)
}

_HEURISTIC_SCALE_H1 = {
    "slope": 1.0,
    "intercept": 0.0,
    "sigma": 0.4,       # Typical heuristic ¹H error (ppm)
}


# ── Direct Python scoring (heuristic mode) ───────────────────────────────────

def _heuristic_dp4_score(
    candidates: list[dict],
    exp_c13: list[float],
    exp_h1: list[float],
) -> list[dict]:
    """
    Compute DP4-like probability scores using CASCADE-predicted ¹³C shifts
    and heuristic ¹H shifts as DFT proxies.

    Args:
        candidates: List of {'smiles': str, 'c13_predicted': list[float],
                             'h1_predicted': list[float]}
        exp_c13: Experimental ¹³C shifts (ppm), unassigned
        exp_h1:  Experimental ¹H shifts (ppm), unassigned

    Returns:
        List of {'smiles', 'dp4_probability', 'c13_error_ppm', 'h1_error_ppm'}
        sorted by probability descending.
    """
    import math

    def _match_hungarian(predicted: list[float], experimental: list[float]) -> float:
        """
        Optimal assignment (Hungarian-like) using greedy nearest-neighbor.
        Returns mean absolute error of the best assignment.
        """
        if not predicted or not experimental:
            return float('inf')
        # Simple greedy: assign each experimental to nearest predicted
        pred = sorted(predicted)
        obs = sorted(experimental)
        matched = []
        used = set()
        for o in obs:
            diffs = [(abs(o - p), i) for i, p in enumerate(pred) if i not in used]
            if not diffs:
                break
            diffs.sort()
            best_diff, best_idx = diffs[0]
            matched.append(best_diff)
            used.add(best_idx)
        return sum(matched) / len(matched) if matched else float('inf')

    def _log_likelihood(predicted: list[float], experimental: list[float],
                        sigma: float) -> float:
        """Gaussian log-likelihood for unassigned shift comparison."""
        if not predicted or not experimental:
            return 0.0
        mae = _match_hungarian(predicted, experimental)
        if mae == float('inf'):
            return -100.0
        # Simple Gaussian: LL = -n/2 * log(2*pi*sigma^2) - n*mae^2 / (2*sigma^2)
        # Simplified to just the exponential term for ranking
        return -mae ** 2 / (2 * sigma ** 2)

    log_likelihoods = []
    for cand in candidates:
        ll_c13 = _log_likelihood(
            cand.get("c13_predicted", []), exp_c13, _CASCADE_SCALE_C13["sigma"]
        )
        ll_h1 = _log_likelihood(
            cand.get("h1_predicted", []), exp_h1, _HEURISTIC_SCALE_H1["sigma"]
        )
        log_likelihoods.append(ll_c13 + ll_h1)

    # Convert log-likelihoods to DP4-style probabilities (softmax)
    max_ll = max(log_likelihoods) if log_likelihoods else 0
    exp_lls = [math.exp(ll - max_ll) for ll in log_likelihoods]
    total = sum(exp_lls)

    results = []
    for cand, ll, exp_ll in zip(candidates, log_likelihoods, exp_lls):
        dp4_prob = round(100 * exp_ll / total, 2) if total > 0 else 0.0
        # Also compute per-nucleus MAE for display
        c13_mae = _match_hungarian(cand.get("c13_predicted", []), exp_c13)
        h1_mae = _match_hungarian(cand.get("h1_predicted", []), exp_h1)
        results.append({
            "smiles": cand["smiles"],
            "dp4_probability_percent": dp4_prob,
            "c13_mae_ppm": round(c13_mae, 3) if c13_mae != float('inf') else None,
            "h1_mae_ppm": round(h1_mae, 3) if h1_mae != float('inf') else None,
            "log_likelihood": round(ll, 4),
        })

    results.sort(key=lambda x: x["dp4_probability_percent"], reverse=True)
    return results


def _predict_c13_for_smiles(smiles: str) -> list[float]:
    """Predict ¹³C shifts for a SMILES using CASCADE (fallback: heuristics)."""
    try:
        from speqtro.tools.cascade import predict_c13_cascade
        result = predict_c13_cascade(smiles)
        if "error" in result:
            raise ValueError(result["error"])
        predictions = result.get("predictions", {})
        shifts = []
        for atom_preds in predictions.values():
            shifts.extend([p["shift_ppm"] for p in atom_preds])
        return sorted(shifts)
    except Exception as e:
        logger.warning("CASCADE prediction failed for %s: %s", smiles, e)
        # Fall back to heuristic ranges (midpoints)
        try:
            from speqtro.tools.nmr import predict_c13_shifts
            result = predict_c13_shifts(smiles)
            preds = result.get("predictions", [])
            return sorted([p["estimated_shift_ppm"] for p in preds])
        except Exception:
            return []


def _predict_h1_for_smiles(smiles: str) -> list[float]:
    """Predict ¹H shifts for a SMILES using heuristic rules."""
    try:
        from speqtro.tools.nmr import predict_h1_shifts
        result = predict_h1_shifts(smiles)
        preds = result.get("predictions", [])
        # Expand each prediction by num_H (repeat midpoint for each proton)
        shifts = []
        for p in preds:
            for _ in range(max(1, int(p.get("num_H", 1)))):
                shifts.append(p["estimated_shift_ppm"])
        return sorted(shifts)
    except Exception:
        return []


# ── Vendored DP4 scorer integration ──────────────────────────────────────────

def _try_vendored_dp4(
    candidates: list[dict],
    exp_c13: list[float],
    exp_h1: list[float],
) -> tuple[str, Optional[list[dict]]]:
    """
    Attempt to score candidates using the vendored full DP4 statistical engine
    (from speqtro.vendors.dp5). This uses the proper Goodman-group DP4 method:
    internal linear scaling + single-Gaussian error model per atom.

    Falls back gracefully if the vendored module is unavailable.

    Returns:
        (scoring_method_string, results_list_or_None)
    """
    try:
        from speqtro.vendors.dp5 import dp4_score_unassigned, DP4Result
    except ImportError:
        logger.debug("Vendored DP4 scorer not available, will use heuristic fallback")
        return ("", None)

    try:
        calc_c_per_isomer = [cand.get("c13_predicted", []) for cand in candidates]
        calc_h_per_isomer = [cand.get("h1_predicted", []) for cand in candidates]

        # Check that at least some candidates have predictions
        has_c = any(len(c) > 0 for c in calc_c_per_isomer) and len(exp_c13) > 0
        has_h = any(len(h) > 0 for h in calc_h_per_isomer) and len(exp_h1) > 0

        if not has_c and not has_h:
            logger.debug("No predicted shifts available for vendored DP4")
            return ("", None)

        # Use vendored DP4 with default B3LYP error parameters
        result: DP4Result = dp4_score_unassigned(
            calc_c_per_isomer=calc_c_per_isomer if has_c else [[] for _ in candidates],
            exp_c=exp_c13 if has_c else [],
            calc_h_per_isomer=calc_h_per_isomer if has_h else [[] for _ in candidates],
            exp_h=exp_h1 if has_h else [],
        )

        # Format results in the same shape as the heuristic scorer
        scores = []
        for i, cand in enumerate(candidates):
            c13_mae = None
            h1_mae = None
            if has_c and i < len(result.c_errors):
                errs = result.c_errors[i]
                c13_mae = round(sum(abs(e) for e in errs) / len(errs), 3) if errs else None
            if has_h and i < len(result.h_errors):
                errs = result.h_errors[i]
                h1_mae = round(sum(abs(e) for e in errs) / len(errs), 3) if errs else None

            scores.append({
                "smiles": cand["smiles"],
                "dp4_probability_percent": round(100.0 * result.combined_dp4[i], 2),
                "carbon_dp4_percent": round(100.0 * result.carbon_dp4[i], 2) if has_c else None,
                "proton_dp4_percent": round(100.0 * result.proton_dp4[i], 2) if has_h else None,
                "c13_mae_ppm": c13_mae,
                "h1_mae_ppm": h1_mae,
            })

        scores.sort(key=lambda x: x["dp4_probability_percent"], reverse=True)

        method = "DP4-vendored (Goodman single-Gaussian, CASCADE 13C + empirical 1H)"
        logger.info("Vendored DP4 scoring succeeded for %d candidates", len(candidates))
        return (method, scores)

    except Exception as e:
        logger.warning("Vendored DP4 scoring failed: %s — falling back to heuristic", e)
        return ("", None)


# ── speQ tool registration ────────────────────────────────────────────────────

@registry.register(
    name="nmr.dp4_score_candidates",
    description=(
        "Score a set of candidate structures against experimental NMR shifts "
        "using DP4-style Bayesian probability. Uses CASCADE 2.0 for ¹³C shift "
        "prediction (heuristic for ¹H) as DFT proxies. "
        "Returns DP4 probability % for each candidate. "
        "For publication-grade DP5 probabilities, use the full DP5 pipeline "
        "with Gaussian-computed shifts (requires dp5.repo_dir config)."
    ),
    category="nmr",
    parameters={
        "candidate_smiles": {
            "type": "array",
            "description": "List of candidate SMILES strings to score.",
        },
        "c13_experimental": {
            "type": "array",
            "description": "Experimental ¹³C NMR shifts (ppm), unassigned list of floats.",
        },
        "h1_experimental": {
            "type": "array",
            "description": "Experimental ¹H NMR shifts (ppm), unassigned list of floats.",
        },
        "use_dft": {
            "type": "boolean",
            "description": (
                "If true, run full DP5 pipeline with DFT-computed shifts (requires "
                "Gaussian and dp5.repo_dir config). Default: false (uses CASCADE proxy)."
            ),
        },
    },
)
def dp4_score_candidates(
    candidate_smiles: list[str],
    c13_experimental: Optional[list] = None,
    h1_experimental: Optional[list] = None,
    use_dft: bool = False,
) -> dict:
    """
    Score candidates by DP4-like probability against experimental NMR shifts.

    Args:
        candidate_smiles: List of SMILES to score.
        c13_experimental: Experimental ¹³C shifts (ppm).
        h1_experimental: Experimental ¹H shifts (ppm).
        use_dft: If True, attempt full DP5 via subprocess (requires Gaussian).

    Returns:
        Dict with candidates ranked by DP4 probability.
    """
    if not candidate_smiles:
        return {"error": "No candidate SMILES provided."}
    if not c13_experimental and not h1_experimental:
        return {"error": "Provide at least c13_experimental or h1_experimental shifts."}

    exp_c13 = [float(x) for x in (c13_experimental or [])]
    exp_h1 = [float(x) for x in (h1_experimental or [])]

    if use_dft:
        repo_dir = _get_repo_dir()
        if repo_dir is None:
            return {
                "error": (
                    "DP5 repository not found for DFT mode. "
                    "Set config key 'dp5.repo_dir' or SPEQ_DP5_DIR env var. "
                    "Falling back: set use_dft=false to use CASCADE proxy instead."
                )
            }
        missing = _check_dp5_data(repo_dir)
        if missing:
            return {
                "error": f"DP5 data files missing from {repo_dir}: {missing}. "
                         "Cannot run full DP5. Use use_dft=false for CASCADE-based scoring."
            }
        # Full DP5 would require Gaussian outputs — not implemented here.
        # The workflow would be: generate 3D conformers → Gaussian NMR calc → DP5 scoring.
        return {
            "error": (
                "Full DFT-based DP5 requires Gaussian NMR calculations. "
                "Run the DP5 pipeline directly: "
                "python PyDP4.py candidate.sdf experimental_nmr.txt -w gmns -s cdcl3 "
                f"(from {repo_dir}). "
                "Use use_dft=false for CASCADE-based DP4 scoring instead."
            ),
            "dp5_repo": str(repo_dir) if repo_dir else None,
        }

    # Predict shifts for each candidate using CASCADE / heuristics
    logger.info("dp4_score_candidates: predicting shifts for %d candidates", len(candidate_smiles))

    candidates_with_preds = []
    for smi in candidate_smiles:
        c13_pred = _predict_c13_for_smiles(smi) if exp_c13 else []
        h1_pred = _predict_h1_for_smiles(smi) if exp_h1 else []
        candidates_with_preds.append({
            "smiles": smi,
            "c13_predicted": c13_pred,
            "h1_predicted": h1_pred,
        })

    # ── Try vendored full DP4 scorer first, fall back to heuristic ────────
    scoring_method, scores = _try_vendored_dp4(
        candidates_with_preds, exp_c13, exp_h1
    )

    if scores is None:
        # Vendored scorer unavailable or failed — use heuristic fallback
        scores = _heuristic_dp4_score(candidates_with_preds, exp_c13, exp_h1)
        scoring_method = "DP4-heuristic (CASCADE 13C + empirical 1H, no DFT)"

    return {
        "ranked_candidates": scores,
        "top_candidate": scores[0] if scores else None,
        "scoring_method": scoring_method,
        "warning": (
            "These probabilities use CASCADE predictions as DFT proxies. "
            "They are suitable for candidate ranking but NOT for publication-grade "
            "structural assignment. Use full DP5 with Gaussian for rigorous results."
        ),
        "n_candidates": len(scores),
        "c13_peaks_used": len(exp_c13),
        "h1_peaks_used": len(exp_h1),
        "citations": [
            {
                "title": "DP4 probability — Smith SG, Goodman JM. JACS 2010, 132, 12946",
                "note": (
                    f"Scoring method: {scoring_method}. "
                    "For full DP5: Ermanis K et al., Org. Biomol. Chem. 2017"
                ),
                "repo": str(_get_repo_dir()) if _get_repo_dir() else "not configured",
            }
        ],
    }

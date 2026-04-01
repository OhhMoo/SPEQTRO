"""
Full multi-spectroscopy elucidation pipeline.

Orchestrates ALL available speQ tools in a single pass:

  1. nmrglue / spectral_input  — parse raw NMR files (Bruker/Varian/JDX)
  2. SSIN                      — IR functional group detection (.jdx)
  3. ChefNMR / pipeline.elucidation — generate candidate structures from NMR
  4. CASCADE 2.0               — ¹³C shift prediction per candidate
  5. DP4-heuristic             — rank candidates by NMR probability
  6. ICEBERG (ms-pred)         — MS/MS prediction + cosine scoring (optional)
  7. Final ensemble scoring    — combine NMR + MS evidence

Registered tool: pipeline.full_elucidation

Claude routes to this tool when the user has multi-spectroscopy data
(NMR + IR and/or MS) and no known product SMILES (EXPLORE mode).
"""

from __future__ import annotations

import logging
from typing import Optional

from speqtro.tools import registry

logger = logging.getLogger("speqtro.modes.pipeline")


# ── Weights for ensemble scoring ─────────────────────────────────────────────
# Adjust these to reflect relative reliability of each evidence type.
_ENSEMBLE_WEIGHTS = {
    "nmr_dp4":       0.50,   # DP4-heuristic NMR probability
    "c13_mad":       0.25,   # ¹³C MAD from CASCADE (lower is better)
    "ms_cosine":     0.20,   # MS/MS cosine similarity
    "ir_compatible": 0.05,   # IR functional group compatibility bonus
}

# Thresholds
_VERDICT_THRESHOLDS = {
    "CONFIRMED":  85.0,
    "LIKELY":     65.0,
    "UNCERTAIN":  40.0,
}


def _verdict(score: float) -> str:
    if score >= _VERDICT_THRESHOLDS["CONFIRMED"]:
        return "CONFIRMED"
    if score >= _VERDICT_THRESHOLDS["LIKELY"]:
        return "LIKELY"
    if score >= _VERDICT_THRESHOLDS["UNCERTAIN"]:
        return "UNCERTAIN"
    return "UNLIKELY"


# ── Sub-step helpers ──────────────────────────────────────────────────────────

def _step_ir_detection(ir_file: Optional[str], python_exe: str) -> dict:
    """Run SSIN IR functional group detection if JDX file is provided."""
    if not ir_file:
        return {"skipped": True, "reason": "No IR file provided"}
    try:
        from speqtro.tools.ssin import detect_functional_groups_ssin
        return detect_functional_groups_ssin(jdx_file=ir_file, python_exe=python_exe)
    except Exception as e:
        logger.warning("SSIN IR step failed: %s", e)
        return {"error": str(e), "skipped": False}


def _step_generate_candidates(
    h1_peaks: list[dict],
    c13_peaks: list[dict],
    n_candidates: int,
    python_exe: str,
) -> dict:
    """Generate candidate structures via ChefNMR (EXPLORE pipeline)."""
    try:
        from speqtro.tools.chefnmr import elucidation_chefnmr
        return elucidation_chefnmr(
            h1_peaks=h1_peaks or None,
            c13_peaks=c13_peaks or None,
            n_candidates=n_candidates,
            python_exe=python_exe,
        )
    except Exception as e:
        logger.warning("ChefNMR step failed: %s", e)
        return {"error": str(e), "smiles_candidates": []}


def _step_cascade_per_candidate(smiles: str, c13_obs: list[float]) -> dict:
    """
    Predict ¹³C shifts for a candidate via CASCADE and compute MAD vs observed.
    Returns {'mad_ppm', 'predicted_shifts', 'error'}.
    """
    try:
        from speqtro.tools.cascade import predict_c13_cascade
        result = predict_c13_cascade(smiles)
        if "error" in result:
            return {"mad_ppm": None, "error": result["error"]}

        pred_groups = result.get("predictions", {})
        pred_shifts: list[float] = []
        for atom_preds in pred_groups.values():
            pred_shifts.extend([p["shift_ppm"] for p in atom_preds])

        if not pred_shifts or not c13_obs:
            return {"mad_ppm": None, "predicted_shifts": pred_shifts, "error": "No shifts to compare"}

        # Greedy nearest-neighbor matching
        obs = sorted(c13_obs)
        pred = sorted(pred_shifts)
        used = set()
        errors = []
        for o in obs:
            diffs = [(abs(o - p), i) for i, p in enumerate(pred) if i not in used]
            if not diffs:
                break
            diffs.sort()
            best_diff, best_idx = diffs[0]
            errors.append(best_diff)
            used.add(best_idx)

        mad = round(sum(errors) / len(errors), 3) if errors else None
        return {"mad_ppm": mad, "predicted_shifts": sorted(pred_shifts)}
    except Exception as e:
        return {"mad_ppm": None, "error": str(e)}


def _step_ms_scoring(
    smiles: str,
    experimental_ms: list[dict],
    collision_energy: float,
    adduct: str,
    python_exe: str,
) -> dict:
    """Predict MS/MS for a candidate and compute cosine similarity vs experimental."""
    if not experimental_ms:
        return {"cosine_similarity": None, "skipped": True}
    try:
        from speqtro.tools.mspred import score_candidate_vs_experimental
        return score_candidate_vs_experimental(
            candidate_smiles=smiles,
            experimental_peaks=experimental_ms,
            collision_energy=collision_energy,
            adduct=adduct,
            python_exe=python_exe,
        )
    except Exception as e:
        return {"cosine_similarity": None, "error": str(e)}


def _step_dp4_ranking(
    candidate_smiles: list[str],
    c13_obs: list[float],
    h1_obs: list[float],
) -> dict:
    """Rank candidates by DP4-heuristic NMR probability."""
    if not candidate_smiles:
        return {"ranked_candidates": [], "error": "No candidates to score"}
    try:
        from speqtro.tools.dp5 import dp4_score_candidates
        return dp4_score_candidates(
            candidate_smiles=candidate_smiles,
            c13_experimental=c13_obs or None,
            h1_experimental=h1_obs or None,
            use_dft=False,
        )
    except Exception as e:
        logger.warning("DP4 scoring step failed: %s", e)
        return {"ranked_candidates": [], "error": str(e)}


def _ir_compatible_bonus(smiles: str, ir_result: dict) -> float:
    """
    Return an IR compatibility bonus in [0, 1]:
    fraction of SSIN-detected functional groups confirmed by SMARTS in the candidate.
    """
    if not ir_result or ir_result.get("skipped"):
        return 0.5  # neutral when no IR data

    detected = ir_result.get("detected_functional_groups", [])
    if not detected:
        return 0.5

    try:
        from rdkit import Chem
        from rdkit.Chem import MolFromSmarts
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.5

        import json
        from pathlib import Path
        fg_path = Path(__file__).resolve().parents[1] / "data" / "functional_group_shifts.json"
        with open(fg_path, encoding="utf-8") as f:
            fg_data = json.load(f)

        fg_map = {fg["name"]: fg.get("smarts") for fg in fg_data.get("functional_groups", [])}

        matched = 0
        for fg_name in detected:
            smarts = fg_map.get(fg_name)
            if smarts:
                patt = Chem.MolFromSmarts(smarts)
                if patt and mol.HasSubstructMatch(patt):
                    matched += 1
                else:
                    # Penalise: SSIN detected it but SMARTS says it's absent
                    matched -= 0.5

        raw = matched / len(detected)
        return max(0.0, min(1.0, raw + 0.5))  # centre at 0.5
    except Exception:
        return 0.5


def _ensemble_score(
    dp4_prob: Optional[float],         # 0–100
    c13_mad: Optional[float],          # ppm (lower is better)
    ms_cosine: Optional[float],        # 0–1
    ir_bonus: float,                   # 0–1
) -> float:
    """
    Compute weighted ensemble score in [0, 100].
    Missing evidence components are replaced by a neutral 50%.
    """
    scores: dict[str, float] = {}

    # DP4 probability (already 0–100, normalise to 0–1)
    scores["nmr_dp4"] = (dp4_prob / 100.0) if dp4_prob is not None else 0.5

    # ¹³C MAD: 0 ppm = perfect (1.0), 5+ ppm = terrible (0.0)
    if c13_mad is not None:
        scores["c13_mad"] = max(0.0, 1.0 - c13_mad / 5.0)
    else:
        scores["c13_mad"] = 0.5  # neutral

    # MS/MS cosine
    scores["ms_cosine"] = ms_cosine if ms_cosine is not None else 0.5

    # IR bonus
    scores["ir_compatible"] = ir_bonus

    weighted = sum(
        scores[k] * _ENSEMBLE_WEIGHTS[k]
        for k in _ENSEMBLE_WEIGHTS
    )
    return round(weighted * 100, 1)


# ── pipeline.full_elucidation tool ───────────────────────────────────────────

@registry.register(
    name="pipeline.full_elucidation",
    description=(
        "Full multi-spectroscopy elucidation pipeline. "
        "Orchestrates: (1) SSIN IR functional group detection, "
        "(2) ChefNMR structure generation from NMR peaks, "
        "(3) CASCADE ¹³C shift scoring per candidate, "
        "(4) DP4-heuristic NMR probability ranking, "
        "(5) ICEBERG MS/MS cosine scoring (if MS data provided). "
        "Returns an ensemble-ranked list of candidate SMILES with per-evidence scores. "
        "Use for EXPLORE mode when no known product SMILES is available."
    ),
    category="pipeline",
    parameters={
        "h1_peaks": {
            "type": "array",
            "description": "¹H NMR peaks: list of {'shift': float, 'integral': float}.",
        },
        "c13_peaks": {
            "type": "array",
            "description": "¹³C NMR peaks: list of {'shift': float}.",
        },
        "ir_file": {
            "type": "string",
            "description": "Absolute path to JCAMP-DX (.jdx) IR spectrum file (optional).",
        },
        "ms_peaks": {
            "type": "array",
            "description": (
                "Experimental MS/MS peaks: list of {'mz': float, 'intensity': float} (optional). "
                "If provided, ICEBERG cosine scoring is performed for each candidate."
            ),
        },
        "solvent": {
            "type": "string",
            "description": "NMR solvent (e.g. 'CDCl3', 'DMSO-d6'). Default: 'CDCl3'.",
        },
        "collision_energy": {
            "type": "number",
            "description": "MS/MS collision energy in eV. Default: 20.",
        },
        "adduct": {
            "type": "string",
            "description": "MS ionization adduct. Default: '[M+H]+'.",
        },
        "n_candidates": {
            "type": "integer",
            "description": "Number of ChefNMR candidates to generate. Default: 10.",
        },
        "candidate_smiles": {
            "type": "array",
            "description": (
                "Optional: provide your own candidate SMILES list instead of "
                "running ChefNMR. Useful when you have prior structural candidates."
            ),
        },
        "python_exe": {
            "type": "string",
            "description": "Python interpreter (must have ChefNMR + SSIN deps). Default: 'python'.",
        },
    },
)
def full_elucidation(
    h1_peaks: Optional[list] = None,
    c13_peaks: Optional[list] = None,
    ir_file: Optional[str] = None,
    ms_peaks: Optional[list] = None,
    solvent: str = "CDCl3",
    collision_energy: float = 20.0,
    adduct: str = "[M+H]+",
    n_candidates: int = 10,
    candidate_smiles: Optional[list] = None,
    python_exe: str = "python",
) -> dict:
    """
    Full multi-spectroscopy elucidation pipeline.

    Orchestration order:
      1. SSIN IR detection  (if ir_file provided)
      2. ChefNMR candidate generation (if candidate_smiles not provided)
      3. CASCADE ¹³C scoring per candidate
      4. DP4-heuristic ranking
      5. ICEBERG MS/MS scoring (if ms_peaks provided)
      6. Ensemble score → final ranked list
    """
    if not h1_peaks and not c13_peaks and not candidate_smiles:
        return {
            "error": "Provide at least h1_peaks, c13_peaks, or candidate_smiles."
        }

    # Normalise peak dicts
    h1 = h1_peaks or []
    c13 = c13_peaks or []

    # Extract flat shift lists for scoring
    c13_obs = [float(p.get("shift", p.get("ppm", 0))) for p in c13]
    h1_obs = [float(p.get("shift", p.get("ppm", 0))) for p in h1]

    pipeline_steps: dict = {}

    # ── Step 1: IR ──────────────────────────────────────────────────────────
    logger.info("pipeline.full_elucidation: step 1 — IR detection")
    ir_result = _step_ir_detection(ir_file, python_exe)
    pipeline_steps["ir"] = ir_result

    # ── Step 2: Candidate generation ────────────────────────────────────────
    logger.info("pipeline.full_elucidation: step 2 — candidate generation")
    if candidate_smiles:
        candidates = [str(s) for s in candidate_smiles]
        pipeline_steps["candidate_generation"] = {
            "source": "user-provided",
            "n_candidates": len(candidates),
        }
    else:
        if not h1 and not c13:
            return {"error": "Cannot run ChefNMR without h1_peaks or c13_peaks."}
        chef_result = _step_generate_candidates(h1, c13, n_candidates, python_exe)
        pipeline_steps["candidate_generation"] = chef_result
        if "error" in chef_result and not chef_result.get("smiles_candidates"):
            return {
                "error": f"Candidate generation failed: {chef_result['error']}",
                "pipeline_steps": pipeline_steps,
            }
        candidates = chef_result.get("smiles_candidates", [])

    if not candidates:
        return {
            "error": "No candidate structures generated.",
            "pipeline_steps": pipeline_steps,
        }

    # ── Step 3 + 4: DP4 ranking (runs CASCADE internally) ──────────────────
    logger.info("pipeline.full_elucidation: steps 3+4 — DP4 ranking for %d candidates", len(candidates))
    dp4_result = _step_dp4_ranking(candidates, c13_obs, h1_obs)
    pipeline_steps["dp4_ranking"] = dp4_result

    # Build DP4 prob lookup
    dp4_probs: dict[str, float] = {}
    for row in dp4_result.get("ranked_candidates", []):
        dp4_probs[row["smiles"]] = row.get("dp4_probability_percent", 0.0)

    # ── Step 3 (CASCADE MAD): run per-candidate if we have ¹³C obs ─────────
    cascade_mads: dict[str, Optional[float]] = {}
    if c13_obs:
        logger.info("pipeline.full_elucidation: CASCADE MAD for %d candidates", len(candidates))
        for smi in candidates:
            casc = _step_cascade_per_candidate(smi, c13_obs)
            cascade_mads[smi] = casc.get("mad_ppm")
    else:
        cascade_mads = {smi: None for smi in candidates}

    # ── Step 5: MS/MS scoring ───────────────────────────────────────────────
    ms_cosines: dict[str, Optional[float]] = {}
    if ms_peaks:
        logger.info("pipeline.full_elucidation: step 5 — MS/MS scoring for %d candidates", len(candidates))
        ms_step_results = {}
        for smi in candidates:
            ms_res = _step_ms_scoring(smi, ms_peaks, collision_energy, adduct, python_exe)
            ms_cosines[smi] = ms_res.get("cosine_similarity")
            ms_step_results[smi] = ms_res
        pipeline_steps["ms_scoring"] = ms_step_results
    else:
        ms_cosines = {smi: None for smi in candidates}
        pipeline_steps["ms_scoring"] = {"skipped": True, "reason": "No MS/MS data provided"}

    # ── Step 6: Ensemble scoring ─────────────────────────────────────────────
    logger.info("pipeline.full_elucidation: step 6 — ensemble scoring")
    ranked: list[dict] = []

    for smi in candidates:
        ir_bonus = _ir_compatible_bonus(smi, ir_result)
        dp4_p = dp4_probs.get(smi)
        c13_mad = cascade_mads.get(smi)
        ms_cos = ms_cosines.get(smi)

        ens_score = _ensemble_score(dp4_p, c13_mad, ms_cos, ir_bonus)

        ranked.append({
            "smiles": smi,
            "ensemble_score": ens_score,
            "verdict": _verdict(ens_score),
            "evidence": {
                "dp4_probability_percent": dp4_p,
                "c13_mad_ppm": c13_mad,
                "ms_cosine_similarity": ms_cos,
                "ir_compatibility_score": round(ir_bonus, 3),
            },
        })

    ranked.sort(key=lambda x: x["ensemble_score"], reverse=True)

    top = ranked[0] if ranked else None

    return {
        "top_candidate": top,
        "ranked_candidates": ranked,
        "n_candidates_evaluated": len(ranked),
        "pipeline_steps": {
            "ir_detection": pipeline_steps.get("ir", {}),
            "candidate_generation": pipeline_steps.get("candidate_generation", {}),
            "dp4_ranking": {
                "method": dp4_result.get("scoring_method"),
                "warning": dp4_result.get("warning"),
            },
            "ms_scoring": pipeline_steps.get("ms_scoring", {}),
        },
        "ensemble_weights": _ENSEMBLE_WEIGHTS,
        "verdict_thresholds": _VERDICT_THRESHOLDS,
        "input_summary": {
            "h1_peaks_count": len(h1),
            "c13_peaks_count": len(c13),
            "ir_file": ir_file,
            "ms_peaks_count": len(ms_peaks) if ms_peaks else 0,
            "solvent": solvent,
        },
        "citations": [
            {
                "tool": "ChefNMR",
                "title": "Atomic Diffusion Models for Small Molecule Structure Elucidation",
                "venue": "NeurIPS 2025",
            },
            {
                "tool": "CASCADE 2.0",
                "title": "Real-time Prediction of 13C NMR Shifts",
                "url": "https://nova.chem.colostate.edu/v2/cascade/",
            },
            {
                "tool": "DP4/DP5",
                "title": "DP5: Quantifying the Probability of the Correct Structure Assignment",
                "note": "Heuristic approximation using CASCADE; not full DFT DP5",
            },
            {
                "tool": "ICEBERG (ms-pred)",
                "title": "ICEBERG: Fragment-level MS/MS Prediction",
                "repo": "https://github.com/samgoldman97/ms-pred",
            },
            {
                "tool": "SSIN",
                "title": "SSIN: Substructure-directed IR Spectrum Interpreter Network",
            },
        ],
    }

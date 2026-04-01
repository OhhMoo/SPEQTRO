"""
Pipeline tools for the VERIFY and EXPLORE workflows.

All chemistry is computed here — Claude only routes to these tools and
reports the output verbatim.

Tools registered:
  pipeline.verify_product  — VERIFY mode: full 4-step verification pipeline
  pipeline.elucidation     — EXPLORE mode: ChefNMR structure candidates
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from speqtro.tools import registry

logger = logging.getLogger("speqtro.modes.verify")

_DATA_DIR = Path(__file__).resolve().parents[1] / "data"


# ── Data loaders ─────────────────────────────────────────────────────────────

def _load_functional_groups() -> list[dict]:
    path = _DATA_DIR / "functional_group_shifts.json"
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("functional_groups", [])
    except Exception as e:
        logger.warning("Could not load functional_group_shifts.json: %s", e)
        return []


def _load_contaminants() -> list[dict]:
    path = _DATA_DIR / "contaminants.json"
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("contaminants", [])
    except Exception as e:
        logger.warning("Could not load contaminants.json: %s", e)
        return []


def _load_solvent_peaks() -> dict:
    path = _DATA_DIR / "solvent_peaks.json"
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Could not load solvent_peaks.json: %s", e)
        return {}


# ── RDKit availability check ─────────────────────────────────────────────────

def _rdkit_available() -> bool:
    try:
        from rdkit import Chem  # noqa: F401
        return True
    except ImportError:
        return False


# ── Step 1: DiagnosticMarkerGenerator ────────────────────────────────────────

def _run_diagnostic_markers(
    product_smiles: str,
    sm_smiles: Optional[str],
    observed_peaks: list[dict],
    tolerance_ppm: float = 0.3,
) -> dict:
    """
    Use RDKit SMARTS to detect functional groups present in product but not SM
    (and vice versa), then check whether observed peaks fall in expected shift ranges.

    Returns dict: {group_name: {expected_range, role, found, at_ppm}}
    """
    fg_defs = _load_functional_groups()
    result: dict[str, dict] = {}

    if not _rdkit_available():
        return {"_error": "RDKit not available — diagnostic markers skipped"}

    from rdkit import Chem

    product_mol = Chem.MolFromSmiles(product_smiles)
    if product_mol is None:
        return {"_error": f"Invalid product SMILES: {product_smiles}"}

    sm_mol = Chem.MolFromSmiles(sm_smiles) if sm_smiles else None

    for fg in fg_defs:
        name = fg["name"]
        smarts = fg.get("smarts")
        nucleus = fg.get("nucleus", "13C")
        shift_range = fg.get("shift_range")
        if not smarts or not shift_range:
            continue

        try:
            patt = Chem.MolFromSmarts(smarts)
            if patt is None:
                continue
        except Exception:
            continue

        in_product = product_mol.HasSubstructMatch(patt)
        in_sm = sm_mol.HasSubstructMatch(patt) if sm_mol else None

        # Determine role: "new" (should appear), "lost" (should disappear), or "present"
        if in_sm is not None:
            if in_product and not in_sm:
                role = "new"
            elif not in_product and in_sm:
                role = "lost"
            elif in_product and in_sm:
                role = "present"
            else:
                continue  # absent in both — skip
        else:
            if not in_product:
                continue
            role = "present"

        lo, hi = float(shift_range[0]), float(shift_range[1])

        if role == "lost":
            # Check that NO observed peak falls in this range
            matching_peaks = [
                p for p in observed_peaks
                if p.get("nucleus", nucleus) == nucleus
                and lo - tolerance_ppm <= float(p.get("shift", p.get("ppm", -9999))) <= hi + tolerance_ppm
            ]
            found = len(matching_peaks) == 0  # "found absent" = good
            result[name] = {
                "expected": "lost",
                "shift_range": f"{lo}–{hi} ppm",
                "found_absent": found,
                "spurious_peaks": [p.get("shift", p.get("ppm")) for p in matching_peaks],
            }
        else:
            # "new" or "present" — check that at least one peak falls in range
            matching_peaks = [
                p for p in observed_peaks
                if p.get("nucleus", nucleus) == nucleus
                and lo - tolerance_ppm <= float(p.get("shift", p.get("ppm", -9999))) <= hi + tolerance_ppm
            ]
            found = len(matching_peaks) > 0
            result[name] = {
                "expected_range": f"{lo}–{hi} ppm",
                "role": role,
                "found": found,
                "at_ppm": matching_peaks[0].get("shift", matching_peaks[0].get("ppm")) if matching_peaks else None,
            }

    return result


# ── Step 2: CascadePeakMatcher ────────────────────────────────────────────────

def _run_cascade_peak_matcher(
    product_smiles: str,
    observed_c13_peaks: list[dict],
) -> dict:
    """
    Predict ¹³C shifts for product_smiles via CASCADE 2.0 (or heuristic fallback),
    then compare to observed peaks.

    Returns: {mean_deviation_ppm, matched, unmatched, per_atom_table, method}
    """
    # Try CASCADE first
    try:
        from speqtro.tools.cascade import predict_c13_cascade
        cascade_result = predict_c13_cascade(product_smiles)
    except Exception as e:
        cascade_result = {"error": str(e)}

    if "error" in cascade_result:
        # Heuristic fallback: skip matching, return unavailable
        return {
            "mean_deviation_ppm": None,
            "matched": None,
            "unmatched": None,
            "per_atom_table": [],
            "method": "CASCADE unavailable",
            "cascade_error": cascade_result["error"],
        }

    # Extract predicted shifts from CASCADE result
    predictions_by_smiles = cascade_result.get("predictions", {})
    predicted_atoms = predictions_by_smiles.get(product_smiles, [])
    if not predicted_atoms:
        # Try first key if SMILES key doesn't match exactly
        if predictions_by_smiles:
            predicted_atoms = next(iter(predictions_by_smiles.values()))

    if not predicted_atoms:
        return {
            "mean_deviation_ppm": None,
            "matched": 0,
            "unmatched": len(observed_c13_peaks),
            "per_atom_table": [],
            "method": "CASCADE 2.0",
            "note": "No predictions returned",
        }

    # Match each predicted atom to nearest observed peak
    observed_shifts = [
        float(p.get("shift", p.get("ppm", 0)))
        for p in observed_c13_peaks
    ]

    per_atom = []
    matched_obs_indices: set[int] = set()
    deviations: list[float] = []

    for atom in predicted_atoms:
        pred_ppm = atom["shift_ppm"]
        atom_id = atom["atom_id"]

        if not observed_shifts:
            per_atom.append({
                "atom_id": atom_id,
                "predicted_ppm": pred_ppm,
                "matched_ppm": None,
                "deviation_ppm": None,
                "matched": False,
            })
            continue

        # Find nearest unmatched observed peak
        diffs = [
            (abs(pred_ppm - obs), i)
            for i, obs in enumerate(observed_shifts)
            if i not in matched_obs_indices
        ]
        if not diffs:
            per_atom.append({
                "atom_id": atom_id,
                "predicted_ppm": pred_ppm,
                "matched_ppm": None,
                "deviation_ppm": None,
                "matched": False,
            })
            continue

        diffs.sort()
        best_diff, best_idx = diffs[0]
        matched_obs_indices.add(best_idx)
        deviations.append(best_diff)
        per_atom.append({
            "atom_id": atom_id,
            "predicted_ppm": pred_ppm,
            "matched_ppm": observed_shifts[best_idx],
            "deviation_ppm": round(best_diff, 3),
            "matched": True,
        })

    unmatched_obs = len(observed_shifts) - len(matched_obs_indices)
    mad = round(sum(deviations) / len(deviations), 3) if deviations else None

    return {
        "mean_deviation_ppm": mad,
        "matched": len([a for a in per_atom if a["matched"]]),
        "unmatched_predicted": len([a for a in per_atom if not a["matched"]]),
        "unmatched_observed": unmatched_obs,
        "per_atom_table": per_atom,
        "method": "CASCADE 2.0 (PAiNN, MMFF geometry)",
    }


# ── Step 3: IntegralPurityEstimator ──────────────────────────────────────────

def _run_integral_purity(
    product_smiles: str,
    h1_peaks: list[dict],
    solvent: str,
) -> dict:
    """
    Estimate purity from ¹H integral ratios vs expected proton count from SMILES.
    Cross-reference residual integrals with contaminants.json.
    Filter solvent residual peaks before analysis.

    Returns: {purity_percent, expected_protons, observed_integral_total,
              assigned_integral, contaminants_detected, note}
    """
    if not h1_peaks:
        return {
            "purity_percent": None,
            "note": "No ¹H peaks provided — purity estimate unavailable",
        }

    if not _rdkit_available():
        return {
            "purity_percent": None,
            "note": "RDKit not available — purity estimate unavailable",
        }

    from rdkit import Chem
    from rdkit.Chem import Descriptors

    mol = Chem.MolFromSmiles(product_smiles)
    if mol is None:
        return {
            "purity_percent": None,
            "note": f"Invalid SMILES: {product_smiles}",
        }

    # Count expected H (including implicit)
    mol_with_h = Chem.AddHs(mol)
    expected_h = sum(1 for a in mol_with_h.GetAtoms() if a.GetAtomicNum() == 1)

    # Filter solvent residual peaks
    solvent_data = _load_solvent_peaks()
    solvent_shifts = {
        entry["shift"]
        for entry in solvent_data.get(solvent, [])
        if entry.get("nucleus") == "1H"
    }
    # Also add water peaks for this solvent
    contaminant_data = _load_contaminants()
    water_entry = next((c for c in contaminant_data if c["name"] == "water"), None)
    if water_entry and "shifts_by_solvent" in water_entry:
        water_shift = water_entry["shifts_by_solvent"].get(solvent)
        if water_shift:
            solvent_shifts.add(water_shift)

    def _is_solvent_peak(peak: dict) -> bool:
        shift = float(peak.get("shift", peak.get("ppm", -99)))
        return any(abs(shift - s) < 0.1 for s in solvent_shifts)

    filtered_peaks = [p for p in h1_peaks if not _is_solvent_peak(p)]

    if not filtered_peaks:
        return {
            "purity_percent": None,
            "note": "All ¹H peaks match solvent residuals — cannot estimate purity",
        }

    # Build integral totals
    total_integral = sum(float(p.get("integral", 1.0)) for p in filtered_peaks)

    # Check for known contaminant peaks
    detected_contaminants = []
    contaminant_integral = 0.0

    for contaminant in contaminant_data:
        if contaminant["name"] == "water":
            continue
        cont_shifts = contaminant.get("shifts", [])
        nucleus = contaminant.get("nucleus", "1H")
        if nucleus != "1H":
            continue
        matched_peaks = []
        for cs in cont_shifts:
            for peak in filtered_peaks:
                shift = float(peak.get("shift", peak.get("ppm", -99)))
                if abs(shift - cs) < 0.15:
                    matched_peaks.append(peak)
        if matched_peaks:
            cont_integral = sum(float(p.get("integral", 1.0)) for p in matched_peaks)
            detected_contaminants.append({
                "name": contaminant["name"],
                "integral": round(cont_integral, 3),
                "note": contaminant.get("note", ""),
            })
            contaminant_integral += cont_integral

    assigned_integral = total_integral - contaminant_integral

    # Estimate purity: assigned / total * 100
    # Normalise so expected_h maps to "pure" integral
    if total_integral > 0:
        purity_pct = round((assigned_integral / total_integral) * 100, 1)
    else:
        purity_pct = None

    return {
        "purity_percent": purity_pct,
        "expected_protons": expected_h,
        "total_integral": round(total_integral, 3),
        "assigned_integral": round(assigned_integral, 3),
        "contaminant_integral": round(contaminant_integral, 3),
        "contaminants_detected": detected_contaminants,
        "solvent_peaks_filtered": list(solvent_shifts),
        "note": "Integral-based purity estimate; assumes flat baseline and no overlap.",
    }


# ── Step 4: VerificationScorer ────────────────────────────────────────────────

def _compute_verification_confidence(
    marker_result: dict,
    peak_match_result: dict,
    purity_result: dict,
) -> dict:
    """
    Compute weighted confidence score:
      50% diagnostic markers
      30% peak match (MAD)
      20% purity

    Returns: {confidence_percent, verdict, breakdown}
    Verdict: CONFIRMED ≥85, LIKELY ≥65, UNCERTAIN ≥40, UNLIKELY <40
    """
    scores: dict[str, Optional[float]] = {}
    breakdown: dict[str, dict] = {}

    # --- Marker score (50%) ---
    marker_error = marker_result.get("_error")
    if marker_error:
        scores["markers"] = None
        breakdown["markers"] = {"score": None, "reason": marker_error, "weight": 0.50}
    else:
        total_markers = 0
        passed_markers = 0
        for name, info in marker_result.items():
            if name.startswith("_"):
                continue
            total_markers += 1
            if info.get("expected") == "lost":
                if info.get("found_absent"):
                    passed_markers += 1
            else:
                if info.get("found"):
                    passed_markers += 1
        if total_markers > 0:
            marker_score = passed_markers / total_markers
        else:
            marker_score = 0.5  # no markers = neutral
        scores["markers"] = marker_score
        breakdown["markers"] = {
            "score": round(marker_score * 100, 1),
            "passed": passed_markers,
            "total": total_markers,
            "weight": 0.50,
        }

    # --- Peak match score (30%) ---
    mad = peak_match_result.get("mean_deviation_ppm")
    if mad is None:
        scores["peak_match"] = None
        breakdown["peak_match"] = {
            "score": None,
            "reason": peak_match_result.get("cascade_error", "Not computed"),
            "weight": 0.30,
        }
    else:
        # MAD → score: 0 ppm = 100%, 5 ppm = 0% (linear)
        peak_score = max(0.0, 1.0 - mad / 5.0)
        scores["peak_match"] = peak_score
        breakdown["peak_match"] = {
            "score": round(peak_score * 100, 1),
            "mean_deviation_ppm": mad,
            "matched": peak_match_result.get("matched"),
            "weight": 0.30,
        }

    # --- Purity score (20%) ---
    purity_pct = purity_result.get("purity_percent")
    if purity_pct is None:
        scores["purity"] = None
        breakdown["purity"] = {
            "score": None,
            "reason": purity_result.get("note", "Not computed"),
            "weight": 0.20,
        }
    else:
        purity_score = min(1.0, purity_pct / 100.0)
        scores["purity"] = purity_score
        breakdown["purity"] = {
            "score": round(purity_score * 100, 1),
            "purity_percent": purity_pct,
            "weight": 0.20,
        }

    # --- Weighted average (skip None components, redistribute weight) ---
    weight_map = {"markers": 0.50, "peak_match": 0.30, "purity": 0.20}
    total_weight = sum(weight_map[k] for k, v in scores.items() if v is not None)
    if total_weight == 0:
        confidence = 50.0  # complete uncertainty
    else:
        weighted_sum = sum(
            scores[k] * weight_map[k]
            for k in scores
            if scores[k] is not None
        )
        confidence = round((weighted_sum / total_weight) * 100, 1)

    if confidence >= 85:
        verdict = "CONFIRMED"
    elif confidence >= 65:
        verdict = "LIKELY"
    elif confidence >= 40:
        verdict = "UNCERTAIN"
    else:
        verdict = "UNLIKELY"

    return {
        "confidence_percent": confidence,
        "verdict": verdict,
        "breakdown": breakdown,
    }


# ── pipeline.verify_product ───────────────────────────────────────────────────

@registry.register(
    name="pipeline.verify_product",
    description=(
        "Full VERIFY pipeline: given an expected product SMILES and observed spectral peaks, "
        "runs 4 sequential sub-steps — (1) diagnostic marker detection via SMARTS, "
        "(2) ¹³C peak matching via CASCADE 2.0, (3) ¹H integral purity estimation, "
        "(4) weighted confidence scoring. Returns a structured verdict JSON. "
        "Claude should report this output verbatim without adding interpretation."
    ),
    category="pipeline",
    parameters={
        "smiles": {
            "type": "string",
            "description": "SMILES of the expected product.",
        },
        "observed_peaks": {
            "type": "array",
            "description": (
                "List of observed peaks. Each peak is a dict with 'shift' (ppm), "
                "'nucleus' ('1H' or '13C'), and optional 'integral', 'multiplicity'."
            ),
        },
        "solvent": {
            "type": "string",
            "description": "NMR solvent (e.g. 'CDCl3', 'DMSO-d6'). Used for purity estimation.",
        },
        "sm_smiles": {
            "type": "string",
            "description": (
                "SMILES of the starting material (optional). "
                "When provided, enables differential marker analysis."
            ),
        },
        "tolerance_ppm": {
            "type": "number",
            "description": "Tolerance in ppm for peak-to-range matching (default: 0.3).",
        },
    },
)
def verify_product(
    smiles: str,
    observed_peaks: list[dict],
    solvent: str = "CDCl3",
    sm_smiles: Optional[str] = None,
    tolerance_ppm: float = 0.3,
) -> dict:
    """
    Run the full VERIFY pipeline and return a structured verdict.

    Steps:
      1. DiagnosticMarkerGenerator — SMARTS-based functional group detection
      2. CascadePeakMatcher       — ¹³C shift prediction + MAD
      3. IntegralPurityEstimator  — ¹H purity from integrals
      4. VerificationScorer       — weighted confidence + verdict
    """
    if not smiles:
        return {"error": "No product SMILES provided."}
    if not observed_peaks:
        return {"error": "No observed peaks provided."}

    # Separate peaks by nucleus
    h1_peaks = [
        p for p in observed_peaks
        if str(p.get("nucleus", "")).strip() in ("1H", "H", "H1", "")
        and p.get("nucleus") is not None
    ]
    c13_peaks = [
        p for p in observed_peaks
        if str(p.get("nucleus", "")).strip() in ("13C", "C", "C13")
    ]

    # If no nucleus tags, try to guess from shift range
    if not h1_peaks and not c13_peaks:
        for p in observed_peaks:
            shift = float(p.get("shift", p.get("ppm", 0)))
            if shift <= 15:
                h1_peaks.append({**p, "nucleus": "1H"})
            else:
                c13_peaks.append({**p, "nucleus": "13C"})

    logger.info(
        "verify_product: smiles=%s, h1_peaks=%d, c13_peaks=%d, solvent=%s",
        smiles, len(h1_peaks), len(c13_peaks), solvent,
    )

    # Step 1: Diagnostic markers
    try:
        marker_result = _run_diagnostic_markers(smiles, sm_smiles, observed_peaks, tolerance_ppm)
    except Exception as e:
        logger.exception("DiagnosticMarkerGenerator failed")
        marker_result = {"_error": f"DiagnosticMarkerGenerator error: {e}"}

    # Step 2: CASCADE peak matching (¹³C only)
    try:
        peak_match_result = _run_cascade_peak_matcher(smiles, c13_peaks)
    except Exception as e:
        logger.exception("CascadePeakMatcher failed")
        peak_match_result = {
            "mean_deviation_ppm": None,
            "cascade_error": f"CascadePeakMatcher error: {e}",
        }

    # Step 3: Purity estimate (¹H only)
    try:
        purity_result = _run_integral_purity(smiles, h1_peaks, solvent)
    except Exception as e:
        logger.exception("IntegralPurityEstimator failed")
        purity_result = {
            "purity_percent": None,
            "note": f"IntegralPurityEstimator error: {e}",
        }

    # Step 4: Weighted confidence score
    try:
        score_result = _compute_verification_confidence(
            marker_result, peak_match_result, purity_result
        )
    except Exception as e:
        logger.exception("VerificationScorer failed")
        score_result = {
            "confidence_percent": None,
            "verdict": "ERROR",
            "error": str(e),
        }

    return {
        "verdict": score_result["verdict"],
        "confidence_percent": score_result["confidence_percent"],
        "score_breakdown": score_result["breakdown"],
        "diagnostic_markers": marker_result,
        "peak_match": {
            "mean_deviation_ppm": peak_match_result.get("mean_deviation_ppm"),
            "matched": peak_match_result.get("matched"),
            "unmatched_observed": peak_match_result.get("unmatched_observed"),
            "method": peak_match_result.get("method"),
            "per_atom_table": peak_match_result.get("per_atom_table", []),
        },
        "purity_percent": purity_result.get("purity_percent"),
        "contaminants": purity_result.get("contaminants_detected", []),
        "citations": [
            {
                "title": "CASCADE 2.0: Real-time Prediction of 13C NMR Shifts",
                "authors": "Paton Lab, Colorado State University",
                "url": "https://nova.chem.colostate.edu/v2/cascade/",
            },
            {
                "title": "functional_group_shifts.json",
                "note": "SMARTS-based functional group detection, speqtro built-in data",
            },
            {
                "title": "contaminants.json",
                "note": "Common NMR contaminant reference, speqtro built-in data",
            },
        ],
    }


# ── pipeline.elucidation ──────────────────────────────────────────────────────

@registry.register(
    name="pipeline.elucidation",
    description=(
        "EXPLORE pipeline: given ¹H and/or ¹³C peak lists, converts peaks to spectrum "
        "vectors and calls ChefNMR to return top-k SMILES candidates. "
        "Use when no product SMILES is available (unknown structure). "
        "Claude should report the candidate list verbatim without picking a winner."
    ),
    category="pipeline",
    parameters={
        "h1_peaks": {
            "type": "array",
            "description": (
                "¹H NMR peaks as list of dicts with 'shift' (ppm) and optional "
                "'integral' and 'multiplicity'."
            ),
        },
        "c13_peaks": {
            "type": "array",
            "description": "¹³C NMR peaks as list of dicts with 'shift' (ppm).",
        },
        "solvent": {
            "type": "string",
            "description": "NMR solvent (used for peak filtering).",
        },
        "n_candidates": {
            "type": "integer",
            "description": "Number of candidate structures to request (default: 10).",
        },
    },
)
def elucidation(
    h1_peaks: list[dict] | None = None,
    c13_peaks: list[dict] | None = None,
    solvent: str = "CDCl3",
    n_candidates: int = 10,
) -> dict:
    """
    EXPLORE pipeline: route to ChefNMR for unknown structure elucidation.
    Returns top-k SMILES candidates. Claude reports the list — does not pick.
    """
    if not h1_peaks and not c13_peaks:
        return {"error": "Provide at least h1_peaks or c13_peaks."}

    try:
        from speqtro.tools.chefnmr import elucidation_chefnmr
        result = elucidation_chefnmr(
            h1_peaks=h1_peaks,
            c13_peaks=c13_peaks,
            n_candidates=n_candidates,
        )
        return result
    except Exception as e:
        logger.exception("pipeline.elucidation failed")
        return {
            "error": f"Elucidation pipeline error: {e}",
            "smiles_candidates": [],
        }

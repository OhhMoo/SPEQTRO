# -*- coding: utf-8 -*-
"""
Vendored DP4 scoring engine.

Adapted from DP4.py by Alexander Howarth (ke291), Goodman group, Cambridge.
Original: https://github.com/KristapsE/DP4-AI  (also DP5 repo)

Stripped to inference-only library code:
  - Removed CLI/GUI, file I/O for Gaussian/NWChem, MakeOutput, PrintNMR
  - Removed Settings/Isomers coupling — functions accept plain numpy arrays
  - Kept: internal scaling, single/multi Gaussian probability, DP4 calculation

Reference:
  Smith SG, Goodman JM. "Assigning Stereochemistry to Single Diastereoisomers
  by GIAO NMR Calculation: The DP4 Probability" JACS 2010, 132, 12946-12959.

License: MIT (follows original DP4-AI license)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Default DP4 error distribution parameters (from original DP4.py)
# These are for B3LYP/6-31G(d,p) level DFT-computed shifts.
# ---------------------------------------------------------------------------
DEFAULT_MEAN_C: float = 0.0
DEFAULT_STDEV_C: float = 2.269372270818724
DEFAULT_MEAN_H: float = 0.0
DEFAULT_STDEV_H: float = 0.18731058105269952


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------
@dataclass
class DP4Result:
    """Results of a DP4 calculation over N candidate isomers."""

    n_isomers: int = 0

    # Per-isomer lists (length = n_isomers)
    c_scaled: list[list[float]] = field(default_factory=list)
    h_scaled: list[list[float]] = field(default_factory=list)
    c_errors: list[list[float]] = field(default_factory=list)
    h_errors: list[list[float]] = field(default_factory=list)
    c_probs: list[list[float]] = field(default_factory=list)
    h_probs: list[list[float]] = field(default_factory=list)

    # Final DP4 probabilities (length = n_isomers, sum to 1.0)
    carbon_dp4: list[float] = field(default_factory=list)
    proton_dp4: list[float] = field(default_factory=list)
    combined_dp4: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal scaling (linear regression)
# ---------------------------------------------------------------------------
def scale_nmr(calc_shifts: Sequence[float], exp_shifts: Sequence[float]) -> list[float]:
    """
    Internally scale calculated shifts against experimental using linear regression.

    Fits:  calc = slope * exp + intercept
    Returns:  scaled_i = (calc_i - intercept) / slope

    This removes systematic DFT method bias so that only random errors remain.
    """
    if len(calc_shifts) < 2 or len(exp_shifts) < 2:
        return list(calc_shifts)
    slope, intercept, _, _, _ = stats.linregress(exp_shifts, calc_shifts)
    if abs(slope) < 1e-12:
        return list(calc_shifts)
    return [(x - intercept) / slope for x in calc_shifts]


# ---------------------------------------------------------------------------
# Probability functions
# ---------------------------------------------------------------------------
def single_gauss_probability(error: float, mean: float, stdev: float) -> float:
    """
    Standard DP4 probability for a single error value.
    P = 2 * Phi(-|z|)  where z = (error - mean) / stdev
    This is the two-tailed p-value of the error under N(mean, stdev).
    """
    z = abs((error - mean) / stdev)
    return float(2.0 * stats.norm.cdf(-z))


def multi_gauss_probability(error: float, means: Sequence[float],
                            stdevs: Sequence[float]) -> float:
    """
    DP4 probability using a mixture-of-Gaussians error model.
    Returns the average pdf value across all Gaussian components.
    """
    if not means:
        raise ValueError("means must be non-empty")
    total = sum(stats.norm(m, s).pdf(error) for m, s in zip(means, stdevs))
    return float(total / len(means))


# ---------------------------------------------------------------------------
# Core DP4 calculation
# ---------------------------------------------------------------------------
def compute_dp4(
    calc_c_shifts: list[list[float]],
    exp_c_shifts: list[list[float]],
    calc_h_shifts: list[list[float]],
    exp_h_shifts: list[list[float]],
    *,
    mean_c: float = DEFAULT_MEAN_C,
    stdev_c: float = DEFAULT_STDEV_C,
    mean_h: float = DEFAULT_MEAN_H,
    stdev_h: float = DEFAULT_STDEV_H,
    multi_gauss_params: Optional[dict] = None,
) -> DP4Result:
    """
    Compute DP4 probabilities for a set of candidate isomers.

    Args:
        calc_c_shifts: Calculated 13C shifts per isomer.
            calc_c_shifts[i] = list of floats for isomer i.
        exp_c_shifts: Experimental 13C shifts per isomer (same assignment).
            exp_c_shifts[i] = list of floats for isomer i.
        calc_h_shifts: Calculated 1H shifts per isomer.
        exp_h_shifts: Experimental 1H shifts per isomer.
        mean_c, stdev_c: Single-Gaussian error parameters for 13C.
        mean_h, stdev_h: Single-Gaussian error parameters for 1H.
        multi_gauss_params: Optional dict with keys 'c_means', 'c_stdevs',
            'h_means', 'h_stdevs' for mixture-of-Gaussians model.
            If provided, overrides single-Gaussian parameters.

    Returns:
        DP4Result with per-isomer probabilities.

    Raises:
        ValueError: If shift lists have inconsistent lengths.
    """
    n = len(calc_c_shifts)
    if n == 0:
        n = len(calc_h_shifts)
    if n == 0:
        raise ValueError("Must provide at least carbon or proton shifts")

    result = DP4Result(n_isomers=n)

    # --- Internal scaling ---
    has_carbon = len(calc_c_shifts) > 0 and len(calc_c_shifts[0]) > 0
    has_proton = len(calc_h_shifts) > 0 and len(calc_h_shifts[0]) > 0

    if has_carbon:
        for calc, exp in zip(calc_c_shifts, exp_c_shifts):
            if len(calc) != len(exp):
                raise ValueError(
                    f"Carbon shift length mismatch: calc={len(calc)}, exp={len(exp)}"
                )
            scaled = scale_nmr(calc, exp) if len(exp) > 1 else list(calc)
            result.c_scaled.append(scaled)
            errors = [s - e for s, e in zip(scaled, exp)]
            result.c_errors.append(errors)

    if has_proton:
        for calc, exp in zip(calc_h_shifts, exp_h_shifts):
            if len(calc) != len(exp):
                raise ValueError(
                    f"Proton shift length mismatch: calc={len(calc)}, exp={len(exp)}"
                )
            scaled = scale_nmr(calc, exp) if len(exp) > 1 else list(calc)
            result.h_scaled.append(scaled)
            errors = [s - e for s, e in zip(scaled, exp)]
            result.h_errors.append(errors)

    # --- Calculate per-atom probabilities ---
    use_multi = multi_gauss_params is not None

    if has_carbon:
        for errors in result.c_errors:
            if use_multi:
                probs = [
                    multi_gauss_probability(
                        e, multi_gauss_params["c_means"], multi_gauss_params["c_stdevs"]
                    )
                    for e in errors
                ]
            else:
                probs = [single_gauss_probability(e, mean_c, stdev_c) for e in errors]
            result.c_probs.append(probs)

    if has_proton:
        for errors in result.h_errors:
            if use_multi:
                probs = [
                    multi_gauss_probability(
                        e, multi_gauss_params["h_means"], multi_gauss_params["h_stdevs"]
                    )
                    for e in errors
                ]
            else:
                probs = [single_gauss_probability(e, mean_h, stdev_h) for e in errors]
            result.h_probs.append(probs)

    # --- Multiply per-atom probabilities → per-isomer likelihoods ---
    c_likelihoods = []
    if has_carbon:
        for probs in result.c_probs:
            likelihood = 1.0
            for p in probs:
                likelihood *= p
            c_likelihoods.append(likelihood)
    else:
        c_likelihoods = [1.0] * n

    h_likelihoods = []
    if has_proton:
        for probs in result.h_probs:
            likelihood = 1.0
            for p in probs:
                likelihood *= p
            h_likelihoods.append(likelihood)
    else:
        h_likelihoods = [1.0] * n

    combined_likelihoods = [h * c for h, c in zip(h_likelihoods, c_likelihoods)]

    # --- Normalize to probabilities ---
    c_sum = sum(c_likelihoods)
    h_sum = sum(h_likelihoods)
    t_sum = sum(combined_likelihoods)

    result.carbon_dp4 = [x / c_sum for x in c_likelihoods] if c_sum > 0 else [0.0] * n
    result.proton_dp4 = [x / h_sum for x in h_likelihoods] if h_sum > 0 else [0.0] * n
    result.combined_dp4 = [x / t_sum for x in combined_likelihoods] if t_sum > 0 else [0.0] * n

    return result


# ---------------------------------------------------------------------------
# Convenience: unassigned shift scoring (Hungarian matching + DP4)
# ---------------------------------------------------------------------------
def dp4_score_unassigned(
    calc_c_per_isomer: list[list[float]],
    exp_c: list[float],
    calc_h_per_isomer: list[list[float]],
    exp_h: list[float],
    **kwargs,
) -> DP4Result:
    """
    Score multiple isomers when experimental shifts are NOT pre-assigned to atoms.

    Uses sorted-order assignment (optimal for DP4 when assignment is unknown):
    both calculated and experimental shifts are sorted, then paired by index.
    This is the standard approach from the original DP4 paper.

    Args:
        calc_c_per_isomer: calc_c_per_isomer[i] = predicted 13C shifts for isomer i
        exp_c: experimental 13C shifts (one flat list, shared across isomers)
        calc_h_per_isomer: calc_h_per_isomer[i] = predicted 1H shifts for isomer i
        exp_h: experimental 1H shifts (one flat list, shared across isomers)
        **kwargs: passed through to compute_dp4()

    Returns:
        DP4Result
    """
    n = max(len(calc_c_per_isomer), len(calc_h_per_isomer))

    sorted_exp_c = sorted(exp_c)
    sorted_exp_h = sorted(exp_h)

    # Build per-isomer assigned pairs via sorted matching
    assigned_calc_c = []
    assigned_exp_c = []
    for i in range(len(calc_c_per_isomer)):
        sc = sorted(calc_c_per_isomer[i])
        # Truncate to min length (handle mismatch in peak count)
        min_len = min(len(sc), len(sorted_exp_c))
        assigned_calc_c.append(sc[:min_len])
        assigned_exp_c.append(sorted_exp_c[:min_len])

    assigned_calc_h = []
    assigned_exp_h = []
    for i in range(len(calc_h_per_isomer)):
        sh = sorted(calc_h_per_isomer[i])
        min_len = min(len(sh), len(sorted_exp_h))
        assigned_calc_h.append(sh[:min_len])
        assigned_exp_h.append(sorted_exp_h[:min_len])

    return compute_dp4(
        assigned_calc_c, assigned_exp_c,
        assigned_calc_h, assigned_exp_h,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Read multi-Gaussian parameter files (original DP4 format)
# ---------------------------------------------------------------------------
def read_param_file(filepath: str) -> dict:
    """
    Read a DP4 multi-Gaussian stats parameter file.

    File format (from original DP4):
      Line 0: type marker (contains 'm' for multi-Gaussian)
      Line 1: C means (comma-separated)
      Line 2: C stdevs (comma-separated)
      Line 3: H means (comma-separated)
      Line 4: H stdevs (comma-separated)

    Returns:
        dict with keys: c_means, c_stdevs, h_means, h_stdevs
    """
    with open(filepath, "r") as f:
        lines = f.readlines()

    return {
        "c_means": [float(x) for x in lines[1].split(",")],
        "c_stdevs": [float(x) for x in lines[2].split(",")],
        "h_means": [float(x) for x in lines[3].split(",")],
        "h_stdevs": [float(x) for x in lines[4].split(",")],
    }

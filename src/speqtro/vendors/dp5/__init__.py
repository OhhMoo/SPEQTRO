"""
Vendored DP4/DP5 scoring engine.

Adapted from the Goodman group DP4-AI / DP5 codebase (Cambridge).
Contains only the inference/scoring code — no DFT job submission,
no GUI, no file I/O for Gaussian/NWChem.

Modules:
    dp4  — DP4 statistical scoring (scipy/numpy only, no heavy deps)

Usage:
    from speqtro.vendors.dp5 import compute_dp4, dp4_score_unassigned, DP4Result

References:
    - Smith SG, Goodman JM. JACS 2010, 132, 12946 (DP4)
    - Ermanis K, Parkes KEB, Ageback T, Goodman JM. Org. Biomol. Chem. 2017 (DP5)
"""
from speqtro.vendors.dp5.dp4 import (
    DP4Result,
    compute_dp4,
    dp4_score_unassigned,
    scale_nmr,
    single_gauss_probability,
    multi_gauss_probability,
    read_param_file,
    DEFAULT_MEAN_C,
    DEFAULT_STDEV_C,
    DEFAULT_MEAN_H,
    DEFAULT_STDEV_H,
)

__all__ = [
    "DP4Result",
    "compute_dp4",
    "dp4_score_unassigned",
    "scale_nmr",
    "single_gauss_probability",
    "multi_gauss_probability",
    "read_param_file",
    "DEFAULT_MEAN_C",
    "DEFAULT_STDEV_C",
    "DEFAULT_MEAN_H",
    "DEFAULT_STDEV_H",
]

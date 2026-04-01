"""
Vendored SSIN (Substructure-directed Spectrum Interpreter Network).

Inference-only subset of the SSIN codebase, adapted for use as a library module
within speqtro. Original source: https://github.com/ngs00/SSIN

Only the components needed for inference are included:
  - model.py: SSIN model class + predict_from_jdx + identify_impt_peaks
  - data.py: IRSpectrum, IRDataset, read_jdx_file, interpol_absorbance, collate
  - chem.py: topk_points, func_groups, atom_nums, get_state_label (no graph code)

All imports are lazy — heavy dependencies (torch, jcamp, scipy) are only
loaded when the submodule is actually accessed.
"""


def __getattr__(name):
    if name == "SSIN":
        from speqtro.vendors.ssin.model import SSIN
        return SSIN
    if name == "predict_from_jdx":
        from speqtro.vendors.ssin.model import predict_from_jdx
        return predict_from_jdx
    if name == "identify_impt_peaks":
        from speqtro.vendors.ssin.model import identify_impt_peaks
        return identify_impt_peaks
    if name in ("IRSpectrum", "IRDataset", "read_jdx_file", "interpol_absorbance", "collate"):
        from speqtro.vendors.ssin import data
        return getattr(data, name)
    if name in ("topk_points", "func_groups", "atom_nums", "get_state_label"):
        from speqtro.vendors.ssin import chem
        return getattr(chem, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

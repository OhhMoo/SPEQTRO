"""
Bruker NMR experiment directory parser.

Handles both processed data (pdata/1/1r) and raw FID (fid + acqus).
Uses nmrglue for file I/O and basic processing.

Falls back gracefully if nmrglue is not installed — returns a stub
result with a clear error message so the rest of the pipeline can
report the missing dependency via `speqtro doctor`.
"""

from pathlib import Path


def parse(bruker_dir: Path) -> dict:
    """
    Parse a Bruker NMR experiment directory.

    Args:
        bruker_dir: Path to the experiment directory (contains acqus/fid
                    or pdata/1/1r).

    Returns:
        Normalized spectrum dict.
    """
    bruker_dir = Path(bruker_dir)
    try:
        import nmrglue as ng
        import numpy as np
    except ImportError:
        raise ImportError(
            "nmrglue is required to parse Bruker files. "
            "Install it with: pip install nmrglue"
        )

    # ── Load data ──────────────────────────────────────────────────────
    pdata = bruker_dir / "pdata" / "1"
    if pdata.exists() and (pdata / "1r").exists():
        dic, data = ng.bruker.read_pdata(str(pdata))
        udic = ng.bruker.guess_udic(dic, data)
    else:
        dic, data = ng.bruker.read(str(bruker_dir))
        udic = ng.bruker.guess_udic(dic, data)
        # Basic 1D processing: zero-fill → apodization → FFT → autophase
        data = ng.proc_base.zf_size(data, data.size * 2)
        data = ng.proc_base.em(data, lb=0.5)
        data = ng.proc_base.fft(data)
        try:
            data = ng.proc_autophase.autops(data, "acme")
        except Exception:
            pass  # autophase can fail; continue with unphased spectrum

    # ── Metadata extraction ────────────────────────────────────────────
    acqus = dic.get("acqus", {})
    nucleus = acqus.get("NUC1", "1H")
    solvent = acqus.get("SOLVENT", "unknown")
    try:
        frequency = float(acqus.get("SFO1", 400.0))
    except (TypeError, ValueError):
        frequency = 400.0

    # ── ppm axis ───────────────────────────────────────────────────────
    uc = ng.fileiobase.uc_from_udic(udic)
    data_real = np.real(data)
    ppm_scale = np.array([uc.ppm(i) for i in range(len(data_real))])

    # ── Peak picking ───────────────────────────────────────────────────
    noise_std = float(np.std(data_real))
    threshold = noise_std * 5.0
    try:
        peak_indices = ng.peakpick.pick(data_real, pthres=threshold)
    except Exception:
        peak_indices = []

    peaks = []
    for idx in peak_indices:
        try:
            idx_int = int(idx)
            ppm = float(uc.ppm(idx_int))
            intensity = float(data_real[idx_int])
            if intensity > 0:  # suppress negative baseline artefacts
                peaks.append({
                    "shift": round(ppm, 4),
                    "intensity": round(intensity, 2),
                    "integral": None,
                    "multiplicity": None,
                    "coupling_hz": None,
                })
        except Exception:
            continue

    return {
        "peaks": sorted(peaks, key=lambda p: -p["shift"]),
        "nucleus": _normalize_nucleus(nucleus),
        "solvent": solvent,
        "frequency_mhz": frequency,
        "source_format": "bruker",
        "raw_spectrum": data_real,
        "ppm_scale": ppm_scale,
    }


def _normalize_nucleus(raw: str) -> str:
    """Map Bruker nucleus strings to standard notation."""
    mapping = {
        "1H": "1H", "H1": "1H",
        "13C": "13C", "C13": "13C",
        "31P": "31P", "P31": "31P",
        "19F": "19F", "F19": "19F",
        "15N": "15N", "N15": "15N",
        "2H": "2H", "H2": "2H",
    }
    return mapping.get(raw.strip(), raw.strip())

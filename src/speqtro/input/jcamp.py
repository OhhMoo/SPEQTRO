"""
JCAMP-DX parser for NMR spectra.

Supports both peak table format (##PEAK TABLE=) and
continuous xy data (##XYDATA= or ##XYPOINTS=).

Does not require nmrglue — pure Python with numpy.
"""

from pathlib import Path
import re


def parse(jcamp_path: Path) -> dict:
    """
    Parse a JCAMP-DX file (.jdx / .dx / .jcamp).

    Returns:
        Normalized spectrum dict.
    """
    try:
        import numpy as np
    except ImportError:
        np = None

    path = Path(jcamp_path)
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = _split_blocks(text)
    # Use first block if multiple (compound JCAMP)
    block = blocks[0] if blocks else text
    return _parse_block(block, np)


def _split_blocks(text: str) -> list[str]:
    """Split compound JCAMP files at ##TITLE= markers."""
    parts = re.split(r"(?=##TITLE=)", text, flags=re.IGNORECASE)
    return [p.strip() for p in parts if p.strip()]


def _parse_block(text: str, np) -> dict:
    meta = _extract_meta(text)
    nucleus = _detect_nucleus(meta)
    solvent = meta.get(".solvent name", meta.get("solvent", "unknown"))
    try:
        frequency = float(meta.get(".observe frequency", meta.get("freq", 0)) or 0)
    except (TypeError, ValueError):
        frequency = None

    peaks = []

    # ── Peak table (preferred) ─────────────────────────────────────────
    pt_match = re.search(
        r"##PEAK TABLE=(.*?)(?=##[A-Z]|\Z)", text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if pt_match:
        peaks = _parse_peak_table(pt_match.group(1))

    # ── XY data ───────────────────────────────────────────────────────
    elif np is not None:
        xy_match = re.search(
            r"##(?:XYDATA|XYPOINTS)=(.*?)(?=##[A-Z]|\Z)", text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if xy_match:
            peaks = _pick_peaks_from_xy(xy_match.group(1), meta, np)

    return {
        "peaks": sorted(peaks, key=lambda p: -p["shift"]),
        "nucleus": nucleus,
        "solvent": solvent,
        "frequency_mhz": frequency if frequency else None,
        "source_format": "jcamp",
        "raw_spectrum": None,
        "ppm_scale": None,
    }


def _extract_meta(text: str) -> dict:
    """Extract ##KEY= VALUE pairs from JCAMP header."""
    meta = {}
    for m in re.finditer(r"##(.+?)=\s*(.+?)(?=\n##|\Z)", text, re.DOTALL):
        key = m.group(1).strip().lower()
        val = m.group(2).strip()
        meta[key] = val
    return meta


def _detect_nucleus(meta: dict) -> str:
    """Guess nucleus from metadata keys."""
    raw = (
        meta.get(".observe nucleus", "") or
        meta.get("nuc1", "") or
        meta.get(".nucleus", "") or
        ""
    ).upper()
    if "13C" in raw or "C13" in raw:
        return "13C"
    if "31P" in raw or "P31" in raw:
        return "31P"
    if "19F" in raw or "F19" in raw:
        return "19F"
    if "15N" in raw or "N15" in raw:
        return "15N"
    return "1H"  # default


def _parse_peak_table(table_text: str) -> list[dict]:
    """Parse ##PEAK TABLE= section: lines of 'ppm, intensity' pairs."""
    peaks = []
    for line in table_text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.lower().startswith("("):
            continue
        # Handle comma or space separation; optional trailing data
        parts = re.split(r"[,\s]+", line)
        if len(parts) >= 1:
            try:
                shift = float(parts[0])
                intensity = float(parts[1]) if len(parts) >= 2 else None
                peaks.append({
                    "shift": round(shift, 4),
                    "intensity": intensity,
                    "integral": None,
                    "multiplicity": None,
                    "coupling_hz": None,
                })
            except ValueError:
                continue
    return peaks


def _pick_peaks_from_xy(xy_text: str, meta: dict, np) -> list[dict]:
    """
    Simple peak-picking from XYDATA (X++(Y..Y) format or X,Y pairs).
    Returns only peaks above 5× noise threshold.
    """
    # Parse X,Y pairs — handle both X,Y and X++(Y..Y) compressed formats
    x_vals, y_vals = [], []
    first_x = float(meta.get("firstx", 0) or 0)
    last_x = float(meta.get("lastx", 1) or 1)
    n_points = int(meta.get("npoints", 0) or 0)

    # Try simple X,Y line format first
    for line in xy_text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"[,\s]+", line)
        if len(parts) >= 2:
            try:
                x_vals.append(float(parts[0]))
                y_vals.append(float(parts[1]))
            except ValueError:
                continue

    if not y_vals and n_points > 0:
        # JCAMP compressed format — generate uniform x axis
        x_vals = list(np.linspace(first_x, last_x, n_points))
        # Extract all numeric tokens from the xy block
        tokens = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", xy_text)
        y_vals = [float(t) for t in tokens[:n_points]]

    if not y_vals:
        return []

    y = np.array(y_vals, dtype=float)
    x = np.array(x_vals[:len(y)], dtype=float)

    threshold = np.std(y) * 5
    peaks = []
    for i in range(1, len(y) - 1):
        if y[i] > threshold and y[i] >= y[i - 1] and y[i] >= y[i + 1]:
            peaks.append({
                "shift": round(float(x[i]), 4),
                "intensity": round(float(y[i]), 2),
                "integral": None,
                "multiplicity": None,
                "coupling_hz": None,
            })

    return peaks

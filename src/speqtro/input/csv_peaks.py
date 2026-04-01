"""
Generic CSV / TSV peak list parser.

Expects at minimum a column containing chemical shift values (ppm).
Optionally: intensity, integral, multiplicity, coupling_hz columns.

Column detection is header-name based (case-insensitive). Falls back
to positional detection (first numeric column = shift).
"""

from pathlib import Path
import csv
import re


def parse(csv_path: Path) -> dict:
    """
    Parse a CSV/TSV file of NMR peaks.

    Returns:
        Normalized spectrum dict.
    """
    path = Path(csv_path)
    text = path.read_text(encoding="utf-8", errors="replace")
    dialect = "excel-tab" if path.suffix.lower() == ".tsv" else _sniff_dialect(text)

    reader = csv.reader(text.splitlines(), dialect=dialect)
    rows = [row for row in reader if any(cell.strip() for cell in row)]

    if not rows:
        return _empty_result()

    # Detect header row
    header_idx = 0
    col_map = {}
    if _looks_like_header(rows[0]):
        col_map = _map_columns(rows[0])
        header_idx = 1

    # If no header mapping, try positional: first numeric col = shift
    data_rows = rows[header_idx:]
    if not col_map:
        col_map = _positional_map(data_rows)

    peaks = []
    for row in data_rows:
        peak = _parse_row(row, col_map)
        if peak is not None:
            peaks.append(peak)

    # Try to extract nucleus/solvent from filename or header comments
    nucleus = _guess_nucleus(text)
    solvent = _guess_solvent(text)

    return {
        "peaks": sorted(peaks, key=lambda p: -p["shift"]),
        "nucleus": nucleus,
        "solvent": solvent,
        "frequency_mhz": None,
        "source_format": "csv",
        "raw_spectrum": None,
        "ppm_scale": None,
    }


def _empty_result() -> dict:
    return {
        "peaks": [],
        "nucleus": "1H",
        "solvent": "unknown",
        "frequency_mhz": None,
        "source_format": "csv",
        "raw_spectrum": None,
        "ppm_scale": None,
    }


def _sniff_dialect(text: str):
    try:
        sample = "\n".join(text.splitlines()[:5])
        return csv.Sniffer().sniff(sample)
    except csv.Error:
        return "excel"


def _looks_like_header(row: list[str]) -> bool:
    """True if the first row appears to be column names (not numbers)."""
    numeric = sum(1 for cell in row if _is_float(cell.strip()))
    return numeric < len(row) / 2


def _map_columns(header_row: list[str]) -> dict:
    """Map column names to indices."""
    col_map = {}
    aliases = {
        "shift": ["shift", "ppm", "chemical shift", "delta", "δ", "position"],
        "intensity": ["intensity", "height", "int", "signal"],
        "integral": ["integral", "integration", "area", "protons", "nH", "n"],
        "multiplicity": ["multiplicity", "mult", "splitting", "pattern", "m"],
        "coupling_hz": ["j", "j (hz)", "coupling", "coupling constant", "hz"],
    }
    for i, cell in enumerate(header_row):
        name = cell.strip().lower()
        for key, names in aliases.items():
            if any(name == a or name.startswith(a) for a in names):
                if key not in col_map:
                    col_map[key] = i
                break
    return col_map


def _positional_map(rows: list[list[str]]) -> dict:
    """Guess column positions from data if no header exists."""
    if not rows:
        return {}
    # Find first column that's consistently numeric (= shift)
    n_cols = max(len(r) for r in rows[:10])
    for i in range(n_cols):
        vals = [r[i] for r in rows[:10] if i < len(r)]
        if sum(_is_float(v.strip()) for v in vals) >= len(vals) * 0.8:
            return {"shift": i}
    return {}


def _parse_row(row: list[str], col_map: dict) -> dict | None:
    if "shift" not in col_map:
        return None
    shift_col = col_map["shift"]
    if shift_col >= len(row):
        return None
    try:
        shift = float(row[shift_col].strip())
    except ValueError:
        return None

    def _get(key):
        idx = col_map.get(key)
        if idx is None or idx >= len(row):
            return None
        val = row[idx].strip()
        return val if val else None

    intensity_raw = _get("intensity")
    integral_raw = _get("integral")
    coupling_raw = _get("coupling_hz")

    return {
        "shift": round(shift, 4),
        "intensity": float(intensity_raw) if intensity_raw and _is_float(intensity_raw) else None,
        "integral": float(integral_raw) if integral_raw and _is_float(integral_raw) else None,
        "multiplicity": _get("multiplicity"),
        "coupling_hz": _parse_coupling(coupling_raw) if coupling_raw else None,
    }


def _parse_coupling(raw: str) -> list[float] | None:
    """Parse coupling constants like '7.2, 3.4' or '7.2 Hz'."""
    nums = re.findall(r"\d+\.?\d*", raw)
    if not nums:
        return None
    return [float(n) for n in nums]


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _guess_nucleus(text: str) -> str:
    """Try to infer nucleus from file content."""
    upper = text.upper()
    if "13C" in upper or "C-13" in upper or "CARBON" in upper:
        return "13C"
    if "31P" in upper or "P-31" in upper:
        return "31P"
    if "19F" in upper or "F-19" in upper:
        return "19F"
    return "1H"


def _guess_solvent(text: str) -> str:
    """Try to extract solvent name from file content."""
    solvents = ["CDCl3", "DMSO-d6", "D2O", "CD3OD", "Acetone-d6", "C6D6", "CD2Cl2"]
    upper = text.upper()
    for s in solvents:
        if s.upper() in upper:
            return s
    return "unknown"

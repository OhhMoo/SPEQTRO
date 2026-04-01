"""
MestReNova multiplet report parser.

MestReNova can export peak/multiplet reports as .txt files with a
characteristic header format. This parser handles both the multiplet
report (with integration and multiplicity) and the peak list export.

Raises NotMestReNovaFormat if the file does not look like MestReNova output,
so the caller (autodetect.py) can fall back to generic csv_peaks.
"""

from pathlib import Path
import re


class NotMestReNovaFormat(Exception):
    """Raised when the file is not a MestReNova export."""


def parse(path: Path) -> dict:
    """
    Parse a MestReNova multiplet report or peak list.

    Raises:
        NotMestReNovaFormat: If the file doesn't match MestReNova patterns.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")

    # MestReNova files usually contain these markers
    if not _is_mestranova(text):
        raise NotMestReNovaFormat(f"File does not appear to be MestReNova output: {path}")

    nucleus = _extract_nucleus(text)
    solvent = _extract_solvent(text)
    frequency = _extract_frequency(text)

    # Try multiplet table first, then peak list
    peaks = _parse_multiplet_table(text)
    if not peaks:
        peaks = _parse_peak_list(text)

    return {
        "peaks": sorted(peaks, key=lambda p: -p["shift"]),
        "nucleus": nucleus,
        "solvent": solvent,
        "frequency_mhz": frequency,
        "source_format": "mestranova",
        "raw_spectrum": None,
        "ppm_scale": None,
    }


def _is_mestranova(text: str) -> bool:
    """Check for MestReNova fingerprints in the file."""
    markers = [
        "mestrelab", "mnova", "mestrenova",
        "multiplet list", "peak list",
        "δ (ppm)", "d (ppm)", "chemical shift",
    ]
    lower = text.lower()
    return any(m in lower for m in markers)


def _extract_nucleus(text: str) -> str:
    m = re.search(r"nucleus[:\s]+([0-9]+[A-Z]+)", text, re.IGNORECASE)
    if m:
        return m.group(1)
    if "13c" in text.lower() or "carbon" in text.lower():
        return "13C"
    return "1H"


def _extract_solvent(text: str) -> str:
    m = re.search(r"solvent[:\s]+([^\n\r,]+)", text, re.IGNORECASE)
    if m:
        raw = m.group(1).strip().rstrip(".")
        return raw
    solvents = ["CDCl3", "DMSO-d6", "D2O", "CD3OD", "Acetone-d6", "C6D6"]
    for s in solvents:
        if s.lower() in text.lower():
            return s
    return "unknown"


def _extract_frequency(text: str) -> float | None:
    m = re.search(r"(\d{2,3}(?:\.\d+)?)\s*MHz", text, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


def _parse_multiplet_table(text: str) -> list[dict]:
    """
    Parse MestReNova multiplet table format:
        δ (ppm)  Integral  Mult.  J (Hz)
        7.26     5.00      m
        3.71     2.00      s
    """
    peaks = []
    # Find the table section
    table_match = re.search(
        r"(?:multiplet|peak)\s+(?:list|table)[^\n]*\n((?:[^\n]+\n)*)",
        text, re.IGNORECASE
    )
    block = table_match.group(1) if table_match else text

    for line in block.splitlines():
        line = line.strip()
        if not line or line.lower().startswith(("δ", "d ", "shift", "#", "no")):
            continue

        # Match: shift  [integral]  [mult]  [J values]
        m = re.match(
            r"([\d.]+(?:[-–][\d.]+)?)"   # shift or range
            r"\s+([\d.]+)?"              # integral (optional)
            r"\s*([sdtqmb][a-z ,]*?)?"  # multiplicity (optional)
            r"\s*([\d., ]+)?"           # J Hz (optional)
            r"\s*$",
            line,
        )
        if m:
            shift_raw = m.group(1)
            # Use midpoint if range given (e.g., "7.20-7.30")
            if "-" in shift_raw or "–" in shift_raw:
                parts = re.split(r"[-–]", shift_raw)
                try:
                    shift = (float(parts[0]) + float(parts[1])) / 2
                except ValueError:
                    continue
            else:
                try:
                    shift = float(shift_raw)
                except ValueError:
                    continue

            integral_raw = m.group(2)
            mult_raw = m.group(3)
            j_raw = m.group(4)

            coupling = None
            if j_raw:
                nums = re.findall(r"\d+\.?\d*", j_raw)
                coupling = [float(n) for n in nums] if nums else None

            peaks.append({
                "shift": round(shift, 4),
                "intensity": None,
                "integral": float(integral_raw) if integral_raw else None,
                "multiplicity": mult_raw.strip() if mult_raw else None,
                "coupling_hz": coupling,
            })

    return peaks


def _parse_peak_list(text: str) -> list[dict]:
    """
    Parse simple MestReNova peak list:
        7.2600   1.00
        3.7100   0.45
    """
    peaks = []
    for line in text.splitlines():
        line = line.strip()
        parts = re.split(r"[\s,;]+", line)
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

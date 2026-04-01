"""
IR JCAMP-DX parser.

Handles ##DATA TYPE= INFRARED SPECTRUM files.  X-axis is wavenumber (cm⁻¹),
Y-axis is transmittance (%T) or absorbance.  Returns a dense (wavenumbers,
intensities) array suitable for SSIN as well as a simplified peak list.

Key differences from the NMR jcamp.py parser:
  - X-axis values are 400–4000 cm⁻¹ (not ppm)
  - Transmittance is converted to absorbance before peak-picking
  - Peaks are maxima in absorbance space (troughs in %T space)
  - Returns raw spectrum arrays, not just peak positions
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


def parse(jcamp_path: Path) -> dict:
    """
    Parse an IR JCAMP-DX file.

    Returns:
        {
            "wavenumbers":  list[float],   # cm⁻¹, ascending
            "intensities":  list[float],   # absorbance, normalised 0–1
            "peaks":        list[dict],    # [{"wavenumber": float, "intensity": float}, ...]
            "y_units":      str,           # "absorbance" | "transmittance" | "unknown"
            "data_type":    str,
            "source_format": "ir_jcamp",
        }
    """
    path = Path(jcamp_path)
    text = path.read_text(encoding="utf-8", errors="replace")
    return _parse_block(text)


def is_ir_jcamp(text_head: str) -> bool:
    """
    Return True if the JCAMP file is an IR spectrum (not NMR).
    Checks DATA TYPE field first, then falls back to x-axis range heuristic.
    """
    upper = text_head.upper()

    # Primary: DATA TYPE field
    dt_match = re.search(r"##\s*DATA\s*TYPE\s*=\s*([^\n\r]+)", text_head, re.IGNORECASE)
    if dt_match:
        val = dt_match.group(1).strip().upper()
        if "INFRARED" in val or "IR SPECTRUM" in val or "RAMAN" in val:
            return True
        if "NMR" in val or "NUCLEAR" in val:
            return False

    # Secondary: XUNITS
    xu_match = re.search(r"##\s*XUNITS\s*=\s*([^\n\r]+)", text_head, re.IGNORECASE)
    if xu_match:
        xu = xu_match.group(1).strip().upper()
        if "1/CM" in xu or "CM-1" in xu or "CM^-1" in xu:
            return True
        if "PPM" in xu or "HZ" in xu:
            return False

    # Tertiary: x-axis range  (IR is 400-4000, NMR is 0-250)
    for key in ("FIRSTX", "LASTX"):
        m = re.search(rf"##\s*{key}\s*=\s*([\d.]+)", text_head, re.IGNORECASE)
        if m:
            val = float(m.group(1))
            if val >= 300:
                return True
            if val <= 250:
                return False

    return False


# ── Private implementation ─────────────────────────────────────────────────

def _extract_meta(text: str) -> dict:
    meta: dict = {}
    for m in re.finditer(r"##(.+?)=\s*(.+?)(?=\n##|\Z)", text, re.DOTALL):
        key = m.group(1).strip().lower()
        val = m.group(2).strip().splitlines()[0].strip()  # first line only
        meta[key] = val
    return meta


def _parse_block(text: str) -> dict:
    meta = _extract_meta(text)

    data_type = (meta.get("data type", "") or meta.get("data type=", "INFRARED SPECTRUM")).strip()

    yunits_raw = (meta.get("yunits", "") or "").upper()
    if "TRANSMIT" in yunits_raw or "%T" in yunits_raw:
        y_units = "transmittance"
    elif "ABSORB" in yunits_raw:
        y_units = "absorbance"
    else:
        y_units = "unknown"

    # ── Try PEAK TABLE first (pre-processed) ───────────────────────────
    pt_match = re.search(
        r"##\s*PEAK\s*TABLE\s*=.*?\n(.*?)(?=##[A-Z]|\Z)",
        text, re.DOTALL | re.IGNORECASE,
    )
    raw_peaks: list[dict] = []
    wavenumbers: list[float] = []
    intensities: list[float] = []

    if pt_match:
        raw_peaks = _parse_peak_table(pt_match.group(1), y_units)
        wavenumbers = [p["wavenumber"] for p in raw_peaks]
        intensities = [p["intensity"] for p in raw_peaks]
    else:
        # ── Parse dense XY data ─────────────────────────────────────
        wavenumbers, raw_y = _parse_xydata(text, meta)
        if wavenumbers and raw_y:
            import numpy as np
            y_arr = np.array(raw_y, dtype=float)
            wn_arr = np.array(wavenumbers, dtype=float)

            # Convert transmittance → absorbance
            if y_units == "transmittance":
                y_arr = _transmittance_to_absorbance(y_arr)
                y_units = "absorbance"  # now in absorbance space
            elif y_units == "unknown":
                # Heuristic: if most values > 1 they're probably %T
                if float(np.median(y_arr)) > 1.0:
                    y_arr = _transmittance_to_absorbance(y_arr)
                    y_units = "absorbance"

            # Normalise to [0, 1]
            max_val = float(y_arr.max())
            if max_val > 0:
                y_norm = y_arr / max_val
            else:
                y_norm = y_arr

            intensities = y_norm.tolist()
            wavenumbers = wn_arr.tolist()

            # Pick peaks
            raw_peaks = _pick_peaks(wn_arr, y_norm)

    # Ensure ascending wavenumber order
    if len(wavenumbers) > 1 and wavenumbers[0] > wavenumbers[-1]:
        wavenumbers = list(reversed(wavenumbers))
        intensities = list(reversed(intensities))
        raw_peaks = sorted(raw_peaks, key=lambda p: p["wavenumber"])

    return {
        "wavenumbers": wavenumbers,
        "intensities": intensities,
        "peaks": raw_peaks,
        "y_units": y_units,
        "data_type": data_type,
        "source_format": "ir_jcamp",
    }


def _transmittance_to_absorbance(y: "np.ndarray") -> "np.ndarray":
    """
    Convert %T → absorbance: A = 2 - log10(T%).
    Clamp T to [0.01, 100] to avoid log(0).
    """
    import numpy as np
    t_pct = np.clip(y, 0.01, 100.0)
    # If values are already 0–1 fraction, scale up
    if float(t_pct.max()) <= 1.0:
        t_pct = t_pct * 100.0
    t_pct = np.clip(t_pct, 0.01, 100.0)
    return 2.0 - np.log10(t_pct)


def _parse_peak_table(table_text: str, y_units: str) -> list[dict]:
    """Parse ##PEAK TABLE= section. Lines may be 'wn, intensity' or 'wn'."""
    peaks = []
    for line in table_text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.upper().startswith("(X"):
            continue
        parts = re.split(r"[,\s;]+", line)
        if not parts:
            continue
        try:
            wn = float(parts[0])
            if not (200 <= wn <= 5000):
                continue
            intensity = float(parts[1]) if len(parts) >= 2 else 1.0
            peaks.append({"wavenumber": round(wn, 2), "intensity": round(intensity, 4)})
        except (ValueError, IndexError):
            continue

    # Normalise intensities
    if peaks:
        max_i = max(p["intensity"] for p in peaks)
        if max_i > 0:
            for p in peaks:
                p["intensity"] = round(p["intensity"] / max_i, 4)
    return peaks


def _parse_xydata(text: str, meta: dict) -> tuple[list[float], list[float]]:
    """
    Parse XYDATA or XYPOINTS section.
    Handles both X,Y line format and X++(Y..Y) compressed JCAMP encoding.
    """
    # Find data block
    block_match = re.search(
        r"##\s*(?:XYDATA|XYPOINTS)\s*=\s*.*?\n(.*?)(?=##[A-Z]|\Z)",
        text, re.DOTALL | re.IGNORECASE,
    )
    if not block_match:
        return [], []

    block = block_match.group(1)

    try:
        first_x = float(meta.get("firstx", 4000))
        last_x = float(meta.get("lastx", 400))
        n_pts = int(float(meta.get("npoints", 0) or 0))
        x_factor = float(meta.get("xfactor", 1.0) or 1.0)
        y_factor = float(meta.get("yfactor", 1.0) or 1.0)
    except (ValueError, TypeError):
        first_x, last_x, n_pts, x_factor, y_factor = 4000.0, 400.0, 0, 1.0, 1.0

    x_vals: list[float] = []
    y_vals: list[float] = []

    # Try simple X,Y per-line format first
    for line in block.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"[,\s]+", line)
        if len(parts) >= 2:
            try:
                x_vals.append(float(parts[0]) * x_factor)
                y_vals.append(float(parts[1]) * y_factor)
                continue
            except ValueError:
                pass

        # X++(Y..Y) compressed format: first token is X, rest are Y values
        parts = re.split(r"[,\s]+", line)
        if len(parts) >= 2:
            try:
                x_start = float(parts[0]) * x_factor
                if n_pts > 1 and len(x_vals) < n_pts:
                    delta = (last_x - first_x) / (n_pts - 1)
                else:
                    delta = (last_x - first_x) / max(n_pts - 1, 1)
                for i, tok in enumerate(parts[1:]):
                    try:
                        y_vals.append(float(tok) * y_factor)
                        x_vals.append(x_start + i * abs(delta))
                    except ValueError:
                        continue
            except ValueError:
                continue

    # If still empty, fall back to uniform grid + all numeric tokens
    if not y_vals and n_pts > 0:
        import numpy as np
        x_vals = list(np.linspace(first_x, last_x, n_pts))
        tokens = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", block)
        y_vals = [float(t) * y_factor for t in tokens[:n_pts]]

    return x_vals, y_vals


def _pick_peaks(
    wavenumbers: "np.ndarray",
    intensities: "np.ndarray",
    min_intensity: float = 0.05,
    min_prominence: float = 0.03,
) -> list[dict]:
    """
    Pick absorbance peaks (local maxima above threshold).
    Returns list sorted by intensity descending.
    """
    import numpy as np

    if len(intensities) < 3:
        return []

    try:
        from scipy.signal import find_peaks
        peak_indices, props = find_peaks(
            intensities,
            height=min_intensity,
            prominence=min_prominence,
        )
    except ImportError:
        # Manual local-maxima fallback
        peak_indices = []
        for i in range(1, len(intensities) - 1):
            if (intensities[i] > min_intensity and
                    intensities[i] >= intensities[i - 1] and
                    intensities[i] >= intensities[i + 1]):
                peak_indices.append(i)
        peak_indices = np.array(peak_indices)

    peaks = []
    for i in peak_indices:
        wn = float(wavenumbers[i])
        it = float(intensities[i])
        if 200 <= wn <= 5000:
            peaks.append({"wavenumber": round(wn, 1), "intensity": round(it, 4)})

    peaks.sort(key=lambda p: p["intensity"], reverse=True)
    return peaks

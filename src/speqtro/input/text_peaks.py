"""
Inline text NMR peak parser.

Handles the many ways a chemist types NMR data on the command line:

  ¹H inline styles:
    "7.26 (5H, m), 3.71 (2H, s), 2.35 (3H, s)"
    "δ 7.26 (m, 5H), 3.71 (s, 2H)"
    "7.26 m 5H; 3.71 s 2H"
    "7.26, 3.71, 2.35"              (shifts only)

  ¹³C inline styles:
    "128.5, 77.2, 45.3, 21.3"
    "δ 128.5 (s), 77.2 (s)"
    "128.5; 77.2; 45.3"

Returns normalized peak list:
  [{"shift": float, "integral": float|None,
    "multiplicity": str|None, "coupling_hz": list|None}, ...]
"""

import re
from typing import Optional


# Recognized multiplicity tokens
_MULT_TOKENS = {
    "s": "s", "singlet": "s",
    "d": "d", "doublet": "d",
    "t": "t", "triplet": "t",
    "q": "q", "quartet": "q",
    "m": "m", "multiplet": "m",
    "dd": "dd", "dt": "dt", "td": "td", "tt": "tt",
    "ddd": "ddd", "ddt": "ddt", "dtd": "dtd",
    "qd": "qd", "dq": "dq",
    "br": "br s", "br s": "br s", "br d": "br d",
    "brs": "br s", "brd": "br d",
    "sept": "sept", "septet": "sept",
    "sex": "sextet", "sextet": "sextet",
    "h": "hept", "hept": "hept",
}


def parse_inline_peaks(text: str) -> list[dict]:
    """
    Parse inline NMR peak text into a normalized peak list.

    Handles comma/semicolon separated entries, optional parenthetical
    annotations (nH, multiplicity, J values), and plain shift lists.

    Args:
        text: Raw NMR string from user input.

    Returns:
        Sorted (high→low ppm) list of normalized peak dicts.
    """
    if not text or not text.strip():
        return []

    # Strip δ prefix, "ppm" labels
    text = re.sub(r"δ\s*", "", text)
    text = re.sub(r"\bppm\b", "", text, flags=re.IGNORECASE)

    # Split into individual peak entries on comma or semicolon,
    # but NOT commas inside parentheses
    entries = _split_entries(text)

    peaks = []
    for entry in entries:
        peak = _parse_entry(entry.strip())
        if peak is not None:
            peaks.append(peak)

    return sorted(peaks, key=lambda p: -p["shift"])


def _split_entries(text: str) -> list[str]:
    """
    Split text into peak entries, respecting parentheses.
    Splits on , or ; that are not inside ().
    """
    entries = []
    depth = 0
    current = []
    for ch in text:
        if ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth -= 1
            current.append(ch)
        elif ch in (",", ";") and depth == 0:
            entry = "".join(current).strip()
            if entry:
                entries.append(entry)
            current = []
        else:
            current.append(ch)
    remainder = "".join(current).strip()
    if remainder:
        entries.append(remainder)
    return entries


def _parse_entry(entry: str) -> Optional[dict]:
    """
    Parse a single peak entry like:
      "7.26 (5H, m)"
      "3.71 (s, 2H)"
      "2.35 (3H, s, J = 7.2)"
      "128.5"
    """
    if not entry:
        return None

    # Extract the shift value (first float in the string)
    shift_match = re.search(r"(\d+\.?\d*)", entry)
    if not shift_match:
        return None

    try:
        shift = float(shift_match.group(1))
    except ValueError:
        return None

    if shift < 0 or shift > 250:
        return None  # Sanity check

    integral = None
    multiplicity = None
    coupling_hz = None

    # Extract parenthetical annotation if present
    paren_match = re.search(r"\(([^)]+)\)", entry)
    if paren_match:
        ann = paren_match.group(1)
        integral, multiplicity, coupling_hz = _parse_annotation(ann)
    else:
        # No parens — check for inline multiplicity token after the shift
        remainder = entry[shift_match.end():].strip()
        multiplicity, coupling_hz = _extract_mult_from_text(remainder)
        integral = _extract_integral_from_text(remainder)

    return {
        "shift": round(shift, 4),
        "intensity": None,
        "integral": integral,
        "multiplicity": multiplicity,
        "coupling_hz": coupling_hz,
    }


def _parse_annotation(ann: str) -> tuple:
    """
    Parse parenthetical annotation: "(5H, m)", "(s, 2H)", "(3H, s, J=7.2)".
    Returns (integral, multiplicity, coupling_hz).
    """
    integral = None
    multiplicity = None
    coupling_hz = None

    # Extract J coupling constants: J = 7.2, 3.4 or J(Hz) = 7.2
    j_match = re.search(
        r"[Jj]\s*(?:\([Hh][Zz]\))?\s*[=:]\s*([\d.,\s]+)", ann
    )
    if j_match:
        nums = re.findall(r"\d+\.?\d*", j_match.group(1))
        coupling_hz = [float(n) for n in nums] if nums else None
        ann = ann[:j_match.start()] + ann[j_match.end():]

    # Extract integral (nH pattern)
    h_match = re.search(r"(\d+\.?\d*)\s*[Hh]\b", ann)
    if h_match:
        try:
            integral = float(h_match.group(1))
        except ValueError:
            pass
        ann = ann[:h_match.start()] + ann[h_match.end():]

    # Extract multiplicity token
    multiplicity, _ = _extract_mult_from_text(ann)

    return integral, multiplicity, coupling_hz


def _extract_mult_from_text(text: str) -> tuple:
    """Find multiplicity token in text. Returns (multiplicity, remaining_text)."""
    # Try longest tokens first (dd before d)
    tokens_sorted = sorted(_MULT_TOKENS.keys(), key=len, reverse=True)
    text_lower = text.lower().strip()
    for token in tokens_sorted:
        # Match whole word
        pattern = r"\b" + re.escape(token) + r"\b"
        if re.search(pattern, text_lower):
            return _MULT_TOKENS[token], text
    return None, text


def _extract_integral_from_text(text: str) -> Optional[float]:
    """Extract nH integral from text like '3H' or '3 H'."""
    m = re.search(r"(\d+\.?\d*)\s*[Hh]\b", text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


# ── Convenience: parse comma-separated 13C shifts ────────────────────────────

def parse_c13_text(text: str) -> list[dict]:
    """
    Parse simple ¹³C shift text: "128.5, 77.2, 45.3, 21.3" or
    "δ 128.5 (s), 77.2 (s), 45.3 (s)".

    Returns normalized peak list (no multiplicity expected for 13C).
    """
    # Full parser handles this fine
    peaks = parse_inline_peaks(text)
    # For 13C, anything 0-230 ppm is reasonable
    return [p for p in peaks if 0 <= p["shift"] <= 240]

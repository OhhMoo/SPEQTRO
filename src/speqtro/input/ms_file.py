"""
MS/MS data parser — supports MGF, mzML, two-column CSV, and inline text.

All parsers return a canonical dict:
    {
        "precursor_mz":     float | None,
        "collision_energy": float | None,   # eV
        "adduct":           str | None,     # e.g. "[M+H]+"
        "charge":           int | None,
        "peaks":            list[dict],     # [{"mz": float, "intensity": float}, ...]
        "source_format":    str,
        "n_spectra":        int,            # number of spectra found in file
        "_warning":         str | None,
    }

Intensities are normalised 0–1 relative to the base peak (highest intensity = 1.0).
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Optional


# ── Public entry point ────────────────────────────────────────────────────────

def parse(source) -> dict:
    """
    Route MS data to the correct parser.

    Args:
        source: File path (str or Path) pointing to .mgf / .mzml / .mzxml / .csv.

    Returns:
        Canonical MS dict (see module docstring).

    Raises:
        ValueError: Unsupported file format.
        FileNotFoundError: Path does not exist.
    """
    p = Path(source)
    if not p.exists():
        raise FileNotFoundError(f"MS file not found: {p}")

    suffix = p.suffix.lower()
    if suffix == ".mgf":
        return parse_mgf(p)
    if suffix in (".mzml", ".mzxml"):
        return parse_mzml(p)
    if suffix in (".csv", ".tsv", ".txt"):
        return parse_ms_csv(p)

    raise ValueError(
        f"Unsupported MS file format: '{suffix}'. "
        "Supported: .mgf, .mzml, .mzxml, .csv/.tsv/.txt"
    )


# ── MGF parser ────────────────────────────────────────────────────────────────

def parse_mgf(path: Path) -> dict:
    """
    Parse an MGF (Mascot Generic Format) file.
    Returns the first spectrum; sets n_spectra to total count.

    MGF format example:
        BEGIN IONS
        PEPMASS=181.0863 100.0
        CHARGE=1+
        COLLISION_ENERGY=20 eV
        181.0863 100.0
        139.0542 45.2
        77.0386  23.1
        END IONS
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")

    # Split into spectrum blocks
    blocks = re.findall(
        r"BEGIN IONS(.*?)END IONS",
        text, re.DOTALL | re.IGNORECASE,
    )
    n_spectra = len(blocks)
    if not blocks:
        return _empty_result("mgf", warning="No IONS blocks found in MGF file.")

    # Parse first block
    block = blocks[0]
    result = _parse_mgf_block(block)
    result["n_spectra"] = n_spectra
    result["source_format"] = "mgf"

    if n_spectra > 1:
        result["_warning"] = (
            f"MGF file contains {n_spectra} spectra. "
            "Only the first spectrum is used. "
            "Export a single spectrum for best results."
        )
    return result


def _parse_mgf_block(block: str) -> dict:
    """Parse a single BEGIN IONS … END IONS block."""
    precursor_mz = None
    collision_energy = None
    adduct = None
    charge = None
    peaks: list[dict] = []

    for line in block.strip().splitlines():
        line = line.strip()
        if not line:
            continue

        # Key=Value metadata lines
        if "=" in line and not re.match(r"^\d", line):
            key, _, val = line.partition("=")
            key = key.strip().upper()
            val = val.strip()

            if key == "PEPMASS":
                # May be "181.0863" or "181.0863 100.0"
                parts = val.split()
                try:
                    precursor_mz = float(parts[0])
                except (ValueError, IndexError):
                    pass

            elif key in ("CHARGE",):
                # "1+" or "2-" or "1"
                m = re.match(r"(\d+)", val)
                if m:
                    charge = int(m.group(1))

            elif key in ("COLLISION_ENERGY", "CE", "ENERGY"):
                collision_energy = _parse_collision_energy(val)

            elif key in ("ADDUCT", "PRECURSOR_TYPE", "IONMODE"):
                adduct = _parse_adduct(val)
            continue

        # Numeric lines: m/z [intensity]
        parts = re.split(r"[\s,\t]+", line)
        if len(parts) >= 1:
            try:
                mz = float(parts[0])
                intensity = float(parts[1]) if len(parts) >= 2 else 1.0
                if mz > 0:
                    peaks.append({"mz": round(mz, 4), "intensity": round(intensity, 4)})
            except ValueError:
                continue

    # Infer adduct from charge if not explicit
    if adduct is None and charge is not None:
        adduct = f"[M+H]+" if charge > 0 else "[M-H]-"

    return {
        "precursor_mz": precursor_mz,
        "collision_energy": collision_energy,
        "adduct": adduct,
        "charge": charge,
        "peaks": _normalize_peaks(peaks),
        "n_spectra": 1,
        "_warning": None,
    }


# ── mzML / mzXML parser ───────────────────────────────────────────────────────

def parse_mzml(path: Path) -> dict:
    """
    Parse an mzML or mzXML file.
    Uses pyteomics if available; falls back to an ElementTree implementation.
    Returns the first MS2 spectrum found (or first spectrum if no MS2).
    """
    suffix = path.suffix.lower()

    try:
        if suffix == ".mzml":
            return _parse_mzml_pyteomics(path)
        else:
            return _parse_mzxml_pyteomics(path)
    except ImportError:
        pass  # pyteomics not installed — use ElementTree fallback

    return _parse_mzml_elementtree(path)


def _parse_mzml_pyteomics(path: Path) -> dict:
    from pyteomics import mzml
    with mzml.MzML(str(path)) as reader:
        for spec in reader:
            ms_level = spec.get("ms level", 1)
            if ms_level != 2:
                continue
            return _extract_mzml_spectrum(spec)
        # No MS2 found — try first spectrum
        reader.reset()
        spec = next(iter(reader), None)
        if spec:
            return _extract_mzml_spectrum(spec)
    return _empty_result("mzml", warning="No spectra found in mzML file.")


def _parse_mzxml_pyteomics(path: Path) -> dict:
    from pyteomics import mzxml
    with mzxml.MzXML(str(path)) as reader:
        for spec in reader:
            if spec.get("msLevel", 1) == 2:
                return _extract_mzxml_spectrum(spec)
        reader.reset()
        spec = next(iter(reader), None)
        if spec:
            return _extract_mzxml_spectrum(spec)
    return _empty_result("mzxml", warning="No spectra found in mzXML file.")


def _extract_mzml_spectrum(spec: dict) -> dict:
    """Convert a pyteomics mzML spectrum dict to canonical format."""
    import numpy as np
    mz_array = spec.get("m/z array", np.array([]))
    int_array = spec.get("intensity array", np.array([]))

    precursor_mz = None
    collision_energy = None
    adduct = None
    charge = None

    precursor_list = spec.get("precursorList", {}).get("precursor", [])
    if precursor_list:
        prec = precursor_list[0]
        si = prec.get("selectedIonList", {}).get("selectedIon", [{}])[0]
        precursor_mz = si.get("selected ion m/z")
        charge = si.get("charge state")
        act = prec.get("activation", {})
        collision_energy = act.get("collision energy")

    peaks = [
        {"mz": round(float(m), 4), "intensity": round(float(i), 4)}
        for m, i in zip(mz_array, int_array)
        if float(i) > 0
    ]

    return {
        "precursor_mz": float(precursor_mz) if precursor_mz else None,
        "collision_energy": float(collision_energy) if collision_energy else None,
        "adduct": adduct,
        "charge": int(charge) if charge else None,
        "peaks": _normalize_peaks(peaks),
        "source_format": "mzml",
        "n_spectra": 1,
        "_warning": None,
    }


def _extract_mzxml_spectrum(spec: dict) -> dict:
    """Convert a pyteomics mzXML spectrum dict to canonical format."""
    import numpy as np
    mz_array = spec.get("m/z array", np.array([]))
    int_array = spec.get("intensity array", np.array([]))

    peaks = [
        {"mz": round(float(m), 4), "intensity": round(float(i), 4)}
        for m, i in zip(mz_array, int_array)
        if float(i) > 0
    ]

    return {
        "precursor_mz": spec.get("precursorMz", [{}])[0].get("precursorMz") if spec.get("precursorMz") else None,
        "collision_energy": None,
        "adduct": None,
        "charge": spec.get("precursorMz", [{}])[0].get("precursorIntensity") if spec.get("precursorMz") else None,
        "peaks": _normalize_peaks(peaks),
        "source_format": "mzxml",
        "n_spectra": 1,
        "_warning": None,
    }


def _parse_mzml_elementtree(path: Path) -> dict:
    """
    Minimal mzML parser using ElementTree.
    Handles the most common Thermo/Waters export format.
    Decodes base64+zlib binary arrays for m/z and intensity.
    """
    import xml.etree.ElementTree as ET
    import base64
    import struct
    import zlib

    NS = "http://psi.hupo.org/ms/mzml"

    try:
        tree = ET.parse(str(path))
    except ET.ParseError as e:
        return _empty_result("mzml", warning=f"mzML XML parse error: {e}")

    root = tree.getroot()

    def _find_all(elem, tag):
        return elem.findall(f"{{{NS}}}{tag}")

    def _find(elem, tag):
        return elem.find(f"{{{NS}}}{tag}")

    def _cv_param(elem, name: str):
        for cv in _find_all(elem, "cvParam"):
            if cv.get("name", "").lower() == name.lower():
                return cv.get("value")
        return None

    def _decode_array(bda_elem) -> list[float]:
        """Decode a <binaryDataArray> element to a list of floats."""
        binary_elem = _find(bda_elem, "binary")
        if binary_elem is None or not binary_elem.text:
            return []

        data = base64.b64decode(binary_elem.text.strip())

        # Check compression
        is_zlib = any(
            cv.get("accession") in ("MS:1000574", "MS:1000576")
            for cv in _find_all(bda_elem, "cvParam")
        )
        if is_zlib:
            try:
                data = zlib.decompress(data)
            except zlib.error:
                pass

        # Check precision (32-bit vs 64-bit)
        is_64bit = any(
            cv.get("accession") == "MS:1000514"
            or "64" in cv.get("name", "")
            for cv in _find_all(bda_elem, "cvParam")
        )
        fmt = "d" if is_64bit else "f"
        n = len(data) // struct.calcsize(fmt)
        return list(struct.unpack(f"<{n}{fmt}", data[:n * struct.calcsize(fmt)]))

    # Find first MS2 spectrum
    spectrum_list = _find(root, "run")
    if spectrum_list is not None:
        spectrum_list = _find(spectrum_list, "spectrumList")

    if spectrum_list is None:
        return _empty_result("mzml", warning="No spectrumList found in mzML.")

    spectra = _find_all(spectrum_list, "spectrum")
    target_spec = None
    for spec in spectra:
        ms_level = _cv_param(spec, "ms level")
        if ms_level == "2":
            target_spec = spec
            break
    if target_spec is None and spectra:
        target_spec = spectra[0]
    if target_spec is None:
        return _empty_result("mzml", warning="No spectra in mzML.")

    # Extract precursor info
    precursor_mz = None
    collision_energy = None
    prec_list = _find(target_spec, "precursorList")
    if prec_list:
        for prec in _find_all(prec_list, "precursor"):
            si_list = _find(prec, "selectedIonList")
            if si_list:
                for si in _find_all(si_list, "selectedIon"):
                    val = _cv_param(si, "selected ion m/z")
                    if val:
                        try:
                            precursor_mz = float(val)
                        except ValueError:
                            pass
            act = _find(prec, "activation")
            if act:
                ce_val = _cv_param(act, "collision energy")
                if ce_val:
                    try:
                        collision_energy = float(ce_val)
                    except ValueError:
                        pass

    # Extract m/z and intensity arrays
    bda_list = _find(target_spec, "binaryDataArrayList")
    mz_array: list[float] = []
    int_array: list[float] = []

    if bda_list:
        for bda in _find_all(bda_list, "binaryDataArray"):
            is_mz = any(
                "m/z" in cv.get("name", "").lower() or cv.get("accession") == "MS:1000514"
                for cv in _find_all(bda, "cvParam")
            )
            is_int = any(
                "intensity" in cv.get("name", "").lower() or cv.get("accession") == "MS:1000515"
                for cv in _find_all(bda, "cvParam")
            )
            decoded = _decode_array(bda)
            if is_mz:
                mz_array = decoded
            elif is_int:
                int_array = decoded

    peaks = [
        {"mz": round(float(m), 4), "intensity": round(float(i), 4)}
        for m, i in zip(mz_array, int_array)
        if float(i) > 0
    ]

    return {
        "precursor_mz": precursor_mz,
        "collision_energy": collision_energy,
        "adduct": None,
        "charge": None,
        "peaks": _normalize_peaks(peaks),
        "source_format": "mzml",
        "n_spectra": len(spectra),
        "_warning": (
            None if len(spectra) <= 1
            else f"mzML contains {len(spectra)} spectra; first MS2 used."
        ),
    }


# ── CSV parser ────────────────────────────────────────────────────────────────

def parse_ms_csv(path: Path) -> dict:
    """
    Parse a two-column (m/z, intensity) CSV/TSV file.
    The first row is treated as a header if it contains non-numeric values.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    try:
        import csv as _csv
        dialect = _csv.Sniffer().sniff(text[:2000]) if text[:2000].strip() else "excel"
    except Exception:
        dialect = "excel"

    reader = csv.reader(text.splitlines(), dialect=dialect)
    rows = [r for r in reader if any(c.strip() for c in r)]

    if not rows:
        return _empty_result("ms_csv", warning="Empty CSV file.")

    # Skip header
    data_start = 1 if not _is_float(rows[0][0].strip()) else 0
    data_rows = rows[data_start:]

    peaks: list[dict] = []
    for row in data_rows:
        if len(row) < 1:
            continue
        try:
            mz = float(row[0].strip())
            intensity = float(row[1].strip()) if len(row) >= 2 else 1.0
            if mz > 0:
                peaks.append({"mz": round(mz, 4), "intensity": round(intensity, 4)})
        except (ValueError, IndexError):
            continue

    return {
        "precursor_mz": None,
        "collision_energy": None,
        "adduct": None,
        "charge": None,
        "peaks": _normalize_peaks(peaks),
        "source_format": "ms_csv",
        "n_spectra": 1,
        "_warning": (
            "No precursor m/z or collision energy in CSV. "
            "Provide --ms-adduct and --ms-ce flags if known."
            if peaks else "No MS peaks parsed from CSV."
        ),
    }


# ── Inline text parser ────────────────────────────────────────────────────────

def parse_ms_inline(text: str) -> dict:
    """
    Parse inline MS/MS text. Handles formats like:
      "m/z 181 [M+H]+, 139, 77"
      "181.0 (100%), 139.1 (45%), 77.0 (23%)"
      "181.0863 100 139.0542 45 77.0386 23"

    Args:
        text: Raw MS string from user input.

    Returns:
        Canonical MS dict.
    """
    if not text or not text.strip():
        return _empty_result("ms_inline", warning="Empty MS text.")

    precursor_mz = None
    adduct = None

    # ── Step 1: extract precursor + adduct ─────────────────────────────
    # Pattern: float followed by [M...]+/- adduct tag
    prec_match = re.search(
        r"m/?z\s*[:=]?\s*([\d.]+)\s*(\[[Mm][^\]]*\][+-][0-9]*)",
        text, re.IGNORECASE,
    )
    if not prec_match:
        # Try: float immediately followed by adduct bracket
        prec_match = re.search(
            r"([\d.]+)\s*(\[[Mm][^\]]*\][+-][0-9]*)",
            text,
        )
    if prec_match:
        try:
            precursor_mz = float(prec_match.group(1))
            adduct = _parse_adduct(prec_match.group(2))
        except ValueError:
            pass

    # ── Step 2: extract m/z intensity pairs (float (percent%)) ─────────
    peaks: list[dict] = []
    pair_matches = re.findall(r"([\d.]+)\s*\(\s*([\d.]+)\s*%?\s*\)", text)
    if pair_matches:
        for mz_str, int_str in pair_matches:
            try:
                mz = float(mz_str)
                intensity = float(int_str) / 100.0  # percent → fraction
                if mz > 10:
                    peaks.append({"mz": round(mz, 4), "intensity": round(intensity, 4)})
            except ValueError:
                continue

    # ── Step 3: bare m/z list if no pairs found ─────────────────────────
    if not peaks:
        # Extract all standalone floats ≥ 10 Da as fragment m/z
        bare = re.findall(r"\b(\d{2,4}\.?\d*)\b", text)
        for mz_str in bare:
            try:
                mz = float(mz_str)
                if 10 < mz < 5000:
                    peaks.append({"mz": round(mz, 4), "intensity": 1.0})
            except ValueError:
                continue
        # Remove duplicates (keep first occurrence)
        seen: set[float] = set()
        deduped = []
        for p in peaks:
            if p["mz"] not in seen:
                seen.add(p["mz"])
                deduped.append(p)
        peaks = deduped

    # If precursor was found and not already in peaks, set intensity to 1.0
    if precursor_mz and not any(abs(p["mz"] - precursor_mz) < 0.02 for p in peaks):
        peaks.insert(0, {"mz": round(precursor_mz, 4), "intensity": 1.0})

    collision_energy = _parse_collision_energy(text)

    return {
        "precursor_mz": precursor_mz,
        "collision_energy": collision_energy,
        "adduct": adduct,
        "charge": None,
        "peaks": _normalize_peaks(peaks),
        "source_format": "ms_inline",
        "n_spectra": 1,
        "_warning": (
            "Inline MS text parsed heuristically. "
            "Provide a .mgf or .mzml file for precise m/z values."
            if peaks else "No m/z values could be extracted from text."
        ),
    }


# ── Shared utilities ─────────────────────────────────────────────────────────

def _normalize_peaks(peaks: list[dict]) -> list[dict]:
    """Sort by m/z, normalise intensities to [0, 1] relative to base peak."""
    if not peaks:
        return []
    peaks = sorted(peaks, key=lambda p: p["mz"])
    max_i = max(p["intensity"] for p in peaks)
    if max_i > 0:
        for p in peaks:
            p["intensity"] = round(p["intensity"] / max_i, 4)
    return peaks


def _parse_adduct(raw: str) -> Optional[str]:
    """Normalise common adduct string variants to canonical form."""
    if not raw:
        return None
    raw = raw.strip()
    # Already canonical
    if re.match(r"^\[M[^\]]*\][+-][0-9]*$", raw):
        return raw
    # Shorthand variants: "MH+", "M+H", "M-H"
    normalisation = {
        "mh+": "[M+H]+", "m+h": "[M+H]+", "[m+h]+": "[M+H]+",
        "m-h": "[M-H]-", "[m-h]-": "[M-H]-", "mh-": "[M-H]-",
        "m+na": "[M+Na]+", "[m+na]+": "[M+Na]+",
        "m+nh4": "[M+NH4]+", "[m+nh4]+": "[M+NH4]+",
        "m+k": "[M+K]+", "[m+k]+": "[M+K]+",
        "m+2h": "[M+2H]2+", "[m+2h]2+": "[M+2H]2+",
        "m+cl": "[M+Cl]-", "[m+cl]-": "[M+Cl]-",
    }
    return normalisation.get(raw.lower(), raw)


def _parse_collision_energy(raw: str) -> Optional[float]:
    """Extract numeric eV value from strings like '20 eV', 'CE=20', '20.0 V'."""
    if not raw:
        return None
    m = re.search(r"(?:ce|collision[_\s]?energy|energy)[\s=:]*(\d+\.?\d*)", raw, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    # Try plain number followed by optional eV/V unit
    m = re.search(r"\b(\d+\.?\d*)\s*(?:ev|v)\b", raw, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _empty_result(source_format: str, warning: str = None) -> dict:
    return {
        "precursor_mz": None,
        "collision_energy": None,
        "adduct": None,
        "charge": None,
        "peaks": [],
        "source_format": source_format,
        "n_spectra": 0,
        "_warning": warning,
    }

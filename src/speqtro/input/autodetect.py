"""
Autodetect router — route any spectroscopic input to the correct parser.

Supported inputs:
  - Bruker experiment directory (contains acqus or pdata/)
  - JCAMP-DX files (.jdx, .dx, .jcamp) — NMR or IR, auto-discriminated
  - MestReNova multiplet export (.txt, .csv, .tsv)
  - Generic CSV peak list
  - Spectrum image (.png, .jpg, .jpeg, .pdf, .tiff) — NMR or IR
  - MGF file (.mgf) — MS/MS
  - mzML / mzXML file (.mzml, .mzxml) — MS/MS
  - Python dict of peaks (pass-through)

NMR parsers return a normalized spectrum dict:
    {
        "peaks": [{"shift": float, "intensity": float|None,
                   "integral": float|None, "multiplicity": str|None,
                   "coupling_hz": list[float]|None}, ...],
        "nucleus": str,          # "1H", "13C", "31P", etc.
        "solvent": str,          # e.g. "CDCl3"
        "frequency_mhz": float|None,
        "source_format": str,
        "raw_spectrum": array|None,
        "ppm_scale": array|None,
    }

IR parsers return:
    {
        "wavenumbers": list[float],   # cm⁻¹, ascending
        "intensities": list[float],   # normalised 0–1 absorbance
        "peaks": list[dict],          # [{"wavenumber": float, "intensity": float}, ...]
        "y_units": str,
        "source_format": str,
        "spectrum_type": "ir",
    }

MS parsers return:
    {
        "precursor_mz": float|None,
        "collision_energy": float|None,
        "adduct": str|None,
        "charge": int|None,
        "peaks": [{"mz": float, "intensity": float}, ...],
        "source_format": str,
        "spectrum_type": "ms",
    }

Use parse_spectrum() for NMR/IR/unknown inputs.
Use parse_ms() for MS files.
Use parse_any() to auto-detect all types including MS.
"""

from pathlib import Path
from typing import Union

# Image suffixes that can be NMR or IR images
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".pdf", ".tiff", ".tif"}
# MS file suffixes
_MS_SUFFIXES = {".mgf", ".mzml", ".mzxml"}
# JCAMP suffixes
_JCAMP_SUFFIXES = {".jdx", ".dx", ".jcamp"}
# Text/CSV suffixes (NMR exports)
_CSV_SUFFIXES = {".txt", ".csv", ".tsv"}


# ── Public entry points ────────────────────────────────────────────────────────

def parse_any(
    source: Union[str, Path, dict, list],
    spectrum_hint: str = "auto",
) -> dict:
    """
    Parse any spectroscopic input, including MS files.

    Args:
        source: File path, dict, or list.
        spectrum_hint: "nmr", "ir", "ms", or "auto" (default).

    Returns:
        Parsed spectrum dict with a "spectrum_type" key added:
        "nmr", "ir", or "ms".
    """
    if isinstance(source, (dict, list)):
        return parse_spectrum(source)

    p = Path(source)
    suffix = p.suffix.lower() if p.is_file() else ""

    if spectrum_hint == "ms" or suffix in _MS_SUFFIXES:
        return parse_ms(p)

    # Images: could be NMR or IR — use hint or fall through to parse_spectrum
    if suffix in _IMAGE_SUFFIXES:
        if spectrum_hint == "ir":
            return _parse_ir_image(p)
        # Default images to NMR spectrum parser (existing behaviour)
        result = parse_spectrum(source)
        result.setdefault("spectrum_type", "nmr")
        return result

    # JCAMP: may be NMR or IR
    if suffix in _JCAMP_SUFFIXES:
        if spectrum_hint == "ir":
            return _parse_ir_jcamp(p)
        if spectrum_hint == "nmr":
            result = _parse_nmr_jcamp(p)
            result.setdefault("spectrum_type", "nmr")
            return result
        # Auto-discriminate
        return _route_jcamp(p)

    # Everything else: NMR path
    result = parse_spectrum(source)
    result.setdefault("spectrum_type", "nmr")
    return result


def parse_spectrum(source: Union[str, Path, dict, list]) -> dict:
    """
    Parse any NMR (or ambiguous) input into a normalized NMR spectrum dict.

    Args:
        source: One of:
            - str/Path pointing to a file or Bruker directory
            - dict already in normalized format (pass-through)
            - list of peaks as [{"shift": ..., "multiplicity": ...}, ...]

    Returns:
        Normalized NMR spectrum dict (see module docstring).

    Raises:
        ValueError: Unsupported format or unparseable file.
        FileNotFoundError: Path does not exist.
    """
    # Pass-through: already parsed dict
    if isinstance(source, dict):
        _validate_normalized(source)
        return source

    # Pass-through: bare peak list
    if isinstance(source, list):
        return {
            "peaks": _normalize_peak_list(source),
            "nucleus": "1H",
            "solvent": "unknown",
            "frequency_mhz": None,
            "source_format": "peak_list",
            "raw_spectrum": None,
            "ppm_scale": None,
            "spectrum_type": "nmr",
        }

    p = Path(source)
    if not p.exists():
        raise FileNotFoundError(f"Path not found: {p}")

    # Bruker directory
    if p.is_dir():
        if (p / "acqus").exists() or (p / "pdata").exists():
            from speqtro.input import bruker
            result = bruker.parse(p)
            result.setdefault("spectrum_type", "nmr")
            return result
        if (p / "procpar").exists():
            raise NotImplementedError(
                "Varian/Agilent format not yet supported. "
                "Export to JCAMP-DX from VnmrJ and retry."
            )
        raise ValueError(f"Directory does not look like a Bruker experiment: {p}")

    suffix = p.suffix.lower()

    # JCAMP-DX: discriminate IR vs NMR
    if suffix in _JCAMP_SUFFIXES:
        return _route_jcamp(p)

    # MS files: delegate to parse_ms
    if suffix in _MS_SUFFIXES:
        return parse_ms(p)

    # MestReNova / CSV / TSV
    if suffix in _CSV_SUFFIXES:
        from speqtro.input import mestrexport, csv_peaks
        try:
            result = mestrexport.parse(p)
        except mestrexport.NotMestReNovaFormat:
            result = csv_peaks.parse(p)
        result.setdefault("spectrum_type", "nmr")
        return result

    # Spectrum image — default to NMR; caller should use parse_any(..., hint="ir") for IR
    if suffix in _IMAGE_SUFFIXES:
        from speqtro.input import image_spectrum
        result = image_spectrum.parse(p)
        result.setdefault("spectrum_type", "nmr")
        return result

    raise ValueError(
        f"Unsupported file format: '{suffix}'. "
        "Supported: Bruker dir, .jdx/.dx/.jcamp, .csv/.txt/.tsv, "
        ".png/.jpg/.jpeg/.pdf/.tiff, .mgf, .mzml, .mzxml"
    )


def parse_ms(source: Union[str, Path]) -> dict:
    """
    Parse an MS/MS file into a normalized MS spectrum dict.

    Supports: .mgf, .mzml, .mzxml, .csv (two-column m/z, intensity).

    Returns:
        {precursor_mz, collision_energy, adduct, charge,
         peaks: [{mz, intensity}], source_format, spectrum_type: "ms"}
    """
    from speqtro.input.ms_file import parse as ms_parse
    result = ms_parse(source)
    result["spectrum_type"] = "ms"
    return result


def parse_ir(source: Union[str, Path]) -> dict:
    """
    Parse an IR spectrum file or image into a normalized IR dict.

    Supports: .jdx/.dx/.jcamp (JCAMP-DX IR), .png/.jpg/.jpeg/.pdf/.tiff (image).

    Returns:
        {wavenumbers, intensities, peaks, y_units, source_format, spectrum_type: "ir"}
    """
    p = Path(source)
    if not p.exists():
        raise FileNotFoundError(f"Path not found: {p}")

    suffix = p.suffix.lower()

    if suffix in _JCAMP_SUFFIXES:
        return _parse_ir_jcamp(p)

    if suffix in _IMAGE_SUFFIXES:
        return _parse_ir_image(p)

    raise ValueError(
        f"Unsupported IR file format: '{suffix}'. "
        "Supported: .jdx/.dx/.jcamp, .png/.jpg/.jpeg/.pdf/.tiff"
    )


# ── Private helpers ────────────────────────────────────────────────────────────

def _route_jcamp(p: Path) -> dict:
    """
    Discriminate between IR and NMR JCAMP-DX files and route accordingly.
    Reads the first 3 kB of the file to check DATA TYPE / XUNITS / axis range.
    """
    text_head = p.read_text(encoding="utf-8", errors="replace")[:3000]

    from speqtro.input.ir_jcamp import is_ir_jcamp
    if is_ir_jcamp(text_head):
        return _parse_ir_jcamp(p)

    return _parse_nmr_jcamp(p)


def _parse_ir_jcamp(p: Path) -> dict:
    from speqtro.input.ir_jcamp import parse as parse_ir_jcamp
    result = parse_ir_jcamp(p)
    result["spectrum_type"] = "ir"
    return result


def _parse_nmr_jcamp(p: Path) -> dict:
    from speqtro.input import jcamp
    result = jcamp.parse(p)
    result.setdefault("spectrum_type", "nmr")
    return result


def _parse_ir_image(p: Path) -> dict:
    from speqtro.input.ir_image import parse as parse_ir_image
    result = parse_ir_image(p)
    result["spectrum_type"] = "ir"
    return result


def _normalize_peak_list(peaks: list) -> list:
    """Ensure each peak dict has the expected NMR keys."""
    normalized = []
    for p in peaks:
        if isinstance(p, (int, float)):
            p = {"shift": float(p)}
        normalized.append({
            "shift": float(p.get("shift", p.get("ppm", 0.0))),
            "intensity": p.get("intensity"),
            "integral": p.get("integral"),
            "multiplicity": p.get("multiplicity"),
            "coupling_hz": p.get("coupling_hz") or p.get("J"),
        })
    return sorted(normalized, key=lambda x: -x["shift"])


def _validate_normalized(d: dict) -> None:
    """Lightly validate a pass-through dict has required keys."""
    if "peaks" not in d:
        raise ValueError("Normalized spectrum dict must contain 'peaks' key.")

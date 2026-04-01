"""
SpectralInput — unified container for all user-provided spectroscopic data.

This is the bridge between the CLI (user intent) and the agent (structured context).
A SpectralInput holds whatever combination of data the user provided:
  - SMILES strings (expected product and/or starting material)
  - ¹H NMR peaks (from inline text or parsed file)
  - ¹³C NMR peaks (from inline text or parsed file)
  - MS/MS data (file, inline text, or pre-parsed peaks)
  - IR spectrum (file or inline wavenumber list)
  - Solvent, spectrometer frequency, reaction type

Usage:
    inp = SpectralInput.from_cli(
        smiles="CC(=O)O",
        h1="7.26 (5H, m), 3.71 (2H, s)",
        c13="128.5, 77.2, 45.3",
        solvent="CDCl3",
    )
    context_str = inp.to_context_string()
    tool_args   = inp.to_dict()   # inject directly into pipeline tool calls
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class SpectralInput:
    """Container for all spectroscopic input data provided by the user."""

    # ── Structure ──────────────────────────────────────────────────────
    smiles: Optional[str] = None          # Expected product SMILES
    sm_smiles: Optional[str] = None       # Starting material SMILES
    molecular_formula: Optional[str] = None

    # ── NMR ───────────────────────────────────────────────────────────
    h1_peaks: list[dict] = field(default_factory=list)   # normalized peak dicts
    c13_peaks: list[dict] = field(default_factory=list)
    h1_source: str = "unknown"    # "inline", "bruker", "jcamp", "csv", ...
    c13_source: str = "unknown"
    solvent: Optional[str] = None
    frequency_mhz: Optional[float] = None
    nucleus: str = "1H"

    # ── MS ────────────────────────────────────────────────────────────
    # Structured MS fields (preferred over ms_text)
    ms_peaks: list[dict] = field(default_factory=list)       # [{mz, intensity}, ...]
    ms_precursor_mz: Optional[float] = None
    ms_collision_energy: Optional[float] = None
    ms_adduct: Optional[str] = None                          # "[M+H]+", etc.
    ms_charge: Optional[int] = None
    ms_source: str = "unknown"                               # "mgf", "mzml", "csv", "inline"
    # Legacy / fallback free-text field
    ms_text: Optional[str] = None
    exact_mass: Optional[float] = None
    ms_adducts: list[str] = field(default_factory=list)       # predicted adduct list

    # ── IR ────────────────────────────────────────────────────────────
    # Structured IR fields
    ir_wavenumbers: list[float] = field(default_factory=list)  # cm⁻¹, ascending
    ir_intensities: list[float] = field(default_factory=list)  # normalised 0–1 absorbance
    ir_peaks: list[dict] = field(default_factory=list)         # [{wavenumber, intensity}, ...]
    ir_y_units: str = "unknown"                                # "absorbance" | "transmittance"
    ir_source: str = "unknown"                                 # "ir_jcamp", "ir_image_extraction", "inline"
    ir_file: Optional[str] = None                              # original file path for SSIN

    # ── Context ───────────────────────────────────────────────────────
    reaction_type: Optional[str] = None
    route_step: Optional[str] = None
    notes: Optional[str] = None

    # ── Mode hint ─────────────────────────────────────────────────────
    mode: str = "verify"   # "verify" | "explore" | "predict" | "score" | "ir"

    # ----------------------------------------------------------------
    # Factory: build from CLI args
    # ----------------------------------------------------------------

    @classmethod
    def from_cli(
        cls,
        smiles: str | None = None,
        sm_smiles: str | None = None,
        h1: str | None = None,
        c13: str | None = None,
        ms: str | None = None,
        ms_adduct: str | None = None,
        ms_ce: float | None = None,
        ir: str | None = None,
        solvent: str | None = None,
        freq: float | None = None,
        formula: str | None = None,
        reaction: str | None = None,
        route_step: str | None = None,
        notes: str | None = None,
        mode: str = "verify",
    ) -> "SpectralInput":
        """
        Build SpectralInput from CLI argument strings.

        NMR/IR args can be either a file path or inline text.
        MS arg can be a file path (.mgf/.mzml/.csv) or inline text
        (e.g. "181 [M+H]+, 139, 77").
        """
        inp = cls(
            smiles=smiles.strip() if smiles else None,
            sm_smiles=sm_smiles.strip() if sm_smiles else None,
            molecular_formula=formula.strip() if formula else None,
            solvent=_normalize_solvent(solvent) if solvent else None,
            frequency_mhz=float(freq) if freq else None,
            ms_text=ms.strip() if ms and not Path(ms).exists() else None,
            reaction_type=reaction.strip() if reaction else None,
            route_step=route_step.strip() if route_step else None,
            notes=notes.strip() if notes else None,
            mode=mode,
        )

        if h1:
            inp.h1_peaks, inp.h1_source = _load_nmr_arg(h1, expected_nucleus="1H")

        if c13:
            inp.c13_peaks, inp.c13_source = _load_nmr_arg(c13, expected_nucleus="13C")

        if ir:
            (
                inp.ir_peaks,
                inp.ir_wavenumbers,
                inp.ir_intensities,
                inp.ir_source,
                inp.ir_file,
                inp.ir_y_units,
            ) = _load_ir_arg(ir)

        if ms:
            (
                inp.ms_peaks,
                inp.ms_precursor_mz,
                inp.ms_collision_energy,
                inp.ms_adduct,
                inp.ms_charge,
                inp.ms_source,
            ) = _load_ms_arg(ms, adduct_hint=ms_adduct, ce_hint=ms_ce)

        # CLI-provided adduct/CE override parsed values
        if ms_adduct and not inp.ms_adduct:
            inp.ms_adduct = ms_adduct.strip()
        if ms_ce is not None and inp.ms_collision_energy is None:
            inp.ms_collision_energy = float(ms_ce)

        return inp

    # ----------------------------------------------------------------
    # Validation
    # ----------------------------------------------------------------

    def has_any_data(self) -> bool:
        return bool(
            self.smiles or self.h1_peaks or self.c13_peaks or
            self.ms_peaks or self.ms_text or
            self.ir_peaks or self.ir_wavenumbers or
            self.molecular_formula
        )

    def summary(self) -> str:
        parts = []
        if self.smiles:
            parts.append(f"SMILES: {self.smiles}")
        if self.h1_peaks:
            parts.append(f"¹H NMR: {len(self.h1_peaks)} peaks ({self.h1_source})")
        if self.c13_peaks:
            parts.append(f"¹³C NMR: {len(self.c13_peaks)} peaks ({self.c13_source})")
        if self.ms_peaks:
            meta = f"{len(self.ms_peaks)} fragments"
            if self.ms_precursor_mz:
                meta += f", precursor {self.ms_precursor_mz:.4f}"
            parts.append(f"MS: {meta} ({self.ms_source})")
        elif self.ms_text:
            parts.append(f"MS: {self.ms_text[:60]}")
        if self.ir_peaks or self.ir_wavenumbers:
            n = len(self.ir_peaks) or len(self.ir_wavenumbers)
            parts.append(f"IR: {n} data points ({self.ir_source})")
        if not parts:
            return "No spectral data provided."
        return "  |  ".join(parts)

    # ----------------------------------------------------------------
    # Context string for system prompt injection
    # ----------------------------------------------------------------

    def to_context_string(self) -> str:
        """
        Render all spectral data as a structured markdown block for
        injection into the agent's system prompt / query context.
        """
        sections: list[str] = ["## Spectral Data\n"]

        # ── Structure ──────────────────────────────────────────────
        if self.smiles or self.sm_smiles or self.molecular_formula:
            sections.append("### Structure")
            if self.smiles:
                sections.append(f"- **Expected product SMILES:** `{self.smiles}`")
            if self.sm_smiles:
                sections.append(f"- **Starting material SMILES:** `{self.sm_smiles}`")
            if self.molecular_formula:
                sections.append(f"- **Molecular formula:** {self.molecular_formula}")
            if self.reaction_type:
                sections.append(f"- **Reaction type:** {self.reaction_type}")
            if self.route_step:
                sections.append(f"- **Route step:** {self.route_step}")
            sections.append("")

        # ── ¹H NMR ─────────────────────────────────────────────────
        if self.h1_peaks:
            header = "### ¹H NMR"
            meta = []
            if self.solvent:
                meta.append(self.solvent)
            if self.frequency_mhz:
                meta.append(f"{self.frequency_mhz:.0f} MHz")
            if meta:
                header += f" ({', '.join(meta)})"
            sections.append(header)
            sections.append(f"*Source: {self.h1_source}*")
            sections.append("")
            sections.append(_peaks_to_table(self.h1_peaks, nucleus="1H"))

            if self.solvent:
                flagged = _flag_solvent_peaks(self.h1_peaks, self.solvent, "1H")
                if flagged:
                    sections.append(
                        f"\n> **Note:** The following peaks are likely solvent "
                        f"residuals ({self.solvent}): "
                        + ", ".join(f"{p:.2f} ppm" for p in flagged)
                    )
            sections.append("")

        # ── ¹³C NMR ────────────────────────────────────────────────
        if self.c13_peaks:
            header = "### ¹³C NMR"
            meta = []
            if self.solvent:
                meta.append(self.solvent)
            if self.frequency_mhz:
                meta.append(f"{self.frequency_mhz / 4:.0f} MHz")  # approx C freq
            if meta:
                header += f" ({', '.join(meta)})"
            sections.append(header)
            sections.append(f"*Source: {self.c13_source}*")
            sections.append("")
            sections.append(_c13_peaks_to_table(self.c13_peaks))

            if self.solvent:
                flagged = _flag_solvent_peaks(self.c13_peaks, self.solvent, "13C")
                if flagged:
                    sections.append(
                        f"\n> **Note:** The following peaks are likely solvent "
                        f"residuals ({self.solvent}): "
                        + ", ".join(f"{p:.2f} ppm" for p in flagged)
                    )
            sections.append("")

        # ── MS ─────────────────────────────────────────────────────
        if self.ms_peaks:
            sections.append("### Mass Spectrometry (MS/MS)")
            meta_parts = [f"*Source: {self.ms_source}*"]
            if self.ms_precursor_mz:
                meta_parts.append(f"Precursor: {self.ms_precursor_mz:.4f} m/z")
            if self.ms_adduct:
                meta_parts.append(f"Adduct: {self.ms_adduct}")
            if self.ms_collision_energy:
                meta_parts.append(f"CE: {self.ms_collision_energy:.1f} eV")
            sections.append("  ".join(meta_parts))
            sections.append("")
            # Top 20 fragments by intensity
            top = sorted(self.ms_peaks, key=lambda x: x["intensity"], reverse=True)[:20]
            rows = ["| m/z | Rel. Intensity |", "|-----|----------------|"]
            for frag in sorted(top, key=lambda x: x["mz"]):
                rows.append(f"| {frag['mz']:.4f} | {frag['intensity']:.3f} |")
            sections.append("\n".join(rows))
            sections.append("")
        elif self.ms_text:
            sections.append("### Mass Spectrometry")
            sections.append(self.ms_text)
            sections.append("")

        # ── IR ─────────────────────────────────────────────────────
        if self.ir_peaks or self.ir_wavenumbers:
            sections.append("### IR Spectrum")
            meta_parts = [f"*Source: {self.ir_source}*"]
            if self.ir_y_units and self.ir_y_units != "unknown":
                meta_parts.append(f"Y-units: {self.ir_y_units}")
            if self.ir_file:
                meta_parts.append(f"File: `{self.ir_file}`")
            sections.append("  ".join(meta_parts))
            sections.append("")
            # Show top peaks by intensity
            if self.ir_peaks:
                top_ir = sorted(self.ir_peaks, key=lambda x: x["intensity"], reverse=True)[:15]
                rows = ["| Wavenumber (cm⁻¹) | Intensity |",
                        "|-------------------|-----------|"]
                for pk in sorted(top_ir, key=lambda x: x["wavenumber"], reverse=True):
                    rows.append(f"| {pk['wavenumber']:.1f} | {pk['intensity']:.3f} |")
                sections.append("\n".join(rows))
            elif self.ir_wavenumbers:
                sections.append(
                    "Spectrum range: "
                    f"{min(self.ir_wavenumbers):.0f}–{max(self.ir_wavenumbers):.0f} cm⁻¹  "
                    f"({len(self.ir_wavenumbers)} points)"
                )
            sections.append("")

        # ── Notes ──────────────────────────────────────────────────
        if self.notes:
            sections.append("### Chemist Notes")
            sections.append(self.notes)
            sections.append("")

        if len(sections) == 1:
            return ""  # Nothing to show

        return "\n".join(sections)

    def to_dict(self) -> dict:
        """
        Serialize to JSON-compatible dict suitable for direct injection
        into pipeline tool call arguments.
        """
        return {
            # Structure
            "smiles": self.smiles,
            "sm_smiles": self.sm_smiles,
            "molecular_formula": self.molecular_formula,
            # NMR
            "h1_peaks": self.h1_peaks,
            "c13_peaks": self.c13_peaks,
            "h1_source": self.h1_source,
            "c13_source": self.c13_source,
            "solvent": self.solvent,
            "frequency_mhz": self.frequency_mhz,
            # MS
            "ms_peaks": self.ms_peaks,
            "ms_precursor_mz": self.ms_precursor_mz,
            "ms_collision_energy": self.ms_collision_energy,
            "ms_adduct": self.ms_adduct,
            "ms_charge": self.ms_charge,
            "ms_source": self.ms_source,
            "ms_text": self.ms_text,
            # IR
            "ir_peaks": self.ir_peaks,
            "ir_wavenumbers": self.ir_wavenumbers,
            "ir_intensities": self.ir_intensities,
            "ir_y_units": self.ir_y_units,
            "ir_source": self.ir_source,
            "ir_file": self.ir_file,
            # Context
            "reaction_type": self.reaction_type,
            "mode": self.mode,
        }


# ── Private helpers ──────────────────────────────────────────────────────────

def _load_nmr_arg(arg: str, expected_nucleus: str) -> tuple[list[dict], str]:
    """
    Determine if arg is a file path or inline text and parse accordingly.
    Returns (peaks, source_label).
    """
    p = Path(arg)
    if p.exists():
        from speqtro.input.autodetect import parse_spectrum
        spectrum = parse_spectrum(p)
        return spectrum.get("peaks", []), spectrum.get("source_format", "file")

    # Inline text
    from speqtro.input.text_peaks import parse_inline_peaks, parse_c13_text
    if expected_nucleus == "13C":
        peaks = parse_c13_text(arg)
    else:
        peaks = parse_inline_peaks(arg)
    return peaks, "inline"


def _load_ir_arg(
    arg: str,
) -> tuple[list[dict], list[float], list[float], str, Optional[str], str]:
    """
    Load IR data from file path or inline comma-separated wavenumbers.

    Returns:
        (ir_peaks, ir_wavenumbers, ir_intensities, source, ir_file_path, y_units)
    """
    p = Path(arg)
    if p.exists():
        from speqtro.input.autodetect import parse_ir
        try:
            result = parse_ir(p)
            return (
                result.get("peaks", []),
                result.get("wavenumbers", []),
                result.get("intensities", []),
                result.get("source_format", "file"),
                str(p),
                result.get("y_units", "unknown"),
            )
        except Exception:
            pass

    # Inline: "1715, 1600, 3300"
    nums = re.findall(r"\d+\.?\d*", arg)
    peaks_wn = [float(n) for n in nums if 400 <= float(n) <= 4000]
    # Without intensities: build minimal peak list
    peaks = [{"wavenumber": wn, "intensity": 1.0} for wn in sorted(peaks_wn)]
    return peaks, sorted(peaks_wn), [1.0] * len(peaks_wn), "inline", None, "unknown"


def _load_ms_arg(
    arg: str,
    adduct_hint: Optional[str] = None,
    ce_hint: Optional[float] = None,
) -> tuple[list[dict], Optional[float], Optional[float], Optional[str], Optional[int], str]:
    """
    Load MS/MS data from a file path or inline fragment text.

    Returns:
        (peaks, precursor_mz, collision_energy, adduct, charge, source_format)
    """
    p = Path(arg)
    if p.exists():
        from speqtro.input.autodetect import parse_ms
        try:
            result = parse_ms(p)
            return (
                result.get("peaks", []),
                result.get("precursor_mz"),
                result.get("collision_energy", ce_hint),
                result.get("adduct") or adduct_hint,
                result.get("charge"),
                result.get("source_format", "file"),
            )
        except Exception:
            pass

    # Inline: "181 [M+H]+, 139, 77" or "181.0 (100%), 139.1 (45%)"
    from speqtro.input.ms_file import parse_ms_inline
    try:
        result = parse_ms_inline(arg)
        return (
            result.get("peaks", []),
            result.get("precursor_mz", None),
            result.get("collision_energy") or ce_hint,
            result.get("adduct") or adduct_hint,
            result.get("charge"),
            result.get("source_format", "inline"),
        )
    except Exception:
        return [], None, ce_hint, adduct_hint, None, "unknown"


def _normalize_solvent(raw: str) -> str:
    """Normalize common solvent name variants."""
    aliases = {
        "cdcl3": "CDCl3", "chloroform": "CDCl3", "chloroform-d": "CDCl3",
        "dmso": "DMSO-d6", "dmso-d6": "DMSO-d6",
        "d2o": "D2O", "water": "D2O",
        "cd3od": "CD3OD", "methanol-d4": "CD3OD", "meod": "CD3OD",
        "acetone": "Acetone-d6", "acetone-d6": "Acetone-d6",
        "c6d6": "C6D6", "benzene-d6": "C6D6",
        "thf-d8": "THF-d8",
        "cd2cl2": "CD2Cl2", "dcm-d2": "CD2Cl2",
        "mecn-d3": "MeCN-d3", "acetonitrile-d3": "MeCN-d3",
    }
    return aliases.get(raw.strip().lower(), raw.strip())


def _peaks_to_table(peaks: list[dict], nucleus: str = "1H") -> str:
    """Render ¹H peaks as a markdown table."""
    rows = ["| δ (ppm) | Integral | Multiplicity | J (Hz) |",
            "|---------|----------|--------------|--------|"]
    for p in peaks:
        shift = f"{p['shift']:.2f}"
        integral = f"{p['integral']:.0f}H" if p.get("integral") else "—"
        mult = p.get("multiplicity") or "—"
        j_vals = p.get("coupling_hz")
        j_str = ", ".join(f"{j:.1f}" for j in j_vals) if j_vals else "—"
        rows.append(f"| {shift} | {integral} | {mult} | {j_str} |")
    return "\n".join(rows)


def _c13_peaks_to_table(peaks: list[dict]) -> str:
    """Render ¹³C peaks as a markdown table."""
    rows = ["| δ (ppm) | Notes |", "|---------|-------|"]
    for p in peaks:
        shift = f"{p['shift']:.1f}"
        notes = p.get("multiplicity") or "—"
        rows.append(f"| {shift} | {notes} |")
    return "\n".join(rows)


def _flag_solvent_peaks(
    peaks: list[dict],
    solvent: str,
    nucleus: str,
    tolerance_ppm: float = 0.15,
) -> list[float]:
    """Return shift values that match known solvent residual peaks."""
    try:
        data_file = Path(__file__).resolve().parents[1] / "data" / "solvent_peaks.json"
        if not data_file.exists():
            return []
        with open(data_file, encoding="utf-8") as f:
            solvent_data = json.load(f)
        ref_peaks = [
            p["shift"] for p in solvent_data.get(solvent, [])
            if p.get("nucleus") == nucleus
        ]
    except Exception:
        return []

    flagged = []
    for peak in peaks:
        for ref in ref_peaks:
            if abs(peak["shift"] - ref) <= tolerance_ppm:
                flagged.append(peak["shift"])
                break
    return flagged

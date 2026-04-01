"""NMR prediction tools using empirical rules and RDKit."""

from speqtro.tools import registry


# ---------------------------------------------------------------------------
# Heuristic ¹H shift tables (atom environment → approx ppm)
# ---------------------------------------------------------------------------

def _classify_h1_environment(atom, mol) -> tuple[str, float, float]:
    """Return (description, low_ppm, high_ppm) for a hydrogen-bearing atom."""
    from rdkit.Chem import rdMolDescriptors
    sym = atom.GetSymbol()
    neighbors = [n.GetSymbol() for n in atom.GetNeighbors()]
    ring_info = mol.GetRingInfo()
    in_ring = ring_info.NumAtomRings(atom.GetIdx()) > 0
    in_aromatic = atom.GetIsAromatic()

    if sym == "C":
        # Check for adjacent electronegative atoms
        has_hetero = any(n in ("O", "N", "S", "F", "Cl", "Br") for n in neighbors)
        # Check for carbonyl neighbors
        has_carbonyl = False
        for nb in atom.GetNeighbors():
            if nb.GetSymbol() == "C":
                for nb2 in nb.GetNeighbors():
                    if nb2.GetSymbol() == "O" and mol.GetBondBetweenAtoms(nb.GetIdx(), nb2.GetIdx()).GetBondTypeAsDouble() == 2.0:
                        has_carbonyl = True

        if in_aromatic:
            return ("aromatic C–H", 6.5, 8.5)
        # Check if alkene
        for bond in atom.GetBonds():
            if bond.GetBondTypeAsDouble() == 2.0:
                other = bond.GetOtherAtom(atom)
                if other.GetSymbol() == "C":
                    return ("vinyl C–H", 4.5, 6.5)
        if has_hetero:
            if "O" in neighbors:
                return ("C–H adjacent to O (OCH)", 3.2, 4.5)
            if "N" in neighbors:
                return ("C–H adjacent to N (NCH)", 2.2, 3.5)
            if any(h in neighbors for h in ("F", "Cl", "Br")):
                return ("C–H adjacent to halogen", 3.0, 4.5)
        if has_carbonyl:
            return ("C–H adjacent to C=O", 2.0, 2.8)
        return ("alkyl C–H", 0.5, 2.0)
    elif sym == "O":
        return ("O–H (alcohol or phenol)", 1.0, 5.0)
    elif sym == "N":
        return ("N–H", 1.0, 5.0)
    return ("other", 0.5, 10.0)


def _midpoint_shift(low: float, high: float) -> float:
    return round((low + high) / 2, 2)


@registry.register(
    name="nmr.predict_h1_shifts",
    description=(
        "Predict 1H NMR chemical shifts for all hydrogen-bearing atoms in a molecule "
        "from its SMILES string. Uses empirical rules based on atom environment "
        "(aromatic, vinyl, aliphatic, adjacent to heteroatoms). "
        "Returns estimated chemical shift ranges in ppm."
    ),
    category="nmr",
    parameters={
        "smiles": "SMILES string of the molecule (e.g. 'CC(=O)O' for acetic acid)",
    },
)
def predict_h1_shifts(smiles: str = "", **kwargs) -> dict:
    """Predict ¹H NMR chemical shifts from SMILES using empirical rules."""
    if not smiles:
        return {"summary": "Error: no SMILES provided"}

    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, rdMolDescriptors
    except ImportError:
        return {"summary": "Error: RDKit not installed. Run: pip install rdkit"}

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"summary": f"Error: could not parse SMILES '{smiles}'"}

    mol = Chem.AddHs(mol)

    predictions = []
    heavy_atom_seen: dict[int, int] = {}  # idx → group counter for deduplication

    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "H":
            continue
        # Get the heavy atom this H is attached to
        parent_atoms = [n for n in atom.GetNeighbors()]
        if not parent_atoms:
            continue
        parent = parent_atoms[0]
        parent_idx = parent.GetIdx()

        # Count how many H on this heavy atom (for multiplicity grouping)
        h_on_parent = sum(1 for n in parent.GetNeighbors() if n.GetSymbol() == "H")

        env_desc, low_ppm, high_ppm = _classify_h1_environment(parent, mol)
        est = _midpoint_shift(low_ppm, high_ppm)

        # Group equivalent Hs on same heavy atom
        group_key = parent_idx
        if group_key not in heavy_atom_seen:
            heavy_atom_seen[group_key] = 1
            predictions.append({
                "atom_idx": parent_idx,
                "atom_symbol": parent.GetSymbol(),
                "num_H": h_on_parent,
                "environment": env_desc,
                "estimated_shift_ppm": est,
                "range_ppm": f"{low_ppm:.1f}–{high_ppm:.1f}",
            })

    if not predictions:
        return {"summary": f"No hydrogen-bearing atoms found in '{smiles}'"}

    # Sort by estimated shift
    predictions.sort(key=lambda x: x["estimated_shift_ppm"])

    lines = [f"¹H NMR prediction for {smiles}:", ""]
    for p in predictions:
        lines.append(
            f"  {p['num_H']}H @ ~{p['estimated_shift_ppm']} ppm "
            f"(range: {p['range_ppm']}) — {p['environment']}"
        )

    return {
        "summary": "\n".join(lines),
        "smiles": smiles,
        "predictions": predictions,
        "note": "Shifts estimated from empirical tables. Use actual NMR instrument for precise values.",
    }


# ---------------------------------------------------------------------------
# ¹³C shift heuristics
# ---------------------------------------------------------------------------

def _classify_c13_environment(atom, mol) -> tuple[str, float, float]:
    """Return (description, low_ppm, high_ppm) for a carbon atom."""
    neighbors = [n.GetSymbol() for n in atom.GetNeighbors()]
    in_aromatic = atom.GetIsAromatic()

    # Check for double bonds
    for bond in atom.GetBonds():
        bt = bond.GetBondTypeAsDouble()
        other = bond.GetOtherAtom(atom)
        if bt == 2.0 and other.GetSymbol() == "O":
            # Carbonyl
            for nb in atom.GetNeighbors():
                if nb.GetSymbol() == "O" and any(
                    nb2.GetSymbol() == "O" or nb2.GetSymbol() == "N"
                    for nb2 in atom.GetNeighbors() if nb2 != nb
                ):
                    return ("ester/amide/acid carbonyl C", 155, 180)
            return ("ketone/aldehyde carbonyl C", 190, 220)
        if bt == 2.0 and other.GetSymbol() == "C":
            if in_aromatic:
                return ("aromatic C", 110, 160)
            return ("alkene C=C", 100, 150)

    if in_aromatic:
        return ("aromatic C", 110, 160)

    has_O = "O" in neighbors
    has_N = "N" in neighbors
    has_X = any(h in neighbors for h in ("F", "Cl", "Br", "I"))

    if has_O:
        return ("C–O (ether/alcohol/ester)", 55, 90)
    if has_N:
        return ("C–N", 40, 70)
    if has_X:
        return ("C–halogen", 25, 65)

    return ("alkyl C", 10, 50)


@registry.register(
    name="nmr.predict_c13_shifts",
    description=(
        "Predict 13C NMR chemical shifts for all carbon atoms in a molecule from its SMILES. "
        "Uses empirical additive rules based on carbon environment "
        "(aromatic, carbonyl, C-O, alkyl). Returns estimated shift ranges in ppm."
    ),
    category="nmr",
    parameters={
        "smiles": "SMILES string of the molecule (e.g. 'CC(=O)O' for acetic acid)",
    },
)
def predict_c13_shifts(smiles: str = "", **kwargs) -> dict:
    """Predict ¹³C NMR chemical shifts from SMILES using empirical rules."""
    if not smiles:
        return {"summary": "Error: no SMILES provided"}

    try:
        from rdkit import Chem
    except ImportError:
        return {"summary": "Error: RDKit not installed. Run: pip install rdkit"}

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"summary": f"Error: could not parse SMILES '{smiles}'"}

    predictions = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "C":
            continue
        env_desc, low_ppm, high_ppm = _classify_c13_environment(atom, mol)
        est = _midpoint_shift(low_ppm, high_ppm)
        num_H = atom.GetTotalNumHs()
        predictions.append({
            "atom_idx": atom.GetIdx(),
            "num_H": num_H,
            "environment": env_desc,
            "estimated_shift_ppm": est,
            "range_ppm": f"{low_ppm:.0f}–{high_ppm:.0f}",
        })

    if not predictions:
        return {"summary": f"No carbon atoms found in '{smiles}'"}

    predictions.sort(key=lambda x: x["estimated_shift_ppm"])

    lines = [f"¹³C NMR prediction for {smiles}:", ""]
    for p in predictions:
        mult = {0: "C", 1: "CH", 2: "CH₂", 3: "CH₃"}.get(p["num_H"], f"CH{p['num_H']}")
        lines.append(
            f"  C{p['atom_idx']} ({mult}) @ ~{p['estimated_shift_ppm']} ppm "
            f"(range: {p['range_ppm']}) — {p['environment']}"
        )

    return {
        "summary": "\n".join(lines),
        "smiles": smiles,
        "predictions": predictions,
        "note": "Shifts estimated from empirical tables. Actual values vary with solvent, temperature, and substitution.",
    }


# ---------------------------------------------------------------------------
# JCAMP-DX file parser
# ---------------------------------------------------------------------------

def _parse_jcamp_header(lines: list[str]) -> dict:
    """Extract key-value pairs from JCAMP-DX header lines."""
    import re
    header = {}
    for line in lines:
        m = re.match(r"##([^=]+)=\s*(.*)", line.strip())
        if m:
            key = m.group(1).strip().upper()
            val = m.group(2).strip()
            header[key] = val
    return header


def _parse_jcamp_xydata(lines: list[str], n_points: int, first_x: float, last_x: float,
                         x_factor: float, y_factor: float) -> tuple[list[float], list[float]]:
    """Parse (X++(Y..Y)) encoded JCAMP-DX spectral data."""
    import re
    x_vals: list[float] = []
    y_vals: list[float] = []
    delta_x = (last_x - first_x) / max(n_points - 1, 1)

    for line in lines:
        line = line.strip()
        if not line or line.startswith("##"):
            break
        tokens = re.split(r"[\s,]+", line)
        if not tokens:
            continue
        try:
            x_start = float(tokens[0]) * x_factor
        except ValueError:
            continue
        for i, tok in enumerate(tokens[1:]):
            try:
                y = float(tok) * y_factor
            except ValueError:
                continue
            x = x_start + i * abs(delta_x)
            x_vals.append(round(x, 5))
            y_vals.append(y)

    return x_vals, y_vals


def _parse_jcamp_xypoints(lines: list[str], x_factor: float, y_factor: float) -> tuple[list[float], list[float]]:
    """Parse (XY..XY) encoded JCAMP-DX spectral data."""
    import re
    x_vals: list[float] = []
    y_vals: list[float] = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("##"):
            break
        tokens = re.split(r"[\s,]+", line)
        for i in range(0, len(tokens) - 1, 2):
            try:
                x = float(tokens[i]) * x_factor
                y = float(tokens[i + 1]) * y_factor
                x_vals.append(round(x, 5))
                y_vals.append(y)
            except (ValueError, IndexError):
                continue
    return x_vals, y_vals


def _find_peaks_simple(x_vals: list[float], y_vals: list[float],
                        min_height_frac: float = 0.02) -> list[dict]:
    """Find local maxima above threshold. Returns list of {ppm, intensity}."""
    if len(y_vals) < 3:
        return []
    max_y = max(y_vals)
    if max_y <= 0:
        return []
    threshold = max_y * min_height_frac
    peaks = []
    for i in range(1, len(y_vals) - 1):
        if y_vals[i] > threshold and y_vals[i] >= y_vals[i - 1] and y_vals[i] >= y_vals[i + 1]:
            # Refine to local maximum within window
            if i > 0 and i < len(y_vals) - 1 and y_vals[i] > y_vals[i - 1] and y_vals[i] > y_vals[i + 1]:
                peaks.append({"ppm": round(x_vals[i], 3), "relative_intensity": round(y_vals[i] / max_y, 3)})
    # Merge peaks within 0.02 ppm (instrument resolution)
    merged = []
    for p in peaks:
        if merged and abs(p["ppm"] - merged[-1]["ppm"]) < 0.02:
            if p["relative_intensity"] > merged[-1]["relative_intensity"]:
                merged[-1] = p
        else:
            merged.append(p)
    return merged


@registry.register(
    name="nmr.parse_jcamp",
    description=(
        "Parse a JCAMP-DX NMR spectrum file (.jdx, .dx) and extract peaks, nucleus type, "
        "frequency, and metadata. Accepts a file path or raw JCAMP-DX text content. "
        "Returns detected peak list (ppm), nucleus (1H/13C), spectrometer frequency, and instrument metadata."
    ),
    category="nmr",
    parameters={
        "file_path": "Absolute path to a .jdx or .dx JCAMP-DX file",
        "jcamp_text": "Raw JCAMP-DX text content (alternative to file_path)",
        "min_height_frac": "Minimum peak height as fraction of tallest peak (default 0.02)",
    },
)
def parse_jcamp(file_path: str = "", jcamp_text: str = "", min_height_frac: float = 0.02, **kwargs) -> dict:
    """Parse JCAMP-DX NMR spectrum file and return peaks + metadata."""
    if not file_path and not jcamp_text:
        return {"summary": "Error: provide file_path or jcamp_text"}

    if file_path and not jcamp_text:
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
                jcamp_text = fh.read()
        except OSError as e:
            return {"summary": f"Error reading file '{file_path}': {e}"}

    lines = jcamp_text.splitlines()
    header = _parse_jcamp_header(lines)

    # Extract key parameters
    def _float(key: str, default: float = 1.0) -> float:
        try:
            return float(header.get(key, default))
        except ValueError:
            return default

    def _int(key: str, default: int = 0) -> int:
        try:
            return int(float(header.get(key, default)))
        except ValueError:
            return default

    x_units = header.get("XUNITS", "PPM").upper()
    y_units = header.get("YUNITS", "ARBITRARY UNITS")
    first_x = _float("FIRSTX")
    last_x = _float("LASTX")
    x_factor = _float("XFACTOR")
    y_factor = _float("YFACTOR")
    n_points = _int("NPOINTS")
    title = header.get("TITLE", "Unknown")
    nucleus = header.get(".OBSERVE NUCLEUS", header.get("$NUCLEUS", "Unknown"))
    obs_freq = header.get(".OBSERVE FREQUENCY", header.get("$OBSERVEFREQUENCY", "Unknown"))
    data_type = header.get("DATA TYPE", header.get("DATATYPE", "NMR SPECTRUM"))

    # Find data section
    data_start = 0
    data_format = "xydata"
    for i, line in enumerate(lines):
        stripped = line.strip().upper()
        if stripped.startswith("##XYDATA="):
            data_start = i + 1
            data_format = "xydata"
            break
        if stripped.startswith("##XYPOINTS="):
            data_start = i + 1
            data_format = "xypoints"
            break

    if data_start == 0:
        return {
            "summary": f"Error: could not find XYDATA or XYPOINTS section in JCAMP file",
            "header": header,
        }

    data_lines = lines[data_start:]

    if data_format == "xydata":
        x_vals, y_vals = _parse_jcamp_xydata(data_lines, n_points, first_x, last_x, x_factor, y_factor)
    else:
        x_vals, y_vals = _parse_jcamp_xypoints(data_lines, x_factor, y_factor)

    if not x_vals:
        return {"summary": "Error: no spectral data could be parsed", "header": header}

    peaks = _find_peaks_simple(x_vals, y_vals, min_height_frac=min_height_frac)

    # Sort peaks by ppm descending (NMR convention)
    peaks.sort(key=lambda p: p["ppm"], reverse=True)

    ppm_range = (round(min(x_vals), 2), round(max(x_vals), 2))
    peak_ppms = [p["ppm"] for p in peaks]

    summary_lines = [
        f"JCAMP-DX NMR file: {title}",
        f"  Nucleus: {nucleus} | Frequency: {obs_freq} MHz | Data type: {data_type}",
        f"  X range: {ppm_range[0]}–{ppm_range[1]} {x_units} | {len(x_vals)} data points",
        f"  {len(peaks)} peaks detected (threshold: {min_height_frac:.0%} of base peak):",
    ]
    for p in peaks[:20]:
        summary_lines.append(f"    {p['ppm']:7.3f} ppm  (rel. intensity: {p['relative_intensity']:.3f})")
    if len(peaks) > 20:
        summary_lines.append(f"    ... and {len(peaks) - 20} more peaks")

    return {
        "summary": "\n".join(summary_lines),
        "title": title,
        "nucleus": nucleus,
        "observe_frequency_mhz": obs_freq,
        "data_type": data_type,
        "x_units": x_units,
        "ppm_range": ppm_range,
        "n_data_points": len(x_vals),
        "peaks": peaks,
        "peak_ppms": peak_ppms,
        "header_fields": header,
    }


# ---------------------------------------------------------------------------
# ¹H multiplicity pattern finder
# ---------------------------------------------------------------------------

_REGION_ASSIGNMENTS = [
    (9.5, 12.0, "aldehyde / carboxylic acid O–H"),
    (8.0, 9.5,  "deshielded aromatic / heteroaromatic H"),
    (6.5, 8.0,  "aromatic H"),
    (4.5, 6.5,  "vinyl / vinylogous H"),
    (3.2, 4.5,  "H on C–O (ether, ester, alcohol α-CH)"),
    (2.0, 3.2,  "H on C adjacent to C=O or C–N"),
    (0.5, 2.0,  "alkyl H (CH₃, CH₂, CH)"),
    (-0.5, 0.5, "silyl or highly shielded H"),
]


def _assign_region(ppm: float) -> str:
    for low, high, label in _REGION_ASSIGNMENTS:
        if low <= ppm <= high:
            return label
    return "unknown region"


def _detect_multiplicity(cluster: list[float]) -> str:
    """Guess multiplicity from a cluster of ppm values."""
    n = len(cluster)
    if n == 1:
        return "singlet (s)"
    if n == 2:
        return "doublet (d)"
    if n == 3:
        return "triplet (t)"
    if n == 4:
        return "quartet (q)"
    if n == 5:
        return "quintet"
    if n == 6:
        return "sextet"
    if n <= 8:
        return f"multiplet (m, {n} lines)"
    return "complex multiplet (m)"


def _cluster_peaks(ppms: list[float], max_gap: float = 0.25) -> list[list[float]]:
    """Group nearby peaks into multiplet clusters."""
    if not ppms:
        return []
    sorted_ppms = sorted(ppms)
    clusters: list[list[float]] = [[sorted_ppms[0]]]
    for p in sorted_ppms[1:]:
        if p - clusters[-1][-1] <= max_gap:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    return clusters


@registry.register(
    name="nmr.find_h1_pattern",
    description=(
        "Identify chemical shift regions, multiplicity patterns, and likely functional groups "
        "from a list of observed 1H NMR peak positions (ppm). "
        "Accepts comma-separated ppm values and optional integration values. "
        "Returns grouped multiplets, region assignments, and structural interpretations."
    ),
    category="nmr",
    parameters={
        "peaks_ppm": "Comma-separated observed 1H NMR peak positions in ppm (e.g. '7.26, 7.18, 3.82, 2.31')",
        "integrations": "Optional comma-separated relative integration values matching peaks_ppm order",
        "coupling_constant_hz": "Expected J coupling constant in Hz for multiplet detection (default 8.0)",
        "spectrometer_mhz": "Spectrometer frequency in MHz, used to convert Hz to ppm spacing (default 400)",
    },
)
def find_h1_pattern(
    peaks_ppm: str = "",
    integrations: str = "",
    coupling_constant_hz: float = 8.0,
    spectrometer_mhz: float = 400.0,
    **kwargs,
) -> dict:
    """Analyze observed ¹H NMR peak positions and identify patterns and functional groups."""
    if not peaks_ppm:
        return {"summary": "Error: no peaks_ppm provided"}

    try:
        ppms = [float(x.strip()) for x in peaks_ppm.split(",") if x.strip()]
    except ValueError as e:
        return {"summary": f"Error parsing peaks_ppm: {e}"}

    if not ppms:
        return {"summary": "Error: no valid ppm values found"}

    ints: list[float] = []
    if integrations:
        try:
            ints = [float(x.strip()) for x in integrations.split(",") if x.strip()]
        except ValueError:
            ints = []

    # Convert J coupling Hz → ppm gap at given field
    j_ppm_gap = coupling_constant_hz / spectrometer_mhz

    # Cluster peaks into multiplets
    clusters = _cluster_peaks(ppms, max_gap=j_ppm_gap * 6)  # allow up to 6*J spacing

    results = []
    for cluster in clusters:
        center = round(sum(cluster) / len(cluster), 3)
        region = _assign_region(center)
        mult = _detect_multiplicity(cluster)

        # Estimate J coupling from line spacing if multiplet
        j_est = None
        if len(cluster) >= 2:
            spacings = [abs(cluster[i + 1] - cluster[i]) * spectrometer_mhz
                        for i in range(len(cluster) - 1)]
            j_est = round(sum(spacings) / len(spacings), 1)

        entry: dict = {
            "center_ppm": center,
            "lines_ppm": [round(p, 3) for p in sorted(cluster, reverse=True)],
            "multiplicity": mult,
            "region": region,
        }
        if j_est is not None:
            entry["estimated_J_Hz"] = j_est

        # Match integration to cluster
        if ints:
            # Find closest integration value by index proximity
            cluster_center_idx = min(range(len(ppms)), key=lambda i: abs(ppms[i] - center))
            if cluster_center_idx < len(ints):
                entry["integration"] = ints[cluster_center_idx]

        results.append(entry)

    # Sort by ppm descending
    results.sort(key=lambda x: x["center_ppm"], reverse=True)

    # Build structural interpretation
    interpretations = []
    for r in results:
        c = r["center_ppm"]
        m = r["multiplicity"]
        reg = r["region"]
        j_str = f", J ≈ {r['estimated_J_Hz']} Hz" if "estimated_J_Hz" in r else ""
        int_str = f", {r['integration']:.1f}H" if "integration" in r else ""
        interpretations.append(f"  δ {c:.2f} ppm ({m}{j_str}{int_str}) — {reg}")

    summary_lines = [
        f"¹H NMR pattern analysis ({len(ppms)} peaks → {len(results)} multiplet groups):",
        "",
    ] + interpretations

    # Add structural hints
    centers = [r["center_ppm"] for r in results]
    hints = []
    if any(9.5 <= c <= 10.5 for c in centers):
        hints.append("Aldehyde C–H likely present (~9.5–10.5 ppm singlet)")
    if any(10.5 <= c <= 13.0 for c in centers):
        hints.append("Carboxylic acid O–H or strongly H-bonded O–H likely (~10–13 ppm)")
    if any(6.5 <= c <= 8.5 for c in centers):
        hints.append("Aromatic ring likely present (6.5–8.5 ppm)")
    if any(4.5 <= c <= 6.5 for c in centers):
        hints.append("Alkene or vinyl H likely present (4.5–6.5 ppm)")
    if any(3.2 <= c <= 4.5 for c in centers):
        hints.append("C–O or N–CH likely present (3.2–4.5 ppm)")

    if hints:
        summary_lines += ["", "Structural hints:"] + [f"  • {h}" for h in hints]

    return {
        "summary": "\n".join(summary_lines),
        "input_peaks_ppm": ppms,
        "multiplet_groups": results,
        "structural_hints": hints,
        "n_groups": len(results),
    }

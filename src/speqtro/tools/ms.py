"""Mass spectrometry tools."""

from speqtro.tools import registry

# Monoisotopic masses of common elements
_ISOTOPE_MASSES = {
    "H": 1.0078250319, "C": 12.0000000, "N": 14.0030740052,
    "O": 15.9949146221, "F": 18.99840322, "P": 30.97376151,
    "S": 31.97207069, "Cl": 34.96885271, "Br": 78.9183376,
    "I": 126.904468, "Si": 27.9769265, "Na": 22.9897693,
}

# Common adducts: name → mass shift
_ADDUCTS = {
    "[M+H]+":  1.007276,
    "[M+Na]+": 22.989218,
    "[M+K]+":  38.963158,
    "[M+NH4]+": 18.034164,
    "[M-H]-":  -1.007276,
    "[M+Cl]-": 34.969402,
    "[M+2H]2+": 1.007276,   # per charge — will be labelled separately
}


def _parse_formula(formula: str) -> dict[str, int]:
    """Parse a molecular formula string into element counts."""
    import re
    tokens = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    counts: dict[str, int] = {}
    for element, count_str in tokens:
        if not element:
            continue
        count = int(count_str) if count_str else 1
        counts[element] = counts.get(element, 0) + count
    return counts


def _monoisotopic_mass(counts: dict[str, int]) -> float:
    mass = 0.0
    for element, count in counts.items():
        iso_mass = _ISOTOPE_MASSES.get(element)
        if iso_mass is None:
            raise ValueError(f"Unknown element: {element}")
        mass += iso_mass * count
    return mass


@registry.register(
    name="ms.calc_exact_mass",
    description=(
        "Calculate the monoisotopic exact mass from a molecular formula and list common "
        "MS adducts ([M+H]+, [M+Na]+, [M+K]+, [M-H]-, etc.)."
    ),
    category="ms",
    parameters={
        "formula": "Molecular formula string, e.g. 'C6H12O6' or 'C10H15NO2'",
    },
)
def calc_exact_mass(formula: str = "", **kwargs) -> dict:
    """Calculate monoisotopic mass and adduct m/z values from molecular formula."""
    if not formula:
        return {"summary": "Error: no formula provided"}

    try:
        counts = _parse_formula(formula.strip())
    except Exception as e:
        return {"summary": f"Error parsing formula '{formula}': {e}"}

    try:
        mono_mass = _monoisotopic_mass(counts)
    except ValueError as e:
        return {"summary": str(e)}

    adducts = {}
    for adduct_name, shift in _ADDUCTS.items():
        if "2+" in adduct_name:
            mz = (mono_mass + 2 * 1.007276) / 2
            adducts[adduct_name] = round(mz, 4)
        else:
            adducts[adduct_name] = round(mono_mass + shift, 4)

    # Neutral losses
    neutral_losses = {
        "M - H2O": round(mono_mass - 18.010565, 4),
        "M - CO2": round(mono_mass - 43.98983, 4),
        "M - NH3": round(mono_mass - 17.026549, 4),
    }

    summary_lines = [f"Formula: {formula} | Monoisotopic mass: {mono_mass:.4f} Da"]
    summary_lines.append("Common adducts (m/z):")
    for name, mz in adducts.items():
        summary_lines.append(f"  {name}: {mz:.4f}")
    summary_lines.append("Neutral losses:")
    for name, mz in neutral_losses.items():
        summary_lines.append(f"  {name}: {mz:.4f}")

    return {
        "summary": "\n".join(summary_lines),
        "formula": formula,
        "monoisotopic_mass": round(mono_mass, 6),
        "adducts": adducts,
        "neutral_losses": neutral_losses,
        "atom_counts": counts,
    }


# ---------------------------------------------------------------------------
# MS fragmentation prediction
# ---------------------------------------------------------------------------

# SMARTS patterns for functional group detection → fragmentation rules
# Each rule: (smarts, description, neutral_loss_formula, neutral_loss_mass)
_FRAG_RULES = [
    # Neutral losses
    ("[OH]",                   "alcohol/phenol O–H",         "H2O",  18.010565),
    ("[CX3](=O)[OX2H1]",      "carboxylic acid",             "H2O",  18.010565),
    ("[CX3](=O)[OX2H1]",      "carboxylic acid decarboxyl.", "CO2",  43.989830),
    ("[NX3;H2,H1]",           "primary/secondary amine",     "NH3",  17.026549),
    ("[CX3](=O)",             "carbonyl (CO loss)",          "CO",   27.994915),
    ("[F]",                   "C–F bond",                    "HF",   20.006229),
    ("[Cl]",                  "C–Cl bond",                   "HCl",  35.976678),
    ("[Br]",                  "C–Br bond",                   "HBr",  79.926161),
    ("[SX2]",                 "thioether/thiol",             "H2S",  33.987721),
    ("[CX4][NX3]",            "N-alkyl amine (CH2=NR)",      "CH2=NR (~28 Da)", 28.018750),
    ("[CX3](=O)[OX2][CX4]",  "ester (acyl loss + OMe etc)", "OCH3 (~31 Da)",   31.018389),
]

# Characteristic fragment ions (m/z values, formula-independent)
_CHAR_FRAGMENTS = [
    (91.0548,  "C7H7+",  "tropylium — benzyl/toluene system"),
    (77.0386,  "C6H5+",  "phenyl cation — monosubstituted benzene"),
    (105.0335, "C7H5O+", "benzoyl cation — ArC(=O)+"),
    (43.0184,  "C2H3O+", "acetyl cation CH3CO+ — methyl ketone"),
    (57.0340,  "C3H5O+", "propanoyl/t-butyl cation"),
    (29.0022,  "CHO+",   "formyl cation — aldehyde"),
]


@registry.register(
    name="ms.fragment_predict",
    description=(
        "Predict common MS/MS fragmentation ions and neutral losses for a molecule given its SMILES. "
        "Identifies functional-group-driven neutral losses (H2O, NH3, CO2, CO, HX) and "
        "characteristic fragment ions (tropylium, phenyl, acylium). "
        "Returns predicted fragment m/z values for [M+H]+ precursor and key daughter ions."
    ),
    category="ms",
    parameters={
        "smiles": "SMILES string of the molecule",
        "precursor_adduct": "Precursor ion adduct: '[M+H]+' (default), '[M-H]-', '[M+Na]+'",
    },
)
def fragment_predict(smiles: str = "", precursor_adduct: str = "[M+H]+", **kwargs) -> dict:
    """Predict MS/MS fragmentation ions from SMILES."""
    if not smiles:
        return {"summary": "Error: no SMILES provided"}

    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors
    except ImportError:
        return {"summary": "Error: RDKit not installed. Run: pip install rdkit"}

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"summary": f"Error: could not parse SMILES '{smiles}'"}

    formula = rdMolDescriptors.CalcMolFormula(mol)
    counts = _parse_formula(formula)
    try:
        mono_mass = _monoisotopic_mass(counts)
    except ValueError as e:
        return {"summary": f"Error computing mass: {e}"}

    # Precursor m/z
    adduct_shift = _ADDUCTS.get(precursor_adduct, 1.007276)
    if "[M+2H]2+" in precursor_adduct:
        precursor_mz = round((mono_mass + 2 * 1.007276) / 2, 4)
    else:
        precursor_mz = round(mono_mass + adduct_shift, 4)

    # Detect functional groups and collect neutral losses
    predicted_losses: list[dict] = []
    seen_losses: set[str] = set()

    for smarts, fgroup, loss_label, loss_mass in _FRAG_RULES:
        patt = Chem.MolFromSmarts(smarts)
        if patt and mol.HasSubstructMatch(patt):
            if loss_label not in seen_losses:
                seen_losses.add(loss_label)
                fragment_mz = round(precursor_mz - loss_mass, 4)
                if fragment_mz > 0:
                    predicted_losses.append({
                        "functional_group": fgroup,
                        "neutral_loss": loss_label,
                        "loss_mass_da": round(loss_mass, 4),
                        "fragment_mz": fragment_mz,
                    })

    # Check for characteristic fragment ions
    char_frags: list[dict] = []
    # Tropylium: benzyl system (ArCH2– or CH2=CH–Ph)
    benzyl_patt = Chem.MolFromSmarts("c1ccccc1C")
    if benzyl_patt and mol.HasSubstructMatch(benzyl_patt):
        char_frags.append({"mz": 91.0548, "formula": "C7H7+", "description": "tropylium ion (benzyl system)"})

    # Phenyl cation
    phenyl_patt = Chem.MolFromSmarts("c1ccccc1")
    if phenyl_patt and mol.HasSubstructMatch(phenyl_patt):
        char_frags.append({"mz": 77.0386, "formula": "C6H5+", "description": "phenyl cation"})

    # Acetyl cation: methyl ketone
    acetyl_patt = Chem.MolFromSmarts("[CH3][CX3](=O)")
    if acetyl_patt and mol.HasSubstructMatch(acetyl_patt):
        char_frags.append({"mz": 43.0184, "formula": "C2H3O+", "description": "acetyl cation CH3CO+"})

    # Benzoyl cation
    benzoyl_patt = Chem.MolFromSmarts("c1ccccc1[CX3](=O)")
    if benzoyl_patt and mol.HasSubstructMatch(benzoyl_patt):
        char_frags.append({"mz": 105.0335, "formula": "C7H5O+", "description": "benzoyl cation ArCO+"})

    # Formyl cation: aldehyde
    aldehyde_patt = Chem.MolFromSmarts("[CX3H1](=O)")
    if aldehyde_patt and mol.HasSubstructMatch(aldehyde_patt):
        char_frags.append({"mz": 29.0022, "formula": "CHO+", "description": "formyl cation (aldehyde)"})

    # Build summary
    lines = [
        f"MS fragmentation prediction for {smiles}",
        f"  Formula: {formula} | Monoisotopic mass: {mono_mass:.4f} Da",
        f"  Precursor {precursor_adduct}: m/z {precursor_mz:.4f}",
        "",
    ]

    if predicted_losses:
        lines.append("Predicted neutral losses (from functional groups):")
        for fl in sorted(predicted_losses, key=lambda x: x["loss_mass_da"]):
            lines.append(
                f"  –{fl['neutral_loss']} ({fl['loss_mass_da']:.4f} Da) → m/z {fl['fragment_mz']:.4f}"
                f"  [{fl['functional_group']}]"
            )
    else:
        lines.append("No common neutral losses predicted for this structure.")

    if char_frags:
        lines.append("")
        lines.append("Characteristic fragment ions:")
        for cf in char_frags:
            lines.append(f"  m/z {cf['mz']:.4f} ({cf['formula']}) — {cf['description']}")

    return {
        "summary": "\n".join(lines),
        "smiles": smiles,
        "formula": formula,
        "monoisotopic_mass": round(mono_mass, 4),
        "precursor_adduct": precursor_adduct,
        "precursor_mz": precursor_mz,
        "predicted_neutral_losses": predicted_losses,
        "characteristic_fragments": char_frags,
    }


# ---------------------------------------------------------------------------
# Formula candidates from exact mass
# ---------------------------------------------------------------------------

@registry.register(
    name="ms.formula_from_mass",
    description=(
        "Generate candidate molecular formulas matching an observed exact mass (monoisotopic). "
        "Searches within a given mass tolerance (default +/-5 ppm) for organic formulas "
        "up to the given carbon/hydrogen/heteroatom limits. "
        "Useful for de-novo formula assignment from high-resolution MS data."
    ),
    category="ms",
    parameters={
        "observed_mass": "Observed monoisotopic mass in Da (e.g. '180.0634')",
        "tolerance_ppm": "Mass tolerance in ppm (default 5)",
        "max_carbons": "Maximum number of carbons to consider (default 20)",
        "adduct": "Adduct to account for: 'neutral' (default), '[M+H]+', '[M-H]-', '[M+Na]+'",
    },
)
def formula_from_mass(
    observed_mass: str = "",
    tolerance_ppm: float = 5.0,
    max_carbons: int = 20,
    adduct: str = "neutral",
    **kwargs,
) -> dict:
    """Generate candidate molecular formulas for an observed exact mass."""
    if not observed_mass:
        return {"summary": "Error: no observed_mass provided"}

    try:
        obs = float(observed_mass)
    except ValueError:
        return {"summary": f"Error: could not parse observed_mass '{observed_mass}'"}

    # Convert observed m/z to neutral monoisotopic mass
    adduct_corrections = {
        "neutral":  0.0,
        "[M+H]+":  -1.007276,
        "[M-H]-":   1.007276,
        "[M+Na]+": -22.989218,
        "[M+K]+":  -38.963158,
    }
    correction = adduct_corrections.get(adduct, 0.0)
    target_mass = obs + correction

    tol_da = target_mass * tolerance_ppm * 1e-6

    # Search CxHyNzOwSv formulas (common organic elements)
    candidates = []
    # Enumerate: C0-max_carbons, H0-2*C+4+2*N, N0-4, O0-6, S0-2
    for c in range(0, max_carbons + 1):
        max_h = 2 * c + 4 + 8  # generous upper bound
        for n in range(0, 5):
            for o in range(0, 7):
                for s in range(0, 3):
                    for h in range(0, max_h + 1):
                        # Quick mass estimate
                        m = (c * 12.0 + h * 1.0078250319 + n * 14.0030740052
                             + o * 15.9949146221 + s * 31.97207069)
                        if abs(m - target_mass) <= tol_da:
                            # Check degree of unsaturation (must be ≥ 0 and integer/half-integer)
                            dou = (2 * c + 2 + 2 * n - h) / 2.0
                            if dou < 0 or (dou % 0.5) != 0:
                                continue
                            formula_parts = []
                            if c: formula_parts.append(f"C{c}" if c > 1 else "C")
                            if h: formula_parts.append(f"H{h}" if h > 1 else "H")
                            if n: formula_parts.append(f"N{n}" if n > 1 else "N")
                            if o: formula_parts.append(f"O{o}" if o > 1 else "O")
                            if s: formula_parts.append(f"S{s}" if s > 1 else "S")
                            candidates.append({
                                "formula": "".join(formula_parts) or "?",
                                "exact_mass": round(m, 6),
                                "mass_error_ppm": round((m - target_mass) / target_mass * 1e6, 2),
                                "degree_of_unsaturation": dou,
                            })

    # Sort by absolute mass error
    candidates.sort(key=lambda x: abs(x["mass_error_ppm"]))
    candidates = candidates[:20]  # top 20

    lines = [
        f"Formula candidates for m/z {obs:.4f} Da (adduct: {adduct}, tol: ±{tolerance_ppm} ppm)",
        f"  Target neutral mass: {target_mass:.6f} Da | Window: ±{tol_da*1000:.2f} mDa",
        f"  Found {len(candidates)} candidate(s):",
        "",
        f"  {'Formula':<14} {'Exact Mass':>12}  {'Error (ppm)':>11}  {'DoU':>5}",
        f"  {'-'*14} {'-'*12}  {'-'*11}  {'-'*5}",
    ]
    for cand in candidates:
        lines.append(
            f"  {cand['formula']:<14} {cand['exact_mass']:>12.6f}  "
            f"{cand['mass_error_ppm']:>+11.2f}  {cand['degree_of_unsaturation']:>5.1f}"
        )

    return {
        "summary": "\n".join(lines),
        "observed_mass": obs,
        "adduct": adduct,
        "target_neutral_mass": round(target_mass, 6),
        "tolerance_ppm": tolerance_ppm,
        "candidates": candidates,
    }

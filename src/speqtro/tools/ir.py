"""Infrared spectroscopy prediction tools using RDKit SMARTS-based functional group detection."""

from speqtro.tools import registry


# ---------------------------------------------------------------------------
# IR absorption band database (functional group -> wavenumber ranges, intensity)
# ---------------------------------------------------------------------------
# Each entry: (smarts, group_name, low_cm, high_cm, intensity, notes)
_IR_BANDS = [
    # O-H stretches
    ("[OX2H][CX4]",                  "alcohol O-H stretch",           3200, 3550, "strong, broad",
     "Characteristic broad absorption; sharper for dilute solutions"),
    ("c[OH]",                         "phenol O-H stretch",            3200, 3500, "strong, broad",
     "Similar to alcohol but slightly lower"),
    ("[CX3](=O)[OX2H1]",             "carboxylic acid O-H stretch",   2500, 3300, "very broad, strong",
     "Very characteristic broad absorption overlapping C-H region"),
    # N-H stretches
    ("[NX3;H2]",                      "primary amine N-H stretch",     3300, 3500, "medium, two bands",
     "Two N-H stretches (symmetric + antisymmetric)"),
    ("[NX3;H1;!$(NC=O)]",            "secondary amine N-H stretch",   3300, 3350, "medium, one band",
     "Single N-H stretch"),
    ("[NX3;H2][CX3]=O",              "primary amide N-H stretch",     3170, 3350, "medium, two bands",
     "Amide N-H slightly lower than amine"),
    ("[NX3;H1][CX3]=O",              "secondary amide N-H stretch",   3280, 3300, "medium",
     "One N-H stretch"),
    # C-H stretches (always present, diagnostic for sp2/sp3)
    ("[CX4;H]",                       "sp3 C-H stretch",               2850, 2960, "medium-strong",
     "CH3 antisym ~2962, CH2 antisym ~2926, CH sym ~2853"),
    ("[cH]",                          "aromatic C-H stretch",          3000, 3100, "medium",
     "Slightly above 3000 cm-1"),
    ("[CX3;H]=[CX3]",                "vinyl C-H stretch",             3010, 3100, "medium",
     ">3000 cm-1, distinguishes from sp3"),
    ("[CX2;H]#[CX2]",               "terminal alkyne C-H stretch",  3300, 3333, "strong",
     "Very sharp, characteristic of terminal alkynes"),
    # C=O stretches (most diagnostic IR region)
    ("[CH1](=O)",                     "aldehyde C=O stretch",          1720, 1740, "strong",
     "Also shows Fermi doublet at 2700-2850 cm-1 (C-H of CHO)"),
    ("[CH1](=O)",                     "aldehyde C-H overtone (Fermi)", 2700, 2850, "weak-medium",
     "Fermi resonance doublet, diagnostic for aldehydes"),
    ("[CX3](=O)[CX4,cX3]",          "ketone C=O stretch",            1705, 1725, "strong",
     "Cyclic ketones shift: 6-ring ~1715, 5-ring ~1740, 4-ring ~1775"),
    ("[CX3](=O)[OX2][CX4,cX3]",    "ester C=O stretch",             1735, 1750, "strong",
     "Also shows C-O stretch 1000-1300; vinyl esters ~1760"),
    ("[CX3](=O)[OX2H1]",            "carboxylic acid C=O stretch",   1700, 1725, "strong",
     "Broad due to H-bonding; anhydrides show two bands ~1800+1850"),
    ("[NX3][CX3](=O)",              "amide C=O stretch (Amide I)",   1630, 1690, "strong",
     "Primary amide ~1680, secondary ~1660, tertiary ~1650; Amide II N-H bend ~1500-1560"),
    ("[cX3][CX3]=O",                "aryl ketone C=O stretch",       1680, 1700, "strong",
     "Conjugation lowers C=O frequency"),
    # C=C stretches
    ("[CX3]=[CX3]",                  "alkene C=C stretch",            1620, 1680, "variable",
     "Symmetrical alkenes may be IR-inactive; terminal alkenes ~1640"),
    ("c1ccccc1",                      "aromatic C=C ring stretch",     1450, 1600, "medium, two bands",
     "Two bands ~1450 and ~1500-1600; substitution pattern affects overtones 1650-2000"),
    # Triple bonds
    ("[CX2]#[CX2]",                 "internal alkyne CC stretch",   2100, 2260, "variable",
     "Weak or absent for symmetric alkynes; terminal alkyne ~2100-2140"),
    ("[CX2]#[NX1]",                 "nitrile CN stretch",           2200, 2260, "strong",
     "Very diagnostic; isonitrile ~2100-2200"),
    # C-O single bond
    ("[CX4][OX2][CX4]",             "ether C-O-C stretch",           1000, 1300, "strong",
     "Broad region; cyclic ethers: THF ~1070, epoxide ~800-950"),
    ("[CX4][OX2H]",                  "alcohol C-O stretch",           1000, 1150, "strong",
     "Primary ~1050, secondary ~1100, tertiary ~1150"),
    ("[CX3](=O)[OX2][CX4]",        "ester C-O stretch",             1000, 1300, "strong, two bands",
     "Two C-O stretches (acyl C-O ~1200, alkyl C-O ~1060)"),
    # N-H bending
    ("[NX3;H2]",                      "primary amine N-H bend",        1550, 1650, "medium-strong",
     "Scissoring mode; overlaps with amide Amide I"),
    ("[NX3][CX3](=O)",              "amide N-H bend (Amide II)",     1500, 1560, "medium-strong",
     "Coupled N-H bending + C-N stretch"),
    # Nitro group
    ("[NX3](=O)=O",                  "nitro group N=O stretch",       1500, 1570, "strong",
     "Antisymmetric ~1550, symmetric ~1370; two strong bands"),
    ("[NX3](=O)=O",                  "nitro group N=O sym stretch",   1350, 1380, "strong",
     "Second nitro band; together with ~1550 diagnostic for -NO2"),
    # C-N single bond
    ("[CX4][NX3]",                   "aliphatic C-N stretch",         1020, 1220, "medium",
     "Weaker than C-O; aromatic C-N ~1180-1360"),
    # S-H
    ("[SX2H]",                        "thiol S-H stretch",             2550, 2600, "weak",
     "Characteristic but weak; lower than O-H"),
    # C-X halogens
    ("[CX4][F]",                      "C-F stretch",                   1000, 1400, "strong",
     "CF3 ~1100-1200 very strong; C-F often strongest band in spectrum"),
    ("[CX4][Cl]",                     "C-Cl stretch",                   600,  800, "strong",
     "Below fingerprint region; C-Cl2 two bands"),
    ("[CX4][Br]",                     "C-Br stretch",                   500,  800, "strong",
     "Lower frequency than C-Cl"),
    # Phosphorus
    ("[P](=O)",                       "P=O stretch",                   1100, 1300, "strong",
     "Phosphonate ~1230, phosphate ~1100"),
    # Sulfone/sulfoxide
    ("[SX3](=O)",                     "sulfoxide S=O stretch",         1030, 1070, "strong",
     "Diagnostic for DMSO-like sulfoxides"),
    ("[SX4](=O)(=O)",                "sulfone S=O stretch",           1120, 1160, "strong, two bands",
     "Two S=O stretches (antisym + sym)"),
]


def _predict_ir_from_smiles(smiles: str) -> list[dict]:
    """Return list of predicted IR bands for a SMILES string."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    bands: list[dict] = []
    seen_groups: set[str] = set()

    for smarts, group_name, low_cm, high_cm, intensity, notes in _IR_BANDS:
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        if mol.HasSubstructMatch(patt):
            key = group_name
            if key not in seen_groups:
                seen_groups.add(key)
                bands.append({
                    "functional_group": group_name,
                    "wavenumber_low": low_cm,
                    "wavenumber_high": high_cm,
                    "wavenumber_range": f"{low_cm}-{high_cm} cm-1",
                    "intensity": intensity,
                    "notes": notes,
                })

    # Sort by wavenumber (highest first -- conventional IR presentation)
    bands.sort(key=lambda b: b["wavenumber_high"], reverse=True)
    return bands


@registry.register(
    name="ir.predict_absorptions",
    description=(
        "Predict infrared (IR) absorption bands for a molecule from its SMILES string. "
        "Uses SMARTS-based functional group detection to identify expected absorption "
        "wavenumber ranges (cm-1), intensities, and diagnostic notes. "
        "Covers O-H, N-H, C-H, C=O, C=C, CN, CC, C-O, C-N, C-X, and more."
    ),
    category="ir",
    parameters={
        "smiles": "SMILES string of the molecule (e.g. 'CC(=O)O' for acetic acid)",
    },
)
def predict_absorptions(smiles: str = "", **kwargs) -> dict:
    """Predict IR absorption bands from SMILES using functional group tables."""
    if not smiles:
        return {"summary": "Error: no SMILES provided"}

    try:
        from rdkit import Chem
    except ImportError:
        return {"summary": "Error: RDKit not installed. Run: pip install rdkit"}

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"summary": f"Error: could not parse SMILES '{smiles}'"}

    bands = _predict_ir_from_smiles(smiles)

    if not bands:
        return {
            "summary": f"No characteristic IR absorptions predicted for '{smiles}'. "
                       f"Molecule may contain only C-H and C-C bonds.",
            "smiles": smiles,
            "bands": [],
        }

    lines = [f"Predicted IR absorptions for {smiles}:", ""]
    lines.append(f"  {'Functional Group':<38} {'Range (cm-1)':<18} {'Intensity'}")
    lines.append(f"  {'-'*38} {'-'*18} {'-'*20}")

    for b in bands:
        lines.append(
            f"  {b['functional_group']:<38} {b['wavenumber_range']:<18} {b['intensity']}"
        )

    lines.append("")
    lines.append("Diagnostic notes:")
    for b in bands:
        if b["notes"]:
            lines.append(f"  • {b['functional_group']}: {b['notes']}")

    return {
        "summary": "\n".join(lines),
        "smiles": smiles,
        "bands": bands,
        "n_bands": len(bands),
    }

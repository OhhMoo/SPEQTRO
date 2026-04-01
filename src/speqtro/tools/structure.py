"""Structure / cheminformatics tools."""

from speqtro.tools import registry


@registry.register(
    name="structure.smiles_to_formula",
    description="Calculate molecular formula, molecular weight, exact mass, and degree of unsaturation from a SMILES string.",
    category="structure",
    parameters={
        "smiles": "SMILES string for the molecule (e.g. 'CCO' for ethanol)",
    },
)
def smiles_to_formula(smiles: str = "", **kwargs) -> dict:
    """Calculate molecular properties from SMILES using RDKit."""
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
    mw = Descriptors.MolWt(mol)
    exact_mass = Descriptors.ExactMolWt(mol)

    # Count C, H, N, O, halogens for DoU
    atom_counts = {}
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        atom_counts[sym] = atom_counts.get(sym, 0) + 1
        atom_counts["H"] = atom_counts.get("H", 0) + atom.GetTotalNumHs()

    C = atom_counts.get("C", 0)
    H = atom_counts.get("H", 0)
    N = atom_counts.get("N", 0)
    X = sum(atom_counts.get(x, 0) for x in ("F", "Cl", "Br", "I"))
    dou = (2 * C + 2 + N - H - X) / 2

    dou_interpretation = []
    if dou == 0:
        dou_interpretation.append("saturated (no rings or double bonds)")
    elif dou == 1:
        dou_interpretation.append("one ring or one C=C/C=O double bond")
    elif dou == 4:
        dou_interpretation.append("likely benzene ring (4 DoU)")
    elif dou > 4:
        dou_interpretation.append(f"aromatic or polycyclic system ({dou:.0f} DoU)")

    summary = (
        f"Formula: {formula} | MW: {mw:.3f} Da | Exact Mass: {exact_mass:.4f} Da | "
        f"DoU: {dou:.1f} ({', '.join(dou_interpretation) or 'mixed'})"
    )

    return {
        "summary": summary,
        "smiles": smiles,
        "formula": formula,
        "molecular_weight": round(mw, 4),
        "exact_mass": round(exact_mass, 6),
        "degree_of_unsaturation": dou,
        "atom_counts": atom_counts,
    }

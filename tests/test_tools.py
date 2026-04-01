"""Tests for speq tools — use mocked/pure-Python implementations, no real datasets."""

import pytest


# ---------------------------------------------------------------------------
# structure.smiles_to_formula
# ---------------------------------------------------------------------------

def test_smiles_to_formula_ethanol():
    pytest.importorskip("rdkit")
    from speqtro.tools.structure import smiles_to_formula
    result = smiles_to_formula(smiles="CCO")
    assert result.get("formula") == "C2H6O"
    assert abs(result["molecular_weight"] - 46.07) < 0.1
    assert "summary" in result
    assert result["degree_of_unsaturation"] == 0.0


def test_smiles_to_formula_benzene():
    pytest.importorskip("rdkit")
    from speqtro.tools.structure import smiles_to_formula
    result = smiles_to_formula(smiles="c1ccccc1")
    assert result["formula"] == "C6H6"
    assert result["degree_of_unsaturation"] == 4.0


def test_smiles_to_formula_aspirin():
    pytest.importorskip("rdkit")
    from speqtro.tools.structure import smiles_to_formula
    result = smiles_to_formula(smiles="CC(=O)Oc1ccccc1C(=O)O")
    assert result["formula"] == "C9H8O4"
    assert abs(result["exact_mass"] - 180.042) < 0.01


def test_smiles_to_formula_empty():
    from speqtro.tools.structure import smiles_to_formula
    result = smiles_to_formula(smiles="")
    assert "Error" in result["summary"]


def test_smiles_to_formula_invalid():
    pytest.importorskip("rdkit")
    from speqtro.tools.structure import smiles_to_formula
    result = smiles_to_formula(smiles="NOTASMILES!!!!")
    assert "Error" in result["summary"]


# ---------------------------------------------------------------------------
# ms.calc_exact_mass
# ---------------------------------------------------------------------------

def test_calc_exact_mass_glucose():
    from speqtro.tools.ms import calc_exact_mass
    result = calc_exact_mass(formula="C6H12O6")
    assert abs(result["monoisotopic_mass"] - 180.063) < 0.01
    assert "[M+H]+" in result["adducts"]
    assert abs(result["adducts"]["[M+H]+"] - 181.070) < 0.01
    assert "[M+Na]+" in result["adducts"]
    assert "[M-H]-" in result["adducts"]


def test_calc_exact_mass_aspirin():
    from speqtro.tools.ms import calc_exact_mass
    result = calc_exact_mass(formula="C9H8O4")
    assert abs(result["monoisotopic_mass"] - 180.042) < 0.01


def test_calc_exact_mass_empty():
    from speqtro.tools.ms import calc_exact_mass
    result = calc_exact_mass(formula="")
    assert "Error" in result["summary"]


def test_calc_exact_mass_unknown_element():
    from speqtro.tools.ms import calc_exact_mass
    result = calc_exact_mass(formula="C6H12X99")
    assert "Error" in result["summary"] or "Unknown" in result["summary"]


# ---------------------------------------------------------------------------
# nmr.predict_h1_shifts
# ---------------------------------------------------------------------------

def test_predict_h1_shifts_ethanol():
    pytest.importorskip("rdkit")
    from speqtro.tools.nmr import predict_h1_shifts
    result = predict_h1_shifts(smiles="CCO")
    assert "summary" in result
    assert result.get("predictions")
    # Should find at least alkyl and O-H environments
    envs = [p["environment"] for p in result["predictions"]]
    assert any("alkyl" in e or "C–O" in e or "O–H" in e for e in envs)


def test_predict_h1_shifts_benzene():
    pytest.importorskip("rdkit")
    from speqtro.tools.nmr import predict_h1_shifts
    result = predict_h1_shifts(smiles="c1ccccc1")
    assert result.get("predictions")
    envs = [p["environment"] for p in result["predictions"]]
    assert any("aromatic" in e for e in envs)
    # Aromatic H should be in 6.5-8.5 range
    for p in result["predictions"]:
        if "aromatic" in p["environment"]:
            assert 6.0 <= p["estimated_shift_ppm"] <= 9.0


def test_predict_h1_shifts_empty():
    from speqtro.tools.nmr import predict_h1_shifts
    result = predict_h1_shifts(smiles="")
    assert "Error" in result["summary"]


# ---------------------------------------------------------------------------
# nmr.predict_c13_shifts
# ---------------------------------------------------------------------------

def test_predict_c13_shifts_acetic_acid():
    pytest.importorskip("rdkit")
    from speqtro.tools.nmr import predict_c13_shifts
    result = predict_c13_shifts(smiles="CC(=O)O")
    assert result.get("predictions")
    assert len(result["predictions"]) == 2  # 2 carbons
    envs = [p["environment"] for p in result["predictions"]]
    assert any("alkyl" in e or "carbonyl" in e or "ester" in e for e in envs)


def test_predict_c13_shifts_empty():
    from speqtro.tools.nmr import predict_c13_shifts
    result = predict_c13_shifts(smiles="")
    assert "Error" in result["summary"]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_has_5_tools():
    from speqtro.tools import registry, ensure_loaded
    ensure_loaded()
    tools = registry.list_tools()
    names = [t.name for t in tools]
    assert "structure.smiles_to_formula" in names
    assert "ms.calc_exact_mass" in names
    assert "nmr.predict_h1_shifts" in names
    assert "nmr.predict_c13_shifts" in names
    assert "database.pubchem_search" in names
    assert len(tools) >= 5


def test_registry_categories():
    from speqtro.tools import registry, ensure_loaded
    ensure_loaded()
    cats = registry.categories()
    assert "nmr" in cats
    assert "ms" in cats
    assert "structure" in cats
    assert "database" in cats
    assert "ir" in cats


# ---------------------------------------------------------------------------
# nmr.parse_jcamp
# ---------------------------------------------------------------------------

_JCAMP_SAMPLE = """\
##TITLE= Ethanol test
##JCAMPDX= 4.24
##DATA TYPE= NMR SPECTRUM
##.OBSERVE NUCLEUS= ^1H
##.OBSERVE FREQUENCY= 400.0
##XUNITS= PPM
##YUNITS= ARBITRARY UNITS
##FIRSTX= 5.0
##LASTX= 0.0
##NPOINTS= 6
##XFACTOR= 1.0
##YFACTOR= 1.0
##XYDATA=(X++(Y..Y))
5.0 0 0 500 0 0 5000
##END=
"""


def test_parse_jcamp_from_text():
    from speqtro.tools.nmr import parse_jcamp
    result = parse_jcamp(jcamp_text=_JCAMP_SAMPLE)
    assert "summary" in result
    assert "Error" not in result["summary"]
    assert result["nucleus"] == "^1H"
    assert result["n_data_points"] > 0


def test_parse_jcamp_empty():
    from speqtro.tools.nmr import parse_jcamp
    result = parse_jcamp()
    assert "Error" in result["summary"]


def test_parse_jcamp_bad_file():
    from speqtro.tools.nmr import parse_jcamp
    result = parse_jcamp(file_path="/nonexistent/path/file.jdx")
    assert "Error" in result["summary"]


# ---------------------------------------------------------------------------
# nmr.find_h1_pattern
# ---------------------------------------------------------------------------

def test_find_h1_pattern_aspirin():
    from speqtro.tools.nmr import find_h1_pattern
    result = find_h1_pattern(peaks_ppm="8.11, 7.55, 7.35, 7.10, 2.35")
    assert "summary" in result
    assert result["n_groups"] >= 2
    hints = result["structural_hints"]
    assert any("aromatic" in h.lower() for h in hints)


def test_find_h1_pattern_empty():
    from speqtro.tools.nmr import find_h1_pattern
    result = find_h1_pattern(peaks_ppm="")
    assert "Error" in result["summary"]


def test_find_h1_pattern_multiplet_grouping():
    from speqtro.tools.nmr import find_h1_pattern
    # Doublet: two peaks 8 Hz apart at 400 MHz = 0.02 ppm
    result = find_h1_pattern(peaks_ppm="3.50, 3.48", spectrometer_mhz=400.0)
    assert result["n_groups"] == 1
    group = result["multiplet_groups"][0]
    assert "doublet" in group["multiplicity"]


def test_find_h1_pattern_aldehyde_hint():
    from speqtro.tools.nmr import find_h1_pattern
    result = find_h1_pattern(peaks_ppm="9.80, 7.72, 7.50")
    assert any("aldehyde" in h.lower() or "carboxylic" in h.lower() for h in result["structural_hints"])


# ---------------------------------------------------------------------------
# ir.predict_absorptions
# ---------------------------------------------------------------------------

def test_ir_predict_aspirin():
    pytest.importorskip("rdkit")
    from speqtro.tools.ir import predict_absorptions
    result = predict_absorptions(smiles="CC(=O)Oc1ccccc1C(=O)O")
    assert "summary" in result
    assert result["n_bands"] > 0
    groups = [b["functional_group"] for b in result["bands"]]
    # Aspirin has ester C=O and carboxylic acid C=O
    assert any("ester" in g or "carboxylic" in g for g in groups)
    assert any("aromatic" in g for g in groups)


def test_ir_predict_ethanol():
    pytest.importorskip("rdkit")
    from speqtro.tools.ir import predict_absorptions
    result = predict_absorptions(smiles="CCO")
    groups = [b["functional_group"] for b in result["bands"]]
    assert any("alcohol" in g for g in groups)


def test_ir_predict_empty():
    from speqtro.tools.ir import predict_absorptions
    result = predict_absorptions(smiles="")
    assert "Error" in result["summary"]


def test_ir_predict_nitrile():
    pytest.importorskip("rdkit")
    from speqtro.tools.ir import predict_absorptions
    result = predict_absorptions(smiles="CC#N")  # acetonitrile
    groups = [b["functional_group"] for b in result["bands"]]
    assert any("nitrile" in g for g in groups)


# ---------------------------------------------------------------------------
# ms.fragment_predict
# ---------------------------------------------------------------------------

def test_fragment_predict_aspirin():
    pytest.importorskip("rdkit")
    from speqtro.tools.ms import fragment_predict
    result = fragment_predict(smiles="CC(=O)Oc1ccccc1C(=O)O")
    assert "summary" in result
    assert result["precursor_mz"] > 0
    # Should predict loss of H2O and/or CO2 from carboxylic acid
    losses = [fl["neutral_loss"] for fl in result["predicted_neutral_losses"]]
    assert any("H2O" in l or "CO2" in l for l in losses)


def test_fragment_predict_aniline():
    pytest.importorskip("rdkit")
    from speqtro.tools.ms import fragment_predict
    result = fragment_predict(smiles="Nc1ccccc1")  # aniline
    losses = [fl["neutral_loss"] for fl in result["predicted_neutral_losses"]]
    assert any("NH3" in l for l in losses)


def test_fragment_predict_toluene_tropylium():
    pytest.importorskip("rdkit")
    from speqtro.tools.ms import fragment_predict
    result = fragment_predict(smiles="Cc1ccccc1")  # toluene
    char_mzs = [cf["mz"] for cf in result["characteristic_fragments"]]
    assert 91.0548 in char_mzs  # tropylium


def test_fragment_predict_empty():
    from speqtro.tools.ms import fragment_predict
    result = fragment_predict(smiles="")
    assert "Error" in result["summary"]


# ---------------------------------------------------------------------------
# ms.formula_from_mass
# ---------------------------------------------------------------------------

def test_formula_from_mass_glucose():
    from speqtro.tools.ms import formula_from_mass
    result = formula_from_mass(observed_mass="180.0634", tolerance_ppm=5)
    assert "summary" in result
    assert result["candidates"]
    formulas = [c["formula"] for c in result["candidates"]]
    # C6H12O6 should be in candidates
    assert any("C6" in f and "O6" in f for f in formulas)


def test_formula_from_mass_mh_adduct():
    from speqtro.tools.ms import formula_from_mass
    # Aspirin [M+H]+: 181.0495
    result = formula_from_mass(observed_mass="181.0495", adduct="[M+H]+", tolerance_ppm=5)
    assert result["candidates"]
    formulas = [c["formula"] for c in result["candidates"]]
    assert any("C9" in f for f in formulas)


def test_formula_from_mass_empty():
    from speqtro.tools.ms import formula_from_mass
    result = formula_from_mass(observed_mass="")
    assert "Error" in result["summary"]

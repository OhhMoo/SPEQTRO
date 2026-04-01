"""chem_utils.py -- vendored from ms-pred, imports fixed for speqtro."""

import re
import numpy as np
import logging
import torch
from rdkit import Chem
from rdkit.Chem import Atom
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from rdkit.Chem.Descriptors import ExactMolWt

try:
    from rdkit.Chem.MolStandardize.tautomer import TautomerCanonicalizer, TautomerTransform
    _RD_TAUTOMER_CANONICALIZER = 'v1'
    _TAUTOMER_TRANSFORMS = (
        TautomerTransform('1,3 heteroatom H shift',
                          '[#7,S,O,Se,Te;!H0]-[#7X2,#6,#15]=[#7,#16,#8,Se,Te]'),
        TautomerTransform('1,3 (thio)keto/enol r', '[O,S,Se,Te;X2!H0]-[C]=[C]'),
    )
except (ModuleNotFoundError, ImportError):
    from rdkit.Chem.MolStandardize.rdMolStandardize import TautomerEnumerator
    _RD_TAUTOMER_CANONICALIZER = 'v2'

P_TBL = Chem.GetPeriodicTable()

ROUND_FACTOR = 4
ELECTRON_MASS = 0.00054858
CHEM_FORMULA_SIZE = "([A-Z][a-z]*)([0-9]*)"

VALID_ELEMENTS = [
    "C", "N", "P", "O", "S", "Si", "I", "H", "Cl", "F",
    "Br", "B", "Se", "Fe", "Co", "As", "Na", "K",
]

ELEMENT_TO_GROUP = {
    "C": 4, "N": 3, "P": 3, "O": 5, "S": 5, "Si": 4,
    "I": 6, "H": 0, "Cl": 6, "F": 6, "Br": 6, "B": 2,
    "Se": 5, "Fe": 7, "Co": 7, "As": 3, "Na": 1, "K": 1,
}
ELEMENT_GROUP_DIM = len(set(ELEMENT_TO_GROUP.values()))
ELEMENT_GROUP_VECTORS = np.eye(ELEMENT_GROUP_DIM)

VALID_ATOM_NUM = [Atom(i).GetAtomicNum() for i in VALID_ELEMENTS]
CHEM_ELEMENT_NUM = len(VALID_ELEMENTS)

ATOM_NUM_TO_ONEHOT = torch.zeros((max(VALID_ATOM_NUM) + 1, CHEM_ELEMENT_NUM))
ATOM_NUM_TO_ONEHOT[VALID_ATOM_NUM, torch.arange(CHEM_ELEMENT_NUM)] = 1

VALID_MONO_MASSES = np.array(
    [P_TBL.GetMostCommonIsotopeMass(i) for i in VALID_ELEMENTS]
)
CHEM_MASSES = VALID_MONO_MASSES[:, None]

ELEMENT_VECTORS = np.eye(len(VALID_ELEMENTS))
ELEMENT_VECTORS_MASS = np.hstack([ELEMENT_VECTORS, CHEM_MASSES])
ELEMENT_TO_MASS = dict(zip(VALID_ELEMENTS, CHEM_MASSES.squeeze()))

ELEMENT_DIM_MASS = len(ELEMENT_VECTORS_MASS[0])
ELEMENT_DIM = len(ELEMENT_VECTORS[0])

COLLISION_PE_DIM = 64
COLLISION_PE_SCALAR = 10000

SIM_PE_DIM = 64
SIM_PE_SCALAR = 10

GRAPHTYPE_LEN = 32

NORM_VEC_MASS = np.array(
    [81, 19, 6, 34, 6, 6, 6, 158, 10, 17, 3, 1, 2, 1, 1, 2, 1, 1, 1471]
)
NORM_VEC = np.array([81, 19, 6, 34, 6, 6, 6, 158, 10, 17, 3, 1, 2, 1, 1, 2, 1, 1])
MAX_ATOM_CT = 160

BINARY_BITS = 8
MAX_ELEMENT_NUM = 64
MAX_H = 6

element_to_ind = dict(zip(VALID_ELEMENTS, np.arange(len(VALID_ELEMENTS))))
element_to_position = dict(zip(VALID_ELEMENTS, ELEMENT_VECTORS))
element_to_position_mass = dict(zip(VALID_ELEMENTS, ELEMENT_VECTORS_MASS))
element_to_group = {k: ELEMENT_GROUP_VECTORS[v] for k, v in ELEMENT_TO_GROUP.items()}

# -- Adduct / ion mappings ---------------------------------------------------

ion2mass = {
    "[M+H]+": ELEMENT_TO_MASS["H"] - ELECTRON_MASS,
    "[M+Na]+": ELEMENT_TO_MASS["Na"] - ELECTRON_MASS,
    "[M+K]+": ELEMENT_TO_MASS["K"] - ELECTRON_MASS,
    "[M-H2O+H]+": -ELEMENT_TO_MASS["O"] - ELEMENT_TO_MASS["H"] - ELECTRON_MASS,
    "[M+H-H2O]+": -ELEMENT_TO_MASS["O"] - ELEMENT_TO_MASS["H"] - ELECTRON_MASS,
    "[M+H3N+H]+": ELEMENT_TO_MASS["N"] + ELEMENT_TO_MASS["H"] * 4 - ELECTRON_MASS,
    "[M+NH4]+": ELEMENT_TO_MASS["N"] + ELEMENT_TO_MASS["H"] * 4 - ELECTRON_MASS,
    "[M]+": 0 - ELECTRON_MASS,
    "[M-H4O2+H]+": -ELEMENT_TO_MASS["O"] * 2 - ELEMENT_TO_MASS["H"] * 3 - ELECTRON_MASS,
    "[M+H-2H2O]+": -ELEMENT_TO_MASS["O"] * 2 - ELEMENT_TO_MASS["H"] * 3 - ELECTRON_MASS,
    "[M-H]-": -ELEMENT_TO_MASS["H"] + ELECTRON_MASS,
    "[M+Cl]-": ELEMENT_TO_MASS["Cl"] + ELECTRON_MASS,
    "[M-H2O-H]-": -ELEMENT_TO_MASS["O"] - ELEMENT_TO_MASS["H"] * 3 + ELECTRON_MASS,
    "[M-H-H2O]-": -ELEMENT_TO_MASS["O"] - ELEMENT_TO_MASS["H"] * 3 + ELECTRON_MASS,
    "[M-H-CO2]-": -ELEMENT_TO_MASS["C"] - ELEMENT_TO_MASS["O"] * 2 - ELEMENT_TO_MASS["H"] + ELECTRON_MASS,
}

ion2onehot_pos = {
    "[M+H]+": 0, "[M+Na]+": 1, "[M+K]+": 2,
    "[M-H2O+H]+": 3, "[M+H-H2O]+": 3,
    "[M+H3N+H]+": 4, "[M+NH4]+": 4,
    "[M]+": 5,
    "[M-H4O2+H]+": 6, "[M+H-2H2O]+": 6,
    "[M-H]-": 7, "[M+Cl]-": 8,
    "[M-H2O-H]-": 9, "[M-H-H2O]-": 9,
    "[M-H-CO2]-": 10,
}

ion_pos2extra_multihot = {v: set() for v in ion2onehot_pos.values()}
for _k, _v in ion2onehot_pos.items():
    _ion_mode = _k[-1]
    _kk = _k.strip(_ion_mode).strip('[M').strip(']')
    _ions = []
    for _idx, _i in enumerate(_kk.split('+')):
        if _i:
            if _idx != 0:
                _i = '+' + _i
            for _jdx, _j in enumerate(_i.split('-')):
                if _j:
                    if _jdx == 0:
                        _ions.append(_j)
                    else:
                        _ions.append('-' + _j)
    if _ion_mode == '+':
        ion_pos2extra_multihot[_v].add(0)
    else:
        ion_pos2extra_multihot[_v].add(1)
    if '+Na' in _ions or '+K' in _ions:
        ion_pos2extra_multihot[_v].add(2)
    if '+H' in _ions:
        ion_pos2extra_multihot[_v].add(3)
    if '-H' in _ions:
        ion_pos2extra_multihot[_v].add(4)
    if '+Cl' in _ions:
        ion_pos2extra_multihot[_v].add(5)
    if '-H2O' in _ions or '-H4O2' in _ions or '-2H2O' in _ions:
        ion_pos2extra_multihot[_v].add(6)
    if '-CO2' in _ions:
        ion_pos2extra_multihot[_v].add(7)
    if '+NH3' in _ions:
        ion_pos2extra_multihot[_v].add(8)

# Equivalent adduct keys
_ori_ions = list(ion2mass.keys())
for _ion in _ori_ions:
    _adduct, _charge = _ion.split(']')
    if not _charge[0].isnumeric():
        _eq_ion = _adduct + ']1' + _charge
        ion2mass[_eq_ion] = ion2mass[_ion]
        if _ion in ion2onehot_pos:
            ion2onehot_pos[_eq_ion] = ion2onehot_pos[_ion]

_ori_ions = list(ion2mass.keys())
for _ion in _ori_ions:
    _adduct, _charge = _ion.split(']')
    if 'H3N' in _adduct:
        _eq_ion = _ion.replace('H3N', 'NH3')
        ion2mass[_eq_ion] = ion2mass[_ion]
        if _ion in ion2onehot_pos:
            ion2onehot_pos[_eq_ion] = ion2onehot_pos[_ion]

instrument2onehot_pos = {
    "Orbitrap": 0,
    "QTOF": 1,
}


# -- Helper functions ---------------------------------------------------------

def is_positive_adduct(adduct_str: str) -> bool:
    return adduct_str[-1] == '+'


def formula_to_dense(chem_formula: str) -> np.ndarray:
    total_onehot = []
    for (chem_symbol, num) in re.findall(CHEM_FORMULA_SIZE, chem_formula):
        num = 1 if num == "" else int(num)
        one_hot = element_to_position[chem_symbol].reshape(1, -1)
        one_hot_repeats = np.repeat(one_hot, repeats=num, axis=0)
        total_onehot.append(one_hot_repeats)
    if len(total_onehot) == 0:
        dense_vec = np.zeros(len(element_to_position))
    else:
        dense_vec = np.vstack(total_onehot).sum(0)
    return dense_vec


def vec_to_formula(form_vec):
    build_str = ""
    for i in np.argwhere(form_vec > 0).flatten():
        el = VALID_ELEMENTS[i]
        ct = int(form_vec[i])
        new_item = f"{el}{ct}" if ct > 1 else f"{el}"
        build_str = build_str + new_item
    return build_str


def uncharged_formula(mol, mol_type="mol") -> str:
    if mol_type == "mol":
        chem_formula = CalcMolFormula(mol)
    elif mol_type == "smiles":
        mol = Chem.MolFromSmiles(mol)
        if mol is None:
            return None
        chem_formula = CalcMolFormula(mol)
    elif mol_type == "inchi":
        mol = Chem.MolFromInchi(mol)
        if mol is None:
            return None
        chem_formula = CalcMolFormula(mol)
    else:
        raise ValueError()
    return re.findall(r"^([^\+,^\-]*)", chem_formula)[0]


def canonical_mol_from_inchi(inchi):
    mol = Chem.MolFromInchi(inchi)
    if mol is None:
        return None
    if _RD_TAUTOMER_CANONICALIZER == 'v1':
        _molvs_t = TautomerCanonicalizer(transforms=_TAUTOMER_TRANSFORMS)
        mol = _molvs_t.canonicalize(mol)
    else:
        _te = TautomerEnumerator()
        mol = _te.Canonicalize(mol)
    return mol


def form_from_smi(smi: str) -> str:
    return uncharged_formula(smi, mol_type="smiles")


def inchi_from_smiles(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    else:
        return Chem.MolToInchi(mol)


def smiles_from_inchi(inchi: str) -> str:
    mol = canonical_mol_from_inchi(inchi)
    if mol is None:
        return ""
    else:
        return Chem.MolToSmiles(mol)


def _mol_from_types(mol, mol_type):
    if mol_type == 'smi':
        mol = Chem.MolFromSmiles(mol)
    elif mol_type == 'inchi':
        mol = canonical_mol_from_inchi(mol)
    elif mol_type == 'mol':
        mol = mol
    else:
        raise ValueError(f"Unknown mol_type={mol_type}")
    return mol


def rm_stereo(mol: str, mol_type='smi') -> str:
    mol = _mol_from_types(mol, mol_type)
    if mol is None:
        return
    else:
        Chem.RemoveStereochemistry(mol)
    if mol_type == 'smi':
        return Chem.MolToSmiles(mol)
    elif mol_type == 'inchi':
        return Chem.MolToInchi(mol)
    else:
        return mol


def mass_from_smi(smi: str) -> float:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return 0
    else:
        return ExactMolWt(mol)


def get_collision_energy(filename):
    import re as _re
    colli_eng = _re.findall('collision +([0-9]+\\.?[0-9]*|nan).*', filename)
    if len(colli_eng) > 1:
        raise ValueError(f'Multiple collision energies found in {filename}')
    if len(colli_eng) == 1:
        colli_eng = colli_eng[0].split()[-1]
    else:
        colli_eng = 'nan'
    return colli_eng


def formula_mass(chem_formula: str) -> float:
    mass = 0
    for (chem_symbol, num) in re.findall(CHEM_FORMULA_SIZE, chem_formula):
        num = 1 if num == "" else int(num)
        mass += ELEMENT_TO_MASS[chem_symbol] * num
    return mass

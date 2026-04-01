"""fragmentation.py -- vendored from ms-pred, imports fixed for speqtro.

Fragment a molecule combinatorially by pulling atoms.
"""
import numpy as np
from collections import Counter, defaultdict
from hashlib import blake2b
from typing import Tuple, List
from rdkit import Chem
from .. import common


TYPEW = {
    Chem.rdchem.BondType.names["AROMATIC"]: 2,
    Chem.rdchem.BondType.names["DOUBLE"]: 2,
    Chem.rdchem.BondType.names["TRIPLE"]: 3,
    Chem.rdchem.BondType.names["SINGLE"]: 1,
}
MAX_BONDS = max(list(TYPEW.values())) + 1
MAX_ATOM_BONDS = 6

# CC bonds are strongest --> higher score = harder to break
HETEROW = {False: 2, True: 1}


class FragmentEngine(object):
    """FragmentEngine."""

    def __init__(
        self,
        mol_str: str,
        max_tree_depth: int = 3,
        max_broken_bonds: int = 6,
        mol_str_type: str = "smiles",
        mol_str_canonicalized: bool = False,
    ):
        # Standardize mol by roundtripping it
        if mol_str_type == "smiles":
            self.smiles = mol_str
            self.mol = Chem.MolFromSmiles(self.smiles)
            if self.mol is None:
                return
            self.inchi = Chem.MolToInchi(self.mol)
            if not mol_str_canonicalized:
                self.mol = common.canonical_mol_from_inchi(self.inchi)
                self.smiles = Chem.MolToSmiles(self.mol)
                self.mol = Chem.MolFromSmiles(self.smiles)

        elif mol_str_type == "inchi":
            self.inchi = mol_str
            self.mol = common.canonical_mol_from_inchi(self.inchi)
            if self.mol is None:
                return
            self.smiles = Chem.MolToSmiles(self.mol)
            self.mol = Chem.MolFromSmiles(self.smiles)
        else:
            raise NotImplementedError()

        if self.mol is None:
            raise RuntimeError(f"Invalid molecule encountered. SMILES: {self.smiles}, InChI: {self.inchi}")

        self.natoms = self.mol.GetNumAtoms()

        # Kekulize the molecule
        Chem.Kekulize(self.mol, clearAromaticFlags=True)

        # Calculate number of hs on each atom and masses
        self.atom_symbols = [i.GetSymbol() for i in self.mol.GetAtoms()]
        self.atom_symbols_ar = np.array(self.atom_symbols)
        self.atom_hs = np.array(
            [i.GetNumImplicitHs() + i.GetNumExplicitHs() for i in self.mol.GetAtoms()]
        )
        self.total_hs = self.atom_hs.sum()
        self.atom_weights = np.array(
            [common.ELEMENT_TO_MASS[sym] if i.GetIsotope() == 0 else
             common.P_TBL.GetMassForIsotope(i.GetSymbol(), i.GetIsotope())
             for sym, i in zip(self.atom_symbols, self.mol.GetAtoms())]
        )
        self.atom_weights_h = (
            self.atom_hs * common.ELEMENT_TO_MASS["H"] + self.atom_weights
        )

        self.full_weight = np.sum(self.atom_weights_h)

        self.bonded_atoms = [[] for _ in self.atom_symbols]
        self.bonded_types = [[] for _ in self.atom_symbols]

        self.bonded_atoms_np = np.zeros((self.natoms, MAX_ATOM_BONDS), dtype=int)
        self.bonded_types_np = np.zeros((self.natoms, MAX_ATOM_BONDS), dtype=int)
        self.num_bonds_np = np.zeros(self.natoms, dtype=int)

        self.bond_to_type = {}
        self.bonds = set()
        self.bonds_list = []
        self.bond_types_list = []
        self.bond_inds_list = []
        self.bondscore = {}
        for bond in self.mol.GetBonds():
            a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            self.bonded_atoms[a1].append(a2)
            self.bonded_atoms[a2].append(a1)

            self.bonded_atoms_np[a1, self.num_bonds_np[a1]] = a2
            self.bonded_atoms_np[a2, self.num_bonds_np[a2]] = a1

            bondbits = 1 << a1 | 1 << a2
            bondscore = (
                TYPEW[bond.GetBondType()]
                * HETEROW[self.atom_symbols[a1] != "C" or self.atom_symbols[a2] != "C"]
            )
            bondtype = TYPEW[bond.GetBondType()]

            self.bonded_types[a1].append(bondtype)
            self.bonded_types[a2].append(bondtype)

            self.bonded_types_np[a1, self.num_bonds_np[a1]] = bondtype
            self.bonded_types_np[a2, self.num_bonds_np[a2]] = bondtype

            self.num_bonds_np[a1] += 1
            self.num_bonds_np[a2] += 1

            self.bond_to_type[bondbits] = bondtype
            self.bondscore[bondbits] = bondscore
            if bondbits not in self.bonds:
                self.bonds_list.append(bondbits)
                self.bond_types_list.append(bondtype)
                self.bond_inds_list.append((a1, a2))
            self.bonds.add(bondbits)

        self.max_broken_bonds = max_broken_bonds
        self.max_tree_depth = max_tree_depth

        self.shift_buckets = (
            np.arange(self.max_broken_bonds * 2 + 1) - self.max_broken_bonds
        )
        self.shift_bucket_inds = np.arange(self.max_broken_bonds * 2 + 1)
        self.shift_bucket_masses = self.shift_buckets * common.ELEMENT_TO_MASS["H"]

        self.frag_to_entry = {}

    def score_fragment(self, fragment):
        score, breaks = 0, 0
        for bond in self.bonds:
            if 0 < (fragment & bond) < bond:
                score += self.bondscore[bond]
                breaks += 1
        return breaks, score

    def single_mass(self, frag: int):
        fragment_mass = 0.0
        for atom in range(self.natoms):
            if frag & (1 << atom):
                fragment_mass += self.atom_weights_h[atom]
        return fragment_mass

    def formula_from_frag(self, frag: int, h_shift=0):
        form_vec = np.zeros(len(common.VALID_ELEMENTS))
        h_pos = common.element_to_ind["H"]
        for atom in range(self.natoms):
            if frag & (1 << atom):
                dense_pos = common.element_to_ind[self.atom_symbols[atom]]
                form_vec[dense_pos] += 1
                form_vec[h_pos] += self.atom_hs[atom]
        form_vec[h_pos] += h_shift
        form = common.vec_to_formula(form_vec)
        return form

    def formula_from_kept_inds(self, kept_inds):
        form_vec = np.zeros(len(common.VALID_ELEMENTS))
        h_count = self.atom_hs[kept_inds].sum()
        h_pos = common.element_to_ind["H"]
        form_vec[h_pos] = h_count
        atom_cts = Counter(self.atom_symbols_ar[kept_inds])
        for atom_type, atom_ct in atom_cts.items():
            form_vec[common.element_to_ind[atom_type]] = atom_ct
        form = common.vec_to_formula(form_vec)
        return form

    def atom_pass_stats(self, frag: int, depth: int = None):
        fragment_mass = 0.0
        form_vec = np.zeros(len(common.VALID_ELEMENTS))
        h_pos = common.element_to_ind["H"]
        for atom in range(self.natoms):
            if frag & (1 << atom):
                fragment_mass += self.atom_weights_h[atom]
                dense_pos = common.element_to_ind[self.atom_symbols[atom]]
                form_vec[dense_pos] += 1
                form_vec[h_pos] += self.atom_hs[atom]
        form = common.vec_to_formula(form_vec)
        frag_hs = int(form_vec[h_pos])

        max_remove = int(min(frag_hs, self.max_broken_bonds))
        max_add = int(min(self.total_hs - frag_hs, self.max_broken_bonds))

        if depth is not None:
            max_remove = int(min(depth, max_remove))
            max_add = int(min(depth, max_add))

        return {
            "form": form,
            "base_mass": float(fragment_mass),
            "frag_hs": frag_hs,
            "max_remove_hs": max_remove,
            "max_add_hs": max_add,
        }

    def wl_hash(self, template_fragment: int) -> int:
        cur_hashes = [
            f"{i}"
            for i, j, k in zip(self.atom_symbols, self.atom_hs, self.bonded_atoms)
        ]

        def get_graph_hash(full_hashes):
            counter = Counter(full_hashes)
            counter_str = str(tuple(sorted(counter.items(), key=lambda x: x[0])))
            counter_hash = _hash_label(counter_str)
            return counter_hash

        graph_hash = get_graph_hash(cur_hashes)
        iterations = self.natoms
        changed = True
        ct = 0

        while ct <= iterations and changed:
            new_hashes = []
            temp_atoms = 0
            for atom in range(self.natoms):
                atombit = 1 << atom
                cur_hash = cur_hashes[atom]

                if not atombit & template_fragment:
                    new_hashes.append(cur_hash)
                    continue

                temp_atoms += 1

                neighbor_labels = []
                for targind in self.bonded_atoms[atom]:
                    targbit = 1 << targind
                    if not targbit & template_fragment:
                        continue
                    targhash = cur_hashes[targind]
                    bondbit = targbit | atombit
                    bondtype = self.bond_to_type[bondbit]
                    neighbor_label = f"{bondtype}_{targhash}"
                    neighbor_labels.append(neighbor_label)
                new_hash_str = cur_hash + "".join(sorted(neighbor_labels))
                new_hash = _hash_label(new_hash_str)
                new_hashes.append(new_hash)

            iterations = temp_atoms
            new_graph_hash = get_graph_hash(new_hashes)
            changed = new_graph_hash != graph_hash

            graph_hash = new_graph_hash
            cur_hash = new_hashes
            ct += 1
        return graph_hash

    def get_frag_masses(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        frag_ids, frag_inds, shift_inds, masses, scores = [], [], [], [], []
        for k, v in self.frag_to_entry.items():
            max_remove, max_add = v["max_remove_hs"], v["max_add_hs"]
            frag_int = v["frag"]
            base_mass = v["base_mass"]
            score = v["score"]
            for num_shift, shift_ind, shift_mass in zip(
                self.shift_buckets, self.shift_bucket_inds, self.shift_bucket_masses
            ):
                if (num_shift >= -max_remove) and (num_shift <= max_add):
                    frag_inds.append(frag_int)
                    shift_inds.append(shift_ind)
                    masses.append(base_mass + shift_mass)
                    scores.append(score)
                    frag_ids.append(k)
        return (
            np.array(frag_ids),
            np.array(frag_inds),
            np.array(shift_inds),
            np.array(masses),
            np.array(scores),
        )

    def get_frag_forms(self):
        masses, form_vecs = [], []
        form_set = set()

        for k, v in self.frag_to_entry.items():
            max_remove, max_add = v["max_remove_hs"], v["max_add_hs"]
            base_mass = v["base_mass"]
            base_form_str = v["form"]
            base_form_vec = common.formula_to_dense(base_form_str)

            for num_shift, shift_ind, shift_mass in zip(
                self.shift_buckets, self.shift_bucket_inds, self.shift_bucket_masses
            ):
                if (num_shift >= -max_remove) and (num_shift <= max_add):
                    new_form_vec = (
                        base_form_vec
                        + num_shift * common.ELEMENT_VECTORS[common.element_to_ind["H"]]
                    )
                    str_code = str(new_form_vec)
                    if str_code in form_set:
                        continue
                    masses.append(base_mass + shift_mass)
                    form_vecs.append(new_form_vec)
                    form_set.add(str_code)

        return np.array(form_vecs), np.array(masses)

    def get_root_frag(self) -> int:
        return (1 << self.natoms) - 1

    def generate_fragments(self):
        cur_id = 0
        frag = (1 << self.natoms) - 1
        root = {
            "frag": frag,
            "id": cur_id,
            "sibling_hashes": [],
            "parents": [],
            "parent_hashes": [],
            "parent_ind_removed": [],
            "max_broken": 0,
            "tree_depth": 0,
            "score": self.score_fragment(frag)[1],
        }
        root.update(self.atom_pass_stats(frag, depth=0))
        frag_hash = self.wl_hash(frag)

        self.frag_to_entry[frag_hash] = root
        current_fragments = [frag_hash]
        new_fragments = []

        for step in range(self.max_tree_depth):
            for frag_hash in current_fragments:
                cur_parent = self.frag_to_entry[frag_hash]["id"]
                fragment = self.frag_to_entry[frag_hash]["frag"]
                parent_broken = self.frag_to_entry[frag_hash]["max_broken"]
                cur_parent_hash = frag_hash

                for atom in range(self.natoms):
                    extended_fragments = self.remove_atom(fragment, atom)
                    sibling_hashes = set([i["new_hash"] for i in extended_fragments])
                    for frag_dict in extended_fragments:
                        removed_atom = frag_dict["removed_atom"]
                        new_frag_hash = frag_dict["new_hash"]
                        rm_bond_t = frag_dict["rm_bond_t"]
                        frag = frag_dict["new_frag"]

                        temp_sibs = list(sibling_hashes.difference([new_frag_hash]))

                        old_entry = self.frag_to_entry.get(new_frag_hash)
                        max_broken = parent_broken + rm_bond_t

                        if old_entry is None:
                            cur_id += 1
                            new_entry = {
                                "frag": frag,
                                "id": cur_id,
                                "parents": [cur_parent],
                                "parent_hashes": [cur_parent_hash],
                                "parent_ind_removed": [removed_atom],
                                "sibling_hashes": [temp_sibs],
                                "max_broken": max_broken,
                                "tree_depth": step + 1,
                                "score": self.score_fragment(frag)[1],
                            }
                            new_entry.update(
                                self.atom_pass_stats(frag, depth=max_broken)
                            )
                            self.frag_to_entry[new_frag_hash] = new_entry
                            new_fragments.append(new_frag_hash)
                        elif old_entry["max_broken"] == max_broken:
                            old_entry["parent_ind_removed"].append(removed_atom)
                            old_entry["parents"].append(cur_parent)
                            old_entry["parent_hashes"].append(cur_parent_hash)
                            old_entry["sibling_hashes"].append(temp_sibs)

            current_fragments = new_fragments
            new_fragments = []

    def remove_atom(self, fragment: int, atom: int) -> List[dict]:
        if not ((1 << atom) & fragment):
            return []

        template_fragment = fragment ^ (1 << atom)
        list_ext_atoms = set([])
        extended_fragments = []
        ext_atom_to_bo = {}

        for a in self.bonded_atoms[atom]:
            if (1 << a) & template_fragment:
                list_ext_atoms.add(a)
                bond_num = (1 << atom) | (1 << a)
                bond_type = self.bond_to_type[bond_num]
                if a in ext_atom_to_bo:
                    raise ValueError("Double iterating bond atoms")
                ext_atom_to_bo[a] = bond_type

        if len(list_ext_atoms) == 1:
            if template_fragment == 0:
                return []
            bo = next(iter(ext_atom_to_bo.values()))
            new_frag_hash = self.wl_hash(template_fragment)
            extended_fragments.append(
                {
                    "new_frag": template_fragment,
                    "new_hash": new_frag_hash,
                    "removed_atom": atom,
                    "rm_bond_t": bo,
                }
            )
        else:
            for a in list_ext_atoms:
                is_ring = False
                rm_bond_t = ext_atom_to_bo[a]
                for frag_dict in extended_fragments:
                    frag_id = frag_dict["new_frag"]
                    if (1 << a) & frag_id:
                        is_ring = True
                if not is_ring:
                    new_fragment = extend(a, self.bonded_atoms, template_fragment)
                    if new_fragment == 0:
                        continue
                    new_frag_hash = self.wl_hash(new_fragment)
                    extended_fragments.append(
                        {
                            "new_frag": new_fragment,
                            "new_hash": new_frag_hash,
                            "removed_atom": atom,
                            "rm_bond_t": rm_bond_t,
                        }
                    )
        return extended_fragments

    def export_edges(self, frag_hashes: List[int]) -> List[Tuple[int, int]]:
        edges = []
        explored = set(frag_hashes)
        for i in frag_hashes:
            entry = self.frag_to_entry[i]
            for p in entry["parent_hashes"]:
                if p in explored:
                    edges.append((p, i))
        return edges

    def export_edges_dict(self, frag_hashes: List[int]) -> Tuple[dict, dict]:
        incoming_edges, outgoing_edges = defaultdict(lambda: []), defaultdict(lambda: [])
        explored = set(frag_hashes)
        for i in frag_hashes:
            entry = self.frag_to_entry[i]
            for p in entry["parent_hashes"]:
                if p in explored:
                    incoming_edges[i].append(p)
                    outgoing_edges[p].append(i)
        return incoming_edges, outgoing_edges

    def get_present_atoms(self, frag: int):
        ret_inds, ret_symbs = [], []
        for atom in range(self.natoms):
            if (1 << atom) & frag:
                ret_inds.append(atom)
                ret_symbs.append(self.atom_symbols[atom])
        return ret_inds, ret_symbs

    def get_present_edges(self, frag: int):
        output_bonds = []
        output_bond_types = []
        for bond, bond_inds in zip(self.bonds_list, self.bond_inds_list):
            if (frag & bond) == bond:
                output_bonds.append(bond_inds)
                output_bond_types.append(self.bond_to_type[bond])
        return output_bond_types, output_bonds

    def get_atoms_hash(self, frag_hash: float):
        entry = self.frag_to_entry[frag_hash]
        frag = entry["frag"]
        return self.get_present_atoms(frag)

    def get_draw_dict(self, frag):
        keep_atoms, _ = self.get_present_atoms(frag)
        _, keep_bonds = self.get_present_edges(frag)
        keep_bonds = [self.mol.GetBondBetweenAtoms(*i).GetIdx() for i in keep_bonds]
        return dict(
            hatoms=keep_atoms, hbonds=keep_bonds, mol=self.mol, smiles=self.smiles
        )

    def frags_to_intens(self, frags: dict):
        mass_to_obj = defaultdict(lambda: {})
        for k, val in frags.items():
            masses = val["base_mass"] + self.shift_bucket_masses
            intens = val["intens"]
            for m, i in zip(masses, intens):
                if i <= 0:
                    continue
                cur_obj = mass_to_obj[m]
                if cur_obj.get("inten", 0) > 0:
                    if cur_obj.get("inten") < i:
                        cur_obj["frag_hash"] = k
                    cur_obj["inten"] += i
                else:
                    cur_obj["inten"] = i
                    cur_obj["frag_hash"] = k

        max_inten = max(*[i["inten"] for i in mass_to_obj.values()], 1e-9)
        mass_to_obj = {
            k: dict(inten=v["inten"] / max_inten, frag_hash=v["frag_hash"])
            for k, v in mass_to_obj.items()
        }

        out_entries = [dict(mz=k, **v) for k, v in mass_to_obj.items()]
        return out_entries


def extend(atom: int, bonded_atoms: list, template_fragment: int):
    stack = [atom]
    new_fragment = 0
    while len(stack) > 0:
        atom = stack.pop()
        for a in bonded_atoms[atom]:
            atombit = 1 << a
            if (not (atombit & template_fragment)) or (atombit & new_fragment):
                continue
            new_fragment = new_fragment | atombit
            stack.append(a)
    return new_fragment


def _hash_label(label, digest_size=32):
    return blake2b(label.encode("ascii"), digest_size=digest_size).hexdigest()


def bit_array(num):
    return np.array(list(f"{num:b}")).astype(float)


def create_new_ids(frags):
    frag_to_id = {
        i: id
        for id, i in enumerate(sorted(frags, key=lambda x: frags[x]["tree_depth"]))
    }
    id_to_frag = {id: i for i, id in frag_to_id.items()}
    return frag_to_id, id_to_frag

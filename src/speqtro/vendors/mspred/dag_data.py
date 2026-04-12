"""dag_data.py -- vendored from ms-pred (PyG version), imports fixed for speqtro.

Fragment dataset to build out model class.
"""
import logging
from pathlib import Path
from typing import List
import json
import copy
import functools
import random

import numpy as np

import torch
from torch.utils.data.dataset import Dataset
from torch_geometric.data import Data, Batch

from . import common
from . import nn_utils
from .magma import fragmentation


class TreeProcessor:
    """TreeProcessor.

    Hold key functionalities to read in a magma dag and process it.
    """

    def __init__(
        self,
        pe_embed_k: int = 10,
        root_encode: str = "gnn",
        binned_targs: bool = False,
        add_hs: bool = False,
        embed_elem_group: bool = False,
    ):
        self.pe_embed_k = pe_embed_k
        self.root_encode = root_encode
        self.binned_targs = binned_targs
        self.add_hs = add_hs
        self.embed_elem_group = embed_elem_group
        self.bins = np.linspace(0, 1500, 15000)

    def get_frag_info(
        self,
        frag: int,
        engine: fragmentation.FragmentEngine,
    ):
        num_atoms = engine.natoms
        kept_atom_inds, kept_atom_symbols = engine.get_present_atoms(frag)
        form = engine.formula_from_kept_inds(kept_atom_inds)

        num_kept = len(kept_atom_inds)
        new_inds = np.arange(num_kept)
        old_inds = kept_atom_inds

        old_to_new = np.zeros(num_atoms, dtype=int)
        old_to_new[old_inds] = new_inds

        new_to_old = np.zeros(num_kept, dtype=int)
        new_to_old[new_inds] = old_inds

        return {
            "new_to_old": new_to_old,
            "old_to_new": old_to_new,
            "form": form,
        }

    def featurize_frag(
        self,
        frag: int,
        engine: fragmentation.FragmentEngine,
        add_random_walk: bool = False,
    ) -> False:
        kept_atom_inds, kept_atom_symbols = engine.get_present_atoms(frag)
        atom_symbols = engine.atom_symbols
        kept_bond_orders, kept_bonds = engine.get_present_edges(frag)

        info = self.get_frag_info(frag, engine)
        old_to_new = info['old_to_new']

        new_bond_inds = np.empty((0, 2), dtype=int)
        if len(kept_bonds) > 0:
            new_bond_inds = old_to_new[np.vstack(kept_bonds)]

        if self.add_hs:
            h_adds = np.array(engine.atom_hs)[kept_atom_inds]
        else:
            h_adds = None

        graph = self.dgl_featurize(
            np.array(atom_symbols)[kept_atom_inds],
            h_adds=h_adds,
            bond_inds=new_bond_inds,
            bond_types=np.array(kept_bond_orders),
            embed_elem_group=self.embed_elem_group,
        )

        if add_random_walk:
            self.add_pe_embed(graph)

        frag_feature_dict = {
            "graph": graph,
        }
        frag_feature_dict.update(info)
        return frag_feature_dict

    def _convert_to_dgl(
        self,
        tree: dict,
        include_targets: bool = True,
        last_row=False,
    ):
        root_smiles = tree["root_canonical_smiles"]
        engine = fragmentation.FragmentEngine(mol_str=root_smiles, mol_str_type="smiles", mol_str_canonicalized=True)
        bottom_depth = engine.max_tree_depth
        if self.root_encode == "gnn":
            root_frag = engine.get_root_frag()
            root_graph_dict = self.featurize_frag(
                frag=root_frag,
                engine=engine,
            )
            root_repr = root_graph_dict["graph"]
        elif self.root_encode == "fp":
            root_repr = common.get_morgan_fp_smi(root_smiles)
        else:
            raise ValueError()

        root_form = common.form_from_smi(root_smiles)

        adduct_mass_shift = np.array([
            common.ion2mass[tree["adduct"]],
            -common.ELECTRON_MASS if common.is_positive_adduct(tree["adduct"]) else common.ELECTRON_MASS
        ])

        masses, inten_frag_ids, dgl_inputs, inten_targets, frag_targets, max_broken = (
            [], [], [], [], [], [],
        )
        forms = []
        max_remove_hs, max_add_hs = [], []
        for k, sub_frag in tree["frags"].items():
            max_broken_num = sub_frag["max_broken"]
            tree_depth = sub_frag["tree_depth"]

            if (not last_row) and (tree_depth == bottom_depth):
                continue

            binary_targs = sub_frag["atoms_pulled"]
            frag = sub_frag["frag"]

            frag_dict = self.featurize_frag(frag, engine)
            forms.append(frag_dict["form"])
            old_to_new = frag_dict["old_to_new"]
            graph = frag_dict["graph"]
            max_broken.append(max_broken_num)

            max_remove_hs.append(sub_frag["max_remove_hs"])
            max_add_hs.append(sub_frag["max_add_hs"])

            inten_frag_ids.append(k)

            targ_vec = np.zeros(graph.num_nodes)
            for j in old_to_new[binary_targs]:
                targ_vec[j] = 1

            graph = frag_dict["graph"]

            dgl_inputs.append(graph)
            masses.append(sub_frag["base_mass"])
            frag_targets.append(torch.from_numpy(targ_vec))

        if include_targets:
            inten_targets = np.array(tree["raw_spec"])

        masses = engine.shift_bucket_masses[None, None, :] + \
                 adduct_mass_shift[None, :, None] + \
                 np.array(masses)[:, None, None]
        max_remove_hs = np.array(max_remove_hs)
        max_add_hs = np.array(max_add_hs)
        max_broken = np.array(max_broken)

        all_form_vecs = [common.formula_to_dense(i) for i in forms]
        all_form_vecs = np.array(all_form_vecs)
        root_form_vec = common.formula_to_dense(root_form)

        out_dict = {
            "root_repr": root_repr,
            "dgl_frags": dgl_inputs,
            "masses": masses,
            "inten_targs": np.array(inten_targets) if include_targets else None,
            "inten_frag_ids": inten_frag_ids,
            "max_remove_hs": max_remove_hs,
            "max_add_hs": max_add_hs,
            "max_broken": max_broken,
            "targs": frag_targets,
            "form_vecs": all_form_vecs,
            "root_form_vec": root_form_vec,
        }
        return out_dict

    def _process_tree(
        self,
        tree: dict,
        include_targets: bool = True,
        last_row=False,
        convert_to_dgl=True,
    ):
        if convert_to_dgl:
            out_dict = self._convert_to_dgl(tree, include_targets, last_row)
            if "collision_energy" in tree:
                out_dict["collision_energy"] = tree["collision_energy"]
            if "instrument" in tree:
                out_dict["instrument"] = tree["instrument"]
        else:
            out_dict = tree

        dgl_inputs = out_dict["dgl_frags"]
        root_repr = out_dict["root_repr"]

        if self.pe_embed_k > 0:
            for graph in dgl_inputs:
                self.add_pe_embed(graph)

            if isinstance(root_repr, Data):
                self.add_pe_embed(root_repr)

        if include_targets and self.binned_targs:
            intens = out_dict["inten_targs"]
            bin_posts = np.clip(np.digitize(intens[:, 0], self.bins), 0, len(self.bins) - 1)
            new_out = np.zeros_like(self.bins)
            for bin_post, inten in zip(bin_posts, intens[:, 1]):
                new_out[bin_post] = max(new_out[bin_post], inten)
            inten_targets = new_out
            out_dict["inten_targs"] = inten_targets
        return out_dict

    def add_pe_embed(self, graph):
        pe_embeds = nn_utils.random_walk_pe(
            graph.edge_index,
            num_nodes=graph.num_nodes,
            k=self.pe_embed_k,
            edge_weight=graph.edge_type.float() if hasattr(graph, 'edge_type') and graph.edge_type is not None else None,
        )
        graph.x = torch.cat((graph.x, pe_embeds), -1).float()
        return graph

    def process_tree_gen(self, tree: dict, convert_to_dgl=True):
        proc_out = self._process_tree(
            tree, include_targets=False, last_row=False, convert_to_dgl=convert_to_dgl
        )
        keys = {
            "root_repr",
            "dgl_frags",
            "targs",
            "max_broken",
            "form_vecs",
            "root_form_vec",
            "collision_energy",
        }
        dgl_tree = {i: proc_out[i] for i in keys}
        return {"dgl_tree": dgl_tree, "tree": tree}

    def process_tree_inten(self, tree, convert_to_dgl=True):
        proc_out = self._process_tree(
            tree, include_targets=True, last_row=True, convert_to_dgl=convert_to_dgl
        )
        keys = {
            "root_repr",
            "dgl_frags",
            "masses",
            "inten_targs",
            "inten_frag_ids",
            "max_remove_hs",
            "max_add_hs",
            "max_broken",
            "form_vecs",
            "root_form_vec",
            "collision_energy",
        }
        dgl_tree = {i: proc_out[i] for i in keys}
        return {"dgl_tree": dgl_tree, "tree": tree}

    def process_tree_inten_pred(self, tree: dict, convert_to_dgl=True):
        proc_out = self._process_tree(
            tree, include_targets=False, last_row=True, convert_to_dgl=convert_to_dgl
        )
        keys = {
            "root_repr",
            "dgl_frags",
            "masses",
            "inten_targs",
            "inten_frag_ids",
            "max_remove_hs",
            "max_add_hs",
            "max_broken",
            "form_vecs",
            "root_form_vec",
            "collision_energy",
        }
        dgl_tree = {i: proc_out[i] for i in keys}
        return {"dgl_tree": dgl_tree, "tree": tree}

    def dgl_featurize(
        self,
        atom_symbols: List[str],
        h_adds: np.ndarray,
        bond_inds: np.ndarray,
        bond_types: np.ndarray,
        embed_elem_group: bool,
    ):
        node_types = [common.element_to_position[el] for el in atom_symbols]
        node_types = np.vstack(node_types)
        num_nodes = node_types.shape[0]

        src, dest = bond_inds[:, 0], bond_inds[:, 1]
        src_tens_, dest_tens_ = torch.from_numpy(src), torch.from_numpy(dest)
        bond_types = torch.from_numpy(bond_types)
        src_tens = torch.cat([src_tens_, dest_tens_])
        dest_tens = torch.cat([dest_tens_, src_tens_])
        bond_types = torch.cat([bond_types, bond_types])
        bond_featurizer = torch.eye(fragmentation.MAX_BONDS)

        bond_types_onehot = bond_featurizer[bond_types.long()]
        node_data = torch.from_numpy(node_types)

        if embed_elem_group:
            node_groups = [common.element_to_group[el] for el in atom_symbols]
            node_groups = np.vstack(node_groups)
            node_data = torch.hstack([node_data, torch.from_numpy(node_groups)])

        if h_adds is None:
            zero_vec = torch.zeros((node_data.shape[0], common.MAX_H))
            node_data = torch.hstack([node_data, zero_vec])
        else:
            h_featurizer = torch.eye(common.MAX_H)
            h_adds_vec = torch.from_numpy(h_adds)
            node_data = torch.hstack([node_data, h_featurizer[h_adds_vec]])

        edge_index = torch.stack([src_tens, dest_tens], dim=0)  # (2, E)
        data = Data(
            x=node_data.float(),
            edge_index=edge_index,
            edge_attr=bond_types_onehot.float(),
            edge_type=bond_types.long(),
        )
        return data

    def get_node_feats(self):
        if self.embed_elem_group:
            return self.pe_embed_k + common.ELEMENT_DIM + common.MAX_H + common.ELEMENT_GROUP_DIM
        else:
            return self.pe_embed_k + common.ELEMENT_DIM + common.MAX_H


class IntenPredDataset:
    """IntenPredDataset — inference-only stub for collate_fn access."""

    @classmethod
    def get_collate_fn(cls):
        return _inten_collate_fn


def _inten_collate_fn(input_list):
    """Collate function for intensity prediction (inference)."""
    names = [j["name"] for j in input_list]
    frag_graphs = [j["dgl_frags"] for j in input_list]
    frag_graphs_e = [j for i in frag_graphs for j in i]
    num_frags = torch.LongTensor([len(i) for i in frag_graphs])
    frag_atoms = torch.LongTensor([i.num_nodes for i in frag_graphs_e])

    root_reprs = [j["root_repr"] for j in input_list]
    if isinstance(root_reprs[0], Data):
        batched_reprs = Batch.from_data_list(root_reprs)
    elif isinstance(root_reprs[0], np.ndarray):
        batched_reprs = torch.FloatTensor(np.vstack(root_reprs)).float()
    else:
        raise NotImplementedError()

    frag_batch = Batch.from_data_list(frag_graphs_e)
    root_inds = torch.arange(len(frag_graphs)).repeat_interleave(num_frags)

    inten_frag_ids = None
    if input_list[0].get("inten_frag_ids") is not None:
        inten_frag_ids = [i["inten_frag_ids"] for i in input_list]

    masses_padded = _unroll_pad(input_list, "masses")
    max_remove_hs_padded = _unroll_pad(input_list, "max_remove_hs")
    max_add_hs_padded = _unroll_pad(input_list, "max_add_hs")

    max_broken = [torch.LongTensor(i["max_broken"]) for i in input_list]
    broken_padded = torch.nn.utils.rnn.pad_sequence(max_broken, batch_first=True)

    supply_adduct = "adduct" in input_list[0]
    if supply_adduct:
        adducts = [j["adduct"] for j in input_list]
        adducts = torch.FloatTensor(adducts)
    supply_instrument = "instrument" in input_list[0]
    if supply_instrument:
        instruments = [j["instrument"] for j in input_list]
        instruments = torch.FloatTensor(instruments)

    collision_engs = [float(j["collision_energy"]) for j in input_list]
    collision_engs = torch.FloatTensor(collision_engs)

    precursor_mzs = [j["precursor"] for j in input_list]
    precursor_mzs = torch.FloatTensor(precursor_mzs)

    form_vecs = _unroll_pad(input_list, "form_vecs")
    root_vecs = _unroll_pad(input_list, "root_form_vec")

    output = {
        "names": names,
        "root_reprs": batched_reprs,
        "frag_graphs": frag_batch,
        "frag_atoms": frag_atoms,
        "inds": root_inds,
        "num_frags": num_frags,
        "masses": masses_padded,
        "broken_bonds": broken_padded,
        "max_add_hs": max_add_hs_padded,
        "max_remove_hs": max_remove_hs_padded,
        "inten_frag_ids": inten_frag_ids,
        "adducts": adducts if supply_adduct else None,
        "collision_engs": collision_engs,
        "instruments": instruments if supply_instrument else None,
        "precursor_mzs": precursor_mzs,
        "root_form_vecs": root_vecs,
        "frag_form_vecs": form_vecs,
    }
    return output


def _unroll_pad(input_list, key):
    if input_list[0].get(key) is not None:
        out = [torch.FloatTensor(i[key]) if i.get(key) is not None
               else torch.FloatTensor(input_list[0][key] * float('nan'))
               for i in input_list]
        out_padded = torch.nn.utils.rnn.pad_sequence(out, batch_first=True)
        return out_padded
    return None

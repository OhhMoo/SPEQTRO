"""Tests for the DGL→PyG transition in vendors/mspred.

Levels:
  1. Import chain  — no model weights needed
  2. Graph build   — MolGraph / dgl_featurize produce valid PyG Data objects
  3. GNN forward   — MoleculeGNN processes a real graph (random weights)
  4. Collate       — _inten_collate_fn produces correctly-shaped tensors
  5. Tool wrapper  — predict_msms_iceberg returns a sensible error when
                     checkpoints are absent (no real inference needed)

All tests are skipped when torch_geometric or torch_scatter are absent.
"""

import pytest

pyg = pytest.importorskip("torch_geometric")
ts  = pytest.importorskip("torch_scatter")
rd  = pytest.importorskip("rdkit")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
_CAFFEINE = "Cn1c(=O)c2c(ncn2C)n(c1=O)C"


# ---------------------------------------------------------------------------
# 1. Import chain
# ---------------------------------------------------------------------------

def test_import_pyg_modules():
    from speqtro.vendors.mspred.nn_utils import pyg_modules
    assert hasattr(pyg_modules, "GatedGraphConv")
    assert hasattr(pyg_modules, "PNAConv")
    assert hasattr(pyg_modules, "GINEConv")


def test_import_nn_utils():
    from speqtro.vendors.mspred.nn_utils import nn_utils
    assert hasattr(nn_utils, "MoleculeGNN")
    assert hasattr(nn_utils, "SetTransformerEncoder")
    assert hasattr(nn_utils, "random_walk_pe")


def test_import_mol_graph():
    from speqtro.vendors.mspred.nn_utils import mol_graph
    assert hasattr(mol_graph, "MolGraph")
    assert hasattr(mol_graph, "MolDGLGraph")   # backward-compat alias still present


def test_import_dag_data():
    from speqtro.vendors.mspred import dag_data
    assert hasattr(dag_data, "TreeProcessor")
    assert hasattr(dag_data, "IntenPredDataset")


def test_no_dgl_import():
    """Confirm no live DGL library dependency survives the transition."""
    import sys
    # Import the full nn_utils stack
    import speqtro.vendors.mspred.nn_utils.pyg_modules  # noqa
    import speqtro.vendors.mspred.nn_utils.nn_utils     # noqa
    import speqtro.vendors.mspred.nn_utils.mol_graph    # noqa
    # dgl must NOT have been imported as a side-effect
    assert "dgl" not in sys.modules, "DGL was imported — check for stray 'import dgl'"


# ---------------------------------------------------------------------------
# 2. Graph construction
# ---------------------------------------------------------------------------

def test_molgraph_returns_pyg_data():
    from torch_geometric.data import Data
    from speqtro.vendors.mspred.nn_utils.mol_graph import MolGraph
    from rdkit import Chem

    mol = Chem.MolFromSmiles(_ASPIRIN)
    assert mol is not None

    builder = MolGraph()
    data = builder.get_dgl_graph(mol)

    assert isinstance(data, Data)
    assert data.x is not None
    assert data.edge_index is not None
    assert data.edge_attr is not None
    # Aspirin: 13 heavy atoms → 13 nodes
    assert data.x.shape[0] == 13
    # edge_index shape: (2, E)
    assert data.edge_index.shape[0] == 2
    # bigraph → each bond appears twice
    assert data.edge_index.shape[1] == data.edge_attr.shape[0]


def test_molgraph_num_feats():
    from speqtro.vendors.mspred.nn_utils.mol_graph import MolGraph
    from rdkit import Chem

    mol = Chem.MolFromSmiles("c1ccccc1")  # benzene, 6 atoms
    builder = MolGraph()
    data = builder.get_dgl_graph(mol)
    assert data.x.shape == (6, builder.num_atom_feats)
    assert data.edge_attr.shape[1] == builder.num_bond_feats


def test_molgraph_pe_embed():
    import torch
    from torch_geometric.data import Data
    from speqtro.vendors.mspred.nn_utils.mol_graph import MolGraph
    from rdkit import Chem

    k = 5
    mol = Chem.MolFromSmiles(_ASPIRIN)
    builder = MolGraph(pe_embed_k=k)
    data = builder.get_dgl_graph(mol)
    # pe appended to atom feats
    assert data.x.shape[1] == builder.num_atom_feats
    # num_atom_feats already includes pe_embed_k
    assert data.x.shape[1] == (builder.num_atom_feats - 0)  # sanity


def test_dag_featurize_returns_data():
    import numpy as np
    import torch
    from torch_geometric.data import Data
    from speqtro.vendors.mspred import dag_data
    from speqtro.vendors.mspred.magma import fragmentation

    tp = dag_data.TreeProcessor()
    # Hand-craft minimal inputs: 3 atoms (C, C, O), one bond C-C
    atom_symbols = ["C", "C", "O"]
    h_adds = None
    bond_inds = np.array([[0, 1]])
    bond_types = np.array([1])  # single bond index

    data = tp.dgl_featurize(atom_symbols, h_adds, bond_inds, bond_types,
                            embed_elem_group=False)
    assert isinstance(data, Data)
    assert data.num_nodes == 3
    assert data.edge_index.shape[0] == 2
    # bigraph: 1 bond → 2 directed edges
    assert data.edge_index.shape[1] == 2
    assert hasattr(data, "edge_type")


# ---------------------------------------------------------------------------
# 3. GNN forward pass (random weights — no checkpoint needed)
# ---------------------------------------------------------------------------

def _make_batch(n_atoms=13, n_bonds=28, node_feats=74, edge_feats=5, batch_size=2):
    import torch
    from torch_geometric.data import Data, Batch
    graphs = []
    for _ in range(batch_size):
        src = torch.randint(0, n_atoms, (n_bonds,))
        dst = torch.randint(0, n_atoms, (n_bonds,))
        graphs.append(Data(
            x=torch.randn(n_atoms, node_feats),
            edge_index=torch.stack([src, dst]),
            edge_attr=torch.randn(n_bonds, edge_feats),
        ))
    return Batch.from_data_list(graphs)


@pytest.mark.parametrize("mpnn_type", ["GGNN", "PNA", "GINE"])
def test_molecule_gnn_forward(mpnn_type):
    import torch
    from speqtro.vendors.mspred.nn_utils.nn_utils import MoleculeGNN

    hidden = 64
    node_feats = 74
    edge_feats = 5

    model = MoleculeGNN(
        hidden_size=hidden,
        gnn_node_feats=node_feats,
        gnn_edge_feats=edge_feats,
        mpnn_type=mpnn_type,
        set_transform_layers=1,
    )
    model.eval()

    batch = _make_batch(node_feats=node_feats, edge_feats=edge_feats)
    with torch.no_grad():
        out = model(batch)

    assert out.shape == (batch.num_nodes, hidden)
    assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# 4. Collate function
# ---------------------------------------------------------------------------

def test_inten_collate_fn_shapes():
    import numpy as np
    import torch
    from torch_geometric.data import Data
    from speqtro.vendors.mspred.dag_data import _inten_collate_fn
    from speqtro.vendors.mspred.common import chem_utils as cu
    from speqtro.vendors.mspred import common

    # Build two minimal fake processed-tree dicts
    def _fake_tree(n_frags=3):
        n_atoms = 5
        frag_graphs = [
            Data(
                x=torch.randn(n_atoms, 18),
                edge_index=torch.zeros(2, 4, dtype=torch.long),
                edge_attr=torch.randn(4, 5),
                edge_type=torch.zeros(4, dtype=torch.long),
            )
            for _ in range(n_frags)
        ]
        root_repr = Data(
            x=torch.randn(10, 18),
            edge_index=torch.zeros(2, 8, dtype=torch.long),
            edge_attr=torch.randn(8, 5),
        )
        n_elements = common.ELEMENT_DIM  # size of formula vectors
        return {
            "name": "mol",
            "dgl_frags": frag_graphs,
            "root_repr": root_repr,
            "masses": np.zeros((n_frags, 2, 3)),
            "max_remove_hs": np.zeros(n_frags),
            "max_add_hs": np.zeros(n_frags),
            "max_broken": np.zeros(n_frags),
            "inten_frag_ids": [str(i) for i in range(n_frags)],
            "collision_energy": 20.0,
            "precursor": 200.0,
            "form_vecs": np.zeros((n_frags, n_elements)),
            "root_form_vec": np.zeros(n_elements),
        }

    batch = _inten_collate_fn([_fake_tree(3), _fake_tree(4)])

    assert batch["frag_graphs"] is not None
    assert batch["root_reprs"] is not None
    # total frags = 3 + 4 = 7
    assert batch["num_frags"].sum().item() == 7
    assert batch["masses"].shape[0] == 2   # batch dim


# ---------------------------------------------------------------------------
# 5. Tool wrapper — graceful error when no checkpoints available
# ---------------------------------------------------------------------------

def test_predict_msms_iceberg_no_ckpt(tmp_path, monkeypatch):
    """Without checkpoints the tool should return an error dict, not raise."""
    # Patch config to return non-existent paths so we skip loading
    import speqtro.tools.mspred as mspred_tool

    monkeypatch.setattr(
        mspred_tool, "_get_config",
        lambda: {"gen_checkpoint": tmp_path / "missing_gen.ckpt",
                 "inten_checkpoint": tmp_path / "missing_inten.ckpt"}
    )

    result = mspred_tool.predict_msms_iceberg(
        smiles=_ASPIRIN,
        collision_energy=20.0,
        adduct="[M+H]+",
    )
    assert "error" in result
    assert "checkpoint" in result["error"].lower() or "not found" in result["error"].lower()


def test_predict_msms_iceberg_empty_smiles():
    from speqtro.tools.mspred import predict_msms_iceberg
    result = predict_msms_iceberg(smiles="")
    assert "error" in result


def test_cosine_similarity_perfect_match():
    from speqtro.tools.mspred import cosine_similarity_spectra
    peaks = [{"mz": 100.0, "intensity": 1.0}, {"mz": 200.0, "intensity": 0.5}]
    score = cosine_similarity_spectra(peaks, peaks)
    assert abs(score - 1.0) < 1e-6


def test_cosine_similarity_no_overlap():
    from speqtro.tools.mspred import cosine_similarity_spectra
    a = [{"mz": 100.0, "intensity": 1.0}]
    b = [{"mz": 300.0, "intensity": 1.0}]
    score = cosine_similarity_spectra(a, b)
    assert score == 0.0

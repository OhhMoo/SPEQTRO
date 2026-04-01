# Spectroscopy Reasoning Agent: Complete Technical Blueprint

---

## Part 1: Naming

### Naming Criteria
- Short (2-4 chars for CLI command)
- Memorable for chemists
- Available as PyPI package + .com domain (verify before committing)
- Sounds like a tool, not a company

### Top Candidates

| Name | CLI cmd | Vibe | Notes |
|---|---|---|---|
| **Spectra** | `sp` | Clean, obvious | Might be taken on PyPI |
| **Peakwise** | `pw` | Smart, analytical | "wise" implies intelligence |
| **Shiftly** | `sf` | Modern, playful | References chemical shifts |
| **Resonance** | `rs` | Elegant, scientific | NMR = nuclear magnetic *resonance* |
| **Elucid** | `el` | Sharp, purposeful | From "structure *elucidation*" |
| **Assigno** | `ag` | Action-oriented | Chemists "assign" peaks to atoms |
| **Nucleus** | `nc` | Bold, fundamental | NMR is about nuclei |
| **Sigmax** | `sx` | Technical, punchy | σ (sigma) = shielding in NMR |
| **Speq** | `speq` | Compact, unique | Spec + Query |
| **Peakstack** | `ps` | Developer-friendly | Peaks = spectral features |

### My top 3 recommendations

1. **`speq`** — Short, unique, pronounceable ("speck"), easy to type, likely available. "Spectroscopy Query" — exactly what it does.

2. **`elucid`** — Premium feel, directly references the core task (structure elucidation), memorable. `el` as the CLI command is fast.

3. **`shiftly`** — Friendly, modern, references chemical shifts. Good for marketing. `sf` as CLI command.

For this document I'll use **`speq`** as the working name.

---

## Part 2: CLI Composition & Architecture

### 2.1 Tech Stack

```
speq-cli/
├── pyproject.toml              # Package config (use hatch or setuptools)
├── README.md                   # Demo GIF + quick start
├── src/
│   └── speq/
│       ├── __init__.py
│       ├── cli.py              # Click CLI entry points
│       ├── agent.py            # Core reasoning agent
│       ├── planner.py          # Claude API integration for planning
│       ├── config.py           # User settings, API keys
│       ├── session.py          # Conversation state, history
│       ├── report.py           # Markdown/HTML report generation
│       ├── display.py          # Rich terminal rendering
│       │
│       ├── core/               # Framework internals
│       │   ├── tool.py         # Base Tool class
│       │   ├── registry.py     # Auto-discovers tools
│       │   ├── result.py       # ToolResult with citations
│       │   ├── molecule.py     # Molecular input handler (SMILES, SDF, MOL)
│       │   └── spectrum.py     # Spectral data loader (CSV, JCAMP-DX, Bruker)
│       │
│       ├── tools/              # ALL TOOLS LIVE HERE (one file each)
│       │   ├── _template.py    # Copy this to make a new tool
│       │   ├── nmr_prediction/
│       │   │   ├── h1_shift_predictor.py
│       │   │   ├── c13_shift_predictor.py
│       │   │   ├── coupling_constant_predictor.py
│       │   │   ├── hsqc_predictor.py
│       │   │   └── cosy_predictor.py
│       │   ├── structure_verification/
│       │   │   ├── dp4_calculator.py
│       │   │   ├── shift_deviation_scorer.py
│       │   │   ├── assignment_validator.py
│       │   │   └── regioisomer_checker.py
│       │   ├── structure_elucidation/
│       │   │   ├── substructure_detector.py
│       │   │   ├── fragment_assembler.py
│       │   │   ├── candidate_ranker.py
│       │   │   └── molecular_formula.py
│       │   ├── mass_spec/
│       │   │   ├── formula_from_hrms.py
│       │   │   ├── fragmentation_predictor.py
│       │   │   ├── isotope_pattern.py
│       │   │   └── adduct_calculator.py
│       │   ├── ir_raman/
│       │   │   ├── functional_group_identifier.py
│       │   │   ├── ir_fingerprint_matcher.py
│       │   │   └── carbonyl_classifier.py
│       │   ├── qm_prediction/
│       │   │   ├── dft_nmr_shifts.py       # GIAO via ORCA/xTB
│       │   │   ├── conformer_search.py      # CREST/RDKit
│       │   │   ├── boltzmann_weighting.py
│       │   │   └── solvent_correction.py
│       │   ├── database/
│       │   │   ├── nmrshiftdb_search.py
│       │   │   ├── sdbs_search.py
│       │   │   ├── pubchem_lookup.py
│       │   │   ├── chembl_lookup.py
│       │   │   └── cas_lookup.py
│       │   ├── molecular/
│       │   │   ├── smiles_parser.py
│       │   │   ├── molecular_formula.py
│       │   │   ├── functional_groups.py
│       │   │   ├── stereocenter_detector.py
│       │   │   └── tautomer_enumerator.py
│       │   └── reporting/
│       │       ├── structure_confirmation.py
│       │       ├── impurity_report.py
│       │       └── batch_comparison.py
│       │
│       └── models/             # Your proprietary ML models (added over time)
│           ├── __init__.py
│           ├── shift_gnn.py    # GNN for chemical shift prediction
│           ├── struct_elucidator.py  # Transformer for spectra→structure
│           └── functional_group_classifier.py
│
├── tests/
│   ├── test_tools/
│   ├── test_agent/
│   ├── test_models/
│   └── benchmarks/             # Accuracy benchmarks
│       ├── nmrshiftdb2_benchmark.py
│       └── structure_elucidation_benchmark.py
│
└── data/
    └── README.md               # Instructions for downloading datasets
```

### 2.2 CLI Commands (Click-based)

```python
# src/speq/cli.py

import click
from rich.console import Console

console = Console()

@click.group()
@click.version_option()
def cli():
    """speq — The spectroscopy reasoning engine for your terminal."""
    pass


# ── Main interactive mode ────────────────────────────────
@cli.command()
@click.argument("query", required=False)
@click.option("--smiles", "-s", help="Molecular structure as SMILES")
@click.option("--h1", type=click.Path(), help="¹H NMR data (CSV, JCAMP-DX)")
@click.option("--c13", type=click.Path(), help="¹³C NMR data")
@click.option("--hsqc", type=click.Path(), help="HSQC 2D NMR data")
@click.option("--ms", type=click.Path(), help="Mass spectrum data")
@click.option("--ir", type=click.Path(), help="IR spectrum data")
@click.option("--sdf", type=click.Path(), help="Molecule file (SDF/MOL)")
@click.option("--continue", "resume", is_flag=True, help="Resume last session")
def ask(query, smiles, h1, c13, hsqc, ms, ir, sdf, resume):
    """Ask a spectroscopy question. Starts interactive mode if no query given."""
    # If no query, launch interactive REPL
    # If query provided, run single-shot and exit
    pass


# ── Setup & Config ────────────────────────────────────────
@cli.command()
@click.option("--api-key", help="Anthropic API key")
def setup(api_key):
    """Interactive setup wizard."""
    pass


@cli.command()
def doctor():
    """Check installation, dependencies, and API connectivity."""
    pass


# ── Tool Management ───────────────────────────────────────
@cli.command()
@click.option("--category", "-c", help="Filter by category")
@click.option("--check", is_flag=True, help="Verify all dependencies")
def tools(category, check):
    """List available tools and their status."""
    pass


# ── Data Management ───────────────────────────────────────
@cli.group()
def data():
    """Download and manage reference datasets."""
    pass

@data.command()
@click.argument("dataset", type=click.Choice([
    "nmrshiftdb2", "sdbs", "nmrexp", "pubchem-nmr",
    "nist-ir", "nist-ms", "functional-groups"
]))
def pull(dataset):
    """Download a reference dataset."""
    pass


# ── Reports ───────────────────────────────────────────────
@cli.group()
def report():
    """Manage and publish analysis reports."""
    pass

@report.command()
def publish():
    """Convert latest report to self-contained HTML."""
    pass


# ── Sessions ──────────────────────────────────────────────
@cli.command()
def sessions():
    """List past analysis sessions."""
    pass


# Entry point
def main():
    cli()
```

### 2.3 What a Session Looks Like (User Experience)

```
$ speq

  ╭──────────────────────────────────────────────╮
  │  speq v0.1.0 — spectroscopy reasoning engine │
  │  Type /help for commands, or ask a question   │
  ╰──────────────────────────────────────────────╯

speq > I ran a reaction and got ¹H NMR in CDCl3: 
       7.35 (m, 5H), 5.12 (s, 2H), 4.21 (q, J=7.1, 2H), 
       1.28 (t, J=7.1, 3H). MW from HRMS is 180.0786.
       What's my product?

  ┌ Planning ──────────────────────────────────────┐
  │ 1. Parse NMR peaks and molecular formula        │
  │ 2. Identify functional groups from shift ranges  │
  │ 3. Determine molecular formula from exact mass   │
  │ 4. Search nmrshiftdb2 for candidates             │
  │ 5. Predict shifts for top candidates              │
  │ 6. Rank by spectral match + formula consistency   │
  └────────────────────────────────────────────────┘

  ▸ Running: molecular_formula_from_hrms ... done (0.1s)
    → C₉H₁₀O₃ (Δ = 0.2 ppm) [tool: formula_from_hrms]

  ▸ Running: functional_group_identifier ... done (0.3s)
    → Monosubstituted phenyl (7.35, 5H)
    → OCH₂ benzyl type (5.12, s, 2H) 
    → Ethyl ester (4.21 q + 1.28 t, J=7.1 Hz)
    [tool: functional_group_identifier]

  ▸ Running: nmrshiftdb_search ... done (1.2s)
    → 3 candidates within C₉H₁₀O₃ ± spectral match
    [tool: nmrshiftdb_search]

  ▸ Running: h1_shift_predictor (3 candidates) ... done (2.1s)
  ▸ Running: shift_deviation_scorer ... done (0.5s)

  ╭─ Result ───────────────────────────────────────╮
  │                                                 │
  │  Proposed structure: Benzyl ethyl carbonate      │
  │  SMILES: O=C(OCc1ccccc1)OCC                     │
  │  Confidence: 94%                                 │
  │                                                 │
  │  Assignment:                                     │
  │  7.35 (m, 5H) → ArH [pred: 7.32-7.38]          │
  │  5.12 (s, 2H) → PhCH₂O [pred: 5.14]            │
  │  4.21 (q, 2H) → OCH₂CH₃ [pred: 4.19]           │
  │  1.28 (t, 3H) → OCH₂CH₃ [pred: 1.30]           │
  │                                                 │
  │  All shifts within 0.05 ppm of prediction.       │
  │  HRMS consistent with C₉H₁₀O₃.                  │
  │                                                 │
  │  Sources: [h1_shift_predictor: GNN-AIMNet2]      │
  │           [formula_from_hrms: exact mass calc]    │
  │           [nmrshiftdb_search: entry #42891]       │
  ╰────────────────────────────────────────────────╯

  Report saved → ~/.speq/reports/2026-03-24_session_001.md

speq > /export sdf
  → Exported to benzyl_ethyl_carbonate.sdf

speq > Could this be an isomer? Check all C9H10O3 possibilities.
  ...
```

### 2.4 Core Tool Base Class

```python
# src/speq/core/tool.py

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


class ToolCategory(Enum):
    NMR_PREDICTION = "nmr_prediction"
    STRUCTURE_VERIFICATION = "structure_verification"
    STRUCTURE_ELUCIDATION = "structure_elucidation"
    MASS_SPEC = "mass_spec"
    IR_RAMAN = "ir_raman"
    QM_PREDICTION = "qm_prediction"
    DATABASE = "database"
    MOLECULAR = "molecular"
    REPORTING = "reporting"


@dataclass
class ToolResult:
    """Every tool returns this. Citations are mandatory."""
    success: bool
    data: Dict[str, Any]
    citations: List[str]         # e.g. ["tool: h1_shift_predictor, method: GNN-v1"]
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None
    execution_time_ms: float = 0

    @classmethod
    def ok(cls, data: dict, citations: list, **kwargs):
        return cls(success=True, data=data, citations=citations, **kwargs)

    @classmethod
    def fail(cls, error: str):
        return cls(success=False, data={}, citations=[], error=error)


class Tool(ABC):
    """Base class for all speq tools. One file = one tool."""

    name: str                    # unique identifier
    description: str             # what Claude sees when selecting tools
    category: ToolCategory
    requires: List[str] = []     # optional dependencies

    @abstractmethod
    def run(self, **kwargs) -> ToolResult:
        """Execute the tool. Must return ToolResult with citations."""
        pass

    def is_available(self) -> bool:
        """Check if all dependencies are installed."""
        for dep in self.requires:
            try:
                __import__(dep)
            except ImportError:
                return False
        return True
```

### 2.5 Example Tool Implementation

```python
# src/speq/tools/nmr_prediction/h1_shift_predictor.py

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from speq.core.tool import Tool, ToolResult, ToolCategory


class H1ShiftPredictor(Tool):
    """Predict ¹H NMR chemical shifts for a molecule."""

    name = "h1_shift_predictor"
    description = (
        "Predicts ¹H NMR chemical shifts for all hydrogen atoms in a molecule. "
        "Returns per-atom shift values in ppm with confidence intervals. "
        "Use for structure verification or candidate ranking."
    )
    category = ToolCategory.NMR_PREDICTION
    requires = ["rdkit"]

    def __init__(self):
        self._model = None  # lazy-loaded

    def _load_model(self):
        """Load the GNN model for shift prediction."""
        if self._model is None:
            try:
                # Try proprietary model first (if installed)
                from speq.models.shift_gnn import ShiftGNN
                self._model = ShiftGNN.load_pretrained("h1_gnn_v1")
                self._method = "speq-GNN-v1"
            except ImportError:
                # Fallback to heuristic (additive increment) method
                from speq.tools.nmr_prediction._heuristic import AdditiveH1
                self._model = AdditiveH1()
                self._method = "additive-increment"

    def run(self, smiles: str, solvent: str = "CDCl3") -> ToolResult:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ToolResult.fail(f"Invalid SMILES: {smiles}")

        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)

        self._load_model()
        predictions = self._model.predict(mol, solvent=solvent)

        # Group by equivalent hydrogens
        grouped = self._group_equivalent_h(mol, predictions)

        return ToolResult.ok(
            data={
                "shifts": grouped,          # [{"shift": 7.35, "count": 5, "atoms": [3,4,5,6,7]}]
                "solvent": solvent,
                "smiles": smiles,
            },
            citations=[
                f"tool: h1_shift_predictor, method: {self._method}, "
                f"solvent: {solvent}, n_atoms: {mol.GetNumAtoms()}"
            ]
        )

    def _group_equivalent_h(self, mol, predictions):
        """Group equivalent protons and average their predicted shifts."""
        # Use RDKit canonical ranking to find equivalent atoms
        # ...
        pass
```

---

## Part 3: Model Training

### 3.1 Training Data Landscape

| Dataset | Size | Content | Access | Use |
|---|---|---|---|---|
| **NMRexp** | 3.3M records | ¹H, ¹³C, ¹⁹F, ³¹P, ²⁹Si, ¹¹B shifts from 200K papers | Open | Primary training set for shift prediction |
| **nmrshiftdb2** | ~54K entries | Peer-reviewed assigned spectra + structures | Open (CC) | Validation, high-quality reference |
| **SDBS** | ~35K spectra | ¹H, ¹³C NMR + IR + MS for same compounds | Free (AIST Japan) | Multi-modal training |
| **2DNMRGym** | 22K+ HSQC | Annotated 2D HSQC spectra | Open | 2D NMR prediction |
| **NIST WebBook** | 16K+ IR spectra | Experimental IR + some NMR/MS | Free | IR model training |
| **PubChem** | 100M+ compounds | Structures, some with NMR data | Open | Database search, candidate generation |
| **DFT-computed shifts** | Generate yourself | GIAO-B3LYP/6-31G* on drug-like molecules | Self-generated | High-accuracy reference, your moat |
| **Patent NMR data** | 177K molecules | Computed IR + NMR from patent structures | Open (recent paper) | Cross-modal training |

**Key insight from recent research**: NMRexp provides 3.3 million experimental NMR records extracted from nearly 200,000 supporting information documents, with over 99% accuracy in metadata extraction — surpassing existing public NMR databases by over an order of magnitude. This is your primary training dataset.

### 3.2 Model 1: Chemical Shift Predictor (GNN)

This is the most important model — the engine behind structure verification and candidate ranking.

**Architecture: Message Passing Neural Network (MPNN)**

```
Input: Molecular graph (atoms = nodes, bonds = edges)
       + atom features (element, hybridization, charge, ring membership, aromaticity)
       + bond features (bond type, ring membership, conjugation)
       + global features (solvent, temperature)

                    ┌─────────────────────┐
                    │   Atom Embedding     │  (element → 64-dim vector)
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Message Passing     │  × 6 layers
                    │  (D-MPNN variant)    │  (aggregate neighbor info)
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Atom-Level Readout  │  (one prediction per H atom)
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Solvent Correction  │  (learned offset per solvent)
                    └──────────┬──────────┘
                               │
                    Output: δ (ppm) per hydrogen atom
                            + uncertainty estimate (σ)
```

**Implementation:**

```python
# src/speq/models/shift_gnn.py

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import Data


class ShiftMPNN(nn.Module):
    """
    Message Passing Neural Network for NMR chemical shift prediction.
    
    Predicts per-atom ¹H or ¹³C chemical shifts from molecular graph.
    Architecture inspired by Chemprop (D-MPNN) with modifications for
    NMR-specific features (solvent, temperature, atom environment).
    """

    def __init__(
        self,
        atom_features_dim: int = 45,    # element(16) + hybridization(6) + ...
        bond_features_dim: int = 12,     # bond_type(4) + ring(1) + ...
        hidden_dim: int = 256,
        n_message_passes: int = 6,
        n_solvents: int = 20,           # CDCl3, DMSO-d6, D2O, etc.
        dropout: float = 0.1,
    ):
        super().__init__()

        # Atom encoder
        self.atom_encoder = nn.Sequential(
            nn.Linear(atom_features_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        # Bond encoder
        self.bond_encoder = nn.Linear(bond_features_dim, hidden_dim)

        # Message passing layers
        self.message_layers = nn.ModuleList([
            MPNNLayer(hidden_dim, dropout) for _ in range(n_message_passes)
        ])

        # Solvent embedding (learned correction per solvent)
        self.solvent_embedding = nn.Embedding(n_solvents, hidden_dim)

        # Per-atom readout → chemical shift
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # atom_emb + solvent_emb
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2),  # [predicted_shift, log_uncertainty]
        )

    def forward(self, data: Data):
        """
        Args:
            data.x: atom features [n_atoms, atom_features_dim]
            data.edge_index: bond connectivity [2, n_bonds]
            data.edge_attr: bond features [n_bonds, bond_features_dim]
            data.solvent_id: solvent index [batch_size]
            data.h_mask: boolean mask for H atoms [n_atoms]
        
        Returns:
            shifts: predicted δ (ppm) for each H atom
            uncertainties: σ for each prediction
        """
        # Encode atoms and bonds
        h = self.atom_encoder(data.x)
        edge_attr = self.bond_encoder(data.edge_attr)

        # Message passing
        for layer in self.message_layers:
            h = layer(h, data.edge_index, edge_attr)

        # Get solvent context
        solvent_emb = self.solvent_embedding(data.solvent_id)
        # Broadcast solvent to all atoms in each molecule
        solvent_per_atom = solvent_emb[data.batch]

        # Concatenate atom embedding with solvent context
        h_with_solvent = torch.cat([h, solvent_per_atom], dim=-1)

        # Predict shift for each atom
        out = self.readout(h_with_solvent)  # [n_atoms, 2]

        # Only return predictions for H atoms (for ¹H model)
        # or C atoms (for ¹³C model)
        h_atoms = out[data.h_mask]
        shifts = h_atoms[:, 0]
        log_sigma = h_atoms[:, 1]
        uncertainties = torch.exp(log_sigma)

        return shifts, uncertainties


class MPNNLayer(MessagePassing):
    """Single message passing layer with edge features."""

    def __init__(self, hidden_dim, dropout):
        super().__init__(aggr="add")
        self.message_nn = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # source + target + edge
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.update_nn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # old + aggregated
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x, edge_index, edge_attr):
        agg = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return self.update_nn(torch.cat([x, agg], dim=-1))

    def message(self, x_j, x_i, edge_attr):
        return self.message_nn(torch.cat([x_j, x_i, edge_attr], dim=-1))
```

### 3.3 Training Process (Step by Step)

#### Step 1: Prepare NMRexp Dataset

```python
# scripts/prepare_nmrexp.py
"""
Download NMRexp (3.3M records) and convert to training format.
Each record: (SMILES, atom_index, experimental_shift, solvent, nucleus)
"""

import pandas as pd
from rdkit import Chem
from speq.data.featurizer import mol_to_graph

# Load NMRexp
raw = pd.read_parquet("data/nmrexp/nmrexp_h1.parquet")

# Filter: only ¹H shifts, valid SMILES, common solvents
valid = raw[
    (raw["nucleus"] == "1H") &
    (raw["smiles"].apply(lambda s: Chem.MolFromSmiles(s) is not None)) &
    (raw["solvent"].isin(COMMON_SOLVENTS))
]

# Convert to PyG Data objects
datasets = []
for smiles, group in valid.groupby("smiles"):
    mol = Chem.MolFromSmiles(smiles)
    graph = mol_to_graph(mol)  # atom features, bond features, edge_index
    graph.shifts = torch.tensor(group["shift"].values, dtype=torch.float)
    graph.atom_indices = torch.tensor(group["atom_idx"].values, dtype=torch.long)
    graph.solvent_id = SOLVENT_MAP[group["solvent"].iloc[0]]
    datasets.append(graph)

# Split: 80% train, 10% val, 10% test
# CRITICAL: split by SCAFFOLD (Murcko), not random
# This ensures model generalizes to new chemical scaffolds
train, val, test = scaffold_split(datasets, [0.8, 0.1, 0.1])

print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
# Expected: ~2.6M / 330K / 330K
```

#### Step 2: Train the ¹H Shift GNN

```bash
# Training command
python train_shift_model.py \
    --nucleus h1 \
    --data data/processed/nmrexp_h1/ \
    --hidden-dim 256 \
    --n-layers 6 \
    --batch-size 128 \
    --lr 3e-4 \
    --epochs 100 \
    --scheduler cosine \
    --early-stopping-patience 10 \
    --loss heteroscedastic_gaussian \
    --split scaffold \
    --output models/h1_gnn_v1/

# Loss function: heteroscedastic Gaussian NLL
# This trains both the shift prediction AND the uncertainty estimate
# L = log(σ) + (y_pred - y_true)² / (2σ²)
# → model learns to predict larger σ for harder atoms
```

**Expected training compute:**
- Dataset: ~2.6M training molecules
- Model: ~2M parameters
- Hardware: Single A100 GPU (or 4x T4 on cloud)
- Time: ~8-12 hours
- Cost: ~$30-50 on cloud GPU

#### Step 3: Train the ¹³C Shift GNN

Same architecture, different target. ¹³C shifts span 0-220 ppm (vs 0-12 for ¹H), so the loss scaling is different.

```bash
python train_shift_model.py \
    --nucleus c13 \
    --data data/processed/nmrexp_c13/ \
    --hidden-dim 256 \
    --n-layers 6 \
    --output models/c13_gnn_v1/
```

#### Step 4: Generate DFT Reference Data (Your Proprietary Moat)

Public ML models are trained on experimental data only. Your edge comes from having DFT-computed shifts as a calibration layer.

```python
# scripts/generate_dft_shifts.py
"""
Run GIAO NMR calculations on a curated set of drug-like molecules.
This creates your proprietary training data that nobody else has.
"""

from speq.qm import ORCARunner

# Pick 5,000 diverse drug-like molecules
molecules = load_diverse_druglike_set(n=5000)

for mol in molecules:
    # Step 1: Conformer search (CREST, fast)
    conformers = crest_conformer_search(mol, ewin=6.0)  # within 6 kcal/mol

    # Step 2: DFT optimization (B3LYP/6-31G*)
    optimized = []
    for conf in conformers[:10]:  # top 10 conformers
        opt = orca_optimize(conf, method="B3LYP", basis="6-31G*")
        optimized.append(opt)

    # Step 3: NMR shielding calculation (GIAO)
    for opt_conf in optimized:
        shieldings = orca_nmr_giao(opt_conf, method="B3LYP", basis="6-311+G(2d,p)")
        save_result(mol.smiles, opt_conf.energy, shieldings)

    # Step 4: Boltzmann-weighted average shifts
    weighted_shifts = boltzmann_average(optimized)
    save_final(mol.smiles, weighted_shifts)
```

**Compute estimate for 5,000 molecules:**
- ~50 conformers × 5,000 mols = 250,000 DFT jobs
- Each GIAO calculation: ~5-30 min on 4 cores
- Total: ~500-2000 CPU-hours
- Cost: ~$500-$2,000 on cloud (e.g., AWS c5.4xlarge)
- Timeline: 1-2 weeks with parallelization

### 3.4 Model 2: Functional Group Classifier (from spectra)

Given raw spectral peaks, identify which functional groups are present.

```
Input: List of (shift_ppm, multiplicity, integral, J_coupling) tuples
Output: Probability for each of ~50 functional group classes

Architecture: 1D CNN on binned spectrum + MLP on peak features

Training data: NMRexp + nmrshiftdb2 (structure→functional group labels from RDKit)
```

```python
class FunctionalGroupClassifier(nn.Module):
    """
    Identifies functional groups from ¹H NMR peak list.
    Multi-label classification (molecule can have multiple groups).
    """

    FUNCTIONAL_GROUPS = [
        "aromatic_monosub", "aromatic_disub_ortho", "aromatic_disub_meta",
        "aromatic_disub_para", "alkyl_ch3", "alkyl_ch2", "alkyl_ch",
        "alcohol_oh", "carboxylic_acid", "aldehyde", "ketone",
        "ester", "amide", "amine_primary", "amine_secondary",
        "amine_tertiary", "ether", "alkene", "alkyne",
        "nitrile", "nitro", "halide_f", "halide_cl", "halide_br",
        # ... ~50 total
    ]

    def __init__(self):
        super().__init__()

        # Branch 1: Binned spectrum (0-15 ppm, 0.01 ppm resolution = 1500 bins)
        self.spectrum_cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # Branch 2: Peak features (shift, mult, integral, J per peak, max 30 peaks)
        self.peak_encoder = nn.Sequential(
            nn.Linear(4, 32),  # per peak: shift, mult_code, integral, J
            nn.ReLU(),
        )
        self.peak_attention = nn.MultiheadAttention(32, num_heads=4, batch_first=True)

        # Combine and classify
        self.classifier = nn.Sequential(
            nn.Linear(128 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, len(self.FUNCTIONAL_GROUPS)),
        )

    def forward(self, binned_spectrum, peak_features, peak_mask):
        # Branch 1
        spec_emb = self.spectrum_cnn(binned_spectrum.unsqueeze(1)).squeeze(-1)

        # Branch 2
        peak_emb = self.peak_encoder(peak_features)
        peak_emb, _ = self.peak_attention(peak_emb, peak_emb, peak_emb,
                                           key_padding_mask=~peak_mask)
        peak_emb = peak_emb.mean(dim=1)  # pool over peaks

        combined = torch.cat([spec_emb, peak_emb], dim=-1)
        logits = self.classifier(combined)
        return torch.sigmoid(logits)  # multi-label probabilities
```

### 3.5 Model 3: Spectra-to-Structure Transformer (Later Phase)

This is the ambitious long-term model — give it spectra, get candidate structures.

**Architecture (inspired by NMR-Solver and the Stanford ACS Cent. Sci. paper):**

```
Input: Tokenized spectral data
       ¹H peaks: [H_7.35_m_5H] [H_5.12_s_2H] [H_4.21_q_2H_J7.1] [H_1.28_t_3H_J7.1]
       ¹³C peaks: [C_170.1] [C_136.2] [C_128.5] [C_128.3] [C_66.8] [C_61.0] [C_14.2]
       Molecular formula: [C9] [H10] [O3]

       ↓ Encoder (Transformer)

       Latent representation

       ↓ Decoder (autoregressive, generates SMILES token by token)

Output: "O=C(OCc1ccccc1)OCC"   (SMILES)
```

This model is trained on the NMRexp dataset + DFT-computed shifts. Training requires more compute (~$500-2000, 1-2 weeks on 4×A100), but builds your most defensible asset.

---

## Part 4: Testing & Evaluation

### 4.1 Benchmark Suite

Create a formal benchmark (like CellType's BixBench) to track progress and compare against baselines.

```python
# tests/benchmarks/speq_bench.py
"""
SpeqBench: Standardized evaluation for spectroscopy AI tools.

Test categories:
1. Shift prediction accuracy (MAE in ppm)
2. Structure verification (correct/incorrect classification)
3. Structure elucidation (top-k accuracy)
4. Functional group identification (F1 score)
5. End-to-end agent accuracy (full workflow)
"""


class SpeqBench:

    # ── Benchmark 1: Shift Prediction ─────────────────────
    def bench_h1_shift_prediction(self, model):
        """
        Test set: 500 molecules from nmrshiftdb2 (held out, scaffold split)
        Metric: Mean Absolute Error (MAE) in ppm
        
        Baselines:
        - Additive increment (heuristic): ~0.35 ppm
        - ChemProp (Chemprop GNN): ~0.22 ppm  
        - ACD/Labs HOSE code: ~0.18 ppm (commercial)
        - DFT GIAO B3LYP/6-311+G**: ~0.15 ppm (slow)
        
        Target: < 0.20 ppm (competitive with commercial, 100x faster than DFT)
        """
        results = []
        for mol, exp_shifts in self.h1_test_set:
            pred_shifts = model.predict(mol)
            mae = mean_absolute_error(exp_shifts, pred_shifts)
            results.append(mae)
        return np.mean(results)

    # ── Benchmark 2: Structure Verification ───────────────
    def bench_structure_verification(self, agent):
        """
        Test set: 200 (structure, spectrum) pairs
        - 100 correct structures
        - 100 wrong structures (regioisomers, stereoisomers, wrong connectivity)
        
        Metric: Classification accuracy (correct vs wrong)
        
        Baselines:
        - DP4 probability: ~85% accuracy
        - ACD/Labs ASV: ~90% accuracy
        
        Target: > 88% accuracy
        """
        correct = 0
        for structure, spectrum, is_correct in self.verification_test_set:
            result = agent.verify_structure(structure, spectrum)
            if result.is_match == is_correct:
                correct += 1
        return correct / len(self.verification_test_set)

    # ── Benchmark 3: Structure Elucidation ────────────────
    def bench_structure_elucidation(self, agent):
        """
        Test set: 100 (spectrum_only → structure) problems
        From published natural product elucidations and textbook problems.
        
        Metric: Top-1 and Top-5 accuracy
        
        Baselines:
        - Random: ~0% (combinatorial explosion)
        - Database search only: ~30% top-5
        - NMR-Solver: ~55% top-15
        - Multimodal transformer (NeurIPS 2025): ~96% top-1 (but simulated data)
        
        Target: > 60% top-5 on real experimental data
        """
        top1, top5 = 0, 0
        for spectrum, true_structure in self.elucidation_test_set:
            candidates = agent.elucidate(spectrum, top_k=5)
            if candidates[0] == true_structure:
                top1 += 1
            if true_structure in candidates[:5]:
                top5 += 1
        return top1 / len(self.elucidation_test_set), top5 / len(self.elucidation_test_set)

    # ── Benchmark 4: End-to-End Agent ─────────────────────
    def bench_agent_e2e(self, agent):
        """
        Test set: 50 real-world spectroscopy questions
        (from published papers, textbooks, and pharma QC labs)
        
        Each question has:
        - Input: natural language query + spectral data
        - Expected: correct structure/answer
        - Grading: automated (SMILES match) + expert review
        
        Metric: % correct answers
        
        Baselines:
        - Claude alone (no tools): ~30%
        - GPT-4o alone (no tools): ~25%
        
        Target: > 75%
        """
        pass
```

### 4.2 Running Benchmarks

```bash
# Run all benchmarks
speq bench run --all

# Run specific benchmark
speq bench run --shift-prediction --model models/h1_gnn_v1.pt

# Compare against baselines
speq bench compare --models h1_gnn_v1,heuristic,dft_giao

# Output:
# ┌──────────────────────────────┬──────────┬──────────────┐
# │ Model                        │ MAE (ppm)│ Time/mol (s) │
# ├──────────────────────────────┼──────────┼──────────────┤
# │ speq-GNN-v1                  │ 0.19     │ 0.02         │
# │ Additive increment (heuristic)│ 0.35    │ 0.001        │
# │ DFT GIAO B3LYP/6-311+G**    │ 0.14     │ 1800         │
# └──────────────────────────────┴──────────┴──────────────┘
```

### 4.3 Continuous Testing

```yaml
# .github/workflows/test.yml
name: Tests & Benchmarks

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install
        run: pip install -e ".[dev]"

      - name: Unit tests
        run: pytest tests/ -x --ignore=tests/benchmarks

      - name: Tool availability check
        run: speq doctor --ci

      - name: Quick benchmark (shift prediction on 50 molecules)
        run: speq bench run --shift-prediction --quick
```

---

## Part 5: Implementation Timeline

### Week 1: Skeleton + First 5 Tools

```
Day 1-2: Project setup
  - pyproject.toml, CLI skeleton (Click + Rich)
  - Tool base class, registry, result format
  - Config system (API key, settings)
  
Day 3-4: First 5 tools
  1. smiles_parser — validate + canonicalize SMILES
  2. molecular_formula — compute formula, MW, exact mass
  3. functional_groups — detect groups from structure (RDKit)
  4. h1_shift_predictor — heuristic version (additive increments)
  5. pubchem_lookup — search PubChem by name/formula/SMILES

Day 5-6: Wire up Claude
  - System prompt with tool descriptions
  - Planning: query → tool selection → execution order
  - Synthesis: tool results → grounded report
  
Day 7: Ship it
  - README with demo GIF (use asciinema or VHS)
  - pip install speq-cli on PyPI
  - GitHub repo public
```

### Week 2-4: Expand to 40+ Tools

```
- NMR prediction tools (additive heuristic, then ML when model ready)
- Mass spec tools (formula from HRMS, isotope pattern)
- Database search tools (nmrshiftdb2, SDBS, ChEMBL)
- IR functional group identifier
- Structure verification (DP4-like scoring)
- Spectral data loaders (JCAMP-DX, Bruker, CSV)
```

### Month 2-3: Train First GNN Model

```
- Download + process NMRexp dataset
- Train ¹H shift GNN (target: < 0.20 ppm MAE)
- Train ¹³C shift GNN
- Integrate models into CLI as speq-models package
- Publish benchmark results
```

### Month 3-5: Community + DFT Data Generation

```
- Run DFT GIAO calculations on 5,000 drug-like molecules
- Train calibration layer (ML prediction + DFT correction)
- Build SpeqBench and publish leaderboard
- Grow to 100+ tools
- First blog posts, demo videos
```

### Month 5-8: Structure Elucidation Model

```
- Train spectra→structure transformer
- Integrate into CLI as /elucidate command
- Publish paper or preprint on results
- Start enterprise conversations
```

---

## Part 6: Week-One Build Checklist

```
□ Create GitHub repo (speq-cli or your chosen name)
□ Set up pyproject.toml with Click, Rich, RDKit dependencies
□ Implement Tool base class + ToolResult + ToolRegistry
□ Implement 5 tools:
  □ smiles_parser
  □ molecular_formula  
  □ functional_groups (RDKit-based)
  □ h1_shift_predictor (heuristic/additive)
  □ pubchem_lookup
□ Wire up Claude API for planning + synthesis
□ Build interactive REPL (Rich-based terminal)
□ Record demo GIF
□ Write README (installation, 3 example queries)
□ Publish to PyPI
□ Post on Twitter/X, Reddit r/chemistry, Hacker News
```

**The single most important thing**: by end of week 1, this should work:

```bash
$ pip install speq-cli
$ export ANTHROPIC_API_KEY=sk-ant-...
$ speq "I have a compound with MW 180.08. ¹H NMR shows
        7.35 (m, 5H), 5.12 (s, 2H), 4.21 (q, 2H), 1.28 (t, 3H).
        What is it?"
```

And it should return a grounded, cited answer. That's your MVP.

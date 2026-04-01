# speq v2 — Spectroscopy Verification Agent: Technical Blueprint

---

## Part 1: Product Vision

### 1.1 The Core Insight

80%+ of NMR usage in synthetic chemistry is not open-ended structure elucidation. It is **verification**: the chemist knows what they tried to make, ran a reaction, and needs to confirm that the expected functional group transformation occurred and that the product is sufficiently pure for the next step.

speq v2 is redesigned around this reality. The primary workflow is **synthetic route tracking with step-by-step spectral verification**. Structure elucidation ("what is this mystery compound?") is available as an escape hatch — not the default mode.

### 1.2 Design Principles

1. **Integrate, don't reinvent.** Use state-of-the-art open-source ML models (ChefNMR, CASCADE 2.0) as tools rather than training from scratch. speq's value is orchestration, not prediction.
2. **Verification-first.** The default question is "did my reaction work?" not "what is this molecule?"
3. **Chemist-native inputs.** Accept Bruker folders, MestReNova exports, JCAMP-DX, spectrum images, peak lists, and SMILES. Don't force the chemist to transcribe peak tables.
4. **Mandatory citations.** Every claim traces back to a specific model, database, or calculation method.
5. **Route-aware context.** The agent tracks multi-step syntheses and accumulates knowledge across steps.

### 1.3 Working Name

**speq** — Spectroscopy Query. CLI command: `speq`. Short, pronounceable ("speck"), unique.

---

## Part 2: User Workflow

### 2.1 The Three Modes

```
┌─────────────────────────────────────────────────────────────────┐
│                        speq v2 Modes                            │
├─────────────────────┬──────────────────┬────────────────────────┤
│   VERIFY (default)  │   BTW (escape)   │   EXPLORE (freeform)  │
│                     │                  │                        │
│ "Did my reaction    │ "Wait, this      │ "What is this          │
│  work? Is the Boc   │  spectrum looks  │  unknown compound      │
│  group attached?"   │  wrong — what    │  from the natural      │
│                     │  did I actually  │  product extract?"     │
│ Input: expected     │  make?"          │                        │
│ structure + NMR     │                  │ Input: spectra only    │
│                     │ Triggered mid-   │ (no expected           │
│ Output: yes/no      │ session when     │  structure)            │
│ with confidence,    │ verification     │                        │
│ purity estimate     │ fails or chemist │ Output: candidate      │
│                     │ flags anomaly    │ structures, ranked     │
└─────────────────────┴──────────────────┴────────────────────────┘
```

### 2.2 VERIFY Mode — The Primary Workflow

This is what 80% of users will use 80% of the time. The workflow has three phases:

**Phase 1: Define the Synthetic Route**

The chemist sets up their synthesis by defining steps. Each step specifies a starting material, a reaction type, and the expected product.

```
$ speq route new "Kinase inhibitor synthesis"

speq > Step 1: Starting material?
user  > CCO (ethanol)

speq > Reaction type?
user  > Esterification with benzoic acid

speq > Expected product SMILES? (or draw at localhost:8080)
user  > O=C(OCC)c1ccccc1

  ✓ Step 1 defined: Esterification
    SM: CCO (ethanol)
    Product: O=C(OCC)c1ccccc1 (ethyl benzoate)
    
    Diagnostic markers for verification:
    • NEW: Ester C=O expected at 165-170 ppm (¹³C), no ¹H
    • NEW: OCH₂ quartet expected at 4.1-4.4 ppm (¹H)
    • LOST: Broad O-H of ethanol at 1-3 ppm (¹H)
    • LOST: Acid O-H of benzoic acid at ~11-12 ppm (¹H)
    • RETAINED: Aromatic protons at 7.4-8.1 ppm (¹H)

speq > Add another step? (y/n)
```

The agent pre-computes the diagnostic spectral markers for each step using shift prediction models. This means it knows what to look for *before* seeing any experimental data.

**Phase 2: Upload and Verify**

After running the reaction, the chemist uploads their NMR data:

```
$ speq verify step-1 --h1 ./bruker/exp_001/ --solvent CDCl3

  ┌ Verification Report: Step 1 — Esterification ─────────┐
  │                                                         │
  │  Functional Group Check                                 │
  │  ═══════════════════                                    │
  │  ✅ Ester C=O:  CONFIRMED                              │
  │     → ¹³C peak at 166.4 ppm (pred: 166.1 ± 1.2 ppm)   │
  │     → Source: CASCADE-2.0, MAE 0.73 ppm                 │
  │                                                         │
  │  ✅ OCH₂ (ethyl):  CONFIRMED                           │
  │     → ¹H quartet at 4.37 ppm, J=7.1 Hz (pred: 4.35)    │
  │     → ¹H triplet at 1.38 ppm, J=7.1 Hz (pred: 1.36)    │
  │                                                         │
  │  ✅ Aromatic H:  5H, monosubstituted pattern            │
  │     → 8.05 (d, 2H), 7.56 (t, 1H), 7.44 (t, 2H)        │
  │                                                         │
  │  ✅ No residual acid O-H at 11-12 ppm                   │
  │  ✅ No residual ethanol O-H                             │
  │                                                         │
  │  Confidence: 97%  │  All diagnostic markers matched.    │
  │                                                         │
  │  Purity Estimate                                        │
  │  ════════════════                                       │
  │  Estimated purity: ~92%                                 │
  │  Minor impurity peaks detected:                         │
  │     → 2.05 ppm (s, ~3% integral) — acetone?            │
  │     → 1.25 ppm (s, ~5% integral) — grease/silicone     │
  │                                                         │
  │  ⚠  Note: Purity by NMR integration assumes            │
  │     quantitative acquisition (D1 ≥ 5×T₁). Check your   │
  │     acquisition parameters for accuracy.                │
  │                                                         │
  │  Sources:                                               │
  │  • ¹H prediction: speq-heuristic v1 (additive)         │
  │  • ¹³C prediction: CASCADE-2.0 (3D-GNN, Paton Lab)     │
  │  • Purity: integral ratio analysis                      │
  └─────────────────────────────────────────────────────────┘

  Report saved → ~/.speq/routes/kinase_inhibitor/step_01.md
```

**Phase 3: Track Progress Across the Route**

As the chemist completes each step, the agent accumulates knowledge:

```
$ speq route status

  Kinase Inhibitor Synthesis
  ══════════════════════════
  Step 1: Esterification          ✅ Verified (97%, purity ~92%)
  Step 2: Nitration               ✅ Verified (94%, purity ~88%)
  Step 3: Reduction               ✅ Verified (91%, purity ~85%)
  Step 4: Amide coupling          ⏳ Awaiting NMR data
  Step 5: Boc deprotection        ⏳ Not started

  Cumulative impurity tracking:
    Grease contaminant persists from Step 1 (not reaction-related)
    Step 2 regioisomer impurity (~3%) — see /btw note
```

### 2.3 BTW Mode — The Escape Hatch

Mid-verification, when something doesn't match expectations, the chemist triggers investigation mode:

```
$ speq verify step-4 --h1 ./bruker/exp_004/

  ┌ Verification Report: Step 4 — Amide Coupling ─────────┐
  │                                                         │
  │  ❌ Amide N-H:  NOT DETECTED                           │
  │     → No signal in expected range 6.5-8.5 ppm          │
  │                                                         │
  │  ⚠  Unexpected peak at 9.82 ppm (s, 1H)               │
  │  ⚠  Unexpected peak at 2.36 ppm (s, 3H)               │
  │                                                         │
  │  Confidence: 12%  │  Product likely NOT formed.         │
  │                                                         │
  │  → Type /btw to investigate what was actually made.     │
  └─────────────────────────────────────────────────────────┘

speq > /btw

  Entering investigation mode...

  Based on the unexpected peaks:
  • 9.82 ppm (s, 1H) — consistent with aldehyde C-H
  • 2.36 ppm (s, 3H) — consistent with aromatic methyl or ketone methyl

  Questions to narrow down:
  1. What coupling reagent did you use? (EDC/HOBt, HATU, DCC, etc.)
  2. What base and solvent?
  3. Was the reaction done under inert atmosphere?

speq > I used HATU in DMF with DIPEA, under nitrogen

  Analyzing with your reaction conditions...

  ▸ Running: ChefNMR structure elucidation ... done (4.2s)
  ▸ Running: CASCADE-2.0 ¹³C prediction for candidates ... done (1.1s)
  ▸ Running: pubchem_lookup for side products ... done (0.8s)

  ╭─ Investigation Result ─────────────────────────────────╮
  │                                                         │
  │  Most likely scenario: Starting amine was oxidized       │
  │  to the corresponding imine/aldehyde under HATU          │
  │  activation conditions. The amide coupling did not       │
  │  proceed — likely due to steric hindrance at the         │
  │  amine nitrogen.                                         │
  │                                                         │
  │  Suggested structure: [SMILES]                           │
  │  Confidence: 73%                                         │
  │                                                         │
  │  Recommendation:                                         │
  │  • Try HATU with longer reaction time (24h vs 12h)       │
  │  • Consider switching to acid chloride activation        │
  │  • Run LCMS to confirm MW of side product                │
  ╰─────────────────────────────────────────────────────────╯
```

### 2.4 EXPLORE Mode — Full Structure Elucidation

For cases with no expected structure (natural products, unknown impurities, degradation studies):

```
$ speq explore --h1 unknown.jdx --c13 unknown_c13.jdx --ms hrms.csv

  Full structure elucidation pipeline...
  ▸ Running: formula_from_hrms ... C₁₂H₁₅NO₃
  ▸ Running: ChefNMR (atomic diffusion) ... 10 candidates
  ▸ Running: CASCADE-2.0 (¹³C verification) ... ranked
  ▸ Running: functional_group_id ... ester, amine, aromatic
  ▸ Running: nmrshiftdb_search ... 2 database hits

  [Full elucidation report with ranked candidates]
```

---

## Part 3: Architecture

### 3.1 System Overview

```
                         ┌─────────────────────┐
                         │     Chemist (User)   │
                         └──────────┬──────────┘
                                    │
                         ┌──────────▼──────────┐
                         │   INPUT LAYER        │
                         │  • Bruker FID/pdata  │
                         │  • JCAMP-DX          │
                         │  • MestReNova export │
                         │  • Peak list (CSV)   │
                         │  • Spectrum image     │
                         │  • SMILES / SDF      │
                         └──────────┬──────────┘
                                    │
                         ┌──────────▼──────────┐
                         │   SPECTRUM PARSER    │
                         │  (nmrglue + custom)  │
                         │                      │
                         │  Raw FID → peaks     │
                         │  Image → peaks (LLM) │
                         │  Peak list → struct  │
                         └──────────┬──────────┘
                                    │
                         ┌──────────▼──────────┐
                         │   AGENT CORE         │
                         │  (Claude API)        │
                         │                      │
                         │  Mode router:        │
                         │  VERIFY / BTW /      │
                         │  EXPLORE             │
                         │                      │
                         │  Route context       │
                         │  Session state       │
                         └──────────┬──────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
   ┌──────────▼──────┐  ┌──────────▼──────┐  ┌──────────▼──────┐
   │ EXTERNAL MODELS  │  │  BUILT-IN TOOLS  │  │   DATABASES     │
   │                  │  │                  │  │                  │
   │ • ChefNMR        │  │ • RDKit func.    │  │ • nmrshiftdb2    │
   │   (elucidation)  │  │   group detect   │  │ • PubChem        │
   │ • CASCADE-2.0    │  │ • HRMS formula   │  │ • SDBS           │
   │   (¹³C shifts)   │  │ • Purity scorer  │  │ • ChEMBL         │
   │ • Future models  │  │ • DP4 calculator │  │                  │
   │   (plug-in)      │  │ • Peak matcher   │  │                  │
   └──────────────────┘  └──────────────────┘  └──────────────────┘
              │                     │                     │
              └─────────────────────┼─────────────────────┘
                                    │
                         ┌──────────▼──────────┐
                         │   REPORT GENERATOR   │
                         │  • Verification card │
                         │  • Investigation log │
                         │  • ELN-ready export  │
                         │  • Route summary     │
                         └─────────────────────┘
```

### 3.2 Project Structure

```
speq/
├── pyproject.toml
├── README.md
├── src/
│   └── speq/
│       ├── __init__.py
│       ├── cli.py                  # Click CLI (verify, route, explore, btw)
│       ├── agent.py                # Core reasoning loop (Claude API)
│       ├── config.py               # API keys, model paths, settings
│       ├── session.py              # Conversation + route state persistence
│       ├── display.py              # Rich terminal rendering
│       │
│       ├── input/                  # INPUT LAYER — all data ingestion
│       │   ├── __init__.py
│       │   ├── bruker.py           # Parse Bruker FID/pdata directories
│       │   ├── jcamp.py            # JCAMP-DX file parser
│       │   ├── mestrexport.py      # MestReNova peak list / multiplet report
│       │   ├── topspinexport.py    # TopSpin peak list export
│       │   ├── csv_peaks.py        # Generic CSV/TSV peak list
│       │   ├── image_spectrum.py   # Spectrum image → peaks via vision LLM
│       │   ├── molecule.py         # SMILES, SDF, MOL, InChI input
│       │   └── autodetect.py       # Sniff file type and route to parser
│       │
│       ├── core/                   # Framework internals
│       │   ├── tool.py             # Base Tool class
│       │   ├── registry.py         # Auto-discovers tools from tools/
│       │   ├── result.py           # ToolResult with mandatory citations
│       │   └── route.py            # Synthetic route data model
│       │
│       ├── modes/                  # THE THREE MODES
│       │   ├── __init__.py
│       │   ├── verify.py           # VERIFY mode — functional group check
│       │   ├── btw.py              # BTW mode — investigation escape hatch
│       │   └── explore.py          # EXPLORE mode — full elucidation
│       │
│       ├── tools/                  # ALL TOOLS (one file each)
│       │   │
│       │   ├── external_models/    # Wrappers around published ML models
│       │   │   ├── chefnmr_wrapper.py
│       │   │   ├── cascade_wrapper.py
│       │   │   └── model_registry.py
│       │   │
│       │   ├── verification/       # Core verification tools
│       │   │   ├── functional_group_checker.py
│       │   │   ├── diagnostic_marker_generator.py
│       │   │   ├── peak_matcher.py
│       │   │   ├── shift_deviation_scorer.py
│       │   │   ├── dp4_calculator.py
│       │   │   └── regioisomer_checker.py
│       │   │
│       │   ├── purity/             # Purity assessment
│       │   │   ├── integral_purity.py
│       │   │   ├── impurity_identifier.py
│       │   │   ├── solvent_peak_library.py
│       │   │   └── common_contaminants.py
│       │   │
│       │   ├── prediction/         # Shift prediction (heuristic + model)
│       │   │   ├── h1_heuristic.py
│       │   │   ├── c13_cascade.py
│       │   │   ├── coupling_predictor.py
│       │   │   └── multiplicity_predictor.py
│       │   │
│       │   ├── elucidation/        # Structure solving (BTW/EXPLORE only)
│       │   │   ├── chefnmr_elucidator.py
│       │   │   ├── fragment_assembler.py
│       │   │   ├── candidate_ranker.py
│       │   │   └── reaction_side_product_db.py
│       │   │
│       │   ├── mass_spec/
│       │   │   ├── formula_from_hrms.py
│       │   │   ├── isotope_pattern.py
│       │   │   └── adduct_calculator.py
│       │   │
│       │   ├── database/
│       │   │   ├── nmrshiftdb_search.py
│       │   │   ├── pubchem_lookup.py
│       │   │   └── known_side_products.py
│       │   │
│       │   └── molecular/
│       │       ├── smiles_parser.py
│       │       ├── molecular_formula.py
│       │       ├── functional_groups_rdkit.py
│       │       ├── reaction_classifier.py
│       │       └── stereocenter_detector.py
│       │
│       └── report/                 # Output generation
│           ├── verification_card.py
│           ├── investigation_log.py
│           ├── route_summary.py
│           └── eln_export.py       # Electronic lab notebook format
│
├── tests/
│   ├── test_input/                 # Parser tests with real file fixtures
│   ├── test_verify/                # Verification pipeline tests
│   ├── test_btw/                   # Investigation mode tests
│   ├── test_tools/
│   └── benchmarks/
│       ├── verification_benchmark.py
│       └── purity_benchmark.py
│
├── models/                         # External model checkpoints (git-ignored)
│   ├── README.md                   # Download instructions
│   ├── chefnmr/                    # ChefNMR checkpoints from Zenodo
│   └── cascade/                    # CASCADE-2.0 pretrained weights
│
└── data/
    ├── solvent_peaks.json          # Reference: common solvent residual peaks
    ├── contaminants.json           # Reference: common contaminants (grease, etc.)
    ├── reaction_types.json         # Reaction type → expected spectral changes
    └── functional_group_shifts.json # Diagnostic shift ranges per group
```

### 3.3 CLI Commands

```python
# src/speq/cli.py

import click
from rich.console import Console

console = Console()

@click.group()
@click.version_option()
def cli():
    """speq — spectroscopy verification agent for synthetic chemistry."""
    pass


# ══════════════════════════════════════════════════════════
# ROUTE MANAGEMENT — Define and track multi-step syntheses
# ══════════════════════════════════════════════════════════

@cli.group()
def route():
    """Define and manage synthetic routes."""
    pass

@route.command()
@click.argument("name")
def new(name):
    """Create a new synthetic route. Starts interactive step definition."""
    pass

@route.command()
@click.argument("route_name", required=False)
def status(route_name):
    """Show verification status of all steps in a route."""
    pass

@route.command()
@click.argument("route_name")
@click.option("--step", "-n", type=int, help="Step number")
@click.option("--sm", help="Starting material SMILES")
@click.option("--product", "-p", help="Expected product SMILES")
@click.option("--reaction", "-r", help="Reaction type (e.g., 'amide_coupling')")
def add_step(route_name, step, sm, product, reaction):
    """Add a step to an existing route."""
    pass


# ══════════════════════════════════════════════════════════
# VERIFY — The primary workflow
# ══════════════════════════════════════════════════════════

@cli.command()
@click.argument("step_ref")  # e.g., "step-1" or "kinase:step-3"
@click.option("--h1", type=click.Path(exists=True), help="¹H NMR data")
@click.option("--c13", type=click.Path(exists=True), help="¹³C NMR data")
@click.option("--hsqc", type=click.Path(exists=True), help="HSQC data")
@click.option("--ms", type=click.Path(exists=True), help="Mass spec data")
@click.option("--ir", type=click.Path(exists=True), help="IR spectrum")
@click.option("--solvent", default="CDCl3", help="NMR solvent")
@click.option("--peaks", type=click.Path(exists=True), help="Peak list file")
@click.option("--image", type=click.Path(exists=True), help="Spectrum image")
def verify(step_ref, h1, c13, hsqc, ms, ir, solvent, peaks, image):
    """Verify a synthetic step against experimental NMR data."""
    pass


# ══════════════════════════════════════════════════════════
# QUICK VERIFY — No route needed, single-shot verification
# ══════════════════════════════════════════════════════════

@cli.command()
@click.option("--smiles", "-s", required=True, help="Expected product SMILES")
@click.option("--h1", type=click.Path(exists=True), help="¹H NMR data")
@click.option("--c13", type=click.Path(exists=True), help="¹³C NMR data")
@click.option("--solvent", default="CDCl3", help="NMR solvent")
@click.option("--peaks", type=click.Path(exists=True), help="Peak list file")
@click.option("--image", type=click.Path(exists=True), help="Spectrum image")
def check(smiles, h1, c13, solvent, peaks, image):
    """Quick one-shot verification: does this spectrum match this structure?"""
    pass


# ══════════════════════════════════════════════════════════
# BTW — Investigation mode (escape hatch)
# ══════════════════════════════════════════════════════════

@cli.command()
@click.option("--h1", type=click.Path(exists=True), help="¹H NMR data")
@click.option("--c13", type=click.Path(exists=True), help="¹³C NMR data")
@click.option("--ms", type=click.Path(exists=True), help="Mass spec data")
@click.option("--context", help="What you expected / what went wrong")
def btw(h1, c13, ms, context):
    """Investigate unexpected results. Opens interactive chat with the agent."""
    pass


# ══════════════════════════════════════════════════════════
# EXPLORE — Full structure elucidation
# ══════════════════════════════════════════════════════════

@cli.command()
@click.option("--h1", type=click.Path(exists=True), help="¹H NMR data")
@click.option("--c13", type=click.Path(exists=True), help="¹³C NMR data")
@click.option("--hsqc", type=click.Path(exists=True), help="HSQC data")
@click.option("--ms", type=click.Path(exists=True), help="Mass spec data")
@click.option("--formula", help="Molecular formula if known")
def explore(h1, c13, hsqc, ms, formula):
    """Full structure elucidation from spectra (no expected structure)."""
    pass


# ══════════════════════════════════════════════════════════
# SETUP & UTILITIES
# ══════════════════════════════════════════════════════════

@cli.command()
def setup():
    """Interactive setup: API keys, model downloads, dependency check."""
    pass

@cli.command()
def doctor():
    """Check installation: dependencies, models, API connectivity."""
    pass

@cli.command()
@click.option("--format", "fmt", type=click.Choice(["md", "html", "eln"]),
              default="md")
def export(fmt):
    """Export latest session/route report."""
    pass


def main():
    cli()
```

---

## Part 4: External Model Integration

### 4.1 Integration Philosophy

speq does not train its own core prediction models. Instead, it wraps published, peer-reviewed, open-source models as tools. This gives speq:

- **Immediate state-of-the-art accuracy** on day one
- **Academic credibility** through proper citation of published methods
- **Faster development** — no GPU training, no dataset curation
- **Easy upgrades** — swap in a better model by updating one wrapper file

speq's proprietary value is the **orchestration layer**: deciding which model to call, when, with what inputs, and how to synthesize multiple model outputs into a chemist-readable answer.

### 4.2 Model: ChefNMR (Structure Elucidation)

**Source:** Princeton, E.Z. Lab (Zhong group)
**Paper:** "Atomic Diffusion Models for Small Molecule Structure Elucidation from NMR Spectra" (NeurIPS 2025)
**Repository:** https://github.com/ml-struct-bio/chefnmr
**License:** MIT
**Checkpoints:** Zenodo (https://zenodo.org/records/17766755)

**What it does:**
ChefNMR takes 1D NMR spectra (¹H and/or ¹³C) plus a molecular formula as input, and generates candidate 3D molecular structures using an atomic diffusion model built on a non-equivariant transformer. It frames structure elucidation as conditional generation — given spectral data, it denoises random atomic coordinates into valid molecular geometries.

**Key capabilities:**
- End-to-end: raw spectra → candidate structures (no intermediate SMILES tokenization)
- Trained on 111,000+ natural products (SpectraNP dataset)
- Multiple pretrained checkpoints for different spectrum resolutions (H10k, C10k, C80)
- Top-10 matching accuracy of ~79% on synthetic benchmarks, ~65% on challenging natural products
- Generates 3D conformations directly, not just connectivity

**How speq uses it:**
- BTW mode: when verification fails and the chemist asks "what did I actually make?"
- EXPLORE mode: full structure elucidation from spectra alone
- NOT used in VERIFY mode (verification doesn't need elucidation)

**Integration approach:**

```python
# src/speq/tools/external_models/chefnmr_wrapper.py

import subprocess
import json
from pathlib import Path
from speq.core.tool import Tool, ToolResult, ToolCategory


class ChefNMRElucidator(Tool):
    """
    Wrapper around ChefNMR atomic diffusion model for structure elucidation.
    
    Calls ChefNMR as a subprocess (Python environment isolation).
    Requires: conda env 'nmr3d' with ChefNMR installed.
    """

    name = "chefnmr_elucidator"
    description = (
        "Predicts molecular structure from 1D NMR spectra and molecular formula. "
        "Uses atomic diffusion model (ChefNMR, NeurIPS 2025). "
        "Returns ranked candidate structures with confidence scores. "
        "Use for structure elucidation when the target structure is unknown."
    )
    category = ToolCategory.STRUCTURE_ELUCIDATION
    requires = []  # External dependency, checked via is_available()

    def __init__(self, chefnmr_dir: str = None, checkpoint: str = None):
        self.chefnmr_dir = Path(chefnmr_dir or "~/.speq/models/chefnmr").expanduser()
        self.checkpoint = checkpoint or "SB-H10kC80-L128-epoch5249.ckpt"

    def is_available(self) -> bool:
        """Check if ChefNMR environment and checkpoint exist."""
        ckpt_path = self.chefnmr_dir / "checkpoints" / self.checkpoint
        if not ckpt_path.exists():
            return False
        # Check conda env exists
        result = subprocess.run(
            ["conda", "run", "-n", "nmr3d", "python", "-c", "import src"],
            capture_output=True, cwd=str(self.chefnmr_dir)
        )
        return result.returncode == 0

    def run(
        self,
        h1_spectrum: list[float] = None,     # binned ¹H spectrum (10k bins)
        c13_spectrum: list[float] = None,     # binned ¹³C spectrum (80 or 10k bins)
        molecular_formula: str = None,         # e.g., "C9H10O3"
        n_candidates: int = 10,
        n_diffusion_steps: int = 50,
    ) -> ToolResult:
        """
        Run ChefNMR structure elucidation.
        
        Returns candidate structures as SMILES with Tanimoto similarities.
        """
        if molecular_formula is None:
            return ToolResult.fail("Molecular formula required for ChefNMR")

        # Prepare input data as HDF5 (ChefNMR's expected format)
        input_path = self._prepare_input(h1_spectrum, c13_spectrum, molecular_formula)

        # Call ChefNMR via subprocess in its conda environment
        cmd = [
            "conda", "run", "-n", "nmr3d", "python", "-m", "src.main",
            "+data=custom_input",
            "+condition=h1c13nmr-10k-80",
            "+model=dit-l",
            "+embedder=hybrid-baseline",
            "+diffusion=edm-train_af3-sample_edm_sde",
            "+guidance=cfg2",
            "+exp=eval_ckpt",
            f"general.ckpt_abs_path={self.chefnmr_dir}/checkpoints/{self.checkpoint}",
            f"dataset_args.test_args.test_samples=1",
            f"test_args.diffusion_samples={n_candidates}",
            f"test_args.num_sampling_steps={n_diffusion_steps}",
            f"dataset_args.custom_input={input_path}",
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=str(self.chefnmr_dir), timeout=120
        )

        if result.returncode != 0:
            return ToolResult.fail(f"ChefNMR failed: {result.stderr[:500]}")

        # Parse output: extract candidate SMILES, similarities, RMSD
        candidates = self._parse_output(result.stdout)

        return ToolResult.ok(
            data={
                "candidates": candidates,
                "molecular_formula": molecular_formula,
                "n_candidates": len(candidates),
            },
            citations=[
                "model: ChefNMR (Xiong et al., NeurIPS 2025), "
                f"checkpoint: {self.checkpoint}, "
                f"diffusion_steps: {n_diffusion_steps}, "
                f"cfg_scale: 2.0"
            ]
        )

    def _prepare_input(self, h1, c13, formula):
        """Convert speq internal format to ChefNMR HDF5 input."""
        # ... HDF5 preparation logic
        pass

    def _parse_output(self, stdout):
        """Extract candidate structures from ChefNMR output."""
        # ... parse SMILES, Tanimoto similarity, RMSD from output
        pass
```

### 4.3 Model: CASCADE 2.0 (¹³C Shift Prediction)

**Source:** Colorado State University, Paton Lab
**Paper:** "CASCADE-2.0: Real Time Prediction of ¹³C-NMR Shifts with sub-ppm Accuracy" (ChemRxiv 2025)
**Repository:** https://github.com/asbhd/CASCADE-2.0
**License:** MIT
**Webserver:** https://nova.chem.colostate.edu/v2/cascade/

**What it does:**
CASCADE 2.0 is a 3D graph neural network that predicts ¹³C NMR chemical shifts from molecular structure. It takes SMILES or SDF as input, automatically generates 3D conformers, and predicts per-carbon shifts with a mean absolute error of 0.73 ppm against experimental data — state-of-the-art for ¹³C prediction.

**Key capabilities:**
- Sub-ppm accuracy (0.73 ppm MAE) for ¹³C shifts
- Trained on 211,000 experimental shifts cross-validated against DFT
- Handles diverse chemistry including heteroatoms (N, O, S, halogens)
- Real-time prediction (~seconds per molecule, no DFT needed)
- Pretrained weights available for direct use

**How speq uses it:**
- VERIFY mode: predict ¹³C shifts for the expected product, compare against experimental
- BTW mode: predict shifts for candidate side-product structures
- EXPLORE mode: verify ¹³C shift consistency of ChefNMR candidates

**Integration approach:**

```python
# src/speq/tools/external_models/cascade_wrapper.py

import numpy as np
from pathlib import Path
from speq.core.tool import Tool, ToolResult, ToolCategory


class CascadeC13Predictor(Tool):
    """
    Wrapper around CASCADE-2.0 for ¹³C NMR chemical shift prediction.
    
    Uses pretrained GNN weights from the Paton Lab.
    Requires: rdkit, tensorflow, KGCNN (installed in speq environment).
    """

    name = "cascade_c13_predictor"
    description = (
        "Predicts ¹³C NMR chemical shifts for all carbon atoms in a molecule. "
        "Uses CASCADE-2.0 3D-GNN (Paton Lab, 2025). "
        "MAE: 0.73 ppm against experimental data. "
        "Input: SMILES string. Output: per-carbon shift predictions."
    )
    category = ToolCategory.NMR_PREDICTION
    requires = ["rdkit", "tensorflow"]

    def __init__(self, model_dir: str = None):
        self.model_dir = Path(model_dir or "~/.speq/models/cascade").expanduser()
        self._model = None

    def _load_model(self):
        if self._model is None:
            from speq.tools.external_models._cascade_loader import load_cascade_model
            self._model = load_cascade_model(self.model_dir)

    def run(self, smiles: str, solvent: str = "CDCl3") -> ToolResult:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ToolResult.fail(f"Invalid SMILES: {smiles}")

        self._load_model()

        # CASCADE workflow: SMILES → 3D conformers → GNN prediction
        # Generate conformer ensemble (CASCADE uses MMFF optimization)
        mol_3d = Chem.AddHs(mol)
        AllChem.EmbedMultipleConfs(mol_3d, numConfs=10, randomSeed=42)
        AllChem.MMFFOptimizeMoleculeConfs(mol_3d)

        # Run prediction through CASCADE GNN
        predictions = self._model.predict(mol_3d)

        # Format per-carbon results
        carbon_shifts = []
        for idx, atom in enumerate(mol.GetAtoms()):
            if atom.GetAtomicNum() == 6:  # Carbon
                carbon_shifts.append({
                    "atom_index": idx,
                    "predicted_shift_ppm": float(predictions[idx]),
                    "atom_symbol": atom.GetSymbol(),
                    "hybridization": str(atom.GetHybridization()),
                    "neighbors": [
                        mol.GetAtomWithIdx(n.GetIdx()).GetSymbol()
                        for n in atom.GetNeighbors()
                    ]
                })

        return ToolResult.ok(
            data={
                "smiles": smiles,
                "solvent": solvent,
                "carbon_shifts": sorted(carbon_shifts,
                                       key=lambda x: x["predicted_shift_ppm"],
                                       reverse=True),
                "n_carbons": len(carbon_shifts),
            },
            citations=[
                "model: CASCADE-2.0 (Bhadauria et al., 2025), "
                "architecture: 3D-GNN with MMFF conformers, "
                f"training_data: 211K experimental shifts, "
                f"reported_MAE: 0.73 ppm (¹³C)"
            ]
        )
```

### 4.4 External Model Registry

A registry system allows adding new models without changing core code:

```python
# src/speq/tools/external_models/model_registry.py

"""
External Model Registry
═══════════════════════
Tracks available external ML models, their capabilities, and status.

To add a new model:
1. Create a wrapper in this directory (see chefnmr_wrapper.py as template)
2. Register it below
3. Run `speq doctor` to verify installation

Models are lazy-loaded — no GPU memory used until first call.
"""

EXTERNAL_MODELS = {
    "chefnmr": {
        "name": "ChefNMR",
        "version": "1.0",
        "task": "structure_elucidation",
        "paper": "Xiong et al., NeurIPS 2025",
        "repo": "https://github.com/ml-struct-bio/chefnmr",
        "license": "MIT",
        "input": "1D NMR spectra (¹H, ¹³C) + molecular formula",
        "output": "Candidate 3D structures as SMILES",
        "accuracy": "~65% top-1 on natural products, ~79% top-10 on SpectraBase",
        "install": "conda env create -f chefnmr/environment.yaml -n nmr3d",
        "checkpoint_url": "https://zenodo.org/records/17766755",
        "gpu_required": True,
        "wrapper": "chefnmr_wrapper.ChefNMRElucidator",
    },
    "cascade_v2": {
        "name": "CASCADE 2.0",
        "version": "2.0",
        "task": "c13_shift_prediction",
        "paper": "Bhadauria et al., ChemRxiv 2025",
        "repo": "https://github.com/asbhd/CASCADE-2.0",
        "license": "MIT",
        "input": "SMILES or SDF",
        "output": "Per-carbon ¹³C chemical shifts (ppm)",
        "accuracy": "0.73 ppm MAE vs experimental",
        "install": "pip install rdkit tensorflow==2.11 KGCNN==2.2.1",
        "checkpoint_url": "included in repo (models/ directory)",
        "gpu_required": False,  # CPU inference is fast enough
        "wrapper": "cascade_wrapper.CascadeC13Predictor",
    },
    # ── Future models to integrate ──────────────────────────
    # "nmrshiftpredict": {
    #     "task": "h1_shift_prediction",
    #     "paper": "...",
    #     "note": "Placeholder for ¹H shift GNN when a suitable
    #              open-source model is published"
    # },
    # "spectro": {
    #     "task": "multimodal_elucidation",
    #     "paper": "Chacko et al., NeurIPS AI4Mat Workshop 2024",
    #     "note": "Multi-modal approach using IR + NMR for elucidation"
    # },
}
```

### 4.5 Models NOT Integrated (and Why)

| Model / Tool | Why Not |
|---|---|
| ACD/Labs | Commercial, closed-source. Cannot redistribute. |
| MestReNova prediction | Commercial plugin. But we accept MestReNova *exports* as input. |
| NMR-Solver | Older method, superseded by ChefNMR in accuracy. |
| Custom GNN (blueprint v1) | Unnecessary in v2 — CASCADE 2.0 already achieves target accuracy. Train only if gaps found. |
| Spectra→SMILES transformer (blueprint v1) | ChefNMR's diffusion approach is superior. No need to replicate. |

### 4.6 When to Train Your Own Models

Despite the integrate-first philosophy, there are specific gaps where custom models add value:

1. **¹H shift prediction** — No strong open-source GNN currently matches CASCADE's ¹³C performance for ¹H. The blueprint v1 GNN architecture (Section 3.2 of original blueprint) remains valid for this. Train on NMRexp (3.3M records) with scaffold splitting. Target: < 0.20 ppm MAE.

2. **Functional group classifier from raw spectra** — The blueprint v1 CNN+attention model (Section 3.4 of original blueprint) is useful for VERIFY mode. This is a simpler model (multi-label classification, not generation), trainable in hours, and fills a gap no published model covers exactly.

3. **Reaction-aware side product predictor** — Given a reaction type and conditions, predict the most likely side products. This is speq-specific and not covered by any existing model. Could be a fine-tuned LLM or a curated rule-based system initially.

---

## Part 5: Core Tool Implementations

### 5.1 Tool Base Class

```python
# src/speq/core/tool.py

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


class ToolCategory(Enum):
    NMR_PREDICTION = "nmr_prediction"
    STRUCTURE_ELUCIDATION = "structure_elucidation"
    STRUCTURE_VERIFICATION = "structure_verification"
    PURITY = "purity"
    MASS_SPEC = "mass_spec"
    DATABASE = "database"
    MOLECULAR = "molecular"
    INPUT_PARSING = "input_parsing"


@dataclass
class ToolResult:
    """Every tool returns this. Citations are mandatory."""
    success: bool
    data: Dict[str, Any]
    citations: List[str]
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
    name: str
    description: str
    category: ToolCategory
    requires: List[str] = []

    @abstractmethod
    def run(self, **kwargs) -> ToolResult:
        pass

    def is_available(self) -> bool:
        for dep in self.requires:
            try:
                __import__(dep)
            except ImportError:
                return False
        return True
```

### 5.2 Diagnostic Marker Generator

This is a new tool not in the original blueprint — it pre-computes what the agent should look for during verification.

```python
# src/speq/tools/verification/diagnostic_marker_generator.py

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from speq.core.tool import Tool, ToolResult, ToolCategory


class DiagnosticMarkerGenerator(Tool):
    """
    Given a starting material and expected product, compute the diagnostic
    spectral markers that indicate successful transformation.
    
    Identifies:
    - NEW peaks: functional groups present in product but not starting material
    - LOST peaks: functional groups present in SM but not product
    - RETAINED peaks: functional groups unchanged between SM and product
    - SHIFTED peaks: groups that remain but change environment
    """

    name = "diagnostic_marker_generator"
    description = (
        "Computes diagnostic NMR markers for verifying a synthetic step. "
        "Compares starting material and product structures to identify "
        "which peaks should appear, disappear, or shift."
    )
    category = ToolCategory.STRUCTURE_VERIFICATION

    # Functional group definitions with characteristic NMR ranges
    DIAGNOSTIC_GROUPS = {
        "aldehyde": {
            "smarts": "[CH]=O",
            "h1_range": (9.4, 10.0),
            "h1_mult": "s or d",
            "c13_range": (190, 205),
            "description": "Aldehyde C-H"
        },
        "carboxylic_acid_oh": {
            "smarts": "[OH]C=O",
            "h1_range": (10.0, 12.0),
            "h1_mult": "broad s",
            "c13_range": (165, 185),
            "description": "Carboxylic acid O-H"
        },
        "ester_carbonyl": {
            "smarts": "[#6]OC(=O)[#6]",
            "h1_range": None,  # No direct H
            "c13_range": (165, 175),
            "description": "Ester C=O"
        },
        "amide_nh": {
            "smarts": "[NH]C=O",
            "h1_range": (6.0, 8.5),
            "h1_mult": "broad",
            "c13_range": (165, 175),
            "description": "Amide N-H"
        },
        "boc_tert_butyl": {
            "smarts": "OC(=O)OC(C)(C)C",
            "h1_range": (1.35, 1.55),
            "h1_mult": "s, 9H",
            "c13_range": (28, 29),
            "description": "Boc tert-butyl (9H singlet)"
        },
        "aromatic_h": {
            "smarts": "[cH]",
            "h1_range": (6.5, 8.5),
            "h1_mult": "varies",
            "c13_range": (110, 160),
            "description": "Aromatic C-H"
        },
        "primary_amine_nh2": {
            "smarts": "[NH2][#6]",
            "h1_range": (0.5, 5.0),
            "h1_mult": "broad",
            "c13_range": None,
            "description": "Primary amine N-H₂"
        },
        "alcohol_oh": {
            "smarts": "[OH][#6]",
            "h1_range": (1.0, 5.5),
            "h1_mult": "broad, variable",
            "c13_range": None,
            "description": "Alcohol O-H"
        },
        "methyl_ester_och3": {
            "smarts": "COC=O",
            "h1_range": (3.6, 3.9),
            "h1_mult": "s, 3H",
            "c13_range": (51, 53),
            "description": "Methyl ester O-CH₃"
        },
        "tms_protecting_group": {
            "smarts": "[Si](C)(C)C",
            "h1_range": (-0.1, 0.3),
            "h1_mult": "s, 9H",
            "c13_range": (-2, 2),
            "description": "TMS group"
        },
        # ... 30+ more functional group patterns
    }

    def run(self, sm_smiles: str, product_smiles: str) -> ToolResult:
        sm = Chem.MolFromSmiles(sm_smiles)
        prod = Chem.MolFromSmiles(product_smiles)

        if sm is None or prod is None:
            return ToolResult.fail("Invalid SMILES for SM or product")

        new_groups = []
        lost_groups = []
        retained_groups = []

        for group_name, info in self.DIAGNOSTIC_GROUPS.items():
            pattern = Chem.MolFromSmarts(info["smarts"])
            in_sm = sm.HasSubstructMatch(pattern)
            in_prod = prod.HasSubstructMatch(pattern)

            marker = {
                "group": group_name,
                "description": info["description"],
                "h1_range": info.get("h1_range"),
                "h1_multiplicity": info.get("h1_mult"),
                "c13_range": info.get("c13_range"),
            }

            if in_prod and not in_sm:
                new_groups.append(marker)
            elif in_sm and not in_prod:
                lost_groups.append(marker)
            elif in_sm and in_prod:
                retained_groups.append(marker)

        return ToolResult.ok(
            data={
                "new_groups": new_groups,      # Should appear in product spectrum
                "lost_groups": lost_groups,    # Should be absent from product
                "retained_groups": retained_groups,
                "sm_smiles": sm_smiles,
                "product_smiles": product_smiles,
            },
            citations=[
                "tool: diagnostic_marker_generator, "
                "method: RDKit SMARTS substructure matching, "
                "shift_ranges: literature reference values"
            ]
        )
```

### 5.3 Purity Estimator

```python
# src/speq/tools/purity/integral_purity.py

import json
from pathlib import Path
from speq.core.tool import Tool, ToolResult, ToolCategory


class IntegralPurityEstimator(Tool):
    """
    Estimates sample purity from ¹H NMR integral ratios.
    
    Compares total integral of expected product peaks versus
    unexpected peaks. Identifies known contaminants (solvents, grease).
    """

    name = "integral_purity_estimator"
    description = (
        "Estimates purity from ¹H NMR integrals. Compares product peak "
        "integrals against total spectrum integral. Identifies common "
        "contaminants by shift position."
    )
    category = ToolCategory.PURITY

    def __init__(self):
        data_dir = Path(__file__).parent.parent.parent / "data"
        with open(data_dir / "solvent_peaks.json") as f:
            self.solvent_peaks = json.load(f)
        with open(data_dir / "contaminants.json") as f:
            self.contaminants = json.load(f)

    def run(
        self,
        observed_peaks: list[dict],      # [{"shift": 7.35, "integral": 5.0, ...}]
        expected_peaks: list[dict],      # From shift predictor
        solvent: str = "CDCl3",
        tolerance_ppm: float = 0.15,
    ) -> ToolResult:
        # Classify each observed peak as: product, solvent, contaminant, or unknown
        product_integral = 0.0
        solvent_integral = 0.0
        contaminant_integral = 0.0
        unknown_peaks = []
        unknown_integral = 0.0
        matched_peaks = []

        solvent_ref = self.solvent_peaks.get(solvent, [])

        for obs in observed_peaks:
            shift = obs["shift"]
            integral = obs.get("integral", 1.0)

            # Check if it matches an expected product peak
            if self._matches_expected(shift, expected_peaks, tolerance_ppm):
                product_integral += integral
                matched_peaks.append(obs)
                continue

            # Check if it's a known solvent residual peak
            if self._matches_solvent(shift, solvent_ref):
                solvent_integral += integral
                continue

            # Check common contaminants (grease, water, plasticizer)
            contam = self._identify_contaminant(shift)
            if contam:
                contaminant_integral += integral
                continue

            # Unknown / impurity
            unknown_peaks.append(obs)
            unknown_integral += integral

        total = product_integral + unknown_integral + contaminant_integral
        purity = (product_integral / total * 100) if total > 0 else 0

        return ToolResult.ok(
            data={
                "purity_percent": round(purity, 1),
                "product_integral": product_integral,
                "impurity_integral": unknown_integral,
                "contaminant_integral": contaminant_integral,
                "unknown_peaks": unknown_peaks,
                "n_expected_matched": len(matched_peaks),
                "n_expected_total": len(expected_peaks),
            },
            citations=[
                "tool: integral_purity_estimator, "
                "method: ¹H NMR integral ratio analysis, "
                "solvent_ref: Gottlieb et al. J. Org. Chem. 1997"
            ],
            warnings=[
                "Purity by NMR integration assumes quantitative acquisition "
                "(relaxation delay D1 ≥ 5×T₁). Actual purity may differ if "
                "acquisition was not quantitative."
            ] if purity < 98 else []
        )

    def _matches_expected(self, shift, expected, tol):
        return any(abs(shift - exp["shift"]) < tol for exp in expected)

    def _matches_solvent(self, shift, solvent_ref):
        return any(abs(shift - s["shift"]) < 0.05 for s in solvent_ref)

    def _identify_contaminant(self, shift):
        for contam in self.contaminants:
            if abs(shift - contam["shift"]) < 0.05:
                return contam["name"]
        return None
```

### 5.4 Reaction Side Product Database

A new tool specific to BTW mode — given a reaction type, suggests likely side products:

```python
# src/speq/tools/elucidation/reaction_side_product_db.py

from speq.core.tool import Tool, ToolResult, ToolCategory


class ReactionSideProductDB(Tool):
    """
    Given a reaction type and conditions, returns common side products
    with their expected spectral signatures.
    
    This is a curated knowledge base, not an ML model. Entries come from
    literature reviews and established organic chemistry knowledge.
    """

    name = "reaction_side_products"
    description = (
        "Looks up common side products for a given reaction type. "
        "Returns side product structures and their diagnostic NMR peaks. "
        "Use when verification fails to check if a known side reaction occurred."
    )
    category = ToolCategory.STRUCTURE_ELUCIDATION

    REACTION_SIDE_PRODUCTS = {
        "suzuki_coupling": {
            "desired": "Pd-catalyzed C-C bond formation between aryl halide and boronic acid",
            "common_side_products": [
                {
                    "name": "Homo-coupling of boronic acid",
                    "mechanism": "Oxidative homo-coupling under aerobic conditions",
                    "diagnostic": "Symmetric biaryl — ¹H NMR shows fewer aromatic signals than expected",
                    "prevention": "Use inert atmosphere, fresh Pd catalyst"
                },
                {
                    "name": "Proto-deboronation",
                    "mechanism": "Boronic acid loses B(OH)₂, replaced by H",
                    "diagnostic": "Parent arene of boronic acid appears — check for extra ArH",
                    "prevention": "Lower temperature, less basic conditions"
                },
                {
                    "name": "Unreacted aryl halide",
                    "mechanism": "Incomplete conversion",
                    "diagnostic": "Starting material peaks persist; check C-X carbon shift",
                    "prevention": "More catalyst, higher temperature, longer time"
                },
            ]
        },
        "amide_coupling": {
            "desired": "Amide bond formation between amine and carboxylic acid",
            "common_side_products": [
                {
                    "name": "Epimerization at α-carbon",
                    "mechanism": "Base-mediated racemization during activation",
                    "diagnostic": "Diastereomeric peaks visible — doubled signals in ¹H",
                    "prevention": "Use HATU instead of EDC, lower temperature, hindered base"
                },
                {
                    "name": "Guanidinylation (HATU side product)",
                    "mechanism": "HATU reacts with amine instead of activated acid",
                    "diagnostic": "Guanidinium peaks at ~7-8 ppm (broad NH), ¹³C ~155 ppm",
                    "prevention": "Add acid first, pre-activate before adding amine"
                },
                {
                    "name": "N-acylurea (DCC side product)",
                    "mechanism": "O→N acyl migration of O-acylisourea",
                    "diagnostic": "Extra carbonyl in ¹³C (~170 ppm), urea NH at 8-9 ppm",
                    "prevention": "Use HOBt additive, switch to EDC"
                },
            ]
        },
        "boc_deprotection": {
            "desired": "Removal of Boc group to reveal free amine",
            "common_side_products": [
                {
                    "name": "Incomplete deprotection",
                    "mechanism": "Insufficient TFA or reaction time",
                    "diagnostic": "Boc tert-butyl singlet at 1.4-1.5 ppm persists",
                    "prevention": "More TFA, longer reaction time"
                },
                {
                    "name": "TFA salt formation",
                    "mechanism": "Product amine protonated by TFA",
                    "diagnostic": "Broad NH₃⁺ at 7-9 ppm, product shifts may change",
                    "prevention": "Basify workup (NaHCO₃ wash)"
                },
            ]
        },
        # ... 30+ more reaction types
    }

    def run(self, reaction_type: str) -> ToolResult:
        key = reaction_type.lower().replace(" ", "_").replace("-", "_")
        entry = self.REACTION_SIDE_PRODUCTS.get(key)

        if entry is None:
            return ToolResult.ok(
                data={"match": False, "available_types": list(self.REACTION_SIDE_PRODUCTS.keys())},
                citations=["tool: reaction_side_products, method: curated knowledge base"]
            )

        return ToolResult.ok(
            data={
                "reaction_type": key,
                "desired_outcome": entry["desired"],
                "side_products": entry["common_side_products"],
            },
            citations=[
                "tool: reaction_side_products, "
                "method: curated organic chemistry knowledge base, "
                "sources: Clayden et al., March's Advanced Organic Chemistry"
            ]
        )
```

---

## Part 6: Input Layer — Handling Real-World NMR Data

### 6.1 The Input Problem

Chemists have NMR data in many forms. The input layer must handle all of them gracefully.

```
┌─────────────────────────────────────────────────────────────┐
│                     INPUT FORMATS                            │
├──────────────────┬──────────────────────────────────────────┤
│ Instrument raw   │ Bruker FID/pdata, JEOL .jdf, Varian .fid│
│ Processed export │ JCAMP-DX (.jdx/.dx), MestReNova peak    │
│                  │ list (.txt), TopSpin peak list           │
│ Generic          │ CSV/TSV peak list, JSON peak list        │
│ Visual           │ Spectrum image (PNG/PDF) via vision LLM  │
│ Text             │ Typed peak table in natural language     │
│ Structure        │ SMILES, SDF, MOL, InChI, drawn structure │
└──────────────────┴──────────────────────────────────────────┘
```

### 6.2 Autodetect Router

```python
# src/speq/input/autodetect.py

from pathlib import Path
from speq.input import bruker, jcamp, mestrexport, csv_peaks, image_spectrum


def load_spectrum(path: str) -> dict:
    """
    Auto-detect file type and parse NMR data into standard internal format.
    
    Returns:
        {
            "peaks": [{"shift": 7.35, "integral": 5.0, "multiplicity": "m",
                        "coupling_hz": None, "width_hz": 2.3}, ...],
            "source_format": "bruker",
            "nucleus": "1H",
            "solvent": "CDCl3",         # if detectable from file
            "frequency_mhz": 400.0,     # if detectable
            "raw_spectrum": np.array,    # full spectrum if available
        }
    """
    p = Path(path)

    # Bruker: directory containing 'acqus' or 'pdata'
    if p.is_dir():
        if (p / "acqus").exists() or (p / "pdata").exists():
            return bruker.parse(p)
        # Check for Varian/Agilent
        if (p / "procpar").exists():
            raise NotImplementedError("Varian/Agilent format — coming soon")

    # JCAMP-DX
    if p.suffix.lower() in (".jdx", ".dx", ".jcamp"):
        return jcamp.parse(p)

    # MestReNova multiplet report (exported as .txt or .csv)
    if p.suffix.lower() in (".txt", ".csv", ".tsv"):
        # Try MestReNova format first, fall back to generic CSV
        try:
            return mestrexport.parse(p)
        except mestrexport.NotMestReNovaFormat:
            return csv_peaks.parse(p)

    # Spectrum image
    if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".pdf", ".tiff"):
        return image_spectrum.parse(p)

    raise ValueError(f"Unsupported file format: {p.suffix}")
```

### 6.3 Bruker Parser (via nmrglue)

```python
# src/speq/input/bruker.py

import nmrglue as ng
import numpy as np
from pathlib import Path


def parse(bruker_dir: Path) -> dict:
    """
    Parse a Bruker NMR experiment directory.
    
    Handles both raw FID (fid + acqus) and processed data (pdata/1/1r).
    Extracts peaks via automatic peak picking.
    """
    bruker_dir = Path(bruker_dir)

    # Try processed data first (pdata/1/)
    pdata = bruker_dir / "pdata" / "1"
    if pdata.exists() and (pdata / "1r").exists():
        dic, data = ng.bruker.read_pdata(str(pdata))
        udic = ng.bruker.guess_udic(dic, data)
    else:
        # Fall back to raw FID + processing
        dic, data = ng.bruker.read(str(bruker_dir))
        udic = ng.bruker.guess_udic(dic, data)
        # Apply processing: zero-fill, apodization, FFT, phase
        data = ng.proc_base.zf_size(data, data.size * 2)
        data = ng.proc_base.em(data, lb=0.5)
        data = ng.proc_base.fft(data)
        data = ng.proc_autophase.autops(data, "acme")

    # Extract metadata
    nucleus = dic.get("acqus", {}).get("NUC1", "1H")
    solvent = dic.get("acqus", {}).get("SOLVENT", "unknown")
    frequency = float(dic.get("acqus", {}).get("SFO1", 400.0))

    # Create a unit converter for ppm
    uc = ng.fileiobase.uc_from_udic(udic)

    # Peak picking
    threshold = np.std(data) * 5  # 5× noise level
    peak_indices = ng.peakpick.pick(data, pthres=threshold)

    peaks = []
    for idx in peak_indices:
        ppm = float(uc.ppm(idx))
        intensity = float(data[idx])
        peaks.append({
            "shift": round(ppm, 4),
            "intensity": intensity,
            "integral": None,       # Requires integration regions
            "multiplicity": None,   # Requires multiplet analysis
            "coupling_hz": None,
        })

    return {
        "peaks": sorted(peaks, key=lambda p: -p["shift"]),
        "source_format": "bruker",
        "nucleus": nucleus,
        "solvent": solvent,
        "frequency_mhz": frequency,
        "raw_spectrum": data.real,
        "ppm_scale": np.array([uc.ppm(i) for i in range(len(data))]),
    }
```

### 6.4 Spectrum Image Parser (Vision LLM)

For chemists who only have a screenshot or PDF of their spectrum:

```python
# src/speq/input/image_spectrum.py

import base64
from pathlib import Path
from anthropic import Anthropic


def parse(image_path: Path) -> dict:
    """
    Extract NMR peak data from a spectrum image using Claude's vision.
    
    This is a best-effort extraction — accuracy is lower than instrument
    file parsing. Suitable for preliminary analysis.
    """
    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode()

    suffix = image_path.suffix.lower()
    media_type = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }.get(suffix, "image/png")

    client = Anthropic()
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data
                    }
                },
                {
                    "type": "text",
                    "text": (
                        "This is a ¹H NMR spectrum. Extract all visible peaks as a JSON array. "
                        "For each peak, provide:\n"
                        '- "shift": chemical shift in ppm (read from x-axis)\n'
                        '- "multiplicity": s/d/t/q/m/dd/dt/br s etc.\n'
                        '- "integral": approximate number of protons if integration shown\n'
                        '- "coupling_hz": J coupling constant if readable\n\n'
                        "Return ONLY the JSON array, no other text. "
                        "Read the axis carefully — NMR spectra have ppm decreasing left to right."
                    )
                }
            ]
        }]
    )

    import json
    peaks = json.loads(message.content[0].text)

    return {
        "peaks": peaks,
        "source_format": "image_extraction",
        "nucleus": "1H",
        "solvent": "unknown",
        "frequency_mhz": None,
        "raw_spectrum": None,
        "extraction_confidence": "approximate",
        "_warning": (
            "Peaks extracted from image via vision LLM. Shifts may have "
            "±0.1-0.2 ppm uncertainty. For high-confidence analysis, "
            "use instrument data files."
        ),
    }
```

---

## Part 7: The VERIFY Pipeline (Detailed)

### 7.1 Step-by-Step Verification Flow

```
                    ┌──────────────┐
                    │  INPUT DATA  │
                    │  (NMR file   │
                    │  or peaks)   │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ PARSE &      │  autodetect.py
                    │ NORMALIZE    │  → standard peak list
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ LOAD ROUTE   │  route.py
                    │ CONTEXT      │  → SM SMILES, product SMILES,
                    │              │     reaction type, step number
                    └──────┬───────┘
                           │
              ┌────────────▼────────────┐
              │ GENERATE DIAGNOSTIC     │  diagnostic_marker_generator.py
              │ MARKERS (if not cached) │  → new/lost/retained groups
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │ PREDICT EXPECTED SHIFTS  │  CASCADE (¹³C) + heuristic (¹H)
              │ FOR PRODUCT STRUCTURE    │  → per-atom predicted shifts
              └────────────┬────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
  ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐
  │ FUNCTIONAL  │  │ PEAK        │  │ PURITY      │
  │ GROUP CHECK │  │ MATCHING    │  │ ANALYSIS    │
  │             │  │             │  │             │
  │ Are the     │  │ Do observed │  │ What % is   │
  │ diagnostic  │  │ peaks match │  │ product vs  │
  │ markers     │  │ predicted   │  │ impurity?   │
  │ present?    │  │ shifts?     │  │             │
  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
         │                 │                 │
         └─────────────────┼─────────────────┘
                           │
              ┌────────────▼────────────┐
              │ AGENT SYNTHESIS         │  Claude API
              │ (combine all results    │
              │  into verification      │
              │  report with confidence │
              │  score)                 │
              └────────────┬────────────┘
                           │
                    ┌──────▼───────┐
                    │ VERIFICATION │
                    │ REPORT       │
                    │              │
                    │ ✅ or ❌     │
                    │ + confidence │
                    │ + purity     │
                    │ + citations  │
                    └──────────────┘
```

### 7.2 Confidence Score Calculation

```python
# src/speq/modes/verify.py (confidence scoring logic)

def compute_verification_confidence(
    diagnostic_results: dict,
    peak_match_results: dict,
    purity_results: dict,
) -> dict:
    """
    Compute overall verification confidence from sub-analyses.
    
    Scoring rubric:
    - Diagnostic markers: 50% weight (did the transformation happen?)
    - Peak matching: 30% weight (does overall spectrum match prediction?)
    - Purity: 20% weight (is the sample clean?)
    
    Returns confidence as percentage + breakdown.
    """
    # Diagnostic marker score
    n_new = len(diagnostic_results["new_groups"])
    n_new_found = sum(1 for g in diagnostic_results["new_groups"] if g["found"])
    n_lost = len(diagnostic_results["lost_groups"])
    n_lost_absent = sum(1 for g in diagnostic_results["lost_groups"] if not g["found"])

    if n_new + n_lost > 0:
        diag_score = (n_new_found + n_lost_absent) / (n_new + n_lost)
    else:
        diag_score = 0.5  # No diagnostic markers — neutral

    # Peak match score (mean absolute deviation vs predicted)
    avg_deviation = peak_match_results.get("mean_deviation_ppm", 999)
    if avg_deviation < 0.1:
        match_score = 1.0
    elif avg_deviation < 0.3:
        match_score = 0.8
    elif avg_deviation < 0.5:
        match_score = 0.5
    else:
        match_score = 0.2

    # Purity score
    purity_pct = purity_results.get("purity_percent", 0)
    purity_score = min(purity_pct / 100, 1.0)

    # Weighted combination
    confidence = (
        0.50 * diag_score +
        0.30 * match_score +
        0.20 * purity_score
    ) * 100

    return {
        "confidence_percent": round(confidence, 1),
        "diagnostic_score": round(diag_score * 100, 1),
        "peak_match_score": round(match_score * 100, 1),
        "purity_score": round(purity_score * 100, 1),
        "verdict": "CONFIRMED" if confidence >= 75 else
                   "LIKELY" if confidence >= 50 else
                   "UNCERTAIN" if confidence >= 25 else
                   "UNLIKELY",
    }
```

---

## Part 8: Testing & Evaluation

### 8.1 Benchmark Suite (Revised for Verification Focus)

```python
# tests/benchmarks/verification_benchmark.py

class VerificationBenchmark:
    """
    Benchmark the VERIFY pipeline against curated test cases.
    
    Test set: Real (structure, spectrum) pairs from published syntheses.
    """

    # ── Benchmark 1: Functional Group Detection ──────────
    def bench_functional_group_detection(self, pipeline):
        """
        Test set: 300 (structure, ¹H NMR peak list) pairs
        - 150 where expected FG IS present
        - 150 where expected FG is NOT present (wrong structure)
        
        Covers: ester, amide, aldehyde, Boc, TMS, aromatic, alcohol,
                amine, nitrile, alkene, carboxylic acid, etc.
        
        Metric: Binary classification accuracy + per-group F1
        
        Target: > 92% overall accuracy
        """
        pass

    # ── Benchmark 2: Purity Estimation ───────────────────
    def bench_purity_estimation(self, pipeline):
        """
        Test set: 100 spectra with known purity (from qNMR studies)
        
        Metric: MAE of purity estimate vs true purity
        
        Target: ±5% MAE for samples with purity > 80%
        """
        pass

    # ── Benchmark 3: End-to-End Verification ─────────────
    def bench_verification_e2e(self, pipeline):
        """
        Test set: 200 synthetic step verifications
        - 100 successful reactions (product matches expectation)
        - 100 failed reactions (wrong product, incomplete, side product)
        
        Metric: Classification accuracy (reaction success vs failure)
        
        Baselines:
        - Chemist with 2 years experience: ~95% (but takes 5-15 min each)
        - Claude alone (no tools): ~55%
        
        Target: > 90% accuracy, < 30 seconds per verification
        """
        pass

    # ── Benchmark 4: BTW Investigation ───────────────────
    def bench_btw_investigation(self, agent):
        """
        Test set: 50 failed reactions with known side products
        
        Metric: Was the correct side product identified in top-3?
        
        Target: > 60% top-3 accuracy
        """
        pass
```

### 8.2 Running Benchmarks

```bash
# Quick check (50 molecules, no GPU models)
speq bench --quick

# Full verification benchmark
speq bench --verify --dataset published_syntheses

# Purity benchmark
speq bench --purity --dataset qnmr_reference

# BTW investigation benchmark (requires ChefNMR)
speq bench --btw --dataset failed_reactions
```

---

## Part 9: Implementation Timeline (Revised)

### Week 1-2: Core Skeleton + VERIFY MVP

```
Day 1-2: Project scaffolding
  □ pyproject.toml with Click, Rich, nmrglue, rdkit, anthropic
  □ Tool base class + registry
  □ Config system (API keys, model paths)
  □ CLI skeleton (verify, check, route, btw, explore)

Day 3-4: Input layer
  □ Bruker parser (nmrglue)
  □ JCAMP-DX parser
  □ CSV peak list parser
  □ Autodetect router
  □ SMILES input handler

Day 5-7: VERIFY mode core
  □ DiagnosticMarkerGenerator (RDKit SMARTS, 20+ groups)
  □ ¹H shift heuristic predictor (additive increments)
  □ Peak matcher (observed vs predicted)
  □ IntegralPurityEstimator
  □ Verification confidence scorer

Day 8-10: Agent wiring
  □ Claude API integration for VERIFY synthesis
  □ System prompt with tool descriptions
  □ Verification report renderer (Rich terminal)
  □ Session persistence

Day 11-14: Ship MVP
  □ Quick-verify command (speq check --smiles "..." --h1 data/)
  □ README with demo GIF
  □ 5 worked examples (esterification, amide coupling, Boc 
    deprotection, Suzuki coupling, alkylation)
  □ Publish to PyPI
```

**The MVP demo:**
```bash
$ pip install speq-cli
$ speq check --smiles "O=C(OCC)c1ccccc1" --h1 ./my_bruker_data/ --solvent CDCl3
```

### Week 3-4: Route Tracking + CASCADE Integration

```
  □ Route data model (multi-step synthesis definition)
  □ Route CLI commands (new, add-step, status)
  □ Step-by-step verification with cumulative tracking
  □ CASCADE 2.0 integration (¹³C shift prediction)
  □ Solvent peak library (Gottlieb reference values)
  □ Common contaminants database
```

### Month 2: BTW Mode + ChefNMR Integration

```
  □ BTW interactive chat mode
  □ ChefNMR wrapper + checkpoint download script
  □ Reaction side product database (20+ reaction types)
  □ Investigation report generator
  □ MestReNova export parser
  □ Spectrum image parser (Claude vision)
```

### Month 3: EXPLORE Mode + Polish

```
  □ Full EXPLORE pipeline (spectra → candidates)
  □ DP4 probability calculator
  □ nmrshiftdb2 database search integration
  □ ELN export format (for pharma users)
  □ Batch verification mode (multiple spectra at once)
  □ Verification benchmark suite (200+ test cases)
```

### Month 4-6: Scale + Proprietary Models

```
  □ Train ¹H shift GNN (fill gap in open-source models)
  □ Train functional group classifier from raw spectra
  □ HSQC/COSY 2D NMR support
  □ Web UI (optional, for non-CLI users)
  □ Enterprise features (team routes, shared databases)
```

---

## Part 10: Data Files

### 10.1 Solvent Residual Peaks Reference

```json
// data/solvent_peaks.json (partial)
{
  "CDCl3": [
    {"shift": 7.26, "nucleus": "1H", "note": "CHCl₃ residual"},
    {"shift": 77.16, "nucleus": "13C", "note": "CDCl₃ triplet center"}
  ],
  "DMSO-d6": [
    {"shift": 2.50, "nucleus": "1H", "note": "DMSO-d₅ residual"},
    {"shift": 39.52, "nucleus": "13C", "note": "DMSO-d₆ septet center"}
  ],
  "D2O": [
    {"shift": 4.79, "nucleus": "1H", "note": "HOD residual"}
  ],
  "CD3OD": [
    {"shift": 3.31, "nucleus": "1H", "note": "CHD₂OD residual"},
    {"shift": 4.87, "nucleus": "1H", "note": "OH residual"},
    {"shift": 49.00, "nucleus": "13C", "note": "CD₃OD septet center"}
  ],
  "Acetone-d6": [
    {"shift": 2.05, "nucleus": "1H", "note": "acetone-d₅ residual"},
    {"shift": 29.84, "nucleus": "13C", "note": "CD₃ carbon"},
    {"shift": 206.26, "nucleus": "13C", "note": "C=O carbon"}
  ]
}
```

### 10.2 Common Contaminants Reference

```json
// data/contaminants.json (partial)
{
  "contaminants": [
    {
      "name": "silicone grease",
      "shift": 0.07,
      "nucleus": "1H",
      "multiplicity": "s",
      "note": "From joints, septa. Persistent."
    },
    {
      "name": "water",
      "shifts_by_solvent": {
        "CDCl3": 1.56,
        "DMSO-d6": 3.33,
        "CD3OD": 4.87,
        "Acetone-d6": 2.84
      },
      "nucleus": "1H",
      "note": "Broad singlet, variable."
    },
    {
      "name": "ethyl acetate",
      "shifts": [4.12, 2.05, 1.26],
      "nucleus": "1H",
      "note": "Common column chromatography solvent."
    },
    {
      "name": "DCM",
      "shift": 5.30,
      "nucleus": "1H",
      "note": "Dichloromethane residual."
    },
    {
      "name": "DMF",
      "shifts": [8.02, 2.96, 2.88],
      "nucleus": "1H",
      "note": "Dimethylformamide. Hard to remove."
    },
    {
      "name": "BHT (butylated hydroxytoluene)",
      "shifts": [6.98, 5.00, 2.27, 1.43],
      "nucleus": "1H",
      "note": "Stabilizer from THF/ether. Distinctive."
    }
  ]
}
```

---

## Part 11: Key Design Decisions (Summary)

| Decision | Choice | Rationale |
|---|---|---|
| Primary workflow | Verification, not elucidation | 80%+ of real NMR use in synthesis is confirming expected products |
| ML models | Integrate published models, not train from scratch | ChefNMR and CASCADE already achieve SOTA. Ship faster, cite properly. |
| ¹³C prediction | CASCADE 2.0 (Paton Lab) | 0.73 ppm MAE, MIT license, pretrained weights available |
| Structure elucidation | ChefNMR (Princeton) | NeurIPS 2025, ~65% top-1 on natural products, atomic diffusion |
| ¹H prediction | Heuristic first, train GNN later | No strong open-source ¹H GNN exists. Additive increment works for verification. |
| Input formats | Bruker + JCAMP + CSV + image | Cover the formats chemists actually have. nmrglue for instrument files. |
| Agent framework | Claude API as planner/synthesizer | Not hardcoded decision trees. Agent selects tools dynamically. |
| Purity analysis | Integral ratio + contaminant lookup | Simple, interpretable, sufficient for screening. Not a replacement for qNMR. |
| BTW escape hatch | Mode switch, not always-on | Keeps the default path fast. Investigation is opt-in when verification fails. |
| Route tracking | Multi-step with cumulative context | Unique differentiator. No existing tool tracks NMR across synthesis steps. |
| Citations | Mandatory on every ToolResult | Scientific reproducibility. Chemists need to know the method behind each claim. |
| Report format | Markdown + optional ELN export | Paste into electronic lab notebooks for regulatory documentation. |

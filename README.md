<p>
  <img src="assets/logo_pix.png" alt="SPEQTRO logo" width="400">
</p>

<p>
  <strong>Autonomous spectroscopy reasoning agent for chemists.</strong>
</p>

| | |
|:--- |:--- |
| <img src="assets/logo_analysis.gif" alt="SPEQTRO thinking animation" width="350"/> | **SPEQTRO** wraps Agentic AI into a CLI tool that helps chemists identify unknown structures from spectroscopic data, predict chemical shifts, search reference databases, and generate reproducible analysis reports — all from the terminal. |

## Features
- **15 built-in tools** across NMR, MS, IR, structure analysis, and database search
- **Interactive REPL** with slash commands, spectroscopy-themed UI, and conversational analysis
- **Web GUI** — `speqtro web` opens a browser interface with chat, verify, predict, and analysis history
- **Multi-format input** — JCAMP-DX, CSV peak lists, Bruker FID, MestReNova exports, and images
- **Vendored ML models** — CASCADE (1H, 13C prediction), ChefNMR (structure elucidation), SSIN (IR classification), ICEBERG/ms-pred (MS/MS fragmentation), DP4/DP5 (structure verification)
- **One-command weight download** — `speqtro fetch-weights` pulls CASCADE, SSIN, DP5, and ICEBERG from HuggingFace
- **Heuristic fallbacks** — tools work without ML models using additive increment rules and empirical tables
- **Mandatory citations** — every result includes method, data sources, and parameters for reproducibility

## Requirements

- Python 3.10 – 3.13
- An Anthropic API

## Installation

### pip (recommended)

```bash
pip install speqtro
```

### With optional extras

```bash
# With RDKit (structure drawing, SMILES validation)
pip install "speqtro[chemistry]"

# With web GUI
pip install "speqtro[web]"

# With MCP server support
pip install "speqtro[mcp]"

# With all ML backends (PyTorch + TensorFlow)
pip install "speqtro[all]"
```

### pipx (isolated environment, no dependency conflicts)

```bash
pipx install speqtro

# With extras
pipx install "speqtro[chemistry]"
```

### Development / editable install

```bash
git clone https://github.com/OhhMoo/SPEQTRO-Agent.git
cd SPEQTRO-Agent
pip install -e ".[dev,chemistry]"
```

---

## Workflow

### Step 1 — First-time setup

Run the interactive wizard after installation. It saves your Anthropic API key, lets you choose the Claude model (Opus for best accuracy, Haiku for speed), and verifies your environment.

```bash
speqtro setup
```

### Step 2 — Download ML model weights

SPEQTRO ships the vendored model code but not the weights (too large for git). Pull everything from HuggingFace in one command:

```bash
speqtro fetch-weights
```

This downloads CASCADE, SSIN, DP5, and ICEBERG into `~/.speqtro/models/`. To download a specific model only:

```bash
speqtro fetch-weights --model cascade2    # 13C NMR predictor
speqtro fetch-weights --model iceberg     # MS/MS fragmentation
speqtro fetch-weights --model ssin        # IR functional group classifier
speqtro fetch-weights --model dp5         # Bayesian NMR scoring
speqtro fetch-weights --model cascade1    # 1H NMR GNN
speqtro fetch-weights --model chefnmr    # NMR structure elucidation (S128 variants, ~8.6 GB)
```

> **ChefNMR** (structure elucidation from NMR) is not on HuggingFace due to licensing. Download the checkpoint from [Zenodo](https://zenodo.org/records/17766755) and place it in `~/Desktop/chefnmr/` or `~/.speqtro/models/chefnmr/`, or configure it explicitly:
> ```bash
> speqtro config set chefnmr.checkpoint /path/to/checkpoint.ckpt
> ```

### Step 3 — Verify your environment

```bash
speqtro doctor
```

This prints a full diagnostic: Python dependencies, loaded tools, ML model weight status (found/missing/path), and API connectivity. Check that CASCADE, SSIN, ICEBERG, and DP5 all show `found` before running analyses.

---

### Example Scenario
In the REPL:

```
speqtro
> I have an unknown compound. 1H NMR in CDCl3: 7.26 (s, 5H), 3.71 (s, 3H), 2.35 (s, 3H).
  13C: 170.5, 55.3, 21.2. What could this be?
```

The full elucidation pipeline runs: ChefNMR generates candidate SMILES → CASCADE scores each against your 13C data → DP4-heuristic ranks by probability → returns a ranked table of candidates.

---

### Scenario 2: Multi-spectroscopy elucidation (NMR + IR + MS)

The most powerful mode. You have NMR peaks, an IR spectrum file, and an MS/MS spectrum from LC-MS.

```
speqtro
> Elucidate this compound:
  1H NMR (CDCl3): 7.35 (m, 5H), 5.12 (s, 2H), 3.72 (s, 3H)
  13C NMR: 171.2, 136.1, 128.5, 128.2, 128.0, 66.8, 52.1
  IR file: /data/compound1.jdx
  MS/MS precursor m/z 180.07, collision energy 20 eV,
  fragment ions: 149.0, 121.0, 105.0, 91.0, 77.0
```

SPEQTRO orchestrates the full pipeline:
1. **SSIN** reads the IR file → detects ester C=O, aromatic C-H
2. **ChefNMR** generates 10 candidate structures from NMR peaks
3. **CASCADE** predicts 13C shifts for each candidate → computes MAD vs observed
4. **DP4-heuristic** ranks candidates by NMR probability
5. **ICEBERG** predicts MS/MS for each candidate → cosine similarity vs your fragment ions
6. **Ensemble scoring** combines all four evidence streams → final ranked output

---

### Scenario 3: Predicting spectra before running an experiment

Planning a synthesis and want to know what the product should look like before you make it:

```bash
# Predict all spectral handles for a target molecule
speqtro predict h1  --smiles "CC(=O)Oc1ccccc1C(=O)O"
speqtro predict c13 --smiles "CC(=O)Oc1ccccc1C(=O)O"
speqtro predict ir  --smiles "CC(=O)Oc1ccccc1C(=O)O"
speqtro predict ms  --smiles "CC(=O)Oc1ccccc1C(=O)O" --adduct "[M+H]+"
```

---

### Reading spectral files directly

SPEQTRO can parse instrument files without manual peak picking:

```bash
# Parse a JCAMP-DX NMR file and send to agent
speqtro --file spectrum.jdx "verify against aspirin"

# Parse a Bruker experiment directory
speqtro --file /data/bruker/1 "what functional groups are present?"

# Parse a MestReNova peak export CSV
speqtro --file peaks.csv "elucidate this structure"
```

---


### Web GUI

```bash
speqtro web
```

Opens a browser-based interface at `http://127.0.0.1:8080` with chat, one-click verify/predict, and a local analysis history log. Requires the `web` extra:


---

## ML Model Weights

The vendored model code is included with the package. Weights are downloaded separately using `speqtro fetch-weights` (HuggingFace) or manually for ChefNMR.

| Model | Task | `fetch-weights` | Config Key | Env Var |
|-------|------|:--------------:|------------|---------|
| CASCADE 1.0 | 1H NMR prediction (GNN) | `--model cascade1` | `cascade1.model_dir` | `SPEQ_CASCADE1_DIR` |
| CASCADE 2.0 | 13C NMR prediction (GNN) | `--model cascade2` | `cascade.model_dir` | `SPEQ_CASCADE_DIR` |
| SSIN | IR functional group detection | `--model ssin` | `ssin.repo_dir` | `SPEQ_SSIN_DIR` |
| DP5 | Bayesian NMR scoring | `--model dp5` | `dp5.repo_dir` | `SPEQ_DP5_DIR` |
| ICEBERG | MS/MS fragmentation (DAG GNN) | `--model iceberg` | `mspred.gen_checkpoint` | `SPEQ_MSPRED_GEN_CKPT` |
| ChefNMR | NMR-to-structure elucidation | `--model chefnmr` | `chefnmr.checkpoint` | `SPEQ_CHEFNMR_CKPT` |

All models gracefully fall back to heuristic methods when weights are absent.

**ICEBERG** uses two checkpoints (generator + intensity predictor). Both are downloaded by `fetch-weights --model iceberg` into `~/.speqtro/models/iceberg/` and discovered automatically. No manual path configuration required.

**ChefNMR** downloads the four S128 (smaller) checkpoints covering all model families (NP, SB, US) into `~/.speqtro/models/chefnmr/` and are discovered automatically. To configure a specific checkpoint explicitly:
```bash
speqtro config set chefnmr.checkpoint /path/to/checkpoint.ckpt
```

---

## Environment Diagnostics

### `speqtro doctor`

Full diagnostic covering: Python dependencies, tool registry, ML model weight status, and Anthropic/PubChem API connectivity.

### `speqtro pytorch`

PyTorch-specific report: version, CUDA/MPS availability, active device, and weight status for SSIN, ICEBERG, and ChefNMR.

### `speqtro tf`

TensorFlow-specific report: version, GPU devices, Keras, h5py, and weight status for CASCADE 1.0 and 2.0.

---

## MCP Server — Use speqtro from Claude.ai, Cursor, and VSCode

speqtro exposes all its spectroscopy tools as a **Model Context Protocol (MCP) server** so you can call them directly from Claude.ai, Cursor, or VSCode Copilot Chat without opening a terminal.

### Install with MCP support

```bash
pip install "speqtro[mcp]"
```

### Available commands

```bash
speqtro mcp list-tools                        # Print all tools as MCP JSON schema
speqtro mcp serve --stdio                     # stdio transport (Claude.ai / Cursor / Claude Code)
speqtro mcp serve --http                      # HTTP/SSE transport on 127.0.0.1:8765
speqtro mcp serve --http --host 0.0.0.0 --port 9000   # Custom host/port
```

### Claude Code

```bash
claude mcp add speqtro -- speqtro mcp serve --stdio
```

### Claude.ai (Desktop)

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "speqtro": {
      "command": "speqtro",
      "args": ["mcp", "serve", "--stdio"]
    }
  }
}
```

### Cursor

Add to `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "speqtro": {
      "command": "speqtro",
      "args": ["mcp", "serve", "--stdio"]
    }
  }
}
```

### VSCode Copilot Chat

Add to `.vscode/mcp.json` in your workspace:

```json
{
  "servers": {
    "speqtro": {
      "type": "stdio",
      "command": "speqtro",
      "args": ["mcp", "serve", "--stdio"]
    }
  }
}
```

---

## Architecture

```
CLI (Typer) → Agent Loop (Claude Agent SDK) → Tool Registry → 15 Tool Modules
                                                             → 6 Vendored ML Models
Web GUI (Starlette) → AgentRunner (SSE streaming) → Tool Registry
MCP Server → Tool Registry
```

### Directory Structure

```
src/speqtro/
├── cli.py                 # Typer entry point + all CLI commands
├── agent/                 # Claude Agent SDK agentic loop
│   ├── loop.py            # Agentic reasoning loop
│   ├── runner.py          # Agent execution (Claude Agent SDK)
│   ├── mcp_server.py      # Model Context Protocol server
│   ├── config.py          # Config (~/.speqtro/config.json)
│   ├── api_check.py       # API connectivity probes
│   └── ...
├── input/                 # Spectral format parsers
│   ├── jcamp.py           # JCAMP-DX
│   ├── csv_peaks.py       # CSV peak lists
│   ├── bruker.py          # Bruker FID
│   ├── autodetect.py      # Format auto-detection
│   └── ...
├── modes/                 # Analysis pipelines
│   ├── verify.py          # VERIFY / EXPLORE pipelines
│   └── pipeline.py        # Full multi-spectroscopy elucidation
├── tools/                 # Tool registry (15 tools)
│   ├── nmr.py             # 1H/13C prediction, JCAMP parsing
│   ├── ms.py              # Exact mass, fragmentation, formula search
│   ├── ir.py              # IR absorption prediction
│   ├── structure.py       # SMILES→formula
│   ├── database.py        # PubChem search
│   ├── cascade.py         # CASCADE 13C predictor
│   ├── chefnmr.py         # ChefNMR elucidation
│   ├── ssin.py            # SSIN IR classifier
│   ├── mspred.py          # ICEBERG MS/MS predictor
│   └── dp5.py             # DP4 structure scoring
├── ui/                    # Terminal interface
│   ├── terminal.py        # Interactive REPL
│   ├── status.py          # Animated thinking spinner
│   └── markdown.py        # Markdown renderer
├── web/                   # Browser-based GUI
│   ├── server.py          # Starlette web server + REST/SSE API
│   ├── leaderboard.py     # SQLite analysis history (~/.speqtro/leaderboard.db)
│   └── static/            # index.html + logo assets
└── vendors/               # Vendored ML model code + weights
    ├── cascade/            # 13C NMR (TensorFlow/kgcnn)
    ├── cascade1/           # 1H NMR (TensorFlow/kgcnn)
    ├── chefnmr/            # Structure elucidation (PyTorch)
    ├── dp5/                # DP4/DP5 scoring (pure SciPy)
    ├── ssin/               # IR classification (PyTorch)
    └── mspred/             # ICEBERG MS/MS fragmentation (PyTorch/DGL)
```

---

## Testing

```bash
pip install -e ".[dev]"
pytest
```

34 unit tests cover the heuristic tool suite (NMR prediction, mass calculation, IR absorption, fragmentation, formula search).

---


## License

MIT — see [LICENSE](LICENSE).

## Third-Party Notices

SPEQTRO vendors code from several open-source ML projects. See [THIRD-PARTY-NOTICES](THIRD-PARTY-NOTICES) and [CITATIONS.md](CITATIONS.md) for full attribution.

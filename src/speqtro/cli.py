"""speqtro CLI — spectroscopy reasoning engine for chemists.

Usage:
    speqtro verify --smiles "CC(=O)Oc1ccccc1C(=O)O" --h1 "7.26,3.71,2.35"
    speqtro verify --cas "50-78-2" --nmr-file spectrum.jdx
    speqtro verify --smiles "CCO" --nmr-file h1.jdx --nmr-file c13.jdx
    speqtro explore --h1 "7.26,3.71,2.35" --c13 "170.5,150.2"
    speqtro predict h1 --smiles "CCO"
    speqtro predict c13 --smiles "CC(=O)O"
    speqtro predict ir --smiles "CCO"
    speqtro predict ms --smiles "CC(=O)Oc1ccccc1C(=O)O"
    speqtro formula --smiles "CCO"
    speqtro mass --formula "C6H12O6"
    speqtro mass-search --mass 180.0634 --tolerance 5
    speqtro tools
    speqtro doctor
    speqtro pytorch
    speqtro tf
"""

import json
import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="speqtro",
    help="Spectroscopy reasoning engine for chemists.",
    invoke_without_command=True,
    no_args_is_help=False,
    add_completion=False,
)
console = Console()


@app.callback(invoke_without_command=True)
def _default(ctx: typer.Context):
    """Launch interactive REPL when no subcommand is given."""
    if ctx.invoked_subcommand is not None:
        return
    # Launch the interactive terminal
    from speqtro.agent.config import Config
    from speqtro.agent.session import Session
    from speqtro.ui.terminal import InteractiveTerminal

    config = Config.load()
    session = Session(config=config, mode="interactive")
    terminal = InteractiveTerminal(session)
    terminal.run()

predict_app = typer.Typer(help="Predict NMR/IR/MS spectra from SMILES.")
app.add_typer(predict_app, name="predict")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _parse_peaks(peak_str: str, nucleus: str) -> list[dict]:
    """Parse comma-separated peak list into dict format."""
    peaks = []
    for p in peak_str.split(","):
        p = p.strip()
        if p:
            peaks.append({"shift": float(p), "nucleus": nucleus})
    return peaks


def _cas_to_smiles(cas: str) -> Optional[str]:
    """Resolve a CAS registry number to SMILES via PubChem REST API."""
    import urllib.request
    import urllib.error

    cas = cas.strip()
    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{cas}/property/IsomericSMILES/JSON"
    )
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
        return data["PropertyTable"]["Properties"][0]["IsomericSMILES"]
    except Exception:
        return None


def _peaks_from_nmr_file(path: Path) -> list[dict]:
    """
    Parse an NMR file (Bruker dir, .jdx, .csv, .txt, image) and return
    observed peaks in the [{shift, nucleus, integral, multiplicity}] format
    expected by verify_product.
    """
    from speqtro.input.autodetect import parse_spectrum

    parsed = parse_spectrum(path)
    nucleus = parsed.get("nucleus", "1H")

    peaks = []
    for p in parsed.get("peaks", []):
        peaks.append({
            "shift": float(p.get("shift", p.get("ppm", 0.0))),
            "nucleus": nucleus,
            "integral": p.get("integral"),
            "multiplicity": p.get("multiplicity"),
        })
    return peaks


def _print_json(data: dict):
    console.print_json(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# speqtro verify
# ---------------------------------------------------------------------------

@app.command()
def verify(
    smiles: Optional[str] = typer.Option(None, "--smiles", "-s", help="Product SMILES"),
    cas: Optional[str] = typer.Option(None, "--cas", help="CAS registry number (resolved to SMILES via PubChem)"),
    sm: Optional[str] = typer.Option(None, "--sm", help="Starting material SMILES"),
    h1: Optional[str] = typer.Option(None, "--h1", help="1H peaks (comma-separated ppm)"),
    c13: Optional[str] = typer.Option(None, "--c13", help="13C peaks (comma-separated ppm)"),
    nmr_file: Optional[List[Path]] = typer.Option(None, "--nmr-file", "-f",
        help="NMR spectrum file (Bruker dir, .jdx, .csv, .txt, image). Repeatable."),
    solvent: str = typer.Option("CDCl3", "--solvent", help="NMR solvent"),
):
    """Verify a proposed product structure against observed NMR peaks.

    Structure input: provide exactly one of --smiles or --cas.
    Spectral input: use --h1/--c13 (ppm lists), --nmr-file (file path), or both.
    """
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from speqtro.modes.verify import verify_product

    # ── Resolve structure input ──────────────────────────────────────────────
    if smiles and cas:
        console.print("[red]Error:[/red] Provide either --smiles or --cas, not both.")
        raise typer.Exit(1)
    if not smiles and not cas:
        console.print("[red]Error:[/red] Provide a product structure: --smiles or --cas")
        raise typer.Exit(1)

    if cas:
        console.print(f"  Resolving CAS [cyan]{cas}[/cyan] via PubChem…")
        smiles = _cas_to_smiles(cas)
        if not smiles:
            console.print(f"[red]Error:[/red] Could not resolve CAS '{cas}' to SMILES (PubChem lookup failed).")
            raise typer.Exit(1)
        console.print(f"  → [green]{smiles}[/green]\n")

    # ── Collect observed peaks ───────────────────────────────────────────────
    observed_peaks: list[dict] = []

    if h1:
        observed_peaks.extend(_parse_peaks(h1, "1H"))
    if c13:
        observed_peaks.extend(_parse_peaks(c13, "13C"))

    if nmr_file:
        for fpath in nmr_file:
            if not fpath.exists():
                console.print(f"[red]Error:[/red] NMR file not found: {fpath}")
                raise typer.Exit(1)
            try:
                file_peaks = _peaks_from_nmr_file(fpath)
            except Exception as e:
                console.print(f"[red]Error parsing {fpath.name}:[/red] {e}")
                raise typer.Exit(1)
            console.print(
                f"  Parsed [cyan]{fpath.name}[/cyan] → "
                f"{len(file_peaks)} peaks "
                f"({file_peaks[0]['nucleus'] if file_peaks else '?'})"
            )
            observed_peaks.extend(file_peaks)

    if not observed_peaks:
        console.print("[red]Error:[/red] Provide spectral data: --h1, --c13, or --nmr-file")
        raise typer.Exit(1)

    console.print(f"\n[bold]Verifying:[/bold] {smiles}")
    console.print(f"  Solvent: {solvent} | Peaks: {len(observed_peaks)}\n")

    result = verify_product(
        smiles=smiles,
        observed_peaks=observed_peaks,
        solvent=solvent,
        sm_smiles=sm,
    )

    verdict = result.get("verdict", "?")
    confidence = result.get("confidence_percent", "?")
    color = {"CONFIRMED": "green", "LIKELY": "yellow", "UNCERTAIN": "cyan"}.get(verdict, "red")

    console.print(f"[bold {color}]VERDICT: {verdict}  ({confidence}% confidence)[/bold {color}]")

    if result.get("peak_match", {}).get("mean_deviation_ppm") is not None:
        console.print(f"  Peak Match MAD: {result['peak_match']['mean_deviation_ppm']:.2f} ppm")
    if result.get("purity_percent") is not None:
        console.print(f"  Purity Estimate: {result['purity_percent']}%")

    console.print()
    _print_json(result)


# ---------------------------------------------------------------------------
# speqtro explore
# ---------------------------------------------------------------------------

@app.command()
def explore(
    h1: str = typer.Option(None, "--h1", help="1H peaks (comma-separated ppm)"),
    c13: str = typer.Option(None, "--c13", help="13C peaks (comma-separated ppm)"),
    solvent: str = typer.Option("CDCl3", "--solvent", help="NMR solvent"),
    candidates: int = typer.Option(10, "-n", "--candidates", help="Number of candidates"),
):
    """Elucidate unknown structure from observed NMR peaks (requires ChefNMR model)."""
    from speqtro.modes.verify import elucidation

    h1_peaks = [{"shift": float(p.strip())} for p in h1.split(",") if p.strip()] if h1 else None
    c13_peaks = [{"shift": float(p.strip())} for p in c13.split(",") if p.strip()] if c13 else None

    if not h1_peaks and not c13_peaks:
        console.print("[red]Error:[/red] Provide at least --h1 or --c13 peaks")
        raise typer.Exit(1)

    console.print(f"\n[bold]Structure Elucidation[/bold]")
    console.print(f"  Solvent: {solvent}")
    if h1_peaks:
        console.print(f"  1H peaks: {len(h1_peaks)}")
    if c13_peaks:
        console.print(f"  13C peaks: {len(c13_peaks)}")
    console.print()

    result = elucidation(
        h1_peaks=h1_peaks,
        c13_peaks=c13_peaks,
        solvent=solvent,
        n_candidates=candidates,
    )

    if "error" in result:
        console.print(f"[red]Error:[/red] {result['error']}")
        raise typer.Exit(1)

    cands = result.get("smiles_candidates", [])
    if cands:
        table = Table(title=f"Top {len(cands)} Candidates")
        table.add_column("#", style="dim")
        table.add_column("SMILES", style="cyan")
        table.add_column("Score", style="green")
        for i, c in enumerate(cands, 1):
            table.add_row(str(i), c.get("smiles", "?"), f"{c.get('score', 0):.3f}")
        console.print(table)
    else:
        console.print("[yellow]No candidates returned.[/yellow]")

    console.print()
    _print_json(result)


# ---------------------------------------------------------------------------
# speqtro predict h1 / c13 / ir / ms
# ---------------------------------------------------------------------------

@predict_app.command("h1")
def predict_h1(
    smiles: str = typer.Option(..., "--smiles", "-s", help="SMILES string"),
):
    """Predict 1H NMR chemical shifts from SMILES (heuristic)."""
    from speqtro.tools.nmr import predict_h1_shifts
    result = predict_h1_shifts(smiles=smiles)
    if "Error" in result.get("summary", ""):
        console.print(f"[red]{result['summary']}[/red]")
        raise typer.Exit(1)

    table = Table(title=f"1H NMR Prediction: {smiles}")
    table.add_column("H count", style="cyan")
    table.add_column("Shift (ppm)", style="green")
    table.add_column("Range", style="dim")
    table.add_column("Environment")
    for p in result.get("predictions", []):
        table.add_row(
            f"{p['num_H']}H",
            f"~{p['estimated_shift_ppm']}",
            p["range_ppm"],
            p["environment"],
        )
    console.print(table)


@predict_app.command("c13")
def predict_c13(
    smiles: str = typer.Option(..., "--smiles", "-s", help="SMILES string"),
):
    """Predict 13C NMR chemical shifts from SMILES (heuristic)."""
    from speqtro.tools.nmr import predict_c13_shifts
    result = predict_c13_shifts(smiles=smiles)
    if "Error" in result.get("summary", ""):
        console.print(f"[red]{result['summary']}[/red]")
        raise typer.Exit(1)

    table = Table(title=f"13C NMR Prediction: {smiles}")
    table.add_column("Atom", style="cyan")
    table.add_column("Shift (ppm)", style="green")
    table.add_column("Range", style="dim")
    table.add_column("Environment")
    for p in result.get("predictions", []):
        mult = {0: "C", 1: "CH", 2: "CH2", 3: "CH3"}.get(p["num_H"], f"CH{p['num_H']}")
        table.add_row(
            f"C{p['atom_idx']} ({mult})",
            f"~{p['estimated_shift_ppm']}",
            p["range_ppm"],
            p["environment"],
        )
    console.print(table)


@predict_app.command("ir")
def predict_ir(
    smiles: str = typer.Option(..., "--smiles", "-s", help="SMILES string"),
):
    """Predict IR absorption bands from SMILES."""
    from speqtro.tools.ir import predict_absorptions
    result = predict_absorptions(smiles=smiles)
    if "Error" in result.get("summary", ""):
        console.print(f"[red]{result['summary']}[/red]")
        raise typer.Exit(1)

    table = Table(title=f"IR Prediction: {smiles}")
    table.add_column("Functional Group")
    table.add_column("Range (cm-1)", style="cyan")
    table.add_column("Intensity", style="green")
    table.add_column("Notes", style="dim")
    for b in result.get("bands", []):
        table.add_row(
            b["functional_group"],
            b["wavenumber_range"],
            b["intensity"],
            b.get("notes", ""),
        )
    console.print(table)


@predict_app.command("ms")
def predict_ms(
    smiles: str = typer.Option(..., "--smiles", "-s", help="SMILES string"),
    adduct: str = typer.Option("[M+H]+", "--adduct", help="Precursor adduct"),
):
    """Predict MS/MS fragmentation from SMILES."""
    from speqtro.tools.ms import fragment_predict
    result = fragment_predict(smiles=smiles, precursor_adduct=adduct)
    if "Error" in result.get("summary", ""):
        console.print(f"[red]{result['summary']}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Precursor {adduct}:[/bold] m/z {result['precursor_mz']:.4f}")
    console.print(f"  Formula: {result['formula']} | Mass: {result['monoisotopic_mass']:.4f} Da\n")

    if result["predicted_neutral_losses"]:
        table = Table(title="Predicted Neutral Losses")
        table.add_column("Loss")
        table.add_column("Mass (Da)", style="cyan")
        table.add_column("Fragment m/z", style="green")
        table.add_column("Functional Group", style="dim")
        for fl in result["predicted_neutral_losses"]:
            table.add_row(fl["neutral_loss"], f"{fl['loss_mass_da']:.4f}",
                          f"{fl['fragment_mz']:.4f}", fl["functional_group"])
        console.print(table)

    if result["characteristic_fragments"]:
        table = Table(title="Characteristic Fragments")
        table.add_column("m/z", style="cyan")
        table.add_column("Formula", style="green")
        table.add_column("Description")
        for cf in result["characteristic_fragments"]:
            table.add_row(f"{cf['mz']:.4f}", cf["formula"], cf["description"])
        console.print(table)


# ---------------------------------------------------------------------------
# speqtro formula
# ---------------------------------------------------------------------------

@app.command()
def formula(
    smiles: str = typer.Option(..., "--smiles", "-s", help="SMILES string"),
):
    """Calculate molecular formula, MW, exact mass, and DoU from SMILES."""
    from speqtro.tools.structure import smiles_to_formula
    result = smiles_to_formula(smiles=smiles)
    if "Error" in result.get("summary", ""):
        console.print(f"[red]{result['summary']}[/red]")
        raise typer.Exit(1)
    console.print(f"\n  [bold]Formula:[/bold]    {result['formula']}")
    console.print(f"  [bold]MW:[/bold]         {result['molecular_weight']:.3f} Da")
    console.print(f"  [bold]Exact Mass:[/bold] {result['exact_mass']:.6f} Da")
    console.print(f"  [bold]DoU:[/bold]        {result['degree_of_unsaturation']:.1f}")
    console.print()


# ---------------------------------------------------------------------------
# speqtro mass / mass-search
# ---------------------------------------------------------------------------

@app.command()
def mass(
    formula_str: str = typer.Option(..., "--formula", "-f", help="Molecular formula (e.g. C6H12O6)"),
):
    """Calculate exact mass and common adducts from a molecular formula."""
    from speqtro.tools.ms import calc_exact_mass
    result = calc_exact_mass(formula=formula_str)
    if "Error" in result.get("summary", ""):
        console.print(f"[red]{result['summary']}[/red]")
        raise typer.Exit(1)

    console.print(f"\n  [bold]Formula:[/bold] {result['formula']}")
    console.print(f"  [bold]Monoisotopic Mass:[/bold] {result['monoisotopic_mass']:.6f} Da\n")

    table = Table(title="Adducts")
    table.add_column("Adduct", style="cyan")
    table.add_column("m/z", style="green")
    for name, mz in result["adducts"].items():
        table.add_row(name, f"{mz:.4f}")
    console.print(table)


@app.command("mass-search")
def mass_search(
    observed: float = typer.Option(..., "--mass", "-m", help="Observed exact mass (Da)"),
    tolerance: float = typer.Option(5.0, "--tolerance", "-t", help="Tolerance in ppm"),
    adduct: str = typer.Option("neutral", "--adduct", help="Adduct correction"),
    max_c: int = typer.Option(20, "--max-c", help="Maximum carbons"),
):
    """Search for molecular formula candidates matching an observed mass."""
    from speqtro.tools.ms import formula_from_mass
    result = formula_from_mass(
        observed_mass=str(observed), tolerance_ppm=tolerance,
        max_carbons=max_c, adduct=adduct,
    )
    if "Error" in result.get("summary", ""):
        console.print(f"[red]{result['summary']}[/red]")
        raise typer.Exit(1)

    cands = result.get("candidates", [])
    console.print(f"\n  Observed: {observed:.4f} Da | Adduct: {adduct} | Tolerance: +/-{tolerance} ppm")
    console.print(f"  {len(cands)} candidate(s) found\n")

    if cands:
        table = Table(title="Formula Candidates")
        table.add_column("Formula", style="cyan")
        table.add_column("Exact Mass", style="green")
        table.add_column("Error (ppm)", style="yellow")
        table.add_column("DoU")
        for c in cands:
            table.add_row(c["formula"], f"{c['exact_mass']:.6f}",
                          f"{c['mass_error_ppm']:+.2f}", f"{c['degree_of_unsaturation']:.1f}")
        console.print(table)


# ---------------------------------------------------------------------------
# speqtro tools
# ---------------------------------------------------------------------------

@app.command()
def tools():
    """List all available spectroscopy tools."""
    from speqtro.tools import registry, ensure_loaded, tool_load_errors
    ensure_loaded()

    console.print(registry.list_tools_table())

    errors = tool_load_errors()
    if errors:
        console.print(f"\n[yellow]Some tool modules failed to load:[/yellow]")
        for mod, err in errors.items():
            console.print(f"  [dim]{mod}:[/dim] {err}")


# ---------------------------------------------------------------------------
# speqtro setup
# ---------------------------------------------------------------------------

@app.command()
def setup():
    """Interactive first-time setup: configure API key, check deps, verify connectivity."""
    from pathlib import Path
    from rich import box
    from rich.panel import Panel
    from rich.text import Text

    from speqtro.agent.config import Config, CONFIG_DIR

    config = Config.load()

    # ── Header ────────────────────────────────────────────────────────
    header = Text()
    header.append("speqtro setup", style="bold cyan")
    header.append("\n\nInteractive configuration for speqtro.\n", style="dim")
    header.append("Press Enter to keep current values.", style="dim")
    console.print(Panel(header, border_style="cyan", box=box.ROUNDED, padding=(1, 3)))

    # ── Step 1: Anthropic API key ─────────────────────────────────────
    console.print("\n[bold]Step 1/4:[/bold] Anthropic API Key\n")
    current_key = config.llm_api_key()
    if current_key:
        masked = current_key[:8] + "..." + current_key[-4:]
        console.print(f"  Current key: [dim]{masked}[/dim]")
    else:
        console.print("  [yellow]No API key configured.[/yellow]")

    try:
        new_key = input("  Enter API key (or Enter to skip): ").strip()
    except (EOFError, KeyboardInterrupt):
        new_key = ""

    if new_key:
        config.set("llm.api_key", new_key)
        console.print("  [green]API key saved.[/green]")
    elif current_key:
        console.print("  [dim]Keeping existing key.[/dim]")
    else:
        console.print("  [yellow]Skipped. Set later with: export ANTHROPIC_API_KEY=sk-ant-...[/yellow]")

    # ── Step 2: Model selection ───────────────────────────────────────
    console.print("\n[bold]Step 2/4:[/bold] Default LLM Model\n")
    models = [
        ("claude-opus-4-6",           "Opus 4.6   -- most capable, highest quality"),
        ("claude-sonnet-4-6",         "Sonnet 4.6 -- balanced speed and quality"),
        ("claude-haiku-4-5-20251001", "Haiku 4.5  -- fastest, lowest cost"),
    ]
    current_model = config.get("llm.model", "claude-opus-4-6")
    for i, (mid, label) in enumerate(models, 1):
        marker = "[green]*[/green]" if mid == current_model else " "
        console.print(f"  {marker} [{i}] {label}")

    try:
        choice = input("\n  Select model [1-3] (or Enter to keep current): ").strip()
    except (EOFError, KeyboardInterrupt):
        choice = ""

    if choice.isdigit() and 1 <= int(choice) <= len(models):
        mid, label = models[int(choice) - 1]
        config.set("llm.model", mid)
        console.print(f"  [green]Model set to {label.split('--')[0].strip()}[/green]")
    else:
        console.print(f"  [dim]Keeping: {current_model}[/dim]")

    # ── Step 3: Save config ──────────────────────────────────────────
    console.print("\n[bold]Step 3/4:[/bold] Saving Configuration\n")
    config.save()
    console.print(f"  [green]Config saved to {CONFIG_DIR / 'config.json'}[/green]")

    # ── Step 4: Run doctor ───────────────────────────────────────────
    console.print("\n[bold]Step 4/4:[/bold] Verifying Installation\n")
    _run_doctor_checks(config)

    # ── Done ─────────────────────────────────────────────────────────
    console.print(Panel(
        "[bold green]Setup complete![/bold green]\n\n"
        "  Launch the interactive REPL:  [cyan]speqtro[/cyan]\n"
        "  Run a quick prediction:       [cyan]speqtro predict h1 --smiles CCO[/cyan]\n"
        "  Check status anytime:         [cyan]speqtro doctor[/cyan]",
        border_style="green",
        box=box.ROUNDED,
        padding=(1, 3),
    ))


# ---------------------------------------------------------------------------
# speqtro doctor
# ---------------------------------------------------------------------------

def _run_doctor_checks(config=None):
    """Core doctor logic, shared between `speqtro doctor` and `speqtro setup`."""
    from pathlib import Path
    from rich import box
    from rich.panel import Panel
    from speqtro.ui.status import ThinkingStatus

    if config is None:
        from speqtro.agent.config import Config
        config = Config.load()

    # ── 1. Python dependencies ────────────────────────────────────────
    dep_checks = [
        ("rdkit",      "RDKit",          "cheminformatics",        "pip install rdkit"),
        ("torch",      "PyTorch",        "ML models (SSIN, ChefNMR, ms-pred)", "pip install torch"),
        ("tensorflow", "TensorFlow",     "CASCADE 13C prediction", "pip install tensorflow-cpu"),
        ("dgl",        "DGL",            "ms-pred graph networks", "pip install dgl"),
        ("lightning",  "Lightning",      "ChefNMR / ms-pred",      "pip install lightning"),
        ("jcamp",      "jcamp",          "IR file parsing",        "pip install jcamp"),
        ("scipy",      "SciPy",          "DP4/DP5 scoring",        "pip install scipy"),
        ("numpy",      "NumPy",          "numerical computing",    "pip install numpy"),
        ("pandas",     "Pandas",         "data handling",          "pip install pandas"),
        ("anthropic",  "Anthropic SDK",  "Claude API (agent mode)","pip install anthropic"),
        ("claude_agent_sdk", "Agent SDK","agentic loop",           "pip install claude-agent-sdk"),
    ]

    dep_table = Table(title="Python Dependencies", box=box.SIMPLE_HEAVY)
    dep_table.add_column("Package", style="cyan", min_width=16)
    dep_table.add_column("Purpose", style="dim")
    dep_table.add_column("Status", min_width=8)
    dep_table.add_column("Version", style="dim")

    n_ok = 0
    with ThinkingStatus(console, "doctor"):
        for module_name, display_name, purpose, install_cmd in dep_checks:
            try:
                mod = __import__(module_name)
                version = getattr(mod, "__version__", "?")
                dep_table.add_row(display_name, purpose, "[green]OK[/green]", version)
                n_ok += 1
            except Exception:
                dep_table.add_row(display_name, purpose, "[red]MISSING[/red]", f"[dim]{install_cmd}[/dim]")

    console.print(dep_table)
    console.print(f"  {n_ok}/{len(dep_checks)} packages installed\n")

    # ── 2. Tool registry ──────────────────────────────────────────────
    tool_table = Table(title="Tool Registry", box=box.SIMPLE_HEAVY)
    tool_table.add_column("Category", style="cyan")
    tool_table.add_column("Tools", style="green")
    tool_table.add_column("Status")

    with ThinkingStatus(console, "doctor_tools"):
        from speqtro.tools import registry, ensure_loaded, tool_load_errors
        ensure_loaded()
        tool_list = registry.list_tools()
        load_errors = tool_load_errors()

        for cat in registry.categories():
            cat_tools = registry.list_tools(category=cat)
            names = ", ".join(t.name.split(".")[-1] for t in cat_tools)
            tool_table.add_row(cat, names, f"[green]{len(cat_tools)} loaded[/green]")

        if load_errors:
            for mod, err in load_errors.items():
                short_err = err[:60] + "..." if len(err) > 60 else err
                tool_table.add_row(mod, "", f"[yellow]failed: {short_err}[/yellow]")

    console.print(tool_table)
    console.print(f"  {len(tool_list)} tools loaded across {len(registry.categories())} categories\n")

    # ── 3. ML model weights ───────────────────────────────────────────
    import os

    home_models = Path.home() / ".speqtro" / "models"

    def _resolve_model_path(config_key, env_var, home_subdir, sentinel_file):
        """Return (path_str, source_label) for the first found location."""
        p = config.get(config_key) or os.environ.get(env_var, "")
        if p and Path(p).exists():
            return str(p), "config/env"
        home_p = home_models / home_subdir
        if home_p.exists() and (home_p / sentinel_file).exists():
            return str(home_p), "~/.speqtro"
        return p or "", None

    model_table = Table(title="ML Model Weights", box=box.SIMPLE_HEAVY)
    model_table.add_column("Model", style="cyan", min_width=12)
    model_table.add_column("Purpose", style="dim")
    model_table.add_column("Status")
    model_table.add_column("Source", style="dim")
    model_table.add_column("Path", style="dim")

    with ThinkingStatus(console, "doctor_models"):
        model_checks = [
            ("CASCADE 1.0", "1H NMR GNN (Paton)",
             *_resolve_model_path("cascade1.model_dir", "SPEQ_CASCADE1_DIR", "cascade1", "best_model.hdf5")),
            ("CASCADE 2.0", "13C NMR GNN (Paton)",
             *_resolve_model_path("cascade.model_dir",  "SPEQ_CASCADE_DIR",  "cascade2", "best_model.h5")),
            ("DP5",         "Bayesian NMR scoring",
             *_resolve_model_path("dp5.repo_dir",        "SPEQ_DP5_DIR",      "dp5",      "c_w_kde_mean_s_0.025.p")),
            ("SSIN",        "IR functional group",
             *_resolve_model_path("ssin.repo_dir",        "SPEQ_SSIN_DIR",     "ssin",     "save/model")),
            ("ChefNMR",     "NMR structure elucidation",
             *_resolve_model_path("chefnmr.checkpoint", "SPEQ_CHEFNMR_CKPT", "chefnmr", "NP-H10kC10k-S128-epoch26599.ckpt")),
            ("ICEBERG",     "MS/MS fragmentation",
             *_resolve_model_path("mspred.gen_checkpoint", "SPEQ_MSPRED_GEN_CKPT",
                                  "iceberg", "iceberg_dag_gen_msg_best.ckpt")),
        ]
        for name, purpose, path, source in model_checks:
            short_path = ("..." + str(path)[-37:]) if len(str(path)) > 40 else str(path)
            if source:
                model_table.add_row(name, purpose, "[green]found[/green]", source, short_path)
            elif path:
                model_table.add_row(name, purpose, "[red]not found[/red]", "", short_path[:40])
            else:
                model_table.add_row(name, purpose, "[yellow]missing[/yellow]",
                                     "run: speqtro fetch-weights", "")

    console.print(model_table)
    console.print()

    # ── 4. API connectivity ───────────────────────────────────────────
    from speqtro.agent.api_check import check_all

    with ThinkingStatus(console, "doctor_api"):
        api_results = check_all(config)

    api_table = Table(title="External Services", box=box.SIMPLE_HEAVY)
    api_table.add_column("Service", style="cyan", min_width=16)
    api_table.add_column("Status")
    api_table.add_column("Latency", style="dim", min_width=10)
    api_table.add_column("Details", style="dim")

    for r in api_results:
        status = "[green]OK[/green]" if r.ok else "[red]FAIL[/red]"
        latency = f"{r.latency_ms:.0f} ms" if r.latency_ms is not None else "--"
        detail = r.note
        if r.error and not r.ok:
            detail = r.error[:60] + "..." if len(r.error) > 60 else r.error
        api_table.add_row(r.name, status, latency, detail)

    console.print(api_table)

    # ── 5. Configuration ──────────────────────────────────────────────
    console.print()
    console.print(config.to_table())
    console.print()


@app.command()
def doctor():
    """Full diagnostic: check dependencies, tools, model weights, and API connectivity."""
    from rich import box
    from rich.panel import Panel
    from rich.text import Text

    header = Text()
    header.append("speqtro doctor", style="bold cyan")
    header.append("  --  system diagnostic report\n", style="dim")
    console.print(Panel(header, border_style="cyan", box=box.ROUNDED, padding=(0, 3)))
    console.print()

    _run_doctor_checks()


# ---------------------------------------------------------------------------
# speqtro fetch-weights
# ---------------------------------------------------------------------------

_HF_REPO = "SPEQTRO/SPEQTRO_0.1.0"

# HuggingFace path → local path (relative to ~/.speqtro/models/)
_WEIGHT_MANIFEST: dict[str, list[tuple[str, str]]] = {
    "cascade1": [
        ("cascade1/best_model.hdf5",    "cascade1/best_model.hdf5"),
        ("cascade1/preprocessor.p",     "cascade1/preprocessor.p"),
    ],
    "cascade2": [
        ("cascade2/best_model.h5",       "cascade2/best_model.h5"),
        ("cascade2/preprocessor_orig.p", "cascade2/preprocessor_orig.p"),
    ],
    "dp5": [
        ("dp5/c_w_kde_mean_s_0.025.p",   "dp5/c_w_kde_mean_s_0.025.p"),
        ("dp5/i_w_kde_mean_s_0.025.p",   "dp5/i_w_kde_mean_s_0.025.p"),
        ("dp5/folded_scaled_errors.p",   "dp5/folded_scaled_errors.p"),
    ],
    "ssin": [
        *[(f"ssin/alcohol/model_{i}.pt",  f"ssin/save/model/alcohol/model_{i}.pt")  for i in range(5)],
        *[(f"ssin/aldehyde/model_{i}.pt", f"ssin/save/model/aldehyde/model_{i}.pt") for i in range(5)],
    ],
    "iceberg": [
        ("iceberg/iceberg_dag_gen_msg_best.ckpt",   "iceberg/iceberg_dag_gen_msg_best.ckpt"),
        ("iceberg/iceberg_dag_inten_msg_best.ckpt", "iceberg/iceberg_dag_inten_msg_best.ckpt"),
    ],
    "chefnmr": [
        ("chefnmr-weights/NP-H10kC10k-S128-epoch26599.ckpt", "chefnmr/NP-H10kC10k-S128-epoch26599.ckpt"),
        ("chefnmr-weights/SB-H10kC80-S128-epoch10099.ckpt",  "chefnmr/SB-H10kC80-S128-epoch10099.ckpt"),
        ("chefnmr-weights/US-H10kC10k-S128-epoch10849.ckpt", "chefnmr/US-H10kC10k-S128-epoch10849.ckpt"),
        ("chefnmr-weights/US-H10kC80-S128-epoch6299.ckpt",   "chefnmr/US-H10kC80-S128-epoch6299.ckpt"),
    ],
}


@app.command("fetch-weights")
def fetch_weights(
    model: str = typer.Option("all", "--model", "-m",
                               help="Model to fetch: all, cascade1, cascade2, ssin, dp5, iceberg, chefnmr"),
    dest: str = typer.Option(None, "--dest",
                              help="Destination dir (default: ~/.speqtro/models)"),
    repo: str = typer.Option(_HF_REPO, "--repo", help="HuggingFace repo ID"),
    force: bool = typer.Option(False, "--force", "-f", is_flag=True,
                                help="Re-download even if file already exists"),
):
    """Download ML model weights from HuggingFace into ~/.speqtro/models/."""
    import os
    from pathlib import Path as _Path

    try:
        from huggingface_hub import hf_hub_download, whoami
        from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError
    except ImportError:
        console.print("[red]Error:[/red] huggingface_hub not installed.")
        console.print("  Run: [cyan]pip install huggingface-hub[/cyan]")
        raise typer.Exit(1)

    base_dir = _Path(dest) if dest else (_Path.home() / ".speqtro" / "models")
    models_to_fetch = list(_WEIGHT_MANIFEST.keys()) if model == "all" else [model]

    for m in models_to_fetch:
        if m not in _WEIGHT_MANIFEST:
            console.print(f"[red]Unknown model:[/red] {m}. Choose from: {', '.join(_WEIGHT_MANIFEST)}")
            raise typer.Exit(1)

    # Check auth early so we fail fast instead of stalling on the first download
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    try:
        whoami(token=token)
    except Exception:
        # Not logged in — downloads may still work for public repos, warn but continue
        if not token:
            console.print(
                "[yellow]Warning:[/yellow] Not logged in to HuggingFace. "
                "Private/gated repos will fail.\n"
                "  Run [cyan]huggingface-cli login[/cyan] or set [cyan]HF_TOKEN[/cyan]."
            )

    total = sum(len(_WEIGHT_MANIFEST[m]) for m in models_to_fetch)
    console.print(f"\nFetching {total} files from [cyan]{repo}[/cyan]")
    console.print(f"Destination: [dim]{base_dir}[/dim]\n")

    ok = skipped = failed = 0
    for m in models_to_fetch:
        console.print(f"[bold]{m}[/bold]")
        for hf_path, local_rel in _WEIGHT_MANIFEST[m]:
            local_path = base_dir / local_rel
            if local_path.exists() and not force:
                console.print(f"  [dim]skip[/dim]  {local_rel}")
                skipped += 1
                continue
            local_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                hf_hub_download(
                    repo_id=repo,
                    filename=hf_path,
                    repo_type="model",
                    local_dir=str(local_path.parent),
                    token=token or None,
                )
                # hf_hub_download preserves repo subdirectory structure under local_dir,
                # so the file lands at local_dir/hf_path (not local_dir/basename).
                downloaded = local_path.parent / hf_path
                if downloaded != local_path and downloaded.exists():
                    downloaded.rename(local_path)
                console.print(f"  [green]done[/green]  {local_rel}")
                ok += 1
            except GatedRepoError:
                console.print(
                    f"  [red]fail[/red]  {local_rel}  "
                    "(gated repo — accept terms at huggingface.co and set HF_TOKEN)"
                )
                failed += 1
            except RepositoryNotFoundError:
                console.print(
                    f"  [red]fail[/red]  {local_rel}  "
                    f"(repo '{repo}' not found or private — check --repo)"
                )
                failed += 1
            except Exception as exc:
                console.print(f"  [red]fail[/red]  {local_rel}  ({exc})")
                failed += 1

    console.print()
    if failed:
        console.print(f"[yellow]{ok} downloaded, {skipped} skipped, {failed} failed[/yellow]")
    else:
        console.print(f"[green]{ok} downloaded, {skipped} already present[/green]")
    console.print("Run [cyan]speqtro doctor[/cyan] to verify model status.")


# ---------------------------------------------------------------------------
# speqtro pytorch
# ---------------------------------------------------------------------------

@app.command()
def pytorch():
    """Show PyTorch environment: version, device, CUDA/MPS info, and dependent models."""
    import os
    from pathlib import Path
    from rich import box
    from rich.panel import Panel
    from rich.text import Text
    from speqtro.ui.status import ThinkingStatus

    header = Text()
    header.append("speqtro pytorch", style="bold cyan")
    header.append("  --  PyTorch environment report\n", style="dim")
    console.print(Panel(header, border_style="cyan", box=box.ROUNDED, padding=(0, 3)))
    console.print()

    with ThinkingStatus(console, "doctor"):
        # ── PyTorch core ──────────────────────────────────────────────
        try:
            import torch
            pt_version = torch.__version__
            pt_ok = True
        except Exception as exc:
            pt_version = None
            pt_ok = False
            pt_exc = str(exc)

        # ── CUDA ──────────────────────────────────────────────────────
        cuda_available = False
        cuda_version = None
        cuda_device_name = None
        if pt_ok:
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                cuda_version = torch.version.cuda
                cuda_device_name = torch.cuda.get_device_name(0)

        # ── MPS (Apple Silicon) ───────────────────────────────────────
        mps_available = pt_ok and hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

        # ── Companion libs ────────────────────────────────────────────
        companion_checks = [
            ("dgl",       "DGL",            "ms-pred graph networks",   "pip install dgl"),
            ("lightning", "Lightning",      "ChefNMR / ms-pred trainer","pip install lightning"),
            ("numpy",     "NumPy",          "tensor interop",           "pip install numpy"),
            ("scipy",     "SciPy",          "DP4/DP5 scoring",          "pip install scipy"),
        ]
        companions = []
        for mod_name, display, purpose, install in companion_checks:
            try:
                m = __import__(mod_name)
                companions.append((display, purpose, True, getattr(m, "__version__", "?")))
            except Exception:
                companions.append((display, purpose, False, install))

    # ── Print: framework status ───────────────────────────────────────
    fw_table = Table(title="PyTorch Framework", box=box.SIMPLE_HEAVY)
    fw_table.add_column("Property", style="cyan", min_width=18)
    fw_table.add_column("Value")

    if pt_ok:
        fw_table.add_row("PyTorch version", f"[green]{pt_version}[/green]")
    else:
        fw_table.add_row("PyTorch", f"[red]NOT INSTALLED[/red]  {pt_exc[:60]}")
        fw_table.add_row("Install", "[dim]pip install torch[/dim]")

    if pt_ok:
        if cuda_available:
            fw_table.add_row("CUDA", f"[green]available[/green]  (CUDA {cuda_version})")
            fw_table.add_row("GPU", cuda_device_name or "unknown")
        else:
            fw_table.add_row("CUDA", "[yellow]not available[/yellow]  (CPU-only)")
        if mps_available:
            fw_table.add_row("MPS (Apple)", "[green]available[/green]")
        active = "cuda:0" if cuda_available else ("mps" if mps_available else "cpu")
        fw_table.add_row("Active device", f"[bold]{active}[/bold]")

    console.print(fw_table)
    console.print()

    # ── Print: companion libraries ────────────────────────────────────
    comp_table = Table(title="Companion Libraries", box=box.SIMPLE_HEAVY)
    comp_table.add_column("Package", style="cyan", min_width=12)
    comp_table.add_column("Purpose", style="dim")
    comp_table.add_column("Status", min_width=8)
    comp_table.add_column("Version / Install", style="dim")

    for display, purpose, ok, info in companions:
        if ok:
            comp_table.add_row(display, purpose, "[green]OK[/green]", info)
        else:
            comp_table.add_row(display, purpose, "[red]MISSING[/red]", info)

    console.print(comp_table)
    console.print()

    # ── Print: dependent models ───────────────────────────────────────
    from speqtro.agent.config import Config
    config = Config.load()
    home_models = Path.home() / ".speqtro" / "models"

    model_table = Table(title="Models Using PyTorch", box=box.SIMPLE_HEAVY)
    model_table.add_column("Model", style="cyan", min_width=12)
    model_table.add_column("Purpose", style="dim")
    model_table.add_column("Weights", min_width=10)
    model_table.add_column("Hint", style="dim")

    pt_models = [
        ("SSIN",    "IR functional group classifier",
         home_models / "ssin" / "save" / "model",
         "speqtro fetch-weights --model ssin"),
        ("ICEBERG", "MS/MS fragmentation GNN",
         home_models / "iceberg",
         "speqtro fetch-weights --model iceberg"),
        ("ChefNMR", "NMR structure elucidation",
         home_models / "chefnmr",
         "speqtro fetch-weights --model chefnmr"),
    ]

    for name, purpose, weight_path, hint in pt_models:
        found = weight_path.exists()
        status = "[green]found[/green]" if found else "[yellow]missing[/yellow]"
        model_table.add_row(name, purpose, status, "" if found else hint)

    console.print(model_table)
    console.print()

    if not pt_ok:
        console.print("[yellow]Install PyTorch:[/yellow]  https://pytorch.org/get-started/locally/")
    elif not cuda_available:
        console.print("[dim]No GPU detected. CPU inference is supported but slower for ICEBERG/SSIN.[/dim]")
    console.print()


# ---------------------------------------------------------------------------
# speqtro tf
# ---------------------------------------------------------------------------

@app.command()
def tf():
    """Show TensorFlow environment: version, GPU info, and dependent models (CASCADE)."""
    import os
    from pathlib import Path
    from rich import box
    from rich.panel import Panel
    from rich.text import Text
    from speqtro.ui.status import ThinkingStatus

    header = Text()
    header.append("speqtro tf", style="bold cyan")
    header.append("  --  TensorFlow environment report\n", style="dim")
    console.print(Panel(header, border_style="cyan", box=box.ROUNDED, padding=(0, 3)))
    console.print()

    with ThinkingStatus(console, "doctor"):
        # ── TensorFlow core ───────────────────────────────────────────
        try:
            import tensorflow as tensorflow_mod
            tf_version = tensorflow_mod.__version__
            tf_ok = True
        except Exception as exc:
            tf_version = None
            tf_ok = False
            tf_exc = str(exc)

        # ── GPU devices ───────────────────────────────────────────────
        gpu_devices = []
        if tf_ok:
            try:
                gpu_devices = tensorflow_mod.config.list_physical_devices("GPU")
            except Exception:
                pass

        # ── Keras ─────────────────────────────────────────────────────
        keras_version = None
        keras_ok = False
        if tf_ok:
            try:
                import keras
                keras_version = keras.__version__
                keras_ok = True
            except Exception:
                # TF 2.x bundles keras as tf.keras
                try:
                    keras_version = tensorflow_mod.keras.__version__
                    keras_ok = True
                except Exception:
                    pass

        # ── h5py (required for .hdf5 / .h5 weights) ──────────────────
        h5py_ok = False
        h5py_version = None
        try:
            import h5py
            h5py_version = h5py.__version__
            h5py_ok = True
        except Exception:
            pass

        # ── NumPy ─────────────────────────────────────────────────────
        numpy_ok = False
        numpy_version = None
        try:
            import numpy as np
            numpy_version = np.__version__
            numpy_ok = True
        except Exception:
            pass

    # ── Print: framework status ───────────────────────────────────────
    fw_table = Table(title="TensorFlow Framework", box=box.SIMPLE_HEAVY)
    fw_table.add_column("Property", style="cyan", min_width=18)
    fw_table.add_column("Value")

    if tf_ok:
        fw_table.add_row("TensorFlow version", f"[green]{tf_version}[/green]")
        if gpu_devices:
            for i, dev in enumerate(gpu_devices):
                fw_table.add_row(f"GPU {i}", f"[green]{dev.name}[/green]")
        else:
            fw_table.add_row("GPU", "[yellow]none detected[/yellow]  (CPU-only)")
        fw_table.add_row(
            "Keras",
            f"[green]{keras_version}[/green]" if keras_ok else "[yellow]not found[/yellow]",
        )
    else:
        fw_table.add_row("TensorFlow", f"[red]NOT INSTALLED[/red]  {tf_exc[:60]}")
        fw_table.add_row("Install (CPU)", "[dim]pip install tensorflow-cpu[/dim]")
        fw_table.add_row("Install (GPU)", "[dim]pip install tensorflow[/dim]")

    console.print(fw_table)
    console.print()

    # ── Print: companion libraries ────────────────────────────────────
    comp_table = Table(title="Companion Libraries", box=box.SIMPLE_HEAVY)
    comp_table.add_column("Package", style="cyan", min_width=12)
    comp_table.add_column("Purpose", style="dim")
    comp_table.add_column("Status", min_width=8)
    comp_table.add_column("Version / Install", style="dim")

    companion_rows = [
        ("h5py",  "load .hdf5 / .h5 weight files", h5py_ok,  h5py_version  or "pip install h5py"),
        ("NumPy", "array interop",                  numpy_ok, numpy_version or "pip install numpy"),
    ]
    for display, purpose, ok, info in companion_rows:
        comp_table.add_row(
            display, purpose,
            "[green]OK[/green]" if ok else "[red]MISSING[/red]",
            info,
        )

    console.print(comp_table)
    console.print()

    # ── Print: dependent models ───────────────────────────────────────
    from speqtro.agent.config import Config
    config = Config.load()
    home_models = Path.home() / ".speqtro" / "models"

    model_table = Table(title="Models Using TensorFlow", box=box.SIMPLE_HEAVY)
    model_table.add_column("Model", style="cyan", min_width=14)
    model_table.add_column("Purpose", style="dim")
    model_table.add_column("Weights", min_width=10)
    model_table.add_column("Hint", style="dim")

    tf_models = [
        ("CASCADE 1.0", "1H NMR GNN (Paton)",
         home_models / "cascade1" / "best_model.hdf5",
         "speqtro fetch-weights --model cascade1"),
        ("CASCADE 2.0", "13C NMR GNN (Paton)",
         home_models / "cascade2" / "best_model.h5",
         "speqtro fetch-weights --model cascade2"),
    ]

    for name, purpose, weight_path, hint in tf_models:
        found = weight_path.exists()
        status = "[green]found[/green]" if found else "[yellow]missing[/yellow]"
        model_table.add_row(name, purpose, status, "" if found else hint)

    console.print(model_table)
    console.print()

    if not tf_ok:
        console.print("[yellow]TensorFlow install guide:[/yellow]  https://www.tensorflow.org/install")
    elif not gpu_devices:
        console.print("[dim]No GPU detected. CASCADE runs on CPU — prediction may be slower for large batches.[/dim]")
    console.print()


# ---------------------------------------------------------------------------
# speq mcp — standalone MCP server
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# speqtro web
# ---------------------------------------------------------------------------

@app.command()
def web(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address"),
    port: int = typer.Option(8080, "--port", "-p", help="Port"),
    no_browser: bool = typer.Option(False, "--no-browser", is_flag=True, help="Don't open browser"),
):
    """Launch the speqtro web interface in your browser."""
    try:
        import uvicorn
        from speqtro.web.server import app as web_app  # noqa: F401
    except ImportError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        console.print("Install web dependencies: [cyan]pip install 'speqtro[web]'[/cyan]")
        raise typer.Exit(1)

    url = f"http://{host}:{port}"
    console.print(f"[bold]speqtro web[/bold] → {url}")

    if not no_browser:
        import webbrowser, threading
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    uvicorn.run(web_app, host=host, port=port)


mcp_app = typer.Typer(help="Standalone MCP server for IDE integrations (Claude.ai, Cursor, VSCode).")
app.add_typer(mcp_app, name="mcp")


@mcp_app.command("serve")
def mcp_serve(
    stdio: bool = typer.Option(False, "--stdio", is_flag=True, help="stdio transport (Claude.ai / Cursor / Claude Code)"),
    http: bool = typer.Option(False, "--http", is_flag=True, help="HTTP/SSE transport (remote MCP)"),
    host: str = typer.Option("127.0.0.1", "--host", help="HTTP host"),
    port: int = typer.Option(8765, "--port", help="HTTP port"),
):
    """Start the speqtro MCP server (--stdio or --http)."""
    import asyncio
    from speqtro.agent.config import Config

    if not stdio and not http:
        console.print("[red]Error:[/red] Specify --stdio or --http")
        raise typer.Exit(1)

    config = Config.load()

    try:
        from speqtro.agent.mcp_server import serve_stdio, serve_http
    except ImportError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)

    if stdio:
        asyncio.run(serve_stdio(config))
    else:
        console.print(f"Starting speqtro MCP HTTP/SSE server on {host}:{port} ...")
        asyncio.run(serve_http(config, host=host, port=port))


@mcp_app.command("list-tools")
def mcp_list_tools():
    """Print all registered tools as MCP JSON schema."""
    from speqtro.agent.config import Config

    try:
        from speqtro.agent.mcp_server import list_tools_json
    except ImportError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)

    config = Config.load()
    print(list_tools_json(config))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def entry():
    """Main entry point for `speqtro` command."""
    app()


if __name__ == "__main__":
    entry()

"""NMRshiftDB2 Java predictor wrapper for ¹H and ¹³C NMR shift prediction.

NMRshiftDB2 (Stefan Kuhn) ships two CDK-based Java predictors:
  predictorh.jar — predict ¹H chemical shifts from a MOL file
  predictorc.jar — predict ¹³C chemical shifts from a MOL file

Both JARs require:
  - JDK 11+ on PATH
  - cdk-2.3.jar in the same directory

Output per atom: atom_id (1-based), min ppm, mean ppm, max ppm.

Configuration (checked in order):
  1. Config key  nmrshiftdb_predictor.jar_dir
  2. Env var     SPEQ_NMRSHIFTDB_JAR_DIR
  3. Auto-detect: <repo-root>/../SPEQTRO_TOOL/nmrshiftdb_predictors_app/app/lib
"""

import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from speqtro.tools import registry

logger = logging.getLogger("speqtro.tools.nmrshiftdb_predictor")

# Solvent display name → value expected by the Java predictor
SOLVENTS = {
    "CDCl3": "Chloroform-D1 (CDCl3)",
    "CD3OD": "Methanol-D4 (CD3OD)",
    "DMSO":  "Dimethylsulphoxide-D6 (DMSO-D6, C2D6SO)",
}

# Classpath separator is ; on Windows, : elsewhere
_CP_SEP = ";" if sys.platform == "win32" else ":"


# ── Config helpers ───────────────────────────────────────────────────────────

def _get_jar_dir() -> Optional[Path]:
    """Locate the directory containing predictorc.jar / predictorh.jar."""
    try:
        from speqtro.agent.config import Config
        cfg = Config.load()
        d = cfg.get("nmrshiftdb_predictor.jar_dir")
        if d:
            return Path(d)
    except Exception:
        pass

    env_dir = os.environ.get("SPEQ_NMRSHIFTDB_JAR_DIR")
    if env_dir:
        return Path(env_dir)

    # Auto-detect: look for the SPEQTRO_TOOL sibling repo
    repo_root = Path(__file__).resolve().parents[4]  # git_repo/
    candidate = (
        repo_root / "SPEQTRO_TOOL" / "nmrshiftdb_predictors_app" / "app" / "lib"
    )
    if candidate.exists() and (candidate / "predictorc.jar").exists():
        return candidate

    return None


def _check_java() -> bool:
    """Return True if java is on PATH."""
    try:
        subprocess.run(
            ["java", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# ── Core prediction logic ────────────────────────────────────────────────────

def _smiles_to_molfile(smiles: str, tmp_dir: str) -> Path:
    """Convert SMILES to a MOL file using RDKit, return the file path."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(mol)

    tmp_path = Path(tmp_dir) / "input.mol"
    Chem.MolToMolFile(mol, str(tmp_path))
    return tmp_path


def _run_jar(jar_name: str, mol_path: Path, jar_dir: Path,
             solvent: Optional[str]) -> list[dict]:
    """
    Call a predictor JAR and parse stdout.

    Returns list of dicts: [{atom_id, min, mean, max}, ...]
    """
    classpath = _CP_SEP.join([
        str(jar_dir / jar_name),
        str(jar_dir / "cdk-2.3.jar"),
    ])

    cmd_parts = ["java", "-cp", classpath, "Test", str(mol_path.absolute())]
    if solvent:
        cmd_parts.append(solvent)

    # shell=True + joined string mirrors the original nmrpredictor.py approach
    # to correctly pass solvent strings that contain spaces/parens
    if solvent:
        shell_cmd = (
            f'java -cp "{classpath}" Test "{mol_path.absolute()}" "{solvent}"'
        )
    else:
        shell_cmd = f'java -cp "{classpath}" Test "{mol_path.absolute()}"'

    result = subprocess.run(
        shell_cmd,
        cwd=str(jar_dir),
        capture_output=True,
        shell=True,
        timeout=60,
    )

    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"Predictor JAR failed (rc={result.returncode}): {stderr}")

    lines = result.stdout.decode(errors="replace").split("\n")
    data = []
    for line in lines[2:]:  # first two lines are headers
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            data.append({
                "atom_id": int(parts[0].replace(":", "")),
                "min":     round(float(parts[1]), 3),
                "mean":    round(float(parts[2]), 3),
                "max":     round(float(parts[3]), 3),
            })
        except (ValueError, IndexError):
            continue

    return data


def _predict(nucleus: str, smiles: str, solvent_key: Optional[str]) -> dict:
    """Shared prediction logic for ¹H and ¹³C."""
    smiles = smiles.strip()
    if not smiles:
        return {"error": "No SMILES provided."}

    if not _check_java():
        return {
            "error": (
                "Java not found on PATH. Install JDK 11+ and ensure 'java' is "
                "accessible, then retry."
            )
        }

    jar_dir = _get_jar_dir()
    if jar_dir is None:
        return {
            "error": (
                "NMRshiftDB predictor JARs not found. "
                "Set config key 'nmrshiftdb_predictor.jar_dir' or env var "
                "SPEQ_NMRSHIFTDB_JAR_DIR to the directory containing "
                "predictorc.jar / predictorh.jar / cdk-2.3.jar."
            )
        }

    jar_name = "predictorh.jar" if nucleus == "H" else "predictorc.jar"
    if not (jar_dir / jar_name).exists():
        return {"error": f"{jar_name} not found in {jar_dir}"}

    solvent_value: Optional[str] = None
    if solvent_key:
        solvent_value = SOLVENTS.get(solvent_key.upper(), solvent_key)

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            mol_path = _smiles_to_molfile(smiles, tmp_dir)
            predictions = _run_jar(jar_name, mol_path, jar_dir, solvent_value)
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        logger.exception("NMRshiftDB predictor failed")
        return {"error": f"Prediction failed: {e}"}

    if not predictions:
        return {"error": "Predictor returned no predictions. Check SMILES validity."}

    nucleus_label = "¹H" if nucleus == "H" else "¹³C"
    solvent_display = solvent_value or "Unreported"
    summary_lines = [f"NMRshiftDB2 {nucleus_label} predictions ({solvent_display}):"]
    for p in predictions:
        summary_lines.append(
            f"  atom {p['atom_id']:3d}: {p['mean']:.2f} ppm "
            f"(range {p['min']:.2f}–{p['max']:.2f})"
        )

    return {
        "summary": "\n".join(summary_lines),
        "smiles": smiles,
        "nucleus": nucleus_label,
        "solvent": solvent_display,
        "predictions": predictions,
        "model": "NMRshiftDB2 statistical predictor (CDK 2.3)",
        "units": "ppm",
        "note": (
            "Each prediction gives min/mean/max ppm based on the NMRshiftDB2 "
            "statistical model. mean is the best estimate."
        ),
    }


# ── Tool registration ────────────────────────────────────────────────────────

@registry.register(
    name="nmr.predict_h1_nmrshiftdb",
    description=(
        "Predict ¹H NMR chemical shifts using the NMRshiftDB2 statistical predictor "
        "(CDK 2.3, Java-based). Input: SMILES string. "
        "Output: per-atom shift predictions with min/mean/max ppm. "
        "Requires JDK 11+ and the NMRshiftDB predictor JARs."
    ),
    category="nmr",
    parameters={
        "type": "object",
        "properties": {
            "smiles": {
                "type": "string",
                "description": "SMILES string of the molecule",
            },
            "solvent": {
                "type": "string",
                "description": (
                    "NMR solvent. Accepted values: CDCl3, CD3OD, DMSO. "
                    "Omit for solvent-independent prediction."
                ),
            },
        },
        "required": ["smiles"],
    },
)
def predict_h1_nmrshiftdb(smiles: str, solvent: str = None) -> dict:
    """Predict ¹H shifts using the NMRshiftDB2 statistical predictor."""
    return _predict("H", smiles, solvent)


@registry.register(
    name="nmr.predict_c13_nmrshiftdb",
    description=(
        "Predict ¹³C NMR chemical shifts using the NMRshiftDB2 statistical predictor "
        "(CDK 2.3, Java-based). Input: SMILES string. "
        "Output: per-atom shift predictions with min/mean/max ppm. "
        "Requires JDK 11+ and the NMRshiftDB predictor JARs."
    ),
    category="nmr",
    parameters={
        "type": "object",
        "properties": {
            "smiles": {
                "type": "string",
                "description": "SMILES string of the molecule",
            },
            "solvent": {
                "type": "string",
                "description": (
                    "NMR solvent. Accepted values: CDCl3, CD3OD, DMSO. "
                    "Omit for solvent-independent prediction."
                ),
            },
        },
        "required": ["smiles"],
    },
)
def predict_c13_nmrshiftdb(smiles: str, solvent: str = None) -> dict:
    """Predict ¹³C shifts using the NMRshiftDB2 statistical predictor."""
    return _predict("C", smiles, solvent)

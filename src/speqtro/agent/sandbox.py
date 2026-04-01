"""Sandboxed Python execution for speqtro code generation."""

import io
import os
import signal
import sys
import traceback
from pathlib import Path
from typing import Any, Optional

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import matplotlib
    matplotlib.use("Agg")
except ImportError:
    pass

_BLOCKED_MODULES = frozenset({
    "subprocess", "shutil", "socket", "http.server", "smtplib", "ctypes",
})

_SAFE_OS_ATTRS = frozenset({
    "path", "listdir", "walk", "getcwd", "sep", "linesep", "stat", "scandir",
})


def _make_safe_import(real_import):
    def _safe_import(name, *args, **kwargs):
        base = name.split(".")[0]
        if name in _BLOCKED_MODULES or base in _BLOCKED_MODULES:
            raise ImportError(f"Import of '{name}' is blocked in the speqtro sandbox.")
        return real_import(name, *args, **kwargs)
    return _safe_import


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _make_safe_open(output_dir: Path, extra_read_dirs: list[Path] = None):
    real_open = open
    output_root = output_dir.resolve()
    cwd_root = Path.cwd().resolve()
    extra_roots = [d.resolve() for d in (extra_read_dirs or [])]

    def _safe_open(file, mode="r", *args, **kwargs):
        if isinstance(file, int):
            return real_open(file, mode, *args, **kwargs)
        path = Path(file).expanduser()
        resolved = path.resolve() if path.is_absolute() else (cwd_root / path).resolve()
        downloads_root = (Path.home() / ".speqtro" / "downloads").resolve()
        tmp_root = Path(os.environ.get("TEMP", "/tmp")).resolve()
        can_read = (
            _is_within(resolved, cwd_root)
            or _is_within(resolved, output_root)
            or _is_within(resolved, downloads_root)
            or _is_within(resolved, tmp_root)
            or any(_is_within(resolved, d) for d in extra_roots)
        )
        if not can_read:
            raise PermissionError(
                f"Sandbox reads restricted to {cwd_root} and {output_root}"
            )
        writes = any(flag in mode for flag in ("w", "a", "x", "+"))
        if writes and not (_is_within(resolved, output_root) or _is_within(resolved, tmp_root)):
            raise PermissionError(f"Sandbox writes restricted to {output_root}")
        if writes:
            resolved.parent.mkdir(parents=True, exist_ok=True)
        return real_open(resolved, mode, *args, **kwargs)

    return _safe_open


class Sandbox:
    """Sandboxed execution environment for generated Python code."""

    def __init__(self, timeout: int = 60, output_dir: Path = None, max_retries: int = 2,
                 extra_read_dirs: list[Path] = None):
        self.timeout = timeout
        self.output_dir = Path(output_dir) if output_dir else Path.cwd() / "outputs"
        self.max_retries = max_retries
        self.extra_read_dirs = [Path(d).resolve() for d in (extra_read_dirs or [])]
        self._namespace: dict[str, Any] = {}
        self._setup_namespace()

    def _setup_namespace(self):
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import json, re, math, collections, itertools, datetime, io, tempfile, csv, os

        if isinstance(__builtins__, dict):
            safe_builtins = dict(__builtins__)
        else:
            safe_builtins = {k: getattr(__builtins__, k) for k in dir(__builtins__)}
        safe_builtins["__import__"] = _make_safe_import(__import__)
        safe_builtins["open"] = _make_safe_open(self.output_dir, self.extra_read_dirs)

        self._namespace = {
            "pd": pd, "np": np, "plt": plt,
            "json": json, "re": re, "math": math,
            "collections": collections, "itertools": itertools,
            "datetime": datetime, "io": io, "tempfile": tempfile, "csv": csv,
            "os": os, "Path": Path,
            "OUTPUT_DIR": self.output_dir,
            "__builtins__": safe_builtins,
        }

        try:
            import scipy.stats as scipy_stats
            self._namespace["scipy_stats"] = scipy_stats
            self._namespace["scipy"] = __import__("scipy")
        except ImportError:
            pass

        try:
            import rdkit
            from rdkit import Chem
            from rdkit.Chem import Descriptors, AllChem
            self._namespace["Chem"] = Chem
            self._namespace["Descriptors"] = Descriptors
            self._namespace["AllChem"] = AllChem
            self._namespace["rdkit"] = rdkit
        except ImportError:
            pass

    def load_datasets(self) -> dict:
        return {}

    def get_variable(self, name: str, default=None):
        return self._namespace.get(name, default)

    def execute(self, code: str) -> dict:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        existing_files = set(self.output_dir.iterdir()) if self.output_dir.exists() else set()

        old_stdout, old_stderr = sys.stdout, sys.stderr
        captured_out = io.StringIO()
        captured_err = io.StringIO()

        result = {"stdout": "", "stderr": "", "result": None, "error": None, "plots": [], "exports": []}

        import threading
        has_alarm = hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread()
        if has_alarm:
            def _timeout_handler(signum, frame):
                raise TimeoutError(f"Execution timed out after {self.timeout}s")
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(self.timeout)

        try:
            sys.stdout = captured_out
            sys.stderr = captured_err
            compiled = compile(code, "<speqtro-sandbox>", "exec")
            exec(compiled, self._namespace)
            if "result" in self._namespace:
                result["result"] = self._namespace["result"]
        except TimeoutError as e:
            result["error"] = str(e)
        except Exception:
            result["error"] = traceback.format_exc()
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            if has_alarm:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

        result["stdout"] = captured_out.getvalue()
        result["stderr"] = captured_err.getvalue()

        if self.output_dir.exists():
            new_files = set(self.output_dir.iterdir()) - existing_files
            for f in sorted(new_files):
                if f.suffix in (".png", ".svg", ".jpg", ".pdf"):
                    result["plots"].append(str(f))
                elif f.suffix in (".csv", ".xlsx", ".json"):
                    result["exports"].append(str(f))

        return result

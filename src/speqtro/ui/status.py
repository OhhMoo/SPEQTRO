"""
Thinking status display for speqtro.

Shows a spinner with rotating spectroscopy-themed words and an elapsed
time counter. The spinner uses a neon-green → cyan gradient.

Usage:
    with ThinkingStatus(console, "planning"):
        result = agent.run(...)
"""

import math
import random
import time
from typing import List

from rich.live import Live
from rich.text import Text

# ---------------------------------------------------------------------------
# Spinner animations
# ---------------------------------------------------------------------------

_BASE = "\u2881\u2822\u2814\u2848\u2814\u2822"  # ⢁⠢⠔⡈⠔⠢
_DNA_FRAMES: List[str] = [_BASE[i:] + _BASE[:i] for i in range(len(_BASE))]

SPINNERS = {
    "benzene_breathing": {
        "frames": ["⬡", "⎔", "⌬", "⬢", "⌬", "⎔"],
        "interval_ms": 125,
    },
    "dna_helix": {
        "frames": _DNA_FRAMES,
        "interval_ms": 125,
    },
    "nmr_pulse": {
        "frames": ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█", "▇", "▆", "▅", "▄", "▃", "▂"],
        "interval_ms": 100,
    },
}

THINKING_WORDS = {
    "planning": [
        "Analyzing spectra",
        "Scanning NMR peaks",
        "Mapping chemical shifts",
        "Evaluating degree of unsaturation",
        "Querying PubChem",
        "Identifying functional groups",
        "Interpreting multiplicity",
        "Reviewing MS fragmentation",
        "Calculating exact mass",
        "Consulting reference spectra",
        "Correlating COSY data",
        "Matching IR absorptions",
        "Scanning literature",
        "Assigning carbon skeleton",
        "Probing heteroatoms",
        "Evaluating coupling constants",
        "Cross-referencing SDBS",
        "Parsing molecular formula",
        "Checking ring current effects",
        "Surveying chemical space",
        "Triaging candidates",
        "Calibrating approach",
        "Weighing structures",
    ],
    "synthesizing": [
        "Elucidating structure",
        "Assembling the puzzle",
        "Connecting spectral evidence",
        "Building molecular picture",
        "Integrating NMR data",
        "Assigning the structure",
        "Mapping atom connectivity",
        "Reconciling chemical shifts",
        "Formulating the assignment",
        "Distilling spectral evidence",
        "Completing the analysis",
        "Crystallizing the structure",
        "Narrating the evidence",
        "Connecting the fragments",
        "Framing the interpretation",
    ],
    "evaluating": [
        "Checking shift assignments",
        "Verifying multiplicity",
        "Validating structure",
        "Auditing peak coverage",
        "Stress-testing the assignment",
        "Cross-checking spectra",
        "Scoring confidence",
        "Reviewing consistency",
        "Checking for blind spots",
        "Verifying DP4 scores",
    ],
    "reasoning": [
        "Reasoning through connectivity",
        "Tracing carbon skeleton",
        "Evaluating tautomers",
        "Considering stereochemistry",
        "Weighing structural alternatives",
        "Modeling ring currents",
        "Analyzing electronic effects",
        "Exploring conformational effects",
        "Pondering symmetry",
        "Probing through-bond coupling",
        "Examining isotope patterns",
        "Assessing diastereotopic protons",
    ],
    "coding": [
        "Writing code",
        "Running RDKit",
        "Computing properties",
        "Parsing spectrum",
        "Fitting peaks",
        "Plotting spectrum",
        "Searching database",
        "Calling PubChem",
        "Running sandbox",
    ],
    "doctor": [
        "Probing environment",
        "Importing dependencies",
        "Checking package versions",
        "Scanning installed modules",
        "Auditing Python environment",
        "Enumerating packages",
    ],
    "doctor_tools": [
        "Loading tool registry",
        "Registering spectroscopy tools",
        "Wiring tool modules",
        "Instantiating tool backends",
        "Indexing ML wrappers",
    ],
    "doctor_models": [
        "Locating model weights",
        "Scanning checkpoint paths",
        "Checking HuggingFace cache",
        "Verifying model files",
        "Resolving weight locations",
    ],
    "doctor_api": [
        "Probing external services",
        "Pinging Anthropic API",
        "Testing network connectivity",
        "Measuring API latency",
        "Checking PubChem endpoint",
        "Validating API credentials",
    ],
}


# ---------------------------------------------------------------------------
# Rotating benzene ring animation
# ---------------------------------------------------------------------------
# 3-row compact hexagon.  One atom (●) orbits clockwise every 1.5 s.
#
#  ○─○     row 0:  atoms at col 1 (top-left) and col 3 (top-right)
# ○   ○    row 1:  atoms at col 0 (left) and col 4 (right)
#  ○─○     row 2:  atoms at col 1 (bottom-left) and col 3 (bottom-right)

_RING_ROWS = [
    " ○─○",
    "○   ○",
    " ○─○",
]

# Clockwise atom positions: (row, col)  starting from top-right
_ATOM_POS = [
    (0, 3),   # 0 — top-right
    (1, 4),   # 1 — right
    (2, 3),   # 2 — bottom-right
    (2, 1),   # 3 — bottom-left
    (1, 0),   # 4 — left
    (0, 1),   # 5 — top-left
]


def _gradient_color(elapsed_s: float) -> str:
    """Return a hex color string that pulses deep teal → bright cyan."""
    r1, g1, b1 = 13, 107, 140   # #0D6B8C deep teal
    r2, g2, b2 = 0, 188, 212    # #00BCD4 bright cyan
    cycle = 2.25
    t = (math.sin((elapsed_s % cycle) * (2 * math.pi / cycle)) + 1) / 2
    r = int(r1 + (r2 - r1) * t)
    g = int(g1 + (g2 - g1) * t)
    b = int(b1 + (b2 - b1) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def _render_ring(active_idx: int, atom_color: str) -> Text:
    """Build a Rich Text object for one benzene ring frame."""
    active = _ATOM_POS[active_idx % 6]
    ring = Text()
    for row_i, row_str in enumerate(_RING_ROWS):
        for col_i, ch in enumerate(row_str):
            if ch == "○":
                if (row_i, col_i) == active:
                    ring.append("●", style=f"bold {atom_color}")
                else:
                    ring.append("○", style="#1a3040")
            elif ch in ("─", "╱", "╲"):
                ring.append(ch, style="#0a1820")
            else:
                ring.append(ch)
        if row_i < len(_RING_ROWS) - 1:
            ring.append("\n")
    return ring


# Phases that belong to the interactive agent page — use the benzene ring.
_RING_PHASES = {"planning", "synthesizing", "evaluating", "reasoning", "coding"}

_SPIN_FRAMES = ["⬡", "⎔", "⌬", "⬢", "⌬", "⎔"]


class _ThinkingRenderable:
    """Self-updating renderable: rotating benzene ring + spectroscopy word + timer.
    Used only for the interactive agent page (planning / evaluating / … phases).
    """

    def __init__(self, words: List[str]):
        self.words = words
        self.start_time = time.time()

    def __rich_console__(self, console, options):
        from rich.table import Table

        elapsed = time.time() - self.start_time

        # One full rotation every 1.5 s  (6 frames × 0.25 s each)
        frame_idx = int(elapsed / 0.25) % 6
        color = _gradient_color(elapsed)
        ring = _render_ring(frame_idx, color)

        word = self.words[int(elapsed / 3) % len(self.words)]
        if elapsed < 60:
            time_str = f"{elapsed:.0f}s"
        else:
            time_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"

        status = Text()
        status.append(f"{word}…", style="#00BCD4")
        status.append("\n\n")
        status.append(f"({time_str})", style="dim")

        t = Table.grid(padding=(0, 2))
        t.add_column(vertical="middle")
        t.add_column(vertical="middle")
        t.add_row(ring, status)

        yield t


class _ThinkingRenderableLine:
    """Self-updating renderable: compact single-line spinner + word + timer.
    Used for doctor / tf / setup pages.
    """

    def __init__(self, words: List[str]):
        self.words = words
        self.start_time = time.time()

    def __rich_console__(self, console, options):
        elapsed = time.time() - self.start_time

        frame = _SPIN_FRAMES[int(elapsed / 0.25) % len(_SPIN_FRAMES)]
        color = _gradient_color(elapsed)
        word = self.words[int(elapsed / 3) % len(self.words)]
        if elapsed < 60:
            time_str = f"{elapsed:.0f}s"
        else:
            time_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"

        line = Text()
        line.append(frame, style=f"bold {color}")
        line.append("  ")
        line.append(f"{word}…", style="#00BCD4")
        line.append("  ")
        line.append(f"({time_str})", style="dim")

        yield line


class ThinkingStatus:
    """Context manager showing a thinking status with animated spinner.

    Args:
        console: Rich Console instance.
        phase: One of the keys in THINKING_WORDS.
    """

    def __init__(self, console, phase: str = "planning"):
        self.console = console
        words = list(THINKING_WORDS.get(phase, THINKING_WORDS["planning"]))
        random.shuffle(words)
        if phase in _RING_PHASES:
            self._renderable = _ThinkingRenderable(words)
        else:
            self._renderable = _ThinkingRenderableLine(words)
        self._live = None
        self._async_task = None

    def __enter__(self):
        self._live = Live(
            self._renderable,
            console=self.console,
            refresh_per_second=8,
            transient=True,
        )
        self._live.__enter__()
        return self

    def __exit__(self, *args):
        self._cancel_async_task()
        if self._live is not None:
            return self._live.__exit__(*args)

    def start_async_refresh(self):
        """Start a background asyncio task that keeps the timer ticking."""
        import asyncio

        async def _refresh_loop():
            try:
                while True:
                    await asyncio.sleep(0.125)
                    if self._live:
                        try:
                            self._live.refresh()
                        except Exception:
                            pass
            except asyncio.CancelledError:
                pass

        try:
            loop = asyncio.get_running_loop()
            self._async_task = loop.create_task(_refresh_loop())
        except RuntimeError:
            pass

    def _cancel_async_task(self):
        if self._async_task is not None:
            self._async_task.cancel()
            self._async_task = None

    def stop(self):
        """Programmatically stop the animation (idempotent)."""
        self._cancel_async_task()
        if self._live is not None:
            try:
                self._live.__exit__(None, None, None)
            except Exception:
                pass
            self._live = None

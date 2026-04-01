"""
Interactive REPL terminal for speqtro.

Provides a rich REPL-style interface for continuous spectroscopy sessions,
with slash-command autocomplete, ghost suggestions, a bottom toolbar, file
history, and a pixel-art NMR spectrum splash screen.
"""

import random
import sys
import time
from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# ---------------------------------------------------------------------------
# Pixel-art SPEQ logo
# ---------------------------------------------------------------------------
# Each letter is a 5-wide × 7-tall bitmap; pixels are doubled (each "1" → "██")
# so the rendered logo is ~10 chars per letter + spacing.
# Main fill uses a vertical lime→cyan gradient; a single dim cyan rule sits under the word.

_PIXEL_FONT: dict[str, list[str]] = {
    "S": [
        " ███ ",
        "█   █",
        "█    ",
        " ███ ",
        "    █",
        "█   █",
        " ███ ",
    ],
    "P": [
        "████ ",
        "█   █",
        "█   █",
        "████ ",
        "█    ",
        "█    ",
        "█    ",
    ],
    "E": [
        "█████",
        "█    ",
        "█    ",
        "████ ",
        "█    ",
        "█    ",
        "█████",
    ],
    "Q": [
        " ███ ",
        "█   █",
        "█   █",
        "█   █",
        "█ ██ ",
        "█  ██",
        " ████",
    ],
    "T": [
        "█████",
        "  █  ",
        "  █  ",
        "  █  ",
        "  █  ",
        "  █  ",
        "  █  ",
    ],
    "R": [
        "████ ",
        "█   █",
        "█   █",
        "████ ",
        "█ █  ",
        "█  █ ",
        "█   █",
    ],
    "O": [
        " ███ ",
        "█   █",
        "█   █",
        "█   █",
        "█   █",
        "█   █",
        " ███ ",
    ],
}

def _logo_row_color(row_i: int, height: int) -> str:
    """Lime (top) → cyan (bottom) vertical gradient for retro terminal header."""
    if height <= 1:
        return "#c8ff3a"
    t = row_i / (height - 1)
    # #c8ff3a (lime) → #00e5ff (cyan)
    r = int(0xC8 + (0x00 - 0xC8) * t)
    g = int(0xFF + (0xE5 - 0xFF) * t)
    b = int(0x3A + (0xFF - 0x3A) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def _render_logo(word: str = "SPEQTRO") -> list[Text]:
    """Return Rich Text rows: doubled pixel-art SPEQTRO with vertical gradient."""
    glyphs = [(ch, _PIXEL_FONT[ch]) for ch in word if ch in _PIXEL_FONT]
    if not glyphs:
        return []
    height = max(len(g) for _, g in glyphs)
    rows: list[Text] = []
    for row_i in range(height):
        color = _logo_row_color(row_i, height)
        line = Text()
        for letter_i, (ch, glyph) in enumerate(glyphs):
            bitmap_row = glyph[row_i] if row_i < len(glyph) else " " * len(glyph[0])
            for cell in bitmap_row:
                line.append("██" if cell == "█" else "  ", style=color if cell == "█" else "")
            if letter_i < len(glyphs) - 1:
                line.append("  ")  # letter gap
        rows.append(line)
    return rows


def _logo_accent_line(logo_width: int) -> Text:
    """One thin neon rule under the logo (replaces a heavy duplicate shadow)."""
    if logo_width < 4:
        return Text()
    # Bottom arc + rule — HUD-style, one line tall; width matches the logo row.
    dash = max(1, logo_width - 2)
    line = Text()
    line.append("╰", style="dim #00a8c4")
    line.append("─" * dash, style="dim #00c8e8")
    line.append("╯", style="dim #00a8c4")
    return line


# ---------------------------------------------------------------------------
# Slash commands + autocomplete
# ---------------------------------------------------------------------------

SLASH_COMMANDS = {
    "/help":    "Show command reference and examples",
    "/tools":   "List all available speqtro tools",
    "/model":   "Switch the LLM model",
    "/config":  "Show active configuration",
    "/clear":   "Clear the screen",
    "/verbose": "Toggle verbose tool output",
    "/verify":  "Verify product from spectral data (guided input)",
    "/exit":    "Exit speqtro",
}

# ---------------------------------------------------------------------------
# Ghost placeholder suggestions (spectroscopy-themed)
# ---------------------------------------------------------------------------

DEFAULT_SUGGESTIONS = [
    "identify the compound from 1H NMR: 7.3 (5H, m), 3.7 (2H, s), 2.1 (3H, s)",
    "predict NMR shifts for aspirin (CC(=O)Oc1ccccc1C(=O)O)",
    "calculate exact mass for C6H12O6 and list MS adducts",
    "what compound has MW 180 and 1H NMR singlets at 3.8 and 5.9 ppm?",
    "search PubChem for vanillin and predict its IR spectrum",
    "interpret 13C NMR: 200 ppm, 130 ppm (×5), 45 ppm",
    "fragment the SMILES CC(=O)Nc1ccc(O)cc1 and predict MS/MS losses",
    "find molecular formula for exact mass 194.0579 (M+H, 5 ppm)",
    "compare predicted vs observed NMR for ibuprofen",
    "parse JCAMP file at /data/sample.jdx and identify peaks",
]

# ---------------------------------------------------------------------------
# Prompt-toolkit style (dark terminal theme)
# ---------------------------------------------------------------------------

try:
    from prompt_toolkit.styles import Style as PtStyle
    PT_STYLE = PtStyle.from_dict({
        "prompt":                          "#50fa7b bold",
        "completion-menu.completion":      "bg:#1e1e2e #cccccc",
        "completion-menu.completion.current": "bg:#00e5ff #1e1e2e bold",
        "completion-menu.meta.completion": "bg:#1e1e2e #666666",
        "scrollbar.background":            "bg:#1e1e2e",
        "scrollbar.button":                "bg:#444466",
        "bottom-toolbar":                  "bg:#1e1e2e #555577",
    })
except ImportError:
    PT_STYLE = None

# ---------------------------------------------------------------------------
# Slash command completer
# ---------------------------------------------------------------------------

try:
    from prompt_toolkit.completion import Completer, Completion

    class SlashCompleter(Completer):
        """Autocomplete slash commands when input starts with /."""

        def get_completions(self, document, complete_event):
            text = document.text_before_cursor.lstrip()
            if not text.startswith("/"):
                return
            for cmd, desc in SLASH_COMMANDS.items():
                if cmd.startswith(text):
                    display_meta = desc[:50]
                    yield Completion(
                        cmd,
                        start_position=-len(text),
                        display_meta=display_meta,
                    )

except ImportError:
    SlashCompleter = None

# ---------------------------------------------------------------------------
# InteractiveTerminal
# ---------------------------------------------------------------------------

class InteractiveTerminal:
    """REPL-style interface for speqtro spectroscopy sessions."""

    def __init__(self, session):
        self.session = session
        self.console = session.console
        self._last_interrupt = 0.0
        self._show_exit_hint = False
        self._suggestions = list(DEFAULT_SUGGESTIONS)
        random.shuffle(self._suggestions)
        self._suggestion_idx = 0
        self._verbose = session.verbose

        # Prompt history file path (session built lazily in run())
        self._history_file = Path.home() / ".speqtro" / "history"
        self._history_file.parent.mkdir(parents=True, exist_ok=True)
        self._prompt_session = None  # built on first run()

    # ------------------------------------------------------------------
    # Bottom toolbar
    # ------------------------------------------------------------------

    def _bottom_toolbar(self):
        try:
            from prompt_toolkit.formatted_text import HTML
        except ImportError:
            return ""

        if self._show_exit_hint:
            return HTML('<style fg="#888888">  Press Ctrl+C again to exit</style>')

        model = self.session.config.get("llm.model", "claude-opus-4-6")
        short_names = {
            "claude-opus-4-6":           "Opus 4.6",
            "claude-sonnet-4-6":         "Sonnet 4.6",
            "claude-haiku-4-5-20251001": "Haiku 4.5",
        }
        model_label = short_names.get(model, model)
        verbose_badge = (
            '<style fg="#1a1a2e" bg="#50fa7b"> verbose </style>'
            if self._verbose else ""
        )
        return HTML(
            f'  <style fg="#ffffff" bg="#005577"> {model_label} </style>'
            f'{verbose_badge}'
            f'<style fg="#444466">  /help for commands  ·  Ctrl+C to interrupt  ·  Ctrl+C × 2 to exit</style>'
        )

    # ------------------------------------------------------------------
    # Ghost placeholder suggestions
    # ------------------------------------------------------------------

    def _current_placeholder(self):
        try:
            from prompt_toolkit.formatted_text import HTML
        except ImportError:
            return ""
        text = self._suggestions[self._suggestion_idx % len(self._suggestions)]
        # Truncate if too long for comfort
        if len(text) > 72:
            text = text[:69] + "…"
        return HTML(f'<style fg="#3a3a5c">{text}</style>')

    def _advance_suggestion(self):
        self._suggestion_idx = (self._suggestion_idx + 1) % len(self._suggestions)

    # ------------------------------------------------------------------
    # Splash screen
    # ------------------------------------------------------------------

    def _show_splash(self, version: str):
        """Print the pixel-art SPEQ logo + welcome panel (retro terminal style)."""
        from speqtro.tools import ensure_loaded, registry

        ensure_loaded()
        n_tools = len(registry.list_tools())

        logo_rows = _render_logo("SPEQTRO")

        header = Text()
        for row in logo_rows:
            header.append_text(row)
            header.append("\n")
        if logo_rows:
            header.append_text(_logo_accent_line(len(logo_rows[0].plain)))
            header.append("\n\n")
        else:
            header.append("\n")
        header.append(
            "Autonomous spectroscopy reasoning agent for chemists",
            style="bold white",
        )
        header.append("\n")
        header.append(f"v{version}", style="dim #6b6b7b")
        header.append("  ·  ", style="dim #4a4a5c")
        header.append(f"{n_tools} tools loaded", style="dim #6b6b7b")
        header.append("\n\n")
        header.append("Type a spectroscopy question, or ", style="dim #5c5c72")
        header.append("/help", style="bold #00e5ff")
        header.append(" for commands.", style="dim #5c5c72")

        self.console.print()
        self.console.print(
            Panel(
                header,
                title=Text("speqtro", style="bold #00e5ff"),
                title_align="center",
                border_style="dim #555555",
                box=box.ROUNDED,
                padding=(1, 3),
            )
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _build_prompt_session(self):
        """Build the prompt_toolkit PromptSession (deferred until run() so that
        importing the module does not require a real terminal)."""
        try:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.history import FileHistory
            from prompt_toolkit.key_binding import KeyBindings

            kb = KeyBindings()

            @kb.add("c-c")
            def _handle_ctrl_c(event):
                now = time.time()
                if now - self._last_interrupt < 1.5:
                    event.app.exit(result="__EXIT__")
                else:
                    self._last_interrupt = now
                    self._show_exit_hint = True
                    event.app.current_buffer.text = ""
                    event.app.invalidate()

            kwargs = dict(
                history=FileHistory(str(self._history_file)),
                complete_while_typing=True,
                multiline=False,
                key_bindings=kb,
            )
            if SlashCompleter is not None:
                kwargs["completer"] = SlashCompleter()
            if PT_STYLE is not None:
                kwargs["style"] = PT_STYLE

            self._prompt_session = PromptSession(**kwargs)
        except Exception:
            self._prompt_session = None

    def run(self):
        from speqtro.agent.loop import AgentLoop
        from speqtro import __version__

        self._show_splash(__version__)
        self._build_prompt_session()

        loop = AgentLoop(self.session)
        term_width = self.console.width

        while True:
            # Separator line above prompt
            self.console.print(f"[dim #3a3a4e]{'─' * term_width}[/]")

            try:
                if self._prompt_session is not None:
                    from prompt_toolkit.formatted_text import HTML
                    query = self._prompt_session.prompt(
                        [("class:prompt", "> ")],
                        bottom_toolbar=self._bottom_toolbar,
                        placeholder=self._current_placeholder(),
                    ).strip()
                    self._show_exit_hint = False
                else:
                    query = input("> ").strip()

            except EOFError:
                self.console.print("\n[dim]Goodbye.[/dim]")
                break
            except KeyboardInterrupt:
                self.console.print("\n[dim]Goodbye.[/dim]")
                break

            if query == "__EXIT__":
                self.console.print("[dim]Goodbye.[/dim]")
                break

            if not query:
                self._advance_suggestion()
                continue

            # Slash commands
            if query.startswith("/") or query in ("help", "exit", "quit"):
                result = self._handle_slash(query.lower().strip(), loop)
                if result == "exit":
                    break
                self._advance_suggestion()
                continue

            # Agent query
            try:
                loop.run(query)
            except KeyboardInterrupt:
                self.console.print("\n[dim]Interrupted.[/dim]")

            self._advance_suggestion()

    # ------------------------------------------------------------------
    # Slash command handler
    # ------------------------------------------------------------------

    def _handle_slash(self, cmd: str, loop) -> str | None:
        # Auto-resolve partial commands (e.g. "/mo" → "/model")
        if cmd.startswith("/") and cmd not in SLASH_COMMANDS:
            prefix = cmd.split()[0]
            matches = [c for c in SLASH_COMMANDS if c.startswith(prefix)]
            if matches:
                cmd = matches[0]

        if cmd in ("/exit", "/quit", "exit", "quit"):
            self.console.print("[dim]Goodbye.[/dim]")
            return "exit"

        if cmd in ("/clear",):
            self.console.clear()

        elif cmd in ("/help", "help", "?"):
            self._show_help()

        elif cmd in ("/tools",):
            from speqtro.tools import registry, ensure_loaded, tool_load_errors
            ensure_loaded()
            self.console.print(registry.list_tools_table())
            errs = tool_load_errors()
            if errs:
                self.console.print(f"[yellow]Warning: {len(errs)} tool(s) failed to load.[/yellow]")

        elif cmd in ("/config",):
            self.console.print(self.session.config.to_table())

        elif cmd in ("/model",):
            self._switch_model()

        elif cmd in ("/verbose",):
            self._verbose = not self._verbose
            self.session.verbose = self._verbose
            state = "[green]on[/green]" if self._verbose else "[dim]off[/dim]"
            self.console.print(f"  Verbose mode: {state}")

        elif cmd in ("/verify",):
            inp = self._guided_verify_input()
            if inp is not None:
                full_query, spectral_input = inp
                try:
                    loop.run(full_query, spectral_input=spectral_input)
                except KeyboardInterrupt:
                    self.console.print("\n[dim]Interrupted.[/dim]")

        else:
            self.console.print(
                f"[yellow]Unknown command:[/yellow] {cmd}  "
                "[dim](type /help for commands)[/dim]"
            )

        return None

    def _show_help(self):
        from speqtro.ui.markdown import LeftMarkdown
        help_md = (
            "**Usage:**\n"
            "Type any spectroscopy question and speqtro will reason through it.\n\n"
            "**Commands:**\n"
            + "\n".join(f"- `{cmd}` — {desc}" for cmd, desc in SLASH_COMMANDS.items())
            + "\n\n"
            "**Shortcuts:**\n"
            "- `Ctrl+C` — interrupt running query\n"
            "- `Ctrl+C × 2` — exit\n"
            "- `↑ / ↓` — browse history\n"
            "- `Tab` — autocomplete slash commands\n"
            "\n"
            "**Example queries:**\n"
            '- `identify compound from 1H NMR peaks at 7.26 and 2.35 ppm`\n'
            '- `predict NMR shifts for SMILES CC(=O)O`\n'
            '- `calculate exact mass for C10H14N2 and show adducts`\n'
            '- `search PubChem for caffeine`\n'
            '- `what IR absorptions does acetone show?`\n'
        )
        self.console.print(Panel(
            LeftMarkdown(help_md),
            title="[bold cyan]speqtro Help[/bold cyan]",
            border_style="cyan",
        ))

    def _guided_verify_input(self):
        """
        Interactive guided prompt for /verify.
        Returns (full_query, SpectralInput) or None if user cancels.
        """
        from speqtro.input.spectral_input import SpectralInput

        self.console.print("\n[bold cyan]speqtro Verify[/bold cyan] — enter spectral data\n")
        self.console.print(
            "[dim]Each field accepts a file path or inline text. "
            "Press Enter to skip.[/dim]\n"
        )

        def _prompt(label: str, hint: str = "") -> str:
            full_label = f"  [cyan]{label}[/cyan]"
            if hint:
                full_label += f" [dim]({hint})[/dim]"
            self.console.print(full_label)
            try:
                return input("  ❯ ").strip()
            except (EOFError, KeyboardInterrupt):
                return ""

        try:
            smiles   = _prompt("Product SMILES", "e.g. CC(=O)Oc1ccccc1")
            sm_smiles = _prompt("Starting material SMILES", "optional")
            h1       = _prompt("¹H NMR", "file path or '7.26 (5H, m), 3.71 (2H, s)'")
            c13      = _prompt("¹³C NMR", "file path or '128.5, 77.2, 45.3'")
            ms       = _prompt("MS data", "e.g. 'm/z 181 [M+H]+'")
            ir       = _prompt("IR peaks (cm⁻¹)", "file path or '1715, 1600'")
            solvent  = _prompt("Solvent", "CDCl3 / DMSO-d6 / D2O / ...")
            reaction = _prompt("Reaction type", "esterification / amide coupling / ...")
            notes    = _prompt("Notes", "any extra context")
        except (EOFError, KeyboardInterrupt):
            self.console.print("\n  [dim]Verify cancelled.[/dim]")
            return None

        inp = SpectralInput.from_cli(
            smiles=smiles or None,
            sm_smiles=sm_smiles or None,
            h1=h1 or None,
            c13=c13 or None,
            ms=ms or None,
            ir=ir or None,
            solvent=solvent or None,
            reaction=reaction or None,
            notes=notes or None,
            mode="verify",
        )

        if not inp.has_any_data():
            self.console.print("  [yellow]No data entered. Verify cancelled.[/yellow]")
            return None

        from rich.panel import Panel
        self.console.print(Panel(
            f"[dim]{inp.summary()}[/dim]",
            title="[bold cyan]Spectral Input[/bold cyan]",
            border_style="cyan",
            padding=(0, 2),
        ))

        # Build query
        if smiles:
            query = (
                f"Verify whether the spectral data is consistent with "
                f"the expected product (SMILES: {smiles})."
            )
        else:
            query = "Analyze the provided spectral data and identify the compound."

        if reaction:
            query += f" Reaction type: {reaction}."
        if solvent:
            query += f" NMR solvent: {solvent}."

        return query, inp

    def _switch_model(self):
        models = [
            ("claude-opus-4-6",           "Opus 4.6   — most capable"),
            ("claude-sonnet-4-6",         "Sonnet 4.6 — balanced"),
            ("claude-haiku-4-5-20251001", "Haiku 4.5  — fastest"),
        ]
        self.console.print("\n[bold]Select model:[/bold]")
        current = self.session.config.get("llm.model")
        for i, (mid, label) in enumerate(models, 1):
            marker = "[green]●[/green]" if mid == current else " "
            self.console.print(f"  {marker} [{i}] {label}")
        try:
            raw = input("\n  Enter number (or Enter to keep current): ").strip()
            if raw.isdigit():
                idx = int(raw) - 1
                if 0 <= idx < len(models):
                    mid, label = models[idx]
                    self.session.config.set("llm.model", mid)
                    self.session.config.save()
                    self.console.print(f"  [green]Model set to {label.split('—')[0].strip()}[/green]")
                    return
            self.console.print("  [dim]Model unchanged.[/dim]")
        except (EOFError, KeyboardInterrupt):
            self.console.print("  [dim]Cancelled.[/dim]")

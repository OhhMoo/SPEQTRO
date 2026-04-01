"""Session management: holds config and shared state for a speqtro session."""

from rich.console import Console
from speqtro.agent.config import Config


class Session:
    """Manages state for a speqtro analysis session."""

    def __init__(self, config: Config = None, verbose: bool = False, mode: str = "batch"):
        self.config = config or Config.load()
        self.verbose = verbose
        self.mode = mode  # "interactive" or "batch"
        self.console = Console()
        self._active_spinner = None

    def log(self, message: str):
        if self.verbose:
            self.console.print(f"  [dim]{message}[/dim]")

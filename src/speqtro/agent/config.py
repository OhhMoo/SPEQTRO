"""Configuration management for speqtro. Stored at ~/.speqtro/config.json."""

import json
import os
import logging
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()
load_dotenv(Path(__file__).resolve().parents[3] / ".env")

from rich.table import Table

CONFIG_DIR = Path.home() / ".speqtro"
CONFIG_FILE = CONFIG_DIR / "config.json"
logger = logging.getLogger("speqtro.config")

DEFAULTS = {
    "llm.provider": "anthropic",
    "llm.model": "claude-opus-4-6",
    "llm.api_key": None,
    "llm.temperature": 0.1,

    "output.format": "markdown",
    "output.verbose": False,

    "ui.spinner": "benzene_breathing",

    "sandbox.timeout": 60,
    "sandbox.output_dir": str(Path.cwd() / "outputs"),
    "sandbox.max_retries": 2,

    "agent.max_sdk_turns": 30,
    "agent.enable_experimental_tools": False,
    "agent.plan_preview": False,
}


class Config:
    """speqtro configuration manager."""

    def __init__(self, data: dict = None):
        self._data = data or {}

    @classmethod
    def load(cls) -> "Config":
        """Load config from disk, creating defaults if needed."""
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE) as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    data = {}
            except (json.JSONDecodeError, OSError):
                data = {}
        else:
            data = {}

        # Check environment variables
        env_mappings = {
            "ANTHROPIC_API_KEY": "llm.api_key",
            "SPEQ_LLM_MODEL": "llm.model",
        }
        for env_var, config_key in env_mappings.items():
            val = os.environ.get(env_var)
            if val and config_key not in data:
                data[config_key] = val

        return cls(data)

    def save(self):
        """Save config to disk."""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, "w") as f:
            json.dump(self._data, f, indent=2)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, DEFAULTS.get(key, default))

    def set(self, key: str, value: Any):
        if key in DEFAULTS and DEFAULTS[key] is not None:
            expected_type = type(DEFAULTS[key])
            if expected_type == bool:
                value = value.lower() in ("true", "1", "yes") if isinstance(value, str) else bool(value)
            elif expected_type == float:
                value = float(value)
            elif expected_type == int:
                value = int(value)
        self._data[key] = value

    def llm_api_key(self, provider: Optional[str] = None) -> Optional[str]:
        return self.get("llm.api_key")

    def llm_preflight_issue(self) -> Optional[str]:
        if self.llm_api_key():
            return None
        if os.environ.get("ANTHROPIC_FOUNDRY_API_KEY") or os.environ.get("ANTHROPIC_FOUNDRY_RESOURCE"):
            return None
        return (
            "Anthropic API key not configured. Set ANTHROPIC_API_KEY or run:\n"
            "  speqtro config set llm.api_key <key>"
        )

    def to_table(self) -> Table:
        table = Table(title="speqtro Configuration")
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Source", style="dim")
        all_keys = sorted(set(list(DEFAULTS.keys()) + list(self._data.keys())))
        for key in all_keys:
            if key in self._data:
                val = self._data[key]
                source = "config"
            elif key in DEFAULTS:
                val = DEFAULTS[key]
                source = "default"
            else:
                continue
            display_val = str(val)
            if "api_key" in key and val and len(str(val)) > 8:
                display_val = str(val)[:4] + "..." + str(val)[-4:]
            table.add_row(key, display_val, source)
        return table

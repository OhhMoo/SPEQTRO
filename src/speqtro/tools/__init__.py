"""Tool registry for speqtro."""

from dataclasses import dataclass, field
import importlib
import logging
from typing import Callable, Optional
from rich.table import Table


_TOOL_MODULES = (
    "nmr", "ms", "structure", "database", "ir", "cascade", "cascade1", "chefnmr",
    # New external tool wrappers
    "ssin",
    "mspred",
    "dp5",
    # Database search tools
    "nmrshiftdb",
    # Pipeline modes (full import path)
    "speqtro.modes.verify",
    "speqtro.modes.pipeline",
)


@dataclass
class Tool:
    name: str
    description: str
    category: str
    function: Callable
    parameters: dict = field(default_factory=dict)
    requires_data: list = field(default_factory=list)

    def run(self, **kwargs):
        return self.function(**kwargs)

    def input_schema(self) -> dict:
        """Return a JSON Schema dict describing this tool's parameters."""
        if not self.parameters:
            return {"type": "object", "properties": {}}
        if self.parameters.get("type") == "object" and isinstance(self.parameters.get("properties"), dict):
            return self.parameters
        properties = {}
        required = []
        for name, desc in self.parameters.items():
            properties[name] = {"type": "string", "description": str(desc)}
            required.append(name)
        schema: dict = {"type": "object", "properties": properties}
        if required:
            schema["required"] = required
        return schema


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, name: str, description: str, category: str,
                 parameters: dict = None, requires_data: list = None):
        def decorator(func):
            self._tools[name] = Tool(
                name=name,
                description=description,
                category=category,
                function=func,
                parameters=parameters or {},
                requires_data=requires_data or [],
            )
            return func
        return decorator

    def get_tool(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list_tools(self, category: str = None) -> list[Tool]:
        tools = list(self._tools.values())
        if category:
            tools = [t for t in tools if t.category == category]
        return sorted(tools, key=lambda t: t.name)

    def categories(self) -> list[str]:
        return sorted(set(t.category for t in self._tools.values()))

    def list_tools_table(self) -> Table:
        table = Table(title="speqtro Tools")
        table.add_column("Tool", style="cyan")
        table.add_column("Category", style="yellow")
        table.add_column("Description")
        for tool in self.list_tools():
            table.add_row(tool.name, tool.category, tool.description)
        return table


registry = ToolRegistry()


def _load_tools():
    logger = logging.getLogger("speqtro.tools")
    errors = {}
    for module_name in _TOOL_MODULES:
        # Entries that already contain a dot are treated as full import paths;
        # short names are prefixed with "speqtro.tools."
        if "." in module_name:
            import_name = module_name
        else:
            import_name = f"speqtro.tools.{module_name}"
        try:
            importlib.import_module(import_name)
        except Exception as exc:
            errors[module_name] = str(exc)
            logger.warning("Failed to load tool module %s: %s", import_name, exc)
    return errors


_loaded = False
_load_errors: dict[str, str] = {}


def ensure_loaded():
    global _loaded, _load_errors
    if not _loaded:
        _load_errors = _load_tools()
        _loaded = True


def tool_load_errors() -> dict[str, str]:
    return dict(_load_errors)

"""MCP tool server for speqtro — wraps ToolRegistry for the Claude Agent SDK.

Two modes of operation:
  1. In-process (AgentRunner): create_speq_mcp_server(session) — existing behaviour.
  2. Standalone (CLI): serve_stdio(config), serve_http(config, host, port),
     list_tools_json(config) — no session or AgentRunner required.
"""

import asyncio
import json
import logging
from typing import Any

logger = logging.getLogger("speqtro.mcp_server")


def _format_tool_result(result: Any, max_chars: int = 8000) -> str:
    if not isinstance(result, dict):
        text = str(result)
        return text[:max_chars] if len(text) > max_chars else text
    parts = []
    summary = result.get("summary", "")
    if summary:
        parts.append(summary)
    skip = {"summary"}
    for key, val in result.items():
        if key in skip:
            continue
        val_str = str(val)
        if len(val_str) > 1500:
            val_str = val_str[:1500] + f"... [{len(val_str)} chars total]"
        parts.append(f"{key}: {val_str}")
    text = "\n".join(parts)
    return text[:max_chars] if len(text) > max_chars else text


def _params_to_json_schema(parameters: dict) -> dict:
    if not parameters:
        return {"type": "object", "properties": {}}
    if parameters.get("type") == "object" and isinstance(parameters.get("properties"), dict):
        return parameters
    properties = {}
    for name, desc in parameters.items():
        properties[name] = {"type": "string", "description": str(desc)}
    return {"type": "object", "properties": properties}


def _make_tool_handler(tool_obj, session):
    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        call_args = dict(args)
        call_args["_session"] = session
        call_args["_prior_results"] = {}
        try:
            result = await asyncio.to_thread(tool_obj.run, **call_args)
            text = _format_tool_result(result)
        except Exception as e:
            logger.warning("Tool %s raised: %s", tool_obj.name, e)
            return {"content": [{"type": "text", "text": f"Error: {e}"}], "is_error": True}
        return {"content": [{"type": "text", "text": text}]}
    return handler


def _make_run_python_handler(session, code_trace_buffer: list | None = None):
    from speqtro.agent.sandbox import Sandbox
    config = session.config
    timeout = int(config.get("sandbox.timeout", 60))
    output_dir = config.get("sandbox.output_dir")
    max_retries = int(config.get("sandbox.max_retries", 2))
    sandbox = Sandbox(timeout=timeout, output_dir=output_dir, max_retries=max_retries)
    sandbox.load_datasets()

    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        code = args.get("code", "")
        if not code.strip():
            return {"content": [{"type": "text", "text": "Error: no code provided"}], "is_error": True}
        exec_result = await asyncio.to_thread(sandbox.execute, code)
        parts = []
        if exec_result.get("stdout"):
            parts.append(exec_result["stdout"])
        if exec_result.get("error"):
            parts.append(f"Error:\n{exec_result['error']}")
        if exec_result.get("plots"):
            parts.append(f"Plots saved: {exec_result['plots']}")
        result_var = sandbox.get_variable("result")
        if result_var and isinstance(result_var, dict):
            if result_var.get("summary"):
                parts.append(f"\nResult: {result_var['summary']}")
        text = "\n".join(parts) if parts else "(no output)"
        text = text[:6000]
        if code_trace_buffer is not None:
            code_trace_buffer.append({
                "tool": "run_python", "code": code,
                "stdout": exec_result.get("stdout", ""),
                "error": exec_result.get("error"),
            })
        return {"content": [{"type": "text", "text": text}], "is_error": bool(exec_result.get("error"))}

    return handler, sandbox


def create_speq_mcp_server(
    session,
    *,
    exclude_categories: set[str] | None = None,
    exclude_tools: set[str] | None = None,
    include_run_python: bool = True,
):
    """Create an in-process MCP server exposing all speqtro tools."""
    from claude_agent_sdk import SdkMcpTool, create_sdk_mcp_server  # noqa: F401 (in-process only)
    from speqtro.tools import registry, ensure_loaded

    ensure_loaded()

    exclude_categories = exclude_categories or set()
    exclude_tools = exclude_tools or set()
    code_trace_buffer: list[dict] = []

    sdk_tools: list[SdkMcpTool] = []
    tool_names: list[str] = []

    for tool_obj in registry.list_tools():
        if tool_obj.category in exclude_categories:
            continue
        if tool_obj.name in exclude_tools:
            continue

        handler = _make_tool_handler(tool_obj, session)
        schema = _params_to_json_schema(tool_obj.parameters)

        sdk_tool = SdkMcpTool(
            name=tool_obj.name,
            description=tool_obj.description,
            input_schema=schema,
            handler=handler,
        )
        sdk_tools.append(sdk_tool)
        tool_names.append(tool_obj.name)

    sandbox = None
    if include_run_python:
        rp_handler, sandbox = _make_run_python_handler(session, code_trace_buffer)
        rp_tool = SdkMcpTool(
            name="run_python",
            description=(
                "Execute Python code in a sandboxed environment. Variables persist between calls. "
                "Pre-imported: pd, np, plt, scipy_stats, json, re, math, Path, OUTPUT_DIR. "
                "RDKit available as Chem, Descriptors, AllChem. "
                "Assign result = {'summary': '...', 'answer': '...'} when done."
            ),
            input_schema={
                "type": "object",
                "properties": {"code": {"type": "string", "description": "Python code to execute"}},
                "required": ["code"],
            },
            handler=rp_handler,
        )
        sdk_tools.append(rp_tool)
        tool_names.append("run_python")

    server = create_sdk_mcp_server(name="speqtro-tools", version="1.0.0", tools=sdk_tools)

    logger.info("Created speqtro MCP server with %d tools", len(sdk_tools))
    return server, sandbox, tool_names, code_trace_buffer


# ---------------------------------------------------------------------------
# Standalone MCP server (no AgentRunner required)
# ---------------------------------------------------------------------------

def _build_standalone_mcp_app():
    """Build an mcp.server.Server loaded with all registered speqtro tools."""
    try:
        from mcp.server import Server
        from mcp import types as mcp_types
    except ImportError as exc:
        raise RuntimeError(
            "mcp package not installed. Run: pip install 'speqtro[mcp]'"
        ) from exc

    from speqtro.tools import registry, ensure_loaded

    ensure_loaded()

    app = Server("speqtro-tools")

    @app.list_tools()
    async def _list_tools() -> list[mcp_types.Tool]:
        tools = []
        for tool_obj in registry.list_tools():
            tools.append(
                mcp_types.Tool(
                    name=tool_obj.name,
                    description=tool_obj.description,
                    inputSchema=tool_obj.input_schema(),
                )
            )
        return tools

    @app.call_tool()
    async def _call_tool(name: str, arguments: dict) -> list[mcp_types.TextContent]:
        tool_obj = registry.get_tool(name)
        if tool_obj is None:
            return [mcp_types.TextContent(type="text", text=f"Error: unknown tool '{name}'")]
        try:
            result = await asyncio.to_thread(tool_obj.run, **arguments)
            text = _format_tool_result(result)
        except Exception as exc:
            logger.warning("Standalone tool %s raised: %s", name, exc)
            text = f"Error: {exc}"
        return [mcp_types.TextContent(type="text", text=text)]

    return app


async def serve_stdio(config) -> None:
    """Run speqtro as an MCP server over stdio (for Claude.ai / Cursor / Claude Code)."""
    try:
        from mcp.server.stdio import stdio_server
    except ImportError as exc:
        raise RuntimeError(
            "mcp package not installed. Run: pip install 'speqtro[mcp]'"
        ) from exc

    app = _build_standalone_mcp_app()
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


async def serve_http(config, host: str = "127.0.0.1", port: int = 8765) -> None:
    """Run speqtro as an MCP server over HTTP/SSE."""
    try:
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.routing import Route
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "HTTP server deps not installed. Run: pip install 'speqtro[mcp]'"
        ) from exc

    app = _build_standalone_mcp_app()
    sse = SseServerTransport("/messages")

    async def handle_sse(request):
        async with sse.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            await app.run(streams[0], streams[1], app.create_initialization_options())

    async def handle_messages(request):
        await sse.handle_post_message(request.scope, request.receive, request._send)

    starlette_app = Starlette(
        routes=[
            Route("/sse", endpoint=handle_sse),
            Route("/messages", endpoint=handle_messages, methods=["POST"]),
        ]
    )

    logger.info("speqtro MCP HTTP/SSE server starting on %s:%d", host, port)
    uvicorn_config = uvicorn.Config(starlette_app, host=host, port=port, log_level="info")
    server = uvicorn.Server(uvicorn_config)
    await server.serve()


def list_tools_json(config) -> str:
    """Return all registered tools serialised as a JSON string."""
    from speqtro.tools import registry, ensure_loaded

    ensure_loaded()

    tools = []
    for tool_obj in registry.list_tools():
        tools.append(
            {
                "name": tool_obj.name,
                "description": tool_obj.description,
                "category": tool_obj.category,
                "inputSchema": tool_obj.input_schema(),
            }
        )
    return json.dumps(tools, indent=2)

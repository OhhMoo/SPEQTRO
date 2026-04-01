"""AgentRunner: query entry point using the Claude Agent SDK."""

import asyncio
import logging
import os
import time
import traceback

from speqtro.agent.types import ExecutionResult, Plan, Step

logger = logging.getLogger("speqtro.runner")


class AgentRunner:
    """Run queries via the Claude Agent SDK agentic loop."""

    def __init__(self, session, trajectory=None, headless: bool = False):
        self.session = session
        self.trajectory = trajectory
        self._headless = headless
        self._active_spinner = None

    def run(self, query: str, context: dict | None = None,
            spectral_input=None) -> ExecutionResult:
        import threading

        loop = asyncio.new_event_loop()
        result_holder: list = [None]
        exc_holder: list = [None]

        def _target():
            asyncio.set_event_loop(loop)
            try:
                result_holder[0] = loop.run_until_complete(
                    self._run_async(query, context, spectral_input)
                )
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                exc_holder[0] = exc
            finally:
                loop.close()

        thread = threading.Thread(target=_target, daemon=True)
        thread.start()

        try:
            while thread.is_alive():
                thread.join(timeout=0.05)
        except KeyboardInterrupt:
            if not loop.is_closed():
                def _cancel_all():
                    for task in asyncio.all_tasks(loop):
                        task.cancel()
                loop.call_soon_threadsafe(_cancel_all)
            thread.join(timeout=5.0)
            if self._active_spinner is not None:
                try:
                    self._active_spinner.stop()
                except Exception:
                    pass
                self._active_spinner = None
            raise

        if exc_holder[0] is not None:
            raise exc_holder[0]
        return result_holder[0]

    async def _run_async(self, query: str, context: dict | None = None,
                         spectral_input=None) -> ExecutionResult:
        from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions
        from speqtro.agent.mcp_server import create_speq_mcp_server
        from speqtro.agent.system_prompt import build_system_prompt

        thinking_status = None
        if not self._headless:
            try:
                from speqtro.ui.status import ThinkingStatus
                thinking_status = ThinkingStatus(self.session.console, phase="planning")
                thinking_status.__enter__()
                thinking_status.start_async_refresh()
                self._active_spinner = thinking_status
            except Exception:
                pass

        t0 = time.time()
        config = self.session.config
        ctx = context or {}

        server, sandbox, tool_names, code_trace_buffer = create_speq_mcp_server(self.session)

        history = None
        if self.trajectory and self.trajectory.turns:
            history = self.trajectory.context_for_planner()

        data_context = None
        structured_context = None
        if spectral_input is not None:
            try:
                data_context = spectral_input.to_context_string()
                structured_context = _build_structured_context(spectral_input)
            except Exception:
                pass

        system_prompt = build_system_prompt(
            self.session,
            tool_names=tool_names,
            data_context=data_context,
            history=history,
        )

        model = config.get("llm.model") or "claude-opus-4-6"
        max_turns = int(config.get("agent.max_sdk_turns", 30))
        allowed_tools = [f"mcp__speq-tools__{name}" for name in tool_names]

        _STRIP_VARS = {"CLAUDECODE", "CLAUDE_CODE_SESSION_ID", "CLAUDE_CODE_PARENT_SESSION_ID"}
        clean_env = {k: v for k, v in os.environ.items() if k not in _STRIP_VARS}
        api_key = config.llm_api_key("anthropic")
        if api_key:
            clean_env["ANTHROPIC_API_KEY"] = api_key
        clean_env["PYTHONWARNINGS"] = "ignore"

        options_kwargs = dict(
            system_prompt=system_prompt,
            model=model,
            max_turns=max_turns,
            mcp_servers={"speqtro-tools": server},
            allowed_tools=allowed_tools,
            permission_mode="bypassPermissions",
            env=clean_env,
            hooks={},
        )

        try:
            options = ClaudeAgentOptions(include_partial_messages=True, **options_kwargs)
        except TypeError:
            options = ClaudeAgentOptions(**options_kwargs)

        full_text: list[str] = []
        tool_calls: list[dict] = []

        try:
            from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock, ToolUseBlock
            try:
                from claude_agent_sdk import ToolResultBlock, StreamEvent
            except ImportError:
                ToolResultBlock = None
                StreamEvent = None

            # Prepend structured JSON context to query so tool args are ready-to-use
            effective_query = query
            if structured_context:
                effective_query = structured_context + "\n\n" + query

            result_msg = None
            async with ClaudeSDKClient(options=options) as client:
                await client.query(effective_query)
                async for message in client.receive_response():
                    if StreamEvent is not None and isinstance(message, StreamEvent):
                        continue
                    if isinstance(message, AssistantMessage):
                        for block in (message.content or []):
                            if isinstance(block, TextBlock):
                                if thinking_status is not None:
                                    thinking_status.stop()
                                    thinking_status = None
                                    self._active_spinner = None
                                text = block.text or ""
                                full_text.append(text)
                                if not self._headless and text.strip():
                                    try:
                                        from speqtro.ui.markdown import LeftMarkdown as Markdown
                                    except ImportError:
                                        from rich.markdown import Markdown
                                    self.session.console.print(Markdown(text))
                            elif isinstance(block, ToolUseBlock):
                                if thinking_status is None and not self._headless:
                                    try:
                                        from speqtro.ui.status import ThinkingStatus
                                        thinking_status = ThinkingStatus(self.session.console, phase="evaluating")
                                        thinking_status.__enter__()
                                        thinking_status.start_async_refresh()
                                        self._active_spinner = thinking_status
                                    except Exception:
                                        pass
                                tool_calls.append({"name": block.name, "input": block.input})
                                if not self._headless:
                                    clean = block.name.replace("mcp__speq-tools__", "")
                                    self.session.console.print(f"  [cyan]▸ {clean}[/cyan]  [dim]{block.input}[/dim]")
                    elif isinstance(message, ResultMessage):
                        if thinking_status is not None:
                            thinking_status.stop()
                            thinking_status = None
                        result_msg = message

        except Exception as e:
            logger.error("Agent SDK query failed: %s\n%s", e, traceback.format_exc())
            duration = time.time() - t0
            if thinking_status is not None:
                thinking_status.stop()
            return self._make_error_result(query, str(e), duration)
        finally:
            if thinking_status is not None:
                thinking_status.stop()

        duration = time.time() - t0
        summary = "\n".join(full_text).strip() or "(Agent produced no text output)"

        steps = [
            Step(id=i, tool=tc["name"].replace("mcp__speq-tools__", ""),
                 description=f"Called {tc['name']}", tool_args=tc.get("input", {}),
                 status="completed")
            for i, tc in enumerate(tool_calls, 1)
        ]
        plan = Plan(query=query, steps=steps)

        cost_usd = getattr(result_msg, "total_cost_usd", 0.0) or 0.0 if result_msg else 0.0

        if not self._headless and result_msg:
            self._print_usage(result_msg, duration)

        return ExecutionResult(
            plan=plan,
            summary=summary,
            raw_results={"tool_calls": tool_calls},
            duration_s=duration,
            iterations=1,
            metadata={
                "sdk_cost_usd": cost_usd,
                "sdk_turns": getattr(result_msg, "num_turns", 0) if result_msg else 0,
                "tool_call_count": len(tool_calls),
            },
        )

    def _print_usage(self, result_msg, duration: float):
        cost = getattr(result_msg, "total_cost_usd", 0)
        turns = getattr(result_msg, "num_turns", 0)
        parts = []
        if cost:
            parts.append(f"${cost:.2f}")
        if turns:
            parts.append(f"{turns} turns")
        if duration >= 60:
            parts.append(f"{int(duration//60)}m {int(duration%60)}s")
        else:
            parts.append(f"{duration:.1f}s")
        self.session.console.print(f"\n  [dim]{' | '.join(parts)}[/dim]")

    async def astream_run(self, query: str, context: dict | None = None, spectral_input=None):
        """Async generator that yields SSE-ready event dicts as the agent runs.

        Event types:
          {"type": "chunk", "text": "…"}          — assistant text fragment
          {"type": "tool",  "name": "tool_name"}  — tool invocation
          {"type": "done",  "duration_s": …, "cost_usd": …, "tool_calls": […]}
          {"type": "error", "text": "…"}
        """
        from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions
        from speqtro.agent.mcp_server import create_speq_mcp_server
        from speqtro.agent.system_prompt import build_system_prompt

        t0 = time.time()
        config = self.session.config
        ctx = context or {}

        server, sandbox, tool_names, code_trace_buffer = create_speq_mcp_server(self.session)

        history = None
        if self.trajectory and self.trajectory.turns:
            history = self.trajectory.context_for_planner()

        data_context = None
        structured_context = None
        if spectral_input is not None:
            try:
                data_context = spectral_input.to_context_string()
                structured_context = _build_structured_context(spectral_input)
            except Exception:
                pass

        system_prompt = build_system_prompt(
            self.session,
            tool_names=tool_names,
            data_context=data_context,
            history=history,
        )

        model = config.get("llm.model") or "claude-opus-4-6"
        max_turns = int(config.get("agent.max_sdk_turns", 30))
        allowed_tools = [f"mcp__speq-tools__{name}" for name in tool_names]

        _STRIP_VARS = {"CLAUDECODE", "CLAUDE_CODE_SESSION_ID", "CLAUDE_CODE_PARENT_SESSION_ID"}
        clean_env = {k: v for k, v in os.environ.items() if k not in _STRIP_VARS}
        api_key = config.llm_api_key("anthropic")
        if api_key:
            clean_env["ANTHROPIC_API_KEY"] = api_key
        clean_env["PYTHONWARNINGS"] = "ignore"

        options_kwargs = dict(
            system_prompt=system_prompt,
            model=model,
            max_turns=max_turns,
            mcp_servers={"speqtro-tools": server},
            allowed_tools=allowed_tools,
            permission_mode="bypassPermissions",
            env=clean_env,
            hooks={},
        )

        try:
            options = ClaudeAgentOptions(include_partial_messages=True, **options_kwargs)
        except TypeError:
            options = ClaudeAgentOptions(**options_kwargs)

        tool_calls: list[str] = []
        result_msg = None

        try:
            from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock, ToolUseBlock
            try:
                from claude_agent_sdk import StreamEvent
            except ImportError:
                StreamEvent = None

            effective_query = query
            if structured_context:
                effective_query = structured_context + "\n\n" + query

            async with ClaudeSDKClient(options=options) as client:
                await client.query(effective_query)
                async for message in client.receive_response():
                    if StreamEvent is not None and isinstance(message, StreamEvent):
                        continue
                    if isinstance(message, AssistantMessage):
                        for block in (message.content or []):
                            if isinstance(block, TextBlock):
                                text = block.text or ""
                                if text:
                                    yield {"type": "chunk", "text": text}
                            elif isinstance(block, ToolUseBlock):
                                clean = block.name.replace("mcp__speq-tools__", "")
                                tool_calls.append(clean)
                                yield {"type": "tool", "name": clean}
                    elif isinstance(message, ResultMessage):
                        result_msg = message

        except Exception as exc:
            logger.error("astream_run failed: %s\n%s", exc, traceback.format_exc())
            yield {"type": "error", "text": str(exc)}
            return

        duration = time.time() - t0
        cost = getattr(result_msg, "total_cost_usd", 0.0) or 0.0 if result_msg else 0.0
        yield {"type": "done", "duration_s": round(duration, 2), "cost_usd": cost, "tool_calls": tool_calls}

    @staticmethod
    def _make_error_result(query: str, error: str, duration: float) -> ExecutionResult:
        return ExecutionResult(
            plan=Plan(query=query, steps=[]),
            summary=f"Agent SDK error: {error}",
            raw_results={"error": error},
            duration_s=duration,
            iterations=1,
        )


def _build_structured_context(spectral_input) -> str:
    """
    Build a structured JSON block from SpectralInput that Claude can use
    directly as tool arguments — avoiding markdown re-parsing.

    Returns a string like:
        [TOOL_ARGS for pipeline.verify_product]
        { "smiles": "...", "observed_peaks": {...}, ... }
        [/TOOL_ARGS]
    """
    import json

    mode = getattr(spectral_input, "mode", "verify")
    blocks: list[str] = []

    # ── pipeline.verify_product args ───────────────────────────────────
    if mode == "verify" and spectral_input.smiles:
        args: dict = {"smiles": spectral_input.smiles}
        if spectral_input.sm_smiles:
            args["sm_smiles"] = spectral_input.sm_smiles
        if spectral_input.h1_peaks or spectral_input.c13_peaks:
            args["observed_peaks"] = {
                "h1": spectral_input.h1_peaks,
                "c13": spectral_input.c13_peaks,
            }
        if spectral_input.solvent:
            args["solvent"] = spectral_input.solvent
        blocks.append(
            "[TOOL_ARGS for pipeline.verify_product]\n"
            + json.dumps(args, indent=2)
            + "\n[/TOOL_ARGS]"
        )

    # ── pipeline.full_elucidation args ─────────────────────────────────
    if mode == "explore":
        args = {}
        if spectral_input.h1_peaks:
            args["h1_peaks"] = spectral_input.h1_peaks
        if spectral_input.c13_peaks:
            args["c13_peaks"] = spectral_input.c13_peaks
        if spectral_input.solvent:
            args["solvent"] = spectral_input.solvent
        if spectral_input.ir_file:
            args["ir_file"] = spectral_input.ir_file
        if spectral_input.ms_peaks:
            args["ms_peaks"] = spectral_input.ms_peaks
            if spectral_input.ms_precursor_mz:
                args["precursor_mz"] = spectral_input.ms_precursor_mz
            if spectral_input.ms_collision_energy:
                args["collision_energy"] = spectral_input.ms_collision_energy
            if spectral_input.ms_adduct:
                args["adduct"] = spectral_input.ms_adduct
        if args:
            blocks.append(
                "[TOOL_ARGS for pipeline.full_elucidation]\n"
                + json.dumps(args, indent=2)
                + "\n[/TOOL_ARGS]"
            )

    # ── ms.predict_msms_iceberg args (any mode with MS data) ───────────
    if spectral_input.ms_peaks and spectral_input.smiles:
        iceberg_args: dict = {
            "smiles": spectral_input.smiles,
            "experimental_peaks": spectral_input.ms_peaks,
        }
        if spectral_input.ms_precursor_mz:
            iceberg_args["precursor_mz"] = spectral_input.ms_precursor_mz
        if spectral_input.ms_collision_energy:
            iceberg_args["collision_energy"] = spectral_input.ms_collision_energy
        if spectral_input.ms_adduct:
            iceberg_args["adduct"] = spectral_input.ms_adduct
        blocks.append(
            "[TOOL_ARGS for ms.predict_msms_iceberg]\n"
            + json.dumps(iceberg_args, indent=2)
            + "\n[/TOOL_ARGS]"
        )

    # ── ir.detect_functional_groups_ssin args ──────────────────────────
    if spectral_input.ir_wavenumbers:
        ssin_args: dict = {
            "wavenumbers": spectral_input.ir_wavenumbers,
            "intensities": spectral_input.ir_intensities,
        }
        if spectral_input.ir_file:
            ssin_args["jdx_file"] = spectral_input.ir_file
        blocks.append(
            "[TOOL_ARGS for ir.detect_functional_groups_ssin]\n"
            + json.dumps(ssin_args, indent=2)
            + "\n[/TOOL_ARGS]"
        )

    return "\n\n".join(blocks) if blocks else ""

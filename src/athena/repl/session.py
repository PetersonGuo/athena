"""DebugSession - main orchestrator for the debugging REPL."""

from __future__ import annotations

import bdb
import os
import signal
import sys
import re
from typing import Any

from athena.agent.llm_client import LLMClient
from athena.agent.system_prompt import build_stop_context
from athena.agent.tool_executor import ToolExecutor
from athena.core.breakpoint_manager import BreakpointManager
from athena.core.debugger import DebuggerCore
from athena.core.expression_evaluator import ExpressionEvaluator
from athena.core.frame_inspector import FrameInspector
from athena.core.source_provider import SourceProvider
from athena.core.watch_manager import WatchManager
from athena.memory.cuda_memory import CudaMemoryProfiler
from athena.memory.leak_detector import LeakDetector
from athena.memory.python_memory import PythonMemoryProfiler
from athena.repl.commands import CommandHandler
from athena.repl.input_handler import InputHandler
from athena.repl.output_formatter import OutputFormatter
from athena.utils.config import DebugConfig


class DebugSession:
    """The main orchestrator. Owns all components and manages the REPL.

    Lifecycle:
    1. DebuggerCore runs the target script via bdb.run()
    2. When a stop condition is met, user_line() -> session.on_debugger_stop()
    3. on_debugger_stop() evaluates watches, builds context, enters REPL
    4. REPL: read input -> slash command or send to LLM -> display -> loop
    5. Execution control action -> exit REPL, resume target
    6. Target runs until next stop -> goto 2

    Runs SYNCHRONOUSLY in the target thread (same approach as pdb).
    """

    def __init__(
        self,
        config: DebugConfig | None = None,
    ):
        self._config = config or DebugConfig.from_env()

        # Core
        self._debugger = DebuggerCore()
        self._frame_inspector = FrameInspector(self._debugger)
        self._evaluator = ExpressionEvaluator(self._debugger)
        self._source_provider = SourceProvider()
        self._watch_manager = WatchManager(self._evaluator)
        self._breakpoint_manager = BreakpointManager(self._debugger)

        # Memory
        self._python_memory = PythonMemoryProfiler(self._config.tracemalloc_nframes)
        self._cuda_memory = CudaMemoryProfiler()
        self._leak_detector = LeakDetector(self._python_memory, self._cuda_memory)

        # Agent
        self._tool_executor = ToolExecutor(
            debugger=self._debugger,
            frame_inspector=self._frame_inspector,
            evaluator=self._evaluator,
            watch_manager=self._watch_manager,
            breakpoint_manager=self._breakpoint_manager,
            source_provider=self._source_provider,
            python_memory=self._python_memory,
            cuda_memory=self._cuda_memory,
            leak_detector=self._leak_detector,
        )

        self._llm = LLMClient(
            model=self._config.model,
            api_key=self._config.api_key,
            base_url=self._config.base_url,
        )
        self._llm.set_tool_executor(self._tool_executor)

        # REPL
        self._formatter = OutputFormatter()
        self._input_handler = InputHandler(self._config.history_file)
        self._command_handler = CommandHandler(self)
        self._llm.set_tool_call_observer(
            lambda name, _args: self._formatter.show_tool_call(name)
        )

        # Wire up
        self._debugger.attach_session(self)

        # State
        self._execution_action: str | None = None
        self._first_stop = True
        self._quitting = False
        self._rerun_requested = False
        self._post_run_script_path: str | None = None

    # --- Public properties for CommandHandler access ---

    @property
    def debugger(self) -> DebuggerCore:
        return self._debugger

    @property
    def frame_inspector(self) -> FrameInspector:
        return self._frame_inspector

    @property
    def formatter(self) -> OutputFormatter:
        return self._formatter

    @property
    def watch_manager(self) -> WatchManager:
        return self._watch_manager

    @property
    def breakpoint_manager(self) -> BreakpointManager:
        return self._breakpoint_manager

    @property
    def python_memory(self) -> PythonMemoryProfiler:
        return self._python_memory

    @property
    def is_quitting(self) -> bool:
        return self._quitting

    @property
    def is_rerun_requested(self) -> bool:
        return self._rerun_requested

    def set_execution_action(self, action: str) -> None:
        """Set the execution control action (called by commands and tools)."""
        self._execution_action = action

    def on_debugger_stop(
        self,
        frame: Any,
        exc_info: tuple | None,
        reason: str,
    ) -> None:
        """Called by DebuggerCore when execution pauses.

        Evaluates watches, builds context, and enters the REPL.
        """
        if self._quitting:
            raise bdb.BdbQuit
        self._post_run_script_path = None

        # Evaluate watch expressions
        watch_results = self._watch_manager.evaluate_all()

        # Get source context
        source_ctx = self._frame_inspector.get_source_context(context_lines=10)

        # Build system prompt with current context
        system_context = build_stop_context(
            source_context=source_ctx,
            stop_reason=reason,
            watch_changes=watch_results,
            exception_info=exc_info,
        )
        self._llm.set_system_prompt(system_context)

        # Show stop banner
        if self._first_stop:
            self._formatter.show_banner()
            self._first_stop = False

        self._formatter.show_stop_banner(
            filename=source_ctx.get("filename", "?"),
            lineno=source_ctx.get("current_line", 0),
            function=source_ctx.get("function", "?"),
            reason=reason,
            source_lines=source_ctx.get("lines"),
        )

        # Show watch changes
        changed_watches = [w for w in watch_results if w.get("changed")]
        if changed_watches:
            for w in changed_watches:
                self._formatter.show_info(
                    f"Watch changed: {w['expression']}: "
                    f"{w.get('previous_value')} -> {w['current_value']}"
                )

        # Enter REPL
        self._repl_loop()

        # Apply execution control
        action = self._execution_action
        self._execution_action = None

        if action == "step":
            self._debugger.do_step()
        elif action == "next":
            self._debugger.do_next()
        elif action == "continue":
            self._debugger.do_continue()
        elif action == "return":
            self._debugger.do_return()
        elif action == "quit":
            self._quitting = True
            self._formatter.show_info("Quitting Athena.")
            raise bdb.BdbQuit
        elif action == "rerun":
            self._rerun_requested = True
            self._formatter.show_info("Restarting target script.")
            raise bdb.BdbQuit
        else:
            # Default: continue
            self._debugger.do_continue()

    def _repl_loop(self) -> None:
        """The interactive REPL loop."""
        self._execution_action = None

        while self._execution_action is None:
            try:
                user_input = self._input_handler.read_input("debug> ")
            except KeyboardInterrupt:
                # Double Ctrl-C to quit
                self._formatter.show_info("Press Ctrl-C again to quit, or type a command.")
                try:
                    user_input = self._input_handler.read_input("debug> ")
                except KeyboardInterrupt:
                    self._execution_action = "quit"
                    break

            if user_input is None:
                # EOF (Ctrl-D): in interactive mode, stay alive until explicit quit.
                if not sys.stdin.isatty():
                    self._execution_action = "quit"
                    break
                self._formatter.show_info(
                    "EOF ignored. Use /continue to resume or /quit to exit."
                )
                continue

            if user_input == "":
                continue

            user_input = user_input.strip()
            if not user_input:
                continue

            # Handle bare quit/exit commands
            if user_input.lower() in ("quit", "exit", "q"):
                self._execution_action = "quit"
                break
            if user_input.lower() in ("rerun", "restart"):
                self._execution_action = "rerun"
                break

            # Check for slash commands
            if user_input.startswith("/"):
                result = self._command_handler.handle(user_input)
                if result is None:
                    # Execution control command was handled
                    break
                if result:
                    self._formatter._console.print(result)
                continue

            # Send to LLM
            try:
                augmented_input = self._augment_user_prompt(user_input)
                full_response = ""
                for chunk in self._llm.send_message_streaming(augmented_input):
                    full_response += chunk
                if full_response.strip():
                    self._formatter.show_model_response(full_response)

                # Check if LLM triggered an execution control action
                action = self._tool_executor.get_execution_action()
                if action:
                    self._execution_action = action
                    break
            except KeyboardInterrupt:
                self._formatter.show_info("Interrupted.")
            except Exception as e:
                self._formatter.show_error(f"LLM API error: {e}")

    def enter_post_run_repl(self, script_path: str, reason: str) -> None:
        """Keep Athena alive after target ends until user explicitly quits."""
        self._post_run_script_path = os.path.abspath(script_path)
        self._formatter.show_info(
            f"Target stopped ({reason}) for {self._post_run_script_path}."
        )
        self._formatter.show_info(
            "Athena is still running. Type /rerun to run again, or /quit to exit."
        )

        self._llm.set_system_prompt(
            "The target process has stopped. You can still analyze pasted code, "
            "inspect/edit files, and suggest fixes. Do not use execution-control tools. "
            f"When a filename is needed, use: {self._post_run_script_path}. "
            "If the user asks to verify with breakpoints/runtime behavior, call rerun_target first."
        )

        while not self._quitting:
            try:
                user_input = self._input_handler.read_input("athena(post)> ")
            except KeyboardInterrupt:
                self._formatter.show_info("Interrupted. Type /quit to exit.")
                continue

            if user_input is None:
                if not sys.stdin.isatty():
                    self._quitting = True
                    break
                self._formatter.show_info("EOF ignored. Type /quit to exit.")
                continue

            user_input = user_input.strip()
            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit", "q"):
                self._quitting = True
                break
            if user_input.lower() in ("rerun", "restart"):
                self._rerun_requested = True
                break

            if user_input.startswith("/"):
                result = self._command_handler.handle(user_input)
                if result is None:
                    # Execution-control slash commands don't apply post-run.
                    if self._execution_action == "quit":
                        self._quitting = True
                        break
                    if self._execution_action == "rerun":
                        self._rerun_requested = True
                        break
                    self._execution_action = None
                    self._formatter.show_info(
                        "Execution controls are unavailable: target is not running."
                    )
                    continue
                if result:
                    self._formatter._console.print(result)
                continue

            try:
                augmented_input = self._augment_user_prompt(user_input)
                full_response = ""
                for chunk in self._llm.send_message_streaming(augmented_input):
                    full_response += chunk
                if full_response.strip():
                    self._formatter.show_model_response(full_response)

                action = self._tool_executor.get_execution_action()
                if action == "quit":
                    self._quitting = True
                    break
                if action == "rerun":
                    self._rerun_requested = True
                    break
            except KeyboardInterrupt:
                self._formatter.show_info("Interrupted. Type /quit to exit.")
            except Exception as e:
                self._formatter.show_error(f"LLM API error: {e}")

    def _augment_user_prompt(self, user_input: str) -> str:
        """Steer generic prompts toward static-first analysis."""
        target_hint = (
            f" Target file: {self._post_run_script_path}."
            if self._post_run_script_path
            else ""
        )

        pasted = self._extract_pasted_code(user_input)
        if pasted is not None:
            return (
                "User pasted/highlighted code. First run static_analyze_snippet on the "
                "pasted code. Then use find_snippet_lines to map snippet to file lines. "
                "If matches are found, set targeted breakpoints near the suspicious lines "
                "and continue with runtime probing."
                f"{target_hint} If no current frame exists, pass filename explicitly.\n\n"
                "Pasted code:\n```python\n"
                f"{pasted}\n```"
            )

        if self._is_generic_prompt(user_input):
            if self._debugger.current_frame is not None:
                return (
                    "User request is generic/high-level bug finding. Follow this sequence:\n"
                    "1) Run static_analyze_file on the active file to identify likely hotspots.\n"
                    "2) Set targeted breakpoints on candidate lines.\n"
                    "3) Continue/step execution to hit those breakpoints.\n"
                    "4) Inspect runtime state (stack, locals, inspect_variable, "
                    "evaluate_expression) to verify or refute hypotheses.\n"
                    "5) Only then provide conclusions/fixes with observed evidence.\n\n"
                    "Do not stop after static analysis.\n\nUser request:\n"
                    f"{user_input}"
                )
            return (
                "User request is generic/high-level. Run static_analyze_file with explicit "
                "filename to identify candidate breakpoints."
                f"{target_hint}\nIf runtime is not currently paused, say so clearly and "
                "call rerun_target to verify with breakpoints in a live run.\n\nUser request:\n"
                f"{user_input}"
            )
        return user_input

    @staticmethod
    def _is_generic_prompt(user_input: str) -> bool:
        markers = [
            r"\.py\b",
            r":\d+",
            r"\bline\s+\d+\b",
            r"\bfunction\b",
            r"\bclass\b",
            r"\btraceback\b",
            r"\bstack\b",
            r"\bframe\b",
            r"\blocal(s)?\b",
            r"\bvariable\b",
            r"/[A-Za-z0-9_.-]+",
            r"\\",
            r"/",
        ]
        lowered = user_input.strip().lower()
        if len(lowered) > 240:
            return False
        return not any(re.search(pat, lowered) for pat in markers)

    @staticmethod
    def _extract_pasted_code(user_input: str) -> str | None:
        fenced = re.search(
            r"```(?:python)?\n(?P<code>[\s\S]+?)\n```",
            user_input,
            flags=re.IGNORECASE,
        )
        if fenced:
            code = fenced.group("code").strip("\n")
            return code if code else None

        if "\n" in user_input:
            lines = [ln for ln in user_input.splitlines() if ln.strip()]
            if len(lines) >= 3:
                score = sum(
                    1
                    for ln in lines
                    if any(tok in ln for tok in ("def ", "class ", "=", "return", "if ", "for ", "while "))
                )
                if score >= 2:
                    return "\n".join(lines)

        return None

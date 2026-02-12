"""DebugSession - main orchestrator for the debugging REPL."""

from __future__ import annotations

import bdb
import os
import re
import signal
import sys
import time
from datetime import UTC, datetime
from typing import Any

from athena import __version__ as athena_version
from athena.agent.llm_client import LLMClient
from athena.agent.system_prompt import PRE_RUN_SYSTEM_PROMPT, build_stop_context
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
from athena.state.models import (
    AgentStateSummary,
    DebuggerBreakpointState,
    DebuggerState,
    MemoryStateSummary,
    PerfCheckpointSummary,
    PerfStateSummary,
    RuntimeStateSummary,
    SCHEMA_VERSION,
    StateEnvelope,
    StateMeta,
)
from athena.state.store import StateStore
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
        self._tool_executor.bind_session_operations({
            "create_perf_checkpoint": self.create_perf_checkpoint,
            "list_perf_checkpoints": self.list_perf_checkpoints,
            "compare_perf_checkpoints": self.compare_perf_checkpoints,
            "save_checkpoint_state": self.save_checkpoint_state,
            "save_operation_checkpoint": self.save_operation_checkpoint,
            "list_checkpoint_states": self.list_checkpoint_states,
            "load_checkpoint_state": self.load_checkpoint_state,
            "queue_checkpoint_restore": self.queue_checkpoint_restore,
            "generate_perf_issue_report": self.generate_perf_issue_report,
        })

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
        self._active_script_path: str | None = None
        self._active_script_args: list[str] = []
        self._queued_restore_selector: str | None = None
        self._perf_checkpoints: list[PerfCheckpointSummary] = []
        self._last_perf_comparison: dict[str, Any] | None = None
        self._state_store = StateStore(
            state_dir=self._config.state_dir,
            max_auto_per_script=self._config.max_auto_states_per_script,
        )

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

    def set_active_target(self, script_path: str, script_args: list[str]) -> None:
        self._active_script_path = os.path.abspath(script_path)
        self._active_script_args = list(script_args)

    def list_states(self) -> list[dict[str, Any]]:
        script = self._active_script_path or self._post_run_script_path
        return self._state_store.list_states(script_path=script)

    def list_checkpoint_states(self) -> list[dict[str, Any]]:
        return self.list_states()

    def save_checkpoint_state(self, name: str | None = None) -> dict[str, Any]:
        return self.save_state(name=name, auto=False, reason="manual")

    def save_operation_checkpoint(
        self,
        operation: str,
        stage: str = "before",
    ) -> dict[str, Any]:
        """Best-effort rolling checkpoint for state-changing debugger operations."""
        op = (operation or "operation").strip() or "operation"
        phase = (stage or "before").strip() or "before"
        return self.save_state(auto=True, reason=f"op:{op}:{phase}")

    def load_checkpoint_state(
        self,
        selector: str = "latest",
        apply_now: bool = True,
    ) -> dict[str, Any]:
        if not apply_now:
            return self.queue_checkpoint_restore(selector=selector)
        if self._can_apply_restore_now():
            return self.load_state(selector=selector)
        queued = self.queue_checkpoint_restore(selector=selector)
        return {
            "status": "queued",
            "queued_selector": queued.get("queued_selector"),
            "message": (
                "Restore queued for next rerun because runtime is not paused "
                "and post-run context is unavailable."
            ),
        }

    def queue_checkpoint_restore(self, selector: str = "latest") -> dict[str, Any]:
        normalized = (selector or "latest").strip() or "latest"
        self._queued_restore_selector = normalized
        return {
            "status": "queued",
            "queued_selector": normalized,
            "message": f"Queued restore selector '{normalized}' for the next rerun.",
        }

    def consume_queued_restore_selector(self) -> str | None:
        selector = self._queued_restore_selector
        self._queued_restore_selector = None
        return selector

    def create_perf_checkpoint(self, label: str | None = None) -> dict[str, Any]:
        resolved_label = label or f"perf_{len(self._perf_checkpoints) + 1}"
        source_ctx = self._frame_inspector.get_source_context(context_lines=0)
        stop_file = source_ctx.get("filename") if "error" not in source_ctx else None
        stop_line = source_ctx.get("current_line") if "error" not in source_ctx else None
        stop_function = source_ctx.get("function") if "error" not in source_ctx else None

        memory = self._python_memory.get_current_memory()
        current_bytes = memory.get("current_bytes") if "error" not in memory else None
        peak_bytes = memory.get("peak_bytes") if "error" not in memory else None

        checkpoint = PerfCheckpointSummary(
            label=resolved_label,
            created_at=datetime.now(UTC).isoformat(),
            wall_time_ns=time.perf_counter_ns(),
            process_time_ns=time.process_time_ns(),
            stop_file=stop_file,
            stop_line=stop_line,
            stop_function=stop_function,
            python_current_bytes=current_bytes,
            python_peak_bytes=peak_bytes,
        )
        self._perf_checkpoints.append(checkpoint)
        return self._perf_checkpoint_to_dict(checkpoint)

    def list_perf_checkpoints(self) -> list[dict[str, Any]]:
        return [self._perf_checkpoint_to_dict(cp) for cp in self._perf_checkpoints]

    def compare_perf_checkpoints(
        self,
        label_a: str,
        label_b: str,
    ) -> dict[str, Any]:
        cp_a = self._find_perf_checkpoint(label_a)
        if cp_a is None:
            return {"error": f"Perf checkpoint not found: {label_a}"}
        cp_b = self._find_perf_checkpoint(label_b)
        if cp_b is None:
            return {"error": f"Perf checkpoint not found: {label_b}"}

        wall_delta_ns = cp_b.wall_time_ns - cp_a.wall_time_ns
        process_delta_ns = cp_b.process_time_ns - cp_a.process_time_ns

        memory_current_delta = None
        if cp_a.python_current_bytes is not None and cp_b.python_current_bytes is not None:
            memory_current_delta = cp_b.python_current_bytes - cp_a.python_current_bytes

        memory_peak_delta = None
        if cp_a.python_peak_bytes is not None and cp_b.python_peak_bytes is not None:
            memory_peak_delta = cp_b.python_peak_bytes - cp_a.python_peak_bytes

        result = {
            "label_a": cp_a.label,
            "label_b": cp_b.label,
            "wall_delta_ns": wall_delta_ns,
            "wall_delta_ms": wall_delta_ns / 1_000_000,
            "process_delta_ns": process_delta_ns,
            "process_delta_ms": process_delta_ns / 1_000_000,
            "memory_current_delta_bytes": memory_current_delta,
            "memory_peak_delta_bytes": memory_peak_delta,
            "checkpoint_a": self._perf_checkpoint_to_dict(cp_a),
            "checkpoint_b": self._perf_checkpoint_to_dict(cp_b),
        }
        self._last_perf_comparison = result
        return result

    def generate_perf_issue_report(
        self,
        title: str | None = None,
        baseline_label: str | None = None,
        current_label: str | None = None,
        selector: str = "latest",
        persist: bool = False,
    ) -> dict[str, Any]:
        script_path = self._active_script_path or self._post_run_script_path or "<script>"
        report_title = title or "Performance regression investigation"
        comparison: dict[str, Any] | None = None

        if baseline_label and current_label:
            comparison = self.compare_perf_checkpoints(baseline_label, current_label)
            if "error" in comparison:
                return comparison
        elif self._last_perf_comparison:
            comparison = self._last_perf_comparison
        elif len(self._perf_checkpoints) >= 2:
            older = self._perf_checkpoints[-2].label
            newer = self._perf_checkpoints[-1].label
            comparison = self.compare_perf_checkpoints(older, newer)

        focus_files = sorted(self._debugger._focus_files)
        focus_functions = sorted(self._debugger._focus_functions)
        breakpoints = self._breakpoint_manager.list_breakpoints()
        restore_selector = (selector or "latest").strip() or "latest"

        lines: list[str] = [
            f"# {report_title}",
            "",
            "## Symptom",
            "- Potential performance issue under investigation.",
            "",
            "## Reproduction",
            f"- Script: `{script_path}`",
            f"- Restore command: `athena {script_path} --restore {restore_selector}`",
            "",
        ]

        if comparison:
            lines.extend([
                "## Checkpoint Comparison",
                f"- Baseline: `{comparison.get('label_a')}`",
                f"- Current: `{comparison.get('label_b')}`",
                f"- Wall delta: `{comparison.get('wall_delta_ms')} ms`",
                f"- CPU process delta: `{comparison.get('process_delta_ms')} ms`",
                f"- Python current memory delta: `{comparison.get('memory_current_delta_bytes')} bytes`",
                f"- Python peak memory delta: `{comparison.get('memory_peak_delta_bytes')} bytes`",
                "",
            ])
        else:
            lines.extend([
                "## Checkpoint Comparison",
                "- No perf checkpoint comparison available yet.",
                "",
            ])

        lines.extend([
            "## Debugger Context",
            f"- Breakpoints: `{len(breakpoints)}`",
            f"- Focus files: `{', '.join(focus_files) if focus_files else '(none)'}`",
            f"- Focus functions: `{', '.join(focus_functions) if focus_functions else '(none)'}`",
            "",
            "## Next Verification Steps",
            "- Restore baseline checkpoint state if needed.",
            "- Rerun and collect fresh perf checkpoints around target code path.",
            "- Compare wall/CPU/memory deltas to confirm regression direction.",
            "",
        ])

        report_text = "\n".join(lines)
        output: dict[str, Any] = {"title": report_title, "report": report_text}

        if persist:
            issue_dir = os.path.join(".athena", "issues")
            os.makedirs(issue_dir, exist_ok=True)
            filename = datetime.now(UTC).strftime("perf_%Y%m%dT%H%M%S%fZ.md")
            path = os.path.join(issue_dir, filename)
            with open(path, "w") as f:
                f.write(report_text)
            output["path"] = path

        return output

    def _can_apply_restore_now(self) -> bool:
        return self._debugger.current_frame is not None or self._post_run_script_path is not None

    def _find_perf_checkpoint(self, label: str) -> PerfCheckpointSummary | None:
        for checkpoint in self._perf_checkpoints:
            if checkpoint.label == label:
                return checkpoint
        return None

    @staticmethod
    def _perf_checkpoint_to_dict(checkpoint: PerfCheckpointSummary) -> dict[str, Any]:
        return {
            "label": checkpoint.label,
            "created_at": checkpoint.created_at,
            "wall_time_ns": checkpoint.wall_time_ns,
            "process_time_ns": checkpoint.process_time_ns,
            "stop_file": checkpoint.stop_file,
            "stop_line": checkpoint.stop_line,
            "stop_function": checkpoint.stop_function,
            "python_current_bytes": checkpoint.python_current_bytes,
            "python_peak_bytes": checkpoint.python_peak_bytes,
        }

    def capture_state_snapshot(self, reason: str) -> StateEnvelope:
        script_path = self._active_script_path or self._post_run_script_path or ""
        script_hash = self._state_store.compute_script_hash(script_path) if script_path else ""

        stop_file = None
        stop_line = None
        stop_function = None
        stack: list[dict[str, Any]] = []
        locals_summary: dict[str, dict[str, str]] = {}
        source_ctx = self._frame_inspector.get_source_context(context_lines=0)
        if "error" not in source_ctx:
            stop_file = source_ctx.get("filename")
            stop_line = source_ctx.get("current_line")
            stop_function = source_ctx.get("function")
            stack = self._frame_inspector.get_stack_frames()
            locals_summary = self._frame_inspector.get_all_locals()

        breakpoints = []
        for bp in self._breakpoint_manager.list_breakpoints():
            line_text = self._source_provider.get_line_text(bp["file"], bp["line"])
            breakpoints.append(DebuggerBreakpointState(
                file=bp["file"],
                line=bp["line"],
                condition=bp.get("condition"),
                enabled=bool(bp.get("enabled", True)),
                temporary=bool(bp.get("temporary", False)),
                hits=int(bp.get("hits", 0)),
                ignore_count=int(bp.get("ignore_count", 0)),
                context_snippet=line_text or "",
            ))

        mem = self._python_memory.get_current_memory()
        py_labels = self._python_memory.get_snapshot_labels()
        cuda_labels = self._cuda_memory.get_snapshot_labels()
        leak_summary = None
        if len(py_labels) >= 2 or len(cuda_labels) >= 2:
            try:
                leak_summary = self._leak_detector.analyze()
            except Exception:
                leak_summary = None

        return StateEnvelope(
            kind="auto",
            reason=reason,
            name=None,
            meta=StateMeta(
                schema_version=SCHEMA_VERSION,
                created_at=datetime.now(UTC).isoformat(),
                athena_version=athena_version,
                python_version=sys.version,
                cwd=os.getcwd(),
                script_path=script_path,
                script_args=list(self._active_script_args),
                model=self._config.model,
                script_hash=script_hash,
            ),
            debugger=DebuggerState(
                breakpoints=breakpoints,
                watches=self._watch_manager.get_watch_list(),
                focus_files=sorted(self._debugger._focus_files),
                focus_functions=sorted(self._debugger._focus_functions),
            ),
            runtime=RuntimeStateSummary(
                stop_reason=reason,
                stop_file=stop_file,
                stop_line=stop_line,
                stop_function=stop_function,
                stack=stack,
                locals_summary=locals_summary,
            ),
            memory=MemoryStateSummary(
                python_current_human=mem.get("current_human") if "error" not in mem else None,
                python_peak_human=mem.get("peak_human") if "error" not in mem else None,
                python_snapshot_labels=py_labels,
                cuda_snapshot_labels=cuda_labels,
                leak_summary=leak_summary,
            ),
            agent=AgentStateSummary(
                conversation_summary=self._llm.get_conversation_summary(),
            ),
            perf=PerfStateSummary(
                checkpoints=[
                    PerfCheckpointSummary(
                        label=cp.label,
                        created_at=cp.created_at,
                        wall_time_ns=cp.wall_time_ns,
                        process_time_ns=cp.process_time_ns,
                        stop_file=cp.stop_file,
                        stop_line=cp.stop_line,
                        stop_function=cp.stop_function,
                        python_current_bytes=cp.python_current_bytes,
                        python_peak_bytes=cp.python_peak_bytes,
                    )
                    for cp in self._perf_checkpoints
                ],
                last_comparison=self._last_perf_comparison,
            ),
        )

    def save_state(
        self,
        name: str | None = None,
        auto: bool = False,
        reason: str = "manual",
    ) -> dict[str, Any]:
        if auto and not self._config.auto_save_state:
            return {"status": "disabled"}

        try:
            state = self.capture_state_snapshot(reason=reason)
            state.kind = "auto" if auto else "manual"
            state.name = None if auto else (name or "manual")
            if auto:
                path = self._state_store.save_auto(state)
                if state.meta.script_path:
                    self._state_store.prune_auto(
                        state.meta.script_path,
                        keep=self._config.max_auto_states_per_script,
                    )
            else:
                path = self._state_store.save_manual(name or "manual", state)
            return {"status": "ok", "path": path}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    def load_state(
        self,
        selector: str = "latest",
        expected_script_path: str | None = None,
    ) -> dict[str, Any]:
        script_path = expected_script_path or self._active_script_path or self._post_run_script_path
        if not script_path:
            return {"error": "No script path available for compatibility checks"}

        loaded = self._state_store.load(
            selector=selector,
            script_path=script_path,
            model=self._config.model,
        )
        if "error" in loaded:
            return loaded

        state: StateEnvelope = loaded["state"]
        apply_report = self._apply_debugger_state(
            state.debugger,
            code_drift=bool(loaded.get("code_drift")),
        )
        conv_summary = state.agent.conversation_summary
        if conv_summary:
            self._llm.set_restored_summary(conv_summary)
        self._perf_checkpoints = list(state.perf.checkpoints)
        self._last_perf_comparison = state.perf.last_comparison
        self._active_script_path = state.meta.script_path or self._active_script_path
        self._active_script_args = list(state.meta.script_args or self._active_script_args)

        return {
            "status": "ok",
            "path": loaded.get("path"),
            "warnings": loaded.get("warnings", []) + apply_report.get("warnings", []),
            "applied": apply_report,
        }

    def _apply_debugger_state(
        self,
        debugger_state: DebuggerState,
        code_drift: bool = False,
    ) -> dict[str, Any]:
        warnings: list[str] = []
        applied_breakpoints = 0
        skipped_breakpoints = 0

        self._breakpoint_manager.clear_all_breakpoints()
        self._watch_manager.clear_all()
        self._debugger.clear_focus()

        for watch in debugger_state.watches:
            self._watch_manager.add_watch(watch)

        if debugger_state.focus_files:
            self._debugger.set_focus_files(debugger_state.focus_files)
        if debugger_state.focus_functions:
            self._debugger.set_focus_functions(debugger_state.focus_functions)

        for bp in debugger_state.breakpoints:
            target_line = bp.line
            if code_drift:
                remapped = self._source_provider.remap_line_by_snippet(
                    filename=bp.file,
                    preferred_line=bp.line,
                    snippet=bp.context_snippet,
                )
                if remapped is not None:
                    target_line = remapped
                else:
                    skipped_breakpoints += 1
                    warnings.append(
                        f"Could not remap breakpoint {bp.file}:{bp.line}; skipped."
                    )
                    continue

            result = self._breakpoint_manager.set_breakpoint(
                filename=bp.file,
                lineno=target_line,
                condition=bp.condition,
            )
            if "error" in result:
                skipped_breakpoints += 1
                warnings.append(
                    f"Failed restoring breakpoint {bp.file}:{target_line}: {result['error']}"
                )
                continue

            self._breakpoint_manager.apply_breakpoint_runtime_state(
                filename=bp.file,
                lineno=target_line,
                enabled=bp.enabled,
                hits=bp.hits,
                ignore_count=bp.ignore_count,
            )
            applied_breakpoints += 1

        return {
            "applied_breakpoints": applied_breakpoints,
            "skipped_breakpoints": skipped_breakpoints,
            "applied_watches": len(debugger_state.watches),
            "warnings": warnings,
        }

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

        auto_save = self.save_state(auto=True, reason=f"stop:{reason}")
        if "error" in auto_save:
            self._formatter.show_info(f"State auto-save warning: {auto_save['error']}")

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
            auto_save = self.save_state(auto=True, reason="quit")
            if "error" in auto_save:
                self._formatter.show_info(f"State auto-save warning: {auto_save['error']}")
            self._quitting = True
            self._formatter.show_info("Quitting Athena.")
            raise bdb.BdbQuit
        elif action == "rerun":
            auto_save = self.save_state(auto=True, reason="rerun")
            if "error" in auto_save:
                self._formatter.show_info(f"State auto-save warning: {auto_save['error']}")
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

    def enter_pre_run_repl(self, script_path: str) -> bool:
        """GDB-style pre-run REPL: set breakpoints, inspect source, then /run.

        Returns True if the user wants to run the script, False if they quit.
        """
        self._formatter.show_banner()
        self._formatter.show_info(
            f"Target: {script_path} (not yet running)"
        )
        self._formatter.show_info(
            "Set breakpoints, inspect source, then type /run to start execution."
        )

        # Configure LLM for pre-run: restricted tool set, pre-run prompt
        self._llm.set_system_prompt(
            PRE_RUN_SYSTEM_PROMPT.format(script_path=script_path)
        )
        self._llm.set_tool_schema_override(
            self._tool_executor.get_pre_run_tool_schemas()
        )

        self._execution_action = None
        while self._execution_action is None:
            try:
                user_input = self._input_handler.read_input("athena> ")
            except KeyboardInterrupt:
                self._formatter.show_info(
                    "Press Ctrl-C again to quit, or type a command."
                )
                try:
                    user_input = self._input_handler.read_input("athena> ")
                except KeyboardInterrupt:
                    self._execution_action = "quit"
                    break

            if user_input is None:
                if not sys.stdin.isatty():
                    self._execution_action = "quit"
                    break
                self._formatter.show_info(
                    "EOF ignored. Use /run to start or /quit to exit."
                )
                continue

            user_input = user_input.strip()
            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit", "q"):
                self._execution_action = "quit"
                break
            if user_input.lower() in ("run",):
                self._execution_action = "run"
                break

            if user_input.startswith("/"):
                result = self._command_handler.handle(user_input)
                if result is None:
                    # Execution control command was handled
                    break
                if result:
                    self._formatter._console.print(result)
                continue

            # Send to LLM (with pre-run tool set)
            try:
                full_response = ""
                for chunk in self._llm.send_message_streaming(user_input):
                    full_response += chunk
                if full_response.strip():
                    self._formatter.show_model_response(full_response)

                action = self._tool_executor.get_execution_action()
                if action:
                    self._execution_action = action
                    break
            except KeyboardInterrupt:
                self._formatter.show_info("Interrupted.")
            except Exception as e:
                self._formatter.show_error(f"LLM API error: {e}")

        # Restore full tool set for runtime
        self._llm.set_tool_schema_override(None)

        if self._execution_action == "run":
            self._execution_action = None
            return True
        if self._execution_action == "quit":
            self._quitting = True
            return False

        # Unexpected action in pre-run (e.g. step/continue) — treat as run
        self._execution_action = None
        return True

    def enter_post_run_repl(self, script_path: str, reason: str) -> None:
        """Keep Athena alive after target ends until user explicitly quits."""
        self._post_run_script_path = os.path.abspath(script_path)
        auto_save = self.save_state(auto=True, reason=f"post_run:{reason}")
        if "error" in auto_save:
            self._formatter.show_info(f"State auto-save warning: {auto_save['error']}")
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
            "Checkpoint tools are available here; load/queue restore state as needed. "
            "If the user asks to verify with breakpoints/runtime behavior, call rerun_target first."
        )

        while not self._quitting:
            try:
                user_input = self._input_handler.read_input("athena> ")
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
                auto_save = self.save_state(auto=True, reason="post_run_quit")
                if "error" in auto_save:
                    self._formatter.show_info(f"State auto-save warning: {auto_save['error']}")
                self._quitting = True
                break
            if user_input.lower() in ("rerun", "restart"):
                auto_save = self.save_state(auto=True, reason="post_run_rerun")
                if "error" in auto_save:
                    self._formatter.show_info(f"State auto-save warning: {auto_save['error']}")
                self._rerun_requested = True
                break

            if user_input.startswith("/"):
                result = self._command_handler.handle(user_input)
                if result is None:
                    # Execution-control slash commands don't apply post-run.
                    if self._execution_action == "quit":
                        auto_save = self.save_state(auto=True, reason="post_run_quit")
                        if "error" in auto_save:
                            self._formatter.show_info(f"State auto-save warning: {auto_save['error']}")
                        self._quitting = True
                        break
                    if self._execution_action == "rerun":
                        auto_save = self.save_state(auto=True, reason="post_run_rerun")
                        if "error" in auto_save:
                            self._formatter.show_info(f"State auto-save warning: {auto_save['error']}")
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
                    auto_save = self.save_state(auto=True, reason="post_run_quit")
                    if "error" in auto_save:
                        self._formatter.show_info(f"State auto-save warning: {auto_save['error']}")
                    self._quitting = True
                    break
                if action == "rerun":
                    auto_save = self.save_state(auto=True, reason="post_run_rerun")
                    if "error" in auto_save:
                        self._formatter.show_info(f"State auto-save warning: {auto_save['error']}")
                    self._rerun_requested = True
                    break
            except KeyboardInterrupt:
                self._formatter.show_info("Interrupted. Type /quit to exit.")
            except Exception as e:
                self._formatter.show_error(f"LLM API error: {e}")

    def _augment_user_prompt(self, user_input: str) -> str:
        """Steer prompts toward reliable runtime workflows."""
        target_hint = (
            f" Target file: {self._post_run_script_path}."
            if self._post_run_script_path
            else ""
        )

        if self._config.perf_mode or self._is_perf_prompt(user_input):
            if self._can_apply_restore_now():
                return (
                    "User requested performance debugging. Follow this checkpoint workflow:\n"
                    "1) Call list_checkpoint_states and list_perf_checkpoints to inspect baselines.\n"
                    "2) If baseline context is missing, call load_checkpoint_state "
                    "(apply_now=true) or queue_checkpoint_restore when needed.\n"
                    "3) Set focused breakpoints around suspected hot paths.\n"
                    "4) Create perf checkpoints before and after the target code path.\n"
                    "5) Compare checkpoints with compare_perf_checkpoints and report wall/CPU/memory deltas.\n"
                    "6) Use runtime verification before suggesting fixes.\n\n"
                    f"{target_hint}\nUser request:\n{user_input}"
                )
            return (
                "User requested performance debugging but runtime is not paused.\n"
                "1) Inspect checkpoint states and queue restore if needed (queue_checkpoint_restore).\n"
                "2) Call rerun_target to re-enter live runtime debugging.\n"
                "3) Collect and compare perf checkpoints before concluding.\n\n"
                f"{target_hint}\nUser request:\n{user_input}"
            )

        pasted = self._extract_pasted_code(user_input)
        if pasted is not None:
            return (
                "User pasted/highlighted code. First run static_analyze_snippet on the "
                "pasted code. Then use find_snippet_lines to map snippet to file lines. "
                "If matches are found, set targeted breakpoints near the suspicious lines "
                "and use runtime tools to verify. State your hypothesis before each tool "
                "call, then analyze results before proceeding. Do not conclude without "
                "observed runtime evidence."
                f"{target_hint} If no current frame exists, pass filename explicitly.\n\n"
                "Pasted code:\n```python\n"
                f"{pasted}\n```"
            )

        if self._is_generic_prompt(user_input):
            if self._debugger.current_frame is not None:
                return (
                    "User request is generic/high-level bug finding. Investigate iteratively:\n"
                    "- Start with static_analyze_file to form initial hypotheses.\n"
                    "- For EACH hypothesis, state it explicitly, then use a runtime tool "
                    "(inspect_variable, evaluate_expression, get_all_locals) to gather evidence.\n"
                    "- After each tool result, analyze what the evidence shows before deciding "
                    "your next action.\n"
                    "- Set breakpoints and continue/step to reach code paths you need to verify.\n"
                    "- Do NOT conclude until you have runtime evidence. Every finding must cite "
                    "a specific observed value, not a guess from reading code.\n\n"
                    f"User request:\n{user_input}"
                )
            return (
                "User request is generic/high-level. Run static_analyze_file with explicit "
                "filename to form hypotheses."
                f"{target_hint}\nRuntime is not currently paused — you cannot verify yet. "
                "Say so explicitly. Set breakpoints at candidate lines and call rerun_target "
                "to start a live run where you can gather runtime evidence.\n\n"
                f"User request:\n{user_input}"
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
    def _is_perf_prompt(user_input: str) -> bool:
        lowered = user_input.lower()
        keywords = (
            "perf",
            "performance",
            "latency",
            "throughput",
            "slow",
            "regression",
            "hot path",
            "cpu",
            "benchmark",
            "timing",
        )
        return any(keyword in lowered for keyword in keywords)

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

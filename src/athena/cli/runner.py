"""Script runner - executes user scripts under the debugger."""

from __future__ import annotations

import os
import sys

from athena.repl.session import DebugSession
from athena.utils.config import DebugConfig


class ScriptRunner:
    """Runs a user's Python script under the debugging agent.

    No code changes needed to the target script. The debugger wraps
    execution using bdb.Bdb.run() which installs trace functions
    transparently.
    """

    def run(
        self,
        script_path: str,
        script_args: list[str] | None = None,
        break_on_entry: bool = False,
        break_on_exception: bool = False,
        initial_breakpoints: list[str] | None = None,
        injected_breaks: list[str] | None = None,
        focus_files: list[str] | None = None,
        focus_functions: list[str] | None = None,
        model: str | None = None,
        trace_memory: bool = False,
    ) -> None:
        script_path = os.path.abspath(script_path)
        if not os.path.isfile(script_path):
            print(f"Error: File not found: {script_path}", file=sys.stderr)
            sys.exit(1)

        while True:
            # Build config
            config = DebugConfig.from_env()
            if model:
                config.model = model

            # Create session
            session = DebugSession(config=config)
            self._bind_session_for_injected_breaks(session)

            # Configure
            if break_on_exception:
                session.debugger._break_on_exception = True

            if trace_memory:
                session.python_memory.start_tracing()

            # Set focus
            if focus_files:
                session.debugger.set_focus_files(focus_files)
            if focus_functions:
                session.debugger.set_focus_functions(focus_functions)

            # Set initial breakpoints
            if initial_breakpoints:
                for bp_spec in initial_breakpoints:
                    result = session.breakpoint_manager.add_from_spec(bp_spec)
                    if "error" in result:
                        print(f"Warning: Could not set breakpoint {bp_spec}: {result['error']}",
                              file=sys.stderr)

            injected_lines = self._resolve_injected_break_lines(
                script_path=script_path,
                specs=injected_breaks or [],
            )

            # If not breaking on entry, skip the first debugger stop callback.
            if not break_on_entry:
                session.debugger._skip_first_stop = True

            # Run the script
            stop_reason = "completed"
            try:
                session.debugger.run_script(
                    script_path,
                    script_args,
                    injected_break_lines=injected_lines,
                )
            except bdb_module.BdbQuit:
                stop_reason = "user_quit"
            except SystemExit:
                stop_reason = "system_exit"
            except Exception as e:
                stop_reason = f"exception: {type(e).__name__}"
                print(f"\nUnhandled exception in target script: {type(e).__name__}: {e}",
                      file=sys.stderr)
                if break_on_exception:
                    # The debugger should have caught this
                    pass
                else:
                    import traceback
                    traceback.print_exc()

            if (
                not session.is_quitting
                and stop_reason != "user_quit"
                and sys.stdin.isatty()
                and sys.stdout.isatty()
            ):
                session.enter_post_run_repl(script_path, stop_reason)

            if session.is_rerun_requested:
                print("\n[athena] Rerunning target script...\n")
                continue
            break

    @staticmethod
    def _bind_session_for_injected_breaks(session: DebugSession) -> None:
        """Bind athena.breakpoint() to this live session."""
        import athena

        athena._session = session

    @staticmethod
    def _resolve_injected_break_lines(script_path: str, specs: list[str]) -> list[int]:
        lines: list[int] = []
        script_abs = os.path.abspath(script_path)
        script_base = os.path.basename(script_abs)

        for spec in specs:
            if ":" not in spec:
                print(f"Warning: Invalid --inject-break spec (expected FILE:LINE): {spec}", file=sys.stderr)
                continue

            file_part, line_part = spec.rsplit(":", 1)
            try:
                lineno = int(line_part)
            except ValueError:
                print(f"Warning: Invalid --inject-break line number: {line_part}", file=sys.stderr)
                continue

            target_abs = os.path.abspath(file_part)
            if target_abs != script_abs and os.path.basename(target_abs) != script_base:
                print(
                    f"Warning: --inject-break currently supports only the entry script; skipping {spec}",
                    file=sys.stderr,
                )
                continue

            if lineno < 1:
                print(f"Warning: --inject-break line must be >= 1: {spec}", file=sys.stderr)
                continue
            lines.append(lineno)

        return sorted(set(lines))


# Import bdb for BdbQuit
import bdb as bdb_module

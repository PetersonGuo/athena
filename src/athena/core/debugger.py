"""Core debugger engine built on bdb.Bdb with selective tracing support.

This debugger requires NO code changes to the target script. It runs the
target via bdb.run() and uses selective frame filtering for low overhead.
On Python 3.12+, sys.monitoring can be used for even lower overhead.
"""

from __future__ import annotations

import bdb
import linecache
import os
import re
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from types import FrameType

    from athena.repl.session import DebugSession


class DebuggerCore(bdb.Bdb):
    """Central debugger engine. Subclasses bdb.Bdb for trace management.

    When a stop condition is met (breakpoint hit, step completed, exception),
    bdb calls user_line/user_return/user_exception. We override these to
    signal the DebugSession which enters the interactive REPL.

    The target code runs in the main thread. When paused, that thread blocks
    in the REPL loop (same approach as pdb).
    """

    def __init__(self, skip: list[str] | None = None):
        skip_modules = skip or [
            "athena.*",
            "openai.*",
            "httpx.*",
            "rich.*",
            "prompt_toolkit.*",
        ]
        super().__init__(skip=skip_modules)
        self._session: DebugSession | None = None
        self._current_frame: FrameType | None = None
        self._current_traceback: Any = None
        self._stop_reason: str = ""
        self._break_on_exception: bool = False
        self._skip_first_stop: bool = False

        # Focus tracing: only stop in files/functions the user cares about.
        # When empty, stops everywhere (like normal pdb).
        self._focus_files: set[str] = set()
        self._focus_functions: set[str] = set()

    def attach_session(self, session: DebugSession) -> None:
        self._session = session

    def set_focus_files(self, files: list[str]) -> None:
        """Only stop in these files (resolved to absolute paths)."""
        self._focus_files = {os.path.abspath(f) for f in files}

    def set_focus_functions(self, functions: list[str]) -> None:
        """Only stop in these function names."""
        self._focus_functions = set(functions)

    def add_focus_file(self, filepath: str) -> None:
        self._focus_files.add(os.path.abspath(filepath))

    def add_focus_function(self, funcname: str) -> None:
        self._focus_functions.add(funcname)

    def clear_focus(self) -> None:
        self._focus_files.clear()
        self._focus_functions.clear()

    def _should_stop_in_frame(self, frame: FrameType) -> bool:
        """Check if this frame matches our focus filters.

        If no focus is set, always stop. If focus is set, only stop
        if the frame's file or function matches.
        """
        if not self._focus_files and not self._focus_functions:
            return True

        filename = self.canonic(frame.f_code.co_filename)
        funcname = frame.f_code.co_name

        if self._focus_files:
            if filename in self._focus_files:
                return True
            # Also match by basename for convenience
            basename = os.path.basename(filename)
            if any(os.path.basename(f) == basename for f in self._focus_files):
                return True

        if self._focus_functions and funcname in self._focus_functions:
            return True

        return False

    def user_line(self, frame: FrameType) -> None:
        """Called when we stop at a line (step, breakpoint, etc.)."""
        if self._skip_first_stop:
            self._skip_first_stop = False
            self.set_continue()
            return

        if not self._should_stop_in_frame(frame):
            self.set_continue()
            return

        self._current_frame = frame
        self._stop_reason = "line"
        if self._session:
            self._session.on_debugger_stop(frame, None, self._stop_reason)

    def user_return(self, frame: FrameType, return_value: Any) -> None:
        """Called when a return is about to happen in a watched frame."""
        if not self._should_stop_in_frame(frame):
            return

        self._current_frame = frame
        self._stop_reason = "return"
        if self._session:
            self._session.on_debugger_stop(frame, None, self._stop_reason)

    def user_exception(
        self,
        frame: FrameType,
        exc_info: tuple[type, BaseException, Any],
    ) -> None:
        """Called when an exception occurs in a watched frame."""
        if not self._break_on_exception:
            return
        self._current_frame = frame
        self._stop_reason = "exception"
        if self._session:
            self._session.on_debugger_stop(frame, exc_info, self._stop_reason)

    # --- Execution control (called by tool executor) ---

    def do_step(self) -> None:
        """Step into the next call."""
        self.set_step()

    def do_next(self) -> None:
        """Step over to the next line in the current frame."""
        if self._current_frame:
            self.set_next(self._current_frame)

    def do_return(self) -> None:
        """Continue until the current function returns."""
        if self._current_frame:
            self.set_return(self._current_frame)

    def do_continue(self) -> None:
        """Continue execution until next breakpoint."""
        self.set_continue()

    def do_until(self, lineno: int) -> None:
        """Continue until a line >= lineno in the current frame."""
        if self._current_frame:
            self.set_until(self._current_frame, lineno)

    @property
    def current_frame(self) -> FrameType | None:
        return self._current_frame

    def run_script(
        self,
        script_path: str,
        script_args: list[str] | None = None,
        injected_break_lines: list[int] | None = None,
    ) -> None:
        """Run a script under the debugger. No code changes needed."""
        script_path = os.path.abspath(script_path)

        # Set up sys.argv
        sys.argv = [script_path] + (script_args or [])

        # Read and compile the script
        with open(script_path) as f:
            source = f.read()
        if injected_break_lines:
            source = self._inject_runtime_breaks(source, injected_break_lines)

        # Ensure linecache can find the source
        linecache.cache[script_path] = (
            len(source),
            None,
            source.splitlines(True),
            script_path,
        )

        code = compile(source, script_path, "exec")

        # Set up the script's global namespace
        script_globals: dict[str, Any] = {
            "__name__": "__main__",
            "__file__": script_path,
            "__builtins__": __builtins__,
        }

        # Add script directory to sys.path
        script_dir = os.path.dirname(script_path)
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)

        # Run under debugger
        self.run(code, script_globals)

    @staticmethod
    def _inject_runtime_breaks(source: str, break_lines: list[int]) -> str:
        """Inject athena.breakpoint() calls before selected line numbers.

        This does not modify files on disk. It only rewrites the source text
        compiled for this process.
        """
        lines = source.splitlines(keepends=True)
        if not lines:
            return source

        for line_no in sorted(set(break_lines), reverse=True):
            if line_no < 1 or line_no > len(lines):
                continue
            target = lines[line_no - 1]
            indent = re.match(r"\s*", target).group(0) if target else ""
            injection = (
                f'{indent}__import__("athena").breakpoint()  # athena:injected-break\n'
            )
            lines.insert(line_no - 1, injection)

        return "".join(lines)

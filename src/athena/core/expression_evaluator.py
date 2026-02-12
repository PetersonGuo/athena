"""Safe expression evaluation in target frame contexts."""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stderr, redirect_stdout
from typing import TYPE_CHECKING, Any

from athena.utils.safe_repr import safe_repr, safe_type_name

if TYPE_CHECKING:
    from athena.core.debugger import DebuggerCore


class ExpressionEvaluator:
    """Evaluates Python expressions/statements in target frame contexts.

    Uses compile() + eval() for expressions and compile() + exec() for
    statements. Evaluation happens in the target frame's locals/globals.
    """

    def __init__(self, debugger: DebuggerCore):
        self._debugger = debugger

    def evaluate(
        self,
        expression: str,
        frame_index: int | None = None,
        timeout_seconds: float = 10.0,
    ) -> dict[str, Any]:
        """Evaluate an expression in the target frame.

        Returns: {expression, result, type} or {expression, error}
        Also captures any stdout/stderr produced during evaluation.
        """
        frame = self._get_frame(frame_index)
        if frame is None:
            return {"expression": expression, "error": "No active frame"}

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            # Try as expression first (has a return value)
            code = compile(expression, "<debug-eval>", "eval")
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                result = eval(code, frame.f_globals, frame.f_locals)  # noqa: S307

            output: dict[str, Any] = {
                "expression": expression,
                "result": safe_repr(result),
                "type": safe_type_name(result),
            }
        except SyntaxError:
            # Fall back to statement execution (assignments, etc.)
            try:
                code = compile(expression, "<debug-eval>", "exec")
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    exec(code, frame.f_globals, frame.f_locals)  # noqa: S102

                output = {
                    "expression": expression,
                    "result": "<statement executed>",
                    "type": "None",
                }
            except Exception as e:
                output = {
                    "expression": expression,
                    "error": f"{type(e).__name__}: {e}",
                }
        except Exception as e:
            output = {
                "expression": expression,
                "error": f"{type(e).__name__}: {e}",
            }

        # Attach captured output if any
        stdout_val = stdout_capture.getvalue()
        stderr_val = stderr_capture.getvalue()
        if stdout_val:
            output["stdout"] = stdout_val[:2000]
        if stderr_val:
            output["stderr"] = stderr_val[:2000]

        return output

    def _get_frame(self, frame_index: int | None = None):
        current = self._debugger.current_frame
        if current is None:
            return None
        if frame_index is None:
            return current
        stack, _ = self._debugger.get_stack(current, None)
        if 0 <= frame_index < len(stack):
            return stack[frame_index][0]
        return None

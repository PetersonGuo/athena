"""Frame introspection: variables, stack, closures, source context."""

from __future__ import annotations

import linecache
import sys
from typing import TYPE_CHECKING, Any

from athena.utils.safe_repr import safe_repr, safe_type_name

if TYPE_CHECKING:
    from types import FrameType

    from athena.core.debugger import DebuggerCore


class FrameInspector:
    """Inspects Python frame objects to extract variable state and context.

    Uses frame.f_locals, f_globals, f_code, and the inspect module
    to extract information without side effects.
    """

    def __init__(self, debugger: DebuggerCore):
        self._debugger = debugger

    def get_stack_frames(self) -> list[dict[str, Any]]:
        """Return the full call stack as a list of frame descriptors."""
        frame = self._debugger.current_frame
        if frame is None:
            return []

        try:
            stack, cur_index = self._debugger.get_stack(frame, None)
        except AttributeError:
            # botframe not set (not running via bdb.run). Walk frames manually.
            stack = []
            f = frame
            while f is not None:
                stack.append((f, f.f_lineno))
                f = f.f_back
            stack.reverse()
            cur_index = len(stack) - 1

        result = []
        for i, (f, lineno) in enumerate(stack):
            result.append({
                "index": i,
                "filename": f.f_code.co_filename,
                "lineno": lineno,
                "function": f.f_code.co_name,
                "is_current": i == cur_index,
                "locals_names": list(f.f_locals.keys()),
            })
        return result

    def get_variable(
        self,
        name: str,
        frame_index: int | None = None,
        max_depth: int = 3,
    ) -> dict[str, Any]:
        """Inspect a specific variable by name in the given frame."""
        frame = self._get_frame(frame_index)
        if frame is None:
            return {"name": name, "error": "No active frame"}

        for namespace_name, namespace in [
            ("local", frame.f_locals),
            ("global", frame.f_globals),
            ("builtin", frame.f_builtins),
        ]:
            if name in namespace:
                obj = namespace[name]
                result: dict[str, Any] = {
                    "name": name,
                    "found_in": namespace_name,
                    "type": safe_type_name(obj),
                    "repr": safe_repr(obj, max_depth),
                    "id": id(obj),
                }
                try:
                    if hasattr(obj, "__len__"):
                        result["length"] = len(obj)
                except Exception:
                    pass
                return result

        return {"name": name, "error": f"Variable {name!r} not found in frame"}

    def get_all_locals(self, frame_index: int | None = None) -> dict[str, dict[str, str]]:
        """Return all local variables in the specified frame."""
        frame = self._get_frame(frame_index)
        if frame is None:
            return {}

        result = {}
        for name, val in frame.f_locals.items():
            result[name] = {
                "type": safe_type_name(val),
                "repr": safe_repr(val, max_depth=2),
            }
        return result

    def get_closure_vars(self, frame_index: int | None = None) -> dict[str, str]:
        """Inspect closure (free) variables for the function in the specified frame."""
        frame = self._get_frame(frame_index)
        if frame is None:
            return {}

        result = {}
        for name in frame.f_code.co_freevars:
            if name in frame.f_locals:
                result[name] = safe_repr(frame.f_locals[name])
        return result

    def get_source_context(
        self,
        frame_index: int | None = None,
        context_lines: int = 10,
    ) -> dict[str, Any]:
        """Get source code around the current execution point."""
        frame = self._get_frame(frame_index)
        if frame is None:
            return {"error": "No active frame"}

        filename = frame.f_code.co_filename
        current_line = frame.f_lineno
        start = max(1, current_line - context_lines)
        end = current_line + context_lines + 1

        lines: dict[int, str] = {}
        for i in range(start, end):
            line = linecache.getline(filename, i, frame.f_globals)
            if line:
                lines[i] = line.rstrip("\n")

        return {
            "filename": filename,
            "current_line": current_line,
            "function": frame.f_code.co_name,
            "lines": lines,
        }

    def _get_frame(self, frame_index: int | None = None) -> FrameType | None:
        """Get the frame at the given stack index.

        If frame_index is None, returns the current (innermost) frame.
        Index 0 is the outermost frame, highest index is the innermost.
        """
        current = self._debugger.current_frame
        if current is None:
            return None

        if frame_index is None:
            return current

        try:
            stack, _ = self._debugger.get_stack(current, None)
        except AttributeError:
            stack = []
            f = current
            while f is not None:
                stack.append((f, f.f_lineno))
                f = f.f_back
            stack.reverse()

        if 0 <= frame_index < len(stack):
            return stack[frame_index][0]
        return None

    def get_current_frame_index(self) -> int:
        """Get the index of the current frame in the stack."""
        current = self._debugger.current_frame
        if current is None:
            return 0
        _, cur_index = self._debugger.get_stack(current, None)
        return cur_index

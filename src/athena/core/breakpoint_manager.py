"""Breakpoint management wrapper around bdb."""

from __future__ import annotations

import bdb
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from athena.core.debugger import DebuggerCore


class BreakpointManager:
    """Manages breakpoints through the bdb interface."""

    def __init__(self, debugger: DebuggerCore):
        self._debugger = debugger

    def set_breakpoint(
        self,
        filename: str | None = None,
        lineno: int = 0,
        condition: str | None = None,
        temporary: bool = False,
    ) -> dict[str, Any]:
        """Set a breakpoint at file:line, optionally with a condition."""
        if filename is None:
            frame = self._debugger.current_frame
            if frame is None:
                return {"error": "No filename specified and no active frame"}
            filename = frame.f_code.co_filename

        filename = os.path.abspath(filename)

        err = self._debugger.set_break(
            filename, lineno, temporary=temporary, cond=condition
        )
        if err:
            return {"error": err}

        return {
            "status": "ok",
            "filename": filename,
            "lineno": lineno,
            "condition": condition,
            "temporary": temporary,
        }

    def remove_breakpoint(
        self,
        bp_number: int | None = None,
        filename: str | None = None,
        lineno: int | None = None,
    ) -> dict[str, Any]:
        """Remove a breakpoint by number, or by file:line."""
        if bp_number is not None:
            # Remove by breakpoint number
            try:
                bp = bdb.Breakpoint.bpbynumber[bp_number]
            except (IndexError, KeyError):
                return {"error": f"Breakpoint #{bp_number} not found"}
            if bp is None:
                return {"error": f"Breakpoint #{bp_number} already deleted"}
            err = self._debugger.clear_break(bp.file, bp.line)
            if err:
                return {"error": err}
            return {"status": "ok", "removed": bp_number}

        if filename and lineno:
            filename = os.path.abspath(filename)
            err = self._debugger.clear_break(filename, lineno)
            if err:
                return {"error": err}
            return {"status": "ok", "removed_at": f"{filename}:{lineno}"}

        return {"error": "Specify bp_number or filename+lineno"}

    def list_breakpoints(self) -> list[dict[str, Any]]:
        """List all currently set breakpoints."""
        result = []
        for bp in bdb.Breakpoint.bpbynumber:
            if bp is None:
                continue
            result.append({
                "number": bp.number,
                "file": bp.file,
                "line": bp.line,
                "enabled": bp.enabled,
                "temporary": bp.temporary,
                "condition": bp.cond,
                "hits": bp.hits,
                "ignore_count": bp.ignore,
            })
        return result

    def add_from_spec(self, spec: str) -> dict[str, Any]:
        """Parse a breakpoint spec like 'file.py:42' or 'file.py:42 if x > 5'."""
        condition = None
        if " if " in spec:
            spec, condition = spec.split(" if ", 1)

        if ":" in spec:
            parts = spec.rsplit(":", 1)
            filename = parts[0]
            try:
                lineno = int(parts[1])
            except ValueError:
                return {"error": f"Invalid line number: {parts[1]}"}
            return self.set_breakpoint(filename, lineno, condition)

        # Treat as function name -- find its source location
        return {"error": f"Function breakpoints not yet supported: {spec}"}

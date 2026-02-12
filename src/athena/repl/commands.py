"""Built-in slash commands that bypass the LLM for speed."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from athena.repl.session import DebugSession


HELP_TEXT = """\
Slash commands (bypass the LLM, instant):
  /locals, /l          Show all local variables
  /stack, /bt          Show call stack
  /source, /src        Show source around current line
  /step, /s            Step into
  /next, /n            Step over
  /continue, /c        Continue execution
  /return, /r          Step out (return)
  /break FILE:LINE     Set breakpoint (e.g., /break main.py:42)
  /breakpoints, /bp    List breakpoints
  /watch EXPR          Add watch expression
  /unwatch EXPR        Remove watch expression
  /watches             Show all watches
  /memory              Quick memory stats
  /snapshot LABEL      Take memory snapshot
  /focus FILE/FUNC     Focus debugging on file or function
  /unfocus             Clear focus (stop everywhere)
  /quit, /q            Quit debugger and program
  /help, /h            Show this help

Everything else is sent to the model as natural language.
"""


class CommandHandler:
    """Handles slash commands in the REPL."""

    def __init__(self, session: DebugSession):
        self._session = session

    def handle(self, command: str) -> str | None:
        """Handle a slash command. Returns None if it's an execution control command.

        Returns a string result for display commands, or None if the
        command triggers execution control (step/next/continue/return/quit).
        """
        parts = command.strip().split(None, 1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        handlers = {
            "/locals": self._locals,
            "/l": self._locals,
            "/stack": self._stack,
            "/bt": self._stack,
            "/source": self._source,
            "/src": self._source,
            "/step": self._step,
            "/s": self._step,
            "/next": self._next,
            "/n": self._next,
            "/continue": self._continue,
            "/c": self._continue,
            "/return": self._return,
            "/r": self._return,
            "/break": self._break,
            "/breakpoints": self._breakpoints,
            "/bp": self._breakpoints,
            "/watch": self._watch,
            "/unwatch": self._unwatch,
            "/watches": self._watches,
            "/memory": self._memory,
            "/snapshot": self._snapshot,
            "/focus": self._focus,
            "/unfocus": self._unfocus,
            "/quit": self._quit,
            "/q": self._quit,
            "/help": self._help,
            "/h": self._help,
        }

        handler = handlers.get(cmd)
        if handler is None:
            return f"Unknown command: {cmd}. Type /help for available commands."

        return handler(arg)

    def _locals(self, _arg: str) -> str:
        self._session.formatter.show_locals(
            self._session.frame_inspector.get_all_locals()
        )
        return ""

    def _stack(self, _arg: str) -> str:
        self._session.formatter.show_stack(
            self._session.frame_inspector.get_stack_frames()
        )
        return ""

    def _source(self, _arg: str) -> str:
        ctx = self._session.frame_inspector.get_source_context(context_lines=15)
        if "error" in ctx:
            return ctx["error"]
        lines = ctx.get("lines", {})
        self._session.formatter._show_source_snippet(
            lines, ctx["current_line"], ctx["filename"]
        )
        return ""

    def _step(self, _arg: str) -> str | None:
        self._session.set_execution_action("step")
        return None

    def _next(self, _arg: str) -> str | None:
        self._session.set_execution_action("next")
        return None

    def _continue(self, _arg: str) -> str | None:
        self._session.set_execution_action("continue")
        return None

    def _return(self, _arg: str) -> str | None:
        self._session.set_execution_action("return")
        return None

    def _break(self, arg: str) -> str:
        if not arg:
            return "Usage: /break FILE:LINE [if CONDITION]"
        result = self._session.breakpoint_manager.add_from_spec(arg)
        if "error" in result:
            return result["error"]
        return f"Breakpoint set at {result.get('filename', '?')}:{result.get('lineno', '?')}"

    def _breakpoints(self, _arg: str) -> str:
        bps = self._session.breakpoint_manager.list_breakpoints()
        if not bps:
            return "No breakpoints set."
        lines = []
        for bp in bps:
            cond = f" if {bp['condition']}" if bp.get("condition") else ""
            status = "enabled" if bp["enabled"] else "disabled"
            lines.append(
                f"  #{bp['number']} {bp['file']}:{bp['line']} "
                f"[{status}] hits={bp['hits']}{cond}"
            )
        return "Breakpoints:\n" + "\n".join(lines)

    def _watch(self, arg: str) -> str:
        if not arg:
            return "Usage: /watch EXPRESSION"
        return self._session.watch_manager.add_watch(arg)

    def _unwatch(self, arg: str) -> str:
        if not arg:
            return "Usage: /unwatch EXPRESSION"
        removed = self._session.watch_manager.remove_watch(arg)
        return f"Removed watch: {arg}" if removed else f"Watch not found: {arg}"

    def _watches(self, _arg: str) -> str:
        watches = self._session.watch_manager.evaluate_all()
        if not watches:
            return "No watch expressions set."
        lines = []
        for w in watches:
            change = " [CHANGED]" if w["changed"] else ""
            err = f" ERROR: {w['error']}" if w["error"] else ""
            lines.append(f"  {w['expression']} = {w['current_value']}{change}{err}")
        return "Watches:\n" + "\n".join(lines)

    def _memory(self, _arg: str) -> str:
        mem = self._session.python_memory.get_current_memory()
        if "error" in mem:
            # Start tracing and try again
            self._session.python_memory.start_tracing()
            mem = self._session.python_memory.get_current_memory()
        if "error" in mem:
            return mem["error"]
        return (
            f"Memory: current={mem['current_human']}, peak={mem['peak_human']}"
        )

    def _snapshot(self, arg: str) -> str:
        label = arg or ""
        result = self._session.python_memory.take_snapshot(label)
        return (
            f"Snapshot '{result['label']}': "
            f"{result['total_size_human']}, {result['num_traces']} traces"
        )

    def _focus(self, arg: str) -> str:
        if not arg:
            files = self._session.debugger._focus_files
            funcs = self._session.debugger._focus_functions
            if not files and not funcs:
                return "No focus set. Usage: /focus FILE or /focus @FUNCTION"
            parts = []
            if files:
                parts.append(f"Files: {', '.join(files)}")
            if funcs:
                parts.append(f"Functions: {', '.join(funcs)}")
            return "Focus: " + "; ".join(parts)

        if arg.startswith("@"):
            self._session.debugger.add_focus_function(arg[1:])
            return f"Focus added: function {arg[1:]}"
        else:
            self._session.debugger.add_focus_file(arg)
            return f"Focus added: file {arg}"

    def _unfocus(self, _arg: str) -> str:
        self._session.debugger.clear_focus()
        return "Focus cleared. Debugger will stop everywhere."

    def _quit(self, _arg: str) -> str | None:
        self._session.set_execution_action("quit")
        return None

    def _help(self, _arg: str) -> str:
        return HELP_TEXT

"""Bridge model tool calls to the debugger core."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from athena.agent.tool_registry import ToolRegistry

if TYPE_CHECKING:
    from athena.core.breakpoint_manager import BreakpointManager
    from athena.core.debugger import DebuggerCore
    from athena.core.expression_evaluator import ExpressionEvaluator
    from athena.core.frame_inspector import FrameInspector
    from athena.core.source_provider import SourceProvider
    from athena.core.watch_manager import WatchManager
    from athena.memory.cuda_memory import CudaMemoryProfiler
    from athena.memory.leak_detector import LeakDetector
    from athena.memory.python_memory import PythonMemoryProfiler

from athena.core.static_analyzer import StaticAnalyzer


# Sentinel returned by execution control tools to signal REPL exit
EXECUTION_CONTROL_SENTINEL = "__EXECUTION_CONTROL__"


class ToolExecutor:
    """Bridges model tool calls to the actual debugger components.

    Registers all tools and dispatches calls. Execution control tools
    (step/next/continue/return) set a flag on the session to exit the REPL.
    """

    def __init__(
        self,
        debugger: DebuggerCore,
        frame_inspector: FrameInspector,
        evaluator: ExpressionEvaluator,
        watch_manager: WatchManager,
        breakpoint_manager: BreakpointManager,
        source_provider: SourceProvider,
        python_memory: PythonMemoryProfiler,
        cuda_memory: CudaMemoryProfiler,
        leak_detector: LeakDetector,
    ):
        self._debugger = debugger
        self._inspector = frame_inspector
        self._evaluator = evaluator
        self._watches = watch_manager
        self._breakpoints = breakpoint_manager
        self._source = source_provider
        self._static = StaticAnalyzer()
        self._python_memory = python_memory
        self._cuda_memory = cuda_memory
        self._leak_detector = leak_detector

        self.registry = ToolRegistry()
        self._execution_action: str | None = None

        self._register_all_tools()

    def get_execution_action(self) -> str | None:
        """Return and clear any pending execution control action."""
        action = self._execution_action
        self._execution_action = None
        return action

    def _register_all_tools(self) -> None:
        r = self.registry

        # === Variable Inspection ===
        r.register(
            "inspect_variable",
            "Inspect a variable's value, type, and attributes in the specified stack frame.",
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The variable name to inspect"},
                    "frame_index": {
                        "type": "integer",
                        "description": "Stack frame index (0=outermost, highest=current). Use get_call_stack to see frames.",
                    },
                    "max_depth": {"type": "integer", "description": "Max depth for recursive inspection (1-5)", "default": 3},
                },
                "required": ["name"],
            },
            self._inspect_variable,
        )

        r.register(
            "get_all_locals",
            "Get all local variables and their values in the specified stack frame.",
            {
                "type": "object",
                "properties": {
                    "frame_index": {"type": "integer", "description": "Stack frame index"},
                },
                "required": [],
            },
            self._get_all_locals,
        )

        r.register(
            "get_closure_vars",
            "Inspect closure (free) variables captured by the function.",
            {
                "type": "object",
                "properties": {
                    "frame_index": {"type": "integer"},
                },
                "required": [],
            },
            self._get_closure_vars,
        )

        # === Expression Evaluation ===
        r.register(
            "evaluate_expression",
            "Evaluate a Python expression in the target frame context. Has access to all local and global variables. Can also execute statements.",
            {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Python expression or statement to evaluate"},
                    "frame_index": {"type": "integer"},
                },
                "required": ["expression"],
            },
            self._evaluate_expression,
        )

        # === Source Code ===
        r.register(
            "get_source_context",
            "Get source code surrounding the current execution point with the current line highlighted.",
            {
                "type": "object",
                "properties": {
                    "frame_index": {"type": "integer"},
                    "context_lines": {"type": "integer", "description": "Lines above and below to show", "default": 15},
                },
                "required": [],
            },
            self._get_source_context,
        )

        r.register(
            "get_source_file",
            "Read source code of a file, or a specific line range.",
            {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "File path. If omitted, uses current file."},
                    "start_line": {"type": "integer", "default": 1},
                    "end_line": {"type": "integer", "description": "End line (inclusive). Omit to read to end."},
                },
                "required": [],
            },
            self._get_source_file,
        )

        r.register(
            "static_analyze_file",
            "Run static analysis on a Python file to find likely issues without executing code.",
            {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "File path. If omitted, uses the current frame's file.",
                    },
                },
                "required": [],
            },
            self._static_analyze_file,
        )

        r.register(
            "static_analyze_snippet",
            "Run static analysis on pasted Python code without executing it.",
            {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Python snippet text to analyze",
                    },
                    "filename_hint": {
                        "type": "string",
                        "description": "Optional pseudo filename for diagnostics",
                        "default": "<snippet>",
                    },
                },
                "required": ["source"],
            },
            self._static_analyze_snippet,
        )

        r.register(
            "find_snippet_lines",
            "Find where a pasted code snippet appears in a file and return line ranges.",
            {
                "type": "object",
                "properties": {
                    "snippet": {
                        "type": "string",
                        "description": "Code snippet to locate",
                    },
                    "filename": {
                        "type": "string",
                        "description": "File path. If omitted, uses current frame file.",
                    },
                },
                "required": ["snippet"],
            },
            self._find_snippet_lines,
        )

        r.register(
            "replace_file_contents",
            "Replace the full text contents of a file. Use this to apply a complete fix after inspecting source.",
            {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "File path to update"},
                    "content": {"type": "string", "description": "New full file contents"},
                    "create_if_missing": {"type": "boolean", "default": False},
                },
                "required": ["filename", "content"],
            },
            self._replace_file_contents,
        )

        r.register(
            "replace_text_in_file",
            "Replace exact text in a file. Prefer this for small targeted edits.",
            {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "File path to update"},
                    "old_text": {"type": "string", "description": "Exact text to replace"},
                    "new_text": {"type": "string", "description": "Replacement text"},
                    "max_replacements": {
                        "type": "integer",
                        "description": "Maximum replacements; -1 means replace all matches",
                        "default": -1,
                    },
                },
                "required": ["filename", "old_text", "new_text"],
            },
            self._replace_text_in_file,
        )

        # === Call Stack ===
        r.register(
            "get_call_stack",
            "Get the full call stack with file, line, function, and local variable names for each frame.",
            {
                "type": "object",
                "properties": {},
                "required": [],
            },
            self._get_call_stack,
        )

        # === Breakpoints ===
        r.register(
            "set_breakpoint",
            "Set a breakpoint at file:line. Optionally with a condition.",
            {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Source file. If omitted, uses current file."},
                    "lineno": {"type": "integer", "description": "Line number"},
                    "condition": {"type": "string", "description": "Condition expression (break only when true)"},
                },
                "required": ["lineno"],
            },
            self._set_breakpoint,
        )

        r.register(
            "remove_breakpoint",
            "Remove a breakpoint by number or by file:line.",
            {
                "type": "object",
                "properties": {
                    "bp_number": {"type": "integer"},
                    "filename": {"type": "string"},
                    "lineno": {"type": "integer"},
                },
                "required": [],
            },
            self._remove_breakpoint,
        )

        r.register(
            "list_breakpoints",
            "List all breakpoints with status, conditions, and hit counts.",
            {
                "type": "object",
                "properties": {},
                "required": [],
            },
            self._list_breakpoints,
        )

        # === Execution Control ===
        r.register(
            "step_into",
            "Step into the next function call. Resumes execution -- the next stop triggers a new debugging interaction.",
            {
                "type": "object",
                "properties": {},
                "required": [],
            },
            self._step_into,
        )

        r.register(
            "step_over",
            "Step over to the next line in the current function. Resumes execution.",
            {
                "type": "object",
                "properties": {},
                "required": [],
            },
            self._step_over,
        )

        r.register(
            "step_out",
            "Continue until the current function returns. Resumes execution.",
            {
                "type": "object",
                "properties": {},
                "required": [],
            },
            self._step_out,
        )

        r.register(
            "continue_execution",
            "Continue execution until the next breakpoint or program end. Resumes execution.",
            {
                "type": "object",
                "properties": {},
                "required": [],
            },
            self._continue_execution,
        )

        # === Watch Expressions ===
        r.register(
            "add_watch",
            "Add a watch expression that auto-evaluates each time the debugger stops. You'll see when values change.",
            {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Python expression to watch"},
                },
                "required": ["expression"],
            },
            self._add_watch,
        )

        r.register(
            "remove_watch",
            "Remove a previously added watch expression.",
            {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"},
                },
                "required": ["expression"],
            },
            self._remove_watch,
        )

        r.register(
            "get_watches",
            "Evaluate all watch expressions and show current values with change indicators.",
            {
                "type": "object",
                "properties": {},
                "required": [],
            },
            self._get_watches,
        )

        # === Python Memory ===
        r.register(
            "memory_snapshot",
            "Take a Python memory snapshot using tracemalloc. Starts tracing if needed. Label for later comparison.",
            {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "description": "Label for this snapshot (e.g., 'before_loop')"},
                },
                "required": [],
            },
            self._memory_snapshot,
        )

        r.register(
            "memory_compare",
            "Compare two memory snapshots to find allocation changes. Useful for finding leaks.",
            {
                "type": "object",
                "properties": {
                    "label_a": {"type": "string", "description": "First (earlier) snapshot label"},
                    "label_b": {"type": "string", "description": "Second (later) snapshot label"},
                    "top_n": {"type": "integer", "default": 20},
                },
                "required": ["label_a", "label_b"],
            },
            self._memory_compare,
        )

        r.register(
            "memory_top_allocations",
            "Show top memory allocations from the most recent snapshot.",
            {
                "type": "object",
                "properties": {
                    "top_n": {"type": "integer", "default": 20},
                    "group_by": {"type": "string", "enum": ["lineno", "filename", "traceback"], "default": "lineno"},
                },
                "required": [],
            },
            self._memory_top_allocations,
        )

        r.register(
            "memory_current",
            "Get current and peak memory usage from tracemalloc.",
            {
                "type": "object",
                "properties": {},
                "required": [],
            },
            self._memory_current,
        )

        r.register(
            "gc_stats",
            "Get garbage collector statistics: object counts, collection stats, top types.",
            {
                "type": "object",
                "properties": {
                    "top_n_types": {"type": "integer", "default": 20},
                },
                "required": [],
            },
            self._gc_stats,
        )

        r.register(
            "object_references",
            "Analyze reference graph for an object: what refers to it and what it refers to. Useful for understanding GC behavior.",
            {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Expression evaluating to the object to analyze"},
                    "direction": {"type": "string", "enum": ["referrers", "referents", "both"], "default": "both"},
                    "max_results": {"type": "integer", "default": 10},
                },
                "required": ["expression"],
            },
            self._object_references,
        )

        # === CUDA Memory ===
        r.register(
            "cuda_memory_stats",
            "Get PyTorch CUDA memory stats: allocated, reserved, peak. Error if no CUDA.",
            {
                "type": "object",
                "properties": {
                    "device": {"type": "integer", "default": 0},
                },
                "required": [],
            },
            self._cuda_memory_stats,
        )

        r.register(
            "cuda_memory_summary",
            "Get detailed PyTorch CUDA memory summary (allocator statistics).",
            {
                "type": "object",
                "properties": {
                    "device": {"type": "integer", "default": 0},
                },
                "required": [],
            },
            self._cuda_memory_summary,
        )

        r.register(
            "cuda_live_tensors",
            "List all live CUDA tensors with shapes, dtypes, sizes, refcounts. Useful for finding tensor leaks.",
            {
                "type": "object",
                "properties": {
                    "device": {"type": "integer", "default": 0},
                    "sort_by": {"type": "string", "enum": ["size", "refcount"], "default": "size"},
                    "top_n": {"type": "integer", "default": 30},
                },
                "required": [],
            },
            self._cuda_live_tensors,
        )

        r.register(
            "cuda_tensor_snapshot",
            "Snapshot all live CUDA tensors for later comparison.",
            {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                },
                "required": ["label"],
            },
            self._cuda_tensor_snapshot,
        )

        r.register(
            "cuda_tensor_compare",
            "Compare two CUDA tensor snapshots to find new/removed tensors.",
            {
                "type": "object",
                "properties": {
                    "label_a": {"type": "string"},
                    "label_b": {"type": "string"},
                },
                "required": ["label_a", "label_b"],
            },
            self._cuda_tensor_compare,
        )

        r.register(
            "detect_leaks",
            "Analyze all snapshots (Python + CUDA) to identify potential memory leaks using heuristics.",
            {
                "type": "object",
                "properties": {},
                "required": [],
            },
            self._detect_leaks,
        )

        # === Focus Control ===
        r.register(
            "set_focus",
            "Focus debugging on specific files or functions. When focus is set, the debugger only stops in matching code.",
            {
                "type": "object",
                "properties": {
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "File paths to focus on",
                    },
                    "functions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Function names to focus on",
                    },
                },
                "required": [],
            },
            self._set_focus,
        )

        r.register(
            "clear_focus",
            "Clear all focus filters. The debugger will stop everywhere again.",
            {
                "type": "object",
                "properties": {},
                "required": [],
            },
            self._clear_focus,
        )

    # === Handler implementations ===

    def _inspect_variable(self, name: str, frame_index: int | None = None, max_depth: int = 3) -> dict:
        return self._inspector.get_variable(name, frame_index, max_depth)

    def _get_all_locals(self, frame_index: int | None = None) -> dict:
        return self._inspector.get_all_locals(frame_index)

    def _get_closure_vars(self, frame_index: int | None = None) -> dict:
        return self._inspector.get_closure_vars(frame_index)

    def _evaluate_expression(self, expression: str, frame_index: int | None = None) -> dict:
        return self._evaluator.evaluate(expression, frame_index)

    def _get_source_context(self, frame_index: int | None = None, context_lines: int = 15) -> dict:
        return self._inspector.get_source_context(frame_index, context_lines)

    def _get_source_file(self, filename: str | None = None, start_line: int = 1, end_line: int | None = None) -> dict:
        if filename is None:
            frame = self._debugger.current_frame
            if frame is None:
                return {"error": "No filename specified and no active frame"}
            filename = frame.f_code.co_filename
        return self._source.get_file_source(filename, start_line, end_line)

    def _static_analyze_file(self, filename: str | None = None) -> dict:
        if filename is None:
            frame = self._debugger.current_frame
            if frame is None:
                return {"error": "No filename specified and no active frame"}
            filename = frame.f_code.co_filename
        return self._static.analyze_file(filename)

    def _static_analyze_snippet(
        self,
        source: str,
        filename_hint: str = "<snippet>",
    ) -> dict:
        return self._static.analyze_source(source, filename=filename_hint)

    def _find_snippet_lines(
        self,
        snippet: str,
        filename: str | None = None,
    ) -> dict:
        if filename is None:
            frame = self._debugger.current_frame
            if frame is None:
                return {"error": "No filename specified and no active frame"}
            filename = frame.f_code.co_filename
        return self._source.find_snippet_lines(filename, snippet)

    def _replace_file_contents(self, filename: str, content: str, create_if_missing: bool = False) -> dict:
        return self._source.write_file_source(
            filename=filename,
            content=content,
            create_if_missing=create_if_missing,
        )

    def _replace_text_in_file(
        self,
        filename: str,
        old_text: str,
        new_text: str,
        max_replacements: int = -1,
    ) -> dict:
        return self._source.replace_text_in_file(
            filename=filename,
            old_text=old_text,
            new_text=new_text,
            max_replacements=max_replacements,
        )

    def _get_call_stack(self) -> list[dict]:
        return self._inspector.get_stack_frames()

    def _set_breakpoint(self, lineno: int, filename: str | None = None, condition: str | None = None) -> dict:
        return self._breakpoints.set_breakpoint(filename, lineno, condition)

    def _remove_breakpoint(self, bp_number: int | None = None, filename: str | None = None, lineno: int | None = None) -> dict:
        return self._breakpoints.remove_breakpoint(bp_number, filename, lineno)

    def _list_breakpoints(self) -> list[dict]:
        return self._breakpoints.list_breakpoints()

    def _step_into(self) -> dict:
        self._execution_action = "step"
        return {"action": "step_into", "message": "Execution will resume. Stepping into next call."}

    def _step_over(self) -> dict:
        self._execution_action = "next"
        return {"action": "step_over", "message": "Execution will resume. Stepping to next line."}

    def _step_out(self) -> dict:
        self._execution_action = "return"
        return {"action": "step_out", "message": "Execution will resume. Running until function returns."}

    def _continue_execution(self) -> dict:
        self._execution_action = "continue"
        return {"action": "continue", "message": "Execution will resume. Running until next breakpoint."}

    def _add_watch(self, expression: str) -> dict:
        msg = self._watches.add_watch(expression)
        return {"message": msg}

    def _remove_watch(self, expression: str) -> dict:
        removed = self._watches.remove_watch(expression)
        return {"removed": removed, "expression": expression}

    def _get_watches(self) -> list[dict]:
        return self._watches.evaluate_all()

    def _memory_snapshot(self, label: str = "") -> dict:
        return self._python_memory.take_snapshot(label)

    def _memory_compare(self, label_a: str, label_b: str, top_n: int = 20) -> dict:
        return self._python_memory.compare_snapshots(label_a, label_b, top_n)

    def _memory_top_allocations(self, top_n: int = 20, group_by: str = "lineno") -> list[dict]:
        return self._python_memory.get_top_allocations(top_n, group_by)

    def _memory_current(self) -> dict:
        return self._python_memory.get_current_memory()

    def _gc_stats(self, top_n_types: int = 20) -> dict:
        return self._python_memory.gc_stats(top_n_types)

    def _object_references(self, expression: str, direction: str = "both", max_results: int = 10) -> dict:
        result = self._evaluator.evaluate(expression)
        if "error" in result:
            return result

        # We need the actual object, not its repr. Re-evaluate to get it.
        frame = self._debugger.current_frame
        if frame is None:
            return {"error": "No active frame"}

        try:
            obj = eval(expression, frame.f_globals, frame.f_locals)  # noqa: S307
        except Exception as e:
            return {"error": f"Could not evaluate expression: {e}"}

        output: dict[str, Any] = {"expression": expression}
        if direction in ("referrers", "both"):
            output["referrers"] = self._python_memory.get_referrers(obj, max_results)
        if direction in ("referents", "both"):
            output["referents"] = self._python_memory.get_referents(obj, max_results)
        return output

    def _cuda_memory_stats(self, device: int = 0) -> dict:
        return self._cuda_memory.get_memory_stats(device)

    def _cuda_memory_summary(self, device: int = 0) -> dict:
        return self._cuda_memory.get_memory_summary(device)

    def _cuda_live_tensors(self, device: int = 0, sort_by: str = "size", top_n: int = 30) -> dict:
        return self._cuda_memory.get_live_tensors(device, sort_by, top_n)

    def _cuda_tensor_snapshot(self, label: str) -> dict:
        return self._cuda_memory.take_tensor_snapshot(label)

    def _cuda_tensor_compare(self, label_a: str, label_b: str) -> dict:
        return self._cuda_memory.compare_tensor_snapshots(label_a, label_b)

    def _detect_leaks(self) -> dict:
        return self._leak_detector.analyze()

    def _set_focus(self, files: list[str] | None = None, functions: list[str] | None = None) -> dict:
        if files:
            self._debugger.set_focus_files(files)
        if functions:
            self._debugger.set_focus_functions(functions)
        return {
            "focus_files": list(self._debugger._focus_files),
            "focus_functions": list(self._debugger._focus_functions),
        }

    def _clear_focus(self) -> dict:
        self._debugger.clear_focus()
        return {"message": "Focus cleared. Debugger will stop everywhere."}

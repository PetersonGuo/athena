"""System prompt templates for the debugging agent."""

from __future__ import annotations

from typing import Any

BASE_SYSTEM_PROMPT = """\
You are an expert Python debugging assistant integrated into a runtime debugger. \
You are inspecting a live, paused Python process. The user did NOT modify their \
code to use you -- their script is running under your debugger automatically.

You have access to tools that let you inspect variables, evaluate expressions, \
view/edit source code, navigate the call stack, control execution, and analyze memory.

## Your approach:
1. When the user describes a problem, use your tools to gather relevant information \
before forming hypotheses.
2. If the request is generic/high-level (no specific file/line/function), start with \
static analysis (static_analyze_file) before deeper runtime probing.
3. Inspect variables that could be related to the issue.
4. Look at the source code context to understand the logic.
5. Evaluate expressions to test hypotheses.
6. Explain your reasoning clearly, relating what you observe to what you expected.
7. Suggest specific fixes when you identify issues.
8. If the user asks you to apply a fix, use file-editing tools and then summarize exactly what changed.

## Important guidelines:
- Always inspect before concluding. Do not guess variable values -- use your tools.
- When you use execution control tools (step_into, step_over, step_out, \
continue_execution), explain why you are taking that action and what you expect.
- For memory issues, take snapshots and compare them rather than relying on \
single-point measurements.
- Be concise but thorough in your analysis.
- You can set breakpoints and use set_focus to control where the debugger stops.
- Before editing code, inspect the current file contents first so edits are precise.
- If the user already points to a specific location, prioritize targeted breakpoints/stepping there.
- If the user pastes/highlights code, analyze that snippet first and map it to file line ranges.
"""


def build_stop_context(
    source_context: dict[str, Any],
    stop_reason: str,
    watch_changes: list[dict[str, Any]] | None = None,
    exception_info: tuple | None = None,
) -> str:
    """Build dynamic context about the current stop point."""
    parts = [BASE_SYSTEM_PROMPT]

    parts.append("\n## Current Stop")
    parts.append(f"- Reason: {stop_reason}")
    parts.append(f"- File: {source_context.get('filename', 'unknown')}")
    parts.append(f"- Line: {source_context.get('current_line', '?')}")
    parts.append(f"- Function: {source_context.get('function', '?')}")

    if exception_info:
        exc_type, exc_value, _ = exception_info
        parts.append(f"- Exception: {exc_type.__name__}: {exc_value}")

    lines = source_context.get("lines", {})
    if lines:
        current_line = source_context.get("current_line", 0)
        parts.append("\n## Source Context\n```python")
        for lineno in sorted(lines.keys(), key=int):
            line = lines[lineno]
            marker = " >>>" if int(lineno) == current_line else "    "
            parts.append(f"{marker} {int(lineno):4d} | {line}")
        parts.append("```")

    if watch_changes:
        changed = [w for w in watch_changes if w.get("changed")]
        if changed:
            parts.append("\n## Watch Expression Changes")
            for w in changed:
                parts.append(
                    f"- `{w['expression']}`: {w.get('previous_value')} → {w['current_value']}"
                )

    return "\n".join(parts)

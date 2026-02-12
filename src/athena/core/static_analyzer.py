"""Static code analysis helpers for generic debugging prompts."""

from __future__ import annotations

import ast
import os
from typing import Any


class StaticAnalyzer:
    """Lightweight AST-based analyzer for common bug patterns."""

    def analyze_file(self, filename: str) -> dict[str, Any]:
        filename = os.path.abspath(filename)
        if not os.path.isfile(filename):
            return {"error": f"File not found: {filename}"}

        try:
            with open(filename) as f:
                source = f.read()
        except Exception as e:
            return {"error": f"Could not read file: {e}"}

        return self.analyze_source(source, filename=filename)

    def analyze_source(self, source: str, filename: str = "<snippet>") -> dict[str, Any]:
        """Analyze source text directly (useful for pasted snippets)."""
        try:
            tree = ast.parse(source, filename=filename)
        except SyntaxError as e:
            return {
                "error": "SyntaxError while parsing source",
                "filename": filename,
                "line": e.lineno,
                "offset": e.offset,
                "message": e.msg,
            }

        lines = source.splitlines()
        issues: list[dict[str, Any]] = []

        for node in ast.walk(tree):
            # Bare except blocks can hide root causes.
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                issues.append(self._issue(
                    node,
                    "bare_except",
                    "Bare except catches all exceptions; narrow the exception type.",
                    lines,
                ))

            # Mutable defaults are shared across calls.
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defaults = list(node.args.defaults) + list(node.args.kw_defaults)
                for default in defaults:
                    if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                        issues.append(self._issue(
                            node,
                            "mutable_default",
                            f"Function '{node.name}' has mutable default argument.",
                            lines,
                        ))
                        break

            # Potential divide-by-zero risks.
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)):
                if not self._is_nonzero_constant(node.right):
                    issues.append(self._issue(
                        node,
                        "possible_zero_division",
                        "Division/modulo denominator is non-constant or may be zero.",
                        lines,
                    ))

            # Infinite loops without an obvious break.
            if isinstance(node, ast.While) and self._is_true_constant(node.test):
                has_break = any(isinstance(sub, ast.Break) for sub in ast.walk(node))
                if not has_break:
                    issues.append(self._issue(
                        node,
                        "possible_infinite_loop",
                        "while True loop has no break statement.",
                        lines,
                    ))

        severity_counts = {"high": 0, "medium": 0, "low": 0}
        for issue in issues:
            severity_counts[issue["severity"]] += 1

        return {
            "filename": filename,
            "issue_count": len(issues),
            "severity_counts": severity_counts,
            "issues": sorted(issues, key=lambda i: (i["line"], i["kind"])),
        }

    @staticmethod
    def _issue(
        node: ast.AST,
        kind: str,
        message: str,
        lines: list[str],
    ) -> dict[str, Any]:
        lineno = getattr(node, "lineno", 0) or 0
        return {
            "kind": kind,
            "severity": _severity_for_kind(kind),
            "line": lineno,
            "message": message,
            "source": lines[lineno - 1].strip() if 0 < lineno <= len(lines) else "",
        }

    @staticmethod
    def _is_nonzero_constant(node: ast.AST) -> bool:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value != 0
        return False

    @staticmethod
    def _is_true_constant(node: ast.AST) -> bool:
        return isinstance(node, ast.Constant) and node.value is True


def _severity_for_kind(kind: str) -> str:
    if kind in {"possible_zero_division", "bare_except"}:
        return "high"
    if kind in {"mutable_default", "possible_infinite_loop"}:
        return "medium"
    return "low"

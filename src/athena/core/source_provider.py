"""Source code retrieval and file editing."""

from __future__ import annotations

import linecache
import os


class SourceProvider:
    """Provides source code retrieval and safe text edits."""

    def get_file_source(
        self,
        filename: str,
        start_line: int = 1,
        end_line: int | None = None,
    ) -> dict:
        """Read source code from a file, optionally a specific line range."""
        filename = os.path.abspath(filename)
        if not os.path.isfile(filename):
            return {"error": f"File not found: {filename}"}

        lines = linecache.getlines(filename)
        if not lines:
            try:
                with open(filename) as f:
                    lines = f.readlines()
            except Exception as e:
                return {"error": f"Could not read file: {e}"}

        total_lines = len(lines)
        if end_line is None:
            end_line = total_lines

        start_line = max(1, start_line)
        end_line = min(total_lines, end_line)

        result_lines: dict[int, str] = {}
        for i in range(start_line - 1, end_line):
            result_lines[i + 1] = lines[i].rstrip("\n")

        return {
            "filename": filename,
            "start_line": start_line,
            "end_line": end_line,
            "total_lines": total_lines,
            "lines": result_lines,
        }

    def find_function_lines(self, filename: str, function_name: str) -> dict:
        """Find the start and end lines of a function in a file.

        Uses simple AST parsing to find function definitions.
        """
        import ast

        filename = os.path.abspath(filename)
        if not os.path.isfile(filename):
            return {"error": f"File not found: {filename}"}

        try:
            with open(filename) as f:
                source = f.read()
            tree = ast.parse(source, filename)
        except Exception as e:
            return {"error": f"Could not parse file: {e}"}

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == function_name:
                    end_lineno = getattr(node, "end_lineno", None)
                    if end_lineno is None:
                        # Estimate end line for older Python
                        end_lineno = node.lineno + 50
                    return {
                        "filename": filename,
                        "function": function_name,
                        "start_line": node.lineno,
                        "end_line": end_lineno,
                    }

        return {"error": f"Function {function_name!r} not found in {filename}"}

    def find_snippet_lines(self, filename: str, snippet: str) -> dict:
        """Find exact snippet matches in a file and return matching line ranges."""
        filename = os.path.abspath(filename)
        if not os.path.isfile(filename):
            return {"error": f"File not found: {filename}"}

        normalized = snippet.strip("\n")
        if not normalized.strip():
            return {"error": "Snippet is empty"}

        try:
            with open(filename) as f:
                source_lines = f.read().splitlines()
        except Exception as e:
            return {"error": f"Could not read file: {e}"}

        snippet_lines = normalized.splitlines()
        n = len(snippet_lines)
        matches: list[dict[str, int]] = []

        for i in range(0, len(source_lines) - n + 1):
            window = source_lines[i:i + n]
            if window == snippet_lines:
                matches.append({"start_line": i + 1, "end_line": i + n})

        # Whitespace-insensitive fallback
        if not matches:
            stripped_snippet = [ln.strip() for ln in snippet_lines]
            for i in range(0, len(source_lines) - n + 1):
                window = [ln.strip() for ln in source_lines[i:i + n]]
                if window == stripped_snippet:
                    matches.append({"start_line": i + 1, "end_line": i + n})

        return {
            "filename": filename,
            "snippet_lines": n,
            "match_count": len(matches),
            "matches": matches,
        }

    def write_file_source(
        self,
        filename: str,
        content: str,
        create_if_missing: bool = False,
    ) -> dict:
        """Replace a file's entire text content."""
        filename = os.path.abspath(filename)
        exists = os.path.isfile(filename)
        if not exists and not create_if_missing:
            return {"error": f"File not found: {filename}"}

        old_content = ""
        if exists:
            try:
                with open(filename) as f:
                    old_content = f.read()
            except Exception as e:
                return {"error": f"Could not read file before writing: {e}"}

        if exists and old_content == content:
            return {
                "filename": filename,
                "status": "no_change",
                "bytes_written": 0,
                "lines_written": len(content.splitlines()),
            }

        try:
            with open(filename, "w") as f:
                f.write(content)
        except Exception as e:
            return {"error": f"Could not write file: {e}"}

        linecache.checkcache(filename)

        return {
            "filename": filename,
            "status": "updated" if exists else "created",
            "bytes_written": len(content.encode("utf-8")),
            "lines_written": len(content.splitlines()),
        }

    def replace_text_in_file(
        self,
        filename: str,
        old_text: str,
        new_text: str,
        max_replacements: int = -1,
    ) -> dict:
        """Replace text in a file and write the result back."""
        filename = os.path.abspath(filename)
        if not os.path.isfile(filename):
            return {"error": f"File not found: {filename}"}

        try:
            with open(filename) as f:
                content = f.read()
        except Exception as e:
            return {"error": f"Could not read file: {e}"}

        if old_text == "":
            return {"error": "old_text must be non-empty"}

        available = content.count(old_text)
        if available == 0:
            return {
                "error": "old_text not found in file",
                "filename": filename,
            }

        if max_replacements == 0:
            return {"error": "max_replacements cannot be 0"}

        new_content = content.replace(
            old_text,
            new_text,
            max_replacements if max_replacements > 0 else -1,
        )
        replacements = available if max_replacements < 0 else min(available, max_replacements)

        write_result = self.write_file_source(filename, new_content, create_if_missing=False)
        if "error" in write_result:
            return write_result

        return {
            "filename": filename,
            "replacements": replacements,
            "status": write_result["status"],
            "bytes_written": write_result["bytes_written"],
            "lines_written": write_result["lines_written"],
        }

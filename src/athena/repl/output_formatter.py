"""Rich-based terminal output formatting for the debug REPL."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text


class OutputFormatter:
    """Formats debug output using rich for beautiful terminal display."""

    def __init__(self):
        self._console = Console()

    def show_banner(self) -> None:
        """Show the startup banner."""
        self._console.print(
            Panel(
                "[bold]Athena[/bold]\n"
                "AI-powered interactive debugger\n\n"
                "Type your questions in natural language, or use /help for commands.\n"
                "Press Ctrl-D to quit, Ctrl-C to cancel input.",
                title="athena",
                border_style="blue",
            )
        )

    def show_stop_banner(
        self,
        filename: str,
        lineno: int,
        function: str,
        reason: str,
        source_lines: dict[int, str] | None = None,
    ) -> None:
        """Show a banner when the debugger stops."""
        location = f"{filename}:{lineno} in {function}()"

        if reason == "exception":
            style = "red"
            title = "Exception"
        elif reason == "return":
            style = "yellow"
            title = "Return"
        else:
            style = "green"
            title = "Breakpoint" if reason == "breakpoint" else "Stopped"

        content = Text()
        content.append(f"{title}: ", style=f"bold {style}")
        content.append(location, style="dim")

        self._console.print()
        self._console.print(Panel(content, border_style=style))

        if source_lines:
            self._show_source_snippet(source_lines, lineno, filename)

    def _show_source_snippet(
        self,
        lines: dict[int, str],
        current_line: int,
        filename: str,
    ) -> None:
        """Show a source code snippet with the current line highlighted."""
        # Build the source text with line numbers
        source_parts = []
        for lineno in sorted(lines.keys(), key=int):
            marker = ">>>" if int(lineno) == current_line else "   "
            source_parts.append(f" {marker} {int(lineno):4d} │ {lines[lineno]}")

        source_text = "\n".join(source_parts)
        self._console.print(source_text)
        self._console.print()

    def show_model_response(self, text: str) -> None:
        """Display a model response as formatted markdown."""
        md = Markdown(text)
        self._console.print(md)
        self._console.print()

    def print_streaming(self, chunk: str) -> None:
        """Print a streaming text chunk without newline."""
        self._console.print(chunk, end="", highlight=False)

    def print_newline(self) -> None:
        self._console.print()

    def show_locals(self, locals_dict: dict[str, dict[str, str]]) -> None:
        """Show local variables in a table."""
        table = Table(title="Local Variables", show_header=True)
        table.add_column("Name", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Value")

        for name, info in sorted(locals_dict.items()):
            table.add_row(name, info["type"], info["repr"])

        self._console.print(table)
        self._console.print()

    def show_stack(self, frames: list[dict[str, Any]]) -> None:
        """Show the call stack."""
        table = Table(title="Call Stack", show_header=True)
        table.add_column("#", style="dim")
        table.add_column("Function", style="cyan")
        table.add_column("Location")
        table.add_column("", style="bold yellow")

        for frame in reversed(frames):
            marker = "►" if frame["is_current"] else ""
            table.add_row(
                str(frame["index"]),
                frame["function"],
                f"{frame['filename']}:{frame['lineno']}",
                marker,
            )

        self._console.print(table)
        self._console.print()

    def show_error(self, message: str) -> None:
        """Show an error message."""
        self._console.print(f"[bold red]Error:[/bold red] {message}")

    def show_info(self, message: str) -> None:
        """Show an informational message."""
        self._console.print(f"[dim]{message}[/dim]")

    def show_tool_call(self, tool_name: str) -> None:
        """Show that a tool is being called (for transparency)."""
        self._console.print(f"  [dim]→ {tool_name}[/dim]", highlight=False)

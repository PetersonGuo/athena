"""REPL input handling with prompt-toolkit."""

from __future__ import annotations

import os
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings


class InputHandler:
    """Handles user input for the REPL.

    Features:
    - readline-like history with file persistence
    - Multi-line input (trailing backslash)
    - Ctrl-C cancels current input
    - Ctrl-D exits the REPL (returns None)
    """

    def __init__(self, history_file: str | None = None):
        if history_file is None:
            history_file = os.path.expanduser("~/.athena_history")

        # Ensure the directory exists
        history_dir = os.path.dirname(history_file)
        if history_dir:
            os.makedirs(history_dir, exist_ok=True)

        self._session: PromptSession = PromptSession(
            history=FileHistory(history_file),
        )

    def read_input(self, prompt: str = "debug> ") -> str | None:
        """Read one logical line of input.

        Returns None on EOF (Ctrl-D).
        Returns empty string on Ctrl-C.
        Supports multi-line input via trailing backslash.
        """
        try:
            lines: list[str] = []
            current_prompt = prompt

            while True:
                line = self._session.prompt(current_prompt)
                if line.endswith("\\"):
                    lines.append(line[:-1])
                    current_prompt = "... "
                else:
                    lines.append(line)
                    break

            return "\n".join(lines)
        except KeyboardInterrupt:
            return ""
        except EOFError:
            return None

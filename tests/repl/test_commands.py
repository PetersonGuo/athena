"""Tests for repl commands."""

from __future__ import annotations

from athena.repl.commands import CommandHandler


class _DummySession:
    def __init__(self):
        self.execution_action = None

    def set_execution_action(self, action: str) -> None:
        self.execution_action = action


def test_rerun_command_sets_execution_action():
    session = _DummySession()
    handler = CommandHandler(session)  # type: ignore[arg-type]
    result = handler.handle("/rerun")
    assert result is None
    assert session.execution_action == "rerun"

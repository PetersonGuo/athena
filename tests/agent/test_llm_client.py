"""Tests for llm_client.py loop guards."""

from __future__ import annotations

import json
from types import SimpleNamespace

from athena.agent.llm_client import LLMClient


class _FakeRegistry:
    def execute(self, _name: str, _input_data: dict) -> str:
        return json.dumps({"ok": True})

    def get_tool_schemas(self) -> list[dict]:
        return []


class _FakeExecutor:
    def __init__(self):
        self.registry = _FakeRegistry()

    def get_execution_action(self):
        return None


def test_repeated_identical_write_calls_are_blocked(monkeypatch):
    monkeypatch.setattr("athena.agent.llm_client.OpenAI", lambda **_: SimpleNamespace())
    client = LLMClient()
    client.set_tool_executor(_FakeExecutor())
    client._reset_tool_call_guards()

    first = client._execute_tool("replace_text_in_file", {"filename": "a.py", "old_text": "x", "new_text": "y"})
    second = client._execute_tool("replace_text_in_file", {"filename": "a.py", "old_text": "x", "new_text": "y"})
    third = client._execute_tool("replace_text_in_file", {"filename": "a.py", "old_text": "x", "new_text": "y"})

    assert '"ok": true' in first
    assert '"ok": true' in second
    assert "Loop guard" in third


def test_non_write_tools_not_blocked(monkeypatch):
    monkeypatch.setattr("athena.agent.llm_client.OpenAI", lambda **_: SimpleNamespace())
    client = LLMClient()
    client.set_tool_executor(_FakeExecutor())
    client._reset_tool_call_guards()

    for _ in range(10):
        out = client._execute_tool("get_call_stack", {})
        assert '"ok": true' in out


def test_tool_round_limit_returns_message(monkeypatch):
    class _InfiniteToolCallClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **_kwargs):
            tc = SimpleNamespace(
                id="tc1",
                function=SimpleNamespace(name="get_call_stack", arguments="{}"),
            )
            message = SimpleNamespace(content=None, tool_calls=[tc])
            choice = SimpleNamespace(message=message)
            return SimpleNamespace(choices=[choice])

    monkeypatch.setattr("athena.agent.llm_client.OpenAI", lambda **_: _InfiniteToolCallClient())
    client = LLMClient()
    client.set_tool_executor(_FakeExecutor())
    client._max_tool_rounds = 2

    text = client.send_message("debug this")
    assert "too many tool-call rounds" in text

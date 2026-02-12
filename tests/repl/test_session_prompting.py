"""Tests for prompt augmentation behavior in session."""

from __future__ import annotations

from athena.repl.session import DebugSession


def test_extract_pasted_code_from_fenced_block():
    user_input = "please debug this:\n```python\ndef f(x):\n    return 1 / x\n```\nthanks"
    code = DebugSession._extract_pasted_code(user_input)
    assert code is not None
    assert "def f(x):" in code


def test_extract_pasted_code_from_multiline_plain_text():
    user_input = "def f(x):\n    y = x + 1\n    return y\n"
    code = DebugSession._extract_pasted_code(user_input)
    assert code is not None
    assert "return y" in code


def test_generic_prompt_detection():
    assert DebugSession._is_generic_prompt("find likely issues")
    assert not DebugSession._is_generic_prompt("check examples/simple_bug.py line 8")

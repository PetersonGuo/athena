"""Tests for expression_evaluator.py."""

from __future__ import annotations

import sys

from athena.core.debugger import DebuggerCore
from athena.core.expression_evaluator import ExpressionEvaluator


def _make_evaluator():
    debugger = DebuggerCore()
    x = 42
    y = "hello"
    z = [1, 2, 3]
    frame = sys._getframe()
    debugger._current_frame = frame
    return ExpressionEvaluator(debugger)


def test_evaluate_simple_expression():
    evaluator = _make_evaluator()
    result = evaluator.evaluate("x + 1")
    assert result["result"] == "43"
    assert "int" in result["type"]


def test_evaluate_string_expression():
    evaluator = _make_evaluator()
    result = evaluator.evaluate("y.upper()")
    assert result["result"] == "'HELLO'"


def test_evaluate_list_expression():
    evaluator = _make_evaluator()
    result = evaluator.evaluate("len(z)")
    assert result["result"] == "3"


def test_evaluate_error():
    evaluator = _make_evaluator()
    result = evaluator.evaluate("1 / 0")
    assert "error" in result
    assert "ZeroDivisionError" in result["error"]


def test_evaluate_statement():
    evaluator = _make_evaluator()
    result = evaluator.evaluate("a = x * 2")
    assert result["result"] == "<statement executed>"


def test_evaluate_with_stdout():
    evaluator = _make_evaluator()
    result = evaluator.evaluate("print('test output')")
    assert "stdout" in result
    assert "test output" in result["stdout"]


def test_evaluate_no_frame():
    debugger = DebuggerCore()
    evaluator = ExpressionEvaluator(debugger)
    result = evaluator.evaluate("1 + 1")
    assert "error" in result

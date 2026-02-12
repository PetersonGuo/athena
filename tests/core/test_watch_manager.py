"""Tests for watch_manager.py."""

from __future__ import annotations

import sys

from athena.core.debugger import DebuggerCore
from athena.core.expression_evaluator import ExpressionEvaluator
from athena.core.watch_manager import WatchManager


def _make_watch_manager():
    debugger = DebuggerCore()
    x = 10
    frame = sys._getframe()
    debugger._current_frame = frame
    evaluator = ExpressionEvaluator(debugger)
    return WatchManager(evaluator)


def test_add_watch():
    wm = _make_watch_manager()
    result = wm.add_watch("x")
    assert "Added" in result


def test_add_duplicate_watch():
    wm = _make_watch_manager()
    wm.add_watch("x")
    result = wm.add_watch("x")
    assert "already exists" in result


def test_remove_watch():
    wm = _make_watch_manager()
    wm.add_watch("x")
    assert wm.remove_watch("x") is True
    assert wm.remove_watch("x") is False


def test_evaluate_all():
    wm = _make_watch_manager()
    wm.add_watch("x")
    results = wm.evaluate_all()
    assert len(results) == 1
    assert results[0]["expression"] == "x"
    assert "10" in results[0]["current_value"]


def test_change_detection():
    wm = _make_watch_manager()
    wm.add_watch("x")

    # First evaluation
    results = wm.evaluate_all()
    assert results[0]["changed"] is False

    # Second evaluation (same value)
    results = wm.evaluate_all()
    assert results[0]["changed"] is False


def test_clear_all():
    wm = _make_watch_manager()
    wm.add_watch("x")
    wm.add_watch("y")
    wm.clear_all()
    assert wm.get_watch_list() == []

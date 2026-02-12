"""Tests for frame_inspector.py."""

from __future__ import annotations

import sys

from athena.core.debugger import DebuggerCore
from athena.core.frame_inspector import FrameInspector


def _make_inspector_with_frame():
    """Create an inspector with a live frame captured inline."""
    debugger = DebuggerCore()

    x = 42
    y = "hello"
    z = [1, 2, 3]
    data = {"key": "value", "num": 123}
    frame = sys._getframe()
    debugger._current_frame = frame
    return FrameInspector(debugger), frame


def test_get_variable_local():
    inspector, frame = _make_inspector_with_frame()
    result = inspector.get_variable("x")
    assert result["name"] == "x"
    assert result["found_in"] == "local"
    assert "42" in result["repr"]


def test_get_variable_not_found():
    inspector, _ = _make_inspector_with_frame()
    result = inspector.get_variable("nonexistent")
    assert "error" in result


def test_get_all_locals():
    inspector, _ = _make_inspector_with_frame()
    locals_dict = inspector.get_all_locals()
    assert "x" in locals_dict
    assert "y" in locals_dict
    assert locals_dict["x"]["type"].endswith("int")


def test_get_source_context():
    inspector, _ = _make_inspector_with_frame()
    ctx = inspector.get_source_context()
    assert "filename" in ctx
    assert "current_line" in ctx
    assert "lines" in ctx
    assert len(ctx["lines"]) > 0


def test_get_stack_frames():
    inspector, _ = _make_inspector_with_frame()
    frames = inspector.get_stack_frames()
    assert len(frames) > 0
    # At least one frame should be marked as current
    current_frames = [f for f in frames if f["is_current"]]
    assert len(current_frames) == 1


def test_get_variable_builtin():
    inspector, _ = _make_inspector_with_frame()
    result = inspector.get_variable("len")
    assert result["name"] == "len"
    assert result["found_in"] == "builtin"

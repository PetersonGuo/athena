"""Tests for the core debugger engine."""

from __future__ import annotations

from athena.core.debugger import DebuggerCore


def test_debugger_creation():
    d = DebuggerCore()
    assert d.current_frame is None


def test_focus_files():
    d = DebuggerCore()
    d.set_focus_files(["test.py", "main.py"])
    assert len(d._focus_files) == 2
    d.clear_focus()
    assert len(d._focus_files) == 0


def test_focus_functions():
    d = DebuggerCore()
    d.set_focus_functions(["main", "process"])
    assert "main" in d._focus_functions
    assert "process" in d._focus_functions


def test_add_focus():
    d = DebuggerCore()
    d.add_focus_file("test.py")
    d.add_focus_function("main")
    assert len(d._focus_files) == 1
    assert len(d._focus_functions) == 1


def test_inject_runtime_breaks():
    source = "def f():\n    x = 1\n    return x\n"
    injected = DebuggerCore._inject_runtime_breaks(source, [2])
    assert '__import__("athena").breakpoint()' in injected
    assert injected.count("athena:injected-break") == 1

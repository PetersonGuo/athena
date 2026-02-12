"""Tests for python_memory.py."""

from __future__ import annotations

from athena.memory.python_memory import PythonMemoryProfiler, _format_bytes


def test_format_bytes():
    assert _format_bytes(0) == "0 B"
    assert _format_bytes(1023) == "1023 B"
    assert "KiB" in _format_bytes(1024)
    assert "MiB" in _format_bytes(1024 * 1024)


def test_start_stop_tracing():
    pm = PythonMemoryProfiler()
    result = pm.start_tracing()
    assert result["status"] == "started"

    result = pm.start_tracing()
    assert result["status"] == "already_tracing"

    result = pm.stop_tracing()
    assert result["status"] == "stopped"


def test_take_snapshot():
    pm = PythonMemoryProfiler()
    result = pm.take_snapshot("test")
    assert result["label"] == "test"
    assert result["total_size_bytes"] >= 0
    pm.stop_tracing()


def test_compare_snapshots():
    pm = PythonMemoryProfiler()
    pm.take_snapshot("before")

    # Allocate some memory
    big_list = [i for i in range(10000)]

    pm.take_snapshot("after")

    result = pm.compare_snapshots("before", "after")
    assert "total_diff_bytes" in result
    assert result["label_a"] == "before"
    assert result["label_b"] == "after"

    pm.stop_tracing()
    del big_list


def test_gc_stats():
    pm = PythonMemoryProfiler()
    stats = pm.gc_stats()
    assert "generation_counts" in stats
    assert "top_types" in stats
    assert stats["total_objects"] > 0


def test_get_current_memory():
    pm = PythonMemoryProfiler()
    # Not tracing yet
    result = pm.get_current_memory()
    assert "error" in result

    pm.start_tracing()
    result = pm.get_current_memory()
    assert "current_bytes" in result
    pm.stop_tracing()


def test_snapshot_not_found():
    pm = PythonMemoryProfiler()
    result = pm.compare_snapshots("nonexistent_a", "nonexistent_b")
    assert "error" in result

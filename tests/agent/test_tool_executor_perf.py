"""Tests for perf/checkpoint tools on ToolExecutor."""

from __future__ import annotations

from unittest.mock import MagicMock

from athena.agent.tool_executor import ToolExecutor


def _make_executor() -> ToolExecutor:
    return ToolExecutor(
        debugger=MagicMock(),
        frame_inspector=MagicMock(),
        evaluator=MagicMock(),
        watch_manager=MagicMock(),
        breakpoint_manager=MagicMock(),
        source_provider=MagicMock(),
        python_memory=MagicMock(),
        cuda_memory=MagicMock(),
        leak_detector=MagicMock(),
    )


def test_perf_checkpoint_tools_are_registered():
    executor = _make_executor()
    names = {
        "create_perf_checkpoint",
        "list_perf_checkpoints",
        "compare_perf_checkpoints",
        "save_checkpoint_state",
        "list_checkpoint_states",
        "load_checkpoint_state",
        "queue_checkpoint_restore",
        "generate_perf_issue_report",
    }
    for name in names:
        assert executor.registry.has_tool(name)


def test_perf_checkpoint_tool_dispatch_calls_session_ops():
    executor = _make_executor()
    calls: dict[str, object] = {}
    executor._source.replace_text_in_file.return_value = {"status": "updated"}

    def _queue_checkpoint_restore(selector="latest"):
        calls["queued"] = selector
        return {"selector": selector}

    executor.bind_session_operations({
        "create_perf_checkpoint": lambda label=None: {"label": label or "x"},
        "list_perf_checkpoints": lambda: [{"label": "a"}],
        "compare_perf_checkpoints": lambda label_a, label_b: {"pair": [label_a, label_b]},
        "save_checkpoint_state": lambda name=None: {"saved": name or "manual"},
        "save_operation_checkpoint": lambda operation, stage="before": calls.update(
            {"operation": operation, "stage": stage}
        ) or {"operation": operation, "stage": stage},
        "list_checkpoint_states": lambda: [{"name": "s1"}],
        "load_checkpoint_state": lambda selector="latest", apply_now=True: {
            "selector": selector,
            "apply_now": apply_now,
        },
        "queue_checkpoint_restore": _queue_checkpoint_restore,
        "generate_perf_issue_report": lambda **kwargs: {"title": kwargs.get("title", "t")},
    })

    queued = executor.registry.execute("queue_checkpoint_restore", {"selector": "checkpoint-a"})
    compared = executor.registry.execute(
        "compare_perf_checkpoints",
        {"label_a": "before", "label_b": "after"},
    )
    loaded = executor.registry.execute(
        "load_checkpoint_state",
        {"selector": "latest", "apply_now": False},
    )
    edited = executor.registry.execute(
        "replace_text_in_file",
        {"filename": "a.py", "old_text": "x", "new_text": "y"},
    )

    assert "checkpoint-a" in queued
    assert calls["queued"] == "checkpoint-a"
    assert "before" in compared and "after" in compared
    assert '"apply_now": false' in loaded.lower()
    assert calls["operation"] == "replace_text_in_file"
    assert calls["stage"] == "before"
    assert '"status": "updated"' in edited

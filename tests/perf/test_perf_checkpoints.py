"""Tests for runtime performance checkpoints in DebugSession."""

from __future__ import annotations

from types import SimpleNamespace

from athena.repl.session import DebugSession
from athena.utils.config import DebugConfig


def _make_session(monkeypatch, tmp_path):
    monkeypatch.setattr("athena.agent.llm_client.OpenAI", lambda **_: SimpleNamespace())
    config = DebugConfig(
        api_key="test",
        state_dir=str(tmp_path / ".athena" / "state"),
        history_file=str(tmp_path / ".athena_history"),
    )
    return DebugSession(config=config)


def test_create_and_list_perf_checkpoints(monkeypatch, tmp_path):
    session = _make_session(monkeypatch, tmp_path)
    session.python_memory.start_tracing()

    first = session.create_perf_checkpoint("before")
    second = session.create_perf_checkpoint("after")
    checkpoints = session.list_perf_checkpoints()

    assert first["label"] == "before"
    assert second["label"] == "after"
    assert len(checkpoints) == 2
    assert checkpoints[0]["label"] == "before"
    assert checkpoints[1]["label"] == "after"


def test_compare_perf_checkpoints_returns_deltas(monkeypatch, tmp_path):
    session = _make_session(monkeypatch, tmp_path)
    session.python_memory.start_tracing()

    session.create_perf_checkpoint("baseline")
    session.create_perf_checkpoint("current")
    result = session.compare_perf_checkpoints("baseline", "current")

    assert "error" not in result
    assert result["label_a"] == "baseline"
    assert result["label_b"] == "current"
    assert result["wall_delta_ns"] >= 0
    assert result["process_delta_ns"] >= 0


def test_compare_perf_checkpoints_missing_label(monkeypatch, tmp_path):
    session = _make_session(monkeypatch, tmp_path)
    result = session.compare_perf_checkpoints("missing-a", "missing-b")
    assert "error" in result

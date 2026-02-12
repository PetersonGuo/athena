"""Tests for perf-specific prompt augmentation."""

from __future__ import annotations

from types import SimpleNamespace

from athena.repl.session import DebugSession
from athena.utils.config import DebugConfig


def _make_session(monkeypatch, tmp_path, perf_mode: bool = False):
    monkeypatch.setattr("athena.agent.llm_client.OpenAI", lambda **_: SimpleNamespace())
    config = DebugConfig(
        api_key="test",
        state_dir=str(tmp_path / ".athena" / "state"),
        history_file=str(tmp_path / ".athena_history"),
        perf_mode=perf_mode,
    )
    return DebugSession(config=config)


def test_perf_prompt_augmentation_includes_checkpoint_workflow(monkeypatch, tmp_path):
    session = _make_session(monkeypatch, tmp_path, perf_mode=False)
    session._post_run_script_path = str(tmp_path / "app.py")
    augmented = session._augment_user_prompt("investigate performance regression")
    assert "list_checkpoint_states" in augmented
    assert "compare_perf_checkpoints" in augmented


def test_perf_mode_forces_perf_workflow_even_without_keywords(monkeypatch, tmp_path):
    session = _make_session(monkeypatch, tmp_path, perf_mode=True)
    session._post_run_script_path = str(tmp_path / "app.py")
    augmented = session._augment_user_prompt("help debug this")
    assert "performance debugging" in augmented.lower()
    assert "create perf checkpoints" in augmented.lower()


def test_non_perf_prompt_keeps_generic_flow(monkeypatch, tmp_path):
    session = _make_session(monkeypatch, tmp_path, perf_mode=False)
    augmented = session._augment_user_prompt("find likely issues")
    assert "static_analyze_file" in augmented
    assert "list_checkpoint_states" not in augmented

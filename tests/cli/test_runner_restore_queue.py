"""Tests for restore selector precedence across reruns."""

from __future__ import annotations

from types import SimpleNamespace

from athena.cli.runner import ScriptRunner


def _build_fake_session_class(queue_on_first: bool):
    class _FakeDebugger:
        def __init__(self):
            self._break_on_exception = False
            self._skip_first_stop = False

        def set_focus_files(self, _files):
            return None

        def set_focus_functions(self, _functions):
            return None

        def run_script(self, *_args, **_kwargs):
            return None

    class _FakeSession:
        instance_index = 0
        load_selectors: list[tuple[int, str]] = []

        def __init__(self, config=None):
            self.config = config
            self.debugger = _FakeDebugger()
            self.breakpoint_manager = SimpleNamespace(add_from_spec=lambda _spec: {})
            self.python_memory = SimpleNamespace(start_tracing=lambda: None)
            self.is_quitting = False
            self.is_rerun_requested = _FakeSession.instance_index == 0
            if _FakeSession.instance_index == 0 and queue_on_first:
                self._queued_selector = "queued-state"
            else:
                self._queued_selector = None
            self._instance = _FakeSession.instance_index
            _FakeSession.instance_index += 1

        def set_active_target(self, _script_path, _script_args):
            return None

        def load_state(self, selector="latest", expected_script_path=None):
            _FakeSession.load_selectors.append((self._instance, selector))
            return {"status": "ok", "path": "/tmp/state.json", "warnings": []}

        def consume_queued_restore_selector(self):
            return self._queued_selector

        def enter_post_run_repl(self, _script_path, _reason):
            return None

    return _FakeSession


def test_runner_prefers_queued_restore_over_initial(monkeypatch, tmp_path):
    fake_session = _build_fake_session_class(queue_on_first=True)
    monkeypatch.setattr("athena.cli.runner.DebugSession", fake_session)

    script = tmp_path / "app.py"
    script.write_text("print('ok')\n")

    ScriptRunner().run(script_path=str(script), restore_selector="cli-state")
    assert fake_session.load_selectors == [(0, "cli-state"), (1, "queued-state")]


def test_runner_falls_back_to_latest_when_no_restore_specified(monkeypatch, tmp_path):
    fake_session = _build_fake_session_class(queue_on_first=False)
    monkeypatch.setattr("athena.cli.runner.DebugSession", fake_session)

    script = tmp_path / "app.py"
    script.write_text("print('ok')\n")

    ScriptRunner().run(script_path=str(script), restore_selector=None)
    assert fake_session.load_selectors == [(1, "latest")]

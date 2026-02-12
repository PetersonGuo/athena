"""Tests for repl commands."""

from __future__ import annotations

from athena.repl.commands import CommandHandler


class _DummySession:
    def __init__(self):
        self.execution_action = None
        self.saved = False
        self.loaded = False
        self.queued = None
        self.operation_checkpoints = []
        self.breakpoint_manager = type("BP", (), {"add_from_spec": lambda _self, _arg: {"filename": "a.py", "lineno": 1}, "list_breakpoints": lambda _self: []})()
        self.watch_manager = type("WM", (), {"add_watch": lambda _self, arg: f"added {arg}", "remove_watch": lambda _self, _arg: True, "evaluate_all": lambda _self: []})()
        self.debugger = type("DBG", (), {"_focus_files": set(), "_focus_functions": set(), "add_focus_file": lambda _self, _f: None, "add_focus_function": lambda _self, _f: None, "clear_focus": lambda _self: None})()

    def set_execution_action(self, action: str) -> None:
        self.execution_action = action

    def save_state(self, name=None, auto=False, reason="manual"):
        self.saved = True
        return {"status": "ok", "path": f"/tmp/{name or 'manual'}.json"}

    def load_state(self, selector="latest"):
        self.loaded = True
        return {"status": "ok", "path": f"/tmp/{selector}.json", "warnings": []}

    def list_states(self):
        return [{
            "created_at": "now",
            "kind": "manual",
            "path": "/tmp/manual.json",
            "name": "manual",
        }]

    def queue_checkpoint_restore(self, selector="latest"):
        self.queued = selector
        return {"status": "queued", "queued_selector": selector, "message": "queued"}

    def create_perf_checkpoint(self, label=None):
        return {
            "label": label or "perf_1",
            "wall_time_ns": 1,
            "process_time_ns": 1,
        }

    def list_perf_checkpoints(self):
        return [
            {
                "label": "baseline",
                "created_at": "now",
                "stop_file": "app.py",
                "stop_line": 10,
                "stop_function": "run",
            }
        ]

    def compare_perf_checkpoints(self, label_a, label_b):
        return {
            "label_a": label_a,
            "label_b": label_b,
            "wall_delta_ms": 1.0,
            "process_delta_ms": 0.5,
            "memory_current_delta_bytes": 10,
            "memory_peak_delta_bytes": 20,
        }

    def generate_perf_issue_report(self, title=None, selector="latest", persist=True):
        return {"title": title or "issue", "path": "/tmp/perf.md", "report": "content"}

    def save_operation_checkpoint(self, operation="op", stage="before"):
        self.operation_checkpoints.append((operation, stage))
        return {"status": "ok"}


def test_rerun_command_sets_execution_action():
    session = _DummySession()
    handler = CommandHandler(session)  # type: ignore[arg-type]
    result = handler.handle("/rerun")
    assert result is None
    assert session.execution_action == "rerun"


def test_save_and_load_state_commands():
    session = _DummySession()
    handler = CommandHandler(session)  # type: ignore[arg-type]

    save_result = handler.handle("/save-state checkpoint-1")
    assert "State saved" in (save_result or "")
    assert session.saved is True

    load_result = handler.handle("/load-state latest")
    assert "State loaded" in (load_result or "")
    assert session.loaded is True


def test_list_states_command():
    session = _DummySession()
    handler = CommandHandler(session)  # type: ignore[arg-type]
    result = handler.handle("/list-states")
    assert "Saved states:" in (result or "")


def test_perf_commands_and_restore_next():
    session = _DummySession()
    handler = CommandHandler(session)  # type: ignore[arg-type]

    queued = handler.handle("/restore-next checkpoint-1")
    assert "queued" in (queued or "")
    assert session.queued == "checkpoint-1"

    checkpoint = handler.handle("/perf-checkpoint before")
    assert "Perf checkpoint 'before'" in (checkpoint or "")

    listed = handler.handle("/perf-checkpoints")
    assert "Perf checkpoints:" in (listed or "")

    compared = handler.handle("/perf-compare baseline current")
    assert "Perf compare baseline -> current" in (compared or "")

    issue = handler.handle("/perf-issue Slow path report")
    assert "/tmp/perf.md" in (issue or "")


def test_mutating_commands_capture_operation_checkpoints():
    session = _DummySession()
    handler = CommandHandler(session)  # type: ignore[arg-type]

    handler.handle("/break app.py:10")
    handler.handle("/watch x")
    handler.handle("/unwatch x")
    handler.handle("/focus app.py")
    handler.handle("/unfocus")

    operations = [name for name, _stage in session.operation_checkpoints]
    assert "command_break" in operations
    assert "command_watch" in operations
    assert "command_unwatch" in operations
    assert "command_focus" in operations
    assert "command_unfocus" in operations

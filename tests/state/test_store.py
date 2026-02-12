"""Tests for persistent state store."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from athena.state.models import (
    AgentStateSummary,
    DebuggerState,
    MemoryStateSummary,
    RuntimeStateSummary,
    SCHEMA_VERSION,
    StateEnvelope,
    StateMeta,
)
from athena.state.store import StateStore


def _make_state(script_path: str, model: str = "m") -> StateEnvelope:
    return StateEnvelope(
        kind="auto",
        reason="test",
        name=None,
        meta=StateMeta(
            schema_version=SCHEMA_VERSION,
            created_at=datetime.now(UTC).isoformat(),
            athena_version="0.1.0",
            python_version="3.x",
            cwd=".",
            script_path=script_path,
            script_args=[],
            model=model,
            script_hash=StateStore.compute_script_hash(script_path),
        ),
        debugger=DebuggerState(),
        runtime=RuntimeStateSummary(stop_reason="test"),
        memory=MemoryStateSummary(),
        agent=AgentStateSummary(),
    )


def test_save_and_load_latest_compatible(tmp_path):
    script = tmp_path / "app.py"
    script.write_text("print('ok')\n")
    store = StateStore(str(tmp_path / ".athena" / "state"), max_auto_per_script=20)
    state = _make_state(str(script), model="model-a")
    path = store.save_auto(state)
    assert path.endswith(".json")

    loaded = store.load("latest", script_path=str(script), model="model-a")
    assert "error" not in loaded
    assert loaded["state"].meta.script_path == str(script)


def test_manual_save_and_lookup_by_name(tmp_path):
    script = tmp_path / "app.py"
    script.write_text("print('ok')\n")
    store = StateStore(str(tmp_path / ".athena" / "state"), max_auto_per_script=20)
    state = _make_state(str(script))
    store.save_manual("checkpoint-a", state)

    loaded = store.load("checkpoint-a", script_path=str(script), model="m")
    assert "error" not in loaded
    assert loaded["state"].meta.script_path == str(script)


def test_prune_auto_keeps_recent(tmp_path):
    script = tmp_path / "app.py"
    script.write_text("print('ok')\n")
    store = StateStore(str(tmp_path / ".athena" / "state"), max_auto_per_script=2)
    for _ in range(4):
        store.save_auto(_make_state(str(script)))
    deleted = store.prune_auto(str(script), keep=2)
    assert deleted >= 2
    states = store.list_states(script_path=str(script))
    autos = [s for s in states if s["kind"] == "auto"]
    assert len(autos) <= 2


def test_load_v1_state_defaults_empty_perf_summary(tmp_path):
    script = tmp_path / "app.py"
    script.write_text("print('ok')\n")
    store = StateStore(str(tmp_path / ".athena" / "state"), max_auto_per_script=20)

    v1_payload = {
        "kind": "manual",
        "reason": "manual",
        "name": "legacy",
        "meta": {
            "schema_version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "athena_version": "0.1.0",
            "python_version": "3.x",
            "cwd": str(tmp_path),
            "script_path": str(script),
            "script_args": [],
            "model": "m",
            "script_hash": StateStore.compute_script_hash(str(script)),
        },
        "debugger": {
            "breakpoints": [],
            "watches": [],
            "focus_files": [],
            "focus_functions": [],
        },
        "runtime": {
            "stop_reason": "legacy",
            "stack": [],
            "locals_summary": {},
        },
        "memory": {},
        "agent": {"conversation_summary": {}},
    }

    state_file = tmp_path / ".athena" / "state" / "legacy.json"
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(v1_payload))

    loaded = store.load(str(state_file), script_path=str(script), model="m")
    assert "error" not in loaded
    assert loaded["state"].perf.checkpoints == []
    assert loaded["state"].perf.last_comparison is None

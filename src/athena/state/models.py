"""Typed models for persisted Athena state."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


SCHEMA_VERSION = 2


@dataclass
class StateMeta:
    schema_version: int
    created_at: str
    athena_version: str
    python_version: str
    cwd: str
    script_path: str
    script_args: list[str]
    model: str
    script_hash: str


@dataclass
class DebuggerBreakpointState:
    file: str
    line: int
    condition: str | None = None
    enabled: bool = True
    temporary: bool = False
    hits: int = 0
    ignore_count: int = 0
    context_snippet: str = ""


@dataclass
class DebuggerState:
    breakpoints: list[DebuggerBreakpointState] = field(default_factory=list)
    watches: list[str] = field(default_factory=list)
    focus_files: list[str] = field(default_factory=list)
    focus_functions: list[str] = field(default_factory=list)


@dataclass
class RuntimeStateSummary:
    stop_reason: str
    stop_file: str | None = None
    stop_line: int | None = None
    stop_function: str | None = None
    stack: list[dict[str, Any]] = field(default_factory=list)
    locals_summary: dict[str, dict[str, str]] = field(default_factory=dict)


@dataclass
class MemoryStateSummary:
    python_current_human: str | None = None
    python_peak_human: str | None = None
    python_snapshot_labels: list[str] = field(default_factory=list)
    cuda_snapshot_labels: list[str] = field(default_factory=list)
    leak_summary: dict[str, Any] | None = None


@dataclass
class AgentStateSummary:
    conversation_summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class PerfCheckpointSummary:
    label: str
    created_at: str
    wall_time_ns: int
    process_time_ns: int
    stop_file: str | None = None
    stop_line: int | None = None
    stop_function: str | None = None
    python_current_bytes: int | None = None
    python_peak_bytes: int | None = None


@dataclass
class PerfStateSummary:
    checkpoints: list[PerfCheckpointSummary] = field(default_factory=list)
    last_comparison: dict[str, Any] | None = None


@dataclass
class StateEnvelope:
    kind: str
    reason: str
    name: str | None
    meta: StateMeta
    debugger: DebuggerState
    runtime: RuntimeStateSummary
    memory: MemoryStateSummary
    agent: AgentStateSummary
    perf: PerfStateSummary = field(default_factory=PerfStateSummary)


def state_to_dict(state: StateEnvelope) -> dict[str, Any]:
    return asdict(state)


def state_from_dict(data: dict[str, Any]) -> StateEnvelope:
    meta = StateMeta(**data["meta"])
    breakpoints = [
        DebuggerBreakpointState(**bp)
        for bp in data.get("debugger", {}).get("breakpoints", [])
    ]
    debugger = DebuggerState(
        breakpoints=breakpoints,
        watches=list(data.get("debugger", {}).get("watches", [])),
        focus_files=list(data.get("debugger", {}).get("focus_files", [])),
        focus_functions=list(data.get("debugger", {}).get("focus_functions", [])),
    )
    runtime = RuntimeStateSummary(**data.get("runtime", {}))
    memory = MemoryStateSummary(**data.get("memory", {}))
    agent = AgentStateSummary(**data.get("agent", {}))
    perf_data = data.get("perf", {}) or {}
    checkpoints = [
        PerfCheckpointSummary(**checkpoint)
        for checkpoint in perf_data.get("checkpoints", [])
    ]
    perf = PerfStateSummary(
        checkpoints=checkpoints,
        last_comparison=perf_data.get("last_comparison"),
    )
    return StateEnvelope(
        kind=data.get("kind", "manual"),
        reason=data.get("reason", "unknown"),
        name=data.get("name"),
        meta=meta,
        debugger=debugger,
        runtime=runtime,
        memory=memory,
        agent=agent,
        perf=perf,
    )

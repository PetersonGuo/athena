"""Filesystem-backed storage for Athena state snapshots."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from athena.state.models import StateEnvelope, state_from_dict, state_to_dict


class StateStore:
    """Persists state snapshots as JSON files under a project-local directory."""

    def __init__(self, state_dir: str, max_auto_per_script: int = 20):
        self._state_dir = Path(state_dir)
        self._max_auto_per_script = max_auto_per_script
        self._state_dir.mkdir(parents=True, exist_ok=True)

    @property
    def state_dir(self) -> Path:
        return self._state_dir

    def save_auto(self, state: StateEnvelope) -> str:
        script_key = _script_key(state.meta.script_path)
        ts = _timestamp_token()
        filename = f"auto_{script_key}_{ts}.json"
        path = self._state_dir / filename
        self._write_state(path, state)
        return str(path)

    def save_manual(self, name: str, state: StateEnvelope) -> str:
        slug = _slug(name or "manual")
        state.name = name
        state.kind = "manual"
        ts = _timestamp_token()
        filename = f"manual_{slug}_{ts}.json"
        path = self._state_dir / filename
        self._write_state(path, state)
        return str(path)

    def load(
        self,
        selector: str,
        script_path: str,
        model: str | None = None,
    ) -> dict[str, Any]:
        selector = (selector or "latest").strip()
        script_path = os.path.abspath(script_path)
        path: Path | None = None
        warnings: list[str] = []

        if selector == "latest":
            entries = self.list_states(script_path=script_path)
            if not entries:
                return {"error": f"No compatible state found for script: {script_path}"}
            path = Path(entries[0]["path"])
        else:
            candidate = Path(selector)
            if candidate.exists():
                path = candidate
            else:
                named = self._find_named(selector, script_path=script_path)
                if named is None:
                    return {"error": f"State not found: {selector}"}
                path = named

        state = self._read_state(path)
        if state is None:
            return {"error": f"Could not parse state file: {path}"}

        if os.path.abspath(state.meta.script_path) != script_path:
            warnings.append(
                "Saved state script_path does not match current script path. "
                "Restore will proceed best-effort."
            )

        if model and state.meta.model != model:
            warnings.append(
                f"Saved model '{state.meta.model}' differs from current model '{model}'."
            )

        current_hash = self.compute_script_hash(script_path)
        if current_hash and state.meta.script_hash and current_hash != state.meta.script_hash:
            warnings.append(
                "Script hash mismatch (code drift detected). Breakpoints will be remapped best-effort."
            )

        return {
            "state": state,
            "path": str(path),
            "warnings": warnings,
            "code_drift": bool(current_hash and state.meta.script_hash and current_hash != state.meta.script_hash),
        }

    def list_states(self, script_path: str | None = None) -> list[dict[str, Any]]:
        states: list[dict[str, Any]] = []
        script_abs = os.path.abspath(script_path) if script_path else None

        for path in sorted(self._state_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            state = self._read_state(path)
            if state is None:
                continue
            if script_abs and os.path.abspath(state.meta.script_path) != script_abs:
                continue
            states.append({
                "path": str(path),
                "kind": state.kind,
                "name": state.name,
                "created_at": state.meta.created_at,
                "script_path": state.meta.script_path,
                "model": state.meta.model,
                "reason": state.reason,
            })
        return states

    def prune_auto(self, script_path: str, keep: int | None = None) -> int:
        keep_n = keep if keep is not None else self._max_auto_per_script
        script_abs = os.path.abspath(script_path)
        auto_files: list[Path] = []
        for path in self._state_dir.glob("auto_*.json"):
            state = self._read_state(path)
            if state is None:
                continue
            if os.path.abspath(state.meta.script_path) == script_abs:
                auto_files.append(path)

        auto_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        deleted = 0
        for path in auto_files[keep_n:]:
            try:
                path.unlink()
                deleted += 1
            except OSError:
                pass
        return deleted

    @staticmethod
    def compute_script_hash(script_path: str) -> str:
        script_abs = os.path.abspath(script_path)
        if not os.path.isfile(script_abs):
            return ""
        h = hashlib.sha256()
        try:
            with open(script_abs, "rb") as f:
                while True:
                    chunk = f.read(8192)
                    if not chunk:
                        break
                    h.update(chunk)
        except OSError:
            return ""
        return h.hexdigest()

    def _find_named(self, name: str, script_path: str) -> Path | None:
        script_abs = os.path.abspath(script_path)
        for entry in self.list_states(script_path=script_abs):
            if entry["name"] == name:
                return Path(entry["path"])
        return None

    def _write_state(self, path: Path, state: StateEnvelope) -> None:
        data = state_to_dict(state)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)

    def _read_state(self, path: Path) -> StateEnvelope | None:
        try:
            with open(path) as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return None
            return state_from_dict(data)
        except Exception:
            return None


def _timestamp_token() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def _slug(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value.strip())
    cleaned = cleaned.strip("-_")
    return cleaned or "state"


def _script_key(script_path: str) -> str:
    base = os.path.basename(script_path) or "script"
    stem = _slug(os.path.splitext(base)[0])
    digest = hashlib.sha1(os.path.abspath(script_path).encode("utf-8")).hexdigest()[:8]
    return f"{stem}_{digest}"

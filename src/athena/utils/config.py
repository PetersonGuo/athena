"""Configuration management for athena."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file


@dataclass
class DebugConfig:
    """Configuration for the debugging agent."""

    api_key: str | None = None
    base_url: str = "https://inference.baseten.co/v1"
    model: str = "zai-org/GLM-4.7"
    max_tool_result_length: int = 8000
    max_conversation_messages: int = 100
    eval_timeout_seconds: float = 10.0
    tracemalloc_nframes: int = 25
    safe_repr_max_depth: int = 3
    safe_repr_max_length: int = 1000
    history_file: str = field(default_factory=lambda: os.path.expanduser("~/.athena_history"))
    state_dir: str = ".athena/state"
    auto_save_state: bool = True
    max_auto_states_per_script: int = 20
    perf_mode: bool = False

    @classmethod
    def from_env(cls) -> DebugConfig:
        """Create config from environment variables."""
        return cls(
            api_key=os.environ.get("BASETEN_API_KEY"),
            base_url=os.environ.get("ATHENA_BASE_URL", "https://inference.baseten.co/v1"),
            model=os.environ.get("ATHENA_MODEL", "zai-org/GLM-4.7"),
            state_dir=os.environ.get("ATHENA_STATE_DIR", ".athena/state"),
            auto_save_state=_parse_bool(os.environ.get("ATHENA_AUTO_SAVE_STATE"), default=True),
            max_auto_states_per_script=_parse_int(
                os.environ.get("ATHENA_MAX_AUTO_STATES_PER_SCRIPT"),
                default=20,
            ),
            perf_mode=_parse_bool(os.environ.get("ATHENA_PERF_MODE"), default=False),
        )


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default

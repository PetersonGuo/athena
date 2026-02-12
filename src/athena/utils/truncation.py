"""Smart truncation utilities for API payloads."""

from __future__ import annotations

import json
from typing import Any


def truncate_string(s: str, max_length: int = 4000) -> str:
    """Truncate a string with an indicator of how much was cut."""
    if len(s) <= max_length:
        return s
    half = max_length // 2 - 50
    return (
        s[:half]
        + f"\n\n... (truncated {len(s) - max_length} chars) ...\n\n"
        + s[-half:]
    )


def truncate_tool_result(result: Any, max_length: int = 8000) -> str:
    """Truncate a tool result for sending back to the API.

    Accepts dicts (serialized to JSON), strings, or other objects.
    """
    if isinstance(result, dict):
        s = json.dumps(result, indent=2, default=str)
    elif isinstance(result, str):
        s = result
    else:
        s = str(result)
    return truncate_string(s, max_length)

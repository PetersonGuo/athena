"""Conversation history management for the LLM API."""

from __future__ import annotations

from typing import Any


class ConversationManager:
    """Manages message history for the OpenAI-compatible API.

    Handles rolling window truncation and tool result size limits
    to stay within context window constraints.
    """

    def __init__(
        self,
        max_messages: int = 100,
        max_tool_result_length: int = 8000,
    ):
        self._messages: list[dict[str, Any]] = []
        self._max_messages = max_messages
        self._max_tool_result_length = max_tool_result_length

    def add_user_message(self, content: str) -> None:
        self._messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: Any) -> None:
        """Add an assistant message. Content can be str or a full message dict."""
        if isinstance(content, dict):
            self._messages.append(content)
        else:
            self._messages.append({"role": "assistant", "content": content})

    def add_tool_result(self, result: dict[str, Any]) -> None:
        """Add a single tool result message (OpenAI format)."""
        self._messages.append(result)

    def add_tool_results(self, results: list[dict[str, Any]]) -> None:
        """Add multiple tool result messages."""
        for result in results:
            self._messages.append(result)

    def get_messages(self) -> list[dict[str, Any]]:
        """Return the message history, trimmed if necessary."""
        if len(self._messages) > self._max_messages:
            keep = self._max_messages - 2
            self._messages = self._messages[:2] + self._messages[-keep:]
        return list(self._messages)

    def clear(self) -> None:
        """Clear all messages."""
        self._messages.clear()

    def truncate_tool_result(self, result: str) -> str:
        """Truncate an overly long tool result."""
        if len(result) <= self._max_tool_result_length:
            return result
        half = self._max_tool_result_length // 2 - 50
        return (
            result[:half]
            + f"\n\n... (truncated {len(result) - self._max_tool_result_length} chars) ...\n\n"
            + result[-half:]
        )

    def export_summary(
        self,
        max_messages: int = 12,
        max_chars_per_message: int = 240,
    ) -> dict[str, Any]:
        """Return a compact summary suitable for persisted state."""
        recent = self._messages[-max_messages:]
        summary_messages: list[dict[str, Any]] = []
        for msg in recent:
            role = msg.get("role", "unknown")
            content = msg.get("content")
            if isinstance(content, str):
                content_preview = content[:max_chars_per_message]
            else:
                content_preview = ""
            tool_calls = msg.get("tool_calls")
            tool_names: list[str] = []
            if isinstance(tool_calls, list):
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    if isinstance(fn, dict) and isinstance(fn.get("name"), str):
                        tool_names.append(fn["name"])
            summary_messages.append({
                "role": role,
                "content_preview": content_preview,
                "tool_calls": tool_names,
            })
        return {
            "message_count": len(self._messages),
            "recent": summary_messages,
        }

    @property
    def message_count(self) -> int:
        return len(self._messages)

"""LLM client using OpenAI-compatible API (Baseten endpoint)."""

from __future__ import annotations

import json
from collections.abc import Generator
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from openai import OpenAI

from athena.agent.conversation import ConversationManager

if TYPE_CHECKING:
    from athena.agent.tool_executor import ToolExecutor


class LLMClient:
    """Manages communication with an OpenAI-compatible API using function calling.

    Handles the tool call loop:
    1. Send user message + tool definitions
    2. Receive response (may contain tool_calls)
    3. Execute tools locally via ToolExecutor
    4. Send tool results back
    5. Repeat until the model responds with text only
    """

    def __init__(
        self,
        model: str = "zai-org/GLM-4.7",
        api_key: str | None = None,
        base_url: str = "https://inference.baseten.co/v1",
        max_tokens: int = 4096,
    ):
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._max_tokens = max_tokens
        self._conversation = ConversationManager()
        self._tool_executor: ToolExecutor | None = None
        self._tool_call_observer: Callable[[str, dict[str, Any]], None] | None = None
        self._system_prompt: str = ""
        self._tool_schema_override: list[dict[str, Any]] | None = None
        self._max_identical_write_calls = 2
        self._write_call_signature_counts: dict[str, int] = {}
        self._restored_summary: dict[str, Any] | None = None

    def set_tool_executor(self, executor: ToolExecutor) -> None:
        self._tool_executor = executor

    def set_tool_call_observer(
        self,
        observer: Callable[[str, dict[str, Any]], None] | None,
    ) -> None:
        """Set an optional callback fired on every tool invocation."""
        self._tool_call_observer = observer

    def set_system_prompt(self, prompt: str) -> None:
        self._system_prompt = prompt

    def set_tool_schema_override(
        self,
        schemas: list[dict[str, Any]] | None,
    ) -> None:
        """Override tool schemas sent to the LLM (e.g. for pre-run filtering).

        Pass None to revert to the default (all tools from the registry).
        """
        self._tool_schema_override = schemas

    def reset_conversation(self) -> None:
        self._conversation.clear()
        self._restored_summary = None

    def set_restored_summary(self, summary: dict[str, Any] | None) -> None:
        self._restored_summary = summary

    def get_conversation_summary(self) -> dict[str, Any]:
        return self._conversation.export_summary()

    def send_message(self, user_message: str) -> str:
        """Send a message and handle the full tool call loop.

        Returns the model's final text response.
        """
        self._reset_tool_call_guards()
        self._conversation.add_user_message(user_message)

        while True:
            messages = self._build_messages()
            kwargs: dict[str, Any] = {
                "model": self._model,
                "max_tokens": self._max_tokens,
                "messages": messages,
            }

            tools = self._get_tools()
            if tools:
                kwargs["tools"] = tools

            response = self._client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            message = choice.message

            # Store assistant response
            self._conversation.add_assistant_message(self._message_to_dict(message))

            if message.tool_calls:
                tool_results = self._process_tool_calls(message.tool_calls)
                for result in tool_results:
                    self._conversation.add_tool_result(result)

                # Check if an execution control action was triggered
                if self._tool_executor and self._tool_executor.get_execution_action():
                    pass  # Continue loop to get final text response
            else:
                return message.content or ""

    def send_message_streaming(self, user_message: str) -> Generator[str, None, str]:
        """Send a message with streaming. Yields text chunks.

        Tool calls are handled between streaming segments.
        Returns the full response text.
        """
        self._reset_tool_call_guards()
        self._conversation.add_user_message(user_message)
        full_response = ""

        while True:
            messages = self._build_messages()
            kwargs: dict[str, Any] = {
                "model": self._model,
                "max_tokens": self._max_tokens,
                "messages": messages,
                "stream": True,
            }

            tools = self._get_tools()
            if tools:
                kwargs["tools"] = tools

            # Collect streaming response
            content_parts: list[str] = []
            tool_calls_accum: dict[int, dict[str, Any]] = {}
            finish_reason = None

            stream = self._client.chat.completions.create(**kwargs)
            for chunk in stream:
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta
                finish_reason = chunk.choices[0].finish_reason

                # Stream text content
                if delta.content:
                    content_parts.append(delta.content)
                    full_response += delta.content
                    yield delta.content

                # Accumulate tool calls
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_accum:
                            tool_calls_accum[idx] = {
                                "id": tc.id or "",
                                "name": "",
                                "arguments": "",
                            }
                        if tc.id:
                            tool_calls_accum[idx]["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                tool_calls_accum[idx]["name"] = tc.function.name
                            if tc.function.arguments:
                                tool_calls_accum[idx]["arguments"] += tc.function.arguments

            # Build the assistant message for conversation history
            assistant_msg: dict[str, Any] = {"role": "assistant"}
            full_content = "".join(content_parts)
            if full_content:
                assistant_msg["content"] = full_content

            if tool_calls_accum:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"],
                        },
                    }
                    for tc in tool_calls_accum.values()
                ]

            self._conversation.add_assistant_message(assistant_msg)

            if finish_reason == "tool_calls" and tool_calls_accum:
                tool_results = self._process_accumulated_tool_calls(tool_calls_accum)
                for result in tool_results:
                    self._conversation.add_tool_result(result)

                # Check for execution control
                if self._tool_executor:
                    action = self._tool_executor.get_execution_action()
                    if action:
                        self._tool_executor._execution_action = action
                        continue
                continue
            else:
                return full_response

    def _build_messages(self) -> list[dict[str, Any]]:
        """Build the messages list including system prompt."""
        messages: list[dict[str, Any]] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        if self._restored_summary:
            messages.append({
                "role": "system",
                "content": (
                    "Restored session summary from previous run. Use as context only; "
                    "verify assumptions with live tools.\n"
                    + json.dumps(self._restored_summary)
                ),
            })
        messages.extend(self._conversation.get_messages())
        return messages

    def _get_tools(self) -> list[dict[str, Any]]:
        if self._tool_schema_override is not None:
            return self._tool_schema_override
        if self._tool_executor:
            return self._tool_executor.registry.get_tool_schemas()
        return []

    def _process_tool_calls(self, tool_calls: list) -> list[dict[str, Any]]:
        """Process tool_calls from a non-streaming response."""
        results = []
        for tc in tool_calls:
            try:
                arguments = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except json.JSONDecodeError:
                arguments = {}

            result = self._execute_tool(tc.function.name, arguments)
            results.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })
        return results

    def _process_accumulated_tool_calls(
        self,
        tool_calls: dict[int, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Process tool calls accumulated during streaming."""
        results = []
        for tc in tool_calls.values():
            try:
                arguments = json.loads(tc["arguments"]) if tc["arguments"] else {}
            except json.JSONDecodeError:
                arguments = {}

            result = self._execute_tool(tc["name"], arguments)
            results.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            })
        return results

    def _execute_tool(self, name: str, input_data: dict[str, Any]) -> str:
        """Execute a single tool and return the result string."""
        if self._tool_executor is None:
            return json.dumps({"error": "No tool executor configured"})
        if self._tool_call_observer:
            self._tool_call_observer(name, input_data)
        if name in {"replace_text_in_file", "replace_file_contents"}:
            signature = self._tool_call_signature(name, input_data)
            count = self._write_call_signature_counts.get(signature, 0) + 1
            self._write_call_signature_counts[signature] = count
            if count > self._max_identical_write_calls:
                return json.dumps({
                    "error": (
                        "Loop guard: repeated identical file-edit call blocked. "
                        "Change tool arguments or inspect runtime state before editing again."
                    ),
                    "tool": name,
                    "repeated_calls": count,
                })
        return self._tool_executor.registry.execute(name, input_data)

    def _reset_tool_call_guards(self) -> None:
        self._write_call_signature_counts.clear()

    @staticmethod
    def _tool_call_signature(name: str, input_data: dict[str, Any]) -> str:
        try:
            payload = json.dumps(input_data, sort_keys=True)
        except TypeError:
            payload = repr(input_data)
        return f"{name}:{payload}"

    @staticmethod
    def _message_to_dict(message: Any) -> dict[str, Any]:
        """Convert an OpenAI message object to a dict for conversation history."""
        msg: dict[str, Any] = {"role": "assistant"}
        if message.content:
            msg["content"] = message.content
        if message.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]
        return msg

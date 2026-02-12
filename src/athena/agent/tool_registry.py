"""Central registry for tools the LLM can invoke."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from athena.utils.truncation import truncate_tool_result


@dataclass
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., Any]


class ToolRegistry:
    """Registry for all debugging tools available to the LLM.

    Each tool has a name, description, input_schema (JSON Schema),
    and a handler function.
    """

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}

    def register(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        handler: Callable[..., Any],
    ) -> None:
        """Register a tool with its schema and handler."""
        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            input_schema=input_schema,
            handler=handler,
        )

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Return all tool schemas in OpenAI function calling format."""
        schemas = []
        for tool in self._tools.values():
            schemas.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
            })
        return schemas

    def execute(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Execute a tool by name, return result as string."""
        tool = self._tools.get(tool_name)
        if tool is None:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

        try:
            result = tool.handler(**tool_input)
        except Exception as e:
            result = {"error": f"{type(e).__name__}: {e}"}

        return truncate_tool_result(result)

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

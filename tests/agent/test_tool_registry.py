"""Tests for tool_registry.py."""

from __future__ import annotations

from athena.agent.tool_registry import ToolRegistry


def test_register_and_list():
    registry = ToolRegistry()
    registry.register(
        "test_tool",
        "A test tool",
        {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]},
        lambda x: {"result": x * 2},
    )
    assert registry.has_tool("test_tool")
    assert "test_tool" in registry.list_tools()


def test_get_tool_schemas():
    registry = ToolRegistry()
    registry.register(
        "tool_a",
        "Tool A",
        {"type": "object", "properties": {}, "required": []},
        lambda: "ok",
    )
    schemas = registry.get_tool_schemas()
    assert len(schemas) == 1
    assert schemas[0]["type"] == "function"
    assert schemas[0]["function"]["name"] == "tool_a"
    assert schemas[0]["function"]["description"] == "Tool A"
    assert schemas[0]["function"]["parameters"]["type"] == "object"


def test_execute_tool():
    registry = ToolRegistry()
    registry.register(
        "add",
        "Add numbers",
        {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]},
        lambda a, b: {"sum": a + b},
    )
    result = registry.execute("add", {"a": 3, "b": 4})
    assert '"sum": 7' in result


def test_execute_unknown_tool():
    registry = ToolRegistry()
    result = registry.execute("nonexistent", {})
    assert "Unknown tool" in result


def test_execute_tool_error():
    registry = ToolRegistry()
    registry.register(
        "fail",
        "Always fails",
        {"type": "object", "properties": {}, "required": []},
        lambda: 1 / 0,
    )
    result = registry.execute("fail", {})
    assert "ZeroDivisionError" in result

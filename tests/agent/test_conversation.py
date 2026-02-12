"""Tests for conversation.py."""

from __future__ import annotations

from athena.agent.conversation import ConversationManager


def test_add_messages():
    cm = ConversationManager()
    cm.add_user_message("Hello")
    cm.add_assistant_message("Hi there")
    messages = cm.get_messages()
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"


def test_truncation():
    cm = ConversationManager(max_messages=4)
    for i in range(10):
        cm.add_user_message(f"msg {i}")
    messages = cm.get_messages()
    assert len(messages) <= 4


def test_clear():
    cm = ConversationManager()
    cm.add_user_message("test")
    cm.clear()
    assert cm.message_count == 0


def test_truncate_tool_result():
    cm = ConversationManager(max_tool_result_length=100)
    short = cm.truncate_tool_result("short text")
    assert short == "short text"

    long = cm.truncate_tool_result("x" * 200)
    assert len(long) < 250  # truncated but includes indicator text
    assert "truncated" in long

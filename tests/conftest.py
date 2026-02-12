"""Shared test fixtures."""

from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock

import pytest

from athena.core.debugger import DebuggerCore
from athena.core.expression_evaluator import ExpressionEvaluator
from athena.core.frame_inspector import FrameInspector
from athena.core.watch_manager import WatchManager


def capture_frame(code_str: str, target_var: str = "_frame") -> types.FrameType:
    """Execute code and capture the frame at the point where _frame is assigned.

    The code should assign sys._getframe() to the target variable.
    """
    namespace: dict[str, Any] = {"sys": sys}
    exec(compile(code_str, "<test>", "exec"), namespace)
    return namespace[target_var]


@pytest.fixture
def debugger():
    """Create a DebuggerCore instance."""
    return DebuggerCore()


@pytest.fixture
def sample_frame():
    """Create a frame with some sample variables for testing."""
    frame = capture_frame(
        """
x = 42
y = "hello"
z = [1, 2, 3]
data = {"key": "value", "num": 123}
_frame = sys._getframe()
"""
    )
    return frame


@pytest.fixture
def debugger_with_frame(debugger, sample_frame):
    """DebuggerCore with a mock current frame set."""
    debugger._current_frame = sample_frame
    return debugger


@pytest.fixture
def frame_inspector(debugger_with_frame):
    """FrameInspector with an active frame."""
    return FrameInspector(debugger_with_frame)


@pytest.fixture
def expression_evaluator(debugger_with_frame):
    """ExpressionEvaluator with an active frame."""
    return ExpressionEvaluator(debugger_with_frame)


@pytest.fixture
def watch_manager(expression_evaluator):
    """WatchManager with an evaluator."""
    return WatchManager(expression_evaluator)


@pytest.fixture
def mock_openai_client():
    """Mock an OpenAI-compatible client to avoid real API calls."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "This is a test response from the model."
    mock_message.tool_calls = None
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client

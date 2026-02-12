"""Watch expression tracking with change detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from athena.utils.safe_repr import safe_repr

if TYPE_CHECKING:
    from athena.core.expression_evaluator import ExpressionEvaluator


@dataclass
class WatchEntry:
    expression: str
    current_value: str = "<not evaluated>"
    previous_value: str | None = None
    current_type: str = ""
    error: str | None = None
    changed: bool = False
    eval_count: int = 0


class WatchManager:
    """Tracks watch expressions across debugger stops.

    Each time the debugger pauses, all watch expressions are re-evaluated
    and compared to their previous values.
    """

    def __init__(self, evaluator: ExpressionEvaluator):
        self._evaluator = evaluator
        self._watches: dict[str, WatchEntry] = {}

    def add_watch(self, expression: str) -> str:
        """Register a watch expression."""
        if expression in self._watches:
            return f"Watch already exists: {expression}"
        self._watches[expression] = WatchEntry(expression=expression)
        return f"Added watch: {expression}"

    def remove_watch(self, expression: str) -> bool:
        """Remove a watch expression."""
        return self._watches.pop(expression, None) is not None

    def get_watch_list(self) -> list[str]:
        """Return list of watched expressions."""
        return list(self._watches.keys())

    def evaluate_all(self, frame_index: int | None = None) -> list[dict[str, Any]]:
        """Evaluate all watches and return results with change detection."""
        results = []
        for expr, entry in self._watches.items():
            result = self._evaluator.evaluate(expr, frame_index)

            entry.previous_value = entry.current_value
            entry.eval_count += 1

            if "error" in result:
                entry.current_value = f"<error: {result['error']}>"
                entry.current_type = ""
                entry.error = result["error"]
                entry.changed = entry.current_value != entry.previous_value
            else:
                entry.current_value = result["result"]
                entry.current_type = result["type"]
                entry.error = None
                entry.changed = (
                    entry.eval_count > 1
                    and entry.current_value != entry.previous_value
                )

            results.append({
                "expression": expr,
                "current_value": entry.current_value,
                "previous_value": entry.previous_value if entry.eval_count > 1 else None,
                "type": entry.current_type,
                "changed": entry.changed,
                "error": entry.error,
            })

        return results

    def get_changes_summary(self) -> str:
        """Return a human-readable summary of what changed since last stop."""
        changes = []
        for expr, entry in self._watches.items():
            if entry.changed:
                changes.append(
                    f"  {expr}: {entry.previous_value} -> {entry.current_value}"
                )
        if not changes:
            return "No watch expressions changed."
        return "Changed watches:\n" + "\n".join(changes)

    def clear_all(self) -> None:
        """Remove all watch expressions."""
        self._watches.clear()

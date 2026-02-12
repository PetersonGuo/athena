"""Safe object representation that never raises exceptions or triggers side effects."""

from __future__ import annotations

from typing import Any


def safe_repr(obj: Any, max_depth: int = 3, max_length: int = 1000, _depth: int = 0) -> str:
    """Produce a repr of obj that is safe and bounded.

    - Never raises an exception (catches __repr__ failures)
    - Truncates long outputs
    - Limits recursion depth
    - Shows type info for truncated objects
    """
    if _depth >= max_depth:
        type_name = type(obj).__qualname__
        try:
            length_info = f", len={len(obj)}" if hasattr(obj, "__len__") else ""
        except Exception:
            length_info = ""
        return f"<{type_name}{length_info} ...>"

    try:
        # For common container types, do element-level truncation
        if isinstance(obj, dict) and _depth < max_depth:
            items = []
            for i, (k, v) in enumerate(obj.items()):
                if i >= 20:
                    items.append(f"... ({len(obj) - 20} more items)")
                    break
                kr = safe_repr(k, max_depth, max_length, _depth + 1)
                vr = safe_repr(v, max_depth, max_length, _depth + 1)
                items.append(f"{kr}: {vr}")
            r = "{" + ", ".join(items) + "}"
        elif isinstance(obj, (list, tuple)) and _depth < max_depth:
            bracket = "[]" if isinstance(obj, list) else "()"
            items = []
            for i, v in enumerate(obj):
                if i >= 20:
                    items.append(f"... ({len(obj) - 20} more items)")
                    break
                items.append(safe_repr(v, max_depth, max_length, _depth + 1))
            r = bracket[0] + ", ".join(items) + bracket[1]
        elif isinstance(obj, set) and _depth < max_depth:
            items = []
            for i, v in enumerate(obj):
                if i >= 20:
                    items.append(f"... ({len(obj) - 20} more items)")
                    break
                items.append(safe_repr(v, max_depth, max_length, _depth + 1))
            r = "{" + ", ".join(items) + "}" if items else "set()"
        else:
            r = repr(obj)

        if len(r) > max_length:
            return r[:max_length] + f"... (truncated, {len(r)} total chars)"
        return r
    except Exception as e:
        type_name = type(obj).__qualname__
        return f"<{type_name} (repr failed: {type(e).__name__}: {e})>"


def safe_type_name(obj: Any) -> str:
    """Get the fully qualified type name of an object, safely."""
    try:
        t = type(obj)
        module = t.__module__
        qualname = t.__qualname__
        if module and module != "builtins":
            return f"{module}.{qualname}"
        return qualname
    except Exception:
        return "<unknown type>"

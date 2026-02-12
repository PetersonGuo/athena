"""Tests for safe_repr utility."""

from __future__ import annotations

from athena.utils.safe_repr import safe_repr, safe_type_name


def test_safe_repr_simple():
    assert safe_repr(42) == "42"
    assert safe_repr("hello") == "'hello'"
    assert safe_repr(None) == "None"


def test_safe_repr_list():
    result = safe_repr([1, 2, 3])
    assert "1" in result
    assert "2" in result
    assert "3" in result


def test_safe_repr_dict():
    result = safe_repr({"a": 1})
    assert "a" in result
    assert "1" in result


def test_safe_repr_depth_limit():
    nested = {"a": {"b": {"c": {"d": "deep"}}}}
    result = safe_repr(nested, max_depth=2)
    assert "..." in result


def test_safe_repr_length_limit():
    long_str = "x" * 2000
    result = safe_repr(long_str, max_length=100)
    assert len(result) < 200
    assert "truncated" in result


def test_safe_repr_large_list():
    big_list = list(range(100))
    result = safe_repr(big_list)
    assert "more items" in result


def test_safe_repr_failing_repr():
    class BadRepr:
        def __repr__(self):
            raise ValueError("nope")

    result = safe_repr(BadRepr())
    assert "repr failed" in result


def test_safe_type_name():
    assert safe_type_name(42) == "int"
    assert safe_type_name("hello") == "str"
    assert safe_type_name([]) == "list"

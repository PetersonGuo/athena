"""Tests for static_analyzer.py."""

from __future__ import annotations

from athena.core.static_analyzer import StaticAnalyzer


def test_static_analyze_file_detects_issues(tmp_path):
    code = """
def bad(a=[]):
    try:
        return 10 / x
    except:
        return 0
"""
    path = tmp_path / "bad.py"
    path.write_text(code.strip() + "\n")

    analyzer = StaticAnalyzer()
    result = analyzer.analyze_file(str(path))

    assert result["issue_count"] >= 3
    kinds = {issue["kind"] for issue in result["issues"]}
    assert "mutable_default" in kinds
    assert "bare_except" in kinds
    assert "possible_zero_division" in kinds


def test_static_analyze_file_missing(tmp_path):
    analyzer = StaticAnalyzer()
    result = analyzer.analyze_file(str(tmp_path / "missing.py"))
    assert "error" in result


def test_static_analyze_source():
    analyzer = StaticAnalyzer()
    result = analyzer.analyze_source(
        "def g(v):\n    return 1 / v\n",
        filename="<paste>",
    )
    assert result["filename"] == "<paste>"
    assert result["issue_count"] >= 1

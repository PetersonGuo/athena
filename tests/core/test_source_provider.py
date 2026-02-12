"""Tests for source_provider.py."""

from __future__ import annotations

from athena.core.source_provider import SourceProvider


def test_write_file_source_updates_existing(tmp_path):
    provider = SourceProvider()
    file_path = tmp_path / "sample.py"
    file_path.write_text("x = 1\n")

    result = provider.write_file_source(str(file_path), "x = 2\n")
    assert result["status"] == "updated"
    assert file_path.read_text() == "x = 2\n"


def test_write_file_source_create_if_missing(tmp_path):
    provider = SourceProvider()
    file_path = tmp_path / "new_file.py"

    result = provider.write_file_source(
        str(file_path),
        "value = 123\n",
        create_if_missing=True,
    )
    assert result["status"] == "created"
    assert file_path.read_text() == "value = 123\n"


def test_replace_text_in_file(tmp_path):
    provider = SourceProvider()
    file_path = tmp_path / "replace_me.py"
    file_path.write_text("a = 1\na = 1\n")

    result = provider.replace_text_in_file(
        str(file_path),
        old_text="a = 1",
        new_text="a = 2",
        max_replacements=1,
    )
    assert result["replacements"] == 1
    assert file_path.read_text() == "a = 2\na = 1\n"


def test_replace_text_in_file_missing_text(tmp_path):
    provider = SourceProvider()
    file_path = tmp_path / "missing.py"
    file_path.write_text("print('hi')\n")

    result = provider.replace_text_in_file(
        str(file_path),
        old_text="not_there",
        new_text="something",
    )
    assert "error" in result
    assert "not found" in result["error"]


def test_find_snippet_lines(tmp_path):
    provider = SourceProvider()
    file_path = tmp_path / "snippet.py"
    file_path.write_text("def f():\n    x = 1\n    return x\n")

    result = provider.find_snippet_lines(
        str(file_path),
        "x = 1\nreturn x",
    )
    assert result["match_count"] == 1
    assert result["matches"][0]["start_line"] == 2
    assert result["matches"][0]["end_line"] == 3


def test_remap_line_by_snippet(tmp_path):
    provider = SourceProvider()
    file_path = tmp_path / "shifted.py"
    file_path.write_text("a = 0\nx = 1\ny = 2\n")

    remapped = provider.remap_line_by_snippet(
        str(file_path),
        preferred_line=10,
        snippet="x = 1",
    )
    assert remapped == 2

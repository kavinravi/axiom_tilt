"""Tests for src.data.clean_filings."""
from pathlib import Path

import pytest

from src.data.clean_filings import (
    extract_sgml_bodies,
    strip_html_tags,
    clean_text,
    process_filing,
    MIN_TEXT_LENGTH,
)


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_extract_sgml_bodies_returns_multiple_blocks():
    raw = (FIXTURES / "sample_sgml.txt").read_text(encoding="utf-8")
    bodies = extract_sgml_bodies(raw)
    assert len(bodies) == 2
    # First body should contain "primary document body"
    assert "primary" in bodies[0]
    # Second body should contain "Exhibit 99.1"
    assert "Exhibit 99.1" in bodies[1]


def test_extract_sgml_bodies_handles_attributes():
    # The regex must match <TEXT> AND <TEXT TYPE="...">
    raw = "<TEXT>plain body</TEXT> middle <TEXT TYPE=\"EX-99\">attr body</TEXT>"
    bodies = extract_sgml_bodies(raw)
    assert len(bodies) == 2
    assert "plain body" in bodies[0]
    assert "attr body" in bodies[1]


def test_extract_sgml_bodies_empty_for_no_text_blocks():
    raw = "<html><body>no SGML envelope</body></html>"
    bodies = extract_sgml_bodies(raw)
    assert bodies == []


def test_strip_html_tags_removes_tags_and_decodes_entities():
    html = "<p>Hello <b>world</b> &amp; friends</p>"
    out = strip_html_tags(html)
    # tags gone
    assert "<" not in out and ">" not in out
    # entity decoded
    assert "&" in out
    # content preserved
    assert "Hello" in out and "world" in out and "friends" in out


def test_strip_html_tags_collapses_whitespace():
    html = "<p>line one\n\nline two</p>   <p>line\tthree</p>"
    out = strip_html_tags(html)
    # Should not contain double-space, tab, or newline
    assert "  " not in out
    assert "\t" not in out
    assert "\n" not in out


def test_clean_text_combines_extraction_and_stripping():
    raw = (FIXTURES / "sample_sgml.txt").read_text(encoding="utf-8")
    text = clean_text(raw)
    # Should contain primary doc content
    assert "primary" in text
    assert "Risk Factors" in text
    # Should NOT contain HTML tags
    assert "<p>" not in text and "<html>" not in text
    # Should be a single string (joined bodies)
    assert isinstance(text, str)


def test_process_filing_writes_output_for_sgml(tmp_path):
    src = tmp_path / "input.txt"
    src.write_text((FIXTURES / "sample_sgml.txt").read_text(encoding="utf-8"))
    dst = tmp_path / "output.txt"
    wrote = process_filing(src, dst)
    assert wrote is True
    assert dst.exists()
    content = dst.read_text(encoding="utf-8")
    assert len(content) >= MIN_TEXT_LENGTH
    assert "primary" in content


def test_process_filing_skips_when_output_exists(tmp_path):
    src = tmp_path / "input.txt"
    src.write_text((FIXTURES / "sample_sgml.txt").read_text(encoding="utf-8"))
    dst = tmp_path / "output.txt"
    dst.write_text("pre-existing non-empty content")  # simulate prior run
    wrote = process_filing(src, dst)
    assert wrote is False  # skipped
    # Existing content preserved
    assert dst.read_text(encoding="utf-8") == "pre-existing non-empty content"


def test_process_filing_drops_short_output(tmp_path):
    # Construct an SGML with a tiny body (below MIN_TEXT_LENGTH)
    src = tmp_path / "tiny.txt"
    src.write_text("<TEXT>short</TEXT>")
    dst = tmp_path / "out.txt"
    wrote = process_filing(src, dst)
    assert wrote is False  # below threshold, not written
    assert not dst.exists()

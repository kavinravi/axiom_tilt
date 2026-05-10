"""Tests for src.data.ingest_filings."""
from pathlib import Path

import pandas as pd
import pytest

from src.data.ingest_filings import (
    parse_master_idx,
    extract_text_from_html,
    accession_from_filename,
    EdgarFiling,
)


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_parse_master_idx_returns_filings_dataframe():
    text = (FIXTURES / "edgar_master.idx").read_text(encoding="latin-1")
    df = parse_master_idx(text)
    assert {"cik", "company", "form_type", "filing_date", "filename"}.issubset(df.columns)
    # CIK should be zero-padded 10-digit string
    assert df["cik"].str.len().eq(10).all()
    # Should have a healthy mix of form types
    assert df["form_type"].nunique() > 5
    assert (df["form_type"] == "10-K").sum() > 0
    assert (df["form_type"] == "10-Q").sum() > 0
    assert (df["form_type"] == "8-K").sum() > 0


def test_parse_master_idx_filters_to_universe_forms():
    text = (FIXTURES / "edgar_master.idx").read_text(encoding="latin-1")
    df = parse_master_idx(text)
    df = df[df["form_type"].isin(["10-K", "10-Q", "8-K"])]
    assert len(df) > 100
    assert df["form_type"].isin(["10-K", "10-Q", "8-K"]).all()


def test_extract_text_from_html_strips_tags_and_returns_text():
    html = (FIXTURES / "sample_10k.html").read_text(encoding="utf-8", errors="ignore")
    text = extract_text_from_html(html)
    assert len(text) > 10_000  # 10-Ks are long
    assert "<" not in text[:1000]  # no raw tags in the cleaned text
    # 10-Ks always contain certain phrases
    assert "Item" in text or "ITEM" in text


def test_accession_from_filename_extracts_correctly():
    fname = "edgar/data/320193/000032019324000123/0000320193-24-000123-index.htm"
    assert accession_from_filename(fname) == "0000320193-24-000123"

    fname2 = "edgar/data/789019/000156459024041223/msft-20240630.htm"
    # Accession is in the second-to-last path segment, in folder form (no dashes)
    # We need to reconstruct from "000156459024041223" -> "0001564590-24-041223"
    assert accession_from_filename(fname2) == "0001564590-24-041223"


def test_edgar_filing_dataclass():
    f = EdgarFiling(
        cik="0000320193",
        company="APPLE INC",
        form_type="10-K",
        filing_date=pd.Timestamp("2024-11-01"),
        filename="edgar/data/320193/000032019324000123/0000320193-24-000123-index.htm",
        accession="0000320193-24-000123",
    )
    assert f.url.startswith("https://www.sec.gov/")
    assert f.local_text_path.name.endswith(".txt")
    assert f.local_raw_path.name == "0000320193-24-000123.htm"

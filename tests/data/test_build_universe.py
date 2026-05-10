"""Tests for src.data.build_universe."""
from pathlib import Path

import pandas as pd

from src.data.build_universe import (
    parse_current_members,
    parse_changes_table,
    load_ticker_to_cik,
    reconstruct_membership,
)


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_parse_current_members_returns_nonempty_dataframe():
    html = (FIXTURES / "wikipedia_sp500.html").read_text(encoding="utf-8")
    df = parse_current_members(html)
    assert len(df) >= 400
    assert {"ticker", "company"}.issubset(df.columns)
    # Tickers should be uppercase and short
    assert all(t.isupper() for t in df["ticker"].head(10))


def test_parse_changes_table_returns_dataframe_with_dates():
    html = (FIXTURES / "wikipedia_sp500.html").read_text(encoding="utf-8")
    df = parse_changes_table(html)
    assert len(df) > 100
    assert {"date", "added_ticker", "removed_ticker"}.issubset(df.columns)
    assert pd.api.types.is_datetime64_any_dtype(df["date"])


def test_load_ticker_to_cik_maps_known_ticker():
    json_path = FIXTURES / "sec_company_tickers.json"
    mapping = load_ticker_to_cik(json_path)
    # Apple's CIK is 320193 — extremely unlikely to ever change
    assert mapping["AAPL"] == "0000320193"
    # CIKs are 10-digit zero-padded strings
    for cik in list(mapping.values())[:20]:
        assert len(cik) == 10
        assert cik.isdigit()


def test_reconstruct_membership_produces_intervals():
    html = (FIXTURES / "wikipedia_sp500.html").read_text(encoding="utf-8")
    cik_map = load_ticker_to_cik(FIXTURES / "sec_company_tickers.json")
    df = reconstruct_membership(
        html=html,
        ticker_to_cik=cik_map,
        start_date="2000-01-01",
        end_date="2025-12-31",
    )
    assert {"ticker", "cik", "date_in", "date_out"}.issubset(df.columns)
    # Every row should have a valid date_in
    assert df["date_in"].notna().all()
    # date_out can be NaT for currently-listed
    # Total distinct tickers should be in the 600-1000 range (current 500 + churn)
    assert 500 <= df["ticker"].nunique() <= 1200
    # No row should have date_in > date_out (when date_out is set)
    valid_out = df.dropna(subset=["date_out"])
    assert (valid_out["date_in"] <= valid_out["date_out"]).all()

"""Tests for src/data/ingest_sharadar.py — Nasdaq Data Link client mocked, no live network."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.data.ingest_sharadar import (
    PIT_DIMENSIONS,
    pull_sharadar_sf1,
    pull_sharadar_tickers,
    resolve_sharadar_tickers,
)


def _secfilings_url(cik: int) -> str:
    return f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik:010d}"


def _sample_tickers() -> pd.DataFrame:
    # SHARADAR/TICKERS has no standalone `cik` column — CIK is embedded in the
    # secfilings EDGAR URL, which is what the module parses.
    return pd.DataFrame(
        {
            "table": ["SF1", "SF1", "SF1"],
            "permaticker": [199059, 199623, 220000],
            "ticker": ["AAPL", "MSFT", "DEFUNCT"],
            "name": ["Apple Inc", "Microsoft Corp", "Defunct Co"],
            "secfilings": [
                _secfilings_url(320193),
                _secfilings_url(789019),
                _secfilings_url(999999),
            ],
            "lastupdated": ["2026-01-01", "2026-01-01", "2010-06-01"],
        }
    )


def _sample_sf1() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL", "MSFT"],
            "dimension": ["ARQ", "ARY", "ARQ"],
            "calendardate": ["2008-12-31", "2008-12-31", "2008-12-31"],
            "datekey": ["2009-01-21", "2009-01-21", "2009-01-22"],
            "reportperiod": ["2008-12-27", "2008-12-27", "2008-12-31"],
            "revenue": [10167000000, 32479000000, 16629000000],
            "netinc": [1605000000, 6119000000, 4174000000],
            "assets": [39572000000, 39572000000, 72793000000],
            "lastupdated": ["2026-01-01", "2026-01-01", "2026-01-01"],
        }
    )


def make_ndl(tables: dict) -> MagicMock:
    """Mock the nasdaqdatalink module. `tables` maps dataset code -> DataFrame."""
    ndl = MagicMock()

    def fake_get_table(code, **kwargs):
        if code not in tables:
            raise AssertionError(f"unexpected get_table code: {code}")
        df = tables[code].copy()
        # Emulate the ticker filter for SF1 pulls
        if code == "SHARADAR/SF1" and "ticker" in kwargs:
            wanted = kwargs["ticker"]
            df = df[df["ticker"].isin(wanted)]
        return df

    ndl.get_table.side_effect = fake_get_table
    return ndl


def test_pull_sharadar_tickers_writes_parquet(tmp_path: Path):
    ndl = make_ndl({"SHARADAR/TICKERS": _sample_tickers()})
    out = tmp_path / "sharadar_tickers.parquet"

    result = pull_sharadar_tickers(ndl, out)

    assert out.exists()
    assert len(result) == 3
    # lastupdated should have been coerced to datetime
    assert pd.api.types.is_datetime64_any_dtype(result["lastupdated"])


def test_pull_sharadar_tickers_derives_cik_from_secfilings(tmp_path: Path):
    """Regression: SHARADAR/TICKERS has no `cik` column — it must be parsed
    out of the secfilings EDGAR URL, or the whole resolve step KeyErrors."""
    ndl = make_ndl({"SHARADAR/TICKERS": _sample_tickers()})
    out = tmp_path / "sharadar_tickers.parquet"

    result = pull_sharadar_tickers(ndl, out)

    assert "cik" in result.columns
    assert result.loc[result["ticker"] == "AAPL", "cik"].iloc[0] == 320193
    assert result.loc[result["ticker"] == "MSFT", "cik"].iloc[0] == 789019


def test_resolve_sharadar_tickers_maps_via_cik():
    universe_ids = pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT"],
            "cik": [320193, 789019],
            "permno": [14593, 10107],
            "gvkey": ["001690", "012141"],
        }
    )
    result = resolve_sharadar_tickers(universe_ids, _sample_tickers())

    # AAPL + MSFT match by CIK; DEFUNCT (cik 999999) is not in our universe
    assert result == ["AAPL", "MSFT"]


def test_resolve_sharadar_tickers_handles_string_cik_in_universe():
    """universe_ids cik may arrive as float/str; resolver coerces both sides."""
    universe_ids = pd.DataFrame({"ticker": ["AAPL"], "cik": [320193.0]})
    result = resolve_sharadar_tickers(universe_ids, _sample_tickers())
    assert result == ["AAPL"]


def test_resolve_sharadar_tickers_raises_on_zero_match():
    universe_ids = pd.DataFrame({"ticker": ["NOPE"], "cik": [11111111]})
    with pytest.raises(RuntimeError, match="No universe CIKs matched"):
        resolve_sharadar_tickers(universe_ids, _sample_tickers())


def test_pull_sharadar_sf1_filters_and_writes(tmp_path: Path):
    ndl = make_ndl({"SHARADAR/SF1": _sample_sf1()})
    out = tmp_path / "sharadar_sf1.parquet"

    pull_sharadar_sf1(ndl, ["AAPL", "MSFT"], out)

    assert out.exists()
    df = pd.read_parquet(out)
    assert len(df) == 3
    assert pd.api.types.is_datetime64_any_dtype(df["datekey"])
    # GFC-era data present — the whole point of using Sharadar over XBRL
    assert df["datekey"].min() == pd.Timestamp("2009-01-21")
    assert df["calendardate"].min() == pd.Timestamp("2008-12-31")


def test_pull_sharadar_sf1_requests_only_pit_dimensions(tmp_path: Path):
    ndl = make_ndl({"SHARADAR/SF1": _sample_sf1()})
    out = tmp_path / "sharadar_sf1.parquet"

    pull_sharadar_sf1(ndl, ["AAPL"], out)

    # Verify the get_table call asked for ARQ/ARY only, never MR* (restated)
    call_kwargs = ndl.get_table.call_args.kwargs
    assert call_kwargs["dimension"] == PIT_DIMENSIONS
    assert "MRQ" not in call_kwargs["dimension"]
    assert "MRY" not in call_kwargs["dimension"]


def test_pull_sharadar_sf1_chunks_large_ticker_lists(tmp_path: Path):
    ndl = make_ndl({"SHARADAR/SF1": _sample_sf1()})
    out = tmp_path / "sharadar_sf1.parquet"

    # 250 tickers with chunk_size 100 -> 3 get_table calls
    many = [f"T{i}" for i in range(250)]
    pull_sharadar_sf1(ndl, many, out, chunk_size=100)

    assert ndl.get_table.call_count == 3

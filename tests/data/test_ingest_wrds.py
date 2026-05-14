"""Tests for src/data/ingest_wrds.py — all WRDS calls mocked, no live network."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.data.ingest_wrds import (
    pull_crsp_daily,
    resolve_universe_ids,
)


def make_conn(query_responses: dict[str, pd.DataFrame]) -> MagicMock:
    """Mock wrds.Connection whose raw_sql returns a DataFrame matching a substring of the SQL.

    Order of dict insertion matters when multiple keywords could match.
    """
    conn = MagicMock()

    def fake_raw_sql(sql: str, date_cols=None):
        for keyword, df in query_responses.items():
            if keyword in sql:
                return df.copy()
        raise AssertionError(f"No mock match for SQL: {sql[:200]}")

    conn.raw_sql.side_effect = fake_raw_sql
    return conn


# --- resolve_universe_ids ---------------------------------------------------


def test_resolve_universe_ids_basic():
    universe = pd.DataFrame(
        {
            "ticker": ["AAPL", "META"],
            "cik": [320193, 1326801],
            "company": ["Apple", "Meta"],
            "date_in": [pd.Timestamp("2000-01-01"), pd.Timestamp("2013-12-23")],
            "date_out": [pd.Timestamp("2025-12-31"), pd.Timestamp("2025-12-31")],
        }
    )
    stocknames = pd.DataFrame(
        {
            "permno": [14593, 13407],
            "ticker": ["AAPL", "META"],
            "namedt": [pd.Timestamp("1980-12-12"), pd.Timestamp("2022-06-09")],
            "nameenddt": [pd.NaT, pd.NaT],
        }
    )

    conn = make_conn({"crsp.stocknames": stocknames})
    result = resolve_universe_ids(universe, conn)

    assert len(result) == 2
    assert {"ticker", "permno"}.issubset(result.columns)
    assert "gvkey" not in result.columns  # fundamentals path removed
    apple = result[result["ticker"] == "AAPL"].iloc[0]
    assert apple["permno"] == 14593
    meta = result[result["ticker"] == "META"].iloc[0]
    assert meta["permno"] == 13407


def test_resolve_universe_ids_low_match_raises():
    universe = pd.DataFrame(
        {
            "ticker": ["AAPL", "ZZZZ", "YYYY", "XXXX", "WWWW"],
            "cik": [320193, 0, 0, 0, 0],
            "company": ["Apple", "", "", "", ""],
            "date_in": [pd.Timestamp("2000-01-01")] * 5,
            "date_out": [pd.Timestamp("2025-12-31")] * 5,
        }
    )
    stocknames = pd.DataFrame(
        {
            "permno": [14593],
            "ticker": ["AAPL"],
            "namedt": [pd.Timestamp("1980-12-12")],
            "nameenddt": [pd.NaT],
        }
    )

    conn = make_conn({"crsp.stocknames": stocknames})
    with pytest.raises(RuntimeError, match="95"):
        resolve_universe_ids(universe, conn)


def test_resolve_universe_ids_ticker_reuse_picks_largest_overlap():
    """A ticker can be reused across periods (e.g., GOOG -> dormant -> different company).
    Universe row should resolve to the permno whose validity overlaps it the most.
    """
    universe = pd.DataFrame(
        {
            "ticker": ["GOOG"],
            "cik": [1652044],
            "company": ["Alphabet"],
            "date_in": [pd.Timestamp("2015-08-25")],
            "date_out": [pd.Timestamp("2025-12-31")],
        }
    )
    stocknames = pd.DataFrame(
        {
            "permno": [11111, 90319],
            "ticker": ["GOOG", "GOOG"],
            "namedt": [pd.Timestamp("1995-01-01"), pd.Timestamp("2014-04-03")],
            "nameenddt": [pd.Timestamp("2004-08-18"), pd.NaT],  # first one ends, second is current
        }
    )

    conn = make_conn({"crsp.stocknames": stocknames})
    result = resolve_universe_ids(universe, conn)
    assert len(result) == 1
    assert result.iloc[0]["permno"] == 90319  # current Alphabet, not the dormant one


# --- pull_crsp_daily --------------------------------------------------------


def test_pull_crsp_daily_basic(tmp_path: Path):
    dsf = pd.DataFrame(
        {
            "permno": [14593, 14593],
            "date": [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")],
            "prc": [75.0, 74.5],
            "ret": [0.01, -0.005],
            "vol": [1000, 1100],
            "shrout": [10000, 10000],
            "openprc": [74.0, 75.0],
            "askhi": [76.0, 75.5],
            "bidlo": [73.0, 74.0],
            "cfacpr": [1.0, 1.0],
            "cfacshr": [1.0, 1.0],
        }
    )
    msedelist = pd.DataFrame(
        {"permno": [], "date": [], "dlret": [], "dlstcd": []}
    )

    conn = make_conn({"crsp.msedelist": msedelist, "crsp.dsf": dsf})
    pull_crsp_daily(conn, [14593], "2020-01-01", "2020-12-31", tmp_path)

    out = tmp_path / "year=2020" / "part-0.parquet"
    assert out.exists()
    df = pd.read_parquet(out)
    assert len(df) == 2
    assert "dlret" in df.columns
    assert df["dlret"].isna().all()


def test_pull_crsp_daily_merges_delisting(tmp_path: Path):
    dsf = pd.DataFrame(
        {
            "permno": [14593],
            "date": [pd.Timestamp("2020-06-30")],
            "prc": [50.0],
            "ret": [-0.01],
            "vol": [1000],
            "shrout": [10000],
            "openprc": [50.0],
            "askhi": [51.0],
            "bidlo": [49.0],
            "cfacpr": [1.0],
            "cfacshr": [1.0],
        }
    )
    msedelist = pd.DataFrame(
        {
            "permno": [14593],
            "date": [pd.Timestamp("2020-06-30")],
            "dlret": [-0.30],
            "dlstcd": [500],
        }
    )

    conn = make_conn({"crsp.msedelist": msedelist, "crsp.dsf": dsf})
    pull_crsp_daily(conn, [14593], "2020-01-01", "2020-12-31", tmp_path)

    df = pd.read_parquet(tmp_path / "year=2020" / "part-0.parquet")
    row = df.iloc[0]
    assert row["dlret"] == -0.30
    assert row["dlstcd"] == 500


def test_pull_crsp_daily_resume_skips_existing(tmp_path: Path):
    """If year partition already exists, skip the pull entirely."""
    existing = tmp_path / "year=2020"
    existing.mkdir(parents=True)
    (existing / "part-0.parquet").touch()

    conn = MagicMock()
    pull_crsp_daily(conn, [14593], "2020-01-01", "2020-12-31", tmp_path)
    conn.raw_sql.assert_not_called()

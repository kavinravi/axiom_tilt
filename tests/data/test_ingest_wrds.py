"""Tests for src/data/ingest_wrds.py — all WRDS calls mocked, no live network."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.data.ingest_wrds import (
    pull_ccm_linktable,
    pull_compustat_funda,
    pull_compustat_fundq,
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
    lnkhist = pd.DataFrame(
        {
            "gvkey": ["001690", "170617"],
            "permno": [14593, 13407],
            "linktype": ["LU", "LU"],
            "linkprim": ["P", "P"],
            "linkdt": [pd.Timestamp("1980-12-12"), pd.Timestamp("2012-05-18")],
            "linkenddt": [pd.NaT, pd.NaT],
        }
    )

    conn = make_conn({"crsp.stocknames": stocknames, "ccmxpf_lnkhist": lnkhist})
    result = resolve_universe_ids(universe, conn)

    assert len(result) == 2
    assert {"ticker", "permno", "gvkey"}.issubset(result.columns)
    apple = result[result["ticker"] == "AAPL"].iloc[0]
    assert apple["permno"] == 14593
    assert apple["gvkey"] == "001690"
    meta = result[result["ticker"] == "META"].iloc[0]
    assert meta["permno"] == 13407
    assert meta["gvkey"] == "170617"


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
    lnkhist = pd.DataFrame(
        {
            "gvkey": ["001690"],
            "permno": [14593],
            "linktype": ["LU"],
            "linkprim": ["P"],
            "linkdt": [pd.Timestamp("1980-12-12")],
            "linkenddt": [pd.NaT],
        }
    )

    conn = make_conn({"crsp.stocknames": stocknames, "ccmxpf_lnkhist": lnkhist})
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
    lnkhist = pd.DataFrame(
        {
            "gvkey": ["160329"],
            "permno": [90319],
            "linktype": ["LU"],
            "linkprim": ["P"],
            "linkdt": [pd.Timestamp("2014-04-03")],
            "linkenddt": [pd.NaT],
        }
    )

    conn = make_conn({"crsp.stocknames": stocknames, "ccmxpf_lnkhist": lnkhist})
    result = resolve_universe_ids(universe, conn)
    assert len(result) == 1
    assert result.iloc[0]["permno"] == 90319  # current Alphabet, not the dormant one
    assert result.iloc[0]["gvkey"] == "160329"


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


# --- Compustat pulls --------------------------------------------------------


def test_pull_compustat_funda_uses_pit_filters(tmp_path: Path):
    expected = pd.DataFrame(
        {
            "gvkey": ["001690"],
            "datadate": [pd.Timestamp("2020-09-26")],
            "rdq": [pd.Timestamp("2020-10-29")],
            "at": [323888.0],
        }
    )
    conn = make_conn({"comp.funda": expected})
    out = tmp_path / "comp_funda.parquet"

    pull_compustat_funda(conn, ["001690"], "1995-01-01", "2025-12-31", out)

    sql_arg = conn.raw_sql.call_args[0][0]
    assert "consol='C'" in sql_arg
    assert "indfmt='INDL'" in sql_arg
    assert "datafmt='STD'" in sql_arg
    assert "popsrc='D'" in sql_arg
    assert "curcd='USD'" in sql_arg
    assert "comp.funda" in sql_arg

    df = pd.read_parquet(out)
    assert len(df) == 1
    assert df["at"].iloc[0] == 323888.0


def test_pull_compustat_fundq_uses_pit_filters(tmp_path: Path):
    expected = pd.DataFrame(
        {
            "gvkey": ["001690"],
            "datadate": [pd.Timestamp("2020-09-26")],
            "rdq": [pd.Timestamp("2020-10-29")],
            "atq": [323888.0],
        }
    )
    conn = make_conn({"comp.fundq": expected})
    out = tmp_path / "comp_fundq.parquet"

    pull_compustat_fundq(conn, ["001690"], "1995-01-01", "2025-12-31", out)

    sql_arg = conn.raw_sql.call_args[0][0]
    assert "consol='C'" in sql_arg
    assert "indfmt='INDL'" in sql_arg
    assert "comp.fundq" in sql_arg


# --- pull_ccm_linktable -----------------------------------------------------


def test_pull_ccm_linktable_writes_parquet(tmp_path: Path):
    expected = pd.DataFrame(
        {
            "gvkey": ["001690"],
            "lpermno": [14593],
            "lpermco": [9999],
            "linktype": ["LU"],
            "linkprim": ["P"],
            "linkdt": [pd.Timestamp("1980-12-12")],
            "linkenddt": [pd.NaT],
        }
    )
    conn = make_conn({"ccmxpf_linktable": expected})
    out = tmp_path / "ccm_linktable.parquet"

    pull_ccm_linktable(conn, out)

    assert out.exists()
    df = pd.read_parquet(out)
    assert len(df) == 1
    assert df["gvkey"].iloc[0] == "001690"

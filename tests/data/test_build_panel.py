"""Tests for src/data/build_panel.py — the leakage-sensitive PIT join.

The single most important test here is the leakage guard: no panel row may ever
see a fundamental whose filing date (`datekey`) is after the trading `date`.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.build_panel import (
    build_panel,
    build_ticker_cik_map,
    flag_universe_membership,
    pit_join,
    resolve_sf1_permno,
    strike_no_fundamentals,
)


def _crsp() -> pd.DataFrame:
    """Two permnos, daily bars across early 2009."""
    dates = pd.date_range("2009-01-01", "2009-03-31", freq="B")
    frames = []
    for permno in (101, 202):
        frames.append(
            pd.DataFrame(
                {
                    "permno": permno,
                    "date": dates,
                    "prc": 10.0 + permno,
                    "ret": 0.001,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _sf1() -> pd.DataFrame:
    """One ARQ filing per ticker; AAPL filed 2009-01-21, MSFT 2009-02-15."""
    return pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT"],
            "dimension": ["ARQ", "ARQ"],
            "datekey": pd.to_datetime(["2009-01-21", "2009-02-15"]),
            "calendardate": pd.to_datetime(["2008-12-31", "2008-12-31"]),
            "revenue": [10_167_000_000, 16_629_000_000],
        }
    )


def _sharadar_tickers() -> pd.DataFrame:
    return pd.DataFrame({"ticker": ["AAPL", "MSFT"], "cik": [320193, 789019]})


def _universe_ids() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT"],
            "cik": ["0000320193", "0000789019"],
            "permno": pd.array([101, 202], dtype="Int64"),
            "date_in": pd.to_datetime(["2009-02-01", "2009-01-01"]),
            "date_out": pd.to_datetime([None, None]),
        }
    )


def test_build_ticker_cik_map_one_row_per_ticker():
    m = build_ticker_cik_map(_sharadar_tickers())
    assert len(m) == 2
    assert m.set_index("ticker")["cik"].to_dict() == {"AAPL": 320193, "MSFT": 789019}


def test_resolve_sf1_permno_maps_via_cik():
    sf1 = resolve_sf1_permno(_sf1(), build_ticker_cik_map(_sharadar_tickers()), _universe_ids())
    assert set(sf1["permno"]) == {101, 202}
    assert sf1["permno"].dtype == "int64"


def test_resolve_sf1_permno_drops_unknown_cik():
    """An SF1 ticker whose cik is not in the universe is dropped, not kept with NaN."""
    sf1 = _sf1()
    sf1.loc[len(sf1)] = {"ticker": "NOPE", "dimension": "ARQ",
                         "datekey": pd.Timestamp("2009-01-10"),
                         "calendardate": pd.Timestamp("2008-12-31"), "revenue": 1}
    tickers = _sharadar_tickers()
    tickers.loc[len(tickers)] = {"ticker": "NOPE", "cik": 99999999}
    resolved = resolve_sf1_permno(sf1, build_ticker_cik_map(tickers), _universe_ids())
    assert len(resolved) == 2
    assert "NOPE" not in set(resolved["ticker"])


def test_pit_join_is_backward_only_no_leakage():
    """Every joined row must have datekey <= date — the core leakage guard."""
    sf1 = resolve_sf1_permno(_sf1(), build_ticker_cik_map(_sharadar_tickers()), _universe_ids())
    panel = pit_join(_crsp(), sf1)
    joined = panel[panel["datekey"].notna()]
    assert len(joined) > 0
    assert (joined["datekey"] <= joined["date"]).all()


def test_pit_join_rows_before_first_filing_have_no_fundamentals():
    """permno 101 (AAPL) filed 2009-01-21 — Jan 1-20 trading days must be NaN."""
    sf1 = resolve_sf1_permno(_sf1(), build_ticker_cik_map(_sharadar_tickers()), _universe_ids())
    panel = pit_join(_crsp(), sf1)
    early = panel[(panel["permno"] == 101) & (panel["date"] < pd.Timestamp("2009-01-21"))]
    assert early["datekey"].isna().all()
    later = panel[(panel["permno"] == 101) & (panel["date"] >= pd.Timestamp("2009-01-21"))]
    assert later["datekey"].notna().all()
    assert (later["revenue"] == 10_167_000_000).all()


def test_strike_no_fundamentals_drops_permnos_with_zero_filings():
    """A permno that never matches any SF1 row is struck entirely, not kept price-only."""
    sf1 = resolve_sf1_permno(_sf1(), build_ticker_cik_map(_sharadar_tickers()), _universe_ids())
    crsp = _crsp()
    # permno 303 exists in CRSP but has no SF1 fundamentals
    extra = crsp[crsp["permno"] == 101].copy()
    extra["permno"] = 303
    crsp = pd.concat([crsp, extra], ignore_index=True)

    panel = pit_join(crsp, sf1)
    assert 303 in set(panel["permno"])
    struck = strike_no_fundamentals(panel)
    assert 303 not in set(struck["permno"])
    assert {101, 202} == set(struck["permno"])


def test_flag_universe_membership():
    """AAPL joins the universe 2009-02-01; rows before that are out-of-universe."""
    sf1 = resolve_sf1_permno(_sf1(), build_ticker_cik_map(_sharadar_tickers()), _universe_ids())
    panel = flag_universe_membership(pit_join(_crsp(), sf1), _universe_ids())
    aapl_early = panel[(panel["permno"] == 101) & (panel["date"] < pd.Timestamp("2009-02-01"))]
    aapl_late = panel[(panel["permno"] == 101) & (panel["date"] >= pd.Timestamp("2009-02-01"))]
    assert not aapl_early["in_universe"].any()
    assert aapl_late["in_universe"].all()
    # MSFT is in from day one
    assert panel[panel["permno"] == 202]["in_universe"].all()


def test_build_panel_end_to_end_writes_partitioned(tmp_path: Path):
    crsp_dir = tmp_path / "crsp_daily"
    crsp = _crsp()
    crsp["year"] = crsp["date"].dt.year
    crsp.to_parquet(crsp_dir, partition_cols=["year"], index=False)

    sf1_path = tmp_path / "sharadar_sf1.parquet"
    # include an ARY row that must be filtered out
    sf1 = _sf1()
    sf1.loc[len(sf1)] = {"ticker": "AAPL", "dimension": "ARY",
                         "datekey": pd.Timestamp("2009-01-21"),
                         "calendardate": pd.Timestamp("2008-12-31"), "revenue": 999}
    sf1.to_parquet(sf1_path, index=False)

    tickers_path = tmp_path / "sharadar_tickers.parquet"
    _sharadar_tickers().to_parquet(tickers_path, index=False)
    universe_path = tmp_path / "universe_ids.parquet"
    _universe_ids().to_parquet(universe_path, index=False)

    out_dir = tmp_path / "panel"
    panel = build_panel(crsp_dir, sf1_path, tickers_path, universe_path, out_dir)

    assert (out_dir / "year=2009").exists()
    # ARY row filtered: no revenue == 999 anywhere
    assert (panel["revenue"] != 999).all()
    # leakage guard holds end-to-end
    joined = panel[panel["datekey"].notna()]
    assert (joined["datekey"] <= joined["date"]).all()
    # fund_age_days is non-negative wherever fundamentals are present
    assert (joined["fund_age_days"] >= 0).all()
    roundtrip = pd.read_parquet(out_dir)
    assert len(roundtrip) == len(panel)


def test_build_panel_raises_on_leakage(tmp_path: Path, monkeypatch):
    """If pit_join ever returned a future-dated fundamental, build_panel must abort."""
    crsp_dir = tmp_path / "crsp_daily"
    crsp = _crsp()
    crsp["year"] = crsp["date"].dt.year
    crsp.to_parquet(crsp_dir, partition_cols=["year"], index=False)
    sf1_path = tmp_path / "sharadar_sf1.parquet"
    _sf1().to_parquet(sf1_path, index=False)
    tickers_path = tmp_path / "sharadar_tickers.parquet"
    _sharadar_tickers().to_parquet(tickers_path, index=False)
    universe_path = tmp_path / "universe_ids.parquet"
    _universe_ids().to_parquet(universe_path, index=False)

    def _leaky_join(crsp_df, sf1_df):
        panel = pit_join(crsp_df, sf1_df)
        panel["datekey"] = panel["date"] + pd.Timedelta(days=30)  # force future dates
        return panel

    monkeypatch.setattr("src.data.build_panel.pit_join", _leaky_join)
    with pytest.raises(RuntimeError, match="LEAKAGE"):
        build_panel(crsp_dir, sf1_path, tickers_path, universe_path, tmp_path / "panel")

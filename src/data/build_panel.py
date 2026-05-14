"""Build the unified point-in-time panel: CRSP daily prices + Sharadar fundamentals.

Output:
  data/processed/panel/year=YYYY/*.parquet   one row per (permno, trading day)

The join is the leakage-sensitive step. Each trading day carries the most
recently *filed* fundamentals — Sharadar's `datekey` (the filing date) must be
<= the trading `date`. This is a backward `merge_asof`: nothing from the future
can leak into a row. The leakage guard in tests/data/test_build_panel.py asserts
this invariant holds.

Crosswalk chain (Sharadar and CRSP use different IDs):
  SF1.ticker --(sharadar_tickers)--> cik --(universe_ids)--> permno
The cik->permno step is date-aware: a handful of CIKs map to more than one
permno over time, so we match on the universe membership window when possible
and fall back to the earliest window otherwise.

Fundamentals dimension: ARQ only (As-Reported Quarterly) — the finest-grained
point-in-time view. ARY (annual) stays in sharadar_sf1.parquet if ever needed.

Usage:
  python -m src.data.build_panel
"""
from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.utils.io import processed_dir, repo_root
from src.utils.logging_utils import configure_logging, get_logger


log = get_logger(__name__)

_FAR_FUTURE = pd.Timestamp("2100-01-01")
PANEL_DIMENSION = "ARQ"


def load_crsp_daily(crsp_dir: Path) -> pd.DataFrame:
    """Read every year=YYYY partition of the CRSP daily pull into one frame."""
    files = sorted(crsp_dir.glob("year=*/*.parquet"))
    if not files:
        raise SystemExit(
            f"No CRSP daily partitions under {crsp_dir} — run `python -m src.data.ingest_wrds --crsp-only` first."
        )
    crsp = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    crsp["permno"] = crsp["permno"].astype("int64")
    crsp["date"] = pd.to_datetime(crsp["date"])
    log.info("CRSP daily: %d rows, %d permnos, %s -> %s",
             len(crsp), crsp["permno"].nunique(), crsp["date"].min().date(), crsp["date"].max().date())
    return crsp


def build_ticker_cik_map(sharadar_tickers: pd.DataFrame) -> pd.DataFrame:
    """Sharadar ticker -> integer cik. One row per Sharadar ticker."""
    m = sharadar_tickers[["ticker", "cik"]].dropna(subset=["ticker", "cik"]).copy()
    m["cik"] = pd.to_numeric(m["cik"], errors="coerce")
    return m.dropna(subset=["cik"]).drop_duplicates("ticker")


def resolve_sf1_permno(
    sf1: pd.DataFrame,
    ticker_cik_map: pd.DataFrame,
    universe_ids: pd.DataFrame,
) -> pd.DataFrame:
    """Attach a `permno` to every SF1 row via ticker -> cik -> permno.

    The cik -> permno step prefers the universe membership window that contains
    the filing date; rows outside any window fall back to the cik's earliest
    window. SF1 rows whose cik is not in the universe at all are dropped.
    """
    sf1 = sf1.merge(ticker_cik_map, on="ticker", how="left").reset_index(drop=True)
    sf1["_rid"] = range(len(sf1))

    xwalk = (
        universe_ids[["cik", "permno", "date_in", "date_out"]]
        .dropna(subset=["cik", "permno"])
        .copy()
    )
    xwalk["cik"] = pd.to_numeric(xwalk["cik"], errors="coerce")
    xwalk["permno"] = xwalk["permno"].astype("int64")

    cand = sf1[["_rid", "cik", "datekey"]].merge(xwalk, on="cik", how="left")
    end = cand["date_out"].fillna(_FAR_FUTURE)
    in_window = (cand["datekey"] >= cand["date_in"]) & (cand["datekey"] <= end)
    primary = (
        cand[in_window].drop_duplicates("_rid").set_index("_rid")["permno"]
    )
    fallback = (
        xwalk.sort_values("date_in").drop_duplicates("cik").set_index("cik")["permno"]
    )

    permno = sf1["_rid"].map(primary)
    need_fb = permno.isna()
    permno = permno.where(~need_fb, sf1["cik"].map(fallback))
    sf1["permno"] = permno

    n_total = len(sf1)
    sf1 = sf1.dropna(subset=["permno"]).copy()
    sf1["permno"] = sf1["permno"].astype("int64")
    sf1 = sf1.drop(columns="_rid")

    log.info(
        "Crosswalk: %d/%d SF1 rows resolved to a permno (%.1f%%); %d dropped (cik not in universe)",
        len(sf1), n_total, 100.0 * len(sf1) / n_total, n_total - len(sf1),
    )
    return sf1


def pit_join(crsp: pd.DataFrame, sf1: pd.DataFrame) -> pd.DataFrame:
    """Backward merge_asof: each (permno, date) gets the latest SF1 row with datekey <= date."""
    crsp = crsp.sort_values("date")
    sf1 = sf1.sort_values("datekey")
    panel = pd.merge_asof(
        crsp, sf1, left_on="date", right_on="datekey", by="permno", direction="backward"
    )
    return panel.reset_index(drop=True)


def strike_no_fundamentals(panel: pd.DataFrame) -> pd.DataFrame:
    """Drop every permno that never carries a single fundamental row.

    Sharadar SF1 doesn't cover the whole survivorship-bias-free universe — mostly
    old delisted names. Rather than keep price-only rows for those, we strike
    them entirely. This reintroduces some survivorship bias (a known, accepted
    trade-off) but keeps every retained name feature-complete.
    """
    keep_permnos = panel.loc[panel["datekey"].notna(), "permno"].unique()
    kept = panel[panel["permno"].isin(keep_permnos)].copy()
    n_before, n_after = panel["permno"].nunique(), kept["permno"].nunique()
    log.info(
        "Struck %d/%d permnos with zero fundamentals (kept %d); %d -> %d rows",
        n_before - n_after, n_before, n_after, len(panel), len(kept),
    )
    return kept


def flag_universe_membership(panel: pd.DataFrame, universe_ids: pd.DataFrame) -> pd.DataFrame:
    """Add an `in_universe` bool: was this permno an index member on this date?"""
    windows = (
        universe_ids[["permno", "date_in", "date_out"]].dropna(subset=["permno"]).copy()
    )
    windows["permno"] = windows["permno"].astype("int64")

    cand = panel[["permno", "date"]].reset_index().merge(windows, on="permno", how="left")
    end = cand["date_out"].fillna(_FAR_FUTURE)
    cand["_in"] = (cand["date"] >= cand["date_in"]) & (cand["date"] <= end)
    flag = cand.groupby("index")["_in"].any()
    panel["in_universe"] = panel.index.map(flag).fillna(False).astype(bool)
    return panel


def build_panel(
    crsp_dir: Path,
    sf1_path: Path,
    sharadar_tickers_path: Path,
    universe_ids_path: Path,
    output_dir: Path,
) -> pd.DataFrame:
    """Orchestrate the full PIT panel build and write it partitioned by year."""
    crsp = load_crsp_daily(crsp_dir)

    sf1 = pd.read_parquet(sf1_path)
    sf1 = sf1[sf1["dimension"] == PANEL_DIMENSION].copy()
    log.info("Sharadar SF1 (%s): %d rows", PANEL_DIMENSION, len(sf1))

    sharadar_tickers = pd.read_parquet(sharadar_tickers_path)
    universe_ids = pd.read_parquet(universe_ids_path)

    ticker_cik_map = build_ticker_cik_map(sharadar_tickers)
    sf1 = resolve_sf1_permno(sf1, ticker_cik_map, universe_ids)

    panel = pit_join(crsp, sf1)
    panel = strike_no_fundamentals(panel)
    panel = flag_universe_membership(panel, universe_ids)

    leak = panel["datekey"].notna() & (panel["datekey"] > panel["date"])
    if leak.any():
        raise RuntimeError(f"LEAKAGE: {leak.sum()} rows have datekey > date — PIT join is broken")

    # Age of the forward-filled fundamental in calendar days. PIT-legal at any
    # age, but downstream feature code may want to drop stale rows.
    panel["fund_age_days"] = (panel["date"] - panel["datekey"]).dt.days

    panel["year"] = panel["date"].dt.year
    if output_dir.exists():
        shutil.rmtree(output_dir)
    panel.to_parquet(output_dir, partition_cols=["year"], index=False)

    n_with_fund = panel["datekey"].notna().sum()
    log.info(
        "Panel: %d rows, %d permnos, %s -> %s | %d rows (%.1f%%) carry fundamentals | %d in-universe",
        len(panel), panel["permno"].nunique(),
        panel["date"].min().date(), panel["date"].max().date(),
        n_with_fund, 100.0 * n_with_fund / len(panel), int(panel["in_universe"].sum()),
    )
    log.info("Wrote %s (partitioned by year)", output_dir)
    return panel


def main() -> None:
    argparse.ArgumentParser(description="Build the unified PIT price+fundamentals panel").parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = repo_root() / "logs" / f"build_panel_{ts}.log"
    configure_logging(log_file=log_file)
    log.info("Panel build starting")
    log.info("Log file: %s", log_file)

    out_dir = processed_dir()
    build_panel(
        crsp_dir=out_dir / "crsp_daily",
        sf1_path=out_dir / "sharadar_sf1.parquet",
        sharadar_tickers_path=out_dir / "sharadar_tickers.parquet",
        universe_ids_path=out_dir / "universe_ids.parquet",
        output_dir=out_dir / "panel",
    )
    log.info("Panel build complete.")


if __name__ == "__main__":
    main()

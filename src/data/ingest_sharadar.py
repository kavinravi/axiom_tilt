"""Pull point-in-time fundamentals from Sharadar Core US Fundamentals (SF1).

Outputs:
  data/processed/sharadar_tickers.parquet  Sharadar ticker metadata (CIK mapping)
  data/processed/sharadar_sf1.parquet      As-Reported fundamentals for the universe

Sharadar SF1 `dimension` field:
  ARQ / ARY / ART  -- As Reported (Quarterly / Annual / TTM): values as
                      ORIGINALLY filed. Point-in-time. We pull these.
  MRQ / MRY / MRT  -- Most Recent: restated values. Look-ahead bias. We skip these.

Key date columns:
  datekey       -- date the data became public (the filing date). PIT join key.
  calendardate  -- standardized period-end (quarter/year aligned).
  reportperiod  -- fiscal period end as reported by the company.

Coverage: ~1998-present, survivorship-bias-free (includes delisted firms) — so it
covers the 2008 GFC, unlike SEC XBRL (which only starts ~2009).

Setup:
  1. Subscribe to Sharadar Core US Fundamentals (SHARADAR/SF1) at data.nasdaq.com.
  2. Put your API key in .env as NASDAQ_DATA_LINK_API_KEY.
  3. pip install -e .  (nasdaqdatalink is in requirements.txt)

Usage:
  python -m src.data.ingest_sharadar
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Iterator

import pandas as pd

from src.utils.env import get_env
from src.utils.io import processed_dir, repo_root
from src.utils.logging_utils import configure_logging, get_logger


log = get_logger(__name__)

# As-Reported dimensions only — point-in-time. MR* dimensions are restated
# and would introduce look-ahead bias if used as model features.
PIT_DIMENSIONS = ["ARQ", "ARY"]

# Date columns in SF1 / TICKERS that should be parsed to datetime.
_DATE_COLS = ("datekey", "calendardate", "reportperiod", "lastupdated", "firstadded")


def _chunk(seq: list, n: int) -> Iterator[list]:
    """Yield successive n-sized chunks from seq."""
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce known Sharadar date columns to datetime in place."""
    for col in _DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _add_cik_column(tickers: pd.DataFrame) -> pd.DataFrame:
    """Derive an integer `cik` column from the `secfilings` URL.

    SHARADAR/TICKERS has no standalone CIK field; the CIK is embedded in the
    secfilings EDGAR URL, e.g. ".../browse-edgar?...&CIK=0002099039".
    """
    if "secfilings" in tickers.columns:
        tickers["cik"] = pd.to_numeric(
            tickers["secfilings"].str.extract(r"CIK=(\d+)", expand=False),
            errors="coerce",
        )
    return tickers


def pull_sharadar_tickers(ndl, output_path: Path) -> pd.DataFrame:
    """Pull the SHARADAR/TICKERS metadata table for SF1 (ticker <-> cik mapping)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tickers = ndl.get_table("SHARADAR/TICKERS", table="SF1", paginate=True)
    tickers = _parse_dates(tickers)
    tickers = _add_cik_column(tickers)
    tickers.to_parquet(output_path, index=False)
    log.info("SHARADAR/TICKERS: %d rows -> %s", len(tickers), output_path)
    return tickers


def resolve_sharadar_tickers(
    universe_ids: pd.DataFrame,
    sharadar_tickers: pd.DataFrame,
) -> list[str]:
    """Map universe CIKs to Sharadar tickers via the TICKERS metadata table.

    One CIK can map to multiple Sharadar tickers (ticker changes over time);
    all of them are returned so SF1 history is complete.
    """
    universe_ciks = set(
        pd.to_numeric(universe_ids["cik"], errors="coerce").dropna().astype(int).tolist()
    )

    st = sharadar_tickers.copy()
    if "cik" not in st.columns:
        st = _add_cik_column(st)
    st["cik"] = pd.to_numeric(st["cik"], errors="coerce")
    matched = st[st["cik"].isin(universe_ciks)]

    ticker_list = sorted(matched["ticker"].dropna().astype(str).unique().tolist())
    log.info(
        "Mapped %d/%d universe CIKs -> %d Sharadar tickers",
        matched["cik"].nunique(),
        len(universe_ciks),
        len(ticker_list),
    )
    if not ticker_list:
        raise RuntimeError(
            "No universe CIKs matched Sharadar TICKERS — check that universe_ids.parquet "
            "has a populated `cik` column."
        )
    return ticker_list


def pull_sharadar_sf1(
    ndl,
    tickers: list[str],
    output_path: Path,
    dimensions: list[str] | None = None,
    chunk_size: int = 100,
) -> None:
    """Pull SHARADAR/SF1 fundamentals for the given tickers, As-Reported dimensions only.

    Tickers are chunked because get_table's filter list has a practical size cap.
    """
    dimensions = dimensions or PIT_DIMENSIONS
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dfs = []
    for chunk in _chunk(tickers, chunk_size):
        df = ndl.get_table(
            "SHARADAR/SF1",
            ticker=chunk,
            dimension=dimensions,
            paginate=True,
        )
        dfs.append(df)

    sf1 = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    sf1 = _parse_dates(sf1)
    sf1.to_parquet(output_path, index=False)

    if len(sf1):
        log.info(
            "SHARADAR/SF1: %d rows, %d tickers, datekey %s -> %s -> %s",
            len(sf1),
            sf1["ticker"].nunique() if "ticker" in sf1.columns else 0,
            sf1["datekey"].min() if "datekey" in sf1.columns else "?",
            sf1["datekey"].max() if "datekey" in sf1.columns else "?",
            output_path,
        )
    else:
        log.warning("SHARADAR/SF1 returned zero rows -> %s", output_path)


def main() -> None:
    argparse.ArgumentParser(description="Pull Sharadar SF1 fundamentals for the universe").parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = repo_root() / "logs" / f"sharadar_{ts}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    configure_logging(log_file=log_file)
    log.info("Sharadar SF1 ingest starting")
    log.info("Log file: %s", log_file)

    import nasdaqdatalink as ndl  # local import so tests don't need the package

    ndl.ApiConfig.api_key = get_env("NASDAQ_DATA_LINK_API_KEY", required=True)

    out_dir = processed_dir()
    tickers_path = out_dir / "sharadar_tickers.parquet"
    sf1_path = out_dir / "sharadar_sf1.parquet"
    universe_ids_path = out_dir / "universe_ids.parquet"

    if not universe_ids_path.exists():
        raise SystemExit(
            f"{universe_ids_path} missing — run `python -m src.data.ingest_wrds --resolve-only` first."
        )

    universe_ids = pd.read_parquet(universe_ids_path)
    sharadar_tickers = pull_sharadar_tickers(ndl, tickers_path)
    ticker_list = resolve_sharadar_tickers(universe_ids, sharadar_tickers)
    pull_sharadar_sf1(ndl, ticker_list, sf1_path)

    log.info("Sharadar SF1 ingest complete.")


if __name__ == "__main__":
    main()

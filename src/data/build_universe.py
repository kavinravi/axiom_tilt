"""Reconstruct point-in-time S&P 500 membership 2000-2025.

Sources:
  - https://en.wikipedia.org/wiki/List_of_S%26P_500_companies (current + changes table)
  - https://www.sec.gov/files/company_tickers.json (ticker -> CIK)

Output:
  data/processed/universe.parquet with columns:
    ticker, cik, company, date_in, date_out

A ticker may appear multiple times if it left and re-joined the index.
"""
from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.utils.env import get_env
from src.utils.io import processed_dir
from src.utils.logging_utils import configure_logging, get_logger


WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

log = get_logger(__name__)


def _fetch(url: str, user_agent: str) -> str:
    resp = requests.get(url, headers={"User-Agent": user_agent}, timeout=30)
    resp.raise_for_status()
    return resp.text


def parse_current_members(html: str) -> pd.DataFrame:
    """Parse the first wikitable on the Wikipedia page (current S&P 500 members)."""
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", {"id": "constituents"})
    if table is None:
        # Fallback: first sortable wikitable
        table = soup.find("table", {"class": lambda c: c and "wikitable" in c})
    if table is None:
        raise RuntimeError("Could not find S&P 500 members table on Wikipedia page")
    df = pd.read_html(StringIO(str(table)))[0]
    df.columns = [c.strip() for c in df.columns]
    # Wikipedia's column names drift; normalize the two we care about
    ticker_col = next(c for c in df.columns if "Symbol" in c or "Ticker" in c)
    name_col = next(c for c in df.columns if "Security" in c or "Company" in c)
    out = pd.DataFrame({
        "ticker": df[ticker_col].astype(str).str.upper().str.replace(".", "-", regex=False),
        "company": df[name_col].astype(str),
    })
    return out.reset_index(drop=True)


def parse_changes_table(html: str) -> pd.DataFrame:
    """Parse the 'Selected changes to the list' historical table.

    Returns long-format rows: (date, added_ticker, removed_ticker).
    A single date can have both an addition and a removal (they're paired changes).
    """
    soup = BeautifulSoup(html, "lxml")
    # The changes table usually has id="changes" or is the second wikitable
    changes_table = soup.find("table", {"id": "changes"})
    if changes_table is None:
        tables = soup.find_all("table", {"class": lambda c: c and "wikitable" in c})
        if len(tables) < 2:
            raise RuntimeError("Could not find changes table on Wikipedia page")
        changes_table = tables[1]

    raw = pd.read_html(StringIO(str(changes_table)), header=[0, 1])[0]
    # Multi-level header has top row {Date, Added, Added, Removed, Removed, Reason}
    # and second row with subcolumn names. Flatten:
    raw.columns = ["_".join([str(x).strip() for x in tup if str(x) != "nan"]) for tup in raw.columns]

    # Identify columns we need
    date_col = next(c for c in raw.columns if "date" in c.lower())
    added_ticker_col = next(
        c for c in raw.columns
        if "added" in c.lower() and ("ticker" in c.lower() or "symbol" in c.lower())
    )
    removed_ticker_col = next(
        c for c in raw.columns
        if "removed" in c.lower() and ("ticker" in c.lower() or "symbol" in c.lower())
    )

    out = pd.DataFrame({
        "date": pd.to_datetime(raw[date_col], errors="coerce"),
        "added_ticker": raw[added_ticker_col].astype(str).str.upper()
            .str.replace(".", "-", regex=False).replace({"NAN": pd.NA}),
        "removed_ticker": raw[removed_ticker_col].astype(str).str.upper()
            .str.replace(".", "-", regex=False).replace({"NAN": pd.NA}),
    })
    out = out.dropna(subset=["date"]).reset_index(drop=True)
    return out


def load_ticker_to_cik(json_path: Path) -> dict[str, str]:
    """Load ticker -> 10-digit zero-padded CIK from SEC's company_tickers.json."""
    with json_path.open() as f:
        data = json.load(f)
    out: dict[str, str] = {}
    for row in data.values():
        ticker = str(row["ticker"]).upper().replace(".", "-")
        cik = str(row["cik_str"]).zfill(10)
        out[ticker] = cik
    return out


def reconstruct_membership(
    html: str,
    ticker_to_cik: dict[str, str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Walk the changes table backward from current members to produce intervals.

    Algorithm:
      1. Start with current members (all open, date_out=NaT).
      2. Walk changes from newest -> oldest:
         - If a ticker was 'added' on date d, it was NOT a member before d.
           Close any open interval for that ticker with date_out=d, set date_in=d.
         - If a ticker was 'removed' on date d, it WAS a member before d.
           Open an interval ending at d with no known date_in (we'll close it at start_date).
      3. Any still-open interval at end gets date_in=start_date.
      4. Trim everything to [start_date, end_date].
    """
    current = parse_current_members(html)
    changes = parse_changes_table(html).sort_values("date", ascending=False)

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    intervals: list[dict] = []
    # current members -> open intervals with date_out = NaT
    for _, row in current.iterrows():
        intervals.append({
            "ticker": row["ticker"],
            "company": row["company"],
            "date_in": pd.NaT,   # filled in by walk
            "date_out": pd.NaT,
        })

    # index intervals by ticker, taking the most-recent open one for each ticker
    def open_interval(ticker: str) -> dict | None:
        for iv in reversed(intervals):
            if iv["ticker"] == ticker and pd.isna(iv["date_in"]):
                return iv
        return None

    for _, ch in changes.iterrows():
        d = ch["date"]
        added = ch["added_ticker"]
        removed = ch["removed_ticker"]

        if pd.notna(added):
            iv = open_interval(added)
            if iv is None:
                # Ticker was added on d but we don't have an open interval — means
                # they were added then later removed before "current". Open one going forward.
                intervals.append({
                    "ticker": added,
                    "company": "",
                    "date_in": d,
                    "date_out": pd.NaT,
                })
            else:
                iv["date_in"] = d

        if pd.notna(removed):
            # They were a member up to d. Open a fresh interval to be closed by older changes.
            intervals.append({
                "ticker": removed,
                "company": "",
                "date_in": pd.NaT,
                "date_out": d,
            })

    # Any still-open date_in -> they were already in at start_date
    for iv in intervals:
        if pd.isna(iv["date_in"]):
            iv["date_in"] = start_ts

    df = pd.DataFrame(intervals)

    # Trim to window. Drop intervals fully outside [start_ts, end_ts].
    df = df[df["date_in"] <= end_ts]
    df = df[df["date_out"].isna() | (df["date_out"] >= start_ts)]
    df["date_in"] = df["date_in"].clip(lower=start_ts)
    df.loc[df["date_out"] > end_ts, "date_out"] = pd.NaT

    # Attach CIKs
    df["cik"] = df["ticker"].map(ticker_to_cik)

    # Drop rows where date_in > date_out
    valid = df["date_out"].isna() | (df["date_in"] <= df["date_out"])
    df = df[valid].reset_index(drop=True)

    return df[["ticker", "cik", "company", "date_in", "date_out"]]


def main() -> None:
    configure_logging()
    user_agent = get_env("SEC_USER_AGENT", required=True)

    log.info("Fetching Wikipedia S&P 500 page")
    html = _fetch(WIKI_URL, user_agent="Mozilla/5.0 axiom-tilt-research")

    log.info("Fetching SEC company_tickers.json")
    sec_json = _fetch(SEC_TICKERS_URL, user_agent=user_agent)
    sec_path = processed_dir().parent / "raw" / "sec" / "company_tickers.json"
    sec_path.parent.mkdir(parents=True, exist_ok=True)
    sec_path.write_text(sec_json)
    ticker_to_cik = load_ticker_to_cik(sec_path)

    log.info("Reconstructing membership intervals")
    df = reconstruct_membership(
        html=html,
        ticker_to_cik=ticker_to_cik,
        start_date="2000-01-01",
        end_date="2025-12-31",
    )

    out_path = processed_dir() / "universe.parquet"
    df.to_parquet(out_path, index=False)

    n_with_cik = df["cik"].notna().sum()
    log.info(
        "Wrote %d intervals (%d unique tickers, %d with CIK match) to %s",
        len(df), df["ticker"].nunique(), n_with_cik, out_path,
    )


if __name__ == "__main__":
    main()

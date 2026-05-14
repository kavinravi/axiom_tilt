"""Pull quarterly + annual fundamentals from Financial Modeling Prep.

**DEPRECATED (2026-05-13):** superseded by src/data/ingest_wrds.py
(Compustat funda + fundq via WRDS). FMP returns the latest restated values,
introducing look-ahead bias. Compustat is point-in-time via (datadate, rdq).
Kept here for replication / fallback. See
docs/superpowers/specs/2026-05-13-wrds-ingestion-design.md.

Output: data/processed/fundamentals.parquet
Long format: (ticker, statement, period, period_end, filing_date, ...)

The puller is per-ticker and resumable. State is tracked in
data/state/fundamentals_done.txt — append-only list of completed tickers.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pandas as pd
import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

from src.utils.config import load_config
from src.utils.env import get_env
from src.utils.io import processed_dir, raw_dir, state_dir
from src.utils.logging_utils import configure_logging, get_logger
from src.utils.rate_limit import TokenBucket


log = get_logger(__name__)
FMP_BASE = "https://financialmodelingprep.com/api/v3"


class StatementType(str, Enum):
    INCOME = "income-statement"
    BALANCE = "balance-sheet-statement"
    CASHFLOW = "cash-flow-statement"


@dataclass
class FmpClient:
    api_key: str
    bucket: TokenBucket
    retry_attempts: int

    def fetch(
        self,
        endpoint: str,
        ticker: str,
        period: str = "quarter",
        limit: int = 200,
    ) -> list[dict]:
        url = f"{FMP_BASE}/{endpoint}/{ticker}"
        params = {"period": period, "limit": limit, "apikey": self.api_key}

        @retry(
            stop=stop_after_attempt(self.retry_attempts),
            wait=wait_exponential(multiplier=2, min=2, max=60),
            retry=retry_if_exception_type((requests.RequestException,)),
            reraise=True,
        )
        def _do() -> list[dict]:
            self.bucket.acquire()
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                raise requests.RequestException(f"rate limited (429)")
            if not resp.ok:
                # Construct error message WITHOUT the URL (would leak api_key in query string)
                raise requests.HTTPError(
                    f"{resp.status_code} {resp.reason} for {endpoint}/{ticker}"
                )
            data = resp.json()
            if isinstance(data, dict) and "Error Message" in data:
                # Treat as empty rather than raising; some tickers have no data
                return []
            return data if isinstance(data, list) else []

        return _do()


def parse_fmp_statement(
    rows: list[dict],
    ticker: str,
    stmt_type: StatementType,
) -> pd.DataFrame:
    """Normalize FMP statement rows into our long schema."""
    if not rows:
        return pd.DataFrame(columns=["ticker", "statement", "period", "period_end", "filing_date"])

    df = pd.DataFrame(rows)
    df["ticker"] = ticker
    df["statement"] = stmt_type.name.lower()
    df["period_end"] = pd.to_datetime(df.get("date"), errors="coerce")
    df["filing_date"] = pd.to_datetime(df.get("fillingDate"), errors="coerce")
    df["period"] = df.get("period", pd.Series(["UNK"] * len(df))).astype(str)
    # Reorder
    front = ["ticker", "statement", "period", "period_end", "filing_date"]
    rest = [c for c in df.columns if c not in front]
    return df[front + rest]


def _load_done(state_file: Path) -> set[str]:
    if not state_file.exists():
        return set()
    return {line.strip() for line in state_file.read_text().splitlines() if line.strip()}


def _append_done(state_file: Path, ticker: str) -> None:
    with state_file.open("a") as f:
        f.write(f"{ticker}\n")


def main() -> None:
    configure_logging()
    cfg = load_config("data")
    api_key = get_env("FMP_API_KEY", required=True)

    universe_path = processed_dir() / "universe.parquet"
    universe = pd.read_parquet(universe_path)
    tickers = sorted(universe["ticker"].dropna().unique().tolist())

    state_file = state_dir() / "fundamentals_done.txt"
    done = _load_done(state_file)
    todo = [t for t in tickers if t not in done]
    log.info("Tickers: %d total, %d done, %d to do", len(tickers), len(done), len(todo))

    client = FmpClient(
        api_key=api_key,
        bucket=TokenBucket(
            rate_per_sec=cfg["fundamentals"]["rate_per_sec"],
            capacity=cfg["fundamentals"]["capacity"],
        ),
        retry_attempts=cfg["fundamentals"]["retry_attempts"],
    )

    raw_root = raw_dir() / "fundamentals"
    raw_root.mkdir(parents=True, exist_ok=True)

    all_frames: list[pd.DataFrame] = []
    for ticker in tqdm(todo, desc="fundamentals"):
        try:
            for stmt in StatementType:
                rows = client.fetch(stmt.value, ticker, period="quarter", limit=200)
                # Persist raw response for debugging
                (raw_root / f"{ticker}_{stmt.name.lower()}.json").write_text(
                    json.dumps(rows)
                )
                df = parse_fmp_statement(rows, ticker, stmt)
                if not df.empty:
                    all_frames.append(df)
            _append_done(state_file, ticker)
        except Exception as e:
            log.warning("Failed %s: %s", ticker, e)

    # Merge with anything already on disk for tickers we just finished
    out_path = processed_dir() / "fundamentals.parquet"
    if all_frames:
        new = pd.concat(all_frames, ignore_index=True)
        if out_path.exists():
            existing = pd.read_parquet(out_path)
            combined = pd.concat([existing, new], ignore_index=True)
        else:
            combined = new
        combined = combined.drop_duplicates(
            subset=["ticker", "statement", "period_end"], keep="last"
        )
        combined.to_parquet(out_path, index=False)
        log.info("Wrote %d rows to %s", len(combined), out_path)
    else:
        log.info("No new data to write.")


if __name__ == "__main__":
    main()

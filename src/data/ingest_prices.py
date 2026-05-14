"""Pull daily OHLCV adjusted prices via yfinance for the universe.

**DEPRECATED (2026-05-13):** superseded by src/data/ingest_wrds.py
(CRSP daily via WRDS). yfinance drops delisted tickers, causing survivorship
bias in backtests. Kept here for replication / fallback when WRDS access is
unavailable. See docs/superpowers/specs/2026-05-13-wrds-ingestion-design.md.

Output: data/processed/prices.parquet
Long format: (date, ticker, open, high, low, close, adj_close, volume)
"""
from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from pathlib import Path

import pandas as pd
import yfinance as yf
from tqdm import tqdm

from src.utils.config import load_config
from src.utils.io import processed_dir
from src.utils.logging_utils import configure_logging, get_logger


log = get_logger(__name__)


def chunk_tickers(tickers: Iterable[str], batch_size: int) -> Iterator[list[str]]:
    buf: list[str] = []
    for t in tickers:
        buf.append(t)
        if len(buf) >= batch_size:
            yield buf
            buf = []
    if buf:
        yield buf


def normalize_yfinance_frame(
    frame: pd.DataFrame,
    ticker: str | None = None,
) -> pd.DataFrame:
    """Convert yfinance output to long-format (date, ticker, ohlcv).

    yfinance returns:
      - flat columns when called with a single ticker -> need `ticker` arg
      - MultiIndex columns (Price, Ticker) when called with multiple
    """
    if isinstance(frame.columns, pd.MultiIndex):
        # Wide -> long via stack
        frame = frame.copy()
        frame.index.name = frame.index.name or "Date"
        long = frame.stack(level="Ticker", future_stack=True).reset_index()
        long.columns = [str(c).strip() for c in long.columns]
        long = long.rename(columns={
            "Date": "date",
            "Ticker": "ticker",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        })
    else:
        if ticker is None:
            raise ValueError("ticker required when frame has flat columns")
        frame = frame.copy()
        frame.index.name = frame.index.name or "Date"
        long = frame.reset_index().rename(columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        })
        long["ticker"] = ticker

    keep = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
    long = long[[c for c in keep if c in long.columns]]
    long = long.dropna(subset=["close"])
    long["date"] = pd.to_datetime(long["date"]).dt.tz_localize(None).dt.normalize()
    return long.reset_index(drop=True)


def fetch_batch(
    tickers: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Pull one batch via yfinance.download."""
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    if raw is None or raw.empty:
        return pd.DataFrame()
    if len(tickers) == 1:
        return normalize_yfinance_frame(raw, ticker=tickers[0])
    return normalize_yfinance_frame(raw)


def main() -> None:
    configure_logging()
    cfg = load_config("data")

    universe_path = processed_dir() / "universe.parquet"
    if not universe_path.exists():
        raise FileNotFoundError(
            f"{universe_path} not found. Run `python -m src.data.build_universe` first."
        )
    universe = pd.read_parquet(universe_path)
    tickers = sorted(universe["ticker"].unique().tolist())
    log.info("Pulling prices for %d tickers", len(tickers))

    batch_size = cfg["prices"]["batch_size"]
    delay = cfg["prices"]["request_delay_sec"]
    start = cfg["start_date"]
    end = cfg["end_date"]

    frames: list[pd.DataFrame] = []
    failed: list[str] = []
    for batch in tqdm(list(chunk_tickers(tickers, batch_size)), desc="batches"):
        try:
            df = fetch_batch(batch, start=start, end=end)
            if not df.empty:
                frames.append(df)
            else:
                failed.extend(batch)
        except Exception as e:
            log.warning("Batch failed (%s): %s", e.__class__.__name__, e)
            failed.extend(batch)
        time.sleep(delay)

    if not frames:
        raise RuntimeError("No price data was successfully pulled.")

    out = pd.concat(frames, ignore_index=True).drop_duplicates(["date", "ticker"])
    out_path = processed_dir() / "prices.parquet"
    out.to_parquet(out_path, index=False)

    log.info(
        "Wrote %d (date, ticker) rows for %d distinct tickers to %s",
        len(out), out["ticker"].nunique(), out_path,
    )
    if failed:
        log.warning("Failed tickers (%d): %s", len(failed), failed[:20])
        # Persist for inspection
        Path(processed_dir() / "prices_failed.txt").write_text("\n".join(failed))


if __name__ == "__main__":
    main()

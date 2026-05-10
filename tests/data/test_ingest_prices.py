"""Tests for src.data.ingest_prices."""
import pandas as pd
import pytest

from src.data.ingest_prices import (
    normalize_yfinance_frame,
    chunk_tickers,
)


def test_chunk_tickers_splits_evenly():
    tickers = [f"T{i}" for i in range(105)]
    chunks = list(chunk_tickers(tickers, batch_size=50))
    assert len(chunks) == 3
    assert len(chunks[0]) == 50
    assert len(chunks[1]) == 50
    assert len(chunks[2]) == 5


def test_normalize_yfinance_frame_long_format():
    # yfinance.download returns a wide multi-index DataFrame for >1 ticker
    idx = pd.date_range("2024-01-02", periods=3, freq="B")
    cols = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Adj Close", "Volume"], ["AAPL", "MSFT"]],
        names=["Price", "Ticker"],
    )
    data = [[1.0]*12, [2.0]*12, [3.0]*12]
    wide = pd.DataFrame(data, index=idx, columns=cols)

    long_df = normalize_yfinance_frame(wide)
    assert {"date", "ticker", "open", "high", "low", "close", "adj_close", "volume"} \
        .issubset(long_df.columns)
    assert len(long_df) == 6  # 3 dates x 2 tickers
    assert set(long_df["ticker"].unique()) == {"AAPL", "MSFT"}


def test_normalize_yfinance_frame_handles_single_ticker():
    # When yfinance.download is called with one ticker it returns a flat-column frame
    idx = pd.date_range("2024-01-02", periods=2, freq="B")
    flat = pd.DataFrame(
        {"Open": [1.0, 2.0], "High": [1.1, 2.1], "Low": [0.9, 1.9],
         "Close": [1.05, 2.05], "Adj Close": [1.05, 2.05], "Volume": [1000, 2000]},
        index=idx,
    )
    long_df = normalize_yfinance_frame(flat, ticker="AAPL")
    assert len(long_df) == 2
    assert (long_df["ticker"] == "AAPL").all()

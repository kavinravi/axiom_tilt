"""Tests for src.data.ingest_fundamentals."""
import json

import pandas as pd
import pytest

from src.data.ingest_fundamentals import (
    parse_fmp_statement,
    StatementType,
)


def test_parse_fmp_income_statement():
    sample = [
        {
            "date": "2024-09-28",
            "symbol": "AAPL",
            "reportedCurrency": "USD",
            "fillingDate": "2024-11-01",
            "acceptedDate": "2024-11-01 06:01:36",
            "calendarYear": "2024",
            "period": "FY",
            "revenue": 391_035_000_000,
            "costOfRevenue": 210_352_000_000,
            "grossProfit": 180_683_000_000,
            "netIncome": 93_736_000_000,
            "eps": 6.08,
        }
    ]
    df = parse_fmp_statement(sample, ticker="AAPL", stmt_type=StatementType.INCOME)
    assert len(df) == 1
    assert df.iloc[0]["ticker"] == "AAPL"
    assert df.iloc[0]["statement"] == "income"
    assert df.iloc[0]["period_end"] == pd.Timestamp("2024-09-28")
    assert df.iloc[0]["filing_date"] == pd.Timestamp("2024-11-01")
    assert df.iloc[0]["revenue"] == 391_035_000_000


def test_parse_fmp_handles_empty_response():
    df = parse_fmp_statement([], ticker="ZZZZ", stmt_type=StatementType.INCOME)
    assert len(df) == 0
    assert "ticker" in df.columns

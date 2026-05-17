"""Tests for src.utils.features — pure helpers behind notebook 05."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.utils.features import (
    compute_days_since_filing,
    compute_doc_count_window,
    compute_forward_returns,
    compute_text_novelty,
    pivot_macro_wide,
)


def test_pivot_macro_wide_creates_one_column_per_series():
    long = pd.DataFrame({
        'date': pd.to_datetime(['2020-01-01', '2020-01-01', '2020-01-02', '2020-01-02']),
        'series': ['VIXCLS', 'DGS10', 'VIXCLS', 'DGS10'],
        'value': [25.0, 1.8, 27.0, 1.9],
    })
    out = pivot_macro_wide(long)
    assert list(out.columns) == ['date', 'macro_dgs10', 'macro_vixcls']  # alpha-sorted, prefixed
    assert len(out) == 2
    assert out.iloc[0]['macro_vixcls'] == 25.0
    assert out.iloc[1]['macro_dgs10'] == 1.9


def test_pivot_macro_wide_forward_fills_missing_dates_when_requested():
    long = pd.DataFrame({
        'date': pd.to_datetime(['2020-01-01', '2020-01-03']),  # gap on 2020-01-02
        'series': ['VIXCLS', 'VIXCLS'],
        'value': [25.0, 27.0],
    })
    out = pivot_macro_wide(long, ffill_dates=pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03']))
    assert len(out) == 3
    assert out.iloc[1]['macro_vixcls'] == 25.0  # ffilled from 2020-01-01


# -------------------------------- compute_forward_returns ----------------------


def test_compute_forward_returns_one_permno_simple_case():
    df = pd.DataFrame({
        'permno': [101] * 6,
        'date': pd.date_range('2020-01-02', periods=6, freq='B'),
        'ret': [0.01, -0.02, 0.03, 0.00, 0.05, -0.01],
    })
    out = compute_forward_returns(df, horizons=(1, 5))
    # fwd_ret_1d: next-day return
    assert out['fwd_ret_1d'].iloc[0] == pytest.approx(-0.02)
    assert pd.isna(out['fwd_ret_1d'].iloc[-1])  # last row has no next day
    # fwd_ret_5d: compounded next 5 returns
    expected_5d = (1.0 + np.array([-0.02, 0.03, 0.00, 0.05, -0.01])).prod() - 1.0
    assert out['fwd_ret_5d'].iloc[0] == pytest.approx(expected_5d)
    # Last 5 rows have no full 5-day forward window
    assert out['fwd_ret_5d'].iloc[1:].isna().sum() == 5


def test_compute_forward_returns_does_not_cross_permnos():
    df = pd.DataFrame({
        'permno': [101, 101, 202, 202],
        'date': pd.to_datetime(['2020-01-02', '2020-01-03', '2020-01-02', '2020-01-03']),
        'ret': [0.01, 0.02, 0.10, 0.20],
    })
    out = compute_forward_returns(df, horizons=(1,))
    # 101's last row's fwd_ret_1d should NOT pull from 202
    assert pd.isna(out.loc[out['permno'] == 101, 'fwd_ret_1d'].iloc[-1])
    assert pd.isna(out.loc[out['permno'] == 202, 'fwd_ret_1d'].iloc[-1])


# -------------------------------- compute_text_novelty -------------------------


def test_compute_text_novelty_identical_vectors_is_zero():
    embed = pd.DataFrame({
        'permno': [101, 101],
        'date': pd.to_datetime(['2020-01-08', '2020-01-15']),  # 7 days apart
        'vec': [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    })
    out = compute_text_novelty(embed, lookback_days=7)
    # First row has no t-7 → NaN. Second row's vec == prior vec → novelty = 0.
    assert pd.isna(out['text_novelty'].iloc[0])
    assert out['text_novelty'].iloc[1] == pytest.approx(0.0, abs=1e-6)


def test_compute_text_novelty_orthogonal_vectors_is_one():
    embed = pd.DataFrame({
        'permno': [101, 101],
        'date': pd.to_datetime(['2020-01-08', '2020-01-15']),
        'vec': [[1.0, 0.0], [0.0, 1.0]],
    })
    out = compute_text_novelty(embed, lookback_days=7)
    # cosine_sim = 0 → novelty = 1 - 0 = 1
    assert out['text_novelty'].iloc[1] == pytest.approx(1.0, abs=1e-6)


def test_compute_text_novelty_does_not_cross_permnos():
    embed = pd.DataFrame({
        'permno': [101, 202],
        'date': pd.to_datetime(['2020-01-08', '2020-01-15']),  # different permnos
        'vec': [[1.0, 0.0], [0.0, 1.0]],
    })
    out = compute_text_novelty(embed, lookback_days=7)
    # Neither row has a t-7 same-permno predecessor → both NaN
    assert out['text_novelty'].isna().all()


# -------------------------------- compute_days_since_filing --------------------


def test_compute_days_since_filing_simple_case():
    filings = pd.DataFrame({
        'cik': ['0000000101', '0000000101'],
        'filing_date': pd.to_datetime(['2020-01-01', '2020-02-01']),
        'form_type': ['10-K', '10-Q'],
    })
    panel = pd.DataFrame({
        'permno': [101, 101, 101],
        'cik': ['0000000101'] * 3,
        'date': pd.to_datetime(['2020-01-05', '2020-02-05', '2020-03-10']),
    })
    out = compute_days_since_filing(filings, panel)
    assert out['days_since_filing'].tolist() == [4, 4, 38]


def test_compute_days_since_filing_returns_nan_before_first_filing():
    filings = pd.DataFrame({
        'cik': ['0000000101'],
        'filing_date': pd.to_datetime(['2020-06-01']),
        'form_type': ['10-K'],
    })
    panel = pd.DataFrame({
        'permno': [101],
        'cik': ['0000000101'],
        'date': pd.to_datetime(['2020-01-15']),  # before any filing
    })
    out = compute_days_since_filing(filings, panel)
    assert pd.isna(out['days_since_filing'].iloc[0])


def test_compute_days_since_filing_excludes_non_kqa_forms():
    filings = pd.DataFrame({
        'cik': ['0000000101', '0000000101'],
        'filing_date': pd.to_datetime(['2020-01-01', '2020-02-01']),
        'form_type': ['DEF 14A', '10-K'],  # 14A excluded; 10-K counted
    })
    panel = pd.DataFrame({
        'permno': [101],
        'cik': ['0000000101'],
        'date': pd.to_datetime(['2020-02-10']),
    })
    out = compute_days_since_filing(filings, panel)
    assert out['days_since_filing'].iloc[0] == 9  # from 2020-02-01, not 2020-01-01


# -------------------------------- compute_doc_count_window ---------------------


def test_compute_doc_count_window_counts_filings_in_window():
    filings = pd.DataFrame({
        'cik': ['0000000101'] * 4,
        'filing_date': pd.to_datetime(['2020-06-01', '2020-06-03', '2020-06-05', '2020-06-15']),
        'form_type': ['8-K', '8-K', '10-Q', '8-K'],
    })
    panel = pd.DataFrame({
        'permno': [101],
        'cik': ['0000000101'],
        'date': pd.to_datetime(['2020-06-07']),  # window = [2020-05-31, 2020-06-07]
    })
    out = compute_doc_count_window(filings, panel, window_days=7)
    # 3 filings in [2020-05-31, 2020-06-07]: 06-01, 06-03, 06-05
    assert out['doc_count_7d'].iloc[0] == 3


def test_compute_doc_count_window_returns_zero_when_no_filings():
    filings = pd.DataFrame({
        'cik': ['0000000101'],
        'filing_date': pd.to_datetime(['2019-01-01']),
        'form_type': ['10-K'],
    })
    panel = pd.DataFrame({
        'permno': [101],
        'cik': ['0000000101'],
        'date': pd.to_datetime(['2020-06-07']),
    })
    out = compute_doc_count_window(filings, panel, window_days=7)
    assert out['doc_count_7d'].iloc[0] == 0


def test_compute_doc_count_window_does_not_count_other_permnos():
    filings = pd.DataFrame({
        'cik': ['0000000999'],  # different cik
        'filing_date': pd.to_datetime(['2020-06-03']),
        'form_type': ['8-K'],
    })
    panel = pd.DataFrame({
        'permno': [101],
        'cik': ['0000000101'],
        'date': pd.to_datetime(['2020-06-07']),
    })
    out = compute_doc_count_window(filings, panel, window_days=7)
    assert out['doc_count_7d'].iloc[0] == 0

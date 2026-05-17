"""Tests for src.utils.features — pure helpers behind notebook 05."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.utils.features import compute_forward_returns, pivot_macro_wide


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

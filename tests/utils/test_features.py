"""Tests for src.utils.features — pure helpers behind notebook 05."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.utils.features import pivot_macro_wide


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

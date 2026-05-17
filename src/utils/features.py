"""Feature helpers for the daily training panel (notebook 05).

Pure functions over pandas DataFrames so the notebook stays a thin glue layer.
See docs/superpowers/specs/2026-05-16-feature-assembly-design.md for design.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def pivot_macro_wide(
    macro_long: pd.DataFrame,
    ffill_dates: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Pivot FRED long-format macro (`date, series, value`) to wide.

    Series become columns prefixed `macro_<series_lower>`. If `ffill_dates` is
    provided, reindex to that date axis and forward-fill (FRED publishes on
    business days; panel rows include all calendar days a permno is active).
    """
    wide = macro_long.pivot(index='date', columns='series', values='value')
    wide.columns = [f'macro_{c.lower()}' for c in wide.columns]
    wide = wide.sort_index().sort_index(axis=1)
    if ffill_dates is not None:
        wide = wide.reindex(ffill_dates).ffill()
    return wide.reset_index().rename(columns={'index': 'date'})


def compute_forward_returns(
    panel: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 5),
    ret_col: str = 'ret',
    permno_col: str = 'permno',
    date_col: str = 'date',
) -> pd.DataFrame:
    """Compute `fwd_ret_{h}d` for each horizon h (trading days).

    For each permno, sorts by date and computes compounded forward returns over
    the next `h` rows via log-return rolling sum (vectorized, no apply).
    Rows in the last `h` of a permno's history get NaN. Delisted permnos
    naturally produce NaN at the tail.

    Returns the input panel with new `fwd_ret_{h}d` columns appended.
    """
    out = panel.sort_values([permno_col, date_col]).copy()
    log_ret = np.log1p(out[ret_col].astype(float))
    for h in horizons:
        # Forward sum of log-returns over horizon h, then expm1.
        # Rolling-sum at index t covers t-h+1..t; we want t+1..t+h, so shift -h.
        rolling_sum = (
            log_ret.groupby(out[permno_col])
            .rolling(window=h, min_periods=h)
            .sum()
            .reset_index(level=0, drop=True)
        )
        out[f'fwd_ret_{h}d'] = np.expm1(rolling_sum.groupby(out[permno_col]).shift(-h))
    return out.reset_index(drop=True)

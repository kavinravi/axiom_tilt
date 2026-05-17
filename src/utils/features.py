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

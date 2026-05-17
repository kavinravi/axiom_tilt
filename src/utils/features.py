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


def compute_text_novelty(
    embed: pd.DataFrame,
    lookback_days: int = 7,
    permno_col: str = 'permno',
    date_col: str = 'date',
    vec_col: str = 'vec',
) -> pd.DataFrame:
    """Compute `text_novelty` = 1 − cosine_similarity(e_{i,t}, e_{i, t−lookback}).

    `lookback_days` is calendar days. For each (permno, date), looks up the
    same permno's vector at exactly `date − lookback_days` (calendar). If no
    such row exists, writes NaN. Embedding panel is assumed forward-filled to
    daily (notebook 03 output) so the lookup hits in steady state.
    """
    out = embed[[permno_col, date_col, vec_col]].copy()
    lookup = out.set_index([permno_col, date_col])[vec_col].to_dict()
    novelty = []
    for permno, date, vec in zip(out[permno_col], out[date_col], out[vec_col]):
        prior_date = date - pd.Timedelta(days=lookback_days)
        prior_vec = lookup.get((permno, prior_date))
        if prior_vec is None:
            novelty.append(np.nan)
            continue
        a = np.asarray(vec, dtype=np.float32)
        b = np.asarray(prior_vec, dtype=np.float32)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0.0:
            novelty.append(np.nan)
            continue
        novelty.append(float(1.0 - np.dot(a, b) / denom))
    out['text_novelty'] = np.asarray(novelty, dtype=np.float32)
    return out[[permno_col, date_col, 'text_novelty']]


FORM_TYPES_KQA = ('10-K', '10-Q', '8-K')


def compute_days_since_filing(
    filings: pd.DataFrame,
    panel: pd.DataFrame,
    form_types: tuple[str, ...] = FORM_TYPES_KQA,
) -> pd.DataFrame:
    """For each (permno, date), days since the most recent filing (10-K/Q/8-K).

    `filings` columns: cik, filing_date, form_type.
    `panel` columns: permno, cik, date.
    Returns the panel with `days_since_filing` (int or NaN if no prior filing).
    Uses `merge_asof` (left, by cik, on filing_date) for vectorized lookup.
    """
    f = (filings[filings['form_type'].isin(form_types)][['cik', 'filing_date']]
         .sort_values('filing_date'))
    # merge_asof requires the global `on` column to be monotonic; sort by date.
    p = panel[['permno', 'cik', 'date']].sort_values('date').reset_index(drop=True)
    merged = pd.merge_asof(
        p,
        f.rename(columns={'filing_date': 'last_filing_date'}),
        left_on='date',
        right_on='last_filing_date',
        by='cik',
        direction='backward',
    )
    merged['days_since_filing'] = (merged['date'] - merged['last_filing_date']).dt.days
    # Re-align to the input panel's row order via key-join (not positional).
    return panel[['permno', 'date']].merge(
        merged[['permno', 'date', 'days_since_filing']],
        on=['permno', 'date'], how='left',
    )


def compute_doc_count_window(
    filings: pd.DataFrame,
    panel: pd.DataFrame,
    window_days: int = 7,
) -> pd.DataFrame:
    """For each (permno, date), count filings on that cik in `[date − window_days, date]`.

    All form types counted (not restricted to K/Q/8K). Returns the panel with
    `doc_count_{window_days}d` column (int32).
    """
    col = f'doc_count_{window_days}d'
    f = filings[['cik', 'filing_date']].sort_values(['cik', 'filing_date']).copy()
    p = panel[['permno', 'cik', 'date']].copy()
    p['_left'] = p['date'] - pd.Timedelta(days=window_days)
    merged = p.merge(f, on='cik', how='left')
    in_window = (merged['filing_date'] >= merged['_left']) & (merged['filing_date'] <= merged['date'])
    merged['_count'] = in_window.astype('int32')
    counts = (
        merged.groupby(['permno', 'date'], as_index=False)['_count'].sum()
        .rename(columns={'_count': col})
    )
    out = panel[['permno', 'date']].merge(counts, on=['permno', 'date'], how='left')
    out[col] = out[col].fillna(0).astype('int32')
    return out

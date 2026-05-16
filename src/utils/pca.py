"""PCA helpers for FinBERT stock-day embeddings (ranker text features).

Design: see docs/superpowers/specs/2026-05-08-text-enhanced-rl-portfolio-design.md §5.3.
Operational decisions: docs/superpowers/plans/2026-05-15-pca-text-features.md.

These functions are pure / sklearn-thin so they unit-test cleanly on synthetic
data. The first-walk orchestration + diagnostics live in
notebooks/04_pca_text_features.ipynb.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def pick_n_components(cum_var: np.ndarray, target: float) -> int:
    """Return locked `n_pca`: smallest n with `cum_var[n-1] >= target`, plus 1 safety.

    If target is unreachable (max cum_var < target), or hit only by the last
    component, cap at `len(cum_var)` — there's no n+1 component beyond full rank.

    Raises ValueError on empty input or target outside [0, 1].
    """
    if len(cum_var) == 0:
        raise ValueError("cum_var must be non-empty")
    if not (0.0 <= target <= 1.0):
        raise ValueError(f"target must be in [0, 1]; got {target}")
    if cum_var[-1] < target:
        return len(cum_var)
    n = int(np.searchsorted(cum_var, target, side="left")) + 1
    return min(n + 1, len(cum_var))


def weekly_snapshots(
    panel: pd.DataFrame,
    permno_col: str = "permno",
    date_col: str = "date",
) -> pd.DataFrame:
    """Return the latest observation per (permno, ISO week ending Friday).

    Used to resample the daily stock-day embed panel to weekly before PCA fit,
    matching the §3.1 rebalance cadence and avoiding the forward-fill bulk.
    """
    df = panel.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["_week"] = df[date_col].dt.to_period("W-FRI")
    df = df.sort_values([permno_col, "_week", date_col])
    out = df.groupby([permno_col, "_week"], as_index=False).tail(1)
    return out.drop(columns="_week").reset_index(drop=True)


def filter_in_universe(panel: pd.DataFrame, universe_ids: pd.DataFrame) -> pd.DataFrame:
    """Keep panel rows where (permno, date) falls inside any universe interval.

    NaT in date_out is treated as "still active" via a far-future sentinel.
    A permno-date that matches multiple intervals (re-joins) is kept once.
    """
    panel_cols = list(panel.columns)
    intervals = universe_ids.dropna(subset=["permno"]).copy()
    intervals["permno"] = intervals["permno"].astype("int64")
    intervals["date_out"] = intervals["date_out"].fillna(pd.Timestamp("2099-12-31"))
    merged = panel.merge(
        intervals[["permno", "date_in", "date_out"]],
        on="permno",
        how="left",
    )
    in_window = (merged["date"] >= merged["date_in"]) & (merged["date"] <= merged["date_out"])
    kept = merged[in_window][panel_cols].drop_duplicates().reset_index(drop=True)
    return kept

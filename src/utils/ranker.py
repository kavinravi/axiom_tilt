"""Helpers for the supervised ranker (notebook 06).

Pure functions over pandas/numpy so the notebook stays a thin orchestration
layer. See docs/superpowers/specs/2026-05-16-supervised-ranker-design.md.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def load_walk_pca(walk_id: int, artifacts_root: Path | None = None) -> tuple[PCA, int]:
    """Load fitted PCA from notebook 04's per-walk artifact."""
    root = Path(artifacts_root) if artifacts_root is not None else Path('artifacts')
    path = root / 'pca-text' / f'walk-{walk_id:03d}' / 'pca.joblib'
    pca: PCA = joblib.load(path)
    return pca, int(pca.n_components_)


def project_text_to_pca(
    embed: pd.DataFrame,
    pca: PCA,
    vec_col: str = 'vec',
) -> pd.DataFrame:
    """Project (permno, date, vec) -> (permno, date, pca_0..pca_{n-1})."""
    X = np.vstack(embed[vec_col].to_numpy()).astype(np.float32)
    Z = pca.transform(X).astype(np.float32)
    cols = [f'pca_{i}' for i in range(Z.shape[1])]
    pca_df = pd.DataFrame(Z, columns=cols)
    keys = embed[['permno', 'date']].reset_index(drop=True)
    return pd.concat([keys, pca_df], axis=1)


def friday_only(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """Keep only Friday rows (weekday == 4) — the rebalance cadence."""
    return df[df[date_col].dt.dayofweek == 4].copy()


def compute_excess_return_buckets(
    df: pd.DataFrame,
    ret_col: str = 'fwd_ret_5d',
    date_col: str = 'date',
    n_buckets: int = 32,
) -> pd.Series:
    """Cross-sectional excess return → percentile rank → integer bucket.

    Returns Int64 series aligned to df.index; NaN where `ret_col` is NaN.
    """
    grp = df.groupby(date_col)[ret_col]
    excess = df[ret_col] - grp.transform('mean')
    pct = excess.groupby(df[date_col]).rank(pct=True, method='average')
    bucket = np.floor(pct * n_buckets).clip(upper=n_buckets - 1)
    bucket = bucket.where(pct.notna())
    return bucket.astype('Int64')


# Identifiers, labels, and non-numeric metadata that must never enter the feature matrix.
NON_FEATURE_COLS = frozenset([
    'permno', 'date', 'cik', 'ret', 'ticker',
    'fiscalperiod', 'datekey', 'calendardate', 'reportperiod', 'lastupdated',
    'dimension', 'in_universe',
    'fwd_ret_1d', 'fwd_ret_5d',
])


def assemble_walk_features(
    panel: pd.DataFrame,
    embed_pca: pd.DataFrame,
    target_col: str = 'fwd_ret_5d',
    date_col: str = 'date',
) -> tuple[pd.DataFrame, pd.Series, list[int], pd.DataFrame]:
    """Inner-join Friday panel rows with PCA embeddings; build (X, y, groups, meta).

    - `X`: numeric feature matrix (PCA + structured + macro + aux), sorted by date.
    - `y`: cross-sectional excess return per Friday (target − per-date mean).
    - `groups`: per-date row counts in `X`'s order (lambdarank group sizes).
    - `meta`: (permno, date, target_col) parallel to X for joining results back.
    """
    merged = (friday_only(panel, date_col)
              .merge(embed_pca, on=['permno', date_col], how='inner')
              .dropna(subset=[target_col])
              .sort_values([date_col, 'permno'])
              .reset_index(drop=True))

    y_excess = (merged[target_col]
                - merged.groupby(date_col)[target_col].transform('mean'))
    feature_cols = [c for c in merged.columns
                    if c not in NON_FEATURE_COLS
                    and pd.api.types.is_numeric_dtype(merged[c])]
    X = merged[feature_cols].copy()
    y = y_excess.rename('y_excess')
    groups = merged.groupby(date_col, sort=False).size().tolist()
    meta = merged[['permno', date_col, target_col]].copy()
    return X, y, groups, meta

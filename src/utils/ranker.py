"""Helpers for the supervised ranker (notebook 06).

Pure functions over pandas/numpy so the notebook stays a thin orchestration
layer. See docs/superpowers/specs/2026-05-16-supervised-ranker-design.md.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRanker, LGBMRegressor
from sklearn.decomposition import PCA

from src.utils.io import repo_root


def load_walk_pca(walk_id: int, artifacts_root: Path | None = None) -> tuple[PCA, int]:
    """Load fitted PCA from notebook 04's per-walk artifact.

    Defaults to `<repo_root>/artifacts` so the call works from any CWD
    (notebooks/, tests/, etc).
    """
    root = Path(artifacts_root) if artifacts_root is not None else repo_root() / 'artifacts'
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


# LightGBM's default label_gain has 31 entries (labels 0..30). We use 32 buckets
# in the notebook, so we need a 32-entry gain vector. Keep the same 2^i - 1 shape.
_LABEL_GAIN_64 = [float(2 ** i - 1) for i in range(64)]


DEFAULT_RANKER_PARAMS: dict[str, Any] = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'label_gain': _LABEL_GAIN_64,
    'num_leaves': 63,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.9,
    'bagging_freq': 5,
    'min_data_in_leaf': 50,
    'lambda_l2': 1.0,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
}


def drop_zero_info_columns(X_train: pd.DataFrame, *also: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    """Drop columns where TRAIN is entirely NaN or zero-variance.

    Returns the input frames with the same set of columns removed from all,
    based on X_train's structure (held constant across train/val/test).
    """
    all_nan = X_train.columns[X_train.isna().all()].tolist()
    zero_var = [c for c in X_train.columns
                if c not in all_nan and X_train[c].nunique(dropna=True) <= 1]
    drop = all_nan + zero_var
    return tuple(df.drop(columns=drop) for df in (X_train, *also))


def build_ranker(params: dict | None = None) -> LGBMRanker:
    """`LGBMRanker` factory with lambdarank defaults; `params` overrides."""
    merged = {**DEFAULT_RANKER_PARAMS, **(params or {})}
    return LGBMRanker(**merged)


DEFAULT_REGRESSOR_PARAMS: dict[str, Any] = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 63,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.9,
    'bagging_freq': 5,
    'min_data_in_leaf': 50,
    'lambda_l2': 1.0,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
}


def build_regressor(params: dict | None = None) -> LGBMRegressor:
    """`LGBMRegressor` factory for the alternative head (predicts continuous
    excess return, then we rank predictions)."""
    merged = {**DEFAULT_REGRESSOR_PARAMS, **(params or {})}
    return LGBMRegressor(**merged)


def compute_grouped_ndcg(
    scores: np.ndarray,
    labels: np.ndarray,
    group_sizes: list[int],
    k: int = 30,
) -> float:
    """Mean per-group NDCG@k. `labels` are integer relevance grades (0..n)."""
    ndcgs: list[float] = []
    idx = 0
    for g in group_sizes:
        s = scores[idx:idx + g]
        t = np.asarray(labels[idx:idx + g], dtype=np.float64)
        kk = min(k, g)
        # Discount: 1 / log2(rank + 2), rank starts at 0
        discounts = 1.0 / np.log2(np.arange(kk) + 2)
        # Actual: top-k by predicted score
        top_k_by_score = np.argsort(s)[::-1][:kk]
        actual_gains = (2.0 ** t[top_k_by_score]) - 1.0
        dcg = float((actual_gains * discounts).sum())
        # Ideal: top-k by true label
        ideal_top = np.sort(t)[::-1][:kk]
        ideal_gains = (2.0 ** ideal_top) - 1.0
        idcg = float((ideal_gains * discounts).sum())
        if idcg > 0:
            ndcgs.append(dcg / idcg)
        idx += g
    return float(np.mean(ndcgs)) if ndcgs else 0.0


def evaluate_ranker(
    model: LGBMRanker,
    X: np.ndarray | pd.DataFrame,
    y_excess: np.ndarray | pd.Series,
    group_dates: np.ndarray | pd.Series,
    top_k: int = 30,
    entity_ids: np.ndarray | pd.Series | None = None,
) -> dict[str, float]:
    """Compute per-group rank IC + decile spread + hit rate + top-K Jaccard.

    `entity_ids` (e.g., permno) is required for meaningful top-K Jaccard
    stability. Without it, the Jaccard metric is NaN (row indices never
    overlap across dates so Jaccard would always be 0, which is misleading).
    """
    scores = model.predict(X)
    df = pd.DataFrame({
        'score': np.asarray(scores, dtype=np.float64),
        'y_excess': np.asarray(y_excess, dtype=np.float64),
        'date': pd.to_datetime(np.asarray(group_dates)),
        'entity': (np.asarray(entity_ids) if entity_ids is not None
                   else np.arange(len(scores))),
    })

    # Per-date Spearman IC.
    ics = (df.groupby('date', sort=True)
           .apply(lambda g: g['score'].corr(g['y_excess'], method='spearman'),
                  include_groups=False)
           .dropna())
    rank_ic_mean = float(ics.mean()) if len(ics) else float('nan')
    rank_ic_ir = (float(ics.mean() / ics.std())
                  if len(ics) > 1 and ics.std() > 0 else float('nan'))

    # Per-date decile spread (top - bottom decile mean), then mean across dates, in bps.
    def _decile_spread(g: pd.DataFrame) -> float:
        if len(g) < 10:
            return float('nan')
        d = pd.qcut(g['score'], 10, labels=False, duplicates='drop')
        top_mask = d == d.max()
        bot_mask = d == d.min()
        return float(g.loc[top_mask, 'y_excess'].mean()
                     - g.loc[bot_mask, 'y_excess'].mean())
    spreads = df.groupby('date').apply(_decile_spread, include_groups=False).dropna()
    decile_spread_bps = float(spreads.mean() * 1e4) if len(spreads) else float('nan')

    # Hit rate: fraction of dates with top-K mean > bottom-K mean.
    def _hit(g: pd.DataFrame) -> float:
        if len(g) < 2 * top_k:
            return float('nan')
        s = g.sort_values('score', ascending=False)
        return float(s.head(top_k)['y_excess'].mean()
                     > s.tail(top_k)['y_excess'].mean())
    hits = df.groupby('date').apply(_hit, include_groups=False).dropna()
    hit_rate = float(hits.mean()) if len(hits) else float('nan')

    # Top-K Jaccard between consecutive dates' top-K sets (by entity_id, e.g. permno).
    # Without entity_ids the metric is meaningless (row indices never overlap), so NaN.
    if entity_ids is None:
        top_k_jaccard = float('nan')
    else:
        df_sorted = df.sort_values(['date', 'score'], ascending=[True, False])
        top_k_sets = {d: set(g.head(top_k)['entity']) for d, g in df_sorted.groupby('date')}
        dates_sorted = sorted(top_k_sets)
        jaccards = []
        for d1, d2 in zip(dates_sorted, dates_sorted[1:]):
            s1, s2 = top_k_sets[d1], top_k_sets[d2]
            if s1 or s2:
                jaccards.append(len(s1 & s2) / len(s1 | s2))
        top_k_jaccard = float(np.mean(jaccards)) if jaccards else float('nan')

    return {
        'rank_ic_mean': rank_ic_mean,
        'rank_ic_ir': rank_ic_ir,
        'decile_spread_bps': decile_spread_bps,
        'hit_rate': hit_rate,
        'top_k_jaccard': top_k_jaccard,
    }

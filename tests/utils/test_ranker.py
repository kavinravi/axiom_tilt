"""Tests for src.utils.ranker — helpers behind notebook 06."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA

from src.utils.ranker import (
    assemble_walk_features,
    build_ranker,
    compute_excess_return_buckets,
    evaluate_ranker,
    friday_only,
    load_walk_pca,
    project_text_to_pca,
)


# -------------------------------- load_walk_pca --------------------------------


def test_load_walk_pca_returns_pca_and_n_components(tmp_path):
    walk_dir = tmp_path / 'artifacts' / 'pca-text' / 'walk-001'
    walk_dir.mkdir(parents=True)
    pca = PCA(n_components=5).fit(np.random.RandomState(0).randn(50, 20))
    joblib.dump(pca, walk_dir / 'pca.joblib')

    loaded_pca, n_pca = load_walk_pca(walk_id=1, artifacts_root=tmp_path / 'artifacts')
    assert isinstance(loaded_pca, PCA)
    assert n_pca == 5


# -------------------------------- project_text_to_pca --------------------------


def test_project_text_to_pca_returns_correct_shape_and_columns():
    rng = np.random.RandomState(0)
    pca = PCA(n_components=3).fit(rng.randn(20, 10))
    embed = pd.DataFrame({
        'permno': [101, 102, 103],
        'date': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03']),
        'vec': [rng.randn(10).astype(np.float32) for _ in range(3)],
    })
    out = project_text_to_pca(embed, pca)
    assert list(out.columns) == ['permno', 'date', 'pca_0', 'pca_1', 'pca_2']
    assert len(out) == 3
    assert out['pca_0'].dtype == np.float32


# -------------------------------- friday_only ----------------------------------


def test_friday_only_keeps_only_weekday_4():
    df = pd.DataFrame({
        # Wed, Thu, Fri, Mon, Fri
        'date': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03',
                                '2020-01-06', '2020-01-10']),
        'x': [1, 2, 3, 4, 5],
    })
    out = friday_only(df)
    assert out['date'].dt.dayofweek.unique().tolist() == [4]
    assert len(out) == 2
    assert out['x'].tolist() == [3, 5]


# -------------------------------- compute_excess_return_buckets ----------------


def test_compute_excess_return_buckets_higher_excess_higher_bucket():
    df = pd.DataFrame({
        'permno': [101, 102, 103, 104, 105, 106],
        'date': pd.to_datetime(['2020-01-03'] * 3 + ['2020-01-10'] * 3),
        'fwd_ret_5d': [0.01, 0.02, 0.03, -0.01, 0.00, 0.01],
    })
    out = compute_excess_return_buckets(df, n_buckets=3)
    # Within each date, larger excess return gets a higher bucket.
    assert out.iloc[0] < out.iloc[2]
    assert out.iloc[3] < out.iloc[5]
    assert out.dropna().astype(int).between(0, 2).all()


def test_compute_excess_return_buckets_drops_nan_rows():
    df = pd.DataFrame({
        'permno': [101, 102],
        'date': pd.to_datetime(['2020-01-03', '2020-01-03']),
        'fwd_ret_5d': [0.01, np.nan],
    })
    out = compute_excess_return_buckets(df, n_buckets=2)
    assert pd.isna(out.iloc[1])


# -------------------------------- assemble_walk_features -----------------------


def test_assemble_walk_features_joins_panel_and_pca_drops_non_features():
    # 2020-01-03 and 2020-01-10 are both Fridays.
    panel = pd.DataFrame({
        'permno': [101, 102, 101, 102],
        'date': pd.to_datetime(['2020-01-03'] * 2 + ['2020-01-10'] * 2),
        'cik': ['a', 'b', 'a', 'b'],
        'ret': [0.01, 0.02, 0.0, 0.01],
        'ticker': ['A', 'B', 'A', 'B'],
        'fwd_ret_5d': [0.01, 0.02, 0.0, 0.01],
        'macro_vixcls': [20.0, 20.0, 22.0, 22.0],
        'text_novelty': [0.1, 0.2, 0.15, 0.25],
        'feature_x': [1.0, 2.0, 3.0, 4.0],
    })
    embed_pca = pd.DataFrame({
        'permno': [101, 102, 101, 102],
        'date': pd.to_datetime(['2020-01-03'] * 2 + ['2020-01-10'] * 2),
        'pca_0': [0.5, 0.6, 0.7, 0.8],
        'pca_1': [1.0, 1.1, 1.2, 1.3],
    })
    X, y, groups, meta = assemble_walk_features(panel, embed_pca)
    # Non-feature columns dropped from X
    for col in ('permno', 'date', 'cik', 'ret', 'ticker', 'fwd_ret_5d'):
        assert col not in X.columns
    assert {'pca_0', 'pca_1', 'macro_vixcls', 'text_novelty', 'feature_x'} <= set(X.columns)
    assert len(X) == len(y) == 4
    assert groups == [2, 2]
    assert {'permno', 'date', 'fwd_ret_5d'} <= set(meta.columns)


def test_assemble_walk_features_drops_non_friday_rows():
    # Mix Friday and non-Friday rows.
    panel = pd.DataFrame({
        'permno': [101, 101],
        'date': pd.to_datetime(['2020-01-03', '2020-01-06']),  # Fri, Mon
        'cik': ['a', 'a'],
        'ret': [0.0, 0.0],
        'ticker': ['A', 'A'],
        'fwd_ret_5d': [0.01, 0.02],
        'feature_x': [1.0, 2.0],
    })
    embed_pca = pd.DataFrame({
        'permno': [101, 101],
        'date': pd.to_datetime(['2020-01-03', '2020-01-06']),
        'pca_0': [0.5, 0.6],
    })
    X, y, groups, meta = assemble_walk_features(panel, embed_pca)
    assert len(X) == 1
    assert meta['date'].iloc[0] == pd.Timestamp('2020-01-03')


# -------------------------------- build_ranker ---------------------------------


def test_build_ranker_returns_lgbm_ranker_with_lambdarank_defaults():
    from lightgbm import LGBMRanker
    model = build_ranker({'num_leaves': 31, 'learning_rate': 0.05, 'n_estimators': 50})
    assert isinstance(model, LGBMRanker)
    assert model.objective == 'lambdarank'
    assert model.num_leaves == 31


# -------------------------------- evaluate_ranker ------------------------------


def test_evaluate_ranker_returns_metric_dict_with_required_keys():
    from lightgbm import LGBMRanker
    rng = np.random.RandomState(0)
    n_dates, n_per_date = 5, 40
    X = rng.randn(n_dates * n_per_date, 8).astype(np.float32)
    y_excess = X[:, 0] * 0.5 + rng.randn(len(X)) * 0.1
    groups = [n_per_date] * n_dates
    dates = pd.to_datetime([f'2020-01-{i * 7 + 3:02d}' for i in range(n_dates)])
    group_dates = np.repeat(dates.to_numpy(), n_per_date)

    pct = pd.Series(y_excess).groupby(group_dates).rank(pct=True)
    labels = np.floor(pct * 4).clip(upper=3).astype(int).values
    model = LGBMRanker(objective='lambdarank', n_estimators=100, verbose=-1)
    model.fit(X, labels, group=groups)

    out = evaluate_ranker(model, X, y_excess, group_dates, top_k=10)
    for k in ('rank_ic_mean', 'rank_ic_ir', 'decile_spread_bps',
              'hit_rate', 'top_k_jaccard'):
        assert k in out
    assert isinstance(out['rank_ic_mean'], float)

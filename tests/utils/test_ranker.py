"""Tests for src.utils.ranker — helpers behind notebook 06."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA

from src.utils.ranker import load_walk_pca, project_text_to_pca


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

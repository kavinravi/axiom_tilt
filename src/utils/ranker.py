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

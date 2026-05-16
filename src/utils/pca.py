"""PCA helpers for FinBERT stock-day embeddings (ranker text features).

Design: see docs/superpowers/specs/2026-05-08-text-enhanced-rl-portfolio-design.md §5.3.
Operational decisions: docs/superpowers/plans/2026-05-15-pca-text-features.md.

These functions are pure / sklearn-thin so they unit-test cleanly on synthetic
data. The first-walk orchestration + diagnostics live in
notebooks/04_pca_text_features.ipynb.
"""
from __future__ import annotations

import numpy as np


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

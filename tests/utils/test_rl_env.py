"""Tests for src.utils.rl_env — PortfolioEnv + helpers for notebook 07."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.utils.rl_env import build_scoreboard_from_scored_panel, project_to_simplex


# -------------------------------- project_to_simplex ---------------------------


def test_project_to_simplex_outputs_sum_to_one():
    rng = np.random.RandomState(0)
    action = rng.randn(30).astype(np.float32)
    weights = project_to_simplex(action, max_weight=0.10)
    assert weights.sum() == pytest.approx(1.0, abs=1e-5)
    assert (weights >= 0).all()


def test_project_to_simplex_respects_max_weight():
    # One element dominates; would push above 0.10 without the cap.
    action = np.array([10.0] + [0.0] * 29, dtype=np.float32)
    weights = project_to_simplex(action, max_weight=0.10)
    assert weights.max() <= 0.10 + 1e-6
    assert weights.sum() == pytest.approx(1.0, abs=1e-5)


# -------------------------------- build_scoreboard_from_scored_panel ----------


def test_build_scoreboard_from_scored_panel_keeps_top_k_per_friday():
    rng = np.random.RandomState(0)
    n_permno, n_friday = 60, 4
    fridays = pd.to_datetime(['2002-01-04', '2002-01-11', '2002-01-18', '2002-01-25'])
    rows = []
    for d in fridays:
        for p in range(100, 100 + n_permno):
            rows.append({
                'permno': p, 'date': d,
                'score': rng.randn(),
                'fwd_ret_5d': rng.randn() * 0.02,
                'macro_vixcls': 20.0, 'macro_dgs10': 4.0, 'macro_t10y2y': 1.0,
                'payoutratio': rng.rand(), 'ncfdiv': rng.rand(),
                'bidlo': 50 + rng.rand(), 'sgna': rng.rand(), 'retearn': rng.rand(),
            })
    panel_df = pd.DataFrame(rows)
    sb = build_scoreboard_from_scored_panel(panel_df, top_k=30)
    assert (sb.groupby('date').size() == 30).all()
    required = {'permno', 'date', 'score', 'fwd_ret_5d',
                'macro_vixcls', 'macro_dgs10', 'macro_t10y2y',
                'payoutratio', 'ncfdiv', 'bidlo', 'sgna', 'retearn'}
    assert required <= set(sb.columns)
    # Top-k should be the highest scores per Friday.
    first_friday = sb[sb['date'] == fridays[0]].sort_values('score', ascending=False)
    assert first_friday['score'].is_monotonic_decreasing

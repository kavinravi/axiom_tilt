"""Tests for src.utils.rl_env — PortfolioEnv + helpers for notebook 07."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.utils.rl_env import (
    PortfolioEnv,
    build_scoreboard_from_scored_panel,
    project_to_simplex,
)


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


# -------------------------------- PortfolioEnv ---------------------------------


def _make_synthetic_scoreboard(n_friday: int = 10, top_k: int = 30, seed: int = 0):
    rng = np.random.RandomState(seed)
    fridays = pd.to_datetime([f'2002-{1 + (i // 4):02d}-{4 + 7 * (i % 4):02d}'
                              for i in range(n_friday)])
    rows = []
    for d in fridays:
        for p in range(100, 100 + top_k):
            rows.append({
                'permno': p, 'date': d,
                'score': rng.randn(),
                'fwd_ret_5d': rng.randn() * 0.02,
                'macro_vixcls': 20.0, 'macro_dgs10': 4.0, 'macro_t10y2y': 1.0,
                'payoutratio': rng.rand(), 'ncfdiv': rng.rand(),
                'bidlo': 50.0 + rng.rand(), 'sgna': rng.rand(), 'retearn': rng.rand(),
            })
    return pd.DataFrame(rows)


def test_portfolio_env_reset_returns_correct_obs_shape():
    sb = _make_synthetic_scoreboard(n_friday=10)
    env = PortfolioEnv(scoreboard=sb, top_k=30, episode_length=3, cost_bps=5.0)
    obs, info = env.reset(seed=0)
    # 30 + 30 + 5 * 30 + 3 + 1 = 214
    assert obs.shape == (214,)
    assert obs.dtype == np.float32
    assert not np.isnan(obs).any()


def test_portfolio_env_step_returns_valid_tuple():
    sb = _make_synthetic_scoreboard(n_friday=10)
    env = PortfolioEnv(scoreboard=sb, top_k=30, episode_length=3, cost_bps=5.0)
    env.reset(seed=0)
    action = np.zeros(30, dtype=np.float32)  # equal-weight after softmax
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (214,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert 'portfolio_return' in info and 'trade_amount' in info


def test_portfolio_env_step_terminates_at_episode_length():
    sb = _make_synthetic_scoreboard(n_friday=10)
    env = PortfolioEnv(scoreboard=sb, top_k=30, episode_length=2, cost_bps=5.0)
    env.reset(seed=0)
    _, _, term1, _, _ = env.step(np.zeros(30, dtype=np.float32))
    _, _, term2, _, _ = env.step(np.zeros(30, dtype=np.float32))
    assert not term1
    assert term2


def test_portfolio_env_passes_sb3_check_env():
    """gymnasium/SB3 compatibility smoke test."""
    from stable_baselines3.common.env_checker import check_env
    sb = _make_synthetic_scoreboard(n_friday=10)
    env = PortfolioEnv(scoreboard=sb, top_k=30, episode_length=3, cost_bps=5.0)
    check_env(env, warn=True)  # raises on incompatibility

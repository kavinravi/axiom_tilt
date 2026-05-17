"""Tests for src.utils.backtest — helpers behind notebook 08."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.utils.backtest import (
    compute_strategy_metrics,
    equal_weight_weights,
    min_variance_weights,
    score_proportional_weights,
)


# -------------------------------- equal_weight_weights -------------------------


def test_equal_weight_weights_sum_to_one_and_uniform():
    w = equal_weight_weights(30)
    assert w.shape == (30,)
    assert w.sum() == pytest.approx(1.0)
    assert np.allclose(w, 1.0 / 30)


# -------------------------------- score_proportional_weights ------------------


def test_score_proportional_weights_higher_score_higher_weight():
    # K=4 needs max_weight >= 0.25 for the simplex to be feasible.
    scores = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    w = score_proportional_weights(scores, max_weight=0.50)
    assert w.sum() == pytest.approx(1.0)
    # Monotonic: higher score → higher weight (modulo the cap)
    assert w[0] < w[1] <= w[2] <= w[3]


def test_score_proportional_weights_respects_max_weight():
    # One score dominates — without cap it'd push above 0.10.
    scores = np.array([100.0] + [0.0] * 29, dtype=np.float32)
    w = score_proportional_weights(scores, max_weight=0.10)
    assert w.max() <= 0.10 + 1e-6
    assert w.sum() == pytest.approx(1.0, abs=1e-5)


# -------------------------------- min_variance_weights ------------------------


def test_min_variance_weights_handles_simple_case():
    # 3 uncorrelated assets, equal variance → minvar = equal-weight.
    rng = np.random.RandomState(0)
    returns_history = rng.randn(50, 3).astype(np.float32) * 0.01
    w = min_variance_weights(returns_history, max_weight=0.5)
    assert w.shape == (3,)
    assert w.sum() == pytest.approx(1.0, abs=1e-4)
    assert (w >= 0).all()
    # All should be roughly 1/3 (within optimization tolerance for noisy data)
    assert np.allclose(w, 1.0 / 3, atol=0.1)


def test_min_variance_weights_respects_max_weight():
    rng = np.random.RandomState(0)
    returns_history = rng.randn(50, 10).astype(np.float32) * 0.01
    w = min_variance_weights(returns_history, max_weight=0.20)
    assert w.max() <= 0.20 + 1e-4
    assert w.sum() == pytest.approx(1.0, abs=1e-4)


# -------------------------------- compute_strategy_metrics --------------------


def test_compute_strategy_metrics_returns_required_keys():
    rng = np.random.RandomState(0)
    weekly_returns = rng.randn(52) * 0.02  # 52 weeks, ~2% weekly vol
    weekly_turnover = np.full(52, 0.2, dtype=np.float32)
    out = compute_strategy_metrics(weekly_returns, weekly_turnover, cost_bps=5.0)
    required = {
        'total_return_gross', 'total_return_net',
        'annualized_return', 'annualized_vol',
        'sharpe', 'sortino',
        'max_drawdown', 'calmar',
        'hit_rate', 'avg_turnover',
    }
    assert required <= set(out.keys())
    assert isinstance(out['sharpe'], float)


def test_compute_strategy_metrics_zero_turnover_means_gross_equals_net():
    weekly_returns = np.array([0.01, -0.005, 0.02, 0.0], dtype=np.float64)
    weekly_turnover = np.zeros(4, dtype=np.float32)
    out = compute_strategy_metrics(weekly_returns, weekly_turnover, cost_bps=5.0)
    assert out['total_return_gross'] == pytest.approx(out['total_return_net'])


def test_compute_strategy_metrics_handles_all_negative_returns():
    """Sortino/Calmar should still produce a finite number when returns are bad."""
    weekly_returns = np.full(52, -0.005, dtype=np.float64)  # -0.5% per week, all neg
    weekly_turnover = np.full(52, 0.1, dtype=np.float32)
    out = compute_strategy_metrics(weekly_returns, weekly_turnover, cost_bps=5.0)
    assert out['annualized_return'] < 0
    assert out['max_drawdown'] < 0

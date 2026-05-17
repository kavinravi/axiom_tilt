"""Backtest helpers for notebook 08.

Pure functions over numpy + pandas. See
docs/superpowers/specs/2026-05-17-backtest-design.md for design.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import cvxpy as cp
    _HAS_CVXPY = True
except ImportError:
    _HAS_CVXPY = False


def equal_weight_weights(k: int) -> np.ndarray:
    """Equal-weight portfolio over `k` assets."""
    return np.full(k, 1.0 / k, dtype=np.float32)


def score_proportional_weights(
    scores: np.ndarray,
    max_weight: float = 0.10,
) -> np.ndarray:
    """Softmax(scores) -> water-fill cap at `max_weight`.

    Uses the same projection logic as PortfolioEnv.project_to_simplex so
    score-prop and PPO weights live in the same feasible set.
    """
    from src.utils.rl_env import project_to_simplex
    return project_to_simplex(np.asarray(scores, dtype=np.float64), max_weight)


def min_variance_weights(
    returns_history: np.ndarray,
    max_weight: float = 0.10,
    ridge: float = 1e-6,
) -> np.ndarray:
    """Solve min w'Σw s.t. sum(w)=1, 0 ≤ w ≤ max_weight via cvxpy.

    `returns_history` shape: (n_periods, k_assets). Ridge term `ridge*I`
    regularizes Σ to avoid singular-matrix failures with short history.
    """
    if not _HAS_CVXPY:
        raise ImportError('cvxpy required for min_variance_weights')

    X = np.asarray(returns_history, dtype=np.float64)
    # Drop columns / rows that are entirely NaN.
    X = np.nan_to_num(X, nan=0.0)
    k = X.shape[1]
    sigma = np.cov(X, rowvar=False) + ridge * np.eye(k)

    w = cp.Variable(k)
    objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(sigma)))
    constraints = [cp.sum(w) == 1, w >= 0, w <= max_weight]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    if w.value is None:
        # Fallback to equal-weight if solver fails.
        return equal_weight_weights(k)
    return np.asarray(w.value, dtype=np.float32)


def compute_strategy_metrics(
    weekly_returns: np.ndarray,
    weekly_turnover: np.ndarray,
    cost_bps: float = 5.0,
    periods_per_year: int = 52,
) -> dict[str, float]:
    """Compute the per-spec §17.5 metrics.

    Both `weekly_returns` and `weekly_turnover` are gross (pre-cost). Net
    returns subtract `(cost_bps / 1e4) * turnover`.
    """
    r_gross = np.asarray(weekly_returns, dtype=np.float64)
    tu = np.asarray(weekly_turnover, dtype=np.float64)
    cost_per_week = (cost_bps / 10_000.0) * tu
    r_net = r_gross - cost_per_week

    eq_gross = np.cumprod(1.0 + r_gross)
    eq_net = np.cumprod(1.0 + r_net)
    total_gross = float(eq_gross[-1] - 1.0) if len(eq_gross) else 0.0
    total_net = float(eq_net[-1] - 1.0) if len(eq_net) else 0.0

    n = len(r_net)
    if n == 0:
        return {k: float('nan') for k in
                ('total_return_gross', 'total_return_net', 'annualized_return',
                 'annualized_vol', 'sharpe', 'sortino', 'max_drawdown',
                 'calmar', 'hit_rate', 'avg_turnover')}

    annualized_return = float((1.0 + total_net) ** (periods_per_year / n) - 1.0)
    annualized_vol = float(np.std(r_net, ddof=1) * np.sqrt(periods_per_year)) if n > 1 else 0.0
    sharpe = float(annualized_return / annualized_vol) if annualized_vol > 0 else float('nan')

    downside = r_net[r_net < 0]
    downside_vol = (float(np.std(downside, ddof=1) * np.sqrt(periods_per_year))
                    if len(downside) > 1 else 0.0)
    sortino = (float(annualized_return / downside_vol)
               if downside_vol > 0 else float('nan'))

    # Max drawdown on net equity curve.
    running_max = np.maximum.accumulate(eq_net)
    drawdown = eq_net / running_max - 1.0
    max_dd = float(drawdown.min())
    calmar = float(annualized_return / abs(max_dd)) if max_dd < 0 else float('nan')

    hit_rate = float((r_net > 0).mean())
    avg_turnover = float(tu.mean())

    return {
        'total_return_gross': total_gross,
        'total_return_net': total_net,
        'annualized_return': annualized_return,
        'annualized_vol': annualized_vol,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_drawdown': max_dd,
        'calmar': calmar,
        'hit_rate': hit_rate,
        'avg_turnover': avg_turnover,
    }

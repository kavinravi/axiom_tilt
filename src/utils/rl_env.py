"""Reinforcement-learning helpers for notebook 07.

Pure functions over numpy + a PortfolioEnv class. See
docs/superpowers/specs/2026-05-17-rl-agent-design.md for design.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


def project_to_simplex(action: np.ndarray, max_weight: float = 0.10) -> np.ndarray:
    """Map a raw real-valued action vector to long-only weights.

    softmax -> water-fill cap: iteratively clip over-cap weights and
    redistribute excess proportionally to the under-cap ones. Naive
    clip+renorm fails when one weight dominates (renorm undoes the clip).
    """
    K = len(action)
    if K * max_weight < 1.0:
        raise ValueError(f'K * max_weight = {K * max_weight} < 1 — infeasible simplex')

    a = np.asarray(action, dtype=np.float64)
    a = a - a.max()  # numerical stability
    w = np.exp(a)
    w = w / w.sum()  # softmax

    # Iterative water-fill (converges in O(K) iterations worst-case).
    for _ in range(K):
        over = w > max_weight + 1e-12
        if not over.any():
            break
        excess = float((w[over] - max_weight).sum())
        w[over] = max_weight
        under = ~over
        under_sum = float(w[under].sum())
        if under_sum <= 0:
            # All names capped; just equal-fill the slack (shouldn't happen if K*cap >= 1).
            w[under] = excess / max(1, under.sum())
        else:
            w[under] = w[under] + excess * (w[under] / under_sum)

    # Final renorm to cancel float drift.
    return (w / w.sum()).astype(np.float32)


TOP_FEATURES = ['payoutratio', 'ncfdiv', 'bidlo', 'sgna', 'retearn']
MACRO_COLS = ['macro_vixcls', 'macro_dgs10', 'macro_t10y2y']


def build_scoreboard_from_scored_panel(
    panel_df: pd.DataFrame,
    top_k: int = 30,
    date_col: str = 'date',
    score_col: str = 'score',
    target_col: str = 'fwd_ret_5d',
) -> pd.DataFrame:
    """Given a Friday-only panel with a pre-computed `score` column, keep
    top-K by score per date. Returns columns:
    [permno, date, score, fwd_ret_5d, *MACRO_COLS, *TOP_FEATURES].
    """
    keep = ['permno', date_col, score_col, target_col, *MACRO_COLS, *TOP_FEATURES]
    df = panel_df[keep].copy()
    df = (df.sort_values([date_col, score_col], ascending=[True, False])
            .groupby(date_col, sort=False, group_keys=False)
            .head(top_k)
            .reset_index(drop=True))
    return df


# Observation dim: K weights + K scores + 5 features * K stocks + 3 macro + 1 recent return.
def _obs_dim(top_k: int) -> int:
    return top_k + top_k + len(TOP_FEATURES) * top_k + len(MACRO_COLS) + 1


class PortfolioEnv(gym.Env):
    """Walk-1 portfolio allocation env (gymnasium-compatible).

    Each step picks weights over the top-K (already filtered by ranker score
    for the current Friday), realizes a 5-day forward return, advances one
    Friday. Reward = portfolio_return - (cost_bps/1e4) * trade_amount.
    """

    metadata = {'render_modes': []}

    def __init__(
        self,
        scoreboard: pd.DataFrame,
        top_k: int = 30,
        episode_length: int = 52,
        cost_bps: float = 5.0,
        max_weight: float = 0.10,
    ):
        super().__init__()
        self.scoreboard = scoreboard.sort_values('date').reset_index(drop=True)
        self.top_k = int(top_k)
        self.episode_length = int(episode_length)
        self.cost_bps = float(cost_bps)
        self.max_weight = float(max_weight)

        self._dates = np.array(sorted(self.scoreboard['date'].unique()))
        self._by_date: dict = {d: g.reset_index(drop=True)
                               for d, g in self.scoreboard.groupby('date', sort=True)}

        # Finite, symmetric bounds per SB3 recommendation. Softmax handles any input
        # but PPO outputs roughly tanh-shaped actions, so [-1, 1] is the natural range.
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.top_k,), dtype=np.float32,
        )
        # Wide finite bounds for observation; post-VecNormalize values stay within ~10.
        self.observation_space = spaces.Box(
            low=-100.0, high=100.0, shape=(_obs_dim(self.top_k),), dtype=np.float32,
        )

        # Initialize state attrs so reset() can be called even before _build_obs.
        self._idx = 0
        self._steps = 0
        self._weights = np.full(self.top_k, 1.0 / self.top_k, dtype=np.float32)
        self._last_return = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        max_start = max(0, len(self._dates) - self.episode_length - 1)
        self._idx = int(self.np_random.integers(0, max_start + 1)) if max_start > 0 else 0
        self._steps = 0
        self._weights = np.full(self.top_k, 1.0 / self.top_k, dtype=np.float32)
        self._last_return = 0.0
        return self._build_obs(), {}

    def step(self, action: np.ndarray):
        cur = self._by_date[self._dates[self._idx]]
        new_weights = project_to_simplex(action, max_weight=self.max_weight)

        rets = cur['fwd_ret_5d'].to_numpy(dtype=np.float32)[:self.top_k]
        rets = np.nan_to_num(rets, nan=0.0)
        portfolio_return = float(np.dot(new_weights, rets))
        # Baseline: equal-weight over the same top-K — strips out market beta so
        # reward measures the agent's alpha vs the "no-skill within top-K" prior.
        eq_weights = np.full(self.top_k, 1.0 / self.top_k, dtype=np.float32)
        baseline_return = float(np.dot(eq_weights, rets))
        excess_return = portfolio_return - baseline_return
        trade_amount = float(np.abs(new_weights - self._weights).sum())
        reward = excess_return - (self.cost_bps / 10_000.0) * trade_amount

        self._weights = new_weights
        self._last_return = portfolio_return
        self._idx += 1
        self._steps += 1
        terminated = self._steps >= self.episode_length or self._idx >= len(self._dates)
        return (self._build_obs(), float(reward), bool(terminated), False,
                {'portfolio_return': portfolio_return,
                 'baseline_return': baseline_return,
                 'excess_return': excess_return,
                 'trade_amount': trade_amount})

    def _build_obs(self) -> np.ndarray:
        # If we've stepped past the last date, obs is from previous date's snapshot
        # (terminated=True is already set; SB3 won't use this obs for action selection).
        idx = min(self._idx, len(self._dates) - 1)
        cur = self._by_date[self._dates[idx]]

        # 1. weights (K)
        w = self._weights
        # 2. ranker scores (K), z-scored within date
        scores = cur['score'].to_numpy(dtype=np.float32)[:self.top_k]
        scores = (scores - scores.mean()) / (scores.std() + 1e-8)
        # 3. top-5 features per stock (5 * K), z-scored within date
        feats = []
        for col in TOP_FEATURES:
            v = cur[col].to_numpy(dtype=np.float32)[:self.top_k]
            v = (v - np.nanmean(v)) / (np.nanstd(v) + 1e-8)
            feats.append(v)
        feats = np.concatenate(feats)
        # 4. macro (3) — same across the date's rows
        macro = cur[MACRO_COLS].iloc[0].to_numpy(dtype=np.float32)
        # 5. recent portfolio return (1)
        recent = np.array([self._last_return], dtype=np.float32)

        obs = np.concatenate([w, scores, feats, macro, recent]).astype(np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

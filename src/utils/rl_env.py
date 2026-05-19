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
# If include_portfolio_state, add `history_len` past portfolio returns.
def _obs_dim(top_k: int, include_portfolio_state: bool = False, history_len: int = 4) -> int:
    base = top_k + top_k + len(TOP_FEATURES) * top_k + len(MACRO_COLS) + 1
    if include_portfolio_state:
        base += history_len
    return base


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
        reward_type: str = 'excess_return',
        sharpe_window: int = 8,
        downside_lambda: float = 5.0,
        action_high: float = 5.0,
        score_bias: float = 0.0,
        baseline_anchor: bool = False,
        tilt_scale: float = 1.0,
        include_portfolio_state: bool = False,
        history_len: int = 4,
        cost_anneal_episodes: int = 0,
    ):
        super().__init__()
        self.scoreboard = scoreboard.sort_values('date').reset_index(drop=True)
        self.top_k = int(top_k)
        self.episode_length = int(episode_length)
        self.cost_bps = float(cost_bps)
        self.max_weight = float(max_weight)
        # Reward variants:
        #   'excess_return'    : alpha vs equal-weight top-K (default)
        #   'sharpe'           : rolling Sharpe proxy of excess_return over `sharpe_window`
        #   'sharpe_total'     : rolling Sharpe proxy of portfolio_return (not excess)
        #   'return_only'      : raw portfolio return (no alpha subtraction)
        #   'downside_penalty' : excess_return - downside_lambda * max(0, -portfolio_return)
        #   'ir_vs_baseline'   : rolling Sharpe of (port_ret - score_prop_ret); active-return signal
        self.reward_type = str(reward_type)
        self.sharpe_window = int(sharpe_window)
        self.downside_lambda = float(downside_lambda)
        self.action_high = float(action_high)
        # score_bias: legacy additive-bias-on-action (autoresearch round 1, dead end with sharpe reward).
        self.score_bias = float(score_bias)
        # baseline_anchor: when True, treat action as a log-tilt added to log(score_prop_weights).
        # action=0 → exactly Score-Prop weights. PPO learns deviations from baseline, not absolute weights.
        self.baseline_anchor = bool(baseline_anchor)
        self.tilt_scale = float(tilt_scale)
        # include_portfolio_state: extend obs with last `history_len` portfolio returns (proprioception).
        self.include_portfolio_state = bool(include_portfolio_state)
        self.history_len = int(history_len)
        # cost_anneal_episodes: ramp effective cost from 0 → cost_bps linearly over N episodes
        # (counted across resets). 0 disables annealing.
        self.cost_anneal_episodes = int(cost_anneal_episodes)

        self._dates = np.array(sorted(self.scoreboard['date'].unique()))
        self._by_date: dict = {d: g.reset_index(drop=True)
                               for d, g in self.scoreboard.groupby('date', sort=True)}

        # Action bounds (symmetric, finite per SB3 recommendation). Default 5.0
        # gives softmax room to concentrate ~99% on one stock; 10.0 even sharper.
        self.action_space = spaces.Box(
            low=-self.action_high, high=self.action_high,
            shape=(self.top_k,), dtype=np.float32,
        )
        # Wide finite bounds for observation; post-VecNormalize values stay within ~10.
        self.observation_space = spaces.Box(
            low=-100.0, high=100.0,
            shape=(_obs_dim(self.top_k, self.include_portfolio_state, self.history_len),),
            dtype=np.float32,
        )

        # Initialize state attrs so reset() can be called even before _build_obs.
        self._idx = 0
        self._steps = 0
        self._episode_count = 0
        self._weights = np.full(self.top_k, 1.0 / self.top_k, dtype=np.float32)
        self._last_return = 0.0
        self._return_history: list = []  # for sharpe-style reward
        self._port_return_history: list = []  # for include_portfolio_state obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        max_start = max(0, len(self._dates) - self.episode_length - 1)
        self._idx = int(self.np_random.integers(0, max_start + 1)) if max_start > 0 else 0
        self._steps = 0
        self._episode_count += 1
        self._weights = np.full(self.top_k, 1.0 / self.top_k, dtype=np.float32)
        self._last_return = 0.0
        self._return_history = []
        self._port_return_history = []
        return self._build_obs(), {}

    def step(self, action: np.ndarray):
        cur = self._by_date[self._dates[self._idx]]
        raw_scores = cur['score'].to_numpy(dtype=np.float32)[:self.top_k]

        # Compute Score-Prop baseline weights (used for tilt-anchor and ir_vs_baseline reward).
        # Only call if needed — cheap but adds a softmax + water-fill per step.
        need_baseline = self.baseline_anchor or self.reward_type == 'ir_vs_baseline'
        baseline_weights = (project_to_simplex(raw_scores, max_weight=self.max_weight)
                            if need_baseline else None)

        if self.score_bias > 0:
            action = np.asarray(action, dtype=np.float32) + self.score_bias * raw_scores

        if self.baseline_anchor:
            # action = 0 → softmax(log baseline) = baseline (water-fill no-op since baseline
            # already satisfies cap). PPO learns deviations from Score-Prop.
            log_baseline = np.log(baseline_weights + 1e-8)
            tilted = log_baseline + self.tilt_scale * np.asarray(action, dtype=np.float32)
            new_weights = project_to_simplex(tilted, max_weight=self.max_weight)
        else:
            new_weights = project_to_simplex(action, max_weight=self.max_weight)

        rets = cur['fwd_ret_5d'].to_numpy(dtype=np.float32)[:self.top_k]
        rets = np.nan_to_num(rets, nan=0.0)
        portfolio_return = float(np.dot(new_weights, rets))
        eq_weights = np.full(self.top_k, 1.0 / self.top_k, dtype=np.float32)
        baseline_return = float(np.dot(eq_weights, rets))  # equal-weight (legacy)
        excess_return = portfolio_return - baseline_return
        trade_amount = float(np.abs(new_weights - self._weights).sum())

        # Cost annealing: ramp 0 → cost_bps over the first `cost_anneal_episodes` resets.
        if self.cost_anneal_episodes > 0:
            progress = min(1.0, self._episode_count / float(self.cost_anneal_episodes))
            effective_cost_bps = self.cost_bps * progress
        else:
            effective_cost_bps = self.cost_bps
        cost = (effective_cost_bps / 10_000.0) * trade_amount

        # Reward shape selector.
        if self.reward_type == 'excess_return':
            reward = excess_return - cost
        elif self.reward_type == 'return_only':
            reward = portfolio_return - cost
        elif self.reward_type == 'downside_penalty':
            downside = max(0.0, -portfolio_return)
            reward = excess_return - self.downside_lambda * downside - cost
        elif self.reward_type == 'sharpe':
            self._return_history.append(excess_return)
            if len(self._return_history) > self.sharpe_window:
                self._return_history = self._return_history[-self.sharpe_window:]
            if len(self._return_history) >= 2:
                arr = np.asarray(self._return_history, dtype=np.float64)
                std = float(arr.std()) + 1e-6
                reward = float(arr.mean() / std) - cost
            else:
                reward = excess_return - cost
        elif self.reward_type == 'sharpe_total':
            self._return_history.append(portfolio_return)
            if len(self._return_history) > self.sharpe_window:
                self._return_history = self._return_history[-self.sharpe_window:]
            if len(self._return_history) >= 2:
                arr = np.asarray(self._return_history, dtype=np.float64)
                std = float(arr.std()) + 1e-6
                reward = float(arr.mean() / std) - cost
            else:
                reward = portfolio_return - cost
        elif self.reward_type == 'ir_vs_baseline':
            # Rolling info ratio vs Score-Prop: mean(active) / std(active) over window.
            sp_return = float(np.dot(baseline_weights, rets))
            active_return = portfolio_return - sp_return
            self._return_history.append(active_return)
            if len(self._return_history) > self.sharpe_window:
                self._return_history = self._return_history[-self.sharpe_window:]
            if len(self._return_history) >= 2:
                arr = np.asarray(self._return_history, dtype=np.float64)
                std = float(arr.std()) + 1e-6
                reward = float(arr.mean() / std) - cost
            else:
                reward = active_return - cost
        else:
            raise ValueError(f'unknown reward_type: {self.reward_type}')

        self._port_return_history.append(portfolio_return)
        if len(self._port_return_history) > self.history_len:
            self._port_return_history = self._port_return_history[-self.history_len:]

        self._weights = new_weights
        self._last_return = portfolio_return
        self._idx += 1
        self._steps += 1
        terminated = self._steps >= self.episode_length or self._idx >= len(self._dates)
        return (self._build_obs(), float(reward), bool(terminated), False,
                {'portfolio_return': portfolio_return,
                 'baseline_return': baseline_return,
                 'excess_return': excess_return,
                 'trade_amount': trade_amount,
                 'effective_cost_bps': effective_cost_bps})

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

        parts = [w, scores, feats, macro, recent]

        # 6. (optional) last `history_len` portfolio returns, oldest-first, zero-padded.
        if self.include_portfolio_state:
            buf = np.zeros(self.history_len, dtype=np.float32)
            h = self._port_return_history[-self.history_len:]
            if h:
                buf[-len(h):] = np.asarray(h, dtype=np.float32)
            parts.append(buf)

        obs = np.concatenate(parts).astype(np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

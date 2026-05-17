"""Reinforcement-learning helpers for notebook 07.

Pure functions over numpy + a PortfolioEnv class. See
docs/superpowers/specs/2026-05-17-rl-agent-design.md for design.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


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

"""Tests for src.utils.rl_env — PortfolioEnv + helpers for notebook 07."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.utils.rl_env import project_to_simplex


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

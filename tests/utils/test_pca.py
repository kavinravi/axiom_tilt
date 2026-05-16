"""Tests for src.utils.pca — pure-function helpers behind notebook 04."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.utils.pca import pick_n_components


# -------------------------------- pick_n_components ----------------------------


def test_pick_n_components_target_in_middle():
    """Target 0.95 reached at index 2 (cum_var[2]=0.95) -> n=3, +1 safety = 4."""
    cum_var = np.array([0.5, 0.8, 0.95, 0.99, 1.0])
    assert pick_n_components(cum_var, target=0.95) == 4


def test_pick_n_components_target_exactly_first_component():
    """First component alone already exceeds target."""
    cum_var = np.array([0.99, 1.0])
    # n=1, +1 safety = 2
    assert pick_n_components(cum_var, target=0.95) == 2


def test_pick_n_components_target_at_last_caps_at_full_rank():
    """Target only hit by the last component -> cap at full rank (no n+1)."""
    cum_var = np.array([0.3, 0.6, 0.9, 0.99])
    assert pick_n_components(cum_var, target=0.99) == 4


def test_pick_n_components_target_unreachable_caps_at_full_rank():
    """Target above max cum_var: cap at len(cum_var)."""
    cum_var = np.array([0.3, 0.6, 0.85])
    assert pick_n_components(cum_var, target=0.99) == 3


def test_pick_n_components_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        pick_n_components(np.array([]), target=0.95)


def test_pick_n_components_invalid_target_raises():
    cum_var = np.array([0.5, 1.0])
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        pick_n_components(cum_var, target=1.5)
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        pick_n_components(cum_var, target=-0.1)

"""Tests for src.utils.pca — pure-function helpers behind notebook 04."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.utils.pca import (
    pick_n_components,
    weekly_snapshots,
)


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


# -------------------------------- weekly_snapshots -----------------------------


def test_weekly_snapshots_keeps_latest_per_week_per_permno():
    """Within one ISO week, keep the latest-date row for each permno."""
    df = pd.DataFrame({
        "permno": [101, 101, 101, 202, 202],
        "date": pd.to_datetime([
            "2020-01-06",  # Mon week 2
            "2020-01-08",  # Wed week 2
            "2020-01-10",  # Fri week 2  <- keep for 101
            "2020-01-07",  # Tue week 2
            "2020-01-09",  # Thu week 2  <- keep for 202
        ]),
        "vec": [[1.0], [2.0], [3.0], [10.0], [20.0]],
    })
    out = weekly_snapshots(df)
    assert len(out) == 2
    assert out.loc[out["permno"] == 101, "date"].iloc[0] == pd.Timestamp("2020-01-10")
    assert out.loc[out["permno"] == 202, "date"].iloc[0] == pd.Timestamp("2020-01-09")


def test_weekly_snapshots_one_row_per_week_across_boundaries():
    df = pd.DataFrame({
        "permno": [101, 101, 101],
        "date": pd.to_datetime([
            "2020-01-10",  # Fri week 2
            "2020-01-17",  # Fri week 3
            "2020-01-24",  # Fri week 4
        ]),
        "vec": [[1.0], [2.0], [3.0]],
    })
    out = weekly_snapshots(df)
    assert len(out) == 3


def test_weekly_snapshots_multiple_permnos_same_week_all_kept():
    df = pd.DataFrame({
        "permno": [101, 202, 303],
        "date": pd.to_datetime(["2020-01-08", "2020-01-09", "2020-01-10"]),
        "vec": [[1.0], [2.0], [3.0]],
    })
    out = weekly_snapshots(df)
    assert len(out) == 3
    assert set(out["permno"]) == {101, 202, 303}

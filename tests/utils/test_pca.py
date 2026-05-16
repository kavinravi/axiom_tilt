"""Tests for src.utils.pca — pure-function helpers behind notebook 04."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.utils.pca import (
    assemble_training_matrix,
    filter_in_universe,
    fit_pca_initial,
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


# -------------------------------- filter_in_universe ---------------------------


def _universe_ids_fixture() -> pd.DataFrame:
    """Three permnos: AAPL/GOOG open-ended, DROP with bounded window."""
    return pd.DataFrame({
        "ticker": ["AAPL", "GOOG", "DROP"],
        "permno": pd.array([101, 202, 303], dtype="Int64"),
        "date_in": pd.to_datetime(["2009-02-01", "2015-08-25", "2010-01-01"]),
        "date_out": pd.to_datetime([None, None, "2012-12-31"]),
    })


def test_filter_in_universe_keeps_rows_inside_window():
    panel = pd.DataFrame({
        "permno": [101, 101, 202],
        "date": pd.to_datetime(["2009-03-01", "2020-06-15", "2018-01-01"]),
    })
    out = filter_in_universe(panel, _universe_ids_fixture())
    assert len(out) == 3


def test_filter_in_universe_drops_rows_before_date_in():
    panel = pd.DataFrame({"permno": [101], "date": pd.to_datetime(["2008-01-01"])})
    out = filter_in_universe(panel, _universe_ids_fixture())
    assert len(out) == 0


def test_filter_in_universe_drops_rows_after_date_out():
    panel = pd.DataFrame({"permno": [303], "date": pd.to_datetime(["2013-01-01"])})
    out = filter_in_universe(panel, _universe_ids_fixture())
    assert len(out) == 0


def test_filter_in_universe_open_ended_window_keeps_current_dates():
    """date_out=NaT means active; rows after date_in are kept."""
    panel = pd.DataFrame({"permno": [101], "date": pd.to_datetime(["2099-01-01"])})
    out = filter_in_universe(panel, _universe_ids_fixture())
    assert len(out) == 1


def test_filter_in_universe_drops_unknown_permno():
    panel = pd.DataFrame({"permno": [999], "date": pd.to_datetime(["2020-01-01"])})
    out = filter_in_universe(panel, _universe_ids_fixture())
    assert len(out) == 0


# -------------------------------- assemble_training_matrix ---------------------


def _write_embed_shard(out_dir: Path, year: int, permno: int, rows: list[dict]) -> None:
    """Write a single per-permno year shard, matching notebook 03's output layout."""
    part = out_dir / f"year={year}" / f"part-permno-{permno:08d}.parquet"
    part.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([
        ("permno", pa.int64()),
        ("date", pa.timestamp("ns")),
        ("vec", pa.list_(pa.float32())),
    ])
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), part)


def test_assemble_training_matrix_filters_universe_window_and_resamples(tmp_path: Path):
    rows_2007 = [
        {"permno": 101, "date": pd.Timestamp("2007-06-07"), "vec": [0.0, 1.0, 0.0]},  # Thu
        {"permno": 101, "date": pd.Timestamp("2007-06-08"), "vec": [1.0, 0.0, 0.0]},  # Fri (same week — wins)
    ]
    rows_2007_other = [
        {"permno": 999, "date": pd.Timestamp("2007-06-08"), "vec": [9.0, 9.0, 9.0]},  # not in universe
    ]
    rows_2008 = [
        {"permno": 101, "date": pd.Timestamp("2008-01-04"), "vec": [2.0, 0.0, 0.0]},  # outside window
    ]
    _write_embed_shard(tmp_path, 2007, 101, rows_2007)
    _write_embed_shard(tmp_path, 2007, 999, rows_2007_other)
    _write_embed_shard(tmp_path, 2008, 101, rows_2008)

    universe_ids = pd.DataFrame({
        "ticker": ["AAPL"],
        "permno": pd.array([101], dtype="Int64"),
        "date_in": pd.to_datetime(["2000-01-01"]),
        "date_out": pd.to_datetime([None]),
    })
    X, meta = assemble_training_matrix(
        embed_dir=tmp_path,
        universe_ids=universe_ids,
        start="2007-01-01",
        end="2007-12-31",
    )
    assert X.shape == (1, 3)
    assert X.dtype == np.float32
    np.testing.assert_array_equal(X[0], [1.0, 0.0, 0.0])
    assert len(meta) == 1
    assert int(meta["permno"].iloc[0]) == 101
    assert meta["date"].iloc[0] == pd.Timestamp("2007-06-08")


def test_assemble_training_matrix_empty_match_returns_zero_rows(tmp_path: Path):
    _write_embed_shard(tmp_path, 2007, 101, [
        {"permno": 101, "date": pd.Timestamp("2007-06-08"), "vec": [1.0, 0.0]},
    ])
    # universe excludes 101
    universe_ids = pd.DataFrame({
        "ticker": ["X"],
        "permno": pd.array([999], dtype="Int64"),
        "date_in": pd.to_datetime(["2000-01-01"]),
        "date_out": pd.to_datetime([None]),
    })
    X, meta = assemble_training_matrix(tmp_path, universe_ids, "2007-01-01", "2007-12-31")
    assert X.shape == (0, 2)
    assert len(meta) == 0


def test_assemble_training_matrix_raises_when_no_shards(tmp_path: Path):
    universe_ids = pd.DataFrame({
        "ticker": ["AAPL"],
        "permno": pd.array([101], dtype="Int64"),
        "date_in": pd.to_datetime(["2000-01-01"]),
        "date_out": pd.to_datetime([None]),
    })
    with pytest.raises(FileNotFoundError, match="no parquet shards"):
        assemble_training_matrix(tmp_path, universe_ids, "2007-01-01", "2007-12-31")


# -------------------------------- fit_pca_initial ------------------------------


def test_fit_pca_initial_recovers_planted_low_rank_signal():
    """5-dim signal embedded into 50 with small noise -> n_pca should land near 5+1."""
    rng = np.random.RandomState(42)
    truth = rng.randn(1000, 5).astype(np.float32)
    proj = rng.randn(5, 50).astype(np.float32)
    noise = rng.randn(1000, 50).astype(np.float32) * 0.01
    X = truth @ proj + noise

    n_pca, cum_var, pca = fit_pca_initial(X, target=0.95)

    assert cum_var.shape == (50,)
    assert 5 <= n_pca <= 10  # tight band — signal dominates
    assert pca.n_components_ == n_pca
    # cum_var[n_pca - 1] meets the target (because we hit it or capped at full rank)
    assert cum_var[n_pca - 1] >= 0.95 - 1e-6 or n_pca == 50


def test_fit_pca_initial_higher_target_yields_more_components():
    rng = np.random.RandomState(0)
    X = rng.randn(500, 20).astype(np.float32)
    n95, _, _ = fit_pca_initial(X, target=0.95)
    n99, _, _ = fit_pca_initial(X, target=0.99)
    assert n99 >= n95

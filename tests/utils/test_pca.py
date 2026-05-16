"""Tests for src.utils.pca + the sklearn contract notebook 04 relies on."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from sklearn.decomposition import PCA

from src.utils.pca import assemble_training_matrix


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


def test_assemble_training_matrix_filters_universe_and_window(tmp_path: Path):
    """Keep rows inside both the (date_in, date_out) interval and the [start, end] window."""
    rows_2007_101 = [
        {"permno": 101, "date": pd.Timestamp("2007-06-07"), "vec": [0.0, 1.0, 0.0]},
        {"permno": 101, "date": pd.Timestamp("2007-06-08"), "vec": [1.0, 0.0, 0.0]},
    ]
    rows_2007_999 = [
        {"permno": 999, "date": pd.Timestamp("2007-06-08"), "vec": [9.0, 9.0, 9.0]},  # not in universe
    ]
    rows_2008_101 = [
        {"permno": 101, "date": pd.Timestamp("2008-01-04"), "vec": [2.0, 0.0, 0.0]},  # outside window
    ]
    _write_embed_shard(tmp_path, 2007, 101, rows_2007_101)
    _write_embed_shard(tmp_path, 2007, 999, rows_2007_999)
    _write_embed_shard(tmp_path, 2008, 101, rows_2008_101)

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
    # Both 2007 rows for permno 101 are kept (no weekly resample).
    assert X.shape == (2, 3)
    assert X.dtype == np.float32
    assert set(meta["permno"]) == {101}
    assert meta["date"].min() == pd.Timestamp("2007-06-07")
    assert meta["date"].max() == pd.Timestamp("2007-06-08")


def test_assemble_training_matrix_respects_date_out(tmp_path: Path):
    """A bounded universe interval drops rows after date_out."""
    _write_embed_shard(tmp_path, 2013, 303, [
        {"permno": 303, "date": pd.Timestamp("2013-01-15"), "vec": [1.0, 2.0]},
    ])
    universe_ids = pd.DataFrame({
        "ticker": ["DROP"],
        "permno": pd.array([303], dtype="Int64"),
        "date_in": pd.to_datetime(["2010-01-01"]),
        "date_out": pd.to_datetime(["2012-12-31"]),
    })
    X, _ = assemble_training_matrix(tmp_path, universe_ids, "2010-01-01", "2014-12-31")
    assert X.shape == (0, 2)


def test_assemble_training_matrix_open_ended_window_keeps_recent(tmp_path: Path):
    """date_out=NaT means active — recent dates are kept."""
    _write_embed_shard(tmp_path, 2024, 101, [
        {"permno": 101, "date": pd.Timestamp("2024-06-15"), "vec": [1.0, 2.0]},
    ])
    universe_ids = pd.DataFrame({
        "ticker": ["AAPL"],
        "permno": pd.array([101], dtype="Int64"),
        "date_in": pd.to_datetime(["2009-02-01"]),
        "date_out": pd.to_datetime([None]),
    })
    X, _ = assemble_training_matrix(tmp_path, universe_ids, "2024-01-01", "2024-12-31")
    assert X.shape == (1, 2)


def test_assemble_training_matrix_empty_match_returns_zero_rows(tmp_path: Path):
    _write_embed_shard(tmp_path, 2007, 101, [
        {"permno": 101, "date": pd.Timestamp("2007-06-08"), "vec": [1.0, 0.0]},
    ])
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


# -------------------------------- sklearn contract (ranker consumer) ----------


def test_pca_joblib_round_trip_preserves_transform(tmp_path: Path):
    """The ranker notebook loads `pca.joblib` and calls .transform() — verify
    that contract: dump, load, transform on holdout, get the same output as the
    in-memory model. This is the actual integration point with downstream code."""
    rng = np.random.RandomState(0)
    X_train = rng.randn(500, 50).astype(np.float32)
    X_holdout = rng.randn(100, 50).astype(np.float32)
    pca = PCA(n_components=10, svd_solver="full").fit(X_train)
    expected = pca.transform(X_holdout)

    path = tmp_path / "pca.joblib"
    joblib.dump(pca, path)
    loaded = joblib.load(path)
    np.testing.assert_array_almost_equal(loaded.transform(X_holdout), expected, decimal=6)
    assert loaded.n_components_ == 10


def test_pca_full_svd_raises_when_n_samples_below_locked_dim():
    """If per-walk training rows fall below the locked `n_pca`, sklearn raises
    rather than silently fitting a degraded model. Capture that contract so a
    refactor that swaps solvers doesn't silently change behavior — the per-walk
    re-fit in notebook 04 depends on a loud failure here."""
    rng = np.random.RandomState(1)
    X = rng.randn(8, 50).astype(np.float32)
    with pytest.raises(ValueError, match="must be between"):
        PCA(n_components=20, svd_solver="full").fit(X)

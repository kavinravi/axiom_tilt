"""PCA helpers for FinBERT stock-day embeddings (ranker text features).

Why a module instead of inlining in the notebook: the per-shard streaming read
below is the only non-trivial bit, and it benefits from unit-test coverage.
Everything else the notebook needs is one or two lines of sklearn — see
`notebooks/04_pca_text_features.ipynb`.

Design (see spec §5.3 / §17.2):
  - Fit `PCA(svd_solver='full')` on the first walk's training matrix; pick
    `n_pca` as smallest n with cum_var[n-1] >= 0.99, plus 1 safety, capped at
    full rank. Lock that dim for all subsequent walks; re-fit components per
    walk and watch `explained_variance_ratio_.sum()` for drift.
  - No L2 normalization before PCA: the stock-day vectors out of notebook 03
    are already mean-pooled FinBERT CLS vectors, and L2-norm would discard the
    magnitude signal the ranker may use.
  - No weekly resample: PCA cares about variance structure, not rebalance
    cadence; the forward-filled daily duplicates add zero variance and sklearn's
    SVD handles them effectively for free. Resampling would also throw away
    real signal (mid-week filings whose Tue/Wed/Thu vectors differ from Fri).
  - Sanity gate: if locked `n_pca >= 200`, the cum-var elbow has likely
    blown out and the production assumption ("text reduces to a handful of
    dims") no longer holds — caller should hard-stop, not warn.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def assemble_training_matrix(
    embed_dir: Path,
    universe_ids: pd.DataFrame,
    start: str,
    end: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Read finbert_stockday_embed shards, gate to (window x universe), return X + meta.

    Streams per-shard: each parquet is filtered to the date window AND the
    universe intervals before concat, so peak memory stays roughly the size of
    one shard rather than the full panel. At ~1M+ stock-days x 768 float32, a
    pre-concat universe filter is the difference between a few hundred MB and
    several GB of RSS.

    `date_out=NaT` in `universe_ids` means "still active" — represented via a
    far-future sentinel for the interval merge.

    Returns:
      X:    float32 array, (n_samples, hidden_dim). Same row order as meta.
      meta: DataFrame with permno and date for each row of X.
    """
    embed_dir = Path(embed_dir)
    shards = sorted(embed_dir.glob("year=*/*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no parquet shards under {embed_dir}")

    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    intervals = universe_ids.dropna(subset=["permno"]).copy()
    intervals["permno"] = intervals["permno"].astype("int64")
    intervals["date_out"] = intervals["date_out"].fillna(pd.Timestamp("2099-12-31"))
    intervals = intervals[["permno", "date_in", "date_out"]]

    # Hidden dim is needed even for the empty path so the caller's shape contract holds.
    hidden = len(pd.read_parquet(shards[0], columns=["vec"])["vec"].iloc[0])

    frames: list[pd.DataFrame] = []
    for s in shards:
        df = pd.read_parquet(s, columns=["permno", "date", "vec"])
        df["date"] = pd.to_datetime(df["date"])
        df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)]
        if not len(df):
            continue
        df = df.merge(intervals, on="permno", how="inner")
        in_window = (df["date"] >= df["date_in"]) & (df["date"] <= df["date_out"])
        df = df[in_window].drop(columns=["date_in", "date_out"])
        # A panel row matches multiple intervals only if universe intervals overlap,
        # which build_universe forbids; this is a defensive backstop.
        df = df.drop_duplicates(subset=["permno", "date"])
        if len(df):
            frames.append(df)

    if not frames:
        return np.empty((0, hidden), dtype=np.float32), pd.DataFrame(columns=["permno", "date"])

    panel = pd.concat(frames, ignore_index=True)
    X = np.stack([np.asarray(v, dtype=np.float32) for v in panel["vec"].values])
    meta = panel[["permno", "date"]].reset_index(drop=True)
    return X, meta

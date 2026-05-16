# PCA Text Features (Scaffolding) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the dimensionality-reduction stage that turns FinBERT stock-day embeddings into ranker-ready low-dim features. Fit on the first walk's training window, lock `n_pca` via cumulative-variance target, support per-walk component re-fit.

**Architecture:** Six pure-function helpers in `src/utils/pca.py` (data assembly + sklearn-thin PCA wrappers), unit-tested with synthetic data so the scaffolding lands before Kavin's GPU output. One orchestration notebook (`notebooks/04_pca_text_features.ipynb`) runs the first-walk fit + diagnostics + walk-2 demo. Notebook has a `USE_SYNTHETIC` switch so it executes end-to-end on Dylan's laptop today and on real embeddings as soon as `data/processed/finbert_stockday_embed/` lands.

**Tech Stack:** scikit-learn (PCA full SVD), numpy, pandas, matplotlib, pyarrow, joblib (transitive via sklearn), nbformat. All already in `requirements.txt`.

**Spec reference:** [docs/superpowers/specs/2026-05-08-text-enhanced-rl-portfolio-design.md](../specs/2026-05-08-text-enhanced-rl-portfolio-design.md) §5.3 (PCA design), §17.2 (per-walk diagnostics).

**Operational decisions (from brainstorm):**

| ID | Decision | Rationale |
|---|---|---|
| A2 | Code lives in `src/utils/pca.py` | Small bend on "utils" rule; PCA is reused by future ranker code; precedent in `src/utils/seed.py` |
| B2 | Fit on weekly Friday snapshots, in-universe only | Matches rebalance cadence; ~156K samples, ample for 768-dim |
| C | `PCA(svd_solver='full')`, no L2-norm, no per-component z-score | Exact, fast at this size, gives full cum-var curve natively |
| D | Two functions: `fit_pca_initial` (returns full cum_var + locked dim) and `fit_pca_walk(X, n_pca)` | Locking semantics from spec §5.3 |
| E3 | First walk produces in-memory frame + a small parquet for inspection; defer multi-walk schema | Avoids premature schema design before ranker exists |
| F | Synthetic-data mode in notebook so it runs without real embeddings | Decouples scaffolding from Kavin's GPU job |
| G | Plan only (no separate spec) | Math is already in spec §5.3; only operational decisions are new |

---

## Scope

This plan produces working PCA scaffolding that can be exercised today on synthetic data and flipped to real data when notebook 03 output lands. No ranker, no walk-forward orchestration, no output-schema commitment. Future work is explicitly out of scope.

## File Structure

**Created:**
- `src/utils/pca.py` (~110 LOC, 6 functions)
- `tests/utils/test_pca.py` (~200 LOC, 14 tests)
- `scripts/build_pca_notebook.py` (one-shot notebook builder, ~150 LOC)
- `notebooks/04_pca_text_features.ipynb` (generated, committed for convenience)

**Modified:**
- None. (`artifacts/` is already gitignored, sklearn already in requirements, no foundation tweaks needed.)

**Out of scope:**
- Notebook 05 (LightGBM ranker) — separate plan once embeddings land
- Walk-forward orchestration loop — needs ranker first
- 3 aux text features (novelty / recency / count) — separate, computed alongside structured features
- Subspace stability tracking per §17.6 — v2

---

## Task 0: Sanity check the environment

No commit. Pure verification before starting.

- [ ] **Step 1: Verify sklearn is importable at a recent version**

Run: `python -c "import sklearn; from sklearn.decomposition import PCA; print(sklearn.__version__)"`

Expected: prints a version >= 1.5 (per requirements.txt).

- [ ] **Step 2: Verify pyarrow + nbformat available**

Run: `python -c "import pyarrow; import nbformat; print('ok', pyarrow.__version__, nbformat.__version__)"`

Expected: prints `ok <pyarrow_version> <nbformat_version>` with no error.

If either fails, run `pip install -r requirements.txt` and retry.

---

## Task 1: `pick_n_components` (TDD)

The pure dim-selection helper. Takes a cumulative-variance array, returns the locked `n_pca` (smallest `n` hitting target, plus 1 safety, capped at full rank).

**Files:**
- Create: `src/utils/pca.py`
- Create: `tests/utils/test_pca.py`

### Step 1: Write failing tests

Create `tests/utils/test_pca.py`:

```python
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
```

### Step 2: Run tests to verify they fail

Run: `pytest tests/utils/test_pca.py -v`

Expected: collection error or ImportError on `src.utils.pca` — the module doesn't exist yet.

### Step 3: Create `src/utils/pca.py` with `pick_n_components`

```python
"""PCA helpers for FinBERT stock-day embeddings (ranker text features).

Design: see docs/superpowers/specs/2026-05-08-text-enhanced-rl-portfolio-design.md §5.3.
Operational decisions: docs/superpowers/plans/2026-05-15-pca-text-features.md.

These functions are pure / sklearn-thin so they unit-test cleanly on synthetic
data. The first-walk orchestration + diagnostics live in
notebooks/04_pca_text_features.ipynb.
"""
from __future__ import annotations

import numpy as np


def pick_n_components(cum_var: np.ndarray, target: float) -> int:
    """Return locked `n_pca`: smallest n with `cum_var[n-1] >= target`, plus 1 safety.

    If target is unreachable (max cum_var < target), or hit only by the last
    component, cap at `len(cum_var)` — there's no n+1 component beyond full rank.

    Raises ValueError on empty input or target outside [0, 1].
    """
    if len(cum_var) == 0:
        raise ValueError("cum_var must be non-empty")
    if not (0.0 <= target <= 1.0):
        raise ValueError(f"target must be in [0, 1]; got {target}")
    if cum_var[-1] < target:
        return len(cum_var)
    n = int(np.searchsorted(cum_var, target, side="left")) + 1
    return min(n + 1, len(cum_var))
```

### Step 4: Run tests to verify pass

Run: `pytest tests/utils/test_pca.py -v`

Expected: 6 passed.

### Step 5: Commit

```bash
git add src/utils/pca.py tests/utils/test_pca.py
git commit -m "utils: add pick_n_components for PCA target-variance dim selection"
```

---

## Task 2: `weekly_snapshots` (TDD)

Take the latest observation per `(permno, ISO week)`. This is the resample that turns the daily stock-day embed panel into the weekly cadence we fit on.

**Files:**
- Modify: `src/utils/pca.py` (append function + import)
- Modify: `tests/utils/test_pca.py` (append tests + import)

### Step 1: Append failing tests

Append to `tests/utils/test_pca.py` (the `from src.utils.pca import` line at top of file should be extended):

First, update the top-of-file import block:

```python
from src.utils.pca import (
    pick_n_components,
    weekly_snapshots,
)
```

Then append at end of file:

```python
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
```

### Step 2: Run, verify the 3 new tests fail (ImportError on `weekly_snapshots`)

Run: `pytest tests/utils/test_pca.py -v`

Expected: ImportError before any test runs (because of the augmented top-of-file import).

### Step 3: Append `weekly_snapshots` to `src/utils/pca.py`

Add `import pandas as pd` to the imports block:

```python
import numpy as np
import pandas as pd
```

Append the function at end of file:

```python
def weekly_snapshots(
    panel: pd.DataFrame,
    permno_col: str = "permno",
    date_col: str = "date",
) -> pd.DataFrame:
    """Return the latest observation per (permno, ISO week ending Friday).

    Used to resample the daily stock-day embed panel to weekly before PCA fit,
    matching the §3.1 rebalance cadence and avoiding the forward-fill bulk.
    """
    df = panel.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["_week"] = df[date_col].dt.to_period("W-FRI")
    df = df.sort_values([permno_col, "_week", date_col])
    out = df.groupby([permno_col, "_week"], as_index=False).tail(1)
    return out.drop(columns="_week").reset_index(drop=True)
```

### Step 4: Run, verify pass

Run: `pytest tests/utils/test_pca.py -v`

Expected: 9 passed.

### Step 5: Commit

```bash
git add src/utils/pca.py tests/utils/test_pca.py
git commit -m "utils: add weekly_snapshots for PCA training-matrix resampling"
```

---

## Task 3: `filter_in_universe` (TDD)

Keep only `(permno, date)` rows that fall inside a `(date_in, date_out)` interval from `universe_ids`. `date_out` NaT means "still active". A permno can have multiple intervals (joined → left → rejoined the index).

**Files:**
- Modify: `src/utils/pca.py` (append function)
- Modify: `tests/utils/test_pca.py` (append tests + import)

### Step 1: Append failing tests

Update top-of-file import:

```python
from src.utils.pca import (
    filter_in_universe,
    pick_n_components,
    weekly_snapshots,
)
```

Append at end of file:

```python
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
```

### Step 2: Verify the 5 new tests fail with ImportError

Run: `pytest tests/utils/test_pca.py -v`

Expected: ImportError.

### Step 3: Append `filter_in_universe` to `src/utils/pca.py`

```python
def filter_in_universe(panel: pd.DataFrame, universe_ids: pd.DataFrame) -> pd.DataFrame:
    """Keep panel rows where (permno, date) falls inside any universe interval.

    NaT in date_out is treated as "still active" via a far-future sentinel.
    A permno-date that matches multiple intervals (re-joins) is kept once.
    """
    panel_cols = list(panel.columns)
    intervals = universe_ids.dropna(subset=["permno"]).copy()
    intervals["permno"] = intervals["permno"].astype("int64")
    intervals["date_out"] = intervals["date_out"].fillna(pd.Timestamp("2099-12-31"))
    merged = panel.merge(
        intervals[["permno", "date_in", "date_out"]],
        on="permno",
        how="left",
    )
    in_window = (merged["date"] >= merged["date_in"]) & (merged["date"] <= merged["date_out"])
    kept = merged[in_window][panel_cols].drop_duplicates().reset_index(drop=True)
    return kept
```

### Step 4: Verify pass

Run: `pytest tests/utils/test_pca.py -v`

Expected: 14 passed.

### Step 5: Commit

```bash
git add src/utils/pca.py tests/utils/test_pca.py
git commit -m "utils: add filter_in_universe for PCA training-matrix universe gating"
```

---

## Task 4: `assemble_training_matrix` (TDD with parquet round-trip)

Read `data/processed/finbert_stockday_embed/year=*/*.parquet`, apply window + universe + weekly resample, return `(X, meta)`. End-to-end deterministic test uses `tmp_path` and a small fake shard.

**Files:**
- Modify: `src/utils/pca.py` (append function + path import)
- Modify: `tests/utils/test_pca.py` (append tests + import)

### Step 1: Append failing tests

Update top-of-file import:

```python
from src.utils.pca import (
    assemble_training_matrix,
    filter_in_universe,
    pick_n_components,
    weekly_snapshots,
)
```

Append at end of file:

```python
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
```

### Step 2: Verify ImportError

Run: `pytest tests/utils/test_pca.py -v`

Expected: ImportError.

### Step 3: Append to `src/utils/pca.py`

Add `from pathlib import Path` to the imports (top of file):

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
```

Append at end of file:

```python
def assemble_training_matrix(
    embed_dir: Path,
    universe_ids: pd.DataFrame,
    start: str,
    end: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Read finbert_stockday_embed shards, gate to (window x universe), weekly resample.

    Returns:
      X:    float32 array, (n_samples, hidden_dim). Same row order as meta.
      meta: DataFrame with permno and date for each row of X.
    """
    embed_dir = Path(embed_dir)
    shards = sorted(embed_dir.glob("year=*/*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no parquet shards under {embed_dir}")

    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    frames: list[pd.DataFrame] = []
    for s in shards:
        df = pd.read_parquet(s, columns=["permno", "date", "vec"])
        df["date"] = pd.to_datetime(df["date"])
        df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)]
        if len(df):
            frames.append(df)

    # Hidden dim is needed even for the empty path so the caller's shape contract holds.
    hidden = len(pd.read_parquet(shards[0], columns=["vec"])["vec"].iloc[0])

    if not frames:
        return np.empty((0, hidden), dtype=np.float32), pd.DataFrame(columns=["permno", "date"])

    panel = pd.concat(frames, ignore_index=True)
    panel = filter_in_universe(panel, universe_ids)
    panel = weekly_snapshots(panel)

    if len(panel) == 0:
        return np.empty((0, hidden), dtype=np.float32), pd.DataFrame(columns=["permno", "date"])

    X = np.stack([np.asarray(v, dtype=np.float32) for v in panel["vec"].values])
    meta = panel[["permno", "date"]].reset_index(drop=True)
    return X, meta
```

### Step 4: Verify pass

Run: `pytest tests/utils/test_pca.py -v`

Expected: 17 passed.

### Step 5: Commit

```bash
git add src/utils/pca.py tests/utils/test_pca.py
git commit -m "utils: add assemble_training_matrix for PCA fit input"
```

---

## Task 5: `fit_pca_initial` (TDD synthetic)

First-walk fit: full SVD, pick locked dim from cum-var curve, return `(n_pca, full_cum_var, fitted_pca_at_n_pca)`.

**Files:**
- Modify: `src/utils/pca.py` (append + sklearn import)
- Modify: `tests/utils/test_pca.py` (append tests + import)

### Step 1: Append failing tests

Update top-of-file import:

```python
from src.utils.pca import (
    assemble_training_matrix,
    filter_in_universe,
    fit_pca_initial,
    pick_n_components,
    weekly_snapshots,
)
```

Append at end:

```python
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
```

### Step 2: Verify ImportError

Run: `pytest tests/utils/test_pca.py -v`

Expected: ImportError.

### Step 3: Append to `src/utils/pca.py`

Add sklearn import to the imports block:

```python
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
```

Append at end:

```python
def fit_pca_initial(X: np.ndarray, target: float = 0.99) -> tuple[int, np.ndarray, PCA]:
    """First-walk PCA fit. Pick n_pca via cum-var target, lock for all walks.

    Two SVD passes: one full-rank (cheap at our scale, gives the §17.2 cum-var
    curve) and one truncated for the production transformer.

    Returns:
      n_pca:   locked dim
      cum_var: full cumulative explained-variance curve, all components
      pca:     fitted PCA with n_components=n_pca, ready to .transform()
    """
    full = PCA(svd_solver="full").fit(X)
    cum_var = np.cumsum(full.explained_variance_ratio_)
    n_pca = pick_n_components(cum_var, target=target)
    pca = PCA(n_components=n_pca, svd_solver="full").fit(X)
    return n_pca, cum_var, pca
```

### Step 4: Verify pass

Run: `pytest tests/utils/test_pca.py -v`

Expected: 19 passed.

### Step 5: Commit

```bash
git add src/utils/pca.py tests/utils/test_pca.py
git commit -m "utils: add fit_pca_initial for first-walk PCA dim selection + lock"
```

---

## Task 6: `fit_pca_walk` (TDD)

Per-walk re-fit at the locked dim. Reports the variance captured by the locked-dim subspace — this is the §17.2 drift sanity check.

**Files:**
- Modify: `src/utils/pca.py` (append)
- Modify: `tests/utils/test_pca.py` (append + import)

### Step 1: Append failing test

Update top-of-file import:

```python
from src.utils.pca import (
    assemble_training_matrix,
    filter_in_universe,
    fit_pca_initial,
    fit_pca_walk,
    pick_n_components,
    weekly_snapshots,
)
```

Append:

```python
# -------------------------------- fit_pca_walk ---------------------------------


def test_fit_pca_walk_locks_dim_and_reports_variance():
    rng = np.random.RandomState(7)
    X = rng.randn(500, 30).astype(np.float32)
    pca, variance_captured = fit_pca_walk(X, n_pca=10)

    assert pca.n_components_ == 10
    assert 0.0 <= variance_captured <= 1.0
    np.testing.assert_almost_equal(
        variance_captured,
        float(pca.explained_variance_ratio_.sum()),
        decimal=6,
    )
```

### Step 2: Verify ImportError

Run: `pytest tests/utils/test_pca.py -v`

Expected: ImportError.

### Step 3: Append to `src/utils/pca.py`

```python
def fit_pca_walk(X: np.ndarray, n_pca: int) -> tuple[PCA, float]:
    """Per-walk re-fit at the locked dim. Returns (fitted_pca, variance_captured).

    `variance_captured` is `explained_variance_ratio_.sum()` — the per-walk
    drift sanity check from spec §17.2. If this falls noticeably below the
    initial target across walks, the locked dim is becoming too tight.
    """
    pca = PCA(n_components=n_pca, svd_solver="full").fit(X)
    return pca, float(pca.explained_variance_ratio_.sum())
```

### Step 4: Verify pass

Run: `pytest tests/utils/test_pca.py -v`

Expected: 20 passed.

### Step 5: Commit

```bash
git add src/utils/pca.py tests/utils/test_pca.py
git commit -m "utils: add fit_pca_walk for per-walk PCA re-estimation at locked dim"
```

---

## Task 7: Notebook `04_pca_text_features.ipynb` (built via nbformat)

The notebook is built by a Python script using `nbformat`, mirroring the pattern from `scripts/build_finbert_notebook.py`. Cell sources live in source control as readable Python strings; the `.ipynb` is the generated artifact (committed for convenience).

**Files:**
- Create: `scripts/build_pca_notebook.py`
- Create: `notebooks/04_pca_text_features.ipynb` (generated)

### Task 7.1: Create the build script

- [ ] **Step 1: Create `scripts/build_pca_notebook.py`**

```python
"""Build notebooks/04_pca_text_features.ipynb from cell definitions.

Run via:
    python scripts/build_pca_notebook.py

Cell content is the source of truth here; the .ipynb is the regenerated artifact.
"""
from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


# =============================================================================
# Cell sources
# =============================================================================

INTRO_MD = """# 04 — Text-feature PCA

Reduce 768-dim FinBERT stock-day embeddings (notebook 03 output) to a ranker-ready
low-dim feature set. Fit on the first walk's training window (2002-2007), pick
`n_pca` at 99% cumulative variance + 1 safety buffer, lock the dim for all
subsequent walks. Re-fit components at each walk boundary.

**Spec:** `docs/superpowers/specs/2026-05-08-text-enhanced-rl-portfolio-design.md` §5.3 / §17.2.
**Plan:** `docs/superpowers/plans/2026-05-15-pca-text-features.md`.

**Mode switch:** `USE_SYNTHETIC=True` runs on planted-signal Gaussian data so the
notebook is executable end-to-end before notebook 03's GPU output lands. Flip to
`False` once `data/processed/finbert_stockday_embed/` is populated.
"""

A_SETUP = """from __future__ import annotations
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.io import processed_dir, repo_root
from src.utils.pca import (
    assemble_training_matrix,
    fit_pca_initial,
    fit_pca_walk,
    pick_n_components,
)

USE_SYNTHETIC = True  # flip to False after notebook 03 output lands

EMBED_DIR = processed_dir() / 'finbert_stockday_embed'
UNIVERSE_PATH = processed_dir() / 'universe_ids.parquet'
ARTIFACTS_DIR = repo_root() / 'artifacts' / 'pca-text'
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Spec §7.2 walk-forward windows
WALK_1_START, WALK_1_END = '2002-01-01', '2007-12-31'
WALK_2_START, WALK_2_END = '2003-01-01', '2008-12-31'

TARGETS = [0.95, 0.98, 0.99]
PROD_TARGET = 0.99
SANITY_MAX_N_PCA = 200  # spec §5.3 sanity check

print(f'USE_SYNTHETIC={USE_SYNTHETIC}')
print(f'embed_dir={EMBED_DIR}')
print(f'walk 1 window: {WALK_1_START} -> {WALK_1_END}')
print(f'production target: {PROD_TARGET}; sensitivity: {TARGETS}')
"""

B_MD = """## B. Load first-walk training matrix

Synthetic mode plants an 8-dim signal in 768-dim with ~2% Gaussian noise. The
cum-var curve should have a real elbow near 8 components; locked `n_pca` should
land at 9 (8 + 1 safety) at the 0.99 target. Real mode reads notebook 03 output
and applies the universe gate + weekly resample via `assemble_training_matrix`.
"""

B_LOAD = """if USE_SYNTHETIC:
    rng = np.random.RandomState(42)
    n_samples_w1 = 6 * 52 * 500  # 6 yr x 52 wk x ~500 names
    n_signal = 8
    hidden = 768
    truth = rng.randn(n_samples_w1, n_signal).astype(np.float32)
    proj = rng.randn(n_signal, hidden).astype(np.float32)
    X_w1 = truth @ proj + rng.randn(n_samples_w1, hidden).astype(np.float32) * 0.02
    meta_w1 = pd.DataFrame({
        'permno': rng.choice(np.arange(10001, 10501), size=n_samples_w1),
        # Synthetic dates are decorative — PCA fit only uses X
        'date': pd.date_range('2002-01-04', periods=n_samples_w1, freq='B')[:n_samples_w1],
    })
    print(f'synthetic walk 1: X={X_w1.shape}, samples={len(meta_w1):,}, planted_dim={n_signal}')
else:
    universe_ids = pd.read_parquet(UNIVERSE_PATH)
    X_w1, meta_w1 = assemble_training_matrix(
        embed_dir=EMBED_DIR,
        universe_ids=universe_ids,
        start=WALK_1_START,
        end=WALK_1_END,
    )
    print(f'real walk 1: X={X_w1.shape}, samples={len(meta_w1):,}')
    if len(meta_w1):
        print(f'  date range: {meta_w1.date.min().date()} -> {meta_w1.date.max().date()}')
        print(f'  unique permnos: {meta_w1.permno.nunique()}')
    assert len(meta_w1) >= 10_000, (
        f'walk 1 has {len(meta_w1)} samples; expected >= 10K. '
        'Did notebook 03 finish? Did universe_ids include enough permnos?'
    )
"""

C_MD = """## C. Fit PCA on first walk; pick `n_pca`; lock

Full-rank SVD gives the §17.2 cum-var curve. `pick_n_components` returns the
smallest `n` with `cum_var[n-1] >= target`, plus 1 safety, capped at full rank.
"""

C_FIT = """n_pca, cum_var, pca_w1 = fit_pca_initial(X_w1, target=PROD_TARGET)
captured_at_lock = float(cum_var[n_pca - 1]) if n_pca <= len(cum_var) else 1.0
print(f'locked n_pca = {n_pca}  (target={PROD_TARGET}, includes +1 safety)')
print(f'variance captured at n_pca: {captured_at_lock:.4f}')

sensitivity = {f'{t:.2f}': int(pick_n_components(cum_var, target=t)) for t in TARGETS}
print('\\nsensitivity (n_pca at each target):')
for k, v in sensitivity.items():
    print(f'  target={k}: n_pca={v}')

# Spec §5.3 sanity check
if n_pca >= SANITY_MAX_N_PCA:
    print(f'\\nWARNING: n_pca={n_pca} >= {SANITY_MAX_N_PCA}. Inspect the scree.')
    print('Options: lower target (95 / 98), L2-normalize embeddings before PCA, or')
    print('skip PCA entirely and use a different reducer (e.g., a small linear projection).')
"""

D_MD = """## D. Diagnostics — cumulative variance, scree

Cum-var on log-x to read the elbow clearly. Scree on log-y so the long noise
tail stays visible (otherwise the first eigenvalue swamps everything).
"""

D_PLOTS = """fig, axes = plt.subplots(1, 2, figsize=(12, 4))

xs = np.arange(1, len(cum_var) + 1)
ax = axes[0]
ax.plot(xs, cum_var, lw=1.2)
for t, color in zip(TARGETS, ['tab:red', 'tab:orange', 'tab:green']):
    ax.axhline(t, color=color, ls='--', lw=0.7, label=f'target={t}')
ax.axvline(n_pca, color='black', ls=':', lw=0.7, label=f'locked n_pca={n_pca}')
ax.set_xlabel('component')
ax.set_ylabel('cumulative explained variance')
ax.set_title('PCA cumulative variance — first walk')
ax.set_xscale('log')
ax.legend(loc='lower right', fontsize=8)
ax.grid(alpha=0.3)

full_evr = np.diff(np.concatenate([[0.0], cum_var]))
ax = axes[1]
ax.plot(xs, full_evr, lw=0.8)
ax.set_yscale('log')
ax.set_xlabel('component')
ax.set_ylabel('explained variance ratio (log)')
ax.set_title('Scree — first walk')
ax.grid(alpha=0.3, which='both')

plt.tight_layout()
plt.show()
"""

E_MD = """## E. Walk-2 re-fit demo at locked dim

`fit_pca_walk(X, n_pca)` runs PCA with the locked dim on the next walk's
training matrix. `variance_captured` per walk is the §17.2 drift sanity check —
if the captured fraction falls noticeably below `PROD_TARGET` across walks, the
locked dim has become too tight (text "topic geometry" is shifting).
"""

E_REFIT = """if USE_SYNTHETIC:
    n_samples_w2 = 6 * 52 * 500
    truth_w2 = rng.randn(n_samples_w2, n_signal).astype(np.float32)
    X_w2 = truth_w2 @ proj + rng.randn(n_samples_w2, hidden).astype(np.float32) * 0.02
else:
    X_w2, _ = assemble_training_matrix(EMBED_DIR, universe_ids, WALK_2_START, WALK_2_END)

pca_w2, var_captured_w2 = fit_pca_walk(X_w2, n_pca=n_pca)
print(f'walk 2 fit: n_components={pca_w2.n_components_}, variance_captured={var_captured_w2:.4f}')
print(f'walk 1 captured:                                {captured_at_lock:.4f}')
print(f'drift (walk 2 minus walk 1):                    {var_captured_w2 - captured_at_lock:+.4f}')
"""

F_MD = """## F. Persist artifacts

`artifacts/pca-text/walk-001/` (gitignored): fitted PCA, full cum-var curve,
summary JSON. The ranker training notebook (TBD) will load these for the
first walk's PCA transformer.
"""

F_PERSIST = """WALK_1_DIR = ARTIFACTS_DIR / 'walk-001'
WALK_1_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(pca_w1, WALK_1_DIR / 'pca.joblib')
np.save(WALK_1_DIR / 'cum_var.npy', cum_var)

summary = {
    'walk_id': 1,
    'window_start': WALK_1_START,
    'window_end': WALK_1_END,
    'target_variance': PROD_TARGET,
    'locked_n_pca': int(n_pca),
    'variance_captured_at_n_pca': captured_at_lock,
    'sensitivity_n_pca': sensitivity,
    'n_train_samples': int(X_w1.shape[0]),
    'hidden_dim': int(X_w1.shape[1]),
    'use_synthetic': USE_SYNTHETIC,
}
(WALK_1_DIR / 'summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
print(f'\\nartifacts -> {WALK_1_DIR.relative_to(repo_root())}')
"""


# =============================================================================
# Build
# =============================================================================

def build_notebook() -> nbformat.NotebookNode:
    nb = new_notebook()
    nb.cells = [
        new_markdown_cell(INTRO_MD),
        new_markdown_cell('## A. Setup'),
        new_code_cell(A_SETUP),
        new_markdown_cell(B_MD),
        new_code_cell(B_LOAD),
        new_markdown_cell(C_MD),
        new_code_cell(C_FIT),
        new_markdown_cell(D_MD),
        new_code_cell(D_PLOTS),
        new_markdown_cell(E_MD),
        new_code_cell(E_REFIT),
        new_markdown_cell(F_MD),
        new_code_cell(F_PERSIST),
    ]
    nb.metadata = {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.11'},
    }
    return nb


def main() -> None:
    out_path = Path(__file__).resolve().parents[1] / 'notebooks' / '04_pca_text_features.ipynb'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nb = build_notebook()
    with out_path.open('w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(f'Wrote {out_path}')

    # Round-trip read to catch JSON / nbformat issues
    with out_path.open(encoding='utf-8') as f:
        loaded = nbformat.read(f, as_version=4)
    print(f'Round-trip OK. Cells: {len(loaded.cells)}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run the build script**

Run: `python scripts/build_pca_notebook.py`

Expected output:
```
Wrote /Users/dylanmassaro/axiom_tilt/.claude/worktrees/cool-mclaren-4b7c3a/notebooks/04_pca_text_features.ipynb
Round-trip OK. Cells: 13
```

If the path differs that's fine — the worktree path will appear. The "Cells: 13" line confirms structure.

- [ ] **Step 3: Validate the notebook is loadable by Jupyter**

Run: `jupyter nbconvert --to notebook --stdout notebooks/04_pca_text_features.ipynb > /dev/null && echo "VALID"`

Expected: prints `VALID`. If it errors with JSON / nbformat issues, inspect with:
```bash
python -c "import nbformat; nb = nbformat.read('notebooks/04_pca_text_features.ipynb', as_version=4); [print(i, c['cell_type'], len(c['source'])) for i, c in enumerate(nb.cells)]"
```

### Task 7.2: Execute the notebook in synthetic mode (smoke test)

The notebook's `USE_SYNTHETIC=True` default means it should run end-to-end on Dylan's laptop without needing notebook 03 output. This is the scaffold-validation gate.

- [ ] **Step 1: Run end-to-end**

Run: `jupyter nbconvert --to notebook --execute --inplace notebooks/04_pca_text_features.ipynb --ExecutePreprocessor.timeout=300`

Expected: completes within a few minutes. CPU-only.

- [ ] **Step 2: Verify the synthetic fit landed sensibly**

Run: `python -c "import json; print(json.dumps(json.loads(open('artifacts/pca-text/walk-001/summary.json').read()), indent=2))"`

Expected output (numbers approximate):
```json
{
  "walk_id": 1,
  "window_start": "2002-01-01",
  "window_end": "2007-12-31",
  "target_variance": 0.99,
  "locked_n_pca": 9,
  "variance_captured_at_n_pca": 0.99...,
  "sensitivity_n_pca": {"0.95": 9, "0.98": 9, "0.99": 9},
  "n_train_samples": 156000,
  "hidden_dim": 768,
  "use_synthetic": true
}
```

Gate: `locked_n_pca` should be in `[8, 12]` for the synthetic case (8-dim signal + safety). If it's drastically larger, the build is broken — investigate before committing.

### Task 7.3: Commit

- [ ] **Step 1: Commit the build script and generated + executed notebook**

```bash
git add scripts/build_pca_notebook.py notebooks/04_pca_text_features.ipynb
git commit -m "notebook: add 04_pca_text_features for first-walk PCA fit"
```

---

## Task 8: Final validation

- [ ] **Step 1: Run the full test suite**

Run: `pytest tests/ -v`

Expected: all tests pass. Total should be ≥ 70 (50 pre-existing + 20 added).

If anything outside `tests/utils/test_pca.py` regressed, the PCA helpers were probably colliding with an import elsewhere — `src/utils/pca.py` is new so this is unlikely.

- [ ] **Step 2: Confirm worktree state is clean**

Run: `git status`

Expected: `nothing to commit, working tree clean`.

- [ ] **Step 3: Verify the notebook can be re-run idempotently**

Run: `jupyter nbconvert --to notebook --execute --inplace notebooks/04_pca_text_features.ipynb --ExecutePreprocessor.timeout=300`

Expected: completes without error and produces the same `summary.json` (synthetic mode is seeded). The `artifacts/pca-text/walk-001/` files are overwritten cleanly.

No commit — this is verification only.

---

## Self-review

**1. Spec coverage** ([spec §5.3](../specs/2026-05-08-text-enhanced-rl-portfolio-design.md)):

| Spec requirement | Implementation |
|---|---|
| Default target 99% (sensitivity 95/98/99) | Task 7, cell C — `PROD_TARGET=0.99`, sensitivity printed at all three |
| Fit on first walk's training window only | `assemble_training_matrix(start=WALK_1_START, end=WALK_1_END)` (Task 4 + notebook B) |
| Smallest `n` with cum_var ≥ target | `pick_n_components` (Task 1) |
| Production dim = `n + 1` (safety) | Inside `pick_n_components` (Task 1) |
| Lock `n_pca` across walks | `fit_pca_walk(X, n_pca)` takes locked dim as input (Task 6) |
| Re-estimate components per walk | `fit_pca_walk` body — fresh `PCA.fit` (Task 6) |
| Diagnostic: cum-var curve all 768 components | Returned by `fit_pca_initial` (Task 5), plotted in cell D (Task 7) |
| Diagnostic: variance captured by locked dim per walk | `variance_captured` from `fit_pca_walk` (Task 6), reported in cell E (Task 7) |
| Sanity: reconsider if n_pca ≥ ~200 | `SANITY_MAX_N_PCA` warning in cell C (Task 7) |
| Top-N PCA loadings per walk | **NOT included** — listed in spec §17.2 as v2 ("subspace stability", v2). Defer. |
| Subspace cosine stability across walks | **NOT included** — v2 per spec §17 MVP/v2 split. Hook is the saved `pca.joblib` + cum_var per walk; future tooling can compute cosine offline. |

**2. Placeholder scan:** No "TBD", "TODO", "implement later" anywhere. Every code block is complete. Every command has an exact expected output.

**3. Type / name consistency:**
- `pick_n_components`, `weekly_snapshots`, `filter_in_universe`, `assemble_training_matrix`, `fit_pca_initial`, `fit_pca_walk` — referenced identically across tasks, tests, and notebook imports.
- `n_pca` (int), `cum_var` (np.ndarray), `pca` (sklearn PCA) — same naming everywhere.
- `meta` DataFrame has columns `(permno, date)` consistently in Task 4 implementation and Task 7 usage.
- `X` is float32 (n_samples, hidden) consistently. Hidden=768 in real mode, 768 in synthetic mode (notebook B).
- `USE_SYNTHETIC` toggle: Task 7 notebook only.

**4. Scope check:**
- Single focused unit (PCA scaffolding for first walk).
- Future ranker code (notebook 05) will import from `src/utils/pca.py` and consume `artifacts/pca-text/walk-001/`.
- Future walk-forward loop will call `fit_pca_walk` per window — API is in place.
- Subspace stability (v2 diagnostic per spec §17.6) is explicitly skipped here; the saved `pca.joblib` per walk is the hook for future tooling.

**5. Ambiguity check:**
- "Weekly snapshot" pinned to ISO week ending Friday (`W-FRI`) — matches Friday-close rebalance in spec §11.
- "In-universe" pinned to `(date_in, date_out)` from `universe_ids.parquet` — NaT in `date_out` = active.
- PCA fit uses `svd_solver='full'` for exactness, not `'auto'` — explicit and stable across sklearn versions.

No issues found.

---

## Execution Handoff

Plan complete and saved to [docs/superpowers/plans/2026-05-15-pca-text-features.md](2026-05-15-pca-text-features.md). Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?

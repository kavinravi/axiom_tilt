# Feature Assembly (Notebook 05) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce `data/processed/training_panel/year=YYYY/part-0.parquet` — a daily-cadence joined panel of structured + aux text + label features, ready for the ranker (notebook 06, Friday-filtered) and the RL agent (notebook 07+, daily).

**Architecture:** Pure helper functions in `src/utils/features.py` (testable with synthetic dataframes, TDD); notebook 05 glues them together with the existing `panel/`, `macro.parquet`, `finbert_stockday_embed/`, and `edgar_index.parquet` inputs. Per-year shard write, atomic (tmp → rename), resumable. PCA text features deliberately not in the panel — derived at ranker time, one matrix multiply per walk.

**Tech Stack:** pandas, numpy, pyarrow, jupyter, pytest. Reuses `src/utils/io.py` path helpers and the per-shard streaming pattern from `src/utils/pca.py`.

**Spec:** `docs/superpowers/specs/2026-05-16-feature-assembly-design.md`

---

## File Structure

| File | Responsibility |
|---|---|
| `src/utils/features.py` (new) | 5 pure helpers: `pivot_macro_wide`, `compute_forward_returns`, `compute_text_novelty`, `compute_days_since_filing`, `compute_doc_count_window` |
| `tests/utils/test_features.py` (new) | Unit tests for the 5 helpers; small synthetic dataframes, no I/O beyond `tmp_path` |
| `notebooks/05_assemble_training_panel.ipynb` (new) | 6 cells (A–F) wiring helpers + I/O |
| `reports/metrics/feature_assembly_summary.json` (notebook output) | Per-year row count, NaN-rate audit, schema fingerprint |
| `data/processed/training_panel/year=YYYY/part-0.parquet` (notebook output) | The training panel itself, gitignored |

`training_panel/` will need a `.gitignore` entry (it's the same "large derivative, R2-synced" category as `finbert_doc_embed/`). Added in Task 9 alongside the final commit.

---

## Task 1: `pivot_macro_wide` helper

**Files:**
- Create: `src/utils/features.py`
- Test: `tests/utils/test_features.py`

- [ ] **Step 1: Write the failing test**

Create `tests/utils/test_features.py`:

```python
"""Tests for src.utils.features — pure helpers behind notebook 05."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.utils.features import pivot_macro_wide


def test_pivot_macro_wide_creates_one_column_per_series():
    long = pd.DataFrame({
        'date': pd.to_datetime(['2020-01-01', '2020-01-01', '2020-01-02', '2020-01-02']),
        'series': ['VIXCLS', 'DGS10', 'VIXCLS', 'DGS10'],
        'value': [25.0, 1.8, 27.0, 1.9],
    })
    out = pivot_macro_wide(long)
    assert list(out.columns) == ['date', 'macro_dgs10', 'macro_vixcls']  # alpha-sorted, prefixed
    assert len(out) == 2
    assert out.iloc[0]['macro_vixcls'] == 25.0
    assert out.iloc[1]['macro_dgs10'] == 1.9


def test_pivot_macro_wide_forward_fills_missing_dates_when_requested():
    long = pd.DataFrame({
        'date': pd.to_datetime(['2020-01-01', '2020-01-03']),  # gap on 2020-01-02
        'series': ['VIXCLS', 'VIXCLS'],
        'value': [25.0, 27.0],
    })
    out = pivot_macro_wide(long, ffill_dates=pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03']))
    assert len(out) == 3
    assert out.iloc[1]['macro_vixcls'] == 25.0  # ffilled from 2020-01-01
```

- [ ] **Step 2: Run test, verify it fails**

```bash
python -m pytest tests/utils/test_features.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.utils.features'`.

- [ ] **Step 3: Implement `pivot_macro_wide`**

Create `src/utils/features.py`:

```python
"""Feature helpers for the daily training panel (notebook 05).

Pure functions over pandas DataFrames so the notebook stays a thin glue layer.
See docs/superpowers/specs/2026-05-16-feature-assembly-design.md for design.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def pivot_macro_wide(
    macro_long: pd.DataFrame,
    ffill_dates: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Pivot FRED long-format macro (`date, series, value`) to wide.

    Series become columns prefixed `macro_<series_lower>`. If `ffill_dates` is
    provided, reindex to that date axis and forward-fill (FRED publishes on
    business days; panel rows include all calendar days a permno is active).
    """
    wide = macro_long.pivot(index='date', columns='series', values='value')
    wide.columns = [f'macro_{c.lower()}' for c in wide.columns]
    wide = wide.sort_index().sort_index(axis=1)
    if ffill_dates is not None:
        wide = wide.reindex(ffill_dates).ffill()
    return wide.reset_index().rename(columns={'index': 'date'})
```

- [ ] **Step 4: Run test, verify it passes**

```bash
python -m pytest tests/utils/test_features.py -v
```
Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/utils/features.py tests/utils/test_features.py
git commit -m "add pivot_macro_wide helper + tests (notebook 05 prep)"
```

---

## Task 2: `compute_forward_returns` helper

**Files:**
- Modify: `src/utils/features.py`
- Modify: `tests/utils/test_features.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/utils/test_features.py`:

```python
from src.utils.features import compute_forward_returns


def test_compute_forward_returns_one_permno_simple_case():
    df = pd.DataFrame({
        'permno': [101] * 6,
        'date': pd.date_range('2020-01-02', periods=6, freq='B'),
        'ret': [0.01, -0.02, 0.03, 0.00, 0.05, -0.01],
    })
    out = compute_forward_returns(df, horizons=(1, 5))
    # fwd_ret_1d: next-day return
    assert out['fwd_ret_1d'].iloc[0] == pytest.approx(-0.02)
    assert pd.isna(out['fwd_ret_1d'].iloc[-1])  # last row has no next day
    # fwd_ret_5d: compounded next 5 returns
    expected_5d = (1.0 + np.array([-0.02, 0.03, 0.00, 0.05, -0.01])).prod() - 1.0
    assert out['fwd_ret_5d'].iloc[0] == pytest.approx(expected_5d)
    # Last 5 rows have no full 5-day forward window
    assert out['fwd_ret_5d'].iloc[1:].isna().sum() == 5


def test_compute_forward_returns_does_not_cross_permnos():
    df = pd.DataFrame({
        'permno': [101, 101, 202, 202],
        'date': pd.to_datetime(['2020-01-02', '2020-01-03', '2020-01-02', '2020-01-03']),
        'ret': [0.01, 0.02, 0.10, 0.20],
    })
    out = compute_forward_returns(df, horizons=(1,))
    # 101's last row's fwd_ret_1d should NOT pull from 202
    assert pd.isna(out.loc[out['permno'] == 101, 'fwd_ret_1d'].iloc[-1])
    assert pd.isna(out.loc[out['permno'] == 202, 'fwd_ret_1d'].iloc[-1])
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
python -m pytest tests/utils/test_features.py -v
```
Expected: 2 new tests FAIL with `ImportError` or `AttributeError`.

- [ ] **Step 3: Implement `compute_forward_returns`**

Append to `src/utils/features.py`:

```python
def compute_forward_returns(
    panel: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 5),
    ret_col: str = 'ret',
    permno_col: str = 'permno',
    date_col: str = 'date',
) -> pd.DataFrame:
    """Compute `fwd_ret_{h}d` for each horizon h (trading days).

    For each permno, sorts by date and computes compounded forward returns over
    the next `h` rows via log-return rolling sum (vectorized, no apply).
    Rows in the last `h` of a permno's history get NaN. Delisted permnos
    naturally produce NaN at the tail.

    Returns the input panel with new `fwd_ret_{h}d` columns appended.
    """
    out = panel.sort_values([permno_col, date_col]).copy()
    log_ret = np.log1p(out[ret_col].astype(float))
    for h in horizons:
        # Forward sum of log-returns over horizon h, then expm1.
        # shift(-h) aligns "sum of returns from t+1..t+h" with row t.
        rolling_sum = (
            log_ret.groupby(out[permno_col])
            .rolling(window=h, min_periods=h)
            .sum()
            .reset_index(level=0, drop=True)
        )
        # Rolling-sum at index t covers t-h+1..t; we want t+1..t+h, so shift -h.
        out[f'fwd_ret_{h}d'] = np.expm1(rolling_sum.groupby(out[permno_col]).shift(-h))
    return out.reset_index(drop=True)
```

- [ ] **Step 4: Run tests, verify they pass**

```bash
python -m pytest tests/utils/test_features.py -v
```
Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/utils/features.py tests/utils/test_features.py
git commit -m "add compute_forward_returns helper + tests"
```

---

## Task 3: `compute_text_novelty` helper

**Files:**
- Modify: `src/utils/features.py`
- Modify: `tests/utils/test_features.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/utils/test_features.py`:

```python
from src.utils.features import compute_text_novelty


def test_compute_text_novelty_identical_vectors_is_zero():
    embed = pd.DataFrame({
        'permno': [101, 101],
        'date': pd.to_datetime(['2020-01-08', '2020-01-15']),  # 7 days apart
        'vec': [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    })
    out = compute_text_novelty(embed, lookback_days=7)
    # First row has no t-7 → NaN. Second row's vec == prior vec → novelty = 0.
    assert pd.isna(out['text_novelty'].iloc[0])
    assert out['text_novelty'].iloc[1] == pytest.approx(0.0, abs=1e-6)


def test_compute_text_novelty_orthogonal_vectors_is_one():
    embed = pd.DataFrame({
        'permno': [101, 101],
        'date': pd.to_datetime(['2020-01-08', '2020-01-15']),
        'vec': [[1.0, 0.0], [0.0, 1.0]],
    })
    out = compute_text_novelty(embed, lookback_days=7)
    # cosine_sim = 0 → novelty = 1 - 0 = 1
    assert out['text_novelty'].iloc[1] == pytest.approx(1.0, abs=1e-6)


def test_compute_text_novelty_does_not_cross_permnos():
    embed = pd.DataFrame({
        'permno': [101, 202],
        'date': pd.to_datetime(['2020-01-08', '2020-01-15']),  # different permnos
        'vec': [[1.0, 0.0], [0.0, 1.0]],
    })
    out = compute_text_novelty(embed, lookback_days=7)
    # Neither row has a t-7 same-permno predecessor → both NaN
    assert out['text_novelty'].isna().all()
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
python -m pytest tests/utils/test_features.py -v
```
Expected: 3 new tests FAIL.

- [ ] **Step 3: Implement `compute_text_novelty`**

Append to `src/utils/features.py`:

```python
def compute_text_novelty(
    embed: pd.DataFrame,
    lookback_days: int = 7,
    permno_col: str = 'permno',
    date_col: str = 'date',
    vec_col: str = 'vec',
) -> pd.DataFrame:
    """Compute `text_novelty` = 1 − cosine_similarity(e_{i,t}, e_{i, t−lookback}).

    `lookback_days` is calendar days. For each (permno, date), looks up the
    same permno's vector at exactly `date − lookback_days` (calendar). If no
    such row exists, writes NaN. Embedding panel is assumed forward-filled to
    daily (notebook 03 output) so the lookup hits in steady state.
    """
    out = embed[[permno_col, date_col, vec_col]].copy()
    lookup = out.set_index([permno_col, date_col])[vec_col].to_dict()
    novelty = []
    for permno, date, vec in zip(out[permno_col], out[date_col], out[vec_col]):
        prior_date = date - pd.Timedelta(days=lookback_days)
        prior_vec = lookup.get((permno, prior_date))
        if prior_vec is None:
            novelty.append(np.nan)
            continue
        a = np.asarray(vec, dtype=np.float32)
        b = np.asarray(prior_vec, dtype=np.float32)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0.0:
            novelty.append(np.nan)
            continue
        novelty.append(float(1.0 - np.dot(a, b) / denom))
    out['text_novelty'] = np.asarray(novelty, dtype=np.float32)
    return out[[permno_col, date_col, 'text_novelty']]
```

- [ ] **Step 4: Run tests, verify they pass**

```bash
python -m pytest tests/utils/test_features.py -v
```
Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/utils/features.py tests/utils/test_features.py
git commit -m "add compute_text_novelty helper + tests (cosine vs t-7d)"
```

---

## Task 4: `compute_days_since_filing` helper

**Files:**
- Modify: `src/utils/features.py`
- Modify: `tests/utils/test_features.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/utils/test_features.py`:

```python
from src.utils.features import compute_days_since_filing


def test_compute_days_since_filing_simple_case():
    filings = pd.DataFrame({
        'cik': ['0000000101', '0000000101'],
        'filing_date': pd.to_datetime(['2020-01-01', '2020-02-01']),
        'form_type': ['10-K', '10-Q'],
    })
    panel = pd.DataFrame({
        'permno': [101, 101, 101],
        'cik': ['0000000101'] * 3,
        'date': pd.to_datetime(['2020-01-05', '2020-02-05', '2020-03-10']),
    })
    out = compute_days_since_filing(filings, panel)
    assert out['days_since_filing'].tolist() == [4, 4, 38]


def test_compute_days_since_filing_returns_nan_before_first_filing():
    filings = pd.DataFrame({
        'cik': ['0000000101'],
        'filing_date': pd.to_datetime(['2020-06-01']),
        'form_type': ['10-K'],
    })
    panel = pd.DataFrame({
        'permno': [101],
        'cik': ['0000000101'],
        'date': pd.to_datetime(['2020-01-15']),  # before any filing
    })
    out = compute_days_since_filing(filings, panel)
    assert pd.isna(out['days_since_filing'].iloc[0])


def test_compute_days_since_filing_excludes_non_kqa_forms():
    filings = pd.DataFrame({
        'cik': ['0000000101', '0000000101'],
        'filing_date': pd.to_datetime(['2020-01-01', '2020-02-01']),
        'form_type': ['DEF 14A', '10-K'],  # 14A excluded; 10-K counted
    })
    panel = pd.DataFrame({
        'permno': [101],
        'cik': ['0000000101'],
        'date': pd.to_datetime(['2020-02-10']),
    })
    out = compute_days_since_filing(filings, panel)
    assert out['days_since_filing'].iloc[0] == 9  # from 2020-02-01, not 2020-01-01
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
python -m pytest tests/utils/test_features.py -v
```
Expected: 3 new tests FAIL.

- [ ] **Step 3: Implement `compute_days_since_filing`**

Append to `src/utils/features.py`:

```python
FORM_TYPES_KQA = ('10-K', '10-Q', '8-K')


def compute_days_since_filing(
    filings: pd.DataFrame,
    panel: pd.DataFrame,
    form_types: tuple[str, ...] = FORM_TYPES_KQA,
) -> pd.DataFrame:
    """For each (permno, date), days since the most recent filing (10-K/Q/8-K).

    `filings` columns: cik, filing_date, form_type.
    `panel` columns: permno, cik, date.
    Returns the panel with `days_since_filing` (int or NaN if no prior filing).
    Uses `merge_asof` (left, by cik, on filing_date) for vectorized lookup.
    """
    f = filings[filings['form_type'].isin(form_types)][['cik', 'filing_date']].copy()
    f = f.sort_values(['cik', 'filing_date'])
    p = panel.sort_values(['cik', 'date']).copy()
    merged = pd.merge_asof(
        p,
        f.rename(columns={'filing_date': 'last_filing_date'}),
        left_on='date',
        right_on='last_filing_date',
        by='cik',
        direction='backward',
    )
    delta = (merged['date'] - merged['last_filing_date']).dt.days
    out = panel[['permno', 'date']].copy()
    out['days_since_filing'] = delta.values
    return out
```

- [ ] **Step 4: Run tests, verify they pass**

```bash
python -m pytest tests/utils/test_features.py -v
```
Expected: all 10 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/utils/features.py tests/utils/test_features.py
git commit -m "add compute_days_since_filing helper + tests (10-K/Q/8-K, merge_asof)"
```

---

## Task 5: `compute_doc_count_window` helper

**Files:**
- Modify: `src/utils/features.py`
- Modify: `tests/utils/test_features.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/utils/test_features.py`:

```python
from src.utils.features import compute_doc_count_window


def test_compute_doc_count_window_counts_filings_in_window():
    filings = pd.DataFrame({
        'cik': ['0000000101'] * 4,
        'filing_date': pd.to_datetime(['2020-06-01', '2020-06-03', '2020-06-05', '2020-06-15']),
        'form_type': ['8-K', '8-K', '10-Q', '8-K'],
    })
    panel = pd.DataFrame({
        'permno': [101],
        'cik': ['0000000101'],
        'date': pd.to_datetime(['2020-06-07']),  # window = [2020-05-31, 2020-06-07]
    })
    out = compute_doc_count_window(filings, panel, window_days=7)
    # 3 filings in [2020-05-31, 2020-06-07]: 06-01, 06-03, 06-05
    assert out['doc_count_7d'].iloc[0] == 3


def test_compute_doc_count_window_returns_zero_when_no_filings():
    filings = pd.DataFrame({
        'cik': ['0000000101'],
        'filing_date': pd.to_datetime(['2019-01-01']),
        'form_type': ['10-K'],
    })
    panel = pd.DataFrame({
        'permno': [101],
        'cik': ['0000000101'],
        'date': pd.to_datetime(['2020-06-07']),
    })
    out = compute_doc_count_window(filings, panel, window_days=7)
    assert out['doc_count_7d'].iloc[0] == 0


def test_compute_doc_count_window_does_not_count_other_permnos():
    filings = pd.DataFrame({
        'cik': ['0000000999'],  # different cik
        'filing_date': pd.to_datetime(['2020-06-03']),
        'form_type': ['8-K'],
    })
    panel = pd.DataFrame({
        'permno': [101],
        'cik': ['0000000101'],
        'date': pd.to_datetime(['2020-06-07']),
    })
    out = compute_doc_count_window(filings, panel, window_days=7)
    assert out['doc_count_7d'].iloc[0] == 0
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
python -m pytest tests/utils/test_features.py -v
```
Expected: 3 new tests FAIL.

- [ ] **Step 3: Implement `compute_doc_count_window`**

Append to `src/utils/features.py`:

```python
def compute_doc_count_window(
    filings: pd.DataFrame,
    panel: pd.DataFrame,
    window_days: int = 7,
) -> pd.DataFrame:
    """For each (permno, date), count filings on that cik in `[date − window_days, date]`.

    All form types counted (not restricted to K/Q/8K). Returns the panel with
    `doc_count_{window_days}d` column (int32).
    """
    col = f'doc_count_{window_days}d'
    f = filings[['cik', 'filing_date']].sort_values(['cik', 'filing_date']).copy()
    # For each panel row, count filings with filing_date in [date - window, date].
    # We can do this with two merge_asof's: one for the right edge (date), one
    # for the left edge (date - window - 1 day). But cleaner: per-cik join then filter.
    p = panel[['permno', 'cik', 'date']].copy()
    p['_left'] = p['date'] - pd.Timedelta(days=window_days)
    merged = p.merge(f, on='cik', how='left')
    in_window = (merged['filing_date'] >= merged['_left']) & (merged['filing_date'] <= merged['date'])
    merged['_count'] = in_window.astype('int32')
    counts = (
        merged.groupby(['permno', 'date'], as_index=False)['_count'].sum()
        .rename(columns={'_count': col})
    )
    out = panel[['permno', 'date']].merge(counts, on=['permno', 'date'], how='left')
    out[col] = out[col].fillna(0).astype('int32')
    return out
```

- [ ] **Step 4: Run tests, verify they pass**

```bash
python -m pytest tests/utils/test_features.py -v
```
Expected: all 13 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/utils/features.py tests/utils/test_features.py
git commit -m "add compute_doc_count_window helper + tests"
```

---

## Task 6: Notebook 05 — cells A (intro + setup) and B (load + universe filter + macro)

**Files:**
- Create: `notebooks/05_assemble_training_panel.ipynb`

- [ ] **Step 1: Create the notebook with cells A + intro markdown only first, verify syntax**

Use NotebookEdit to create the notebook with these cells in order. For brevity here the markdown headers are inline:

**Cell intro (markdown):**

```markdown
# 05 — Assemble training panel

Join the existing PIT panel (CRSP + Sharadar) with macro indicators, aux text
features (text_novelty, days_since_filing, doc_count_7d), and forward-return
labels. Output: `data/processed/training_panel/year=YYYY/part-0.parquet`,
daily cadence, partitioned by year. Consumed by notebook 06 (ranker,
Friday-filtered) and notebook 07+ (RL agent, daily).

**Spec:** `docs/superpowers/specs/2026-05-16-feature-assembly-design.md`.

**Resumable:** existing year-shards are skipped on re-run. Delete a year's
`part-0.parquet` to re-do.

PCA text features are *not* in this panel — they're derived at ranker time
from `finbert_stockday_embed` + the walk's `pca.joblib`. See spec §6.
```

**Cell A markdown:**

```markdown
## A. Setup
```

**Cell A code:**

```python
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from src.utils.io import processed_dir, repo_root
from src.utils.features import (
    pivot_macro_wide,
    compute_forward_returns,
    compute_text_novelty,
    compute_days_since_filing,
    compute_doc_count_window,
)

PANEL_DIR = processed_dir() / 'panel'
MACRO_PATH = processed_dir() / 'macro.parquet'
EMBED_DIR = processed_dir() / 'finbert_stockday_embed'
EDGAR_INDEX_PATH = processed_dir() / 'edgar_index.parquet'
UNIVERSE_PATH = processed_dir() / 'universe_ids.parquet'
OUT_DIR = processed_dir() / 'training_panel'
SUMMARY_PATH = repo_root() / 'reports' / 'metrics' / 'feature_assembly_summary.json'

OUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

# Validation thresholds (spec §8).
MAX_TEXT_NOVELTY_NAN_RATE = 0.05
MAX_DAYS_SINCE_FILING_NAN_RATE = 0.10

YEARS = sorted({int(p.name.split('=')[1]) for p in PANEL_DIR.glob('year=*')})
print(f'panel years: {YEARS[0]}..{YEARS[-1]} ({len(YEARS)} years)')
print(f'out_dir: {OUT_DIR}')
```

**Cell B markdown:**

```markdown
## B. Load shared inputs (universe, macro, filings, embed dir handle)

Loaded once outside the per-year loop. Universe + filings are small; the
embed shards are accessed per-year inside the loop to keep peak memory bounded.
```

**Cell B code:**

```python
universe_ids = pd.read_parquet(UNIVERSE_PATH)
universe_ids['permno'] = universe_ids['permno'].astype('Int64')
universe_ids['date_out'] = universe_ids['date_out'].fillna(pd.Timestamp('2099-12-31'))
print(f'universe_ids: {len(universe_ids)} rows, {universe_ids["permno"].nunique()} permnos')

macro_long = pd.read_parquet(MACRO_PATH)
print(f'macro: {len(macro_long)} rows, series={sorted(macro_long["series"].unique())}')

edgar_index = pd.read_parquet(EDGAR_INDEX_PATH)
edgar_index['filing_date'] = pd.to_datetime(edgar_index['filing_date'])
# Restrict to ciks in our universe — drops noise from the broader EDGAR pull.
edgar_index = edgar_index[edgar_index['cik'].isin(universe_ids['cik'])]
print(f'edgar_index (universe-filtered): {len(edgar_index)} filings')
```

Run via NotebookEdit using edit_mode=insert for each cell after the previous.

- [ ] **Step 2: Compile-check cells A and B**

```bash
python -c "
import nbformat
nb = nbformat.read('notebooks/05_assemble_training_panel.ipynb', as_version=4)
for i, c in enumerate(nb.cells):
    if c.cell_type == 'code':
        compile(c.source, f'<cell {i}>', 'exec')
print(f'all {sum(1 for c in nb.cells if c.cell_type == \"code\")} code cells compile')
"
```
Expected: `all 2 code cells compile`.

- [ ] **Step 3: Commit**

```bash
git add notebooks/05_assemble_training_panel.ipynb
git commit -m "notebook 05: cells A (setup) + B (load shared inputs)"
```

---

## Task 7: Notebook 05 — cell C (per-year join + aux text features)

**Files:**
- Modify: `notebooks/05_assemble_training_panel.ipynb`

- [ ] **Step 1: Add cells C markdown + C code via NotebookEdit insert**

**Cell C markdown:**

```markdown
## C. Per-year assembly helper

`assemble_year(year)` does: read `panel/year=YYYY/`, universe-filter, left-join
macro (pivoted + ffilled over that year's date range), left-join cik via
universe_ids (so we can attach filings later), then attach aux text features
(text_novelty from the year's embed shard; days_since_filing + doc_count_7d
from edgar_index). Returns the per-year dataframe ready for labels (cell D)
and persist (cell F).
```

**Cell C code:**

```python
def _read_year_embed(year: int) -> pd.DataFrame:
    """Read finbert_stockday_embed shards for one year. Also peeks at year-1's
    last 7 days so the year's earliest rows can find a t-7 predecessor."""
    shards = sorted(EMBED_DIR.glob(f'year={year}/*.parquet'))
    prior_shards = sorted(EMBED_DIR.glob(f'year={year - 1}/*.parquet'))
    frames = []
    for s in shards:
        frames.append(pd.read_parquet(s, columns=['permno', 'date', 'vec']))
    if prior_shards:
        for s in prior_shards:
            df = pd.read_parquet(s, columns=['permno', 'date', 'vec'])
            df = df[df['date'] >= pd.Timestamp(f'{year - 1}-12-24')]  # last week of prior year
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=['permno', 'date', 'vec'])
    embed = pd.concat(frames, ignore_index=True)
    embed['date'] = pd.to_datetime(embed['date'])
    return embed


def assemble_year(year: int) -> pd.DataFrame:
    # 1. Read panel year.
    panel_shards = sorted(PANEL_DIR.glob(f'year={year}/*.parquet'))
    base = pd.concat([pd.read_parquet(s) for s in panel_shards], ignore_index=True)
    base['date'] = pd.to_datetime(base['date'])

    # 2. Universe filter (interval merge, same logic as src/utils/pca.py).
    intervals = universe_ids[['permno', 'cik', 'date_in', 'date_out']].copy()
    intervals['permno'] = intervals['permno'].astype('int64')
    merged = base.merge(intervals, on='permno', how='inner')
    in_window = (merged['date'] >= merged['date_in']) & (merged['date'] <= merged['date_out'])
    base = (merged[in_window]
            .drop(columns=['date_in', 'date_out'])
            .drop_duplicates(subset=['permno', 'date']))

    # 3. Macro: pivot, ffill to this year's calendar dates, left-join.
    year_dates = pd.DatetimeIndex(sorted(base['date'].unique()))
    macro_w = pivot_macro_wide(macro_long, ffill_dates=year_dates)
    base = base.merge(macro_w, on='date', how='left')

    # 4. Aux text features.
    embed = _read_year_embed(year)
    novelty = compute_text_novelty(embed, lookback_days=7)
    # Drop the borrowed t-1y rows from the output (we only kept them for lookups).
    novelty = novelty[novelty['date'].dt.year == year]
    base = base.merge(novelty, on=['permno', 'date'], how='left')

    dsf = compute_days_since_filing(edgar_index, base[['permno', 'cik', 'date']])
    base = base.merge(dsf, on=['permno', 'date'], how='left')

    docc = compute_doc_count_window(edgar_index, base[['permno', 'cik', 'date']], window_days=7)
    base = base.merge(docc, on=['permno', 'date'], how='left')

    return base
```

- [ ] **Step 2: Compile-check**

```bash
python -c "
import nbformat
nb = nbformat.read('notebooks/05_assemble_training_panel.ipynb', as_version=4)
for i, c in enumerate(nb.cells):
    if c.cell_type == 'code':
        compile(c.source, f'<cell {i}>', 'exec')
print('all code cells compile')
"
```
Expected: `all code cells compile`.

- [ ] **Step 3: Commit**

```bash
git add notebooks/05_assemble_training_panel.ipynb
git commit -m "notebook 05: cell C (per-year assemble helper — universe filter, macro join, aux text features)"
```

---

## Task 8: Notebook 05 — cell D (labels) + cell E (validation gates)

**Files:**
- Modify: `notebooks/05_assemble_training_panel.ipynb`

- [ ] **Step 1: Add cells D + E**

**Cell D markdown:**

```markdown
## D. Per-year label computation

Forward returns require lookahead, so labels for year Y need ret values from
year Y+1. The year-by-year persist below handles this by stitching a small
prefix of year Y+1's panel onto year Y before computing labels, then dropping
the prefix before writing.
```

**Cell D code:**

```python
def attach_labels(year_df: pd.DataFrame, year: int, max_horizon: int = 5) -> pd.DataFrame:
    """Compute fwd_ret_1d and fwd_ret_5d for year_df, using year+1's prefix
    so the last `max_horizon` trading days of year_df get full labels."""
    next_year_shards = sorted(PANEL_DIR.glob(f'year={year + 1}/*.parquet'))
    if next_year_shards:
        prefix = pd.concat([pd.read_parquet(s) for s in next_year_shards], ignore_index=True)
        prefix['date'] = pd.to_datetime(prefix['date'])
        # Keep ~10 trading days (calendar buffer of 14) so the rolling has room.
        prefix = prefix[prefix['date'] <= pd.Timestamp(f'{year + 1}-01-15')]
        prefix = prefix.merge(
            universe_ids[['permno', 'date_in', 'date_out']],
            on='permno', how='inner',
        )
        in_window = (prefix['date'] >= prefix['date_in']) & (prefix['date'] <= prefix['date_out'])
        prefix = prefix[in_window].drop(columns=['date_in', 'date_out'])
        prefix = prefix[['permno', 'date', 'ret']].drop_duplicates(subset=['permno', 'date'])
        stitched = pd.concat(
            [year_df[['permno', 'date', 'ret']], prefix],
            ignore_index=True,
        )
    else:
        stitched = year_df[['permno', 'date', 'ret']]

    labeled = compute_forward_returns(stitched, horizons=(1, 5))
    labeled = labeled[labeled['date'].dt.year == year]
    return year_df.merge(
        labeled[['permno', 'date', 'fwd_ret_1d', 'fwd_ret_5d']],
        on=['permno', 'date'], how='left',
    )
```

**Cell E markdown:**

```markdown
## E. Validation gates

Per-year hard stops (spec §8). Raises on any gate failure — the year's
partial output is not persisted.
```

**Cell E code:**

```python
def validate_year(df: pd.DataFrame, year: int) -> None:
    assert len(df) > 0, f'year {year}: empty dataframe'
    expected_cols = {
        'permno', 'date', 'ret', 'macro_vixcls', 'macro_dgs10', 'macro_dgs3mo',
        'macro_t10y2y', 'text_novelty', 'days_since_filing', 'doc_count_7d',
        'fwd_ret_1d', 'fwd_ret_5d',
    }
    missing = expected_cols - set(df.columns)
    assert not missing, f'year {year}: missing columns {missing}'

    novelty_nan = df['text_novelty'].isna().mean()
    assert novelty_nan <= MAX_TEXT_NOVELTY_NAN_RATE, (
        f'year {year}: text_novelty NaN rate {novelty_nan:.3f} '
        f'> threshold {MAX_TEXT_NOVELTY_NAN_RATE}'
    )

    dsf_nan = df['days_since_filing'].isna().mean()
    assert dsf_nan <= MAX_DAYS_SINCE_FILING_NAN_RATE, (
        f'year {year}: days_since_filing NaN rate {dsf_nan:.3f} '
        f'> threshold {MAX_DAYS_SINCE_FILING_NAN_RATE}'
    )

    # fwd_ret_5d NaN should be bounded — roughly: (last 5 trading days per permno)
    # / total rows. With ~250 trading days and ~600 permnos -> at most ~6 / 250
    # of rows. Use 3% as the floor.
    fwd_nan = df['fwd_ret_5d'].isna().mean()
    assert fwd_nan <= 0.05, (
        f'year {year}: fwd_ret_5d NaN rate {fwd_nan:.3f} > 0.05 '
        '(expected only tail rows of each permno)'
    )
```

- [ ] **Step 2: Compile-check**

```bash
python -c "
import nbformat
nb = nbformat.read('notebooks/05_assemble_training_panel.ipynb', as_version=4)
for i, c in enumerate(nb.cells):
    if c.cell_type == 'code':
        compile(c.source, f'<cell {i}>', 'exec')
print('all code cells compile')
"
```
Expected: `all code cells compile`.

- [ ] **Step 3: Commit**

```bash
git add notebooks/05_assemble_training_panel.ipynb
git commit -m "notebook 05: cells D (labels) + E (validation gates)"
```

---

## Task 9: Notebook 05 — cell F (per-year loop + persist + summary), gitignore, final commit

**Files:**
- Modify: `notebooks/05_assemble_training_panel.ipynb`
- Modify: `.gitignore`

- [ ] **Step 1: Add cell F**

**Cell F markdown:**

```markdown
## F. Per-year loop, atomic persist, summary

For each year, skip if `training_panel/year=YYYY/part-0.parquet` exists.
Otherwise: assemble → label → validate → atomic write (`*.tmp → rename`).
After the loop, write per-year row counts + NaN audit to
`reports/metrics/feature_assembly_summary.json`.
```

**Cell F code:**

```python
def shard_path(year: int) -> Path:
    return OUT_DIR / f'year={year}' / 'part-0.parquet'


def process_year(year: int) -> dict:
    out_path = shard_path(year)
    if out_path.exists():
        n = pq.read_metadata(out_path).num_rows
        print(f'year={year}: exists ({n} rows) — skipping')
        return {'year': year, 'n_rows': int(n), 'status': 'skipped'}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = assemble_year(year)
    df = attach_labels(df, year)
    validate_year(df, year)

    tmp = out_path.with_suffix('.parquet.tmp')
    df.to_parquet(tmp, compression='zstd', index=False)
    tmp.rename(out_path)
    print(f'year={year}: wrote {len(df)} rows -> {out_path}')
    return {
        'year': year,
        'n_rows': int(len(df)),
        'n_permnos': int(df['permno'].nunique()),
        'nan_rate_text_novelty': float(df['text_novelty'].isna().mean()),
        'nan_rate_days_since_filing': float(df['days_since_filing'].isna().mean()),
        'nan_rate_fwd_ret_5d': float(df['fwd_ret_5d'].isna().mean()),
        'status': 'written',
    }


per_year = [process_year(y) for y in YEARS]
SUMMARY_PATH.write_text(json.dumps({'years': per_year}, indent=2))
print(f'\nsummary -> {SUMMARY_PATH.relative_to(repo_root())}')
```

- [ ] **Step 2: Compile-check all cells**

```bash
python -c "
import nbformat
nb = nbformat.read('notebooks/05_assemble_training_panel.ipynb', as_version=4)
for i, c in enumerate(nb.cells):
    if c.cell_type == 'code':
        compile(c.source, f'<cell {i}>', 'exec')
print(f'all {sum(1 for c in nb.cells if c.cell_type == \"code\")} code cells compile')
"
```
Expected: `all 6 code cells compile`.

- [ ] **Step 3: Gitignore the panel output**

Add to `.gitignore` in the `# Large derivatives that go to Cloudflare R2, not git` block:

```
data/processed/training_panel/
```

- [ ] **Step 4: Run the full test suite**

```bash
python -m pytest tests/ -q
```
Expected: PASS (existing tests + the 13 new feature tests).

- [ ] **Step 5: Commit**

```bash
git add notebooks/05_assemble_training_panel.ipynb .gitignore
git commit -m "notebook 05: cell F (per-year loop, atomic persist, summary); gitignore training_panel/"
```

---

## Task 10: Push branch + hand off for notebook execution

**Files:** none (git operation only)

- [ ] **Step 1: Push the branch**

```bash
git -c credential.helper= -c credential.helper='!gh auth git-credential' push -u origin feature-assembly
```
Expected: branch created on origin.

- [ ] **Step 2: Hand off to the user**

The notebook is not executed by the agent — the user runs it in Cursor for live
metrics (per `feedback_notebook_execution`). Tell the user:

- Branch `feature-assembly` is pushed.
- Run `notebooks/05_assemble_training_panel.ipynb` cell by cell.
- Watch for cell-E asserts on the first year — those validate the joins
  and aux feature logic on real data.
- Once all years are written cleanly, the summary JSON should show low NaN
  rates (text_novelty < 5%, days_since_filing < 10%, fwd_ret_5d ~1%).
- After the user confirms, we open a PR + merge to main + add
  `data/processed/training_panel/` to the R2 sync (separate small follow-up).

---

## Self-Review (Completed)

**Spec coverage:**
- §1 goal ✓ (Task 6+)
- §2 inputs ✓ (cell B loads each)
- §3 output schema ✓ (cell C joins all sources; cell D appends labels; cell F persists)
- §4 computation details ✓ (Tasks 1–5 implement each helper)
- §5 macro ✓ (Task 1)
- §6 no PCA in panel ✓ (intro markdown calls it out; not in the code)
- §7 cell structure ✓ (Tasks 6–9)
- §8 validation gates ✓ (Task 8 cell E + thresholds in cell A)
- §9 resumability + atomicity ✓ (Task 9 `process_year` checks exists + tmp→rename)
- §10 downstream contracts — informational only, no code needed
- §11 out of scope — informational
- §12 risks — memory bounded by per-year loop ✓

**Placeholder scan:** No TBD / TODO / "fill in" / "similar to" patterns. Every step contains exact code or exact commands.

**Type consistency:** Helper signatures match across the plan: `pivot_macro_wide(macro_long, ffill_dates)`, `compute_forward_returns(panel, horizons)`, `compute_text_novelty(embed, lookback_days)`, `compute_days_since_filing(filings, panel, form_types)`, `compute_doc_count_window(filings, panel, window_days)`. Notebook imports match (Task 6 cell A) and call sites in Task 7 use the same signatures.

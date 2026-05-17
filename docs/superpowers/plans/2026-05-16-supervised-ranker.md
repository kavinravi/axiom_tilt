# Supervised Ranker (Notebook 06) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Walk-1 MVP of the LightGBM lambdarank ranker — pure helpers in `src/utils/ranker.py` + orchestration in `notebooks/06_supervised_ranker.ipynb`. Output: `artifacts/ranker/walk-001/{model.joblib, hp.json, summary.json, feature_importance.csv, optuna_study.pkl}`.

**Architecture:** Pure functions for feature assembly + metrics live in `src/utils/ranker.py`, fully unit-tested. Notebook reads PCA + panel + embeddings, calls helpers, runs Optuna (15 trials on val NDCG@30), final fit with early stopping, evaluates OOS on 2009, persists.

**Tech Stack:** `lightgbm >= 4.2`, `optuna >= 3.5`, `scikit-learn`, `pandas`, `pyarrow`, `joblib`. Conventions per the `machine-learning` skill (no test-set tuning, early-stopping callback, Optuna study persistence, gain-based feature importance).

**Spec:** `docs/superpowers/specs/2026-05-16-supervised-ranker-design.md`.

---

### Task 1: `load_walk_pca` + `project_text_to_pca` helpers

**Files:**
- Modify: `src/utils/ranker.py` (create)
- Test: `tests/utils/test_ranker.py` (create)

- [ ] **Step 1: Write failing tests for both helpers**

```python
# tests/utils/test_ranker.py
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import joblib
from sklearn.decomposition import PCA

from src.utils.ranker import load_walk_pca, project_text_to_pca


def test_load_walk_pca_returns_pca_and_n_components(tmp_path, monkeypatch):
    # Set up fake artifacts/pca-text/walk-001/pca.joblib
    walk_dir = tmp_path / 'artifacts' / 'pca-text' / 'walk-001'
    walk_dir.mkdir(parents=True)
    pca = PCA(n_components=5).fit(np.random.RandomState(0).randn(50, 20))
    joblib.dump(pca, walk_dir / 'pca.joblib')

    monkeypatch.chdir(tmp_path)
    loaded_pca, n_pca = load_walk_pca(walk_id=1)
    assert isinstance(loaded_pca, PCA)
    assert n_pca == 5


def test_project_text_to_pca_returns_correct_shape():
    rng = np.random.RandomState(0)
    pca = PCA(n_components=3).fit(rng.randn(20, 10))
    embed = pd.DataFrame({
        'permno': [101, 102, 103],
        'date': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03']),
        'vec': [rng.randn(10).astype(np.float32) for _ in range(3)],
    })
    out = project_text_to_pca(embed, pca)
    assert list(out.columns) == ['permno', 'date', 'pca_0', 'pca_1', 'pca_2']
    assert len(out) == 3
```

- [ ] **Step 2: Run tests, verify they fail**

`python -m pytest tests/utils/test_ranker.py -q` → expect "ModuleNotFoundError" or function-not-defined.

- [ ] **Step 3: Implement both helpers**

```python
# src/utils/ranker.py
"""Helpers for the supervised ranker (notebook 06).

Pure functions over pandas DataFrames + numpy arrays so the notebook stays
a thin orchestration layer. See docs/superpowers/specs/2026-05-16-supervised-ranker-design.md.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def load_walk_pca(walk_id: int, artifacts_root: Path | None = None) -> tuple[PCA, int]:
    """Load fitted PCA from notebook 04's artifacts."""
    root = artifacts_root or Path('artifacts')
    path = root / 'pca-text' / f'walk-{walk_id:03d}' / 'pca.joblib'
    pca: PCA = joblib.load(path)
    return pca, int(pca.n_components_)


def project_text_to_pca(embed: pd.DataFrame, pca: PCA, vec_col: str = 'vec') -> pd.DataFrame:
    """Project a (permno, date, vec) embedding frame to (permno, date, pca_0..pca_{n-1})."""
    X = np.vstack(embed[vec_col].to_numpy()).astype(np.float32)
    Z = pca.transform(X).astype(np.float32)
    cols = [f'pca_{i}' for i in range(Z.shape[1])]
    out = pd.DataFrame(Z, columns=cols, index=embed.index)
    return pd.concat([embed[['permno', 'date']].reset_index(drop=True),
                      out.reset_index(drop=True)], axis=1)
```

- [ ] **Step 4: Run tests, verify they pass**

`python -m pytest tests/utils/test_ranker.py -q` → 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/utils/ranker.py tests/utils/test_ranker.py
git commit -m "add load_walk_pca + project_text_to_pca helpers (notebook 06 prep)"
```

---

### Task 2: `friday_only` + `compute_excess_return_buckets` helpers

**Files:**
- Modify: `src/utils/ranker.py`
- Test: `tests/utils/test_ranker.py`

- [ ] **Step 1: Write failing tests**

```python
from src.utils.ranker import friday_only, compute_excess_return_buckets


def test_friday_only_keeps_only_weekday_4():
    df = pd.DataFrame({
        'date': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03',
                                '2020-01-06', '2020-01-10']),  # Wed, Thu, Fri, Mon, Fri
    })
    out = friday_only(df)
    assert out['date'].dt.dayofweek.unique().tolist() == [4]
    assert len(out) == 2


def test_compute_excess_return_buckets_centers_per_date():
    df = pd.DataFrame({
        'permno': [101, 102, 103, 104, 105, 106],
        'date': pd.to_datetime(['2020-01-03'] * 3 + ['2020-01-10'] * 3),
        'fwd_ret_5d': [0.01, 0.02, 0.03, -0.01, 0.00, 0.01],
    })
    out = compute_excess_return_buckets(df, n_buckets=3)
    # Per-date ranks: 2020-01-03 -> [0, 1, 2]; 2020-01-10 -> [0, 1, 2]
    assert out.iloc[0] < out.iloc[2]  # lower excess return -> lower bucket
    assert out.iloc[2] > out.iloc[0]
    assert out.iloc[3] < out.iloc[5]
    assert out.dropna().between(0, 2).all()  # 3 buckets -> labels 0,1,2


def test_compute_excess_return_buckets_drops_nan_rows():
    df = pd.DataFrame({
        'permno': [101, 102],
        'date': pd.to_datetime(['2020-01-03', '2020-01-03']),
        'fwd_ret_5d': [0.01, np.nan],
    })
    out = compute_excess_return_buckets(df, n_buckets=2)
    assert pd.isna(out.iloc[1])
```

- [ ] **Step 2: Run tests, verify they fail**

`python -m pytest tests/utils/test_ranker.py -q -k "friday_only or excess_return"`

- [ ] **Step 3: Implement helpers**

```python
def friday_only(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """Keep only Friday rows (weekday == 4)."""
    return df[df[date_col].dt.dayofweek == 4].copy()


def compute_excess_return_buckets(
    df: pd.DataFrame,
    ret_col: str = 'fwd_ret_5d',
    date_col: str = 'date',
    n_buckets: int = 32,
) -> pd.Series:
    """Cross-sectional excess return (vs mean) -> percentile rank -> bucket.

    Returns int label in [0, n_buckets-1] aligned to df.index; NaN where
    `ret_col` is NaN.
    """
    s = df[ret_col]
    grp = df.groupby(date_col)[ret_col]
    excess = s - grp.transform('mean')
    # Per-date percentile rank in [0, 1], NaN-safe.
    pct = excess.groupby(df[date_col]).rank(pct=True, method='average')
    # Bucket: floor(pct * n_buckets), clipped to [0, n_buckets-1].
    bucket = np.floor(pct * n_buckets).clip(upper=n_buckets - 1)
    bucket = bucket.where(pct.notna())
    return bucket.astype('Int64')
```

- [ ] **Step 4: Run tests, verify they pass**

- [ ] **Step 5: Commit**

```bash
git add -u
git commit -m "add friday_only + compute_excess_return_buckets helpers"
```

---

### Task 3: `assemble_walk_features` helper

**Files:**
- Modify: `src/utils/ranker.py`
- Test: `tests/utils/test_ranker.py`

- [ ] **Step 1: Write failing test**

```python
from src.utils.ranker import assemble_walk_features


def test_assemble_walk_features_joins_panel_and_pca_drops_non_features():
    panel = pd.DataFrame({
        'permno': [101, 102, 101, 102],
        'date': pd.to_datetime(['2020-01-03'] * 2 + ['2020-01-10'] * 2),
        'cik': ['a', 'b', 'a', 'b'],
        'ret': [0.01, 0.02, 0.0, 0.01],
        'ticker': ['A', 'B', 'A', 'B'],
        'fwd_ret_5d': [0.01, 0.02, 0.0, 0.01],
        'macro_vixcls': [20.0, 20.0, 22.0, 22.0],
        'text_novelty': [0.1, 0.2, 0.15, 0.25],
        'feature_x': [1.0, 2.0, 3.0, 4.0],
    })
    embed_pca = pd.DataFrame({
        'permno': [101, 102, 101, 102],
        'date': pd.to_datetime(['2020-01-03'] * 2 + ['2020-01-10'] * 2),
        'pca_0': [0.5, 0.6, 0.7, 0.8],
        'pca_1': [1.0, 1.1, 1.2, 1.3],
    })
    X, y, groups, meta = assemble_walk_features(panel, embed_pca)
    # Non-feature cols dropped from X
    for col in ['permno', 'date', 'cik', 'ret', 'ticker', 'fwd_ret_5d']:
        assert col not in X.columns
    # PCA + macro + text + structured retained
    assert {'pca_0', 'pca_1', 'macro_vixcls', 'text_novelty', 'feature_x'} <= set(X.columns)
    assert len(X) == len(y) == 4
    # groups: counts per Friday date in order
    assert groups == [2, 2]
    # meta has permno/date for joining back
    assert {'permno', 'date'} <= set(meta.columns)
```

- [ ] **Step 2: Run test, verify it fails**

- [ ] **Step 3: Implement helper**

```python
# Columns that are never features (identifiers, labels, or non-numeric metadata)
NON_FEATURE_COLS = frozenset([
    'permno', 'date', 'cik', 'ret', 'ticker',
    'fiscalperiod', 'datekey', 'calendardate', 'reportperiod', 'lastupdated',
    'dimension', 'in_universe',
    'fwd_ret_1d', 'fwd_ret_5d',
])


def assemble_walk_features(
    panel: pd.DataFrame,
    embed_pca: pd.DataFrame,
    target_col: str = 'fwd_ret_5d',
    date_col: str = 'date',
) -> tuple[pd.DataFrame, pd.Series, list[int], pd.DataFrame]:
    """Inner-join Friday panel rows with PCA embeddings, drop NaN target rows,
    build (X, y, group_sizes, meta)."""
    merged = friday_only(panel).merge(embed_pca, on=['permno', date_col], how='inner')
    merged = merged.dropna(subset=[target_col]).sort_values([date_col, 'permno']).reset_index(drop=True)

    y_excess = (merged[target_col]
                - merged.groupby(date_col)[target_col].transform('mean'))
    feature_cols = [c for c in merged.columns
                    if c not in NON_FEATURE_COLS
                    and pd.api.types.is_numeric_dtype(merged[c])]
    X = merged[feature_cols].copy()
    y = y_excess.copy()
    groups = merged.groupby(date_col, sort=False).size().tolist()
    meta = merged[['permno', date_col, target_col]].copy()
    return X, y, groups, meta
```

- [ ] **Step 4: Run test, verify it passes**

- [ ] **Step 5: Commit**

```bash
git add -u
git commit -m "add assemble_walk_features helper (Friday filter + PCA join + group sizes)"
```

---

### Task 4: `build_ranker` factory + `evaluate_ranker` helper

**Files:**
- Modify: `src/utils/ranker.py`
- Test: `tests/utils/test_ranker.py`

- [ ] **Step 1: Write failing tests**

```python
from src.utils.ranker import build_ranker, evaluate_ranker


def test_build_ranker_returns_lgbm_ranker_with_lambdarank():
    from lightgbm import LGBMRanker
    model = build_ranker({'num_leaves': 31, 'learning_rate': 0.05, 'n_estimators': 50})
    assert isinstance(model, LGBMRanker)
    assert model.objective == 'lambdarank'
    assert model.num_leaves == 31


def test_evaluate_ranker_returns_metric_dict_with_required_keys():
    from lightgbm import LGBMRanker
    rng = np.random.RandomState(0)
    n_dates, n_per_date = 5, 30
    X = rng.randn(n_dates * n_per_date, 8).astype(np.float32)
    # Synthetic signal so rank IC > 0.
    y_excess = X[:, 0] * 0.5 + rng.randn(len(X)) * 0.1
    groups = [n_per_date] * n_dates
    dates = pd.to_datetime([f'2020-01-{i*7+3:02d}' for i in range(n_dates)])
    group_dates = np.repeat(dates, n_per_date)

    # Use buckets as the lambdarank label proxy.
    pct = pd.Series(y_excess).groupby(group_dates).rank(pct=True)
    labels = np.floor(pct * 4).clip(upper=3).astype(int).values
    model = LGBMRanker(objective='lambdarank', n_estimators=100, verbose=-1)
    model.fit(X, labels, group=groups)

    out = evaluate_ranker(model, X, y_excess, group_dates)
    for k in ['rank_ic_mean', 'rank_ic_ir', 'decile_spread_bps',
              'hit_rate', 'top_k_jaccard']:
        assert k in out
    assert isinstance(out['rank_ic_mean'], float)
```

- [ ] **Step 2: Run tests, verify they fail**

- [ ] **Step 3: Implement helpers**

```python
from lightgbm import LGBMRanker


DEFAULT_RANKER_PARAMS: dict[str, Any] = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'eval_at': [30],
    'num_leaves': 63,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.9,
    'bagging_freq': 5,
    'min_data_in_leaf': 50,
    'lambda_l2': 1.0,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
}


def build_ranker(params: dict | None = None) -> LGBMRanker:
    """LGBMRanker factory with lambdarank defaults; `params` overrides."""
    merged = {**DEFAULT_RANKER_PARAMS, **(params or {})}
    return LGBMRanker(**merged)


def evaluate_ranker(
    model: LGBMRanker,
    X: np.ndarray | pd.DataFrame,
    y_excess: np.ndarray | pd.Series,
    group_dates: np.ndarray | pd.Series,
    top_k: int = 30,
) -> dict[str, float]:
    """Per-group rank IC + decile spread + hit rate + top-K Jaccard stability."""
    scores = model.predict(X)
    df = pd.DataFrame({
        'score': scores,
        'y_excess': np.asarray(y_excess, dtype=np.float64),
        'date': pd.to_datetime(group_dates),
    })

    # Per-date rank IC (Spearman).
    ics = (df.groupby('date')
           .apply(lambda g: g['score'].rank().corr(g['y_excess'].rank())))
    ics = ics.dropna()
    rank_ic_mean = float(ics.mean()) if len(ics) else float('nan')
    rank_ic_ir = float(ics.mean() / ics.std()) if len(ics) > 1 and ics.std() > 0 else float('nan')

    # Decile spread (top 10% mean - bottom 10% mean) across all pooled rows, in bps.
    df['decile'] = df.groupby('date')['score'].transform(
        lambda s: pd.qcut(s, 10, labels=False, duplicates='drop'))
    top = df[df['decile'] == df['decile'].max()]['y_excess'].mean()
    bot = df[df['decile'] == 0]['y_excess'].mean()
    decile_spread_bps = float((top - bot) * 1e4)

    # Hit rate: fraction of dates where top-K mean > bot-K mean.
    def _hit(g: pd.DataFrame) -> float:
        if len(g) < 2 * top_k:
            return float('nan')
        sorted_g = g.sort_values('score', ascending=False)
        return float(sorted_g.head(top_k)['y_excess'].mean() >
                     sorted_g.tail(top_k)['y_excess'].mean())
    hits = df.groupby('date').apply(_hit).dropna()
    hit_rate = float(hits.mean()) if len(hits) else float('nan')

    # Top-K Jaccard stability between consecutive dates' top-K sets.
    df_sorted = df.sort_values(['date', 'score'], ascending=[True, False])
    top_k_by_date = {d: set(g.head(top_k).index) for d, g in df_sorted.groupby('date')}
    dates_sorted = sorted(top_k_by_date)
    jaccards = []
    for d1, d2 in zip(dates_sorted, dates_sorted[1:]):
        s1, s2 = top_k_by_date[d1], top_k_by_date[d2]
        if s1 or s2:
            jaccards.append(len(s1 & s2) / len(s1 | s2))
    top_k_jaccard = float(np.mean(jaccards)) if jaccards else float('nan')

    return {
        'rank_ic_mean': rank_ic_mean,
        'rank_ic_ir': rank_ic_ir,
        'decile_spread_bps': decile_spread_bps,
        'hit_rate': hit_rate,
        'top_k_jaccard': top_k_jaccard,
    }
```

- [ ] **Step 4: Run tests, verify they pass**

`python -m pytest tests/utils/test_ranker.py -q` → all green.

- [ ] **Step 5: Commit**

```bash
git add -u
git commit -m "add build_ranker factory + evaluate_ranker (rank IC, decile spread, top-K Jaccard)"
```

---

### Task 5: Notebook cells A + B — setup + walk-1 assembly

**Files:**
- Create: `notebooks/06_supervised_ranker.ipynb`

- [ ] **Step 1: Create notebook with intro markdown + cell A (setup)**

Use the existing notebook helper (from earlier in the session — notebook 05 follows the same layout). Cells:

```markdown
# 06 — Supervised ranker (walk 1 MVP)

LightGBM `LGBMRanker(lambdarank)` over PCA text + structured + macro + aux text
features. Walk-1 only (train 2002–2007, val 2008, test 2009). Outputs land in
`artifacts/ranker/walk-001/`. Spec: `docs/superpowers/specs/2026-05-16-supervised-ranker-design.md`.
```

```python
# Cell A: Setup
from __future__ import annotations
import json, joblib
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
from lightgbm import early_stopping
import matplotlib.pyplot as plt

from src.utils.io import processed_dir, repo_root
from src.utils.ranker import (
    load_walk_pca, project_text_to_pca, friday_only,
    compute_excess_return_buckets, assemble_walk_features,
    build_ranker, evaluate_ranker, DEFAULT_RANKER_PARAMS,
)

WALK_ID = 1
TRAIN_START, TRAIN_END = '2002-01-01', '2007-12-31'
VAL_START, VAL_END     = '2008-01-01', '2008-12-31'
TEST_START, TEST_END   = '2009-01-01', '2009-12-31'

PANEL_DIR = processed_dir() / 'training_panel'
EMBED_DIR = processed_dir() / 'finbert_stockday_embed'
OUT_DIR   = repo_root() / 'artifacts' / 'ranker' / f'walk-{WALK_ID:03d}'
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_OPTUNA_TRIALS = 15
N_BUCKETS = 32
TOP_K = 30
RANDOM_STATE = 42
print(f'walk {WALK_ID}: train {TRAIN_START}..{TRAIN_END}, val {VAL_START}..{VAL_END}, test {TEST_START}..{TEST_END}')
print(f'out_dir: {OUT_DIR}')
```

- [ ] **Step 2: Add cell B (load + assemble train/val/test)**

```python
# Cell B: Load PCA + assemble train/val/test feature matrices
pca, n_pca = load_walk_pca(WALK_ID)
print(f'PCA loaded: n_components={n_pca}')

def _load_years(parquet_dir: Path, start: str, end: str, columns=None) -> pd.DataFrame:
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    years = list(range(s.year, e.year + 1))
    frames = []
    for y in years:
        for p in sorted(parquet_dir.glob(f'year={y}/*.parquet')):
            df = pd.read_parquet(p, columns=columns)
            df['date'] = pd.to_datetime(df['date'])
            df = df[(df['date'] >= s) & (df['date'] <= e)]
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def _project_for(window_start: str, window_end: str) -> pd.DataFrame:
    embed = _load_years(EMBED_DIR, window_start, window_end, columns=['permno', 'date', 'vec'])
    return project_text_to_pca(embed, pca)

panel_train = _load_years(PANEL_DIR, TRAIN_START, TRAIN_END)
panel_val   = _load_years(PANEL_DIR, VAL_START, VAL_END)
panel_test  = _load_years(PANEL_DIR, TEST_START, TEST_END)
print(f'panel rows: train={len(panel_train)}, val={len(panel_val)}, test={len(panel_test)}')

embed_train_pca = _project_for(TRAIN_START, TRAIN_END)
embed_val_pca   = _project_for(VAL_START, VAL_END)
embed_test_pca  = _project_for(TEST_START, TEST_END)

X_train, y_train_excess, groups_train, meta_train = assemble_walk_features(panel_train, embed_train_pca)
X_val,   y_val_excess,   groups_val,   meta_val   = assemble_walk_features(panel_val,   embed_val_pca)
X_test,  y_test_excess,  groups_test,  meta_test  = assemble_walk_features(panel_test,  embed_test_pca)

# lambdarank labels (32 buckets on the train/val targets).
buckets_train = compute_excess_return_buckets(meta_train.assign(date=meta_train['date']), n_buckets=N_BUCKETS)
buckets_val   = compute_excess_return_buckets(meta_val.assign(date=meta_val['date']),     n_buckets=N_BUCKETS)
print(f'feature matrix: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}')
print(f'groups: train={len(groups_train)} dates, val={len(groups_val)}, test={len(groups_test)}')
```

- [ ] **Step 3: Run cells A + B in the notebook**

Confirm prints look sane: PCA n_components=79, X_train has ~196 columns, groups_train ≈ 300 Friday dates over 6 years.

- [ ] **Step 4: Commit**

```bash
git add notebooks/06_supervised_ranker.ipynb
git commit -m "notebook 06: cells A (setup) + B (load + assemble walk-1)"
```

---

### Task 6: Notebook cell C — Optuna study

**Files:**
- Modify: `notebooks/06_supervised_ranker.ipynb`

- [ ] **Step 1: Add cell C (Optuna study)**

```python
# Cell C: Optuna hyperparameter search (15 trials, val NDCG@30, early stopping)
def objective(trial: optuna.Trial) -> float:
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 15, 127),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 200),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 5.0),
        'n_estimators': 500,
    }
    model = build_ranker(params)
    model.fit(
        X_train, buckets_train.astype(int).values,
        group=groups_train,
        eval_set=[(X_val, buckets_val.astype(int).values)],
        eval_group=[groups_val],
        eval_at=[TOP_K],
        callbacks=[early_stopping(stopping_rounds=30, verbose=False)],
    )
    # best_score_ -> {'valid_0': {'ndcg@30': x}}
    return float(model.best_score_['valid_0'][f'ndcg@{TOP_K}'])

study = optuna.create_study(
    direction='maximize',
    pruner=optuna.pruners.MedianPruner(),
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
)
study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)

print(f'best NDCG@{TOP_K}: {study.best_value:.4f}')
print(f'best params: {study.best_params}')
joblib.dump(study, OUT_DIR / 'optuna_study.pkl')
```

- [ ] **Step 2: Run cell C in the notebook**

If it completes with a positive best NDCG@30, proceed. Note approximate wall time.

- [ ] **Step 3: Commit**

```bash
git add notebooks/06_supervised_ranker.ipynb
git commit -m "notebook 06: cell C (Optuna study, 15 trials, val NDCG@30)"
```

---

### Task 7: Notebook cells D + E — final fit + OOS evaluation

**Files:**
- Modify: `notebooks/06_supervised_ranker.ipynb`

- [ ] **Step 1: Add cell D (final fit with early stopping)**

```python
# Cell D: Final fit with best HPs + early stopping on val
best_params = {**study.best_params, 'n_estimators': 2000}
model = build_ranker(best_params)
model.fit(
    X_train, buckets_train.astype(int).values,
    group=groups_train,
    eval_set=[(X_val, buckets_val.astype(int).values)],
    eval_group=[groups_val],
    eval_at=[TOP_K],
    callbacks=[early_stopping(stopping_rounds=50, verbose=True)],
)
best_iter = int(model.best_iteration_)
val_ndcg = float(model.best_score_['valid_0'][f'ndcg@{TOP_K}'])
print(f'best iteration: {best_iter}, val NDCG@{TOP_K}: {val_ndcg:.4f}')
```

- [ ] **Step 2: Add cell E (OOS evaluation + feature importance plot)**

```python
# Cell E: OOS test eval — rank IC, decile spread, hit rate, top-K Jaccard
test_metrics = evaluate_ranker(model, X_test, y_test_excess, meta_test['date'], top_k=TOP_K)
print('test metrics:')
for k, v in test_metrics.items():
    print(f'  {k}: {v:.4f}')

assert test_metrics['rank_ic_mean'] > 0, (
    f"sanity gate failed: test rank IC mean = {test_metrics['rank_ic_mean']:.4f} <= 0"
)

# Feature importance (gain).
fi = pd.DataFrame({
    'feature': X_train.columns,
    'gain': model.booster_.feature_importance(importance_type='gain'),
}).sort_values('gain', ascending=False)
print('top 20 features by gain:')
print(fi.head(20).to_string(index=False))

# Inline plot
fig, ax = plt.subplots(figsize=(8, 7))
fi.head(20).iloc[::-1].plot.barh(x='feature', y='gain', ax=ax, legend=False)
ax.set_title('Top 20 features by gain (walk 1)')
plt.tight_layout()
plt.show()
```

- [ ] **Step 3: Run cells D + E in the notebook**

If sanity gate fails (rank IC <= 0), STOP and ask for help. If positive, proceed.

- [ ] **Step 4: Commit**

```bash
git add notebooks/06_supervised_ranker.ipynb
git commit -m "notebook 06: cells D (final fit) + E (OOS eval + feature importance)"
```

---

### Task 8: Notebook cell F — persistence + summary

**Files:**
- Modify: `notebooks/06_supervised_ranker.ipynb`

- [ ] **Step 1: Add cell F (persist all artifacts)**

```python
# Cell F: Persist model + HPs + summary + feature importance
joblib.dump(
    {'model': model, 'feature_names': X_train.columns.tolist()},
    OUT_DIR / 'model.joblib',
)
(OUT_DIR / 'hp.json').write_text(json.dumps({
    **study.best_params,
    f'val_ndcg_at_{TOP_K}': val_ndcg,
    'best_iteration': best_iter,
}, indent=2))
fi.to_csv(OUT_DIR / 'feature_importance.csv', index=False)

summary = {
    'walk_id': WALK_ID,
    'train_window': [TRAIN_START, TRAIN_END],
    'val_window': [VAL_START, VAL_END],
    'test_window': [TEST_START, TEST_END],
    'n_features': X_train.shape[1],
    'n_pca': n_pca,
    'n_train_rows': len(X_train),
    'n_val_rows': len(X_val),
    'n_test_rows': len(X_test),
    'best_iteration': best_iter,
    f'val_ndcg_at_{TOP_K}': val_ndcg,
    **{f'test_{k}': v for k, v in test_metrics.items()},
    'passed_sanity': bool(test_metrics['rank_ic_mean'] > 0),
}
(OUT_DIR / 'summary.json').write_text(json.dumps(summary, indent=2))
print(f'wrote artifacts to {OUT_DIR.relative_to(repo_root())}')
print(json.dumps(summary, indent=2))
```

- [ ] **Step 2: Run cell F**

Verify all 5 files exist in `artifacts/ranker/walk-001/`.

- [ ] **Step 3: Commit**

```bash
git add notebooks/06_supervised_ranker.ipynb
git commit -m "notebook 06: cell F (persist model + hp.json + summary.json + feature importance)"
```

---

### Task 9: Run the full notebook end-to-end + commit summary

- [ ] **Step 1: Restart kernel, run all cells in order**

Check that everything completes without errors and the sanity gate passes.

- [ ] **Step 2: Inspect `artifacts/ranker/walk-001/summary.json`**

If `passed_sanity == true` and metrics look reasonable (rank IC mean roughly in [0.01, 0.10] range), we're good. If anomalous (e.g., rank IC > 0.5 — suspicious data leakage), stop and investigate.

- [ ] **Step 3: Commit the summary JSON**

```bash
git add artifacts/ranker/walk-001/summary.json
git commit -m "notebook 06: walk-1 summary (rank IC + NDCG@30 + decile spread)"
```

Note: `artifacts/` is already in `.gitignore`. Force-add the summary specifically so the audit trail is in git, without committing the binary `model.joblib` or `optuna_study.pkl`.

```bash
git add -f artifacts/ranker/walk-001/summary.json
```

---

### Task 10: Complete development branch

- [ ] **Step 1: Run full test suite**

```bash
python -m pytest tests/ -q
```

Expect all green (existing 83 + new ranker tests).

- [ ] **Step 2: Use finishing-a-development-branch skill**

Announce: "I'm using the finishing-a-development-branch skill to complete this work."

Follow the skill: verify tests pass, detect environment, present 4 options, execute user's choice.

---

## Self-review notes

- **Spec coverage:** §3 (inputs), §4 (features), §5 (target), §6 (training), §7 (outputs), §8 (preprocessing), §9 (diagnostics), §10 (validation gates), §11 (file layout) all map to tasks 1-9.
- **Placeholder scan:** no TBDs, no "implement similar to above" — every code step has full code.
- **Type consistency:** `assemble_walk_features` returns `(X, y, groups, meta)` consistently across tasks 3, 5, 6, 7. `evaluate_ranker` signature `(model, X, y_excess, group_dates, top_k)` consistent in tasks 4, 7.

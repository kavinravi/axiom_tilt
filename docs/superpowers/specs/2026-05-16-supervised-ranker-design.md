# Supervised Ranker (Notebook 06) — Design

**Date:** 2026-05-16
**Parent spec:** [`2026-05-08-text-enhanced-rl-portfolio-design.md`](2026-05-08-text-enhanced-rl-portfolio-design.md) §7
**Status:** Design

## 1. Goal

Train a `LGBMRanker(objective='lambdarank')` per walk that scores the cross-section of stocks every Friday, using PCA-projected text + structured + macro + auxiliary text features. Output: per-walk model artifact + diagnostics consumed by notebook 07 (RL agent).

MVP scope is **walk 1 only** (train 2002–2007, val 2008, test 2009). Walks 2–16 are deferred to a follow-up loop notebook (notebook 06b or extension of 06), unblocking RL work in parallel.

## 2. Conventions

This notebook follows the `machine-learning` skill conventions:
- **sklearn `Pipeline`** wrapping the ranker for reproducibility / serialization in one artifact.
- **Optuna study** persisted to disk (joblib) for resumability.
- **Early stopping** via LightGBM callback on val NDCG@30.
- **No tuning on test** — Optuna optimizes val NDCG@30; test is held out.
- **Model versioning** — write `model.joblib` + `hp.json` + `summary.json` together so a walk's full state can be reloaded.

## 3. Inputs

| Source | Path | Notes |
|---|---|---|
| Training panel | `data/processed/training_panel/year=YYYY/part-0.parquet` | Output of notebook 05; 2002–2024 |
| Stock-day text embeddings | `data/processed/finbert_stockday_embed/year=YYYY/*.parquet` | 768-dim CLS vec per (permno, date) |
| Walk-specific PCA | `artifacts/pca-text/walk-001/pca.joblib` | Fitted in notebook 04; `n_pca = 79` |

## 4. Feature schema

Per-row features fed to the ranker (after assembly):

| Block | Columns | Dim |
|---|---|---|
| PCA text | `pca_0..pca_{n-1}` | 79 (walk 1) |
| Structured (panel) | All numeric panel cols minus `permno, date, cik, ret, ticker, in_universe, fiscalperiod, datekey, calendardate, reportperiod, lastupdated, dimension` and `fwd_ret_*` | ~110 |
| Macro | `macro_vixcls, macro_dgs10, macro_dgs3mo, macro_t10y2y` | 4 |
| Aux text | `text_novelty, days_since_filing, doc_count_7d` | 3 |

Categorical handling: panel `ticker` is dropped (high-cardinality string with no predictive value beyond `permno` identity, which we deliberately omit to avoid memorization).

## 5. Target

Per Friday rebalance date `t`:
1. `excess_t = fwd_ret_5d_t - mean(fwd_ret_5d_t across that date's eligible universe)`
2. Cross-sectional percentile rank of `excess_t` within Friday `t`.
3. Bucket into 32 quantiles → integer label `0..31` for `lambdarank`.

Drop rows where `fwd_ret_5d` or `text_novelty` is NaN before bucketing. (LightGBM tolerates NaN in features, but the target and PCA inputs must be present.)

## 6. Training scheme

**Walk 1 (MVP):**
- Train: 2002-01-01 → 2007-12-31, Friday-filtered
- Val: 2008-01-01 → 2008-12-31, Friday-filtered (used for Optuna + early stopping)
- Test: 2009-01-01 → 2009-12-31, Friday-filtered (held out)

**Group**: each Friday date is one `lambdarank` group. `group_sizes` is a list of per-Friday counts in order.

**Optuna search space** (~15 trials, walk 1 only; freeze for walks 2-16):
- `num_leaves`: 15 to 127
- `learning_rate`: 0.01 to 0.1 (log scale)
- `min_data_in_leaf`: 20 to 200
- `feature_fraction`: 0.6 to 1.0
- `bagging_fraction`: 0.6 to 1.0
- `lambda_l2`: 0.0 to 5.0

Objective: NDCG@30 on val. Pruner: `MedianPruner`. Study persisted to `artifacts/ranker/walk-001/optuna_study.pkl`.

**Final fit** (after Optuna):
- `n_estimators = 2000`, `early_stopping_rounds = 50` on val NDCG@30.

## 7. Outputs

Per walk N, write to `artifacts/ranker/walk-{N:03d}/`:

| File | Content |
|---|---|
| `model.joblib` | Fitted `LGBMRanker` (no scaler — see §8) + feature-name list, persisted via `joblib.dump({'model': model, 'feature_names': cols})` |
| `hp.json` | Best Optuna params + best NDCG@30 on val |
| `summary.json` | Walk metadata, train/val/test sizes, OOS metrics |
| `feature_importance.csv` | LightGBM gain importance, sorted desc |
| `optuna_study.pkl` | Study object for inspection / resumption |

`summary.json` schema:
```json
{
  "walk_id": 1, "train_window": ["2002-01-01", "2007-12-31"],
  "val_window": ["2008-01-01", "2008-12-31"],
  "test_window": ["2009-01-01", "2009-12-31"],
  "n_features": 196, "n_pca": 79,
  "n_train_rows": ..., "n_val_rows": ..., "n_test_rows": ...,
  "best_iteration": 421,
  "val_ndcg_30": 0.xx, "test_ndcg_30": 0.xx,
  "test_rank_ic_mean": 0.xx, "test_rank_ic_ir": 0.xx,
  "test_decile_spread_bps": ..., "test_hit_rate": 0.xx,
  "passed_sanity": true
}
```

## 8. Preprocessing

LightGBM is scale-invariant, so no StandardScaler. Pipeline is essentially `LGBMRanker` wrapped for I/O consistency.

NaN policy:
- **Target / PCA features**: drop row before training. (PCA can't propagate NaN.)
- **Structured / macro / aux text**: pass through. LightGBM handles natively.

## 9. Diagnostics (MVP per parent §17.3)

On test set:
- **Rank IC**: per-Friday Spearman correlation of predicted score vs realized `excess_t`. Report mean + IR (mean / std).
- **Decile spread**: top-decile mean realized return − bottom-decile mean. In basis points.
- **Hit rate**: fraction of Fridays with positive top-30 minus bottom-30 spread.
- **Top-K Jaccard stability** between consecutive Fridays' top-30 sets (mean).
- **Feature importance**: LightGBM gain, dumped to CSV. Plot top 20 inline.

Deferred to v2 (per parent §17.3): SHAP, feature-importance drift across walks.

## 10. Validation gates

Per walk:
- `n_train_rows > 0` and `n_val_rows > 0`.
- Val NDCG@30 must improve over a baseline of `0.5` (random ranking gives ~0.5 on this scale; if below, model is broken).
- **Test rank IC mean > 0** — sanity check. If negative, abort and flag.

Failure = raise, partial outputs not persisted.

## 11. File / module structure

```
src/utils/ranker.py             # pure helpers (TDD'd)
tests/utils/test_ranker.py      # 6-8 unit tests
notebooks/06_supervised_ranker.ipynb
artifacts/ranker/walk-001/      # output dir (gitignored — large)
```

`src/utils/ranker.py` exports:
- `load_walk_pca(walk_id: int) -> tuple[PCA, int]`
- `project_text_to_pca(embed_df: pd.DataFrame, pca: PCA) -> pd.DataFrame` — returns `(permno, date, pca_0...pca_{n-1})`
- `assemble_walk_features(panel: pd.DataFrame, embed_pca: pd.DataFrame) -> pd.DataFrame`
- `friday_only(df: pd.DataFrame, date_col='date') -> pd.DataFrame`
- `compute_excess_return_buckets(df, ret_col='fwd_ret_5d', n_buckets=32) -> pd.Series` — returns int label aligned to df
- `build_ranker(params: dict) -> LGBMRanker`
- `evaluate_ranker(model, X_test, y_excess, group_dates) -> dict` — returns rank_ic_mean, rank_ic_ir, decile_spread_bps, hit_rate, top_k_jaccard

Notebook cells:
- A. Setup (paths, walk_id config)
- B. Load + assemble walk-1 features (train/val/test)
- C. Optuna study (15 trials, persist)
- D. Final fit with best HPs + early stopping
- E. OOS eval (rank IC, decile spread, Jaccard, feature importance plot)
- F. Persist all artifacts; print summary

## 12. Risks / mitigations

- **Memory**: full 2002-2007 Friday panel × ~196 features × ~300 permnos × ~52 weeks × 6 years ≈ 470k rows × 196 cols → ~700 MB float32. Fits.
- **Optuna time**: 15 trials × ~60s/trial (rough) ≈ 15 min. Cap each trial at `n_estimators=500` with early stopping to keep tight.
- **Joblib + Pipeline reload**: LightGBM versions can drift; pin in `requirements.txt` if not already.
- **PCA-feature dim drift across walks**: walks 2-16 may yield different `n_pca` if we don't lock. Parent spec §7.2 explicitly says lock after walk 1. Notebook 04 already enforces `SANITY_MAX_N_PCA=100` and the walk-1 `n_pca=79` becomes the locked value.

## 13. Out of scope

- Walks 2-16 (deferred to a loop after walk 1 validates).
- SHAP / feature-importance drift (v2).
- Eligibility filters (ADV, price, halts) — defer until backtest, where they matter for execution realism.

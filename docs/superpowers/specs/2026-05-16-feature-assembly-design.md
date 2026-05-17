# Feature Assembly (Notebook 05) — Design Spec

**Date:** 2026-05-16
**Status:** Draft, pending user review
**Repo:** `axiom_tilt`
**Branch:** `feature-assembly`
**Related spec:** `docs/superpowers/specs/2026-05-08-text-enhanced-rl-portfolio-design.md` §5–§7
**Predecessor:** Notebook 04 (per-walk PCA artifacts at `artifacts/pca-text/walk-001/...walk-016/`)

## 1. Goal

Produce a single daily-cadence training panel that joins the existing structured
PIT panel with newly-computed auxiliary text features and forward-return labels.
The output is the canonical input both for the supervised ranker (notebook 06,
which filters to Fridays) and for the RL agent's daily state (notebook 07+).

**Out of scope:** PCA-transformed text features are *not* materialized in the
panel — see §6.

## 2. Inputs

| Path | Purpose | Cadence |
|---|---|---|
| `data/processed/panel/year=*/*.parquet` | CRSP + Sharadar PIT join (returns, prices, fundamentals) | daily |
| `data/processed/macro.parquet` | Macro indicators (long format: `(date, series, value)`; series = `VIXCLS`, `DGS3MO`, `DGS10`, `T10Y2Y`) | daily, business days only |
| `data/processed/finbert_stockday_embed/year=*/part-permno-*.parquet` | 768-dim FinBERT vec per (permno, date), forward-filled | daily |
| `data/processed/edgar_index.parquet` | Per-filing metadata (cik, accession, filing_date, form_type) | event |
| `data/processed/universe_ids.parquet` | Permno × date intervals defining the eligible universe | interval |

## 3. Output

**Path:** `data/processed/training_panel/year=YYYY/part-0.parquet`

Daily cadence, partitioned by year. Atomic per-year write (`*.tmp → rename`),
resumable like notebook 02: re-run skips year-shards already on disk.

**Schema (one row per `(permno, date)`):**

| Column | Type | Source |
|---|---|---|
| `permno` | `int64` | panel |
| `date` | `timestamp[ns]` | panel |
| `<all structured columns>` | various | passed through from `panel/` (CRSP daily + Sharadar SF1 join, ~80 cols) |
| `macro_vix`, `macro_dgs3mo`, `macro_dgs10`, `macro_t10y2y` | `float64` | pivoted + ffilled + left-joined from `macro.parquet` on `date` (see §5) |
| `text_novelty` | `float32` | computed: cosine distance between `e_{i,t}` and `e_{i, t−7 calendar days}` |
| `days_since_filing` | `int32` | computed: days since most recent 10-K / 10-Q / 8-K from `edgar_index` |
| `doc_count_7d` | `int32` | computed: filings released in the last 7 calendar days for permno `i` |
| `fwd_ret_1d` | `float32` | computed: `ret` shifted -1 trading day per permno |
| `fwd_ret_5d` | `float32` | computed: product of next 5 trading days' `(1 + ret)` minus 1 per permno (≈ Friday-to-Friday for Friday rows) |

Universe-gated: only rows where `(permno, date)` falls inside a `universe_ids`
interval are kept.

## 4. Computation details

### 4.1 Structured + macro join

- Read each year-shard of `panel/`, filter to universe (per §4 of notebook 04's
  `assemble_training_matrix` logic), left-merge `macro.parquet` on `date`.
- No transformation beyond column passthrough — the ranker decides what to use.

### 4.2 Aux text features

**`text_novelty`** — for each `(permno, date)`:
- Look up `e_{i,t}` in `finbert_stockday_embed`.
- Look up `e_{i, t−7d}` for the same permno (where `t−7d` is calendar days).
  Since `stockday_embed` is forward-filled to daily, a vector exists for any
  active permno-date.
- Compute `1 − cosine_similarity(e_{i,t}, e_{i, t−7d})`.
- First 7 days for any permno (no t−7d available): write `NaN`.

**`days_since_filing`** — for each `(permno, date)`:
- Join `edgar_index` to permnos via `universe_ids.parquet` (which carries both
  `cik` and `permno` per row). For each `(permno, date)`, find max
  `filing_date ≤ date` among `form_type ∈ {10-K, 10-Q, 8-K}`.
- Compute `(date − max_filing_date).days`; if no prior filing, write `NaN`.

**`doc_count_7d`** — for each `(permno, date)`:
- Count filings for that permno with `filing_date ∈ [date − 7d, date]`.
- All form types (not just K/Q/8K). `int32`.

Computation strategy: per-permno groupby with vectorized lag / window operations
where possible; fall back to per-year shard loops for the cosine that needs the
prior embedding lookup (avoids loading the full 3 GB embed panel at once).

### 4.3 Labels

- `fwd_ret_1d`: `groupby(permno)['ret'].shift(-1)`.
- `fwd_ret_5d`: `groupby(permno)['ret'].rolling(5, min_periods=5).apply(lambda r: (1+r).prod() - 1).shift(-5)` — or equivalent vectorized form using log-returns + 5-period rolling sum + expm1.

Trading-day cadence (gaps for weekends/holidays are honored by `groupby(permno).shift(-k)`).

**Edge:** delisted permnos produce `NaN` at the tail of their date range (last 5
days of activity have no fwd_ret_5d). Kept as `NaN`; downstream ranker drops
NaN-label rows before training.

## 5. Macro features

`macro.parquet` is in long format `(date, series, value)` with four FRED series:
`VIXCLS` (VIX close), `DGS3MO` (3-month treasury yield), `DGS10` (10-year
treasury yield), `T10Y2Y` (10y–2y spread). Step 1 in the macro join: pivot to
wide so each series becomes a column. Step 2: forward-fill across non-business
days (FRED publishes on business days; weekend/holiday rows in the panel get
the prior trading day's value). Step 3: left-join to the panel on `date`.

The four series add `macro_vix`, `macro_dgs3mo`, `macro_dgs10`, `macro_t10y2y`
to the output schema.

## 6. Why PCA text features are NOT in the panel

Per spec §7.2 ("Re-fit PCA at each walk boundary"), each of the 16 walks has
its own `pca.joblib`. Pre-applying PCA in the panel would mean either:
- Storing 16 versions of the text columns (`pca_w01_*` ... `pca_w16_*`), or
- Writing 16 separate panel files (`training_panel/walk=NNN/...`).

Both add storage and complexity for no compute gain — the PCA `.transform()` is
a single matrix multiply. Instead, the ranker notebook (06) does:
```python
panel = read_training_panel(walk_train_start, walk_train_end)
embed = read_stockday_embed(walk_train_start, walk_train_end)
pca = joblib.load(f'artifacts/pca-text/walk-{walk_id:03d}/pca.joblib')
panel = panel.merge(embed, on=['permno', 'date'])
panel[[f'pca_{i}' for i in range(n_pca)]] = pca.transform(np.stack(panel['vec']))
```

Cost: one extra join + matrix-multiply per walk. Benefit: notebook 05's output
is a single daily artifact, the RL agent (which doesn't use PCA text) shares
the same panel.

## 7. Notebook structure

| Cell | Role |
|---|---|
| A | Setup, paths, year range derived from `panel/` shards |
| B | Load `panel/` + `macro.parquet` for one year, universe-filter, retain base columns |
| C | Compute aux text features: `text_novelty` (lagged embed lookup), `days_since_filing`, `doc_count_7d` (edgar_index window scan) |
| D | Compute labels: `fwd_ret_1d`, `fwd_ret_5d` via per-permno shift / rolling |
| E | Validation: row count vs `panel/` baseline, NaN audit per column, sample (permno, date) lookup round-trip |
| F | Persist `training_panel/year=YYYY/part-0.parquet` atomically, write summary to `reports/metrics/feature_assembly_summary.json`. Re-run is resumable per-year. |

Cells B–F are wrapped in a `for year in YEARS` loop. Heavy reads (`finbert_stockday_embed`,
`edgar_index`) are loaded once outside the loop and queried per year.

## 8. Validation gates (cell E)

- **Row count:** total panel rows ≥ universe-filtered baseline from `panel/`
  (no rows lost in aux-feature joins; left-joins, not inner).
- **NaN audit:** `text_novelty` NaN rate ≤ 5% (first-7-days of each permno is
  expected). `days_since_filing` NaN rate ≤ 10% (some permnos have no filings
  in the corpus — IPOs near end of range, etc.). Labels NaN at edges only.
- **Schema:** column count + dtypes match the spec table above.
- **Round-trip:** read back one year, query a random `(permno, friday)` row,
  confirm all columns populated and `fwd_ret_5d ≈ (1 + ret[t+1])·...·(1 + ret[t+5]) − 1`.

If any gate fails → raise (notebook stops, no panel persisted for that year).

## 9. Resumability + atomicity

- Per-year write to `*.tmp`, rename on success. Existing `part-0.parquet` for a
  year → skip (`process_year(year)` checks first).
- To re-do a year: `rm data/processed/training_panel/year=YYYY/part-0.parquet`.
- Validation gates fire per-year, so a partial run leaves earlier years
  on-disk and aborts cleanly on the bad year.

## 10. Downstream contracts

**Ranker (notebook 06):**
1. Load `training_panel` for `[walk_train_start, walk_train_end]`.
2. Filter `df = df[df.date.dt.dayofweek == 4]` (Friday).
3. Drop rows where `fwd_ret_5d.isna()`.
4. Join `finbert_stockday_embed` on `(permno, date)`.
5. Load walk's `pca.joblib`, apply `.transform()`.
6. Train LightGBM on structured + macro + aux text + PCA text → `fwd_ret_5d`.

**RL agent (notebook 07+):**
- Same `training_panel`, daily cadence (no Friday filter).
- State vector = structured + macro + aux text + top-K mask from ranker scores.

## 11. Out of scope

- PCA text materialization (see §6).
- Feature engineering beyond the spec's aux text features (no rolling vol,
  no momentum signals — those can be added by the ranker as features it
  computes on the fly, or in a v2 of this notebook).
- Test/validate split — the panel is daily everywhere; the ranker / RL stages
  enforce walk-forward splits.
- Per-walk panels.

## 12. Risks / open items

- Memory: `finbert_stockday_embed` is 3 GB on disk. Loading the full panel into
  pandas for the cosine lookup may be borderline. Per-year loop with shard-level
  filtering keeps peak RAM bounded; if it bites, switch to per-permno groupby.

# Text-Enhanced RL Portfolio Allocation — Design Spec

**Date:** 2026-05-08
**Status:** Active design — data layer implemented (CRSP + Sharadar + panel); FinBERT FT in progress; ranker/RL/backtest not yet built
**Repo:** `axiom_tilt`

## 1. Research Question

> Can fine-tuned financial-text representations improve **stock selection** enough that an RL agent can convert them into better after-cost portfolio performance than non-text and non-RL baselines?

The project answers two sub-questions:
1. Does financial text improve cross-sectional ranking of a wide equity universe?
2. Does an RL agent improve dynamic allocation among top-ranked candidates relative to non-RL baselines once transaction costs and path dependence matter?

## 2. Architecture (Option A — text in ranker, indirect RL exposure)

```
                                    ┌──────────────────────────┐
financial docs ─► FinBERT ─► CLS ──►│ PCA + agg over docs      │──┐
                                    └──────────────────────────┘  │
                                              │                   │
                                              ▼                   ▼
                                    novelty / recency / count    ranker text features
                                              │                   │
                                              │           ┌───────┴────────────────┐
                                              │           │ structured features    │
                                              │           │ (momentum, vol, ...)   │
                                              │           └───────┬────────────────┘
                                              │                   ▼
                                              │           ┌──────────────┐
                                              │           │ LightGBM     │
                                              │           │ Ranker       │
                                              │           │ (lambdarank) │
                                              │           └──────┬───────┘
                                              │                  │ score per stock
                                              │                  ▼
                                              │           top-K selection
                                              │                  │
                                              ▼                  ▼
                                    ┌─────────────────────────────────────┐
                                    │ RL state: portfolio + ranker scores │
                                    │ + structured features for K names   │
                                    │ + 3 text-derived scalars per name   │
                                    └────────────────┬────────────────────┘
                                                     ▼
                                            RL policy → target weights
                                                     │
                                                     ▼
                                            constraint layer (projection)
                                                     │
                                                     ▼
                                            execution + after-cost reward
```

Text reaches RL only through:
- top-K membership (which names are in the candidate set)
- ranker scores per candidate
- 3 lightweight scalars: text novelty, days-since-filing, doc-count this period

Raw 768-dim CLS embeddings never enter RL state.

## 3. Data

### 3.1 Universe and timing

- **Universe:** ~400 liquid US equities. S&P 500–style point-in-time membership with survivorship awareness (CRSP/Compustat-style, or reconstructed from index history).
- **Cadence:** Daily observation; **weekly rebalance** (RL action emitted weekly). Ranker scores recompute daily.
- **Holding horizon:** One week, with rolling state continuity.
- **Backtest:** Walk-forward expanding-window, weekly rebalance, realistic execution lag (next-day open after decision).
- **Period:** ~2000-01-01 to 2025-12-31 (~25 years). Constrained by financial text availability.

### 3.2 Text sources (priority order)

1. **SEC EDGAR full-text** — 10-K, 10-Q, 8-K. Free, bulk download. ~25 years of clean coverage. Primary corpus.
2. **Earnings call transcripts** — quarterly. Coverage from ~2002 via reasonable scraping or paid feed (Refinitiv/FactSet if available; otherwise SeekingAlpha-style with care).
3. **Curated financial news** — optional, lower priority. HF datasets (`ashraq/financial-news`, `zeroshot/twitter-financial-news-sentiment`) for breadth.

For FinBERT MLM continued pretraining, EDGAR alone gives billions of tokens — sufficient.

### 3.3 Price and fundamentals

*(Updated 2026-05-13 — see `docs/superpowers/specs/2026-05-13-wrds-ingestion-design.md` for the full data-layer history. yfinance and FMP were both tried and dropped.)*

- **Prices:** CRSP daily via Wharton WRDS — `src/data/ingest_wrds.py`. Prices, returns, volume, shares, OHLC, split/dividend factors, plus delisting returns from `crsp.msedelist`. Covers the full survivorship-bias-free universe (live + delisted), 1995–2024.
- **Fundamentals:** Sharadar SF1 (Core US Fundamentals) via Nasdaq Data Link — `src/data/ingest_sharadar.py`. Point-in-time via `datekey` (filing date), As-Reported dimensions only (ARQ/ARY — no restated values), ~1993 onward so the 2008 GFC is covered. WRDS Compustat was inaccessible at the school's subscription tier; SEC XBRL was tried but only covers ~2009+ (no GFC) — both abandoned.
- **Risk-free rate / macro:** FRED via `src/data/ingest_macro.py`.

**Unified panel:** `src/data/build_panel.py` materializes `data/processed/panel/` — CRSP daily left-joined to Sharadar SF1 via a backward `merge_asof` (`datekey <= date`) with an explicit leakage guard. Permnos with no Sharadar coverage (~245 of 826, mostly old delisted names) are struck — an accepted partial survivorship-bias trade-off, since FMP was rejected for look-ahead/survivorship issues.

### 3.4 Directory layout

```
data/
  raw/
    edgar/         # downloaded 10-K, 10-Q, 8-K (gitignored)
    sec/           # company_tickers.json — universe build input
  interim/
    edgar_text/    # cleaned + deduped filings (gitignored)
  processed/
    crsp_daily/    # CRSP daily prices, year-partitioned (gitignored)
    sharadar_sf1.parquet, sharadar_tickers.parquet  # fundamentals (gitignored)
    panel/         # unified PIT panel, year-partitioned (gitignored)
    finbert_tok/   # tokenized chunks (gitignored)
    universe.parquet, universe_ids.parquet, macro.parquet, edgar_index.parquet
  embeddings/      # stock-day text vectors (post FinBERT + PCA + aggregation)
```

Small `data/processed/` artifacts stay tracked for laptop sync; large derivatives
(`crsp_daily/`, `panel/`, `finbert_tok/`, Sharadar parquets, raw corpora) are
gitignored and shared via Cloudflare R2 — see the WRDS ingestion spec.

## 4. FinBERT Fine-Tuning

### 4.1 Objective

Continued **MLM (masked language model) pretraining** on a large unlabeled financial-text corpus. No supervised head. Output: a domain-adapted encoder whose `[CLS]` (or pooled mean) representation captures financial document semantics.

### 4.2 Base model

`yiyanghkust/finbert-pretrain` if available (this is the MLM-only base, not the sentiment variant). Fallback: `bert-base-uncased` and treat the entire fine-tune as pure domain adaptation.

### 4.3 Training corpus

- EDGAR filings (10-K, 10-Q, 8-K), 2000–2024
- Earnings call transcripts where obtainable
- Optional: financial news sample
- Total target: 1B+ tokens

Documents are chunked to 512 tokens with 64-token stride; sentences are not artificially split to keep semantic coherence.

### 4.4 Training config (target Blackwell 5090, 32GB)

| Setting | Value |
|---|---|
| Precision | bf16 |
| Batch size | 64 sequences × 512 tokens (gradient accumulation if needed) |
| Optimizer | AdamW, lr 5e-5, weight decay 0.01 |
| Schedule | Linear warmup 6%, then cosine decay |
| MLM masking | 15% probability, 80/10/10 split |
| Epochs | 1–3 over corpus (validate by held-out perplexity) |
| Eval | Held-out perplexity on 5% of corpus (non-overlapping documents) |
| Checkpointing | Every 5000 steps, keep best on val perplexity |

Single-GPU with `accelerate launch` and `transformers` `Trainer`. No LoRA — full fine-tune.

### 4.5 Output artifacts

- Fine-tuned encoder weights (saved to `artifacts/finbert-mlm/`)
- Tokenizer (kept identical to base)
- Eval log: train/val perplexity curves

## 5. Text Feature Engineering

### 5.1 Document embeddings

For each document `d`:
- Tokenize and chunk (512 tokens, 64 stride)
- Run encoder, take `[CLS]` of each chunk
- **Document embedding** = mean-pool chunk CLS vectors → 768-dim

### 5.2 Stock-day aggregation

For stock `i` on day `t`:
- Collect all documents `D(i, t)` with publication date ≤ `t` and within rolling window of 30 days
- **Aggregated embedding** `e_{i,t}` = exponentially-decay-weighted mean of document embeddings, decay half-life 14 days
- Stock-days with no recent documents inherit the previous day's embedding (forward-fill)

### 5.3 Dimensionality reduction

PCA dim is chosen by **cumulative-variance threshold**, not hardcoded.

**Procedure (run once on the first walk's training window):**
1. Default target: **99%** explained variance (sensitivity range: 95% / 98% / 99%).
2. Fit a full PCA on aggregated stock-day embeddings from the first walk's training window only (no leakage).
3. Find the smallest `n` such that cumulative explained variance ≥ target.
4. Set the production PCA dim to **`n + 1`** (one-component safety buffer).
5. Lock this `n_pca` for all subsequent walk-forward windows.

**Walk-forward behavior:**
- The dim `n_pca` is fixed after the first calibration so the ranker's input schema is stable across walks.
- The PCA *components* (the actual fit) are re-estimated at each walk-forward boundary using only that window's training data.
- This keeps the schema constant while still adapting the projection to new training data.

**Diagnostic outputs to record:**
- Cumulative explained variance curve from the first-walk fit
- Chosen `n_pca` and the variance it captures
- Per-walk: variance captured by the locked `n_pca` (sanity check that 99% target still holds in later windows)

**Sanity check before locking:** if the chosen `n_pca` is so large it defeats the purpose of dim reduction (e.g., ≥ ~200 of 768), reconsider — review the scree / cum-var plot, lower the target (98% or 95%), or skip PCA and feed CLS through a different reducer. PCA only helps if the variance is genuinely concentrated.

### 5.4 Auxiliary text features (RL state inputs, also available to ranker)

For each candidate `i` on rebalance date `t`:
- **`text_novelty_{i,t}`**: cosine distance between `e_{i,t}` and `e_{i, t-7d}`. Captures regime change in the company's narrative.
- **`days_since_filing_{i,t}`**: integer days since the most recent 10-K/Q/8-K release.
- **`doc_count_{i,t}`**: number of documents released in last 7 calendar days.

These are 3 scalars per name, computed once per rebalance.

## 6. Structured Features

Per stock-day, computed from prices/fundamentals:
- **Momentum:** 1m, 3m, 6m, 12m return (skip-1m for 12m)
- **Reversal:** 1-week return
- **Realized volatility:** 21-day, 63-day
- **Beta:** 252-day rolling vs market
- **Liquidity:** ADV (21-day average dollar volume), Amihud illiquidity
- **Valuation:** P/E, P/B, EV/EBITDA, dividend yield
- **Profitability:** ROA, ROE, gross margin
- **Leverage:** Debt-to-equity, interest coverage
- **Sector:** GICS one-hot or target-encoded

All features standardized cross-sectionally per day (z-score within universe). Missingness handled by feature-specific defaults (zero for z-scores, sector medians for fundamentals).

## 7. Supervised Ranker

### 7.1 Model

**LightGBM `LGBMRanker`** with `lambdarank` objective.

Inputs:
- `n_pca` PCA text features (dim chosen via §5.3, locked after first walk)
- ~25 structured features
- 3 auxiliary text features (novelty, recency, doc-count)

Target: cross-sectional rank of next-week excess return (vs equal-weighted universe), bucketed into 32 quantiles for `lambdarank`.

Group: each (rebalance-date) is a `lambdarank` group.

### 7.2 Training scheme

- **Walk-forward expanding window.** Initial train: 2002–2007. Validate: 2008. Test: 2009. Step forward 1 year at a time.
- Hyperparameter tuning on a fixed validation period (2008) at the start; freeze HPs for all subsequent walks.
- Re-fit PCA, FinBERT (optional re-FT), and ranker at each walk boundary.

### 7.3 Diagnostics

- Rank IC (per-period Spearman correlation of predicted vs realized returns)
- Decile spread (top decile minus bottom decile mean return)
- Top-K selection stability (Jaccard between consecutive weeks' top-K sets)
- Feature importance (gain-based and SHAP)

## 8. Top-K Selection

- **K = 30** as default.
- Each rebalance date, take the 30 names with the highest ranker scores from the eligible universe.
- Eligibility filters: minimum ADV (e.g., $5M), minimum price ($5), no halts/suspensions in last 5 days.

## 9. RL Portfolio Control

### 9.1 State

For each rebalance date `t`, observation tensor concatenates:

| Block | Dim | Source |
|---|---|---|
| Current portfolio weights over top-K | K=30 | engine |
| Ranker scores for top-K | 30 | ranker |
| Structured features for top-K | 30 × ~25 | feature pipeline |
| Auxiliary text features for top-K | 30 × 3 | feature pipeline |
| Recent portfolio return (1w, 4w, 12w) | 3 | engine |
| Recent turnover (1w, 4w) | 2 | engine |
| Volatility / regime indicators | ~5 | market data |
| Estimated transaction-cost level | 1 | cost model |

Total: roughly 880-dim state vector. Names are sorted by ranker score for permutation invariance.

### 9.2 Action

**Continuous action: target weights over top-K names** (default), output dim = K.

Actions are projected onto the long-only simplex via softmax + max-weight clip + renormalization.

(Trade-delta actions are an extension, not part of MVP.)

### 9.3 Reward

```
reward_t = portfolio_return_t
         - λ_cost · trading_cost_t
         - λ_vol  · realized_vol_penalty_t
         - λ_dd   · drawdown_penalty_t
```

Defaults: `λ_cost = 1.0`, `λ_vol = 0.1`, `λ_dd = 0.1`. Tune via ablation.

### 9.4 Algorithm

**PPO** (`stable-baselines3`) for online, on-policy training. Rationale: well-understood, robust to hyperparameters, native continuous-action support, good for the moderate state dimension.

**Synthetic episodes via block bootstrap** to address RL data scarcity *(v2; see §16 for MVP scope)*:
- Resample weekly trajectories using stationary block bootstrap (mean block length 8 weeks)
- Generate 1000+ synthetic trajectories per training window
- Train PPO over the synthetic distribution; evaluate on the real walk-forward sequence

**Behavior cloning warm-start** *(v2)*: initialize policy from equal-weight-on-top-K behavior for 100 steps before PPO begins, to avoid early random-policy disasters.

For MVP: PPO trains on real walk-forward trajectories only, with random restart of episode start dates within each training window for some trajectory diversity.

### 9.5 Eval

PPO policy evaluated on real (non-bootstrapped) walk-forward windows. Train on bootstrap → evaluate on real history.

## 10. Constraint Layer

After RL outputs raw action `a_t`:

1. Apply softmax for long-only normalization
2. Clip per-name weights at `w_max = 0.10`
3. Renormalize to sum to 1
4. Apply turnover cap: if `||w_t - w_{t-1}||_1 > τ_max`, scale move toward `w_{t-1}`. Default `τ_max = 0.30`.
5. Apply ADV-based liquidity cap: each trade ≤ 5% of name's 21-day ADV.

The constraint layer is enforcement only; it does not re-optimize. RL retains decision authority over feasible region.

## 11. Backtest Engine

- Weekly rebalance, Friday close decision → Monday open execution
- **Transaction costs:** linear `5 bps` per side default; sensitivity analysis at 2/5/10/20 bps
- Survivorship-aware (delistings → forced liquidation at last available price)
- Mark-to-market daily for path metrics; rebalance only weekly

### 11.1 Metrics

- Annualized return, vol, Sharpe, Sortino
- Max drawdown
- Calmar
- Turnover (annualized)
- Average holdings count
- Cost-adjusted return (gross vs net)
- Hit rate (weekly)

## 12. Baselines (required)

| Baseline | Description |
|---|---|
| EW-Universe | Equal-weight all ~400 names, weekly rebalance |
| EW-TopK | Equal-weight ranker top-30 |
| Ranker + Static | Top-30 with momentum-tilt static weights (e.g., score-proportional) |
| Ranker + MV | Top-30 with mean-variance optimizer (constrained, weekly re-est) |
| No-Text Ranker + RL | Drop all PCA + 3 aux text features from ranker; rest identical |
| Full Pipeline | Text + ranker + RL (this project) |

## 13. Ablations

- **Text source:** EDGAR-only vs EDGAR + transcripts
- **FinBERT FT:** pretrained vanilla vs continued-MLM-fine-tuned
- **Text-feature variant:** PCA (locked dim) vs (PCA + MLP-head scalar) vs (no text features)
- **PCA variance target:** 95% vs 98% vs 99% (changes locked `n_pca`)
- **Aux text features:** which of {novelty, recency, count} matter — drop one at a time
- **K:** 20, 30, 40
- **Cost level:** 2, 5, 10, 20 bps
- **Reward shaping:** with/without `λ_vol`, `λ_dd`
- **Bootstrap:** RL trained on real-only vs real + synthetic
- **Action space:** target weights vs trade deltas (extension)

## 14. Repo Layout

```
src/
  data/         # ingestion (EDGAR, prices, fundamentals)
  text/         # cleaning, chunking, FinBERT FT, embedding generation
  features/     # structured features, PCA, aux text features, panel align
  models/       # supervised ranker (LightGBM)
  policy/       # RL env, PPO training, bootstrap sampler
  portfolio/    # constraint layer, action projection
  backtest/     # simulation engine, metrics, attribution, plots
  utils/        # config loading, logging, CV splits
configs/        # YAML per component (data, model_text, model_tabular, policy, ...)
notebooks/      # 01-data audit → 06-result viz
tests/          # alignment, no-lookahead, candidate selection, projection, backtest
data/           # raw, interim, processed, embeddings (tracked)
artifacts/      # model checkpoints (gitignored)
docs/           # design specs, reports
```

## 15. Tech Stack

- **Python:** 3.11
- **GPU target:** NVIDIA RTX 5090 (Blackwell, sm_120)
- **PyTorch:** ≥2.7.0 with cu128 wheels
- **Hugging Face:** `transformers` ≥4.46, `tokenizers` ≥0.20, `accelerate` ≥1.0, `datasets` ≥3.0
- **Tabular ML:** `lightgbm` ≥4.5, `xgboost` ≥2.1, `scikit-learn` ≥1.5
- **RL:** `stable-baselines3` ≥2.4, `gymnasium` ≥1.0, `d3rlpy` ≥2.6 (offline RL extension)
- **Optimization:** `cvxpy` ≥1.5, `scipy` ≥1.13
- **Backtest metrics:** `empyrical` ≥0.5.5
- **Plotting:** `matplotlib`, `plotly`

## 16. MVP Scope

**First end-to-end pass should include:**
1. EDGAR ingestion + price data
2. FinBERT MLM continued pretraining on EDGAR
3. PCA of aggregated stock-day embeddings (dim chosen via cum-var threshold, see §5.3)
4. 3 auxiliary text features
5. Structured features (the full ~25)
6. LightGBM ranker, single walk-forward window
7. K=30 top-K selection
8. PPO RL on real-only trajectories (no bootstrap yet)
9. Linear-cost backtest at 5 bps
10. Baselines: EW-Universe, EW-TopK, Full Pipeline
11. MVP-tier logging per §17 (W&B + parquet mirror) wired through every stage from day one

**Deferred to v2:**
- Earnings call transcripts
- Block-bootstrap synthetic episodes
- BC warm-start
- Trade-delta action space
- Reward-shaping ablations
- Full multi-cost-level sensitivity

## 17. Logging & Diagnostics

**Goal:** capture enough per-walk metrics that the project can later support a research paper — not just final backtest numbers, but how each stage behaves over time, where data drifts, and where stability breaks down.

**Tooling:**
- **Primary:** Weights & Biases (`wandb`). One run per `{stage, walk_id}`, grouped by experiment name. Wandb is already gitignored in this repo.
- **Mirror dump:** all numeric metrics also written to `reports/metrics/{stage}_{walk}.parquet` for paper-ready plotting independent of the tracker (and so a future "I lost my W&B account" doesn't lose data).
- MLflow is acceptable as a fallback if the user later prefers it; pick one and stick with it.

### 17.1 FinBERT MLM
- Per epoch: train loss, validation perplexity, learning rate, gradient norm
- Final model perplexity and delta vs base-model perplexity on a held-out financial-text sample
- (If FinBERT is re-FT'd at each walk:) final perplexity per walk and delta vs prior walk

### 17.2 PCA / text features
- **Cumulative-variance curve** (all 768 components) per walk — the raw signal you flagged interest in
- Variance captured by the locked `n_pca` per walk (sanity check that the 99% target still holds in later windows)
- Top-N PCA loadings per walk (qualitative interpretability — what dimensions dominate)
- **Subspace stability:** cosine similarity between top-`k` PCA components of walk `w` vs walk `1` (drift indicator — is the "topic geometry" of financial text changing over time?)
- Distribution stats of aux text features (novelty / recency / doc-count) per walk

### 17.3 LightGBM ranker
- Training: lambdarank loss curve, NDCG@K on validation
- OOS per walk: **rank IC**, IC mean, IC information ratio, decile spread, hit rate, top-K Jaccard stability between consecutive weeks
- **Feature importance per walk** (both gain-based and SHAP) — track top-N ranking over time
- Calibration plot: predicted-score quantile buckets vs realized return
- Best HPs from initial tune (logged once, reused across walks)

### 17.4 RL (PPO)
- Episode-level: raw return, after-cost return, reward decomposed into components (return, cost, vol penalty, dd penalty)
- Policy stats: entropy, KL divergence vs previous policy, value-function loss, gradient norm
- Action stats: max weight, weight-concentration Herfindahl index, per-step turnover, holdings count
- Per walk: final-policy snapshot — median weights over the walk, sector exposures, weight correlation across rebalance dates

### 17.5 Backtest
- Per-walk OOS: annualized return, vol, Sharpe, Sortino, Calmar, max drawdown, turnover, average holdings count
- Gross vs net return decomposition (cost attribution)
- Equity curve per walk and concatenated full-period
- Weekly hit rate, winning-weeks ratio

### 17.6 Regime / data-drift (paper-ready context)
- Universe size per walk
- Document volume per walk (filings + transcripts) — text-data density over time
- Market-regime indicators per walk: VIX mean, term spread, market return, dispersion
- Cross-sectional stock-return vol per walk
- **Cross-walk stability composites:** PCA subspace cosine, ranker feature-importance Spearman, RL policy weight correlation

### MVP vs v2 split for logging

**MVP (in v1):**
- §17.1 in full
- §17.2 cum-var curve + locked-`n_pca` variance check (skip subspace stability)
- §17.3 loss + IC + decile spread + hit rate (skip SHAP and importance drift)
- §17.4 reward components + entropy + turnover (skip policy snapshots)
- §17.5 in full

**v2 (research-paper extras):**
- §17.2 subspace stability
- §17.3 SHAP and feature-importance drift over walks
- §17.4 policy snapshots
- §17.6 regime indicators and cross-walk stability composites

This split keeps the MVP runnable without spending a week on instrumentation, while reserving the rich research-paper diagnostics for a clearly bounded second pass.

## 18. Defaults Marked for User Review

These are sensible defaults but not load-bearing — user should flag if they want different values:

- Universe: ~400 names, S&P 500 reconstructed
- Period: 2000–2025
- K = 30
- Rebalance: weekly (Friday close → Monday open)
- Linear cost: 5 bps
- Max weight: 10%, turnover cap: 30%
- PPO as RL algo
- PCA variance target: 99% (then `n_pca = n + 1`, locked after first walk; see §5.3). Alternative considered: re-pick `n_pca` per walk — rejected for schema-stability reasons but available as an ablation if the user wants it.
- Text aggregation half-life: 14 days
- Reward weights: cost 1.0, vol 0.1, dd 0.1

## 19. Open Decisions

None blocking the implementation plan. The above is a complete spec; ablations and v2 work expand it.

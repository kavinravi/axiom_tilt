# axiom_tilt — Continuation TODO

> **Handoff for fresh Claude session.** Read this first to ramp without the prior transcript.

## Project at a glance

`axiom_tilt` is a research project: text-enhanced RL portfolio allocation. Two-stage architecture — (1) FinBERT + LightGBM ranker over ~400 S&P 500 names produces top-K each week, (2) PPO RL agent allocates among the top-K under transaction costs and constraints.

**Source of truth documents:**
- Design spec: `docs/superpowers/specs/2026-05-08-text-enhanced-rl-portfolio-design.md`
- Data-ingestion plan: `docs/superpowers/plans/2026-05-09-data-ingestion.md`
- README: `README.md`

**Current branch:** `data-ingestion` (NOT yet merged to `main`). Pushed to GitHub.

## Status of the 7 data-ingestion tasks

| # | Task | Status | Notes |
|---|---|---|---|
| 0 | Foundation (pyproject, env loader, paths, logging, rate limiter, config) | ✅ done | 8 unit tests passing |
| 1 | Universe reconstruction (Wikipedia S&P 500 + SEC ticker→CIK) | ✅ done | 884 intervals, 858 tickers, 73% CIK match. `data/processed/universe.parquet` |
| 2 | Prices via yfinance (daily OHLCV adjusted) | ✅ done | 3.6M rows, 670 tickers, 2000–2025. `data/processed/prices.parquet` |
| 3 | Fundamentals via FMP | ⛔ **deferred** | FMP deprecated v3 endpoints on 2025-08-31. See "Action items" below |
| 4 | EDGAR text (10-K/10-Q/8-K) | ✅ done | 226,919 filings, 614 CIKs, 746 GB raw on disk. `data/processed/edgar_index.parquet`. Finished 2026-05-10 08:39 in ~8 hrs |
| 5 | Macro via FRED (DGS3MO, DGS10, VIXCLS, T10Y2Y) | ✅ done | 26K rows × 4 series. `data/processed/macro.parquet` |
| 6 | Cross-coverage integration audit | ✅ done | See `docs/audits/2026-05-10-data-coverage.md`. EDGAR ≥90% PASS; **prices 78% BELOW 95% threshold** (yfinance gap on delisted names → survivorship bias) |

## Currently running

Nothing. EDGAR pull completed at 2026-05-10 08:39:00 (started ~00:40, ~8 hours total). Final log line: `Wrote edgar_index with 226919 rows`. See historical context below if launching a re-pull is ever needed.

### Historical: EDGAR launch context (kept for re-pull if needed)
- Mode: 6 worker threads, 8 req/sec rate limit, observed ~7/sec sustained
- Filings: 226,919 (8-K: 174,404 / 10-Q: 39,761 / 10-K: 12,754)
- Final disk: 746 GB raw SGML envelopes under `data/raw/edgar/{cik}/{accession}.{txt,htm}` (gitignored)
- Resume mechanism: `data/state/edgar_done.txt` tracked completed accessions
- To re-run: `nohup python -m src.data.ingest_filings > logs/edgar_full_$(date +%Y%m%d_%H%M%S).log 2>&1 &`
- Storage: `data/raw/edgar/{cik}/{accession}.{txt,htm}` — gitignored bulk SGML envelopes
- Resume: `data/state/edgar_done.txt` tracks completed accessions; safe to interrupt

**Architecture decision baked in:** EDGAR fetch only saves raw bytes — text extraction is deferred to a notebook so the network-bound fetch parallelizes well. `extract_text_from_html` and `extract_text_from_sgml` exist in `src/data/ingest_filings.py` for that future notebook to import.

**Health-check commands:**
```bash
# Progress (count of completed filings)
wc -l data/state/edgar_done.txt

# Disk usage (will grow toward ~600 GB)
du -sh data/raw/edgar/

# Recent activity
tail -20 logs/edgar_full_*.log

# Process alive?
ps -p $(cat logs/edgar.pid)
```

**If stuck (no new lines in `edgar_done.txt` for >5 min):**
```bash
kill $(cat logs/edgar.pid)
# wait ~30 min for SEC throttle to reset, then re-launch:
nohup python -m src.data.ingest_filings > logs/edgar_full_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo $! > logs/edgar.pid
```

## Action items — do these before model work

### 1. Rotate the FMP API key (security)
The user's FMP API key leaked into a prior conversation transcript via a 403 error message that included the URL with `apikey=...` query string. Code in `src/data/ingest_fundamentals.py` is patched to scrub keys from future error logs, but the previously-leaked key value should be rotated. Steps:
- Visit https://site.financialmodelingprep.com/developer/dashboard
- Issue new key, revoke old one
- Update `FMP_API_KEY` in `.env` (gitignored)

### 2. Decide on fundamentals path
FMP's v3 statement endpoints (income/balance/cashflow) returned `403 Legacy Endpoint` for new keys. Three options documented in **design spec §3.3** + **plan Task 3 header**:

1. Migrate `src/data/ingest_fundamentals.py` to FMP's `/stable/` endpoints
2. Upgrade FMP tier (Starter $14/mo or higher) — confirm with FMP support whether the tier covers needed endpoints
3. Switch fundamentals provider to AlphaVantage (their `OVERVIEW`/`INCOME_STATEMENT`/etc. work, but free tier is 25 req/day — too tight for full universe)

This blocks the structured features that include valuation/profitability/leverage. Until resolved, the ranker can only run on text features + price-derived features. **Pick a path before starting ranker training.**

### 3. Survivorship bias in prices (NEW, from audit)
Audit found prices coverage at **78.1%** vs 95% target. The 188 missing tickers are almost all delisted historical S&P 500 members (ABK, AGN, APC, ALXN, etc.) that yfinance silently drops. **Backtest results on this dataset will be optimistically biased.** Two fixes (in `docs/audits/2026-05-10-data-coverage.md`):
1. Swap yfinance → Polygon ($29/mo Stocks Starter) or CRSP via WRDS for survivorship-clean prices
2. Backfill historical CIKs from `https://www.sec.gov/Archives/edgar/cik-lookup-data.txt` to map the 237 unmapped tickers in the universe

For research-paper grade results, fix #1 is necessary. For v1 development you can proceed with the bias and document it.

### 4. Re-running the audit
The cross-coverage audit at `docs/audits/2026-05-10-data-coverage.md` ends with a one-liner that reproduces the coverage numbers — useful when re-checking after fixes:
```bash
python -c "
import pandas as pd
universe = pd.read_parquet('data/processed/universe.parquet')
prices = pd.read_parquet('data/processed/prices.parquet')
edgar = pd.read_parquet('data/processed/edgar_index.parquet')
macro = pd.read_parquet('data/processed/macro.parquet')

uni_tickers = set(universe['ticker'])
uni_ciks = set(universe['cik'].dropna())
print(f'Universe: {len(universe)} intervals, {len(uni_tickers)} tickers, {len(uni_ciks)} CIKs')
print(f'Prices coverage: {len(set(prices[\"ticker\"]) & uni_tickers)}/{len(uni_tickers)} ({100*len(set(prices[\"ticker\"]) & uni_tickers)/len(uni_tickers):.0f}%)')
print(f'EDGAR coverage: {len(set(edgar[\"cik\"]) & uni_ciks)}/{len(uni_ciks)} ({100*len(set(edgar[\"cik\"]) & uni_ciks)/len(uni_ciks):.0f}%)')
print(f'Macro: {macro[\"series\"].nunique()} series, {macro[\"date\"].min().date()}–{macro[\"date\"].max().date()}')
"
```
Expected coverage thresholds: prices ≥95%, EDGAR ≥90% of universe CIKs.

## FinBERT MLM fine-tune (✅ DONE 2026-05-15)

3 epochs continued-MLM on the EDGAR corpus.
- Final train loss ~1.85, val loss ~1.81, val perplexity ~6.16. No overfitting.
- Encoder + tokenizer saved to `artifacts/finbert-mlm/` (4.1 GB, synced to R2; not in git).
- Train/val curves and summary in `reports/metrics/`.
- Notebook of record: `notebooks/01_finbert_finetune.ipynb`.

Execution history kept below for the next time MLM continued pretraining is rerun.

### Step 1: Run Phase 1 (text extraction) — CLI ✅ DONE 2026-05-11
```bash
python -m src.data.clean_filings --workers 16
```
- Multiprocessing-based; **prefer `--workers 16`** on the 9950X (16 physical cores).
  Default `cpu_count - 2` = 30 oversubscribes SMT and contends on single-disk I/O.
- Measured: 21 min wall time for 226,919 filings → 221,844 written, 5,075 skipped
  (too-short or below MIN_TEXT_LENGTH=500).
- Output: `data/interim/edgar_text/{cik}/{accession}.txt` — 103 GB total (~14% of raw).
- Per-file resume; safe to interrupt and re-run.
- Quality caveat (was an issue for MLM, blocking for embedding): output retains
  XBRL namespace residue (`iso4217:USD xbrli:shares`), financial-table number
  dumps, residual HTML attribute fragments (the `<[^>]+>` regex fails when an
  attribute value contains `>`), and partially-decoded base64 image
  attachments. The median 10-Q was 2.1 MB of "text" of which ~70% was noise.
  **Resolved 2026-05-15** via `src/data/refilter_text.py` (sliding-window
  content-density filter + BeautifulSoup HTML fallback). Output:
  `data/interim/edgar_text_v2/` — 53 GB across 226,915 files (vs 243 GB v1).
  Notebook 02 reads v2; v1 is retained as the canonical anchor.

### Step 2: Open the notebook in Cursor and execute Sections A–D
- `notebooks/01_finbert_finetune.ipynb`
- Section A: setup, GPU check (expects RTX 5090 + bf16)
- Section B: verify ≥200K cleaned files from Step 1
- Section C: tokenize (~30-60 min) → `data/processed/finbert_tok/`
- Section D: 50-step dry run (~2 min). **GATE: confirm loss is finite and trending down BEFORE proceeding to Section E.** If OOM: bump grad-accum to 2, drop batch to 32.

### Step 3: Run Section E (full training, ~36-48 hr)
- One long cell.
- TensorBoard sidecar at port 6006 streams live metrics.
- The `CursorSafeProgressCallback` mirrors logs to `logs/finbert_finetune_<timestamp>.log` and keeps notebook output bounded.

### Step 4: Run Section F (eval + save)
- Saves encoder + tokenizer to `artifacts/finbert-mlm/`
- Saves train/val curves to `reports/metrics/finbert_finetune.parquet`
- Saves summary JSON to `reports/metrics/finbert_finetune_summary.json`

### Recovery
- Training crash: `trainer.train(resume_from_checkpoint=True)` resumes from latest checkpoint
- OOM at full training: drop `per_device_train_batch_size=32` + `gradient_accumulation_steps=2`
- HF Hub unreachable: notebook auto-falls back to `bert-base-uncased`

### Validation gates (Section F outputs to look for)
- `artifacts/finbert-mlm/config.json` exists
- `artifacts/finbert-mlm/model.safetensors` (or `pytorch_model.bin`) exists
- `reports/metrics/finbert_finetune.parquet` exists and loads
- Val perplexity printed in Section F is finite and lower than initial random-init

## Next phase — after FinBERT FT completes

**STOP and confirm with the user before starting the next phase.**

**Roadmap from the spec (post-FinBERT):**

1. **Embedding generation notebook** — `notebooks/02_finbert_embed.ipynb` (not yet built). Sliding-window inference using the FT'd encoder over each cleaned filing; emit `[CLS]` per chunk → mean-pool to document embedding → stock-day aggregation per spec §5.2 (exponentially-decay-weighted mean, 14-day half-life).

2. **Text feature engineering** (spec §5) — PCA dim selected via 99% cumulative-variance threshold on the first walk's training window, then locked. Plus 3 aux scalars: text novelty, days-since-filing, doc-count.

3. **Structured features** (spec §6) — momentum, vol, beta, liquidity, etc. (≈25 features). **Note: valuation/profitability/leverage features depend on fundamentals — currently deferred (FMP v3 endpoints deprecated; WRDS Compustat awaiting school approval).**

4. **LightGBM ranker** (spec §7) — `lambdarank` over text-PCA + structured + 3 aux text scalars. Walk-forward expanding window starting 2002.

5. **Top-K selection** (spec §8) — K=30, ADV ≥ $5M, price ≥ $5.

6. **PPO RL allocator** (spec §9) — ~880-dim state, continuous target weights over top-30, after-cost reward.

7. **Constraint layer + backtest** (spec §10–11) — softmax→clip→renorm→turnover cap→ADV cap; 5 bps linear cost; weekly Friday close → Monday open execution.

8. **Per-walk logging & diagnostics** (spec §17) — W&B as primary, parquet mirror for paper-ready plots. MVP-tier vs v2 split is in the spec.

## Key user preferences (NON-NEGOTIABLE)

- **No bloat.** Single-responsibility modules, no premature abstraction (no `BaseFooProvider`, no plugin systems).
- **Notebooks for model code, `.py` for data fetching.** `src/` is for `data/` and `utils/` only.
- **TDD for deterministic units** (parsers, rate limiters, helpers); smoke tests for API clients; no mock-of-the-thing-being-tested.
- **Commit messages: simple, no Claude/AI attribution.**
- **Push to GitHub** when work blocks need to sync to laptop. The `data-ingestion` branch is currently pushed but not merged to `main`.
- **Confirm before model work.** Don't auto-progress from data ingestion into FinBERT/ranker/RL.

## Repo state

- Branch `data-ingestion` is ahead of `main` by ~25 commits. Decide whether to merge or PR-review when ingestion is fully done (after audit + maybe fundamentals fix).
- Tracked data: `data/processed/{universe,prices,macro,edgar_index}.parquet`. Bulk EDGAR text + raw API dumps are gitignored under `data/raw/`.
- 23 unit tests passing in `tests/utils/` and `tests/data/`. No model-code tests yet (model code goes in notebooks).

## Skills/plugins note

If the next session has a `machine-learning` skill that wasn't visible during data ingestion, use it for the modeling phase. Otherwise rely on `superpowers:brainstorming` → `superpowers:writing-plans` → `superpowers:subagent-driven-development` for each model component.

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
| 4 | EDGAR text (10-K/10-Q/8-K) | 🟡 running overnight | See "Currently running" below |
| 5 | Macro via FRED (DGS3MO, DGS10, VIXCLS, T10Y2Y) | ✅ done | 26K rows × 4 series. `data/processed/macro.parquet` |
| 6 | Cross-coverage integration audit | ⏳ pending | Run after EDGAR finishes (plan §6.1) |

## Currently running

**EDGAR full-history pull** (launched 2026-05-10 ~00:40):
- Process: PID was in `logs/edgar.pid` (check with `cat logs/edgar.pid`); kill with `kill $(cat logs/edgar.pid)`
- Log: `logs/edgar_full_*.log` (latest)
- Mode: 6 worker threads, 8 req/sec rate limit, observed ~5/sec sustained
- ETA: ~30–35 hours total (~600K filings × 25 years)
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

### 3. Cross-coverage audit (Task 6)
Once EDGAR finishes, run the audit from plan §6.1:
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

## Next phase — model work

**STOP after the audit and check in with the user before starting any model code.** They explicitly said:
- Notebooks (`.ipynb`) for all model/training/eval code, NOT `.py` modules. `notebooks/` directory exists with `.gitkeep` — empty, ready for content.
- They want to verify a "machine-learning" skill is loaded before model work begins (not currently visible — may be a plugin issue or skill-name confusion; ask them to confirm what skill they meant).

**Roadmap from the spec:**

1. **Text post-processing notebook** — extract clean text from `data/raw/edgar/*` SGML envelopes. Hooks already exist (`extract_text_from_sgml`, `extract_text_from_html` in `src/data/ingest_filings.py`). Output to `data/interim/edgar_text/{cik}/{accession}.txt`.

2. **FinBERT MLM continued pretraining** (spec §4) — full fine-tune (no LoRA), bf16, batch 64×512, 1–3 epochs, target 1B+ tokens of EDGAR. Hardware target: RTX 5090 Blackwell (`requirements.txt` already pinned for cu128).

3. **Text feature engineering** (spec §5) — document embeddings → exponentially-decay-weighted stock-day aggregation (14-day half-life) → PCA dim selected via 99% cumulative-variance threshold on the first walk's training window, then locked.

4. **Structured features** (spec §6) — momentum, vol, beta, liquidity, etc. (≈25 features). **Note: valuation/profitability/leverage features depend on fundamentals — see Action item #2.**

5. **LightGBM ranker** (spec §7) — `lambdarank` over text-PCA + structured + 3 aux text scalars. Walk-forward expanding window starting 2002.

6. **Top-K selection** (spec §8) — K=30, ADV ≥ $5M, price ≥ $5.

7. **PPO RL allocator** (spec §9) — ~880-dim state, continuous target weights over top-30, after-cost reward.

8. **Constraint layer + backtest** (spec §10–11) — softmax→clip→renorm→turnover cap→ADV cap; 5 bps linear cost; weekly Friday close → Monday open execution.

9. **Per-walk logging & diagnostics** (spec §17) — W&B as primary, parquet mirror for paper-ready plots. MVP-tier vs v2 split is in the spec.

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

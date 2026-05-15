# axiom_tilt

Research project on text-enhanced reinforcement-learning portfolio allocation.

> Can fine-tuned financial-text representations help an RL agent allocate a weekly long-only US-equity portfolio better than non-text or non-RL baselines, after realistic trading costs?

The pipeline is split in two stages so each component has a clear job:

1. **Ranking stage** — FinBERT (continued-pretrained on EDGAR) produces document embeddings; these get aggregated to stock-week features alongside structured market and fundamentals signals, and a LightGBM `lambdarank` ranks the universe each week.
2. **Allocation stage** — a PPO agent picks weights over the top-K ranked names under transaction costs and portfolio constraints.

The full design rationale lives in [`docs/superpowers/specs/2026-05-08-text-enhanced-rl-portfolio-design.md`](docs/superpowers/specs/2026-05-08-text-enhanced-rl-portfolio-design.md). This README is the practical entry point.

## Status

| Stage | What | State |
|---|---|---|
| Data | S&P 500 universe (Wikipedia + SEC ticker→CIK) | ✅ |
| Data | Prices: CRSP daily via WRDS (survivorship-clean, 1998+) | ✅ |
| Data | Fundamentals: Sharadar SF1 via Nasdaq Data Link (PIT, 1998+) | ✅ |
| Data | EDGAR 10-K / 10-Q / 8-K SGML | ✅ (226,919 filings) |
| Data | Macro: FRED (DGS3MO, DGS10, VIXCLS, T10Y2Y) | ✅ |
| Data | Point-in-time panel build | ✅ |
| Model | FinBERT MLM continued pretraining on EDGAR | ✅ val perplexity ~6.16 |
| Model | Document → stock-day embedding generation | ⏳ next |
| Model | LightGBM `lambdarank` over text + structured features | ⏳ |
| Model | Top-K selection (K=30) | ⏳ |
| Model | PPO RL allocator + constraint layer | ⏳ |
| Eval | Walk-forward backtest with after-cost reward | ⏳ |

See [`TODO.md`](TODO.md) for the working punch list.

## How the storage works

Two-tier setup because the raw data is too large to live in git:

- **Tracked in git:** code, notebooks, small reference parquets (universe, macro, edgar index), training-run metrics, design specs.
- **Synced to Cloudflare R2** (`scripts/sync_to_r2.sh`, `sync_from_r2.sh`): raw EDGAR filings (~750 GB), tokenized FinBERT corpus, fine-tuned encoder weights (~4 GB), CRSP daily panel, Sharadar parquets.

R2 is the single source of truth for bulk data so a fresh clone can be made workable with `rclone` + the project's `.env`. See [`docs/collaborator-setup.md`](docs/collaborator-setup.md).

## Quickstart

```bash
# 1. Install
pip install -r requirements.txt

# 2. Secrets (WRDS, Nasdaq Data Link, FRED, R2)
cp .env.example .env
# fill in credentials — see docs/wrds-setup.md and docs/collaborator-setup.md

# 3. Pull bulk data from R2 (skip if regenerating from sources)
./scripts/sync_from_r2.sh

# 4. Run the tests
pytest

# 5. Open the FinBERT fine-tune notebook to inspect training run
jupyter notebook notebooks/01_finbert_finetune.ipynb
```

To re-run an ingestion step from scratch:

```bash
python -m src.data.build_universe    # S&P 500 + SEC ticker→CIK
python -m src.data.ingest_wrds       # CRSP daily prices
python -m src.data.ingest_sharadar   # SF1 fundamentals
python -m src.data.ingest_filings    # EDGAR SGML (~8 hr, resumable)
python -m src.data.clean_filings     # SGML → text (~20 min, 16 workers)
python -m src.data.ingest_macro      # FRED series
python -m src.data.build_panel       # join everything into the PIT panel
```

## Repo layout

```
src/
  data/      ingestion: WRDS, Sharadar, EDGAR, FRED, panel build
  utils/     env loading, paths, logging, rate limiter, seeding
notebooks/
  01_finbert_finetune.ipynb     completed MLM continued pretraining
scripts/
  sync_to_r2.sh / sync_from_r2.sh
docs/
  superpowers/specs/    design docs (the source of truth for what to build)
  superpowers/plans/    execution plans per phase
  wrds-setup.md         WRDS account + pgpass setup
  collaborator-setup.md R2 + env onboarding for new machines
tests/                  pytest suite (50 passing)
reports/metrics/        training-run summaries + parquet curves
data/processed/         small tracked artifacts (universe, macro, edgar_index)
```

`artifacts/`, `data/raw/`, `data/interim/`, and the large `data/processed/` derivatives are gitignored — they live in R2.

## Hardware

Training is tuned for an RTX 5090 (Blackwell, sm_120, bf16); `requirements.txt` pins the PyTorch cu128 extra index for that. CPU-only ingestion runs fine on anything.

## Why this framing

There are two unsatisfying extremes for RL in portfolios: letting RL output weights over the entire universe (too high-dimensional and unstable to train), or letting RL pick a single risk-aversion knob while a classical optimizer does the real work (RL becomes decorative).

This project picks a middle ground. The supervised ranker handles cross-sectional selection — the part that benefits most from supervised learning over years of labeled data. RL handles sequential allocation among ~30 candidates — the part that genuinely needs to reason about turnover, costs, and path dependence. Text signal feeds the ranker so that the candidate set actually reflects something the agent couldn't extract from prices alone.

The detailed motivation, training protocol, and ablation plan are in the design spec linked above.

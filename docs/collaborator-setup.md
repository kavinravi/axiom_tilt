# Collaborator setup

Get a fresh laptop from zero to running notebooks, with all shared data in place.

## Prerequisites

- git
- Python 3.11+ (owner runs 3.12)
- **~400 GB free disk.** The synced bundle is ~95-110 GB compressed, but
  `edgar_text` unpacks from its ~50 GB tarball to ~243 GB of loose files and
  `edgar_text_v2` adds another ~53 GB. Budget for the download + both
  unpacked trees + a bit for your own work.

## Steps

### 1. Clone and install

```bash
git clone https://github.com/kavinravi/axiom_tilt.git
cd axiom_tilt
pip install -r requirements.txt
pip install -e .
```

`requirements.txt` pins torch, transformers, WRDS, etc. `pip install -e .` then makes `src/` importable as the `axiom_tilt` package. Both steps are needed.

### 2. Set up your `.env`

```bash
cp .env.example .env
```

`.env` holds per-person credentials and is gitignored — never commit it. It has
three keys:

- `SEC_USER_AGENT` — only needed to re-run the EDGAR filings pull (`ingest_filings.py`)
- `WRDS_USERNAME` — only needed to re-run the WRDS pull (`ingest_wrds.py`)
- `NASDAQ_DATA_LINK_API_KEY` — only needed to re-run the Sharadar pull (`ingest_sharadar.py`)

**If you're only consuming the data the owner already pulled (the common case),
you can leave `.env` blank** — none of those keys are needed to read the synced
parquets or run the model. Fill in only the key(s) for any pull you intend to
re-run yourself.

### 3. Get a WRDS account from your school

Same process as the project owner — see `docs/wrds-setup.md`. You'll need your own account; we don't share credentials.

You only need WRDS if you want to re-run the data pulls yourself. To just consume the data the owner already pulled, you can skip the WRDS setup.

### 4. Get Cloudflare R2 credentials from the project owner

Ask the owner for three values (delivered out-of-band — 1Password, Signal, or in person; **never in this repo**). The owner issues you a **read-only** R2 token — you pull data, you don't push:

- R2 Access Key ID
- R2 Secret Access Key
- R2 Endpoint URL (looks like `https://<account-id>.r2.cloudflarestorage.com`)

### 5. Install rclone

```bash
# Linux / WSL
curl https://rclone.org/install.sh | sudo bash

# macOS (Homebrew)
brew install rclone
```

### 6. Configure rclone

```bash
mkdir -p ~/.config/rclone
cp docs/rclone-r2-template.conf ~/.config/rclone/rclone.conf
# Edit ~/.config/rclone/rclone.conf and paste in the three values from step 4.
# The section header MUST be exactly [r2] on its own first line.
```

Verify:
```bash
rclone lsd r2:
# Should list:
#   -1 ... -1 axiom-tilt-data
```

### 7. Pull shared data

```bash
./scripts/sync_from_r2.sh
```

This downloads ~95-110 GB:
- CRSP daily prices (`data/processed/crsp_daily/`, via WRDS)
- Sharadar SF1 fundamentals (`data/processed/sharadar_sf1.parquet`)
- The unified PIT panel (`data/processed/panel/`)
- Cleaned EDGAR text bundle (v1) — pulled as a ~50 GB tarball, auto-unpacked to
  `data/interim/edgar_text/` (~243 GB on disk). This is the raw SGML
  extraction, retained as the canonical anchor for re-running the refilter.
- Refiltered EDGAR text bundle (v2) — pulled as a smaller tarball,
  auto-unpacked to `data/interim/edgar_text_v2/` (~53 GB on disk). XBRL fact
  dumps, residual HTML, and base64 attachments are stripped. **Notebook 02
  reads this directory**, not v1.
- Tokenized FinBERT dataset (`data/processed/finbert_tok/`)
- Trained FinBERT model (`artifacts/finbert-mlm/`)

Run time: several hours on home internet, depending on your connection.
Subsequent syncs are diff-only and fast.

### 8. Verify

```bash
python -c "
import duckdb
print('CRSP daily rows:',
      duckdb.sql(\"SELECT COUNT(*) FROM 'data/processed/crsp_daily/year=*/*.parquet'\").df().iloc[0, 0])
print('Panel rows:',
      duckdb.sql(\"SELECT COUNT(*) FROM 'data/processed/panel/year=*/*.parquet'\").df().iloc[0, 0])
"
```

If both numbers print, you're set up.

### 9. (Optional) FinBERT inference

```bash
python -c "
from transformers import AutoTokenizer, AutoModelForMaskedLM
tok = AutoTokenizer.from_pretrained('artifacts/finbert-mlm')
mdl = AutoModelForMaskedLM.from_pretrained('artifacts/finbert-mlm')
print('Model loaded:', mdl.config.model_type, mdl.num_parameters() / 1e6, 'M params')
"
```

## What's NOT shared

These are NOT pulled by sync:

- `data/raw/edgar/` — the raw SGML filings. Empty on the owner's machine too (deleted after extraction); re-derivable via `python -m src.data.ingest_filings` if you ever need raw SGML. The *cleaned* text (`data/interim/edgar_text/`) IS shared.
- `artifacts/finbert-mlm/checkpoint-*/` — intermediate training checkpoints; only the final model is shared.
- `.env` — per-person credentials, never leaves your machine.

## Day-to-day

Just run `./scripts/sync_from_r2.sh` when the owner pings you that there's new data. Otherwise treat it like a normal git repo.

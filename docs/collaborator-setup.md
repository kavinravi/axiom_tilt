# Collaborator setup

Get a fresh laptop from zero to running notebooks, with all shared data in place.

## Prerequisites

- git
- Python 3.12 (other versions probably work; not tested)
- ~80 GB free disk for shared data + a bit more for your work

## Steps

### 1. Clone and install

```bash
git clone https://github.com/kavinravi/axiom_tilt.git
cd axiom_tilt
pip install -e .
```

### 2. Get a WRDS account from your school

Same process as the project owner — see `docs/wrds-setup.md`. You'll need your own account; we don't share credentials.

You only need WRDS if you want to re-run the data pulls yourself. To just consume the data the owner already pulled, you can skip the WRDS setup.

### 3. Get Cloudflare R2 credentials from the project owner

Ask the owner for three values (delivered out-of-band — 1Password, Signal, or in person; **never in this repo**):

- R2 Access Key ID
- R2 Secret Access Key
- R2 Endpoint URL (looks like `https://<account-id>.r2.cloudflarestorage.com`)

### 4. Install rclone

```bash
# Linux / WSL
curl https://rclone.org/install.sh | sudo bash

# macOS (Homebrew)
brew install rclone
```

### 5. Configure rclone

```bash
mkdir -p ~/.config/rclone
cp docs/rclone-r2-template.conf ~/.config/rclone/rclone.conf
# Edit ~/.config/rclone/rclone.conf and paste in the three values from step 3.
```

Verify:
```bash
rclone lsd r2:
# Should list:
#   -1 ... -1 axiom-tilt-data
```

### 6. Pull shared data

```bash
./scripts/sync_from_r2.sh
```

This downloads ~70 GB:
- WRDS parquets (CRSP daily, Compustat, link table)
- Cleaned EDGAR text bundle (unpacked to `data/interim/edgar_text/`)
- Tokenized FinBERT dataset (`data/processed/finbert_tok/`)
- Trained FinBERT model (`artifacts/finbert-mlm/`)
- Archived legacy data (`data/archive/`)

Run time: ~1-3 hr on home internet. Subsequent syncs are diff-only.

### 7. Verify

```bash
python -c "
import duckdb
print('CRSP daily rows:',
      duckdb.sql(\"SELECT COUNT(*) FROM 'data/processed/crsp_daily/year=*/*.parquet'\").df().iloc[0, 0])
print('Compustat quarterly rows:',
      duckdb.sql(\"SELECT COUNT(*) FROM 'data/processed/comp_fundq.parquet'\").df().iloc[0, 0])
"
```

If both numbers print, you're set up.

### 8. (Optional) FinBERT inference

```bash
python -c "
from transformers import AutoTokenizer, AutoModelForMaskedLM
tok = AutoTokenizer.from_pretrained('artifacts/finbert-mlm')
mdl = AutoModelForMaskedLM.from_pretrained('artifacts/finbert-mlm')
print('Model loaded:', mdl.config.model_type, mdl.num_parameters() / 1e6, 'M params')
"
```

## What's NOT shared

These stay on the owner's machine and are NOT pulled by sync:

- `data/raw/edgar/` (243 GB SGML filings) — re-derivable via `python -m src.data.ingest_filings`.
- `artifacts/finbert-mlm/checkpoint-*/` (intermediate training checkpoints) — only the final model is shared.
- `.env` (user-specific credentials).

## Day-to-day

Just run `./scripts/sync_from_r2.sh` when the owner pings you that there's new data. Otherwise treat it like a normal git repo.

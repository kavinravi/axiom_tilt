#!/usr/bin/env bash
# Push local data + artifacts -> Cloudflare R2 bucket axiom-tilt-data.
# Run after WRDS pulls, FinBERT training milestones, or any data update worth sharing.
#
# Requires: rclone configured with an [r2] remote (see docs/rclone-r2-template.conf).
# Skips:    data/raw/edgar/ (243 GB, re-derivable via src/data/ingest_filings.py)
#           artifacts/finbert-mlm/checkpoint-*/ (1.3 GB each, reproducible via training resume)

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

if ! command -v rclone >/dev/null 2>&1; then
    echo "rclone not installed. Run: curl https://rclone.org/install.sh | sudo bash" >&2
    exit 1
fi

if ! rclone listremotes | grep -q '^r2:$'; then
    echo "rclone remote 'r2' not configured." >&2
    echo "Copy docs/rclone-r2-template.conf -> ~/.config/rclone/rclone.conf and fill in your R2 keys." >&2
    exit 1
fi

# Pack the 227K-file edgar_text dir into one zstd-compressed archive.
# rclone syncs O(1000s) of files efficiently; O(100K+) overwhelms it.
if [ -d data/interim/edgar_text ] && [ ! -f data/interim/edgar_text.tar.zst ]; then
    echo "Packing data/interim/edgar_text/ (this takes a few minutes)..."
    tar -I 'zstd -T0 -19' -cf data/interim/edgar_text.tar.zst \
        -C data/interim edgar_text
fi

echo "Syncing data/processed/ -> r2:axiom-tilt-data/data/processed/"
rclone sync data/processed/ r2:axiom-tilt-data/data/processed/ --progress

if [ -f data/interim/edgar_text.tar.zst ]; then
    echo "Syncing data/interim/edgar_text.tar.zst -> r2:axiom-tilt-data/data/interim/"
    rclone sync data/interim/edgar_text.tar.zst \
        r2:axiom-tilt-data/data/interim/edgar_text.tar.zst --progress
fi

if [ -d data/archive ]; then
    echo "Syncing data/archive/ -> r2:axiom-tilt-data/data/archive/"
    rclone sync data/archive/ r2:axiom-tilt-data/data/archive/ --progress
fi

if [ -d artifacts/finbert-mlm ]; then
    echo "Syncing artifacts/finbert-mlm/ -> r2 (excluding intermediate checkpoints)"
    rclone sync artifacts/finbert-mlm/ r2:axiom-tilt-data/artifacts/finbert-mlm/ \
        --progress --exclude 'checkpoint-*/**'
fi

echo "Sync complete."

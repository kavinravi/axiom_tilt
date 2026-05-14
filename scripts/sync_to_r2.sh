#!/usr/bin/env bash
# Push local data + artifacts -> Cloudflare R2 bucket axiom-tilt-data.
# Run after WRDS pulls, FinBERT training milestones, or any data update worth sharing.
#
# Requires: rclone configured with an [r2] remote (see docs/rclone-r2-template.conf).
# Skips:    data/raw/edgar/ (raw SGML — huge, re-derivable via ingest_filings.py)
#           artifacts/finbert-mlm/checkpoint-*/ (reproducible via training resume)
# The bucket is meant to be a complete "recreate the project" snapshot otherwise.

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

if ! command -v rclone >/dev/null 2>&1; then
    echo "rclone not installed. Run: curl https://rclone.org/install.sh | sudo bash" >&2
    exit 1
fi

if ! command -v zstd >/dev/null 2>&1; then
    echo "zstd not installed (needed to pack edgar_text). Run: sudo apt install zstd" >&2
    exit 1
fi

if ! rclone listremotes | grep -q '^r2:$'; then
    echo "rclone remote 'r2' not configured." >&2
    echo "Copy docs/rclone-r2-template.conf -> ~/.config/rclone/rclone.conf and fill in your R2 keys." >&2
    exit 1
fi

# --s3-no-check-bucket: the R2 API token has Object Read/Write but not
# bucket-creation rights (by design). rclone's multipart-upload path probes
# CreateBucket otherwise, which 403s on large files. The bucket already exists.
RCLONE_OPTS=(--progress --s3-no-check-bucket)

# Pack the many-file edgar_text dir into one zstd-compressed archive.
# rclone syncs O(1000s) of files efficiently; O(100K+) overwhelms it.
# edgar_text is large (hundreds of GB of text) — this pack step is slow.
# Pack to a .tmp name and rename on success, so a failed/interrupted pack never
# leaves a partial tarball that the existence check below would mistake for done.
if [ -d data/interim/edgar_text ] && [ ! -f data/interim/edgar_text.tar.zst ]; then
    echo "Packing data/interim/edgar_text/ (large — this is slow)..."
    tar -I 'zstd -T0 -19' -cf data/interim/edgar_text.tar.zst.tmp \
        -C data/interim edgar_text
    mv data/interim/edgar_text.tar.zst.tmp data/interim/edgar_text.tar.zst
fi

echo "Syncing data/processed/ -> r2:axiom-tilt-data/data/processed/"
rclone sync data/processed/ r2:axiom-tilt-data/data/processed/ "${RCLONE_OPTS[@]}"

# data/raw/sec/ is a small universe-build input; data/raw/edgar/ (raw SGML) is
# deliberately not synced — it's huge and re-derivable via ingest_filings.py.
if [ -d data/raw/sec ]; then
    echo "Syncing data/raw/sec/ -> r2:axiom-tilt-data/data/raw/sec/"
    rclone sync data/raw/sec/ r2:axiom-tilt-data/data/raw/sec/ "${RCLONE_OPTS[@]}"
fi

if [ -f data/interim/edgar_text.tar.zst ]; then
    echo "Syncing data/interim/edgar_text.tar.zst -> r2:axiom-tilt-data/data/interim/"
    rclone sync data/interim/edgar_text.tar.zst \
        r2:axiom-tilt-data/data/interim/edgar_text.tar.zst "${RCLONE_OPTS[@]}"
fi

if [ -d artifacts/finbert-mlm ]; then
    echo "Syncing artifacts/finbert-mlm/ -> r2 (excluding intermediate checkpoints)"
    rclone sync artifacts/finbert-mlm/ r2:axiom-tilt-data/artifacts/finbert-mlm/ \
        "${RCLONE_OPTS[@]}" --exclude 'checkpoint-*/**'
fi

echo "Sync complete."

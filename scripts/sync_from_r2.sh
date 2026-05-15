#!/usr/bin/env bash
# Pull data + artifacts from Cloudflare R2 bucket axiom-tilt-data -> local repo.
# Collaborator runs this after first clone and whenever the owner pushes updates.
#
# Requires: rclone configured with an [r2] remote (see docs/rclone-r2-template.conf).

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

if ! command -v rclone >/dev/null 2>&1; then
    echo "rclone not installed. Run: curl https://rclone.org/install.sh | sudo bash" >&2
    exit 1
fi

if ! command -v zstd >/dev/null 2>&1; then
    echo "zstd not installed (needed to unpack edgar_text). Run: sudo apt install zstd" >&2
    exit 1
fi

if ! rclone listremotes | grep -q '^r2:$'; then
    echo "rclone remote 'r2' not configured." >&2
    echo "Copy docs/rclone-r2-template.conf -> ~/.config/rclone/rclone.conf and fill in the R2 keys you got from the project owner." >&2
    exit 1
fi

# --s3-no-check-bucket: R2 tokens lack bucket-creation rights by design; without
# this rclone probes CreateBucket and 403s. The bucket already exists.
RCLONE_OPTS=(--progress --s3-no-check-bucket)

echo "Pulling data/processed/ from R2..."
rclone copy r2:axiom-tilt-data/data/processed/ data/processed/ "${RCLONE_OPTS[@]}"

echo "Pulling data/raw/sec/ from R2..."
rclone copy r2:axiom-tilt-data/data/raw/sec/ data/raw/sec/ "${RCLONE_OPTS[@]}"

echo "Pulling data/interim/ from R2..."
rclone copy r2:axiom-tilt-data/data/interim/ data/interim/ "${RCLONE_OPTS[@]}"

echo "Pulling artifacts/ from R2..."
rclone copy r2:axiom-tilt-data/artifacts/ artifacts/ "${RCLONE_OPTS[@]}"

# Unpack the tarred edgar_text bundles if present and not already unpacked.
# v1 (raw SGML extraction) is the canonical anchor; v2 is the refiltered
# embedding-ready corpus and is what notebook 02 reads.
if [ -f data/interim/edgar_text.tar.zst ] && [ ! -d data/interim/edgar_text ]; then
    echo "Unpacking data/interim/edgar_text.tar.zst (large — this is slow)..."
    tar -I 'zstd -d' -xf data/interim/edgar_text.tar.zst -C data/interim/
fi

if [ -f data/interim/edgar_text_v2.tar.zst ] && [ ! -d data/interim/edgar_text_v2 ]; then
    echo "Unpacking data/interim/edgar_text_v2.tar.zst (this is slow)..."
    tar -I 'zstd -d' -xf data/interim/edgar_text_v2.tar.zst -C data/interim/
fi

echo "Sync complete."
echo ""
echo "Verify with:"
echo "  python -c \"import duckdb; print(duckdb.sql(\\\"SELECT COUNT(*) FROM 'data/processed/crsp_daily/year=*/*.parquet'\\\").df())\""

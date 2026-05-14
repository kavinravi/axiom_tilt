#!/usr/bin/env bash
# One-time migration: archive yfinance prices + FMP fundamentals before swapping in WRDS.
# Safe to run repeatedly; skips files that are already archived.

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

mkdir -p data/archive
today=$(date +%F)

if [ -f data/processed/prices.parquet ]; then
    target="data/archive/prices_yfinance_${today}.parquet"
    mv data/processed/prices.parquet "$target"
    echo "Archived yfinance prices -> $target"
else
    echo "No data/processed/prices.parquet to archive (already moved or never created)."
fi

# FMP fundamentals output (if it exists). Path was historically
# data/processed/fundamentals.parquet per src/data/ingest_fundamentals.py docstring,
# though the active runs may have written raw JSON to data/raw/fundamentals/.
if [ -f data/processed/fundamentals.parquet ]; then
    target="data/archive/fundamentals_fmp_${today}.parquet"
    mv data/processed/fundamentals.parquet "$target"
    echo "Archived FMP fundamentals -> $target"
else
    echo "No data/processed/fundamentals.parquet to archive."
fi

echo "Migration complete."

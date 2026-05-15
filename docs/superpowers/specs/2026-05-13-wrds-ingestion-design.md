# WRDS Data Ingestion + Shared Storage — Design Spec

**Date:** 2026-05-13
**Status:** Partially implemented — see Addendum below for what changed
**Repo:** `axiom_tilt`
**Branch:** `data-ingestion`
**Supersedes:** `docs/superpowers/specs/2026-05-11-tiingo-prices-backfill-design.md` (tiingo backfill becomes moot once CRSP daily is in)

## Addendum (2026-05-13, post-implementation) — Compustat → Sharadar pivot

The fundamentals half of this spec did not survive contact with the school's WRDS
subscription tier. What actually shipped:

- **WRDS Compustat is inaccessible.** The school's WRDS tier has no `comp.funda` /
  `comp.fundq` access (and the CRSP/Compustat-merged and sample schemas were either
  linkage-only or 25-firm samples). `pull_compustat_funda`, `pull_compustat_fundq`,
  `detect_compustat_schema`, the CCM link table pull, and **gvkey resolution** were
  all removed — WRDS's role is now **CRSP daily prices + the ticker→permno crosswalk
  only**. Sections 4, 6, 8 below describe code that no longer exists.
- **SEC XBRL was tried as a bridge and dropped.** `ingest_edgar_xbrl.py` pulled the
  companyfacts API but XBRL only covers ~2009+ — no GFC. Removed entirely.
- **Fundamentals now come from Sharadar SF1** (Nasdaq Data Link, paid). PIT via
  `datekey`, ARQ/ARY as-reported dimensions, coverage back to ~1993. New module:
  `src/data/ingest_sharadar.py`.
- **New: `src/data/build_panel.py`** materializes the PIT panel —
  `merge_asof` of CRSP daily ← Sharadar SF1 (backward, `datekey <= date`), with a
  leakage guard. Output: `data/processed/panel/year=YYYY/`. Permnos with zero
  Sharadar coverage (~245 of 826, mostly old delisted names) are struck from the
  panel — accepted partial survivorship bias, since FMP was off the table.
- **yfinance prices and FMP fundamentals** (`ingest_prices.py`, `ingest_fundamentals.py`)
  were deleted outright, not archived — they were never used in the final pipeline,
  so the `migrate_to_wrds.sh` archival step (Section 9) is moot and also removed.

The R2 / rclone sharing pipeline (Sections 5, 10, 11) is still the plan and unaffected.

## 1. Goal

Replace the project's current yfinance prices and FMP fundamentals with academic-grade CRSP and Compustat data pulled from Wharton's WRDS, restricted to the historical S&P 500 universe (858 distinct tickers, 1995-2025). Set up a Cloudflare R2 + rclone sharing pipeline so a collaborator can pull the ~70 GB of derived artifacts (WRDS parquets + cleaned EDGAR text + tokenized FinBERT dataset + trained model) without going through GitHub.

The WRDS pull is gated on two correctness wins:

1. **CRSP fixes survivorship bias.** yfinance silently drops delisted tickers; CRSP returns include delisting returns (`msedelist.dlret`). Critical for an unbiased 30-year backtest.
2. **Compustat fixes look-ahead bias.** FMP returns the latest restated fundamental values; Compustat is point-in-time via `datadate` + report-date (`rdq`). Models trained on FMP see future restatements that didn't exist on the training date.

## 2. Scope

**In scope:**
- `src/data/ingest_wrds.py` — CLI module with pulls for CRSP daily (`dsf` + `msedelist` merge), Compustat annual (`funda`) and quarterly (`fundq`), and the CCM link table (`ccmxpf_linktable`).
- Universe-driven permno/gvkey resolution from `data/processed/universe.parquet`.
- Tests with mocked `wrds.Connection` (no live network in CI).
- Migration: archive existing `prices.parquet` (yfinance) and FMP fundamentals to `data/archive/`.
- R2 bucket provisioning docs + `rclone` config template.
- `scripts/sync_to_r2.sh` and `scripts/sync_from_r2.sh`.
- `docs/wrds-setup.md` and `docs/collaborator-setup.md`.

**Out of scope:**
- Inspection notebook (`notebooks/02_wrds_data_audit.ipynb`) — deferred until first successful pull.
- IBES analyst estimates — Phase 2 if existing signals lack lift.
- Live data updates beyond 2025 — covered later by either restoring yfinance for the tail or pulling tiingo.
- Downstream modeling that consumes WRDS data.

## 3. Architecture

```
┌─────────────────────┐         ┌────────────────────────┐
│   WRDS (Wharton)    │         │  Cloudflare R2 bucket  │
│  PostgreSQL hosted  │         │  axiom-tilt-data       │
└──────────┬──────────┘         └───────────┬────────────┘
           │ wrds python pkg                │ rclone sync
           ▼                                ▼
  ┌──────────────────────────────────────────────────────────┐
  │  Local repo                                              │
  │   data/                                                  │
  │     processed/                                           │
  │       crsp_daily/year=YYYY/part-*.parquet  (partitioned) │
  │       comp_funda.parquet                                 │
  │       comp_fundq.parquet                                 │
  │       ccm_linktable.parquet                              │
  │       universe.parquet      (unchanged)                  │
  │       macro.parquet         (unchanged, FRED)            │
  │     interim/edgar_text/                                  │
  │     processed/finbert_tok/                               │
  │     archive/                                             │
  │       prices_yfinance_2026-05-13.parquet                 │
  │       fundamentals_fmp_2026-05-13.parquet                │
  │     raw/edgar/   ← NOT in R2 (re-derivable from SEC)     │
  │   artifacts/finbert-mlm/   (final model only)            │
  └──────────────────────────────────────────────────────────┘
```

## 4. WRDS datasets and pull semantics

**Pull range:** `1995-01-01` to `2025-12-31` for all tables (5-year lookback before universe starts in 2000, for momentum/value features that need history windows).

| WRDS table | Output file | Approx size | Filters / notes |
|---|---|---|---|
| `crsp.dsf` | `data/processed/crsp_daily/year=YYYY/part-*.parquet` | ~1 GB total | Year-partitioned. `WHERE permno IN (...universe permnos...) AND date BETWEEN start AND end`. Columns: `permno, date, prc, ret, vol, shrout, openprc, askhi, bidlo, cfacpr, cfacshr`. |
| `crsp.msedelist` | merged into `crsp_daily` as `dlret` column | inline | `LEFT JOIN` on `permno` matching the delisting date row into the corresponding `dsf` row. Drops survivorship bias. |
| `comp.funda` | `data/processed/comp_funda.parquet` | ~50 MB | Annual fundamentals. Filter `consol='C' AND indfmt='INDL' AND datafmt='STD' AND popsrc='D' AND curcd='USD'` (the standard PIT filter set). `WHERE gvkey IN (...universe gvkeys...) AND datadate BETWEEN start AND end`. |
| `comp.fundq` | `data/processed/comp_fundq.parquet` | ~200 MB | Quarterly fundamentals, same filters. |
| `crsp.ccmxpf_linktable` | `data/processed/ccm_linktable.parquet` | ~5 MB | Full table, no filter (it's small and we want all link periods). Columns: `gvkey, lpermno, lpermco, linktype, linkprim, linkdt, linkenddt`. |

**Universe → permno/gvkey resolution:**

1. Read `data/processed/universe.parquet` (columns: `ticker, cik, company, date_in, date_out`).
2. For each row, query `crsp.stocknames` by ticker + nameenddt overlap to get a candidate `permno`.
3. For each row, query `crsp.ccmxpf_lnkhist` by `lpermno` to get the matching `gvkey` (with `linkprim IN ('P','C')` and date overlap).
4. Write the resolved table to `data/processed/universe_ids.parquet` (columns: `ticker, cik, date_in, date_out, permno, gvkey`).
5. Use this table to derive the permno and gvkey filter sets for all subsequent pulls.

**Sanity gate after universe resolution:** assert ≥ 95% of universe rows resolve to a permno. If below, log unmatched tickers and halt (manual investigation needed; usually a ticker change like FB → META or BRK.B vs BRK-B).

## 5. Storage layout

**Local repository (gitignored data directories):**

```
data/
  processed/
    crsp_daily/year=2000/part-0.parquet   ... year=2025/...
    comp_funda.parquet
    comp_fundq.parquet
    ccm_linktable.parquet
    universe_ids.parquet            ← new, joined resolution table
    universe.parquet                ← unchanged
    macro.parquet                   ← unchanged
  interim/
    edgar_text/                     (existing, ~15 GB, 227K files)
  processed/
    finbert_tok/                    (existing, ~47 GB HF Arrow)
  archive/
    prices_yfinance_2026-05-13.parquet
    fundamentals_fmp_2026-05-13.parquet
  raw/
    edgar/                          (existing, ~243 GB SGML — NOT shared)
artifacts/
  finbert-mlm/                      (existing; final model synced, ckpts local-only)
```

**Cloudflare R2 bucket `axiom-tilt-data`** (synced from local):

| Local path | In R2? | Reason |
|---|---|---|
| `data/processed/*.parquet` (all) | ✅ | Source of truth for collaborator. |
| `data/processed/crsp_daily/year=*/...` | ✅ | All year partitions. |
| `data/interim/edgar_text.tar.zst` | ✅ | Tarred + zstd'd to one archive (rclone hates syncing 227K small files). |
| `data/processed/finbert_tok/` | ✅ | HF Arrow shards (~96 files of ~500 MB each — syncs fine without packing). |
| `artifacts/finbert-mlm/*.json, *.bin, *.safetensors, vocab.txt, ...` (final model only) | ✅ | ~500 MB. Excludes `checkpoint-*/` subdirs (~1.3 GB each, reproducible via training resume). |
| `data/archive/*` | ✅ | One-time snapshot of pre-WRDS data for reproducibility. |
| `data/raw/edgar/` | ❌ | 243 GB SGML, re-derivable by running `python -m src.data.ingest_filings`. |
| `data/processed/edgar_index.parquet` | ✅ | Small, useful, already in repo's processed/. |

## 6. Code module: `src/data/ingest_wrds.py`

### Public API

```python
def resolve_universe_ids(
    universe: pd.DataFrame,
    conn: wrds.Connection,
) -> pd.DataFrame:
    """Resolve (ticker, cik, date_in, date_out) → (permno, gvkey).

    Returns a DataFrame with all universe rows + resolved permno (nullable) + gvkey (nullable).
    Logs and counts unresolved rows.
    """

def pull_crsp_daily(
    conn: wrds.Connection,
    permnos: list[int],
    start: str,
    end: str,
    output_dir: Path,
) -> None:
    """Pull CRSP daily prices + returns, merge delisting returns from msedelist,
    write year-partitioned parquet to {output_dir}/year=YYYY/part-0.parquet."""

def pull_compustat_funda(
    conn: wrds.Connection,
    gvkeys: list[str],
    start: str,
    end: str,
    output_path: Path,
) -> None:
    """Pull Compustat annual fundamentals (consol=C, indfmt=INDL, datafmt=STD, curcd=USD).
    Single parquet output."""

def pull_compustat_fundq(
    conn: wrds.Connection,
    gvkeys: list[str],
    start: str,
    end: str,
    output_path: Path,
) -> None:
    """Pull Compustat quarterly fundamentals. Same filters as funda. Single parquet."""

def pull_ccm_linktable(
    conn: wrds.Connection,
    output_path: Path,
) -> None:
    """Pull full CCM link table — small, no filtering."""
```

### CLI

```bash
python -m src.data.ingest_wrds --start 1995-01-01 --end 2025-12-31 --all
python -m src.data.ingest_wrds --resolve-only       # just universe → permno/gvkey
python -m src.data.ingest_wrds --crsp-only
python -m src.data.ingest_wrds --compustat-only
python -m src.data.ingest_wrds --linktable-only
```

Order when `--all`: linktable → resolve → CRSP → Compustat funda → Compustat fundq. (Linktable + resolution must run first because permno/gvkey filters depend on them.)

### Logging

Per `src/utils/logging_utils.py`: `configure_logging("wrds_ingest")` writes structured logs to `logs/wrds_ingest_<timestamp>.log`. Progress via `tqdm` for chunked pulls.

### Chunking

`pull_crsp_daily` writes one parquet per year (≈31 files) to bound peak memory. Each year's query is independent; failure on year N doesn't lose work on years 1..N-1. Resume logic: check which `year=YYYY/` partitions exist and skip them.

## 7. WRDS authentication

The `wrds` Python package reads `~/.pgpass` (PostgreSQL standard) for credentials. First-time setup:

```bash
python -c "import wrds; wrds.Connection()"
```

It prompts for username and password, then asks "Would you like to create a .pgpass file for you?" — answer `y`. Creates `~/.pgpass` with mode 0600. Future `wrds.Connection()` calls auto-load from there. No interactive prompts in code.

User stores their WRDS username in `.env` as `WRDS_USERNAME=<theirs>` so the module's `wrds.Connection(wrds_username=...)` call knows whose account to use.

`.pgpass` is OUTSIDE the repo (in `$HOME`). No credentials ever in git.

## 8. Tests

`tests/data/test_ingest_wrds.py`:

| Test | Strategy |
|---|---|
| `test_resolve_universe_ids` | Mock `conn.raw_sql` to return synthetic `crsp.stocknames` and `ccmxpf_lnkhist` rows. Verify universe input → expected (permno, gvkey) output. Include a ticker change case (e.g., FB → META mid-period). |
| `test_pull_crsp_daily_basic` | Mock returns synthetic `dsf` + `msedelist`. Verify merge produces expected `dlret` column, year-partitioning, schema. |
| `test_pull_crsp_daily_no_delisting` | All permnos non-delisted. Verify `dlret` column is all NaN, no rows dropped. |
| `test_pull_compustat_funda_filters` | Inspect the SQL produced (capture mock call args). Verify it includes `consol='C' AND indfmt='INDL' AND datafmt='STD' AND popsrc='D' AND curcd='USD'`. |
| `test_pull_compustat_fundq_filters` | Same as above for fundq. |
| `test_pull_ccm_linktable_schema` | Mock returns a few rows. Verify output parquet has expected columns. |
| `test_resume_skips_existing_partitions` | Pre-create one year's parquet partition. Verify the pull doesn't re-query that year. |
| `test_universe_resolution_low_match_halts` | Mock returns very few matches (<95%). Verify the function logs and raises. |

No live network. Run in CI offline. Use `unittest.mock.patch` for `wrds.Connection` and its methods.

## 9. Migration: yfinance / FMP retirement

One-shot migration script `scripts/migrate_to_wrds.sh`:

```bash
mkdir -p data/archive
mv data/processed/prices.parquet data/archive/prices_yfinance_2026-05-13.parquet
# FMP fundamentals output (check actual path — likely data/processed/fundamentals.parquet)
mv data/processed/fundamentals.parquet data/archive/fundamentals_fmp_2026-05-13.parquet 2>/dev/null || true
```

Edit `src/data/ingest_prices.py` and `src/data/ingest_fundamentals.py`: add a `**DEPRECATED**` block at the top of each module docstring pointing at `ingest_wrds.py`. Keep the code (don't delete) for reproducibility / replication purposes.

No downstream code rewrite needed — the modeling stack (`src/text/*.py`, `src/models/*.py`, etc.) is currently empty after the earlier cleanup commits.

## 10. R2 setup + sync scripts

### Bucket provisioning

User (one-time):

1. Cloudflare account → R2 → enable.
2. **Create bucket** `axiom-tilt-data`, location hint Automatic.
3. **Manage R2 API Tokens** → create token with Object Read & Write, bucket-scoped to `axiom-tilt-data`. Save Access Key ID, Secret Access Key, Endpoint URL.

### rclone config

Stored at `~/.config/rclone/rclone.conf` (NOT in repo). Template at `docs/rclone-r2-template.conf`:

```ini
[r2]
type = s3
provider = Cloudflare
access_key_id = <PASTE_ACCESS_KEY_ID>
secret_access_key = <PASTE_SECRET_ACCESS_KEY>
endpoint = <PASTE_ENDPOINT_URL>
acl = private
```

Install rclone (Ubuntu/WSL):
```bash
curl https://rclone.org/install.sh | sudo bash
```

### scripts/sync_to_r2.sh (push local → R2; you run this)

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# Pack 227K-file edgar_text dir once (idempotent: only repacks if missing)
if [ -d data/interim/edgar_text ] && [ ! -f data/interim/edgar_text.tar.zst ]; then
  echo "Tarring edgar_text..."
  tar -I 'zstd -T0 -19' -cf data/interim/edgar_text.tar.zst -C data/interim edgar_text
fi

rclone sync data/processed/                  r2:axiom-tilt-data/data/processed/  --progress
rclone sync data/interim/edgar_text.tar.zst  r2:axiom-tilt-data/data/interim/    --progress
rclone sync data/archive/                    r2:axiom-tilt-data/data/archive/    --progress
rclone sync artifacts/finbert-mlm/           r2:axiom-tilt-data/artifacts/finbert-mlm/ \
       --progress --exclude 'checkpoint-*/**'
```

### scripts/sync_from_r2.sh (pull R2 → local; collaborator runs this)

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

rclone copy r2:axiom-tilt-data/data/processed/ data/processed/ --progress
rclone copy r2:axiom-tilt-data/data/interim/   data/interim/   --progress
rclone copy r2:axiom-tilt-data/data/archive/   data/archive/   --progress
rclone copy r2:axiom-tilt-data/artifacts/      artifacts/      --progress

if [ -f data/interim/edgar_text.tar.zst ]; then
  tar -I 'zstd -d' -xf data/interim/edgar_text.tar.zst -C data/interim/
fi
```

## 11. Collaborator onboarding

`docs/collaborator-setup.md` covers:

1. `git clone https://github.com/kavinravi/axiom_tilt.git`
2. `cd axiom_tilt && pip install -e .`
3. Get a WRDS account (via collaborator's home institution).
4. First-time WRDS auth: `python -c "import wrds; wrds.Connection()"` (creates `.pgpass`).
5. Get R2 credentials from project owner (out-of-band: 1Password, Signal — never in repo).
6. Install rclone: `curl https://rclone.org/install.sh | sudo bash`
7. Copy `docs/rclone-r2-template.conf` → `~/.config/rclone/rclone.conf`, fill in the three R2 values.
8. `./scripts/sync_from_r2.sh` (downloads ~70 GB; ~1-3 hr on home internet).
9. Verify:
   ```bash
   python -c "import duckdb; print(duckdb.sql(\"SELECT COUNT(*) FROM 'data/processed/crsp_daily/year=*/*.parquet'\").df())"
   ```

## 12. Implementation order

For the eventual `writing-plans` step:

1. `src/data/ingest_wrds.py` skeleton: imports, CLI argparse, placeholder functions.
2. `tests/data/test_ingest_wrds.py` fixtures for synthetic WRDS responses.
3. Universe resolver (`resolve_universe_ids`) + tests.
4. CRSP daily pull with `msedelist` merge + tests.
5. Compustat funda + fundq pulls + tests.
6. CCM link table pull + tests.
7. Wire all into CLI; integration test for `--all` ordering.
8. `scripts/migrate_to_wrds.sh`; archive yfinance/FMP data; deprecation docstrings.
9. `scripts/sync_to_r2.sh`, `scripts/sync_from_r2.sh`.
10. `docs/wrds-setup.md`, `docs/collaborator-setup.md`, `docs/rclone-r2-template.conf`.
11. Add `wrds` to `requirements.txt`.
12. `.gitignore` entry for `data/raw/edgar/`, `data/processed/crsp_daily/`, etc. (anything not already covered).

## 13. Risks and open questions

1. **WRDS schema drift.** Table names like `crsp.dsf` and `crsp.msedelist` have changed at major schema migrations. Our pulls lock to currently-documented names. If WRDS migrates, add a small probe in `resolve_universe_ids` that queries `information_schema.tables` to discover the actual names, with a fallback list.
2. **WRDS query timeouts.** Very large `WHERE permno IN (...)` clauses (>1000 elements) can time out. Mitigation: chunk the permno list at 500 and concatenate results.
3. **rclone first-sync time.** 70 GB at typical home upload speeds (50 Mbps) ≈ 3 hr. Subsequent syncs are diff-only and fast. Run overnight if needed.
4. **R2 cost.** Storage: $0.015/GB/mo × 70 GB ≈ $1.05/mo. Egress: $0. Operations: trivial for our access pattern. Expected monthly bill: $1-2.
5. **Universe gaps for foreign-listed tickers.** Some S&P 500 members are dual-listed or have ADR variants. CRSP coverage is excellent for US primary listings; document any unmatched rows in `data/processed/universe_unmatched.parquet`.
6. **Compustat coverage gaps.** Small fraction of stocks lack Compustat (e.g., REITs reporting on different fiscal years). Sanity-check after pull: `comp_fundq` row count / (univ rows × quarters in range) ≥ 0.8.
7. **Collaborator WRDS access.** Both researchers need their own WRDS accounts to re-run pulls (no shared credentials). For consuming pulled data, no WRDS access needed.

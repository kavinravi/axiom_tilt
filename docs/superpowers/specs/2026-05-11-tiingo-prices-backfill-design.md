# Tiingo Prices Backfill — Design Spec

**Date:** 2026-05-11
**Status:** Draft, pending user review
**Repo:** `axiom_tilt`
**Related spec:** `docs/superpowers/specs/2026-05-08-text-enhanced-rl-portfolio-design.md`
**Audit motivating this work:** `docs/audits/2026-05-10-data-coverage.md`

## 1. Problem

The data-coverage audit on 2026-05-10 found `prices.parquet` covers only **78.1% of universe tickers** (670 of 858). The 188 missing tickers are almost entirely **delisted historical S&P 500 members** (ABK, AGN, APC, ALXN, APOL, ARNC, ABMD, ALTR, ...) that yfinance silently drops.

This creates **survivorship bias**: the dataset over-represents companies that survived to today and under-represents bankruptcies/acquisitions/take-privates. A backtest on this data produces optimistically biased Sharpe and drawdown numbers — unusable for paper-grade claims.

## 2. Goal

**Maximize price coverage from accessible sources and document residual gaps explicitly** so downstream code can either skip them or note them in writeups.

Hard target: **≥95% of universe tickers, each at ≥99% of their expected universe-interval trading days**, after Tiingo backfill.

Honest caveat: 100% elimination is not achievable from yfinance + Tiingo alone (some delistings predate accessible APIs, some symbols got reused, etc.). The residual is documented in `data/processed/prices_missing.csv`; closing it requires CRSP via WRDS or a similar paid source. See §9.

## 3. Source choice

**Tiingo "Power" plan ($10/mo)** is the cheapest API with full historical coverage including delisted tickers. Selected over:
- yfinance — what we already have; doesn't cover delistings (the problem)
- Polygon Stocks Starter ($29/mo) — works but 3× more expensive for what we need
- CRSP via WRDS — gold standard, free if school subscribes, but blocked on user's school-IT response; can supplement later
- Alpha Vantage — free tier 25 req/day is too tight; premium tier $50/mo is more than Tiingo

Tiingo endpoint: `GET https://api.tiingo.com/tiingo/daily/{ticker}/prices?startDate=YYYY-MM-DD&endDate=YYYY-MM-DD`. Returns JSON list of dicts per trading day with `date, open, high, low, close, volume, adjClose, adjOpen, adjHigh, adjLow, adjVolume, divCash, splitFactor`.

Auth via `Authorization: Token <TIINGO_API_KEY>` header (NOT query string — avoids the key-leak bug we hit with FMP).

## 4. Architecture

One new module `src/data/backfill_prices.py`. Single responsibility: identify yfinance-truncated tickers, pull from Tiingo within each ticker's universe interval, merge into `prices.parquet` with a `source` column tagging provenance per row.

```
universe.parquet          prices.parquet (yfinance, no `source` col)
       │                          │
       └──────────┬───────────────┘
                  ▼
       identify_backfill_targets(threshold=0.99)
                  │
                  ▼ list[(ticker, date_in, date_out)]
       For each ticker:
         TiingoClient.fetch_daily(ticker, date_in, date_out)
         parse_tiingo_response → DataFrame with source="tiingo"
         _append_done(state_file, ticker)   # resumable
                  │
                  ▼
       Merge: existing yfinance (source="yfinance") ∪ Tiingo
       drop_duplicates on (date, ticker) keeping tiingo on conflict
                  │
                  ├──► data/processed/prices.parquet (overwrite, +source col)
                  │
                  └──► compute_residual_gaps(threshold=0.99)
                       │
                       ▼
              data/processed/prices_missing.csv
              (tickers still <99% covered after Tiingo — for CRSP follow-up)
```

## 5. Components

### 5.1 `identify_backfill_targets(universe, prices, threshold=0.99) -> list[BackfillTarget]`

For each ticker in universe:
1. Compute `effective_end = today if pd.isna(date_out) else min(date_out, today)`. (NaT in `date_out` means the ticker is currently still in the universe.)
2. Compute `expected_trading_days` from `date_in` to `effective_end` using a US trading calendar (`pandas_market_calendars.get_calendar("XNYS").schedule(...)` if the package is available; otherwise `pd.bdate_range` as a 5-day-week approximation — slight over-count, but the 99% threshold absorbs it).
3. Compute `actual_trading_days` from `prices.parquet` rows where `ticker == t` AND `date_in <= date <= effective_end`.
4. If `expected_trading_days == 0` (degenerate interval), skip the ticker.
5. If `actual / expected < threshold`, add `BackfillTarget(ticker, date_in, effective_end)` to the target list.

For tickers entirely absent from prices: `actual = 0`, always flagged (unless the interval itself is degenerate).

Returns a dataclass `BackfillTarget(ticker: str, start_date: pd.Timestamp, end_date: pd.Timestamp)`.

This function is pure (no I/O) and is unit-testable with synthetic DataFrames.

### 5.2 `TiingoClient` (dataclass)

Same shape as `EdgarClient` / `FmpClient`:
- Fields: `api_key: str`, `bucket: TokenBucket`, `retry_attempts: int`
- Method: `fetch_daily(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> list[dict]`
- Auth via `Authorization` header (NOT query string)
- Tenacity retry on `requests.RequestException` for 429/5xx
- 404 → raise `FileNotFoundError` (terminal, not retried — same pattern as EDGAR fix)
- Rate limit via shared `TokenBucket`; config default 20 req/sec (Tiingo Power has 50k/day, no req/sec cap published, 20/sec is polite)

### 5.3 `parse_tiingo_response(rows, ticker) -> pd.DataFrame`

Maps Tiingo's field names to our schema:

| Tiingo field | Our column |
|---|---|
| `date` | `date` (cast to datetime, tz-strip, normalize) |
| `open` | `open` |
| `high` | `high` |
| `low` | `low` |
| `close` | `close` |
| `adjClose` | `adj_close` |
| `volume` | `volume` |
| (constant) | `ticker` |
| (constant) | `source = "tiingo"` |

Drops `adjOpen`/`adjHigh`/`adjLow`/`adjVolume`/`divCash`/`splitFactor` to match the existing schema.

Empty input → empty DataFrame with correct columns.

### 5.4 `compute_residual_gaps(universe, prices, threshold=0.99) -> pd.DataFrame`

Same coverage check as §5.1 but RUN POST-MERGE on the final `prices.parquet`. Returns:

| Column | Description |
|---|---|
| `ticker` | universe ticker |
| `cik` | from universe (may be NaN) |
| `expected_days` | trading days in universe interval |
| `actual_days` | rows in merged prices for this ticker |
| `coverage_pct` | `actual / expected` × 100 |
| `reason` | see assignment rules below |

`reason` assignment (in priority order):
- `entirely_missing` — `actual_days == 0` for this ticker after Tiingo backfill (neither source had any data)
- `not_in_tiingo` — ticker was in the backfill target list and Tiingo returned 404 (recorded by `main()` in a separate set passed into this function)
- `partial_tiingo_coverage` — Tiingo returned some rows but final coverage still <99% (gaps within the interval; common for thinly-traded names with halts/suspensions)

Written to `data/processed/prices_missing.csv` (CSV, not parquet — small file, human-readable for follow-up).

### 5.5 `main()`

Orchestration:
1. `configure_logging()`, `load_config("data")`, `get_env("TIINGO_API_KEY", required=True)`
2. Read `universe.parquet` and `prices.parquet`
3. Schema migration: if `prices` has no `source` column, add `source="yfinance"` to all rows (one-time)
4. `targets = identify_backfill_targets(universe, prices, threshold=cfg["backfill"]["coverage_threshold"])`
5. Load resume state from `data/state/backfill_done.txt`; filter targets to remaining
6. Loop over remaining targets with `tqdm`:
   - `client.fetch_daily(ticker, start, end)` (retry-wrapped)
   - `parse_tiingo_response(rows, ticker)` → DataFrame
   - Append to `new_frames` list; `_append_done(state_file, ticker)` on success
   - Log warning on Tiingo 404 (ticker not in Tiingo); continue
7. Concat `new_frames`, merge with existing prices via union-keep-last (Tiingo wins on (date,ticker) conflict)
8. Write merged `prices.parquet`
9. `gaps_df = compute_residual_gaps(...)`; write `prices_missing.csv` if non-empty
10. Log summary: `Backfilled N tickers, M new rows. Coverage now X%. Residual gaps: K tickers in prices_missing.csv`

## 6. Config and env

`.env.example` adds:
```
# Tiingo (https://www.tiingo.com/account/api/token) — for prices backfill.
# Power plan ($10/mo) required for full historical incl. delisted.
TIINGO_API_KEY=""
```

`configs/data.yaml` adds:
```yaml
# Tiingo backfill (fills survivorship-bias gaps in yfinance prices)
backfill:
  rate_per_sec: 20.0
  capacity: 20
  retry_attempts: 3
  coverage_threshold: 0.99   # require ≥99% of universe-interval trading days
```

## 7. Schema migration

Before this work: `prices.parquet` schema is `[date, ticker, open, high, low, close, adj_close, volume]`.

After: `[date, ticker, open, high, low, close, adj_close, volume, source]` where `source ∈ {"yfinance", "tiingo"}`.

Migration happens in `main()` step 3 (one-time, idempotent — re-running is safe).

Downstream consumers (none yet — feature engineering hasn't started) should treat `source` as informational; selecting on it is fine but not required.

## 8. Testing

`tests/data/test_backfill_prices.py` — 4 unit tests:

1. `test_identify_backfill_targets_flags_truncated_ticker`: synthetic universe + prices where ticker `XYZ` has interval 2010-2015 (~1260 trading days expected) but only 800 days in prices → flagged.
2. `test_identify_backfill_targets_skips_complete_ticker`: ticker with ≥99% expected coverage → not flagged.
3. `test_identify_backfill_targets_flags_absent_ticker`: ticker in universe, no rows in prices → flagged with full interval.
4. `test_parse_tiingo_response_maps_schema`: fixture JSON → DataFrame with correct columns and `source="tiingo"`.

Test fixture: `tests/fixtures/tiingo_sample.json` — a saved Tiingo daily-prices response for one ticker (~5 rows). Can be captured with a free Tiingo account (the free tier's daily-prices endpoint works for limited tickers).

No network in tests.

## 9. Expected results

Pre-Tiingo coverage (from audit): 670 / 858 tickers = 78.1%.

Tiingo's claimed historical coverage on delisted US equities is strong. Realistic expectation: **85–95% of universe tickers at ≥99% coverage post-backfill**.

Residual gap (5–15% of universe still ungap-filled) goes to `prices_missing.csv` for follow-up via CRSP/WRDS. Common reasons for residual:
- Very old delistings predating Tiingo's data window
- Symbol changes / M&A renames the ticker-string lookup misses (e.g., ACE → CB)
- OTC/foreign listings Tiingo doesn't carry

§10 lists what would be needed to close the residual.

## 10. Follow-ups (not in this scope)

- **CRSP via WRDS** once school access is granted — would close the last 5-15%.
- **Symbol-rename resolution** — manual map of historical name changes (e.g., `{"ACE": "CB", "AGN": ...}`) to retry Tiingo with renamed symbols. Adds maybe a dozen tickers.
- **`prices_meta.json` sidecar** with global provenance (audit doc follow-up #4): fetch timestamps per source, code commit SHA, schema version. Useful for paper reproducibility.

## 11. MVP scope

Everything in §5 (one module, four functions, `main()` orchestrator) plus §8 tests. Estimated ~150 LOC for the module, ~80 LOC for tests.

**Not in MVP:**
- Symbol-rename resolution (§10)
- Adjusted O/H/L columns (Tiingo provides them; our schema doesn't — keep schema stable)
- Per-row `fetched_at` timestamp (use file-level `_meta.json` later)

## 12. Defaults marked for user review

- Coverage threshold: 99% (1% tolerance for trading-calendar quirks). Tightening to 100% would also work but might thrash on calendar edge cases between sources. Loosening to 95% would miss real gaps.
- Tiingo rate: 20 req/sec, 20 capacity. Polite default; Tiingo Power has no published per-second cap.
- Residual report format: CSV (small, human-readable) rather than parquet.
- Source column values: `"yfinance"` and `"tiingo"` (lowercase, strings). Could be enum-like but YAGNI for now.

## 13. Open decisions

None blocking.

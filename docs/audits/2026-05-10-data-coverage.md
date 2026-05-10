# Data Coverage Audit — 2026-05-10

Cross-coverage check after the data-ingestion phase (plan §6.1). EDGAR full-history pull finished at 08:39:00 on 2026-05-10 in ~8 hours total (226,919 filings).

## Summary

| Source | Status | Coverage | Threshold |
|---|---|---|---|
| Universe | ⚠️ | 884 intervals, 858 unique tickers, 623 unique CIKs (237 tickers without CIK) | n/a |
| Prices | ❌ BELOW | 670 / 858 = **78.1%** of universe tickers | ≥95% |
| EDGAR | ✅ PASS | 614 / 623 = **98.6%** of universe CIKs | ≥90% |
| Macro | ✅ | 4 series, 26,073 rows, 2000-01-03 → 2025-12-31 | n/a |

## Detail

### Universe (`data/processed/universe.parquet`)
- intervals: 884
- unique tickers: 858
- unique CIKs: 623
- tickers without CIK: 237 (mostly delisted historical S&P 500 members not in SEC's current `company_tickers.json`)

### Prices (`data/processed/prices.parquet`)
- rows: 3,604,149
- tickers: 670
- date range: 2000-01-03 → 2025-12-30
- coverage: **78.1% of universe tickers**
- missing sample (delisted-heavy): `ABK, ABMD, ACAS, ACE, ADS, AGN, AKS, ALTR, ALXN, ANR, ANSS, APC, APOL, ARG, ARNC`

### EDGAR (`data/processed/edgar_index.parquet`)
- filings: 226,919
- CIKs: 614
- date range: 2000-01-03 → 2025-12-31
- coverage: **98.6% of universe CIKs**
- form types:
  - 8-K: 174,404
  - 10-Q: 39,761
  - 10-K: 12,754
- missing CIK sample: APC, SE, POM, LIFE, MI, SII, SGP, WB, BUD (mostly foreign listings filing 20-F instead of 10-K, or recent additions that file under different umbrellas)

### Macro (`data/processed/macro.parquet`)
- series: DGS10, DGS3MO, T10Y2Y, VIXCLS
- date range: 2000-01-03 → 2025-12-31
- rows: 26,073

## Findings — survivorship bias risk

The 78.1% prices coverage is the load-bearing concern. Looking at the missing names:

- **ABK** (Ambac Financial) — bankruptcy 2010
- **AGN** (Allergan) — acquired by AbbVie 2020
- **APC** (Anadarko Petroleum) — acquired by Occidental 2019
- **ALXN** (Alexion) — acquired by AstraZeneca 2021
- **APOL** (Apollo Education Group) — went private 2017
- **ARNC** (Arconic) — split into multiple entities
- **ABMD** (Abiomed) — acquired by J&J 2022
- **ALTR** (Altera) — acquired by Intel 2015

These are almost all **delisted historical S&P 500 members** that yfinance silently drops. The universe reconstruction correctly captures their membership intervals (they show up in `universe.parquet`), but we lack price data for their late-life trading windows.

**Why this matters for the research question:** the dataset is currently survivorship-biased toward survivors. A backtest run on this would over-represent good outcomes (because the dead names are missing or have truncated histories), giving optimistically biased Sharpe / drawdown numbers. For a research-paper grade result this needs fixing.

## Follow-ups (priority order)

1. **Prices: swap yfinance → Polygon or CRSP.** Polygon's $29/mo "Stocks Starter" tier covers full historical including delisted; CRSP via WRDS is gold standard but requires academic affiliation. Either provides survivorship-clean OHLCV that yfinance lacks.

2. **Universe: backfill historical CIKs.** SEC publishes `https://www.sec.gov/Archives/edgar/cik-lookup-data.txt` which includes historical filers. Cross-reference with the 237 unmapped tickers — many likely have CIKs that just aren't in the active-filers JSON.

3. **EDGAR: investigate the 9 missing CIKs.** Most look like foreign listings (BUD = AB InBev, SE = Sea Ltd, WB = Weibo) that file 20-F instead of 10-K — these are intentionally outside the form-type filter. Confirm and document so we don't re-investigate later.

4. **Add provenance/freshness to processed parquets.** Each parquet should carry a `_meta.json` sidecar with: source URL, fetch timestamp, code commit SHA. Useful for paper reproducibility.

5. **Fundamentals (still deferred from yesterday).** FMP v3 endpoints deprecated 2025-08-31; see design spec §3.3 for the three resolution options.

## Reproduce this audit

From the repo root:

```bash
python -c "
import pandas as pd
from pathlib import Path
universe = pd.read_parquet('data/processed/universe.parquet')
prices = pd.read_parquet('data/processed/prices.parquet')
macro = pd.read_parquet('data/processed/macro.parquet')
edgar = pd.read_parquet('data/processed/edgar_index.parquet')
uni_tickers = set(universe['ticker']); uni_ciks = set(universe['cik'].dropna().astype(str))
print('prices coverage:', f\"{100*len(set(prices['ticker']) & uni_tickers)/len(uni_tickers):.1f}%\")
print('edgar coverage:',  f\"{100*len(set(edgar['cik'].astype(str)) & uni_ciks)/len(uni_ciks):.1f}%\")
print('macro series:', macro['series'].nunique(), '/ rows:', len(macro))
"
```

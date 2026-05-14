"""Pull point-in-time fundamentals from SEC EDGAR XBRL company-facts API.

Outputs:
  data/raw/edgar_xbrl/CIK{cik:010d}.json   raw JSON per company (debug/reparse)
  data/processed/edgar_xbrl_facts.parquet  parsed long-format facts (all CIKs)

Long format columns:
  (cik, concept, xbrl_tag, xbrl_taxonomy, units, period_start, period_end,
   filed, fiscal_year, fiscal_period, form, accn, value)

`filed` is the actual SEC filing date and is the ONLY column that should be used
for point-in-time joins. `period_end` is the fiscal period the value refers to;
it does NOT tell you when the value became public.

Source: https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json
The SEC's free XBRL API returns every fact a company has ever reported, each
tagged with its filing date — so we get a true PIT view, including amendments
and restatements.

Setup:
  Requires SEC_USER_AGENT in .env (same one used by ingest_filings.py).
  SEC enforces 10 req/sec; we run at 8 req/sec for headroom.

Usage:
  python -m src.data.ingest_edgar_xbrl          # full: download + parse
  python -m src.data.ingest_edgar_xbrl --parse-only   # reparse existing raw JSONs
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

from src.utils.env import get_env
from src.utils.io import processed_dir, raw_dir, repo_root, state_dir
from src.utils.logging_utils import configure_logging, get_logger
from src.utils.rate_limit import TokenBucket


log = get_logger(__name__)

XBRL_BASE = "https://data.sec.gov/api/xbrl/companyfacts"


# Project field -> ordered (taxonomy, tag) candidates. First non-empty match wins.
# `us-gaap` covers most US filers; `ifrs-full` would cover foreign filers (TBD).
# `dei` (Document and Entity Information) is used for some metadata-like facts.
CONCEPT_MAP: dict[str, list[tuple[str, str]]] = {
    # Balance sheet
    "assets_total": [("us-gaap", "Assets")],
    "liabilities_total": [("us-gaap", "Liabilities")],
    "stockholders_equity": [
        ("us-gaap", "StockholdersEquity"),
        ("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
    ],
    "cash_and_equivalents": [
        ("us-gaap", "CashAndCashEquivalentsAtCarryingValue"),
        ("us-gaap", "Cash"),
        ("us-gaap", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"),
    ],
    "long_term_debt": [
        ("us-gaap", "LongTermDebt"),
        ("us-gaap", "LongTermDebtNoncurrent"),
    ],
    "short_term_debt": [
        ("us-gaap", "ShortTermBorrowings"),
        ("us-gaap", "LongTermDebtCurrent"),
    ],
    "current_assets": [("us-gaap", "AssetsCurrent")],
    "current_liabilities": [("us-gaap", "LiabilitiesCurrent")],
    "inventory": [("us-gaap", "InventoryNet")],
    "accounts_receivable": [("us-gaap", "AccountsReceivableNetCurrent")],
    "accounts_payable": [("us-gaap", "AccountsPayableCurrent")],
    # Income statement
    "revenue": [
        ("us-gaap", "Revenues"),
        ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
        ("us-gaap", "SalesRevenueNet"),
        ("us-gaap", "SalesRevenueGoodsNet"),
    ],
    "cogs": [
        ("us-gaap", "CostOfRevenue"),
        ("us-gaap", "CostOfGoodsSold"),
        ("us-gaap", "CostOfGoodsAndServicesSold"),
    ],
    "gross_profit": [("us-gaap", "GrossProfit")],
    "operating_income": [("us-gaap", "OperatingIncomeLoss")],
    "net_income": [("us-gaap", "NetIncomeLoss")],
    "interest_expense": [("us-gaap", "InterestExpense")],
    "income_tax_expense": [("us-gaap", "IncomeTaxExpenseBenefit")],
    "depreciation_amortization": [
        ("us-gaap", "DepreciationDepletionAndAmortization"),
        ("us-gaap", "DepreciationAndAmortization"),
    ],
    "eps_basic": [("us-gaap", "EarningsPerShareBasic")],
    "eps_diluted": [("us-gaap", "EarningsPerShareDiluted")],
    # Cash flow
    "cash_from_operations": [("us-gaap", "NetCashProvidedByUsedInOperatingActivities")],
    "capex": [("us-gaap", "PaymentsToAcquirePropertyPlantAndEquipment")],
    "dividends_paid": [
        ("us-gaap", "PaymentsOfDividends"),
        ("us-gaap", "PaymentsOfDividendsCommonStock"),
    ],
    # Shares
    "shares_outstanding": [
        ("us-gaap", "CommonStockSharesOutstanding"),
        ("dei", "EntityCommonStockSharesOutstanding"),
    ],
}


@retry(
    retry=retry_if_exception_type((requests.RequestException,)),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    stop=stop_after_attempt(5),
)
def _fetch_company_facts(
    cik: int,
    bucket: TokenBucket,
    headers: dict,
    session: requests.Session,
) -> dict | None:
    """GET /companyfacts/CIK{cik:010d}.json. Returns None on 404 (no XBRL for this CIK)."""
    bucket.acquire()
    url = f"{XBRL_BASE}/CIK{cik:010d}.json"
    resp = session.get(url, headers=headers, timeout=30)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def parse_company_facts(facts_json: dict, cik: int) -> pd.DataFrame:
    """Flatten companyfacts JSON to long-format DataFrame, restricted to mapped concepts.

    For each project concept, walks the CONCEPT_MAP candidates in order and uses
    the first taxonomy/tag pair that has data. Each fact entry produces one row.
    """
    rows: list[dict] = []
    facts = facts_json.get("facts", {})

    for project_field, candidates in CONCEPT_MAP.items():
        for taxonomy, tag in candidates:
            tag_dict = facts.get(taxonomy, {}).get(tag)
            if not tag_dict:
                continue
            units = tag_dict.get("units", {})
            for unit, entries in units.items():
                for entry in entries:
                    rows.append(
                        {
                            "cik": cik,
                            "concept": project_field,
                            "xbrl_tag": tag,
                            "xbrl_taxonomy": taxonomy,
                            "units": unit,
                            "period_start": entry.get("start"),
                            "period_end": entry.get("end"),
                            "filed": entry.get("filed"),
                            "fiscal_year": entry.get("fy"),
                            "fiscal_period": entry.get("fp"),
                            "form": entry.get("form"),
                            "accn": entry.get("accn"),
                            "value": entry.get("val"),
                        }
                    )
            break  # first matching candidate wins; don't double-count this concept

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    for col in ("period_start", "period_end", "filed"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def pull_xbrl_for_universe(
    universe_ids_path: Path,
    raw_output_dir: Path,
    state_path: Path,
    user_agent: str,
    rate_per_sec: float = 8.0,
    capacity: int = 8,
) -> int:
    """Pull XBRL companyfacts for every CIK in universe_ids.parquet.

    Resumable: appends each successfully-processed CIK to state_path. Re-running
    skips CIKs already in the state file. Returns the count of CIKs that had no
    XBRL data (404 from SEC).
    """
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    state_path.parent.mkdir(parents=True, exist_ok=True)

    universe = pd.read_parquet(universe_ids_path)
    ciks = sorted(universe["cik"].dropna().astype(int).unique().tolist())
    log.info("Universe has %d unique CIKs", len(ciks))

    done: set[int] = set()
    if state_path.exists():
        done = {
            int(line.strip())
            for line in state_path.read_text().splitlines()
            if line.strip()
        }
        log.info("Resume: %d CIKs already done", len(done))

    todo = [c for c in ciks if c not in done]
    log.info("Pulling XBRL for %d CIKs (skipping %d)", len(todo), len(done))

    bucket = TokenBucket(rate_per_sec=rate_per_sec, capacity=capacity)
    headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
    session = requests.Session()

    no_xbrl = 0
    with state_path.open("a") as state_f:
        for cik in tqdm(todo, desc="XBRL pull"):
            try:
                facts = _fetch_company_facts(cik, bucket, headers, session)
            except Exception as e:
                log.warning("CIK %010d failed: %s", cik, e)
                continue

            if facts is None:
                no_xbrl += 1
                state_f.write(f"{cik}\n")
                state_f.flush()
                continue

            raw_path = raw_output_dir / f"CIK{cik:010d}.json"
            raw_path.write_text(json.dumps(facts))
            state_f.write(f"{cik}\n")
            state_f.flush()

    log.info("Pull complete. CIKs with no XBRL data: %d", no_xbrl)
    return no_xbrl


def parse_all_raw(raw_output_dir: Path, parsed_output_path: Path) -> int:
    """Parse all raw JSONs in raw_output_dir into one long-format parquet."""
    parsed_output_path.parent.mkdir(parents=True, exist_ok=True)

    dfs: list[pd.DataFrame] = []
    files = sorted(raw_output_dir.glob("CIK*.json"))
    if not files:
        log.warning("No raw JSONs found in %s; nothing to parse", raw_output_dir)
        return 0

    for f in tqdm(files, desc="parsing"):
        cik = int(f.stem.replace("CIK", ""))
        try:
            with f.open() as fh:
                facts = json.load(fh)
            df = parse_company_facts(facts, cik)
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            log.warning("Parse failed for %s: %s", f.name, e)

    if not dfs:
        log.warning("No parsed data — output not written.")
        return 0

    combined = pd.concat(dfs, ignore_index=True)
    combined.to_parquet(parsed_output_path, index=False)
    log.info(
        "Wrote %d rows for %d CIKs to %s",
        len(combined),
        combined["cik"].nunique(),
        parsed_output_path,
    )
    return len(combined)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull SEC EDGAR XBRL fundamentals for the universe")
    parser.add_argument(
        "--parse-only",
        action="store_true",
        help="Skip download; re-parse existing raw JSONs into parquet only.",
    )
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = repo_root() / "logs" / f"edgar_xbrl_{ts}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    configure_logging(log_file=log_file)
    log.info("EDGAR XBRL ingest starting (mode=%s)", "parse-only" if args.parse_only else "full")
    log.info("Log file: %s", log_file)

    raw_xbrl_dir = raw_dir() / "edgar_xbrl"
    parsed_path = processed_dir() / "edgar_xbrl_facts.parquet"
    state_path = state_dir() / "edgar_xbrl_done.txt"
    universe_ids_path = processed_dir() / "universe_ids.parquet"

    if args.parse_only:
        parse_all_raw(raw_xbrl_dir, parsed_path)
        return

    if not universe_ids_path.exists():
        raise SystemExit(
            f"{universe_ids_path} missing — run `python -m src.data.ingest_wrds --resolve-only` first."
        )

    user_agent = get_env("SEC_USER_AGENT", required=True)
    pull_xbrl_for_universe(
        universe_ids_path=universe_ids_path,
        raw_output_dir=raw_xbrl_dir,
        state_path=state_path,
        user_agent=user_agent,
    )
    parse_all_raw(raw_xbrl_dir, parsed_path)


if __name__ == "__main__":
    main()

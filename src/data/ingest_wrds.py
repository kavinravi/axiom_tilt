"""Pull CRSP + Compustat from Wharton WRDS, replacing yfinance/FMP.

Outputs:
  data/processed/universe_ids.parquet          (ticker -> permno + gvkey resolution)
  data/processed/crsp_daily/year=YYYY/part-0.parquet
                                              (daily prices + delisting returns)
  data/processed/comp_funda.parquet            (annual Compustat, PIT filters)
  data/processed/comp_fundq.parquet            (quarterly Compustat, PIT filters)
  data/processed/ccm_linktable.parquet         (CRSP/Compustat link table)

Setup:
  1. WRDS account: pip install wrds; python -c "import wrds; wrds.Connection()"
     The package prompts for username/password and offers to write ~/.pgpass.
  2. Set WRDS_USERNAME in .env.

Usage:
  python -m src.data.ingest_wrds --all
  python -m src.data.ingest_wrds --resolve-only
  python -m src.data.ingest_wrds --crsp-only
  python -m src.data.ingest_wrds --compustat-only
  python -m src.data.ingest_wrds --linktable-only
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

from src.utils.env import get_env
from src.utils.io import processed_dir, repo_root
from src.utils.logging_utils import configure_logging, get_logger


log = get_logger(__name__)

# Standard Compustat point-in-time filter set.
# Reference: WRDS Compustat user guide; this is the same filter used in the
# Fama-French, Hou-Xue-Zhang, and Frazzini-Pedersen papers.
COMPUSTAT_PIT_FILTERS = (
    "consol='C' AND indfmt='INDL' AND datafmt='STD' "
    "AND popsrc='D' AND curcd='USD'"
)

# CRSP/Compustat link filters — primary + linked link types only.
CCM_LINK_FILTERS = "linkprim IN ('P', 'C') AND linktype IN ('LU', 'LC')"

# Minimum fraction of universe rows that must resolve to a permno.
# Below this, halt so the user can investigate ticker mismatches.
MIN_UNIVERSE_MATCH_RATE = 0.95

# Compustat schemas to try in priority order. `comp` is the premium daily refresh
# (= comp_na_daily_all) which requires an expensive subscription tier. Most schools
# only subscribe to the annual snapshot.
COMPUSTAT_SCHEMA_CANDIDATES = [
    "comp",
    "comp_na_annual_all",
    "comp_na_monthly_all",
    "compa",
    "comp_a",
]


def _chunk(seq: list, n: int) -> Iterator[list]:
    """Yield successive n-sized chunks from seq."""
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def _sql_in_list(values: Iterable, quote: bool = False) -> str:
    """Format values as a comma-separated SQL IN-clause body.

    quote=True for string values (wraps each in single quotes).
    quote=False for numeric values.
    """
    if quote:
        return ",".join(f"'{v}'" for v in values)
    return ",".join(str(v) for v in values)


def _interval_overlap_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    key: str,
    left_start: str,
    left_end: str,
    right_start: str,
    right_end: str,
) -> pd.DataFrame:
    """Inner join `left` and `right` on `key` where their date intervals overlap.

    Open-ended (NaT) right_end is treated as infinity.
    """
    merged = left.merge(right, on=key, how="left")
    sentinel = pd.Timestamp("2099-12-31")
    re = merged[right_end].fillna(sentinel)
    rs = merged[right_start]
    ls = merged[left_start]
    le = merged[left_end]
    overlap = (ls <= re) & (le >= rs)
    return merged[overlap].copy()


def resolve_universe_ids(
    universe: pd.DataFrame,
    conn,
) -> pd.DataFrame:
    """Resolve (ticker, cik, date_in, date_out) -> (permno, gvkey).

    Joins universe -> crsp.stocknames by ticker + date overlap, then
    permno -> crsp.ccmxpf_lnkhist by date overlap to get gvkey.

    Returns the FULL universe with resolved permno (nullable Int64) and
    gvkey (nullable str). Rows that cannot be resolved keep their original
    fields with permno/gvkey = NA.

    Raises RuntimeError if fewer than MIN_UNIVERSE_MATCH_RATE of rows
    resolve to a permno.
    """
    log.info("Resolving %d universe rows to permno/gvkey", len(universe))

    universe = universe.copy()
    universe["date_in"] = pd.to_datetime(universe["date_in"])
    universe["date_out"] = pd.to_datetime(universe["date_out"])
    universe = universe.reset_index(drop=True)
    universe["_row_idx"] = universe.index

    sentinel = pd.Timestamp("2099-12-31")

    # --- Step 1: ticker -> permno via crsp.stocknames -----------------------
    tickers = sorted(universe["ticker"].dropna().unique().tolist())
    ticker_sql = _sql_in_list(tickers, quote=True)
    stocknames = conn.raw_sql(
        f"""
        SELECT permno, ticker, namedt, nameenddt
        FROM crsp.stocknames
        WHERE ticker IN ({ticker_sql})
        """,
        date_cols=["namedt", "nameenddt"],
    )
    matched_tickers = stocknames["ticker"].nunique() if len(stocknames) else 0
    log.info(
        "crsp.stocknames returned %d rows; %d/%d universe tickers had at least one stocknames row",
        len(stocknames),
        matched_tickers,
        len(tickers),
    )

    # LEFT JOIN universe to stocknames on ticker; compute date-interval overlap.
    sn_merged = universe.merge(stocknames, on="ticker", how="left")
    rs = sn_merged["namedt"]
    re = sn_merged["nameenddt"].fillna(sentinel)
    ls = sn_merged["date_in"]
    le = sn_merged["date_out"]
    interval_start = pd.concat([ls, rs], axis=1).max(axis=1)
    interval_end = pd.concat([le, re], axis=1).min(axis=1)
    sn_merged["_overlap"] = (interval_end - interval_start).dt.days

    # Keep only rows with strictly positive overlap (excludes both
    # no-ticker-match rows where namedt is NaT, and rows where intervals miss).
    sn_valid = sn_merged[sn_merged["_overlap"] > 0].copy()
    # For each universe row (_row_idx), pick the stocknames row with the
    # largest overlap (handles ticker reuse across distinct companies).
    sn_best = (
        sn_valid.sort_values("_overlap", ascending=False)
        .drop_duplicates(subset="_row_idx", keep="first")[["_row_idx", "permno"]]
    )

    with_permno = universe.merge(sn_best, on="_row_idx", how="left")
    permno_resolved = with_permno["permno"].notna().sum()
    log.info(
        "After stocknames overlap-join: %d/%d universe rows have a permno",
        permno_resolved,
        len(universe),
    )

    # --- Step 2: permno -> gvkey via crsp.ccmxpf_lnkhist --------------------
    permnos = sorted(with_permno["permno"].dropna().astype(int).unique().tolist())
    if permnos:
        permno_sql = _sql_in_list(permnos)
        lnkhist = conn.raw_sql(
            f"""
            SELECT gvkey, lpermno AS permno, linktype, linkprim, linkdt, linkenddt
            FROM crsp.ccmxpf_lnkhist
            WHERE lpermno IN ({permno_sql})
              AND {CCM_LINK_FILTERS}
            """,
            date_cols=["linkdt", "linkenddt"],
        )
        log.info(
            "ccmxpf_lnkhist returned %d rows for %d permnos",
            len(lnkhist),
            len(permnos),
        )
    else:
        lnkhist = pd.DataFrame()

    if not lnkhist.empty:
        lnkhist["permno"] = lnkhist["permno"].astype(int)
        wp_with_permno = with_permno.dropna(subset=["permno"]).copy()
        wp_with_permno["permno"] = wp_with_permno["permno"].astype(int)
        lk_merged = wp_with_permno.merge(lnkhist, on="permno", how="left")
        rs = lk_merged["linkdt"]
        re = lk_merged["linkenddt"].fillna(sentinel)
        ls = lk_merged["date_in"]
        le = lk_merged["date_out"]
        interval_start = pd.concat([ls, rs], axis=1).max(axis=1)
        interval_end = pd.concat([le, re], axis=1).min(axis=1)
        lk_merged["_overlap"] = (interval_end - interval_start).dt.days
        lk_valid = lk_merged[lk_merged["_overlap"] > 0].copy()
        lk_best = (
            lk_valid.sort_values("_overlap", ascending=False)
            .drop_duplicates(subset="_row_idx", keep="first")[["_row_idx", "gvkey"]]
        )
        result = with_permno.merge(lk_best, on="_row_idx", how="left")
    else:
        result = with_permno.copy()
        result["gvkey"] = pd.NA

    result = result.drop(columns=["_row_idx"])
    keep_cols = [
        c
        for c in [
            "ticker",
            "cik",
            "company",
            "date_in",
            "date_out",
            "permno",
            "gvkey",
        ]
        if c in result.columns
    ]
    result = result[keep_cols]

    resolved = result["permno"].notna().sum()
    match_rate = resolved / len(universe) if len(universe) else 0.0
    log.info(
        "Resolution summary: %d/%d universe rows -> permno (%.1f%%); %d -> gvkey",
        resolved,
        len(universe),
        match_rate * 100,
        result["gvkey"].notna().sum(),
    )

    if match_rate < MIN_UNIVERSE_MATCH_RATE:
        unresolved = result[result["permno"].isna()][["ticker", "date_in", "date_out"]]
        log.error(
            "Unresolved universe rows: %d. Sample (first 30 tickers): %s",
            len(unresolved),
            unresolved["ticker"].tolist()[:30],
        )
        # Save the full unresolved list to disk for inspection.
        unresolved_path = repo_root() / "logs" / "wrds_unresolved_tickers.csv"
        unresolved_path.parent.mkdir(parents=True, exist_ok=True)
        unresolved.to_csv(unresolved_path, index=False)
        log.error("Full unresolved list written to %s", unresolved_path)
        raise RuntimeError(
            f"Universe permno resolution rate {match_rate:.1%} < "
            f"{MIN_UNIVERSE_MATCH_RATE:.0%}. "
            f"Inspect {unresolved_path} to see which tickers/periods didn't match."
        )

    return result


def pull_crsp_daily(
    conn,
    permnos: list[int],
    start: str,
    end: str,
    output_dir: Path,
    chunk_size: int = 500,
) -> None:
    """Pull CRSP daily prices + returns, merge delisting returns, year-partitioned parquet.

    Writes one parquet per year to {output_dir}/year=YYYY/part-0.parquet.
    Re-running skips years whose partition already exists (resume-friendly).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    start_year = int(start.split("-")[0])
    end_year = int(end.split("-")[0])

    pending_years = [
        y for y in range(start_year, end_year + 1)
        if not (output_dir / f"year={y}" / "part-0.parquet").exists()
    ]
    if not pending_years:
        log.info("All year partitions already exist in %s — nothing to do.", output_dir)
        return

    permno_sql = _sql_in_list(permnos)
    msedelist = conn.raw_sql(
        f"""
        SELECT permno, dlstdt AS date, dlret, dlstcd
        FROM crsp.msedelist
        WHERE permno IN ({permno_sql})
        """,
        date_cols=["date"],
    )
    log.info("crsp.msedelist returned %d delisting rows", len(msedelist))

    for year in pending_years:
        out_part = output_dir / f"year={year}" / "part-0.parquet"

        out_part.parent.mkdir(parents=True, exist_ok=True)
        year_start = max(f"{year}-01-01", start)
        year_end = min(f"{year}-12-31", end)

        dfs = []
        for chunk in _chunk(permnos, chunk_size):
            chunk_sql = _sql_in_list(chunk)
            df = conn.raw_sql(
                f"""
                SELECT permno, date, prc, ret, vol, shrout,
                       openprc, askhi, bidlo, cfacpr, cfacshr
                FROM crsp.dsf
                WHERE permno IN ({chunk_sql})
                  AND date BETWEEN '{year_start}' AND '{year_end}'
                """,
                date_cols=["date"],
            )
            dfs.append(df)

        year_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        if year_df.empty:
            log.warning("year=%d had zero CRSP rows", year)
            continue

        # Merge delisting returns onto the matching (permno, date) row.
        if not msedelist.empty:
            year_df = year_df.merge(
                msedelist[["permno", "date", "dlret", "dlstcd"]],
                on=["permno", "date"],
                how="left",
            )
        else:
            year_df["dlret"] = pd.NA
            year_df["dlstcd"] = pd.NA

        year_df.to_parquet(out_part, index=False)
        log.info("year=%d: %d rows -> %s", year, len(year_df), out_part)


def detect_compustat_schema(conn) -> str:
    """Find which Compustat schema this WRDS account can read.

    Tries each candidate in COMPUSTAT_SCHEMA_CANDIDATES with a 1-row probe.
    Returns the first one that works; raises if none do.
    """
    for schema in COMPUSTAT_SCHEMA_CANDIDATES:
        try:
            conn.raw_sql(f"SELECT 1 FROM {schema}.funda LIMIT 1")
            log.info("Compustat: using schema '%s'", schema)
            return schema
        except Exception as e:
            first_line = str(e).strip().splitlines()[0] if str(e).strip() else "(empty)"
            log.info("Compustat schema '%s' inaccessible: %s", schema, first_line[:160])
            continue
    raise RuntimeError(
        f"No accessible Compustat schema found in {COMPUSTAT_SCHEMA_CANDIDATES}. "
        "Run `conn.list_libraries()` interactively to see what's available, then "
        "add the right one to COMPUSTAT_SCHEMA_CANDIDATES."
    )


def pull_compustat_funda(
    conn,
    gvkeys: list[str],
    start: str,
    end: str,
    output_path: Path,
    chunk_size: int = 500,
    schema: str = "comp",
) -> None:
    """Pull Compustat annual fundamentals with standard PIT filters."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dfs = []
    for chunk in _chunk(gvkeys, chunk_size):
        gvkey_sql = _sql_in_list(chunk, quote=True)
        df = conn.raw_sql(
            f"""
            SELECT *
            FROM {schema}.funda
            WHERE {COMPUSTAT_PIT_FILTERS}
              AND gvkey IN ({gvkey_sql})
              AND datadate BETWEEN '{start}' AND '{end}'
            """,
            date_cols=["datadate", "rdq"],
        )
        dfs.append(df)

    funda = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    funda.to_parquet(output_path, index=False)
    log.info("%s.funda: %d rows -> %s", schema, len(funda), output_path)


def pull_compustat_fundq(
    conn,
    gvkeys: list[str],
    start: str,
    end: str,
    output_path: Path,
    chunk_size: int = 500,
    schema: str = "comp",
) -> None:
    """Pull Compustat quarterly fundamentals with standard PIT filters."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dfs = []
    for chunk in _chunk(gvkeys, chunk_size):
        gvkey_sql = _sql_in_list(chunk, quote=True)
        df = conn.raw_sql(
            f"""
            SELECT *
            FROM {schema}.fundq
            WHERE {COMPUSTAT_PIT_FILTERS}
              AND gvkey IN ({gvkey_sql})
              AND datadate BETWEEN '{start}' AND '{end}'
            """,
            date_cols=["datadate", "rdq"],
        )
        dfs.append(df)

    fundq = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    fundq.to_parquet(output_path, index=False)
    log.info("%s.fundq: %d rows -> %s", schema, len(fundq), output_path)


def pull_ccm_linktable(conn, output_path: Path) -> None:
    """Pull the full CRSP/Compustat link table (~30K rows, no filter needed)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = conn.raw_sql(
        """
        SELECT gvkey, lpermno, lpermco, linktype, linkprim, linkdt, linkenddt
        FROM crsp.ccmxpf_linktable
        """,
        date_cols=["linkdt", "linkenddt"],
    )
    df.to_parquet(output_path, index=False)
    log.info("ccm_linktable: %d rows -> %s", len(df), output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull WRDS data for the project universe")
    parser.add_argument("--start", default="1995-01-01", help="Pull start date (default 1995-01-01)")
    parser.add_argument("--end", default="2025-12-31", help="Pull end date (default 2025-12-31)")

    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--all", action="store_true", help="Run linktable, resolve, CRSP, Compustat")
    grp.add_argument("--resolve-only", action="store_true", help="Just universe -> permno/gvkey")
    grp.add_argument("--crsp-only", action="store_true", help="Just CRSP daily + delisting")
    grp.add_argument("--compustat-only", action="store_true", help="Just Compustat funda + fundq")
    grp.add_argument("--linktable-only", action="store_true", help="Just CCM link table")

    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = repo_root() / "logs" / f"wrds_ingest_{ts}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    configure_logging(log_file=log_file)

    log.info("WRDS ingest starting: start=%s end=%s mode=%s", args.start, args.end, _selected_mode(args))
    log.info("Log file: %s", log_file)

    import wrds  # local import so tests don't require the package installed

    wrds_username = get_env("WRDS_USERNAME", required=True)
    log.info("Connecting to WRDS as %s", wrds_username)
    conn = wrds.Connection(wrds_username=wrds_username)

    try:
        out_dir = processed_dir()
        ccm_path = out_dir / "ccm_linktable.parquet"
        ids_path = out_dir / "universe_ids.parquet"
        crsp_dir = out_dir / "crsp_daily"
        funda_path = out_dir / "comp_funda.parquet"
        fundq_path = out_dir / "comp_fundq.parquet"

        run_linktable = args.all or args.linktable_only
        run_resolve = args.all or args.resolve_only
        run_crsp = args.all or args.crsp_only
        run_compustat = args.all or args.compustat_only

        if run_linktable:
            pull_ccm_linktable(conn, ccm_path)

        if run_resolve:
            universe = pd.read_parquet(out_dir / "universe.parquet")
            ids = resolve_universe_ids(universe, conn)
            ids.to_parquet(ids_path, index=False)
            log.info("universe_ids: %d rows -> %s", len(ids), ids_path)

        if run_crsp or run_compustat:
            if not ids_path.exists():
                raise SystemExit(
                    f"{ids_path} missing — run with --resolve-only or --all first."
                )
            ids = pd.read_parquet(ids_path)
            permnos = sorted(ids["permno"].dropna().astype(int).unique().tolist())
            gvkeys = sorted(ids["gvkey"].dropna().astype(str).unique().tolist())
            log.info("Resolved IDs in scope: %d permnos, %d gvkeys", len(permnos), len(gvkeys))

            if run_crsp:
                pull_crsp_daily(conn, permnos, args.start, args.end, crsp_dir)
            if run_compustat:
                schema = detect_compustat_schema(conn)
                pull_compustat_funda(conn, gvkeys, args.start, args.end, funda_path, schema=schema)
                pull_compustat_fundq(conn, gvkeys, args.start, args.end, fundq_path, schema=schema)

        log.info("WRDS ingest complete.")
    finally:
        conn.close()


def _selected_mode(args: argparse.Namespace) -> str:
    if args.all:
        return "all"
    if args.resolve_only:
        return "resolve-only"
    if args.crsp_only:
        return "crsp-only"
    if args.compustat_only:
        return "compustat-only"
    if args.linktable_only:
        return "linktable-only"
    return "unknown"


if __name__ == "__main__":
    main()

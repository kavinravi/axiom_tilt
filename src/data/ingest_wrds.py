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

    Returns the universe with resolved permno (nullable Int64) and gvkey (nullable str).
    Raises RuntimeError if fewer than MIN_UNIVERSE_MATCH_RATE of rows resolve.
    """
    log.info("Resolving %d universe rows to permno/gvkey", len(universe))

    universe = universe.copy()
    universe["date_in"] = pd.to_datetime(universe["date_in"])
    universe["date_out"] = pd.to_datetime(universe["date_out"])

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
    log.info(
        "crsp.stocknames returned %d rows for %d unique tickers",
        len(stocknames),
        len(tickers),
    )

    if stocknames.empty:
        raise RuntimeError(
            "crsp.stocknames returned zero rows — check tickers or WRDS access."
        )

    # Universe x stocknames: overlap join on ticker + date interval.
    u2p = _interval_overlap_join(
        left=universe,
        right=stocknames,
        key="ticker",
        left_start="date_in",
        left_end="date_out",
        right_start="namedt",
        right_end="nameenddt",
    )
    # Multiple stocknames rows can match (ticker reuse across periods). Pick the
    # one whose interval-overlap is largest with the universe row.
    sentinel = pd.Timestamp("2099-12-31")
    re = u2p["nameenddt"].fillna(sentinel)
    overlap_days = (
        u2p[["date_out", "nameenddt"]]
        .assign(re=re)
        .apply(lambda r: min(r["date_out"], r["re"]), axis=1)
        - u2p[["date_in", "namedt"]].max(axis=1)
    ).dt.days.clip(lower=0)
    u2p = u2p.assign(_overlap=overlap_days)
    u2p = (
        u2p.sort_values("_overlap", ascending=False)
        .drop_duplicates(subset=["ticker", "date_in", "date_out"], keep="first")
        .drop(columns=["_overlap", "namedt", "nameenddt"])
    )

    permnos = sorted(u2p["permno"].dropna().astype(int).unique().tolist())
    if not permnos:
        raise RuntimeError("No permnos resolved from stocknames join.")
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
    log.info("ccmxpf_lnkhist returned %d rows for %d permnos", len(lnkhist), len(permnos))

    if not lnkhist.empty:
        # u2p has (ticker, date_in, date_out, permno). Join with lnkhist on permno + date overlap.
        u2p["permno"] = u2p["permno"].astype(int)
        lnkhist["permno"] = lnkhist["permno"].astype(int)
        p2g = _interval_overlap_join(
            left=u2p,
            right=lnkhist,
            key="permno",
            left_start="date_in",
            left_end="date_out",
            right_start="linkdt",
            right_end="linkenddt",
        )
        p2g = p2g.drop_duplicates(
            subset=["ticker", "date_in", "date_out"], keep="first"
        )
        # Reattach unmatched (permno-resolved but no gvkey) rows so we don't lose them.
        unmatched = u2p[
            ~u2p.set_index(["ticker", "date_in", "date_out"]).index.isin(
                p2g.set_index(["ticker", "date_in", "date_out"]).index
            )
        ].copy()
        unmatched["gvkey"] = pd.NA
        result = pd.concat([p2g, unmatched], ignore_index=True)
    else:
        result = u2p.copy()
        result["gvkey"] = pd.NA

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
        "Resolution: %d/%d universe rows -> permno (%.1f%%); %d -> gvkey",
        resolved,
        len(universe),
        match_rate * 100,
        result["gvkey"].notna().sum(),
    )

    if match_rate < MIN_UNIVERSE_MATCH_RATE:
        unresolved = result[result["permno"].isna()]["ticker"].tolist()
        log.error("Unresolved tickers: %s", unresolved[:20])
        raise RuntimeError(
            f"Universe permno resolution rate {match_rate:.1%} < "
            f"{MIN_UNIVERSE_MATCH_RATE:.0%}. Check ticker spellings or WRDS access."
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


def pull_compustat_funda(
    conn,
    gvkeys: list[str],
    start: str,
    end: str,
    output_path: Path,
    chunk_size: int = 500,
) -> None:
    """Pull Compustat annual fundamentals with standard PIT filters."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dfs = []
    for chunk in _chunk(gvkeys, chunk_size):
        gvkey_sql = _sql_in_list(chunk, quote=True)
        df = conn.raw_sql(
            f"""
            SELECT *
            FROM comp.funda
            WHERE {COMPUSTAT_PIT_FILTERS}
              AND gvkey IN ({gvkey_sql})
              AND datadate BETWEEN '{start}' AND '{end}'
            """,
            date_cols=["datadate", "rdq"],
        )
        dfs.append(df)

    funda = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    funda.to_parquet(output_path, index=False)
    log.info("comp.funda: %d rows -> %s", len(funda), output_path)


def pull_compustat_fundq(
    conn,
    gvkeys: list[str],
    start: str,
    end: str,
    output_path: Path,
    chunk_size: int = 500,
) -> None:
    """Pull Compustat quarterly fundamentals with standard PIT filters."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dfs = []
    for chunk in _chunk(gvkeys, chunk_size):
        gvkey_sql = _sql_in_list(chunk, quote=True)
        df = conn.raw_sql(
            f"""
            SELECT *
            FROM comp.fundq
            WHERE {COMPUSTAT_PIT_FILTERS}
              AND gvkey IN ({gvkey_sql})
              AND datadate BETWEEN '{start}' AND '{end}'
            """,
            date_cols=["datadate", "rdq"],
        )
        dfs.append(df)

    fundq = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    fundq.to_parquet(output_path, index=False)
    log.info("comp.fundq: %d rows -> %s", len(fundq), output_path)


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
                pull_compustat_funda(conn, gvkeys, args.start, args.end, funda_path)
                pull_compustat_fundq(conn, gvkeys, args.start, args.end, fundq_path)

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

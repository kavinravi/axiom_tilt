"""Pull CRSP daily prices from Wharton WRDS, plus the ticker->permno crosswalk.

Outputs:
  data/processed/universe_ids.parquet          (ticker -> permno resolution)
  data/processed/crsp_daily/year=YYYY/part-0.parquet
                                              (daily prices + delisting returns)

CRSP is the project's point-in-time price source for the full survivorship-bias-free
universe (live and delisted names). Fundamentals come from Sharadar SF1 — see
src/data/ingest_sharadar.py; this module does not touch fundamentals.

Setup:
  1. WRDS account: pip install wrds; python -c "import wrds; wrds.Connection()"
     The package prompts for username/password and offers to write ~/.pgpass.
  2. Set WRDS_USERNAME in .env.

Usage:
  python -m src.data.ingest_wrds --all            # resolve + CRSP daily
  python -m src.data.ingest_wrds --resolve-only    # just universe -> permno
  python -m src.data.ingest_wrds --crsp-only       # just CRSP daily + delisting
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


def resolve_universe_ids(universe: pd.DataFrame, conn) -> pd.DataFrame:
    """Resolve (ticker, cik, date_in, date_out) -> permno via crsp.stocknames.

    Joins universe -> crsp.stocknames by ticker + date-interval overlap. Returns
    the FULL universe with a resolved permno (nullable Int64); rows that cannot
    be resolved keep their original fields with permno = NA.

    Raises RuntimeError if fewer than MIN_UNIVERSE_MATCH_RATE of rows resolve.
    """
    log.info("Resolving %d universe rows to permno", len(universe))

    universe = universe.copy()
    universe["date_in"] = pd.to_datetime(universe["date_in"])
    universe["date_out"] = pd.to_datetime(universe["date_out"])
    universe = universe.reset_index(drop=True)
    universe["_row_idx"] = universe.index

    sentinel = pd.Timestamp("2099-12-31")

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

    result = universe.merge(sn_best, on="_row_idx", how="left")
    result = result.drop(columns=["_row_idx"])
    keep_cols = [
        c
        for c in ["ticker", "cik", "company", "date_in", "date_out", "permno"]
        if c in result.columns
    ]
    result = result[keep_cols]

    resolved = result["permno"].notna().sum()
    match_rate = resolved / len(universe) if len(universe) else 0.0
    log.info(
        "Resolution summary: %d/%d universe rows -> permno (%.1f%%)",
        resolved,
        len(universe),
        match_rate * 100,
    )

    if match_rate < MIN_UNIVERSE_MATCH_RATE:
        unresolved = result[result["permno"].isna()][["ticker", "date_in", "date_out"]]
        log.error(
            "Unresolved universe rows: %d. Sample (first 30 tickers): %s",
            len(unresolved),
            unresolved["ticker"].tolist()[:30],
        )
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull WRDS CRSP data for the project universe")
    parser.add_argument("--start", default="1995-01-01", help="Pull start date (default 1995-01-01)")
    parser.add_argument("--end", default="2025-12-31", help="Pull end date (default 2025-12-31)")

    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--all", action="store_true", help="Run resolve + CRSP")
    grp.add_argument("--resolve-only", action="store_true", help="Just universe -> permno")
    grp.add_argument("--crsp-only", action="store_true", help="Just CRSP daily + delisting")

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
        ids_path = out_dir / "universe_ids.parquet"
        crsp_dir = out_dir / "crsp_daily"

        run_resolve = args.all or args.resolve_only
        run_crsp = args.all or args.crsp_only

        if run_resolve:
            universe = pd.read_parquet(out_dir / "universe.parquet")
            ids = resolve_universe_ids(universe, conn)
            ids.to_parquet(ids_path, index=False)
            log.info("universe_ids: %d rows -> %s", len(ids), ids_path)

        if run_crsp:
            if not ids_path.exists():
                raise SystemExit(
                    f"{ids_path} missing — run with --resolve-only or --all first."
                )
            ids = pd.read_parquet(ids_path)
            permnos = sorted(ids["permno"].dropna().astype(int).unique().tolist())
            log.info("Resolved IDs in scope: %d permnos", len(permnos))
            pull_crsp_daily(conn, permnos, args.start, args.end, crsp_dir)

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
    return "unknown"


if __name__ == "__main__":
    main()

"""Inspect Compustat-sample libraries to see if any usable fundamentals data exists.

Prints table lists, row counts, and date coverage for each of:
  compsamp, compsamp_all, compsamp_computext, compsamp_snapshot
"""
from __future__ import annotations

import os
import sys

import wrds


CANDIDATE_LIBS = ["compsamp", "compsamp_all", "compsamp_computext", "compsamp_snapshot"]
CANDIDATE_TABLES = ["funda", "fundq", "company", "names"]


def main() -> None:
    username = os.environ.get("WRDS_USERNAME")
    if not username:
        from src.utils.env import get_env

        username = get_env("WRDS_USERNAME", required=True)

    conn = wrds.Connection(wrds_username=username)

    for lib in CANDIDATE_LIBS:
        print()
        print(f"=== Tables in {lib} ===")
        try:
            tables = sorted(conn.list_tables(lib))
            for t in tables:
                print(f"  {t}")
            print(f"  ({len(tables)} total)")
        except Exception as e:
            print(f"  (error: {str(e).splitlines()[0][:120]})")

    print()
    print("=== Row counts and date coverage ===")
    for lib in CANDIDATE_LIBS:
        for tbl in CANDIDATE_TABLES:
            try:
                count = conn.raw_sql(f"SELECT COUNT(*) AS n FROM {lib}.{tbl}").iloc[0, 0]
                try:
                    dates = conn.raw_sql(
                        f"SELECT MIN(datadate) AS dmin, MAX(datadate) AS dmax FROM {lib}.{tbl}"
                    )
                    dmin, dmax = dates.iloc[0, 0], dates.iloc[0, 1]
                    try:
                        firms = conn.raw_sql(
                            f"SELECT COUNT(DISTINCT gvkey) AS n FROM {lib}.{tbl}"
                        ).iloc[0, 0]
                    except Exception:
                        firms = "?"
                    print(f"  {lib}.{tbl}: {count:>10,} rows, {firms} firms, {dmin} -> {dmax}")
                except Exception:
                    print(f"  {lib}.{tbl}: {count:>10,} rows (no datadate column)")
            except Exception as e:
                msg = str(e).splitlines()[0][:80]
                print(f"  {lib}.{tbl}: (error: {msg})")

    conn.close()


if __name__ == "__main__":
    main()

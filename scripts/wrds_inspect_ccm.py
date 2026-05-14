"""Show the tables inside crsp_a_ccm and a column-name preview of funda/fundq.

Run before the Compustat pull to confirm the merged-library schema matches
what ingest_wrds.py expects.
"""
from __future__ import annotations

import os
import sys

import wrds


def main() -> None:
    username = os.environ.get("WRDS_USERNAME")
    if not username:
        from src.utils.env import get_env

        username = get_env("WRDS_USERNAME", required=True)

    conn = wrds.Connection(wrds_username=username)
    print()
    print("=== Tables in crsp_a_ccm ===")
    try:
        tables = conn.list_tables("crsp_a_ccm")
        for t in sorted(tables):
            print(f"  {t}")
    except Exception as e:
        print(f"  (error listing tables: {e})")

    for tbl in ("funda", "fundq"):
        print()
        print(f"=== Columns in crsp_a_ccm.{tbl} (first 30) ===")
        try:
            df = conn.raw_sql(f"SELECT * FROM crsp_a_ccm.{tbl} LIMIT 1")
            for c in list(df.columns)[:30]:
                print(f"  {c}")
            print(f"  ... ({len(df.columns)} columns total)")
        except Exception as e:
            print(f"  (error: {str(e).strip().splitlines()[0]})")

    conn.close()


if __name__ == "__main__":
    main()

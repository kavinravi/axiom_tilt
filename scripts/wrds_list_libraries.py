"""Print the WRDS libraries your account can see, filtered to Compustat-likely ones."""
from __future__ import annotations

import os
import sys

import wrds


def main() -> None:
    username = os.environ.get("WRDS_USERNAME")
    if not username:
        print("WRDS_USERNAME not set in env. Trying .env...", file=sys.stderr)
        from src.utils.env import get_env

        username = get_env("WRDS_USERNAME", required=True)

    print(f"Connecting as {username}...")
    conn = wrds.Connection(wrds_username=username)
    libs = sorted(conn.list_libraries())

    print()
    print(f"Total libraries visible: {len(libs)}")

    comp_like = [l for l in libs if any(kw in l.lower() for kw in ("comp", "fund", "ccm"))]
    print()
    print(f"=== Compustat / fundamentals / CCM ({len(comp_like)}) ===")
    for l in comp_like:
        print(f"  {l}")

    other = [l for l in libs if any(kw in l.lower() for kw in ("crsp", "sec", "wrdsapp", "merg"))]
    other = [l for l in other if l not in comp_like]
    print()
    print(f"=== Other potentially-relevant ({len(other)}) ===")
    for l in other:
        print(f"  {l}")

    print()
    print(f"=== All {len(libs)} libraries ===")
    for l in libs:
        print(f"  {l}")

    conn.close()


if __name__ == "__main__":
    main()

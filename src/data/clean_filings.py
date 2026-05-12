"""Extract clean text from raw EDGAR SGML envelopes.

Phase 1 of the FinBERT FT pipeline. Reads files from
data/raw/edgar/{cik}/{accession}.{txt,htm}, extracts <TEXT>...</TEXT> bodies,
strips HTML tags, collapses whitespace, and writes plain text to
data/interim/edgar_text/{cik}/{accession}.txt.

Parallelized via multiprocessing for speed (measured: 21 min on 16 workers
for 226,919 filings / 746 GB raw, AMD Ryzen 9 9950X + SSD).
Per-file resume: skips already-extracted output files.

Run via:
    python -m src.data.clean_filings              # default workers
    python -m src.data.clean_filings --workers 8  # override pool size
"""
from __future__ import annotations

import argparse
import html as html_module
import os
import re
from multiprocessing import Pool
from pathlib import Path

from tqdm import tqdm

from src.utils.io import edgar_raw_dir, interim_dir
from src.utils.logging_utils import configure_logging, get_logger


log = get_logger(__name__)

MIN_TEXT_LENGTH = 500  # drop filings smaller than this after cleaning

_TEXT_BLOCK_RE = re.compile(r"<TEXT[^>]*>(.*?)</TEXT>", flags=re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def extract_sgml_bodies(raw: str) -> list[str]:
    """Return all <TEXT>...</TEXT> body strings from an SGML envelope.

    Handles both bare <TEXT> and attribute-bearing <TEXT TYPE="...">.
    Returns [] if no blocks are found (caller falls back to treating whole input as html).
    """
    return [m.group(1) for m in _TEXT_BLOCK_RE.finditer(raw)]


def strip_html_tags(body: str) -> str:
    """Strip HTML tags, decode entities, collapse whitespace."""
    # Drop tags first
    no_tags = _TAG_RE.sub(" ", body)
    # Decode HTML entities (&amp; -> &, etc.)
    decoded = html_module.unescape(no_tags)
    # Collapse all whitespace runs to a single space
    return _WHITESPACE_RE.sub(" ", decoded).strip()


def clean_text(raw: str) -> str:
    """Full pipeline: extract bodies + strip + concatenate.

    For SGML envelopes: extracts all <TEXT> bodies, strips each, joins with newlines.
    For non-envelope input: treats the whole thing as HTML body.
    """
    bodies = extract_sgml_bodies(raw)
    if not bodies:
        # No SGML envelope — process input as a single HTML body
        return strip_html_tags(raw)
    cleaned = [strip_html_tags(b) for b in bodies]
    # Drop empty cleaned bodies; join the rest with newlines
    return "\n".join(c for c in cleaned if c)


def process_filing(input_path: Path, output_path: Path) -> bool:
    """Process one filing. Returns True if a new file was written, False otherwise.

    Skip cases:
      - Output already exists with non-empty content (resume)
      - Cleaned text below MIN_TEXT_LENGTH (mostly index pages or thin exhibits)
    """
    if output_path.exists() and output_path.stat().st_size > 0:
        return False

    try:
        raw = input_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        # Some files are latin-1; retry
        try:
            raw = input_path.read_text(encoding="latin-1", errors="ignore")
        except Exception as e:
            log.warning("Could not read %s: %s", input_path, e)
            return False

    text = clean_text(raw)
    if len(text) < MIN_TEXT_LENGTH:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return True


def iter_input_paths(raw_root: Path) -> list[Path]:
    """Walk raw EDGAR directory, return all .txt and .htm files."""
    out: list[Path] = []
    for ext in ("*.txt", "*.htm", "*.html"):
        out.extend(raw_root.rglob(ext))
    return out


def _output_path_for(input_path: Path, raw_root: Path, interim_root: Path) -> Path:
    """Mirror the {cik}/{accession}.{ext} structure under interim/ but force .txt extension."""
    rel = input_path.relative_to(raw_root)
    # Force .txt extension on output
    rel_txt = rel.with_suffix(".txt")
    return interim_root / rel_txt


def _worker(args: tuple[Path, Path]) -> bool:
    input_path, output_path = args
    return process_filing(input_path, output_path)


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(
        description="Extract clean text from raw EDGAR SGML envelopes (Phase 1 of FinBERT FT)."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) - 2),
        help="Number of multiprocessing workers (default: cpu_count - 2)",
    )
    args = parser.parse_args()

    raw_root = edgar_raw_dir()
    interim_root = interim_dir() / "edgar_text"
    interim_root.mkdir(parents=True, exist_ok=True)

    log.info("Scanning raw filings under %s", raw_root)
    inputs = iter_input_paths(raw_root)
    log.info("Found %d candidate files", len(inputs))

    jobs = [(p, _output_path_for(p, raw_root, interim_root)) for p in inputs]

    log.info("Starting multiprocessing pool with %d workers", args.workers)
    written = 0
    with Pool(processes=args.workers) as pool:
        for ok in tqdm(pool.imap_unordered(_worker, jobs, chunksize=16),
                        total=len(jobs), desc="filings"):
            if ok:
                written += 1

    log.info("Done. Wrote %d new files (skipped %d already-done or too-short).",
              written, len(jobs) - written)


if __name__ == "__main__":
    main()

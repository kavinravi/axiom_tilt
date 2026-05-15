"""Post-process cleaned EDGAR text to strip noise that `clean_filings.py` left behind.

Reads `data/interim/edgar_text/{cik}/{accession}.txt` and writes
`data/interim/edgar_text_v2/{cik}/{accession}.txt` — same layout, cleaner content.

Three kinds of noise observed in the v1 cleaned corpus:

  1. **XBRL inline data dumps.** Modern (post-2009) 10-K/10-Q filings embed
     iXBRL fact references that appear in the text as
     `0000831001 us-gaap:CoreDepositsMember 2021-03-31 iso4217:USD ...`
     — sometimes hundreds of thousands of these in a row.

  2. **Residual HTML.** `clean_filings.py` used a regex `<[^>]+>` to strip tags,
     which fails when an attribute value contains `>` (e.g. iXBRL
     `<ix:nonNumeric ... contextRef="abc>def">`). Long tails of
     `style="background-color:#cceeff;..." <span ...>` survive.

  3. **Binary residue.** Embedded base64 blobs (images, fonts) sometimes get
     partially decoded into the text body — high-entropy garbage like
     `&K^*C,1 T"\\F(-1+B]L#...`.

All three present as runs of tokens that are NOT English-prose tokens. The
filter uses a sliding window over whitespace-tokenized text and drops regions
where the fraction of word-like tokens falls below a threshold.

Run via:
    python -m src.data.refilter_text                  # default workers
    python -m src.data.refilter_text --workers 8      # override pool size
"""
from __future__ import annotations

import argparse
import os
import re
from multiprocessing import Pool
from pathlib import Path

from bs4 import BeautifulSoup
from tqdm import tqdm

from src.utils.io import interim_dir
from src.utils.logging_utils import configure_logging, get_logger


log = get_logger(__name__)


WINDOW_TOKENS = 100
CONTENT_THRESHOLD = 0.40
LOCAL_RADIUS = 5  # non-content tokens need a content token within this radius
MIN_OUTPUT_LENGTH = 500

_WHITESPACE_RE = re.compile(r"\s+")
_HTML_LIKE_RE = re.compile(r"<\s*/?\s*[a-zA-Z]")


def is_content_token(t: str) -> bool:
    """Heuristic: token looks like an English-prose word.

    Rejects: XBRL refs (`us-gaap:Foo`), HTML attributes (`style="..."`,
    `colspan="3"`), URLs/paths (`http://...`, `/a/b`), pure numbers, dates,
    binary garbage.
    """
    n = len(t)
    if n < 2 or n > 40:
        return False
    if ":" in t or "=" in t or '"' in t or "/" in t:
        return False
    if not t[0].isalpha():
        return False
    alpha = sum(1 for c in t if c.isalpha())
    return alpha / n >= 0.6


def refilter_text(text: str) -> str:
    """Apply the noise filter to a single document.

    Steps:
      1. If text contains any HTML-tag-like residue, parse with BeautifulSoup
         to extract plain text (robust to `>` inside attribute values, unlike
         the regex used in the v1 cleaner).
      2. Whitespace-collapse.
      3. Sliding-window content-density filter: a token is kept iff the
         centered window of WINDOW_TOKENS contains at least CONTENT_THRESHOLD
         word-like tokens.
    """
    if _HTML_LIKE_RE.search(text):
        try:
            text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
        except Exception:
            # Some filings have malformed SGML markup (e.g., `<![H3[...]]>`-style
            # marked sections) that html.parser rejects outright. Fall back to
            # leaving the text alone — the windowed content filter below will
            # still drop most surviving HTML/attribute fragments because they
            # tokenize into non-content (`:` / `=` / `"`) tokens.
            pass
    text = _WHITESPACE_RE.sub(" ", text).strip()
    tokens = text.split()
    n = len(tokens)
    if n == 0:
        return ""

    flags = [is_content_token(t) for t in tokens]

    # Short document — single-window decision
    if n <= WINDOW_TOKENS:
        return " ".join(tokens) if sum(flags) / n >= CONTENT_THRESHOLD else ""

    # Build prefix sum over content flags for O(n) window queries
    cum = [0] * (n + 1)
    for i, f in enumerate(flags):
        cum[i + 1] = cum[i] + (1 if f else 0)

    half = WINDOW_TOKENS // 2
    keep = [False] * n
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half)
        if (cum[hi] - cum[lo]) / (hi - lo) >= CONTENT_THRESHOLD:
            keep[i] = True

    # Local-radius pass: a non-content token survives the windowed pass only if
    # there's an actual prose word on BOTH sides within LOCAL_RADIUS. Requiring
    # *both* sides catches the boundary leak where the first few tokens of a
    # noise run see content backwards (into prose) but only noise forward.
    for i in range(n):
        if not keep[i] or flags[i]:
            continue
        lo = max(0, i - LOCAL_RADIUS)
        hi = min(n, i + LOCAL_RADIUS + 1)
        back_has = any(flags[j] for j in range(lo, i))
        forward_has = any(flags[j] for j in range(i + 1, hi))
        if not (back_has and forward_has):
            keep[i] = False

    return " ".join(t for t, k in zip(tokens, keep) if k)


def process_filing(input_path: Path, output_path: Path) -> bool:
    """Refilter one file. Returns True if a new output was written.

    All read / filter / write errors are caught and logged so one bad file
    can't kill the multiprocessing pool.
    """
    if output_path.exists() and output_path.stat().st_size > 0:
        return False
    try:
        text = input_path.read_text(encoding="utf-8", errors="ignore")
        filtered = refilter_text(text)
    except Exception as e:
        log.warning("refilter failed for %s: %s: %s", input_path, type(e).__name__, e)
        return False
    if len(filtered) < MIN_OUTPUT_LENGTH:
        return False
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(filtered, encoding="utf-8")
    except Exception as e:
        log.warning("write failed for %s: %s: %s", output_path, type(e).__name__, e)
        return False
    return True


def _worker(args: tuple[Path, Path]) -> bool:
    return process_filing(*args)


def _iter_paths(in_root: Path, out_root: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for p in in_root.rglob("*.txt"):
        rel = p.relative_to(in_root)
        pairs.append((p, out_root / rel))
    return pairs


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) - 2),
        help="multiprocessing pool size (default: cpu_count - 2)",
    )
    parser.add_argument(
        "--in-dir",
        default=str(interim_dir() / "edgar_text"),
        help="input directory (default: data/interim/edgar_text)",
    )
    parser.add_argument(
        "--out-dir",
        default=str(interim_dir() / "edgar_text_v2"),
        help="output directory (default: data/interim/edgar_text_v2)",
    )
    args = parser.parse_args()

    in_root = Path(args.in_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    log.info("scanning %s", in_root)
    jobs = _iter_paths(in_root, out_root)
    log.info("found %d candidate files", len(jobs))

    written = 0
    with Pool(processes=args.workers) as pool:
        for ok in tqdm(pool.imap_unordered(_worker, jobs, chunksize=32),
                        total=len(jobs), desc="refilter"):
            if ok:
                written += 1
    log.info("done: wrote %d new files (skipped %d already-done or too-short).",
              written, len(jobs) - written)


if __name__ == "__main__":
    main()

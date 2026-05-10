"""Pull SEC EDGAR filings (10-K, 10-Q, 8-K) for the universe.

Strategy:
  1. For each quarter from start_date.year/Q1 to end_date.year/Q4, fetch
     https://www.sec.gov/Archives/edgar/full-index/{YYYY}/QTR{n}/master.idx
  2. Filter to (CIK in universe) AND (form_type in configured set).
  3. For each surviving filing, fetch the primary document and extract text.
  4. Save text to data/raw/edgar/{cik}/{accession}.txt
  5. Track completed accessions in data/state/edgar_done.txt for resume.

Usage:
  python -m src.data.ingest_filings           # full run (overnight)
  python -m src.data.ingest_filings --year 2024  # single year
"""
from __future__ import annotations

import argparse
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

from src.utils.config import load_config
from src.utils.env import get_env
from src.utils.io import edgar_raw_dir, processed_dir, raw_dir, state_dir
from src.utils.logging_utils import configure_logging, get_logger
from src.utils.rate_limit import TokenBucket


log = get_logger(__name__)
SEC_BASE = "https://www.sec.gov"
INDEX_URL = "{base}/Archives/edgar/full-index/{year}/QTR{quarter}/master.idx"


@dataclass
class EdgarFiling:
    cik: str           # 10-digit zero-padded
    company: str
    form_type: str
    filing_date: pd.Timestamp
    filename: str      # path relative to SEC_BASE, e.g. edgar/data/.../...txt
    accession: str     # e.g. 0000320193-24-000123

    @property
    def url(self) -> str:
        return f"{SEC_BASE}/Archives/{self.filename}"

    @property
    def local_text_path(self) -> Path:
        return edgar_raw_dir() / self.cik / f"{self.accession}.txt"

    @property
    def local_raw_path(self) -> Path:
        """Path where the raw SGML envelope or HTML is stored on disk."""
        # Preserve original extension (.txt or .htm) so downstream parsing
        # can dispatch correctly without re-querying SEC.
        suffix = Path(self.filename).suffix or ".txt"
        return edgar_raw_dir() / self.cik / f"{self.accession}{suffix}"


def parse_master_idx(text: str) -> pd.DataFrame:
    """Parse SEC's master.idx (pipe-delimited after a header)."""
    # Skip header lines until we find the dashes
    lines = text.splitlines()
    start = 0
    for i, line in enumerate(lines):
        if line.startswith("---"):
            start = i + 1
            break
    body = "\n".join(lines[start:])
    df = pd.read_csv(
        StringIO(body),
        sep="|",
        names=["cik", "company", "form_type", "date_filed", "filename"],
        dtype=str,
    )
    df["cik"] = df["cik"].str.strip().str.zfill(10)
    df["company"] = df["company"].str.strip()
    df["form_type"] = df["form_type"].str.strip()
    df["filing_date"] = pd.to_datetime(df["date_filed"].str.strip(), errors="coerce")
    df["filename"] = df["filename"].str.strip()
    df = df.dropna(subset=["filing_date"])
    return df.drop(columns=["date_filed"])


_ACCESSION_FROM_PATH = re.compile(r"(\d{18})")


def accession_from_filename(filename: str) -> str:
    """Extract accession from an EDGAR filing path.

    Two common forms in master.idx:
      edgar/data/320193/0000320193-24-000123.txt              -> direct
      edgar/data/320193/000032019324000123/...                 -> folder form
    """
    # Try direct dashed form first
    m = re.search(r"(\d{10}-\d{2}-\d{6})", filename)
    if m:
        return m.group(1)
    # Fall back to folder form -> reconstruct
    m = _ACCESSION_FROM_PATH.search(filename)
    if m:
        raw = m.group(1)
        return f"{raw[:10]}-{raw[10:12]}-{raw[12:]}"
    raise ValueError(f"Could not extract accession from: {filename}")


def extract_text_from_html(html: str) -> str:
    """Strip HTML tags and return clean text. Drops scripts/styles."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "head", "meta"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    # Collapse whitespace
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def _load_done(state_file: Path) -> set[str]:
    if not state_file.exists():
        return set()
    return {line.strip() for line in state_file.read_text().splitlines() if line.strip()}


def _append_done(state_file: Path, accession: str) -> None:
    with state_file.open("a") as f:
        f.write(f"{accession}\n")


_state_lock = threading.Lock()


def _append_done_locked(state_file: Path, accession: str) -> None:
    """Thread-safe wrapper around _append_done — serializes file writes."""
    with _state_lock:
        _append_done(state_file, accession)


@dataclass
class EdgarClient:
    user_agent: str
    bucket: TokenBucket
    retry_attempts: int

    def _headers(self) -> dict[str, str]:
        return {"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"}

    def fetch(self, url: str) -> bytes:
        @retry(
            stop=stop_after_attempt(self.retry_attempts),
            wait=wait_exponential(multiplier=2, min=2, max=60),
            retry=retry_if_exception_type((requests.RequestException,)),
            reraise=True,
        )
        def _do() -> bytes:
            self.bucket.acquire()
            resp = requests.get(url, headers=self._headers(), timeout=60)
            # Retry only on transient codes (rate limit + gateway errors)
            if resp.status_code in (429, 502, 503, 504):
                raise requests.RequestException(f"transient {resp.status_code}")
            # 4xx client errors are terminal — raise non-retryable to short-circuit tenacity
            if 400 <= resp.status_code < 500:
                raise FileNotFoundError(f"{resp.status_code} for {url}")
            resp.raise_for_status()
            return resp.content
        return _do()


def iter_quarters(start_year: int, end_year: int):
    for y in range(start_year, end_year + 1):
        for q in range(1, 5):
            yield y, q


def collect_filings_for_universe(
    client: EdgarClient,
    universe_ciks: set[str],
    form_types: set[str],
    start_year: int,
    end_year: int,
) -> list[EdgarFiling]:
    """Walk all quarterly indexes; return matched filings."""
    out: list[EdgarFiling] = []
    quarters = list(iter_quarters(start_year, end_year))
    for year, q in tqdm(quarters, desc="indexes"):
        url = INDEX_URL.format(base=SEC_BASE, year=year, quarter=q)
        try:
            raw = client.fetch(url)
        except Exception as e:
            log.warning("Failed to fetch %s: %s", url, e)
            continue
        text = raw.decode("latin-1", errors="replace")
        df = parse_master_idx(text)
        df = df[df["cik"].isin(universe_ciks)]
        df = df[df["form_type"].isin(form_types)]
        for _, row in df.iterrows():
            try:
                acc = accession_from_filename(row["filename"])
            except ValueError:
                continue
            out.append(EdgarFiling(
                cik=row["cik"],
                company=row["company"],
                form_type=row["form_type"],
                filing_date=row["filing_date"],
                filename=row["filename"],
                accession=acc,
            ))
    return out


def fetch_and_save_filing(client: EdgarClient, filing: EdgarFiling) -> bool:
    """Fetch the filing, save raw bytes to disk. Returns True on success.

    Text extraction (HTML/SGML stripping) happens later in a separate notebook
    so the fetch step stays network-bound and parallelizes well. The raw SGML
    envelope is preserved so we can re-extract with different strategies later.
    """
    out_path = filing.local_raw_path
    if out_path.exists():
        return True
    try:
        raw = client.fetch(filing.url)
    except Exception as e:
        log.warning("Fetch failed %s: %s", filing.url, e)
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(raw)
    return True


def extract_text_from_sgml(sgml: str) -> str:
    """SEC's old SGML envelope wraps individual documents in <DOCUMENT>...</DOCUMENT>.

    Concatenate the text from each <TEXT> body, after HTML stripping.
    """
    bodies: list[str] = []
    for match in re.finditer(r"<TEXT[^>]*>(.*?)</TEXT>", sgml, flags=re.DOTALL | re.IGNORECASE):
        body = match.group(1)
        bodies.append(extract_text_from_html(body))
    if not bodies:
        # Fallback: treat whole envelope as html-ish
        return extract_text_from_html(sgml)
    return "\n\n".join(bodies)


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=None,
                        help="Restrict to a single year (else use config range)")
    args = parser.parse_args()

    cfg = load_config("data")
    user_agent = get_env("SEC_USER_AGENT", required=True)

    universe_path = processed_dir() / "universe.parquet"
    universe = pd.read_parquet(universe_path)
    universe_ciks = set(universe["cik"].dropna().astype(str).str.zfill(10).tolist())
    log.info("Universe CIKs: %d", len(universe_ciks))

    start_year = int(cfg["start_date"][:4])
    end_year = int(cfg["end_date"][:4])
    if args.year is not None:
        start_year = end_year = args.year
        log.info("Restricting to year %d", args.year)

    client = EdgarClient(
        user_agent=user_agent,
        bucket=TokenBucket(
            rate_per_sec=cfg["edgar"]["rate_per_sec"],
            capacity=cfg["edgar"]["capacity"],
        ),
        retry_attempts=cfg["edgar"]["retry_attempts"],
    )

    log.info("Walking quarterly indexes %d-%d...", start_year, end_year)
    filings = collect_filings_for_universe(
        client=client,
        universe_ciks=universe_ciks,
        form_types=set(cfg["edgar"]["form_types"]),
        start_year=start_year,
        end_year=end_year,
    )
    log.info("Matched %d filings to download", len(filings))

    state_file = state_dir() / "edgar_done.txt"
    done = _load_done(state_file)
    todo = [f for f in filings if f.accession not in done]
    log.info("Already done: %d, To do: %d", len(done), len(todo))

    n_workers = cfg["edgar"].get("workers", 6)
    n_ok = 0

    def _fetch_one(filing: EdgarFiling) -> tuple[str, bool]:
        ok = fetch_and_save_filing(client, filing)
        if ok:
            _append_done_locked(state_file, filing.accession)
        return filing.accession, ok

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_fetch_one, f): f for f in todo}
        for i, fut in enumerate(tqdm(as_completed(futures), total=len(todo), desc="filings")):
            try:
                _, ok = fut.result()
                if ok:
                    n_ok += 1
            except Exception as e:
                log.warning("Worker exception: %s", e)
            if (i + 1) % cfg["edgar"]["checkpoint_every_n"] == 0:
                log.info("checkpoint: %d/%d ok", n_ok, i + 1)

    # Build / update the index parquet (load done set ONCE — not per filing)
    done = _load_done(state_file)
    data_root = raw_dir().parent
    index_rows = [
        {
            "cik": f.cik,
            "company": f.company,
            "form_type": f.form_type,
            "filing_date": f.filing_date,
            "accession": f.accession,
            "filename": f.filename,
            "raw_path": str(f.local_raw_path.relative_to(data_root)),
            "text_path": str(f.local_text_path.relative_to(data_root)),
        }
        for f in filings
        if f.accession in done
    ]
    idx_df = pd.DataFrame(index_rows)
    idx_df.to_parquet(processed_dir() / "edgar_index.parquet", index=False)
    log.info("Wrote edgar_index with %d rows", len(idx_df))


if __name__ == "__main__":
    main()

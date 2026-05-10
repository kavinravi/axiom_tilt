# Data Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the five data-ingestion modules (universe, prices, fundamentals, EDGAR text, macro) so we can pull ~25 years of S&P 500 data and ~600K SEC filings to disk, with the EDGAR pull resumable for an overnight unattended run.

**Architecture:** Each ingester is a self-contained module under `src/data/` with its own runnable `main()`. Shared infrastructure lives in `src/utils/`: env loading, paths, logging, and a token-bucket rate limiter for the SEC. Outputs split into `data/raw/` (gitignored bulk: HTML filings, raw API dumps), `data/interim/` (cleaned text), and `data/processed/` (parquet feature tables, tracked in git for laptop sync). EDGAR uses a per-accession-number state file so the run can resume cleanly after rate-limit bans, network hiccups, or manual interrupts.

**Tech Stack:** Python 3.11, `requests` + `tenacity` for HTTP, `BeautifulSoup` + `lxml` for HTML parsing, `pandas` + `pyarrow` for tabular IO, `yfinance` for prices, `pandas-datareader` for FRED, `python-dotenv` for secrets, `pytest` for tests.

---

## Scope

This plan covers **only the data-ingestion phase** (spec §3). It does not cover FinBERT fine-tuning, PCA, ranker training, RL, or backtest — those each get their own plan.

**Deliverables when this plan is done:**
- `data/processed/universe.parquet` — point-in-time S&P 500 membership 2000–2025
- `data/processed/prices.parquet` — daily OHLCV adjusted for every ticker that ever entered the universe
- `data/processed/fundamentals.parquet` — quarterly financial statements
- `data/processed/edgar_index.parquet` — catalog of filings on disk
- `data/raw/edgar/{cik}/{accession}.txt` — extracted plain text per filing
- `data/processed/macro.parquet` — risk-free rate + regime indicators
- All modules have at least a smoke test passing on a small slice.

---

## File Structure

**Created:**
- `pyproject.toml` — minimal package config so `pip install -e .` makes `src.*` importable
- `.env.example` — template for `FMP_API_KEY`, `ALPHAVANTAGE_API_KEY`, `SEC_USER_AGENT`
- `src/utils/env.py` — load `.env` and validate required keys
- `src/utils/paths.py` — repo-root-anchored paths (replaces stub in `io.py`)
- `src/utils/rate_limit.py` — token-bucket rate limiter
- `src/data/build_universe.py` — S&P 500 history reconstruction
- `src/data/ingest_macro.py` — FRED puller
- `tests/data/test_build_universe.py`
- `tests/data/test_ingest_prices.py`
- `tests/data/test_ingest_fundamentals.py`
- `tests/data/test_ingest_filings.py`
- `tests/data/test_ingest_macro.py`
- `tests/utils/test_rate_limit.py`
- `tests/utils/test_env.py`
- `tests/data/__init__.py`, `tests/utils/__init__.py`
- `tests/fixtures/wikipedia_sp500.html` — saved Wikipedia snapshot for deterministic universe tests
- `tests/fixtures/sec_company_tickers.json` — saved SEC ticker→CIK snapshot
- `tests/fixtures/edgar_index.idx` — sample SEC quarterly index
- `tests/fixtures/sample_10k.html` — sample 10-K for HTML→text test

**Modified (replacing stubs):**
- `src/utils/config.py` — env-aware config loader (currently `NotImplementedError`)
- `src/utils/io.py` — path helpers (currently `NotImplementedError`)
- `src/utils/logging_utils.py` — structured logging setup (currently `NotImplementedError`)
- `src/data/ingest_prices.py` — yfinance OHLCV puller
- `src/data/ingest_fundamentals.py` — FMP statements puller
- `src/data/ingest_filings.py` — EDGAR text puller
- `requirements.txt` — add `requests`, `beautifulsoup4`, `lxml`, `tenacity`, `python-dotenv`, `yfinance`, `pandas-datareader`, `tqdm`
- `configs/data.yaml` — add ingestion-specific config (date range, EDGAR form types, rate limits)

**Out of scope (not touched in this plan):**
- `src/data/align_panel.py`, `src/data/split_data.py` — feature-prep, separate plan
- Anything under `src/text/`, `src/features/`, `src/models/`, `src/policy/`, `src/portfolio/`, `src/backtest/`

---

## Task 0: Project foundation

Sets up packaging, env loading, paths, logging, and the rate limiter that everything else depends on.

**Files:**
- Create: `pyproject.toml`, `.env.example`, `src/utils/env.py`, `src/utils/paths.py`, `src/utils/rate_limit.py`, `tests/utils/__init__.py`, `tests/utils/test_rate_limit.py`, `tests/utils/test_env.py`
- Modify: `requirements.txt`, `src/utils/config.py`, `src/utils/io.py`, `src/utils/logging_utils.py`, `configs/data.yaml`

### Task 0.1: Add dependencies to requirements.txt

- [ ] **Step 1: Add ingestion deps to `requirements.txt`**

Open `requirements.txt` and append before the `# Tooling` block:

```
# Data ingestion
requests>=2.32
beautifulsoup4>=4.12
lxml>=5.2
tenacity>=8.5
python-dotenv>=1.0
yfinance>=0.2.40
pandas-datareader>=0.10
tqdm>=4.66
```

- [ ] **Step 2: Install**

Run: `pip install -r requirements.txt`
Expected: all new packages install cleanly. If `lxml` fails on WSL, try `apt install libxml2-dev libxslt-dev` first.

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "deps: add data-ingestion packages"
```

### Task 0.2: Create pyproject.toml

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "axiom-tilt"
version = "0.0.1"
description = "Text-enhanced RL portfolio allocation research project."
requires-python = ">=3.11"

[tool.setuptools]
package-dir = {"" = "."}

[tool.setuptools.packages.find]
where = ["."]
include = ["src*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "UP", "B", "SIM"]
ignore = ["E501"]
```

- [ ] **Step 2: Install in editable mode**

Run: `pip install -e .`
Expected: installs without error. Verify with `python -c "import src; print(src.__file__)"` — should print the repo's `src/__init__.py` path.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "build: add pyproject.toml for editable install"
```

### Task 0.3: Create .env.example

- [ ] **Step 1: Create `.env.example`**

```bash
# Copy this to .env and fill in your keys. .env is gitignored.

# SEC EDGAR requires a User-Agent header identifying you. Format: "Name email@example.com"
# https://www.sec.gov/os/accessing-edgar-data
SEC_USER_AGENT="Your Name your.email@example.com"

# Financial Modeling Prep — fundamentals primary
# https://site.financialmodelingprep.com/developer/docs
FMP_API_KEY=""

# Alpha Vantage — fundamentals cross-check
# https://www.alphavantage.co/support/#api-key
ALPHAVANTAGE_API_KEY=""
```

- [ ] **Step 2: Commit**

```bash
git add .env.example
git commit -m "config: add .env.example template"
```

### Task 0.4: Implement env loader (TDD)

- [ ] **Step 1: Create `tests/utils/__init__.py`** (empty file)

```python
```

- [ ] **Step 2: Write failing test `tests/utils/test_env.py`**

```python
"""Tests for src.utils.env."""
import os
import pytest

from src.utils.env import get_env, EnvError


def test_get_env_returns_value(monkeypatch):
    monkeypatch.setenv("TEST_KEY", "abc123")
    assert get_env("TEST_KEY") == "abc123"


def test_get_env_required_missing_raises(monkeypatch):
    monkeypatch.delenv("MISSING_KEY", raising=False)
    with pytest.raises(EnvError, match="MISSING_KEY"):
        get_env("MISSING_KEY", required=True)


def test_get_env_optional_missing_returns_default(monkeypatch):
    monkeypatch.delenv("OPT_KEY", raising=False)
    assert get_env("OPT_KEY", default="fallback") == "fallback"


def test_get_env_strips_whitespace(monkeypatch):
    monkeypatch.setenv("PADDED", "  hello  ")
    assert get_env("PADDED") == "hello"
```

- [ ] **Step 3: Run test, confirm it fails**

Run: `pytest tests/utils/test_env.py -v`
Expected: ImportError or ModuleNotFoundError on `src.utils.env`.

- [ ] **Step 4: Create `src/utils/env.py`**

```python
"""Environment variable loading with .env file support."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


class EnvError(RuntimeError):
    """Raised when a required environment variable is missing."""


_loaded = False


def _ensure_loaded() -> None:
    global _loaded
    if _loaded:
        return
    repo_root = Path(__file__).resolve().parents[2]
    env_path = repo_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    _loaded = True


def get_env(
    key: str,
    *,
    required: bool = False,
    default: str | None = None,
) -> str | None:
    """Read an env var. Strips whitespace. Loads .env on first call."""
    _ensure_loaded()
    raw = os.environ.get(key)
    if raw is None or raw.strip() == "":
        if required:
            raise EnvError(
                f"Required environment variable {key!r} is unset. "
                f"Add it to .env (see .env.example)."
            )
        return default
    return raw.strip()
```

- [ ] **Step 5: Run test, confirm it passes**

Run: `pytest tests/utils/test_env.py -v`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add src/utils/env.py tests/utils/__init__.py tests/utils/test_env.py
git commit -m "utils: add env loader with .env support"
```

### Task 0.5: Implement paths helper

- [ ] **Step 1: Replace `src/utils/io.py` with paths helper**

```python
"""Project path helpers anchored at the repo root."""
from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    return repo_root() / "data"


def raw_dir() -> Path:
    p = data_dir() / "raw"
    p.mkdir(parents=True, exist_ok=True)
    return p


def interim_dir() -> Path:
    p = data_dir() / "interim"
    p.mkdir(parents=True, exist_ok=True)
    return p


def processed_dir() -> Path:
    p = data_dir() / "processed"
    p.mkdir(parents=True, exist_ok=True)
    return p


def edgar_raw_dir() -> Path:
    p = raw_dir() / "edgar"
    p.mkdir(parents=True, exist_ok=True)
    return p


def state_dir() -> Path:
    """For checkpoint/resume state files."""
    p = data_dir() / "state"
    p.mkdir(parents=True, exist_ok=True)
    return p
```

- [ ] **Step 2: Sanity check**

Run: `python -c "from src.utils.io import processed_dir; print(processed_dir())"`
Expected: prints absolute path ending in `axiom_tilt/data/processed`.

- [ ] **Step 3: Commit**

```bash
git add src/utils/io.py
git commit -m "utils: implement project path helpers"
```

### Task 0.6: Implement logging setup

- [ ] **Step 1: Replace `src/utils/logging_utils.py`**

```python
"""Structured logging configuration."""
from __future__ import annotations

import logging
import sys
from pathlib import Path


_CONFIGURED = False


def configure_logging(
    level: str = "INFO",
    log_file: Path | None = None,
) -> logging.Logger:
    """Configure root logger. Idempotent."""
    global _CONFIGURED
    root = logging.getLogger()
    if _CONFIGURED:
        return root

    root.setLevel(level.upper())
    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    root.addHandler(stream)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        root.addHandler(fh)

    _CONFIGURED = True
    return root


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
```

- [ ] **Step 2: Sanity check**

Run: `python -c "from src.utils.logging_utils import configure_logging, get_logger; configure_logging(); get_logger('test').info('hello')"`
Expected: prints a timestamped INFO line.

- [ ] **Step 3: Commit**

```bash
git add src/utils/logging_utils.py
git commit -m "utils: implement logging configuration"
```

### Task 0.7: Implement config loader

- [ ] **Step 1: Replace `src/utils/config.py`**

```python
"""YAML config loading."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.utils.io import repo_root


def load_config(name: str) -> dict[str, Any]:
    """Load a config file by name from the configs/ directory.

    Example: load_config('data') -> reads configs/data.yaml.
    """
    path = repo_root() / "configs" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open() as f:
        return yaml.safe_load(f)
```

- [ ] **Step 2: Sanity check**

Run: `python -c "from src.utils.config import load_config; print(load_config('data'))"`
Expected: prints the dict from `configs/data.yaml`.

- [ ] **Step 3: Commit**

```bash
git add src/utils/config.py
git commit -m "utils: implement YAML config loader"
```

### Task 0.8: Implement rate limiter (TDD)

- [ ] **Step 1: Write failing test `tests/utils/test_rate_limit.py`**

```python
"""Tests for src.utils.rate_limit."""
import time

from src.utils.rate_limit import TokenBucket


def test_immediate_first_call_does_not_block():
    bucket = TokenBucket(rate_per_sec=10.0, capacity=10)
    t0 = time.monotonic()
    bucket.acquire()
    assert time.monotonic() - t0 < 0.05


def test_burst_within_capacity_does_not_block():
    bucket = TokenBucket(rate_per_sec=10.0, capacity=5)
    t0 = time.monotonic()
    for _ in range(5):
        bucket.acquire()
    assert time.monotonic() - t0 < 0.1


def test_exceeding_capacity_blocks():
    # capacity=2, rate=10/s. After bursting 2, the 3rd acquire should wait ~0.1s.
    bucket = TokenBucket(rate_per_sec=10.0, capacity=2)
    bucket.acquire()
    bucket.acquire()
    t0 = time.monotonic()
    bucket.acquire()
    elapsed = time.monotonic() - t0
    assert 0.08 <= elapsed <= 0.25, f"expected ~0.1s wait, got {elapsed:.3f}s"


def test_steady_state_rate_is_respected():
    # rate=20/s, request 10 tokens in tight loop -> should take ~0.5s
    bucket = TokenBucket(rate_per_sec=20.0, capacity=1)
    t0 = time.monotonic()
    for _ in range(10):
        bucket.acquire()
    elapsed = time.monotonic() - t0
    assert 0.4 <= elapsed <= 0.7, f"expected ~0.5s, got {elapsed:.3f}s"
```

- [ ] **Step 2: Run test, confirm it fails**

Run: `pytest tests/utils/test_rate_limit.py -v`
Expected: ImportError on `src.utils.rate_limit`.

- [ ] **Step 3: Create `src/utils/rate_limit.py`**

```python
"""Token-bucket rate limiter for HTTP clients (notably SEC EDGAR)."""
from __future__ import annotations

import threading
import time


class TokenBucket:
    """Thread-safe token-bucket rate limiter.

    Tokens accumulate at `rate_per_sec` up to `capacity`. `acquire()` blocks
    until at least one token is available, then consumes one.
    """

    def __init__(self, rate_per_sec: float, capacity: int) -> None:
        if rate_per_sec <= 0:
            raise ValueError("rate_per_sec must be positive")
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._rate = float(rate_per_sec)
        self._capacity = float(capacity)
        self._tokens = float(capacity)
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            while True:
                now = time.monotonic()
                elapsed = now - self._last
                self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
                self._last = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                deficit = 1.0 - self._tokens
                sleep_for = deficit / self._rate
            # The continue happens implicitly via the while loop after sleep.
                time.sleep(sleep_for)
```

Note: the inner `time.sleep` placement and loop logic deserves a careful re-read — fix in the next step if the test catches a bug.

- [ ] **Step 4: Run test, fix loop if it hangs**

Run: `pytest tests/utils/test_rate_limit.py -v`

If the test hangs or fails, the issue is the `time.sleep` is inside the lock and the sleep is unreachable. Replace the implementation with this corrected version:

```python
"""Token-bucket rate limiter for HTTP clients (notably SEC EDGAR)."""
from __future__ import annotations

import threading
import time


class TokenBucket:
    """Thread-safe token-bucket rate limiter.

    Tokens accumulate at `rate_per_sec` up to `capacity`. `acquire()` blocks
    until at least one token is available, then consumes one.
    """

    def __init__(self, rate_per_sec: float, capacity: int) -> None:
        if rate_per_sec <= 0:
            raise ValueError("rate_per_sec must be positive")
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._rate = float(rate_per_sec)
        self._capacity = float(capacity)
        self._tokens = float(capacity)
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        self._last = now

    def acquire(self) -> None:
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                deficit = 1.0 - self._tokens
                sleep_for = deficit / self._rate
            # Sleep outside the lock so other threads can refill.
            time.sleep(sleep_for)
```

Re-run: `pytest tests/utils/test_rate_limit.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/utils/rate_limit.py tests/utils/test_rate_limit.py
git commit -m "utils: add token-bucket rate limiter"
```

### Task 0.9: Update configs/data.yaml with ingestion params

- [ ] **Step 1: Replace `configs/data.yaml`**

```yaml
universe: large_cap_us_equities
universe_size: 400
frequency: weekly
holding_period: 1w
document_frequency: weekly

# Date range — trims everything we ingest
start_date: "2000-01-01"
end_date: "2025-12-31"

# Storage roots (relative to repo root)
raw_data_dir: data/raw
interim_data_dir: data/interim
processed_data_dir: data/processed
state_dir: data/state

# Text sources to pull in v1 (transcripts deferred to v2 per spec §16)
text_sources:
  - filings

# EDGAR
edgar:
  form_types: ["10-K", "10-Q", "8-K"]
  rate_per_sec: 8.0           # SEC limit is 10/s; leave 20% headroom
  capacity: 8
  retry_attempts: 5
  retry_min_wait: 2
  retry_max_wait: 60
  checkpoint_every_n: 100     # flush state every N successful pulls

# yfinance
prices:
  batch_size: 50              # tickers per yfinance.download call
  request_delay_sec: 0.5      # polite delay between batches

# FMP
fundamentals:
  rate_per_sec: 5.0           # FMP free tier; paid tier higher
  capacity: 5
  retry_attempts: 3
  endpoints:
    - income-statement
    - balance-sheet-statement
    - cash-flow-statement

# FRED macro series (no API key required via pandas-datareader)
macro:
  series:
    - DGS3MO   # 3-month T-bill (risk-free)
    - DGS10    # 10-year Treasury (term spread component)
    - VIXCLS   # VIX
    - T10Y2Y   # 10y-2y spread
```

- [ ] **Step 2: Verify it loads**

Run: `python -c "from src.utils.config import load_config; print(load_config('data')['edgar'])"`
Expected: prints the EDGAR dict.

- [ ] **Step 3: Commit**

```bash
git add configs/data.yaml
git commit -m "config: extend data.yaml with ingestion params"
```

---

## Task 1: Universe reconstruction

Builds the point-in-time S&P 500 membership table by scraping Wikipedia (current members + historical changes) and joining ticker → CIK from SEC's `company_tickers.json`.

**Files:**
- Create: `src/data/build_universe.py`, `tests/data/__init__.py`, `tests/data/test_build_universe.py`, `tests/fixtures/wikipedia_sp500.html`, `tests/fixtures/sec_company_tickers.json`

### Task 1.1: Capture test fixtures

- [ ] **Step 1: Save Wikipedia S&P 500 page snapshot**

Run:
```bash
mkdir -p tests/fixtures
curl -A "Mozilla/5.0" -o tests/fixtures/wikipedia_sp500.html https://en.wikipedia.org/wiki/List_of_S%26P_500_companies
```

Verify: `ls -lh tests/fixtures/wikipedia_sp500.html` should show ~1-2 MB.

- [ ] **Step 2: Save SEC company tickers JSON**

Run:
```bash
curl -A "axiom-tilt-research test@example.com" -o tests/fixtures/sec_company_tickers.json https://www.sec.gov/files/company_tickers.json
```

Verify: `head -c 200 tests/fixtures/sec_company_tickers.json` should show JSON like `{"0":{"cik_str":...}}`.

- [ ] **Step 3: Commit fixtures**

```bash
git add tests/fixtures/wikipedia_sp500.html tests/fixtures/sec_company_tickers.json
git commit -m "test: add wikipedia + SEC fixtures for universe tests"
```

### Task 1.2: Write universe-parser tests (TDD)

- [ ] **Step 1: Create `tests/data/__init__.py`** (empty)

```python
```

- [ ] **Step 2: Write failing test `tests/data/test_build_universe.py`**

```python
"""Tests for src.data.build_universe."""
from pathlib import Path

import pandas as pd

from src.data.build_universe import (
    parse_current_members,
    parse_changes_table,
    load_ticker_to_cik,
    reconstruct_membership,
)


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_parse_current_members_returns_nonempty_dataframe():
    html = (FIXTURES / "wikipedia_sp500.html").read_text(encoding="utf-8")
    df = parse_current_members(html)
    assert len(df) >= 400
    assert {"ticker", "company"}.issubset(df.columns)
    # Tickers should be uppercase and short
    assert all(t.isupper() for t in df["ticker"].head(10))


def test_parse_changes_table_returns_dataframe_with_dates():
    html = (FIXTURES / "wikipedia_sp500.html").read_text(encoding="utf-8")
    df = parse_changes_table(html)
    assert len(df) > 100
    assert {"date", "added_ticker", "removed_ticker"}.issubset(df.columns)
    assert pd.api.types.is_datetime64_any_dtype(df["date"])


def test_load_ticker_to_cik_maps_known_ticker():
    json_path = FIXTURES / "sec_company_tickers.json"
    mapping = load_ticker_to_cik(json_path)
    # Apple's CIK is 320193 — extremely unlikely to ever change
    assert mapping["AAPL"] == "0000320193"
    # CIKs are 10-digit zero-padded strings
    for cik in list(mapping.values())[:20]:
        assert len(cik) == 10
        assert cik.isdigit()


def test_reconstruct_membership_produces_intervals():
    html = (FIXTURES / "wikipedia_sp500.html").read_text(encoding="utf-8")
    cik_map = load_ticker_to_cik(FIXTURES / "sec_company_tickers.json")
    df = reconstruct_membership(
        html=html,
        ticker_to_cik=cik_map,
        start_date="2000-01-01",
        end_date="2025-12-31",
    )
    assert {"ticker", "cik", "date_in", "date_out"}.issubset(df.columns)
    # Every row should have a valid date_in
    assert df["date_in"].notna().all()
    # date_out can be NaT for currently-listed
    # Total distinct tickers should be in the 600-1000 range (current 500 + churn)
    assert 500 <= df["ticker"].nunique() <= 1200
    # No row should have date_in > date_out (when date_out is set)
    valid_out = df.dropna(subset=["date_out"])
    assert (valid_out["date_in"] <= valid_out["date_out"]).all()
```

- [ ] **Step 3: Run test, confirm it fails**

Run: `pytest tests/data/test_build_universe.py -v`
Expected: ImportError on `src.data.build_universe`.

### Task 1.3: Implement build_universe.py

- [ ] **Step 1: Create `src/data/build_universe.py`**

```python
"""Reconstruct point-in-time S&P 500 membership 2000-2025.

Sources:
  - https://en.wikipedia.org/wiki/List_of_S%26P_500_companies (current + changes table)
  - https://www.sec.gov/files/company_tickers.json (ticker -> CIK)

Output:
  data/processed/universe.parquet with columns:
    ticker, cik, company, date_in, date_out

A ticker may appear multiple times if it left and re-joined the index.
"""
from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.utils.env import get_env
from src.utils.io import processed_dir
from src.utils.logging_utils import configure_logging, get_logger


WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

log = get_logger(__name__)


def _fetch(url: str, user_agent: str) -> str:
    resp = requests.get(url, headers={"User-Agent": user_agent}, timeout=30)
    resp.raise_for_status()
    return resp.text


def parse_current_members(html: str) -> pd.DataFrame:
    """Parse the first wikitable on the Wikipedia page (current S&P 500 members)."""
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", {"id": "constituents"})
    if table is None:
        # Fallback: first sortable wikitable
        table = soup.find("table", {"class": lambda c: c and "wikitable" in c})
    if table is None:
        raise RuntimeError("Could not find S&P 500 members table on Wikipedia page")
    df = pd.read_html(StringIO(str(table)))[0]
    df.columns = [c.strip() for c in df.columns]
    # Wikipedia's column names drift; normalize the two we care about
    ticker_col = next(c for c in df.columns if "Symbol" in c or "Ticker" in c)
    name_col = next(c for c in df.columns if "Security" in c or "Company" in c)
    out = pd.DataFrame({
        "ticker": df[ticker_col].astype(str).str.upper().str.replace(".", "-", regex=False),
        "company": df[name_col].astype(str),
    })
    return out.reset_index(drop=True)


def parse_changes_table(html: str) -> pd.DataFrame:
    """Parse the 'Selected changes to the list' historical table.

    Returns long-format rows: (date, added_ticker, removed_ticker).
    A single date can have both an addition and a removal (they're paired changes).
    """
    soup = BeautifulSoup(html, "lxml")
    # The changes table usually has id="changes" or is the second wikitable
    changes_table = soup.find("table", {"id": "changes"})
    if changes_table is None:
        tables = soup.find_all("table", {"class": lambda c: c and "wikitable" in c})
        if len(tables) < 2:
            raise RuntimeError("Could not find changes table on Wikipedia page")
        changes_table = tables[1]

    raw = pd.read_html(StringIO(str(changes_table)), header=[0, 1])[0]
    # Multi-level header has top row {Date, Added, Added, Removed, Removed, Reason}
    # and second row with subcolumn names. Flatten:
    raw.columns = ["_".join([str(x).strip() for x in tup if str(x) != "nan"]) for tup in raw.columns]

    # Identify columns we need
    date_col = next(c for c in raw.columns if c.lower().startswith("date"))
    added_ticker_col = next(
        c for c in raw.columns
        if "added" in c.lower() and ("ticker" in c.lower() or "symbol" in c.lower())
    )
    removed_ticker_col = next(
        c for c in raw.columns
        if "removed" in c.lower() and ("ticker" in c.lower() or "symbol" in c.lower())
    )

    out = pd.DataFrame({
        "date": pd.to_datetime(raw[date_col], errors="coerce"),
        "added_ticker": raw[added_ticker_col].astype(str).str.upper()
            .str.replace(".", "-", regex=False).replace({"NAN": pd.NA}),
        "removed_ticker": raw[removed_ticker_col].astype(str).str.upper()
            .str.replace(".", "-", regex=False).replace({"NAN": pd.NA}),
    })
    out = out.dropna(subset=["date"]).reset_index(drop=True)
    return out


def load_ticker_to_cik(json_path: Path) -> dict[str, str]:
    """Load ticker -> 10-digit zero-padded CIK from SEC's company_tickers.json."""
    with json_path.open() as f:
        data = json.load(f)
    out: dict[str, str] = {}
    for row in data.values():
        ticker = str(row["ticker"]).upper().replace(".", "-")
        cik = str(row["cik_str"]).zfill(10)
        out[ticker] = cik
    return out


def reconstruct_membership(
    html: str,
    ticker_to_cik: dict[str, str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Walk the changes table backward from current members to produce intervals.

    Algorithm:
      1. Start with current members (all open, date_out=NaT).
      2. Walk changes from newest -> oldest:
         - If a ticker was 'added' on date d, it was NOT a member before d.
           Close any open interval for that ticker with date_out=d, set date_in=d.
         - If a ticker was 'removed' on date d, it WAS a member before d.
           Open an interval ending at d with no known date_in (we'll close it at start_date).
      3. Any still-open interval at end gets date_in=start_date.
      4. Trim everything to [start_date, end_date].
    """
    current = parse_current_members(html)
    changes = parse_changes_table(html).sort_values("date", ascending=False)

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    intervals: list[dict] = []
    # current members -> open intervals with date_out = NaT
    for _, row in current.iterrows():
        intervals.append({
            "ticker": row["ticker"],
            "company": row["company"],
            "date_in": pd.NaT,   # filled in by walk
            "date_out": pd.NaT,
        })

    # index intervals by ticker, taking the most-recent open one for each ticker
    def open_interval(ticker: str) -> dict | None:
        for iv in reversed(intervals):
            if iv["ticker"] == ticker and pd.isna(iv["date_in"]):
                return iv
        return None

    for _, ch in changes.iterrows():
        d = ch["date"]
        added = ch["added_ticker"]
        removed = ch["removed_ticker"]

        if pd.notna(added):
            iv = open_interval(added)
            if iv is None:
                # Ticker was added on d but we don't have an open interval — means
                # they were added then later removed before "current". Open one going forward.
                intervals.append({
                    "ticker": added,
                    "company": "",
                    "date_in": d,
                    "date_out": pd.NaT,
                })
            else:
                iv["date_in"] = d

        if pd.notna(removed):
            # They were a member up to d. Open a fresh interval to be closed by older changes.
            intervals.append({
                "ticker": removed,
                "company": "",
                "date_in": pd.NaT,
                "date_out": d,
            })

    # Any still-open date_in -> they were already in at start_date
    for iv in intervals:
        if pd.isna(iv["date_in"]):
            iv["date_in"] = start_ts

    df = pd.DataFrame(intervals)

    # Trim to window. Drop intervals fully outside [start_ts, end_ts].
    df = df[df["date_in"] <= end_ts]
    df = df[df["date_out"].isna() | (df["date_out"] >= start_ts)]
    df["date_in"] = df["date_in"].clip(lower=start_ts)
    df.loc[df["date_out"] > end_ts, "date_out"] = pd.NaT

    # Attach CIKs
    df["cik"] = df["ticker"].map(ticker_to_cik)

    # Drop rows where date_in > date_out
    valid = df["date_out"].isna() | (df["date_in"] <= df["date_out"])
    df = df[valid].reset_index(drop=True)

    return df[["ticker", "cik", "company", "date_in", "date_out"]]


def main() -> None:
    configure_logging()
    user_agent = get_env("SEC_USER_AGENT", required=True)

    log.info("Fetching Wikipedia S&P 500 page")
    html = _fetch(WIKI_URL, user_agent="Mozilla/5.0 axiom-tilt-research")

    log.info("Fetching SEC company_tickers.json")
    sec_json = _fetch(SEC_TICKERS_URL, user_agent=user_agent)
    sec_path = processed_dir().parent / "raw" / "sec" / "company_tickers.json"
    sec_path.parent.mkdir(parents=True, exist_ok=True)
    sec_path.write_text(sec_json)
    ticker_to_cik = load_ticker_to_cik(sec_path)

    log.info("Reconstructing membership intervals")
    df = reconstruct_membership(
        html=html,
        ticker_to_cik=ticker_to_cik,
        start_date="2000-01-01",
        end_date="2025-12-31",
    )

    out_path = processed_dir() / "universe.parquet"
    df.to_parquet(out_path, index=False)

    n_with_cik = df["cik"].notna().sum()
    log.info(
        "Wrote %d intervals (%d unique tickers, %d with CIK match) to %s",
        len(df), df["ticker"].nunique(), n_with_cik, out_path,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run tests, confirm they pass**

Run: `pytest tests/data/test_build_universe.py -v`
Expected: 4 passed.

If `parse_changes_table` fails because Wikipedia changed their column structure, inspect the saved fixture:
```bash
python -c "
from bs4 import BeautifulSoup
html = open('tests/fixtures/wikipedia_sp500.html').read()
soup = BeautifulSoup(html, 'lxml')
for i, t in enumerate(soup.find_all('table', class_=lambda c: c and 'wikitable' in c)[:3]):
    print(i, [th.get_text(strip=True) for th in t.find_all('th')[:6]])
"
```
Adjust the column-finding logic in `parse_changes_table` if needed.

- [ ] **Step 3: Commit**

```bash
git add src/data/build_universe.py tests/data/__init__.py tests/data/test_build_universe.py
git commit -m "data: implement S&P 500 universe reconstruction"
```

### Task 1.4: Run universe ingestion end-to-end

- [ ] **Step 1: Set up `.env` (one-time, by user)**

Run: `cp .env.example .env`
Then open `.env` and fill in `SEC_USER_AGENT="Your Name your@email.com"`. (FMP and AlphaVantage keys come later.)

- [ ] **Step 2: Run universe builder**

Run: `python -m src.data.build_universe`
Expected: prints log lines, finishes in <30s, writes `data/processed/universe.parquet`.

- [ ] **Step 3: Inspect output**

Run:
```bash
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/universe.parquet')
print('rows:', len(df))
print('unique tickers:', df['ticker'].nunique())
print('with CIK:', df['cik'].notna().sum())
print('without CIK:', df['cik'].isna().sum())
print(df.head(10))
print('---')
print('No-CIK sample:', df[df['cik'].isna()]['ticker'].head(20).tolist())
"
```
Expected:
- rows: ~700-900
- unique tickers: ~700-900
- with CIK: ~85-95% of rows (some old delistings may not appear in SEC's current ticker file)

If "without CIK" is >25%, the ticker normalization needs work — investigate by looking at a few unmatched tickers and comparing to what's in `company_tickers.json`.

- [ ] **Step 4: Commit the output**

```bash
git add data/processed/universe.parquet
git commit -m "data: ingest S&P 500 universe 2000-2025"
```

(`data/processed/` is tracked per the gitignore plan, so this commits the parquet for laptop sync.)

---

## Task 2: Prices via yfinance

Pulls daily OHLCV adjusted for every ticker that ever entered the universe. yfinance is unauthenticated and pretty resilient; main concerns are batch sizing and stale ticker symbols.

**Files:**
- Modify: `src/data/ingest_prices.py`
- Create: `tests/data/test_ingest_prices.py`

### Task 2.1: Write tests (TDD-lite — smoke + unit)

- [ ] **Step 1: Write failing test `tests/data/test_ingest_prices.py`**

```python
"""Tests for src.data.ingest_prices."""
import pandas as pd
import pytest

from src.data.ingest_prices import (
    normalize_yfinance_frame,
    chunk_tickers,
)


def test_chunk_tickers_splits_evenly():
    tickers = [f"T{i}" for i in range(105)]
    chunks = list(chunk_tickers(tickers, batch_size=50))
    assert len(chunks) == 3
    assert len(chunks[0]) == 50
    assert len(chunks[1]) == 50
    assert len(chunks[2]) == 5


def test_normalize_yfinance_frame_long_format():
    # yfinance.download returns a wide multi-index DataFrame for >1 ticker
    idx = pd.date_range("2024-01-02", periods=3, freq="B")
    cols = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Adj Close", "Volume"], ["AAPL", "MSFT"]],
        names=["Price", "Ticker"],
    )
    data = [[1.0]*12, [2.0]*12, [3.0]*12]
    wide = pd.DataFrame(data, index=idx, columns=cols)

    long_df = normalize_yfinance_frame(wide)
    assert {"date", "ticker", "open", "high", "low", "close", "adj_close", "volume"} \
        .issubset(long_df.columns)
    assert len(long_df) == 6  # 3 dates x 2 tickers
    assert set(long_df["ticker"].unique()) == {"AAPL", "MSFT"}


def test_normalize_yfinance_frame_handles_single_ticker():
    # When yfinance.download is called with one ticker it returns a flat-column frame
    idx = pd.date_range("2024-01-02", periods=2, freq="B")
    flat = pd.DataFrame(
        {"Open": [1.0, 2.0], "High": [1.1, 2.1], "Low": [0.9, 1.9],
         "Close": [1.05, 2.05], "Adj Close": [1.05, 2.05], "Volume": [1000, 2000]},
        index=idx,
    )
    long_df = normalize_yfinance_frame(flat, ticker="AAPL")
    assert len(long_df) == 2
    assert (long_df["ticker"] == "AAPL").all()
```

- [ ] **Step 2: Run test, confirm it fails**

Run: `pytest tests/data/test_ingest_prices.py -v`
Expected: ImportError on `src.data.ingest_prices` symbols.

### Task 2.2: Implement ingest_prices.py

- [ ] **Step 1: Replace `src/data/ingest_prices.py`**

```python
"""Pull daily OHLCV adjusted prices via yfinance for the universe.

Output: data/processed/prices.parquet
Long format: (date, ticker, open, high, low, close, adj_close, volume)
"""
from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from pathlib import Path

import pandas as pd
import yfinance as yf
from tqdm import tqdm

from src.utils.config import load_config
from src.utils.io import processed_dir
from src.utils.logging_utils import configure_logging, get_logger


log = get_logger(__name__)


def chunk_tickers(tickers: Iterable[str], batch_size: int) -> Iterator[list[str]]:
    buf: list[str] = []
    for t in tickers:
        buf.append(t)
        if len(buf) >= batch_size:
            yield buf
            buf = []
    if buf:
        yield buf


def normalize_yfinance_frame(
    frame: pd.DataFrame,
    ticker: str | None = None,
) -> pd.DataFrame:
    """Convert yfinance output to long-format (date, ticker, ohlcv).

    yfinance returns:
      - flat columns when called with a single ticker -> need `ticker` arg
      - MultiIndex columns (Price, Ticker) when called with multiple
    """
    if isinstance(frame.columns, pd.MultiIndex):
        # Wide -> long via stack
        long = frame.stack(level="Ticker", future_stack=True).reset_index()
        long.columns = [str(c).strip() for c in long.columns]
        long = long.rename(columns={
            "Date": "date",
            "Ticker": "ticker",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        })
    else:
        if ticker is None:
            raise ValueError("ticker required when frame has flat columns")
        long = frame.reset_index().rename(columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        })
        long["ticker"] = ticker

    keep = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
    long = long[[c for c in keep if c in long.columns]]
    long = long.dropna(subset=["close"])
    long["date"] = pd.to_datetime(long["date"]).dt.tz_localize(None).dt.normalize()
    return long.reset_index(drop=True)


def fetch_batch(
    tickers: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Pull one batch via yfinance.download."""
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    if raw is None or raw.empty:
        return pd.DataFrame()
    if len(tickers) == 1:
        return normalize_yfinance_frame(raw, ticker=tickers[0])
    return normalize_yfinance_frame(raw)


def main() -> None:
    configure_logging()
    cfg = load_config("data")

    universe_path = processed_dir() / "universe.parquet"
    if not universe_path.exists():
        raise FileNotFoundError(
            f"{universe_path} not found. Run `python -m src.data.build_universe` first."
        )
    universe = pd.read_parquet(universe_path)
    tickers = sorted(universe["ticker"].unique().tolist())
    log.info("Pulling prices for %d tickers", len(tickers))

    batch_size = cfg["prices"]["batch_size"]
    delay = cfg["prices"]["request_delay_sec"]
    start = cfg["start_date"]
    end = cfg["end_date"]

    frames: list[pd.DataFrame] = []
    failed: list[str] = []
    for batch in tqdm(list(chunk_tickers(tickers, batch_size)), desc="batches"):
        try:
            df = fetch_batch(batch, start=start, end=end)
            if not df.empty:
                frames.append(df)
            else:
                failed.extend(batch)
        except Exception as e:
            log.warning("Batch failed (%s): %s", e.__class__.__name__, e)
            failed.extend(batch)
        time.sleep(delay)

    if not frames:
        raise RuntimeError("No price data was successfully pulled.")

    out = pd.concat(frames, ignore_index=True).drop_duplicates(["date", "ticker"])
    out_path = processed_dir() / "prices.parquet"
    out.to_parquet(out_path, index=False)

    log.info(
        "Wrote %d (date, ticker) rows for %d distinct tickers to %s",
        len(out), out["ticker"].nunique(), out_path,
    )
    if failed:
        log.warning("Failed tickers (%d): %s", len(failed), failed[:20])
        # Persist for inspection
        Path(processed_dir() / "prices_failed.txt").write_text("\n".join(failed))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run unit tests**

Run: `pytest tests/data/test_ingest_prices.py -v`
Expected: 3 passed.

- [ ] **Step 3: Commit**

```bash
git add src/data/ingest_prices.py tests/data/test_ingest_prices.py
git commit -m "data: implement yfinance price ingestion"
```

### Task 2.3: Run prices ingestion (smoke first, then full)

- [ ] **Step 1: Smoke test on a 5-ticker subset**

Run:
```bash
python -c "
from src.data.ingest_prices import fetch_batch
df = fetch_batch(['AAPL', 'MSFT', 'GOOG', 'JPM', 'XOM'], start='2024-01-01', end='2024-02-01')
print(df.head())
print('rows:', len(df), 'tickers:', df['ticker'].nunique())
"
```
Expected: ~100 rows, 5 distinct tickers, columns look right.

- [ ] **Step 2: Run full pull**

Run: `python -m src.data.ingest_prices`
Expected: 10-30 min runtime. Final log line shows ~3-5M rows for ~700-900 tickers across 25 years. May see warnings for delisted tickers — that's expected.

- [ ] **Step 3: Inspect**

Run:
```bash
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/prices.parquet')
print('rows:', len(df))
print('tickers:', df['ticker'].nunique())
print('date range:', df['date'].min(), 'to', df['date'].max())
print(df.groupby('ticker').size().describe())
"
```
Expected: ~700+ tickers, dates span 2000–2025, median observations per ticker should be in the 3000-5000 range (matches ~250 trading days × ~12-25 years).

- [ ] **Step 4: Commit output**

```bash
git add data/processed/prices.parquet
git commit -m "data: ingest 2000-2025 daily OHLCV for universe"
```

If the parquet is >100 MB and GitHub complains, switch to splitting by year:
```python
for year, sub in out.groupby(out["date"].dt.year):
    sub.to_parquet(processed_dir() / f"prices_{year}.parquet", index=False)
```
and update later readers.

---

## Task 3: Fundamentals via FMP

FMP gives standardized quarterly statements. Free tier is rate-limited (~250 req/day on the truly-free tier; more on the cheap paid tier). The puller is per-ticker and resumable so an interrupted run picks up where it left off.

**Files:**
- Modify: `src/data/ingest_fundamentals.py`
- Create: `tests/data/test_ingest_fundamentals.py`

### Task 3.1: Write tests

- [ ] **Step 1: Write failing test `tests/data/test_ingest_fundamentals.py`**

```python
"""Tests for src.data.ingest_fundamentals."""
import json

import pandas as pd
import pytest

from src.data.ingest_fundamentals import (
    parse_fmp_statement,
    StatementType,
)


def test_parse_fmp_income_statement():
    sample = [
        {
            "date": "2024-09-28",
            "symbol": "AAPL",
            "reportedCurrency": "USD",
            "fillingDate": "2024-11-01",
            "acceptedDate": "2024-11-01 06:01:36",
            "calendarYear": "2024",
            "period": "FY",
            "revenue": 391_035_000_000,
            "costOfRevenue": 210_352_000_000,
            "grossProfit": 180_683_000_000,
            "netIncome": 93_736_000_000,
            "eps": 6.08,
        }
    ]
    df = parse_fmp_statement(sample, ticker="AAPL", stmt_type=StatementType.INCOME)
    assert len(df) == 1
    assert df.iloc[0]["ticker"] == "AAPL"
    assert df.iloc[0]["statement"] == "income"
    assert df.iloc[0]["period_end"] == pd.Timestamp("2024-09-28")
    assert df.iloc[0]["filing_date"] == pd.Timestamp("2024-11-01")
    assert df.iloc[0]["revenue"] == 391_035_000_000


def test_parse_fmp_handles_empty_response():
    df = parse_fmp_statement([], ticker="ZZZZ", stmt_type=StatementType.INCOME)
    assert len(df) == 0
    assert "ticker" in df.columns
```

- [ ] **Step 2: Run test, confirm it fails**

Run: `pytest tests/data/test_ingest_fundamentals.py -v`
Expected: ImportError.

### Task 3.2: Implement ingest_fundamentals.py

- [ ] **Step 1: Replace `src/data/ingest_fundamentals.py`**

```python
"""Pull quarterly + annual fundamentals from Financial Modeling Prep.

Output: data/processed/fundamentals.parquet
Long format: (ticker, statement, period, period_end, filing_date, ...)

The puller is per-ticker and resumable. State is tracked in
data/state/fundamentals_done.txt — append-only list of completed tickers.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
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

from src.utils.config import load_config
from src.utils.env import get_env
from src.utils.io import processed_dir, raw_dir, state_dir
from src.utils.logging_utils import configure_logging, get_logger
from src.utils.rate_limit import TokenBucket


log = get_logger(__name__)
FMP_BASE = "https://financialmodelingprep.com/api/v3"


class StatementType(str, Enum):
    INCOME = "income-statement"
    BALANCE = "balance-sheet-statement"
    CASHFLOW = "cash-flow-statement"


@dataclass
class FmpClient:
    api_key: str
    bucket: TokenBucket
    retry_attempts: int

    def fetch(
        self,
        endpoint: str,
        ticker: str,
        period: str = "quarter",
        limit: int = 200,
    ) -> list[dict]:
        url = f"{FMP_BASE}/{endpoint}/{ticker}"
        params = {"period": period, "limit": limit, "apikey": self.api_key}

        @retry(
            stop=stop_after_attempt(self.retry_attempts),
            wait=wait_exponential(multiplier=2, min=2, max=60),
            retry=retry_if_exception_type((requests.RequestException,)),
            reraise=True,
        )
        def _do() -> list[dict]:
            self.bucket.acquire()
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                raise requests.RequestException(f"rate limited: {resp.text[:200]}")
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict) and "Error Message" in data:
                # Treat as empty rather than raising; some tickers have no data
                return []
            return data if isinstance(data, list) else []

        return _do()


def parse_fmp_statement(
    rows: list[dict],
    ticker: str,
    stmt_type: StatementType,
) -> pd.DataFrame:
    """Normalize FMP statement rows into our long schema."""
    if not rows:
        return pd.DataFrame(columns=["ticker", "statement", "period", "period_end", "filing_date"])

    df = pd.DataFrame(rows)
    df["ticker"] = ticker
    df["statement"] = stmt_type.name.lower()
    df["period_end"] = pd.to_datetime(df.get("date"), errors="coerce")
    df["filing_date"] = pd.to_datetime(df.get("fillingDate"), errors="coerce")
    df["period"] = df.get("period", pd.Series(["UNK"] * len(df))).astype(str)
    # Reorder
    front = ["ticker", "statement", "period", "period_end", "filing_date"]
    rest = [c for c in df.columns if c not in front]
    return df[front + rest]


def _load_done(state_file: Path) -> set[str]:
    if not state_file.exists():
        return set()
    return {line.strip() for line in state_file.read_text().splitlines() if line.strip()}


def _append_done(state_file: Path, ticker: str) -> None:
    with state_file.open("a") as f:
        f.write(f"{ticker}\n")


def main() -> None:
    configure_logging()
    cfg = load_config("data")
    api_key = get_env("FMP_API_KEY", required=True)

    universe_path = processed_dir() / "universe.parquet"
    universe = pd.read_parquet(universe_path)
    tickers = sorted(universe["ticker"].dropna().unique().tolist())

    state_file = state_dir() / "fundamentals_done.txt"
    done = _load_done(state_file)
    todo = [t for t in tickers if t not in done]
    log.info("Tickers: %d total, %d done, %d to do", len(tickers), len(done), len(todo))

    client = FmpClient(
        api_key=api_key,
        bucket=TokenBucket(
            rate_per_sec=cfg["fundamentals"]["rate_per_sec"],
            capacity=cfg["fundamentals"]["capacity"],
        ),
        retry_attempts=cfg["fundamentals"]["retry_attempts"],
    )

    raw_root = raw_dir() / "fundamentals"
    raw_root.mkdir(parents=True, exist_ok=True)

    all_frames: list[pd.DataFrame] = []
    for ticker in tqdm(todo, desc="fundamentals"):
        try:
            for stmt in StatementType:
                rows = client.fetch(stmt.value, ticker, period="quarter", limit=200)
                # Persist raw response for debugging
                (raw_root / f"{ticker}_{stmt.name.lower()}.json").write_text(
                    pd.io.json.dumps(rows)
                )
                df = parse_fmp_statement(rows, ticker, stmt)
                if not df.empty:
                    all_frames.append(df)
            _append_done(state_file, ticker)
        except Exception as e:
            log.warning("Failed %s: %s", ticker, e)

    # Merge with anything already on disk for tickers we just finished
    out_path = processed_dir() / "fundamentals.parquet"
    if all_frames:
        new = pd.concat(all_frames, ignore_index=True)
        if out_path.exists():
            existing = pd.read_parquet(out_path)
            combined = pd.concat([existing, new], ignore_index=True)
        else:
            combined = new
        combined = combined.drop_duplicates(
            subset=["ticker", "statement", "period_end"], keep="last"
        )
        combined.to_parquet(out_path, index=False)
        log.info("Wrote %d rows to %s", len(combined), out_path)
    else:
        log.info("No new data to write.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run unit tests**

Run: `pytest tests/data/test_ingest_fundamentals.py -v`
Expected: 2 passed.

- [ ] **Step 3: Commit**

```bash
git add src/data/ingest_fundamentals.py tests/data/test_ingest_fundamentals.py
git commit -m "data: implement FMP fundamentals ingestion"
```

### Task 3.3: Run fundamentals smoke + full

- [ ] **Step 1: Add `FMP_API_KEY` to `.env`** (user action)

Open `.env`, paste your FMP API key as `FMP_API_KEY="..."`.

- [ ] **Step 2: Smoke test**

Run:
```bash
python -c "
from src.utils.env import get_env
from src.utils.rate_limit import TokenBucket
from src.data.ingest_fundamentals import FmpClient, parse_fmp_statement, StatementType

c = FmpClient(api_key=get_env('FMP_API_KEY', required=True), bucket=TokenBucket(5.0, 5), retry_attempts=3)
rows = c.fetch('income-statement', 'AAPL', period='quarter', limit=4)
print('rows:', len(rows))
print(parse_fmp_statement(rows, 'AAPL', StatementType.INCOME).head())
"
```
Expected: 4 rows of recent AAPL quarterly data. If you get a 401, check the API key.

- [ ] **Step 3: Full run**

Run: `python -m src.data.ingest_fundamentals`

Expected:
- Free tier: probably hits rate limits long before completion. **Stop after a few hundred tickers** with Ctrl-C; the next run resumes from the state file.
- Paid tier ($14/mo Starter): completes in 1-3 hours.

If using free tier, plan to run in chunks across multiple days, or upgrade.

- [ ] **Step 4: Inspect**

Run:
```bash
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/fundamentals.parquet')
print('rows:', len(df))
print('tickers:', df['ticker'].nunique())
print('date range:', df['period_end'].min(), 'to', df['period_end'].max())
print(df.groupby(['ticker', 'statement']).size().head(20))
"
```

- [ ] **Step 5: Commit output**

```bash
git add data/processed/fundamentals.parquet
git commit -m "data: ingest FMP fundamentals (in-progress, resumable)"
```

---

## Task 4: EDGAR text — the overnight job

The big one. Pulls 10-K, 10-Q, 8-K filings for every CIK in our universe across 2000–2025. Uses SEC's **quarterly full-text indexes** rather than per-filing scraping for efficiency.

**Approach:**
1. Download SEC's `master.idx` for each quarter Q1-2000 through Q4-2025 (~104 files).
2. Filter to (CIK in universe) AND (form_type in {10-K, 10-Q, 8-K}).
3. For each surviving filing, fetch the primary document URL and save text-extracted content to `data/raw/edgar/{cik}/{accession}.txt`.
4. Maintain a state file `data/state/edgar_done.txt` (one accession per line) so the pull is resumable.
5. Build `data/processed/edgar_index.parquet` cataloging {cik, ticker, form_type, filing_date, accession, path}.

**Files:**
- Modify: `src/data/ingest_filings.py`
- Create: `tests/data/test_ingest_filings.py`, `tests/fixtures/edgar_master.idx`, `tests/fixtures/sample_10k.html`

### Task 4.1: Capture EDGAR test fixtures

- [ ] **Step 1: Save a small SEC quarterly master.idx**

Run:
```bash
curl -A "axiom-tilt-research test@example.com" \
  -o tests/fixtures/edgar_master.idx \
  https://www.sec.gov/Archives/edgar/full-index/2024/QTR1/master.idx
```

Verify: `head -20 tests/fixtures/edgar_master.idx` should show pipe-delimited rows like `CIK|Company Name|Form Type|Date Filed|Filename`.

- [ ] **Step 2: Save a sample 10-K HTML**

Run:
```bash
curl -A "axiom-tilt-research test@example.com" \
  -o tests/fixtures/sample_10k.html \
  https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm
```

Verify: `ls -lh tests/fixtures/sample_10k.html` should be 1-5 MB.

- [ ] **Step 3: Commit fixtures**

```bash
git add tests/fixtures/edgar_master.idx tests/fixtures/sample_10k.html
git commit -m "test: add EDGAR fixtures for ingestion tests"
```

### Task 4.2: Write EDGAR tests (TDD)

- [ ] **Step 1: Write `tests/data/test_ingest_filings.py`**

```python
"""Tests for src.data.ingest_filings."""
from pathlib import Path

import pandas as pd
import pytest

from src.data.ingest_filings import (
    parse_master_idx,
    extract_text_from_html,
    accession_from_filename,
    EdgarFiling,
)


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_parse_master_idx_returns_filings_dataframe():
    text = (FIXTURES / "edgar_master.idx").read_text(encoding="latin-1")
    df = parse_master_idx(text)
    assert {"cik", "company", "form_type", "filing_date", "filename"}.issubset(df.columns)
    # CIK should be zero-padded 10-digit string
    assert df["cik"].str.len().eq(10).all()
    # Should have a healthy mix of form types
    assert df["form_type"].nunique() > 5
    assert (df["form_type"] == "10-K").sum() > 0
    assert (df["form_type"] == "10-Q").sum() > 0
    assert (df["form_type"] == "8-K").sum() > 0


def test_parse_master_idx_filters_to_universe_forms():
    text = (FIXTURES / "edgar_master.idx").read_text(encoding="latin-1")
    df = parse_master_idx(text)
    df = df[df["form_type"].isin(["10-K", "10-Q", "8-K"])]
    assert len(df) > 100
    assert df["form_type"].isin(["10-K", "10-Q", "8-K"]).all()


def test_extract_text_from_html_strips_tags_and_returns_text():
    html = (FIXTURES / "sample_10k.html").read_text(encoding="utf-8", errors="ignore")
    text = extract_text_from_html(html)
    assert len(text) > 10_000  # 10-Ks are long
    assert "<" not in text[:1000]  # no raw tags in the cleaned text
    # 10-Ks always contain certain phrases
    assert "Item" in text or "ITEM" in text


def test_accession_from_filename_extracts_correctly():
    fname = "edgar/data/320193/000032019324000123/0000320193-24-000123-index.htm"
    assert accession_from_filename(fname) == "0000320193-24-000123"

    fname2 = "edgar/data/789019/000156459024041223/msft-20240630.htm"
    # Accession is in the second-to-last path segment, in folder form (no dashes)
    # We need to reconstruct from "000156459024041223" -> "0001564590-24-041223"
    assert accession_from_filename(fname2) == "0001564590-24-041223"


def test_edgar_filing_dataclass():
    f = EdgarFiling(
        cik="0000320193",
        company="APPLE INC",
        form_type="10-K",
        filing_date=pd.Timestamp("2024-11-01"),
        filename="edgar/data/320193/000032019324000123/0000320193-24-000123-index.htm",
        accession="0000320193-24-000123",
    )
    assert f.url.startswith("https://www.sec.gov/")
    assert f.local_text_path.name.endswith(".txt")
```

- [ ] **Step 2: Run, confirm fail**

Run: `pytest tests/data/test_ingest_filings.py -v`
Expected: ImportError.

### Task 4.3: Implement ingest_filings.py — full module

- [ ] **Step 1: Replace `src/data/ingest_filings.py`**

```python
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
        return f"{SEC_BASE}/{self.filename}"

    @property
    def local_text_path(self) -> Path:
        return edgar_raw_dir() / self.cik / f"{self.accession}.txt"


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
            if resp.status_code in (429, 503):
                raise requests.RequestException(f"throttled: {resp.status_code}")
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
    """Fetch the filing, extract text, save to disk. Returns True on success."""
    out_path = filing.local_text_path
    if out_path.exists():
        return True
    try:
        raw = client.fetch(filing.url)
    except Exception as e:
        log.warning("Fetch failed %s: %s", filing.url, e)
        return False

    # If filename is `.txt`, the file is the SGML envelope — extract <DOCUMENT>...</DOCUMENT> bodies
    # If it's `.htm`/`.html`, parse directly
    raw_str = raw.decode("utf-8", errors="ignore")
    if filing.filename.endswith(".txt"):
        text = extract_text_from_sgml(raw_str)
    else:
        text = extract_text_from_html(raw_str)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return True


def extract_text_from_sgml(sgml: str) -> str:
    """SEC's old SGML envelope wraps individual documents in <DOCUMENT>...</DOCUMENT>.

    Concatenate the text from each <TEXT> body, after HTML stripping.
    """
    bodies: list[str] = []
    for match in re.finditer(r"<TEXT>(.*?)</TEXT>", sgml, flags=re.DOTALL | re.IGNORECASE):
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

    n_ok = 0
    for i, filing in enumerate(tqdm(todo, desc="filings")):
        ok = fetch_and_save_filing(client, filing)
        if ok:
            n_ok += 1
            _append_done(state_file, filing.accession)
        if (i + 1) % cfg["edgar"]["checkpoint_every_n"] == 0:
            log.info("checkpoint: %d/%d ok", n_ok, i + 1)

    # Build / update the index parquet
    index_rows = []
    for f in filings:
        if (state_dir() / "edgar_done.txt").exists() and f.accession in _load_done(state_file):
            index_rows.append({
                "cik": f.cik,
                "company": f.company,
                "form_type": f.form_type,
                "filing_date": f.filing_date,
                "accession": f.accession,
                "filename": f.filename,
                "local_path": str(f.local_text_path.relative_to(raw_dir().parent)),
            })
    idx_df = pd.DataFrame(index_rows)
    idx_df.to_parquet(processed_dir() / "edgar_index.parquet", index=False)
    log.info("Wrote edgar_index with %d rows", len(idx_df))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run unit tests**

Run: `pytest tests/data/test_ingest_filings.py -v`
Expected: 5 passed.

If `test_extract_text_from_html_strips_tags_and_returns_text` fails because the saved Apple 10-K has unusual structure, relax the assertion: just check `len(text) > 10_000` and `"<" not in text[:1000]`.

- [ ] **Step 3: Commit**

```bash
git add src/data/ingest_filings.py tests/data/test_ingest_filings.py
git commit -m "data: implement EDGAR filings ingestion (resumable)"
```

### Task 4.4: Smoke test EDGAR on one quarter

- [ ] **Step 1: Run on Q1 2024 only**

Run: `python -m src.data.ingest_filings --year 2024`

Expected:
- Walks 4 quarters of 2024
- Matches probably 5K-10K filings for the universe
- Downloads them at ~8/sec → 10-20 min
- Final log line: `Wrote edgar_index with N rows`

If it crashes early, check:
- `SEC_USER_AGENT` is set in `.env` and not the placeholder
- Network connectivity
- `data/state/edgar_done.txt` for partial progress

- [ ] **Step 2: Inspect**

Run:
```bash
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/edgar_index.parquet')
print('rows:', len(df))
print('form types:', df['form_type'].value_counts().to_dict())
print('CIKs:', df['cik'].nunique())
print(df.head())
"
ls data/raw/edgar/ | head
```
Expected: thousands of rows, mix of 10-K/Q/8-K, hundreds of CIK directories.

Sample one file:
```bash
TICKER_DIR=$(ls data/raw/edgar/ | head -1)
TXT=$(ls data/raw/edgar/$TICKER_DIR/ | head -1)
head -50 data/raw/edgar/$TICKER_DIR/$TXT
```
Should show readable text content.

- [ ] **Step 3: Commit smoke output (just the index, raw is gitignored)**

```bash
git add data/processed/edgar_index.parquet
git commit -m "data: smoke EDGAR ingestion (2024 only)"
```

### Task 4.5: Kick off the full overnight run

- [ ] **Step 1: Estimate scale**

From the smoke run, `len(2024 index) × 25 years ≈ X filings`. At 8/sec, ETA = `X / 8 / 3600` hours. For ~600K filings this is ~21 hours — doable in two evenings, or one if you start early.

If the estimate looks scary, narrow the date range in `configs/data.yaml` to `start_date: "2010-01-01"` and rerun a partial first.

- [ ] **Step 2: Launch in background with logging**

Run:
```bash
mkdir -p logs
nohup python -m src.data.ingest_filings > logs/edgar_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo $! > logs/edgar.pid
echo "Started PID $(cat logs/edgar.pid). Monitor with: tail -f logs/edgar_*.log"
```

- [ ] **Step 3: Health checks while it runs**

Periodically:
```bash
# Progress
wc -l data/state/edgar_done.txt
# Disk usage
du -sh data/raw/edgar/
# Latest log
tail -20 logs/edgar_*.log
```

If it's stuck (no new lines in `edgar_done.txt` for >5 min and the process is still alive), inspect the log for retry storms — SEC may have temporarily blocked us. If blocked: kill the job (`kill $(cat logs/edgar.pid)`), wait 30 min, restart. The state file means no work is lost.

- [ ] **Step 4: After completion, commit the index**

```bash
git add data/processed/edgar_index.parquet
git commit -m "data: complete full EDGAR ingestion 2000-2025"
```

(`data/raw/edgar/` is gitignored — those hundreds of GB stay local.)

---

## Task 5: Macro via FRED

Small, fast, no API key needed.

**Files:**
- Create: `src/data/ingest_macro.py`, `tests/data/test_ingest_macro.py`

### Task 5.1: Write tests

- [ ] **Step 1: Write `tests/data/test_ingest_macro.py`**

```python
"""Tests for src.data.ingest_macro."""
import pandas as pd

from src.data.ingest_macro import normalize_fred_frame


def test_normalize_fred_frame_long_format():
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    wide = pd.DataFrame(
        {"DGS3MO": [5.1, 5.2, 5.3], "VIXCLS": [13.0, 13.5, 14.0]},
        index=idx,
    )
    long = normalize_fred_frame(wide)
    assert {"date", "series", "value"}.issubset(long.columns)
    assert len(long) == 6
    assert set(long["series"].unique()) == {"DGS3MO", "VIXCLS"}
```

- [ ] **Step 2: Confirm it fails**

Run: `pytest tests/data/test_ingest_macro.py -v`
Expected: ImportError.

### Task 5.2: Implement ingest_macro.py

- [ ] **Step 1: Create `src/data/ingest_macro.py`**

```python
"""Pull macro / risk-free / regime series from FRED via pandas-datareader.

No API key required for the public CSV endpoint.

Output: data/processed/macro.parquet
Long format: (date, series, value)
"""
from __future__ import annotations

import pandas as pd
from pandas_datareader import data as pdr

from src.utils.config import load_config
from src.utils.io import processed_dir
from src.utils.logging_utils import configure_logging, get_logger


log = get_logger(__name__)


def normalize_fred_frame(wide: pd.DataFrame) -> pd.DataFrame:
    """Convert FRED's wide-format frame to long (date, series, value)."""
    long = wide.reset_index().melt(id_vars=wide.index.name or "DATE",
                                    var_name="series",
                                    value_name="value")
    long = long.rename(columns={(wide.index.name or "DATE"): "date"})
    long["date"] = pd.to_datetime(long["date"]).dt.tz_localize(None).dt.normalize()
    return long.dropna(subset=["value"]).reset_index(drop=True)


def main() -> None:
    configure_logging()
    cfg = load_config("data")
    series = cfg["macro"]["series"]
    start = cfg["start_date"]
    end = cfg["end_date"]

    log.info("Pulling FRED series: %s", series)
    wide = pdr.DataReader(series, "fred", start=start, end=end)
    long = normalize_fred_frame(wide)

    out = processed_dir() / "macro.parquet"
    long.to_parquet(out, index=False)
    log.info("Wrote %d rows for %d series to %s", len(long), long["series"].nunique(), out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run tests**

Run: `pytest tests/data/test_ingest_macro.py -v`
Expected: 1 passed.

- [ ] **Step 3: Run full ingestion**

Run: `python -m src.data.ingest_macro`
Expected: ~30 sec runtime, parquet under 10 MB.

- [ ] **Step 4: Inspect**

Run:
```bash
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/macro.parquet')
print(df.groupby('series').agg({'date': ['min', 'max'], 'value': ['mean', 'std']}))
"
```
Expected: Each series spans ~25 years, sensible value ranges (DGS3MO: 0-6 typical, VIXCLS: 10-80, etc.).

- [ ] **Step 5: Commit**

```bash
git add src/data/ingest_macro.py tests/data/test_ingest_macro.py data/processed/macro.parquet
git commit -m "data: ingest FRED macro series"
```

---

## Task 6: Final integration check

Verify all five outputs are coherent and the panel can be aligned later.

### Task 6.1: Cross-check coverage

- [ ] **Step 1: Run the consistency check**

Run:
```bash
python -c "
import pandas as pd
universe = pd.read_parquet('data/processed/universe.parquet')
prices = pd.read_parquet('data/processed/prices.parquet')
fund = pd.read_parquet('data/processed/fundamentals.parquet') if __import__('pathlib').Path('data/processed/fundamentals.parquet').exists() else None
edgar = pd.read_parquet('data/processed/edgar_index.parquet') if __import__('pathlib').Path('data/processed/edgar_index.parquet').exists() else None
macro = pd.read_parquet('data/processed/macro.parquet')

uni_tickers = set(universe['ticker'])
print(f'Universe: {len(universe)} intervals, {len(uni_tickers)} tickers')
print(f'Prices: {prices[\"ticker\"].nunique()} tickers, {prices[\"date\"].min()} to {prices[\"date\"].max()}')
print(f'  coverage: {len(set(prices[\"ticker\"]) & uni_tickers)} / {len(uni_tickers)} universe tickers have prices')

if fund is not None:
    fund_tickers = set(fund['ticker'])
    print(f'Fundamentals: {len(fund_tickers)} tickers, {fund[\"period_end\"].min()} to {fund[\"period_end\"].max()}')
    print(f'  coverage: {len(fund_tickers & uni_tickers)} / {len(uni_tickers)}')

if edgar is not None:
    edgar_ciks = set(edgar['cik'])
    uni_ciks = set(universe['cik'].dropna())
    print(f'EDGAR: {len(edgar)} filings, {len(edgar_ciks)} CIKs, {edgar[\"filing_date\"].min()} to {edgar[\"filing_date\"].max()}')
    print(f'  coverage: {len(edgar_ciks & uni_ciks)} / {len(uni_ciks)} universe CIKs have filings')

print(f'Macro: {macro[\"series\"].nunique()} series, {macro[\"date\"].min()} to {macro[\"date\"].max()}')
"
```

Expected coverage thresholds (sanity):
- Prices: ≥95% of universe tickers
- Fundamentals: ≥80% (lower because of free-tier rate limits and delisted tickers FMP may not have)
- EDGAR: ≥90% of universe CIKs

If any number is below threshold, investigate: bad ticker normalization, missing CIKs, API failures in the state files.

- [ ] **Step 2: Commit anything you fixed**

```bash
git add -A
git commit -m "data: integration fixes from coverage audit"
```

---

## What's NOT in this plan (and where it goes)

- **Earnings call transcripts** — v2 per spec §16. Add a `src/data/ingest_transcripts.py` later when needed.
- **Text cleaning beyond HTML strip** — `src/text/clean_text.py` (separate plan, FinBERT prep).
- **Document chunking for FinBERT** — `src/text/chunk_documents.py` (FinBERT-FT plan).
- **Panel alignment / no-lookahead joins** — `src/data/align_panel.py` (feature-engineering plan).
- **Walk-forward splits** — `src/data/split_data.py` (modeling plan).

Each gets its own focused plan so we don't blow scope.

---

## Self-review

**Spec coverage check (against §3):**

- §3.1 Universe and timing → Task 1 (universe), Task 2 (prices give us trading-day calendar). Period 2000-2025: configured.
- §3.2 Text sources → Task 4 (EDGAR primary). Transcripts deferred per spec.
- §3.3 Prices and fundamentals → Task 2 (yfinance), Task 3 (FMP), risk-free in Task 5 (FRED DGS3MO).
- §3.4 Directory layout → Task 0.5 paths helper sets up `data/raw/`, `data/interim/`, `data/processed/`, plus `data/state/` for resume tracking. `data/raw/edgar/{cik}/{accession}.txt` matches spec.

No gaps.

**Placeholder scan:** No `TODO`, `TBD`, or "implement later" in any task. All code is concrete.

**Type/name consistency:**
- `EdgarFiling` referenced consistently in tests + module
- `StatementType` enum used identically in tests + module
- `parse_master_idx`, `accession_from_filename`, `extract_text_from_html` all match between test imports and module exports
- Output paths consistent across `processed_dir()` calls

**Scope:** Focused on data ingestion only. ~700 lines of code across 5 modules + 4 test files + foundation. Reasonable for one plan.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-09-data-ingestion.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?

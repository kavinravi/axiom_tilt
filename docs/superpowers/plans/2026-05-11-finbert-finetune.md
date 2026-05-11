# FinBERT MLM Fine-Tune Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce two static artifacts so the user (or a Cursor IDE agent) can run FinBERT MLM continued pretraining unattended: (1) `src/data/clean_filings.py` that extracts plain text from raw EDGAR SGML envelopes via multiprocessing+regex, and (2) `notebooks/01_finbert_finetune.ipynb` that handles tokenization, dry-run, full training, and eval with a Cursor-IDE-safe progress callback.

**Architecture:** Two-file deliverable. The `.py` runs first via CLI (`python -m src.data.clean_filings`, ~6 hr CPU). The `.ipynb` is opened in Cursor and run cell-by-cell (Phase 2 tokenization → Phase 3 dry-run → Phase 3 full training ~36-48 hr GPU → Phase 4 eval). The notebook itself is *built* by a Python script (`scripts/build_finbert_notebook.py`) using the `nbformat` library — keeps cell content under source control as a normal Python file and avoids hand-editing brittle JSON.

**Tech Stack:** Python 3.11, `transformers` ≥4.46, `datasets` ≥3.0, `accelerate` ≥1.0, `torch` ≥2.7 (cu128 for Blackwell sm_120), `nbformat` (bundled with Jupyter), `multiprocessing` from stdlib.

**CRITICAL:** This plan produces artifacts only. The implementer MUST NOT execute `clean_filings.py` on the full corpus, MUST NOT execute the notebook, and MUST NOT launch any long-running training. Validation runs are bounded (single-file smoke, dry-run sanity) only.

---

## Scope

Two deliverables and one build script. No model training, no full-corpus extraction. After this plan completes, the user runs the artifacts themselves in Cursor.

## File Structure

**Created:**
- `src/data/clean_filings.py` — multiprocessing regex extractor (~150 LOC)
- `tests/data/test_clean_filings.py` — TDD tests (~8 tests)
- `tests/fixtures/sample_sgml.txt` — synthetic small SGML envelope for tests
- `scripts/build_finbert_notebook.py` — Python script that builds the notebook via nbformat (~500 LOC, mostly cell-string content)
- `notebooks/01_finbert_finetune.ipynb` — generated artifact (committed for convenience)

**Modified:**
- `.gitignore` — add `data/processed/finbert_tok/` (large, regeneratable Arrow dir)
- `requirements.txt` — explicit `nbformat>=5.10` (bundled but pin)

**Out of scope:**
- Embedding generation (Phase 5 — separate notebook `02_finbert_embed.ipynb`, later)
- Walk-forward FT
- Off-the-shelf FinBERT ablation

---

## Task 0: Foundation tweaks

**Files:**
- Modify: `requirements.txt`, `.gitignore`

### Task 0.1: Add nbformat to requirements

- [ ] **Step 1: Open `requirements.txt` and add nbformat under the Tooling block**

Append before any existing dev-tools section:

```
# Notebook tooling
nbformat>=5.10
jupyter>=1.0
```

(If `jupyter` is already listed, only add `nbformat`.)

- [ ] **Step 2: Install**

Run: `pip install -r requirements.txt`
Expected: clean install. `nbformat` may already be present via transitive deps; explicit pin guarantees it.

- [ ] **Step 3: Verify nbformat works**

Run: `python -c "import nbformat; print(nbformat.__version__)"`
Expected: prints a version ≥5.10.

### Task 0.2: Update .gitignore

- [ ] **Step 1: Open `.gitignore` and locate the existing `data/` block**

The existing block looks like:
```
data/raw/
data/interim/
data/embeddings/
```

- [ ] **Step 2: Add `data/processed/finbert_tok/` to that block**

Result:
```
data/raw/
data/interim/
data/embeddings/
data/processed/finbert_tok/
```

(Other parquets under `data/processed/` remain tracked.)

- [ ] **Step 3: Commit Task 0**

```bash
git add requirements.txt .gitignore
git commit -m "build: add nbformat dep and gitignore finbert_tok"
```

---

## Task 1: `clean_filings.py` with TDD

**Files:**
- Create: `src/data/clean_filings.py`
- Test: `tests/data/test_clean_filings.py`
- Test fixture: `tests/fixtures/sample_sgml.txt`

### Task 1.1: Create the SGML test fixture

- [ ] **Step 1: Create `tests/fixtures/sample_sgml.txt` with synthetic SGML content**

```
<SEC-DOCUMENT>0000123456-24-000001.txt : 20240101
<SEC-HEADER>0000123456-24-000001.hdr.sgml : 20240101
ACCESSION NUMBER: 0000123456-24-000001
CONFORMED SUBMISSION TYPE: 10-K
</SEC-HEADER>
<DOCUMENT>
<TYPE>10-K
<TEXT>
<html><body>
<p>This is the <b>primary</b> document body.</p>
<p>It contains some &amp; HTML entities and normal text.</p>
<p>Item 1A. Risk Factors. We face various risks including market risk, credit risk, and operational risk.</p>
<p>The company reported revenue of $1.2 billion in fiscal year 2024.</p>
<p>Management discussion and analysis follows.</p>
<p>This text needs to be at least 500 characters long to pass the minimum length filter in the cleaner. Adding more filler content to ensure we exceed the threshold for a realistic test case that mirrors what real SEC filings contain. The 10-K form is comprehensive and includes financial statements, footnotes, risk factors, and management discussion. End of body.</p>
</body></html>
</TEXT>
</DOCUMENT>
<DOCUMENT>
<TYPE>EX-99.1
<TEXT TYPE="EX-99.1">
<html><body>
<p>Exhibit 99.1 - Earnings Release</p>
<p>Short exhibit body.</p>
</body></html>
</TEXT>
</DOCUMENT>
</SEC-DOCUMENT>
```

This fixture intentionally has two `<TEXT>` blocks (one bare, one with attributes) to verify both regex paths.

- [ ] **Step 2: Commit fixture**

```bash
git add tests/fixtures/sample_sgml.txt
git commit -m "test: add synthetic SGML fixture for clean_filings tests"
```

### Task 1.2: Write failing tests for clean_filings

- [ ] **Step 1: Create `tests/data/test_clean_filings.py`**

```python
"""Tests for src.data.clean_filings."""
from pathlib import Path

import pytest

from src.data.clean_filings import (
    extract_sgml_bodies,
    strip_html_tags,
    clean_text,
    process_filing,
    MIN_TEXT_LENGTH,
)


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_extract_sgml_bodies_returns_multiple_blocks():
    raw = (FIXTURES / "sample_sgml.txt").read_text(encoding="utf-8")
    bodies = extract_sgml_bodies(raw)
    assert len(bodies) == 2
    # First body should contain "primary document body"
    assert "primary" in bodies[0]
    # Second body should contain "Exhibit 99.1"
    assert "Exhibit 99.1" in bodies[1]


def test_extract_sgml_bodies_handles_attributes():
    # The regex must match <TEXT> AND <TEXT TYPE="...">
    raw = "<TEXT>plain body</TEXT> middle <TEXT TYPE=\"EX-99\">attr body</TEXT>"
    bodies = extract_sgml_bodies(raw)
    assert len(bodies) == 2
    assert "plain body" in bodies[0]
    assert "attr body" in bodies[1]


def test_extract_sgml_bodies_empty_for_no_text_blocks():
    raw = "<html><body>no SGML envelope</body></html>"
    bodies = extract_sgml_bodies(raw)
    assert bodies == []


def test_strip_html_tags_removes_tags_and_decodes_entities():
    html = "<p>Hello <b>world</b> &amp; friends</p>"
    out = strip_html_tags(html)
    # tags gone
    assert "<" not in out and ">" not in out
    # entity decoded
    assert "&" in out
    # content preserved
    assert "Hello" in out and "world" in out and "friends" in out


def test_strip_html_tags_collapses_whitespace():
    html = "<p>line one\n\nline two</p>   <p>line\tthree</p>"
    out = strip_html_tags(html)
    # Should not contain double-space, tab, or newline
    assert "  " not in out
    assert "\t" not in out
    assert "\n" not in out


def test_clean_text_combines_extraction_and_stripping():
    raw = (FIXTURES / "sample_sgml.txt").read_text(encoding="utf-8")
    text = clean_text(raw)
    # Should contain primary doc content
    assert "primary" in text
    assert "Risk Factors" in text
    # Should NOT contain HTML tags
    assert "<p>" not in text and "<html>" not in text
    # Should be a single string (joined bodies)
    assert isinstance(text, str)


def test_process_filing_writes_output_for_sgml(tmp_path):
    src = tmp_path / "input.txt"
    src.write_text((FIXTURES / "sample_sgml.txt").read_text(encoding="utf-8"))
    dst = tmp_path / "output.txt"
    wrote = process_filing(src, dst)
    assert wrote is True
    assert dst.exists()
    content = dst.read_text(encoding="utf-8")
    assert len(content) >= MIN_TEXT_LENGTH
    assert "primary" in content


def test_process_filing_skips_when_output_exists(tmp_path):
    src = tmp_path / "input.txt"
    src.write_text((FIXTURES / "sample_sgml.txt").read_text(encoding="utf-8"))
    dst = tmp_path / "output.txt"
    dst.write_text("pre-existing non-empty content")  # simulate prior run
    wrote = process_filing(src, dst)
    assert wrote is False  # skipped
    # Existing content preserved
    assert dst.read_text(encoding="utf-8") == "pre-existing non-empty content"


def test_process_filing_drops_short_output(tmp_path):
    # Construct an SGML with a tiny body (below MIN_TEXT_LENGTH)
    src = tmp_path / "tiny.txt"
    src.write_text("<TEXT>short</TEXT>")
    dst = tmp_path / "out.txt"
    wrote = process_filing(src, dst)
    assert wrote is False  # below threshold, not written
    assert not dst.exists()
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `pytest tests/data/test_clean_filings.py -v`
Expected: ImportError or ModuleNotFoundError on `src.data.clean_filings`.

### Task 1.3: Implement `clean_filings.py`

- [ ] **Step 1: Create `src/data/clean_filings.py`**

```python
"""Extract clean text from raw EDGAR SGML envelopes.

Phase 1 of the FinBERT FT pipeline. Reads files from
data/raw/edgar/{cik}/{accession}.{txt,htm}, extracts <TEXT>...</TEXT> bodies,
strips HTML tags, collapses whitespace, and writes plain text to
data/interim/edgar_text/{cik}/{accession}.txt.

Parallelized via multiprocessing for speed (~6 hr on 16 cores for 226K filings).
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
```

- [ ] **Step 2: Run tests, confirm they pass**

Run: `pytest tests/data/test_clean_filings.py -v`
Expected: 9 passed.

If `test_process_filing_drops_short_output` fails: verify `MIN_TEXT_LENGTH = 500` constant exists at module level and `process_filing` checks `len(text) < MIN_TEXT_LENGTH`.

If `test_extract_sgml_bodies_handles_attributes` fails: verify the regex pattern uses `<TEXT[^>]*>` (not bare `<TEXT>`).

### Task 1.4: Smoke test on one real filing

- [ ] **Step 1: Run a focused single-file smoke**

```bash
python -c "
from pathlib import Path
from src.data.clean_filings import process_filing
import sys, os

# Find one real filing
raw_root = Path('data/raw/edgar')
sample = next(raw_root.rglob('*.txt'), None)
if sample is None:
    print('No raw filings found; skipping smoke')
    sys.exit(0)

# Output to /tmp so we don't touch interim/
tmp_out = Path('/tmp/smoke_clean_filings.txt')
if tmp_out.exists():
    tmp_out.unlink()

wrote = process_filing(sample, tmp_out)
print(f'input: {sample}')
print(f'wrote: {wrote}, size: {tmp_out.stat().st_size if tmp_out.exists() else 0} bytes')
print(f'first 300 chars:')
print(tmp_out.read_text()[:300] if tmp_out.exists() else '(no output)')
"
```

Expected:
- `wrote: True`
- Size > 500 bytes
- First 300 chars: readable financial prose, no `<` characters

**DO NOT run `python -m src.data.clean_filings`** — that's the user's job tomorrow (it takes ~6 hours).

### Task 1.5: Commit Task 1

- [ ] **Step 1: Commit**

```bash
git add src/data/clean_filings.py tests/data/test_clean_filings.py
git commit -m "data: add clean_filings.py multiprocessing SGML extractor"
```

---

## Task 2: FinBERT FT notebook (built via nbformat script)

The notebook is constructed by a Python script that uses `nbformat` to build the .ipynb. This keeps cell content in source control as a normal `.py` file and avoids hand-editing the brittle .ipynb JSON.

**Files:**
- Create: `scripts/build_finbert_notebook.py` (~500 LOC)
- Create: `notebooks/01_finbert_finetune.ipynb` (generated)

### Task 2.1: Create the build script

- [ ] **Step 1: Create `scripts/` directory if missing**

```bash
mkdir -p scripts
```

- [ ] **Step 2: Create `scripts/build_finbert_notebook.py`**

This script defines all notebook cells as Python strings and emits a valid `.ipynb` using `nbformat`. Run it whenever the notebook content needs to change — re-running regenerates the .ipynb.

```python
"""Build notebooks/01_finbert_finetune.ipynb from cell definitions.

Run via:
    python scripts/build_finbert_notebook.py

The notebook is the source of truth for FinBERT MLM fine-tune execution. This
build script is the source of truth for the notebook's cell contents — keeping
cell strings under source control avoids hand-editing brittle .ipynb JSON.
"""
from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


# =============================================================================
# Section A: Setup
# =============================================================================

A_INTRO_MD = """# FinBERT MLM Continued Pretraining

End-to-end FinBERT MLM fine-tune on the EDGAR corpus.

**Phases handled by this notebook:**
- Phase 2: Tokenize cleaned text → HF Datasets Arrow, 95/5 random split
- Phase 3 dry-run: 50-step Trainer sanity check
- Phase 3 full: 3 epochs FT (~36–48 hrs on RTX 5090)
- Phase 4: Held-out eval + persistence

**Phase 1 (text extraction) runs separately via the CLI:**
```bash
python -m src.data.clean_filings
```
Run that FIRST and wait for completion (~6 hr) before starting this notebook.

**Cursor IDE notes:**
- Output is bounded by the `CursorSafeProgressCallback` (Section E) — uses `clear_output(wait=True)` to replace, not append.
- Full progress is mirrored to `logs/finbert_finetune_<timestamp>.log`.
- TensorBoard sidecar (Section E) shows the live metrics view.

**Spec:** `docs/superpowers/specs/2026-05-11-finbert-finetune-design.md`
"""

A_IMPORTS = """import os
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from datasets import Dataset, DatasetDict
from IPython.display import HTML, clear_output, display
import matplotlib.pyplot as plt

from src.utils.io import (
    interim_dir,
    processed_dir,
    repo_root,
)
from src.utils.logging_utils import configure_logging

# Silence noisy warnings — keep notebook output focused
warnings.filterwarnings(\"ignore\", category=FutureWarning)
"""

A_GPU_CHECK = """# Confirm GPU + bf16 support on Blackwell sm_120
assert torch.cuda.is_available(), \"CUDA not available — FinBERT FT requires a CUDA GPU.\"
device_name = torch.cuda.get_device_name(0)
bf16_ok = torch.cuda.is_bf16_supported()
print(f\"CUDA device: {device_name}\")
print(f\"bf16 supported: {bf16_ok}\")
print(f\"PyTorch version: {torch.__version__}\")
print(f\"CUDA version: {torch.version.cuda}\")
assert bf16_ok, \"bf16 not supported on this GPU — Blackwell required for the config in this notebook.\"
"""

A_PATHS = """# Project paths
INTERIM_TEXT_DIR = interim_dir() / \"edgar_text\"
TOKENIZED_DIR = processed_dir() / \"finbert_tok\"
ARTIFACTS_DIR = repo_root() / \"artifacts\" / \"finbert-mlm\"
LOGS_DIR = repo_root() / \"logs\"
METRICS_DIR = repo_root() / \"reports\" / \"metrics\"

for d in (TOKENIZED_DIR, ARTIFACTS_DIR, LOGS_DIR, METRICS_DIR):
    d.mkdir(parents=True, exist_ok=True)

print(f\"Cleaned text source: {INTERIM_TEXT_DIR}\")
print(f\"Tokenized dataset (will create): {TOKENIZED_DIR}\")
print(f\"Artifacts (will create): {ARTIFACTS_DIR}\")
"""

# =============================================================================
# Section B: Verify Phase 1 output
# =============================================================================

B_INTRO_MD = """## Section B — Verify Phase 1 (text extraction) is complete

Phase 1 must run before this notebook. It reads raw SGML from `data/raw/edgar/` and writes clean text to `data/interim/edgar_text/{cik}/{accession}.txt`.

If the assertions below fail, run:
```bash
python -m src.data.clean_filings
```
Wait for it to finish (~6 hr on 14-16 workers), then re-run these cells.
"""

B_VERIFY = """assert INTERIM_TEXT_DIR.exists(), (
    f\"{INTERIM_TEXT_DIR} not found. Run `python -m src.data.clean_filings` first.\"
)

cleaned_files = list(INTERIM_TEXT_DIR.rglob(\"*.txt\"))
print(f\"Cleaned filings found: {len(cleaned_files):,}\")

assert len(cleaned_files) >= 200_000, (
    f\"Only {len(cleaned_files):,} cleaned files; expected >= 200K. \"
    \"Phase 1 may not have completed — check `python -m src.data.clean_filings` log.\"
)

# Sanity peek
import random
sample = random.choice(cleaned_files)
sample_text = sample.read_text(encoding=\"utf-8\")
print(f\"\\nSample: {sample.name} ({len(sample_text):,} chars)\")
print(\"First 500 chars:\")
print(sample_text[:500])
"""

# =============================================================================
# Section C: Tokenize + chunk + split
# =============================================================================

C_INTRO_MD = """## Section C — Phase 2: Tokenize + chunk + 95/5 split

Tokenize all cleaned text files, chunk into 512-token windows with 64-token stride, and create a 95/5 train/val split saved to `data/processed/finbert_tok/`.

This cell takes 30-60 min. Run it once; the saved Arrow dataset is reusable across training runs.
"""

C_TOKENIZER = """MODEL_NAME = \"yiyanghkust/finbert-pretrain\"
FALLBACK_MODEL_NAME = \"bert-base-uncased\"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model_name = MODEL_NAME
    print(f\"Loaded tokenizer from {MODEL_NAME}\")
except Exception as e:
    print(f\"Failed to load {MODEL_NAME}: {e}\")
    print(f\"Falling back to {FALLBACK_MODEL_NAME}\")
    tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL_NAME)
    base_model_name = FALLBACK_MODEL_NAME
"""

C_TOKENIZE_AND_SPLIT = """CHUNK_LEN = 512
STRIDE = 64
MIN_CHUNK_TOKENS = 128


def tokenize_file_to_chunks(path: Path):
    \"\"\"Yield {input_ids, attention_mask} dicts for chunks of one cleaned text file.\"\"\"
    text = path.read_text(encoding=\"utf-8\")
    if not text.strip():
        return
    ids = tokenizer(text, add_special_tokens=False, return_attention_mask=False)[\"input_ids\"]
    if len(ids) < MIN_CHUNK_TOKENS:
        return
    # Sliding-window chunks (always grab CLS+SEP slot at edges)
    cls = tokenizer.cls_token_id
    sep = tokenizer.sep_token_id
    body_len = CHUNK_LEN - 2  # reserve for CLS, SEP
    for start in range(0, len(ids), body_len - STRIDE):
        chunk = ids[start:start + body_len]
        if len(chunk) < MIN_CHUNK_TOKENS:
            break
        chunk = [cls] + chunk + [sep]
        # Pad to CHUNK_LEN
        attn = [1] * len(chunk)
        if len(chunk) < CHUNK_LEN:
            pad_len = CHUNK_LEN - len(chunk)
            chunk = chunk + [tokenizer.pad_token_id] * pad_len
            attn = attn + [0] * pad_len
        yield {\"input_ids\": chunk, \"attention_mask\": attn}


def generator():
    for p in cleaned_files:
        yield from tokenize_file_to_chunks(p)


print(\"Building HF Dataset by streaming through cleaned files...\")
ds = Dataset.from_generator(generator)
print(f\"Total tokenized chunks: {len(ds):,}\")

# 95/5 split
split = ds.train_test_split(test_size=0.05, seed=42)
dataset_dict = DatasetDict({\"train\": split[\"train\"], \"validation\": split[\"test\"]})
print(f\"Train: {len(dataset_dict['train']):,} chunks\")
print(f\"Val:   {len(dataset_dict['validation']):,} chunks\")
print(f\"Approx train tokens: {len(dataset_dict['train']) * CHUNK_LEN / 1e9:.2f}B\")

dataset_dict.save_to_disk(str(TOKENIZED_DIR))
print(f\"Saved to {TOKENIZED_DIR}\")
"""

C_LOAD_BACK = """# Re-load from disk to confirm save worked (also lets you resume from this point later)
from datasets import load_from_disk
dataset_dict = load_from_disk(str(TOKENIZED_DIR))
print(\"Loaded back from disk:\")
print(dataset_dict)
"""

# =============================================================================
# Section D: Dry run (50 steps)
# =============================================================================

D_INTRO_MD = \"\"\"## Section D — Phase 3 dry run (50 steps)

**Critical gate.** Before launching the 36-48 hr full training, verify:
1. Model loads
2. Training loop runs without OOM
3. Loss is finite and trends down

If OOM at batch 64: bump `gradient_accumulation_steps=2` and drop `per_device_train_batch_size=32` (effective batch unchanged).
\"\"\"

D_MODEL_LOAD = \"\"\"model = AutoModelForMaskedLM.from_pretrained(base_model_name)
print(f\"Loaded model from {base_model_name}\")
print(f\"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M\")

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
)
\"\"\"

D_DRY_RUN = \"\"\"DRY_RUN_DIR = ARTIFACTS_DIR / \"dry-run\"
DRY_RUN_DIR.mkdir(parents=True, exist_ok=True)

dry_args = TrainingArguments(
    output_dir=str(DRY_RUN_DIR),
    overwrite_output_dir=True,
    max_steps=50,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    gradient_accumulation_steps=1,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_ratio=0.0,
    lr_scheduler_type=\"cosine\",
    bf16=True,
    save_strategy=\"no\",
    eval_strategy=\"no\",
    logging_strategy=\"steps\",
    logging_steps=10,
    disable_tqdm=True,
    report_to=[\"none\"],
    dataloader_num_workers=2,
    seed=42,
)

dry_trainer = Trainer(
    model=model,
    args=dry_args,
    train_dataset=dataset_dict[\"train\"],
    eval_dataset=dataset_dict[\"validation\"].select(range(min(64, len(dataset_dict[\"validation\"])))),
    data_collator=data_collator,
)

print(\"Starting 50-step dry run...\")
dry_trainer.train()
print(\"\\nDry run complete.\")

# Inspect final loss
log_hist = dry_trainer.state.log_history
last_loss = next((e[\"loss\"] for e in reversed(log_hist) if \"loss\" in e), None)
print(f\"Final training loss: {last_loss}\")
assert last_loss is not None and np.isfinite(last_loss), \"Dry run loss is invalid; do NOT proceed to full training.\"
print(\"\\nGate passed. Proceed to Section E for full training.\")
\"\"\"

# =============================================================================
# Section E: Full training
# =============================================================================

E_INTRO_MD = \"\"\"## Section E — Phase 3 full training (3 epochs, ~36–48 hr on 5090)

**This is the long cell.** Launch once and monitor live progress via the rendered widget and the TensorBoard sidecar.

The `CursorSafeProgressCallback` keeps the cell output bounded — it uses `clear_output(wait=True)` to replace the displayed progress table on each log event, mirroring the full log to `logs/finbert_finetune_<timestamp>.log` so nothing is lost.
\"\"\"

E_CALLBACK = \"\"\"class CursorSafeProgressCallback(TrainerCallback):
    \\\"\\\"\\\"Bounded-output progress callback for Cursor IDE.

    Re-renders a single display widget on each log step (replace, not append).
    Mirrors the full log to a file under logs/ so nothing is lost.
    \\\"\\\"\\\"

    def __init__(self, log_file: Path, plot_every_n_logs: int = 5, recent_rows: int = 20):
        self.log_file = log_file
        self.plot_every_n_logs = plot_every_n_logs
        self.recent_rows = recent_rows
        self.history: list[dict] = []
        self.file_logger = logging.getLogger(f\\\"finbert_ft_{id(self)}\\\")
        self.file_logger.setLevel(logging.INFO)
        # Clear handlers from any prior runs of this cell
        self.file_logger.handlers = []
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter(\\\"%(asctime)s | %(message)s\\\"))
        self.file_logger.addHandler(fh)

    def _render(self):
        clear_output(wait=True)
        df = pd.DataFrame(self.history)
        recent = df.tail(self.recent_rows)
        display(HTML(
            f\\\"<b>FinBERT MLM FT</b> — {len(self.history)} log events; full log at {self.log_file}<br>\\\"
            + recent.to_html(index=False, float_format=lambda v: f\\\"{v:.4f}\\\" if isinstance(v, float) else str(v))
        ))
        if len(self.history) >= 2 and len(self.history) % self.plot_every_n_logs == 0:
            train = df[df[\\\"loss\\\"].notna()] if \\\"loss\\\" in df.columns else pd.DataFrame()
            val = df[df[\\\"eval_loss\\\"].notna()] if \\\"eval_loss\\\" in df.columns else pd.DataFrame()
            fig, ax = plt.subplots(figsize=(8, 4))
            if not train.empty:
                ax.plot(train[\\\"step\\\"], train[\\\"loss\\\"], label=\\\"train\\\", alpha=0.7)
            if not val.empty:
                ax.plot(val[\\\"step\\\"], val[\\\"eval_loss\\\"], label=\\\"val\\\", marker=\\\"o\\\", alpha=0.7)
            ax.set_xlabel(\\\"step\\\")
            ax.set_ylabel(\\\"loss\\\")
            ax.legend()
            ax.grid(True, alpha=0.3)
            display(fig)
            plt.close(fig)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        row = {\\\"step\\\": state.global_step, **logs}
        self.history.append(row)
        self.file_logger.info(\\\"step=%d %s\\\", state.global_step, logs)
        self._render()


print(\\\"Callback class defined.\\\")
\"\"\"

E_TENSORBOARD = \"\"\"# Optional: TensorBoard sidecar for streaming metrics (separate UI in Cursor)
# Comment out if you prefer to keep the notebook output as the sole monitor.
%load_ext tensorboard
%tensorboard --logdir {ARTIFACTS_DIR}/runs --port 6006
\"\"\"

E_FULL_TRAIN = \"\"\"timestamp = datetime.now().strftime(\\\"%Y%m%d_%H%M%S\\\")
log_file = LOGS_DIR / f\\\"finbert_finetune_{timestamp}.log\\\"

# Reload the model from base so the dry-run weights don't carry over
model = AutoModelForMaskedLM.from_pretrained(base_model_name)

full_args = TrainingArguments(
    output_dir=str(ARTIFACTS_DIR),
    overwrite_output_dir=False,
    num_train_epochs=3,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    gradient_accumulation_steps=1,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_ratio=0.06,
    lr_scheduler_type=\\\"cosine\\\",
    bf16=True,
    save_strategy=\\\"steps\\\",
    save_steps=5000,
    save_total_limit=3,
    eval_strategy=\\\"steps\\\",
    eval_steps=5000,
    logging_strategy=\\\"steps\\\",
    logging_steps=100,
    disable_tqdm=True,
    report_to=[\\\"tensorboard\\\"],
    metric_for_best_model=\\\"eval_loss\\\",
    load_best_model_at_end=True,
    dataloader_num_workers=4,
    seed=42,
)

trainer = Trainer(
    model=model,
    args=full_args,
    train_dataset=dataset_dict[\\\"train\\\"],
    eval_dataset=dataset_dict[\\\"validation\\\"],
    data_collator=data_collator,
    callbacks=[CursorSafeProgressCallback(log_file=log_file)],
)

print(f\\\"Starting full FinBERT MLM FT. Log file: {log_file}\\\")
print(f\\\"Expected runtime on RTX 5090: ~36-48 hr for 3 epochs.\\\")
print(f\\\"If the kernel is interrupted, resume with: trainer.train(resume_from_checkpoint=True)\\\")

trainer.train()
print(\\\"\\\\nTraining complete.\\\")
\"\"\"

# =============================================================================
# Section F: Eval + save
# =============================================================================

F_INTRO_MD = \"\"\"## Section F — Phase 4: Eval + save artifacts

Final held-out perplexity, persist the model + tokenizer, and dump loss curves to a parquet for paper-ready plotting later.
\"\"\"

F_FINAL_EVAL = \"\"\"# Final held-out perplexity
eval_metrics = trainer.evaluate(eval_dataset=dataset_dict[\\\"validation\\\"])
val_loss = eval_metrics[\\\"eval_loss\\\"]
val_ppl = float(np.exp(val_loss))
print(f\\\"Final val loss: {val_loss:.4f}\\\")
print(f\\\"Final val perplexity: {val_ppl:.2f}\\\")
\"\"\"

F_SAVE_METRICS = \"\"\"# Save train/val curves to parquet (spec §17.1)
log_hist = pd.DataFrame(trainer.state.log_history)
metrics_path = METRICS_DIR / \\\"finbert_finetune.parquet\\\"
log_hist.to_parquet(metrics_path, index=False)
print(f\\\"Saved {len(log_hist)} log events to {metrics_path}\\\")

# Save a small summary alongside
summary = {
    \\\"base_model\\\": base_model_name,
    \\\"final_train_loss\\\": next((e[\\\"loss\\\"] for e in reversed(trainer.state.log_history) if \\\"loss\\\" in e), None),
    \\\"final_val_loss\\\": val_loss,
    \\\"final_val_perplexity\\\": val_ppl,
    \\\"total_steps\\\": trainer.state.global_step,
    \\\"total_epochs\\\": trainer.state.epoch,
    \\\"timestamp_finished\\\": datetime.now().isoformat(),
}
(METRICS_DIR / \\\"finbert_finetune_summary.json\\\").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
\"\"\"

F_PLOT = \"\"\"# Final loss plot (separate from the live callback plot)
df = log_hist.copy()
fig, ax = plt.subplots(figsize=(10, 5))
train = df[df[\\\"loss\\\"].notna()] if \\\"loss\\\" in df.columns else pd.DataFrame()
val = df[df[\\\"eval_loss\\\"].notna()] if \\\"eval_loss\\\" in df.columns else pd.DataFrame()
if not train.empty:
    ax.plot(train[\\\"step\\\"], train[\\\"loss\\\"], label=\\\"train\\\", alpha=0.7)
if not val.empty:
    ax.plot(val[\\\"step\\\"], val[\\\"eval_loss\\\"], label=\\\"val\\\", marker=\\\"o\\\")
ax.set_xlabel(\\\"step\\\")
ax.set_ylabel(\\\"loss\\\")
ax.set_title(\\\"FinBERT MLM Fine-Tune — Loss\\\")
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
\"\"\"

F_PERSIST = \"\"\"# Save the encoder + tokenizer
trainer.save_model(str(ARTIFACTS_DIR))
tokenizer.save_pretrained(str(ARTIFACTS_DIR))
print(f\\\"\\\\nEncoder + tokenizer saved to {ARTIFACTS_DIR}\\\")

# Sanity check: list saved files
for p in sorted(ARTIFACTS_DIR.iterdir()):
    if p.is_file():
        print(f\\\"  {p.name} ({p.stat().st_size / 1e6:.1f} MB)\\\")
\"\"\"


# =============================================================================
# Build the notebook
# =============================================================================

def build_notebook() -> nbformat.NotebookNode:
    nb = new_notebook()
    cells = [
        # Section A
        new_markdown_cell(A_INTRO_MD),
        new_markdown_cell(\"## Section A — Setup\"),
        new_code_cell(A_IMPORTS),
        new_code_cell(A_GPU_CHECK),
        new_code_cell(A_PATHS),
        # Section B
        new_markdown_cell(B_INTRO_MD),
        new_code_cell(B_VERIFY),
        # Section C
        new_markdown_cell(C_INTRO_MD),
        new_code_cell(C_TOKENIZER),
        new_code_cell(C_TOKENIZE_AND_SPLIT),
        new_code_cell(C_LOAD_BACK),
        # Section D
        new_markdown_cell(D_INTRO_MD),
        new_code_cell(D_MODEL_LOAD),
        new_code_cell(D_DRY_RUN),
        # Section E
        new_markdown_cell(E_INTRO_MD),
        new_code_cell(E_CALLBACK),
        new_code_cell(E_TENSORBOARD),
        new_code_cell(E_FULL_TRAIN),
        # Section F
        new_markdown_cell(F_INTRO_MD),
        new_code_cell(F_FINAL_EVAL),
        new_code_cell(F_SAVE_METRICS),
        new_code_cell(F_PLOT),
        new_code_cell(F_PERSIST),
    ]
    nb.cells = cells
    nb.metadata = {
        \"kernelspec\": {
            \"display_name\": \"Python 3\",
            \"language\": \"python\",
            \"name\": \"python3\",
        },
        \"language_info\": {\"name\": \"python\", \"version\": \"3.11\"},
    }
    return nb


def main() -> None:
    out_path = Path(__file__).resolve().parents[1] / \"notebooks\" / \"01_finbert_finetune.ipynb\"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nb = build_notebook()
    with out_path.open(\"w\", encoding=\"utf-8\") as f:
        nbformat.write(nb, f)
    print(f\"Wrote {out_path}\")
    # Round-trip read to catch JSON / nbformat issues
    with out_path.open(encoding=\"utf-8\") as f:
        loaded = nbformat.read(f, as_version=4)
    print(f\"Round-trip OK. Cells: {len(loaded.cells)}\")


if __name__ == \"__main__\":
    main()
```

NOTE TO IMPLEMENTER: this build script uses heavy string escaping (triple-quoted strings inside triple-quoted blocks). When transcribing into the file, be careful that:
- The outer cell-content strings use plain `\"\"\"...\"\"\"` triple quotes
- Anywhere the cell content itself needs triple quotes (e.g., callback docstrings), use single-quoted `\\\"\\\"\\\"` triple quotes escaped via backslashes
- The notebook's markdown cells use plain triple-quoted strings (no escaping needed)

If pasting introduces subtle quoting errors, run `python -c "import ast; ast.parse(open('scripts/build_finbert_notebook.py').read())"` first to surface syntax errors before trying to execute.

- [ ] **Step 3: Run the build script**

Run: `python scripts/build_finbert_notebook.py`
Expected output:
```
Wrote /home/kavin-ravi/CodingStuff/axiom_tilt/notebooks/01_finbert_finetune.ipynb
Round-trip OK. Cells: 24
```

The round-trip read confirms the notebook is valid nbformat JSON.

- [ ] **Step 4: Verify the notebook can be opened by Jupyter without errors**

Run: `jupyter nbconvert --to notebook --stdout notebooks/01_finbert_finetune.ipynb > /dev/null && echo "VALID"`
Expected: prints `VALID`. (This validates the structure without executing any cells.)

If invalid: the most likely cause is a string-escaping issue in `build_finbert_notebook.py`. Inspect with `python -c "import nbformat; nb = nbformat.read('notebooks/01_finbert_finetune.ipynb', as_version=4); [print(i, c['cell_type'], len(c['source'])) for i, c in enumerate(nb.cells)]"` to see cell structure.

### Task 2.2: Commit notebook artifacts

- [ ] **Step 1: Commit the build script and generated notebook**

```bash
git add scripts/build_finbert_notebook.py notebooks/01_finbert_finetune.ipynb
git commit -m "notebook: add finbert MLM fine-tune notebook and build script"
```

---

## Task 3: Update TODO.md with handoff info

**Files:**
- Modify: `TODO.md`

### Task 3.1: Append a FinBERT-FT-handoff section to TODO.md

- [ ] **Step 1: Append to `TODO.md` (after the existing "Action items" section)**

```markdown

## Tomorrow's execution order (FinBERT FT)

You'll have a Cursor IDE agent execute these in sequence while you're at class:

### Step 1: Run Phase 1 (text extraction) — CLI, ~6 hr
```bash
python -m src.data.clean_filings
```
- Multiprocessing-based; uses `cpu_count - 2` workers by default.
- Output: `data/interim/edgar_text/{cik}/{accession}.txt` (~30-50 GB total)
- Per-file resume; safe to interrupt and re-run.

### Step 2: Open the notebook in Cursor and execute Sections A–D
- `notebooks/01_finbert_finetune.ipynb`
- Section A: setup, GPU check (expects RTX 5090 + bf16)
- Section B: verify ≥200K cleaned files from Step 1
- Section C: tokenize (~30-60 min) → `data/processed/finbert_tok/`
- Section D: 50-step dry run (~2 min). **GATE: confirm loss is finite and trending down BEFORE proceeding to Section E.** If OOM: bump grad-accum to 2, drop batch to 32.

### Step 3: Run Section E (full training, ~36-48 hr)
- One long cell.
- TensorBoard sidecar at port 6006 streams live metrics.
- The `CursorSafeProgressCallback` mirrors logs to `logs/finbert_finetune_<timestamp>.log` and keeps notebook output bounded.

### Step 4: Run Section F (eval + save)
- Saves encoder + tokenizer to `artifacts/finbert-mlm/`
- Saves train/val curves to `reports/metrics/finbert_finetune.parquet`
- Saves summary JSON to `reports/metrics/finbert_finetune_summary.json`

### Recovery
- Training crash: `trainer.train(resume_from_checkpoint=True)` resumes from latest checkpoint
- OOM at full training: drop `per_device_train_batch_size=32` + `gradient_accumulation_steps=2`
- HF Hub unreachable: notebook auto-falls back to `bert-base-uncased`

### Validation gates (Section F outputs to look for)
- `artifacts/finbert-mlm/config.json` exists
- `artifacts/finbert-mlm/model.safetensors` (or `pytorch_model.bin`) exists
- `reports/metrics/finbert_finetune.parquet` exists and loads
- Val perplexity printed in Section F is finite and lower than initial random-init
```

- [ ] **Step 2: Commit TODO update**

```bash
git add TODO.md
git commit -m "docs: add finbert FT execution handoff steps to TODO"
```

---

## Task 4: Final push to origin

### Task 4.1: Push the branch

- [ ] **Step 1: Push**

Run: `git push origin data-ingestion 2>&1 | tail -3`
Expected: clean push, no large-file errors.

---

## Self-review

**1. Spec coverage check (vs `2026-05-11-finbert-finetune-design.md`):**
- §3 architecture (Phases 1-4) → covered by Task 1 (Phase 1 = `clean_filings.py`) and Task 2 (Phases 2-4 = notebook).
- §4.1 `clean_filings.py` requirements → Task 1.3 implementation matches: multiprocessing, regex extraction, resume support, MIN_TEXT_LENGTH=500, argparse `--workers`.
- §4.2 notebook sections A-F → all six sections in `scripts/build_finbert_notebook.py` cell strings.
- §5.1 `disable_tqdm=True` → present in both `D_DRY_RUN` and `E_FULL_TRAIN` Trainer configs.
- §5.2 `CursorSafeProgressCallback` → `E_CALLBACK` cell implements it.
- §5.3 TensorBoard sidecar → `E_TENSORBOARD` cell.
- §5.4 `save_total_limit=3` → `E_FULL_TRAIN` config.
- §6 Trainer configuration → mirrored exactly in `E_FULL_TRAIN`.
- §7.1 text extraction algorithm → matches `clean_filings.py` (regex, html.unescape, whitespace collapse, MIN_TEXT_LENGTH skip).
- §7.2 tokenization + chunking → `C_TOKENIZE_AND_SPLIT` cell with CHUNK_LEN=512, STRIDE=64, MIN_CHUNK_TOKENS=128.
- §8 expected outputs → all paths used in cells match.
- §9 timing budget → documented in TODO update + B_INTRO_MD.
- §10 failure modes → notebook handles HF Hub fallback (C_TOKENIZER), grad-accum guidance (D_INTRO_MD), resume via `resume_from_checkpoint` (E_FULL_TRAIN message).
- §12 validation gates → TODO update step 4 enumerates them.

No gaps.

**2. Placeholder scan:**
- No "TBD", "TODO", "implement later" in tasks.
- All code blocks are complete.
- The single non-code "NOTE TO IMPLEMENTER" in Task 2.1 step 2 is a debugging hint, not a placeholder.

**3. Type / name consistency:**
- `MIN_TEXT_LENGTH` used identically in module and test.
- `extract_sgml_bodies`, `strip_html_tags`, `clean_text`, `process_filing` — same names across module, tests, and any callers.
- `CursorSafeProgressCallback` referenced consistently between E_CALLBACK definition and E_FULL_TRAIN instantiation.
- `dataset_dict`, `tokenizer`, `base_model_name`, `data_collator`, `model`, `trainer` — variables defined in earlier cells (C, D) and reused in later cells (E, F). Notebook cell execution order matters; documented in section intros.
- `ARTIFACTS_DIR`, `TOKENIZED_DIR`, `LOGS_DIR`, `METRICS_DIR`, `INTERIM_TEXT_DIR` — defined in A_PATHS, used everywhere.

No type/name drift.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-11-finbert-finetune.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?

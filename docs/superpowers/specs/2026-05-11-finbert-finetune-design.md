# FinBERT MLM Fine-Tune — Design Spec

**Date:** 2026-05-11
**Status:** Draft, pending user review
**Repo:** `axiom_tilt`
**Related spec:** `docs/superpowers/specs/2026-05-08-text-enhanced-rl-portfolio-design.md` §4
**Deadline context:** Wednesday partner meeting (3 days out); project due Friday

## 1. Goal

Continued masked-language-model (MLM) pretraining of `yiyanghkust/finbert-pretrain` on the project's full EDGAR corpus (226,919 filings, ~1B+ tokens). Output is a domain-adapted encoder whose `[CLS]` representation captures financial-document semantics for downstream use in the supervised ranker (Architecture A from §2 of the parent spec).

No supervised head. No LoRA / PEFT — full fine-tune (FinBERT is ~110M params, fits comfortably on a 5090 with bf16).

## 2. Leakage / bias position

**MLM is unsupervised — no label leakage.** Including a company's filings in the FT corpus and later embedding the same company's filings is NOT a methodological violation. The encoder learns language patterns, not company-level return targets.

**There IS a mild *temporal representational shift*** when FT'd on the full 2000–2025 corpus: the encoder at the 2009 walk boundary has seen 2020 language patterns (COVID lexicon, etc.). This is acknowledged as a known limitation in the writeup; a planned v2 robustness check runs the pipeline with the off-the-shelf `ProsusAI/finbert` encoder (no FT) as an ablation.

Strict walk-forward FT (re-train at each walk boundary on only pre-boundary data) is the gold standard but is out of scope for the Friday deliverable. Standard practice in academic NLP-finance.

## 3. Architecture / pipeline

```
Phase 1: text extraction          Phase 2: tokenize + split (in notebook)
┌────────────────────────────┐    ┌──────────────────────────────────┐
│ raw SGML (746 GB, 226K     │    │ tokenize via AutoTokenizer       │
│ filings under              │───▶│ chunk to 512 with 64 stride      │
│ data/raw/edgar/{cik}/...) │    │ 95/5 random split (per filing)   │
│ → multiprocessing regex    │    │ HF Datasets save_to_disk         │
│ strip → plain text         │    │ → data/processed/finbert_tok/    │
│ → data/interim/edgar_text/ │    │   (arrow, memory-mapped)         │
│ {cik}/{accession}.txt      │    │                                  │
│ ~6 hr on 16-core CPU       │    │ ~30–60 min                       │
└────────────────────────────┘    └──────────────────────────────────┘
                                                │
                                                ▼
                            Phase 3: MLM fine-tune (Trainer + bf16)
                            ┌────────────────────────────────────────┐
                            │ AutoModelForMaskedLM.from_pretrained() │
                            │ batch 64 × 512, lr 5e-5, AdamW         │
                            │ cosine warmup 6%, 3 epochs             │
                            │ bf16, no LoRA                          │
                            │ checkpoints every 5K steps             │
                            │ ~36–48 hr on RTX 5090                  │
                            └────────────────────────────────────────┘
                                                │
                                                ▼
                            Phase 4: eval + persistence
                            ┌────────────────────────────────────────┐
                            │ held-out perplexity                    │
                            │ train/val loss curves → parquet        │
                            │ encoder → artifacts/finbert-mlm/       │
                            └────────────────────────────────────────┘
```

## 4. Deliverables (two files)

### 4.1 `src/data/clean_filings.py` — Phase 1 (data prep, CLI)

Multiprocessing regex-based extractor. Reads raw SGML envelopes under `data/raw/edgar/{cik}/{accession}.{txt,htm}`, extracts `<TEXT>...</TEXT>` body blocks, strips HTML tags with regex, collapses whitespace, writes one plain-text file per filing to `data/interim/edgar_text/{cik}/{accession}.txt`.

- `multiprocessing.Pool(processes=os.cpu_count() - 2)` for parallelism
- Per-file resume: skip already-extracted output files
- Progress via `tqdm` on the iterator
- `argparse` flag `--workers N` to override default pool size
- Run via `python -m src.data.clean_filings`
- Why regex not BeautifulSoup: BeautifulSoup on a 30 MB SGML envelope is ~20–60 seconds (DOM build is the bottleneck). Regex `<TEXT[^>]*>(.*?)</TEXT>` + `re.sub(r"<[^>]+>", " ", body)` is <2 seconds per filing. For 226K filings the difference is days vs hours.
- Drop filings smaller than 500 chars after extraction (mostly index pages, exhibit metadata)

### 4.2 `notebooks/01_finbert_finetune.ipynb` — Phases 2-4 (model code, interactive)

Self-contained Jupyter notebook. Open in Cursor IDE, run cell by cell. Sections:

#### Section A — Setup (~5 cells)
- Imports (`torch`, `transformers`, `datasets`, `numpy`, `pandas`, etc.)
- GPU check (`torch.cuda.is_available()`, prints device name → expects "RTX 5090")
- bf16 support check (`torch.cuda.is_bf16_supported()`)
- Project paths via `src.utils.io`
- Markdown cell explaining: "Run `python -m src.data.clean_filings` BEFORE this notebook if `data/interim/edgar_text/` is empty"

#### Section B — Phase 1 verification (~2 cells)
- `assert Path("data/interim/edgar_text").exists()`
- Count cleaned files: `len(list(Path("data/interim/edgar_text").rglob("*.txt")))` — fail if < 200K (signals Phase 1 wasn't run / incomplete)
- Sample a random cleaned file, print first 500 chars to verify it looks like financial prose

#### Section C — Phase 2: tokenize + chunk + split (~5 cells)
- Load tokenizer: `AutoTokenizer.from_pretrained("yiyanghkust/finbert-pretrain")` with fallback to `bert-base-uncased` if HF Hub is unreachable
- Build dataset by streaming through cleaned files, chunking to 512 tokens with 64 stride
- Use `datasets.Dataset.from_generator()` or similar; chunk on the fly to avoid loading all text in memory
- Random 95/5 split via `dataset.train_test_split(test_size=0.05, seed=42)`
- Save to `data/processed/finbert_tok/` via `save_to_disk()` (Arrow format, memory-mapped)
- Print summary: train/val token counts, expected step count

#### Section D — Phase 3: dry run (~3 cells)

**Critical** — verify training loop works on 50 steps before launching the 36-hr full run.

- `AutoModelForMaskedLM.from_pretrained(model_name)`
- `DataCollatorForLanguageModeling(tokenizer, mlm_probability=0.15)`
- `TrainingArguments(... max_steps=50, bf16=True, ...)` — same as full config but bounded
- `trainer.train()` — confirm runs without OOM, loss is finite, gradients flow
- If OOM: bump `gradient_accumulation_steps` to 2 (effective batch 32 per step, 64 cumulative)
- Markdown cell: gate to confirm dry run passed before proceeding

#### Section E — Phase 3: full training (~3 cells)

**One long-running cell.** ~36–48 hours on 5090.

- Re-create `TrainingArguments` with `num_train_epochs=3` and full step budget
- Custom `TrainerCallback` instance (see §5) wired in
- TensorBoard sidecar cell: `%load_ext tensorboard` then `%tensorboard --logdir artifacts/finbert-mlm/runs --port 6006`
- `trainer.train()`

#### Section F — Phase 4: eval + save (~5 cells)
- `trainer.evaluate()` for final held-out perplexity
- Convert `trainer.state.log_history` → DataFrame → save to `reports/metrics/finbert_finetune.parquet` (spec §17.1)
- Plot final loss curve via matplotlib
- `trainer.save_model("artifacts/finbert-mlm/")` (saves encoder + tokenizer)
- Print final summary: train loss, val loss, val perplexity, total steps, total runtime

## 5. Cursor IDE OOM mitigation (load-bearing)

Long-running HF Trainer cells crash Jupyter clients via two mechanisms: (a) tqdm bars buffering carriage-return lines, (b) per-step log rows accumulating in output. Fixes:

### 5.1 `disable_tqdm=True` in TrainingArguments
Kills the tqdm output stream entirely. Progress is reported through the custom callback below.

### 5.2 Custom `TrainerCallback` with `clear_output`

In Section E, define inline:

```python
from transformers import TrainerCallback
from IPython.display import clear_output, display, HTML
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import logging
from pathlib import Path
from datetime import datetime


class CursorSafeProgressCallback(TrainerCallback):
    """Bounded-output progress callback for Cursor IDE.

    Re-renders a single display widget on each log step (no append). Mirrors
    full log to a file under logs/ so nothing is lost.
    """

    def __init__(self, log_file: Path, plot_every_n_logs: int = 5):
        self.log_file = log_file
        self.plot_every_n_logs = plot_every_n_logs
        self.history: list[dict] = []
        self.file_logger = logging.getLogger("finbert_ft")
        self.file_logger.setLevel(logging.INFO)
        if not self.file_logger.handlers:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
            self.file_logger.addHandler(fh)

    def _render(self):
        clear_output(wait=True)
        df = pd.DataFrame(self.history)
        # Tail to keep display small
        recent = df.tail(20)
        display(HTML(
            f"<b>FinBERT MLM FT</b> — {len(self.history)} log events; full log at {self.log_file}<br>"
            + recent.to_html(index=False)
        ))
        # Inline matplotlib loss plot, re-rendered (not appended)
        if len(self.history) >= 2 and len(self.history) % self.plot_every_n_logs == 0:
            train = df[df["loss"].notna()] if "loss" in df.columns else pd.DataFrame()
            val = df[df["eval_loss"].notna()] if "eval_loss" in df.columns else pd.DataFrame()
            fig, ax = plt.subplots(figsize=(8, 4))
            if not train.empty:
                ax.plot(train["step"], train["loss"], label="train", alpha=0.7)
            if not val.empty:
                ax.plot(val["step"], val["eval_loss"], label="val", marker="o", alpha=0.7)
            ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.legend(); ax.grid(True, alpha=0.3)
            display(fig)
            plt.close(fig)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        row = {"step": state.global_step, **logs}
        self.history.append(row)
        self.file_logger.info("step=%d %s", state.global_step, logs)
        self._render()
```

Why this works:
- `clear_output(wait=True)` **replaces** the cell's output rather than appending. Output buffer stays bounded.
- All logs land in `logs/finbert_finetune_<timestamp>.log` for full history.
- Matplotlib figures are explicitly closed via `plt.close(fig)` to free memory.
- Only the last 20 rows are shown in the HTML widget; full DataFrame stays in `self.history` only.

### 5.3 TensorBoard sidecar
A separate cell loads tensorboard inline:
```python
%load_ext tensorboard
%tensorboard --logdir artifacts/finbert-mlm/runs --port 6006
```
Streams full training metrics in a separate widget. The notebook cells stay clean.

### 5.4 Checkpoint disk budget
`save_total_limit=3` in TrainingArguments → max 3 checkpoints on disk × ~440 MB each ≈ 1.3 GB. Plus the final model. Way under any reasonable disk budget.

## 6. Trainer configuration

```python
TrainingArguments(
    output_dir="artifacts/finbert-mlm/",
    overwrite_output_dir=False,
    num_train_epochs=3,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    gradient_accumulation_steps=1,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_ratio=0.06,
    lr_scheduler_type="cosine",
    bf16=True,
    save_strategy="steps",
    save_steps=5000,
    save_total_limit=3,
    eval_strategy="steps",
    eval_steps=5000,
    logging_strategy="steps",
    logging_steps=100,
    disable_tqdm=True,
    report_to=["tensorboard"],
    metric_for_best_model="eval_loss",
    load_best_model_at_end=True,
    dataloader_num_workers=4,
    seed=42,
)
```

If dry-run OOMs at batch 64:
- First try `gradient_accumulation_steps=2`, `per_device_train_batch_size=32`. Effective batch unchanged.
- If still OOM: drop to batch 16 with grad accum 4. Slower throughput but tractable.

## 7. Data prep details

### 7.1 Text extraction (Phase 1)

For each filing under `data/raw/edgar/{cik}/{accession}.{txt,htm}`:

1. Read raw bytes (encoding-tolerant: try `utf-8` with `errors="ignore"`, fall back to `latin-1`)
2. If file ends with `.txt` (SEC SGML envelope): extract all `<TEXT[^>]*>(.*?)</TEXT>` matches, concatenate
3. If file ends with `.htm`/`.html`: process as single body
4. Strip HTML tags: `re.sub(r"<[^>]+>", " ", body)`
5. Decode HTML entities (`&amp;` → `&`, etc.) via `html.unescape`
6. Collapse whitespace: `re.sub(r"\s+", " ", text).strip()`
7. Drop files <500 chars (mostly index pages)
8. Write to `data/interim/edgar_text/{cik}/{accession}.txt` (UTF-8)
9. Resume: if output file already exists with non-empty content, skip

### 7.2 Tokenization + chunking (Phase 2)

- Tokenizer: `AutoTokenizer.from_pretrained("yiyanghkust/finbert-pretrain")` (returns `BertTokenizerFast`)
- Chunk strategy: process one cleaned-text file at a time → tokenize without truncation → slide a window of 512 tokens with 64-token stride → yield each window as one training example
- Drop chunks <128 tokens (too short to be useful for MLM)
- Implementation: `datasets.Dataset.from_generator(generator_fn, features=...)` where generator yields `{"input_ids": [...], "attention_mask": [...]}` dicts
- After tokenization: 95/5 random train/val split via `dataset.train_test_split(test_size=0.05, seed=42)`
- Save: `dataset_dict.save_to_disk("data/processed/finbert_tok/")`

## 8. Expected outputs

- `data/interim/edgar_text/{cik}/{accession}.txt` — ~226K cleaned plain-text files (~30–50 GB total)
- `data/processed/finbert_tok/` — HF Datasets Arrow directory (~2 GB tokenized)
- `artifacts/finbert-mlm/` — fine-tuned encoder + tokenizer (~440 MB)
- `artifacts/finbert-mlm/runs/` — TensorBoard event files
- `reports/metrics/finbert_finetune.parquet` — train/val loss curves (per spec §17.1)
- `logs/finbert_finetune_<timestamp>.log` — file mirror of progress callback output

`data/interim/` and `artifacts/` are gitignored. `data/processed/finbert_tok/` is large (~2 GB) — gitignore that subdirectory.

## 9. Timing budget

| Phase | Estimated wall time | Where it runs |
|---|---|---|
| Phase 1: text extraction | 6 hr | CPU (multiprocessing) |
| Phase 2: tokenization + split | 30–60 min | CPU |
| Phase 3 dry run | 2 min | GPU |
| Phase 3 full training | 36–48 hr | GPU (RTX 5090) |
| Phase 4: eval + save | 30 min | GPU |
| **Total** | **~44–56 hr** | — |

Sunday night launch → finishes mid-Tuesday to Wednesday morning. Tight but feasible for Wednesday partner meeting. **If timing slips past Wednesday: 2-epoch fallback** — epoch 1 captures most domain adaptation; epoch 3 is polish.

## 10. Failure modes and mitigations

| Failure | Mitigation |
|---|---|
| Phase 1 multiprocessing crashes mid-run | Resume support: skips already-extracted files |
| `yiyanghkust/finbert-pretrain` unreachable on HF Hub | Fallback to `bert-base-uncased` (notebook handles with try/except) |
| Dry-run OOM at batch 64 | Drop to batch 32 with grad-accum 2 (config explicitly handles) |
| Trainer crashes during long run | Resume from latest checkpoint via `trainer.train(resume_from_checkpoint=True)` |
| Cursor IDE freezes from output buffer | Mitigated by §5 (disable_tqdm + clear_output callback) |
| Disk fills up with checkpoints | `save_total_limit=3` caps to ~1.3 GB |
| Loss diverges (NaN) | Trainer auto-detects; checkpoint reload + lower LR (1e-5) and resume |

## 11. What this spec does NOT cover

- Walk-forward FT (v2 robustness; spec §16)
- Off-the-shelf FinBERT ablation comparison (v2 robustness; run in parallel as a separate experiment when bandwidth permits)
- Embedding-generation pipeline (Phase 5 — separate notebook `02_finbert_embed.ipynb`, post-FT)
- Hyperparameter sweeps (out of scope; defaults are well-established for FinBERT-class FT)
- W&B integration (TensorBoard is sufficient for tonight; W&B is a v2 nice-to-have)

## 12. Validation gates

Before user can claim "FinBERT FT is done":
1. Phase 1 produced ≥200K cleaned text files
2. Phase 2 produced `data/processed/finbert_tok/` Arrow directory with both train/val splits
3. Dry-run cell completed without OOM and showed declining loss across 50 steps
4. Full training reached ≥1 complete epoch with val perplexity below initial random-init perplexity
5. `artifacts/finbert-mlm/` contains `config.json`, `pytorch_model.bin` or `model.safetensors`, tokenizer files
6. `reports/metrics/finbert_finetune.parquet` exists and is loadable

## 13. Defaults marked for user review

- Drop filings <500 chars after extraction (index pages, exhibit metadata)
- Drop tokenized chunks <128 tokens
- Tokenizer base: `yiyanghkust/finbert-pretrain` with `bert-base-uncased` fallback
- Batch size 64 × 512 (drops to 32+grad-accum-2 if OOM)
- 3 epochs (2-epoch fallback if timing slips)
- Logging every 100 steps, eval every 5000, checkpoint every 5000
- TensorBoard for live metrics (W&B is v2)

## 14. Open decisions

None blocking implementation.

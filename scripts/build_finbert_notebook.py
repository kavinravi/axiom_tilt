from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


# ---------------------------------------------------------------------------
# Section A — Setup
# ---------------------------------------------------------------------------

A_INTRO_MD = '''# FinBERT MLM Continued Pretraining

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
'''

A_SECTION_MD = '''## Section A — Setup'''

A_IMPORTS = '''import os
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
warnings.filterwarnings("ignore", category=FutureWarning)
'''

A_GPU_CHECK = '''# Confirm GPU + bf16 support on Blackwell sm_120
assert torch.cuda.is_available(), "CUDA not available — FinBERT FT requires a CUDA GPU."
device_name = torch.cuda.get_device_name(0)
bf16_ok = torch.cuda.is_bf16_supported()
print(f"CUDA device: {device_name}")
print(f"bf16 supported: {bf16_ok}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")
assert bf16_ok, "bf16 not supported on this GPU — Blackwell required for the config in this notebook."
'''

A_PATHS = '''# Project paths
INTERIM_TEXT_DIR = interim_dir() / "edgar_text"
TOKENIZED_DIR = processed_dir() / "finbert_tok"
ARTIFACTS_DIR = repo_root() / "artifacts" / "finbert-mlm"
LOGS_DIR = repo_root() / "logs"
METRICS_DIR = repo_root() / "reports" / "metrics"

for d in (TOKENIZED_DIR, ARTIFACTS_DIR, LOGS_DIR, METRICS_DIR):
    d.mkdir(parents=True, exist_ok=True)

print(f"Cleaned text source: {INTERIM_TEXT_DIR}")
print(f"Tokenized dataset (will create): {TOKENIZED_DIR}")
print(f"Artifacts (will create): {ARTIFACTS_DIR}")
'''


# ---------------------------------------------------------------------------
# Section B — Verify Phase 1
# ---------------------------------------------------------------------------

B_INTRO_MD = '''## Section B — Verify Phase 1 (text extraction) is complete

Phase 1 must run before this notebook. It reads raw SGML from `data/raw/edgar/` and writes clean text to `data/interim/edgar_text/{cik}/{accession}.txt`.

If the assertions below fail, run:
```bash
python -m src.data.clean_filings
```
Wait for it to finish (~6 hr on 14-16 workers), then re-run these cells.
'''

B_VERIFY = '''assert INTERIM_TEXT_DIR.exists(), (
    f"{INTERIM_TEXT_DIR} not found. Run `python -m src.data.clean_filings` first."
)

cleaned_files = list(INTERIM_TEXT_DIR.rglob("*.txt"))
print(f"Cleaned filings found: {len(cleaned_files):,}")

assert len(cleaned_files) >= 200_000, (
    f"Only {len(cleaned_files):,} cleaned files; expected >= 200K. "
    "Phase 1 may not have completed — check `python -m src.data.clean_filings` log."
)

# Sanity peek
import random
sample = random.choice(cleaned_files)
sample_text = sample.read_text(encoding="utf-8")
print(f"\\nSample: {sample.name} ({len(sample_text):,} chars)")
print("First 500 chars:")
print(sample_text[:500])
'''


# ---------------------------------------------------------------------------
# Section C — Tokenize + chunk + split
# ---------------------------------------------------------------------------

C_INTRO_MD = '''## Section C — Phase 2: Tokenize + chunk + 95/5 split

Tokenize all cleaned text files, chunk into 512-token windows with 64-token stride, and create a 95/5 train/val split saved to `data/processed/finbert_tok/`.

This cell takes 30-60 min. Run it once; the saved Arrow dataset is reusable across training runs.
'''

C_TOKENIZER = '''MODEL_NAME = "yiyanghkust/finbert-pretrain"
FALLBACK_MODEL_NAME = "bert-base-uncased"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model_name = MODEL_NAME
    print(f"Loaded tokenizer from {MODEL_NAME}")
except Exception as e:
    print(f"Failed to load {MODEL_NAME}: {e}")
    print(f"Falling back to {FALLBACK_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL_NAME)
    base_model_name = FALLBACK_MODEL_NAME
'''

C_TOKENIZE_AND_SPLIT = '''CHUNK_LEN = 512
STRIDE = 64
MIN_CHUNK_TOKENS = 128


def tokenize_file_to_chunks(path: Path):
    """Yield {input_ids, attention_mask} dicts for chunks of one cleaned text file."""
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return
    ids = tokenizer(text, add_special_tokens=False, return_attention_mask=False)["input_ids"]
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
        yield {"input_ids": chunk, "attention_mask": attn}


def generator():
    for p in cleaned_files:
        yield from tokenize_file_to_chunks(p)


print("Building HF Dataset by streaming through cleaned files...")
ds = Dataset.from_generator(generator)
print(f"Total tokenized chunks: {len(ds):,}")

# 95/5 split
split = ds.train_test_split(test_size=0.05, seed=42)
dataset_dict = DatasetDict({"train": split["train"], "validation": split["test"]})
print(f"Train: {len(dataset_dict['train']):,} chunks")
print(f"Val:   {len(dataset_dict['validation']):,} chunks")
print(f"Approx train tokens: {len(dataset_dict['train']) * CHUNK_LEN / 1e9:.2f}B")

dataset_dict.save_to_disk(str(TOKENIZED_DIR))
print(f"Saved to {TOKENIZED_DIR}")
'''

C_LOAD_BACK = '''# Re-load from disk to confirm save worked (also lets you resume from this point later)
from datasets import load_from_disk
dataset_dict = load_from_disk(str(TOKENIZED_DIR))
print("Loaded back from disk:")
print(dataset_dict)
'''


# ---------------------------------------------------------------------------
# Section D — Dry run
# ---------------------------------------------------------------------------

D_INTRO_MD = '''## Section D — Phase 3 dry run (50 steps)

**Critical gate.** Before launching the 36-48 hr full training, verify:
1. Model loads
2. Training loop runs without OOM
3. Loss is finite and trends down

If OOM at batch 64: bump `gradient_accumulation_steps=2` and drop `per_device_train_batch_size=32` (effective batch unchanged).
'''

D_MODEL_LOAD = '''model = AutoModelForMaskedLM.from_pretrained(base_model_name)
print(f"Loaded model from {base_model_name}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
)
'''

D_DRY_RUN = '''DRY_RUN_DIR = ARTIFACTS_DIR / "dry-run"
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
    lr_scheduler_type="cosine",
    bf16=True,
    save_strategy="no",
    eval_strategy="no",
    logging_strategy="steps",
    logging_steps=10,
    disable_tqdm=True,
    report_to=["none"],
    dataloader_num_workers=2,
    seed=42,
)

dry_trainer = Trainer(
    model=model,
    args=dry_args,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["validation"].select(range(min(64, len(dataset_dict["validation"])))),
    data_collator=data_collator,
)

print("Starting 50-step dry run...")
dry_trainer.train()
print("\\nDry run complete.")

# Inspect final loss
log_hist = dry_trainer.state.log_history
last_loss = next((e["loss"] for e in reversed(log_hist) if "loss" in e), None)
print(f"Final training loss: {last_loss}")
assert last_loss is not None and np.isfinite(last_loss), "Dry run loss is invalid; do NOT proceed to full training."
print("\\nGate passed. Proceed to Section E for full training.")
'''


# ---------------------------------------------------------------------------
# Section E — Full training
# ---------------------------------------------------------------------------

E_INTRO_MD = '''## Section E — Phase 3 full training (3 epochs, ~36–48 hr on 5090)

**This is the long cell.** Launch once and monitor live progress via the rendered widget and the TensorBoard sidecar.

The `CursorSafeProgressCallback` keeps the cell output bounded — it uses `clear_output(wait=True)` to replace the displayed progress table on each log event, mirroring the full log to `logs/finbert_finetune_<timestamp>.log` so nothing is lost.
'''

E_CALLBACK = '''class CursorSafeProgressCallback(TrainerCallback):
    """Bounded-output progress callback for Cursor IDE.

    Re-renders a single display widget on each log step (replace, not append).
    Mirrors the full log to a file under logs/ so nothing is lost.
    """

    def __init__(self, log_file: Path, plot_every_n_logs: int = 5, recent_rows: int = 20):
        self.log_file = log_file
        self.plot_every_n_logs = plot_every_n_logs
        self.recent_rows = recent_rows
        self.history: list[dict] = []
        self.file_logger = logging.getLogger(f"finbert_ft_{id(self)}")
        self.file_logger.setLevel(logging.INFO)
        # Clear handlers from any prior runs of this cell
        self.file_logger.handlers = []
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
        self.file_logger.addHandler(fh)

    def _render(self):
        clear_output(wait=True)
        df = pd.DataFrame(self.history)
        recent = df.tail(self.recent_rows)
        display(HTML(
            f"<b>FinBERT MLM FT</b> — {len(self.history)} log events; full log at {self.log_file}<br>"
            + recent.to_html(index=False, float_format=lambda v: f"{v:.4f}" if isinstance(v, float) else str(v))
        ))
        if len(self.history) >= 2 and len(self.history) % self.plot_every_n_logs == 0:
            train = df[df["loss"].notna()] if "loss" in df.columns else pd.DataFrame()
            val = df[df["eval_loss"].notna()] if "eval_loss" in df.columns else pd.DataFrame()
            fig, ax = plt.subplots(figsize=(8, 4))
            if not train.empty:
                ax.plot(train["step"], train["loss"], label="train", alpha=0.7)
            if not val.empty:
                ax.plot(val["step"], val["eval_loss"], label="val", marker="o", alpha=0.7)
            ax.set_xlabel("step")
            ax.set_ylabel("loss")
            ax.legend()
            ax.grid(True, alpha=0.3)
            display(fig)
            plt.close(fig)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        row = {"step": state.global_step, **logs}
        self.history.append(row)
        self.file_logger.info("step=%d %s", state.global_step, logs)
        self._render()


print("Callback class defined.")
'''

E_TENSORBOARD = '''# Optional: TensorBoard sidecar for streaming metrics (separate UI in Cursor)
# Comment out if you prefer to keep the notebook output as the sole monitor.
%load_ext tensorboard
%tensorboard --logdir {ARTIFACTS_DIR}/runs --port 6006
'''

E_FULL_TRAIN = '''timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOGS_DIR / f"finbert_finetune_{timestamp}.log"

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

trainer = Trainer(
    model=model,
    args=full_args,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["validation"],
    data_collator=data_collator,
    callbacks=[CursorSafeProgressCallback(log_file=log_file)],
)

print(f"Starting full FinBERT MLM FT. Log file: {log_file}")
print(f"Expected runtime on RTX 5090: ~36-48 hr for 3 epochs.")
print(f"If the kernel is interrupted, resume with: trainer.train(resume_from_checkpoint=True)")

trainer.train()
print("\\nTraining complete.")
'''


# ---------------------------------------------------------------------------
# Section F — Eval + save
# ---------------------------------------------------------------------------

F_INTRO_MD = '''## Section F — Phase 4: Eval + save artifacts

Final held-out perplexity, persist the model + tokenizer, and dump loss curves to a parquet for paper-ready plotting later.
'''

F_FINAL_EVAL = '''# Final held-out perplexity
eval_metrics = trainer.evaluate(eval_dataset=dataset_dict["validation"])
val_loss = eval_metrics["eval_loss"]
val_ppl = float(np.exp(val_loss))
print(f"Final val loss: {val_loss:.4f}")
print(f"Final val perplexity: {val_ppl:.2f}")
'''

F_SAVE_METRICS = '''# Save train/val curves to parquet (spec §17.1)
log_hist = pd.DataFrame(trainer.state.log_history)
metrics_path = METRICS_DIR / "finbert_finetune.parquet"
log_hist.to_parquet(metrics_path, index=False)
print(f"Saved {len(log_hist)} log events to {metrics_path}")

# Save a small summary alongside
summary = {
    "base_model": base_model_name,
    "final_train_loss": next((e["loss"] for e in reversed(trainer.state.log_history) if "loss" in e), None),
    "final_val_loss": val_loss,
    "final_val_perplexity": val_ppl,
    "total_steps": trainer.state.global_step,
    "total_epochs": trainer.state.epoch,
    "timestamp_finished": datetime.now().isoformat(),
}
(METRICS_DIR / "finbert_finetune_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
'''

F_PLOT = '''# Final loss plot (separate from the live callback plot)
df = log_hist.copy()
fig, ax = plt.subplots(figsize=(10, 5))
train = df[df["loss"].notna()] if "loss" in df.columns else pd.DataFrame()
val = df[df["eval_loss"].notna()] if "eval_loss" in df.columns else pd.DataFrame()
if not train.empty:
    ax.plot(train["step"], train["loss"], label="train", alpha=0.7)
if not val.empty:
    ax.plot(val["step"], val["eval_loss"], label="val", marker="o")
ax.set_xlabel("step")
ax.set_ylabel("loss")
ax.set_title("FinBERT MLM Fine-Tune — Loss")
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
'''

F_PERSIST = '''# Save the encoder + tokenizer
trainer.save_model(str(ARTIFACTS_DIR))
tokenizer.save_pretrained(str(ARTIFACTS_DIR))
print(f"\\nEncoder + tokenizer saved to {ARTIFACTS_DIR}")

# Sanity check: list saved files
for p in sorted(ARTIFACTS_DIR.iterdir()):
    if p.is_file():
        print(f"  {p.name} ({p.stat().st_size / 1e6:.1f} MB)")
'''


# ---------------------------------------------------------------------------
# Notebook builder
# ---------------------------------------------------------------------------

def build_notebook():
    nb = new_notebook()
    cells = [
        # Section A
        new_markdown_cell(A_INTRO_MD),
        new_markdown_cell(A_SECTION_MD),
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
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
    }
    return nb


def main():
    out_path = Path(__file__).resolve().parents[1] / "notebooks" / "01_finbert_finetune.ipynb"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nb = build_notebook()
    with out_path.open("w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"Wrote {out_path}")
    # Round-trip read to catch JSON / nbformat issues
    with out_path.open(encoding="utf-8") as f:
        loaded = nbformat.read(f, as_version=4)
    print(f"Round-trip OK. Cells: {len(loaded.cells)}")


if __name__ == "__main__":
    main()

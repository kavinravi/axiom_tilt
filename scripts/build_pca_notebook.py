"""Build notebooks/04_pca_text_features.ipynb from cell definitions.

Run via:
    python scripts/build_pca_notebook.py

Cell content is the source of truth here; the .ipynb is the regenerated artifact.
"""
from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


# =============================================================================
# Cell sources
# =============================================================================

INTRO_MD = """# 04 — Text-feature PCA

Reduce 768-dim FinBERT stock-day embeddings (notebook 03 output) to a ranker-ready
low-dim feature set. Fit on the first walk's training window (2002-2007), pick
`n_pca` at 99% cumulative variance + 1 safety buffer, lock the dim for all
subsequent walks. Re-fit components at each walk boundary.

**Spec:** `docs/superpowers/specs/2026-05-08-text-enhanced-rl-portfolio-design.md` §5.3 / §17.2.
**Plan:** `docs/superpowers/plans/2026-05-15-pca-text-features.md`.

**Mode switch:** `USE_SYNTHETIC=True` runs on planted-signal Gaussian data so the
notebook is executable end-to-end before notebook 03's GPU output lands. Flip to
`False` once `data/processed/finbert_stockday_embed/` is populated.
"""

A_SETUP = """from __future__ import annotations
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.io import processed_dir, repo_root
from src.utils.pca import (
    assemble_training_matrix,
    fit_pca_initial,
    fit_pca_walk,
    pick_n_components,
)

USE_SYNTHETIC = True  # flip to False after notebook 03 output lands

EMBED_DIR = processed_dir() / 'finbert_stockday_embed'
UNIVERSE_PATH = processed_dir() / 'universe_ids.parquet'
ARTIFACTS_DIR = repo_root() / 'artifacts' / 'pca-text'
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Spec §7.2 walk-forward windows
WALK_1_START, WALK_1_END = '2002-01-01', '2007-12-31'
WALK_2_START, WALK_2_END = '2003-01-01', '2008-12-31'

TARGETS = [0.95, 0.98, 0.99]
PROD_TARGET = 0.99
SANITY_MAX_N_PCA = 200  # spec §5.3 sanity check

print(f'USE_SYNTHETIC={USE_SYNTHETIC}')
print(f'embed_dir={EMBED_DIR}')
print(f'walk 1 window: {WALK_1_START} -> {WALK_1_END}')
print(f'production target: {PROD_TARGET}; sensitivity: {TARGETS}')
"""

B_MD = """## B. Load first-walk training matrix

Synthetic mode plants an 8-dim signal in 768-dim with ~2% Gaussian noise. The
cum-var curve should have a real elbow near 8 components; locked `n_pca` should
land at 9 (8 + 1 safety) at the 0.99 target. Real mode reads notebook 03 output
and applies the universe gate + weekly resample via `assemble_training_matrix`.
"""

B_LOAD = """if USE_SYNTHETIC:
    rng = np.random.RandomState(42)
    # Smoke-scale, not production-scale: 5K samples is plenty to identify an 8-dim
    # signal in 768-dim and keeps the full-SVD pass under a few seconds. Real mode
    # works at ~150K x 768 (one walk's worth of weekly snapshots).
    n_samples_w1 = 5_000
    n_signal = 8
    hidden = 768
    truth = rng.randn(n_samples_w1, n_signal).astype(np.float32)
    proj = rng.randn(n_signal, hidden).astype(np.float32)
    X_w1 = truth @ proj + rng.randn(n_samples_w1, hidden).astype(np.float32) * 0.02
    meta_w1 = pd.DataFrame({
        'permno': rng.choice(np.arange(10001, 10501), size=n_samples_w1),
        # Synthetic dates are decorative — PCA fit only uses X. Use random dates
        # inside walk 1 to avoid overflowing pandas ns timestamps on a long freq=B range.
        'date': pd.to_datetime('2002-01-04') + pd.to_timedelta(
            rng.randint(0, 6 * 365, size=n_samples_w1), unit='D'
        ),
    })
    print(f'synthetic walk 1: X={X_w1.shape}, samples={len(meta_w1):,}, planted_dim={n_signal}')
else:
    universe_ids = pd.read_parquet(UNIVERSE_PATH)
    X_w1, meta_w1 = assemble_training_matrix(
        embed_dir=EMBED_DIR,
        universe_ids=universe_ids,
        start=WALK_1_START,
        end=WALK_1_END,
    )
    print(f'real walk 1: X={X_w1.shape}, samples={len(meta_w1):,}')
    if len(meta_w1):
        print(f'  date range: {meta_w1.date.min().date()} -> {meta_w1.date.max().date()}')
        print(f'  unique permnos: {meta_w1.permno.nunique()}')
    assert len(meta_w1) >= 10_000, (
        f'walk 1 has {len(meta_w1)} samples; expected >= 10K. '
        'Did notebook 03 finish? Did universe_ids include enough permnos?'
    )
"""

C_MD = """## C. Fit PCA on first walk; pick `n_pca`; lock

Full-rank SVD gives the §17.2 cum-var curve. `pick_n_components` returns the
smallest `n` with `cum_var[n-1] >= target`, plus 1 safety, capped at full rank.
"""

C_FIT = """n_pca, cum_var, pca_w1 = fit_pca_initial(X_w1, target=PROD_TARGET)
captured_at_lock = float(cum_var[n_pca - 1]) if n_pca <= len(cum_var) else 1.0
print(f'locked n_pca = {n_pca}  (target={PROD_TARGET}, includes +1 safety)')
print(f'variance captured at n_pca: {captured_at_lock:.4f}')

sensitivity = {f'{t:.2f}': int(pick_n_components(cum_var, target=t)) for t in TARGETS}
print('\\nsensitivity (n_pca at each target):')
for k, v in sensitivity.items():
    print(f'  target={k}: n_pca={v}')

# Spec §5.3 sanity check
if n_pca >= SANITY_MAX_N_PCA:
    print(f'\\nWARNING: n_pca={n_pca} >= {SANITY_MAX_N_PCA}. Inspect the scree.')
    print('Options: lower target (95 / 98), L2-normalize embeddings before PCA, or')
    print('skip PCA entirely and use a different reducer (e.g., a small linear projection).')
"""

D_MD = """## D. Diagnostics — cumulative variance, scree

Cum-var on log-x to read the elbow clearly. Scree on log-y so the long noise
tail stays visible (otherwise the first eigenvalue swamps everything).
"""

D_PLOTS = """fig, axes = plt.subplots(1, 2, figsize=(12, 4))

xs = np.arange(1, len(cum_var) + 1)
ax = axes[0]
ax.plot(xs, cum_var, lw=1.2)
for t, color in zip(TARGETS, ['tab:red', 'tab:orange', 'tab:green']):
    ax.axhline(t, color=color, ls='--', lw=0.7, label=f'target={t}')
ax.axvline(n_pca, color='black', ls=':', lw=0.7, label=f'locked n_pca={n_pca}')
ax.set_xlabel('component')
ax.set_ylabel('cumulative explained variance')
ax.set_title('PCA cumulative variance — first walk')
ax.set_xscale('log')
ax.legend(loc='lower right', fontsize=8)
ax.grid(alpha=0.3)

full_evr = np.diff(np.concatenate([[0.0], cum_var]))
ax = axes[1]
ax.plot(xs, full_evr, lw=0.8)
ax.set_yscale('log')
ax.set_xlabel('component')
ax.set_ylabel('explained variance ratio (log)')
ax.set_title('Scree — first walk')
ax.grid(alpha=0.3, which='both')

plt.tight_layout()
plt.show()
"""

E_MD = """## E. Walk-2 re-fit demo at locked dim

`fit_pca_walk(X, n_pca)` runs PCA with the locked dim on the next walk's
training matrix. `variance_captured` per walk is the §17.2 drift sanity check —
if the captured fraction falls noticeably below `PROD_TARGET` across walks, the
locked dim has become too tight (text "topic geometry" is shifting).
"""

E_REFIT = """if USE_SYNTHETIC:
    n_samples_w2 = 5_000  # smoke-scale, same as walk 1
    truth_w2 = rng.randn(n_samples_w2, n_signal).astype(np.float32)
    X_w2 = truth_w2 @ proj + rng.randn(n_samples_w2, hidden).astype(np.float32) * 0.02
else:
    X_w2, _ = assemble_training_matrix(EMBED_DIR, universe_ids, WALK_2_START, WALK_2_END)

pca_w2, var_captured_w2 = fit_pca_walk(X_w2, n_pca=n_pca)
print(f'walk 2 fit: n_components={pca_w2.n_components_}, variance_captured={var_captured_w2:.4f}')
print(f'walk 1 captured:                                {captured_at_lock:.4f}')
print(f'drift (walk 2 minus walk 1):                    {var_captured_w2 - captured_at_lock:+.4f}')
"""

F_MD = """## F. Persist artifacts

`artifacts/pca-text/walk-001/` (gitignored): fitted PCA, full cum-var curve,
summary JSON. The ranker training notebook (TBD) will load these for the
first walk's PCA transformer.
"""

F_PERSIST = """WALK_1_DIR = ARTIFACTS_DIR / 'walk-001'
WALK_1_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(pca_w1, WALK_1_DIR / 'pca.joblib')
np.save(WALK_1_DIR / 'cum_var.npy', cum_var)

summary = {
    'walk_id': 1,
    'window_start': WALK_1_START,
    'window_end': WALK_1_END,
    'target_variance': PROD_TARGET,
    'locked_n_pca': int(n_pca),
    'variance_captured_at_n_pca': captured_at_lock,
    'sensitivity_n_pca': sensitivity,
    'n_train_samples': int(X_w1.shape[0]),
    'hidden_dim': int(X_w1.shape[1]),
    'use_synthetic': USE_SYNTHETIC,
}
(WALK_1_DIR / 'summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
print(f'\\nartifacts -> {WALK_1_DIR.relative_to(repo_root())}')
"""


# =============================================================================
# Build
# =============================================================================

def build_notebook() -> nbformat.NotebookNode:
    nb = new_notebook()
    nb.cells = [
        new_markdown_cell(INTRO_MD),
        new_markdown_cell('## A. Setup'),
        new_code_cell(A_SETUP),
        new_markdown_cell(B_MD),
        new_code_cell(B_LOAD),
        new_markdown_cell(C_MD),
        new_code_cell(C_FIT),
        new_markdown_cell(D_MD),
        new_code_cell(D_PLOTS),
        new_markdown_cell(E_MD),
        new_code_cell(E_REFIT),
        new_markdown_cell(F_MD),
        new_code_cell(F_PERSIST),
    ]
    nb.metadata = {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.11'},
    }
    return nb


def main() -> None:
    out_path = Path(__file__).resolve().parents[1] / 'notebooks' / '04_pca_text_features.ipynb'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nb = build_notebook()
    with out_path.open('w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(f'Wrote {out_path}')

    # Round-trip read to catch JSON / nbformat issues
    with out_path.open(encoding='utf-8') as f:
        loaded = nbformat.read(f, as_version=4)
    print(f'Round-trip OK. Cells: {len(loaded.cells)}')


if __name__ == '__main__':
    main()

"""Build walk-001 PCA + training_panel + ranker + scoreboard.

Stand-alone version of notebooks 04 → 05 → 06 → 07[A-B], scoped to walk 1 only.
Skips notebook 06's optuna HP search; uses default ranker params so it finishes
in ~10 minutes on CPU. The resulting scoreboard.parquet feeds run_one_dylan.py.

Checkpointed: each output is skipped if it already exists. Re-run to resume.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from lightgbm import early_stopping

from src.utils.io import processed_dir, repo_root
from src.utils.pca import assemble_training_matrix
from src.utils.features import (
    pivot_macro_wide,
    compute_forward_returns,
    compute_text_novelty,
    compute_days_since_filing,
    compute_doc_count_window,
)
from src.utils.ranker import (
    DEFAULT_RANKER_PARAMS,
    assemble_walk_features,
    build_ranker,
    compute_excess_return_buckets,
    drop_zero_info_columns,
    evaluate_ranker,
    project_text_to_pca,
    friday_only,
)
from src.utils.rl_env import build_scoreboard_from_scored_panel

WALK_ID = 1
TRAIN_START, TRAIN_END = '2002-01-01', '2007-12-31'
VAL_START,   VAL_END   = '2008-01-01', '2008-12-31'
TEST_START,  TEST_END  = '2009-01-01', '2009-12-31'
TOP_K = 30
N_BUCKETS = 32
PCA_TARGET = 0.95
N_PCA_SAFETY = 1
RANDOM_STATE = 42

ROOT = repo_root()
EMBED_DIR = processed_dir() / 'finbert_stockday_embed'
PANEL_DIR = processed_dir() / 'panel'
TRAINING_PANEL_DIR = processed_dir() / 'training_panel'
MACRO_PATH = processed_dir() / 'macro.parquet'
EDGAR_INDEX_PATH = processed_dir() / 'edgar_index.parquet'
UNIVERSE_PATH = processed_dir() / 'universe_ids.parquet'

PCA_DIR = ROOT / 'artifacts' / 'pca-text' / f'walk-{WALK_ID:03d}'
PCA_PATH = PCA_DIR / 'pca.joblib'

RANKER_DIR = ROOT / 'artifacts' / 'ranker' / f'walk-{WALK_ID:03d}'
RANKER_PATH = RANKER_DIR / 'model.joblib'

RL_DIR = ROOT / 'artifacts' / 'rl' / f'walk-{WALK_ID:03d}'
SCOREBOARD_PATH = RL_DIR / 'scoreboard.parquet'


def step1_fit_pca():
    if PCA_PATH.exists():
        print(f'[1/4] PCA exists -> {PCA_PATH.relative_to(ROOT)}; skip')
        return
    print(f'[1/4] fit walk-1 PCA on {TRAIN_START}..{TRAIN_END}')
    PCA_DIR.mkdir(parents=True, exist_ok=True)

    universe_ids = pd.read_parquet(UNIVERSE_PATH)
    universe_ids['permno'] = universe_ids['permno'].astype('Int64')

    t0 = time.time()
    X, meta = assemble_training_matrix(
        embed_dir=EMBED_DIR,
        universe_ids=universe_ids,
        start=TRAIN_START,
        end=TRAIN_END,
    )
    print(f'  assemble matrix: X={X.shape}, meta_rows={len(meta)} ({time.time()-t0:.1f}s)')

    t0 = time.time()
    pca_full = PCA(svd_solver='full').fit(X)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    if cum_var[-1] < PCA_TARGET:
        n_pca = len(cum_var)
    else:
        n_pca = int(np.searchsorted(cum_var, PCA_TARGET, side='left')) + 1
        n_pca = min(n_pca + N_PCA_SAFETY, len(cum_var))
    print(f'  full SVD: n_components_full={pca_full.n_components_}, '
          f'cum_var[final]={cum_var[-1]:.4f}; locked n_pca={n_pca} '
          f'(cum_var={cum_var[n_pca-1]:.4f}) ({time.time()-t0:.1f}s)')

    pca = PCA(n_components=n_pca, svd_solver='full').fit(X)
    joblib.dump(pca, PCA_PATH)
    np.save(PCA_DIR / 'cum_var.npy', cum_var)
    (PCA_DIR / 'summary.json').write_text(json.dumps({
        'walk_id': WALK_ID, 'window_start': TRAIN_START, 'window_end': TRAIN_END,
        'n_train_samples': int(X.shape[0]), 'n_meta_rows': int(len(meta)),
        'hidden_dim': int(X.shape[1]), 'locked_n_pca': int(n_pca),
        'target_variance': PCA_TARGET, 'variance_captured': float(pca.explained_variance_ratio_.sum()),
        'use_synthetic': False,
    }, indent=2))
    print(f'  saved -> {PCA_PATH.relative_to(ROOT)}')


def step2_training_panel():
    # Build for walk-1 train + val + test (2002-2009).
    YEARS = list(range(2002, 2010))
    needed = [TRAINING_PANEL_DIR / f'year={y}' / 'part-0.parquet' for y in YEARS]
    if all(p.exists() for p in needed):
        print(f'[2/4] training_panel exists for years {YEARS}; skip')
        return
    print(f'[2/4] assemble training_panel for years {YEARS}')
    TRAINING_PANEL_DIR.mkdir(parents=True, exist_ok=True)

    universe_ids = pd.read_parquet(UNIVERSE_PATH)
    universe_ids['permno'] = universe_ids['permno'].astype('Int64')
    universe_ids['date_out'] = universe_ids['date_out'].fillna(pd.Timestamp('2099-12-31'))

    macro_long = pd.read_parquet(MACRO_PATH)

    edgar_index = pd.read_parquet(EDGAR_INDEX_PATH)
    edgar_index['filing_date'] = pd.to_datetime(edgar_index['filing_date'])
    edgar_index = edgar_index[edgar_index['cik'].isin(universe_ids['cik'])]
    print(f'  edgar_index: {len(edgar_index):,} filings (universe-filtered)')

    def _normalize_cik(s):
        return s.map(lambda x: f'{int(x):010d}' if pd.notna(x) else None)

    def _read_year_embed(year: int) -> pd.DataFrame:
        shards = sorted(EMBED_DIR.glob(f'year={year}/*.parquet'))
        prior_shards = sorted(EMBED_DIR.glob(f'year={year - 1}/*.parquet'))
        frames = []
        for s in shards:
            frames.append(pd.read_parquet(s, columns=['permno', 'date', 'vec']))
        if prior_shards:
            for s in prior_shards:
                df = pd.read_parquet(s, columns=['permno', 'date', 'vec'])
                df = df[df['date'] >= pd.Timestamp(f'{year - 1}-12-24')]
                frames.append(df)
        if not frames:
            return pd.DataFrame(columns=['permno', 'date', 'vec'])
        embed = pd.concat(frames, ignore_index=True)
        embed['date'] = pd.to_datetime(embed['date'])
        return embed

    for year in YEARS:
        out_path = TRAINING_PANEL_DIR / f'year={year}' / 'part-0.parquet'
        if out_path.exists():
            print(f'  year={year}: exists; skip')
            continue
        t0 = time.time()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        panel_shards = sorted(PANEL_DIR.glob(f'year={year}/*.parquet'))
        if not panel_shards:
            raise FileNotFoundError(f'no panel shards for year={year}')
        base = pd.concat([pd.read_parquet(s) for s in panel_shards], ignore_index=True)
        base['date'] = pd.to_datetime(base['date'])
        base['cik'] = _normalize_cik(base['cik'])

        intervals = universe_ids.dropna(subset=['permno'])[['permno', 'date_in', 'date_out']].copy()
        intervals['permno'] = intervals['permno'].astype('int64')
        merged = base.merge(intervals, on='permno', how='inner')
        in_window = (merged['date'] >= merged['date_in']) & (merged['date'] <= merged['date_out'])
        base = (merged[in_window]
                .drop(columns=['date_in', 'date_out'])
                .drop_duplicates(subset=['permno', 'date']))

        year_dates = pd.DatetimeIndex(sorted(base['date'].unique()))
        macro_w = pivot_macro_wide(macro_long, ffill_dates=year_dates)
        base = base.merge(macro_w, on='date', how='left')

        embed = _read_year_embed(year)
        novelty = compute_text_novelty(embed, lookback_days=7)
        novelty = novelty[novelty['date'].dt.year == year]
        base = base.merge(novelty, on=['permno', 'date'], how='left')

        dsf = compute_days_since_filing(edgar_index, base[['permno', 'cik', 'date']])
        base = base.merge(dsf[['permno', 'date', 'days_since_filing']],
                          on=['permno', 'date'], how='left')

        dcw = compute_doc_count_window(edgar_index, base[['permno', 'cik', 'date']],
                                       window_days=7)
        base = base.merge(dcw[['permno', 'date', 'doc_count_7d']],
                          on=['permno', 'date'], how='left')

        # Forward returns. compute_forward_returns walks intra-year; for the last
        # 5 trading days of the year we need a peek at the next year's prefix.
        next_year_panel = sorted(PANEL_DIR.glob(f'year={year + 1}/*.parquet'))
        if next_year_panel:
            prefix = pd.concat([pd.read_parquet(s) for s in next_year_panel], ignore_index=True)
            prefix['date'] = pd.to_datetime(prefix['date'])
            prefix = prefix[prefix['date'] < pd.Timestamp(f'{year + 1}-01-15')]
            prefix = prefix[['permno', 'date', 'ret']].drop_duplicates(subset=['permno', 'date'])
            stitched = pd.concat([base[['permno', 'date', 'ret']], prefix], ignore_index=True)
        else:
            stitched = base[['permno', 'date', 'ret']]

        labeled = compute_forward_returns(stitched, horizons=(1, 5))
        labeled = labeled[labeled['date'].dt.year == year]
        base = base.merge(labeled[['permno', 'date', 'fwd_ret_1d', 'fwd_ret_5d']],
                          on=['permno', 'date'], how='left')

        tmp = out_path.with_suffix('.parquet.tmp')
        base.to_parquet(tmp, compression='zstd', index=False)
        tmp.rename(out_path)
        print(f'  year={year}: wrote {len(base):,} rows ({time.time()-t0:.1f}s)')


def step3_ranker():
    if RANKER_PATH.exists():
        print(f'[3/4] ranker exists -> {RANKER_PATH.relative_to(ROOT)}; skip')
        return
    print(f'[3/4] train LightGBM ranker on walk-1 (default params, no optuna)')
    RANKER_DIR.mkdir(parents=True, exist_ok=True)

    pca = joblib.load(PCA_PATH)
    print(f'  PCA loaded: n_components={pca.n_components_}')

    def _load_panel(s, e):
        s, e = pd.Timestamp(s), pd.Timestamp(e)
        frames = []
        for y in range(s.year, e.year + 1):
            for p in sorted((TRAINING_PANEL_DIR / f'year={y}').glob('*.parquet')):
                df = pd.read_parquet(p)
                df['date'] = pd.to_datetime(df['date'])
                df = df[(df['date'] >= s) & (df['date'] <= e)]
                frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _load_embed(s, e):
        s, e = pd.Timestamp(s), pd.Timestamp(e)
        frames = []
        for y in range(s.year, e.year + 1):
            for p in sorted((EMBED_DIR / f'year={y}').glob('*.parquet')):
                df = pd.read_parquet(p, columns=['permno', 'date', 'vec'])
                df['date'] = pd.to_datetime(df['date'])
                df = df[(df['date'] >= s) & (df['date'] <= e)]
                frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    t0 = time.time()
    panel_tr = _load_panel(TRAIN_START, TRAIN_END)
    panel_va = _load_panel(VAL_START, VAL_END)
    panel_te = _load_panel(TEST_START, TEST_END)
    embed_tr = _load_embed(TRAIN_START, TRAIN_END)
    embed_va = _load_embed(VAL_START, VAL_END)
    embed_te = _load_embed(TEST_START, TEST_END)
    print(f'  loaded panels + embeds ({time.time()-t0:.1f}s)')

    t0 = time.time()
    epca_tr = project_text_to_pca(embed_tr, pca)
    epca_va = project_text_to_pca(embed_va, pca)
    epca_te = project_text_to_pca(embed_te, pca)
    X_tr, y_tr, g_tr, m_tr = assemble_walk_features(panel_tr, epca_tr)
    X_va, y_va, g_va, m_va = assemble_walk_features(panel_va, epca_va)
    X_te, y_te, g_te, m_te = assemble_walk_features(panel_te, epca_te)
    print(f'  X_train={X_tr.shape}, X_val={X_va.shape}, X_test={X_te.shape} ({time.time()-t0:.1f}s)')

    X_tr, X_va, X_te = drop_zero_info_columns(X_tr, X_va, X_te)
    print(f'  feature count after drop_zero_info: {X_tr.shape[1]}')

    buckets_tr = compute_excess_return_buckets(m_tr, n_buckets=N_BUCKETS).astype(int).values
    buckets_va = compute_excess_return_buckets(m_va, n_buckets=N_BUCKETS).astype(int).values

    t0 = time.time()
    params = {**DEFAULT_RANKER_PARAMS, 'n_estimators': 2000}
    model = build_ranker(params)
    model.fit(
        X_tr, buckets_tr,
        group=g_tr,
        eval_set=[(X_va, buckets_va)],
        eval_group=[g_va],
        eval_at=[TOP_K],
        callbacks=[early_stopping(stopping_rounds=50, verbose=False)],
    )
    print(f'  fit done in {time.time()-t0:.1f}s; best_iter={model.best_iteration_}, '
          f'val ndcg@{TOP_K}={model.best_score_["valid_0"][f"ndcg@{TOP_K}"]:.4f}')

    test_metrics = evaluate_ranker(model, X_te, y_te, m_te['date'],
                                   top_k=TOP_K, entity_ids=m_te['permno'])
    print(f'  test metrics: {test_metrics}')

    bundle = {'model': model, 'feature_names': X_tr.columns.tolist()}
    joblib.dump(bundle, RANKER_PATH)
    (RANKER_DIR / 'summary.json').write_text(json.dumps({
        'walk_id': WALK_ID, 'best_iter': int(model.best_iteration_),
        'val_ndcg': float(model.best_score_['valid_0'][f'ndcg@{TOP_K}']),
        'test_metrics': test_metrics,
        'n_features': int(X_tr.shape[1]),
        'feature_names': X_tr.columns.tolist(),
    }, indent=2, default=float))
    print(f'  saved -> {RANKER_PATH.relative_to(ROOT)}')


def step4_scoreboard():
    if SCOREBOARD_PATH.exists():
        print(f'[4/4] scoreboard exists -> {SCOREBOARD_PATH.relative_to(ROOT)}; skip')
        return
    print(f'[4/4] build scoreboard.parquet for walk-1 (train+val+test)')
    RL_DIR.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(RANKER_PATH)
    model = bundle['model']
    feat_names = bundle['feature_names']
    pca = joblib.load(PCA_PATH)
    print(f'  ranker: {len(feat_names)} features; PCA n_components={pca.n_components_}')

    def _load_panel(s, e):
        s, e = pd.Timestamp(s), pd.Timestamp(e)
        frames = []
        for y in range(s.year, e.year + 1):
            for p in sorted((TRAINING_PANEL_DIR / f'year={y}').glob('*.parquet')):
                df = pd.read_parquet(p)
                df['date'] = pd.to_datetime(df['date'])
                df = df[(df['date'] >= s) & (df['date'] <= e)]
                frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _load_embed(s, e):
        s, e = pd.Timestamp(s), pd.Timestamp(e)
        frames = []
        for y in range(s.year, e.year + 1):
            for p in sorted((EMBED_DIR / f'year={y}').glob('*.parquet')):
                df = pd.read_parquet(p, columns=['permno', 'date', 'vec'])
                df['date'] = pd.to_datetime(df['date'])
                df = df[(df['date'] >= s) & (df['date'] <= e)]
                frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    panel = _load_panel(TRAIN_START, TEST_END)
    embed = _load_embed(TRAIN_START, TEST_END)
    embed_pca = project_text_to_pca(embed, pca)

    fri = friday_only(panel).merge(embed_pca, on=['permno', 'date'], how='inner')
    fri = fri.dropna(subset=['fwd_ret_5d']).copy()

    X = pd.DataFrame({c: fri[c] if c in fri.columns else np.nan for c in feat_names})
    fri['score'] = model.predict(X)

    scoreboard = build_scoreboard_from_scored_panel(fri, top_k=TOP_K)
    scoreboard.to_parquet(SCOREBOARD_PATH, compression='zstd', index=False)
    print(f'  wrote {len(scoreboard):,} rows ({scoreboard["date"].nunique()} Fridays) '
          f'-> {SCOREBOARD_PATH.relative_to(ROOT)}')


if __name__ == '__main__':
    print(f'bootstrap walk-{WALK_ID:03d} pipeline')
    print(f'repo root: {ROOT}')
    step1_fit_pca()
    step2_training_panel()
    step3_ranker()
    step4_scoreboard()
    print('done')

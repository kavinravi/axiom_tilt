"""Extend each walk's scoreboard to cover its test year.

Notebook 07 builds scoreboards for train+val only (artifacts/rl/walk-N/scoreboard.parquet).
To run the full-period backtest we also need test-year rows. This script mirrors
notebook 08 cell A's extension logic but loops over all 16 walks.

For each walk N:
  - test year = 2009 + N - 1
  - if scoreboard already contains rows in that year, skip
  - else: load walk-N's ranker + PCA, load panel + embeds for the test year,
    project to PCA, score, build top-K scoreboard, append, persist

usage: python experiments/extend_scoreboards.py [walk_start] [walk_end]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import joblib
import numpy as np
import pandas as pd

from src.utils.io import processed_dir, repo_root
from src.utils.ranker import friday_only, load_walk_pca, project_text_to_pca
from src.utils.rl_env import build_scoreboard_from_scored_panel

REPO_ROOT = repo_root()
PANEL_DIR = processed_dir() / 'training_panel'
EMBED_DIR = processed_dir() / 'finbert_stockday_embed'
RANKER_ROOT = REPO_ROOT / 'artifacts' / 'ranker'
RL_ROOT = REPO_ROOT / 'artifacts' / 'rl'
TOP_K = 30


def load_years(dir_: Path, start: str, end: str, cols=None) -> pd.DataFrame:
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    frames = []
    for y in range(s.year, e.year + 1):
        for p in sorted((dir_ / f'year={y}').glob('*.parquet')):
            df = pd.read_parquet(p, columns=cols)
            df['date'] = pd.to_datetime(df['date'])
            df = df[(df['date'] >= s) & (df['date'] <= e)]
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def extend_walk(walk_id: int) -> dict:
    test_year = 2009 + walk_id - 1
    sb_path = RL_ROOT / f'walk-{walk_id:03d}' / 'scoreboard.parquet'
    if not sb_path.exists():
        return {'walk': walk_id, 'status': 'no_scoreboard'}

    sb = pd.read_parquet(sb_path)
    sb['date'] = pd.to_datetime(sb['date'])
    existing_test = sb[(sb['date'] >= f'{test_year}-01-01') &
                       (sb['date'] <= f'{test_year}-12-31')]
    if len(existing_test) > 0:
        return {'walk': walk_id, 'status': 'already_has_test', 'rows': len(existing_test)}

    ranker_path = RANKER_ROOT / f'walk-{walk_id:03d}' / 'model.joblib'
    if not ranker_path.exists():
        return {'walk': walk_id, 'status': 'no_ranker'}

    bundle = joblib.load(ranker_path)
    model = bundle['model']
    features = bundle['feature_names']
    pca, _ = load_walk_pca(walk_id)

    test_s, test_e = f'{test_year}-01-01', f'{test_year}-12-31'
    panel = load_years(PANEL_DIR, test_s, test_e)
    embed = load_years(EMBED_DIR, test_s, test_e, cols=['permno', 'date', 'vec'])
    if len(panel) == 0 or len(embed) == 0:
        return {'walk': walk_id, 'status': 'no_data', 'panel': len(panel), 'embed': len(embed)}

    embed_pca = project_text_to_pca(embed, pca)
    fri = friday_only(panel).merge(embed_pca, on=['permno', 'date'], how='inner')
    fri = fri.dropna(subset=['fwd_ret_5d']).copy()
    if len(fri) == 0:
        return {'walk': walk_id, 'status': 'no_fridays_with_data'}

    X = pd.DataFrame({c: fri[c] if c in fri.columns else np.nan for c in features})
    fri['score'] = model.predict(X)
    new_scores = build_scoreboard_from_scored_panel(fri, top_k=TOP_K)

    sb_combined = pd.concat([sb, new_scores], ignore_index=True)
    sb_combined = sb_combined.sort_values(['date', 'permno']).reset_index(drop=True)
    sb_combined.to_parquet(sb_path, compression='zstd', index=False)

    return {'walk': walk_id, 'status': 'extended',
            'added_rows': len(new_scores),
            'added_fridays': int(new_scores['date'].nunique())}


def main():
    walk_start = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    walk_end = int(sys.argv[2]) if len(sys.argv) > 2 else 16

    print(f'=== EXTENDING SCOREBOARDS: walks {walk_start}..{walk_end} ===\n')
    for w in range(walk_start, walk_end + 1):
        r = extend_walk(w)
        print(f'  walk {w:2d}: {r}')


if __name__ == '__main__':
    main()

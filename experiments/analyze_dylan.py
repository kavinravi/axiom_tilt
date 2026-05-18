"""Post-sweep diagnostic for the Dylan autoresearch session.

Reads results_dylan.tsv, prints the leaderboard, and for the top configs walks
through their training/eval logs to extract: (a) gross vs net Sharpe split,
(b) weight concentration over time, (c) overlap with Score-Prop's picks, (d)
turnover dynamics. The goal is causal attribution — "this lever is why."

Usage:
    python experiments/analyze_dylan.py           # leaderboard only
    python experiments/analyze_dylan.py --top 5   # also deep-dive top-N
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.utils.io import repo_root
from src.utils.rl_env import PortfolioEnv, project_to_simplex
from src.utils.backtest import score_proportional_weights, compute_strategy_metrics


REPO_ROOT = repo_root()
EXP_DIR = REPO_ROOT / 'experiments'
RESULTS = EXP_DIR / 'results_dylan.tsv'
TOP_K = 30
TEST_START, TEST_END = '2009-01-01', '2009-12-31'


def leaderboard() -> pd.DataFrame:
    df = pd.read_csv(RESULTS, sep='\t')
    df = df.sort_values('excess_sharpe', ascending=False).reset_index(drop=True)
    print(f'{len(df)} runs in {RESULTS.name}')
    print()
    cols = ['exp_id', 'reward_type', 'action_high', 'max_weight',
            'train_cost_bps', 'episode_length', 'sharpe_window', 'gamma',
            'ent_coef', 'turnover_lambda', 'total_timesteps',
            'ppo_sharpe', 'sp_sharpe', 'excess_sharpe', 'ppo_turnover',
            'ppo_annret', 'status']
    print(df[cols].to_string(index=False))
    print()
    wins = df[df['excess_sharpe'] > 0]
    print(f'wins: {len(wins)} / {len(df)}')
    if len(wins) > 0:
        print()
        print('=== WINNERS ===')
        print(wins[cols].to_string(index=False))
    return df


def deep_dive(exp_id: str, cfg: dict, out_dir: Path):
    """Reconstruct PPO + Score-Prop weight series for the test year and dump
    a per-week JSONL trace so we can plot/compare offline."""
    sb = pd.read_parquet(REPO_ROOT / 'artifacts/rl/walk-001/scoreboard.parquet')
    sb['date'] = pd.to_datetime(sb['date'])
    sb_test = sb[(sb['date'] >= TEST_START) & (sb['date'] <= TEST_END)].copy()
    by_date = {d: g.reset_index(drop=True) for d, g in sb_test.groupby('date')}
    dates = sorted(by_date.keys())

    model = PPO.load(out_dir / 'best_model.zip')
    eval_cost = cfg.get('eval_cost_bps', 5.0)

    def _env_fn():
        env = PortfolioEnv(scoreboard=sb_test, top_k=TOP_K,
                           episode_length=cfg['episode_length'],
                           cost_bps=eval_cost, max_weight=cfg['max_weight'],
                           reward_type=cfg['reward_type'],
                           sharpe_window=cfg.get('sharpe_window', 16),
                           action_high=cfg['action_high'],
                           turnover_lambda=cfg.get('turnover_lambda', 0.0))
        env.reset(seed=0)
        return Monitor(env)
    vec = DummyVecEnv([_env_fn])
    vec = VecNormalize.load(str(out_dir / 'vec_normalize.pkl'), vec)
    vec.training = False
    vec.norm_reward = False

    trace = []
    prev_p = np.full(TOP_K, 1.0 / TOP_K, dtype=np.float32)
    prev_s = np.full(TOP_K, 1.0 / TOP_K, dtype=np.float32)
    last_ret_p = 0.0
    for date in dates:
        cur = by_date[date]
        scores = cur['score'].to_numpy(dtype=np.float32)[:TOP_K]
        permnos = cur['permno'].to_numpy()[:TOP_K]
        rets = np.nan_to_num(cur['fwd_ret_5d'].to_numpy(dtype=np.float32)[:TOP_K], nan=0.0)

        tmp = PortfolioEnv(scoreboard=sb_test, top_k=TOP_K,
                           episode_length=cfg['episode_length'],
                           cost_bps=eval_cost, max_weight=cfg['max_weight'],
                           reward_type=cfg['reward_type'],
                           sharpe_window=cfg.get('sharpe_window', 16),
                           action_high=cfg['action_high'],
                           turnover_lambda=cfg.get('turnover_lambda', 0.0))
        tmp.reset(seed=0)
        tmp._idx = tmp._dates.tolist().index(date)
        tmp._weights = np.asarray(prev_p, dtype=np.float32)
        tmp._last_return = float(last_ret_p)
        obs = tmp._build_obs()
        obs_norm = vec.normalize_obs(obs)
        action, _ = model.predict(obs_norm, deterministic=True)
        new_p = project_to_simplex(np.asarray(action, dtype=np.float64),
                                   max_weight=cfg['max_weight']).astype(np.float32)
        new_s = score_proportional_weights(scores, max_weight=cfg['max_weight'])

        p_ret = float(np.dot(new_p, rets))
        s_ret = float(np.dot(new_s, rets))
        p_turn = float(np.abs(new_p - prev_p).sum())
        s_turn = float(np.abs(new_s - prev_s).sum())

        # Concentration: HHI and effective N.
        hhi_p = float((new_p ** 2).sum())
        hhi_s = float((new_s ** 2).sum())
        eff_n_p = 1.0 / hhi_p if hhi_p > 0 else float('nan')
        eff_n_s = 1.0 / hhi_s if hhi_s > 0 else float('nan')

        # Overlap with Score-Prop's top picks.
        top_p = set(permnos[np.argsort(new_p)[::-1][:10]].tolist())
        top_s = set(permnos[np.argsort(new_s)[::-1][:10]].tolist())
        overlap = len(top_p & top_s) / 10.0

        trace.append({
            'date': str(date.date()),
            'ppo_ret': p_ret, 'sp_ret': s_ret, 'ret_diff': p_ret - s_ret,
            'ppo_turn': p_turn, 'sp_turn': s_turn, 'turn_diff': p_turn - s_turn,
            'ppo_hhi': hhi_p, 'sp_hhi': hhi_s,
            'ppo_eff_n': eff_n_p, 'sp_eff_n': eff_n_s,
            'top10_overlap': overlap,
        })

        prev_p, prev_s, last_ret_p = new_p, new_s, p_ret

    trace_path = out_dir / 'trace_dylan.jsonl'
    with trace_path.open('w') as f:
        for row in trace:
            f.write(json.dumps(row) + '\n')

    tdf = pd.DataFrame(trace)
    print(f'\n--- {exp_id} deep-dive ---')
    print(f'  weeks: {len(tdf)}')
    print(f'  PPO  mean turnover: {tdf["ppo_turn"].mean():.4f}, mean HHI: {tdf["ppo_hhi"].mean():.4f}, '
          f'mean eff-N: {tdf["ppo_eff_n"].mean():.2f}')
    print(f'  SP   mean turnover: {tdf["sp_turn"].mean():.4f}, mean HHI: {tdf["sp_hhi"].mean():.4f}, '
          f'mean eff-N: {tdf["sp_eff_n"].mean():.2f}')
    print(f'  top10 overlap (mean): {tdf["top10_overlap"].mean():.2f}')
    print(f'  gross ret diff (PPO - SP, mean): {tdf["ret_diff"].mean() * 1e4:.2f} bps/wk')
    p_m = compute_strategy_metrics(tdf['ppo_ret'].to_numpy(),
                                   tdf['ppo_turn'].to_numpy(), cost_bps=eval_cost)
    s_m = compute_strategy_metrics(tdf['sp_ret'].to_numpy(),
                                   tdf['sp_turn'].to_numpy(), cost_bps=eval_cost)
    print(f'  PPO net Sharpe (recompute): {p_m["sharpe"]:.4f}, ann_ret={p_m["annualized_return"]:.4f}')
    print(f'  SP  net Sharpe (recompute): {s_m["sharpe"]:.4f}, ann_ret={s_m["annualized_return"]:.4f}')
    p_gross = float(np.mean(tdf['ppo_ret']) * 52 / max(np.std(tdf['ppo_ret']) * np.sqrt(52), 1e-9))
    s_gross = float(np.mean(tdf['sp_ret']) * 52 / max(np.std(tdf['sp_ret']) * np.sqrt(52), 1e-9))
    print(f'  PPO gross Sharpe (no costs): {p_gross:.4f}')
    print(f'  SP  gross Sharpe (no costs): {s_gross:.4f}')
    print(f'  trace -> {trace_path.relative_to(REPO_ROOT)}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--top', type=int, default=0,
                   help='Number of top runs to deep-dive (0 = leaderboard only)')
    p.add_argument('--ids', nargs='*', default=None,
                   help='Specific exp_ids to deep-dive (overrides --top)')
    args = p.parse_args()

    df = leaderboard()
    if not args.top and not args.ids:
        sys.exit(0)
    targets = args.ids or df.head(args.top)['exp_id'].tolist()
    print(f'\ndeep-diving: {targets}')

    for exp_id in targets:
        cfg_path = EXP_DIR / 'configs_dylan' / f'{exp_id.lstrip("d_")}.json'
        if not cfg_path.exists():
            # Some configs may be saved with the d_ prefix
            cfg_path = EXP_DIR / 'configs_dylan' / f'{exp_id}.json'
        if not cfg_path.exists():
            print(f'  {exp_id}: no config file found, skip')
            continue
        cfg = json.loads(cfg_path.read_text())
        out_dir = EXP_DIR / 'runs' / exp_id
        if not (out_dir / 'best_model.zip').exists():
            print(f'  {exp_id}: no best_model.zip in {out_dir}, skip')
            continue
        deep_dive(exp_id, cfg, out_dir)

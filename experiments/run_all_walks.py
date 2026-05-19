"""Multi-walk training with a custom config.

Mirrors notebook 07's cell-G train-one-walk loop but reads the algorithm/HP
config from a JSON file, so an autoresearch winner can be trained across all
16 walks without editing the notebook.

Saves to artifacts/rl_round2/walk-{N:03d}/cost-005bps/ to avoid clobbering
the existing artifacts/rl/walk-*/cost-005bps/ outputs from notebook 07.

usage: python experiments/run_all_walks.py <config.json> [walk_start] [walk_end]
       walk_start, walk_end default to 1, 16 inclusive.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback, ProgressBarCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.utils.io import repo_root
from src.utils.rl_env import PortfolioEnv

REPO_ROOT = repo_root()
OUT_ROOT = REPO_ROOT / 'artifacts' / 'rl_round2'
OUT_ROOT.mkdir(parents=True, exist_ok=True)
TOP_K = 30


def walk_windows(walk_id: int):
    """Train 2002-01-01..(2007+N-1)-12-31, val (2008+N-1)."""
    train_end = 2007 + walk_id - 1
    return ('2002-01-01', f'{train_end}-12-31',
            f'{train_end + 1}-01-01', f'{train_end + 1}-12-31')


def make_env_fn(scoreboard, cfg, seed):
    def thunk():
        env = PortfolioEnv(
            scoreboard=scoreboard, top_k=TOP_K,
            episode_length=cfg['episode_length'], cost_bps=cfg['cost_bps'],
            max_weight=cfg['max_weight'], reward_type=cfg['reward_type'],
            sharpe_window=cfg.get('sharpe_window', 8),
            action_high=cfg['action_high'],
            score_bias=cfg.get('score_bias', 0.0),
            baseline_anchor=cfg.get('baseline_anchor', False),
            tilt_scale=cfg.get('tilt_scale', 1.0),
            include_portfolio_state=cfg.get('include_portfolio_state', False),
            history_len=cfg.get('history_len', 4),
            cost_anneal_episodes=cfg.get('cost_anneal_episodes', 0),
        )
        env.reset(seed=seed)
        return Monitor(env)
    return thunk


def train_walk(walk_id: int, cfg: dict) -> dict:
    cost_bps = int(cfg['cost_bps'])
    out_dir = OUT_ROOT / f'walk-{walk_id:03d}' / f'cost-{cost_bps:03d}bps'
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'ckpts').mkdir(exist_ok=True)
    (out_dir / 'tb').mkdir(exist_ok=True)

    if (out_dir / 'final_policy.zip').exists():
        existing = json.loads((out_dir / 'training_metrics.json').read_text())
        print(f'walk {walk_id}: exists, skipping')
        return existing

    tr_s, tr_e, vl_s, vl_e = walk_windows(walk_id)
    sb_path = REPO_ROOT / 'artifacts' / 'rl' / f'walk-{walk_id:03d}' / 'scoreboard.parquet'
    sb = pd.read_parquet(sb_path)
    sb['date'] = pd.to_datetime(sb['date'])
    sb_tr = sb[(sb['date'] >= tr_s) & (sb['date'] <= tr_e)].copy()
    sb_vl = sb[(sb['date'] >= vl_s) & (sb['date'] <= vl_e)].copy()
    print(f'  walk {walk_id}: train Fri={sb_tr["date"].nunique()}, '
          f'val Fri={sb_vl["date"].nunique()}')

    seed = int(cfg.get('seed', 42))
    n_envs = cfg.get('n_envs', 4)
    train_vec = DummyVecEnv([make_env_fn(sb_tr, cfg, seed + i) for i in range(n_envs)])
    train_vec = VecNormalize(train_vec, norm_obs=True, norm_reward=False, clip_obs=10.0)
    val_vec = DummyVecEnv([make_env_fn(sb_vl, cfg, seed + 1000)])
    val_vec = VecNormalize(val_vec, norm_obs=True, norm_reward=False, clip_obs=10.0,
                           training=False)

    eval_cb = EvalCallback(val_vec, best_model_save_path=str(out_dir),
                           log_path=str(out_dir),
                           eval_freq=max(cfg.get('eval_freq', 10_000) // n_envs, 1),
                           n_eval_episodes=1, deterministic=True)
    ckpt_cb = CheckpointCallback(save_freq=max(200_000 // n_envs, 1),
                                 save_path=str(out_dir / 'ckpts'), name_prefix='ppo')
    callbacks = [eval_cb, ckpt_cb, ProgressBarCallback()]

    common = dict(
        policy='MlpPolicy', env=train_vec,
        policy_kwargs=dict(net_arch=cfg['net_arch']),
        learning_rate=cfg['learning_rate'], gamma=cfg['gamma'],
        device='cpu', verbose=0, seed=seed,
        tensorboard_log=str(out_dir / 'tb'),
    )

    algo = cfg['algo'].upper()
    if algo == 'PPO':
        model = PPO(**common,
                    n_steps=cfg.get('n_steps', 2048),
                    batch_size=cfg.get('batch_size', 64),
                    n_epochs=cfg.get('n_epochs', 5),
                    gae_lambda=cfg.get('gae_lambda', 0.95),
                    clip_range=cfg.get('clip_range', 0.15),
                    ent_coef=cfg.get('ent_coef', 0.005),
                    vf_coef=cfg.get('vf_coef', 0.5),
                    max_grad_norm=cfg.get('max_grad_norm', 0.5),
                    target_kl=cfg.get('target_kl', 0.03))
    elif algo == 'SAC':
        model = SAC(**common,
                    buffer_size=cfg.get('buffer_size', 100_000),
                    batch_size=cfg.get('batch_size', 256),
                    tau=cfg.get('tau', 0.005),
                    train_freq=cfg.get('train_freq', 1),
                    gradient_steps=cfg.get('gradient_steps', 1),
                    ent_coef=cfg.get('ent_coef', 'auto'))
    else:
        raise ValueError(f'unknown algo: {algo}')

    t0 = time.time()
    model.learn(total_timesteps=cfg['total_timesteps'],
                callback=callbacks, progress_bar=False)
    elapsed = time.time() - t0

    model.save(out_dir / 'final_policy')
    train_vec.save(str(out_dir / 'vec_normalize.pkl'))

    metrics = {
        'walk_id': walk_id, 'cost_bps': cost_bps,
        'total_timesteps': cfg['total_timesteps'], 'n_envs': n_envs,
        'wall_time_sec': elapsed, 'wall_time_hr': elapsed / 3600.0,
        'best_val_mean_reward': float(eval_cb.best_mean_reward),
    }
    (out_dir / 'training_metrics.json').write_text(json.dumps(metrics, indent=2))
    print(f'  walk {walk_id}: done; wall={elapsed/60:.1f}m; '
          f'best_val_mean_reward={metrics["best_val_mean_reward"]:.4f}')
    return metrics


def main():
    if len(sys.argv) < 2:
        print('usage: python run_all_walks.py <config.json> [walk_start] [walk_end]')
        sys.exit(1)
    cfg_path = Path(sys.argv[1])
    cfg = json.loads(cfg_path.read_text())

    walk_start = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    walk_end = int(sys.argv[3]) if len(sys.argv) > 3 else 16

    print(f'=== MULTI-WALK TRAINING: walks {walk_start}..{walk_end} ===')
    print(f'config: {cfg.get("exp_id", cfg_path.stem)}')
    print(f'output: {OUT_ROOT.relative_to(REPO_ROOT)}/')
    print()

    summary = []
    for walk_id in range(walk_start, walk_end + 1):
        print(f'\n=== walk {walk_id} ===')
        summary.append(train_walk(walk_id, cfg))

    summary_file = OUT_ROOT / f'all_walks_summary_{cfg.get("exp_id", "run")}.json'
    summary_file.write_text(json.dumps({'config': cfg, 'walks': summary}, indent=2))
    print(f'\nfinished. summary -> {summary_file.relative_to(REPO_ROOT)}')


if __name__ == '__main__':
    main()

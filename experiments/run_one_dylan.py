"""Autoresearch runner — Dylan's parallel session.

Fork of `run_one.py` with extensions:
- separates `train_cost_bps` (env reward during training) from `eval_cost_bps`
  (metrics in backtest). Lets us train an agent that internalizes a high cost
  to drive lower turnover, then score it at the real cost.
- supports the new `sharpe_turnover` reward (env extension in rl_env.py) and
  passes `turnover_lambda` through.
- logs to `results_dylan.tsv` with an extended schema; per-run dirs go under
  `experiments/runs/d_<exp_id>/`.
- prefixes exp ids with `d_` so configs and run dirs never collide with the
  parallel `run_one.py` session.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import EvalCallback, ProgressBarCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.utils.io import repo_root
from src.utils.rl_env import PortfolioEnv, project_to_simplex
from src.utils.backtest import (
    compute_strategy_metrics,
    score_proportional_weights,
)


REPO_ROOT = repo_root()
RL_DIR = REPO_ROOT / 'artifacts' / 'rl' / 'walk-001'
EXP_DIR = REPO_ROOT / 'experiments'
EXP_DIR.mkdir(exist_ok=True, parents=True)
RESULTS_TSV = EXP_DIR / 'results_dylan.tsv'

WALK_ID = 1
TRAIN_START, TRAIN_END = '2002-01-01', '2007-12-31'
VAL_START,   VAL_END   = '2008-01-01', '2008-12-31'
TEST_START,  TEST_END  = '2009-01-01', '2009-12-31'
TOP_K = 30


def _load_scoreboards():
    sb = pd.read_parquet(RL_DIR / 'scoreboard.parquet')
    sb['date'] = pd.to_datetime(sb['date'])
    sb_train = sb[(sb['date'] >= TRAIN_START) & (sb['date'] <= TRAIN_END)].copy()
    sb_val   = sb[(sb['date'] >= VAL_START)   & (sb['date'] <= VAL_END)].copy()
    sb_test  = sb[(sb['date'] >= TEST_START)  & (sb['date'] <= TEST_END)].copy()
    return sb_train, sb_val, sb_test


def _build_env(scoreboard, cfg, cost_bps, seed):
    return Monitor(PortfolioEnv(
        scoreboard=scoreboard, top_k=TOP_K,
        episode_length=cfg['episode_length'], cost_bps=cost_bps,
        max_weight=cfg['max_weight'], reward_type=cfg['reward_type'],
        sharpe_window=cfg.get('sharpe_window', 16),
        downside_lambda=cfg.get('downside_lambda', 5.0),
        action_high=cfg['action_high'],
        score_bias=cfg.get('score_bias', 0.0),
        turnover_lambda=cfg.get('turnover_lambda', 0.0),
    ))


def train_one(cfg: dict, out_dir: Path, seed: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    sb_train, sb_val, _ = _load_scoreboards()

    train_cost = cfg.get('train_cost_bps', cfg.get('cost_bps', 5.0))

    n_envs = cfg.get('n_envs', 4)
    train_vec = DummyVecEnv([(lambda s=seed + i: _build_env(sb_train, cfg, train_cost, s))
                             for i in range(n_envs)])
    train_vec = VecNormalize(train_vec, norm_obs=True, norm_reward=False, clip_obs=10.0)

    val_vec = DummyVecEnv([lambda: _build_env(sb_val, cfg, train_cost, seed + 1000)])
    val_vec = VecNormalize(val_vec, norm_obs=True, norm_reward=False, clip_obs=10.0,
                           training=False)

    eval_cb = EvalCallback(val_vec, best_model_save_path=str(out_dir),
                           log_path=str(out_dir),
                           eval_freq=max(cfg.get('eval_freq', 10_000) // n_envs, 1),
                           n_eval_episodes=1, deterministic=True)
    callbacks = [eval_cb, ProgressBarCallback()]

    algo_name = cfg['algo'].upper()
    common = dict(
        policy='MlpPolicy', env=train_vec,
        policy_kwargs=dict(net_arch=cfg['net_arch']),
        learning_rate=cfg['learning_rate'],
        gamma=cfg['gamma'],
        device='cpu', verbose=0, seed=seed,
    )

    if algo_name == 'PPO':
        model = PPO(
            **common,
            n_steps=cfg.get('n_steps', 2048),
            batch_size=cfg.get('batch_size', 64),
            n_epochs=cfg.get('n_epochs', 5),
            gae_lambda=cfg.get('gae_lambda', 0.95),
            clip_range=cfg.get('clip_range', 0.15),
            ent_coef=cfg.get('ent_coef', 0.02),
            vf_coef=cfg.get('vf_coef', 0.5),
            max_grad_norm=cfg.get('max_grad_norm', 0.5),
            target_kl=cfg.get('target_kl', 0.03),
        )
    elif algo_name == 'SAC':
        model = SAC(
            **common,
            buffer_size=cfg.get('buffer_size', 100_000),
            batch_size=cfg.get('batch_size', 256),
            tau=cfg.get('tau', 0.005),
            train_freq=cfg.get('train_freq', 1),
            gradient_steps=cfg.get('gradient_steps', 1),
            ent_coef=cfg.get('ent_coef', 'auto'),
        )
    elif algo_name == 'TD3':
        action_noise = NormalActionNoise(mean=np.zeros(TOP_K),
                                         sigma=cfg.get('action_noise', 0.1) * np.ones(TOP_K))
        model = TD3(
            **common,
            buffer_size=cfg.get('buffer_size', 100_000),
            batch_size=cfg.get('batch_size', 100),
            tau=cfg.get('tau', 0.005),
            train_freq=cfg.get('train_freq', 1),
            gradient_steps=cfg.get('gradient_steps', 1),
            action_noise=action_noise,
        )
    else:
        raise ValueError(f'unknown algo: {algo_name}')

    t0 = time.time()
    model.learn(total_timesteps=cfg['total_timesteps'],
                callback=callbacks, progress_bar=False)
    elapsed = time.time() - t0

    model.save(out_dir / 'final_policy')
    train_vec.save(str(out_dir / 'vec_normalize.pkl'))
    return {
        'wall_time_min': elapsed / 60.0,
        'best_val_mean_reward': float(eval_cb.best_mean_reward),
    }


def backtest_against_score_prop(cfg: dict, out_dir: Path, seed: int) -> dict:
    """Deterministic 2009 backtest at eval_cost_bps. Returns metrics dict."""
    _, _, sb_test = _load_scoreboards()
    by_date = {d: g.reset_index(drop=True) for d, g in sb_test.groupby('date')}
    dates = sorted(by_date.keys())

    eval_cost = cfg.get('eval_cost_bps', 5.0)

    algo_class = {'PPO': PPO, 'SAC': SAC, 'TD3': TD3}[cfg['algo'].upper()]
    model = algo_class.load(out_dir / 'best_model.zip')

    def _env_fn():
        env = PortfolioEnv(scoreboard=sb_test, top_k=TOP_K,
                           episode_length=cfg['episode_length'],
                           cost_bps=eval_cost, max_weight=cfg['max_weight'],
                           reward_type=cfg['reward_type'],
                           sharpe_window=cfg.get('sharpe_window', 16),
                           action_high=cfg['action_high'],
                           score_bias=cfg.get('score_bias', 0.0),
                           turnover_lambda=cfg.get('turnover_lambda', 0.0))
        env.reset(seed=seed)
        return Monitor(env)
    vec = DummyVecEnv([_env_fn])
    vec = VecNormalize.load(str(out_dir / 'vec_normalize.pkl'), vec)
    vec.training = False
    vec.norm_reward = False

    def _runner(weight_fn):
        prev_w = np.full(TOP_K, 1.0 / TOP_K, dtype=np.float32)
        last_ret = 0.0
        rets, turn = [], []
        for date in dates:
            cur = by_date[date]
            new_w = weight_fn(date, cur, prev_w, last_ret)
            rs = np.nan_to_num(cur['fwd_ret_5d'].to_numpy(dtype=np.float32)[:TOP_K], nan=0.0)
            r = float(np.dot(new_w, rs))
            t = float(np.abs(new_w - prev_w).sum())
            rets.append(r); turn.append(t)
            prev_w = new_w; last_ret = r
        return np.array(rets), np.array(turn)

    def _ppo_fn(date, cur, prev_w, last_ret):
        tmp = PortfolioEnv(scoreboard=sb_test, top_k=TOP_K,
                           episode_length=cfg['episode_length'],
                           cost_bps=eval_cost, max_weight=cfg['max_weight'],
                           reward_type=cfg['reward_type'],
                           sharpe_window=cfg.get('sharpe_window', 16),
                           action_high=cfg['action_high'],
                           score_bias=cfg.get('score_bias', 0.0),
                           turnover_lambda=cfg.get('turnover_lambda', 0.0))
        tmp.reset(seed=0)
        tmp._idx = tmp._dates.tolist().index(date)
        tmp._weights = np.asarray(prev_w, dtype=np.float32)
        tmp._last_return = float(last_ret)
        obs = tmp._build_obs()
        obs_norm = vec.normalize_obs(obs)
        action, _ = model.predict(obs_norm, deterministic=True)
        return project_to_simplex(np.asarray(action, dtype=np.float64),
                                  max_weight=cfg['max_weight'])

    def _sp_fn(date, cur, prev_w, last_ret):
        scores = cur['score'].to_numpy(dtype=np.float32)[:TOP_K]
        return score_proportional_weights(scores, max_weight=cfg['max_weight'])

    ppo_rets, ppo_turn = _runner(_ppo_fn)
    sp_rets,  sp_turn  = _runner(_sp_fn)

    ppo_m = compute_strategy_metrics(ppo_rets, ppo_turn, cost_bps=eval_cost)
    sp_m  = compute_strategy_metrics(sp_rets,  sp_turn,  cost_bps=eval_cost)

    return {
        'ppo_sharpe': ppo_m['sharpe'],
        'sp_sharpe':  sp_m['sharpe'],
        'excess_sharpe': ppo_m['sharpe'] - sp_m['sharpe'],
        'ppo_annret': ppo_m['annualized_return'],
        'sp_annret':  sp_m['annualized_return'],
        'ppo_maxdd':  ppo_m['max_drawdown'],
        'ppo_vol':    ppo_m['annualized_vol'],
        'ppo_turnover': ppo_m['avg_turnover'],
    }


COLS = [
    'exp_id', 'algo', 'reward_type', 'lr', 'net_arch', 'action_high',
    'max_weight', 'train_cost_bps', 'eval_cost_bps', 'episode_length',
    'sharpe_window', 'gamma', 'ent_coef', 'turnover_lambda',
    'total_timesteps', 'seed', 'wall_time_min', 'best_val_reward',
    'ppo_sharpe', 'sp_sharpe', 'excess_sharpe',
    'ppo_annret', 'sp_annret', 'ppo_maxdd', 'ppo_vol', 'ppo_turnover',
    'status', 'notes',
]


def _log_to_tsv(row: dict):
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text('\t'.join(COLS) + '\n')
    with RESULTS_TSV.open('a') as f:
        f.write('\t'.join(str(row.get(k, '')) for k in COLS) + '\n')


def main():
    if len(sys.argv) < 2:
        print('usage: python run_one_dylan.py <config.json>')
        sys.exit(1)
    cfg_path = Path(sys.argv[1])
    cfg = json.loads(cfg_path.read_text())
    exp_id = cfg.get('exp_id', cfg_path.stem)
    if not exp_id.startswith('d_'):
        exp_id = 'd_' + exp_id
    out_dir = EXP_DIR / 'runs' / exp_id

    seed = int(cfg.get('seed', 42))

    print(f'=== EXPERIMENT {exp_id} (seed={seed}) ===')
    print(f'config: {json.dumps(cfg, indent=2)}')
    print()

    train_metrics = train_one(cfg, out_dir, seed)
    bt = backtest_against_score_prop(cfg, out_dir, seed)

    summary = {
        'exp_id':         exp_id,
        'algo':           cfg['algo'],
        'reward_type':    cfg['reward_type'],
        'lr':             cfg['learning_rate'],
        'net_arch':       str(cfg['net_arch']),
        'action_high':    cfg['action_high'],
        'max_weight':     cfg['max_weight'],
        'train_cost_bps': cfg.get('train_cost_bps', cfg.get('cost_bps', 5.0)),
        'eval_cost_bps':  cfg.get('eval_cost_bps', 5.0),
        'episode_length': cfg['episode_length'],
        'sharpe_window':  cfg.get('sharpe_window', 16),
        'gamma':          cfg['gamma'],
        'ent_coef':       cfg.get('ent_coef', 0.02),
        'turnover_lambda': cfg.get('turnover_lambda', 0.0),
        'total_timesteps': cfg['total_timesteps'],
        'seed':           seed,
        'wall_time_min':  round(train_metrics['wall_time_min'], 1),
        'best_val_reward': round(train_metrics['best_val_mean_reward'], 4),
        'ppo_sharpe':     round(bt['ppo_sharpe'], 4),
        'sp_sharpe':      round(bt['sp_sharpe'], 4),
        'excess_sharpe':  round(bt['excess_sharpe'], 4),
        'ppo_annret':     round(bt['ppo_annret'], 4),
        'sp_annret':      round(bt['sp_annret'], 4),
        'ppo_maxdd':      round(bt['ppo_maxdd'], 4),
        'ppo_vol':        round(bt['ppo_vol'], 4),
        'ppo_turnover':   round(bt['ppo_turnover'], 4),
        'status':         'WIN' if bt['excess_sharpe'] > 0 else 'LOSS',
        'notes':          cfg.get('notes', ''),
    }
    _log_to_tsv(summary)

    print('--- RESULT ---')
    for k, v in summary.items():
        print(f'  {k}: {v}')
    print(f'\nlogged to {RESULTS_TSV.relative_to(REPO_ROOT)}')


if __name__ == '__main__':
    main()

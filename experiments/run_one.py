"""Autoresearch experiment runner: one PPO/SAC/TD3 training + one walk-1 backtest
vs Score-Prop. Reads HP config from sys.argv[1] (path to JSON) and prints a
compact summary at end. Logs to experiments/results.tsv (tab-separated).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Unbuffered stdout so tail -f shows progress in real time.
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import joblib
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
RESULTS_TSV = EXP_DIR / 'results.tsv'

WALK_ID = 1
TRAIN_START, TRAIN_END = '2002-01-01', '2007-12-31'
VAL_START,   VAL_END   = '2008-01-01', '2008-12-31'
TEST_START,  TEST_END  = '2009-01-01', '2009-12-31'
TOP_K = 30
RANDOM_STATE = 42


def _load_scoreboards():
    sb = pd.read_parquet(RL_DIR / 'scoreboard.parquet')
    sb['date'] = pd.to_datetime(sb['date'])
    sb_train = sb[(sb['date'] >= TRAIN_START) & (sb['date'] <= TRAIN_END)].copy()
    sb_val   = sb[(sb['date'] >= VAL_START)   & (sb['date'] <= VAL_END)].copy()
    sb_test  = sb[(sb['date'] >= TEST_START)  & (sb['date'] <= TEST_END)].copy()
    return sb_train, sb_val, sb_test


def _build_env(scoreboard, cfg, seed):
    return Monitor(PortfolioEnv(
        scoreboard=scoreboard, top_k=TOP_K,
        episode_length=cfg['episode_length'], cost_bps=cfg['cost_bps'],
        max_weight=cfg['max_weight'], reward_type=cfg['reward_type'],
        sharpe_window=cfg.get('sharpe_window', 8),
        downside_lambda=cfg.get('downside_lambda', 5.0),
        action_high=cfg['action_high'],
        score_bias=cfg.get('score_bias', 0.0),
    ))


def train_one(cfg: dict, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    sb_train, sb_val, _ = _load_scoreboards()

    n_envs = cfg.get('n_envs', 4)
    train_vec = DummyVecEnv([(lambda s=RANDOM_STATE + i: _build_env(sb_train, cfg, s))
                             for i in range(n_envs)])
    train_vec = VecNormalize(train_vec, norm_obs=True, norm_reward=False, clip_obs=10.0)

    val_vec = DummyVecEnv([lambda: _build_env(sb_val, cfg, RANDOM_STATE + 1000)])
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
        device='cpu', verbose=0, seed=RANDOM_STATE,
    )

    if algo_name == 'PPO':
        model = PPO(
            **common,
            n_steps=cfg.get('n_steps', 2048),
            batch_size=cfg.get('batch_size', 64),
            n_epochs=cfg.get('n_epochs', 5),
            gae_lambda=cfg.get('gae_lambda', 0.95),
            clip_range=cfg.get('clip_range', 0.15),
            ent_coef=cfg.get('ent_coef', 0.005),
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


def backtest_against_score_prop(cfg: dict, out_dir: Path) -> dict:
    """Run deterministic 2009 backtest: PPO vs Score-Prop. Return metrics dict."""
    _, _, sb_test = _load_scoreboards()
    by_date = {d: g.reset_index(drop=True) for d, g in sb_test.groupby('date')}
    dates = sorted(by_date.keys())

    algo_class = {'PPO': PPO, 'SAC': SAC, 'TD3': TD3}[cfg['algo'].upper()]
    model = algo_class.load(out_dir / 'best_model.zip')

    def _env_fn():
        env = PortfolioEnv(scoreboard=sb_test, top_k=TOP_K,
                           episode_length=cfg['episode_length'],
                           cost_bps=cfg['cost_bps'], max_weight=cfg['max_weight'],
                           reward_type=cfg['reward_type'], action_high=cfg['action_high'],
                           score_bias=cfg.get('score_bias', 0.0))
        env.reset(seed=RANDOM_STATE)
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
                           cost_bps=cfg['cost_bps'], max_weight=cfg['max_weight'],
                           reward_type=cfg['reward_type'], action_high=cfg['action_high'],
                           score_bias=cfg.get('score_bias', 0.0))
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

    ppo_m = compute_strategy_metrics(ppo_rets, ppo_turn, cost_bps=cfg['cost_bps'])
    sp_m  = compute_strategy_metrics(sp_rets,  sp_turn,  cost_bps=cfg['cost_bps'])

    return {
        'ppo_sharpe': ppo_m['sharpe'],
        'sp_sharpe':  sp_m['sharpe'],
        'excess_sharpe': ppo_m['sharpe'] - sp_m['sharpe'],
        'ppo_annret': ppo_m['annualized_return'],
        'sp_annret':  sp_m['annualized_return'],
        'ppo_maxdd':  ppo_m['max_drawdown'],
        'sp_maxdd':   sp_m['max_drawdown'],
        'ppo_vol':    ppo_m['annualized_vol'],
        'ppo_turnover': ppo_m['avg_turnover'],
    }


def _log_to_tsv(row: dict):
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text('\t'.join(row.keys()) + '\n')
    with RESULTS_TSV.open('a') as f:
        f.write('\t'.join(str(v) for v in row.values()) + '\n')


def main():
    if len(sys.argv) < 2:
        print('usage: python run_one.py <config.json>')
        sys.exit(1)
    cfg_path = Path(sys.argv[1])
    cfg = json.loads(cfg_path.read_text())
    exp_id = cfg.get('exp_id', cfg_path.stem)
    out_dir = EXP_DIR / 'runs' / exp_id

    print(f'=== EXPERIMENT {exp_id} ===')
    print(f'config: {json.dumps(cfg, indent=2)}')
    print()

    train_metrics = train_one(cfg, out_dir)
    bt = backtest_against_score_prop(cfg, out_dir)

    summary = {
        'exp_id': exp_id,
        'algo': cfg['algo'],
        'reward_type': cfg['reward_type'],
        'lr': cfg['learning_rate'],
        'net_arch': str(cfg['net_arch']),
        'action_high': cfg['action_high'],
        'total_timesteps': cfg['total_timesteps'],
        'wall_time_min': round(train_metrics['wall_time_min'], 1),
        'best_val_reward': round(train_metrics['best_val_mean_reward'], 4),
        'ppo_sharpe': round(bt['ppo_sharpe'], 4),
        'sp_sharpe': round(bt['sp_sharpe'], 4),
        'excess_sharpe': round(bt['excess_sharpe'], 4),
        'ppo_annret': round(bt['ppo_annret'], 4),
        'sp_annret': round(bt['sp_annret'], 4),
        'ppo_maxdd': round(bt['ppo_maxdd'], 4),
        'ppo_vol': round(bt['ppo_vol'], 4),
        'ppo_turnover': round(bt['ppo_turnover'], 4),
        'status': 'WIN' if bt['excess_sharpe'] > 0 else 'LOSS',
    }
    _log_to_tsv(summary)

    print('--- RESULT ---')
    for k, v in summary.items():
        print(f'  {k}: {v}')
    print(f'\nlogged to {RESULTS_TSV.relative_to(REPO_ROOT)}')


if __name__ == '__main__':
    main()

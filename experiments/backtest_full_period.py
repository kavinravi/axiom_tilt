"""Full-period (2009-2024) concatenated backtest for an autoresearch winner.

For each walk N in 1..16:
  - load test-year scoreboard (artifacts/rl/walk-N/scoreboard.parquet)
  - slice to test year (2009 + N - 1)
  - load trained PPO from artifacts/rl_round2/walk-N/cost-005bps/best_model.zip
  - deterministic playthrough -> weekly returns + turnover
  - run Score-Prop on the same dates as control

Concatenate per-walk returns into one 16-year series per strategy. Compute
full-period Sharpe / Sortino / MDD / Calmar / hit-rate / turnover (same
metrics as notebook 08 cell H, so results are directly comparable).

usage: python experiments/backtest_full_period.py <config.json>
       (config used at training time — needs episode_length, baseline_anchor,
        tilt_scale, cost_bps, action_high)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.utils.io import repo_root
from src.utils.rl_env import PortfolioEnv, project_to_simplex
from src.utils.backtest import (
    compute_strategy_metrics,
    equal_weight_weights,
    score_proportional_weights,
)

REPO_ROOT = repo_root()
TRAINING_ROOT = REPO_ROOT / 'artifacts' / 'rl_round2'
SCOREBOARD_ROOT = REPO_ROOT / 'artifacts' / 'rl'  # scoreboards still live here
OUT_DIR = REPO_ROOT / 'artifacts' / 'backtest_round2'
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = 30
RANDOM_STATE = 42


def env_kwargs_from_cfg(cfg: dict, scoreboard):
    """Build PortfolioEnv constructor kwargs from a training config (so that
    backtest-time obs matches training-time obs exactly)."""
    return dict(
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
        # cost_anneal intentionally omitted: backtest always at full cost.
    )


def make_ppo_weights_fn(cfg, walk_id, sb_test):
    """Returns a `weights_fn(date, cur, prev_w, last_ret)` for the PPO model."""
    cost_bps = int(cfg['cost_bps'])
    cv_dir = TRAINING_ROOT / f'walk-{walk_id:03d}' / f'cost-{cost_bps:03d}bps'
    algo_cls = {'PPO': PPO, 'SAC': SAC}[cfg['algo'].upper()]
    model = algo_cls.load(cv_dir / 'best_model.zip')

    env_kwargs = env_kwargs_from_cfg(cfg, sb_test)

    def _env_fn():
        env = PortfolioEnv(**env_kwargs)
        env.reset(seed=RANDOM_STATE)
        return Monitor(env)
    vec = DummyVecEnv([_env_fn])
    vec = VecNormalize.load(str(cv_dir / 'vec_normalize.pkl'), vec)
    vec.training = False
    vec.norm_reward = False

    def _ppo_fn(date, cur, prev_w, last_ret):
        tmp = PortfolioEnv(**env_kwargs)
        tmp.reset(seed=0)
        tmp._idx = tmp._dates.tolist().index(date)
        tmp._weights = np.asarray(prev_w, dtype=np.float32)
        tmp._last_return = float(last_ret)
        obs = tmp._build_obs()
        obs_norm = vec.normalize_obs(obs)
        action, _ = model.predict(obs_norm, deterministic=True)
        # Replicate training-time action -> weights mapping (tilt if enabled).
        if cfg.get('baseline_anchor', False):
            raw_scores = cur['score'].to_numpy(dtype=np.float32)[:TOP_K]
            baseline_w = project_to_simplex(raw_scores, max_weight=cfg['max_weight'])
            tilted = (np.log(baseline_w + 1e-8)
                      + cfg.get('tilt_scale', 1.0) * np.asarray(action, dtype=np.float32))
            return project_to_simplex(np.asarray(tilted, dtype=np.float64),
                                      max_weight=cfg['max_weight'])
        return project_to_simplex(np.asarray(action, dtype=np.float64),
                                  max_weight=cfg['max_weight'])

    return _ppo_fn


def run_strategy(test_dates, by_test, weights_fn):
    prev_w = equal_weight_weights(TOP_K)
    last_ret = 0.0
    rets, turn = [], []
    for date in test_dates:
        cur = by_test[date]
        new_w = weights_fn(date, cur, prev_w, last_ret)
        rs = np.nan_to_num(cur['fwd_ret_5d'].to_numpy(dtype=np.float32)[:TOP_K], nan=0.0)
        r = float(np.dot(new_w, rs))
        t = float(np.abs(new_w - prev_w).sum())
        rets.append(r); turn.append(t)
        prev_w = new_w
        last_ret = r
    return np.array(rets), np.array(turn)


def main():
    if len(sys.argv) < 2:
        print('usage: python backtest_full_period.py <config.json>')
        sys.exit(1)
    cfg = json.loads(Path(sys.argv[1]).read_text())
    exp_id = cfg.get('exp_id', Path(sys.argv[1]).stem)
    cost_bps = int(cfg['cost_bps'])

    print(f'=== FULL-PERIOD BACKTEST: {exp_id} ===')
    print(f'training root: {TRAINING_ROOT.relative_to(REPO_ROOT)}/')
    print(f'output: {OUT_DIR.relative_to(REPO_ROOT)}/')
    print()

    ppo_all_rets, ppo_all_turn, ppo_dates = [], [], []
    sp_all_rets,  sp_all_turn,  sp_dates  = [], [], []

    for walk_id in range(1, 17):
        test_year = 2009 + walk_id - 1
        sb_path = SCOREBOARD_ROOT / f'walk-{walk_id:03d}' / 'scoreboard.parquet'
        if not sb_path.exists():
            print(f'walk {walk_id}: no scoreboard, skipping')
            continue
        sb = pd.read_parquet(sb_path)
        sb['date'] = pd.to_datetime(sb['date'])
        sb_test = sb[(sb['date'] >= f'{test_year}-01-01') &
                     (sb['date'] <= f'{test_year}-12-31')].copy().reset_index(drop=True)
        if len(sb_test) == 0:
            print(f'walk {walk_id}: no rows in {test_year}, skipping')
            continue
        by_test = {d: g.reset_index(drop=True) for d, g in sb_test.groupby('date')}
        test_dates = sorted(by_test.keys())

        model_path = (TRAINING_ROOT / f'walk-{walk_id:03d}' /
                      f'cost-{cost_bps:03d}bps' / 'best_model.zip')
        if not model_path.exists():
            print(f'walk {walk_id}: no PPO model at {model_path.relative_to(REPO_ROOT)}, '
                  'skipping')
            continue

        ppo_fn = make_ppo_weights_fn(cfg, walk_id, sb_test)
        ppo_r, ppo_t = run_strategy(test_dates, by_test, ppo_fn)

        def _sp_fn(date, cur, prev_w, last_ret):
            scores = cur['score'].to_numpy(dtype=np.float32)[:TOP_K]
            return score_proportional_weights(scores, max_weight=cfg['max_weight'])
        sp_r, sp_t = run_strategy(test_dates, by_test, _sp_fn)

        ppo_all_rets.extend(ppo_r); ppo_all_turn.extend(ppo_t); ppo_dates.extend(test_dates)
        sp_all_rets.extend(sp_r);   sp_all_turn.extend(sp_t);   sp_dates.extend(test_dates)
        print(f'walk {walk_id} ({test_year}): {len(test_dates)} Fri, '
              f'PPO ann={(1+np.mean(ppo_r))**52-1:.3f}, SP ann={(1+np.mean(sp_r))**52-1:.3f}')

    ppo_rets = np.array(ppo_all_rets)
    ppo_turn_arr = np.array(ppo_all_turn)
    sp_rets = np.array(sp_all_rets)
    sp_turn_arr = np.array(sp_all_turn)

    ppo_m = compute_strategy_metrics(ppo_rets, ppo_turn_arr, cost_bps=cost_bps)
    sp_m  = compute_strategy_metrics(sp_rets,  sp_turn_arr,  cost_bps=cost_bps)

    print('\n=== FULL-PERIOD (2009-2024) ===')
    print(f"{'metric':<22s} {'PPO':>12s} {'Score-Prop':>12s} {'PPO-SP':>10s}")
    for k in ['total_return_net', 'annualized_return', 'annualized_vol',
              'sharpe', 'sortino', 'max_drawdown', 'calmar', 'hit_rate', 'avg_turnover']:
        d = ppo_m[k] - sp_m[k] if k not in ('max_drawdown', 'avg_turnover') else None
        d_str = f'{d:+.4f}' if d is not None else ''
        print(f'  {k:<20s} {ppo_m[k]:>12.4f} {sp_m[k]:>12.4f} {d_str:>10s}')
    print(f'\n  n_weeks: {len(ppo_rets)}')

    # Persist.
    out_summary = {
        'exp_id': exp_id,
        'config': cfg,
        'n_weeks': int(len(ppo_rets)),
        'PPO': {k: float(v) for k, v in ppo_m.items() if not isinstance(v, str)},
        'Score-Prop': {k: float(v) for k, v in sp_m.items() if not isinstance(v, str)},
    }
    (OUT_DIR / f'summary_{exp_id}.json').write_text(json.dumps(out_summary, indent=2))
    pd.DataFrame({
        'date': ppo_dates,
        'ppo_return_gross': ppo_rets,
        'ppo_turnover': ppo_turn_arr,
        'sp_return_gross': sp_rets,
        'sp_turnover': sp_turn_arr,
    }).to_parquet(OUT_DIR / f'weekly_{exp_id}.parquet', compression='zstd', index=False)
    print(f"\nwrote -> {(OUT_DIR / f'summary_{exp_id}.json').relative_to(REPO_ROOT)}")


if __name__ == '__main__':
    main()

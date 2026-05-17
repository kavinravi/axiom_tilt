# RL Agent (Notebook 07) — Design

**Date:** 2026-05-17
**Parent spec:** [`2026-05-08-text-enhanced-rl-portfolio-design.md`](2026-05-08-text-enhanced-rl-portfolio-design.md) §9-10
**Backtest:** notebook 08 (separate, this spec only covers env + training)
**Status:** Design

## 1. Goal

Train a PPO policy that allocates weights across the walk-1 ranker's top-30
candidate set each week. Walk-1 only (train 2002-2007, val 2008, holdout for
notebook 08 backtest: 2009). Parameterized over transaction cost (5/2/10/20 bps)
so one overnight run produces all four cost-variant policies for the paper.

Outputs land in `artifacts/rl/walk-001/cost-{bps:03d}bps/` per cost variant.

## 2. Conventions

Per the `stable-baselines3` skill: gymnasium-compatible env, `check_env()`
validation before training, `EvalCallback` to save best-by-val, `CheckpointCallback`
for rolling backups, TensorBoard logging, `VecNormalize` for observation
standardization.

## 3. Inputs

| Source | Path | Notes |
|---|---|---|
| Training panel | `data/processed/training_panel/year=YYYY/part-0.parquet` | 2002-2009 used here |
| Walk-1 ranker | `artifacts/ranker/walk-001/model.joblib` | LightGBM lambdarank, ~199 features |
| Walk-1 PCA | `artifacts/pca-text/walk-001/pca.joblib` | n_pca=79 |
| Stock-day embed | `data/processed/finbert_stockday_embed/year=YYYY/*.parquet` | for projecting text features |

## 4. Environment design

### 4.1 `PortfolioEnv` (gymnasium.Env)

Walk-1 only. Operates on a precomputed per-Friday scoreboard (top-30 by walk-1
ranker for each Friday in 2002-2008). At `reset`, picks a random Friday in the
training window as the episode start, initializes equal-weight over that
Friday's top-30, returns the state.

At `step(action)`:
1. Project `action` to long-only simplex (see §4.3).
2. Compute realized 5-day forward return per current top-30 stock (from `fwd_ret_5d`).
3. Compute trade amount = `|new_weights - prev_weights|.sum()`.
4. `reward = portfolio_return - (cost_bps / 10_000) * trade_amount`.
5. Advance to next Friday; refresh top-30 (may change). Carry over weights for
   stocks still in top-30; redistribute weights of dropped stocks pro-rata to
   the new entrants (forced trade, naturally penalized via the cost term next step).
6. `terminated = True` after 52 Fridays (1-year episodes).

### 4.2 Observation space (≈215 dim)

`Box(shape=(214,), dtype=float32, -inf, inf)`:

| Block | Dim | Description |
|---|---|---|
| Current portfolio weights | 30 | sums to 1, post-projection |
| Ranker scores (current top-30) | 30 | normalized to z-scores within the date for scale invariance |
| Top-5 features × 30 stocks | 150 | features by walk-1 gain importance: `payoutratio`, `ncfdiv`, `bidlo`, `sgna`, `retearn`; per-row z-scored within date |
| Macro | 3 | `macro_vixcls`, `macro_dgs10`, `macro_t10y2y` |
| Recent portfolio 1-week return | 1 | realized last-step return |

NaN handling: `np.nan_to_num(state, nan=0.0)` after assembly. (LightGBM tolerated
NaN; PPO/MLP can't.)

### 4.3 Action space

`Box(shape=(30,), dtype=float32, -inf, inf)`. Raw action is unbounded; we
project to weights via:
```
weights = softmax(action)           # long-only, sums to 1
weights = clip(weights, max=0.10)   # per-name max 10%
weights = weights / weights.sum()   # renormalize
```

Per spec §10, constraint layer is enforcement, not re-optimization. RL retains
decision authority over the feasible region.

### 4.4 Reward

```
reward = portfolio_return - (cost_bps / 10_000) * trade_amount
```

Where:
- `portfolio_return = (new_weights * fwd_ret_5d_top30).sum()` for the current week
- `trade_amount = |new_weights - prev_weights|.sum()` (gross turnover, L1)

For MVP, no volatility or drawdown penalty. Those are mentioned in spec §9.3
but listed as ablations for v2.

### 4.5 Episode structure

- Length: 52 Fridays (1 calendar year of weekly rebalances)
- Reset: random start Friday from the training window
- For the val env (used by `EvalCallback`), reset starts always at first Friday
  of 2008 — deterministic evaluation.

## 5. Training scheme

### 5.1 PPO config

```python
PPO(
    'MlpPolicy',
    vec_env,
    policy_kwargs=dict(net_arch=[256, 128]),  # > obs dim to avoid compression
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    device='cpu',          # see policy size justification in §5.2
    tensorboard_log=OUT_DIR / 'tb',
)
```

### 5.2 Why CPU, not GPU

Policy has ~85k params, env step is data-manipulation-heavy. Per-step inference
is ~50μs CPU, with GPU adding host↔device transfer overhead per step. Net loss
for tiny MLP + small batches. GPU only helps when the policy net is significantly
larger or the env outputs images. CPU stays.

### 5.3 Vec env

`DummyVecEnv` with `n_envs=4`. Lightweight, in-process, no subprocess overhead.
Each env steps independently, PPO accumulates rollouts across all four.

### 5.4 Callbacks

- `EvalCallback(val_env, eval_freq=10_000, n_eval_episodes=1, deterministic=True)`
  → saves best policy in `OUT_DIR / 'best_policy'`.
- `CheckpointCallback(save_freq=200_000, save_path=OUT_DIR / 'ckpts')`.
- `ProgressBarCallback()` — wall-clock progress in the notebook.

### 5.5 Budget

`total_timesteps = 2_000_000` per cost variant. At ~1-5k steps/sec on CPU with
n_envs=4, that's ~2-6 hours per variant. With 4 cost variants (5, 2, 10, 20 bps),
total ~10-24 hours. Fits the 18-hour overnight window with margin if some come
in fast.

### 5.6 Observation normalization

`VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)`. Saved
to `OUT_DIR / 'vec_normalize.pkl'` so notebook 08 backtest can apply the same
normalization to 2009 observations.

## 6. Cost-variant loop

```python
COSTS_BPS = [5, 2, 10, 20]  # primary first

for cost_bps in COSTS_BPS:
    out_dir = ARTIFACTS_ROOT / 'rl' / 'walk-001' / f'cost-{cost_bps:03d}bps'
    if (out_dir / 'final_policy.zip').exists():
        print(f'cost-{cost_bps}bps: exists — skipping')
        continue
    train_one(cost_bps, out_dir)
```

`train_one` builds env, normalizes, sets up callbacks, runs PPO, saves all
artifacts to `out_dir`.

## 7. Outputs per cost variant

Per `OUT_DIR = artifacts/rl/walk-001/cost-{bps:03d}bps/`:

| File / dir | Content |
|---|---|
| `final_policy.zip` | SB3-saved policy after `total_timesteps` |
| `best_policy.zip` | Best policy by val Sharpe (EvalCallback) |
| `vec_normalize.pkl` | Observation normalization stats |
| `training_metrics.json` | Final mean reward, best val mean reward, wall time, n_timesteps |
| `tb/` | TensorBoard logs (reward, value loss, entropy, KL) |
| `ckpts/` | Rolling checkpoints every 200k steps |

`artifacts/rl/walk-001/all_costs_summary.json` consolidates per-variant
training_metrics.json for the paper.

## 8. Validation gates

- `check_env(env, warn=True)` before training (catches space/spec mismatches).
- `total_timesteps > 0` and env produces non-NaN states.
- Sanity: best val episode reward > -1.0 (any policy clearly worse than that
  is broken). True quality assessed in notebook 08's backtest, not here.

## 9. File / module structure

```
src/utils/rl_env.py         # PortfolioEnv + helpers (TDD'd)
tests/utils/test_rl_env.py  # env tests + check_env smoke test
notebooks/07_rl_agent.ipynb
artifacts/rl/walk-001/      # gitignored (large); only summary jsons force-added
```

`src/utils/rl_env.py` exports:
- `PortfolioEnv(scoreboard, cost_bps, ...)` — main env class
- `project_to_simplex(action, max_weight=0.10) -> np.ndarray` — pure helper
- `build_scoreboard(walk_id, panel_dir, embed_dir, start, end) -> pd.DataFrame`
  — precompute per-Friday top-30 with required features, called once at env setup

Notebook 07 cells:
- A. Setup (paths, constants, COSTS_BPS list)
- B. Build scoreboard (once for train window 2002-2008)
- C. Helper: `train_one(cost_bps, out_dir)`
- D. Loop over `COSTS_BPS`
- E. Cross-cost-variant diagnostics (table + reward curve plot)

## 10. Risks / mitigations

- **Per-Friday top-30 mutation**: stocks drop out of top-30 forcing redistribution.
  Mitigated by Jaccard 0.76 — only ~7 names churn per week. Pro-rata redistribution
  to new entrants is simple and naturally costed.
- **Episode length vs convergence**: 52 Fridays is short. If PPO oscillates,
  could extend to multi-year episodes. Defer; not blocking.
- **CPU throughput**: if 2M timesteps takes > 6h per variant, we cut the slow
  variants (10/20 bps) and report only 5/2 in the paper.
- **VecNormalize gotcha**: stats are stateful. Must save AND load `vec_normalize.pkl`
  in notebook 08 or backtest results will silently differ from training-time observations.

## 11. Out of scope (deferred)

- Walks 2-16 (single-walk MVP only)
- Volatility / drawdown reward penalties (spec §9.3)
- Behavior cloning warm-start (spec §9.4 v2)
- Bootstrap-augmented training trajectories (spec §9.4 v2)
- ADV liquidity cap (spec §10) — backtest reports turnover but doesn't cap it
- Sector-level constraints
- Larger policy + GPU (optional ablation if time remains)

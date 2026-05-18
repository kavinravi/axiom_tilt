# Backtest (Notebook 08) — Design

**Date:** 2026-05-17
**Parent spec:** [`2026-05-08-text-enhanced-rl-portfolio-design.md`](2026-05-08-text-enhanced-rl-portfolio-design.md) §11, §17.5
**Scope:** Walk-1 test window (2009) deterministic backtest of all trained policies + baselines
**Status:** Design

## 1. Goal

Apples-to-apples comparison of 8 portfolio-selection strategies on the walk-1
test window (Jan 1 - Dec 31, 2009), under a standardized 5 bps round-trip
execution cost. Output the table and charts the paper needs to make its case.

## 2. Strategies compared

| # | Name | Description |
|---|------|-------------|
| 1 | EW-Top30 | Equal-weight on walk-1 ranker top-30 |
| 2 | Score-Prop | Weight ∝ ranker score (softmax-normalized over top-30) |
| 3 | MinVar | Minimum-variance weights on top-30 (26-week trailing cov, cvxpy QP, max-weight 10%) |
| 4 | 60/40 (proxy) | 60% EW-Top30 + 40% bond carry (DGS10/52 weekly approximation) |
| 5-8 | PPO-{5,2,10,20}bps | Best-by-val PPO policies from notebook 07 |

All evaluated at the same 5 bps execution cost. The cost-bps variants of PPO
test "which TRAINING cost produces a policy that performs best at 5bps
execution" — a cost-robustness check.

## 3. Inputs

| Source | Path |
|---|---|
| Scoreboard | `artifacts/rl/walk-001/scoreboard.parquet` (already includes 2009 from notebook 07) |
| PPO policies | `artifacts/rl/walk-001/cost-{XXX}bps/best_model.zip` |
| Vec normalizers | `artifacts/rl/walk-001/cost-{XXX}bps/vec_normalize.pkl` |

## 4. Execution

**Deterministic 52-Friday playthrough** of 2009. Each strategy:
1. At each Friday: compute weights (per strategy logic).
2. Apply weights to `fwd_ret_5d` of the top-30 → portfolio return for the week.
3. Compute trade amount = `|new_weights - prev_weights|.sum()`.
4. Net return = portfolio_return - `0.0005 * trade_amount`.

For the 60/40 proxy: bond return = `macro_dgs10[date] / 5200` (weekly carry from
annual yield). Combined as `0.6 * equity_return + 0.4 * bond_return`. Bond
side has no trading cost; equity side incurs cost on the 60% portion only.

## 5. Metrics (per spec §17.5)

| Metric | Formula |
|---|---|
| Total return (gross / net) | Cumulative compounded weekly returns |
| Annualized return | `(1 + total)^(52/n_weeks) - 1` |
| Annualized vol | `std(weekly_returns) * sqrt(52)` |
| Sharpe (rf=0) | `annualized_return / annualized_vol` |
| Sortino (rf=0) | `annualized_return / annualized_downside_std` |
| Max drawdown | `min(equity / running_max - 1)` |
| Calmar | `annualized_return / |max_dd|` |
| Hit rate | `% weeks with positive net return` |
| Avg turnover | `mean(weekly trade_amount)` |
| Net excess vs EW-Top30 | `annualized_net_return - annualized_EW_return` (paper's headline number) |

## 6. Outputs

`artifacts/backtest/walk-001/`:

| File | Content |
|---|---|
| `summary_table.csv` | One row per strategy, all metrics |
| `per_strategy_metrics.json` | Same data, nested JSON for paper figure regeneration |
| `equity_curves.png` | Cumulative net return per strategy over 2009 |
| `drawdown.png` | Drawdown curves per strategy |
| `sharpe_bars.png` | Bar chart of Sharpe across strategies |
| `weekly_returns.parquet` | Per-strategy per-week net return + turnover (for reproducibility) |

## 7. File / module structure

```
src/utils/backtest.py        # pure helpers (TDD'd)
tests/utils/test_backtest.py # unit tests
notebooks/08_backtest.ipynb
artifacts/backtest/walk-001/ # gitignored except summary_table.csv (force-add)
```

`src/utils/backtest.py` exports:
- `equal_weight_weights(k: int) -> np.ndarray`
- `score_proportional_weights(scores: np.ndarray) -> np.ndarray` — softmax normalization
- `min_variance_weights(returns_history: np.ndarray, max_weight: float = 0.10) -> np.ndarray` — cvxpy QP
- `compute_strategy_metrics(weekly_returns, weekly_turnover, cost_bps=5.0) -> dict` — full metrics dict
- `run_strategy_episode(scoreboard, weights_fn, cost_bps=5.0) -> dict` — deterministic loop

## 8. Notebook 08 cells

- A. Setup (paths, constants, load scoreboard, load PPO policies)
- B. Helper: `run_one_strategy(name, weights_fn)` — wraps the deterministic loop
- C. Run all 8 strategies, collect per-week return + turnover
- D. Compute metrics table; print formatted comparison
- E. Plot equity curves
- F. Plot drawdown curves
- G. Plot Sharpe bar chart
- H. Persist all artifacts

## 9. Risks / caveats

- **60/40 is a proxy**, not a real bond-index comparison. Bond side uses yield
  carry; assumes constant yield (no duration math). Acceptable for MVP but
  note in paper.
- **MinVar can be ill-conditioned** when 26-week history is short or covariance
  is near-singular. Mitigation: ridge-regularize Σ + ε·I before solving.
- **VecNormalize stats are stateful** — must reload `vec_normalize.pkl` per
  PPO variant or backtest results will silently differ from training-time.
- **Sample of one year** — 2009 was a recovery year, so results may look better
  than long-run expected. Acknowledge in paper.

## 10. Out of scope

- Multi-walk concatenated backtest (walks 2-16). Would require running the
  ranker + PPO loops for those walks; deferred.
- Behavior cloning / bootstrap-augmented PPO comparisons (spec §9.4 v2).
- Sector-attribution analysis.
- Statistical significance testing on strategy differences.

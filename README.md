# Text-Enhanced RL Portfolio Allocation

This repository is for a portfolio-learning project centered on a practical question:

> Can financial text improve weekly equity allocation when reinforcement learning is responsible for the actual trading decision under transaction costs and portfolio constraints?

The design is intentionally split into two stages:
- a text + tabular prediction stack ranks a broad universe of stocks,
- an RL agent allocates among the top-ranked names over time.

That keeps the text modeling and cross-sectional prediction realistic, while giving RL a real sequential control problem instead of a decorative optimizer-tuning role.

## Core Idea

The project is not "RL over all 400 stock weights."

It is:

1. collect and encode financial documents with FinBERT,
2. combine text features with classic market and fundamental signals,
3. train a supervised model to rank the full universe each week,
4. keep the top `K` names,
5. let RL choose weights or trade deltas only within that reduced candidate set,
6. enforce portfolio constraints before execution,
7. evaluate the strategy after costs.

This gives each part of the system a distinct job:
- `FinBERT`: extract financial-text signal,
- supervised model: narrow the universe to the most promising names,
- `RL`: manage dynamic allocation and trading under frictions.

## Why This Framing

There are two bad extremes:

1. RL directly outputs weights for 400 names.
2. RL only chooses a few abstract optimizer knobs while the real portfolio logic lives elsewhere.

The first is too high-dimensional and unstable. The second makes RL feel bolted on.

This project uses a middle ground where RL still gets real spotlight:
- the supervised model handles universe-wide ranking,
- RL handles the sequential portfolio decision on a manageable subset.

## Research Question

The full research question is:

> Can fine-tuned financial-text representations improve stock selection enough that an RL agent can convert them into better after-cost portfolio performance than non-text and non-RL baselines?

That means the project should answer both:
- does financial text improve ranking quality,
- and does RL improve dynamic allocation among top-ranked candidates once trading costs and path dependence matter.

## End-to-End Pipeline

1. Ingest weekly financial documents, price data, and fundamentals.
2. Fine-tune or adapt FinBERT on financial text.
3. Generate stock-week text embeddings from the `[CLS]` token or pooled document representation.
4. Merge text features with typical structured signals.
5. Train a supervised ranker over the full 400-name universe.
6. Select the top `K` names each week.
7. Feed the reduced candidate set and portfolio state into an RL agent.
8. Have RL output target weights or trade deltas.
9. Project the action into the feasible set or pass it through a constrained portfolio layer.
10. Compute after-cost reward and update the policy through time.

## System Design

### 1. Data and Timing

Build a strict `ticker-week` panel with no lookahead.

Recommended defaults:
- Universe: roughly 400 liquid U.S. equities, or an S&P 500-style survivorship-aware subset.
- Rebalance frequency: weekly.
- Holding horizon: one week with rolling portfolio state.
- Evaluation: walk-forward or expanding-window backtest.

At time `t`, the system should only use information available by `t`:
- market data,
- fundamentals,
- financial documents released before the rebalance decision.

### 2. Financial Text Modeling

Use financial documents such as:
- 10-K and 10-Q filings,
- 8-K filings,
- earnings call transcripts,
- press releases,
- optional curated news.

The text workflow is:

1. fetch and clean documents by week,
2. tokenize with FinBERT,
3. use the `[CLS]` embedding or pooled representation as the document feature,
4. aggregate multiple documents into one stock-week text vector.

Recommended scope:
- start with pretrained FinBERT as a baseline,
- add fine-tuning on financial documents as the main text-model contribution,
- treat more advanced contrastive objectives as an extension if time permits.

### 3. Structured Features

Merge the text embeddings with typical signals such as:
- momentum,
- short-term reversal,
- realized volatility,
- beta or risk estimates,
- liquidity measures,
- valuation,
- profitability,
- leverage,
- optional sentiment features.

These features should be standardized cross-sectionally by week and checked for leakage, missingness, and outliers.

### 4. Supervised Ranking Model

The supervised model ranks the full universe before RL gets involved.

Its job is to answer:
- which names look strongest this week based on text + structured features?

Recommended first models:
- LightGBM,
- XGBoost,
- or another strong tabular learner.

Useful targets:
- next-week excess return,
- return scaled by recent volatility,
- cross-sectional rank,
- or top-vs-bottom bucket labels.

Useful diagnostics:
- rank IC,
- decile spread,
- hit rate,
- calibration,
- top-`K` selection stability.

This stage is critical because it reduces the RL problem from "allocate across 400 names" to "allocate across the most promising candidates."

### 5. Candidate Selection

Each week, keep the top `K` ranked stocks from the full universe.

Pick one main value of `K` for the core experiment, such as:
- `K = 30`, or
- `K = 40`.

This should not vary casually across runs. It is part of the formal design.

The selected set becomes the action universe for RL that week.

### 6. RL Portfolio Control

This is the center of the project.

RL should observe:
- current candidate scores,
- candidate-level features or compressed embeddings,
- current portfolio weights,
- recent portfolio returns,
- recent turnover,
- volatility or market-regime indicators,
- estimated transaction costs.

RL then outputs one of:
- target weights over the top `K` names, or
- trade deltas relative to the current portfolio.

Recommended default:
- start with target weights because the formulation is cleaner,
- keep trade deltas as a realism extension if time allows.

This makes RL responsible for the actual sequential portfolio decision rather than just a control overlay.

### 7. Constraint Handling

The RL action should not be executed raw.

Instead, map it into a feasible portfolio using:
- simplex projection,
- softmax-based long-only normalization,
- clipping and renormalization,
- or a constrained portfolio layer.

Representative constraints:
- long-only,
- fully invested,
- max weight per name,
- turnover penalty or cap,
- optional liquidity screen.

The important design principle is that the constraint layer should enforce feasibility without taking the core allocation decision away from RL.

### 8. Reward and Backtest

The reward should reflect realized portfolio performance after costs.

A simple version is:

`reward_t = portfolio_return_t - trading_cost_t`

A stronger version is:

`reward_t = portfolio_return_t - trading_cost_t - lambda_vol * volatility_penalty_t - lambda_dd * drawdown_penalty_t`

Core evaluation metrics:
- annualized return,
- annualized volatility,
- Sharpe ratio,
- Sortino ratio,
- max drawdown,
- turnover,
- average number of holdings,
- cost-adjusted return.

The backtest should use:
- weekly rebalancing,
- realistic execution lag,
- transaction costs every rebalance,
- walk-forward retraining for the ranker,
- out-of-sample evaluation for the RL policy.

## Why RL Is Central Here

In this design, RL is not just choosing a regime label or risk-aversion scalar.

RL owns:
- dynamic reallocation among the selected names,
- adaptation to current holdings,
- adaptation to recent returns and costs,
- sequential tradeoffs between reward and turnover.

That means the project is legitimately about RL-based portfolio control, with text-enhanced ranking as the upstream signal engine.

## Minimum Viable Version

If scope needs to stay tight, build this first:

1. Gather weekly financial documents and structured data.
2. Generate FinBERT embeddings.
3. Merge with quant and sentiment features.
4. Train a supervised ranker on the full universe.
5. Select top `K = 30` or similar each week.
6. Train one RL agent to output long-only target weights on that reduced set.
7. Enforce max-position and turnover constraints with a simple projection layer.
8. Backtest after linear transaction costs.
9. Compare against non-RL allocation baselines.

This version is already a coherent project where RL has real ownership.

## Baselines and Ablations

Required baselines:
- equal-weight over the full universe,
- equal-weight over top `K`,
- supervised ranker plus static weighting rule,
- supervised ranker plus mean-variance or constrained optimizer,
- no-text version of the ranker,
- full text + ranker + RL pipeline.

Useful ablations:
- pretrained FinBERT vs fine-tuned FinBERT,
- remove text features,
- change `K`,
- RL target weights vs simple equal-weight-on-top-`K`,
- with and without transaction costs,
- with and without volatility or drawdown penalties.

The point is to show both:
- whether text improves candidate selection,
- and whether RL improves allocation among those candidates.

## Interpretability

A strong final result should explain:
- what the text model contributes to ranking quality,
- how stable the top-`K` set is over time,
- how the RL policy changes exposures across regimes,
- how much turnover RL creates,
- whether gains survive realistic costs.

Suggested diagnostics:
- feature importance for the ranker,
- performance of top-`K` candidates before RL allocation,
- action or weight concentration over time,
- turnover decomposition,
- sector and factor exposure plots,
- cumulative performance with and without costs.

## Suggested Repo Workflow

The current source tree maps naturally onto this design:
- `src/data`: document, price, and fundamentals ingestion and alignment
- `src/text`: cleaning, chunking, embedding, and FinBERT fine-tuning
- `src/features`: structured signals and merged feature matrix
- `src/models`: supervised ranking model
- `src/policy`: RL state building, training, and evaluation
- `src/portfolio`: action projection, constraints, and optional portfolio layer
- `src/backtest`: simulation engine, metrics, attribution, and plots
- `configs/`: experiment configuration
- `tests/`: alignment, leakage, backtest, and constraint checks

## Practical Scope Recommendation

The safest first version is:
- long-only,
- top `K` fixed in advance,
- weekly target weights,
- one RL algorithm,
- linear transaction costs,
- simple max-position constraints,
- one strong supervised ranker,
- pretrained FinBERT baseline plus optional fine-tuning extension.

That is large enough to be interesting and still narrow enough to execute well.

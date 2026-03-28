# Final Project Recommendation: Text-Enhanced Portfolio Construction with FinBERT + Contextual Bandits/RL

## Executive Summary

This project asks a practical question:

> **Can information extracted from financial text improve portfolio decisions after accounting for trading frictions such as turnover, bid-ask spread, and market impact?**

The proposed system combines three ideas:

1. **Natural language processing (NLP)** to turn financial documents into numerical features.
2. **Traditional quantitative finance signals** such as momentum, volatility, valuation, and liquidity.
3. **A decision policy** that chooses *portfolio construction targets* each week, rather than directly choosing hundreds of stock weights.

The central recommendation is to **avoid making the RL agent output raw weights for ~400 stocks**. That formulation is realistic, but too difficult and unstable for a final project. Instead, the strongest design is a **hybrid architecture**:

- a text model produces stock-level features,
- a scoring model estimates expected usefulness of each stock,
- a contextual bandit or RL policy chooses **high-level portfolio targets**,
- and a constrained optimizer converts those targets into actual stock weights.

This keeps the project:
- ambitious enough to be interesting,
- financially realistic,
- more interpretable,
- and much more feasible to implement and evaluate well.

---

## Why this design is stronger than “pick one of five baskets” or “output all stock weights directly”

### Option A: Choose among five prebuilt baskets
This is easy to train and easy to explain, but too restrictive. It simplifies the action space so much that the RL part risks becoming less meaningful. It is useful as a **baseline**, but not ideal as the main method.

### Option B: Output direct stock weights for the full universe
This is the most realistic in principle, but difficult in practice:
- the action space is very large,
- the reward is noisy,
- transaction costs matter a lot,
- and the policy can become unstable or hard to interpret.

### Recommended middle ground: choose portfolio targets, then optimize weights
The best compromise is for the RL/bandit agent to choose:
- risk budget,
- turnover budget,
- number of names to hold,
- factor tilts,
- sector deviation limits,
- concentration limits.

Then a portfolio optimizer uses these targets plus the predicted stock scores to generate actual weights.

This design separates the problem into two parts:

- **Forecasting / representation learning**: What does the text say about a stock’s future outlook?
- **Decision-making / portfolio construction**: Given those signals, how aggressively should the portfolio lean into them?

That separation is both financially sensible and easier to defend academically.

---

# Stage-by-Stage System Design

## Stage 0: Define the prediction horizon, investment universe, and evaluation protocol

### Purpose
Before modeling anything, define exactly:
- which stocks are eligible,
- what information is available at each point in time,
- what the decision frequency is,
- and how success will be measured.

### Plain-English explanation for a finance professor
This stage makes sure the experiment is fair and realistic. It prevents the model from “cheating” by seeing future information and ensures that results reflect an actual investment process.

### Plain-English explanation for an ML professor
This stage defines the supervised learning and sequential decision setup: the input-output timing, train/validation/test split, and the constraints under which the model operates.

### Recommended choices
- **Universe**: S&P 500 constituents, or a survivorship-bias-aware approximation if full historical membership is difficult.
- **Decision frequency**: Weekly (every Friday close or next Monday open).
- **Holding period**: One week.
- **Backtest period**: 2004–2023.
- **Features available at time _t_**:
  - financial documents available by time _t_,
  - accounting/fundamental data known by time _t_,
  - price/volume data known by time _t_.
- **Outcomes**:
  - next-week return,
  - next-week excess return,
  - next-week Sharpe-like reward or return divided by recent volatility.

### Deliverables
- A clean stock-week panel.
- A documented no-lookahead pipeline.
- A walk-forward or rolling evaluation scheme.

### Tech stack / implementation details
- **Languages**: Python
- **Storage**: Parquet files for panel data
- **Dataframes**: pandas / polars
- **Date alignment**: pandas market calendars, exchange calendars
- **Experiment tracking**: MLflow or Weights & Biases
- **Config management**: YAML + Hydra or plain YAML configs

---

## Stage 1: Collect and align financial text data

### Purpose
Gather financial documents that can plausibly contain useful information about firms.

### Candidate text sources
- 10-K and 10-Q filings
- 8-K filings
- earnings call transcripts
- press releases
- news articles (optional if available cleanly)

### Plain-English explanation for a finance professor
The goal is to convert qualitative disclosures into something measurable. Investors often react not just to raw numbers, but to management tone, risk language, guidance changes, and discussion of business conditions.

### Plain-English explanation for an ML professor
This stage builds the raw text corpus and aligns each document to the correct stock and timestamp so that later embeddings are temporally valid.

### Key implementation concerns
- Timestamp every document properly.
- Map documents to tickers correctly.
- Use only documents available before the portfolio decision date.
- Aggregate multiple documents if several appear in the same week.

### Output
A table like:

| ticker | doc_date | doc_type | text | filing_week | usable_for_week |
|---|---|---|---|---|---|

### Tech stack / implementation details
- **SEC access**: sec-edgar-downloader, sec-api, WRDS, or Kaggle/local dataset if already available
- **Transcript/news**: vendor dataset or pre-collected CSV/JSON
- **Text cleaning**:
  - strip HTML,
  - remove boilerplate,
  - normalize whitespace,
  - optionally truncate very long docs into chunks
- **Suggested libraries**:
  - `beautifulsoup4`
  - `lxml`
  - `regex`
  - `pandas`
  - `pyarrow`

### Notes
If full-document processing is too expensive, use:
- management discussion sections,
- risk factor sections,
- earnings-call prepared remarks,
- or chunk-and-pool embeddings.

---

## Stage 2: Build text embeddings with FinBERT

### Purpose
Turn each document into a fixed-length numerical vector that summarizes its meaning.

### Recommended design
Use **FinBERT** as the base encoder, because it is already adapted to financial language.

Instead of using it only for sentiment classification, use it to create **dense feature vectors** (for example, the `[CLS]` token embedding or pooled embedding) for each document.

### Plain-English explanation for a finance professor
This is the step where written language gets translated into numbers. The numbers are not simple word counts; they are a compact representation of meaning, tone, and context learned from large amounts of financial text.

### Plain-English explanation for an ML professor
FinBERT acts as a domain-specific transformer encoder. The output embedding becomes a reusable representation for downstream prediction and policy learning.

### Recommended enhancement: contrastive fine-tuning
Instead of using off-the-shelf FinBERT only, fine-tune it so that documents associated with **similar future weekly outcomes** are closer together in embedding space, while documents associated with very different future outcomes are farther apart.

### Intuition
You are not merely asking, “Is this document positive or negative?”
You are asking, “Does this document contain information associated with similar future investment outcomes?”

### Possible labeling choices
Use future one-week outcomes such as:
- next-week excess return,
- future weekly Sharpe proxy,
- quantile buckets (top / middle / bottom),
- risk-adjusted abnormal return.

### Output
For each stock-week, produce:
- one or more document embeddings,
- then aggregate into a single stock-week text feature vector.

### Aggregation choices
If multiple documents exist in a week:
- mean pooling,
- weighted mean (e.g., 10-K > press release),
- attention pooling,
- latest-document-only baseline.

### Tech stack / implementation details
- **Core libraries**:
  - `transformers`
  - `torch`
  - `datasets`
  - `sentencepiece` if needed
- **Model**:
  - ProsusAI/finbert or another finance-domain BERT variant
- **Training setup**:
  - mixed precision if GPU available
  - batch size based on VRAM
  - chunking for long documents
- **Loss options**:
  - supervised contrastive loss,
  - triplet loss,
  - classification loss on outcome buckets plus embedding regularization
- **Artifacts to save**:
  - fine-tuned encoder weights
  - tokenizer config
  - stock-week embedding parquet file

### Practical simplification
If contrastive fine-tuning becomes too heavy, use:
1. off-the-shelf FinBERT embeddings,
2. then compare against contrastively fine-tuned embeddings as an extension.

---

## Stage 3: Create classic quantitative finance features

### Purpose
Text should not be the only input. Include standard quantitative predictors so the model can be compared against what a traditional systematic strategy would use.

### Plain-English explanation for a finance professor
This stage adds the standard signals that quantitative investors often use already, such as momentum, volatility, valuation, liquidity, and size.

### Plain-English explanation for an ML professor
These are structured tabular features that complement the unstructured text embeddings.

### Suggested features
#### Price-based
- 1-week, 1-month, 3-month, 6-month, 12-month momentum
- short-term reversal
- realized volatility
- beta / market sensitivity
- idiosyncratic volatility

#### Volume / liquidity
- average daily dollar volume
- turnover
- Amihud illiquidity proxy
- bid-ask spread proxy if available

#### Fundamentals / valuation
- book-to-market
- earnings yield
- profitability / ROE
- leverage
- asset growth
- analyst revision proxies if available

#### Risk / portfolio controls
- sector
- market cap
- factor exposures
- recent drawdown

### Output
A unified stock-week feature matrix:
- text embedding columns,
- numeric quant factor columns,
- metadata columns.

### Tech stack / implementation details
- **Feature engineering**: pandas / polars / numpy
- **Standardization**:
  - cross-sectional z-scoring by week
  - winsorization to reduce outlier distortion
- **Feature store format**:
  - Parquet by year or by split
- **Recommended checks**:
  - missingness analysis
  - factor correlation heatmaps
  - leakage checks

---

## Stage 4: Train a stock scoring model before any RL step

### Purpose
Before building a policy, estimate how attractive each stock appears based on the available features.

### Why this stage matters
A good portfolio policy needs good stock-level information. If the policy receives poor signals, no clever RL formulation will rescue it.

### Plain-English explanation for a finance professor
This stage produces a forecast or ranking for each stock. Think of it as the “research signal” that tells the portfolio construction process which names look more promising.

### Plain-English explanation for an ML professor
This is a supervised learning layer mapping stock-week features to future outcomes. It provides either:
- predicted expected return,
- predicted risk-adjusted score,
- or a ranking score.

### Recommended target choices
- next-week excess return
- next-week return / recent volatility
- ordinal bucket (top decile vs bottom decile)
- cross-sectional rank target

### Candidate models
- linear regression / ridge / lasso
- XGBoost / LightGBM
- MLP
- shallow transformer or fusion network on text + tabular features

### Strong recommendation
Use a **simple, strong tabular model** first:
- LightGBM or XGBoost for tabular + embedding inputs.

This gives a strong baseline and is much easier to debug than end-to-end RL from raw embeddings.

### Output
For each stock-week:
- predicted alpha or score,
- uncertainty estimate if available,
- rank percentile.

### Tech stack / implementation details
- **Libraries**:
  - `lightgbm`
  - `xgboost`
  - `scikit-learn`
- **Evaluation**:
  - rank IC / Spearman correlation,
  - decile spread,
  - long-short spread,
  - calibration plots
- **Model storage**:
  - joblib or pickle for tree models
  - YAML/JSON for hyperparameters

---

## Stage 5: Define the contextual bandit / RL action space

### Purpose
This is where the project becomes sequential decision-making rather than just prediction.

### Core recommendation
Do **not** let the policy choose 400+ raw stock weights directly.

Instead, let it choose a compact set of **portfolio construction targets**.

### Suggested action dimensions
The policy can output weekly choices such as:
- target gross exposure
- target cash level
- target number of holdings
- turnover penalty strength
- max stock weight
- factor tilt strength toward momentum/value/quality
- sector neutrality strictness
- concentration penalty

### Plain-English explanation for a finance professor
The policy is not hand-picking every position from scratch. It is deciding how aggressive or conservative the portfolio construction process should be this week, and which styles or constraints should matter more.

### Plain-English explanation for an ML professor
This converts the problem from a massive continuous action space into a low-dimensional control problem. The policy learns how to set allocation preferences conditional on the current market and cross-sectional opportunity set.

### Why this is the best design
It preserves:
- sequential decision-making,
- sensitivity to market regime,
- and real portfolio controls,

while making training far more stable and interpretable.

### State representation
The policy state can include:
- summary statistics of stock scores,
- market regime indicators,
- cross-sectional dispersion,
- recent realized turnover,
- recent strategy drawdown,
- factor environment,
- volatility regime.

### Action space options
#### Option 1: Discrete contextual bandit
Choose one of a small number of action templates:
- conservative,
- balanced,
- aggressive,
- high-momentum tilt,
- defensive quality tilt.

#### Option 2: Continuous low-dimensional control
Output continuous targets like:
- turnover penalty lambda,
- max names,
- factor tilt coefficients.

### Strongest recommendation for the course project
Start with a **contextual bandit** or **one-step weekly decision** formulation.
That is easier to train and easier to explain than full multi-step deep RL, while still being sequential in spirit.

### Tech stack / implementation details
- **Simplest option**:
  - contextual bandit with LinUCB, Thompson Sampling, or a small neural policy
- **More advanced option**:
  - offline RL / policy learning with logged historical transitions
- **Libraries**:
  - `contextualbandits`
  - `scikit-learn`
  - `d3rlpy`
  - `torch`
- **Recommended starting point**:
  - discrete action templates + contextual bandit

---

## Stage 6: Convert policy outputs into actual portfolio weights with an optimizer

### Purpose
Take the stock scores from Stage 4 and the action targets from Stage 5, then solve for implementable weights.

### Plain-English explanation for a finance professor
This is the actual portfolio construction engine. It chooses which stocks to hold and in what size, while respecting constraints like diversification and trading cost.

### Plain-English explanation for an ML professor
This is a constrained optimization layer that maps predictions and control parameters to feasible portfolio weights.

### Inputs
- predicted stock scores
- previous week portfolio weights
- liquidity and risk estimates
- policy-chosen targets
- portfolio constraints

### Constraints to include
- long-only or long-short, depending on project scope
- sum of weights = 1
- max weight per stock
- max weight per sector
- turnover limit
- liquidity filters
- optional cardinality / sparse holdings
- transaction cost penalty
- market-impact approximation

### Objective example
Maximize:

`expected_score - risk_penalty - turnover_penalty - transaction_cost_penalty`

### Why this stage is important
A portfolio can look attractive on paper but fail in practice because it trades too much or concentrates too heavily. This stage makes the strategy realistic.

### Tech stack / implementation details
- **Optimization libraries**:
  - `cvxpy`
  - `numpy`
  - `scipy`
- **Risk model options**:
  - diagonal volatility estimate,
  - covariance shrinkage,
  - factor covariance model
- **Transaction cost modeling**:
  - linear turnover cost,
  - spread proxy,
  - square-root impact approximation as an optional extension

### Practical simplification
Use:
- long-only,
- max 5% per name,
- max 25% turnover per week,
- linear transaction cost model.

That is sufficient for a strong final project.

---

## Stage 7: Define rewards and backtesting procedure

### Purpose
Evaluate whether the strategy is actually good after costs.

### Plain-English explanation for a finance professor
The reward is the portfolio’s realized performance after accounting for trading frictions and risk. The backtest simulates how the strategy would have performed over time.

### Plain-English explanation for an ML professor
This is the environment evaluation loop: the policy acts, the optimizer constructs a portfolio, and the next-week realized returns determine reward.

### Recommended reward
Weekly reward such as:

`portfolio_return - trading_cost - lambda_1 * volatility - lambda_2 * drawdown_penalty`

Alternative:
- portfolio Sharpe contribution proxy,
- excess return penalized by turnover and drawdown.

### Evaluation metrics
#### Return and risk
- annualized return
- annualized volatility
- Sharpe ratio
- Sortino ratio
- max drawdown
- Calmar ratio

#### Trading realism
- average turnover
- average number of holdings
- effective diversification
- cost-adjusted return

#### Predictive usefulness
- rank IC
- hit rate
- decile spread

### Backtest design
Use:
- expanding-window or rolling-window retraining,
- realistic lags,
- weekly rebalancing,
- transaction costs every rebalance.

### Tech stack / implementation details
- **Backtesting**:
  - custom vectorized engine in pandas/numpy,
  - or `vectorbt` if you want convenience
- **Performance analysis**:
  - `empyrical`
  - `pyfolio`-style metrics
  - custom plots in matplotlib / plotly

---

## Stage 8: Baselines and ablation studies

### Purpose
Show that each component matters.

### Required baselines
1. **Equal-weight baseline**
2. **Price/factor-only model**
3. **Vanilla FinBERT embedding baseline**
4. **Five-basket selection baseline**
5. **Recommended hybrid policy** (text + quant + policy targets + optimizer)

### Useful ablations
- remove text features
- remove contrastive fine-tuning
- remove optimizer constraints
- remove transaction cost penalty
- replace contextual bandit with static optimizer
- compare direct stock scoring vs factor-tilt action space

### Why this matters
A good final project should not just show one profitable backtest. It should show **what component created the improvement**.

---

## Stage 9: Interpretability and diagnostics

### Purpose
Make the model explainable enough for academic evaluation.

### For a finance professor
You should be able to answer:
- which factors the portfolio is exposed to,
- how much turnover it generates,
- whether text helps in certain regimes,
- and whether the strategy is simply rediscovering momentum or size.

### For an ML professor
You should be able to answer:
- whether embeddings improve predictive performance,
- whether the bandit/policy changes behavior by regime,
- and whether the gains are robust across validation splits.

### Suggested diagnostics
- feature importance for tabular model
- embedding visualization (UMAP / t-SNE)
- performance by market regime
- turnover decomposition
- sector exposure charts
- cumulative return plots with and without costs
- action frequency histogram for the contextual bandit

### Tech stack / implementation details
- `shap`
- `matplotlib`
- `plotly`
- `umap-learn`
- `seaborn` optional for diagnostics only

---

# Recommended Implementation Path

## Minimum viable project
If time becomes tight, build this version:

1. Collect stock-week text and quant data.
2. Generate FinBERT embeddings.
3. Train a LightGBM stock scoring model.
4. Create 5–10 portfolio construction templates.
5. Use a contextual bandit to choose the template each week.
6. Backtest with transaction costs.
7. Compare against equal-weight and factor-only baselines.

This is already a strong project.

## Full recommended version
If time permits, extend to:

1. Contrastively fine-tune FinBERT.
2. Replace fixed templates with continuous target outputs.
3. Use a constrained optimizer with dynamic turnover and factor tilts.
4. Run full ablations and interpretability analysis.

---

# Recommended Repository Structure

```text
text-enhanced-portfolio-rl/
├── README.md
├── requirements.txt
├── environment.yml
├── .gitignore
├── configs/
│   ├── data.yaml
│   ├── model_text.yaml
│   ├── model_tabular.yaml
│   ├── bandit.yaml
│   ├── optimizer.yaml
│   ├── backtest.yaml
│   └── experiment.yaml
├── data/
│   ├── raw/
│   │   ├── filings/
│   │   ├── transcripts/
│   │   ├── prices/
│   │   ├── fundamentals/
│   │   └── metadata/
│   ├── interim/
│   │   ├── cleaned_text/
│   │   ├── aligned_panel/
│   │   └── features/
│   └── processed/
│       ├── train/
│       ├── valid/
│       ├── test/
│       └── embeddings/
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_text_cleaning.ipynb
│   ├── 03_embedding_exploration.ipynb
│   ├── 04_feature_checks.ipynb
│   ├── 05_baseline_backtest.ipynb
│   └── 06_result_visualization.ipynb
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── ingest_filings.py
│   │   ├── ingest_prices.py
│   │   ├── ingest_fundamentals.py
│   │   ├── align_panel.py
│   │   └── split_data.py
│   ├── text/
│   │   ├── clean_text.py
│   │   ├── chunk_documents.py
│   │   ├── finbert_embed.py
│   │   ├── contrastive_dataset.py
│   │   └── train_contrastive_finbert.py
│   ├── features/
│   │   ├── make_quant_features.py
│   │   ├── aggregate_text_features.py
│   │   ├── normalize_features.py
│   │   └── build_feature_matrix.py
│   ├── models/
│   │   ├── train_lightgbm.py
│   │   ├── train_xgboost.py
│   │   ├── evaluate_scores.py
│   │   └── predict_scores.py
│   ├── policy/
│   │   ├── build_state.py
│   │   ├── action_templates.py
│   │   ├── train_bandit.py
│   │   ├── evaluate_bandit.py
│   │   └── offline_rl.py
│   ├── portfolio/
│   │   ├── risk_model.py
│   │   ├── transaction_costs.py
│   │   ├── optimizer.py
│   │   └── constraints.py
│   ├── backtest/
│   │   ├── engine.py
│   │   ├── metrics.py
│   │   ├── attribution.py
│   │   └── plots.py
│   └── utils/
│       ├── io.py
│       ├── logging_utils.py
│       ├── seed.py
│       └── config.py
├── reports/
│   ├── figures/
│   ├── tables/
│   └── final_report.md
├── artifacts/
│   ├── models/
│   ├── scalers/
│   ├── embeddings/
│   └── backtests/
└── tests/
    ├── test_alignment.py
    ├── test_no_lookahead.py
    ├── test_optimizer.py
    └── test_backtest_engine.py
```

---

# README-Style Build Plan

## 1. Data pipeline
Run ingestion scripts to collect and align:
- text documents,
- prices,
- fundamentals,
- and ticker metadata.

**Outputs**
- cleaned documents,
- stock-week mapping,
- aligned panel dataset.

## 2. Text representation
Generate embeddings using off-the-shelf FinBERT, then optionally run contrastive fine-tuning.

**Outputs**
- document embeddings,
- stock-week pooled text features.

## 3. Feature engineering
Compute classic quant features and merge them with text embeddings.

**Outputs**
- final stock-week feature matrix.

## 4. Stock scoring
Train supervised models to predict next-week outcome or ranking.

**Outputs**
- stock-level predicted scores,
- rank IC and decile analyses.

## 5. Policy learning
Train a contextual bandit or offline RL policy to choose weekly portfolio construction targets.

**Outputs**
- selected action per week,
- action-value diagnostics.

## 6. Portfolio optimization
Convert scores + policy targets into weights using constraints and transaction costs.

**Outputs**
- portfolio weights by week,
- turnover and cost estimates.

## 7. Backtest and analysis
Run full backtest and compare against baselines.

**Outputs**
- return/risk metrics,
- cumulative return plots,
- attribution tables,
- ablation results.

---

# Recommended Tech Stack Summary

## Core language
- **Python 3.10+**

## Data manipulation
- **pandas**
- **polars** (optional for speed)
- **numpy**
- **pyarrow**

## NLP / deep learning
- **PyTorch**
- **Hugging Face transformers**
- **datasets**
- **tokenizers**

## Tabular prediction
- **LightGBM**
- **XGBoost**
- **scikit-learn**

## Bandits / RL
- **contextualbandits**
- **d3rlpy**
- **PyTorch**

## Optimization
- **cvxpy**
- **scipy**

## Backtesting / metrics
- **empyrical**
- **vectorbt** (optional)
- **matplotlib**
- **plotly**

## Experiment management
- **Weights & Biases** or **MLflow**
- **Hydra** or simple YAML configs

## Testing / engineering
- **pytest**
- **black**
- **ruff**
- **pre-commit**

---

# Concrete implementation details by stage

## Stage 0 implementation details
- Build one row per `ticker-week`.
- Store split labels: train, validation, test.
- Ensure all labels come strictly after the feature timestamp.

## Stage 1 implementation details
- Save raw docs as JSON or Parquet with text and metadata.
- Use unique document IDs.
- Keep filing type and section info.

## Stage 2 implementation details
- Start with frozen FinBERT embeddings.
- Save 768-dimensional vectors.
- Later test contrastive fine-tuning with outcome buckets.

## Stage 3 implementation details
- Standardize features cross-sectionally within each week.
- Winsorize extreme values.
- Store missingness masks if needed.

## Stage 4 implementation details
- Train LightGBM on stock-week rows.
- Use walk-forward validation.
- Evaluate cross-sectional ranking, not just MSE.

## Stage 5 implementation details
- Build weekly state vectors from market and cross-sectional summaries.
- Define 5–10 action templates initially.
- Train contextual bandit on historical reward of each action.

## Stage 6 implementation details
- Use `cvxpy` for the constrained weight solve.
- Penalize deviation from previous weights.
- Include max position and max turnover constraints.

## Stage 7 implementation details
- Simulate weekly rebalancing.
- Apply transaction costs based on turnover.
- Compute cumulative net performance.

## Stage 8 implementation details
- Store each baseline under its own config.
- Use identical test windows and cost assumptions.
- Create one summary table of all metrics.

## Stage 9 implementation details
- Plot feature importance and action usage.
- Compare factor exposures with and without text.
- Report where text helps most: high-volatility periods, earnings season, etc.

---

# Suggested timeline

## Week 1
- finalize scope,
- build aligned stock-week dataset,
- collect and clean text.

## Week 2
- generate FinBERT embeddings,
- engineer quant features,
- train baseline stock scoring model.

## Week 3
- implement optimizer,
- build equal-weight and factor-only baselines,
- run first realistic backtest.

## Week 4
- add contextual bandit policy,
- compare fixed-template actions,
- run ablations.

## Week 5
- attempt contrastive fine-tuning,
- improve diagnostics and plots,
- finalize report.

---

# Final recommendation in one sentence

The strongest version of this project is:

> **Use FinBERT-based text embeddings and classic quant signals to score stocks each week, train a contextual bandit or low-dimensional RL policy to choose portfolio construction targets rather than raw stock weights, and then use a constrained optimizer to map those targets into implementable portfolios evaluated with realistic transaction costs and risk metrics.**

This formulation is:
- more realistic than choosing one of five fixed baskets,
- more feasible than directly outputting hundreds of stock weights,
- and easier to explain to both finance and ML audiences.

---

# Optional one-paragraph proposal version

This project proposes a hybrid text-and-portfolio-learning framework for weekly equity allocation. Financial documents such as SEC filings and earnings-call transcripts will be encoded using FinBERT, optionally contrastively fine-tuned so embeddings better reflect future weekly risk-adjusted outcomes. These text features will be merged with classic quantitative signals such as momentum, volatility, liquidity, and valuation to produce stock-level scores. Rather than having a reinforcement learning agent directly output portfolio weights for hundreds of stocks, the project will train a contextual bandit or low-dimensional RL policy that selects portfolio construction targets—such as factor tilts, turnover budget, concentration level, and risk aggressiveness—which are then converted into actual stock weights using a constrained optimizer. The final strategy will be evaluated on S&P 500 data from 2004–2023 using after-cost Sharpe ratio, max drawdown, turnover, and related metrics, and compared against equal-weight, factor-only, vanilla-FinBERT, and coarse basket-selection baselines.

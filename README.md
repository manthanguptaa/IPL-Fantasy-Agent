# IPL Fantasy Agent

An ML-powered system for generating optimal Dream11 fantasy cricket teams for IPL matches. Combines probabilistic player forecasting, Monte Carlo simulation, constrained optimization, and test-time training to select the best XI with Captain and Vice-Captain.

## How It Works

```
Historical Match Data
        |
   Feature Engineering (100+ features: rolling averages, venue, opponent, form)
        |
   Quantile Forecasting (CatBoost ensemble: q10, q50, q90 per player)
        |
   Monte Carlo Simulation (5000 scenarios per candidate team)
        |
   Candidate Reranking (8 diverse teams scored on expected, ceiling, floor, captain leverage)
        |
   Test-Time Training (residual adapter corrects biases from observed match outcomes)
        |
   Output: 11 players + Captain + Vice-Captain
```

## Performance

Backtested on 200 IPL matches:

| Metric | Value |
|--------|-------|
| Mean Selected Score | 564.7 |
| Mean Oracle Score | 853.7 |
| Mean Regret | 289.0 |
| Oracle Capture Rate | ~66% |
| Player Overlap with Oracle | 54.4% |
| Captain Accuracy | 12.0% |
| VC Accuracy | 10.0% |

| Regret Breakdown | Points | % of Total |
|------------------|--------|------------|
| Team Selection (wrong players) | 195.3 | 68% |
| Captain Selection | 70.5 | 25% |
| VC Selection | 20.2 | 7% |

## Quick Start

### Prerequisites

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) (fast Python package manager):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install

```bash
git clone https://github.com/manthanguptaa/IPL-Fantasy-Agent.git
cd IPL-Fantasy-Agent
uv sync
```

### Generate a team for an IPL 2026 match

```bash
uv run python scripts/generate_match_team.py \
    --team1 CSK --team2 MI \
    --venue "Wankhede Stadium" \
    --won-toss MI --toss-decision bowl
```

Team abbreviations: CSK, MI, RCB, DC, GT, KKR, LSG, PBKS, RR, SRH

Options:
- `--no-rerank` — skip simulation reranking, use single optimizer solve
- `--exclude "Player Name"` — remove injured/unavailable players
- `--top-k 5` — show top 5 alternative teams
- `--candidates 12` — generate more candidate teams for reranking
- `--no-adapt` — skip residual adaptation (test-time training)

### Record match results (for test-time training)

After each match, record actual fantasy points to improve future predictions:

```bash
# From a CSV file (player_name,actual_points)
uv run python scripts/record_match_result.py \
    --match-id "ipl2026_01" \
    --match-date "2026-03-28" \
    --actuals data/match_results/ipl2026_01.csv

# Or inline
uv run python scripts/record_match_result.py \
    --match-id "ipl2026_01" \
    --match-date "2026-03-28" \
    --inline "Virat Kohli:45,Jasprit Bumrah:62,..."
```

### Run backtests

```bash
# Backtest the optimizer on historical matches
uv run python scripts/run_backtest.py --mode optimized --n-matches 50

# Backtest the reranking pipeline
uv run python scripts/run_reranking_backtest.py --n-matches 50
```

### Run tests

```bash
uv run pytest tests/ -v
```

## Data Setup

The system requires historical cricket match data to train models and generate features.

### 1. Raw match data

Place ball-by-ball JSON match files (Cricsheet format) in directories:

```
ipl_json/          # IPL matches (required)
t20s_json/         # International T20s (recommended for recent form)
bbl_json/          # Big Bash League (optional)
...
```

### 2. Normalize and build features

```bash
# Convert JSON to normalized CSV
python scripts/normalize_cricket_json.py

# Build training dataset
python scripts/build_training_dataset.py

# Engineer features (rolling averages, venue stats, etc.)
python scripts/build_feature_dataset.py

# Add toss information
python scripts/update_features_with_toss.py
```

### 3. Train models

```bash
# Train quantile ensemble (q10, q25, q50, q75, q90)
uv run python -m src.ipl_fantasy.quantile_model \
    tmp/full_player_match_features_v4.csv \
    tmp/quantile_models
```

## Project Structure

```
src/ipl_fantasy/
    quantile_model.py         # CatBoost quantile ensemble (q10/q25/q50/q75/q90)
    enhanced_prediction.py    # Role-specific ceiling weighting
    team_optimizer.py         # Dream11 constraint optimizer (PuLP)
    improved_optimizer.py     # Multi-objective ceiling-weighted optimizer
    simulation.py             # Monte Carlo simulation engine
    team_reranker.py          # Candidate generation + simulation reranking
    residual_adapter.py       # Test-time training via residual correction
    build_features.py         # 100+ engineered features
    credit_estimation.py      # Dream11 credit estimation from history
    backtesting.py            # Oracle regret evaluation
    normalize_cricket_json.py # Raw data normalization
    player_roles.py           # Player role inference

src/match_predictor/
    predict_match.py          # Unified prediction pipeline (all 4 models)
    player_performance.py     # Player runs & wickets models
    team_score.py             # Team innings total model
    match_winner.py           # Match winner classifier
    victory_margin.py         # Victory margin regressor
    backtest.py               # IPL 2025 backtest harness
    data_prep.py              # Match-level data aggregation & features

scripts/
    generate_match_team.py    # Live match team generation pipeline
    record_match_result.py    # Post-match result recording for test-time training
    run_backtest.py           # Historical backtest runner
    run_reranking_backtest.py # Reranking pipeline backtest
    train_baseline_model.py   # Model training script
    analyze_breakouts.py      # Breakout performance analysis

data/
    ipl_2026_squads.csv       # IPL 2026 auction squads (all 10 teams)
    venue_profiles.csv        # Venue pitch characteristics
    player_roles.csv          # Inferred player roles

tests/                        # 57 unit tests
```

## Architecture

The system has four layers:

1. **Probabilistic Forecasting** — CatBoost quantile models predict floor/expected/ceiling fantasy points per player using 26 optimal features including role-stratified opponent stats
2. **Candidate Team Reranking** — generates 8 diverse teams via varied optimizer configs, simulates each under Monte Carlo, reranks using composite metric (expected + ceiling + floor + captain leverage + Sharpe)
3. **Constrained Optimization** — PuLP linear programming solver selects the best 11 under Dream11 constraints (100 credits, role bounds, max 7 per team, max 4 overseas)
4. **Test-Time Training** — residual adapter observes (predicted, actual) pairs after each match and learns systematic biases the offline model misses, progressively enabling per-player EMA corrections, per-role bias, and contextual ridge regression

## Match Prediction Models

A separate prediction pipeline (`src/match_predictor/`) forecasts match outcomes using 4 GradientBoosting models trained on historical IPL data (2017–2024) and backtested on IPL 2025 (74 matches).

| Model | Task | Algorithm | Key Features | IPL 2025 Performance |
|-------|------|-----------|--------------|---------------------|
| Player Runs | Predict individual batting runs | GBRegressor (500 trees, depth 5) | Rolling batting averages, strike rate, venue/opponent history | MAE 13.59 |
| Player Wickets | Predict individual wickets taken | GBRegressor (400 trees, depth 3) | Rolling bowling averages, economy rate, venue/opponent history | MAE 0.58 |
| Team Score | Predict team innings total | GBRegressor (500 trees, depth 4) | Venue profile, team rolling form, player strength aggregates, season inflation, first innings score | MAE 27.26 |
| Match Winner | Predict which team wins | GBClassifier (300 trees, depth 3) | Team form differentials, player strength comparisons, toss, venue | 61.4% accuracy |
| Victory Margin | Predict signed run margin | GBRegressor (300 trees, depth 3) | All winner features + dew/chase interaction, scoring form differentials | Direction accuracy 61.4% |

### Running the match predictor

```bash
# Run all 4 models and print match forecasts
uv run python -m src.match_predictor.predict_match

# Run full backtest on IPL 2025
uv run python -m src.match_predictor.backtest
```

## Key Design Decisions

- **Why quantile regression?** Fantasy cricket is high-variance. Knowing a player's ceiling matters as much as their expected score, especially for captaincy decisions.
- **Why constrained optimization over heuristics?** Dream11 has hard constraints (11 players, 100 credits, role bounds, max 7 per team). PuLP guarantees optimal solutions under these constraints.
- **Why test-time training?** The offline CatBoost model can't capture in-season shifts like new batting positions, changed team roles, or venue/condition drift. The residual adapter corrects for these biases using just 1-15 observed matches.
- **Why estimated credits?** Dream11 credit prices aren't publicly available historically. The credit estimator uses performance tiers and role bonuses as a proxy.

## Dream11 Scoring Rules

| Action | Points |
|--------|--------|
| Run scored | +1 |
| Boundary (4) | +1 bonus |
| Six | +2 bonus |
| 30 runs | +4 bonus |
| Half-century | +8 bonus |
| Century | +16 bonus |
| Duck (BAT/WK/AR) | -2 |
| Wicket | +25 |
| 3-wicket haul | +4 bonus |
| 4-wicket haul | +8 bonus |
| 5-wicket haul | +16 bonus |
| Maiden over | +12 |
| Catch | +8 |
| Stumping | +12 |
| Run out (direct) | +12 |
| Captain | 2x points |
| Vice-Captain | 1.5x points |

## License

MIT

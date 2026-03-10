# IPL Fantasy Agent

An ML-powered system for generating optimal Dream11 fantasy cricket teams for IPL matches. Combines probabilistic player forecasting, Monte Carlo simulation, constrained optimization, and reinforcement learning to select the best XI with Captain and Vice-Captain.

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
   Contextual Bandit (LinUCB learns which strategy works for each match context)
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

### Install

```bash
pip install -e ".[dev]"
```

### Generate a team for an IPL 2026 match

```bash
python scripts/generate_match_team.py \
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

### Run backtests

```bash
# Backtest the optimizer on historical matches
python scripts/run_backtest.py --mode optimized --n-matches 50

# Backtest the reranking pipeline
python scripts/run_reranking_backtest.py --n-matches 50

# Backtest the RL agent
python scripts/run_live_rl.py --n-matches 200 --alpha 0.8
```

### Run tests

```bash
pytest tests/ -v
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
# Train baseline CatBoost model
python scripts/train_baseline_model.py

# Train quantile ensemble (q10, q50, q90)
python -c "
from src.ipl_fantasy.quantile_model import QuantileModelEnsemble
import pandas as pd
df = pd.read_csv('tmp/full_player_match_features_v4.csv', low_memory=False)
train = df[df['match_date'] < '2024-01-01']
val = df[df['match_date'] >= '2024-01-01']
ensemble = QuantileModelEnsemble()
ensemble.fit(train, val)
ensemble.save('tmp/quantile_models')
"
```

## Project Structure

```
src/ipl_fantasy/
    quantile_model.py         # CatBoost quantile ensemble (q10/q50/q90)
    enhanced_prediction.py    # Role-specific ceiling weighting
    team_optimizer.py         # Dream11 constraint optimizer (PuLP)
    improved_optimizer.py     # Multi-objective ceiling-weighted optimizer
    simulation.py             # Monte Carlo simulation engine
    team_reranker.py          # Candidate generation + simulation reranking
    rl_policy.py              # LinUCB contextual bandit
    reward_model.py           # RL reward computation
    build_features.py         # 100+ engineered features
    credit_estimation.py      # Dream11 credit estimation from history
    backtesting.py            # Oracle regret evaluation
    normalize_cricket_json.py # Raw data normalization
    player_roles.py           # Player role inference

scripts/
    generate_match_team.py    # Live match team generation pipeline
    run_backtest.py           # Historical backtest runner
    run_reranking_backtest.py # Reranking pipeline backtest
    run_live_rl.py            # RL agent training loop
    train_baseline_model.py   # Model training script
    analyze_breakouts.py      # Breakout performance analysis

data/
    ipl_2026_squads.csv       # IPL 2026 auction squads (all 10 teams)
    venue_profiles.csv        # Venue pitch characteristics
    player_roles.csv          # Inferred player roles

tests/                        # 40 unit tests
docs/                         # Design documents
```

## Architecture

The system implements a 5-phase roadmap:

1. **Probabilistic Forecasting** — CatBoost quantile models predict floor/expected/ceiling fantasy points per player using 23 optimal features
2. **Candidate Team Reranking** — generates 8 diverse teams via varied optimizer configs, simulates each under Monte Carlo, reranks using composite metric (expected + ceiling + floor + captain leverage + Sharpe)
3. **Reward Modeling** — blended reward signal combining score, regret, and captain quality for RL training
4. **Contextual Bandit** — LinUCB agent with 12-dimensional match context learns which optimization strategy to deploy per match
5. **LLM Sidecar** — (planned) pitch report extraction and team explanation generation

See [implementation-direction.md](implementation-direction.md) for the full design rationale.

## Key Design Decisions

- **Why quantile regression?** Fantasy cricket is high-variance. Knowing a player's ceiling matters as much as their expected score, especially for captaincy decisions.
- **Why constrained optimization over heuristics?** Dream11 has hard constraints (11 players, 100 credits, role bounds, max 7 per team). PuLP guarantees optimal solutions under these constraints.
- **Why LinUCB over deep RL?** With only ~74 matches per season and 8 strategy arms, LinUCB converges faster than neural approaches and provides interpretable arm selection.
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

See [problem-statement.md](problem-statement.md) for the full specification.

## License

MIT

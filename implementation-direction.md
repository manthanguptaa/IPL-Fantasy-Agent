# IPL Fantasy Agent - Implementation Direction

## Goal

Build the first working version of the IPL Fantasy Agent as a probabilistic fantasy-team engine that can output a valid Dream11 team for a match, along with Captain and Vice-Captain, using structured cricket data, simulation, and constrained optimization.

This document captures the implementation direction we are taking before reward modeling, RL, or LLM-heavy orchestration is introduced.

## Chosen Direction

We are starting with **Approach B**:

- probabilistic player forecasting
- Monte Carlo simulation
- constrained Dream11 team optimization
- simulation-aware Captain and Vice-Captain selection

We are **not** starting with a pure mean-only regression system, and we are **not** starting with RL as the primary decision engine.

## Why We Chose Approach B

Dream11 team selection is not only about average projected points.

The following factors matter a lot:

- player volatility
- ceiling versus floor
- toss-dependent role changes
- playing XI uncertainty
- Captain and Vice-Captain upside
- batting-position and bowling-overs uncertainty

A mean-only model misses too much of this structure. A probabilistic system gives us a better foundation for fantasy decisions, especially for captaincy and for differentiating between safe and high-upside picks.

At the same time, this approach is still practical and explainable because the final team is generated through explicit Dream11 constraints rather than through a black-box end-to-end model.

## High-Level Architecture

The first version of the system will follow this pipeline:

1. Ingest match inputs and player data.
2. Predict a distribution of fantasy outcomes for each player.
3. Simulate many possible match outcomes.
4. Evaluate candidate Dream11 teams under those simulated outcomes.
5. Select the final XI.
6. Pick Captain and Vice-Captain using simulation-aware ranking.

## System Components

### 1. Data Ingestion Layer

This layer prepares all structured inputs required for a specific match.

It should ingest:

- current player pool
- team squads
- player roles
- Dream11 credits
- venue
- opponent
- toss result if available
- confirmed playing XI if available
- weather context
- historical IPL player performance
- recent T20 form from the last 12 months

### 2. Player Forecasting Layer

This layer predicts a **distribution** of fantasy outcomes for each player, not just one point estimate.

The model should produce one or more of the following:

- expected points
- lower quantile or floor
- upper quantile or ceiling
- variance or uncertainty estimate
- lineup probability
- probability of low-score or zero-score outcomes

For the first version, this can be implemented with:

- LightGBM
- CatBoost
- XGBoost
- quantile regression models
- residual-based uncertainty estimation on top of a tabular regressor

### 3. Simulation Layer

This layer uses the player-level forecasts to generate many plausible match outcomes.

Suggested setup:

- run 1,000 to 10,000 Monte Carlo simulations per match
- sample player points from predicted distributions
- optionally incorporate toss and lineup uncertainty when those are not yet confirmed

This simulation layer allows the system to estimate:

- expected team score
- team score variance
- upside potential
- downside risk
- Captain and Vice-Captain leverage

### 4. Constraint Optimization Layer

This layer selects a valid Dream11 team under hard constraints.

The optimizer must enforce:

- exactly 11 players
- maximum 100 credits
- 1-4 wicket-keepers
- 3-6 batters
- 1-4 all-rounders
- 3-6 bowlers
- maximum 7 players from one real team

Recommended tools:

- OR-Tools
- PuLP

The optimization may initially target:

- expected simulated score

Later it can be extended to optimize:

- expected score with risk penalty
- expected score with upside bonus
- multiple lineup styles such as safe, balanced, or high-upside

### 5. Captain and Vice-Captain Selection Layer

Captain and Vice-Captain should not be selected using only mean projected points.

Instead, the first version should use simulated outcomes to rank players in the chosen XI by:

- expected fantasy contribution
- ceiling
- volatility
- consistency
- sensitivity to toss and role uncertainty

This allows the system to make stronger decisions on the biggest fantasy multiplier lever.

### 6. Backtesting and Evaluation Layer

We will evaluate the system primarily at the team level, not only at the player level.

The key metrics are:

- actual score of the selected XI
- oracle best-valid XI score after the match
- regret versus oracle
- Captain and Vice-Captain regret
- comparison versus manually selected teams or benchmark teams

Secondary diagnostics:

- player-level RMSE
- player-level MAE
- calibration of uncertainty estimates
- calibration of lineup probability

## Data Direction

The data strategy will be a composite stack rather than a single dataset.

### Core historical data

Use structured IPL historical match data with enough detail to derive Dream11-style outcomes and player features.

Primary use cases:

- historical player performance
- derived fantasy labels
- opponent and venue history
- batting and bowling role trends

### Recent-form data

Add T20 cricket data from the last 12 months on top of IPL history.

This is important because:

- current form matters a lot in fantasy
- player roles evolve across seasons
- a player may arrive in the IPL with recent T20 form that is more relevant than older IPL history

This recent-form layer should focus on T20 cricket first, not ODI or Test cricket.

### Current squads and player pool

Use the IPL official website as the source for:

- current teams
- current squads
- fixture structure

This provides the broad candidate universe for current-season selection.

### Match-time context

Use additional sources or APIs for:

- confirmed playing XI
- toss
- venue metadata
- weather
- live match context

### Dream11 credits

Dream11 credit prices should be included whenever available.

If complete historical pricing is not available, we should:

- capture prices prospectively going forward
- use available fantasy credit APIs or proxies where helpful
- keep the system flexible so credits can be improved later without redesigning the model

## What We Are Building First

The first shippable system should do the following for one match:

1. Take in a structured player pool and match context.
2. Predict probabilistic fantasy outcomes for each player.
3. Run simulation.
4. Output the best valid Dream11 XI.
5. Select Captain and Vice-Captain.
6. Estimate predicted total points and basic confidence.

## What We Are Explicitly Not Building Yet

To keep the first version focused, we are not starting with:

- reward modeling
- contextual bandits
- full RL
- ownership or differentiation optimization
- LLM-based team selection
- fully automated unstructured news reasoning in the core loop

These are later extensions, not the foundation.

## Planned Roadmap

### Phase 1 - Probabilistic V1

Build the first end-to-end version with:

- supervised player forecasting
- uncertainty estimation
- Monte Carlo simulation
- Dream11 optimizer
- Captain and Vice-Captain logic
- backtesting against historical matches

This is the immediate implementation direction.

### Phase 2 - Candidate Team Reranking

Once Phase 1 is stable:

- generate top-K candidate valid teams
- compare them under simulated outcomes
- rerank candidate teams using more nuanced objectives such as upside, floor, and captaincy leverage

### Phase 3 - Reward Modeling

After a stable base system exists:

- train a team-level reward model
- score complete teams rather than only individual players
- incorporate oracle regret and captaincy quality into team-level learning

### Phase 4 - Contextual Bandit / RL Layer

Only after the previous layers are working well:

- use a bandit or offline RL layer to choose between candidate teams
- personalize for safe versus aggressive contest styles
- learn selection policy from historical outcomes

### Phase 5 - LLM Sidecar

LLMs may later be added for:

- extracting structured signals from pitch reports and news
- summarizing why a team was selected
- generating user-facing explanations

The LLM should remain a support layer, not the core decision engine.

## Why RL Is Deferred

RL is not the starting point because:

- fantasy cricket offers sparse rewards
- historical counterfactual data is limited
- the action space is large
- evaluation is harder without a strong baseline

A probabilistic forecasting plus optimization system gives us:

- a strong baseline
- measurable backtests
- interpretable outputs
- a clean foundation for later reward modeling and bandit-style improvements

## Success Criteria for This Direction

This implementation direction is successful if the first system can:

- ingest a real upcoming IPL match
- produce a valid Dream11 XI
- produce a reasonable Captain and Vice-Captain
- outperform simple heuristic team selection baselines
- show meaningful signal in historical backtests
- create a stable base for later reward modeling

## Final Summary

We are building the IPL Fantasy Agent as a **probabilistic team-selection system**.

The first version will combine:

- structured cricket data
- probabilistic player forecasting
- Monte Carlo simulation
- constrained team optimization
- simulation-aware Captain and Vice-Captain selection

This gives us a stronger and more realistic starting point than a mean-only regressor, while still keeping the architecture practical enough to implement, debug, and improve over time.

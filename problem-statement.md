# IPL Fantasy Agent - Revised Problem Statement

## Objective

Given the available player pool for a specific IPL match, automatically select the optimal valid 11-player Dream11 team, including Captain and Vice-Captain, that is predicted to maximize total fantasy points.

This is a constrained prediction and optimization problem:

1. Predict expected fantasy points for each available player.
2. Select the highest-value valid XI under Dream11 constraints.
3. Assign Captain and Vice-Captain to maximize upside while balancing consistency and downside risk.

## Dream11 Constraints

The team-selection system must satisfy the standard Dream11 cricket roster constraints:

- Squad size: exactly 11 players
- Budget: maximum 100 credits
- Role mix: 1-4 wicket-keepers, 3-6 batters, 1-4 all-rounders, 3-6 bowlers
- Team cap: maximum 7 players from one real team
- Captain: receives a 2x points multiplier
- Vice-Captain: receives a 1.5x points multiplier

## Scoring System

For IPL, use Dream11's dedicated `TATA IPL Fantasy Cricket Points System`, not the generic Dream11 T20 table.

### Core Batting Points

- Run scored: +1
- Boundary bonus: +1
- Six bonus: +2
- Half-century bonus: +8
- Century bonus: +16
- Dismissal for a duck: -2

Important batting notes:

- A player scoring a century gets the century bonus only and does not also get the half-century bonus.
- If overthrow runs are scored, run points go to the batter on strike.
- If an overthrow reaches the boundary, the batter does not receive an additional boundary bonus for that overthrow.

### Core Bowling Points

- Wicket excluding run-out: +25
- 4-wicket bonus: +8
- 5-wicket bonus: +16
- Maiden over: +8

### Core Fielding Points

- Catch: +8
- Stumping: +12
- Direct-hit run-out: +12
- Run-out, not a direct hit: +6 to thrower and +6 to catcher
- 3-catch bonus: +4

### Other Points

- In announced lineups: +4
- Captain multiplier: 2x
- Vice-Captain multiplier: 1.5x

### Rate Adjustments

#### Economy rate points

Applies to players who bowl at least 2 overs.

- Below 4 runs per over: +6
- 4 to 4.99 runs per over: +4
- 5 to 5.99 runs per over: +2
- 10 to 11 runs per over: -2
- 11.01 to 12 runs per over: -4
- Above 12 runs per over: -6

#### Strike rate points

Applies to non-bowlers who face at least 10 balls.

- Above 170 runs per 100 balls: +6
- 150.01 to 170 runs per 100 balls: +4
- 130 to 150 runs per 100 balls: +2
- 60 to 70 runs per 100 balls: -2
- 50 to 59.99 runs per 100 balls: -4
- Below 50 runs per 100 balls: -6

## Inputs

The system should ingest the following pre-match and near-lock inputs.

### Required inputs

1. Player pool
   The available players for the specific fixture. In practice this may begin as the broader probable squad and then collapse to the final playing group after lineups are announced.

2. Dream11 credit prices
   Match-specific player credit values.

3. Historical IPL performance data
   Per-player, per-match historical data with enough detail to derive Dream11-style fantasy points and features.

4. Match context
   Venue, opponent, pitch type or pitch report, home or away context, weather, and expected innings conditions.

5. Recent form
   Last 3 to 10 matches, with recent matches weighted more heavily than older ones.

6. Playing XI announcements
   The most important final pre-lock signal. The system should sharply re-rank players once confirmed lineups are available.

7. Toss result
   Batting first versus chasing can materially change expected value for top-order batters, death bowlers, and all-rounders.

### Recommended engineered features

- Rolling fantasy points over last 3, 5, and 10 matches
- Exponentially weighted recent form
- Venue-specific batting and bowling performance
- Opponent-specific matchup history
- Batting position expectation
- Bowling quota expectation
- Powerplay, middle overs, and death overs role
- Dismissal mode tendencies
- Catch and stumping involvement rates
- Probability of making the playing XI
- Probability of batting first or bowling first before toss

## Output

For each match, the agent should return:

```text
Team: [11 player names with roles]
Captain: [Player X]
Vice-Captain: [Player Y]
Predicted Points: [Total score]
Confidence: [High / Medium / Low]
```

The system may also optionally return:

- Remaining credits
- Team composition by role
- Players by real team
- Top rejected alternatives
- Key reasons for Captain and Vice-Captain selection

## Core Technical Framing

This is best treated as a three-stage pipeline.

### Step 1: Prediction

Predict each player's expected fantasy output under the current match context.

```text
predict_points(player, match_context) -> expected_points
```

This can be implemented with:

- Gradient boosting models
- Random forests
- Linear baselines with strong feature engineering
- Quantile regression for floor and ceiling estimates
- Ensemble models combining role-specific predictors

The model should ideally predict more than just mean points:

- Expected points
- Floor
- Ceiling
- Variance or uncertainty
- Probability of zero or near-zero score

### Step 2: Constrained Optimization

Use the predicted player values and Dream11 rules to select the optimal valid XI.

```text
select_11(predictions, credits, constraints) -> valid_team
```

This is a constrained combinatorial optimization problem and can be solved with:

- PuLP
- OR-Tools
- SciPy mixed-integer optimization tooling

The optimizer should enforce:

- Exact XI size
- Budget ceiling
- Role bounds
- Max 7 from one team
- Optional lock or exclude controls

### Step 3: Captaincy Selection

Choose Captain and Vice-Captain from the selected XI.

```text
select_c_vc(team, predictions, risk_profile) -> (captain, vice_captain)
```

Naively, this is the top two projected scorers. A better version should consider:

- Ceiling
- Role stability
- Variance
- Toss dependence
- Probability of batting high enough or bowling enough overs

## Evaluation

### Primary evaluation

The most important evaluation should be post-match team quality, not only per-player regression error.

1. Oracle team score
   After the match, compute the best possible valid Dream11 XI using actual player fantasy points.

2. Agent score
   Score the XI selected by the agent using actual realized fantasy points.

3. Regret

```text
regret = oracle_score - agent_score
```

Lower regret is better.

4. Captaincy regret
   Compare the chosen Captain and Vice-Captain against the best possible Captain and Vice-Captain assignments under realized outcomes.

5. Benchmark comparison
   Compare the agent team against:

- your manually chosen team
- a friend or public benchmark team
- any best-known team posted after the match

### Secondary evaluation

Secondary metrics are still useful for model debugging:

- RMSE on per-player fantasy point predictions
- MAE on per-player fantasy point predictions
- Calibration of lineup probability
- Calibration of floor and ceiling estimates

## Ground Truth for Backtesting

For every completed match, store:

- Actual fantasy points for each available player
- The highest-scoring valid XI after the match
- The best possible Captain and Vice-Captain assignment
- The agent's team score
- Your manual team score, if available
- Any benchmark or public team score, if available

This produces a much stronger evaluation loop than only comparing predicted versus actual player points.

## Major Challenges

### Playing XI uncertainty

Before final lineups, some players may not start. The agent should model lineup probability and then switch to deterministic availability once the playing XI is announced.

### Toss dependency

Toss can change projected value significantly:

- top-order batters may gain or lose value depending on chasing context
- death bowlers can become more valuable under specific innings scenarios
- some all-rounders have asymmetric value batting first versus bowling first

### Small sample sizes

New players, role changes, new venues, and season transitions make historical data noisy. The system must be robust to sparse histories.

### Role ambiguity

Some players are labeled as all-rounders but may not reliably bowl. Others may float in batting order. Fantasy value depends on actual usage, not only nominal category.

### Safe picks versus differentiation

Pure expected-point maximization is suitable for baseline optimization, but contests may reward differentiated lineups. A later version of the system may optimize for contest strategy, not just expected value.

## Data Requirements

Cricsheet is useful as raw structured cricket data, but it is not ideal if the immediate goal is a clean historical fantasy-modeling dataset. Prefer a source that is easier to load into tabular training data.

### Recommended historical dataset options

1. Kaggle - IPL Ball By Ball 2008 to 2024
   Useful for long-range ball-by-ball historical modeling in CSV form.

2. Kaggle - IPL Ball-By-Ball Dataset 2020 to 2025
   Useful if you want a recent-season focused dataset.

3. GitHub - ritesh-ojha/IPL-DATASET
   Useful as a lightweight CSV plus JSON source for local experimentation.

4. GitHub - saikat-7/IPL-Match-Data-2008-2023
   Especially useful because it appears to include ball-by-ball data, batting cards, bowling cards, partnership data, and player information.

5. GitHub - imran789924/IPL-statistics
   Useful as an additional historical delivery-level source.

### Preferred data architecture

Build the project around three layers:

1. Raw layer
   Historical ball-by-ball, scorecards, squads, venue information, and pricing snapshots.

2. Feature layer
   Per-player, per-match derived statistics and fantasy features.

3. Serving layer
   Match-day player pool, latest credits, lineup status, toss, context, and predictions.

## Recommended System Behavior

### Pre-toss mode

Generate a provisional team using expected starting players and a toss-agnostic or toss-probabilistic forecast.

### Post-toss mode

Update projections with toss result and innings order.

### Post-lineup mode

Re-run optimization immediately after confirmed playing XI announcements. This should be treated as the final and highest-confidence version before lock.

## Confidence Labeling

The output confidence should be based on the quality and completeness of information available.

- High
  Confirmed playing XIs, toss known, stable player roles, good historical coverage

- Medium
  Toss known but lineups partly uncertain, or several players have unstable roles

- Low
  Pre-toss, lineup uncertainty is high, recent role changes are unresolved, or multiple players have weak sample histories

## Success Definition

The agent is successful if it consistently:

- beats your manual team over a season
- reduces regret versus the oracle best-valid XI
- makes strong Captain and Vice-Captain decisions
- adapts correctly to toss and lineup information

The season-level output should emphasize:

- cumulative agent score
- cumulative oracle gap
- win-loss record versus your team
- best and worst captaincy decisions
- short weekly summaries suitable for posting publicly

## Sources Used for This Revision

- Dream11 fantasy cricket rules and role constraints
- Dream11 official fantasy cricket points system
- Dream11 current generic cricket how-to-play page
- ICC Dream11 explainer for historical confirmation of common roster structure
- Kaggle and GitHub IPL dataset listings used to identify alternatives to Cricsheet

## Final Summary

This project should be framed as an IPL fantasy decision engine with:

- player-level point prediction
- hard-constraint XI optimization
- Captain and Vice-Captain leverage selection
- post-match oracle-gap evaluation

The strongest version of the system is not just a fantasy point predictor. It is a full decision pipeline that ingests match context, reacts to toss and confirmed lineups, and selects the highest-value valid team under Dream11 IPL rules.

"""
Backtest all 4 match prediction models on IPL 2025 data.
Produces detailed match-by-match results and aggregate metrics.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, log_loss

from .data_prep import (
    load_ipl_data,
    load_feature_data,
    load_venue_profiles,
    build_match_level_data,
    build_team_rolling_features,
    merge_venue_features,
)
from .player_performance import (
    train_runs_model,
    train_wickets_model,
    prepare_player_data,
    RUNS_FEATURES,
    WICKETS_FEATURES,
)
from .team_score import (
    build_player_strength_features,
    build_innings_dataset,
    train_team_score_model,
    SCORE_FEATURES,
)
from .match_winner import (
    build_match_winner_dataset,
    train_match_winner_model,
    WINNER_FEATURES,
)
from .victory_margin import (
    build_margin_dataset,
    train_margin_model,
    MARGIN_FEATURES,
)

TEST_SEASON = "2025"


def run_backtest():
    # ── Load & prep ──────────────────────────────────────────────
    print("Loading data...")
    player_df = load_ipl_data()
    feature_df = load_feature_data()
    venue_df = load_venue_profiles()

    match_df = build_match_level_data(player_df)
    match_df = merge_venue_features(match_df, venue_df)
    team_rolling = build_team_rolling_features(match_df)
    player_strength = build_player_strength_features(feature_df)

    test_matches = match_df[match_df["season"].astype(str) == TEST_SEASON].copy()
    print(f"IPL 2025: {len(test_matches)} matches to backtest\n")

    # ── Train models (on pre-2025 data) ─────────────────────────
    print("=" * 70)
    print("TRAINING MODELS (2017-2024)")
    print("=" * 70)

    model_runs, _ = train_runs_model(feature_df, TEST_SEASON)
    model_wickets, _ = train_wickets_model(feature_df, TEST_SEASON)

    innings_df = build_innings_dataset(match_df, team_rolling, player_strength)
    model_score, _ = train_team_score_model(innings_df, TEST_SEASON)

    winner_df = build_match_winner_dataset(match_df, team_rolling, player_strength)
    model_winner, _ = train_match_winner_model(winner_df, TEST_SEASON)

    margin_full = build_margin_dataset(match_df, winner_df)
    model_margin, _ = train_margin_model(margin_full, TEST_SEASON)

    # ── Backtest: Player Performance ─────────────────────────────
    print("\n\n" + "=" * 70)
    print("BACKTEST RESULTS — IPL 2025 (74 matches)")
    print("=" * 70)

    pf = prepare_player_data(feature_df)
    test_players = pf[pf["season"].astype(str) == TEST_SEASON].copy()
    test_players["pred_runs"] = np.maximum(
        model_runs.predict(test_players[RUNS_FEATURES].fillna(0)), 0
    )
    test_players["pred_wickets"] = np.maximum(
        model_wickets.predict(test_players[WICKETS_FEATURES].fillna(0)), 0
    )

    print("\n┌─────────────────────────────────────────────────────┐")
    print("│  MODEL 1: Player Runs Prediction                    │")
    print("└─────────────────────────────────────────────────────┘")
    runs_mae = mean_absolute_error(test_players["runs"], test_players["pred_runs"])
    runs_rmse = np.sqrt(mean_squared_error(test_players["runs"], test_players["pred_runs"]))
    baseline_runs_mae = mean_absolute_error(
        test_players["runs"], [test_players["runs"].mean()] * len(test_players)
    )
    print(f"  Players evaluated: {len(test_players)}")
    print(f"  MAE:      {runs_mae:.2f} runs")
    print(f"  RMSE:     {runs_rmse:.2f} runs")
    print(f"  Baseline: {baseline_runs_mae:.2f} runs (predict mean)")
    print(f"  Improvement: {(1 - runs_mae / baseline_runs_mae) * 100:.1f}%")

    # Top predicted run scorers vs actual
    top_pred = test_players.groupby("player_name").agg(
        total_pred_runs=("pred_runs", "sum"),
        total_actual_runs=("runs", "sum"),
        matches=("match_id", "count"),
    ).sort_values("total_pred_runs", ascending=False).head(15)
    print("\n  Top 15 Predicted Run Scorers vs Actual (season total):")
    print(f"  {'Player':<25s}  {'Pred':>6s}  {'Actual':>6s}  {'Matches':>7s}  {'Err':>6s}")
    print("  " + "-" * 55)
    for name, row in top_pred.iterrows():
        err = row["total_pred_runs"] - row["total_actual_runs"]
        print(f"  {name:<25s}  {row['total_pred_runs']:>6.0f}  {int(row['total_actual_runs']):>6d}  {int(row['matches']):>7d}  {err:>+6.0f}")

    print("\n┌─────────────────────────────────────────────────────┐")
    print("│  MODEL 2: Player Wickets Prediction                 │")
    print("└─────────────────────────────────────────────────────┘")
    wkts_mae = mean_absolute_error(test_players["wickets"], test_players["pred_wickets"])
    wkts_rmse = np.sqrt(mean_squared_error(test_players["wickets"], test_players["pred_wickets"]))
    baseline_wkts_mae = mean_absolute_error(
        test_players["wickets"], [test_players["wickets"].mean()] * len(test_players)
    )
    print(f"  Players evaluated: {len(test_players)}")
    print(f"  MAE:      {wkts_mae:.2f} wickets")
    print(f"  RMSE:     {wkts_rmse:.2f} wickets")
    print(f"  Baseline: {baseline_wkts_mae:.2f} wickets (predict mean)")
    print(f"  Improvement: {(1 - wkts_mae / baseline_wkts_mae) * 100:.1f}%")

    top_bowlers = test_players.groupby("player_name").agg(
        total_pred_wkts=("pred_wickets", "sum"),
        total_actual_wkts=("wickets", "sum"),
        matches=("match_id", "count"),
    ).sort_values("total_pred_wkts", ascending=False).head(15)
    print("\n  Top 15 Predicted Wicket Takers vs Actual (season total):")
    print(f"  {'Player':<25s}  {'Pred':>6s}  {'Actual':>6s}  {'Matches':>7s}  {'Err':>6s}")
    print("  " + "-" * 55)
    for name, row in top_bowlers.iterrows():
        err = row["total_pred_wkts"] - row["total_actual_wkts"]
        print(f"  {name:<25s}  {row['total_pred_wkts']:>6.1f}  {int(row['total_actual_wkts']):>6d}  {int(row['matches']):>7d}  {err:>+6.1f}")

    # ── Backtest: Team Score ─────────────────────────────────────
    print("\n┌─────────────────────────────────────────────────────┐")
    print("│  MODEL 3: Team Score Prediction                     │")
    print("└─────────────────────────────────────────────────────┘")
    test_innings = innings_df[innings_df["season"].astype(str) == TEST_SEASON].copy()
    test_innings = test_innings.dropna(subset=["rolling_runs_scored_avg_5"])
    test_innings["pred_score"] = np.maximum(
        model_score.predict(test_innings[SCORE_FEATURES].fillna(0)), 50
    )
    score_mae = mean_absolute_error(test_innings["runs_scored"], test_innings["pred_score"])
    score_rmse = np.sqrt(mean_squared_error(test_innings["runs_scored"], test_innings["pred_score"]))
    baseline_score_mae = mean_absolute_error(
        test_innings["runs_scored"], [test_innings["runs_scored"].mean()] * len(test_innings)
    )
    print(f"  Innings evaluated: {len(test_innings)}")
    print(f"  MAE:      {score_mae:.2f} runs")
    print(f"  RMSE:     {score_rmse:.2f} runs")
    print(f"  Baseline: {baseline_score_mae:.2f} runs (predict mean)")

    # Breakdown by innings
    for inn in [1, 2]:
        sub = test_innings[test_innings["innings"] == inn]
        if len(sub) > 0:
            mae_inn = mean_absolute_error(sub["runs_scored"], sub["pred_score"])
            print(f"  Innings {inn} MAE: {mae_inn:.2f} ({len(sub)} innings)")

    # Error distribution
    test_innings["error"] = test_innings["pred_score"] - test_innings["runs_scored"]
    print(f"\n  Error distribution:")
    print(f"    Mean error (bias): {test_innings['error'].mean():+.1f} runs")
    print(f"    Std of error:      {test_innings['error'].std():.1f} runs")
    print(f"    Within 20 runs:    {(test_innings['error'].abs() <= 20).mean()*100:.0f}%")
    print(f"    Within 30 runs:    {(test_innings['error'].abs() <= 30).mean()*100:.0f}%")
    print(f"    Within 40 runs:    {(test_innings['error'].abs() <= 40).mean()*100:.0f}%")

    # Match-by-match team scores
    print("\n  All match innings predictions:")
    print(f"  {'Date':<12s} {'Team':>28s}  Inn  Pred  Actual  Error")
    print("  " + "-" * 75)
    for mid in test_innings["match_id"].unique():
        m = test_innings[test_innings["match_id"] == mid].sort_values("innings")
        for _, row in m.iterrows():
            err = row["pred_score"] - row["runs_scored"]
            print(
                f"  {str(row['match_date'])[:10]:<12s} {row['team']:>28s}  "
                f"{int(row['innings']):>3d}  {row['pred_score']:>5.0f}  "
                f"{int(row['runs_scored']):>6d}  {err:>+6.0f}"
            )

    # ── Backtest: Match Winner ───────────────────────────────────
    print("\n┌─────────────────────────────────────────────────────┐")
    print("│  MODEL 4: Match Winner Prediction                   │")
    print("└─────────────────────────────────────────────────────┘")
    test_winner = winner_df[winner_df["season"].astype(str) == TEST_SEASON].copy()
    test_winner = test_winner.dropna(subset=["t1_rolling_won_avg_5"])
    test_winner = test_winner[test_winner["winner"].notna()]

    X_test = test_winner[WINNER_FEATURES].fillna(0)
    test_winner["pred_bat_first_wins"] = model_winner.predict(X_test)
    test_winner["prob_bat_first"] = model_winner.predict_proba(X_test)[:, 1]
    test_winner["pred_winner"] = np.where(
        test_winner["prob_bat_first"] > 0.5,
        test_winner["team_bat_first"],
        test_winner["team_bat_second"],
    )
    test_winner["correct"] = (test_winner["pred_winner"] == test_winner["winner"]).astype(int)

    acc = test_winner["correct"].mean()
    baseline_acc = max(
        (test_winner["winner"] == test_winner["team_bat_first"]).mean(),
        (test_winner["winner"] == test_winner["team_bat_second"]).mean(),
    )
    try:
        ll = log_loss(test_winner["bat_first_won"], test_winner["prob_bat_first"])
    except Exception:
        ll = float("nan")

    print(f"  Matches evaluated: {len(test_winner)}")
    print(f"  Accuracy:   {acc:.3f} ({int(acc * len(test_winner))}/{len(test_winner)})")
    print(f"  Baseline:   {baseline_acc:.3f} (always pick bat-second)")
    print(f"  Log Loss:   {ll:.3f}")

    # Confidence calibration
    print(f"\n  Confidence calibration:")
    for threshold in [0.55, 0.60, 0.65, 0.70]:
        confident = test_winner[
            (test_winner["prob_bat_first"] > threshold)
            | (test_winner["prob_bat_first"] < (1 - threshold))
        ]
        if len(confident) > 0:
            conf_acc = confident["correct"].mean()
            print(f"    P > {threshold:.0%}: {conf_acc:.0%} accuracy ({len(confident)} matches)")

    # Match-by-match
    print(f"\n  Match-by-match predictions:")
    print(
        f"  {'Date':<12s} {'Bat First':>25s} vs {'Bat Second':<25s}  "
        f"{'P(BF)':>5s}  {'Predicted':>25s}  {'Actual':>25s}  {'':>2s}"
    )
    print("  " + "-" * 135)
    for _, row in test_winner.sort_values("match_date").iterrows():
        mark = "OK" if row["correct"] else "X"
        print(
            f"  {str(row['match_date'])[:10]:<12s} {row['team_bat_first']:>25s} vs "
            f"{row['team_bat_second']:<25s}  {row['prob_bat_first']:>5.2f}  "
            f"{row['pred_winner']:>25s}  {row['winner']:>25s}  {mark:>2s}"
        )

    # ── Backtest: Victory Margin ─────────────────────────────────
    print("\n┌─────────────────────────────────────────────────────┐")
    print("│  MODEL 5: Victory Margin Prediction                 │")
    print("└─────────────────────────────────────────────────────┘")
    test_margin = margin_full[margin_full["season"].astype(str) == TEST_SEASON].copy()
    test_margin = test_margin.dropna(subset=["t1_rolling_won_avg_5"])
    test_margin = test_margin[test_margin["winner"].notna()]
    test_margin = test_margin[test_margin["signed_margin"] != 0]

    X_test_m = test_margin[MARGIN_FEATURES].fillna(0)
    test_margin["pred_margin"] = model_margin.predict(X_test_m)

    margin_mae = mean_absolute_error(test_margin["signed_margin"], test_margin["pred_margin"])
    margin_rmse = np.sqrt(mean_squared_error(test_margin["signed_margin"], test_margin["pred_margin"]))
    baseline_margin_mae = mean_absolute_error(
        test_margin["signed_margin"],
        [test_margin["signed_margin"].mean()] * len(test_margin),
    )

    # Direction accuracy: does the sign match?
    direction_correct = (
        (test_margin["pred_margin"] > 0) == (test_margin["signed_margin"] > 0)
    ).mean()

    print(f"  Matches evaluated: {len(test_margin)}")
    print(f"  MAE:      {margin_mae:.2f} runs")
    print(f"  RMSE:     {margin_rmse:.2f} runs")
    print(f"  Baseline: {baseline_margin_mae:.2f} runs (predict mean)")
    print(f"  Direction accuracy (sign match): {direction_correct:.3f}")

    # Error distribution
    test_margin["margin_error"] = test_margin["pred_margin"] - test_margin["signed_margin"]
    print(f"\n  Error distribution:")
    print(f"    Mean error (bias): {test_margin['margin_error'].mean():+.1f} runs")
    print(f"    Within 15 runs:    {(test_margin['margin_error'].abs() <= 15).mean()*100:.0f}%")
    print(f"    Within 25 runs:    {(test_margin['margin_error'].abs() <= 25).mean()*100:.0f}%")
    print(f"    Within 40 runs:    {(test_margin['margin_error'].abs() <= 40).mean()*100:.0f}%")

    # Match-by-match margins
    print(f"\n  Match-by-match margin predictions:")
    print(
        f"  {'Date':<12s} {'Bat First':>25s} vs {'Bat Second':<25s}  "
        f"{'Pred':>6s}  {'Actual':>6s}  {'Error':>6s}  {'Dir':>3s}"
    )
    print("  " + "-" * 100)
    for _, row in test_margin.sort_values("match_date").iterrows():
        err = row["pred_margin"] - row["signed_margin"]
        dir_ok = "OK" if (row["pred_margin"] > 0) == (row["signed_margin"] > 0) else "X"
        print(
            f"  {str(row['match_date'])[:10]:<12s} {row['team_bat_first']:>25s} vs "
            f"{row['team_bat_second']:<25s}  {row['pred_margin']:>+6.0f}  "
            f"{int(row['signed_margin']):>+6d}  {err:>+6.0f}  {dir_ok:>3s}"
        )

    # ── Summary ──────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  BACKTEST SUMMARY — IPL 2025")
    print("=" * 70)
    print(f"""
  ┌──────────────────────┬───────────┬───────────┬────────────┐
  │ Model                │ Metric    │ Score     │ Baseline   │
  ├──────────────────────┼───────────┼───────────┼────────────┤
  │ Player Runs          │ MAE       │ {runs_mae:>7.2f}   │ {baseline_runs_mae:>8.2f}   │
  │ Player Wickets       │ MAE       │ {wkts_mae:>7.2f}   │ {baseline_wkts_mae:>8.2f}   │
  │ Team Score           │ MAE       │ {score_mae:>7.2f}   │ {baseline_score_mae:>8.2f}   │
  │ Match Winner         │ Accuracy  │ {acc:>7.1%}   │ {baseline_acc:>8.1%}   │
  │ Victory Margin       │ MAE       │ {margin_mae:>7.2f}   │ {baseline_margin_mae:>8.2f}   │
  │ Victory Margin       │ Direction │ {direction_correct:>7.1%}   │    50.0%   │
  └──────────────────────┴───────────┴───────────┴────────────┘
    """)


if __name__ == "__main__":
    run_backtest()

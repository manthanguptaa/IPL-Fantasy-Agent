"""
Model 1: Player Performance Predictor
Predicts individual player runs scored and wickets taken in a match.
Uses the existing rolling features from the curated feature set.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path

from .data_prep import load_feature_data, load_venue_profiles

# Features for runs prediction (batting-focused)
RUNS_FEATURES = [
    "prior_matches_all",
    "prior_matches_ipl",
    "rolling_runs_avg_3_all",
    "rolling_runs_avg_5_all",
    "rolling_points_avg_10_all",
    "rolling_batting_points_avg_5_all",
    "rolling_balls_faced_avg_5_all",
    "rolling_strike_rate_5_all",
    "batting_match_rate_5_all",
    "venue_points_avg_all",
    "opponent_points_avg_all",
    "prior_matches_at_venue",
    "points_trend_3_vs_10_all",
]

# Features for wickets prediction (bowling-focused)
WICKETS_FEATURES = [
    "prior_matches_all",
    "prior_matches_ipl",
    "rolling_wickets_avg_3_all",
    "rolling_wickets_avg_5_all",
    "rolling_points_avg_10_all",
    "rolling_bowling_points_avg_5_all",
    "rolling_balls_bowled_avg_5_all",
    "rolling_economy_rate_5_all",
    "bowling_match_rate_5_all",
    "venue_points_avg_all",
    "opponent_points_avg_all",
    "prior_matches_at_venue",
    "points_trend_3_vs_10_all",
]


def prepare_player_data(feature_df: pd.DataFrame):
    """Prepare train/test splits for player prediction models."""
    df = feature_df.copy()

    # Filter to players who were in the playing XI
    df = df[df["playing_xi"] == 1].copy()

    # Drop rows with insufficient history (need at least some rolling features)
    df = df[df["prior_matches_all"] >= 3].copy()

    return df


def train_runs_model(df: pd.DataFrame, test_season: str = "2025"):
    """Train a model to predict individual player runs."""
    df = prepare_player_data(df)

    # Split by season
    train = df[df["season"].astype(str) < test_season].copy()
    test = df[df["season"].astype(str) == test_season].copy()

    # Filter to players who batted (balls_faced > 0) for training
    # But predict for all players at inference time
    train_bat = train[train["balls_faced"] > 0].copy()

    X_train = train_bat[RUNS_FEATURES].fillna(0)
    y_train = train_bat["runs"]
    X_test = test[RUNS_FEATURES].fillna(0)
    y_test = test["runs"]

    model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=20,
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    preds = np.maximum(preds, 0)  # runs can't be negative

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    print(f"Runs Model — MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    print(f"  Baseline (predict mean): MAE={mean_absolute_error(y_test, [y_test.mean()]*len(y_test)):.2f}")
    print(f"  Train: {len(train_bat)} rows, Test: {len(test)} rows")

    # Feature importance
    feat_imp = pd.Series(model.feature_importances_, index=RUNS_FEATURES).sort_values(ascending=False)
    print(f"\n  Top features:\n{feat_imp.head(5).to_string()}")

    return model, {"mae": mae, "rmse": rmse, "test_preds": preds, "test_actual": y_test.values}


def train_wickets_model(df: pd.DataFrame, test_season: str = "2025"):
    """Train a model to predict individual player wickets taken."""
    df = prepare_player_data(df)

    train = df[df["season"].astype(str) < test_season].copy()
    test = df[df["season"].astype(str) == test_season].copy()

    # Filter to players who bowled for training
    train_bowl = train[train["balls_bowled"] > 0].copy()

    X_train = train_bowl[WICKETS_FEATURES].fillna(0)
    y_train = train_bowl["wickets"]
    X_test = test[WICKETS_FEATURES].fillna(0)
    y_test = test["wickets"]

    model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=20,
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    preds = np.maximum(preds, 0)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    print(f"\nWickets Model — MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    print(f"  Baseline (predict mean): MAE={mean_absolute_error(y_test, [y_test.mean()]*len(y_test)):.2f}")
    print(f"  Train: {len(train_bowl)} rows, Test: {len(test)} rows")

    feat_imp = pd.Series(model.feature_importances_, index=WICKETS_FEATURES).sort_values(ascending=False)
    print(f"\n  Top features:\n{feat_imp.head(5).to_string()}")

    return model, {"mae": mae, "rmse": rmse, "test_preds": preds, "test_actual": y_test.values}


def evaluate_predictions(model_runs, model_wickets, df: pd.DataFrame, test_season: str = "2025"):
    """Show sample predictions for a few well-known players."""
    df = prepare_player_data(df)
    test = df[df["season"].astype(str) == test_season].copy()

    test["pred_runs"] = np.maximum(model_runs.predict(test[RUNS_FEATURES].fillna(0)), 0)
    test["pred_wickets"] = np.maximum(model_wickets.predict(test[WICKETS_FEATURES].fillna(0)), 0)

    # Show top predicted run scorers per match (sample last 3 matches)
    last_matches = test["match_id"].unique()[-3:]
    for mid in last_matches:
        m = test[test["match_id"] == mid].copy()
        print(f"\n--- Match {mid} ({m['match_date'].iloc[0].date()}) ---")
        print(f"    {m['team'].unique()[0]} vs {m['opponent'].unique()[0]}")

        # Top 5 predicted run scorers
        top_bat = m.nlargest(5, "pred_runs")[["player_name", "team", "pred_runs", "runs"]]
        print(f"\n  Top predicted run scorers:")
        for _, row in top_bat.iterrows():
            print(f"    {row['player_name']:25s} ({row['team'][:3]}) — pred: {row['pred_runs']:.1f}, actual: {row['runs']}")

        # Top 5 predicted wicket takers
        top_bowl = m.nlargest(5, "pred_wickets")[["player_name", "team", "pred_wickets", "wickets"]]
        print(f"\n  Top predicted wicket takers:")
        for _, row in top_bowl.iterrows():
            print(f"    {row['player_name']:25s} ({row['team'][:3]}) — pred: {row['pred_wickets']:.1f}, actual: {int(row['wickets'])}")


if __name__ == "__main__":
    print("=" * 60)
    print("Player Performance Prediction Model")
    print("=" * 60)

    df = load_feature_data()
    print(f"Loaded {len(df)} player-match rows\n")

    model_runs, runs_metrics = train_runs_model(df)
    model_wickets, wickets_metrics = train_wickets_model(df)

    print("\n" + "=" * 60)
    print("Sample Predictions (IPL 2025)")
    print("=" * 60)
    evaluate_predictions(model_runs, model_wickets, df)

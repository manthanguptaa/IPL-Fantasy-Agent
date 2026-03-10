"""
Model 3: Match Winner Predictor
Predicts which team will win a match.
Uses team form, player strength, venue, and toss information.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss, classification_report

from .data_prep import (
    load_ipl_data,
    load_feature_data,
    load_venue_profiles,
    build_match_level_data,
    build_team_rolling_features,
    build_season_avg_scores,
    merge_venue_features,
)
from .team_score import build_player_strength_features


def build_match_winner_dataset(
    match_df: pd.DataFrame,
    team_rolling: pd.DataFrame,
    player_strength: pd.DataFrame | None = None,
    season_avg: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build a dataset where each row is a match.
    Features compare team1 (batting first) vs team2 (batting second).
    Target: 1 if team batting first wins, 0 otherwise.
    """
    df = match_df.copy()

    # Rolling features for team batting first
    rolling_cols = [
        "rolling_runs_scored_avg_5",
        "rolling_runs_conceded_avg_5",
        "rolling_won_avg_5",
        "rolling_run_diff_avg_5",
    ]

    t1_feats = team_rolling[["match_id", "team"] + rolling_cols].copy()
    t1_feats = t1_feats.rename(columns={c: f"t1_{c}" for c in rolling_cols})
    t1_feats = t1_feats.rename(columns={"team": "team_bat_first"})
    df = df.merge(t1_feats, on=["match_id", "team_bat_first"], how="left")

    t2_feats = team_rolling[["match_id", "team"] + rolling_cols].copy()
    t2_feats = t2_feats.rename(columns={c: f"t2_{c}" for c in rolling_cols})
    t2_feats = t2_feats.rename(columns={"team": "team_bat_second"})
    df = df.merge(t2_feats, on=["match_id", "team_bat_second"], how="left")

    # Player strength features
    if player_strength is not None:
        ps_cols = [c for c in player_strength.columns if c not in ("match_id", "team")]

        t1_ps = player_strength.copy()
        t1_ps = t1_ps.rename(columns={c: f"t1_{c}" for c in ps_cols})
        t1_ps = t1_ps.rename(columns={"team": "team_bat_first"})
        df = df.merge(t1_ps, on=["match_id", "team_bat_first"], how="left")

        t2_ps = player_strength.copy()
        t2_ps = t2_ps.rename(columns={c: f"t2_{c}" for c in ps_cols})
        t2_ps = t2_ps.rename(columns={"team": "team_bat_second"})
        df = df.merge(t2_ps, on=["match_id", "team_bat_second"], how="left")

    # Derived comparison features
    df["form_diff"] = df["t1_rolling_won_avg_5"].fillna(0.5) - df["t2_rolling_won_avg_5"].fillna(0.5)
    df["run_diff_diff"] = (
        df["t1_rolling_run_diff_avg_5"].fillna(0) - df["t2_rolling_run_diff_avg_5"].fillna(0)
    )
    df["scoring_diff"] = (
        df["t1_rolling_runs_scored_avg_5"].fillna(160) - df["t2_rolling_runs_scored_avg_5"].fillna(160)
    )

    if player_strength is not None:
        df["batting_strength_diff"] = (
            df["t1_team_sum_batting_pts"].fillna(0) - df["t2_team_sum_batting_pts"].fillna(0)
        )
        df["bowling_strength_diff"] = (
            df["t1_team_sum_bowling_pts"].fillna(0) - df["t2_team_sum_bowling_pts"].fillna(0)
        )
        df["experience_diff"] = (
            df["t1_team_experience"].fillna(0) - df["t2_team_experience"].fillna(0)
        )
        df["overall_strength_diff"] = (
            df["t1_team_avg_points"].fillna(0) - df["t2_team_avg_points"].fillna(0)
        )

    # Toss features
    df["toss_won_bat_first"] = (df["toss_winner"] == df["team_bat_first"]).astype(int)

    # Season scoring inflation
    if season_avg is not None:
        df = df.merge(season_avg, on="season", how="left")

    return df


WINNER_FEATURES = [
    # Venue
    "avg_total_runs",
    "batting_friendly",
    "dew_factor",
    # Team1 (bat first) form
    "t1_rolling_won_avg_5",
    "t1_rolling_run_diff_avg_5",
    "t1_rolling_runs_scored_avg_5",
    # Team2 (bat second) form
    "t2_rolling_won_avg_5",
    "t2_rolling_run_diff_avg_5",
    "t2_rolling_runs_scored_avg_5",
    # Comparison features
    "form_diff",
    "run_diff_diff",
    "scoring_diff",
    # Player strength comparisons
    "batting_strength_diff",
    "bowling_strength_diff",
    "experience_diff",
    "overall_strength_diff",
    # Toss
    "toss_won_bat_first",
    # Season inflation
    "season_avg_score_lag",
]


def train_match_winner_model(df: pd.DataFrame, test_season: str = "2025"):
    """Train model to predict match winner (1 = bat first wins)."""
    # Drop no-result / tied matches
    df = df.dropna(subset=["bat_first_won", "t1_rolling_won_avg_5"]).copy()
    df = df[df["winner"].notna()].copy()

    train = df[df["season"].astype(str) < test_season]
    test = df[df["season"].astype(str) == test_season]

    X_train = train[WINNER_FEATURES].fillna(0)
    y_train = train["bat_first_won"]
    X_test = test[WINNER_FEATURES].fillna(0)
    y_test = test["bat_first_won"]

    model = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=20,
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    ll = log_loss(y_test, probs)
    baseline_acc = max(y_test.mean(), 1 - y_test.mean())

    print(f"Match Winner Model — Accuracy: {acc:.3f}, Log Loss: {ll:.3f}")
    print(f"  Baseline (majority class): {baseline_acc:.3f}")
    print(f"  Train: {len(train)} matches, Test: {len(test)} matches")

    feat_imp = pd.Series(model.feature_importances_, index=WINNER_FEATURES).sort_values(ascending=False)
    print(f"\n  Top features:\n{feat_imp.head(7).to_string()}")

    return model, {
        "accuracy": acc,
        "log_loss": ll,
        "test_preds": preds,
        "test_probs": probs,
        "test_actual": y_test.values,
        "test_df": test,
    }


def show_predictions(results: dict):
    """Display sample match predictions with probabilities."""
    test = results["test_df"].copy()
    test["pred_winner_is_bat_first"] = results["test_preds"]
    test["prob_bat_first_wins"] = results["test_probs"]

    print("\n  Sample Match Predictions (last 15 matches):")
    print(f"  {'Bat First':>25s} vs {'Bat Second':<25s}  P(BF)  Pred Winner            Actual Winner          {'':>2s}")
    print("  " + "-" * 115)

    last_matches = test["match_id"].unique()[-15:]
    correct = 0
    total = 0
    for mid in last_matches:
        row = test[test["match_id"] == mid].iloc[0]
        prob = row["prob_bat_first_wins"]
        pred_team = row["team_bat_first"] if prob > 0.5 else row["team_bat_second"]
        actual = row["winner"]
        mark = "OK" if pred_team == actual else "X"
        if pred_team == actual:
            correct += 1
        total += 1
        print(
            f"  {row['team_bat_first']:>25s} vs {row['team_bat_second']:<25s}  "
            f"{prob:.2f}   {pred_team:<25s}  {actual:<25s} {mark}"
        )

    print(f"\n  Last {total} matches: {correct}/{total} correct ({correct/total*100:.0f}%)")


if __name__ == "__main__":
    print("=" * 60)
    print("Match Winner Prediction Model")
    print("=" * 60)

    player_df = load_ipl_data()
    feature_df = load_feature_data()
    venue_df = load_venue_profiles()

    match_df = build_match_level_data(player_df)
    match_df = merge_venue_features(match_df, venue_df)
    team_rolling = build_team_rolling_features(match_df)
    player_strength = build_player_strength_features(feature_df)

    season_avg = build_season_avg_scores(match_df)
    df = build_match_winner_dataset(match_df, team_rolling, player_strength, season_avg)
    print(f"Built {len(df)} match rows\n")

    model, results = train_match_winner_model(df)
    show_predictions(results)

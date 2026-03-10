"""
Model 2: Team Total Score Predictor
Predicts the total runs a team will score in their innings.
Uses venue characteristics, team rolling form, toss info, and aggregated player strength.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .data_prep import (
    load_ipl_data,
    load_feature_data,
    load_venue_profiles,
    build_match_level_data,
    build_team_rolling_features,
    build_season_avg_scores,
    merge_venue_features,
)


def build_player_strength_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate player-level rolling features per team per match.
    This gives the model a sense of team batting/bowling strength based on
    the actual players in the XI.
    """
    df = feature_df[feature_df["playing_xi"] == 1].copy()

    agg = (
        df.groupby(["match_id", "team"])
        .agg(
            team_avg_batting_pts=("rolling_batting_points_avg_5_all", "mean"),
            team_sum_batting_pts=("rolling_batting_points_avg_5_all", "sum"),
            team_avg_bowling_pts=("rolling_bowling_points_avg_5_all", "mean"),
            team_sum_bowling_pts=("rolling_bowling_points_avg_5_all", "sum"),
            team_avg_runs_form=("rolling_runs_avg_5_all", "mean"),
            team_sum_runs_form=("rolling_runs_avg_5_all", "sum"),
            team_avg_sr=("rolling_strike_rate_5_all", "mean"),
            team_avg_economy=("rolling_economy_rate_5_all", "mean"),
            team_avg_points=("rolling_points_avg_10_all", "mean"),
            team_max_batting=("rolling_batting_points_avg_5_all", "max"),
            team_experience=("prior_matches_ipl", "mean"),
        )
        .reset_index()
    )
    return agg


def build_innings_dataset(
    match_df: pd.DataFrame,
    team_rolling: pd.DataFrame,
    player_strength: pd.DataFrame | None = None,
    season_avg: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build a dataset where each row is one team's innings in a match.
    Target: runs scored by that team.
    """
    rows = []
    for _, match in match_df.iterrows():
        for innings_num, (team_col, opp_col, score_col, opp_score_col) in enumerate(
            [
                ("team_bat_first", "team_bat_second", "score_bat_first", "score_bat_second"),
                ("team_bat_second", "team_bat_first", "score_bat_second", "score_bat_first"),
            ],
            start=1,
        ):
            rows.append(
                {
                    "match_id": match["match_id"],
                    "match_date": match["match_date"],
                    "season": match["season"],
                    "venue": match["venue"],
                    "team": match[team_col],
                    "opponent": match[opp_col],
                    "innings": innings_num,
                    "runs_scored": match[score_col],
                    "opponent_score": match[opp_score_col],
                    "toss_winner": match["toss_winner"],
                    "toss_decision": match["toss_decision"],
                    # Venue features
                    "avg_total_runs": match.get("avg_total_runs", np.nan),
                    "avg_total_sixes": match.get("avg_total_sixes", np.nan),
                    "pace_economy": match.get("pace_economy", np.nan),
                    "spin_economy": match.get("spin_economy", np.nan),
                    "batting_friendly": match.get("batting_friendly", np.nan),
                    "spin_friendly": match.get("spin_friendly", np.nan),
                    "dew_factor": match.get("dew_factor", np.nan),
                }
            )

    innings_df = pd.DataFrame(rows)

    # Merge team rolling features
    rolling_cols = [
        "rolling_runs_scored_avg_5",
        "rolling_runs_conceded_avg_5",
        "rolling_won_avg_5",
        "rolling_run_diff_avg_5",
    ]
    team_feats = team_rolling[["match_id", "team"] + rolling_cols].copy()
    innings_df = innings_df.merge(team_feats, on=["match_id", "team"], how="left", suffixes=("", "_team"))

    # Opponent rolling features
    opp_feats = team_rolling[["match_id", "team"] + rolling_cols].copy()
    opp_feats = opp_feats.rename(columns={c: f"opp_{c}" for c in rolling_cols})
    opp_feats = opp_feats.rename(columns={"team": "opponent"})
    innings_df = innings_df.merge(opp_feats, on=["match_id", "opponent"], how="left")

    # Merge player strength features for batting team
    if player_strength is not None:
        bat_str = player_strength.copy()
        innings_df = innings_df.merge(bat_str, on=["match_id", "team"], how="left")

        # Opponent bowling strength (how good is the opposition bowling?)
        opp_str = player_strength.copy()
        opp_str = opp_str.rename(columns={
            c: f"opp_{c}" for c in opp_str.columns if c not in ("match_id", "team")
        })
        opp_str = opp_str.rename(columns={"team": "opponent"})
        innings_df = innings_df.merge(opp_str, on=["match_id", "opponent"], how="left")

    # Derived features
    innings_df["won_toss"] = (innings_df["toss_winner"] == innings_df["team"]).astype(int)
    innings_df["chose_bat"] = (innings_df["toss_decision"] == "bat").astype(int)
    innings_df["is_second_innings"] = (innings_df["innings"] == 2).astype(int)

    innings_df["rolling_strength_diff"] = (
        innings_df["rolling_run_diff_avg_5"].fillna(0) - innings_df["opp_rolling_run_diff_avg_5"].fillna(0)
    )

    # Season scoring inflation (lagged by 1 season)
    if season_avg is not None:
        innings_df = innings_df.merge(season_avg, on="season", how="left")

    # First innings score for 2nd innings prediction
    first_inn_scores = match_df.set_index("match_id")["score_bat_first"].to_dict()
    innings_df["first_innings_score"] = innings_df.apply(
        lambda r: first_inn_scores.get(r["match_id"], 0) if r["innings"] == 2 else 0,
        axis=1,
    )

    return innings_df


SCORE_FEATURES = [
    # Venue
    "avg_total_runs",
    "batting_friendly",
    "spin_friendly",
    "dew_factor",
    "pace_economy",
    "spin_economy",
    # Team form
    "rolling_runs_scored_avg_5",
    "rolling_runs_conceded_avg_5",
    "rolling_won_avg_5",
    "rolling_run_diff_avg_5",
    # Opponent form
    "opp_rolling_runs_scored_avg_5",
    "opp_rolling_runs_conceded_avg_5",
    "opp_rolling_won_avg_5",
    # Match context
    "won_toss",
    "chose_bat",
    "is_second_innings",
    "rolling_strength_diff",
    # Player strength (batting team)
    "team_sum_batting_pts",
    "team_avg_runs_form",
    "team_sum_runs_form",
    "team_avg_sr",
    "team_max_batting",
    "team_experience",
    # Opponent bowling strength
    "opp_team_sum_bowling_pts",
    "opp_team_avg_economy",
    "opp_team_avg_points",
    # Improvements: season inflation + first innings score
    "season_avg_score_lag",
    "first_innings_score",
]


def train_team_score_model(innings_df: pd.DataFrame, test_season: str = "2025"):
    """Train model to predict team innings total."""
    df = innings_df.dropna(subset=["rolling_runs_scored_avg_5"]).copy()

    train = df[df["season"].astype(str) < test_season]
    test = df[df["season"].astype(str) == test_season]

    X_train = train[SCORE_FEATURES].fillna(0)
    y_train = train["runs_scored"]
    X_test = test[SCORE_FEATURES].fillna(0)
    y_test = test["runs_scored"]

    model = GradientBoostingRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        min_samples_leaf=10,
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    preds = np.maximum(preds, 50)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    baseline_mae = mean_absolute_error(y_test, [y_test.mean()] * len(y_test))

    print(f"Team Score Model — MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    print(f"  Baseline (predict mean): MAE={baseline_mae:.2f}")
    print(f"  Train: {len(train)} innings, Test: {len(test)} innings")

    feat_imp = pd.Series(model.feature_importances_, index=SCORE_FEATURES).sort_values(ascending=False)
    print(f"\n  Top features:\n{feat_imp.head(10).to_string()}")

    return model, {
        "mae": mae,
        "rmse": rmse,
        "test_preds": preds,
        "test_actual": y_test.values,
        "test_df": test,
    }


def show_predictions(results: dict):
    """Display sample match predictions."""
    test = results["test_df"].copy()
    test["pred_score"] = results["test_preds"]

    print("\n  Sample Match Predictions (last 10 matches):")
    print(f"  {'Team':>30s}  Inn  Pred  Actual  Error")
    print("  " + "-" * 65)

    last_matches = test["match_id"].unique()[-10:]
    for mid in last_matches:
        m = test[test["match_id"] == mid].sort_values("innings")
        for _, row in m.iterrows():
            err = row["pred_score"] - row["runs_scored"]
            print(
                f"  {row['team']:>30s}  {int(row['innings']):>3d}  {row['pred_score']:>5.0f}  {int(row['runs_scored']):>6d}  {err:>+6.0f}"
            )
        print()


if __name__ == "__main__":
    print("=" * 60)
    print("Team Score Prediction Model")
    print("=" * 60)

    player_df = load_ipl_data()
    feature_df = load_feature_data()
    venue_df = load_venue_profiles()

    match_df = build_match_level_data(player_df)
    match_df = merge_venue_features(match_df, venue_df)
    team_rolling = build_team_rolling_features(match_df)
    player_strength = build_player_strength_features(feature_df)
    season_avg = build_season_avg_scores(match_df)

    innings_df = build_innings_dataset(match_df, team_rolling, player_strength, season_avg)
    print(f"Built {len(innings_df)} innings rows\n")

    model, results = train_team_score_model(innings_df)
    show_predictions(results)

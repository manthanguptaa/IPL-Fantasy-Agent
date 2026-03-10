"""
Model 4: Victory Margin Predictor
Predicts the margin of victory — either runs (if batting first wins)
or wickets remaining (if batting second wins).
Also outputs a combined "expected margin" in runs-equivalent.
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
    merge_venue_features,
)
from .team_score import build_player_strength_features
from .match_winner import build_match_winner_dataset, WINNER_FEATURES


def build_margin_dataset(match_df: pd.DataFrame, winner_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build dataset for victory margin prediction.
    Uses the same features as match winner + adds the predicted win probability.
    Target: signed run margin (positive = bat first won by X runs, negative = bat second won).
    """
    df = winner_df.copy()

    # Signed margin: positive means batting first team won by that many runs
    # score_bat_first - score_bat_second
    df["signed_margin"] = df["score_bat_first"] - df["score_bat_second"]

    # Absolute margin
    df["abs_margin"] = df["signed_margin"].abs()

    # Win type
    df["won_by_runs"] = (df["signed_margin"] > 0).astype(int)  # batting first team won
    df["won_by_wickets"] = (df["signed_margin"] < 0).astype(int)  # chasing team won

    # For wickets margin, we have wickets_lost_bat_second
    # If chasing team won, wickets remaining = 10 - wickets_lost
    df["wickets_remaining"] = np.where(
        df["signed_margin"] < 0,
        10 - df["wickets_lost_bat_second"].fillna(0),
        0,
    )

    return df


MARGIN_FEATURES = WINNER_FEATURES + [
    "avg_total_sixes",
    "spin_friendly",
]


def train_margin_model(df: pd.DataFrame, test_season: str = "2025"):
    """Train model to predict victory margin (signed, in runs)."""
    df = df.dropna(subset=["t1_rolling_won_avg_5"]).copy()
    df = df[df["winner"].notna()].copy()
    # Remove super overs / ties with 0 margin
    df = df[df["signed_margin"] != 0].copy()

    train = df[df["season"].astype(str) < test_season]
    test = df[df["season"].astype(str) == test_season]

    X_train = train[MARGIN_FEATURES].fillna(0)
    y_train = train["signed_margin"]
    X_test = test[MARGIN_FEATURES].fillna(0)
    y_test = test["signed_margin"]

    model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=20,
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    baseline_mae = mean_absolute_error(y_test, [y_test.mean()] * len(y_test))

    print(f"Victory Margin Model — MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    print(f"  Baseline (predict mean): MAE={baseline_mae:.2f}")
    print(f"  Train: {len(train)}, Test: {len(test)}")

    # Also check: does the sign of the prediction match the actual winner?
    sign_correct = ((preds > 0) == (y_test.values > 0)).mean()
    print(f"  Winner direction accuracy (from margin sign): {sign_correct:.3f}")

    feat_imp = pd.Series(model.feature_importances_, index=MARGIN_FEATURES).sort_values(ascending=False)
    print(f"\n  Top features:\n{feat_imp.head(7).to_string()}")

    return model, {
        "mae": mae,
        "rmse": rmse,
        "test_preds": preds,
        "test_actual": y_test.values,
        "test_df": test,
    }


def show_predictions(results: dict):
    """Display match-by-match margin predictions."""
    test = results["test_df"].copy()
    test["pred_margin"] = results["test_preds"]

    print("\n  Match-by-Match Margin Predictions (last 15 matches):")
    print(f"  {'Bat First':>25s} vs {'Bat Second':<25s}  Pred   Actual  Interpretation")
    print("  " + "-" * 110)

    last_matches = test["match_id"].unique()[-15:]
    for mid in last_matches:
        row = test[test["match_id"] == mid].iloc[0]
        pred_m = row["pred_margin"]
        actual_m = row["signed_margin"]

        # Interpret prediction
        if pred_m > 0:
            pred_str = f"{row['team_bat_first']} by ~{abs(pred_m):.0f} runs"
        else:
            pred_str = f"{row['team_bat_second']} by ~{abs(pred_m):.0f} runs equiv."

        # Interpret actual
        if actual_m > 0:
            actual_str = f"{row['team_bat_first']} by {int(abs(actual_m))} runs"
        else:
            wkts = int(row.get("wickets_remaining", 0))
            actual_str = f"{row['team_bat_second']} by {wkts} wkts ({int(abs(actual_m))} runs diff)"

        print(f"  {row['team_bat_first']:>25s} vs {row['team_bat_second']:<25s}  {pred_m:>+5.0f}  {actual_m:>+5.0f}   {pred_str}")
        print(f"  {'':>25s}    {'':25s}  {'':5s}  {'':5s}   Actual: {actual_str}")
        print()


if __name__ == "__main__":
    print("=" * 60)
    print("Victory Margin Prediction Model")
    print("=" * 60)

    player_df = load_ipl_data()
    feature_df = load_feature_data()
    venue_df = load_venue_profiles()

    match_df = build_match_level_data(player_df)
    match_df = merge_venue_features(match_df, venue_df)
    team_rolling = build_team_rolling_features(match_df)
    player_strength = build_player_strength_features(feature_df)

    winner_df = build_match_winner_dataset(match_df, team_rolling, player_strength)
    margin_df = build_margin_dataset(match_df, winner_df)
    print(f"Built {len(margin_df)} match rows\n")

    model, results = train_margin_model(margin_df)
    show_predictions(results)

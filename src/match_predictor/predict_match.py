"""
Unified Match Prediction Pipeline
Runs all 4 models and produces a combined match forecast:
1. Player performance (runs & wickets per player)
2. Team totals
3. Match winner
4. Victory margin
"""

import pandas as pd
import numpy as np
from pathlib import Path

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
    RUNS_FEATURES,
    WICKETS_FEATURES,
    prepare_player_data,
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


class MatchPredictor:
    """Trains all models and generates match predictions."""

    def __init__(self, test_season: str = "2025"):
        self.test_season = test_season
        self.models = {}
        self.data = {}

    def load_data(self):
        """Load and prepare all datasets."""
        print("Loading data...")
        player_df = load_ipl_data()
        feature_df = load_feature_data()
        venue_df = load_venue_profiles()

        print("Building match-level data...")
        match_df = build_match_level_data(player_df)
        match_df = merge_venue_features(match_df, venue_df)
        team_rolling = build_team_rolling_features(match_df)
        player_strength = build_player_strength_features(feature_df)

        self.data = {
            "player_df": player_df,
            "feature_df": feature_df,
            "venue_df": venue_df,
            "match_df": match_df,
            "team_rolling": team_rolling,
            "player_strength": player_strength,
        }
        print(f"  {player_df['match_id'].nunique()} matches loaded\n")

    def train_all(self):
        """Train all 4 prediction models."""
        d = self.data

        # 1. Player runs & wickets
        print("=" * 50)
        print("Training Player Performance Models")
        print("=" * 50)
        self.models["runs"], _ = train_runs_model(d["feature_df"], self.test_season)
        self.models["wickets"], _ = train_wickets_model(d["feature_df"], self.test_season)

        # 2. Team score
        print("\n" + "=" * 50)
        print("Training Team Score Model")
        print("=" * 50)
        innings_df = build_innings_dataset(d["match_df"], d["team_rolling"], d["player_strength"])
        self.models["team_score"], _ = train_team_score_model(innings_df, self.test_season)

        # 3. Match winner
        print("\n" + "=" * 50)
        print("Training Match Winner Model")
        print("=" * 50)
        winner_df = build_match_winner_dataset(d["match_df"], d["team_rolling"], d["player_strength"])
        self.models["winner"], _ = train_match_winner_model(winner_df, self.test_season)

        # 4. Victory margin
        print("\n" + "=" * 50)
        print("Training Victory Margin Model")
        print("=" * 50)
        margin_df = build_margin_dataset(d["match_df"], winner_df)
        self.models["margin"], _ = train_margin_model(margin_df, self.test_season)

    def predict_match(self, match_id: int) -> dict:
        """
        Generate a full prediction for a specific match.
        Returns a dict with all predictions.
        """
        d = self.data

        # Get match info
        match_row = d["match_df"][d["match_df"]["match_id"] == match_id]
        if match_row.empty:
            raise ValueError(f"Match {match_id} not found")
        match_row = match_row.iloc[0]

        # Player predictions
        feature_df = prepare_player_data(d["feature_df"])
        players = feature_df[feature_df["match_id"] == match_id].copy()

        if not players.empty:
            players["pred_runs"] = np.maximum(
                self.models["runs"].predict(players[RUNS_FEATURES].fillna(0)), 0
            )
            players["pred_wickets"] = np.maximum(
                self.models["wickets"].predict(players[WICKETS_FEATURES].fillna(0)), 0
            )

        # Team score predictions
        innings_df = build_innings_dataset(d["match_df"], d["team_rolling"], d["player_strength"])
        innings = innings_df[innings_df["match_id"] == match_id].copy()
        if not innings.empty:
            innings["pred_score"] = np.maximum(
                self.models["team_score"].predict(innings[SCORE_FEATURES].fillna(0)), 50
            )

        # Match winner prediction
        winner_df = build_match_winner_dataset(d["match_df"], d["team_rolling"], d["player_strength"])
        winner_row = winner_df[winner_df["match_id"] == match_id].copy()
        if not winner_row.empty:
            X = winner_row[WINNER_FEATURES].fillna(0)
            win_prob = self.models["winner"].predict_proba(X)[0, 1]
        else:
            win_prob = 0.5

        # Victory margin prediction
        margin_df = build_margin_dataset(d["match_df"], winner_df)
        margin_row = margin_df[margin_df["match_id"] == match_id].copy()
        if not margin_row.empty:
            X = margin_row[MARGIN_FEATURES].fillna(0)
            pred_margin = self.models["margin"].predict(X)[0]
        else:
            pred_margin = 0

        return {
            "match_id": match_id,
            "match_date": match_row["match_date"],
            "team_bat_first": match_row["team_bat_first"],
            "team_bat_second": match_row["team_bat_second"],
            "actual_winner": match_row["winner"],
            "actual_score_first": match_row["score_bat_first"],
            "actual_score_second": match_row["score_bat_second"],
            # Predictions
            "win_prob_bat_first": win_prob,
            "predicted_winner": match_row["team_bat_first"] if win_prob > 0.5 else match_row["team_bat_second"],
            "predicted_margin": pred_margin,
            "innings_predictions": innings,
            "player_predictions": players,
        }

    def print_match_forecast(self, match_id: int):
        """Pretty-print a full match forecast."""
        pred = self.predict_match(match_id)

        print("\n" + "=" * 70)
        print(f"  MATCH FORECAST: {pred['team_bat_first']} vs {pred['team_bat_second']}")
        print(f"  Date: {pred['match_date']}")
        print("=" * 70)

        # Match winner
        prob = pred["win_prob_bat_first"]
        t1, t2 = pred["team_bat_first"], pred["team_bat_second"]
        print(f"\n  WINNER PREDICTION:")
        print(f"    {t1}: {prob*100:.0f}%")
        print(f"    {t2}: {(1-prob)*100:.0f}%")
        print(f"    => Predicted winner: {pred['predicted_winner']}")
        print(f"    => Actual winner:    {pred['actual_winner']}")

        # Victory margin
        margin = pred["predicted_margin"]
        if margin > 0:
            print(f"\n  MARGIN: {t1} to win by ~{abs(margin):.0f} runs")
        else:
            print(f"\n  MARGIN: {t2} to win (run-equiv margin: ~{abs(margin):.0f})")
        actual_margin = pred["actual_score_first"] - pred["actual_score_second"]
        if actual_margin > 0:
            print(f"  Actual: {t1} won by {actual_margin} runs")
        else:
            print(f"  Actual: {t2} won (deficit: {abs(actual_margin)} runs)")

        # Team scores
        innings = pred["innings_predictions"]
        if not innings.empty:
            print(f"\n  TEAM SCORE PREDICTIONS:")
            for _, inn in innings.sort_values("innings").iterrows():
                print(
                    f"    {inn['team']:>30s} (Inn {int(inn['innings'])}): "
                    f"Predicted {inn['pred_score']:.0f} | Actual {int(inn['runs_scored'])}"
                )

        # Player predictions
        players = pred["player_predictions"]
        if not players.empty:
            for team in [t1, t2]:
                tp = players[players["team"] == team].copy()
                if tp.empty:
                    continue
                print(f"\n  PLAYER PREDICTIONS — {team}:")
                print(f"    {'Player':<25s}  Pred Runs  Actual  |  Pred Wkts  Actual")
                print("    " + "-" * 65)
                # Sort by predicted runs
                tp = tp.sort_values("pred_runs", ascending=False)
                for _, p in tp.head(11).iterrows():
                    print(
                        f"    {p['player_name']:<25s}  {p['pred_runs']:>8.1f}  {int(p['runs']):>6d}  |  "
                        f"{p['pred_wickets']:>9.2f}  {int(p['wickets']):>6d}"
                    )

        print("\n" + "=" * 70)


def main():
    predictor = MatchPredictor(test_season="2025")
    predictor.load_data()
    predictor.train_all()

    # Print forecasts for last 5 test matches
    match_df = predictor.data["match_df"]
    test_matches = match_df[match_df["season"].astype(str) == "2025"]["match_id"].values
    print("\n\n" + "#" * 70)
    print("#  FULL MATCH FORECASTS (last 5 IPL 2025 matches)")
    print("#" * 70)

    for mid in test_matches[-5:]:
        predictor.print_match_forecast(mid)


if __name__ == "__main__":
    main()

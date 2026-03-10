#!/usr/bin/env python3
"""Generate optimal Dream11 team for a match.

This script:
1. Loads the trained prediction model
2. Generates predictions for all players in a match
3. Runs the Dream11 optimizer to select the best team
4. Outputs the team with Captain and Vice-Captain
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ipl_fantasy.team_optimizer import (
    Dream11Optimizer,
    Dream11Constraints,
    Player,
    estimate_credits_from_points,
)


def load_model(model_path: Path):
    """Load the trained CatBoost model."""
    return joblib.load(model_path)


def load_features_config(config_path: Path) -> dict:
    """Load the feature configuration."""
    return json.loads(config_path.read_text())


def prepare_player_features(
    players_df: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """Prepare feature matrix for prediction."""
    X = players_df[features].copy()
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = X[col].fillna("")
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)
    return X


def get_latest_player_features(
    features_df: pd.DataFrame,
    team1: str,
    team2: str,
) -> pd.DataFrame:
    """
    Get the latest features for players from two teams.

    This extracts the most recent feature snapshot for each player
    who has played for either team.
    """
    # Filter to players from these teams
    team_df = features_df[features_df["team"].isin([team1, team2])].copy()

    if team_df.empty:
        raise ValueError(f"No players found for teams: {team1}, {team2}")

    # Sort by date and get latest per player
    team_df = team_df.sort_values("match_date", ascending=False)
    latest = team_df.groupby("player_name").first().reset_index()

    return latest


def create_player_pool(
    players_df: pd.DataFrame,
    predictions: list[float],
    credits_col: str | None = None,
) -> list[Player]:
    """Create Player objects from dataframe and predictions."""
    players = []

    for i, (_, row) in enumerate(players_df.iterrows()):
        pred_points = predictions[i]

        # Get credits (use estimated if not available)
        if credits_col and credits_col in row:
            credits = float(row[credits_col])
        else:
            credits = estimate_credits_from_points(pred_points)

        # Get role
        role = row.get("player_role", "BAT")
        if role not in ("WK", "BAT", "AR", "BOWL"):
            role = "BAT"

        player = Player(
            name=row["player_name"],
            team=row["team"],
            role=role,
            predicted_points=pred_points,
            credits=credits,
        )
        players.append(player)

    return players


def generate_team(
    model,
    features: list[str],
    players_df: pd.DataFrame,
    n_teams: int = 1,
) -> None:
    """Generate and print optimal Dream11 team(s)."""
    # Prepare features
    X = prepare_player_features(players_df, features)

    # Make predictions
    predictions = model.predict(X)

    # Create player pool
    player_pool = create_player_pool(players_df, predictions)

    print(f"\nPlayer Pool: {len(player_pool)} players")
    print(f"Teams: {set(p.team for p in player_pool)}")

    # Check role distribution
    role_counts = {}
    for p in player_pool:
        role_counts[p.role] = role_counts.get(p.role, 0) + 1
    print(f"Roles: {role_counts}")

    # Optimize
    optimizer = Dream11Optimizer()

    if n_teams == 1:
        result = optimizer.optimize(player_pool)
        print("\n" + result.get_team_summary())
    else:
        results = optimizer.optimize_multiple(player_pool, n_teams=n_teams)
        for i, result in enumerate(results, 1):
            print(f"\n{'='*60}")
            print(f"TEAM {i}")
            print(result.get_team_summary())


def demo_with_sample_data(model, features: list[str], features_df: pd.DataFrame):
    """Run a demo with sample match data."""
    # Get unique team pairs from recent matches
    recent = features_df.sort_values("match_date", ascending=False)
    recent_matches = recent.drop_duplicates(subset=["match_id"]).head(10)

    print("\nRecent matches in dataset:")
    for _, match in recent_matches.iterrows():
        print(f"  {match['match_date']}: {match['team']} vs {match['opponent']}")

    # Use the most recent match
    latest_match_id = recent_matches.iloc[0]["match_id"]
    match_df = features_df[features_df["match_id"] == latest_match_id]

    team1 = match_df["team"].iloc[0]
    team2 = match_df["opponent"].iloc[0]
    match_date = match_df["match_date"].iloc[0]

    print(f"\nGenerating team for: {team1} vs {team2} ({match_date})")

    # Get players from this match
    players_df = match_df.copy()

    generate_team(model, features, players_df, n_teams=1)


def main():
    parser = argparse.ArgumentParser(description="Generate optimal Dream11 team")
    parser.add_argument(
        "--model",
        default="tmp/experiments_final/model.joblib",
        help="Path to trained model",
    )
    parser.add_argument(
        "--config",
        default="tmp/experiments_final/stable_features.json",
        help="Path to feature config",
    )
    parser.add_argument(
        "--features-data",
        default="tmp/full_player_match_features_v3.csv",
        help="Path to features dataset (for demo mode)",
    )
    parser.add_argument(
        "--team1",
        help="First team name",
    )
    parser.add_argument(
        "--team2",
        help="Second team name",
    )
    parser.add_argument(
        "--n-teams",
        type=int,
        default=1,
        help="Number of teams to generate",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo with most recent match from dataset",
    )
    args = parser.parse_args()

    # Load model and config
    print(f"Loading model from {args.model}...")
    model = load_model(Path(args.model))

    print(f"Loading feature config from {args.config}...")
    config = load_features_config(Path(args.config))
    features = config["features"]
    print(f"Using {len(features)} features")

    # Load features data
    print(f"Loading features data from {args.features_data}...")
    features_df = pd.read_csv(args.features_data, low_memory=False)
    print(f"Loaded {len(features_df)} rows")

    if args.demo:
        demo_with_sample_data(model, features, features_df)
    elif args.team1 and args.team2:
        # Get latest features for specified teams
        players_df = get_latest_player_features(features_df, args.team1, args.team2)
        print(f"\nGenerating team for: {args.team1} vs {args.team2}")
        generate_team(model, features, players_df, n_teams=args.n_teams)
    else:
        print("\nUsage:")
        print("  Demo mode:   python generate_team.py --demo")
        print("  Match mode:  python generate_team.py --team1 'Mumbai Indians' --team2 'Chennai Super Kings'")


if __name__ == "__main__":
    main()

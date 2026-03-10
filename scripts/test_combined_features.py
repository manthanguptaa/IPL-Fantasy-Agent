#!/usr/bin/env python3
"""Test combinations of features that showed improvement."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.add_features_and_test import (
    add_opponent_weakness_features,
    add_batting_first_feature,
    add_toss_advantage_feature,
)


def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all new features."""
    df = add_opponent_weakness_features(df)
    df = add_toss_advantage_feature(df)
    # Skip batting_first as it hurt captain accuracy
    return df


def add_best_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add only the best performing features."""
    df = add_toss_advantage_feature(df)
    df = add_opponent_weakness_features(df)
    return df


def test_combined_features(
    features_path: str,
    n_matches: int = 50,
):
    """Test combined features."""
    from src.ipl_fantasy.quantile_model import QuantileModelEnsemble, OPTIMAL_FEATURES
    from src.ipl_fantasy.backtesting import Backtester
    from src.ipl_fantasy.team_optimizer import Player
    from src.ipl_fantasy.enhanced_prediction import OPTIMAL_CONFIG
    from src.ipl_fantasy.credit_estimation import estimate_credits_from_history

    print(f"\n{'='*60}")
    print("Testing COMBINED features (won_toss + opponent_weakness)")
    print(f"{'='*60}")

    # Load data
    print(f"\nLoading data from {features_path}...")
    df = pd.read_csv(features_path, low_memory=False)

    # Add combined features
    df = add_best_features(df)

    # Filter to IPL
    df = df[df['competition'] == 'Indian Premier League'].copy()

    # Sort and split
    df = df.sort_values(['match_date', 'match_id', 'player_name']).reset_index(drop=True)

    # Get unique matches for backtest
    matches = df.groupby('match_id').first().reset_index()
    matches = matches.sort_values('match_date', ascending=False)
    test_match_ids = set(matches.head(n_matches)['match_id'])

    # Split into train and test
    test_df = df[df['match_id'].isin(test_match_ids)].copy()
    train_df = df[~df['match_id'].isin(test_match_ids)].copy()

    print(f"  Train matches: {train_df['match_id'].nunique()}")
    print(f"  Test matches: {test_df['match_id'].nunique()}")

    # Extended features
    new_features = ['won_toss', 'opponent_weakness_avg_5',
                    'opponent_batting_weakness_avg_5', 'opponent_bowling_weakness_avg_5']
    available_new_features = [f for f in new_features if f in df.columns]
    extended_features = OPTIMAL_FEATURES + available_new_features

    print(f"  Total features: {len(extended_features)}")
    print(f"  New features: {available_new_features}")

    # Train model
    print(f"\nTraining model...")
    ensemble = QuantileModelEnsemble(features=extended_features)
    metrics = ensemble.fit(train_df, test_df.head(1000))

    print(f"  RMSE: {metrics.get('mean_rmse', 'N/A'):.4f}")
    print(f"  MAE: {metrics.get('mean_mae', 'N/A'):.4f}")

    # Create prediction function
    def create_predict_fn(ensemble, config):
        def predict_fn(match_df):
            predictions = ensemble.predict(match_df)

            players = []
            for i, pred in enumerate(predictions):
                role = pred.role if pred.role in ("WK", "BAT", "AR", "BOWL") else "BAT"
                ceiling_weight = config.role_ceiling_weights.get(role, 0.3)

                weighted_prediction = (
                    (1 - ceiling_weight) * pred.expected +
                    ceiling_weight * pred.q90
                )

                captain_value = (
                    (1 - config.captain_ceiling_weight) * pred.expected +
                    config.captain_ceiling_weight * pred.q90
                )

                row = match_df.iloc[i] if i < len(match_df) else {}
                avg_all = row.get("rolling_points_avg_10_all", 30.0)
                avg_recent = row.get("rolling_points_avg_5_all", None)

                credits = estimate_credits_from_history(
                    player_name=pred.player_name,
                    player_role=role,
                    avg_points_all=avg_all if pd.notna(avg_all) else 30.0,
                    avg_points_recent=avg_recent if pd.notna(avg_recent) else None,
                )

                player = Player(
                    name=pred.player_name,
                    team=pred.team,
                    role=role,
                    predicted_points=weighted_prediction,
                    credits=credits,
                    ceiling=captain_value,
                    floor=pred.q10,
                    variance=pred.variance,
                )
                players.append(player)

            return players

        return predict_fn

    # Run backtest
    print(f"\nRunning backtest on {n_matches} matches...")
    backtester = Backtester()
    predict_fn = create_predict_fn(ensemble, OPTIMAL_CONFIG)

    summary = backtester.backtest_multiple(
        test_df,
        predict_fn,
        n_matches=n_matches,
        competition_filter=None,
    )

    print(f"\n{'='*60}")
    print("COMBINED FEATURES RESULTS")
    print(f"{'='*60}")
    print(f"  Mean Selected Score: {summary.mean_selected_score:.1f}")
    print(f"  Mean Oracle Score: {summary.mean_oracle_score:.1f}")
    print(f"  Mean Total Regret: {summary.mean_total_regret:.1f}")
    print(f"  Mean Team Regret: {summary.mean_team_regret:.1f}")
    print(f"  Mean Captain Regret: {summary.mean_captain_regret:.1f}")
    print(f"  Mean VC Regret: {summary.mean_vc_regret:.1f}")
    print(f"  Player Overlap: {summary.mean_overlap_pct:.1f}%")
    print(f"  Captain Accuracy: {summary.top_captain_rate:.1f}%")
    print(f"  VC Accuracy: {summary.top_vc_rate:.1f}%")

    # Comparison with baseline
    baseline_regret = 295.4  # From previous test
    improvement = baseline_regret - summary.mean_total_regret
    pct = improvement / baseline_regret * 100

    print(f"\n{'='*60}")
    print("IMPROVEMENT VS BASELINE")
    print(f"{'='*60}")
    print(f"  Baseline regret: 295.4")
    print(f"  New regret: {summary.mean_total_regret:.1f}")
    print(f"  Improvement: {improvement:.1f} pts ({pct:.1f}%)")

    # Save improved model
    output_dir = Path("tmp/quantile_models_improved")
    output_dir.mkdir(parents=True, exist_ok=True)
    ensemble.save(output_dir)
    print(f"\nImproved model saved to {output_dir}")

    return summary


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test combined features")
    parser.add_argument(
        "--features",
        default="tmp/full_player_match_features_v3.csv",
        help="Path to features CSV",
    )
    parser.add_argument(
        "--n-matches",
        type=int,
        default=50,
        help="Number of matches to test on",
    )
    args = parser.parse_args()

    test_combined_features(args.features, args.n_matches)


if __name__ == "__main__":
    main()

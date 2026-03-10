#!/usr/bin/env python3
"""Run backtesting to evaluate team selection quality."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ipl_fantasy.backtesting import Backtester, BacktestSummary
from src.ipl_fantasy.quantile_model import QuantileModelEnsemble
from src.ipl_fantasy.team_optimizer import Player, estimate_credits_from_points
from src.ipl_fantasy.enhanced_prediction import (
    create_enhanced_predict_fn,
    PredictionConfig,
    OPTIMAL_CONFIG,
)


def create_baseline_predict_fn(ensemble: QuantileModelEnsemble):
    """Create a baseline prediction function (expected value only)."""

    def predict_fn(match_df: pd.DataFrame) -> list[Player]:
        """Generate predictions for a match."""
        predictions = ensemble.predict(match_df)

        players = []
        for pred in predictions:
            role = pred.role if pred.role in ("WK", "BAT", "AR", "BOWL") else "BAT"
            player = Player(
                name=pred.player_name,
                team=pred.team,
                role=role,
                predicted_points=pred.expected,
                credits=estimate_credits_from_points(pred.expected),
                ceiling=pred.q90,
                floor=pred.q10,
                variance=pred.variance,
            )
            players.append(player)

        return players

    return predict_fn


def main():
    parser = argparse.ArgumentParser(description="Run backtesting on historical matches")
    parser.add_argument(
        "--model-dir",
        default="tmp/quantile_models",
        help="Directory with quantile models",
    )
    parser.add_argument(
        "--features",
        default="tmp/full_player_match_features_v3.csv",
        help="Path to features CSV",
    )
    parser.add_argument(
        "--output-dir",
        default="tmp/backtest_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--n-matches",
        type=int,
        default=100,
        help="Number of matches to backtest",
    )
    parser.add_argument(
        "--competition",
        default="Indian Premier League",
        help="Competition to filter (or 'all')",
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "optimized"],
        default="optimized",
        help="Prediction mode: baseline (expected only) or optimized (ceiling-weighted)",
    )
    args = parser.parse_args()

    # Load models
    print(f"Loading quantile models from {args.model_dir}...")
    ensemble = QuantileModelEnsemble.load(args.model_dir)

    # Load features
    print(f"Loading features from {args.features}...")
    features_df = pd.read_csv(args.features, low_memory=False)
    print(f"  Total rows: {len(features_df)}")

    # Create prediction function based on mode
    if args.mode == "baseline":
        print("Using baseline prediction (expected value only)")
        predict_fn = create_baseline_predict_fn(ensemble)
    else:
        print("Using optimized prediction (role-weighted ceiling)")
        predict_fn = create_enhanced_predict_fn(ensemble, OPTIMAL_CONFIG)

    # Run backtest
    print(f"\nRunning backtest on {args.n_matches} matches...")
    backtester = Backtester()

    competition_filter = None if args.competition == "all" else args.competition

    summary = backtester.backtest_multiple(
        features_df,
        predict_fn,
        n_matches=args.n_matches,
        competition_filter=competition_filter,
    )

    # Print summary
    print("\n" + summary.get_summary())

    # Save results
    backtester.save_results(summary, args.output_dir)

    # Show some example matches
    print("\nSample match results:")
    for result in summary.match_results[:5]:
        print(f"\n  {result.match_date}: {result.team1} vs {result.team2}")
        print(f"    Selected: {result.selected_score_with_cv:.0f} pts (C: {result.selected_captain})")
        print(f"    Oracle: {result.oracle_score_with_cv:.0f} pts (C: {result.oracle_captain})")
        print(f"    Regret: {result.total_regret:.0f} pts ({result.overlap_count}/11 overlap)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run improved backtesting with ceiling-weighted optimization.

Compares baseline optimizer with improved ceiling-weighted approach.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ipl_fantasy.backtesting import Backtester, BacktestSummary
from src.ipl_fantasy.quantile_model import QuantileModelEnsemble
from src.ipl_fantasy.team_optimizer import Player, estimate_credits_from_points
from src.ipl_fantasy.improved_optimizer import (
    ImprovedDream11Optimizer,
    OptimizationConfig,
)


def create_predict_fn_baseline(ensemble: QuantileModelEnsemble):
    """Create baseline prediction function."""

    def predict_fn(match_df: pd.DataFrame) -> list[Player]:
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


def create_predict_fn_ceiling_weighted(
    ensemble: QuantileModelEnsemble,
    ceiling_weight: float = 0.3,
):
    """Create ceiling-weighted prediction function.

    Adjusts predicted_points to include ceiling potential.
    """

    def predict_fn(match_df: pd.DataFrame) -> list[Player]:
        predictions = ensemble.predict(match_df)

        players = []
        for pred in predictions:
            role = pred.role if pred.role in ("WK", "BAT", "AR", "BOWL") else "BAT"

            # Ceiling-weighted prediction
            weighted_prediction = (
                (1 - ceiling_weight) * pred.expected +
                ceiling_weight * pred.q90
            )

            player = Player(
                name=pred.player_name,
                team=pred.team,
                role=role,
                predicted_points=weighted_prediction,
                credits=estimate_credits_from_points(pred.expected),  # Use expected for credits
                ceiling=pred.q90,
                floor=pred.q10,
                variance=pred.variance,
            )
            players.append(player)

        return players

    return predict_fn


def run_comparison_backtest(
    features_path: str,
    model_dir: str,
    n_matches: int = 50,
    competition: str = "Indian Premier League",
) -> dict:
    """
    Run backtest comparing baseline vs improved approaches.
    """
    # Load models and data
    print(f"Loading quantile models from {model_dir}...")
    ensemble = QuantileModelEnsemble.load(model_dir)

    print(f"Loading features from {features_path}...")
    features_df = pd.read_csv(features_path, low_memory=False)
    print(f"  Total rows: {len(features_df)}")

    competition_filter = None if competition == "all" else competition

    # Test different configurations
    configs = [
        {
            "name": "Baseline",
            "predict_fn": create_predict_fn_baseline(ensemble),
            "optimizer": Backtester(),
        },
        {
            "name": "Ceiling-Weighted (0.2)",
            "predict_fn": create_predict_fn_ceiling_weighted(ensemble, 0.2),
            "optimizer": Backtester(),
        },
        {
            "name": "Ceiling-Weighted (0.3)",
            "predict_fn": create_predict_fn_ceiling_weighted(ensemble, 0.3),
            "optimizer": Backtester(),
        },
        {
            "name": "Ceiling-Weighted (0.4)",
            "predict_fn": create_predict_fn_ceiling_weighted(ensemble, 0.4),
            "optimizer": Backtester(),
        },
        {
            "name": "Improved Optimizer (0.3 ceiling)",
            "predict_fn": create_predict_fn_baseline(ensemble),
            "optimizer": Backtester(
                optimizer=ImprovedDream11Optimizer(
                    config=OptimizationConfig(
                        expected_weight=0.6,
                        ceiling_weight=0.3,
                        floor_weight=0.1,
                        captain_ceiling_weight=0.5,
                    )
                )
            ),
        },
        {
            "name": "Improved Optimizer (0.4 ceiling)",
            "predict_fn": create_predict_fn_baseline(ensemble),
            "optimizer": Backtester(
                optimizer=ImprovedDream11Optimizer(
                    config=OptimizationConfig(
                        expected_weight=0.5,
                        ceiling_weight=0.4,
                        floor_weight=0.1,
                        captain_ceiling_weight=0.6,
                    )
                )
            ),
        },
    ]

    results = []
    print(f"\nRunning backtest comparison on {n_matches} matches...")
    print("=" * 80)

    for config in configs:
        print(f"\nTesting: {config['name']}...")

        try:
            summary = config["optimizer"].backtest_multiple(
                features_df,
                config["predict_fn"],
                n_matches=n_matches,
                competition_filter=competition_filter,
            )

            result = {
                "name": config["name"],
                "mean_selected_score": summary.mean_selected_score,
                "mean_oracle_score": summary.mean_oracle_score,
                "mean_team_regret": summary.mean_team_regret,
                "mean_captain_regret": summary.mean_captain_regret,
                "mean_vc_regret": summary.mean_vc_regret,
                "mean_total_regret": summary.mean_total_regret,
                "mean_overlap_pct": summary.mean_overlap_pct,
                "top_captain_rate": summary.top_captain_rate,
            }
            results.append(result)

            print(f"  Selected: {result['mean_selected_score']:.1f} | "
                  f"Oracle: {result['mean_oracle_score']:.1f} | "
                  f"Regret: {result['mean_total_regret']:.1f} | "
                  f"Overlap: {result['mean_overlap_pct']:.1f}% | "
                  f"Cap Rate: {result['top_captain_rate']:.1f}%")

        except Exception as e:
            print(f"  ERROR: {e}")

    # Summary comparison
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Configuration':<35} {'Regret':>10} {'Overlap':>10} {'Cap Rate':>10}")
    print("-" * 80)

    for r in results:
        print(f"{r['name']:<35} {r['mean_total_regret']:>10.1f} "
              f"{r['mean_overlap_pct']:>9.1f}% {r['top_captain_rate']:>9.1f}%")

    # Find best configuration
    best = min(results, key=lambda x: x["mean_total_regret"])
    baseline = results[0]

    improvement = baseline["mean_total_regret"] - best["mean_total_regret"]
    print("\n" + "=" * 80)
    print(f"Best configuration: {best['name']}")
    print(f"Improvement over baseline: {improvement:.1f} points ({improvement/baseline['mean_total_regret']*100:.1f}%)")

    # Save results
    output_dir = Path("tmp/improved_backtest_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}")

    return {"results": results, "best": best}


def main():
    parser = argparse.ArgumentParser(description="Run improved backtesting comparison")
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
        "--n-matches",
        type=int,
        default=50,
        help="Number of matches to backtest",
    )
    parser.add_argument(
        "--competition",
        default="Indian Premier League",
        help="Competition to filter",
    )
    args = parser.parse_args()

    run_comparison_backtest(
        args.features,
        args.model_dir,
        args.n_matches,
        args.competition,
    )


if __name__ == "__main__":
    main()

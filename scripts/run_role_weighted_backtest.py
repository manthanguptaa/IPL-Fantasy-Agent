#!/usr/bin/env python3
"""Run backtesting with role-specific ceiling weighting.

Based on breakout analysis:
- All-rounders (AR) have highest upside (2.23x breakout ratio)
- Wicket-keepers (WK) have high upside (2.24x breakout ratio)
- Batsmen (BAT) moderate upside (2.21x breakout ratio)
- Bowlers (BOWL) lower upside (2.14x breakout ratio)
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


# Role-specific ceiling weights based on breakout analysis
ROLE_CEILING_WEIGHTS = {
    "AR": 0.40,   # Highest variance/upside
    "WK": 0.38,   # High variance
    "BAT": 0.30,  # Moderate variance
    "BOWL": 0.25, # Lower variance
}


def create_role_weighted_predict_fn(
    ensemble: QuantileModelEnsemble,
    role_weights: dict[str, float] = None,
):
    """Create prediction function with role-specific ceiling weighting."""
    weights = role_weights or ROLE_CEILING_WEIGHTS

    def predict_fn(match_df: pd.DataFrame) -> list[Player]:
        predictions = ensemble.predict(match_df)

        players = []
        for pred in predictions:
            role = pred.role if pred.role in ("WK", "BAT", "AR", "BOWL") else "BAT"
            ceiling_weight = weights.get(role, 0.3)

            # Role-specific ceiling-weighted prediction
            weighted_prediction = (
                (1 - ceiling_weight) * pred.expected +
                ceiling_weight * pred.q90
            )

            player = Player(
                name=pred.player_name,
                team=pred.team,
                role=role,
                predicted_points=weighted_prediction,
                credits=estimate_credits_from_points(pred.expected),
                ceiling=pred.q90,
                floor=pred.q10,
                variance=pred.variance,
            )
            players.append(player)

        return players

    return predict_fn


def create_aggressive_captain_predict_fn(
    ensemble: QuantileModelEnsemble,
    base_ceiling_weight: float = 0.3,
    captain_ceiling_weight: float = 0.5,
):
    """Create prediction with aggressive ceiling weighting for captain selection.

    Uses higher ceiling weight to identify breakout candidates.
    """

    def predict_fn(match_df: pd.DataFrame) -> list[Player]:
        predictions = ensemble.predict(match_df)

        players = []
        for pred in predictions:
            role = pred.role if pred.role in ("WK", "BAT", "AR", "BOWL") else "BAT"

            # Use higher ceiling weight for team selection
            weighted_prediction = (
                (1 - base_ceiling_weight) * pred.expected +
                base_ceiling_weight * pred.q90
            )

            # Store captain-specific value in ceiling for C/VC selection
            # Higher ceiling means more captain value
            captain_value_ceiling = (
                (1 - captain_ceiling_weight) * pred.expected +
                captain_ceiling_weight * pred.q90
            )

            player = Player(
                name=pred.player_name,
                team=pred.team,
                role=role,
                predicted_points=weighted_prediction,
                credits=estimate_credits_from_points(pred.expected),
                ceiling=captain_value_ceiling,  # Used for C/VC selection
                floor=pred.q10,
                variance=pred.variance,
            )
            players.append(player)

        return players

    return predict_fn


def run_role_weighted_backtest(
    features_path: str,
    model_dir: str,
    n_matches: int = 50,
    competition: str = "Indian Premier League",
) -> dict:
    """Run backtest with role-weighted ceiling approach."""
    print(f"Loading quantile models from {model_dir}...")
    ensemble = QuantileModelEnsemble.load(model_dir)

    print(f"Loading features from {features_path}...")
    features_df = pd.read_csv(features_path, low_memory=False)
    print(f"  Total rows: {len(features_df)}")

    competition_filter = None if competition == "all" else competition

    # Test configurations
    configs = [
        {
            "name": "Baseline (expected only)",
            "predict_fn": create_role_weighted_predict_fn(
                ensemble,
                {"AR": 0.0, "WK": 0.0, "BAT": 0.0, "BOWL": 0.0}
            ),
        },
        {
            "name": "Uniform ceiling (0.3)",
            "predict_fn": create_role_weighted_predict_fn(
                ensemble,
                {"AR": 0.3, "WK": 0.3, "BAT": 0.3, "BOWL": 0.3}
            ),
        },
        {
            "name": "Role-weighted ceiling",
            "predict_fn": create_role_weighted_predict_fn(ensemble),
        },
        {
            "name": "Aggressive role-weighted",
            "predict_fn": create_role_weighted_predict_fn(
                ensemble,
                {"AR": 0.50, "WK": 0.45, "BAT": 0.35, "BOWL": 0.30}
            ),
        },
        {
            "name": "AR/WK focused",
            "predict_fn": create_role_weighted_predict_fn(
                ensemble,
                {"AR": 0.55, "WK": 0.50, "BAT": 0.25, "BOWL": 0.20}
            ),
        },
        {
            "name": "Aggressive captain focus",
            "predict_fn": create_aggressive_captain_predict_fn(
                ensemble,
                base_ceiling_weight=0.35,
                captain_ceiling_weight=0.60,
            ),
        },
    ]

    results = []
    print(f"\nRunning backtest on {n_matches} matches...")
    print("=" * 85)

    for config in configs:
        print(f"\nTesting: {config['name']}...")

        try:
            backtester = Backtester()
            summary = backtester.backtest_multiple(
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
                "top_vc_rate": summary.top_vc_rate,
            }
            results.append(result)

            print(f"  Score: {result['mean_selected_score']:.1f} | "
                  f"Regret: {result['mean_total_regret']:.1f} | "
                  f"Overlap: {result['mean_overlap_pct']:.1f}% | "
                  f"Cap: {result['top_captain_rate']:.0f}%")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 85)
    print("COMPARISON SUMMARY")
    print("=" * 85)
    print(f"{'Configuration':<30} {'Score':>8} {'Regret':>8} {'Team-R':>8} "
          f"{'Cap-R':>6} {'Overlap':>8} {'CapRate':>8}")
    print("-" * 85)

    for r in results:
        print(f"{r['name']:<30} {r['mean_selected_score']:>8.1f} "
              f"{r['mean_total_regret']:>8.1f} {r['mean_team_regret']:>8.1f} "
              f"{r['mean_captain_regret']:>6.1f} {r['mean_overlap_pct']:>7.1f}% "
              f"{r['top_captain_rate']:>7.0f}%")

    # Best configuration
    best = min(results, key=lambda x: x["mean_total_regret"])
    baseline = results[0]

    improvement = baseline["mean_total_regret"] - best["mean_total_regret"]
    print("\n" + "=" * 85)
    print(f"Best configuration: {best['name']}")
    print(f"Improvement over baseline: {improvement:.1f} points ({improvement/baseline['mean_total_regret']*100:.1f}%)")

    # Save results
    output_dir = Path("tmp/role_weighted_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}")

    return {"results": results, "best": best}


def main():
    parser = argparse.ArgumentParser(description="Run role-weighted backtesting")
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

    run_role_weighted_backtest(
        args.features,
        args.model_dir,
        args.n_matches,
        args.competition,
    )


if __name__ == "__main__":
    main()

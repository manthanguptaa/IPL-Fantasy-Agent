#!/usr/bin/env python3
"""Analyze breakout performances to understand prediction failures.

This script identifies players who scored much higher than predicted
and looks for patterns that could improve predictions.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ipl_fantasy.quantile_model import QuantileModelEnsemble


def analyze_breakouts(
    features_path: str,
    model_dir: str,
    n_matches: int = 50,
    breakout_threshold: float = 1.5,  # 50% above prediction
) -> dict:
    """
    Analyze breakout performances where actual >> predicted.

    Args:
        features_path: Path to features CSV
        model_dir: Path to quantile models
        n_matches: Number of recent matches to analyze
        breakout_threshold: Ratio of actual/predicted to consider a breakout

    Returns:
        Dictionary with analysis results
    """
    # Load data
    print(f"Loading features from {features_path}...")
    df = pd.read_csv(features_path, low_memory=False)
    print(f"  Total rows: {len(df)}")

    # Load models
    print(f"Loading models from {model_dir}...")
    ensemble = QuantileModelEnsemble.load(model_dir)

    # Filter to IPL and recent matches
    df = df[df["competition"] == "Indian Premier League"]
    df = df.sort_values("match_date", ascending=False)

    # Get unique matches
    matches = df.groupby("match_id").first().reset_index()
    matches = matches.sort_values("match_date", ascending=False).head(n_matches)
    match_ids = set(matches["match_id"])

    df = df[df["match_id"].isin(match_ids)]
    print(f"  Analyzing {len(match_ids)} matches, {len(df)} player-match rows")

    # Generate predictions
    predictions = ensemble.predict(df)

    # Create comparison DataFrame
    comparison = []
    for i, (_, row) in enumerate(df.iterrows()):
        pred = predictions[i]
        actual = row["dream11_points_total"]
        expected = pred.expected

        ratio = actual / expected if expected > 0 else 0
        is_breakout = ratio >= breakout_threshold and actual >= 50

        comparison.append({
            "match_id": row["match_id"],
            "match_date": row["match_date"],
            "player_name": row["player_name"],
            "team": row["team"],
            "opponent": row["opponent"],
            "player_role": row["player_role"],
            "actual": actual,
            "expected": expected,
            "q10": pred.q10,
            "q90": pred.q90,
            "ratio": ratio,
            "residual": actual - expected,
            "is_breakout": is_breakout,
            "above_ceiling": actual > pred.q90,
            # Include key features for analysis
            "rolling_points_avg_5_all": row.get("rolling_points_avg_5_all", 0),
            "rolling_points_avg_10_all": row.get("rolling_points_avg_10_all", 0),
            "venue_points_avg_all": row.get("venue_points_avg_all", 0),
            "opponent_points_avg_all": row.get("opponent_points_avg_all", 0),
        })

    comp_df = pd.DataFrame(comparison)

    # Analysis
    results = {
        "total_players": len(comp_df),
        "breakout_count": comp_df["is_breakout"].sum(),
        "above_ceiling_count": comp_df["above_ceiling"].sum(),
        "above_ceiling_pct": comp_df["above_ceiling"].mean() * 100,
    }

    # Breakout analysis
    breakouts = comp_df[comp_df["is_breakout"]]
    print(f"\n{'='*60}")
    print(f"BREAKOUT ANALYSIS ({len(breakouts)} breakouts)")
    print(f"{'='*60}")

    if len(breakouts) > 0:
        print(f"\nBreakout rate: {len(breakouts)/len(comp_df)*100:.1f}%")
        print(f"Mean breakout actual: {breakouts['actual'].mean():.1f}")
        print(f"Mean breakout expected: {breakouts['expected'].mean():.1f}")
        print(f"Mean ratio: {breakouts['ratio'].mean():.2f}x")

        # By role
        print("\nBreakouts by role:")
        role_breakouts = breakouts.groupby("player_role").agg({
            "player_name": "count",
            "actual": "mean",
            "expected": "mean",
            "ratio": "mean",
        }).rename(columns={"player_name": "count"})
        print(role_breakouts.to_string())

        # Top breakout performances
        print("\nTop 10 breakout performances:")
        top_breakouts = breakouts.nlargest(10, "residual")
        for _, row in top_breakouts.iterrows():
            print(f"  {row['player_name']:25} {row['player_role']:4} "
                  f"Actual: {row['actual']:5.0f} Expected: {row['expected']:5.1f} "
                  f"Ratio: {row['ratio']:.2f}x")

        results["breakouts"] = breakouts.to_dict("records")

    # Above ceiling analysis
    above_ceiling = comp_df[comp_df["above_ceiling"]]
    print(f"\n{'='*60}")
    print(f"CEILING CALIBRATION ({len(above_ceiling)} above q90)")
    print(f"{'='*60}")
    print(f"Above ceiling rate: {len(above_ceiling)/len(comp_df)*100:.1f}% (target: ~10%)")

    if len(above_ceiling) > 0:
        # By role
        print("\nAbove ceiling by role:")
        role_ceiling = comp_df.groupby("player_role").agg({
            "above_ceiling": "mean",
        })
        role_ceiling["above_ceiling_pct"] = role_ceiling["above_ceiling"] * 100
        print(role_ceiling.to_string())

    # Prediction errors by player role
    print(f"\n{'='*60}")
    print("PREDICTION ERRORS BY ROLE")
    print(f"{'='*60}")
    role_errors = comp_df.groupby("player_role").agg({
        "actual": "mean",
        "expected": "mean",
        "residual": ["mean", "std"],
    })
    role_errors.columns = ["actual_mean", "expected_mean", "residual_mean", "residual_std"]
    role_errors["mae"] = comp_df.groupby("player_role")["residual"].apply(lambda x: np.abs(x).mean())
    print(role_errors.to_string())

    # High-value missed players (should have been selected but weren't predicted well)
    print(f"\n{'='*60}")
    print("HIGH-VALUE MISSED PLAYERS (actual > 80, expected < 40)")
    print(f"{'='*60}")
    missed = comp_df[(comp_df["actual"] > 80) & (comp_df["expected"] < 40)]
    print(f"Count: {len(missed)}")
    if len(missed) > 0:
        for _, row in missed.head(10).iterrows():
            print(f"  {row['player_name']:25} {row['player_role']:4} "
                  f"Actual: {row['actual']:5.0f} Expected: {row['expected']:5.1f}")

    # Save detailed analysis
    output_dir = Path("tmp/breakout_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    comp_df.to_csv(output_dir / "prediction_comparison.csv", index=False)
    breakouts.to_csv(output_dir / "breakouts.csv", index=False) if len(breakouts) > 0 else None

    print(f"\nResults saved to {output_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze breakout performances")
    parser.add_argument(
        "--features",
        default="tmp/full_player_match_features_v3.csv",
        help="Path to features CSV",
    )
    parser.add_argument(
        "--model-dir",
        default="tmp/quantile_models",
        help="Directory with quantile models",
    )
    parser.add_argument(
        "--n-matches",
        type=int,
        default=50,
        help="Number of matches to analyze",
    )
    args = parser.parse_args()

    analyze_breakouts(args.features, args.model_dir, args.n_matches)


if __name__ == "__main__":
    main()

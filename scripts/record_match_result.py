"""Record actual match results and update the residual adapter.

After each match completes, run this script with the actual fantasy points
to update the test-time training layer for future matches.

Usage:
    python scripts/record_match_result.py \
        --match-id "ipl2026_01" \
        --match-date "2026-03-28" \
        --actuals data/match_results/ipl2026_01.csv \
        --predictions tmp/last_predictions.json

    # CSV format: player_name,actual_points
    # predictions JSON is auto-saved by generate_match_team.py

    # Or provide actuals inline:
    python scripts/record_match_result.py \
        --match-id "ipl2026_01" \
        --match-date "2026-03-28" \
        --predictions tmp/last_predictions.json \
        --inline "Virat Kohli:45,Jasprit Bumrah:62,..."
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.ipl_fantasy.residual_adapter import ResidualAdapter

DEFAULT_ADAPTER_PATH = project_root / "tmp" / "residual_adapter.json"
DEFAULT_PREDICTIONS_PATH = project_root / "tmp" / "last_predictions.json"


def load_actuals_csv(path: str) -> dict[str, float]:
    """Load actual points from CSV (player_name, actual_points)."""
    df = pd.read_csv(path)
    return dict(zip(df["player_name"], df["actual_points"].astype(float)))


def parse_inline_actuals(inline: str) -> dict[str, float]:
    """Parse 'Name1:pts1,Name2:pts2,...' format."""
    actuals = {}
    for pair in inline.split(","):
        pair = pair.strip()
        if ":" not in pair:
            continue
        name, pts = pair.rsplit(":", 1)
        actuals[name.strip()] = float(pts.strip())
    return actuals


def main():
    parser = argparse.ArgumentParser(description="Record match results for test-time training")
    parser.add_argument("--match-id", required=True, help="Unique match identifier")
    parser.add_argument("--match-date", required=True, help="Match date (YYYY-MM-DD)")
    parser.add_argument("--actuals", default=None, help="CSV with actual points (player_name,actual_points)")
    parser.add_argument("--inline", default=None, help="Inline actuals: 'Name:pts,Name:pts,...'")
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS_PATH),
                        help="Path to saved predictions JSON")
    parser.add_argument("--adapter-state", default=str(DEFAULT_ADAPTER_PATH),
                        help="Path to adapter state file")
    args = parser.parse_args()

    # Load actuals
    if args.actuals:
        actuals = load_actuals_csv(args.actuals)
    elif args.inline:
        actuals = parse_inline_actuals(args.inline)
    else:
        print("ERROR: Provide --actuals (CSV) or --inline actuals.")
        sys.exit(1)

    print(f"Match: {args.match_id} ({args.match_date})")
    print(f"Actual points for {len(actuals)} players loaded.")

    # Load predictions
    pred_path = Path(args.predictions)
    if not pred_path.exists():
        print(f"ERROR: Predictions file not found: {pred_path}")
        print("  Run generate_match_team.py first (it saves predictions automatically).")
        sys.exit(1)

    predictions = json.loads(pred_path.read_text())
    print(f"Loaded predictions for {len(predictions)} players.")

    # Load or create adapter
    adapter = ResidualAdapter.load(args.adapter_state)
    prev_matches = adapter.total_matches
    print(f"Adapter state: {prev_matches} matches previously observed.")

    # Observe
    summary = adapter.observe(
        match_id=args.match_id,
        match_date=args.match_date,
        predictions=predictions,
        actuals=actuals,
    )

    # Save
    adapter.save(args.adapter_state)
    print(f"\nAdapter updated and saved to {args.adapter_state}")

    # Print summary
    print(f"\n--- Match Summary ---")
    print(f"  Players matched: {summary['n_players']}")
    print(f"  Mean residual:   {summary['mean_residual']:+.1f} pts")
    print(f"  Std residual:    {summary['std_residual']:.1f} pts")
    print(f"  Total matches:   {summary['total_matches_observed']}")
    print(f"  Players tracked: {summary['players_tracked']}")

    # Show biggest misses
    matched = {
        name: {
            "predicted": predictions[name]["predicted"],
            "actual": actuals[name],
            "residual": actuals[name] - predictions[name]["predicted"],
        }
        for name in predictions
        if name in actuals
    }
    if matched:
        sorted_by_miss = sorted(matched.items(), key=lambda x: abs(x[1]["residual"]), reverse=True)
        print(f"\n  Top prediction misses:")
        print(f"  {'Player':<28} {'Predicted':>9} {'Actual':>7} {'Residual':>9}")
        print(f"  {'-'*55}")
        for name, info in sorted_by_miss[:10]:
            print(f"  {name:<28} {info['predicted']:>9.1f} {info['actual']:>7.1f} {info['residual']:>+9.1f}")

    print(f"\n{adapter.get_summary()}")


if __name__ == "__main__":
    main()

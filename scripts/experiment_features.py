"""
Experiment: Test two feature improvements for fantasy optimizer.

1. Verify batting position feature impact (already in OPTIMAL_FEATURES)
2. Test role-stratified opponent features (new)

Backtests against IPL 2025 data and compares to baseline.
"""
import sys
import json
from pathlib import Path

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ipl_fantasy.quantile_model import (
    QuantileModelEnsemble,
    OPTIMAL_FEATURES,
    TARGET_COLUMN,
)
from src.ipl_fantasy.enhanced_prediction import (
    create_enhanced_predict_fn,
    OPTIMAL_CONFIG,
)
from src.ipl_fantasy.backtesting import Backtester
from src.ipl_fantasy.team_optimizer import Dream11Constraints


FEATURES_CSV = Path("tmp/full_player_match_features_v4.csv")


def load_and_split(df: pd.DataFrame):
    """Split into train (pre-2025) and test (IPL 2025)."""
    ipl = df[df["competition"] == "Indian Premier League"].copy()
    # Train: everything before IPL 2025 (all competitions for broader data)
    # Test: only IPL 2025
    train = df[df["match_date"] < ipl[ipl["season"].astype(str) == "2025"]["match_date"].min()].copy()
    test = ipl[ipl["season"].astype(str) == "2025"].copy()
    return train, test


def add_role_stratified_opponent_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add per-role opponent features.

    Instead of a single opponent_points_avg_all, compute:
    - opponent_points_avg_bat: avg points vs this opponent for batters
    - opponent_points_avg_bowl: avg points vs this opponent for bowlers
    - opponent_points_avg_ar: avg points vs this opponent for all-rounders
    - opponent_points_avg_wk: avg points vs this opponent for wicket-keepers
    - opponent_role_relative: player's vs-opponent avg minus role avg vs-opponent

    These are computed from historical data, respecting temporal ordering.
    """
    df = df.sort_values(["match_date", "match_id"]).reset_index(drop=True)

    # For each player-match row, we need: avg fantasy points of players of the SAME ROLE
    # against this specific opponent, computed from prior matches only.
    # This is computationally expensive, so we approximate using expanding means.

    # Group by (player_role, opponent) and compute expanding mean of dream11_points_total
    # shifted by 1 to avoid leakage
    role_opponent_avgs = {}
    opponent_role_cols = {}

    for role in ["BAT", "BOWL", "AR", "WK"]:
        col_name = f"opponent_points_avg_{role.lower()}"
        opponent_role_cols[role] = col_name
        df[col_name] = np.nan

    df["opponent_role_relative"] = np.nan

    # Compute per-role opponent averages using expanding window
    # Group by opponent and player_role, compute shifted expanding mean
    for (opponent, role), group in df.groupby(["opponent", "player_role"]):
        if role not in opponent_role_cols:
            continue
        col_name = opponent_role_cols[role]
        # Expanding mean of dream11_points_total shifted by group
        shifted_mean = group[TARGET_COLUMN].expanding().mean().shift(1)
        df.loc[group.index, col_name] = shifted_mean

    # For opponent_role_relative: difference between this player's opponent avg
    # and the role-specific opponent avg
    for role in ["BAT", "BOWL", "AR", "WK"]:
        col_name = opponent_role_cols[role]
        mask = df["player_role"] == role
        df.loc[mask, "opponent_role_relative"] = (
            df.loc[mask, "opponent_points_avg_all"].fillna(0) -
            df.loc[mask, col_name].fillna(0)
        )

    return df


def run_backtest(
    df: pd.DataFrame,
    features: list[str],
    label: str,
    n_matches: int | None = None,
) -> dict:
    """Train model with given features and backtest on IPL 2025."""
    train, test = load_and_split(df)

    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: {label}")
    print(f"{'='*60}")
    print(f"  Features: {len(features)}")
    print(f"  Train rows: {len(train)}, Test rows: {len(test)}")
    print(f"  Test matches: {test['match_id'].nunique()}")

    # Train ensemble with specified features
    ensemble = QuantileModelEnsemble(features=features)

    # Use chronological split for validation within train set
    train_sorted = train.sort_values("match_date")
    split_idx = int(len(train_sorted) * 0.8)
    train_split = train_sorted.iloc[:split_idx]
    val_split = train_sorted.iloc[split_idx:]

    metrics = ensemble.fit(train_split, val_split)
    print(f"  Val RMSE: {metrics.get('mean_rmse', 'N/A'):.2f}")
    print(f"  Val MAE: {metrics.get('mean_mae', 'N/A'):.2f}")

    # Create predict function
    predict_fn = create_enhanced_predict_fn(ensemble, OPTIMAL_CONFIG)

    # Backtest
    backtester = Backtester(constraints=Dream11Constraints())
    summary = backtester.backtest_multiple(
        test,
        predict_fn,
        n_matches=n_matches,
        competition_filter=None,  # Already filtered to IPL 2025
    )

    print(f"\n  RESULTS:")
    print(f"  Matches evaluated: {summary.n_matches}")
    print(f"  Mean selected score: {summary.mean_selected_score:.1f}")
    print(f"  Mean oracle score: {summary.mean_oracle_score:.1f}")
    print(f"  Mean total regret: {summary.mean_total_regret:.1f}")
    print(f"  Mean team regret: {summary.mean_team_regret:.1f}")
    print(f"  Mean captain regret: {summary.mean_captain_regret:.1f}")
    print(f"  Mean VC regret: {summary.mean_vc_regret:.1f}")
    print(f"  Player overlap: {summary.mean_overlap_pct:.1f}%")
    print(f"  Captain accuracy: {summary.top_captain_rate:.1f}%")
    print(f"  VC accuracy: {summary.top_vc_rate:.1f}%")

    return {
        "label": label,
        "n_features": len(features),
        "n_matches": summary.n_matches,
        "mean_selected": summary.mean_selected_score,
        "mean_oracle": summary.mean_oracle_score,
        "mean_total_regret": summary.mean_total_regret,
        "mean_team_regret": summary.mean_team_regret,
        "mean_captain_regret": summary.mean_captain_regret,
        "mean_vc_regret": summary.mean_vc_regret,
        "overlap_pct": summary.mean_overlap_pct,
        "captain_acc": summary.top_captain_rate,
        "vc_acc": summary.top_vc_rate,
        "val_rmse": metrics.get("mean_rmse", None),
        "val_mae": metrics.get("mean_mae", None),
    }


def main():
    print("Loading features CSV...")
    df = pd.read_csv(FEATURES_CSV, low_memory=False)
    print(f"  Total rows: {len(df)}")

    results = []

    # ── Experiment 1: Baseline (current OPTIMAL_FEATURES) ──
    baseline_result = run_backtest(df, OPTIMAL_FEATURES, "Baseline (current)")
    results.append(baseline_result)

    # ── Experiment 2: Without batting position (ablation) ──
    features_no_batpos = [f for f in OPTIMAL_FEATURES if f != "rolling_batting_position_avg_5_all"]
    ablation_result = run_backtest(df, features_no_batpos, "Without batting position (ablation)")
    results.append(ablation_result)

    # ── Experiment 3: Role-stratified opponent features ──
    print("\n\nAdding role-stratified opponent features to dataset...")
    df_enhanced = add_role_stratified_opponent_features(df)

    # New features to add
    new_opponent_features = [
        "opponent_points_avg_bat",
        "opponent_points_avg_bowl",
        "opponent_points_avg_ar",
        "opponent_points_avg_wk",
        "opponent_role_relative",
    ]

    # Check coverage
    ipl25 = df_enhanced[
        (df_enhanced["competition"] == "Indian Premier League") &
        (df_enhanced["season"].astype(str) == "2025")
    ]
    for col in new_opponent_features:
        non_null = ipl25[col].notna().sum()
        print(f"  {col}: {non_null}/{len(ipl25)} non-null ({non_null/len(ipl25)*100:.0f}%)")

    # Test: replace opponent_points_avg_all with role-stratified version
    features_role_opponent = [
        f for f in OPTIMAL_FEATURES if f != "opponent_points_avg_all"
    ] + new_opponent_features
    role_opponent_result = run_backtest(
        df_enhanced, features_role_opponent,
        "Role-stratified opponent (replace generic)"
    )
    results.append(role_opponent_result)

    # ── Experiment 4: Add role-stratified ON TOP of generic ──
    features_both_opponent = OPTIMAL_FEATURES + new_opponent_features
    both_opponent_result = run_backtest(
        df_enhanced, features_both_opponent,
        "Both generic + role-stratified opponent"
    )
    results.append(both_opponent_result)

    # ── Experiment 5: Just opponent_role_relative as extra feature ──
    features_relative = OPTIMAL_FEATURES + ["opponent_role_relative"]
    relative_result = run_backtest(
        df_enhanced, features_relative,
        "Baseline + opponent_role_relative only"
    )
    results.append(relative_result)

    # ── Summary comparison ──
    print("\n\n" + "=" * 90)
    print("  FEATURE EXPERIMENT COMPARISON")
    print("=" * 90)
    print(f"  {'Experiment':<45s}  {'Regret':>8s}  {'Team':>8s}  {'Capt':>8s}  {'Score':>8s}  {'Overlap':>8s}  {'C%':>5s}")
    print("  " + "-" * 88)

    baseline_regret = results[0]["mean_total_regret"]
    for r in results:
        delta = r["mean_total_regret"] - baseline_regret
        delta_str = f"({delta:+.1f})" if r["label"] != "Baseline (current)" else ""
        print(
            f"  {r['label']:<45s}  {r['mean_total_regret']:>7.1f}{delta_str:>8s}  "
            f"{r['mean_team_regret']:>8.1f}  {r['mean_captain_regret']:>8.1f}  "
            f"{r['mean_selected']:>8.1f}  {r['overlap_pct']:>7.1f}%  {r['captain_acc']:>4.0f}%"
        )

    print("\n  CONCLUSION:")
    best = min(results, key=lambda r: r["mean_total_regret"])
    if best["label"] == "Baseline (current)":
        print("  => Neither feature improved over baseline. No changes needed.")
    else:
        improvement = baseline_regret - best["mean_total_regret"]
        print(f"  => Best: '{best['label']}' — {improvement:.1f} pts less regret than baseline")

    # Save results
    output_path = Path("tmp/feature_experiment_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()

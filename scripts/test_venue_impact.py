"""Measure the impact of venue adjustments on backtest regret.

Compares three strategies on the same matches:
  1. Baseline (no venue adjustment)
  2. Venue-adjusted predictions
  3. Oracle (best possible)

Usage:
    python scripts/test_venue_impact.py [--n-matches 100]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.ipl_fantasy.backtesting import Backtester
from src.ipl_fantasy.credit_estimation import estimate_credits_from_history
from src.ipl_fantasy.enhanced_prediction import OPTIMAL_CONFIG, create_enhanced_predict_fn
from src.ipl_fantasy.improved_optimizer import ImprovedDream11Optimizer, OptimizationConfig
from src.ipl_fantasy.quantile_model import QuantileModelEnsemble
from src.ipl_fantasy.team_optimizer import Dream11Constraints, Player

# Import venue functions from the match pipeline
sys.path.insert(0, str(project_root / "scripts"))
from generate_match_team import load_venue_profiles, match_venue_profile, apply_venue_adjustments


def _actual_points(match_df: pd.DataFrame) -> dict[str, float]:
    return {
        row["player_name"]: row["dream11_points_total"]
        for _, row in match_df.iterrows()
        if pd.notna(row.get("dream11_points_total", None))
    }


def _score_team(result, actual: dict[str, float]) -> float:
    names = [p.name for p in result.selected_players]
    cap = result.captain.name if result.captain else ""
    vc = result.vice_captain.name if result.vice_captain else ""
    base = sum(actual.get(n, 0) for n in names)
    return base + actual.get(cap, 0) + actual.get(vc, 0) * 0.5


def _oracle_score(match_df: pd.DataFrame, actual: dict[str, float], constraints) -> float:
    backtester = Backtester(constraints=constraints)
    pairs = []
    for _, row in match_df.iterrows():
        name = row.get("player_name", "")
        role = row.get("player_role", "BAT")
        if role not in ("WK", "BAT", "AR", "BOWL"):
            role = "BAT"
        avg = row.get("rolling_points_avg_10_all", row.get("rolling_points_avg_5_all", 30.0))
        rec = row.get("rolling_points_avg_5_all", None)
        cred = estimate_credits_from_history(
            player_name=name, player_role=role,
            avg_points_all=avg if pd.notna(avg) else 30.0,
            avg_points_recent=rec if pd.notna(rec) else None,
        )
        p = Player(name=name, team=row.get("team", ""), role=role,
                    predicted_points=actual.get(name, 0), credits=cred)
        pairs.append((p, actual.get(name, 0)))

    oracle = backtester.calculate_oracle_team(pairs)
    onames = [p.name for p in oracle.selected_players]
    ocap = oracle.captain.name if oracle.captain else ""
    ovc = oracle.vice_captain.name if oracle.vice_captain else ""
    return backtester.calculate_score_with_cv(onames, ocap, ovc, actual)


def main():
    parser = argparse.ArgumentParser(description="Test venue adjustment impact")
    parser.add_argument("--features", default="tmp/full_player_match_features_v4.csv")
    parser.add_argument("--models", default="tmp/quantile_models")
    parser.add_argument("--n-matches", type=int, default=100)
    parser.add_argument("--competition", default="Indian Premier League")
    args = parser.parse_args()

    print(f"Loading features from {args.features}...")
    df = pd.read_csv(args.features, low_memory=False)
    if args.competition:
        df = df[df["competition"] == args.competition]
    print(f"  {len(df)} rows, {df['match_id'].nunique()} matches")

    print(f"Loading models from {args.models}...")
    ensemble = QuantileModelEnsemble.load(args.models)
    predict_fn = create_enhanced_predict_fn(ensemble, config=OPTIMAL_CONFIG, use_improved_credits=True)
    constraints = Dream11Constraints()
    optimizer = ImprovedDream11Optimizer(constraints=constraints, config=OptimizationConfig())

    venue_profiles = load_venue_profiles()
    print(f"  {len(set(v['venue_key'] for v in venue_profiles.values()))} venue profiles loaded")

    matches = df.groupby("match_id").first().reset_index()
    matches = matches.sort_values("match_date", ascending=False).head(args.n_matches)
    matches = matches.sort_values("match_date", ascending=True)
    match_ids = matches["match_id"].tolist()
    print(f"  Evaluating {len(match_ids)} matches\n")

    base_scores, base_regrets = [], []
    venue_scores, venue_regrets = [], []
    venue_matched_count = 0

    for i, mid in enumerate(match_ids):
        match_df = df[df["match_id"] == mid]
        actual = _actual_points(match_df)
        if len(actual) < 15:
            continue

        venue = match_df["venue"].iloc[0] if "venue" in match_df.columns else ""
        venue_profile = match_venue_profile(venue, venue_profiles)
        if venue_profile:
            venue_matched_count += 1

        try:
            players = predict_fn(match_df)
        except Exception as e:
            print(f"  Match {i+1}: prediction error — {e}")
            continue

        try:
            oracle = _oracle_score(match_df, actual, constraints)
        except Exception:
            continue

        # Baseline (no venue adjustment)
        try:
            bl_result = optimizer.optimize_ceiling_weighted(players)
            bl_score = _score_team(bl_result, actual)
        except Exception:
            bl_score = 0.0

        # Venue-adjusted
        try:
            bowling_styles = {}
            for _, row in match_df.iterrows():
                bs = row.get("bowling_style", "")
                bowling_styles[row["player_name"]] = bs if pd.notna(bs) else ""
            adj_players = apply_venue_adjustments(players, venue_profile, bowling_styles)
            va_result = optimizer.optimize_ceiling_weighted(adj_players)
            va_score = _score_team(va_result, actual)
        except Exception:
            va_score = bl_score

        base_scores.append(bl_score)
        base_regrets.append(oracle - bl_score)
        venue_scores.append(va_score)
        venue_regrets.append(oracle - va_score)

        if (i + 1) % 10 == 0 or i == 0:
            r10_bl = np.mean(base_regrets[-10:])
            r10_va = np.mean(venue_regrets[-10:])
            print(
                f"  Match {i+1:>3}/{len(match_ids)}: "
                f"BL={bl_score:>6.1f} VA={va_score:>6.1f} Oracle={oracle:>6.1f}  "
                f"(rolling10 regret: BL={r10_bl:.1f}, VA={r10_va:.1f})"
            )

    n = len(base_scores)
    if n == 0:
        print("No matches evaluated.")
        return

    bl_mean = np.mean(base_regrets)
    va_mean = np.mean(venue_regrets)
    delta = va_mean - bl_mean
    wins = sum(1 for v, b in zip(venue_regrets, base_regrets) if v < b)
    ties = sum(1 for v, b in zip(venue_regrets, base_regrets) if v == b)
    losses = n - wins - ties

    print(f"\n{'=' * 65}")
    print(f"VENUE ADJUSTMENT IMPACT")
    print(f"{'=' * 65}")
    print(f"Matches evaluated: {n}")
    print(f"Venues matched to profile: {venue_matched_count}/{n}")
    print(f"\n{'Metric':<30} {'Baseline':>12} {'Venue-Adj':>12} {'Delta':>10}")
    print("-" * 65)
    print(f"{'Mean score':<30} {np.mean(base_scores):>12.1f} {np.mean(venue_scores):>12.1f} {np.mean(venue_scores)-np.mean(base_scores):>+10.1f}")
    print(f"{'Mean regret':<30} {bl_mean:>12.1f} {va_mean:>12.1f} {delta:>+10.1f}")
    print(f"{'Median regret':<30} {np.median(base_regrets):>12.1f} {np.median(venue_regrets):>12.1f}")
    print(f"{'P90 regret':<30} {np.percentile(base_regrets, 90):>12.1f} {np.percentile(venue_regrets, 90):>12.1f}")

    print(f"\nVenue-Adj vs Baseline: {wins}W / {ties}T / {losses}L")
    if wins + losses > 0:
        print(f"  Venue-Adj better in {wins/(wins+losses)*100:.0f}% of non-tied matches")

    # Breakdown by venue-matched vs unmatched
    matched_delta = []
    unmatched_delta = []
    matched_idx = 0
    for i, mid in enumerate(match_ids):
        if i >= n:
            break
        match_df = df[df["match_id"] == mid]
        actual = _actual_points(match_df)
        if len(actual) < 15:
            continue
        venue = match_df["venue"].iloc[0] if "venue" in match_df.columns else ""
        profile = match_venue_profile(venue, venue_profiles)
        d = base_regrets[matched_idx] - venue_regrets[matched_idx]
        if profile:
            matched_delta.append(d)
        else:
            unmatched_delta.append(d)
        matched_idx += 1
        if matched_idx >= n:
            break

    if matched_delta:
        print(f"\n  Venue-matched matches ({len(matched_delta)}): avg improvement = {np.mean(matched_delta):+.1f} pts")
    if unmatched_delta:
        print(f"  Unmatched matches ({len(unmatched_delta)}): avg improvement = {np.mean(unmatched_delta):+.1f} pts (should be ~0)")

    print("=" * 65)


if __name__ == "__main__":
    main()

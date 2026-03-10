"""Run the Live RL loop over historical matches.

This script:
1. Loads the trained quantile models and feature dataset.
2. Iterates chronologically through matches (online learning).
3. For each match, the contextual bandit selects a strategy arm.
4. After observing actual outcomes, the bandit updates.
5. Reports learning curves and compares RL vs static baseline.

Usage:
    python scripts/run_live_rl.py [--n-matches 50] [--alpha 1.5] [--output tmp/rl_results]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.ipl_fantasy.rl_policy import LiveRLAgent, build_default_arms, extract_match_context
from src.ipl_fantasy.team_reranker import RerankingConfig
from src.ipl_fantasy.enhanced_prediction import create_enhanced_predict_fn, OPTIMAL_CONFIG
from src.ipl_fantasy.improved_optimizer import ImprovedDream11Optimizer, OptimizationConfig
from src.ipl_fantasy.quantile_model import QuantileModelEnsemble
from src.ipl_fantasy.backtesting import Backtester
from src.ipl_fantasy.team_optimizer import Dream11Constraints
from src.ipl_fantasy.reward_model import compute_reward, RewardConfig


def run_static_baseline(
    ensemble: QuantileModelEnsemble,
    match_df: pd.DataFrame,
    actual_points: dict[str, float],
    constraints: Dream11Constraints,
) -> tuple[float, float]:
    """Run the static baseline (balanced arm) and return (selected_score, oracle_score)."""
    predict_fn = create_enhanced_predict_fn(ensemble, config=OPTIMAL_CONFIG, use_improved_credits=True)
    players = predict_fn(match_df)

    optimizer = ImprovedDream11Optimizer(constraints=constraints, config=OptimizationConfig())
    result = optimizer.optimize_ceiling_weighted(players)

    selected_names = [p.name for p in result.selected_players]
    cap_name = result.captain.name if result.captain else ""
    vc_name = result.vice_captain.name if result.vice_captain else ""

    base = sum(actual_points.get(p, 0) for p in selected_names)
    cap_bonus = actual_points.get(cap_name, 0)
    vc_bonus = actual_points.get(vc_name, 0) * 0.5
    selected_score = base + cap_bonus + vc_bonus

    backtester = Backtester(constraints=constraints)
    from src.ipl_fantasy.credit_estimation import estimate_credits_from_history
    from src.ipl_fantasy.team_optimizer import Player

    oracle_pairs = []
    for _, row in match_df.iterrows():
        name = row.get("player_name", "")
        role = row.get("player_role", "BAT")
        if role not in ("WK", "BAT", "AR", "BOWL"):
            role = "BAT"
        avg_all = row.get("rolling_points_avg_10_all", row.get("rolling_points_avg_5_all", 30.0))
        avg_recent = row.get("rolling_points_avg_5_all", None)
        credits = estimate_credits_from_history(
            player_name=name, player_role=role,
            avg_points_all=avg_all if pd.notna(avg_all) else 30.0,
            avg_points_recent=avg_recent if pd.notna(avg_recent) else None,
        )
        player = Player(name=name, team=row.get("team", ""), role=role,
                        predicted_points=actual_points.get(name, 0), credits=credits)
        oracle_pairs.append((player, actual_points.get(name, 0)))

    oracle_result = backtester.calculate_oracle_team(oracle_pairs)
    oracle_names = [p.name for p in oracle_result.selected_players]
    oracle_cap = oracle_result.captain.name if oracle_result.captain else ""
    oracle_vc = oracle_result.vice_captain.name if oracle_result.vice_captain else ""
    oracle_score = backtester.calculate_score_with_cv(oracle_names, oracle_cap, oracle_vc, actual_points)

    return selected_score, oracle_score


def main():
    parser = argparse.ArgumentParser(description="Run Live RL backtest")
    parser.add_argument("--features", default="tmp/full_player_match_features_v4.csv",
                        help="Path to features CSV")
    parser.add_argument("--models", default="tmp/quantile_models",
                        help="Path to trained quantile models")
    parser.add_argument("--n-matches", type=int, default=50,
                        help="Number of matches to evaluate")
    parser.add_argument("--alpha", type=float, default=1.5,
                        help="LinUCB exploration parameter")
    parser.add_argument("--competition", default="Indian Premier League",
                        help="Competition filter")
    parser.add_argument("--output", default="tmp/rl_results",
                        help="Output directory for results")
    parser.add_argument("--resume", default=None,
                        help="Path to saved agent state to resume from")
    parser.add_argument("--rerank", action="store_true",
                        help="Use Phase 2 reranking (generate K teams, simulate, rerank)")
    parser.add_argument("--rerank-candidates", type=int, default=8,
                        help="Number of candidate teams for reranking")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading features from {args.features}...")
    df = pd.read_csv(args.features, low_memory=False)
    print(f"  Total rows: {len(df)}")

    if args.competition:
        df = df[df["competition"] == args.competition]
        print(f"  After filtering to {args.competition}: {len(df)} rows")

    # Get unique matches sorted chronologically (oldest first for online learning)
    matches = df.groupby("match_id").first().reset_index()
    matches = matches.sort_values("match_date", ascending=True)

    if args.n_matches:
        # Take the most recent N matches
        matches = matches.tail(args.n_matches)

    match_ids = matches["match_id"].tolist()
    print(f"  Evaluating {len(match_ids)} matches chronologically")

    print(f"\nInitializing Live RL agent (alpha={args.alpha})...")
    agent = LiveRLAgent.from_models(
        ensemble_path=args.models,
        agent_path=args.resume,
        alpha=args.alpha,
    )

    arms = build_default_arms()
    print(f"  {len(arms)} strategy arms:")
    for i, arm in enumerate(arms):
        print(f"    [{i}] {arm.name}")

    rerank_cfg = None
    if args.rerank:
        rerank_cfg = RerankingConfig(
            n_candidates=args.rerank_candidates,
            n_simulations=3000,
        )
        print(f"  Reranking enabled: {args.rerank_candidates} candidates per match")

    mode_label = "LIVE RL + RERANKING" if args.rerank else "LIVE RL"
    print(f"\n{'=' * 70}")
    print(f"STARTING {mode_label} LOOP")
    print("=" * 70)

    rl_scores = []
    rl_regrets = []
    baseline_scores = []
    baseline_regrets = []
    arm_choices = []
    constraints = Dream11Constraints()

    for i, match_id in enumerate(match_ids):
        match_df = df[df["match_id"] == match_id]

        actual_points = {
            row["player_name"]: row["dream11_points_total"]
            for _, row in match_df.iterrows()
            if pd.notna(row.get("dream11_points_total", None))
        }

        if len(actual_points) < 15:
            continue

        try:
            sim_cap, sim_vc = None, None
            if args.rerank:
                result, arm_idx, context, sim_cap, sim_vc = (
                    agent.select_team_with_reranking(match_df, rerank_cfg)
                )
            else:
                result, arm_idx, context = agent.select_team(match_df)

            exp = agent.observe(
                match_df, arm_idx, context, result, actual_points,
                override_captain=sim_cap, override_vc=sim_vc,
            )

            rl_scores.append(exp.selected_score)
            rl_regrets.append(exp.regret)
            arm_choices.append(arm_idx)

            bl_score, bl_oracle = run_static_baseline(
                agent.ensemble, match_df, actual_points, constraints,
            )
            baseline_scores.append(bl_score)
            baseline_regrets.append(bl_oracle - bl_score)

            if (i + 1) % 5 == 0 or i == 0:
                recent_rl = np.mean(rl_regrets[-10:]) if rl_regrets else 0
                recent_bl = np.mean(baseline_regrets[-10:]) if baseline_regrets else 0
                print(
                    f"  Match {i+1:>3}/{len(match_ids)}: "
                    f"arm={arms[arm_idx].name:<20} "
                    f"RL_regret={exp.regret:>6.1f}  "
                    f"BL_regret={bl_oracle - bl_score:>6.1f}  "
                    f"(rolling10: RL={recent_rl:.1f}, BL={recent_bl:.1f})"
                )

        except Exception as e:
            print(f"  Match {i+1}: ERROR - {e}")
            continue

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    if not rl_scores:
        print("No matches were successfully evaluated.")
        return

    print(f"\nMatches evaluated: {len(rl_scores)}")

    print(f"\nRL AGENT:")
    print(f"  Mean selected score: {np.mean(rl_scores):.1f}")
    print(f"  Mean regret:         {np.mean(rl_regrets):.1f}")
    print(f"  Median regret:       {np.median(rl_regrets):.1f}")

    print(f"\nSTATIC BASELINE (balanced):")
    print(f"  Mean selected score: {np.mean(baseline_scores):.1f}")
    print(f"  Mean regret:         {np.mean(baseline_regrets):.1f}")
    print(f"  Median regret:       {np.median(baseline_regrets):.1f}")

    rl_mean_regret = np.mean(rl_regrets)
    bl_mean_regret = np.mean(baseline_regrets)
    delta = rl_mean_regret - bl_mean_regret
    print(f"\nRL vs BASELINE:")
    print(f"  Regret difference: {delta:+.1f} ({'RL better' if delta < 0 else 'Baseline better'})")
    print(f"  Score difference:  {np.mean(rl_scores) - np.mean(baseline_scores):+.1f}")

    if len(rl_regrets) >= 10:
        mid = len(rl_regrets) // 2
        first_half = np.mean(rl_regrets[:mid])
        second_half = np.mean(rl_regrets[mid:])
        print(f"\nLEARNING TREND:")
        print(f"  First half avg regret:  {first_half:.1f}")
        print(f"  Second half avg regret: {second_half:.1f}")
        print(f"  Improvement: {first_half - second_half:+.1f}")

    print(agent.get_summary())

    print("\nArm Selection Distribution:")
    from collections import Counter
    arm_counts = Counter(arm_choices)
    for arm_idx, count in sorted(arm_counts.items(), key=lambda x: -x[1]):
        pct = count / len(arm_choices) * 100
        avg_regret = np.mean([r for r, a in zip(rl_regrets, arm_choices) if a == arm_idx])
        print(f"  {arms[arm_idx].name:<25} {count:>4} ({pct:>5.1f}%)  avg_regret={avg_regret:.1f}")

    results = {
        "n_matches": len(rl_scores),
        "alpha": args.alpha,
        "rl": {
            "mean_score": float(np.mean(rl_scores)),
            "mean_regret": float(np.mean(rl_regrets)),
            "median_regret": float(np.median(rl_regrets)),
        },
        "baseline": {
            "mean_score": float(np.mean(baseline_scores)),
            "mean_regret": float(np.mean(baseline_regrets)),
            "median_regret": float(np.median(baseline_regrets)),
        },
        "delta_regret": float(delta),
        "arm_distribution": {arms[k].name: v for k, v in arm_counts.items()},
    }
    (output_dir / "rl_results.json").write_text(json.dumps(results, indent=2))

    match_detail = []
    for j in range(len(rl_scores)):
        match_detail.append({
            "match_index": j,
            "arm": arms[arm_choices[j]].name,
            "rl_score": rl_scores[j],
            "rl_regret": rl_regrets[j],
            "baseline_score": baseline_scores[j],
            "baseline_regret": baseline_regrets[j],
        })
    pd.DataFrame(match_detail).to_csv(output_dir / "match_detail.csv", index=False)

    agent.save(output_dir / "agent_state")
    agent.buffer.save(output_dir / "experience_buffer.json")

    print(f"\nResults saved to {output_dir}")
    print("Done.")


if __name__ == "__main__":
    main()

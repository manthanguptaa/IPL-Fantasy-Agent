"""Backtest the Phase 2 reranking pipeline against historical matches.

Compares three strategies:
  1. Static baseline (single balanced optimizer)
  2. Reranked best team (generate K → simulate → rerank → pick #1)
  3. Oracle (best possible team with actual points)

Usage:
    python scripts/run_reranking_backtest.py [--n-matches 50] [--candidates 8]
"""
from __future__ import annotations

import argparse
import json
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
from src.ipl_fantasy.team_reranker import (
    RerankingConfig,
    get_reranking_summary,
    select_best_team,
    select_top_k,
)


def _actual_points_from_df(match_df: pd.DataFrame) -> dict[str, float]:
    return {
        row["player_name"]: row["dream11_points_total"]
        for _, row in match_df.iterrows()
        if pd.notna(row.get("dream11_points_total", None))
    }


def _score_team(
    result,
    actual_points: dict[str, float],
    captain: str | None = None,
    vc: str | None = None,
) -> float:
    """Score a team using actual points with C/VC multipliers."""
    names = [p.name for p in result.selected_players]
    cap = captain or (result.captain.name if result.captain else "")
    v = vc or (result.vice_captain.name if result.vice_captain else "")
    base = sum(actual_points.get(n, 0) for n in names)
    return base + actual_points.get(cap, 0) + actual_points.get(v, 0) * 0.5


def _oracle_score(
    match_df: pd.DataFrame,
    actual_points: dict[str, float],
    constraints: Dream11Constraints,
) -> float:
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
                    predicted_points=actual_points.get(name, 0), credits=cred)
        pairs.append((p, actual_points.get(name, 0)))

    oracle = backtester.calculate_oracle_team(pairs)
    onames = [p.name for p in oracle.selected_players]
    ocap = oracle.captain.name if oracle.captain else ""
    ovc = oracle.vice_captain.name if oracle.vice_captain else ""
    return backtester.calculate_score_with_cv(onames, ocap, ovc, actual_points)


def main():
    parser = argparse.ArgumentParser(description="Backtest Phase 2 reranking")
    parser.add_argument("--features", default="tmp/full_player_match_features_v4.csv")
    parser.add_argument("--models", default="tmp/quantile_models")
    parser.add_argument("--n-matches", type=int, default=50)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--simulations", type=int, default=3000)
    parser.add_argument("--competition", default="Indian Premier League")
    parser.add_argument("--output", default="tmp/reranking_results")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading features from {args.features}...")
    df = pd.read_csv(args.features, low_memory=False)
    if args.competition:
        df = df[df["competition"] == args.competition]
    print(f"  {len(df)} rows, {df['match_id'].nunique()} matches")

    print(f"Loading models from {args.models}...")
    ensemble = QuantileModelEnsemble.load(args.models)

    predict_fn = create_enhanced_predict_fn(ensemble, config=OPTIMAL_CONFIG, use_improved_credits=True)
    constraints = Dream11Constraints()
    rerank_cfg = RerankingConfig(
        n_candidates=args.candidates,
        n_simulations=args.simulations,
    )

    matches = df.groupby("match_id").first().reset_index()
    matches = matches.sort_values("match_date", ascending=False).head(args.n_matches)
    matches = matches.sort_values("match_date", ascending=True)
    match_ids = matches["match_id"].tolist()
    print(f"  Evaluating {len(match_ids)} matches\n")

    baseline_scores, baseline_regrets = [], []
    reranked_scores, reranked_regrets = [], []
    detail_rows = []

    for i, mid in enumerate(match_ids):
        match_df = df[df["match_id"] == mid]
        actual = _actual_points_from_df(match_df)
        if len(actual) < 15:
            continue

        try:
            players = predict_fn(match_df)
        except Exception as e:
            print(f"  Match {i+1}: prediction error — {e}")
            continue

        try:
            oracle = _oracle_score(match_df, actual, constraints)
        except Exception:
            continue

        try:
            bl_opt = ImprovedDream11Optimizer(constraints=constraints, config=OptimizationConfig())
            bl_result = bl_opt.optimize_ceiling_weighted(players)
            bl_score = _score_team(bl_result, actual)
        except Exception:
            bl_score = 0.0

        try:
            best = select_best_team(players, constraints, rerank_cfg)
            rr_score = _score_team(
                best.result, actual,
                captain=best.sim_captain,
                vc=best.sim_vc,
            )
        except Exception as e:
            if args.verbose:
                print(f"  Match {i+1}: reranking error — {e}")
            rr_score = bl_score  # fallback

        bl_regret = oracle - bl_score
        rr_regret = oracle - rr_score

        baseline_scores.append(bl_score)
        baseline_regrets.append(bl_regret)
        reranked_scores.append(rr_score)
        reranked_regrets.append(rr_regret)

        detail_rows.append({
            "match_index": i,
            "match_id": mid,
            "oracle": oracle,
            "baseline_score": bl_score,
            "baseline_regret": bl_regret,
            "reranked_score": rr_score,
            "reranked_regret": rr_regret,
            "reranked_label": best.label if rr_score != bl_score else "same",
            "reranked_captain": best.sim_captain if rr_score != bl_score else "",
        })

        if (i + 1) % 5 == 0 or i == 0:
            r10_bl = np.mean(baseline_regrets[-10:])
            r10_rr = np.mean(reranked_regrets[-10:])
            print(
                f"  Match {i+1:>3}/{len(match_ids)}: "
                f"BL={bl_score:>6.1f} RR={rr_score:>6.1f} "
                f"Oracle={oracle:>6.1f}  "
                f"(rolling10 regret: BL={r10_bl:.1f}, RR={r10_rr:.1f})"
            )

    n = len(baseline_scores)
    if n == 0:
        print("No matches evaluated.")
        return

    bl_mean_regret = np.mean(baseline_regrets)
    rr_mean_regret = np.mean(reranked_regrets)
    delta = rr_mean_regret - bl_mean_regret
    wins = sum(1 for r, b in zip(reranked_regrets, baseline_regrets) if r < b)
    ties = sum(1 for r, b in zip(reranked_regrets, baseline_regrets) if r == b)
    losses = n - wins - ties

    print("\n" + "=" * 65)
    print("RERANKING BACKTEST RESULTS")
    print("=" * 65)
    print(f"Matches evaluated: {n}")
    print(f"\n{'Metric':<30} {'Baseline':>12} {'Reranked':>12} {'Delta':>10}")
    print("-" * 65)
    print(f"{'Mean score':<30} {np.mean(baseline_scores):>12.1f} {np.mean(reranked_scores):>12.1f} {np.mean(reranked_scores)-np.mean(baseline_scores):>+10.1f}")
    print(f"{'Mean regret':<30} {bl_mean_regret:>12.1f} {rr_mean_regret:>12.1f} {delta:>+10.1f}")
    print(f"{'Median regret':<30} {np.median(baseline_regrets):>12.1f} {np.median(reranked_regrets):>12.1f}")
    print(f"{'P90 regret':<30} {np.percentile(baseline_regrets, 90):>12.1f} {np.percentile(reranked_regrets, 90):>12.1f}")

    print(f"\nReranked vs Baseline: {wins}W / {ties}T / {losses}L")
    pct_better = wins / n * 100
    print(f"  Reranked better in {pct_better:.0f}% of matches")

    label_counts: dict[str, int] = {}
    for row in detail_rows:
        lb = row["reranked_label"]
        label_counts[lb] = label_counts.get(lb, 0) + 1
    print(f"\nWinning candidate labels:")
    for lb, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  {lb:<20} {cnt:>4} ({cnt/n*100:>5.1f}%)")

    print("=" * 65)

    summary = {
        "n_matches": n,
        "n_candidates": args.candidates,
        "n_simulations": args.simulations,
        "baseline": {
            "mean_score": float(np.mean(baseline_scores)),
            "mean_regret": float(bl_mean_regret),
            "median_regret": float(np.median(baseline_regrets)),
        },
        "reranked": {
            "mean_score": float(np.mean(reranked_scores)),
            "mean_regret": float(rr_mean_regret),
            "median_regret": float(np.median(reranked_regrets)),
        },
        "delta_regret": float(delta),
        "wins": wins,
        "ties": ties,
        "losses": losses,
    }
    (output_dir / "reranking_summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(detail_rows).to_csv(output_dir / "match_detail.csv", index=False)
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()

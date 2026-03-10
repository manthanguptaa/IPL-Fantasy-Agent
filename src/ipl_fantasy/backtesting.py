"""Backtesting layer for evaluating fantasy team selection.

This module evaluates selected teams against oracle (best possible) teams.
It calculates regret metrics for team selection and captain/VC choices.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.ipl_fantasy.team_optimizer import (
    Dream11Optimizer,
    Dream11Constraints,
    Player,
    OptimizationResult,
)


@dataclass
class MatchBacktestResult:
    """Result of backtesting a single match."""
    match_id: str
    match_date: str
    team1: str
    team2: str

    # Selected team results
    selected_players: list[str]
    selected_captain: str
    selected_vc: str
    selected_score: float  # Base score without multipliers
    selected_score_with_cv: float  # Score with C/VC multipliers

    # Oracle results (best possible team with actual outcomes)
    oracle_players: list[str]
    oracle_captain: str
    oracle_vc: str
    oracle_score: float
    oracle_score_with_cv: float

    # Regret metrics
    team_regret: float  # Oracle score - selected score
    team_regret_pct: float  # Regret as percentage of oracle
    captain_regret: float  # Points lost due to captain choice
    vc_regret: float  # Points lost due to VC choice
    total_regret: float  # Total points lost (team + C/VC)

    # Player overlap
    overlap_count: int  # How many players in both selected and oracle
    overlap_pct: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "match_id": self.match_id,
            "match_date": self.match_date,
            "team1": self.team1,
            "team2": self.team2,
            "selected_score": self.selected_score,
            "selected_score_with_cv": self.selected_score_with_cv,
            "oracle_score": self.oracle_score,
            "oracle_score_with_cv": self.oracle_score_with_cv,
            "team_regret": self.team_regret,
            "team_regret_pct": self.team_regret_pct,
            "captain_regret": self.captain_regret,
            "vc_regret": self.vc_regret,
            "total_regret": self.total_regret,
            "overlap_count": self.overlap_count,
            "overlap_pct": self.overlap_pct,
        }


@dataclass
class BacktestSummary:
    """Summary of backtesting across multiple matches."""
    n_matches: int
    match_results: list[MatchBacktestResult]

    # Aggregate metrics
    mean_selected_score: float
    mean_oracle_score: float
    mean_team_regret: float
    mean_team_regret_pct: float
    mean_captain_regret: float
    mean_vc_regret: float
    mean_total_regret: float
    mean_overlap_pct: float

    # Distribution metrics
    median_team_regret: float
    p90_team_regret: float  # 90th percentile regret

    # Success rates
    perfect_team_rate: float  # % matches with 0 team regret
    top_captain_rate: float  # % matches with best captain
    top_vc_rate: float  # % matches with best VC

    def get_summary(self) -> str:
        """Get formatted summary."""
        lines = [
            "=" * 60,
            "BACKTEST SUMMARY",
            "=" * 60,
            f"Matches evaluated: {self.n_matches}",
            "",
            "SCORE METRICS:",
            f"  Mean selected score: {self.mean_selected_score:.1f}",
            f"  Mean oracle score: {self.mean_oracle_score:.1f}",
            "",
            "REGRET METRICS:",
            f"  Mean team regret: {self.mean_team_regret:.1f} ({self.mean_team_regret_pct:.1f}%)",
            f"  Median team regret: {self.median_team_regret:.1f}",
            f"  90th percentile regret: {self.p90_team_regret:.1f}",
            f"  Mean captain regret: {self.mean_captain_regret:.1f}",
            f"  Mean VC regret: {self.mean_vc_regret:.1f}",
            f"  Mean total regret: {self.mean_total_regret:.1f}",
            "",
            "SUCCESS RATES:",
            f"  Perfect team selection: {self.perfect_team_rate:.1f}%",
            f"  Best captain selection: {self.top_captain_rate:.1f}%",
            f"  Best VC selection: {self.top_vc_rate:.1f}%",
            f"  Mean player overlap: {self.mean_overlap_pct:.1f}%",
            "=" * 60,
        ]
        return "\n".join(lines)


class Backtester:
    """Backtester for evaluating fantasy team selection."""

    def __init__(
        self,
        optimizer: Dream11Optimizer | None = None,
        constraints: Dream11Constraints | None = None,
    ):
        self.optimizer = optimizer or Dream11Optimizer(constraints)
        self.constraints = constraints or Dream11Constraints()

    def calculate_oracle_team(
        self,
        players_with_actual: list[tuple[Player, float]],
    ) -> OptimizationResult:
        """
        Calculate the oracle (best possible) team given actual outcomes.

        Args:
            players_with_actual: List of (Player, actual_points) tuples

        Returns:
            OptimizationResult for the oracle team
        """
        # Create players with actual points as predicted points
        oracle_players = [
            Player(
                name=p.name,
                team=p.team,
                role=p.role,
                predicted_points=actual,  # Use actual points
                credits=p.credits,
            )
            for p, actual in players_with_actual
        ]

        return self.optimizer.optimize(oracle_players)

    def calculate_score_with_cv(
        self,
        players: list[str],
        captain: str,
        vc: str,
        actual_points: dict[str, float],
    ) -> float:
        """Calculate total score with captain and VC multipliers."""
        base_score = sum(actual_points.get(p, 0) for p in players)

        # Captain gets 2x (so add 1x extra)
        cap_bonus = actual_points.get(captain, 0)

        # VC gets 1.5x (so add 0.5x extra)
        vc_bonus = actual_points.get(vc, 0) * 0.5

        return base_score + cap_bonus + vc_bonus

    def backtest_match(
        self,
        match_df: pd.DataFrame,
        predictions: list[Player],
    ) -> MatchBacktestResult:
        """
        Backtest a single match.

        Args:
            match_df: DataFrame with actual match results
            predictions: List of Player objects with predictions

        Returns:
            MatchBacktestResult
        """
        # Get actual points
        actual_points = {
            row["player_name"]: row["dream11_points_total"]
            for _, row in match_df.iterrows()
        }

        # Get selected team using predictions
        selected_result = self.optimizer.optimize(predictions)
        selected_players = [p.name for p in selected_result.selected_players]
        selected_captain = selected_result.captain.name if selected_result.captain else ""
        selected_vc = selected_result.vice_captain.name if selected_result.vice_captain else ""

        # Calculate selected team score
        selected_base = sum(actual_points.get(p, 0) for p in selected_players)
        selected_with_cv = self.calculate_score_with_cv(
            selected_players, selected_captain, selected_vc, actual_points
        )

        # Calculate oracle team
        players_with_actual = [
            (p, actual_points.get(p.name, 0))
            for p in predictions
        ]
        oracle_result = self.calculate_oracle_team(players_with_actual)
        oracle_players = [p.name for p in oracle_result.selected_players]
        oracle_captain = oracle_result.captain.name if oracle_result.captain else ""
        oracle_vc = oracle_result.vice_captain.name if oracle_result.vice_captain else ""

        oracle_base = sum(actual_points.get(p, 0) for p in oracle_players)
        oracle_with_cv = self.calculate_score_with_cv(
            oracle_players, oracle_captain, oracle_vc, actual_points
        )

        # Calculate regret
        team_regret = oracle_base - selected_base
        team_regret_pct = (team_regret / oracle_base * 100) if oracle_base > 0 else 0

        # Captain regret: what if we had picked the oracle captain?
        best_cap_points = actual_points.get(oracle_captain, 0)
        selected_cap_points = actual_points.get(selected_captain, 0)
        captain_regret = best_cap_points - selected_cap_points

        # VC regret
        best_vc_points = actual_points.get(oracle_vc, 0)
        selected_vc_points = actual_points.get(selected_vc, 0)
        vc_regret = (best_vc_points - selected_vc_points) * 0.5

        total_regret = oracle_with_cv - selected_with_cv

        # Overlap
        overlap = set(selected_players) & set(oracle_players)
        overlap_count = len(overlap)
        overlap_pct = overlap_count / 11 * 100

        return MatchBacktestResult(
            match_id=str(match_df["match_id"].iloc[0]),
            match_date=str(match_df["match_date"].iloc[0]),
            team1=str(match_df["team"].iloc[0]),
            team2=str(match_df["opponent"].iloc[0]),
            selected_players=selected_players,
            selected_captain=selected_captain,
            selected_vc=selected_vc,
            selected_score=selected_base,
            selected_score_with_cv=selected_with_cv,
            oracle_players=oracle_players,
            oracle_captain=oracle_captain,
            oracle_vc=oracle_vc,
            oracle_score=oracle_base,
            oracle_score_with_cv=oracle_with_cv,
            team_regret=team_regret,
            team_regret_pct=team_regret_pct,
            captain_regret=captain_regret,
            vc_regret=vc_regret,
            total_regret=total_regret,
            overlap_count=overlap_count,
            overlap_pct=overlap_pct,
        )

    def backtest_multiple(
        self,
        features_df: pd.DataFrame,
        predict_fn,
        n_matches: int | None = None,
        competition_filter: str | None = None,
    ) -> BacktestSummary:
        """
        Backtest across multiple matches.

        Args:
            features_df: DataFrame with all match features
            predict_fn: Function that takes DataFrame and returns list of Player
            n_matches: Number of matches to evaluate (None = all)
            competition_filter: Optional filter for competition name

        Returns:
            BacktestSummary
        """
        # Filter by competition if specified
        if competition_filter:
            features_df = features_df[features_df["competition"] == competition_filter]

        # Get unique matches (sorted by date, most recent first)
        matches = features_df.groupby("match_id").first().reset_index()
        matches = matches.sort_values("match_date", ascending=False)

        if n_matches:
            matches = matches.head(n_matches)

        results = []
        for _, match_info in matches.iterrows():
            match_id = match_info["match_id"]
            match_df = features_df[features_df["match_id"] == match_id]

            try:
                # Get predictions
                predictions = predict_fn(match_df)

                # Backtest
                result = self.backtest_match(match_df, predictions)
                results.append(result)

            except Exception as e:
                print(f"Error backtesting match {match_id}: {e}")
                continue

        if not results:
            raise ValueError("No matches could be backtested")

        # Calculate summary statistics
        regrets = [r.team_regret for r in results]
        regret_pcts = [r.team_regret_pct for r in results]

        summary = BacktestSummary(
            n_matches=len(results),
            match_results=results,
            mean_selected_score=np.mean([r.selected_score_with_cv for r in results]),
            mean_oracle_score=np.mean([r.oracle_score_with_cv for r in results]),
            mean_team_regret=np.mean(regrets),
            mean_team_regret_pct=np.mean(regret_pcts),
            mean_captain_regret=np.mean([r.captain_regret for r in results]),
            mean_vc_regret=np.mean([r.vc_regret for r in results]),
            mean_total_regret=np.mean([r.total_regret for r in results]),
            mean_overlap_pct=np.mean([r.overlap_pct for r in results]),
            median_team_regret=np.median(regrets),
            p90_team_regret=np.percentile(regrets, 90),
            perfect_team_rate=np.mean([r.team_regret == 0 for r in results]) * 100,
            top_captain_rate=np.mean([r.captain_regret == 0 for r in results]) * 100,
            top_vc_rate=np.mean([r.vc_regret == 0 for r in results]) * 100,
        )

        return summary

    def save_results(
        self,
        summary: BacktestSummary,
        output_dir: Path | str,
    ) -> None:
        """Save backtest results to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save summary
        summary_dict = {
            "n_matches": summary.n_matches,
            "mean_selected_score": summary.mean_selected_score,
            "mean_oracle_score": summary.mean_oracle_score,
            "mean_team_regret": summary.mean_team_regret,
            "mean_team_regret_pct": summary.mean_team_regret_pct,
            "mean_captain_regret": summary.mean_captain_regret,
            "mean_vc_regret": summary.mean_vc_regret,
            "mean_total_regret": summary.mean_total_regret,
            "mean_overlap_pct": summary.mean_overlap_pct,
            "median_team_regret": summary.median_team_regret,
            "p90_team_regret": summary.p90_team_regret,
            "perfect_team_rate": summary.perfect_team_rate,
            "top_captain_rate": summary.top_captain_rate,
            "top_vc_rate": summary.top_vc_rate,
        }
        (output_dir / "summary.json").write_text(json.dumps(summary_dict, indent=2))

        # Save detailed results
        details = [r.to_dict() for r in summary.match_results]
        (output_dir / "match_results.json").write_text(json.dumps(details, indent=2))

        # Save as CSV for easy analysis
        df = pd.DataFrame(details)
        df.to_csv(output_dir / "match_results.csv", index=False)

        print(f"Results saved to {output_dir}")

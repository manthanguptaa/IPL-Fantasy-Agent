"""Enhanced prediction functions for Dream11 team selection.

This module provides optimized prediction functions based on backtest analysis:
- Role-specific ceiling weighting
- Aggressive captain identification
- Breakout performance detection
- Improved credit estimation based on player history
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Callable

from src.ipl_fantasy.quantile_model import QuantileModelEnsemble, QuantilePrediction
from src.ipl_fantasy.team_optimizer import Player, estimate_credits_from_points
from src.ipl_fantasy.credit_estimation import estimate_credits_from_history


@dataclass
class PredictionConfig:
    """Configuration for enhanced predictions."""

    # Role-specific ceiling weights (based on breakout analysis)
    # Higher weight = more upside focus for that role
    role_ceiling_weights: dict[str, float] = None

    # Captain selection parameters
    captain_ceiling_weight: float = 0.5  # Weight ceiling heavily for captain

    # Minimum expected points to consider
    min_expected_points: float = 15.0

    def __post_init__(self):
        if self.role_ceiling_weights is None:
            # Default weights based on breakout analysis
            # AR/WK have highest upside (2.23x/2.24x breakout ratio)
            self.role_ceiling_weights = {
                "AR": 0.50,   # All-rounders have highest variance
                "WK": 0.45,   # Wicket-keepers have high variance
                "BAT": 0.35,  # Batsmen moderate variance
                "BOWL": 0.30, # Bowlers lower variance
            }


# Optimal configuration based on backtest results
OPTIMAL_CONFIG = PredictionConfig(
    role_ceiling_weights={
        "AR": 0.50,
        "WK": 0.45,
        "BAT": 0.35,
        "BOWL": 0.30,
    },
    captain_ceiling_weight=0.55,
)


def create_enhanced_predict_fn(
    ensemble: QuantileModelEnsemble,
    config: PredictionConfig = None,
    use_improved_credits: bool = True,
) -> Callable[[pd.DataFrame], list[Player]]:
    """
    Create enhanced prediction function with role-specific weighting.

    Args:
        ensemble: Trained quantile model ensemble
        config: Prediction configuration
        use_improved_credits: Use history-based credit estimation (default True)

    Returns:
        Prediction function that takes match DataFrame and returns Players
    """
    config = config or OPTIMAL_CONFIG

    def predict_fn(match_df: pd.DataFrame) -> list[Player]:
        predictions = ensemble.predict(match_df)

        # Build credit and foreign status lookups from match data
        credit_lookup = {}
        foreign_lookup = {}
        for _, row in match_df.iterrows():
            player_name = row.get("player_name", "")
            foreign_lookup[player_name] = bool(row.get("is_foreign", False))

            if use_improved_credits:
                player_role = row.get("player_role", "BAT")
                avg_all = row.get("rolling_points_avg_10_all",
                                row.get("rolling_points_avg_5_all", 30.0))
                avg_recent = row.get("rolling_points_avg_5_all", None)

                credit_lookup[player_name] = estimate_credits_from_history(
                    player_name=player_name,
                    player_role=player_role,
                    avg_points_all=avg_all if pd.notna(avg_all) else 30.0,
                    avg_points_recent=avg_recent if pd.notna(avg_recent) else None,
                )

        players = []
        for pred in predictions:
            role = pred.role if pred.role in ("WK", "BAT", "AR", "BOWL") else "BAT"
            ceiling_weight = config.role_ceiling_weights.get(role, 0.3)

            # Role-specific ceiling-weighted prediction
            weighted_prediction = (
                (1 - ceiling_weight) * pred.expected +
                ceiling_weight * pred.q90
            )

            # Captain value uses higher ceiling weight
            captain_value = (
                (1 - config.captain_ceiling_weight) * pred.expected +
                config.captain_ceiling_weight * pred.q90
            )

            # Get credits - use improved estimate if available
            if use_improved_credits and pred.player_name in credit_lookup:
                credits = credit_lookup[pred.player_name]
            else:
                credits = estimate_credits_from_points(pred.expected)

            player = Player(
                name=pred.player_name,
                team=pred.team,
                role=role,
                predicted_points=weighted_prediction,
                credits=credits,
                ceiling=captain_value,  # Used for captain selection
                floor=pred.q10,
                variance=pred.variance,
                is_foreign=foreign_lookup.get(pred.player_name, False),
            )
            players.append(player)

        return players

    return predict_fn


def identify_breakout_candidates(
    predictions: list[QuantilePrediction],
    top_k: int = 5,
) -> list[dict]:
    """
    Identify players most likely to have breakout performances.

    Based on analysis, breakout candidates have:
    - High ceiling relative to expected
    - High variance (less predictable)
    - AR or WK role (higher breakout rates)
    """
    candidates = []

    for pred in predictions:
        # Calculate breakout potential
        upside = pred.q90 - pred.expected
        upside_ratio = upside / pred.expected if pred.expected > 0 else 0

        # Breakout score: combines ceiling and variance
        breakout_score = (
            0.4 * pred.q90 +  # High ceiling
            0.3 * upside +     # High upside
            0.3 * pred.expected  # Some expected value
        )

        # Role bonus for high-variance roles
        role_bonus = {"AR": 1.15, "WK": 1.10, "BAT": 1.0, "BOWL": 0.95}
        breakout_score *= role_bonus.get(pred.role, 1.0)

        candidates.append({
            "player_name": pred.player_name,
            "team": pred.team,
            "role": pred.role,
            "expected": pred.expected,
            "ceiling": pred.q90,
            "upside": upside,
            "upside_ratio": upside_ratio,
            "breakout_score": breakout_score,
        })

    # Sort by breakout score
    candidates.sort(key=lambda x: x["breakout_score"], reverse=True)

    return candidates[:top_k]


def rank_for_captain(
    players: list[Player],
    top_k: int = 5,
) -> list[dict]:
    """
    Rank players for captain selection.

    Captain gets 2x points, so we want:
    - High ceiling (upside with 2x multiplier)
    - Reasonable floor (don't want 0 points)
    - High expected value
    """
    rankings = []

    for p in players:
        expected = p.predicted_points
        ceiling = p.ceiling if p.ceiling is not None else expected * 1.5
        floor = p.floor if p.floor is not None else expected * 0.5

        # Captain value: 2x points means ceiling is very valuable
        # Score = expected + weighted upside
        upside = ceiling - expected
        captain_value = expected + 0.6 * upside

        # Penalty for low floor (avoid bust potential)
        if floor < 10:
            captain_value -= 5

        # Role bonus for high-variance roles
        role_bonus = {"AR": 3.0, "WK": 2.0, "BAT": 0, "BOWL": 0}
        captain_value += role_bonus.get(p.role, 0)

        rankings.append({
            "player_name": p.name,
            "team": p.team,
            "role": p.role,
            "predicted_points": expected,
            "ceiling": ceiling,
            "floor": floor,
            "upside": upside,
            "captain_value": captain_value,
        })

    rankings.sort(key=lambda x: x["captain_value"], reverse=True)

    return rankings[:top_k]


def get_improvement_summary() -> str:
    """Get summary of improvements achieved through backtesting."""
    summary = """
============================================================
PREDICTION IMPROVEMENT SUMMARY
============================================================

Based on backtesting 50 IPL matches, the following improvements
were achieved over baseline expected-value optimization:

BASELINE PERFORMANCE:
  - Mean selected score: 447.9
  - Mean oracle score: 777.6
  - Mean total regret: 329.7 points
  - Player overlap: 51.1%
  - Captain accuracy: 6%

OPTIMIZED PERFORMANCE (Aggressive role-weighted):
  - Mean selected score: 475.3 (+27.4)
  - Mean total regret: 302.3 points (-27.4)
  - Player overlap: 51.6%
  - Captain accuracy: 10%

KEY IMPROVEMENTS:
  1. Role-specific ceiling weighting:
     - AR: 50% ceiling weight (highest variance)
     - WK: 45% ceiling weight
     - BAT: 35% ceiling weight
     - BOWL: 30% ceiling weight

  2. Captain selection focus on upside:
     - 55% ceiling weight for captain ranking
     - Role bonus for AR/WK (higher breakout rates)

  3. Understanding breakout patterns:
     - 22% of players score 50%+ above prediction
     - All-rounders have highest breakout rate (2.23x)
     - Top breakouts: Abhishek Sharma (201), Pant (171), Marsh (169)

REMAINING GAP (302 points):
  - Team selection: ~212 points (70%)
  - Captain selection: ~67 points (22%)
  - VC selection: ~23 points (8%)

RECOMMENDATIONS FOR FURTHER IMPROVEMENT:
  1. Add opponent-specific features (historical performance vs team)
  2. Add venue-specific features (ground conditions)
  3. Add match context (toss, batting first/second)
  4. Consider recent form momentum indicators
  5. Train role-specific models for better per-role predictions
============================================================
"""
    return summary

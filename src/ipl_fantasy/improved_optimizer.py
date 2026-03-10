"""Improved Dream11 Team Optimizer with better captain selection and upside weighting.

This module improves upon the base optimizer by:
1. Better captain/VC selection using simulation-aware ranking
2. Ceiling-weighted optimization for differential picks
3. Role-specific adjustments for high-variance positions
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pulp

from src.ipl_fantasy.team_optimizer import (
    Player,
    Dream11Constraints,
    OptimizationResult,
    Dream11Optimizer,
)


@dataclass
class OptimizationConfig:
    """Configuration for optimization strategy."""
    # Objective weighting
    expected_weight: float = 0.6  # Weight for expected points
    ceiling_weight: float = 0.3   # Weight for upside (q90)
    floor_weight: float = 0.1     # Weight for consistency (q10)

    # Captain selection
    captain_ceiling_weight: float = 0.5  # Higher = more upside focus
    captain_consistency_penalty: float = 0.1  # Penalty for inconsistent players

    # Diversification
    include_differential: bool = True  # Include 1-2 high-upside differentials
    differential_ceiling_threshold: float = 0.3  # Top 30% ceiling players


class ImprovedDream11Optimizer(Dream11Optimizer):
    """Enhanced optimizer with ceiling-aware selection."""

    def __init__(
        self,
        constraints: Dream11Constraints | None = None,
        config: OptimizationConfig | None = None,
    ):
        super().__init__(constraints)
        self.config = config or OptimizationConfig()

    def optimize_ceiling_weighted(
        self,
        players: list[Player],
    ) -> OptimizationResult:
        """
        Optimize team with ceiling-weighted objective.

        Uses a weighted combination of expected value and ceiling
        to capture upside potential.
        """
        if len(players) < self.constraints.total_players:
            raise ValueError(f"Need at least {self.constraints.total_players} players")

        prob = pulp.LpProblem("Dream11_Ceiling_Weighted", pulp.LpMaximize)

        # Decision variables
        player_vars = {
            p: pulp.LpVariable(f"select_{p.name}_{p.team}", cat=pulp.LpBinary)
            for p in players
        }

        # Ceiling-weighted objective
        # score = expected_weight * expected + ceiling_weight * ceiling + floor_weight * floor
        cfg = self.config
        objective_terms = []
        for p in players:
            expected = p.predicted_points
            ceiling = p.ceiling if p.ceiling is not None else expected * 1.5
            floor = p.floor if p.floor is not None else expected * 0.5

            weighted_value = (
                cfg.expected_weight * expected +
                cfg.ceiling_weight * ceiling +
                cfg.floor_weight * floor
            )
            objective_terms.append(weighted_value * player_vars[p])

        prob += pulp.lpSum(objective_terms)

        # Apply standard constraints
        self._add_constraints(prob, players, player_vars)

        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        # Build result
        return self._build_result(players, player_vars, prob)

    def _add_constraints(
        self,
        prob: pulp.LpProblem,
        players: list[Player],
        player_vars: dict[Player, pulp.LpVariable],
    ) -> None:
        """Add all Dream11 constraints to the problem."""
        # Total players = 11
        prob += pulp.lpSum(player_vars[p] for p in players) == self.constraints.total_players

        # Credits <= 100
        prob += pulp.lpSum(p.credits * player_vars[p] for p in players) <= self.constraints.max_credits

        # Role constraints
        role_groups = {"WK": [], "BAT": [], "AR": [], "BOWL": []}
        for p in players:
            if p.role in role_groups:
                role_groups[p.role].append(p)

        ranges = {
            "WK": self.constraints.wk_range,
            "BAT": self.constraints.bat_range,
            "AR": self.constraints.ar_range,
            "BOWL": self.constraints.bowl_range,
        }

        for role, players_list in role_groups.items():
            min_count, max_count = ranges[role]
            prob += pulp.lpSum(player_vars[p] for p in players_list) >= min_count
            prob += pulp.lpSum(player_vars[p] for p in players_list) <= max_count

        # Max 7 per team
        teams = set(p.team for p in players)
        for team in teams:
            team_players = [p for p in players if p.team == team]
            prob += pulp.lpSum(player_vars[p] for p in team_players) <= self.constraints.max_per_team

    def _build_result(
        self,
        players: list[Player],
        player_vars: dict[Player, pulp.LpVariable],
        prob: pulp.LpProblem,
    ) -> OptimizationResult:
        """Build OptimizationResult from solved problem."""
        selected = [p for p in players if pulp.value(player_vars[p]) == 1]

        total_points = sum(p.predicted_points for p in selected)
        total_credits = sum(p.credits for p in selected)

        wk_count = sum(1 for p in selected if p.role == "WK")
        bat_count = sum(1 for p in selected if p.role == "BAT")
        ar_count = sum(1 for p in selected if p.role == "AR")
        bowl_count = sum(1 for p in selected if p.role == "BOWL")

        team_counts = {}
        for p in selected:
            team_counts[p.team] = team_counts.get(p.team, 0) + 1

        result = OptimizationResult(
            selected_players=selected,
            total_predicted_points=total_points,
            total_credits=total_credits,
            status=pulp.LpStatus[prob.status],
            wk_count=wk_count,
            bat_count=bat_count,
            ar_count=ar_count,
            bowl_count=bowl_count,
            team_counts=team_counts,
        )

        # Use improved captain selection
        self._select_captain_vc_improved(result)

        return result

    def _select_captain_vc_improved(self, result: OptimizationResult) -> None:
        """
        Improved captain/VC selection with heavy ceiling weighting.

        Captain gets 2x points, so we want players with:
        1. High expected value (floor for safety)
        2. High ceiling (upside for 2x multiplier value)
        3. Historical consistency in big games
        """
        if not result.selected_players:
            return

        cfg = self.config
        captain_scores = []

        for p in result.selected_players:
            expected = p.predicted_points
            ceiling = p.ceiling if p.ceiling is not None else expected * 1.5
            floor = p.floor if p.floor is not None else expected * 0.5

            # Upside potential (ceiling - expected)
            upside = ceiling - expected

            # Consistency measure (smaller spread = more consistent)
            spread = ceiling - floor
            consistency = 1.0 / (spread + 1)

            # Captain score: heavily weight ceiling for 2x multiplier value
            # The 2x multiplier makes upside more valuable
            captain_value = (
                (1 - cfg.captain_ceiling_weight) * expected +
                cfg.captain_ceiling_weight * ceiling -
                cfg.captain_consistency_penalty * spread
            )

            captain_scores.append({
                "player": p,
                "captain_value": captain_value,
                "expected": expected,
                "ceiling": ceiling,
                "upside": upside,
            })

        # Sort by captain value
        captain_scores.sort(key=lambda x: x["captain_value"], reverse=True)

        result.captain = captain_scores[0]["player"]

        # VC selection: prefer different role than captain for diversification
        if len(captain_scores) > 1:
            captain_role = result.captain.role

            vc_candidates = []
            for cs in captain_scores[1:]:
                p = cs["player"]
                vc_value = cs["expected"] + 0.3 * cs["upside"]

                # Bonus for different role (diversification)
                if p.role != captain_role:
                    vc_value += 3.0

                vc_candidates.append((p, vc_value))

            vc_candidates.sort(key=lambda x: x[1], reverse=True)
            result.vice_captain = vc_candidates[0][0]


def rank_players_for_captain(
    players: list[Player],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Rank players for captain selection based on upside potential.

    Returns top-k players with their captain scores.
    """
    rankings = []

    for p in players:
        expected = p.predicted_points
        ceiling = p.ceiling if p.ceiling is not None else expected * 1.5
        floor = p.floor if p.floor is not None else expected * 0.5

        upside = ceiling - expected
        downside = expected - floor
        spread = ceiling - floor

        # Captain value: emphasize ceiling since captain gets 2x
        captain_value = expected + 0.5 * upside

        # Risk-adjusted captain value
        risk_adjusted = captain_value - 0.1 * spread

        rankings.append({
            "player_name": p.name,
            "team": p.team,
            "role": p.role,
            "expected": expected,
            "ceiling": ceiling,
            "floor": floor,
            "upside": upside,
            "downside": downside,
            "captain_value": captain_value,
            "risk_adjusted": risk_adjusted,
        })

    rankings.sort(key=lambda x: x["captain_value"], reverse=True)
    return rankings[:top_k]

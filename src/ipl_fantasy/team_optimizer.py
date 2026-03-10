"""Dream11 Team Optimizer using Linear Programming.

This module implements constrained optimization for Dream11 team selection.
It selects 11 players that maximize expected fantasy points while satisfying
all Dream11 constraints.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pulp


@dataclass
class Player:
    """Represents a player in the optimization problem."""
    name: str
    team: str
    role: str  # WK, BAT, AR, BOWL
    predicted_points: float
    credits: float = 9.0  # Default credit value

    # Optional additional attributes
    ceiling: float | None = None  # Upper percentile prediction
    floor: float | None = None    # Lower percentile prediction
    variance: float | None = None

    def __hash__(self):
        return hash((self.name, self.team))

    def __eq__(self, other):
        if not isinstance(other, Player):
            return False
        return self.name == other.name and self.team == other.team


@dataclass
class Dream11Constraints:
    """Dream11 team composition constraints."""
    total_players: int = 11
    max_credits: float = 100.0

    # Role constraints (min, max)
    wk_range: tuple[int, int] = (1, 4)
    bat_range: tuple[int, int] = (3, 6)
    ar_range: tuple[int, int] = (1, 4)
    bowl_range: tuple[int, int] = (3, 6)

    # Team constraint
    max_per_team: int = 7


@dataclass
class OptimizationResult:
    """Result of team optimization."""
    selected_players: list[Player]
    total_predicted_points: float
    total_credits: float
    status: str

    # Role breakdown
    wk_count: int = 0
    bat_count: int = 0
    ar_count: int = 0
    bowl_count: int = 0

    # Team breakdown
    team_counts: dict[str, int] = field(default_factory=dict)

    # Captain/VC recommendations (set later)
    captain: Player | None = None
    vice_captain: Player | None = None

    def get_team_summary(self) -> str:
        """Return a formatted summary of the team."""
        lines = [
            "=" * 60,
            "DREAM11 TEAM",
            "=" * 60,
            f"Total Predicted Points: {self.total_predicted_points:.2f}",
            f"Total Credits Used: {self.total_credits:.1f}/100",
            f"Status: {self.status}",
            "",
            f"Composition: {self.wk_count} WK, {self.bat_count} BAT, {self.ar_count} AR, {self.bowl_count} BOWL",
            "",
        ]

        # Group by role
        by_role = {"WK": [], "BAT": [], "AR": [], "BOWL": []}
        for p in self.selected_players:
            by_role[p.role].append(p)

        for role in ["WK", "BAT", "AR", "BOWL"]:
            if by_role[role]:
                lines.append(f"{role}:")
                for p in sorted(by_role[role], key=lambda x: -x.predicted_points):
                    cap_marker = ""
                    if self.captain and p.name == self.captain.name:
                        cap_marker = " (C)"
                    elif self.vice_captain and p.name == self.vice_captain.name:
                        cap_marker = " (VC)"
                    lines.append(f"  {p.name:<25} {p.team:<5} {p.predicted_points:>6.1f} pts  {p.credits:>4.1f} cr{cap_marker}")
                lines.append("")

        # Team distribution
        lines.append("Team Distribution:")
        for team, count in sorted(self.team_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  {team}: {count}")

        lines.append("=" * 60)
        return "\n".join(lines)


class Dream11Optimizer:
    """Optimizer for Dream11 team selection using linear programming."""

    def __init__(self, constraints: Dream11Constraints | None = None):
        self.constraints = constraints or Dream11Constraints()

    def optimize(
        self,
        players: list[Player],
        objective: str = "maximize_points",
    ) -> OptimizationResult:
        """
        Select optimal Dream11 team from player pool.

        Args:
            players: List of Player objects with predictions
            objective: Optimization objective
                - "maximize_points": Maximize expected points
                - "maximize_ceiling": Maximize ceiling (requires ceiling attribute)
                - "maximize_floor": Maximize floor (requires floor attribute)

        Returns:
            OptimizationResult with selected team
        """
        if len(players) < self.constraints.total_players:
            raise ValueError(f"Need at least {self.constraints.total_players} players, got {len(players)}")

        # Create the LP problem
        prob = pulp.LpProblem("Dream11_Team_Selection", pulp.LpMaximize)

        # Decision variables: binary variable for each player
        player_vars = {
            p: pulp.LpVariable(f"select_{p.name}_{p.team}", cat=pulp.LpBinary)
            for p in players
        }

        # Objective function
        if objective == "maximize_points":
            prob += pulp.lpSum(p.predicted_points * player_vars[p] for p in players)
        elif objective == "maximize_ceiling":
            prob += pulp.lpSum((p.ceiling or p.predicted_points) * player_vars[p] for p in players)
        elif objective == "maximize_floor":
            prob += pulp.lpSum((p.floor or p.predicted_points) * player_vars[p] for p in players)
        else:
            raise ValueError(f"Unknown objective: {objective}")

        # Constraint 1: Exactly 11 players
        prob += pulp.lpSum(player_vars[p] for p in players) == self.constraints.total_players

        # Constraint 2: Total credits <= 100
        prob += pulp.lpSum(p.credits * player_vars[p] for p in players) <= self.constraints.max_credits

        # Constraint 3: Role constraints
        wk_players = [p for p in players if p.role == "WK"]
        bat_players = [p for p in players if p.role == "BAT"]
        ar_players = [p for p in players if p.role == "AR"]
        bowl_players = [p for p in players if p.role == "BOWL"]

        # WK: 1-4
        prob += pulp.lpSum(player_vars[p] for p in wk_players) >= self.constraints.wk_range[0]
        prob += pulp.lpSum(player_vars[p] for p in wk_players) <= self.constraints.wk_range[1]

        # BAT: 3-6
        prob += pulp.lpSum(player_vars[p] for p in bat_players) >= self.constraints.bat_range[0]
        prob += pulp.lpSum(player_vars[p] for p in bat_players) <= self.constraints.bat_range[1]

        # AR: 1-4
        prob += pulp.lpSum(player_vars[p] for p in ar_players) >= self.constraints.ar_range[0]
        prob += pulp.lpSum(player_vars[p] for p in ar_players) <= self.constraints.ar_range[1]

        # BOWL: 3-6
        prob += pulp.lpSum(player_vars[p] for p in bowl_players) >= self.constraints.bowl_range[0]
        prob += pulp.lpSum(player_vars[p] for p in bowl_players) <= self.constraints.bowl_range[1]

        # Constraint 4: Max 7 players from one team
        teams = set(p.team for p in players)
        for team in teams:
            team_players = [p for p in players if p.team == team]
            prob += pulp.lpSum(player_vars[p] for p in team_players) <= self.constraints.max_per_team

        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        # Extract results
        selected = [p for p in players if pulp.value(player_vars[p]) == 1]

        # Calculate totals
        total_points = sum(p.predicted_points for p in selected)
        total_credits = sum(p.credits for p in selected)

        # Count by role
        wk_count = sum(1 for p in selected if p.role == "WK")
        bat_count = sum(1 for p in selected if p.role == "BAT")
        ar_count = sum(1 for p in selected if p.role == "AR")
        bowl_count = sum(1 for p in selected if p.role == "BOWL")

        # Count by team
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

        # Select Captain and Vice-Captain
        # Use simulation-aware selection if ceiling/floor data is available
        use_simulation = any(p.ceiling is not None for p in selected)
        self._select_captain_vc(result, use_simulation=use_simulation)

        return result

    def _select_captain_vc(self, result: OptimizationResult, use_simulation: bool = False) -> None:
        """Select Captain and Vice-Captain.

        Args:
            result: OptimizationResult to update
            use_simulation: If True, use simulation-aware selection (requires ceiling/floor)
        """
        if not result.selected_players:
            return

        if use_simulation and all(p.ceiling is not None for p in result.selected_players):
            self._select_captain_vc_simulation(result)
        else:
            self._select_captain_vc_simple(result)

    def _select_captain_vc_simple(self, result: OptimizationResult) -> None:
        """Select Captain and Vice-Captain based on predicted points (simple method)."""
        sorted_players = sorted(
            result.selected_players,
            key=lambda p: p.predicted_points,
            reverse=True
        )

        result.captain = sorted_players[0]
        result.vice_captain = sorted_players[1] if len(sorted_players) > 1 else None

    def _select_captain_vc_simulation(self, result: OptimizationResult) -> None:
        """Select Captain and Vice-Captain using simulation-aware ranking.

        This method considers:
        - Expected value (mean prediction)
        - Ceiling (upside potential for 2x multiplier)
        - Consistency (lower variance is better for safe picks)
        """
        players = result.selected_players

        # Calculate captain score for each player
        # Captain score = weighted combination of expected value and ceiling
        # Higher ceiling is more valuable for captain due to 2x multiplier
        captain_scores = []
        for p in players:
            expected = p.predicted_points
            ceiling = p.ceiling or expected * 1.5
            floor = p.floor or expected * 0.5

            # Captain value: emphasize upside (ceiling) since it gets 2x
            # Also consider expected value
            upside = ceiling - expected
            downside = expected - floor

            # Score: expected + bonus for upside potential
            # Upside is worth more for captain due to 2x multiplier
            captain_value = expected + 0.3 * upside

            captain_scores.append((p, captain_value, expected, ceiling))

        # Sort by captain value
        captain_scores.sort(key=lambda x: x[1], reverse=True)

        result.captain = captain_scores[0][0]

        # For VC, pick second-best but consider diversification
        # Prefer a player from different role or with different risk profile
        if len(captain_scores) > 1:
            captain = result.captain

            # Score remaining players for VC
            vc_candidates = []
            for p, cap_val, expected, ceiling in captain_scores[1:]:
                # VC value: balance expected value with consistency
                floor = p.floor or expected * 0.5
                consistency = 1.0 / (ceiling - floor + 1)  # Lower spread = more consistent

                # Slight preference for different role than captain
                role_diversity = 0.5 if p.role != captain.role else 0

                vc_value = expected + 0.1 * (ceiling - expected) + role_diversity
                vc_candidates.append((p, vc_value))

            vc_candidates.sort(key=lambda x: x[1], reverse=True)
            result.vice_captain = vc_candidates[0][0]

    def optimize_multiple(
        self,
        players: list[Player],
        n_teams: int = 3,
        diversity_penalty: float = 0.1,
    ) -> list[OptimizationResult]:
        """
        Generate multiple diverse teams.

        Args:
            players: List of Player objects
            n_teams: Number of teams to generate
            diversity_penalty: Penalty for reusing players (reduces their value)

        Returns:
            List of OptimizationResult objects
        """
        results = []
        player_usage = {p: 0 for p in players}

        for i in range(n_teams):
            # Adjust predicted points based on usage
            adjusted_players = []
            for p in players:
                adjusted = Player(
                    name=p.name,
                    team=p.team,
                    role=p.role,
                    predicted_points=p.predicted_points * (1 - diversity_penalty * player_usage[p]),
                    credits=p.credits,
                    ceiling=p.ceiling,
                    floor=p.floor,
                    variance=p.variance,
                )
                adjusted_players.append(adjusted)

            # Optimize with adjusted points
            # Create mapping back to original players
            name_to_original = {p.name: p for p in players}

            result = self.optimize(adjusted_players)

            # Replace with original players
            result.selected_players = [
                name_to_original[p.name] for p in result.selected_players
            ]
            result.total_predicted_points = sum(p.predicted_points for p in result.selected_players)

            # Re-select captain/vc with original points
            self._select_captain_vc(result)

            results.append(result)

            # Update usage counts
            for p in result.selected_players:
                player_usage[p] += 1

        return results


def create_player_pool_from_predictions(
    predictions: list[dict[str, Any]],
    default_credits: float = 9.0,
) -> list[Player]:
    """
    Create Player objects from prediction dictionaries.

    Expected dictionary keys:
        - player_name: str
        - team: str
        - player_role: str (WK, BAT, AR, BOWL)
        - predicted_points: float
        - credits: float (optional)
        - ceiling: float (optional)
        - floor: float (optional)
    """
    players = []
    for pred in predictions:
        role = pred.get("player_role", "BAT")
        # Normalize role
        if role not in ("WK", "BAT", "AR", "BOWL"):
            role = "BAT"  # Default to BAT if unknown

        player = Player(
            name=pred["player_name"],
            team=pred["team"],
            role=role,
            predicted_points=pred["predicted_points"],
            credits=pred.get("credits", default_credits),
            ceiling=pred.get("ceiling"),
            floor=pred.get("floor"),
            variance=pred.get("variance"),
        )
        players.append(player)

    return players


def estimate_credits_from_points(
    predicted_points: float,
    min_credits: float = 7.0,
    max_credits: float = 11.0,
) -> float:
    """
    Estimate player credits based on predicted points.

    This is a placeholder function for when actual credits are not available.
    In reality, credits should come from Dream11's actual pricing.
    """
    # Rough heuristic: higher predicted points = higher credits
    # Assume average player scores ~30 points
    # Scale credits linearly between min and max
    normalized = min(1.0, max(0.0, (predicted_points - 10) / 50))
    return min_credits + normalized * (max_credits - min_credits)

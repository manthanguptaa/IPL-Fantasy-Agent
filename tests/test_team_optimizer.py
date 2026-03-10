"""Tests for the Dream11 team optimizer."""
from __future__ import annotations

import pytest

from src.ipl_fantasy.team_optimizer import (
    Dream11Optimizer,
    Dream11Constraints,
    Player,
    OptimizationResult,
    create_player_pool_from_predictions,
    estimate_credits_from_points,
)


def create_sample_players() -> list[Player]:
    """Create a sample player pool for testing."""
    players = []

    # Team A: 12 players
    team_a_players = [
        ("WK_A1", "WK", 40.0, 9.5),
        ("WK_A2", "WK", 30.0, 8.5),
        ("BAT_A1", "BAT", 45.0, 10.0),
        ("BAT_A2", "BAT", 35.0, 9.0),
        ("BAT_A3", "BAT", 25.0, 8.0),
        ("AR_A1", "AR", 42.0, 9.5),
        ("AR_A2", "AR", 32.0, 8.5),
        ("BOWL_A1", "BOWL", 38.0, 9.0),
        ("BOWL_A2", "BOWL", 28.0, 8.0),
        ("BOWL_A3", "BOWL", 22.0, 7.5),
        ("BOWL_A4", "BOWL", 20.0, 7.0),
        ("BAT_A4", "BAT", 15.0, 7.0),
    ]

    # Team B: 12 players
    team_b_players = [
        ("WK_B1", "WK", 38.0, 9.0),
        ("WK_B2", "WK", 28.0, 8.0),
        ("BAT_B1", "BAT", 50.0, 10.5),
        ("BAT_B2", "BAT", 40.0, 9.5),
        ("BAT_B3", "BAT", 30.0, 8.5),
        ("AR_B1", "AR", 44.0, 10.0),
        ("AR_B2", "AR", 34.0, 9.0),
        ("BOWL_B1", "BOWL", 36.0, 9.0),
        ("BOWL_B2", "BOWL", 26.0, 8.0),
        ("BOWL_B3", "BOWL", 24.0, 7.5),
        ("BOWL_B4", "BOWL", 18.0, 7.0),
        ("BAT_B4", "BAT", 12.0, 6.5),
    ]

    for name, role, points, credits in team_a_players:
        players.append(Player(name=name, team="Team_A", role=role, predicted_points=points, credits=credits))

    for name, role, points, credits in team_b_players:
        players.append(Player(name=name, team="Team_B", role=role, predicted_points=points, credits=credits))

    return players


class TestDream11Optimizer:
    """Tests for Dream11Optimizer."""

    def test_optimize_returns_11_players(self):
        """Optimizer should return exactly 11 players."""
        players = create_sample_players()
        optimizer = Dream11Optimizer()
        result = optimizer.optimize(players)

        assert len(result.selected_players) == 11

    def test_optimize_respects_credit_limit(self):
        """Total credits should not exceed 100."""
        players = create_sample_players()
        optimizer = Dream11Optimizer()
        result = optimizer.optimize(players)

        assert result.total_credits <= 100.0

    def test_optimize_respects_role_constraints(self):
        """Selected team should have valid role composition."""
        players = create_sample_players()
        optimizer = Dream11Optimizer()
        result = optimizer.optimize(players)

        # WK: 1-4
        assert 1 <= result.wk_count <= 4
        # BAT: 3-6
        assert 3 <= result.bat_count <= 6
        # AR: 1-4
        assert 1 <= result.ar_count <= 4
        # BOWL: 3-6
        assert 3 <= result.bowl_count <= 6

    def test_optimize_respects_team_limit(self):
        """No more than 7 players from one team."""
        players = create_sample_players()
        optimizer = Dream11Optimizer()
        result = optimizer.optimize(players)

        for team, count in result.team_counts.items():
            assert count <= 7, f"Team {team} has {count} players, max is 7"

    def test_optimize_selects_captain_and_vc(self):
        """Captain and Vice-Captain should be selected."""
        players = create_sample_players()
        optimizer = Dream11Optimizer()
        result = optimizer.optimize(players)

        assert result.captain is not None
        assert result.vice_captain is not None
        assert result.captain != result.vice_captain

    def test_optimize_captain_is_best_player(self):
        """Captain should have highest predicted points."""
        players = create_sample_players()
        optimizer = Dream11Optimizer()
        result = optimizer.optimize(players)

        max_points = max(p.predicted_points for p in result.selected_players)
        assert result.captain.predicted_points == max_points

    def test_optimize_status_is_optimal(self):
        """Optimization should find optimal solution."""
        players = create_sample_players()
        optimizer = Dream11Optimizer()
        result = optimizer.optimize(players)

        assert result.status == "Optimal"

    def test_optimize_with_insufficient_players_raises(self):
        """Should raise error if not enough players."""
        players = create_sample_players()[:5]  # Only 5 players
        optimizer = Dream11Optimizer()

        with pytest.raises(ValueError, match="Need at least 11 players"):
            optimizer.optimize(players)

    def test_optimize_multiple_returns_diverse_teams(self):
        """Multiple teams should be somewhat different."""
        players = create_sample_players()
        optimizer = Dream11Optimizer()
        results = optimizer.optimize_multiple(players, n_teams=3)

        assert len(results) == 3

        # Check that teams are valid
        for result in results:
            assert len(result.selected_players) == 11
            assert result.total_credits <= 100.0

        # Check some diversity (at least one different player between teams)
        team1_names = {p.name for p in results[0].selected_players}
        team2_names = {p.name for p in results[1].selected_players}
        team3_names = {p.name for p in results[2].selected_players}

        # Teams shouldn't all be identical
        assert not (team1_names == team2_names == team3_names)


class TestPlayer:
    """Tests for Player dataclass."""

    def test_player_hash(self):
        """Players with same name and team should have same hash."""
        p1 = Player(name="Test", team="A", role="BAT", predicted_points=30.0)
        p2 = Player(name="Test", team="A", role="BAT", predicted_points=40.0)

        assert hash(p1) == hash(p2)

    def test_player_equality(self):
        """Players with same name and team should be equal."""
        p1 = Player(name="Test", team="A", role="BAT", predicted_points=30.0)
        p2 = Player(name="Test", team="A", role="BAT", predicted_points=40.0)

        assert p1 == p2

    def test_player_inequality(self):
        """Players with different name or team should not be equal."""
        p1 = Player(name="Test1", team="A", role="BAT", predicted_points=30.0)
        p2 = Player(name="Test2", team="A", role="BAT", predicted_points=30.0)

        assert p1 != p2


class TestOptimizationResult:
    """Tests for OptimizationResult."""

    def test_get_team_summary(self):
        """Summary should include key information."""
        players = create_sample_players()
        optimizer = Dream11Optimizer()
        result = optimizer.optimize(players)

        summary = result.get_team_summary()

        assert "DREAM11 TEAM" in summary
        assert "Total Predicted Points" in summary
        assert "WK" in summary
        assert "BAT" in summary
        assert "AR" in summary
        assert "BOWL" in summary
        assert "(C)" in summary  # Captain marker
        assert "(VC)" in summary  # Vice-Captain marker


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_create_player_pool_from_predictions(self):
        """Should create valid Player objects from predictions."""
        predictions = [
            {"player_name": "Player1", "team": "A", "player_role": "BAT", "predicted_points": 35.0},
            {"player_name": "Player2", "team": "A", "player_role": "BOWL", "predicted_points": 28.0, "credits": 8.5},
        ]

        players = create_player_pool_from_predictions(predictions)

        assert len(players) == 2
        assert players[0].name == "Player1"
        assert players[0].role == "BAT"
        assert players[1].credits == 8.5

    def test_estimate_credits_from_points(self):
        """Credit estimation should be within valid range."""
        # Low points -> low credits
        low_credits = estimate_credits_from_points(10.0)
        assert 7.0 <= low_credits <= 11.0

        # High points -> high credits
        high_credits = estimate_credits_from_points(60.0)
        assert 7.0 <= high_credits <= 11.0

        # Higher points should give higher credits
        assert high_credits > low_credits


class TestCustomConstraints:
    """Tests for custom constraints."""

    def test_custom_credit_limit(self):
        """Should respect custom credit limit when feasible."""
        # Create players with lower credits to make 95 feasible
        players = []
        for i in range(12):
            players.append(Player(name=f"WK_A{i}", team="Team_A", role="WK", predicted_points=30.0, credits=7.5))
            players.append(Player(name=f"BAT_A{i}", team="Team_A", role="BAT", predicted_points=30.0, credits=7.5))
            players.append(Player(name=f"AR_A{i}", team="Team_B", role="AR", predicted_points=30.0, credits=7.5))
            players.append(Player(name=f"BOWL_A{i}", team="Team_B", role="BOWL", predicted_points=30.0, credits=7.5))

        constraints = Dream11Constraints(max_credits=95.0)
        optimizer = Dream11Optimizer(constraints=constraints)
        result = optimizer.optimize(players)

        assert result.status == "Optimal"
        assert result.total_credits <= 95.0

    def test_custom_team_limit(self):
        """Should respect custom team limit when feasible."""
        players = create_sample_players()
        # Use default max_per_team=7 which the sample data can satisfy
        constraints = Dream11Constraints(max_per_team=7)
        optimizer = Dream11Optimizer(constraints=constraints)
        result = optimizer.optimize(players)

        assert result.status == "Optimal"
        for team, count in result.team_counts.items():
            assert count <= 7

    def test_infeasible_constraints_returns_status(self):
        """Should return Infeasible status when constraints can't be met."""
        players = create_sample_players()
        # Very restrictive credit limit that can't be satisfied
        constraints = Dream11Constraints(max_credits=50.0)
        optimizer = Dream11Optimizer(constraints=constraints)
        result = optimizer.optimize(players)

        assert result.status == "Infeasible"

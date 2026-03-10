"""Monte Carlo simulation layer for fantasy team evaluation.

This module simulates many possible match outcomes by sampling from
player point distributions. It enables:
- Team score variance estimation
- Upside/downside risk analysis
- Captain leverage calculation
- Simulation-aware team comparison
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

from src.ipl_fantasy.quantile_model import QuantilePrediction
from src.ipl_fantasy.team_optimizer import Player, OptimizationResult


@dataclass
class PlayerDistribution:
    """Distribution of fantasy points for a player."""
    player_name: str
    team: str
    role: str
    credits: float

    # Distribution parameters
    mean: float
    std: float
    floor: float  # 10th percentile
    ceiling: float  # 90th percentile

    # Optional: full distribution for more accurate sampling
    q10: float = 0.0
    q25: float = 0.0
    q50: float = 0.0
    q75: float = 0.0
    q90: float = 0.0

    def sample(self, n: int = 1, method: str = "normal") -> np.ndarray:
        """
        Sample n fantasy point outcomes.

        Args:
            n: Number of samples
            method: Sampling method
                - "normal": Normal distribution with mean/std
                - "truncated_normal": Normal bounded by floor/ceiling
                - "quantile": Sample using quantile interpolation

        Returns:
            Array of n sampled fantasy point values
        """
        if method == "normal":
            samples = np.random.normal(self.mean, self.std, n)
            return np.maximum(0, samples)  # Fantasy points can't be negative

        elif method == "truncated_normal":
            # Truncated normal between floor and ceiling
            a = (self.floor - self.mean) / self.std if self.std > 0 else -np.inf
            b = (self.ceiling - self.mean) / self.std if self.std > 0 else np.inf
            samples = stats.truncnorm.rvs(a, b, loc=self.mean, scale=self.std, size=n)
            return np.maximum(0, samples)

        elif method == "quantile":
            # Sample uniformly from quantiles and interpolate
            u = np.random.uniform(0, 1, n)
            # Define quantile points
            quantiles = [0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0]
            values = [
                max(0, self.q10 - (self.q25 - self.q10)),  # Extrapolate below q10
                self.q10,
                self.q25,
                self.q50,
                self.q75,
                self.q90,
                self.q90 + (self.q90 - self.q75),  # Extrapolate above q90
            ]
            samples = np.interp(u, quantiles, values)
            return np.maximum(0, samples)

        else:
            raise ValueError(f"Unknown sampling method: {method}")

    @classmethod
    def from_quantile_prediction(
        cls,
        pred: QuantilePrediction,
        credits: float = 9.0,
    ) -> "PlayerDistribution":
        """Create distribution from quantile prediction."""
        return cls(
            player_name=pred.player_name,
            team=pred.team,
            role=pred.role,
            credits=credits,
            mean=pred.expected,
            std=math.sqrt(pred.variance) if pred.variance > 0 else pred.iqr / 1.35,
            floor=pred.q10,
            ceiling=pred.q90,
            q10=pred.q10,
            q25=pred.q25,
            q50=pred.q50,
            q75=pred.q75,
            q90=pred.q90,
        )


@dataclass
class SimulationResult:
    """Result of Monte Carlo simulation for a team."""
    players: list[PlayerDistribution]
    n_simulations: int

    # Team-level statistics
    mean_score: float
    std_score: float
    median_score: float
    floor_score: float  # 10th percentile
    ceiling_score: float  # 90th percentile

    # Score distribution
    score_samples: np.ndarray = field(default_factory=lambda: np.array([]))

    # Captain analysis
    captain_scores: dict[str, float] = field(default_factory=dict)
    best_captain: str = ""
    best_captain_expected: float = 0.0

    # VC analysis
    best_vc: str = ""
    best_vc_expected: float = 0.0

    def get_percentile(self, p: float) -> float:
        """Get the p-th percentile of simulated scores."""
        if len(self.score_samples) == 0:
            return self.mean_score
        return float(np.percentile(self.score_samples, p * 100))

    def probability_above(self, threshold: float) -> float:
        """Calculate probability of scoring above threshold."""
        if len(self.score_samples) == 0:
            return 0.5
        return float(np.mean(self.score_samples > threshold))

    def get_summary(self) -> str:
        """Get formatted summary of simulation results."""
        lines = [
            f"Simulation Results ({self.n_simulations:,} simulations)",
            "=" * 50,
            f"Mean Score: {self.mean_score:.1f}",
            f"Std Dev: {self.std_score:.1f}",
            f"Median: {self.median_score:.1f}",
            f"Floor (10%): {self.floor_score:.1f}",
            f"Ceiling (90%): {self.ceiling_score:.1f}",
            "",
            f"Best Captain: {self.best_captain} (Expected: {self.best_captain_expected:.1f})",
            f"Best VC: {self.best_vc} (Expected: {self.best_vc_expected:.1f})",
        ]
        return "\n".join(lines)


class MatchSimulator:
    """Monte Carlo simulator for fantasy match outcomes."""

    def __init__(
        self,
        n_simulations: int = 10000,
        sampling_method: str = "truncated_normal",
        random_seed: int | None = None,
    ):
        self.n_simulations = n_simulations
        self.sampling_method = sampling_method
        if random_seed is not None:
            np.random.seed(random_seed)

    def simulate_team(
        self,
        players: list[PlayerDistribution],
    ) -> SimulationResult:
        """
        Simulate fantasy outcomes for a team of 11 players.

        Args:
            players: List of 11 PlayerDistribution objects

        Returns:
            SimulationResult with team-level statistics
        """
        n = self.n_simulations

        # Sample points for each player
        player_samples = {
            p.player_name: p.sample(n, method=self.sampling_method)
            for p in players
        }

        # Calculate team scores (sum of all players)
        team_scores = np.zeros(n)
        for samples in player_samples.values():
            team_scores += samples

        # Calculate statistics
        mean_score = float(np.mean(team_scores))
        std_score = float(np.std(team_scores))
        median_score = float(np.median(team_scores))
        floor_score = float(np.percentile(team_scores, 10))
        ceiling_score = float(np.percentile(team_scores, 90))

        # Captain analysis: Find best captain based on simulation
        captain_scores = {}
        for p in players:
            # Captain gets 2x points
            # Calculate expected team score with this player as captain
            cap_contribution = player_samples[p.player_name]  # Extra 1x for captain
            cap_team_scores = team_scores + cap_contribution
            captain_scores[p.player_name] = float(np.mean(cap_team_scores))

        best_captain = max(captain_scores, key=captain_scores.get)
        best_captain_expected = captain_scores[best_captain]

        # VC analysis: Find best VC (excluding captain)
        vc_scores = {k: v for k, v in captain_scores.items() if k != best_captain}
        if vc_scores:
            # VC gets 1.5x, so 0.5x extra contribution
            for p in players:
                if p.player_name != best_captain:
                    vc_contribution = player_samples[p.player_name] * 0.5
                    vc_team_scores = team_scores + player_samples[best_captain] + vc_contribution
                    vc_scores[p.player_name] = float(np.mean(vc_team_scores))
            best_vc = max(vc_scores, key=vc_scores.get)
            best_vc_expected = vc_scores[best_vc]
        else:
            best_vc = ""
            best_vc_expected = 0.0

        return SimulationResult(
            players=players,
            n_simulations=n,
            mean_score=mean_score,
            std_score=std_score,
            median_score=median_score,
            floor_score=floor_score,
            ceiling_score=ceiling_score,
            score_samples=team_scores,
            captain_scores=captain_scores,
            best_captain=best_captain,
            best_captain_expected=best_captain_expected,
            best_vc=best_vc,
            best_vc_expected=best_vc_expected,
        )

    def compare_teams(
        self,
        teams: list[list[PlayerDistribution]],
        team_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Compare multiple teams using simulation.

        Args:
            teams: List of teams (each team is a list of PlayerDistribution)
            team_names: Optional names for each team

        Returns:
            Comparison statistics
        """
        if team_names is None:
            team_names = [f"Team_{i+1}" for i in range(len(teams))]

        results = []
        for team in teams:
            result = self.simulate_team(team)
            results.append(result)

        # Compare teams
        comparison = {
            "teams": [],
            "best_mean": {"name": "", "score": 0},
            "best_ceiling": {"name": "", "score": 0},
            "best_floor": {"name": "", "score": 0},
            "lowest_variance": {"name": "", "std": float("inf")},
        }

        for name, result in zip(team_names, results):
            team_info = {
                "name": name,
                "mean": result.mean_score,
                "std": result.std_score,
                "floor": result.floor_score,
                "ceiling": result.ceiling_score,
                "captain": result.best_captain,
                "vc": result.best_vc,
            }
            comparison["teams"].append(team_info)

            if result.mean_score > comparison["best_mean"]["score"]:
                comparison["best_mean"] = {"name": name, "score": result.mean_score}
            if result.ceiling_score > comparison["best_ceiling"]["score"]:
                comparison["best_ceiling"] = {"name": name, "score": result.ceiling_score}
            if result.floor_score > comparison["best_floor"]["score"]:
                comparison["best_floor"] = {"name": name, "score": result.floor_score}
            if result.std_score < comparison["lowest_variance"]["std"]:
                comparison["lowest_variance"] = {"name": name, "std": result.std_score}

        return comparison

    def simulate_captain_choices(
        self,
        players: list[PlayerDistribution],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Analyze captain choices using simulation.

        Args:
            players: List of PlayerDistribution objects
            top_k: Number of top captain choices to return

        Returns:
            List of captain analysis dictionaries
        """
        n = self.n_simulations

        # Sample all players
        player_samples = {
            p.player_name: p.sample(n, method=self.sampling_method)
            for p in players
        }

        # Base team score (without captain multiplier)
        base_score = np.zeros(n)
        for samples in player_samples.values():
            base_score += samples

        # Analyze each captain choice
        captain_analysis = []
        for p in players:
            samples = player_samples[p.player_name]

            # Team score with this captain (2x their points)
            cap_team_score = base_score + samples  # Add extra 1x for captain

            analysis = {
                "player_name": p.player_name,
                "team": p.team,
                "role": p.role,
                "mean_contribution": float(np.mean(samples)),
                "cap_mean_score": float(np.mean(cap_team_score)),
                "cap_std_score": float(np.std(cap_team_score)),
                "cap_ceiling": float(np.percentile(cap_team_score, 90)),
                "cap_floor": float(np.percentile(cap_team_score, 10)),
                "upside": float(np.percentile(samples, 90) - np.mean(samples)),
                "consistency": 1.0 / (float(np.std(samples)) + 1),  # Higher is more consistent
            }
            captain_analysis.append(analysis)

        # Sort by expected team score with this captain
        captain_analysis.sort(key=lambda x: x["cap_mean_score"], reverse=True)

        return captain_analysis[:top_k]


def create_distributions_from_predictions(
    predictions: list[QuantilePrediction],
    credits: dict[str, float] | None = None,
    default_credits: float = 9.0,
) -> list[PlayerDistribution]:
    """
    Create PlayerDistribution objects from QuantilePredictions.

    Args:
        predictions: List of QuantilePrediction objects
        credits: Optional dictionary mapping player names to credits
        default_credits: Default credit value if not specified

    Returns:
        List of PlayerDistribution objects
    """
    credits = credits or {}
    distributions = []

    for pred in predictions:
        player_credits = credits.get(pred.player_name, default_credits)
        dist = PlayerDistribution.from_quantile_prediction(pred, player_credits)
        distributions.append(dist)

    return distributions

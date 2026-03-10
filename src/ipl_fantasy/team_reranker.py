"""Phase 2 — Candidate Team Generation & Simulation-Based Reranking.

Generates K diverse candidate Dream11 teams using different optimization
strategies, simulates each under Monte Carlo, then reranks using a
composite objective that captures expected value, upside, floor safety,
and captain leverage.

Pipeline:
    1. Generate candidates via varied optimizer configs + diversity penalty.
    2. For each candidate, run Monte Carlo with simulation-optimal C/VC.
    3. Score each candidate on a composite reranking metric.
    4. Return the top team (or top-N for further selection).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.ipl_fantasy.improved_optimizer import ImprovedDream11Optimizer, OptimizationConfig
from src.ipl_fantasy.simulation import (
    MatchSimulator,
    PlayerDistribution,
    SimulationResult,
)
from src.ipl_fantasy.team_optimizer import (
    Dream11Constraints,
    Dream11Optimizer,
    OptimizationResult,
    Player,
)



@dataclass
class RerankingConfig:
    """Controls candidate generation and reranking behaviour."""

    n_candidates: int = 8
    diversity_penalty: float = 0.12

    n_simulations: int = 5000
    sampling_method: str = "truncated_normal"

    # Composite score = Σ weight_i × z(metric_i)
    w_expected: float = 0.30
    w_ceiling: float = 0.25
    w_floor: float = 0.10
    w_captain_leverage: float = 0.20
    w_sharpe: float = 0.15


# Pre-baked optimizer configs used to seed diverse candidates
_CANDIDATE_CONFIGS: list[tuple[str, OptimizationConfig]] = [
    (
        "expected",
        OptimizationConfig(
            expected_weight=0.85, ceiling_weight=0.10, floor_weight=0.05,
            captain_ceiling_weight=0.35,
        ),
    ),
    (
        "balanced",
        OptimizationConfig(
            expected_weight=0.60, ceiling_weight=0.30, floor_weight=0.10,
            captain_ceiling_weight=0.50,
        ),
    ),
    (
        "ceiling",
        OptimizationConfig(
            expected_weight=0.30, ceiling_weight=0.60, floor_weight=0.10,
            captain_ceiling_weight=0.70,
        ),
    ),
    (
        "floor",
        OptimizationConfig(
            expected_weight=0.55, ceiling_weight=0.10, floor_weight=0.35,
            captain_ceiling_weight=0.30,
        ),
    ),
    (
        "captain_heavy",
        OptimizationConfig(
            expected_weight=0.50, ceiling_weight=0.35, floor_weight=0.15,
            captain_ceiling_weight=0.85,
        ),
    ),
]



@dataclass
class RankedTeam:
    """A candidate team enriched with simulation statistics."""

    label: str
    result: OptimizationResult

    sim_result: SimulationResult | None = None
    sim_mean: float = 0.0
    sim_ceiling: float = 0.0
    sim_floor: float = 0.0
    sim_std: float = 0.0
    captain_leverage: float = 0.0
    sharpe: float = 0.0

    sim_captain: str = ""
    sim_vc: str = ""

    composite_score: float = 0.0

    def player_names(self) -> list[str]:
        return [p.name for p in self.result.selected_players]

    def get_summary(self) -> str:
        cap_marker = lambda p: (
            " (C)" if p.name == self.sim_captain else
            " (VC)" if p.name == self.sim_vc else ""
        )
        lines = [
            f"--- {self.label}  [composite={self.composite_score:.3f}] ---",
            f"  Expected: {self.sim_mean:.1f}  Ceiling: {self.sim_ceiling:.1f}  "
            f"Floor: {self.sim_floor:.1f}  Sharpe: {self.sharpe:.2f}  "
            f"CaptLev: {self.captain_leverage:.1f}",
        ]
        by_role: dict[str, list[Player]] = {"WK": [], "BAT": [], "AR": [], "BOWL": []}
        for p in self.result.selected_players:
            by_role.get(p.role, by_role["BAT"]).append(p)
        for role in ("WK", "BAT", "AR", "BOWL"):
            if by_role[role]:
                names = ", ".join(
                    f"{p.name}{cap_marker(p)}" for p in
                    sorted(by_role[role], key=lambda x: -x.predicted_points)
                )
                lines.append(f"  {role}: {names}")
        return "\n".join(lines)



def _player_to_distribution(p: Player) -> PlayerDistribution:
    """Convert a Player to a PlayerDistribution for simulation."""
    expected = p.predicted_points
    ceiling = p.ceiling if p.ceiling is not None else expected * 1.5
    floor = p.floor if p.floor is not None else expected * 0.5
    var = p.variance if p.variance is not None else ((ceiling - floor) / 2.56) ** 2
    std = math.sqrt(max(var, 1e-6))
    q50 = (expected + (ceiling + floor) / 2) / 2  # blend

    return PlayerDistribution(
        player_name=p.name,
        team=p.team,
        role=p.role,
        credits=p.credits,
        mean=expected,
        std=std,
        floor=floor,
        ceiling=ceiling,
        q10=floor,
        q25=expected * 0.75 if floor < expected else floor,
        q50=q50,
        q75=expected * 1.25 if ceiling > expected else ceiling,
        q90=ceiling,
    )


def generate_candidates(
    players: list[Player],
    constraints: Dream11Constraints | None = None,
    config: RerankingConfig | None = None,
) -> list[RankedTeam]:
    """
    Produce K diverse candidate teams.

    Strategy:
      - First, generate one team per pre-baked optimizer config.
      - Then, fill remaining slots using diversity-penalised re-solves
        with the balanced config.
      - Deduplicate identical player sets.

    Args:
        players: Full player pool with predictions.
        constraints: Dream11 constraints.
        config: Reranking configuration.

    Returns:
        List of RankedTeam objects (not yet simulated).
    """
    config = config or RerankingConfig()
    constraints = constraints or Dream11Constraints()

    candidates: list[RankedTeam] = []
    seen_sets: set[frozenset[str]] = set()

    def _add_if_new(result: OptimizationResult, label: str) -> bool:
        names = frozenset(p.name for p in result.selected_players)
        if names in seen_sets:
            return False
        seen_sets.add(names)
        candidates.append(RankedTeam(label=label, result=result))
        return True

    # Phase A: one team per config archetype
    for label, opt_cfg in _CANDIDATE_CONFIGS:
        if len(candidates) >= config.n_candidates:
            break
        optimizer = ImprovedDream11Optimizer(constraints=constraints, config=opt_cfg)
        try:
            result = optimizer.optimize_ceiling_weighted(players)
            _add_if_new(result, label)
        except Exception:
            continue

    # Phase B: also use base optimizer with different objectives
    base_optimizer = Dream11Optimizer(constraints=constraints)
    for obj, label in [
        ("maximize_points", "base_expected"),
        ("maximize_ceiling", "base_ceiling"),
        ("maximize_floor", "base_floor"),
    ]:
        if len(candidates) >= config.n_candidates:
            break
        try:
            result = base_optimizer.optimize(players, objective=obj)
            _add_if_new(result, label)
        except Exception:
            continue

    # Phase C: diversity-penalised re-solves to fill remaining slots
    if len(candidates) < config.n_candidates:
        balanced_cfg = OptimizationConfig(
            expected_weight=0.60, ceiling_weight=0.30, floor_weight=0.10,
        )
        optimizer = ImprovedDream11Optimizer(constraints=constraints, config=balanced_cfg)
        player_usage: dict[str, int] = {}
        for c in candidates:
            for p in c.result.selected_players:
                player_usage[p.name] = player_usage.get(p.name, 0) + 1

        attempts = 0
        while len(candidates) < config.n_candidates and attempts < 15:
            attempts += 1
            adjusted = []
            for p in players:
                usage = player_usage.get(p.name, 0)
                penalty = 1.0 - config.diversity_penalty * usage
                adjusted.append(Player(
                    name=p.name,
                    team=p.team,
                    role=p.role,
                    predicted_points=p.predicted_points * max(penalty, 0.1),
                    credits=p.credits,
                    ceiling=p.ceiling,
                    floor=p.floor,
                    variance=p.variance,
                ))
            try:
                result = optimizer.optimize_ceiling_weighted(adjusted)
                # Swap back to original Player objects
                name_map = {p.name: p for p in players}
                result.selected_players = [name_map[p.name] for p in result.selected_players]
                result.total_predicted_points = sum(
                    p.predicted_points for p in result.selected_players
                )
                if result.captain:
                    result.captain = name_map.get(result.captain.name, result.captain)
                if result.vice_captain:
                    result.vice_captain = name_map.get(result.vice_captain.name, result.vice_captain)

                added = _add_if_new(result, f"diverse_{attempts}")
                if added:
                    for p in result.selected_players:
                        player_usage[p.name] = player_usage.get(p.name, 0) + 1
            except Exception:
                continue

    return candidates



def simulate_candidates(
    candidates: list[RankedTeam],
    config: RerankingConfig | None = None,
) -> list[RankedTeam]:
    """
    Run Monte Carlo simulation on each candidate and populate stats.

    Also reassigns captain/VC based on simulation-optimal picks.
    """
    config = config or RerankingConfig()
    simulator = MatchSimulator(
        n_simulations=config.n_simulations,
        sampling_method=config.sampling_method,
    )

    for team in candidates:
        dists = [_player_to_distribution(p) for p in team.result.selected_players]
        sim = simulator.simulate_team(dists)
        team.sim_result = sim

        base_mean = sim.mean_score
        team.sim_ceiling = sim.ceiling_score
        team.sim_floor = sim.floor_score
        team.sim_std = sim.std_score

        team.sim_captain = sim.best_captain
        team.sim_vc = sim.best_vc

        cap_dist = next(
            (d for d in dists if d.player_name == sim.best_captain), None
        )
        vc_dist = next(
            (d for d in dists if d.player_name == sim.best_vc), None
        )
        captain_extra = cap_dist.mean if cap_dist else 0.0
        vc_extra = (vc_dist.mean * 0.5) if vc_dist else 0.0
        team.sim_mean = base_mean + captain_extra + vc_extra
        team.captain_leverage = captain_extra + vc_extra

        team.sharpe = team.sim_mean / team.sim_std if team.sim_std > 0 else 0.0

    return candidates


def rerank(
    candidates: list[RankedTeam],
    config: RerankingConfig | None = None,
) -> list[RankedTeam]:
    """
    Score and sort candidates by composite reranking metric.

    Each raw metric is z-scored across the candidate pool first, then
    combined with the configured weights.
    """
    config = config or RerankingConfig()

    if not candidates:
        return candidates

    means = np.array([t.sim_mean for t in candidates])
    ceilings = np.array([t.sim_ceiling for t in candidates])
    floors = np.array([t.sim_floor for t in candidates])
    leverages = np.array([t.captain_leverage for t in candidates])
    sharpes = np.array([t.sharpe for t in candidates])

    def z_score(arr: np.ndarray) -> np.ndarray:
        s = arr.std()
        if s < 1e-9:
            return np.zeros_like(arr)
        return (arr - arr.mean()) / s

    z_mean = z_score(means)
    z_ceil = z_score(ceilings)
    z_floor = z_score(floors)
    z_lev = z_score(leverages)
    z_sharpe = z_score(sharpes)

    for i, team in enumerate(candidates):
        team.composite_score = (
            config.w_expected * z_mean[i]
            + config.w_ceiling * z_ceil[i]
            + config.w_floor * z_floor[i]
            + config.w_captain_leverage * z_lev[i]
            + config.w_sharpe * z_sharpe[i]
        )

    candidates.sort(key=lambda t: t.composite_score, reverse=True)
    return candidates



def select_best_team(
    players: list[Player],
    constraints: Dream11Constraints | None = None,
    config: RerankingConfig | None = None,
) -> RankedTeam:
    """
    Full pipeline: generate → simulate → rerank → return best.

    Args:
        players: Player pool with predictions.
        constraints: Dream11 constraints.
        config: Reranking config.

    Returns:
        The highest-scoring RankedTeam after simulation reranking.
    """
    config = config or RerankingConfig()
    candidates = generate_candidates(players, constraints, config)
    candidates = simulate_candidates(candidates, config)
    candidates = rerank(candidates, config)
    return candidates[0]


def select_top_k(
    players: list[Player],
    k: int = 3,
    constraints: Dream11Constraints | None = None,
    config: RerankingConfig | None = None,
) -> list[RankedTeam]:
    """
    Return the top-K reranked teams.

    Useful for presenting alternatives or feeding into Phase 4 RL selection.
    """
    config = config or RerankingConfig()
    candidates = generate_candidates(players, constraints, config)
    candidates = simulate_candidates(candidates, config)
    candidates = rerank(candidates, config)
    return candidates[:k]


def get_reranking_summary(teams: list[RankedTeam]) -> str:
    """Pretty-print the reranked candidate list."""
    lines = [
        "=" * 65,
        "CANDIDATE TEAM RERANKING",
        "=" * 65,
        f"Candidates evaluated: {len(teams)}",
        "",
    ]
    for rank, team in enumerate(teams, 1):
        lines.append(f"#{rank}  {team.get_summary()}")
        lines.append("")

    if teams:
        best = teams[0]
        lines.append(f"SELECTED: {best.label}")
        lines.append(
            f"  Captain: {best.sim_captain}  |  VC: {best.sim_vc}  |  "
            f"Expected: {best.sim_mean:.1f}"
        )
    lines.append("=" * 65)
    return "\n".join(lines)

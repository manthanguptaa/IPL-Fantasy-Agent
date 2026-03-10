"""Reward model for reinforcement learning in fantasy team selection.

Computes scalar rewards from match outcomes to train the contextual bandit.
Supports multiple reward formulations:
- Score-based: raw selected team score
- Regret-based: negative gap to oracle
- Blended: weighted combination with captain quality bonus
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RewardConfig:
    """Configuration for reward computation."""
    # Reward formulation weights
    score_weight: float = 0.4       # Weight on raw score
    regret_weight: float = 0.4      # Weight on negative regret
    captain_weight: float = 0.2     # Weight on captain quality

    # Normalization constants (approximate, from backtest baselines)
    score_mean: float = 500.0
    score_std: float = 120.0
    regret_mean: float = 300.0
    regret_std: float = 100.0
    captain_score_mean: float = 50.0
    captain_score_std: float = 30.0


def compute_reward(
    selected_score: float,
    oracle_score: float,
    captain_actual: float,
    oracle_captain_actual: float,
    config: RewardConfig | None = None,
) -> float:
    """
    Compute a scalar reward from match outcome.

    Args:
        selected_score: Total team score with C/VC multipliers.
        oracle_score: Oracle team score with C/VC multipliers.
        captain_actual: Actual fantasy points scored by chosen captain.
        oracle_captain_actual: Actual points scored by oracle captain.
        config: Reward configuration.

    Returns:
        Normalized scalar reward in roughly [-2, 2] range.
    """
    config = config or RewardConfig()

    score_z = (selected_score - config.score_mean) / config.score_std

    regret = oracle_score - selected_score
    regret_z = -(regret - config.regret_mean) / config.regret_std

    captain_gap = oracle_captain_actual - captain_actual
    captain_z = -captain_gap / config.captain_score_std

    reward = (
        config.score_weight * score_z
        + config.regret_weight * regret_z
        + config.captain_weight * captain_z
    )
    return float(reward)


def compute_simple_reward(
    selected_score: float,
    oracle_score: float,
) -> float:
    """Simplified reward: fraction of oracle score captured."""
    if oracle_score <= 0:
        return 0.0
    return selected_score / oracle_score


def compute_regret_reward(
    selected_score: float,
    oracle_score: float,
    baseline_regret: float = 300.0,
) -> float:
    """
    Reward based on regret improvement over baseline.

    Returns positive reward when regret is below baseline,
    negative when above.
    """
    regret = oracle_score - selected_score
    return (baseline_regret - regret) / baseline_regret

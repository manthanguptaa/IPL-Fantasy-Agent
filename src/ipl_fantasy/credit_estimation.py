"""Dream11 Credit Estimation based on player performance.

Since Dream11 credits are not publicly available, this module estimates
credits based on historical performance patterns observed in the data.

Based on research:
- Credit range: 7.0 to 11.0 (most players 8.0-10.5)
- Top players (Kohli, Bumrah): 10.0-10.5
- Good performers: 9.0-9.5
- Average players: 8.0-8.5
- Budget picks: 7.0-7.5

Factors affecting credits:
1. Historical average performance
2. Recent form (last 5-10 matches)
3. Player role (WK/AR slightly higher due to multi-role scoring)
4. Match importance and consistency
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class CreditConfig:
    """Configuration for credit estimation."""
    min_credits: float = 7.0
    max_credits: float = 11.0

    # Typical credit ranges by tier
    elite_threshold: float = 55.0  # Average points for elite tier
    good_threshold: float = 40.0   # Average points for good tier
    average_threshold: float = 25.0  # Average points for average tier

    # Role adjustments (WK/AR tend to score more consistently)
    role_bonus: dict = None

    def __post_init__(self):
        if self.role_bonus is None:
            self.role_bonus = {
                "WK": 0.3,   # WK-batsmen score well
                "AR": 0.4,   # All-rounders have dual scoring
                "BAT": 0.0,  # Baseline
                "BOWL": 0.1, # Bowlers can have high-impact games
            }


# Known star players and their approximate credits (based on public info)
KNOWN_PLAYER_CREDITS = {
    # Top-tier batsmen (10.0-10.5)
    "V Kohli": 10.5,
    "RG Sharma": 10.0,
    "KL Rahul": 10.0,
    "SA Yadav": 9.5,
    "S Gill": 9.5,

    # Top-tier bowlers (9.0-10.0)
    "JJ Bumrah": 10.0,
    "Mohammed Shami": 9.5,
    "YS Chahal": 9.0,
    "Rashid Khan": 9.5,
    "R Ashwin": 9.0,

    # Top-tier all-rounders (9.5-10.5)
    "RA Jadeja": 10.0,
    "HH Pandya": 10.0,
    "AR Patel": 9.5,
    "Washington Sundar": 9.0,
    "SN Thakur": 9.0,

    # Top-tier wicket-keepers (9.0-10.0)
    "RR Pant": 9.5,
    "MS Dhoni": 9.0,
    "KD Karthik": 8.5,
    "SV Samson": 9.0,
    "Q de Kock": 9.5,

    # Good performers (8.5-9.0)
    "R Parag": 8.5,
    "Abhishek Sharma": 8.5,
    "T Head": 9.0,
    "DA Warner": 9.5,
    "F du Plessis": 9.0,
    "JC Buttler": 10.0,
    "GJ Maxwell": 9.0,
    "DJ Mitchell": 8.5,
}


def estimate_credits_from_history(
    player_name: str,
    player_role: str,
    avg_points_all: float,
    avg_points_recent: float = None,
    config: CreditConfig = None,
) -> float:
    """
    Estimate Dream11 credits based on player history.

    Args:
        player_name: Player name
        player_role: Player role (WK, BAT, AR, BOWL)
        avg_points_all: Historical average fantasy points
        avg_points_recent: Recent average (last 5-10 matches)
        config: Credit estimation configuration

    Returns:
        Estimated credit value (7.0-11.0)
    """
    config = config or CreditConfig()

    # Check if player has known credits
    if player_name in KNOWN_PLAYER_CREDITS:
        return KNOWN_PLAYER_CREDITS[player_name]

    # Use recent average if available, else use all-time
    avg_points = avg_points_recent if avg_points_recent is not None else avg_points_all

    # Base credit calculation
    if avg_points >= config.elite_threshold:
        # Elite tier: 9.5-10.5
        base_credits = 9.5 + min(1.0, (avg_points - config.elite_threshold) / 30)
    elif avg_points >= config.good_threshold:
        # Good tier: 8.5-9.5
        ratio = (avg_points - config.good_threshold) / (config.elite_threshold - config.good_threshold)
        base_credits = 8.5 + ratio
    elif avg_points >= config.average_threshold:
        # Average tier: 7.5-8.5
        ratio = (avg_points - config.average_threshold) / (config.good_threshold - config.average_threshold)
        base_credits = 7.5 + ratio
    else:
        # Budget tier: 7.0-7.5
        base_credits = max(7.0, 7.0 + avg_points / 50)

    # Apply role bonus
    role_bonus = config.role_bonus.get(player_role, 0.0)
    credits = base_credits + role_bonus

    # Clamp to valid range
    return max(config.min_credits, min(config.max_credits, credits))


def add_credits_to_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add estimated credits column to features DataFrame.

    Args:
        df: Features DataFrame with player_name, player_role, and rolling averages

    Returns:
        DataFrame with 'estimated_credits' column added
    """
    credits = []

    for _, row in df.iterrows():
        player_name = row.get("player_name", "")
        player_role = row.get("player_role", "BAT")

        # Use rolling averages if available
        avg_all = row.get("rolling_points_avg_10_all",
                         row.get("rolling_points_avg_5_all", 30.0))
        avg_recent = row.get("rolling_points_avg_5_all", None)

        credit = estimate_credits_from_history(
            player_name=player_name,
            player_role=player_role,
            avg_points_all=avg_all,
            avg_points_recent=avg_recent,
        )
        credits.append(credit)

    df = df.copy()
    df["estimated_credits"] = credits

    return df


def create_credit_lookup(df: pd.DataFrame) -> dict[str, float]:
    """
    Create a lookup dictionary of player credits from features.

    Uses the most recent match data for each player.
    """
    # Sort by date and get most recent entry per player
    df_sorted = df.sort_values("match_date", ascending=False)
    latest = df_sorted.groupby("player_name").first().reset_index()

    credits = {}
    for _, row in latest.iterrows():
        player_name = row["player_name"]
        player_role = row.get("player_role", "BAT")
        avg_all = row.get("rolling_points_avg_10_all",
                         row.get("rolling_points_avg_5_all", 30.0))
        avg_recent = row.get("rolling_points_avg_5_all", None)

        credits[player_name] = estimate_credits_from_history(
            player_name=player_name,
            player_role=player_role,
            avg_points_all=avg_all,
            avg_points_recent=avg_recent,
        )

    return credits


def analyze_credit_distribution(df: pd.DataFrame) -> dict:
    """Analyze the distribution of estimated credits."""
    df_with_credits = add_credits_to_features(df)

    credits = df_with_credits["estimated_credits"]

    return {
        "min": credits.min(),
        "max": credits.max(),
        "mean": credits.mean(),
        "median": credits.median(),
        "std": credits.std(),
        "by_role": df_with_credits.groupby("player_role")["estimated_credits"].mean().to_dict(),
        "distribution": {
            "7.0-7.5": (credits < 7.5).sum(),
            "7.5-8.0": ((credits >= 7.5) & (credits < 8.0)).sum(),
            "8.0-8.5": ((credits >= 8.0) & (credits < 8.5)).sum(),
            "8.5-9.0": ((credits >= 8.5) & (credits < 9.0)).sum(),
            "9.0-9.5": ((credits >= 9.0) & (credits < 9.5)).sum(),
            "9.5-10.0": ((credits >= 9.5) & (credits < 10.0)).sum(),
            "10.0+": (credits >= 10.0).sum(),
        }
    }


if __name__ == "__main__":
    import sys

    # Test with sample data
    features_path = sys.argv[1] if len(sys.argv) > 1 else "tmp/full_player_match_features_v3.csv"

    print(f"Loading features from {features_path}...")
    df = pd.read_csv(features_path, low_memory=False)

    # Filter to recent IPL matches
    df = df[df["competition"] == "Indian Premier League"]
    df = df.sort_values("match_date", ascending=False)

    # Analyze credit distribution
    print("\nCredit Distribution Analysis:")
    analysis = analyze_credit_distribution(df.head(5000))

    print(f"  Min: {analysis['min']:.1f}")
    print(f"  Max: {analysis['max']:.1f}")
    print(f"  Mean: {analysis['mean']:.1f}")
    print(f"  Median: {analysis['median']:.1f}")

    print("\n  By Role:")
    for role, avg in analysis["by_role"].items():
        print(f"    {role}: {avg:.2f}")

    print("\n  Distribution:")
    for bucket, count in analysis["distribution"].items():
        print(f"    {bucket}: {count}")

    # Show some example credits
    print("\nExample Player Credits (most recent match):")
    credits = create_credit_lookup(df.head(1000))

    # Sort by credit value
    sorted_credits = sorted(credits.items(), key=lambda x: x[1], reverse=True)

    print("\n  Top 20 (highest credits):")
    for name, credit in sorted_credits[:20]:
        print(f"    {name:30} {credit:.1f}")

    print("\n  Bottom 10 (lowest credits):")
    for name, credit in sorted_credits[-10:]:
        print(f"    {name:30} {credit:.1f}")

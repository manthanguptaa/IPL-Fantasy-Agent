"""
Data preparation for match prediction models.
Aggregates player-level data into match-level and team-level features.
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORE_DATA = PROJECT_ROOT / "tmp" / "core_player_match_base.csv"
FEATURES_DATA = PROJECT_ROOT / "tmp" / "full_player_match_features_curated_best.csv"
VENUE_DATA = PROJECT_ROOT / "data" / "venue_profiles.csv"


def load_ipl_data() -> pd.DataFrame:
    """Load core player-match data filtered to IPL only."""
    df = pd.read_csv(CORE_DATA, parse_dates=["match_date"])
    df = df[df["competition"] == "Indian Premier League"].copy()
    df = df.sort_values(["match_date", "match_id", "team"]).reset_index(drop=True)
    return df


def load_feature_data() -> pd.DataFrame:
    """Load full feature dataset filtered to IPL only."""
    df = pd.read_csv(FEATURES_DATA, parse_dates=["match_date"])
    df = df[df["competition"] == "Indian Premier League"].copy()
    df = df.sort_values(["match_date", "match_id", "team"]).reset_index(drop=True)
    return df


def load_venue_profiles() -> pd.DataFrame:
    """Load venue profiles with scoring characteristics."""
    return pd.read_csv(VENUE_DATA)


def build_match_level_data(player_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate player-level stats into match-level data.
    Each row = one match with stats for both teams.
    """
    # Team-level aggregation per match
    team_stats = (
        player_df.groupby(["match_id", "match_date", "venue", "city", "season", "team", "opponent"])
        .agg(
            team_runs=("runs", "sum"),
            team_balls_faced=("balls_faced", "sum"),
            team_fours=("fours", "sum"),
            team_sixes=("sixes", "sum"),
            team_wickets_lost=("duck", "sum"),  # placeholder, will fix below
            team_wickets_taken=("wickets", "sum"),
            team_catches=("catches", "sum"),
            team_maidens=("maidens", "sum"),
            team_runs_conceded=("runs_conceded", "sum"),
            player_count=("player_name", "count"),
            total_balls_bowled=("balls_bowled", "sum"),
        )
        .reset_index()
    )

    # The actual wickets lost by a team = wickets taken by the opponent
    # team_runs = batting total, team_wickets_taken = bowling wickets taken against opponent
    # So team_runs IS the batting score, team_wickets_taken IS the wickets they took (opponent lost)

    # Get toss and winner info (one row per match)
    match_meta = (
        player_df.groupby("match_id")
        .agg(
            toss_winner=("toss_winner", "first"),
            toss_decision=("toss_decision", "first"),
            winner=("winner", "first"),
        )
        .reset_index()
    )

    team_stats = team_stats.merge(match_meta, on="match_id", how="left")

    # Build match-level rows: team1 vs team2
    matches = []
    for match_id, grp in team_stats.groupby("match_id"):
        if len(grp) != 2:
            continue  # skip incomplete matches

        t1, t2 = grp.iloc[0], grp.iloc[1]

        # Determine batting order from toss
        # If toss winner chose to bat, they bat first
        # If toss winner chose to field, opponent bats first
        toss_winner = t1["toss_winner"]
        toss_decision = t1["toss_decision"]

        if toss_winner == t1["team"]:
            if toss_decision == "bat":
                bat_first, bat_second = t1, t2
            else:
                bat_first, bat_second = t2, t1
        elif toss_winner == t2["team"]:
            if toss_decision == "bat":
                bat_first, bat_second = t2, t1
            else:
                bat_first, bat_second = t1, t2
        else:
            # Toss winner not matching either team name — just use order as-is
            bat_first, bat_second = t1, t2

        winner = t1["winner"]
        bat_first_won = 1 if winner == bat_first["team"] else 0
        # Margin: if batting first team won, margin = runs difference
        # If batting second team won, they chased successfully
        margin_runs = bat_first["team_runs"] - bat_second["team_runs"]

        matches.append(
            {
                "match_id": match_id,
                "match_date": t1["match_date"],
                "venue": t1["venue"],
                "city": t1["city"],
                "season": t1["season"],
                "toss_winner": toss_winner,
                "toss_decision": toss_decision,
                "winner": winner,
                # Batting first team
                "team_bat_first": bat_first["team"],
                "score_bat_first": bat_first["team_runs"],
                "fours_bat_first": bat_first["team_fours"],
                "sixes_bat_first": bat_first["team_sixes"],
                "wickets_lost_bat_first": bat_first["team_wickets_taken"],  # opponent took these
                # Wait: team_wickets_taken = wickets THIS team's bowlers took
                # So wickets lost by bat_first = bat_second's team_wickets_taken
                # Let me fix this...
                # Batting second team
                "team_bat_second": bat_second["team"],
                "score_bat_second": bat_second["team_runs"],
                "fours_bat_second": bat_second["team_fours"],
                "sixes_bat_second": bat_second["team_sixes"],
                # Match result
                "bat_first_won": bat_first_won,
                "run_margin": margin_runs,  # positive = bat first scored more
            }
        )

    match_df = pd.DataFrame(matches)

    # Fix wickets: wickets lost by team A = wickets taken by team B
    # We need to re-derive from team_stats
    wickets_map = team_stats.set_index(["match_id", "team"])["team_wickets_taken"].to_dict()
    match_df["wickets_lost_bat_first"] = match_df.apply(
        lambda r: wickets_map.get((r["match_id"], r["team_bat_second"]), 0), axis=1
    )
    match_df["wickets_lost_bat_second"] = match_df.apply(
        lambda r: wickets_map.get((r["match_id"], r["team_bat_first"]), 0), axis=1
    )

    return match_df.sort_values("match_date").reset_index(drop=True)


def build_season_avg_scores(match_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-season average first-innings score, lagged by 1 season to avoid leakage."""
    season_avg = (
        match_df.groupby("season")
        .agg(season_avg_score=("score_bat_first", "mean"))
        .reset_index()
    )
    season_avg = season_avg.sort_values("season")
    season_avg["season_avg_score_lag"] = season_avg["season_avg_score"].shift(1)
    return season_avg[["season", "season_avg_score_lag"]]


def build_team_rolling_features(match_df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Add rolling team-level features: recent form, avg scores, win rates.
    Features are computed separately for each team.
    """
    # Build a long-form team performance table (each team appears once per match)
    team_rows = []
    for _, row in match_df.iterrows():
        # Batting first team
        team_rows.append(
            {
                "match_id": row["match_id"],
                "match_date": row["match_date"],
                "team": row["team_bat_first"],
                "opponent": row["team_bat_second"],
                "venue": row["venue"],
                "runs_scored": row["score_bat_first"],
                "runs_conceded": row["score_bat_second"],
                "won": 1 if row["bat_first_won"] == 1 else 0,
                "batted_first": 1,
            }
        )
        # Batting second team
        team_rows.append(
            {
                "match_id": row["match_id"],
                "match_date": row["match_date"],
                "team": row["team_bat_second"],
                "opponent": row["team_bat_first"],
                "venue": row["venue"],
                "runs_scored": row["score_bat_second"],
                "runs_conceded": row["score_bat_first"],
                "won": 1 if row["bat_first_won"] == 0 else 0,
                "batted_first": 0,
            }
        )

    team_df = pd.DataFrame(team_rows).sort_values(["team", "match_date"]).reset_index(drop=True)

    # Rolling features per team (shift by 1 to avoid leakage)
    for col in ["runs_scored", "runs_conceded", "won"]:
        team_df[f"rolling_{col}_avg_{window}"] = (
            team_df.groupby("team")[col]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=2).mean())
        )

    team_df[f"rolling_run_diff_avg_{window}"] = (
        team_df[f"rolling_runs_scored_avg_{window}"] - team_df[f"rolling_runs_conceded_avg_{window}"]
    )

    return team_df


def merge_venue_features(match_df: pd.DataFrame, venue_df: pd.DataFrame) -> pd.DataFrame:
    """Add venue characteristics to match data via fuzzy venue name matching."""
    # Build a mapping from venue alias substrings to venue profiles
    venue_map = {}
    for _, row in venue_df.iterrows():
        aliases = str(row["venue_aliases"]).split(";")
        for alias in aliases:
            venue_map[alias.strip()] = row

    def find_venue(venue_name: str):
        if pd.isna(venue_name):
            return None
        # Exact match first
        if venue_name in venue_map:
            return venue_map[venue_name]
        # Substring match
        for alias, profile in venue_map.items():
            if alias in venue_name or venue_name in alias:
                return profile
        return None

    venue_features = []
    for venue in match_df["venue"]:
        profile = find_venue(venue)
        if profile is not None:
            venue_features.append(
                {
                    "avg_total_runs": profile["avg_total_runs"],
                    "avg_total_sixes": profile["avg_total_sixes"],
                    "pace_economy": profile["pace_economy"],
                    "spin_economy": profile["spin_economy"],
                    "batting_friendly": profile["batting_friendly"],
                    "spin_friendly": profile["spin_friendly"],
                    "dew_factor": profile["dew_factor"],
                }
            )
        else:
            venue_features.append(
                {
                    "avg_total_runs": np.nan,
                    "avg_total_sixes": np.nan,
                    "pace_economy": np.nan,
                    "spin_economy": np.nan,
                    "batting_friendly": np.nan,
                    "spin_friendly": np.nan,
                    "dew_factor": np.nan,
                }
            )

    venue_feat_df = pd.DataFrame(venue_features)
    return pd.concat([match_df.reset_index(drop=True), venue_feat_df], axis=1)


def prepare_all_data():
    """Main entry point: prepare all datasets needed for the 4 prediction tasks."""
    print("Loading IPL data...")
    player_df = load_ipl_data()
    feature_df = load_feature_data()
    venue_df = load_venue_profiles()

    print(f"  {len(player_df)} player-match rows, {player_df['match_id'].nunique()} matches")

    print("Building match-level data...")
    match_df = build_match_level_data(player_df)
    print(f"  {len(match_df)} matches built")

    print("Adding venue features...")
    match_df = merge_venue_features(match_df, venue_df)

    print("Building team rolling features...")
    team_rolling = build_team_rolling_features(match_df)

    print("Done!")
    return {
        "player_df": player_df,
        "feature_df": feature_df,
        "match_df": match_df,
        "team_rolling": team_rolling,
        "venue_df": venue_df,
    }


if __name__ == "__main__":
    data = prepare_all_data()
    print("\nMatch-level data sample:")
    print(data["match_df"][["match_date", "team_bat_first", "team_bat_second", "score_bat_first", "score_bat_second", "winner"]].tail(10).to_string())
    print("\nTeam rolling features sample:")
    print(data["team_rolling"].tail(10).to_string())

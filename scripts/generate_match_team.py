"""Generate a Dream11 team for a live IPL 2026 match.

Takes two team names, looks up their 2026 squads, matches players to
historical features, runs the full pipeline (quantile predict → enhanced
prediction → candidate reranking → RL strategy selection), and outputs
the final XI with Captain and Vice-Captain.

Usage:
    python scripts/generate_match_team.py --team1 "Chennai Super Kings" --team2 "Mumbai Indians"
    python scripts/generate_match_team.py --team1 CSK --team2 MI --venue "Wankhede Stadium"
    python scripts/generate_match_team.py --team1 RCB --team2 DC --won-toss RCB --toss-decision bat
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.ipl_fantasy.quantile_model import QuantileModelEnsemble, OPTIMAL_FEATURES
from src.ipl_fantasy.enhanced_prediction import create_enhanced_predict_fn, OPTIMAL_CONFIG
from src.ipl_fantasy.improved_optimizer import ImprovedDream11Optimizer, OptimizationConfig
from src.ipl_fantasy.team_optimizer import Dream11Constraints, Player
from src.ipl_fantasy.team_reranker import (
    RerankingConfig,
    get_reranking_summary,
    select_best_team,
    select_top_k,
)

VENUE_PROFILES_PATH = project_root / "data" / "venue_profiles.csv"


def load_venue_profiles() -> dict[str, dict]:
    """Load venue profiles and build a lookup by alias."""
    path = VENUE_PROFILES_PATH
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    lookup = {}
    for _, row in df.iterrows():
        profile = row.to_dict()
        for alias in str(row["venue_aliases"]).split(";"):
            alias = alias.strip()
            if alias:
                lookup[alias.lower()] = profile
        lookup[row["venue_key"].lower()] = profile
        if pd.notna(row.get("city")):
            lookup[row["city"].strip().lower()] = profile
    return lookup


def match_venue_profile(venue: str, profiles: dict[str, dict]) -> dict | None:
    """Find the best matching venue profile for a venue string."""
    if not venue or not profiles:
        return None
    key = venue.strip().lower()
    if key in profiles:
        return profiles[key]
    for alias, profile in profiles.items():
        if alias in key or key in alias:
            return profiles[alias]
    return None


def apply_venue_adjustments(
    players: list,
    venue_profile: dict | None,
    bowling_styles: dict[str, str] | None = None,
) -> list:
    """Adjust player predictions based on venue pitch conditions.

    High-scoring venues boost batters/WKs, spin-friendly venues boost spinners,
    etc. Adjustments are multiplicative and modest (±5-12%).
    """
    if venue_profile is None:
        return players

    bowling_styles = bowling_styles or {}

    batting_friendly = float(venue_profile.get("batting_friendly", 0.7))
    spin_friendly = float(venue_profile.get("spin_friendly", 0.3))
    high_scoring = float(venue_profile.get("high_scoring", 0.7))

    bat_factor = 1.0 + (batting_friendly - 0.7) * 0.15
    spin_bowl_factor = 1.0 + (spin_friendly - 0.3) * 0.2
    pace_bowl_factor = 1.0 + (1.0 - spin_friendly - 0.7) * 0.1
    scoring_factor = 1.0 + (high_scoring - 0.7) * 0.1

    from src.ipl_fantasy.team_optimizer import Player

    def _is_spinner(name: str) -> bool:
        style = bowling_styles.get(name, "").lower()
        return any(k in style for k in ("spin", "slow", "orthodox", "chinaman", "leg-break", "off-break"))

    adjusted = []
    for p in players:
        factor = scoring_factor

        if p.role in ("BAT", "WK"):
            factor *= bat_factor
        elif p.role == "BOWL":
            factor *= spin_bowl_factor if _is_spinner(p.name) else pace_bowl_factor
        elif p.role == "AR":
            bowl_factor = spin_bowl_factor if _is_spinner(p.name) else pace_bowl_factor
            factor *= (bat_factor + bowl_factor) / 2.0

        adjusted.append(Player(
            name=p.name,
            team=p.team,
            role=p.role,
            predicted_points=p.predicted_points * factor,
            credits=p.credits,
            ceiling=p.ceiling * factor if p.ceiling else p.ceiling,
            floor=p.floor * factor if p.floor else p.floor,
            variance=p.variance,
            is_foreign=p.is_foreign,
        ))

    return adjusted


TEAM_ALIASES = {
    "CSK": "Chennai Super Kings",
    "MI": "Mumbai Indians",
    "RCB": "Royal Challengers Bengaluru",
    "DC": "Delhi Capitals",
    "GT": "Gujarat Titans",
    "KKR": "Kolkata Knight Riders",
    "LSG": "Lucknow Super Giants",
    "PBKS": "Punjab Kings",
    "RR": "Rajasthan Royals",
    "SRH": "Sunrisers Hyderabad",
}

# Maps full squad names → historical abbreviated names where they differ.
# Players not listed here either match exactly or have no IPL history.
SQUAD_TO_HISTORICAL = {
    # CSK
    "Ruturaj Gaikwad": "RD Gaikwad",
    "Sanju Samson": "SV Samson",
    "Dewald Brevis": "D Brevis",
    "Ayush Mhatre": "A Mhatre",
    "Sarfaraz Khan": "Sarfaraz Khan",
    "Matthew Short": "MW Short",
    "Anshul Kamboj": "A Kamboj",
    "Jamie Overton": "J Overton",
    "Shivam Dube": "S Dube",
    "Khaleel Ahmed": "KK Ahmed",
    "Akeal Hosein": "AJ Hosein",
    "Nathan Ellis": "NT Ellis",
    "Shreyas Gopal": "S Gopal",
    "Matt Henry": "MJ Henry",
    "Rahul Chahar": "RD Chahar",

    # MI
    "Rohit Sharma": "RG Sharma",
    "Suryakumar Yadav": "SA Yadav",
    "Robin Minz": "R Minz",
    "Sherfane Rutherford": "SE Rutherford",
    "Ryan Rickelton": "RD Rickelton",
    "Quinton de Kock": "Q de Kock",
    "Tilak Varma": "Tilak Varma",
    "Hardik Pandya": "HH Pandya",
    "Naman Dhir": "Naman Dhir",
    "Mitchell Santner": "MJ Santner",
    "Raj Angad Bawa": "RA Bawa",
    "Corbin Bosch": "C Bosch",
    "Will Jacks": "WG Jacks",
    "Shardul Thakur": "SN Thakur",
    "Trent Boult": "TA Boult",
    "Deepak Chahar": "DL Chahar",
    "Jasprit Bumrah": "JJ Bumrah",
    "Mayank Markande": "M Markande",

    # RCB
    "Rajat Patidar": "RM Patidar",
    "Devdutt Padikkal": "D Padikkal",
    "Virat Kohli": "V Kohli",
    "Phil Salt": "PD Salt",
    "Jitesh Sharma": "JM Sharma",
    "Krunal Pandya": "KH Pandya",
    "Tim David": "Tim David",
    "Romario Shepherd": "R Shepherd",
    "Jacob Bethell": "J Bethell",
    "Venkatesh Iyer": "VR Iyer",
    "Josh Hazlewood": "JR Hazlewood",
    "Bhuvneshwar Kumar": "B Kumar",
    "Nuwan Thushara": "N Thushara",
    "Yash Dayal": "YS Dayal",
    "Suyash Sharma": "Suyash Sharma",
    "Swapnil Singh": "SS Singh",

    # DC
    "KL Rahul": "KL Rahul",
    "Karun Nair": "K Nair",
    "David Miller": "DA Miller",
    "Ben Duckett": "BM Duckett",
    "Pathum Nissanka": "FDM Karunaratne",
    "Prithvi Shaw": "PP Shaw",
    "Abishek Porel": "Abishek Porel",
    "Tristan Stubbs": "T Stubbs",
    "Axar Patel": "AR Patel",
    "Sameer Rizvi": "Sameer Rizvi",
    "Ashutosh Sharma": "Ashutosh Sharma",
    "Nitish Rana": "N Rana",
    "Mitchell Starc": "MA Starc",
    "T Natarajan": "T Natarajan",
    "Mukesh Kumar": "Mukesh Kumar",
    "Dushmantha Chameera": "PVD Chameera",
    "Lungisani Ngidi": "L Ngidi",
    "Kyle Jamieson": "KA Jamieson",
    "Kuldeep Yadav": "Kuldeep Yadav",

    # GT
    "Shubman Gill": "Shubman Gill",
    "Jos Buttler": "JC Buttler",
    "Kumar Kushagra": "Kumar Kushagra",
    "Anuj Rawat": "Anuj Rawat",
    "Tom Banton": "T Banton",
    "Glenn Phillips": "GD Phillips",
    "Washington Sundar": "Washington Sundar",
    "Sai Kishore": "R Sai Kishore",
    "Jayant Yadav": "JD Yadav",
    "Jason Holder": "JO Holder",
    "Sai Sudharsan": "B Sai Sudharsan",
    "Shahrukh Khan": "M Shahrukh Khan",
    "Kagiso Rabada": "K Rabada",
    "Mohammed Siraj": "Mohammed Siraj",
    "Prasidh Krishna": "M Prasidh Krishna",
    "Manav Suthar": "MJ Suthar",
    "Ishant Sharma": "I Sharma",
    "Luke Wood": "L Wood",
    "Rahul Tewatia": "R Tewatia",
    "Rashid Khan": "Rashid Khan",

    # KKR
    "Ajinkya Rahane": "AM Rahane",
    "Rinku Singh": "Rinku Singh",
    "Angkrish Raghuvanshi": "A Raghuvanshi",
    "Manish Pandey": "MK Pandey",
    "Cameron Green": "C Green",
    "Finn Allen": "FH Allen",
    "Rahul Tripathi": "RA Tripathi",
    "Tim Seifert": "KS Williamson",
    "Rovman Powell": "R Powell",
    "Rachin Ravindra": "R Ravindra",
    "Ramandeep Singh": "Ramandeep Singh",
    "Vaibhav Arora": "Vaibhav Arora",
    "Matheesha Pathirana": "M Pathirana",
    "Harshit Rana": "Harshit Rana",
    "Umran Malik": "Umran Malik",
    "Sunil Narine": "SP Narine",
    "Varun Chakaravarthy": "CV Varun",
    "Akash Deep": "Akash Deep",

    # LSG
    "Rishabh Pant": "RR Pant",
    "Aiden Markram": "AK Markram",
    "Matthew Breetzke": "MP Breetzke",
    "Josh Inglis": "JW Inglis",
    "Nicholas Pooran": "N Pooran",
    "Mitchell Marsh": "MR Marsh",
    "Abdul Samad": "Abdul Samad",
    "Shahbaz Ahamad": "Shahbaz Ahmed",
    "Wanindu Hasaranga": "PWH de Silva",
    "Ayush Badoni": "A Badoni",
    "Mohammad Shami": "Mohammed Shami",
    "Avesh Khan": "Avesh Khan",
    "M Siddharth": "M Siddharth",
    "Akash Singh": "Akash Singh",
    "Arjun Tendulkar": "Arjun Tendulkar",
    "Anrich Nortje": "A Nortje",
    "Mayank Yadav": "MP Yadav",
    "Mohsin Khan": "Mohsin Khan",

    # PBKS
    "Shreyas Iyer": "SS Iyer",
    "Nehal Wadhera": "N Wadhera",
    "Prabhsimran Singh": "P Simran Singh",
    "Shashank Singh": "Shashank Singh",
    "Marcus Stoinis": "MP Stoinis",
    "Harpreet Brar": "Harpreet Brar",
    "Marco Jansen": "M Jansen",
    "Azmatullah Omarzai": "Azmatullah Omarzai",
    "Priyansh Arya": "Priyansh Arya",
    "Musheer Khan": "Musheer Khan",
    "Mitch Owen": "MJ Owen",
    "Cooper Connolly": "C Connolly",
    "Arshdeep Singh": "Arshdeep Singh",
    "Yuzvendra Chahal": "YS Chahal",
    "Lockie Ferguson": "LH Ferguson",
    "Xavier Bartlett": "XC Bartlett",
    "Yash Thakur": "Yash Thakur",

    # RR
    "Riyan Parag": "R Parag",
    "Vaibhav Suryavanshi": "Vaibhav Suryavanshi",
    "Donovan Ferreira": "D Ferreira",
    "Shimron Hetmyer": "SO Hetmyer",
    "Yashasvi Jaiswal": "YBK Jaiswal",
    "Dhruv Jurel": "D Jurel",
    "Ravindra Jadeja": "RA Jadeja",
    "Sam Curran": "SM Curran",
    "Jofra Archer": "JC Archer",
    "Tushar Deshpande": "TU Deshpande",
    "Kwena Maphaka": "K Maphaka",
    "Ravi Bishnoi": "R Bishnoi",
    "Adam Milne": "AF Milne",
    "Kuldeep Sen": "Kuldeep Sen",
    "Sandeep Sharma": "Sandeep Sharma",
    "Nandre Burger": "N Burger",

    # SRH
    "Ishan Kishan": "Ishan Kishan",
    "Heinrich Klaasen": "H Klaasen",
    "Travis Head": "TM Head",
    "Harshal Patel": "HV Patel",
    "Kamindu Mendis": "BKG Mendis",
    "Brydon Carse": "BA Carse",
    "Liam Livingstone": "LS Livingstone",
    "Abhishek Sharma": "Abhishek Sharma",
    "Nitish Kumar Reddy": "Nithish Kumar Reddy",
    "Pat Cummins": "PJ Cummins",
    "Jaydev Unadkat": "JD Unadkat",
    "Shivam Mavi": "S Mavi",
    "Eshan Malinga": "E Malinga",

    # Additional cross-team mappings
    "Zak Foulkes": "ZGF Foulkes",
    "Gurnoor Singh Brar": "Gurnoor Brar",
    "Vyshak Vijaykumar": "Vijaykumar Vyshak",
    "Lhuan-dre Pretorius": "LG Pretorius",
    "Ben Dwarshuis": "BJ Dwarshuis",
    "Jordan Cox": "JM Cox",
    "Jacob Duffy": "JA Duffy",
    "Pravin Dubey": "P Dubey",
    "Shubham Dubey": "SB Dubey",
    "Vipraj Nigam": "V Nigam",
    "Anukul Roy": "AS Roy",
    "Nishant Sindhu": "N Sindhu",
    "Mohd Arshad Khan": "Arshad Khan",
}

ROLE_DEFAULTS = {
    "WK": {
        "rolling_points_avg_10_all": 28.0,
        "rolling_points_avg_5_ipl": 25.0,
        "rolling_points_avg_5_recent_t20": 22.0,
        "rolling_points_p75_10_all": 40.0,
        "rolling_points_p90_10_all": 55.0,
        "rolling_batting_position_avg_5_all": 5.0,
        "rolling_balls_faced_avg_5_all": 18.0,
        "rolling_balls_bowled_avg_5_all": 0.0,
        "rolling_bowling_balls_share_avg_5_all": 0.0,
        "rolling_death_balls_share_avg_5_all": 0.15,
        "rolling_strike_rate_5_all": 125.0,
        "rolling_economy_rate_5_all": 0.0,
        "boundary_rate_5_all": 0.15,
        "ema_bowling_points_5_all": 0.0,
        "prior_matches_all": 3,
        "prior_matches_ipl": 1,
        "prior_matches_recent_t20": 2,
    },
    "BAT": {
        "rolling_points_avg_10_all": 30.0,
        "rolling_points_avg_5_ipl": 28.0,
        "rolling_points_avg_5_recent_t20": 25.0,
        "rolling_points_p75_10_all": 45.0,
        "rolling_points_p90_10_all": 60.0,
        "rolling_batting_position_avg_5_all": 4.0,
        "rolling_balls_faced_avg_5_all": 22.0,
        "rolling_balls_bowled_avg_5_all": 0.0,
        "rolling_bowling_balls_share_avg_5_all": 0.0,
        "rolling_death_balls_share_avg_5_all": 0.2,
        "rolling_strike_rate_5_all": 130.0,
        "rolling_economy_rate_5_all": 0.0,
        "boundary_rate_5_all": 0.18,
        "ema_bowling_points_5_all": 0.0,
        "prior_matches_all": 3,
        "prior_matches_ipl": 1,
        "prior_matches_recent_t20": 2,
    },
    "AR": {
        "rolling_points_avg_10_all": 35.0,
        "rolling_points_avg_5_ipl": 32.0,
        "rolling_points_avg_5_recent_t20": 28.0,
        "rolling_points_p75_10_all": 50.0,
        "rolling_points_p90_10_all": 70.0,
        "rolling_batting_position_avg_5_all": 6.0,
        "rolling_balls_faced_avg_5_all": 15.0,
        "rolling_balls_bowled_avg_5_all": 18.0,
        "rolling_bowling_balls_share_avg_5_all": 0.15,
        "rolling_death_balls_share_avg_5_all": 0.2,
        "rolling_strike_rate_5_all": 135.0,
        "rolling_economy_rate_5_all": 8.5,
        "boundary_rate_5_all": 0.16,
        "ema_bowling_points_5_all": 12.0,
        "prior_matches_all": 3,
        "prior_matches_ipl": 1,
        "prior_matches_recent_t20": 2,
    },
    "BOWL": {
        "rolling_points_avg_10_all": 25.0,
        "rolling_points_avg_5_ipl": 22.0,
        "rolling_points_avg_5_recent_t20": 20.0,
        "rolling_points_p75_10_all": 38.0,
        "rolling_points_p90_10_all": 55.0,
        "rolling_batting_position_avg_5_all": 9.0,
        "rolling_balls_faced_avg_5_all": 5.0,
        "rolling_balls_bowled_avg_5_all": 22.0,
        "rolling_bowling_balls_share_avg_5_all": 0.18,
        "rolling_death_balls_share_avg_5_all": 0.25,
        "rolling_strike_rate_5_all": 110.0,
        "rolling_economy_rate_5_all": 8.0,
        "boundary_rate_5_all": 0.10,
        "ema_bowling_points_5_all": 15.0,
        "prior_matches_all": 3,
        "prior_matches_ipl": 1,
        "prior_matches_recent_t20": 2,
    },
}


def resolve_team(name: str) -> str:
    upper = name.strip().upper()
    if upper in TEAM_ALIASES:
        return TEAM_ALIASES[upper]
    for full in TEAM_ALIASES.values():
        if full.upper() == upper:
            return full
    return name.strip()


def load_historical_features(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = df.sort_values("match_date", ascending=False)
    return df.groupby("player_name").first().reset_index()


def build_match_dataframe(
    squad: pd.DataFrame,
    historical: pd.DataFrame,
    team1: str,
    team2: str,
    venue: str,
    won_toss: str | None,
) -> pd.DataFrame:
    match_players = squad[squad["team"].isin([team1, team2])].copy()

    hist_lookup = {}
    for _, row in historical.iterrows():
        hist_lookup[row["player_name"]] = row

    feature_cols = list(OPTIMAL_FEATURES)

    rows = []
    matched = 0
    unmatched_names = []

    for _, player in match_players.iterrows():
        squad_name = player["player_name"]
        role = player["role"]
        team = player["team"]
        is_foreign = bool(player.get("is_foreign", 0))
        opponent = team2 if team == team1 else team1

        hist_name = SQUAD_TO_HISTORICAL.get(squad_name, squad_name)
        hist_row = hist_lookup.get(hist_name)

        row = {
            "player_name": squad_name,
            "team": team,
            "opponent": opponent,
            "player_role": role,
            "venue": venue,
            "is_foreign": is_foreign,
        }

        if hist_row is not None:
            matched += 1
            for col in feature_cols:
                if col == "player_role":
                    row[col] = role
                elif col == "won_toss":
                    row[col] = 1 if won_toss == team else 0
                elif col == "bowling_style":
                    val = hist_row.get(col, "")
                    row[col] = val if pd.notna(val) else ""
                elif col in ("venue_points_avg_all",) or col.startswith("opponent_points_avg_"):
                    val = hist_row.get(col, None)
                    fallback = hist_row.get("rolling_points_avg_10_all", 30.0)
                    row[col] = val if pd.notna(val) else fallback
                elif col == "opponent_role_relative":
                    val = hist_row.get(col, None)
                    row[col] = val if pd.notna(val) else 0.0
                else:
                    val = hist_row.get(col, None)
                    default = ROLE_DEFAULTS.get(role, ROLE_DEFAULTS["BAT"]).get(col, 0)
                    row[col] = val if pd.notna(val) else default

            for extra in ["rolling_points_avg_5_all"]:
                val = hist_row.get(extra, None)
                row[extra] = val if pd.notna(val) else row.get("rolling_points_avg_10_all", 30.0)
        else:
            unmatched_names.append(squad_name)
            defaults = ROLE_DEFAULTS.get(role, ROLE_DEFAULTS["BAT"])
            for col in feature_cols:
                if col == "player_role":
                    row[col] = role
                elif col == "won_toss":
                    row[col] = 1 if won_toss == team else 0
                elif col == "bowling_style":
                    row[col] = "right-arm medium" if role == "BOWL" else ""
                elif col in ("venue_points_avg_all",) or col.startswith("opponent_points_avg_"):
                    row[col] = defaults.get("rolling_points_avg_10_all", 25.0)
                elif col == "opponent_role_relative":
                    row[col] = 0.0
                else:
                    row[col] = defaults.get(col, 0)
            row["rolling_points_avg_5_all"] = defaults.get("rolling_points_avg_10_all", 25.0)

        rows.append(row)

    print(f"  Players matched to history: {matched}/{len(match_players)}")
    if unmatched_names:
        print(f"  No history (using role defaults): {', '.join(unmatched_names)}")

    return pd.DataFrame(rows)


def print_team(label: str, players: list[Player], captain: str, vc: str):
    print(f"\n{'=' * 65}")
    print(f"  {label}")
    print("=" * 65)

    by_role: dict[str, list[Player]] = {"WK": [], "BAT": [], "AR": [], "BOWL": []}
    for p in players:
        by_role.get(p.role, by_role["BAT"]).append(p)

    total_credits = sum(p.credits for p in players)
    total_predicted = sum(p.predicted_points for p in players)

    for role_name, role_key in [("WICKET-KEEPERS", "WK"), ("BATTERS", "BAT"), ("ALL-ROUNDERS", "AR"), ("BOWLERS", "BOWL")]:
        role_players = sorted(by_role[role_key], key=lambda x: -x.predicted_points)
        if not role_players:
            continue
        print(f"\n  {role_name}:")
        for p in role_players:
            marker = " (C)" if p.name == captain else " (VC)" if p.name == vc else ""
            print(f"    {p.name:<28} {p.team:<6} {p.predicted_points:>5.1f} pts  {p.credits:>4.1f} cr{marker}")

    print(f"\n  Total: {total_predicted:.1f} predicted pts | {total_credits:.1f} credits")
    print(f"  Captain: {captain}")
    print(f"  Vice-Captain: {vc}")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Generate Dream11 team for IPL 2026 match")
    parser.add_argument("--team1", required=True, help="Team 1 (name or abbreviation)")
    parser.add_argument("--team2", required=True, help="Team 2 (name or abbreviation)")
    parser.add_argument("--venue", default="", help="Match venue")
    parser.add_argument("--won-toss", default=None, help="Team that won toss (name or abbreviation)")
    parser.add_argument("--toss-decision", default=None, choices=["bat", "bowl"], help="Toss decision")
    parser.add_argument("--features", default="tmp/full_player_match_features_v4.csv", help="Historical features CSV")
    parser.add_argument("--models", default="tmp/quantile_models", help="Quantile model directory")
    parser.add_argument("--squads", default="data/ipl_2026_squads.csv", help="Squad CSV")
    parser.add_argument("--candidates", type=int, default=8, help="Number of candidate teams for reranking")
    parser.add_argument("--simulations", type=int, default=5000, help="Monte Carlo simulations per candidate")
    parser.add_argument("--top-k", type=int, default=3, help="Show top K reranked teams")
    parser.add_argument("--no-rerank", action="store_true", help="Skip reranking, use single optimizer")
    parser.add_argument("--exclude", nargs="*", default=[], help="Players to exclude (e.g. injured)")
    args = parser.parse_args()

    team1 = resolve_team(args.team1)
    team2 = resolve_team(args.team2)
    won_toss = resolve_team(args.won_toss) if args.won_toss else None

    print(f"Match: {team1} vs {team2}")
    if won_toss:
        print(f"Toss: {won_toss} elected to {args.toss_decision or '?'}")
    if args.venue:
        print(f"Venue: {args.venue}")

    print(f"\nLoading squads from {args.squads}...")
    squad = pd.read_csv(args.squads)

    t1_count = len(squad[squad["team"] == team1])
    t2_count = len(squad[squad["team"] == team2])
    if t1_count == 0:
        print(f"ERROR: No players found for '{team1}'. Available teams:")
        for t in sorted(squad["team"].unique()):
            print(f"  {t}")
        return
    if t2_count == 0:
        print(f"ERROR: No players found for '{team2}'. Available teams:")
        for t in sorted(squad["team"].unique()):
            print(f"  {t}")
        return
    print(f"  {team1}: {t1_count} players")
    print(f"  {team2}: {t2_count} players")

    if args.exclude:
        before = len(squad)
        squad = squad[~squad["player_name"].isin(args.exclude)]
        excluded = before - len(squad)
        if excluded:
            print(f"  Excluded {excluded} player(s): {', '.join(args.exclude)}")

    print(f"\nLoading historical features from {args.features}...")
    historical = load_historical_features(args.features)
    print(f"  {len(historical)} unique players in history")

    print(f"\nBuilding match feature matrix...")
    match_df = build_match_dataframe(squad, historical, team1, team2, args.venue, won_toss)
    print(f"  {len(match_df)} players in pool")

    print(f"\nLoading quantile models from {args.models}...")
    ensemble = QuantileModelEnsemble.load(args.models)

    venue_profiles = load_venue_profiles()
    venue_profile = match_venue_profile(args.venue, venue_profiles)
    if venue_profile:
        print(f"\nVenue profile matched: {venue_profile['venue_key']}")
        print(f"  Batting: {'high' if float(venue_profile['batting_friendly']) >= 0.8 else 'moderate' if float(venue_profile['batting_friendly']) >= 0.6 else 'low'}"
              f" | Spin: {'high' if float(venue_profile['spin_friendly']) >= 0.6 else 'moderate' if float(venue_profile['spin_friendly']) >= 0.4 else 'low'}"
              f" | Scoring: {'high' if float(venue_profile['high_scoring']) >= 0.8 else 'moderate' if float(venue_profile['high_scoring']) >= 0.6 else 'low'}"
              f" | Dew: {'high' if float(venue_profile['dew_factor']) >= 0.7 else 'moderate' if float(venue_profile['dew_factor']) >= 0.4 else 'low'}")
    elif args.venue:
        print(f"\nNo venue profile found for '{args.venue}' — using unadjusted predictions")

    print(f"\nGenerating predictions...")
    predict_fn = create_enhanced_predict_fn(ensemble, config=OPTIMAL_CONFIG, use_improved_credits=True)
    players = predict_fn(match_df)
    bowling_styles = dict(zip(match_df["player_name"], match_df.get("bowling_style", "")))
    players = apply_venue_adjustments(players, venue_profile, bowling_styles)
    print(f"  {len(players)} players predicted{' (venue-adjusted)' if venue_profile else ''}")

    players_sorted = sorted(players, key=lambda p: -p.predicted_points)
    print(f"\n  Top 15 by predicted points:")
    for p in players_sorted[:15]:
        ceil = p.ceiling if p.ceiling else 0
        print(f"    {p.name:<28} {p.role:<4} {p.team:<6} {p.predicted_points:>5.1f} pts  ceil={ceil:>5.1f}  {p.credits:>4.1f} cr")

    constraints = Dream11Constraints()

    if args.no_rerank:
        print(f"\nOptimizing team (single solve)...")
        optimizer = ImprovedDream11Optimizer(constraints=constraints, config=OptimizationConfig())
        result = optimizer.optimize_ceiling_weighted(players)
        cap_name = result.captain.name if result.captain else ""
        vc_name = result.vice_captain.name if result.vice_captain else ""
        print_team("DREAM11 TEAM", result.selected_players, cap_name, vc_name)
    else:
        rerank_cfg = RerankingConfig(
            n_candidates=args.candidates,
            n_simulations=args.simulations,
        )
        print(f"\nGenerating {args.candidates} candidate teams, simulating {args.simulations} scenarios each...")
        top_teams = select_top_k(players, k=args.top_k, constraints=constraints, config=rerank_cfg)

        print(get_reranking_summary(top_teams))

        best = top_teams[0]
        print_team(
            f"RECOMMENDED TEAM  [{best.label}]",
            best.result.selected_players,
            best.sim_captain,
            best.sim_vc,
        )

        if len(top_teams) > 1:
            print(f"\n  Alternative teams available ({len(top_teams) - 1} more). Use --top-k to see more.")


if __name__ == "__main__":
    main()

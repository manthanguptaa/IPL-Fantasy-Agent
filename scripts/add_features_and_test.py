#!/usr/bin/env python3
"""Add new features and test their impact on model performance.

Tests each feature addition individually to measure improvement.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def add_opponent_weakness_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add opponent team weakness features.

    This is different from opponent_points_avg_all (player vs opponent).
    This measures how many points the opponent CONCEDES to all players.

    E.g., if MI has weak bowling, all batsmen should score more vs MI.
    """
    print("Adding opponent weakness features...")

    # Sort by date for proper rolling calculation
    df = df.sort_values(['match_date', 'match_id']).copy()

    # Calculate points conceded by each team in each match
    # Group by match and opponent to get total points scored AGAINST each team
    match_points_against = df.groupby(['match_id', 'opponent']).agg({
        'dream11_points_total': 'sum',
        'batting_points': 'sum',
        'bowling_points': 'sum',
    }).reset_index()
    match_points_against.columns = [
        'match_id', 'team',
        'points_conceded', 'batting_points_conceded', 'bowling_points_conceded'
    ]

    # Add match date for sorting
    match_dates = df[['match_id', 'match_date']].drop_duplicates()
    match_points_against = match_points_against.merge(match_dates, on='match_id')
    match_points_against = match_points_against.sort_values('match_date')

    # Calculate rolling average of points conceded by each team
    team_weakness = []
    for team in match_points_against['team'].unique():
        team_data = match_points_against[match_points_against['team'] == team].copy()
        team_data['opponent_weakness_avg_5'] = (
            team_data['points_conceded']
            .shift(1)  # Don't include current match
            .rolling(window=5, min_periods=1)
            .mean()
        )
        team_data['opponent_batting_weakness_avg_5'] = (
            team_data['batting_points_conceded']
            .shift(1)
            .rolling(window=5, min_periods=1)
            .mean()
        )
        team_data['opponent_bowling_weakness_avg_5'] = (
            team_data['bowling_points_conceded']
            .shift(1)
            .rolling(window=5, min_periods=1)
            .mean()
        )
        team_weakness.append(team_data)

    team_weakness_df = pd.concat(team_weakness, ignore_index=True)
    team_weakness_df = team_weakness_df[['match_id', 'team',
                                          'opponent_weakness_avg_5',
                                          'opponent_batting_weakness_avg_5',
                                          'opponent_bowling_weakness_avg_5']]

    # Merge back - the 'team' in weakness_df is the OPPONENT of the player
    # So we merge on opponent
    df = df.merge(
        team_weakness_df.rename(columns={'team': 'opponent'}),
        on=['match_id', 'opponent'],
        how='left'
    )

    # Fill NaN with overall average
    for col in ['opponent_weakness_avg_5', 'opponent_batting_weakness_avg_5',
                'opponent_bowling_weakness_avg_5']:
        df[col] = df[col].fillna(df[col].mean())

    print(f"  Added: opponent_weakness_avg_5 (mean: {df['opponent_weakness_avg_5'].mean():.1f})")

    return df


def add_batting_first_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add batting first indicator.

    batting_first = 1 if team batted first, 0 otherwise
    """
    print("Adding batting first feature...")

    df = df.copy()

    # Determine which team batted first based on toss
    # If toss_decision == 'bat', toss_winner batted first
    # If toss_decision == 'field', the other team batted first

    def get_batting_first(row):
        if pd.isna(row['toss_winner']) or pd.isna(row['toss_decision']):
            return 0.5  # Unknown

        toss_winner = row['toss_winner']
        toss_decision = row['toss_decision'].lower() if isinstance(row['toss_decision'], str) else ''
        player_team = row['team']

        if 'bat' in toss_decision:
            # Toss winner chose to bat
            return 1.0 if player_team == toss_winner else 0.0
        elif 'field' in toss_decision or 'bowl' in toss_decision:
            # Toss winner chose to field
            return 0.0 if player_team == toss_winner else 1.0
        else:
            return 0.5  # Unknown decision

    df['batting_first'] = df.apply(get_batting_first, axis=1)

    print(f"  Added: batting_first (mean: {df['batting_first'].mean():.2f})")

    return df


def add_toss_advantage_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add toss advantage indicator.

    won_toss = 1 if player's team won the toss, 0 otherwise
    """
    print("Adding toss advantage feature...")

    df = df.copy()

    def get_toss_advantage(row):
        if pd.isna(row['toss_winner']):
            return 0.5  # Unknown
        return 1.0 if row['team'] == row['toss_winner'] else 0.0

    df['won_toss'] = df.apply(get_toss_advantage, axis=1)

    print(f"  Added: won_toss (mean: {df['won_toss'].mean():.2f})")

    return df


def test_feature_impact(
    features_path: str,
    feature_name: str,
    add_feature_fn,
    n_matches: int = 50,
):
    """
    Test the impact of adding a feature.

    Trains model with and without the feature and compares backtest results.
    """
    from src.ipl_fantasy.quantile_model import QuantileModelEnsemble, OPTIMAL_FEATURES
    from src.ipl_fantasy.backtesting import Backtester
    from src.ipl_fantasy.team_optimizer import Player
    from src.ipl_fantasy.enhanced_prediction import create_enhanced_predict_fn, OPTIMAL_CONFIG
    from src.ipl_fantasy.credit_estimation import estimate_credits_from_history

    print(f"\n{'='*60}")
    print(f"Testing feature: {feature_name}")
    print(f"{'='*60}")

    # Load data
    print(f"\nLoading data from {features_path}...")
    df = pd.read_csv(features_path, low_memory=False)

    # Add the new feature
    df = add_feature_fn(df)

    # Filter to IPL
    df = df[df['competition'] == 'Indian Premier League'].copy()

    # Sort and split
    df = df.sort_values(['match_date', 'match_id', 'player_name']).reset_index(drop=True)

    # Get unique matches for backtest
    matches = df.groupby('match_id').first().reset_index()
    matches = matches.sort_values('match_date', ascending=False)
    test_match_ids = set(matches.head(n_matches)['match_id'])

    # Split into train and test
    test_df = df[df['match_id'].isin(test_match_ids)].copy()
    train_df = df[~df['match_id'].isin(test_match_ids)].copy()

    print(f"  Train matches: {train_df['match_id'].nunique()}")
    print(f"  Test matches: {test_df['match_id'].nunique()}")

    # Determine new features to add
    if feature_name == 'opponent_weakness':
        new_features = ['opponent_weakness_avg_5', 'opponent_batting_weakness_avg_5',
                       'opponent_bowling_weakness_avg_5']
    elif feature_name == 'batting_first':
        new_features = ['batting_first']
    elif feature_name == 'won_toss':
        new_features = ['won_toss']
    else:
        new_features = []

    # Check which features exist in the data
    available_new_features = [f for f in new_features if f in df.columns]
    print(f"  New features available: {available_new_features}")

    # Train model WITH new features
    extended_features = OPTIMAL_FEATURES + available_new_features

    print(f"\nTraining model with {len(extended_features)} features...")
    ensemble = QuantileModelEnsemble(features=extended_features)
    metrics = ensemble.fit(train_df, test_df.head(1000))  # Small validation set

    print(f"  RMSE: {metrics.get('mean_rmse', 'N/A')}")

    # Create prediction function with new features
    def create_predict_fn_with_features(ensemble, config):
        def predict_fn(match_df):
            predictions = ensemble.predict(match_df)

            players = []
            for i, pred in enumerate(predictions):
                role = pred.role if pred.role in ("WK", "BAT", "AR", "BOWL") else "BAT"
                ceiling_weight = config.role_ceiling_weights.get(role, 0.3)

                weighted_prediction = (
                    (1 - ceiling_weight) * pred.expected +
                    ceiling_weight * pred.q90
                )

                captain_value = (
                    (1 - config.captain_ceiling_weight) * pred.expected +
                    config.captain_ceiling_weight * pred.q90
                )

                # Get credits from match_df
                row = match_df.iloc[i] if i < len(match_df) else {}
                avg_all = row.get("rolling_points_avg_10_all", 30.0)
                avg_recent = row.get("rolling_points_avg_5_all", None)

                credits = estimate_credits_from_history(
                    player_name=pred.player_name,
                    player_role=role,
                    avg_points_all=avg_all if pd.notna(avg_all) else 30.0,
                    avg_points_recent=avg_recent if pd.notna(avg_recent) else None,
                )

                player = Player(
                    name=pred.player_name,
                    team=pred.team,
                    role=role,
                    predicted_points=weighted_prediction,
                    credits=credits,
                    ceiling=captain_value,
                    floor=pred.q10,
                    variance=pred.variance,
                )
                players.append(player)

            return players

        return predict_fn

    # Run backtest
    print(f"\nRunning backtest on {n_matches} matches...")
    backtester = Backtester()
    predict_fn = create_predict_fn_with_features(ensemble, OPTIMAL_CONFIG)

    summary = backtester.backtest_multiple(
        test_df,
        predict_fn,
        n_matches=n_matches,
        competition_filter=None,  # Already filtered
    )

    print(f"\nResults with {feature_name}:")
    print(f"  Mean Selected Score: {summary.mean_selected_score:.1f}")
    print(f"  Mean Oracle Score: {summary.mean_oracle_score:.1f}")
    print(f"  Mean Total Regret: {summary.mean_total_regret:.1f}")
    print(f"  Mean Team Regret: {summary.mean_team_regret:.1f}")
    print(f"  Mean Captain Regret: {summary.mean_captain_regret:.1f}")
    print(f"  Player Overlap: {summary.mean_overlap_pct:.1f}%")
    print(f"  Captain Accuracy: {summary.top_captain_rate:.1f}%")

    return {
        'feature': feature_name,
        'selected_score': summary.mean_selected_score,
        'oracle_score': summary.mean_oracle_score,
        'total_regret': summary.mean_total_regret,
        'team_regret': summary.mean_team_regret,
        'captain_regret': summary.mean_captain_regret,
        'overlap_pct': summary.mean_overlap_pct,
        'captain_rate': summary.top_captain_rate,
    }


def run_baseline(features_path: str, n_matches: int = 50):
    """Run baseline backtest without new features."""
    from src.ipl_fantasy.quantile_model import QuantileModelEnsemble, OPTIMAL_FEATURES
    from src.ipl_fantasy.backtesting import Backtester
    from src.ipl_fantasy.team_optimizer import Player
    from src.ipl_fantasy.enhanced_prediction import OPTIMAL_CONFIG
    from src.ipl_fantasy.credit_estimation import estimate_credits_from_history

    print(f"\n{'='*60}")
    print("BASELINE (no new features)")
    print(f"{'='*60}")

    # Load data
    print(f"\nLoading data from {features_path}...")
    df = pd.read_csv(features_path, low_memory=False)

    # Filter to IPL
    df = df[df['competition'] == 'Indian Premier League'].copy()

    # Sort and split
    df = df.sort_values(['match_date', 'match_id', 'player_name']).reset_index(drop=True)

    # Get unique matches for backtest
    matches = df.groupby('match_id').first().reset_index()
    matches = matches.sort_values('match_date', ascending=False)
    test_match_ids = set(matches.head(n_matches)['match_id'])

    # Split into train and test
    test_df = df[df['match_id'].isin(test_match_ids)].copy()
    train_df = df[~df['match_id'].isin(test_match_ids)].copy()

    print(f"  Train matches: {train_df['match_id'].nunique()}")
    print(f"  Test matches: {test_df['match_id'].nunique()}")

    # Train baseline model
    print(f"\nTraining baseline model with {len(OPTIMAL_FEATURES)} features...")
    ensemble = QuantileModelEnsemble(features=OPTIMAL_FEATURES)
    metrics = ensemble.fit(train_df, test_df.head(1000))

    print(f"  RMSE: {metrics.get('mean_rmse', 'N/A')}")

    # Create prediction function
    def create_predict_fn(ensemble, config):
        def predict_fn(match_df):
            predictions = ensemble.predict(match_df)

            players = []
            for i, pred in enumerate(predictions):
                role = pred.role if pred.role in ("WK", "BAT", "AR", "BOWL") else "BAT"
                ceiling_weight = config.role_ceiling_weights.get(role, 0.3)

                weighted_prediction = (
                    (1 - ceiling_weight) * pred.expected +
                    ceiling_weight * pred.q90
                )

                captain_value = (
                    (1 - config.captain_ceiling_weight) * pred.expected +
                    config.captain_ceiling_weight * pred.q90
                )

                row = match_df.iloc[i] if i < len(match_df) else {}
                avg_all = row.get("rolling_points_avg_10_all", 30.0)
                avg_recent = row.get("rolling_points_avg_5_all", None)

                credits = estimate_credits_from_history(
                    player_name=pred.player_name,
                    player_role=role,
                    avg_points_all=avg_all if pd.notna(avg_all) else 30.0,
                    avg_points_recent=avg_recent if pd.notna(avg_recent) else None,
                )

                player = Player(
                    name=pred.player_name,
                    team=pred.team,
                    role=role,
                    predicted_points=weighted_prediction,
                    credits=credits,
                    ceiling=captain_value,
                    floor=pred.q10,
                    variance=pred.variance,
                )
                players.append(player)

            return players

        return predict_fn

    # Run backtest
    print(f"\nRunning backtest on {n_matches} matches...")
    backtester = Backtester()
    predict_fn = create_predict_fn(ensemble, OPTIMAL_CONFIG)

    summary = backtester.backtest_multiple(
        test_df,
        predict_fn,
        n_matches=n_matches,
        competition_filter=None,
    )

    print(f"\nBaseline Results:")
    print(f"  Mean Selected Score: {summary.mean_selected_score:.1f}")
    print(f"  Mean Oracle Score: {summary.mean_oracle_score:.1f}")
    print(f"  Mean Total Regret: {summary.mean_total_regret:.1f}")
    print(f"  Mean Team Regret: {summary.mean_team_regret:.1f}")
    print(f"  Mean Captain Regret: {summary.mean_captain_regret:.1f}")
    print(f"  Player Overlap: {summary.mean_overlap_pct:.1f}%")
    print(f"  Captain Accuracy: {summary.top_captain_rate:.1f}%")

    return {
        'feature': 'BASELINE',
        'selected_score': summary.mean_selected_score,
        'oracle_score': summary.mean_oracle_score,
        'total_regret': summary.mean_total_regret,
        'team_regret': summary.mean_team_regret,
        'captain_regret': summary.mean_captain_regret,
        'overlap_pct': summary.mean_overlap_pct,
        'captain_rate': summary.top_captain_rate,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test feature additions")
    parser.add_argument(
        "--features",
        default="tmp/full_player_match_features_v3.csv",
        help="Path to features CSV",
    )
    parser.add_argument(
        "--n-matches",
        type=int,
        default=50,
        help="Number of matches to test on",
    )
    args = parser.parse_args()

    results = []

    # Run baseline first
    baseline = run_baseline(args.features, args.n_matches)
    results.append(baseline)

    # Test each feature
    features_to_test = [
        ('opponent_weakness', add_opponent_weakness_features),
        ('batting_first', add_batting_first_feature),
        ('won_toss', add_toss_advantage_feature),
    ]

    for feature_name, add_fn in features_to_test:
        result = test_feature_impact(
            args.features,
            feature_name,
            add_fn,
            args.n_matches,
        )
        results.append(result)

    # Summary comparison
    print(f"\n{'='*80}")
    print("FEATURE COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"{'Feature':<20} {'Score':>10} {'Regret':>10} {'Team-R':>10} {'Cap-R':>8} {'Overlap':>10} {'CapRate':>8}")
    print("-" * 80)

    for r in results:
        print(f"{r['feature']:<20} {r['selected_score']:>10.1f} {r['total_regret']:>10.1f} "
              f"{r['team_regret']:>10.1f} {r['captain_regret']:>8.1f} "
              f"{r['overlap_pct']:>9.1f}% {r['captain_rate']:>7.1f}%")

    # Show improvements
    print(f"\n{'='*80}")
    print("IMPROVEMENT VS BASELINE")
    print(f"{'='*80}")

    baseline_regret = baseline['total_regret']
    for r in results[1:]:
        improvement = baseline_regret - r['total_regret']
        pct = improvement / baseline_regret * 100 if baseline_regret > 0 else 0
        sign = '+' if improvement > 0 else ''
        print(f"{r['feature']:<20} Regret change: {sign}{-improvement:.1f} pts ({sign}{-pct:.1f}%)")


if __name__ == "__main__":
    main()

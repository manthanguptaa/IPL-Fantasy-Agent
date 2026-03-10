import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class NormalizeCricketJsonTests(unittest.TestCase):
    def test_normalize_single_match_returns_player_rows(self):
        from src.ipl_fantasy.normalize_cricket_json import normalize_match_file

        sample_match = ROOT / "ipl_json" / "981019.json"
        rows = normalize_match_file(
            sample_match,
            source_dataset="ipl_json",
            player_roles={
                "DA Warner": {"role": "BAT"},
                "V Kohli": {"role": "BAT"},
                "B Kumar": {"role": "BOWL"},
                "CJ Jordan": {"role": "BOWL"},
            },
        )

        self.assertEqual(len(rows), 22)

        by_name = {row["player_name"]: row for row in rows}
        self.assertIn("DA Warner", by_name)
        self.assertIn("V Kohli", by_name)

        warner = by_name["DA Warner"]
        self.assertEqual(warner["match_id"], "981019")
        self.assertEqual(warner["competition"], "Indian Premier League")
        self.assertEqual(warner["source_dataset"], "ipl_json")
        self.assertEqual(warner["venue"], "M Chinnaswamy Stadium")
        self.assertEqual(warner["team"], "Sunrisers Hyderabad")
        self.assertEqual(warner["opponent"], "Royal Challengers Bangalore")
        self.assertEqual(warner["player_role"], "BAT")
        self.assertEqual(warner["playing_xi"], 1)
        self.assertEqual(warner["batting_position"], 1)
        self.assertEqual(warner["batting_order_bucket"], "opener")
        self.assertAlmostEqual(warner["batting_balls_share"], 0.3167, places=4)
        self.assertGreaterEqual(warner["runs"], 0)
        self.assertGreaterEqual(warner["dream11_points_total"], 0)

        kumar = by_name["B Kumar"]
        self.assertEqual(kumar["player_role"], "BOWL")
        self.assertAlmostEqual(kumar["bowling_balls_share"], 0.2, places=4)
        self.assertEqual(kumar["powerplay_balls"], 12)
        self.assertEqual(kumar["middle_balls"], 0)
        self.assertEqual(kumar["death_balls"], 12)

    def test_normalize_dataset_dir_aggregates_rows(self):
        from src.ipl_fantasy.normalize_cricket_json import normalize_dataset_dir

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_dir = ROOT / "ipl_json"
            for match_id in ("981019.json", "980901.json"):
                (tmp_path / match_id).write_text((source_dir / match_id).read_text())

            rows = normalize_dataset_dir(tmp_path)

        self.assertEqual(len(rows), 44)
        self.assertTrue(all(row["source_dataset"] == tmp_path.name for row in rows))
        self.assertEqual({row["match_id"] for row in rows}, {"981019", "980901"})

    def test_write_training_dataset_csv_writes_stable_headers(self):
        from src.ipl_fantasy.normalize_cricket_json import normalize_dataset_dir, write_training_dataset_csv

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_dir = ROOT / "ipl_json"
            (tmp_path / "981019.json").write_text((source_dir / "981019.json").read_text())

            rows = normalize_dataset_dir(tmp_path)
            output_path = tmp_path / "training.csv"
            write_training_dataset_csv(rows, output_path)

            with output_path.open(newline="") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames
                csv_rows = list(reader)

        self.assertIsNotNone(fieldnames)
        self.assertIn("match_id", fieldnames)
        self.assertIn("player_name", fieldnames)
        self.assertIn("player_role", fieldnames)
        self.assertIn("batting_position", fieldnames)
        self.assertIn("bowling_balls_share", fieldnames)
        self.assertIn("dream11_points_total", fieldnames)
        self.assertEqual(len(csv_rows), 22)

    def test_substitute_only_fielders_are_excluded_from_output(self):
        from src.ipl_fantasy.normalize_cricket_json import normalize_match_file

        sample_match = ROOT / "ipl_json" / "1082591.json"
        rows = normalize_match_file(sample_match, source_dataset="ipl_json")

        substitute_rows = [row for row in rows if not row["team"]]
        self.assertFalse(substitute_rows)
        self.assertNotIn("CJ Jordan", {row["player_name"] for row in rows})

    def test_cli_builds_training_csv(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "training.csv"
            script_path = ROOT / "scripts" / "build_training_dataset.py"

            result = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--dataset",
                    str(ROOT / "ipl_json"),
                    "--limit",
                    "1",
                    "--output",
                    str(output_path),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertTrue(output_path.exists())

    def test_normalize_match_file_can_filter_gender(self):
        from src.ipl_fantasy.normalize_cricket_json import normalize_match_file

        sample_female_match = ROOT / "t20s_json" / "1477599.json"
        rows = normalize_match_file(
            sample_female_match,
            source_dataset="t20s_json",
            gender_filter={"male"},
        )

        self.assertEqual(rows, [])


class FeatureEngineeringTests(unittest.TestCase):
    def test_build_feature_rows_uses_only_past_matches(self):
        from src.ipl_fantasy.build_features import build_feature_rows

        base_rows = [
            {
                "match_id": "1",
                "match_date": "2024-01-01",
                "competition": "Indian Premier League",
                "source_dataset": "ipl_json",
                "season": "2024",
                "team_type": "club",
                "match_type": "T20",
                "venue": "Venue A",
                "city": "City A",
                "team": "Mumbai Indians",
                "opponent": "Chennai Super Kings",
                "player_name": "Player A",
                "player_role": "BAT",
                "playing_xi": "1",
                "toss_winner": "Mumbai Indians",
                "toss_decision": "bat",
                "winner": "Mumbai Indians",
                "batting_position": "1",
                "batting_order_bucket": "opener",
                "runs": "40",
                "balls_faced": "25",
                "batting_balls_share": "0.25",
                "fours": "4",
                "sixes": "2",
                "duck": "0",
                "balls_bowled": "0",
                "overs_bowled": "0.0",
                "bowling_balls_share": "0.0",
                "powerplay_balls": "0",
                "middle_balls": "0",
                "death_balls": "0",
                "powerplay_balls_share": "0.0",
                "middle_balls_share": "0.0",
                "death_balls_share": "0.0",
                "maidens": "0",
                "runs_conceded": "0",
                "wickets": "0",
                "catches": "1",
                "stumpings": "0",
                "run_out_direct": "0",
                "run_out_assist": "0",
                "batting_points": "48",
                "bowling_points": "0",
                "fielding_points": "8",
                "other_points": "4",
                "dream11_points_total": "60",
            },
            {
                "match_id": "2",
                "match_date": "2024-01-10",
                "competition": "Indian Premier League",
                "source_dataset": "ipl_json",
                "season": "2024",
                "team_type": "club",
                "match_type": "T20",
                "venue": "Venue A",
                "city": "City A",
                "team": "Mumbai Indians",
                "opponent": "Royal Challengers Bangalore",
                "player_name": "Player A",
                "player_role": "BAT",
                "playing_xi": "1",
                "toss_winner": "Royal Challengers Bangalore",
                "toss_decision": "field",
                "winner": "Mumbai Indians",
                "batting_position": "2",
                "batting_order_bucket": "opener",
                "runs": "10",
                "balls_faced": "12",
                "batting_balls_share": "0.1",
                "fours": "1",
                "sixes": "0",
                "duck": "0",
                "balls_bowled": "0",
                "overs_bowled": "0.0",
                "bowling_balls_share": "0.0",
                "powerplay_balls": "0",
                "middle_balls": "0",
                "death_balls": "0",
                "powerplay_balls_share": "0.0",
                "middle_balls_share": "0.0",
                "death_balls_share": "0.0",
                "maidens": "0",
                "runs_conceded": "0",
                "wickets": "0",
                "catches": "0",
                "stumpings": "0",
                "run_out_direct": "0",
                "run_out_assist": "0",
                "batting_points": "11",
                "bowling_points": "0",
                "fielding_points": "0",
                "other_points": "2",
                "dream11_points_total": "13",
            },
        ]

        feature_rows = build_feature_rows(base_rows)

        self.assertEqual(len(feature_rows), 2)
        first, second = feature_rows
        self.assertEqual(first["prior_matches_all"], 0)
        self.assertEqual(first["rolling_points_avg_3_all"], 0.0)
        self.assertEqual(second["prior_matches_all"], 1)
        self.assertEqual(second["player_role"], "BAT")
        self.assertEqual(second["rolling_points_avg_3_all"], 60.0)
        self.assertEqual(second["rolling_runs_avg_3_all"], 40.0)
        self.assertEqual(second["rolling_points_avg_3_ipl"], 60.0)
        self.assertEqual(second["rolling_points_avg_10_all"], 60.0)
        self.assertEqual(second["rolling_points_std_5_all"], 0.0)
        self.assertEqual(second["rolling_runs_avg_5_all"], 40.0)
        self.assertEqual(second["rolling_batting_points_avg_5_all"], 48.0)
        self.assertEqual(second["rolling_fielding_points_avg_5_all"], 8.0)
        self.assertEqual(second["batting_match_rate_5_all"], 1.0)
        self.assertEqual(second["bowling_match_rate_5_all"], 0.0)
        self.assertEqual(second["rolling_strike_rate_5_all"], 160.0)
        self.assertEqual(second["rolling_economy_rate_5_all"], 0.0)
        self.assertEqual(second["prior_matches_at_venue"], 1)
        self.assertEqual(second["venue_points_avg_all"], 60.0)
        self.assertEqual(second["prior_matches_vs_opponent"], 0)
        self.assertEqual(second["opponent_points_avg_all"], 0.0)
        self.assertEqual(second["points_trend_3_vs_10_all"], 0.0)
        self.assertEqual(second["batting_position_known_rate_5_all"], 1.0)
        self.assertEqual(second["rolling_batting_position_avg_5_all"], 1.0)
        self.assertEqual(second["rolling_batting_balls_share_avg_5_all"], 0.25)
        self.assertEqual(second["rolling_bowling_balls_share_avg_5_all"], 0.0)
        self.assertEqual(second["rolling_powerplay_balls_share_avg_5_all"], 0.0)

    def test_write_feature_dataset_csv_writes_rows(self):
        from src.ipl_fantasy.build_features import build_feature_rows, write_feature_dataset_csv

        base_rows = [
            {
                "match_id": "1",
                "match_date": "2024-01-01",
                "competition": "Indian Premier League",
                "source_dataset": "ipl_json",
                "season": "2024",
                "team_type": "club",
                "match_type": "T20",
                "venue": "Venue A",
                "city": "City A",
                "team": "Mumbai Indians",
                "opponent": "Chennai Super Kings",
                "player_name": "Player A",
                "player_role": "BAT",
                "playing_xi": "1",
                "toss_winner": "Mumbai Indians",
                "toss_decision": "bat",
                "winner": "Mumbai Indians",
                "batting_position": "1",
                "batting_order_bucket": "opener",
                "runs": "40",
                "balls_faced": "25",
                "batting_balls_share": "0.25",
                "fours": "4",
                "sixes": "2",
                "duck": "0",
                "balls_bowled": "0",
                "overs_bowled": "0.0",
                "bowling_balls_share": "0.0",
                "powerplay_balls": "0",
                "middle_balls": "0",
                "death_balls": "0",
                "powerplay_balls_share": "0.0",
                "middle_balls_share": "0.0",
                "death_balls_share": "0.0",
                "maidens": "0",
                "runs_conceded": "0",
                "wickets": "0",
                "catches": "1",
                "stumpings": "0",
                "run_out_direct": "0",
                "run_out_assist": "0",
                "batting_points": "48",
                "bowling_points": "0",
                "fielding_points": "8",
                "other_points": "4",
                "dream11_points_total": "60",
            }
        ]

        feature_rows = build_feature_rows(base_rows)
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "features.csv"
            write_feature_dataset_csv(feature_rows, output_path)
            with output_path.open(newline="") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames
                rows = list(reader)

        self.assertIsNotNone(fieldnames)
        self.assertIn("prior_matches_all", fieldnames)
        self.assertIn("rolling_points_avg_3_all", fieldnames)
        self.assertIn("rolling_points_avg_10_all", fieldnames)
        self.assertIn("rolling_batting_points_avg_5_all", fieldnames)
        self.assertIn("venue_points_avg_all", fieldnames)
        self.assertIn("player_role", fieldnames)
        self.assertIn("rolling_batting_position_avg_5_all", fieldnames)
        self.assertIn("rolling_batting_balls_share_avg_5_all", fieldnames)
        self.assertEqual(len(rows), 1)


class DatasetCurationTests(unittest.TestCase):
    def test_filter_rows_for_ipl_model_removes_low_signal_t20i_rows(self):
        from src.ipl_fantasy.curate_training_rows import filter_rows_for_ipl_model

        rows = [
            {
                "source_dataset": "t20s_json",
                "competition": "Balkan Cup",
                "team": "Bulgaria",
                "opponent": "Romania",
                "player_name": "Player A",
            },
            {
                "source_dataset": "t20s_json",
                "competition": "England tour of India",
                "team": "India",
                "opponent": "England",
                "player_name": "Player B",
            },
            {
                "source_dataset": "ipl_json",
                "competition": "Indian Premier League",
                "team": "Mumbai Indians",
                "opponent": "Chennai Super Kings",
                "player_name": "Player C",
            },
        ]

        filtered = filter_rows_for_ipl_model(rows)

        names = {row["player_name"] for row in filtered}
        self.assertNotIn("Player A", names)
        self.assertIn("Player B", names)
        self.assertIn("Player C", names)


if __name__ == "__main__":
    unittest.main()

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sample_rows():
    return [
        {
            "match_id": "1",
            "match_date": "2023-01-01",
            "competition": "Indian Premier League",
            "source_dataset": "ipl_json",
            "season": "2023",
            "team_type": "club",
            "match_type": "T20",
            "venue": "Venue A",
            "city": "City A",
            "team": "Mumbai Indians",
            "opponent": "Chennai Super Kings",
            "player_name": "Player A",
            "playing_xi": "1",
            "toss_winner": "Mumbai Indians",
            "toss_decision": "bat",
            "winner": "Mumbai Indians",
            "runs": "20",
            "balls_faced": "10",
            "fours": "2",
            "sixes": "1",
            "duck": "0",
            "balls_bowled": "0",
            "overs_bowled": "0.0",
            "maidens": "0",
            "runs_conceded": "0",
            "wickets": "0",
            "catches": "0",
            "stumpings": "0",
            "run_out_direct": "0",
            "run_out_assist": "0",
            "batting_points": "24",
            "bowling_points": "0",
            "fielding_points": "0",
            "other_points": "4",
            "dream11_points_total": "28",
            "prior_matches_all": "0",
            "prior_matches_ipl": "0",
            "prior_matches_recent_t20": "0",
            "rolling_points_avg_3_all": "0.0",
            "rolling_points_avg_5_all": "0.0",
            "rolling_runs_avg_3_all": "0.0",
            "rolling_wickets_avg_3_all": "0.0",
            "rolling_points_avg_3_ipl": "0.0",
            "rolling_points_avg_3_recent_t20": "0.0",
        },
        {
            "match_id": "2",
            "match_date": "2023-01-10",
            "competition": "Indian Premier League",
            "source_dataset": "ipl_json",
            "season": "2023",
            "team_type": "club",
            "match_type": "T20",
            "venue": "Venue A",
            "city": "City A",
            "team": "Mumbai Indians",
            "opponent": "Royal Challengers Bangalore",
            "player_name": "Player A",
            "playing_xi": "1",
            "toss_winner": "Royal Challengers Bangalore",
            "toss_decision": "field",
            "winner": "Mumbai Indians",
            "runs": "45",
            "balls_faced": "24",
            "fours": "5",
            "sixes": "2",
            "duck": "0",
            "balls_bowled": "0",
            "overs_bowled": "0.0",
            "maidens": "0",
            "runs_conceded": "0",
            "wickets": "0",
            "catches": "1",
            "stumpings": "0",
            "run_out_direct": "0",
            "run_out_assist": "0",
            "batting_points": "54",
            "bowling_points": "0",
            "fielding_points": "8",
            "other_points": "4",
            "dream11_points_total": "66",
            "prior_matches_all": "1",
            "prior_matches_ipl": "1",
            "prior_matches_recent_t20": "0",
            "rolling_points_avg_3_all": "28.0",
            "rolling_points_avg_5_all": "28.0",
            "rolling_runs_avg_3_all": "20.0",
            "rolling_wickets_avg_3_all": "0.0",
            "rolling_points_avg_3_ipl": "28.0",
            "rolling_points_avg_3_recent_t20": "0.0",
        },
        {
            "match_id": "3",
            "match_date": "2024-04-01",
            "competition": "Indian Premier League",
            "source_dataset": "ipl_json",
            "season": "2024",
            "team_type": "club",
            "match_type": "T20",
            "venue": "Venue B",
            "city": "City B",
            "team": "Chennai Super Kings",
            "opponent": "Mumbai Indians",
            "player_name": "Player B",
            "playing_xi": "1",
            "toss_winner": "Chennai Super Kings",
            "toss_decision": "bat",
            "winner": "Chennai Super Kings",
            "runs": "15",
            "balls_faced": "12",
            "fours": "1",
            "sixes": "0",
            "duck": "0",
            "balls_bowled": "24",
            "overs_bowled": "4.0",
            "maidens": "0",
            "runs_conceded": "20",
            "wickets": "2",
            "catches": "0",
            "stumpings": "0",
            "run_out_direct": "0",
            "run_out_assist": "0",
            "batting_points": "16",
            "bowling_points": "50",
            "fielding_points": "0",
            "other_points": "8",
            "dream11_points_total": "74",
            "prior_matches_all": "2",
            "prior_matches_ipl": "2",
            "prior_matches_recent_t20": "0",
            "rolling_points_avg_3_all": "47.0",
            "rolling_points_avg_5_all": "47.0",
            "rolling_runs_avg_3_all": "32.5",
            "rolling_wickets_avg_3_all": "0.0",
            "rolling_points_avg_3_ipl": "47.0",
            "rolling_points_avg_3_recent_t20": "0.0",
        },
        {
            "match_id": "4",
            "match_date": "2024-05-01",
            "competition": "Indian Premier League",
            "source_dataset": "ipl_json",
            "season": "2024",
            "team_type": "club",
            "match_type": "T20",
            "venue": "Venue C",
            "city": "City C",
            "team": "Chennai Super Kings",
            "opponent": "Delhi Capitals",
            "player_name": "Player B",
            "playing_xi": "1",
            "toss_winner": "Delhi Capitals",
            "toss_decision": "field",
            "winner": "Delhi Capitals",
            "runs": "5",
            "balls_faced": "7",
            "fours": "0",
            "sixes": "0",
            "duck": "0",
            "balls_bowled": "18",
            "overs_bowled": "3.0",
            "maidens": "0",
            "runs_conceded": "28",
            "wickets": "1",
            "catches": "0",
            "stumpings": "0",
            "run_out_direct": "0",
            "run_out_assist": "0",
            "batting_points": "5",
            "bowling_points": "25",
            "fielding_points": "0",
            "other_points": "2",
            "dream11_points_total": "32",
            "prior_matches_all": "3",
            "prior_matches_ipl": "3",
            "prior_matches_recent_t20": "0",
            "rolling_points_avg_3_all": "56.0",
            "rolling_points_avg_5_all": "56.0",
            "rolling_runs_avg_3_all": "26.6667",
            "rolling_wickets_avg_3_all": "0.6667",
            "rolling_points_avg_3_ipl": "56.0",
            "rolling_points_avg_3_recent_t20": "0.0",
        },
    ]


class TrainBaselineModelTests(unittest.TestCase):
    def test_split_rows_by_date_uses_recent_rows_for_validation(self):
        from src.ipl_fantasy.train_baseline_model import split_rows_by_date

        train_rows, val_rows = split_rows_by_date(_sample_rows(), validation_fraction=0.5)

        self.assertEqual({row["match_id"] for row in train_rows}, {"1", "2"})
        self.assertEqual({row["match_id"] for row in val_rows}, {"3", "4"})

    def test_prepare_training_matrices_excludes_target_column(self):
        from src.ipl_fantasy.train_baseline_model import prepare_training_matrices

        X, y, feature_columns = prepare_training_matrices(_sample_rows())

        self.assertEqual(len(X), 4)
        self.assertEqual(len(y), 4)
        self.assertNotIn("dream11_points_total", feature_columns)
        self.assertIn("prior_matches_all", feature_columns)
        self.assertNotIn("runs", feature_columns)
        self.assertNotIn("wickets", feature_columns)
        self.assertNotIn("batting_points", feature_columns)
        self.assertNotIn("winner", feature_columns)

    def test_train_and_evaluate_returns_metrics(self):
        from src.ipl_fantasy.train_baseline_model import split_rows_by_date, train_and_evaluate

        train_rows, val_rows = split_rows_by_date(_sample_rows(), validation_fraction=0.5)
        result = train_and_evaluate(train_rows, val_rows)

        self.assertIn("rmse", result["metrics"])
        self.assertIn("mae", result["metrics"])
        self.assertGreaterEqual(result["metrics"]["rmse"], 0.0)
        self.assertGreaterEqual(result["metrics"]["mae"], 0.0)
        self.assertEqual(result["model_name"], "catboost")
        self.assertEqual(result["train_row_count"], 2)
        self.assertEqual(result["validation_row_count"], 2)

    def test_cli_trains_and_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "features.csv"
            output_dir = tmp_path / "model_out"
            script_path = ROOT / "scripts" / "train_baseline_model.py"

            with input_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(_sample_rows()[0].keys()))
                writer.writeheader()
                writer.writerows(_sample_rows())

            result = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--input",
                    str(input_path),
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertTrue((output_dir / "metrics.json").exists())
            self.assertTrue((output_dir / "model.joblib").exists())

            metrics = json.loads((output_dir / "metrics.json").read_text())
            self.assertIn("rmse", metrics["metrics"])
            self.assertIn("feature_columns", metrics)
            self.assertEqual(metrics["model_name"], "catboost")


if __name__ == "__main__":
    unittest.main()

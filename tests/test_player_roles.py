import tempfile
import unittest
from pathlib import Path


class PlayerRoleTests(unittest.TestCase):
    def test_infer_role_from_text_prefers_wicketkeeper(self):
        from src.ipl_fantasy.player_roles import infer_role_from_text

        text = "Player profile. Role: Wicketkeeper Batter. Batting style: right hand bat."
        self.assertEqual(infer_role_from_text(text), "WK")

    def test_infer_role_from_text_detects_all_rounder(self):
        from src.ipl_fantasy.player_roles import infer_role_from_text

        text = "Personal information. Role: Batting all-rounder."
        self.assertEqual(infer_role_from_text(text), "AR")

    def test_infer_role_from_text_detects_bowler(self):
        from src.ipl_fantasy.player_roles import infer_role_from_text

        text = "Profile. Playing role: Left-arm wrist-spin bowler."
        self.assertEqual(infer_role_from_text(text), "BOWL")

    def test_infer_role_from_text_detects_batter(self):
        from src.ipl_fantasy.player_roles import infer_role_from_text

        text = "India | Top order Batter"
        self.assertEqual(infer_role_from_text(text), "BAT")

    def test_infer_role_from_text_ignores_ranking_tab_noise(self):
        from src.ipl_fantasy.player_roles import infer_role_from_text

        text = (
            "role: Batter. batting avg 46.85. "
            "ICC RANKINGS Batting Bowling All-Rounder."
        )
        self.assertEqual(infer_role_from_text(text), "BAT")

    def test_infer_role_from_text_uses_explicit_role_before_bowling_style(self):
        from src.ipl_fantasy.player_roles import infer_role_from_text

        text = "Role\nBatsman\nBowling Style\nRight-arm legbreak"
        self.assertEqual(infer_role_from_text(text), "BAT")

    def test_load_parallel_search_results_handles_broken_warning_suffix(self):
        from src.ipl_fantasy.player_roles import load_parallel_search_results

        broken_payload = """{
  "search_id": "abc",
  "status": "ok",
  "results": [
    {
      "url": "https://www.cricbuzz.com/profiles/1413/virat-kohli",
      "title": "Virat Kohli Profile",
      "excerpts": [
        "Role\\nBatsman"
      ]
    }
  ],
  "warnings": [
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "result.json"
            path.write_text(broken_payload)
            results = load_parallel_search_results(path)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "Virat Kohli Profile")

    def test_resolve_role_from_search_results_prefers_stronger_domains(self):
        from src.ipl_fantasy.player_roles import resolve_role_from_search_results

        results = [
            {
                "url": "https://en.wikipedia.org/wiki/Player_X",
                "title": "Player X",
                "excerpts": ["Player X is a right-handed batter and occasional bowler."],
            },
            {
                "url": "https://www.icc-cricket.com/player-x",
                "title": "Player X | ICC",
                "excerpts": ["role: Batter"],
            },
        ]

        resolved = resolve_role_from_search_results(results)
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved["role"], "BAT")
        self.assertIn("icc-cricket.com", resolved["source_url"])

    def test_resolve_role_from_search_results_skips_low_quality_urls(self):
        from src.ipl_fantasy.player_roles import resolve_role_from_search_results

        results = [
            {
                "url": "https://www.espncricinfo.com/series/foo/match-squads",
                "title": "Match Squads",
                "excerpts": ["Chris Jordan | Wicketkeeper Batter"],
            },
            {
                "url": "https://www.icc-cricket.com/rankings/65748/rashid-khan",
                "title": "Rashid Khan | Player Rankings - ICC",
                "excerpts": ["role: Bowler"],
            },
        ]

        resolved = resolve_role_from_search_results(results)
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved["role"], "BOWL")
        self.assertIn("icc-cricket.com", resolved["source_url"])


if __name__ == "__main__":
    unittest.main()

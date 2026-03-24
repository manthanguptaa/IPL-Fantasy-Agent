"""Tests for the residual adapter (test-time training layer)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.ipl_fantasy.residual_adapter import (
    AdapterConfig,
    ResidualAdapter,
)
from src.ipl_fantasy.team_optimizer import Player


def _make_player(name: str, role: str = "BAT", pts: float = 40.0) -> Player:
    return Player(
        name=name, team="TeamA", role=role,
        predicted_points=pts, credits=8.5,
        ceiling=pts * 1.4, floor=pts * 0.5,
        variance=100.0, is_foreign=False,
    )


def _make_predictions(*players: tuple[str, str, float]) -> dict:
    return {
        name: {"predicted": pts, "role": role, "team": "TeamA"}
        for name, role, pts in players
    }


def _make_actuals(*players: tuple[str, float]) -> dict:
    return {name: pts for name, pts in players}


class TestResidualAdapterBasic:
    def test_fresh_adapter_no_op(self):
        adapter = ResidualAdapter()
        players = [_make_player("A"), _make_player("B")]
        adjusted = adapter.adjust(players)
        assert len(adjusted) == 2
        for orig, adj in zip(players, adjusted):
            assert adj.predicted_points == orig.predicted_points

    def test_observe_single_match(self):
        adapter = ResidualAdapter()
        preds = _make_predictions(("Kohli", "BAT", 50.0), ("Bumrah", "BOWL", 35.0))
        actuals = _make_actuals(("Kohli", 70.0), ("Bumrah", 20.0))

        summary = adapter.observe("m1", "2026-03-28", preds, actuals)

        assert summary["n_players"] == 2
        assert summary["total_matches_observed"] == 1
        assert adapter.player_ema["Kohli"] == pytest.approx(20.0)  # 70-50
        assert adapter.player_ema["Bumrah"] == pytest.approx(-15.0)  # 20-35

    def test_adjust_after_observation(self):
        adapter = ResidualAdapter()
        preds = _make_predictions(("Kohli", "BAT", 50.0))
        actuals = _make_actuals(("Kohli", 70.0))
        adapter.observe("m1", "2026-03-28", preds, actuals)

        players = [_make_player("Kohli", "BAT", 48.0)]
        adjusted = adapter.adjust(players)

        # Kohli should be adjusted upward (residual was +20)
        assert adjusted[0].predicted_points > 48.0

    def test_unknown_player_no_adjustment(self):
        adapter = ResidualAdapter()
        preds = _make_predictions(("Kohli", "BAT", 50.0))
        actuals = _make_actuals(("Kohli", 70.0))
        adapter.observe("m1", "2026-03-28", preds, actuals)

        players = [_make_player("NewPlayer", "BAT", 30.0)]
        adjusted = adapter.adjust(players)

        # No player-level correction, but no crash
        assert adjusted[0].predicted_points == 30.0  # no data for this player yet

    def test_safety_cap(self):
        config = AdapterConfig(max_correction_frac=0.20)
        adapter = ResidualAdapter(config=config)

        # Huge residual
        preds = _make_predictions(("X", "BAT", 30.0))
        actuals = _make_actuals(("X", 100.0))
        adapter.observe("m1", "2026-03-28", preds, actuals)

        players = [_make_player("X", "BAT", 30.0)]
        adjusted = adapter.adjust(players)

        max_shift = 0.20 * 30.0  # 6 points max
        assert adjusted[0].predicted_points <= 30.0 + max_shift + 0.01

    def test_ema_decay_over_matches(self):
        adapter = ResidualAdapter(AdapterConfig(ema_alpha=0.5))

        # Match 1: +20 residual
        adapter.observe("m1", "2026-03-28",
                        _make_predictions(("A", "BAT", 30.0)),
                        _make_actuals(("A", 50.0)))

        # Match 2: -10 residual
        adapter.observe("m2", "2026-03-29",
                        _make_predictions(("A", "BAT", 30.0)),
                        _make_actuals(("A", 20.0)))

        # EMA: 0.5 * (-10) + 0.5 * 20 = 5.0
        assert adapter.player_ema["A"] == pytest.approx(5.0)


class TestRoleBias:
    def test_role_bias_not_active_early(self):
        config = AdapterConfig(min_matches_role=5)
        adapter = ResidualAdapter(config=config)

        # Only 1 match
        preds = _make_predictions(("A", "BAT", 30.0))
        actuals = _make_actuals(("A", 50.0))
        adapter.observe("m1", "2026-03-28", preds, actuals)

        biases = adapter.get_role_biases()
        # Bias exists internally but won't be applied in adjust() until 5 matches
        assert adapter.total_matches < config.min_matches_role

    def test_role_bias_active_after_threshold(self):
        config = AdapterConfig(min_matches_role=3)
        adapter = ResidualAdapter(config=config)

        for i in range(3):
            preds = _make_predictions(("A", "BOWL", 25.0), ("B", "BOWL", 20.0))
            actuals = _make_actuals(("A", 35.0), ("B", 30.0))
            adapter.observe(f"m{i}", f"2026-03-{28+i}", preds, actuals)

        biases = adapter.get_role_biases()
        assert biases["BOWL"] > 0  # bowlers consistently underestimated


class TestPersistence:
    def test_save_and_load(self, tmp_path):
        adapter = ResidualAdapter()
        preds = _make_predictions(("Kohli", "BAT", 50.0), ("Bumrah", "BOWL", 35.0))
        actuals = _make_actuals(("Kohli", 70.0), ("Bumrah", 20.0))
        adapter.observe("m1", "2026-03-28", preds, actuals)

        save_path = tmp_path / "adapter.json"
        adapter.save(save_path)

        loaded = ResidualAdapter.load(save_path)
        assert loaded.total_matches == 1
        assert loaded.player_ema["Kohli"] == pytest.approx(adapter.player_ema["Kohli"])
        assert loaded.player_ema["Bumrah"] == pytest.approx(adapter.player_ema["Bumrah"])
        assert loaded.player_obs_count["Kohli"] == 1

    def test_load_missing_file_returns_fresh(self, tmp_path):
        adapter = ResidualAdapter.load(tmp_path / "nonexistent.json")
        assert adapter.total_matches == 0

    def test_roundtrip_preserves_config(self, tmp_path):
        config = AdapterConfig(ema_alpha=0.6, ridge_lambda=5.0, max_correction_frac=0.25)
        adapter = ResidualAdapter(config=config)
        adapter.observe("m1", "2026-03-28",
                        _make_predictions(("A", "BAT", 30.0)),
                        _make_actuals(("A", 40.0)))

        path = tmp_path / "adapter.json"
        adapter.save(path)
        loaded = ResidualAdapter.load(path)
        assert loaded.config.ema_alpha == 0.6
        assert loaded.config.ridge_lambda == 5.0
        assert loaded.config.max_correction_frac == 0.25


class TestMultiMatchScenario:
    def test_multi_match_season_simulation(self):
        """Simulate 10 matches where a player consistently outperforms predictions."""
        adapter = ResidualAdapter(AdapterConfig(
            ema_alpha=0.4,
            min_matches_role=3,
            min_matches_ridge=15,
        ))

        # Player consistently scores 15 pts above prediction
        for i in range(10):
            preds = _make_predictions(
                ("Star", "AR", 40.0),
                ("Steady", "BAT", 35.0),
                ("Cold", "BOWL", 30.0),
            )
            actuals = _make_actuals(
                ("Star", 55.0),     # +15 every match
                ("Steady", 36.0),   # +1 (noise)
                ("Cold", 20.0),     # -10 every match
            )
            adapter.observe(f"m{i}", f"2026-04-{i+1:02d}", preds, actuals)

        # Star should have large positive correction
        assert adapter.player_ema["Star"] > 10.0
        # Cold should have large negative correction
        assert adapter.player_ema["Cold"] < -5.0

        # Adjust predictions
        players = [
            _make_player("Star", "AR", 40.0),
            _make_player("Steady", "BAT", 35.0),
            _make_player("Cold", "BOWL", 30.0),
        ]
        adjusted = adapter.adjust(players)

        adj_by_name = {p.name: p for p in adjusted}
        assert adj_by_name["Star"].predicted_points > 40.0
        assert adj_by_name["Cold"].predicted_points < 30.0

    def test_ceiling_and_floor_also_adjusted(self):
        adapter = ResidualAdapter()
        preds = _make_predictions(("A", "BAT", 40.0))
        actuals = _make_actuals(("A", 60.0))
        adapter.observe("m1", "2026-03-28", preds, actuals)

        player = _make_player("A", "BAT", 40.0)
        original_ceil = player.ceiling
        original_floor = player.floor

        adjusted = adapter.adjust([player])[0]
        # Ceiling should increase (but less than predicted_points shift)
        assert adjusted.ceiling > original_ceil
        # Floor should increase (but even less)
        assert adjusted.floor > original_floor

    def test_diagnostics(self):
        adapter = ResidualAdapter()
        for i in range(5):
            preds = _make_predictions(("A", "BAT", 30.0), ("B", "BOWL", 25.0))
            actuals = _make_actuals(("A", 45.0), ("B", 15.0))
            adapter.observe(f"m{i}", f"2026-04-{i+1:02d}", preds, actuals)

        summary = adapter.get_summary()
        assert "RESIDUAL ADAPTER" in summary
        assert "Matches observed: 5" in summary

        top = adapter.get_top_corrections(5)
        assert len(top) == 2
        assert top[0]["player"] in ("A", "B")

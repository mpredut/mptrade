"""Regressions for the Kraken shadow test: configuration, P&L and forward window."""
from __future__ import annotations

import contextlib
import dataclasses
import importlib.util
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
KRAKEN_DIR = ROOT / "kraken"
sys.path.insert(0, str(KRAKEN_DIR))
sys.path.insert(0, str(ROOT))

SPEC = importlib.util.spec_from_file_location(
    "kraken_shadow_live_under_test", KRAKEN_DIR / "shadow_live.py",
)
shadow = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(shadow)


class ShadowLiveTest(unittest.TestCase):
    def test_decision_distance_counts_changed_order_events(self):
        current = [
            {"bar": 1, "side": "buy", "kind": "ENTRY"},
            {"bar": 4, "side": "buy", "kind": "DCA"},
        ]
        candidate = [
            {"bar": 1, "side": "buy", "kind": "ENTRY"},
            {"bar": 5, "side": "buy", "kind": "DCA"},
            {"bar": 9, "side": "sell", "kind": "TP"},
        ]

        self.assertEqual(shadow._decision_distance(current, current), 0)
        self.assertEqual(shadow._decision_distance(current, candidate), 2)

    def test_overlay_candidate_is_only_enabled_at_its_native_interval(self):
        @dataclasses.dataclass(frozen=True)
        class Params:
            takeprofit_pct: float = 5.0
            dca_drop_pct: float = 1.25
            dca_spacing_growth_pct: float = 0.0
            reentry_drop_pct: float = 0.0
            stop_loss_pct: float = 12.5
            tp_trail_profit_floor_pct: float = 0.0
            dca_vol_scale_k: float = 0.0
            dca_vol_ref: float = 2.0
            dca_vol_interval: int = 240
            dca_trend_brake: bool = False
            dca_brake_min_pct: float = 1.5
            tp_trail_adaptive: bool = False
            tp_trail_k: float = 2.0
            tp_trail_min: float = 1.5
            tp_trail_max: float = 8.0
            tp_trail_vol_interval: int = 240
            trend_interval: int = 240
            trend_overlay: bool = False
            trend_topup: float = 2000.0
            trend_trail_pct: float = 5.0
            trend_exit_break: bool = False
            tp_regime_gate: bool = False

        with patch(
            "strategies.spot_dca.StratParams.from_env", return_value=Params(),
        ), patch.object(shadow, "_load_runtime_config"):
            variants_60 = shadow._variants(60)
            variants_240 = shadow._variants(240)

        self.assertEqual(
            list(variants_60),
            [
                "current", "tp4", "dca15", "dca_progressive025",
                "reentry4", "trail_profit_floor_sl18", "trail_profit_floor_sl125",
            ],
        )
        self.assertEqual(
            list(variants_240),
            [
                "current", "tp4", "dca15", "dca_progressive025",
                "reentry4", "trail_profit_floor_sl18", "trail_profit_floor_sl125",
                "A_trail", "dca_vol_m1", "tp_regime_gate",
                "overlay650t8_regime_v2", "B_dcabrake_regime_v2",
                "overlay_safe_combo",
            ],
        )
        safe_overlay = variants_240["overlay_safe_combo"]
        self.assertTrue(safe_overlay.trend_overlay)
        self.assertEqual(safe_overlay.trend_topup, 350.0)
        self.assertEqual(safe_overlay.trend_trail_pct, 6.0)
        self.assertEqual(safe_overlay.takeprofit_pct, 4.0)
        self.assertEqual(safe_overlay.dca_spacing_growth_pct, 0.25)
        progressive = variants_60["dca_progressive025"]
        self.assertEqual(progressive.dca_spacing_growth_pct, 0.25)
        self.assertEqual(variants_60["reentry4"].reentry_drop_pct, 4.0)
        profit_floor = variants_60["trail_profit_floor_sl18"]
        self.assertEqual(profit_floor.tp_trail_profit_floor_pct, 1.0)
        self.assertEqual(profit_floor.stop_loss_pct, 18.0)
        # Decoupled: profit floor with the baseline stop (NOT widened to 18)
        floor_only = variants_60["trail_profit_floor_sl125"]
        self.assertEqual(floor_only.tp_trail_profit_floor_pct, 1.0)
        self.assertEqual(floor_only.stop_loss_pct, variants_60["current"].stop_loss_pct)
        self.assertNotEqual(floor_only.stop_loss_pct, 18.0)
        adaptive = variants_240["A_trail"]
        self.assertTrue(adaptive.tp_trail_adaptive)
        self.assertEqual(adaptive.tp_trail_vol_interval, 240)
        vol_scaled = variants_240["dca_vol_m1"]
        self.assertEqual(vol_scaled.dca_vol_scale_k, -1.0)
        self.assertEqual(vol_scaled.dca_vol_ref, 2.0)
        self.assertTrue(variants_240["tp_regime_gate"].tp_regime_gate)
        candidate = variants_240["overlay650t8_regime_v2"]
        self.assertTrue(candidate.trend_overlay)
        brake = variants_240["B_dcabrake_regime_v2"]
        self.assertTrue(brake.dca_trend_brake)
        self.assertEqual(candidate.trend_topup, 650.0)
        self.assertEqual(candidate.trend_trail_pct, 8.0)

    def test_runtime_config_matches_live_precedence(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            config_path = Path(tmp) / "config.env"
            env_path.write_text(
                "KRAKEN_PAIR=FROM_ENV\nSTRAT_TAKEPROFIT_PCT=4.5\n",
                encoding="utf-8",
            )
            config_path.write_text(
                "KRAKEN_PAIR=FROM_CONFIG\nSTRAT_TAKEPROFIT_PCT=5.0\n"
                "STRAT_DCA_DROP_PCT=1.25  # versionat\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                shadow._load_runtime_config(str(env_path), str(config_path))
                self.assertEqual(os.environ["KRAKEN_PAIR"], "FROM_ENV")
                self.assertEqual(os.environ["STRAT_TAKEPROFIT_PCT"], "4.5")
                self.assertEqual(os.environ["STRAT_DCA_DROP_PCT"], "1.25")

    def test_forward_history_keeps_anchor_after_api_window_moves(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(shadow, "LOG_DIR", tmp):
            first = [(100, 1.0, 2.0, 0.5, 1.5), (160, 1.5, 2.5, 1.0, 2.0)]
            later = [(160, 1.5, 2.5, 1.0, 2.0), (220, 2.0, 3.0, 1.5, 2.5)]

            shadow._merge_forward_history("PAIR", 1, 100, first)
            merged = shadow._merge_forward_history("PAIR", 1, 100, later)

            self.assertEqual([bar[0] for bar in merged], [100, 160, 220])

    def test_forward_history_fails_closed_when_a_gap_loses_bars(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(shadow, "LOG_DIR", tmp):
            shadow._merge_forward_history(
                "PAIR", 1, 100,
                [(100, 1.0, 2.0, 0.5, 1.5), (160, 1.5, 2.5, 1.0, 2.0)],
            )
            with self.assertRaisesRegex(RuntimeError, "gap"):
                shadow._merge_forward_history(
                    "PAIR", 1, 100, [(400, 4.0, 5.0, 3.5, 4.5)],
                )

    def test_display_compares_total_pnl_including_open_position(self):
        block = {
            "bars": 10,
            "buyhold_pct": 2.0,
            "configs": {
                "current": {"net_pct": 0.0, "total_pct": 1.0,
                            "maxdd_pct": 2.0, "cycles": 0},
                "tp4": {"net_pct": 0.0, "total_pct": 2.0,
                        "maxdd_pct": 2.0, "cycles": 0},
                "dca15": {"net_pct": 0.0, "total_pct": 0.0,
                          "maxdd_pct": 2.0, "cycles": 0},
            },
        }
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            shadow._print_block("TEST", block)
        rendered = out.getvalue()
        self.assertIn("+1.00pp", rendered)
        self.assertIn("-1.00pp", rendered)

    def test_single_shot_failure_returns_nonzero(self):
        with patch.object(shadow, "_load_runtime_config"), \
                patch.object(shadow, "snapshot", side_effect=RuntimeError("fetch failed")), \
                patch.dict(os.environ, {"KRAKEN_PAIR": "HYPEUSD"}), \
                patch.object(sys, "argv", ["shadow_live.py", "--quiet"]), \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(shadow.main(), 1)


if __name__ == "__main__":
    unittest.main()

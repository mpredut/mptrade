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

    @staticmethod
    def _live_params():
        @dataclasses.dataclass(frozen=True)
        class Params:
            takeprofit_pct: float = 4.0
            dca_drop_pct: float = 1.25
            dca_spacing_growth_pct: float = 0.25
            reentry_drop_pct: float = 2.2
            stop_loss_pct: float = 18.0
            tp_trail_profit_floor_pct: float = 1.0
            dca_vol_scale_k: float = 0.0
            dca_vol_ref: float = 2.0
            dca_vol_interval: int = 240
            dca_trend_brake: bool = False
            dca_brake_min_pct: float = 1.5
            tp_trend_hold: bool = True
            tp_trail_adaptive: bool = True
            tp_trail_vol_interval: int = 240
            trend_interval: int = 240
            trend_overlay: bool = False
            trend_topup: float = 2000.0
            trend_trail_pct: float = 5.0
            trend_exit_break: bool = False
            tp_regime_gate: bool = True

        return Params()

    def test_variants_isolate_each_promoted_change_against_live(self):
        live = self._live_params()
        with patch(
            "strategies.spot_dca.StratParams.from_env", return_value=live,
        ), patch.object(shadow, "_load_runtime_config"):
            variants = shadow._variants(240)

        self.assertEqual(
            list(variants),
            [
                "current", "pre0923", "rev_tp5", "rev_spacing0", "rev_trail_fixed",
                "rev_gate_off", "rev_floor0", "rev_sl125", "dca15", "reentry4",
                "dca_vol_m1", "overlay650t8_regime_v2", "B_dcabrake_regime_v2",
                "overlay_safe_combo",
            ],
        )
        self.assertIs(variants["current"], live)
        # No candidate may collapse onto the live configuration: it would log noise only.
        for name, params in variants.items():
            if name != "current":
                self.assertNotEqual(params, live, name)
        # Each rev_* variant differs from live in exactly one field.
        reverted = {
            "rev_tp5": ("takeprofit_pct", 5.0),
            "rev_spacing0": ("dca_spacing_growth_pct", 0.0),
            "rev_trail_fixed": ("tp_trail_adaptive", False),
            "rev_gate_off": ("tp_regime_gate", False),
            "rev_floor0": ("tp_trail_profit_floor_pct", 0.0),
            "rev_sl125": ("stop_loss_pct", 12.5),
        }
        for name, (field, value) in reverted.items():
            params = variants[name]
            self.assertEqual(getattr(params, field), value, name)
            changed = [f.name for f in dataclasses.fields(params)
                       if getattr(params, f.name) != getattr(live, f.name)]
            self.assertEqual(changed, [field], name)
        pre = variants["pre0923"]
        for field, value in reverted.values():
            self.assertEqual(getattr(pre, field), value, field)
        self.assertEqual(variants["reentry4"].reentry_drop_pct, 4.0)
        vol_scaled = variants["dca_vol_m1"]
        self.assertEqual(vol_scaled.dca_vol_scale_k, -1.0)
        self.assertEqual(vol_scaled.dca_vol_ref, 2.0)
        candidate = variants["overlay650t8_regime_v2"]
        self.assertTrue(candidate.trend_overlay)
        self.assertEqual(candidate.trend_topup, 650.0)
        self.assertEqual(candidate.trend_trail_pct, 8.0)
        self.assertTrue(variants["B_dcabrake_regime_v2"].dca_trend_brake)
        safe_overlay = variants["overlay_safe_combo"]
        self.assertTrue(safe_overlay.trend_overlay)
        self.assertEqual(safe_overlay.trend_topup, 350.0)
        self.assertEqual(safe_overlay.trend_trail_pct, 6.0)

    def test_interval_the_live_config_cannot_replay_is_skipped(self):
        with patch(
            "strategies.spot_dca.StratParams.from_env", return_value=self._live_params(),
        ), patch.object(shadow, "_load_runtime_config"):
            self.assertIsNone(shadow._replay_interval_error(240))
            self.assertIn("240", shadow._replay_interval_error(60))
            with patch.dict(os.environ, {"KRAKEN_PAIR": "HYPEUSD"}), \
                    patch.object(shadow, "snapshot") as snapshot, \
                    patch.object(sys, "argv", ["shadow_live.py", "--interval", "60"]), \
                    contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(shadow.main(), 0)
            snapshot.assert_not_called()
            self.assertIn("skipped", out.getvalue())

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
                "STRAT_DCA_DROP_PCT=1.25  # versioned\n",
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
                patch.object(shadow, "_replay_interval_error", return_value=None), \
                patch.object(shadow, "snapshot", side_effect=RuntimeError("fetch failed")), \
                patch.dict(os.environ, {"KRAKEN_PAIR": "HYPEUSD"}), \
                patch.object(sys, "argv", ["shadow_live.py", "--quiet"]), \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(shadow.main(), 1)


if __name__ == "__main__":
    unittest.main()

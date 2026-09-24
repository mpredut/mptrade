"""kraken/replay.py — the backtest engine that runs the LIVE STRATEGY over OHLC.
Characterisation: on a deterministic series the engine runs without network,
notifications or state files, and returns sane metrics (accounting from the live _apply_fill)."""
import os
import importlib.util
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "kraken"))
sys.path.insert(0, ROOT)
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

# T212, Kraken and Hyperliquid expose generic modules named `strategy`,
# `market_data` and `notify`. Build the replay's Kraken import graph in isolation
# so full-suite collection order cannot substitute another venue's strategy.
_COLLIDING_MODULES = ("strategy", "market_data", "notify")
_PRELOADED_MODULES = {
    name: sys.modules.pop(name) for name in _COLLIDING_MODULES if name in sys.modules
}
try:
    _SPEC = importlib.util.spec_from_file_location(
        "kraken_replay_under_test", os.path.join(ROOT, "kraken", "replay.py")
    )
    rp = importlib.util.module_from_spec(_SPEC)
    sys.modules[_SPEC.name] = rp
    _SPEC.loader.exec_module(rp)
    strat = rp._strat
finally:
    for _name in _COLLIDING_MODULES:
        sys.modules.pop(_name, None)
    sys.modules.update(_PRELOADED_MODULES)


def _series():
    pts = []; p = 100.0
    for _ in range(8): p *= 0.985; pts.append((p, p*1.002, p*0.99, p*0.995))
    for _ in range(6): p *= 1.02;  pts.append((p, p*1.01, p*0.998, p*1.005))
    for _ in range(5): p *= 0.95;  pts.append((p, p*1.002, p*0.97, p*0.98))
    for _ in range(8): p *= 1.03;  pts.append((p, p*1.015, p*0.995, p*1.01))
    return pts


def _params(**over):
    d = dict(currency="USD", entry_amount=100.0, entry_discount_pct=0.2, dca_amount=50.0,
             dca_drop_pct=2.0, check_minutes=2.0, takeprofit_pct=3.0, max_budget=500.0,
             max_dca_buys=3, enable_takeprofit=True, order_ttl_min=10.0, stop_loss_pct=12.5,
             adopt_cost=0.0, adopt_qty=0.0, reentry_drop_pct=0.0, reentry_tolerance_pct=0.05,
             reentry_adaptive=False, reentry_sl_bounce_pct=1.5, tp_tranches=[])
    d.update(over)
    return strat.StratParams(**d)


class ReplayEngineTest(unittest.TestCase):
    def test_cash_account_rejects_entry_that_cannot_pay_its_fee(self):
        result = rp.run_replay([(100, 101, 99, 100)] * 3, _params(), initial_cash=100)
        self.assertEqual(result["fills"], 0)
        self.assertGreater(result["funding"]["refused_buys"], 0)
        self.assertEqual(result["funding"]["final_cash"], 100)

    def test_cash_account_rejects_invalid_capital(self):
        for cash in (0, -1, True, float("nan"), float("inf")):
            with self.subTest(cash=cash), self.assertRaises(ValueError):
                rp.run_replay([(100, 101, 99, 100)], _params(), initial_cash=cash)

    def test_cash_account_equity_matches_cash_plus_marked_inventory(self):
        bars = [(100, 101, 99, 100), (100, 101, 90, 94), (94, 95, 89, 90)]
        result = rp.run_replay(bars, _params(), initial_cash=120,
                               execution=rp.ExecutionModel(partial_fill_ratio=0.5))
        self.assertGreaterEqual(result["funding"]["minimum_cash"], 0)
        self.assertAlmostEqual(result["funding"]["final_cash"] + result["open_qty"] * 90,
                               120 + result["total"], places=6)

    def test_trailing_cancels_unfilled_entry_remainder_before_market_exit(self):
        bars = [(100, 101, 99, 100), (100, 107, 99, 106),
                (106, 107, 102, 102), (102, 103, 95, 102)]
        for policy in ("buy_first", "sell_first"):
            with self.subTest(policy=policy), patch.object(strat, "log"):
                result = rp.run_replay(
                    bars, _params(tp_trend_hold=True, tp_trail_pct=3.0,
                                  reentry_drop_pct=2.2), bar_minutes=240,
                    execution=rp.ExecutionModel(
                        partial_fill_ratio=0.5, intrabar_policy=policy),
                    include_decision_trace=True)
            self.assertEqual(result["ambiguous_bars"], 0)
            self.assertEqual(result["fills"], 2)  # One partial BUY and one full MARKET SELL.
            self.assertEqual(result["open_qty"], 0.0)
            exits = [o for o in result["decision_trace"] if o["side"] == "sell"]
            self.assertEqual(len(exits), 1)
            self.assertTrue(exits[0]["market"])

    def test_partial_limit_fill_remains_for_the_next_bar(self):
        bars = [(100, 101, 99, 100), (100, 101, 99, 100)]
        full = rp.run_replay(bars, _params(), fee_pct=0.26)
        partial = rp.run_replay(
            bars, _params(), fee_pct=0.26,
            execution=rp.ExecutionModel(partial_fill_ratio=0.5),
        )
        self.assertEqual(full["fills"], 1)
        self.assertEqual(partial["fills"], 1)
        self.assertAlmostEqual(partial["open_qty"], full["open_qty"] / 2)

    def test_runs_and_returns_metrics(self):
        res = rp.run_replay(_series(), _params(), fee_pct=0.26, bar_minutes=60)
        for k in ("realized", "net", "fees", "total", "cycles", "wins", "maxdd",
                  "open_qty", "return_pct", "max_drawdown_pct", "sharpe", "sortino",
                  "calmar", "cvar_95_pct", "exposure_pct", "profit_factor",
                  "expectancy", "turnover_pct", "fills"):
            self.assertIn(k, res)
        self.assertEqual(res["cycles"], 2)          # motorul inchide 2 cicluri pe seria asta
        self.assertEqual(res["wins"], 1)
        self.assertEqual(res["open_qty"], 0.0)      # pozitie inchisa la final
        self.assertLess(res["net"], res["realized"]) # net = brut - fee-uri
        self.assertGreaterEqual(res["fees"], 0.0)
        self.assertGreaterEqual(res["maxdd"], 0.0)
        self.assertAlmostEqual(res["net_pnl"], res["total"])

    def test_decision_trace_is_explicitly_opt_in(self):
        plain = rp.run_replay(_series(), _params(), fee_pct=0.26, bar_minutes=60)
        traced = rp.run_replay(
            _series(), _params(), fee_pct=0.26, bar_minutes=60,
            include_decision_trace=True,
        )

        self.assertNotIn("decision_trace", plain)
        self.assertTrue(traced["decision_trace"])
        self.assertEqual(
            set(traced["decision_trace"][0]),
            {"bar", "side", "kind", "market", "price", "qty"},
        )

    def test_no_state_file_written(self):
        before = set(os.listdir(os.path.join(ROOT, "kraken")))
        rp.run_replay(_series(), _params(), fee_pct=0.26)
        after = set(os.listdir(os.path.join(ROOT, "kraken")))
        new_state = [f for f in (after - before) if "REPLAY" in f or f.startswith(".state")]
        self.assertEqual(new_state, [], f"replay must NOT write state: {new_state}")

    def test_existing_replay_state_cannot_contaminate_result(self):
        clean = rp.run_replay(_series(), _params(), fee_pct=0.26)
        with tempfile.TemporaryDirectory() as tmp:
            state_path = os.path.join(tmp, ".state_REPLAY.json")
            contaminated = strat._new_state()
            contaminated.update({"qty": 99.0, "cost": 1.0, "realized_net": 999999.0})
            with open(state_path, "w", encoding="utf-8") as handle:
                json.dump(contaminated, handle)
            with patch.object(strat, "state_path_for", return_value=state_path):
                replayed = rp.run_replay(_series(), _params(), fee_pct=0.26)
        self.assertEqual(replayed, clean)

    def test_order_created_at_close_cannot_fill_on_same_bar(self):
        # A low of 1 would fill any BUY, but the order is only decided at close=100.
        # The harness must only be able to execute it on the next bar.
        first_bar = (100.0, 200.0, 1.0, 100.0)
        one_bar = rp.run_replay([first_bar], _params(), fee_pct=0.26)
        self.assertEqual(one_bar["fills"], 0)
        self.assertEqual(one_bar["open_qty"], 0.0)

        second_bar = (100.0, 101.0, 99.0, 100.0)
        two_bars = rp.run_replay([first_bar, second_bar], _params(), fee_pct=0.26)
        self.assertEqual(two_bars["fills"], 1)
        self.assertGreater(two_bars["open_qty"], 0.0)

    def test_stale_entry_is_replaced_after_a_rally_like_live_ttl(self):
        # The ENTRY is placed 0.2% under 100; the next bars rally without touching it.
        # Live cancels it after order_ttl_min and re-places it near the current price,
        # so the pullback to 110.5 on the last bar must fill the re-placed entry.
        bars = [(100.0, 100.5, 99.9, 100.0)]
        bars += [(p, p * 1.004, p * 0.9995, p) for p in (103.0, 106.0, 109.0, 111.0)]
        bars += [(111.0, 111.2, 110.5, 111.0)]
        trace = rp.run_replay(bars, _params(), fee_pct=0.26, bar_minutes=240,
                              include_decision_trace=True)
        entries = [e for e in trace["decision_trace"] if e["kind"] == "ENTRY"]
        self.assertGreater(len(entries), 1)
        self.assertGreater(entries[-1]["price"], 110.0)
        self.assertGreater(trace["open_qty"], 0.0)

        # Bars shorter than the TTL keep the order, as live would.
        short = rp.run_replay(bars, _params(order_ttl_min=300.0), fee_pct=0.26,
                              bar_minutes=240, include_decision_trace=True)
        self.assertEqual(
            [e["kind"] for e in short["decision_trace"]].count("ENTRY"), 1)
        self.assertEqual(short["open_qty"], 0.0)

    def test_market_stop_fills_at_next_open_even_below_reference_price(self):
        bars = [
            (100.0, 101.0, 99.0, 100.0),  # places the entry
            (100.0, 101.0, 99.0, 100.0),  # umple intrarea
            (80.0, 82.0, 78.0, 80.0),     # triggers STOP MARKET
            (70.0, 71.0, 69.0, 70.0),     # gap down: fills at open, does not wait for the limit
        ]

        result = rp.run_replay(
            bars, _params(stop_loss_pct=10.0), fee_pct=0.26, bar_minutes=60,
        )

        self.assertEqual(result["fills"], 2)
        self.assertEqual(result["cycles"], 1)
        self.assertEqual(result["open_qty"], 0.0)
        self.assertLess(result["net"], 0.0)

    def test_market_slippage_makes_gap_stop_more_conservative(self):
        bars = [
            (100.0, 101.0, 99.0, 100.0),
            (100.0, 101.0, 99.0, 100.0),
            (80.0, 82.0, 78.0, 80.0),
            (70.0, 71.0, 69.0, 70.0),
        ]
        plain = rp.run_replay(bars, _params(stop_loss_pct=10), fee_pct=0.26)
        stressed = rp.run_replay(
            bars, _params(stop_loss_pct=10), fee_pct=0.26,
            execution=rp.ExecutionModel(spread_bps=20, market_slippage_bps=80),
        )
        self.assertLess(stressed["return_pct"], plain["return_pct"])

    def test_market_exit_uses_taker_fee_while_limits_use_maker_fee(self):
        bars = [
            (100.0, 101.0, 99.0, 100.0),
            (100.0, 101.0, 99.0, 100.0),
            (80.0, 82.0, 78.0, 80.0),
            (70.0, 71.0, 69.0, 70.0),
        ]
        from offline.backtests.execution import FeeModel

        uniform = rp.run_replay(
            bars, _params(stop_loss_pct=10), fee_pct=0.16, bar_minutes=60,
        )
        maker_taker = rp.run_replay(
            bars, _params(stop_loss_pct=10), fee_pct=0.16, bar_minutes=60,
            fee_model=FeeModel(limit_fee_pct=0.16, market_fee_pct=0.40),
        )

        self.assertGreater(maker_taker["fees"], uniform["fees"])
        self.assertLess(maker_taker["return_pct"], uniform["return_pct"])

    def test_overlay_requires_replay_bars_at_configured_trend_interval(self):
        with self.assertRaisesRegex(ValueError, "trend_interval"):
            rp.run_replay(
                _series(), _params(trend_overlay=True, trend_interval=240),
                fee_pct=0.26, bar_minutes=60,
            )

    def test_tp_regime_gate_requires_configured_trend_interval(self):
        with self.assertRaisesRegex(ValueError, "trend_interval"):
            rp.run_replay(
                _series(),
                _params(
                    tp_trend_hold=True,
                    tp_regime_gate=True,
                    trend_interval=240,
                ),
                fee_pct=0.26,
                bar_minutes=60,
            )

    def test_adaptive_features_require_their_configured_bar_interval(self):
        with self.assertRaisesRegex(ValueError, "tp_trail_vol_interval"):
            rp.run_replay(
                _series(), _params(tp_trail_adaptive=True, tp_trail_vol_interval=240),
                fee_pct=0.26, bar_minutes=60,
            )
        with self.assertRaisesRegex(ValueError, "trend_interval"):
            rp.run_replay(
                _series(), _params(dca_trend_brake=True, trend_interval=240),
                fee_pct=0.26, bar_minutes=60,
            )
        with self.assertRaisesRegex(ValueError, "dca_vol_interval"):
            rp.run_replay(
                _series(),
                _params(dca_vol_scale_k=-1.0, dca_vol_interval=240),
                fee_pct=0.26, bar_minutes=60,
            )

    def test_adaptive_reentry_requires_bar_interval_for_time_scaling(self):
        with self.assertRaisesRegex(ValueError, "reentry_adaptive"):
            rp.run_replay(_series(), _params(reentry_adaptive=True), fee_pct=0.26)


class PercentageSizingReplayTest(unittest.TestCase):
    def test_equivalent_fixed_and_percentage_models_are_financially_identical(self):
        fixed = _params()
        percent = _params(
            total_budget=1000.0, alloc_pct=50.0,
            entry_pct=20.0, dca_pct=10.0,
        )
        fixed_result = rp.run_replay(
            _series(), fixed, fee_pct=0.26, bar_minutes=60,
            include_decision_trace=True,
        )
        percent_result = rp.run_replay(
            _series(), percent, fee_pct=0.26, bar_minutes=60,
            include_decision_trace=True,
        )
        self.assertEqual(percent_result, fixed_result)


if __name__ == "__main__":
    unittest.main(verbosity=2)

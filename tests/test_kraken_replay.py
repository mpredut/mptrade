"""kraken/replay.py — the backtest engine that runs the LIVE STRATEGY over OHLC.
Characterisation: on a deterministic series the engine runs without network,
notifications or state files, and returns sane metrics (accounting from the live _apply_fill)."""
import os
import importlib.util
import json
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "kraken"))
sys.path.insert(0, ROOT)
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

from strategies import spot_replay as rp
strat = rp._strat


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
        self.assertEqual(res["cycles"], 2)          # the engine closes 2 cycles on this series
        self.assertEqual(res["wins"], 1)
        self.assertEqual(res["open_qty"], 0.0)      # position closed at the end
        self.assertLess(res["net"], res["realized"]) # net = gross - fees
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
            (100.0, 101.0, 99.0, 100.0),  # fills the entry
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

    def test_tp_dynamic_flat_requires_configured_trend_interval(self):
        with self.assertRaisesRegex(ValueError, "trend_interval"):
            rp.run_replay(
                _series(),
                _params(
                    tp_dynamic_flat=True,
                    trend_interval=240,
                ),
                fee_pct=0.26,
                bar_minutes=60,
            )

    def test_surge_and_hybrid_reentry_require_their_configured_trend_cadence(self):
        for feature in ("surge_guard", "reentry_hybrid_enabled"):
            for minutes in (None, 60):
                with self.subTest(feature=feature, minutes=minutes), self.assertRaisesRegex(
                    ValueError, "trend_interval",
                ):
                    rp.run_replay(
                        _series(), _params(**{feature: True}, trend_interval=240),
                        bar_minutes=minutes,
                    )

    def test_unused_surge_settings_do_not_restrict_replay_cadence(self):
        for overrides in (
            dict(surge_guard=False, surge_dynamic=True),
            dict(surge_guard=True, surge_window_hours=0.0),
            dict(surge_guard=True, surge_move_pct=0.0),
        ):
            with self.subTest(overrides=overrides):
                rp._validate_replay(_series(), _params(**overrides), bar_minutes=60)

    def test_long_surge_window_keeps_its_own_replay_history(self):
        strategy_type = rp._strat.Strategy
        created = []

        def capture_strategy(*args, **kwargs):
            strategy = strategy_type(*args, **kwargs)
            created.append(strategy)
            return strategy

        params = _params(surge_guard=True, surge_window_hours=96.0, trend_interval=60)
        bar = (100.0, 100.0, 100.0, 100.0)
        with patch.object(rp._strat, "Strategy", side_effect=capture_strategy):
            for warmup_count, expected_move in ((95, None), (96, 25.0)):
                with self.subTest(warmup_count=warmup_count):
                    created.clear()
                    rp.run_replay(
                        [(125.0, 125.0, 125.0, 125.0)], params,
                        bar_minutes=60, warmup_ohlc=[bar] * warmup_count,
                    )
                    self.assertTrue(created)
                    for strategy in created:
                        self.assertEqual(len(strategy._shadow_prices), 90)
                        move = strategy._surge_window_move_pct(125.0)
                        if expected_move is None:
                            self.assertIsNone(move)
                        else:
                            self.assertAlmostEqual(move, expected_move)

    def test_long_surge_window_can_arm_without_position_gain_threshold(self):
        params = _params(
            tp_trend_hold=True, takeprofit_pct=5.0, tp_trail_pct=8.0,
            surge_guard=True, surge_gain_pct=18.0, surge_move_pct=10.0,
            surge_window_hours=96.0, trend_interval=60,
        )
        client = MagicMock()
        client.pair_precision.return_value = rp.PairPrecision(5, 8, 0.0, "REPLAY")
        client.free_balance.return_value = 1.0
        strategy = rp._strat.Strategy(
            client, "REPLAY", params, dry_run=True,
            initial_state=rp._strat._new_state(), replay_mode=True,
        )
        strategy._save = lambda: None
        strategy.s.update(qty=1.0, cost=100.0, entry_price=100.0)
        strategy._shadow_prices.extend((index * 3600, 100.0) for index in range(96))
        strategy._surge_replay_closes.extend([100.0] * 96)

        strategy.step(115.0, timestamp=96 * 3600)

        self.assertEqual(len(strategy._shadow_prices), 90)
        self.assertTrue(strategy.s["surge_active"])

    def test_adaptive_features_require_their_configured_bar_interval(self):
        with self.assertRaisesRegex(ValueError, "tp_trail_vol_interval"):
            rp.run_replay(
                _series(), _params(surge_guard=True, surge_dynamic=True,
                                   trend_interval=60, tp_trail_vol_interval=240),
                bar_minutes=60,
            )
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

    def test_dca_drop_series_adapts_threshold_per_bar(self):
        bars = [
            (100.0, 100.5, 99.5, 100.0),
            (100.0, 100.5, 99.0, 100.0),  # fills ENTRY at 99.8
            (99.0, 99.0, 98.0, 98.0),     # price drops to 98.0 (1.8% below 99.8) -> places DCA limit at 97.8
            (98.0, 98.0, 97.0, 97.0),     # low 97.0 fills DCA limit if placed
        ]
        # Standard params: drop=1.0%. 98.0 is -1.8% -> triggers DCA BUY
        p_std = _params(dca_drop_pct=1.0)
        res_std = rp.run_replay(bars, p_std, fee_pct=0.26, bar_minutes=60)

        # Dynamic series with 3.0% drop on bar 2: 98.0 is NOT -3.0% -> skips DCA BUY
        res_dyn = rp.run_replay(
            bars, p_std, fee_pct=0.26, bar_minutes=60,
            dca_drop_series=[1.0, 1.0, 3.0, 3.0],
        )
        self.assertEqual(res_std["fills"], 2)  # ENTRY + DCA filled
        self.assertEqual(res_dyn["fills"], 1)  # Only initial ENTRY filled


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

    def test_make_kraken_replay_adapter_walk_forward_compatibility(self):
        from offline.backtests.evaluation import evaluate_segment
        params = _params()
        records = [
            {"timestamp": 1000 + i * 3600, "open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0}
            for i in range(10)
        ]
        adapter = rp.make_spot_replay_adapter(params, fee_pct=0.25, bar_minutes=60)
        res = evaluate_segment(records, adapter, validate_contract=True)
        self.assertIn("metrics", res)
        self.assertIn("return_pct", res["metrics"])
        self.assertIn("fills", res["metrics"])
        self.assertEqual(res["metrics"]["account_currency"], "USD")
        self.assertIn("ambiguous_bars", res["metrics"])

    def test_fetch_ohlc_candles_parsing_and_drop_forming_bar(self):
        import json
        from unittest.mock import MagicMock, patch
        from kraken.market_data import fetch_ohlc_candles
        from kraken.backtest import fetch_candles

        fake_payload = {
            "error": [],
            "result": {
                "HYPEUSD": [
                    ["1700000000", "20.0", "22.0", "19.5", "21.0", "100.0", "50", "20.5"],
                    ["1700003600", "21.0", "23.0", "20.5", "22.5", "120.0", "60", "21.5"],
                    ["1700007200", "22.5", "24.0", "22.0", "23.0", "80.0", "30", "22.8"],
                ],
                "last": 1700007200,
            },
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(fake_payload).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp):
            records_closed = fetch_ohlc_candles("HYPEUSD", 60, drop_incomplete=True)
            self.assertEqual(len(records_closed), 2)
            self.assertEqual(records_closed[0]["timestamp"], 1700000000)
            self.assertEqual(records_closed[0]["close"], 21.0)
            self.assertEqual(records_closed[1]["close"], 22.5)

            records_all = fetch_ohlc_candles("HYPEUSD", 60, drop_incomplete=False)
            self.assertEqual(len(records_all), 3)
            self.assertEqual(records_all[-1]["close"], 23.0)

            candles = fetch_candles("HYPEUSD", 60)
            self.assertEqual(len(candles), 2)
            self.assertEqual(candles[0], (20.0, 22.0, 19.5, 21.0))

    def test_backtest_helpers_guard_zero_budget_and_invalid_windows(self):
        import importlib.util
        from kraken.backtest_trail import bh

        hl_spec = importlib.util.spec_from_file_location(
            "hl_backtest_under_test", os.path.join(ROOT, "hyperliquid", "backtest.py"),
        )
        hl_bt = importlib.util.module_from_spec(hl_spec)
        hl_spec.loader.exec_module(hl_bt)

        self.assertEqual(bh([]), 0.0)
        self.assertEqual(bh([(0.0, 0.0, 0.0, 0.0)]), 0.0)
        self.assertEqual(
            hl_bt.precompute_signal([10.0, 11.0, 12.0], [1, 2, 3], "builtin", 0, 2, 0.01, 24, 8),
            ["neutral", "neutral", "neutral"],
        )


import contextlib
import csv
import hashlib
import io

DATASET = os.path.join(ROOT, "offline", "research", "hype_dataset",
                       "HYPEUSDC_240m_hlspot.csv")
BARS = 800
BUDGET = 3900.0
GOLDEN_ORDERS = 14
GOLDEN_TRACE_HASH = "69fd0a5053d74c58"
GOLDEN_METRICS = {
    "realized": 1844.33939624,
    "net": 1817.57411476,
    "fees": 26.76528149,
    "total": 1817.57411476,
    "final_upnl": 0.0,
    "cycles": 4,
    "wins": 4,
    "maxdd": 297.20377694,
    "open_qty": 0.0,
    "fills": 13,
}


def _base_v2_params():
    return strat.StratParams(
        currency="USD", entry_amount=650.0, entry_discount_pct=0.8,
        dca_amount=325.0, dca_drop_pct=1.25, check_minutes=2.0,
        takeprofit_pct=5.0, max_budget=BUDGET, max_dca_buys=10,
        enable_takeprofit=True, order_ttl_min=10.0, stop_loss_pct=12.5,
        adopt_cost=0.0, adopt_qty=0.0, reentry_drop_pct=2.2,
        reentry_tolerance_pct=0.05, reentry_adaptive=False,
        reentry_sl_bounce_pct=1.5, tp_tranches=[], tp_trend_hold=True,
        tp_trail_pct=3.0, sl_mhdsl_enabled=False,
    )


def _load_ohlc(n):
    with open(DATASET, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return [(float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"]))
            for r in rows][:n]


class BaseV2GoldenTest(unittest.TestCase):
    def _run_with_trace(self):
        ohlc = _load_ohlc(BARS)
        trace = []
        orig_place = strat.Strategy._place

        def _rec(self, side, vol, price, **kw):
            trace.append((side, kw.get("kind"),
                          None if price is None else round(price, 4),
                          round(vol, 6), bool(kw.get("market"))))
            return orig_place(self, side, vol, price, **kw)

        strat.Strategy._place = _rec
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                metrics = rp.run_replay(ohlc, _base_v2_params(),
                                        fee_pct=0.26, bar_minutes=240)
        finally:
            strat.Strategy._place = orig_place
        return trace, metrics

    def test_decision_trace_is_byte_identical(self):
        trace, _ = self._run_with_trace()
        self.assertEqual(len(trace), GOLDEN_ORDERS,
                         "the number of orders placed changed -> a behaviour regression")
        digest = hashlib.sha256(repr(trace).encode()).hexdigest()[:16]
        self.assertEqual(digest, GOLDEN_TRACE_HASH,
                         "the exact decision trace changed -> the refactor altered base v2")

    def test_final_metrics_match_golden(self):
        _, m = self._run_with_trace()
        for key, expected in GOLDEN_METRICS.items():
            got = m.get(key)
            if isinstance(expected, float):
                self.assertAlmostEqual(got, expected, places=6, msg=f"metric '{key}' differs")
            else:
                self.assertEqual(got, expected, f"metric '{key}' differs")

    def test_sell_first_ambiguous_bar_closes_cycle_without_orphan_dca_market_dump(self):
        bars = [
            (100.0, 101.0, 99.0, 100.0),   # decide ENTRY (~99.8)
            (100.0, 101.0, 99.0, 100.0),   # fill ENTRY, decide TP (~102.794)
            (98.0, 100.0, 97.5, 97.5),     # dip places DCA (~97.305) while TP stays open
            (100.0, 104.0, 96.0, 100.0),   # ambiguous bar: touches both TP and DCA
        ]
        res = rp.run_replay(
            bars, _params(max_dca_buys=2), fee_pct=0.26, bar_minutes=60,
            execution=rp.ExecutionModel(intrabar_policy="sell_first"),
            include_decision_trace=True,
        )
        self.assertEqual(res["ambiguous_bars"], 1)
        self.assertEqual(res["cycles"], 1)
        market_sells = [
            d for d in res["decision_trace"] if d["side"] == "sell" and d["market"]
        ]
        self.assertEqual(market_sells, [])

    def test_resized_final_dca_reserves_buy_fee_in_cash_account(self):
        bars = [
            (100.0, 101.0, 99.0, 100.0),
            (100.0, 101.0, 99.0, 100.0),  # fills ENTRY ($100 + $0.26 fee -> $49.74 cash left)
            (96.0, 97.0, 95.0, 96.0),     # dip places resized DCA within $49.74 cash + fee
            (96.0, 97.0, 94.0, 95.0),     # fills resized DCA cleanly without cash refusal
        ]
        res = rp.run_replay(
            bars, _params(entry_amount=100.0, dca_amount=50.0, max_budget=150.0, max_dca_buys=1),
            fee_pct=0.26, bar_minutes=60, initial_cash=150.0,
        )
        self.assertEqual(res["fills"], 2)
        self.assertEqual(res["funding"]["refused_buys"], 0)
        self.assertGreaterEqual(res["funding"]["final_cash"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

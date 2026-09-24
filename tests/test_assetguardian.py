import copy
import os
import unittest
from unittest import mock

os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

import assetguardian as ag
from providers.strategy_executor import OrderStatus


class MemoryState:
    def __init__(self, value=None):
        self.value = copy.deepcopy(value or {})

    def load(self):
        return copy.deepcopy(self.value)

    def save(self, value):
        self.value = copy.deepcopy(value)


class AssetGuardianTest(unittest.TestCase):
    def test_config_validation(self):
        with self.subTest(msg="test_required_config_rejects_missing_and_empty_values"):
            key = "AG_REQUIRED_CONFIG_REGRESSION_TEST"
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop(key, None)
                with self.assertRaisesRegex(ValueError, key):
                    ag._required_config(key)
            for value in ("", "   ", "\t"):
                with self.subTest(value=value), mock.patch.dict(os.environ, {key: value}):
                    with self.assertRaisesRegex(ValueError, key):
                        ag._required_config(key)

        with self.subTest(msg="test_required_numeric_config_has_no_implicit_fallback"):
            key = "AG_REQUIRED_NUMERIC_REGRESSION_TEST"
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop(key, None)
                with self.assertRaisesRegex(ValueError, key):
                    ag._required_float_config(key)

        with self.subTest(msg="test_config_preflight_reports_every_missing_setting"):
            env = {name: "1" for name in ag.REQUIRED_CONFIG_KEYS}
            env["AG_BUY_TIERS"] = ""
            env["AG_SELL_TIERS"] = "   "
            with mock.patch.dict(os.environ, env, clear=True):
                with self.assertRaises(ValueError) as raised:
                    ag._validate_required_config_presence()
            message = str(raised.exception)
            self.assertIn("AG_BUY_TIERS", message)
            self.assertIn("AG_SELL_TIERS", message)

        with self.subTest(msg="test_every_assetguardian_setting_is_registered_as_mandatory"):
            self.assertEqual(
                set(ag.REQUIRED_CONFIG_KEYS),
                {
                    "AG_CHECK_INTERVAL_SEC",
                    "AG_REFERENCE_MINUTES_BACK",
                    "AG_BUY_USE_CASH_RATIO",
                    "AG_BUY_TIERS",
                    "AG_SELL_TIERS",
                    "AG_SELL_REARM_GROWTH_PCT",
                    "AG_ORDER_MAX_AGE_SEC",
                    "AG_RECOVERY_RESET_PCT",
                    "AG_NEAR_TRIGGER_SEC",
                    "AG_ACTIVE_TRIGGER_SEC",
                    "AG_NEAR_TRIGGER_DISTANCE_PCT",
                    "AG_TREND_DEFER_MAX_SEC",
                    "AG_ORDER_MISSING_CONFIRMATIONS",
                },
            )


    def setUp(self):
        self.sell_provider = mock.Mock()
        self.sell_provider.open_orders.return_value = []
        self._real_trend_defer_ready = ag._trend_defer_ready
        self._trend_patch = mock.patch.object(
            ag, "_trend_defer_ready",
            side_effect=lambda _symbol, _kind, _side, _threshold, state:
                (True, state),
        )
        self._trend_patch.start()
        self.addCleanup(self._trend_patch.stop)
        ag._last_evaluation.clear()
        ag._last_evaluation.update({
            symbol: {"growth": None, "drawdown": None, "pending_tier": False}
            for symbol in ag.TRACKED_SYMBOLS
        })
        ag._local_price_samples.clear()
        ag._local_price_samples.update({symbol: [] for symbol in ag.TRACKED_SYMBOLS})

    def test_symbol_window_behaviors(self):
        with self.subTest(msg="test_symbol_window_normalizes_ms_and_returns_price_min_max"):
            now = 2_000_000_000
            rows = [
                [(now - 120) * 1000, 90],
                [(now - 60) * 1000, 110],
                [(now - 30) * 1000, 100],
            ]
            with mock.patch.object(ag, "_read_symbol_price_rows", return_value=rows), \
                 mock.patch.object(ag.time, "time", return_value=now):
                minimum, maximum = ag._get_symbol_window_extrema("TAOUSDC", 105, 10)
            self.assertEqual(minimum["price"], 90.0)
            self.assertEqual(maximum["price"], 110.0)

        with self.subTest(msg="test_symbol_window_skips_malformed_nonfinite_stale_and_future_rows"):
            now = 2_000_000_000
            rows = [
                ["bad", 1],
                [(now - 50) * 1000, "NaN"],
                [(now - 40) * 1000, "Infinity"],
                [(now - 700) * 1000, 10],
                [(now + 1) * 1000, 500],
                [(now - 30) * 1000, 100],
                [(now - 20) * 1000, 120],
            ]
            with mock.patch.object(ag, "_read_symbol_price_rows", return_value=rows), \
                 mock.patch.object(ag.time, "time", return_value=now):
                minimum, maximum = ag._get_symbol_window_extrema("TAOUSDC", 110, 10)
            self.assertEqual(minimum["price"], 100.0)
            self.assertEqual(maximum["price"], 120.0)

    def test_evaluation_triggers_behaviors(self):
        with self.subTest(msg="test_tao_drawdown_triggers_tao_buy_not_btc"):
            extrema = (
                {"timestamp": 1, "price": 224.0},
                {"timestamp": 2, "price": 242.0},
            )
            provider = mock.Mock()
            provider.free_balance.return_value = 1000.0
            state = MemoryState()
            with mock.patch.object(ag.api, "get_current_price", return_value=225.0), \
                 mock.patch.object(ag, "_get_symbol_window_extrema", return_value=extrema), \
                 mock.patch.object(ag, "STATE", state), \
                 mock.patch.object(ag.mkt, "provider_by_name", return_value=provider), \
                 mock.patch.object(
                     ag, "buy_with_all_cash",
                     return_value={"orderId": 701, "status": "NEW"}) as buy, \
                 mock.patch.object(ag, "sell_asset") as sell:
                self.assertTrue(ag.evaluate_symbol("TAOUSDC"))
            buy.assert_called_once()
            self.assertEqual(buy.call_args.kwargs["buy_symbol"], "TAOUSDC")
            self.assertEqual(buy.call_args.kwargs["cash_amount"], 348.25)
            self.assertTrue(
                buy.call_args.kwargs["client_order_id"].startswith("AGB"))
            sell.assert_not_called()
            buy_state = state.value["symbols"]["TAOUSDC"]["buy"]
            self.assertEqual(buy_state["completed_tiers"], [])
            self.assertEqual(buy_state["pending"]["order_id"], "701")

        with self.subTest(msg="test_two_asset_replay_routes_only_tao_drawdown_to_tao"):
            state = MemoryState()
            provider = mock.Mock()
            provider.free_balance.return_value = 1000.0
            prices = {"BTCUSDC": 100.0, "TAOUSDC": 225.0}
            extrema = {
                "BTCUSDC": (
                    {"timestamp": 1, "price": 99.0},
                    {"timestamp": 2, "price": 101.0},
                ),
                "TAOUSDC": (
                    {"timestamp": 1, "price": 224.0},
                    {"timestamp": 2, "price": 242.0},
                ),
            }
            with mock.patch.object(
                    ag.api, "get_current_price", side_effect=lambda symbol: prices[symbol]), \
                 mock.patch.object(
                     ag, "_get_symbol_window_extrema",
                     side_effect=lambda symbol, *_args, **_kwargs: extrema[symbol]), \
                 mock.patch.object(ag, "STATE", state), \
                 mock.patch.object(ag.mkt, "provider_by_name", return_value=provider), \
                 mock.patch.object(
                     ag, "buy_with_all_cash",
                     return_value={"orderId": 702, "status": "NEW"}) as buy, \
                 mock.patch.object(ag, "sell_asset") as sell:
                self.assertTrue(ag.evaluate_and_maybe_sell_or_buy(
                    symbols=("BTCUSDC", "TAOUSDC")))
            buy.assert_called_once()
            self.assertEqual(buy.call_args.kwargs["buy_symbol"], "TAOUSDC")
            self.assertEqual(buy.call_args.kwargs["cash_amount"], 348.25)
            sell.assert_not_called()

        with self.subTest(msg="test_default_15pct_tao_growth_selects_first_sell_tier"):
            extrema = (
                {"timestamp": 1, "price": 100.0},
                {"timestamp": 2, "price": 116.0},
            )
            state = MemoryState()
            provider = mock.Mock()
            provider.free_balance.return_value = 10.0
            provider.open_orders.return_value = []
            with mock.patch.object(ag.api, "get_current_price", return_value=115.0), \
                 mock.patch.object(ag, "_get_symbol_window_extrema", return_value=extrema), \
                 mock.patch.object(ag, "STATE", state), \
                 mock.patch.object(ag.mkt, "provider_by_name", return_value=provider), \
                 mock.patch.object(ag, "_submit_sell_tier", return_value=True) as submit, \
                 mock.patch.object(ag, "buy_with_all_cash") as buy:
                self.assertTrue(ag.evaluate_symbol("TAOUSDC"))
            tier = submit.call_args.kwargs["tier"]
            self.assertEqual(tier, (15.0, 0.30, 3.0))
            self.assertEqual(submit.call_args.kwargs["current_price"], 115.0)
            self.assertEqual(
                state.value["symbols"]["TAOUSDC"]["sell"]["trough_price"], 100.0)
            buy.assert_not_called()

        with self.subTest(msg="test_nonfinite_current_price_cannot_trigger_any_order"):
            with mock.patch.object(ag.api, "get_current_price", return_value=float("inf")), \
                 mock.patch.object(ag, "sell_asset") as sell, \
                 mock.patch.object(ag, "buy_with_all_cash") as buy:
                self.assertFalse(ag.evaluate_symbol("TAOUSDC"))
            sell.assert_not_called()
            buy.assert_not_called()

        with self.subTest(msg="test_insufficient_price_baseline_cannot_trigger_any_order"):
            now = 2_000_000
            with mock.patch.object(ag, "_read_symbol_price_rows", return_value=[]), \
                 mock.patch.object(ag.time, "time", return_value=now), \
                 mock.patch.object(ag.api, "get_current_price", return_value=225.0), \
                 mock.patch.object(ag, "sell_asset") as sell, \
                 mock.patch.object(ag, "buy_with_all_cash") as buy:
                self.assertFalse(ag.evaluate_symbol("TAOUSDC"))
            sell.assert_not_called()
            buy.assert_not_called()


    def test_campaign_tier_tracking_behaviors(self):
        with self.subTest(msg="test_completed_tier_is_isolated_per_symbol"):
            state = MemoryState({
                "version": 2,
                "symbols": {
                    "TAOUSDC": {
                        "peak_price": 242, "peak_ts": 2, "initial_cash": 1000,
                        "completed_tiers": [7],
                    }
                },
            })
            maximum = {"timestamp": 2, "price": 242}
            with mock.patch.object(ag, "STATE", state):
                tao_tier, _ = ag._campaign_tier("TAOUSDC", 8, maximum, 650)
                btc_tier, _ = ag._campaign_tier("BTCUSDC", 8, maximum, 650)
                root = ag._load_state_root()
            self.assertIsNone(tao_tier)
            self.assertEqual(btc_tier, (7.0, 0.35))
            self.assertEqual(
                root["symbols"]["TAOUSDC"]["buy"]["completed_tiers"], [7])

        with self.subTest(msg="test_higher_peak_does_not_repeat_completed_tier_or_replenish_budget"):
            state = MemoryState({
                "version": 2,
                "symbols": {
                    "TAOUSDC": {
                        "peak_price": 242, "peak_ts": 2, "initial_cash": 1000,
                        "completed_tiers": [7],
                    }
                },
            })
            with mock.patch.object(ag, "STATE", state):
                tier, campaign = ag._campaign_tier(
                    "TAOUSDC", 10.5, {"timestamp": 3, "price": 250}, 500)
            self.assertEqual(tier, (10.0, 0.35))
            self.assertEqual(campaign["completed_tiers"], [7])
            self.assertEqual(campaign["initial_cash"], 1000)
            self.assertEqual(campaign["peak_price"], 250)

        with self.subTest(msg="test_guard_refused_shallow_tier_does_not_block_deeper_tiers"):
            # Tier 7 respects the profit guard and keeps being refused (never completed).
            # Deeper tiers are designed to bypass the guard, so crossing them must pick
            # them; walking tiers from the shallowest picked 7 forever.
            state = MemoryState({
                "version": 2,
                "symbols": {
                    "TAOUSDC": {
                        "peak_price": 324.1, "peak_ts": 2, "initial_cash": 1000,
                        "completed_tiers": [], "attempts_by_tier": {"7": 287},
                    }
                },
            })
            maximum = {"timestamp": 2, "price": 324.1}
            with mock.patch.object(ag, "STATE", state):
                shallow, _ = ag._campaign_tier("TAOUSDC", 9.6, maximum, 1000)
                middle, _ = ag._campaign_tier("TAOUSDC", 10.5, maximum, 1000)
                deepest, _ = ag._campaign_tier("TAOUSDC", 14.2, maximum, 1000)
            self.assertEqual(shallow, (7.0, 0.35))
            self.assertEqual(middle, (10.0, 0.35))
            self.assertEqual(deepest, (14.0, 0.30))
            self.assertEqual(ag.GUARDED_BUY_THRESHOLD, 7.0)

        with self.subTest(msg="test_legacy_global_campaign_maps_only_to_btc"):
            state = MemoryState({
                "peak_value": 100,
                "peak_ts": 2,
                "initial_cash": 1000,
                "completed_tiers": [7],
            })
            with mock.patch.object(ag, "STATE", state):
                root = ag._load_state_root()
                tier, campaign = ag._campaign_tier(
                    ag.LEGACY_BUY_SYMBOL, 8,
                    {"timestamp": 3, "price": 100}, 500)
            self.assertIn(ag.LEGACY_BUY_SYMBOL, root["symbols"])
            self.assertNotIn("TAOUSDC", root["symbols"])
            self.assertIsNone(tier)
            self.assertEqual(campaign["completed_tiers"], [7.0])

    def test_evaluation_cycle_and_interval_behaviors(self):
        with self.subTest(msg="test_same_tao_tier_is_not_repeated"):
            state = MemoryState()
            extrema = (
                {"timestamp": 1, "price": 224.0},
                {"timestamp": 2, "price": 242.0},
            )
            provider = mock.Mock()
            provider.free_balance.return_value = 1000.0
            with mock.patch.object(ag.api, "get_current_price", return_value=225.0), \
                 mock.patch.object(ag, "_get_symbol_window_extrema", return_value=extrema), \
                 mock.patch.object(ag, "STATE", state), \
                 mock.patch.object(ag.mkt, "provider_by_name", return_value=provider), \
                 mock.patch.object(
                     ag, "buy_with_all_cash",
                     return_value={"orderId": 703, "status": "NEW"}) as buy, \
                 mock.patch.object(
                     ag.mkt, "order_status",
                     return_value=OrderStatus("open", 0.0, 0.0, 0.0)):
                self.assertTrue(ag.evaluate_symbol("TAOUSDC"))
                self.assertFalse(ag.evaluate_symbol("TAOUSDC"))
            self.assertEqual(buy.call_count, 1)

        with self.subTest(msg="test_refused_tier_is_recalculated_and_uses_fast_interval"):
            state = MemoryState()
            extrema = (
                {"timestamp": 1, "price": 224.0},
                {"timestamp": 2, "price": 242.0},
            )
            provider = mock.Mock()
            provider.free_balance.return_value = 1000.0
            with mock.patch.object(ag.api, "get_current_price", return_value=225.0), \
                 mock.patch.object(ag, "_get_symbol_window_extrema", return_value=extrema), \
                 mock.patch.object(ag, "STATE", state), \
                 mock.patch.object(ag.mkt, "provider_by_name", return_value=provider), \
                 mock.patch.object(ag, "buy_with_all_cash", return_value=False):
                self.assertFalse(ag.evaluate_symbol("TAOUSDC"))
                self.assertEqual(ag._next_check_seconds(), ag.ACTIVE_TRIGGER_SECONDS)
            self.assertEqual(
                state.value["symbols"]["TAOUSDC"]["buy"]["completed_tiers"], [])

        with self.subTest(msg="test_near_tier_interval_is_computed_per_symbol"):
            state = MemoryState({
                "version": 3,
                "symbols": {"TAOUSDC": {"buy": {"completed_tiers": [7]}}},
            })
            ag._last_evaluation["TAOUSDC"] = {
                "growth": 0.0, "drawdown": -8.5, "pending_tier": False,
            }
            with mock.patch.object(ag, "STATE", state):
                self.assertEqual(ag._next_check_seconds(), ag.NEAR_TRIGGER_SECONDS)

        with self.subTest(msg="test_evaluator_accepts_at_most_one_order_per_cycle"):
            with mock.patch.object(ag, "evaluate_symbol", side_effect=[True, True]) as evaluate:
                self.assertTrue(ag.evaluate_and_maybe_sell_or_buy(
                    symbols=("BTCUSDC", "TAOUSDC")))
            self.assertEqual(evaluate.call_count, 1)

    def test_buy_campaign_lifecycle_and_reconciliation(self):
        with self.subTest(msg="test_guardian_owns_retry_for_buy"):
            provider = mock.Mock()
            provider.free_balance.return_value = 1000.0
            with mock.patch.object(ag.mkt, "provider_by_name", return_value=provider), \
                 mock.patch.object(ag.api, "get_current_price", return_value=100.0), \
                 mock.patch.object(ag.mkt, "place", return_value={"orderId": 7}) as place:
                self.assertTrue(ag.buy_with_all_cash("TAOUSDC", cash_ratio=0.5))
            self.assertTrue(place.call_args.kwargs["caller_owns_retry"])
            self.assertTrue(place.call_args.kwargs["bypass_profit_reference"])
            self.assertFalse(place.call_args.kwargs["wait_for_trend"])
            self.assertNotIn("bypass_profit_guard", place.call_args.kwargs)
            self.assertEqual(place.call_args.kwargs["motivation"],
                             "assetguardian_drawdown_buy")
            self.assertEqual(place.call_args.args[0], "TAOUSDC")

        with self.subTest(msg="test_buy_submit_acceptance_is_pending_until_terminal_fill"):
            state = MemoryState()
            campaign = {
                "peak_price": 242.0,
                "peak_ts": 2.0,
                "initial_cash": 1000.0,
                "completed_tiers": [],
            }
            with mock.patch.object(ag, "STATE", state), \
                 mock.patch.object(
                     ag, "buy_with_all_cash",
                     return_value={"orderId": 710, "status": "NEW", "origQty": "1.5"}), \
                 mock.patch.object(ag.mkt, "order_status") as status:
                accepted = ag._submit_buy_tier(
                    "TAOUSDC", 225.0, 348.25, (7.0, 0.35), campaign)

            self.assertTrue(accepted)
            status.assert_not_called()
            buy_state = state.value["symbols"]["TAOUSDC"]["buy"]
            self.assertEqual(buy_state["completed_tiers"], [])
            self.assertEqual(buy_state["pending"]["order_id"], "710")

        with self.subTest(msg="test_buy_terminal_fill_completes_tier_and_records_actual_cost"):
            state = MemoryState({
                "version": 3,
                "symbols": {"TAOUSDC": {"buy": {
                    "peak_price": 242.0, "peak_ts": 2.0,
                    "initial_cash": 1000.0, "completed_tiers": [],
                    "pending": {
                        "threshold": 7.0, "allocation": 0.35,
                        "client_order_id": "AGB_terminal", "order_id": "711",
                        "requested_qty": 1.5, "created_at": 10.0,
                    },
                }}},
            })
            with mock.patch.object(ag, "STATE", state), \
                 mock.patch.object(
                     ag.mkt, "order_status",
                     return_value=OrderStatus("closed", 1.5, 337.5, 0.4)):
                campaign = ag._symbol_campaign("TAOUSDC", "buy")
                campaign, outcome = ag._reconcile_pending_buy("TAOUSDC", campaign)

            self.assertEqual(outcome, "terminal")
            self.assertNotIn("pending", campaign)
            self.assertEqual(campaign["completed_tiers"], [7.0])
            self.assertAlmostEqual(campaign["spent_quote_by_tier"]["7"], 337.9)

        with self.subTest(msg="test_buy_terminal_partial_retries_only_remaining_quote_budget"):
            state = MemoryState({
                "version": 3,
                "symbols": {"TAOUSDC": {"buy": {
                    "peak_price": 242.0, "peak_ts": 2.0,
                    "initial_cash": 1000.0, "completed_tiers": [],
                    "pending": {
                        "threshold": 7.0, "allocation": 0.35,
                        "client_order_id": "AGB_partial", "order_id": "712",
                        "requested_qty": 1.5, "created_at": 10.0,
                    },
                }}},
            })
            with mock.patch.object(ag, "STATE", state), \
                 mock.patch.object(
                     ag.mkt, "order_status",
                     return_value=OrderStatus("canceled", 0.5, 112.5, 0.2)):
                campaign = ag._symbol_campaign("TAOUSDC", "buy")
                campaign, outcome = ag._reconcile_pending_buy("TAOUSDC", campaign)
                remaining, target, spent = ag._buy_remaining_cash(
                    campaign, 7.0, 0.35)

            self.assertEqual(outcome, "terminal")
            self.assertEqual(ag._completed_tier_values(campaign), set())
            self.assertAlmostEqual(target, 348.25)
            self.assertAlmostEqual(spent, 112.7)
            self.assertAlmostEqual(remaining, 235.55)

        with self.subTest(msg="test_buy_ambiguous_submit_confirmed_absent_is_not_marked_complete"):
            state = MemoryState()
            campaign = {
                "peak_price": 242.0,
                "peak_ts": 2.0,
                "initial_cash": 1000.0,
                "completed_tiers": [],
            }
            with mock.patch.object(ag, "STATE", state), \
                 mock.patch.object(ag, "buy_with_all_cash", return_value=None), \
                 mock.patch.object(ag.mkt, "order_by_client_id", return_value=None):
                self.assertFalse(ag._submit_buy_tier(
                    "TAOUSDC", 225.0, 348.25, (7.0, 0.35), campaign))
                buy_state = ag._symbol_campaign("TAOUSDC", "buy")
                buy_state, outcome = ag._reconcile_pending_buy("TAOUSDC", buy_state)

            self.assertEqual(outcome, "retryable")
            self.assertNotIn("pending", buy_state)
            self.assertEqual(ag._completed_tier_values(buy_state), set())

        with self.subTest(msg="test_confirmed_missing_buy_is_revalidated_and_resubmitted_same_cycle_same_id"):
            campaign = {
                "peak_price": 242.0,
                "peak_ts": 2.0,
                "initial_cash": 1000.0,
                "completed_tiers": [],
                "spent_quote_by_tier": {},
                "attempts_by_tier": {"7": 1},
            }
            client_id = ag._buy_client_order_id("TAOUSDC", campaign, 7.0, 1)
            campaign["pending"] = {
                "intent_id": client_id,
                "client_order_id": client_id,
                "symbol": "TAOUSDC",
                "side": "BUY",
                "kind": "ASSET_GUARDIAN_BUY_TIER",
                "threshold": 7.0,
                "allocation": 0.35,
                "requested_qty": 1.5,
                "requested_price": 225.0,
                "attempt": 1,
                "created_at": 1000.0,
            }
            state = MemoryState({
                "version": 3,
                "symbols": {"TAOUSDC": {"buy": campaign}},
            })
            extrema = (
                {"timestamp": 1.0, "price": 224.0},
                {"timestamp": 2.0, "price": 242.0},
            )
            provider = mock.Mock()
            provider.free_balance.return_value = 1000.0
            provider.open_orders.return_value = []
            with mock.patch.object(ag, "STATE", state), \
                 mock.patch.object(ag.mkt, "order_by_client_id", return_value=None), \
                 mock.patch.object(ag.api, "get_current_price", return_value=225.0), \
                 mock.patch.object(ag, "_get_symbol_window_extrema", return_value=extrema), \
                 mock.patch.object(ag.mkt, "provider_by_name", return_value=provider), \
                 mock.patch.object(
                     ag, "buy_with_all_cash",
                     return_value={"orderId": 713, "status": "NEW"}) as buy:
                accepted = ag.evaluate_symbol("TAOUSDC")

            self.assertTrue(accepted)
            buy.assert_called_once()
            self.assertEqual(buy.call_args.kwargs["client_order_id"], client_id)
            retried = state.value["symbols"]["TAOUSDC"]["buy"]
            self.assertEqual(retried["attempts_by_tier"]["7"], 1)
            self.assertEqual(retried["pending"]["order_id"], "713")
            self.assertEqual(retried["completed_tiers"], [])

        with self.subTest(msg="test_trend_wait_is_persisted_and_rechecked_without_sleep"):
            state = MemoryState({
                "version": 3,
                "symbols": {"TAOUSDC": {"buy": {"completed_tiers": []}}},
            })
            manager = mock.Mock()
            manager.should_wait.return_value = True
            initial = {"completed_tiers": []}
            with mock.patch.object(ag, "STATE", state), \
                 mock.patch.object(ag.cm, "get_short_trend_manager", return_value=manager), \
                 mock.patch.object(ag.time, "time", return_value=1000.0), \
                 mock.patch.object(ag.time, "sleep") as sleep:
                ready1, deferred = self._real_trend_defer_ready(
                    "TAOUSDC", "buy", "BUY", 7.0, initial)
            with mock.patch.object(ag, "STATE", state), \
                 mock.patch.object(ag.cm, "get_short_trend_manager", return_value=manager), \
                 mock.patch.object(
                     ag.time, "time",
                     return_value=1000.0 + ag.TREND_DEFER_MAX_SECONDS + 1), \
                 mock.patch.object(ag.time, "sleep") as sleep2:
                ready2, released = self._real_trend_defer_ready(
                    "TAOUSDC", "buy", "BUY", 7.0, deferred)

            self.assertFalse(ready1)
            self.assertTrue(ready2)
            self.assertNotIn("trend_defer", released)
            sleep.assert_not_called()
            sleep2.assert_not_called()

    def test_sell_campaign_initialization_and_constraints(self):
        with self.subTest(msg="test_guardian_owns_retry_for_asset_specific_sell"):
            provider = mock.Mock()
            provider.free_balance.return_value = 0.5
            with mock.patch.object(ag.mkt, "provider_by_name", return_value=provider), \
                 mock.patch.object(ag.mkt, "place", return_value={"orderId": 8}) as place:
                order = ag.sell_asset(
                    "TAOUSDC", qty=0.2, current_price=250.0,
                    client_order_id="AGS_test", tier_threshold=15.0)
            self.assertEqual(order, {"orderId": 8})
            self.assertTrue(place.call_args.kwargs["caller_owns_retry"])
            self.assertTrue(place.call_args.kwargs["bypass_quantity_policy"])
            self.assertNotIn("bypass_profit_reference", place.call_args.kwargs)
            self.assertNotIn("bypass_profit_guard", place.call_args.kwargs)
            self.assertEqual(place.call_args.kwargs["motivation"],
                             "assetguardian_growth_exit_tier_15")
            self.assertEqual(place.call_args.kwargs["client_order_id"], "AGS_test")
            self.assertEqual(place.call_args.args[:4], ("TAOUSDC", "SELL", 250.0, 0.2))

        with self.subTest(msg="test_sell_campaign_freezes_trough_and_initial_quantity"):
            state = MemoryState()
            with mock.patch.object(ag, "STATE", state):
                tier1, campaign, growth1 = ag._sell_campaign_tier(
                    "TAOUSDC", 115.0, {"timestamp": 10, "price": 100.0}, 10.0,
                    self.sell_provider)
                ag._complete_campaign_tier("TAOUSDC", "sell", campaign, 15.0)
                tier2, campaign2, growth2 = ag._sell_campaign_tier(
                    "TAOUSDC", 126.0, {"timestamp": 20, "price": 120.0}, 7.0,
                    self.sell_provider)

            self.assertEqual(tier1, (15.0, 0.30, 3.0))
            self.assertEqual(growth1, 15.0)
            self.assertEqual(tier2, (25.0, 0.30, 3.0))
            self.assertAlmostEqual(growth2, 26.0)
            self.assertEqual(campaign2["trough_price"], 100.0)
            self.assertEqual(campaign2["initial_qty"], 10.0)

        with self.subTest(msg="test_sell_jump_crosses_only_one_incomplete_tier_at_a_time"):
            state = MemoryState()
            with mock.patch.object(ag, "STATE", state):
                tier1, campaign, _ = ag._sell_campaign_tier(
                    "TAOUSDC", 136.0, {"timestamp": 10, "price": 100.0}, 10.0,
                    self.sell_provider)
                ag._complete_campaign_tier("TAOUSDC", "sell", campaign, 15.0)
                tier2, campaign, _ = ag._sell_campaign_tier(
                    "TAOUSDC", 136.0, {"timestamp": 10, "price": 100.0}, 7.0,
                    self.sell_provider)
                ag._complete_campaign_tier("TAOUSDC", "sell", campaign, 25.0)
                tier3, _, _ = ag._sell_campaign_tier(
                    "TAOUSDC", 136.0, {"timestamp": 10, "price": 100.0}, 4.0,
                    self.sell_provider)

            self.assertEqual(tier1, (15.0, 0.30, 3.0))
            self.assertEqual(tier2, (25.0, 0.30, 3.0))
            self.assertEqual(tier3, (35.0, 0.40, 4.0))

        with self.subTest(msg="test_sell_rearm_uses_frozen_trough_not_new_rolling_minimum"):
            state = MemoryState({
                "version": 3,
                "symbols": {"TAOUSDC": {"sell": {
                    "trough_price": 100.0, "trough_ts": 10,
                    "initial_qty": 10.0, "completed_tiers": [15.0],
                }}},
            })
            with mock.patch.object(ag, "STATE", state):
                tier, campaign, growth = ag._sell_campaign_tier(
                    "TAOUSDC", 105.0,
                    {"timestamp": 20, "price": 80.0}, 7.0,
                    self.sell_provider)

            self.assertIsNone(tier)
            self.assertEqual(campaign, {})
            self.assertEqual(growth, 5.0)
            self.assertNotIn("TAOUSDC", state.value.get("symbols", {}))

        with self.subTest(msg="test_existing_exchange_sell_blocks_new_campaign_and_duplicate"):
            state = MemoryState()
            provider = mock.Mock()
            provider.open_orders.return_value = [
                {"orderId": "orphan-1", "side": "SELL", "status": "NEW"}
            ]
            with mock.patch.object(ag, "STATE", state):
                tier, campaign, growth = ag._sell_campaign_tier(
                    "TAOUSDC", 115.0, {"timestamp": 10, "price": 100.0}, 10.0,
                    provider)

            self.assertIsNone(tier)
            self.assertEqual(campaign, {})
            self.assertEqual(growth, 15.0)
            self.assertEqual(state.value, {})

        with self.subTest(msg="test_zero_free_qty_with_open_sell_preserves_campaign"):
            existing = {
                "trough_price": 100.0, "trough_ts": 10,
                "initial_qty": 10.0, "completed_tiers": [15.0],
            }
            state = MemoryState({
                "version": 3,
                "symbols": {"TAOUSDC": {"sell": existing}},
            })
            provider = mock.Mock()
            provider.open_orders.return_value = [
                {"orderId": "locked-1", "side": "SELL", "status": "NEW"}
            ]
            with mock.patch.object(ag, "STATE", state):
                tier, campaign, growth = ag._sell_campaign_tier(
                    "TAOUSDC", 126.0, {"timestamp": 10, "price": 100.0}, 0.0,
                    provider)

            self.assertIsNone(tier)
            self.assertIsNone(growth)
            self.assertEqual(campaign["completed_tiers"], [15.0])
            self.assertIn("TAOUSDC", state.value["symbols"])

        with self.subTest(msg="test_open_sell_blocks_next_tier_of_existing_campaign"):
            state = MemoryState({
                "version": 3,
                "symbols": {"TAOUSDC": {"sell": {
                    "trough_price": 100.0, "trough_ts": 10,
                    "initial_qty": 10.0, "completed_tiers": [15.0],
                    "total_filled_qty": 3.0,
                }}},
            })
            provider = mock.Mock()
            provider.open_orders.return_value = [
                {"orderId": "other-2", "side": "SELL", "status": "NEW"}
            ]
            with mock.patch.object(ag, "STATE", state):
                tier, campaign, growth = ag._sell_campaign_tier(
                    "TAOUSDC", 126.0, {"timestamp": 20, "price": 110.0}, 7.0,
                    provider)

            self.assertIsNone(tier)
            self.assertEqual(growth, 26.0)
            self.assertEqual(campaign["completed_tiers"], [15.0])

        with self.subTest(msg="test_reconciled_balance_increase_rearms_sell_campaign"):
            state = MemoryState({
                "version": 3,
                "symbols": {"TAOUSDC": {"sell": {
                    "trough_price": 100.0, "trough_ts": 10,
                    "initial_qty": 10.0, "completed_tiers": [15.0],
                    "total_filled_qty": 3.0,
                }}},
            })
            with mock.patch.object(ag, "STATE", state):
                tier, campaign, growth = ag._sell_campaign_tier(
                    "TAOUSDC", 126.0, {"timestamp": 20, "price": 110.0}, 8.0,
                    self.sell_provider)

            self.assertIsNone(tier)
            self.assertEqual(growth, 26.0)
            self.assertEqual(campaign, {})
            self.assertNotIn("TAOUSDC", state.value.get("symbols", {}))

    def test_sell_order_reconciliation_and_lifecycle(self):
        with self.subTest(msg="test_open_sell_order_stays_pending_and_does_not_complete_tier"):
            state = MemoryState({
                "version": 3,
                "symbols": {"TAOUSDC": {"sell": {
                    "trough_price": 100.0, "trough_ts": 10,
                    "initial_qty": 10.0, "completed_tiers": [],
                    "pending": {
                        "threshold": 15.0, "client_order_id": "AGS_open",
                        "order_id": "77", "requested_qty": 3.0,
                    },
                }}},
            })
            with mock.patch.object(ag, "STATE", state), \
                 mock.patch.object(
                     ag.mkt, "order_status",
                     return_value=OrderStatus("open", 1.0, 115.0, 0.1)):
                campaign = ag._symbol_campaign("TAOUSDC", "sell")
                campaign, outcome = ag._reconcile_pending_sell("TAOUSDC", campaign)

            self.assertEqual(outcome, "active")
            self.assertEqual(campaign["pending"]["filled_qty"], 1.0)
            self.assertEqual(ag._completed_tier_values(campaign), set())

        with self.subTest(msg="test_owned_sell_over_ttl_is_canceled_once_and_partial_is_reconciled"):
            created_at = 10_000.0
            state = MemoryState({
                "version": 3,
                "symbols": {"TAOUSDC": {"sell": {
                    "trough_price": 100.0, "trough_ts": 10,
                    "initial_qty": 10.0, "completed_tiers": [],
                    "pending": {
                        "threshold": 15.0, "client_order_id": "AGS_expired",
                        "order_id": "ttl-1", "requested_qty": 3.0,
                        "created_at": created_at,
                    },
                }}},
            })
            statuses = (
                OrderStatus("open", 1.0, 115.0, 0.1),
                OrderStatus("canceled", 1.0, 115.0, 0.1),
            )
            with mock.patch.object(ag, "STATE", state), \
                 mock.patch.object(
                     ag.time, "time",
                     return_value=created_at + ag.ORDER_MAX_AGE_SECONDS + 1), \
                 mock.patch.object(
                     ag.mkt, "order_status", side_effect=statuses) as status_call, \
                 mock.patch.object(ag.mkt, "cancel_order") as cancel:
                campaign = ag._symbol_campaign("TAOUSDC", "sell")
                campaign, outcome = ag._reconcile_pending_sell("TAOUSDC", campaign)

            self.assertEqual(outcome, "terminal")
            cancel.assert_called_once_with(
                "TAOUSDC", "ttl-1", provider_name="binance")
            self.assertEqual(status_call.call_count, 2)
            self.assertNotIn("pending", campaign)
            self.assertEqual(campaign["filled_qty_by_tier"]["15"], 1.0)
            self.assertEqual(ag._completed_tier_values(campaign), set())

        with self.subTest(msg="test_owned_sell_under_ttl_is_not_canceled"):
            created_at = 20_000.0
            state = MemoryState({
                "version": 3,
                "symbols": {"TAOUSDC": {"sell": {
                    "trough_price": 100.0, "trough_ts": 10,
                    "initial_qty": 10.0, "completed_tiers": [],
                    "pending": {
                        "threshold": 15.0, "client_order_id": "AGS_young",
                        "order_id": "ttl-2", "requested_qty": 3.0,
                        "created_at": created_at,
                    },
                }}},
            })
            with mock.patch.object(ag, "STATE", state), \
                 mock.patch.object(ag.time, "time", return_value=created_at + 30), \
                 mock.patch.object(
                     ag.mkt, "order_status",
                     return_value=OrderStatus("open", 0.0, 0.0, 0.0)), \
                 mock.patch.object(ag.mkt, "cancel_order") as cancel:
                campaign = ag._symbol_campaign("TAOUSDC", "sell")
                campaign, outcome = ag._reconcile_pending_sell("TAOUSDC", campaign)

            self.assertEqual(outcome, "active")
            self.assertNotIn("cancel_attempted_at", campaign["pending"])
            cancel.assert_not_called()

        with self.subTest(msg="test_ambiguous_ttl_cancel_is_not_repeated_and_blocks_replacement"):
            created_at = 30_000.0
            state = MemoryState({
                "version": 3,
                "symbols": {"TAOUSDC": {"sell": {
                    "trough_price": 100.0, "trough_ts": 10,
                    "initial_qty": 10.0, "completed_tiers": [],
                    "pending": {
                        "threshold": 15.0, "client_order_id": "AGS_ambiguous_cancel",
                        "order_id": "ttl-3", "requested_qty": 3.0,
                        "created_at": created_at,
                    },
                }}},
            })
            with mock.patch.object(ag, "STATE", state), \
                 mock.patch.object(
                     ag.time, "time",
                     return_value=created_at + ag.ORDER_MAX_AGE_SECONDS + 1), \
                 mock.patch.object(
                     ag.mkt, "order_status",
                     return_value=OrderStatus("open", 0.0, 0.0, 0.0)), \
                 mock.patch.object(
                     ag.mkt, "cancel_order", side_effect=RuntimeError("timeout")) as cancel:
                campaign = ag._symbol_campaign("TAOUSDC", "sell")
                campaign, outcome1 = ag._reconcile_pending_sell("TAOUSDC", campaign)
                campaign, outcome2 = ag._reconcile_pending_sell("TAOUSDC", campaign)

            self.assertEqual((outcome1, outcome2), ("active", "active"))
            self.assertEqual(cancel.call_count, 1)
            self.assertIn("cancel_attempted_at", campaign["pending"])
            self.assertEqual(ag._completed_tier_values(campaign), set())

        with self.subTest(msg="test_filled_sell_order_completes_tier_only_after_terminal_status"):
            state = MemoryState({
                "version": 3,
                "symbols": {"TAOUSDC": {"sell": {
                    "trough_price": 100.0, "trough_ts": 10,
                    "initial_qty": 10.0, "completed_tiers": [],
                    "pending": {
                        "threshold": 15.0, "client_order_id": "AGS_filled",
                        "order_id": "78", "requested_qty": 3.0,
                    },
                }}},
            })
            with mock.patch.object(ag, "STATE", state), \
                 mock.patch.object(
                     ag.mkt, "order_status",
                     return_value=OrderStatus("closed", 3.0, 345.0, 0.2)):
                campaign = ag._symbol_campaign("TAOUSDC", "sell")
                campaign, outcome = ag._reconcile_pending_sell("TAOUSDC", campaign)

            self.assertEqual(outcome, "terminal")
            self.assertEqual(campaign["completed_tiers"], [15.0])
            self.assertNotIn("pending", campaign)
            self.assertEqual(campaign["terminal_orders"][-1]["order_id"], "78")

        with self.subTest(msg="test_terminal_partial_sell_retries_only_unfilled_remainder"):
            state = MemoryState({
                "version": 3,
                "symbols": {"TAOUSDC": {"sell": {
                    "trough_price": 100.0, "trough_ts": 10,
                    "initial_qty": 10.0, "completed_tiers": [],
                    "pending": {
                        "threshold": 15.0, "client_order_id": "AGS_partial",
                        "order_id": "79", "requested_qty": 3.0,
                    },
                }}},
            })
            with mock.patch.object(ag, "STATE", state), \
                 mock.patch.object(
                     ag.mkt, "order_status",
                     return_value=OrderStatus("canceled", 1.0, 115.0, 0.1)):
                campaign = ag._symbol_campaign("TAOUSDC", "sell")
                campaign, outcome = ag._reconcile_pending_sell("TAOUSDC", campaign)
                tier, _, _ = ag._sell_campaign_tier(
                    "TAOUSDC", 116.0, {"timestamp": 20, "price": 110.0}, 9.0,
                    self.sell_provider)

            self.assertEqual(outcome, "terminal")
            self.assertEqual(ag._completed_tier_values(campaign), set())
            self.assertEqual(tier, (15.0, 0.30, 2.0))

        with self.subTest(msg="test_ambiguous_submit_is_recovered_by_deterministic_client_id"):
            state = MemoryState({
                "version": 3,
                "symbols": {"TAOUSDC": {"sell": {
                    "trough_price": 100.0, "trough_ts": 10,
                    "initial_qty": 10.0, "completed_tiers": [],
                    "pending": {
                        "threshold": 15.0, "client_order_id": "AGS_recover",
                        "requested_qty": 3.0,
                    },
                }}},
            })
            with mock.patch.object(ag, "STATE", state), \
                 mock.patch.object(
                     ag.mkt, "order_by_client_id", return_value={"orderId": 80}), \
                 mock.patch.object(
                     ag.mkt, "order_status",
                     return_value=OrderStatus("open", 0.0, 0.0, 0.0)):
                campaign = ag._symbol_campaign("TAOUSDC", "sell")
                campaign, outcome = ag._reconcile_pending_sell("TAOUSDC", campaign)

            self.assertEqual(outcome, "active")
            self.assertEqual(campaign["pending"]["order_id"], "80")
            self.assertEqual(ag._completed_tier_values(campaign), set())

        with self.subTest(msg="test_submit_acceptance_without_fill_remains_pending"):
            state = MemoryState()
            with mock.patch.object(ag, "STATE", state):
                tier, campaign, _ = ag._sell_campaign_tier(
                    "TAOUSDC", 115.0, {"timestamp": 10, "price": 100.0}, 10.0,
                    self.sell_provider)
                with mock.patch.object(
                        ag, "sell_asset",
                        return_value={"orderId": 81, "status": "NEW", "origQty": "3"}), \
                     mock.patch.object(
                         ag.mkt, "order_status",
                         return_value=OrderStatus("open", 0.0, 0.0, 0.0)):
                    self.assertTrue(ag._submit_sell_tier(
                        "TAOUSDC", current_price=115.0, tier=tier, state=campaign))
                campaign = ag._symbol_campaign("TAOUSDC", "sell")

            self.assertEqual(campaign["pending"]["order_id"], "81")
            self.assertEqual(ag._completed_tier_values(campaign), set())
            self.assertLessEqual(len(campaign["pending"]["client_order_id"]), 36)

if __name__ == "__main__":
    unittest.main()

import assetguardian as ag
import copy
import json
import order_guard
import os
import pytest
import time
import unittest
from botcore import required_env
from external_order_guard import check_external_order_guards
from llm_order_guard import check_llm_order_guards
from providers.strategy_executor import OrderStatus
from qlimit_order_guard import resolve_cached_gaussian_horizon
from types import SimpleNamespace
from unittest import mock




class MemoryState:
    def __init__(self, value=None):
        self.value = copy.deepcopy(value or {})

    def load(self):
        return copy.deepcopy(self.value)

    def save(self, value):
        self.value = copy.deepcopy(value)


def _make_refuse_place(reason="guard_blocked"):
    def _refuse_place(*_args, **kwargs):
        ctx = kwargs.get("_outcome_context")
        if isinstance(ctx, dict):
            ctx.update(accepted=False, reason=reason, state="refused")
        return None

    return _refuse_place


class AssetGuardianTest(unittest.TestCase):
    def test_config_validation(self):
        with self.subTest(msg="test_required_config_rejects_missing_and_empty_values"):
            key = "AG_REQUIRED_CONFIG_REGRESSION_TEST"
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop(key, None)
                with self.assertRaisesRegex(ValueError, key):
                    required_env(key)
            for value in ("", "   ", "\t"):
                with self.subTest(value=value), mock.patch.dict(os.environ, {key: value}):
                    with self.assertRaisesRegex(ValueError, key):
                        required_env(key)

        with self.subTest(msg="test_required_numeric_config_has_no_implicit_fallback"):
            key = "AG_REQUIRED_NUMERIC_REGRESSION_TEST"
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop(key, None)
                with self.assertRaisesRegex(ValueError, key):
                    ag.required_float_env(key)

        with self.subTest(msg="test_config_preflight_reports_every_missing_setting"):
            env = {name: "1" for name in ag.REQUIRED_CONFIG_KEYS}
            env["AG_CHECK_INTERVAL_SEC"] = ""
            env["AG_BUY_USE_CASH_RATIO"] = "   "
            with mock.patch.dict(os.environ, env, clear=True):
                with self.assertRaises(ValueError) as raised:
                    ag.validate_required_env_keys(ag.REQUIRED_CONFIG_KEYS, label="AssetGuardian")
            message = str(raised.exception)
            self.assertIn("AG_CHECK_INTERVAL_SEC", message)
            self.assertIn("AG_BUY_USE_CASH_RATIO", message)

        with self.subTest(msg="test_every_assetguardian_setting_is_registered_as_mandatory"):
            self.assertEqual(
                set(ag.REQUIRED_CONFIG_KEYS),
                {
                    "AG_CHECK_INTERVAL_SEC",
                    "AG_REFERENCE_MINUTES_BACK",
                    "AG_BUY_USE_CASH_RATIO",
                    "AG_SELL_REARM_GROWTH_PCT",
                    "AG_ORDER_MAX_AGE_SEC",
                    "AG_RECOVERY_RESET_PCT",
                    "AG_NEAR_TRIGGER_SEC",
                    "AG_ACTIVE_TRIGGER_SEC",
                    "AG_NEAR_TRIGGER_DISTANCE_PCT",
                    "AG_TREND_DEFER_MAX_SEC",
                    "AG_ORDER_MISSING_CONFIRMATIONS",
                    "AG_PORTFOLIO_GROWTH_PCT",
                    "AG_PORTFOLIO_ASSET_MIN_GROWTH_PCT",
                    "AG_PORTFOLIO_SELL_ALLOCATION",
                    "AG_PORTFOLIO_SELL_MAX_DISCOUNT",
                    "AG_RTRADE_BLOCK_CANCEL_SEC",
                    "AG_P3B_MAX_AGE_SEC",
                    "AG_P3B_SELL_DEFENSIVE_MAX_DISCOUNT",
                    "AG_P3B_SELL_TRIM_MAX_DISCOUNT",
                    "AG_P3B_BUY_ACCUMULATE_MAX_DISCOUNT",
                    "AG_P3B_BUY_DEFENSIVE_MAX_PENALTY",
                },
            )


    def setUp(self):
        state_patch = mock.patch.object(ag, "STATE", MemoryState())
        state_patch.start()
        self.addCleanup(state_patch.stop)
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
        self._asset_val_patch = mock.patch.object(
            ag, "_read_total_asset_value_rows", return_value=[]
        )
        self._asset_val_patch.start()
        self.addCleanup(self._asset_val_patch.stop)
        self._p3b_patch = mock.patch.object(
            ag, "_read_p3b_sentiment", return_value=None
        )
        self._p3b_patch.start()
        self.addCleanup(self._p3b_patch.stop)
        ag._last_evaluation.clear()
        ag._last_evaluation.update({
            symbol: {"growth": None, "drawdown": None, "pending_tier": False}
            for symbol in ag.TRACKED_SYMBOLS
        })
        ag._local_price_samples.clear()
        ag._local_price_samples.update({symbol: [] for symbol in ag.TRACKED_SYMBOLS})
        ag._rtrade_sell_block_since.clear()

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
            self.assertEqual(buy.call_args.kwargs["cash_amount"], 199.0)
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
            self.assertEqual(buy.call_args.kwargs["cash_amount"], 199.0)
            sell.assert_not_called()

        with self.subTest(msg="test_default_9pct_tao_growth_selects_first_sell_tier"):
            extrema = (
                {"timestamp": 1, "price": 100.0},
                {"timestamp": 2, "price": 110.0},
            )
            state = MemoryState()
            provider = mock.Mock()
            provider.free_balance.return_value = 10.0
            provider.open_orders.return_value = []
            with mock.patch.object(ag.api, "get_current_price", return_value=109.5), \
                 mock.patch.object(ag, "_get_symbol_window_extrema", return_value=extrema), \
                 mock.patch.object(ag, "_get_portfolio_growth_percent", return_value=0.0), \
                 mock.patch.object(ag, "STATE", state), \
                 mock.patch.object(ag.mkt, "provider_by_name", return_value=provider), \
                 mock.patch.object(ag, "_submit_sell_tier", return_value=True) as submit, \
                 mock.patch.object(ag, "buy_with_all_cash") as buy:
                self.assertTrue(ag.evaluate_symbol("TAOUSDC"))
            tier = submit.call_args.kwargs["tier"]
            self.assertEqual(tier, (9.0, 0.12, 1.2))
            self.assertEqual(submit.call_args.kwargs["current_price"], 109.5)
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
            self.assertEqual(btc_tier, (7.0, 0.20))
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
            self.assertEqual(tier, (10.0, 0.30))
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
            self.assertEqual(shallow, (7.0, 0.20))
            self.assertEqual(middle, (10.0, 0.30))
            self.assertEqual(deepest, (14.0, 0.50))
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
            self.assertEqual(
                place.call_args.kwargs["bypass_guards"],
                {"G_PROF", "G_QLIMIT1", "G_QLIMIT2", "GI_OBOOK", "GI_WHALE", "GI_FUNDING", "GLLM1", "GLLM2", "GLLM3"},
            )
            self.assertTrue(place.call_args.kwargs["wait_for_trend"])
            self.assertEqual(place.call_args.kwargs["motivation"],
                             "assetguardian_drawdown_buy")
            self.assertEqual(place.call_args.args[0], "TAOUSDC")
            with mock.patch.object(ag.mkt, "provider_by_name", return_value=provider), \
                 mock.patch.object(ag.api, "get_current_price", return_value=100.0), \
                 mock.patch.object(ag.mkt, "place", return_value={"orderId": 8}) as place_guarded:
                self.assertTrue(ag.buy_with_all_cash("TAOUSDC", cash_ratio=0.5, respect_guard=True))
            self.assertEqual(place_guarded.call_args.kwargs["bypass_guards"], {"G_QLIMIT1"})

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
                campaign, outcome = ag._reconcile_pending_campaign("TAOUSDC", "buy", campaign)

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
                campaign, outcome = ag._reconcile_pending_campaign("TAOUSDC", "buy", campaign)
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
                buy_state, outcome = ag._reconcile_pending_campaign("TAOUSDC", "buy", buy_state)

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
            client_id = ag._ag_client_order_id("buy", "TAOUSDC", campaign, 7.0, 1)
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
            # Even after TREND_DEFER_MAX_SECONDS, while should_wait (GT_S) is True,
            # AssetGuardian must keep deferring and never bypass GT_S on a timer.
            with mock.patch.object(ag, "STATE", state), \
                 mock.patch.object(ag.cm, "get_short_trend_manager", return_value=manager), \
                 mock.patch.object(
                     ag.time, "time",
                     return_value=1000.0 + ag.TREND_DEFER_MAX_SECONDS + 1), \
                 mock.patch.object(ag.time, "sleep") as sleep2:
                ready2, still_deferred = self._real_trend_defer_ready(
                    "TAOUSDC", "buy", "BUY", 7.0, deferred)
            # Once should_wait becomes False (trend turns favorable), it releases immediately.
            manager.should_wait.return_value = False
            with mock.patch.object(ag, "STATE", state), \
                 mock.patch.object(ag.cm, "get_short_trend_manager", return_value=manager), \
                 mock.patch.object(
                     ag.time, "time",
                     return_value=1000.0 + ag.TREND_DEFER_MAX_SECONDS + 5), \
                 mock.patch.object(ag.time, "sleep") as sleep3:
                ready3, released = self._real_trend_defer_ready(
                    "TAOUSDC", "buy", "BUY", 7.0, still_deferred)

            self.assertFalse(ready1)
            self.assertFalse(ready2)
            self.assertIn("trend_defer", still_deferred)
            self.assertTrue(ready3)
            self.assertNotIn("trend_defer", released)
            sleep.assert_not_called()
            sleep2.assert_not_called()
            sleep3.assert_not_called()

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
            self.assertEqual(place.call_args.kwargs["bypass_guards"], {"G_QLIMIT1"})
            self.assertEqual(place.call_args.kwargs["module"], "assetguardian")
            self.assertEqual(place.call_args.kwargs["motivation"],
                             "assetguardian_growth_exit_tier_15")
            self.assertEqual(place.call_args.kwargs["client_order_id"], "AGS_test")
            self.assertEqual(place.call_args.args[:4], ("TAOUSDC", "SELL", 250.0, 0.2))

        with self.subTest(msg="test_sell_campaign_freezes_trough_and_initial_quantity"):
            state = MemoryState()
            with mock.patch.object(ag, "STATE", state), \
                 mock.patch.object(ag, "SELL_TIERS", ((15.0, 0.30), (25.0, 0.30), (35.0, 0.40))):
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
            with mock.patch.object(ag, "STATE", state), \
                 mock.patch.object(ag, "SELL_TIERS", ((15.0, 0.30), (25.0, 0.30), (35.0, 0.40))):
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
                    "TAOUSDC", 102.0,
                    {"timestamp": 20, "price": 80.0}, 7.0,
                    self.sell_provider)

            self.assertIsNone(tier)
            self.assertEqual(campaign, {})
            self.assertEqual(growth, 2.0)
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
                campaign, outcome = ag._reconcile_pending_campaign("TAOUSDC", "sell", campaign)

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
                campaign, outcome = ag._reconcile_pending_campaign("TAOUSDC", "sell", campaign)

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
                campaign, outcome = ag._reconcile_pending_campaign("TAOUSDC", "sell", campaign)

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
                campaign, outcome1 = ag._reconcile_pending_campaign("TAOUSDC", "sell", campaign)
                campaign, outcome2 = ag._reconcile_pending_campaign("TAOUSDC", "sell", campaign)

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
                campaign, outcome = ag._reconcile_pending_campaign("TAOUSDC", "sell", campaign)

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
                 mock.patch.object(ag, "SELL_TIERS", ((15.0, 0.30), (25.0, 0.30), (35.0, 0.40))), \
                 mock.patch.object(
                     ag.mkt, "order_status",
                     return_value=OrderStatus("canceled", 1.0, 115.0, 0.1)):
                campaign = ag._symbol_campaign("TAOUSDC", "sell")
                campaign, outcome = ag._reconcile_pending_campaign("TAOUSDC", "sell", campaign)
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
                campaign, outcome = ag._reconcile_pending_campaign("TAOUSDC", "sell", campaign)

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

        _refuse_place = _make_refuse_place("guard_blocked")

        with self.subTest(msg="test_locally_refused_sell_clears_pending_and_reuses_attempt"):
            state = MemoryState()
            provider = mock.Mock()
            provider.free_balance.return_value = 10.0

            with mock.patch.object(ag, "STATE", state), \
                 mock.patch.object(ag.api, "get_current_price", return_value=225.0), \
                 mock.patch.object(ag.mkt, "provider_by_name", return_value=provider), \
                 mock.patch.object(ag.mkt, "place", side_effect=_refuse_place):
                tier, campaign, _ = ag._sell_campaign_tier(
                    "TAOUSDC", 115.0, {"timestamp": 10, "price": 100.0}, 10.0,
                    self.sell_provider)
                self.assertFalse(ag._submit_sell_tier(
                    "TAOUSDC", current_price=115.0, tier=tier, state=campaign))
                campaign = ag._symbol_campaign("TAOUSDC", "sell")

            self.assertNotIn("pending", campaign)
            self.assertEqual(campaign.get("attempts_by_tier", {}).get("9", 0), 0)

        with self.subTest(msg="test_locally_refused_buy_clears_pending_and_reuses_attempt"):
            state = MemoryState()
            campaign = {
                "peak_price": 242.0,
                "peak_ts": 2.0,
                "initial_cash": 1000.0,
                "completed_tiers": [],
            }
            provider = mock.Mock()
            provider.free_balance.return_value = 1000.0

            with mock.patch.object(ag, "STATE", state), \
                 mock.patch.object(ag.api, "get_current_price", return_value=225.0), \
                 mock.patch.object(ag.mkt, "provider_by_name", return_value=provider), \
                 mock.patch.object(ag.mkt, "place", side_effect=_refuse_place):
                self.assertFalse(ag._submit_buy_tier(
                    "TAOUSDC", 225.0, 348.25, (7.0, 0.35), campaign))
                buy_state = ag._symbol_campaign("TAOUSDC", "buy")

            self.assertNotIn("pending", buy_state)
            self.assertEqual(buy_state.get("attempts_by_tier", {}).get("7", 0), 0)

    def test_portfolio_growth_hybrid_sell_and_rtrade_preemption(self):
        with self.subTest(msg="test_portfolio_growth_triggers_4_1pct_price_notifier_aligned_sell_tier"):
            extrema = (
                {"timestamp": 1, "price": 269.3},
                {"timestamp": 2, "price": 281.2},
            )
            state = MemoryState()
            provider = mock.Mock()
            provider.free_balance.return_value = 10.0
            provider.open_orders.return_value = []
            with mock.patch.object(ag.api, "get_current_price", return_value=281.1), \
                 mock.patch.object(ag, "_get_symbol_window_extrema", return_value=extrema), \
                 mock.patch.object(ag, "_get_portfolio_growth_percent", return_value=2.96), \
                 mock.patch.object(ag, "STATE", state), \
                 mock.patch.object(ag.mkt, "provider_by_name", return_value=provider), \
                 mock.patch.object(ag, "_submit_sell_tier", return_value=True) as submit, \
                 mock.patch.object(ag, "buy_with_all_cash") as buy:
                self.assertTrue(ag.evaluate_symbol("TAOUSDC"))
            tier = submit.call_args.kwargs["tier"]
            self.assertEqual(tier, (4.1, 0.20, 2.0))
            buy.assert_not_called()

        with self.subTest(msg="test_4_4pct_asset_growth_without_portfolio_growth_waits_for_9pct_tier"):
            extrema = (
                {"timestamp": 1, "price": 269.3},
                {"timestamp": 2, "price": 281.2},
            )
            state = MemoryState()
            provider = mock.Mock()
            provider.free_balance.return_value = 10.0
            provider.open_orders.return_value = []
            with mock.patch.object(ag.api, "get_current_price", return_value=281.1), \
                 mock.patch.object(ag, "_get_symbol_window_extrema", return_value=extrema), \
                 mock.patch.object(ag, "_get_portfolio_growth_percent", return_value=0.8), \
                 mock.patch.object(ag, "STATE", state), \
                 mock.patch.object(ag.mkt, "provider_by_name", return_value=provider), \
                 mock.patch.object(ag, "_submit_sell_tier") as submit, \
                 mock.patch.object(ag, "buy_with_all_cash") as buy:
                self.assertFalse(ag.evaluate_symbol("TAOUSDC"))
            submit.assert_not_called()
            buy.assert_not_called()

        with self.subTest(msg="test_rtrade_blocking_sell_order_retries_first_then_cancels_after_grace_timeout"):
            state = MemoryState()
            provider = mock.Mock()
            provider.free_balance.return_value = 10.0
            rtrade_order = {
                "orderId": 716499872,
                "clientOrderId": "RT_TAO_999",
                "side": "SELL",
                "status": "NEW",
            }
            provider.open_orders.return_value = [rtrade_order]
            ag._rtrade_sell_block_since.clear()
            with mock.patch.object(ag, "STATE", state), \
                 mock.patch.object(ag.time, "time", return_value=1000.0), \
                 mock.patch.object(ag.mkt, "cancel_order") as cancel:
                tier1, _, _ = ag._sell_campaign_tier(
                    "TAOUSDC", 110.0, {"timestamp": 10, "price": 100.0}, 10.0, provider)
            self.assertIsNone(tier1)
            cancel.assert_not_called()
            self.assertIn("TAOUSDC", ag._rtrade_sell_block_since)
            self.assertEqual(ag._next_check_seconds(), ag.ACTIVE_TRIGGER_SECONDS)

            # After grace period (120s), assetguardian preempts rtrade and proceeds with its tier
            def _cancel_side_effect(*_args, **_kwargs):
                provider.open_orders.return_value = []

            with mock.patch.object(ag, "STATE", state), \
                 mock.patch.object(
                     ag.time, "time",
                     return_value=1000.0 + ag.RTRADE_BLOCK_CANCEL_SECONDS + 1.0), \
                 mock.patch.object(
                     ag.mkt, "cancel_order", side_effect=_cancel_side_effect) as cancel2:
                tier2, _, _ = ag._sell_campaign_tier(
                    "TAOUSDC", 110.0, {"timestamp": 10, "price": 100.0}, 10.0, provider)
            cancel2.assert_called_once_with(
                "TAOUSDC", "716499872", provider_name="binance")
            self.assertEqual(tier2, (9.0, 0.12, 1.2))

        with self.subTest(msg="test_terminal_buy_and_sell_tolerate_none_or_nan_cost_and_fee"):
            state = MemoryState()
            raw_status = SimpleNamespace(status="canceled", filled_qty=None, cost=None, fee=float("nan"))
            with mock.patch.object(ag, "STATE", state):
                buy_camp = ag._record_terminal_buy(
                    "TAOUSDC",
                    {"pending": {"threshold": 7.0, "order_id": "b1", "client_order_id": "AGB_1"}},
                    raw_status,
                )
                sell_camp = ag._record_terminal_sell(
                    "TAOUSDC",
                    {"pending": {"threshold": 15.0, "order_id": "s1", "client_order_id": "AGS_1"}},
                    raw_status,
                )
            self.assertEqual(buy_camp["spent_quote_by_tier"]["7"], 0.0)
            self.assertEqual(buy_camp["terminal_orders"][-1]["cost"], 0.0)
            self.assertEqual(buy_camp["terminal_orders"][-1]["fee"], 0.0)
            self.assertEqual(sell_camp["filled_qty_by_tier"]["15"], 0.0)
            self.assertEqual(sell_camp["terminal_orders"][-1]["cost"], 0.0)
            self.assertEqual(sell_camp["terminal_orders"][-1]["fee"], 0.0)

        with self.subTest(msg="test_dust_remainder_after_99pct_partial_fill_completes_buy_and_sell_tiers"):
            state = MemoryState()
            with mock.patch.object(ag, "STATE", state):
                buy_camp = ag._record_terminal_buy(
                    "TAOUSDC",
                    {
                        "initial_cash": 1000.0,
                        "completed_tiers": [],
                        "pending": {
                            "threshold": 7.0,
                            "allocation": 0.35,
                            "order_id": "b-99",
                            "client_order_id": "AGB_99",
                        },
                    },
                    OrderStatus("canceled", 1.54, 345.0, 0.5),
                )
                sell_camp = ag._record_terminal_sell(
                    "TAOUSDC",
                    {
                        "initial_qty": 10.0,
                        "completed_tiers": [],
                        "pending": {
                            "threshold": 15.0,
                            "allocation": 0.30,
                            "order_id": "s-99",
                            "client_order_id": "AGS_99",
                        },
                    },
                    OrderStatus("canceled", 2.97, 341.55, 0.3),
                )
            self.assertIn(7.0, buy_camp["completed_tiers"])
            self.assertIn(15.0, sell_camp["completed_tiers"])

    def test_p3b_sentiment_dynamic_threshold_scaling(self):
        with self.subTest(msg="test_p3b_defensive_compresses_sell_threshold_down_to_half_preserving_canonical_tier"):
            # TAO grew +4.65% (from 269.3 to 281.82), below static 9% tier and with 0% portfolio growth.
            # When P3B flips to DEFENSIVE (conf=1.0), effective threshold for 9% becomes 4.5%,
            # triggering canonical tier 9.0 (so it will not sell again when reaching 9%).
            extrema = (
                {"timestamp": 1, "price": 269.3},
                {"timestamp": 2, "price": 282.0},
            )
            state = MemoryState()
            provider = mock.Mock()
            provider.free_balance.return_value = 10.0
            provider.open_orders.return_value = []
            defensive_p3b = {
                "recommended_action": "DEFENSIVE",
                "market_bias": "BEARISH",
                "risk_level": "HIGH",
                "confidence": 1.0,
                "ts": 1000.0,
            }
            with mock.patch.object(ag.api, "get_current_price", return_value=281.82), \
                 mock.patch.object(ag, "_get_symbol_window_extrema", return_value=extrema), \
                 mock.patch.object(ag, "_get_portfolio_growth_percent", return_value=0.2), \
                 mock.patch.object(ag, "_read_p3b_sentiment", return_value=defensive_p3b), \
                 mock.patch.object(ag, "STATE", state), \
                 mock.patch.object(ag.mkt, "provider_by_name", return_value=provider), \
                 mock.patch.object(ag, "_submit_sell_tier", return_value=True) as submit, \
                 mock.patch.object(ag, "buy_with_all_cash") as buy:
                self.assertTrue(ag.evaluate_symbol("TAOUSDC"))
            tier = submit.call_args.kwargs["tier"]
            # Canonical tier 9.0 is returned (not 4.5), preserving SSOT tier accounting.
            self.assertEqual(tier, (9.0, 0.12, 1.2))
            buy.assert_not_called()

            # Once canonical tier 9.0 is completed, even if P3B returns to HOLD (1.0x) and
            # growth reaches +9.5%, tier 9.0 is not repeated; next tier is 12.0.
            with mock.patch.object(ag, "STATE", state), \
                 mock.patch.object(ag, "_read_p3b_sentiment", return_value=None):
                sell_camp = ag._symbol_campaign("TAOUSDC", "sell")
                ag._complete_campaign_tier("TAOUSDC", "sell", sell_camp, 9.0)
                next_tier, _, _ = ag._sell_campaign_tier(
                    "TAOUSDC", 295.0, {"timestamp": 1, "price": 269.3}, 8.8, provider,
                    portfolio_growth_pct=0.2)
                self.assertIsNone(next_tier)
                tier_12, _, _ = ag._sell_campaign_tier(
                    "TAOUSDC", 302.0, {"timestamp": 1, "price": 269.3}, 8.8, provider,
                    portfolio_growth_pct=0.2)
                self.assertEqual(tier_12, (12.0, 0.16, 1.6))

        with self.subTest(msg="test_p3b_trim_profits_compresses_moderately_while_bullish_keeps_static"):
            sell_mult_trim, buy_mult_trim, _ = ag._p3b_threshold_multipliers({
                "recommended_action": "TRIM_PROFITS",
                "market_bias": "CAUTION",
                "risk_level": "MODERATE",
                "confidence": 1.0,
                "ts": 1000.0,
            })
            self.assertAlmostEqual(sell_mult_trim, 0.65)
            self.assertAlmostEqual(ag._trigger_threshold_for_sell_tier(9.0, sell_mult_trim), 5.85)
            self.assertAlmostEqual(buy_mult_trim, 1.175)

            sell_mult_bull, buy_mult_bull, _ = ag._p3b_threshold_multipliers({
                "recommended_action": "ACCUMULATE",
                "market_bias": "BULLISH",
                "risk_level": "LOW",
                "confidence": 1.0,
                "ts": 1000.0,
            })
            self.assertAlmostEqual(sell_mult_bull, 1.0)
            self.assertAlmostEqual(buy_mult_bull, 0.65)
            self.assertAlmostEqual(ag._trigger_threshold_for_buy_tier(7.0, buy_mult_bull), 4.55)

        with self.subTest(msg="test_p3b_defensive_delays_buy_tiers_while_accumulate_triggers_earlier"):
            state = MemoryState()
            maximum = {"timestamp": 2, "price": 280.0}
            defensive_p3b = {
                "recommended_action": "DEFENSIVE",
                "market_bias": "BEARISH",
                "risk_level": "HIGH",
                "confidence": 1.0,
                "ts": 1000.0,
            }
            accumulate_p3b = {
                "recommended_action": "ACCUMULATE",
                "market_bias": "BULLISH",
                "risk_level": "LOW",
                "confidence": 1.0,
                "ts": 1000.0,
            }
            with mock.patch.object(ag, "STATE", state):
                # At -7.5% drawdown, DEFENSIVE requires 7.0 * 1.35 = 9.45% -> no BUY yet
                tier_def, _ = ag._campaign_tier(
                    "TAOUSDC", 7.5, maximum, 1000.0, p3b_assessment=defensive_p3b)
                self.assertIsNone(tier_def)
                # At -5.0% drawdown, ACCUMULATE requires 7.0 * 0.65 = 4.55% -> triggers canonical 7.0 BUY tier
                tier_acc, _ = ag._campaign_tier(
                    "TAOUSDC", 5.0, maximum, 1000.0, p3b_assessment=accumulate_p3b)
                self.assertEqual(tier_acc, (7.0, 0.20))

        with self.subTest(msg="test_dynamic_progressive_tiers_and_24h_volatility_scaling"):
            for forbidden in (
                "DEFAULT_BUY_THRESHOLDS",
                "DEFAULT_SELL_THRESHOLDS",
                "DEFAULT_BUY_INITIAL_ALLOCATION",
                "DEFAULT_SELL_INITIAL_ALLOCATION",
                "BUY_TIERS_RAW",
                "SELL_TIERS_RAW",
            ):
                self.assertFalse(hasattr(ag, forbidden), forbidden)
            self.assertEqual(
                ag._progressive_tier_allocations(3, 0.20, mode="convex"),
                (0.20, 0.30, 0.50),
            )
            self.assertEqual(
                ag._progressive_tier_allocations(5, 0.12, mode="linear"),
                (0.12, 0.16, 0.20, 0.24, 0.28),
            )
            # Verify tiers recalculate dynamically when underlying config bounds change
            with mock.patch.object(ag, "RECOVERY_RESET_PERCENT", 4.0):
                self.assertEqual(
                    ag._build_dynamic_buy_tiers(),
                    ((8.0, 0.20), (12.0, 0.30), (17.0, 0.50)),
                )
            min_mult, max_mult = ag._volatility_mult_bounds()
            # Low-volatility 24h series (like BTC ~0.12% per 15m bucket) scales mult down to min_mult
            low_vol_rows = [
                {"timestamp": 1_000_000.0 + i * 300.0, "price": 80_000.0 * (1.0 + (0.0006 if i % 6 < 3 else -0.0006))}
                for i in range(80)
            ]
            low_mult = ag._compute_symbol_volatility_multiplier(low_vol_rows)
            self.assertEqual(low_mult, min_mult)
            # At min_mult (0.76x), canonical -7.0% BUY tier triggers at -5.32% (catching -6.2% BTC drop)
            self.assertAlmostEqual(
                ag._trigger_threshold_for_buy_tier(7.0, buy_mult=1.0, volatility_mult=low_mult),
                round(7.0 * min_mult, 4),
            )
            # High-volatility 24h series scales mult up to max_mult
            high_vol_rows = [
                {"timestamp": 1_000_000.0 + i * 300.0, "price": 240.0 * (1.012 if (i // 3) % 2 == 0 else 1.0)}
                for i in range(80)
            ]
            high_mult = ag._compute_symbol_volatility_multiplier(high_vol_rows)
            self.assertEqual(high_mult, max_mult)

        with self.subTest(msg="test_combined_p3b_and_portfolio_growth_compress_all_sell_thresholds"):
            # 1. Neutral P3B (1.0x) + Portfolio Growth >= 2.2% compresses static tiers by 20%
            self.assertAlmostEqual(
                ag._trigger_threshold_for_sell_tier(4.1, 1.0, portfolio_growth_pct=2.5),
                4.10,
            )
            self.assertAlmostEqual(
                ag._trigger_threshold_for_sell_tier(9.0, 1.0, portfolio_growth_pct=2.5),
                7.20,
            )
            self.assertAlmostEqual(
                ag._trigger_threshold_for_sell_tier(12.0, 1.0, portfolio_growth_pct=2.5),
                9.60,
            )
            self.assertAlmostEqual(
                ag._trigger_threshold_for_sell_tier(15.0, 1.0, portfolio_growth_pct=2.5),
                12.00,
            )

            # 2. TRIM_PROFITS (0.65x) lowers portfolio activation threshold from 2.2% to 1.43%
            self.assertAlmostEqual(ag._effective_portfolio_growth_threshold(0.65), 1.43)
            active_trim = ag._effective_sell_tiers(portfolio_growth_pct=1.50, sell_mult=0.65)
            self.assertEqual(active_trim[0], (4.1, 0.20))
            self.assertAlmostEqual(
                ag._trigger_threshold_for_sell_tier(4.1, 0.65, portfolio_growth_pct=1.50),
                3.50,
            )
            self.assertAlmostEqual(
                ag._trigger_threshold_for_sell_tier(9.0, 0.65, portfolio_growth_pct=1.50),
                4.68,
            )
            self.assertAlmostEqual(
                ag._trigger_threshold_for_sell_tier(12.0, 0.65, portfolio_growth_pct=1.50),
                6.24,
            )
            self.assertAlmostEqual(
                ag._trigger_threshold_for_sell_tier(15.0, 0.65, portfolio_growth_pct=1.50),
                7.80,
            )

            # 3. DEFENSIVE (0.50x) lowers portfolio activation threshold from 2.2% to 1.10%
            self.assertAlmostEqual(ag._effective_portfolio_growth_threshold(0.50), 1.10)
            active_def = ag._effective_sell_tiers(portfolio_growth_pct=1.20, sell_mult=0.50)
            self.assertEqual(active_def[0], (4.1, 0.20))
            self.assertAlmostEqual(
                ag._trigger_threshold_for_sell_tier(4.1, 0.50, portfolio_growth_pct=1.20),
                3.50,
            )
            self.assertAlmostEqual(
                ag._trigger_threshold_for_sell_tier(9.0, 0.50, portfolio_growth_pct=1.20),
                4.00,
            )
            self.assertAlmostEqual(
                ag._trigger_threshold_for_sell_tier(12.0, 0.50, portfolio_growth_pct=1.20),
                4.80,
            )
            self.assertAlmostEqual(
                ag._trigger_threshold_for_sell_tier(15.0, 0.50, portfolio_growth_pct=1.20),
                6.00,
            )

        with self.subTest(msg="test_persisted_last_refusal_deduplicates_audit_across_restart"):
            state = MemoryState()
            provider = mock.Mock()
            provider.free_balance.return_value = 10.0

            audit_mock = mock.Mock()
            with mock.patch.object(ag, "STATE", state), \
                 mock.patch.object(ag.api, "get_current_price", return_value=115.0), \
                 mock.patch.object(ag.mkt, "provider_by_name", return_value=provider), \
                 mock.patch.object(ag.mkt, "place", side_effect=_make_refuse_place("profit_guard")), \
                 mock.patch.object(ag.ORDER_LIFECYCLE, "audit", audit_mock):
                ag.ORDER_LIFECYCLE._last_refusal_by_intent.clear()
                tier, campaign, _ = ag._sell_campaign_tier(
                    "TAOUSDC", 115.0, {"timestamp": 10, "price": 100.0}, 10.0,
                    self.sell_provider)
                self.assertFalse(ag._submit_sell_tier(
                    "TAOUSDC", current_price=115.0, tier=tier, state=campaign))
                self.assertEqual(audit_mock.record.call_count, 2)

                # Simulate process restart by clearing in-memory _last_refusal_by_intent
                ag.ORDER_LIFECYCLE._last_refusal_by_intent.clear()
                campaign_reloaded = ag._symbol_campaign("TAOUSDC", "sell")
                self.assertIn("last_refusal", campaign_reloaded)
                self.assertFalse(ag._submit_sell_tier(
                    "TAOUSDC", current_price=115.0, tier=tier, state=campaign_reloaded))
                # No duplicate submit_requested / submit_refused emitted after reload
                self.assertEqual(audit_mock.record.call_count, 2)


if __name__ == "__main__":
    unittest.main()

# ============================================================================
# Domain Regressions & Input Hygiene (Targeted Validation)
# ============================================================================


def test_assetguardian__read_p3b_sentiment_recovers_from_previous_when_primary_is_iter13_s1(tmp_path):
    """[ITER13 #1] AssetGuardian _read_p3b_sentiment recovers from .previous when primary is corrupted,."""
    now_ts = time.time()
    # 1. AssetGuardian _read_p3b_sentiment recovers from .previous when primary is corrupted,
    #    and _p3b_threshold_multipliers honors GLLM2 bypass.
    p3b_file = tmp_path / "sentiment_eval.json"
    p3b_prev = tmp_path / "sentiment_eval.json.previous"
    p3b_file.write_text("{corrupted_json", encoding="utf-8")
    p3b_prev.write_text(
        json.dumps(
            {
                "recommended_action": "DEFENSIVE",
                "market_bias": "BEARISH",
                "risk_level": "HIGH",
                "confidence": 0.90,
                "ts": now_ts - 60.0,
            }
        ),
        encoding="utf-8",
    )
    recovered = ag._read_p3b_sentiment(path=str(p3b_file), now_ts=now_ts)
    assert isinstance(recovered, dict)
    assert recovered["recommended_action"] == "DEFENSIVE"
    assert recovered["confidence"] == pytest.approx(0.90)

    # When GLLM2 is bypassed via bypass_guards, _p3b_threshold_multipliers returns neutral (1.0, 1.0, None)
    sell_m, buy_m, eff_assess = ag._p3b_threshold_multipliers(
        assessment=recovered, bypass_guards={"GLLM2"}
    )
    assert sell_m == pytest.approx(1.0)
    assert buy_m == pytest.approx(1.0)
    assert eff_assess is None

    # Without GLLM2 bypass, defensive assessment compresses sell_mult (< 1.0) and expands buy_mult (> 1.0)
    sell_m2, buy_m2, eff_assess2 = ag._p3b_threshold_multipliers(
        assessment=recovered, bypass_guards=set()
    )
    assert sell_m2 < 1.0
    assert buy_m2 > 1.0
    assert eff_assess2 is recovered

def test_resolve_caller_module_short_codes_and_client_order_id_prefix_fallback_iter20_s1():
    """[ITER20 #1] resolve_caller_module short codes and client_order_id -> prefix fallback."""
    # 1. resolve_caller_module short codes and client_order_id -> prefix fallback
    assert order_guard.resolve_caller_module("AG") == "assetguardian"
    assert order_guard.resolve_caller_module("MT") == "monitortrades"
    assert order_guard.resolve_caller_module("TA") == "tradeall"
    assert order_guard.resolve_caller_module("RT") == "rtrade"
    assert order_guard.resolve_caller_module("asset_guardian") == "assetguardian"
    assert (
        order_guard.resolve_caller_module(
            None,
            client_order_id="OR_12345_0",
            prefix="TA_OR",
        )
        == "tradeall"
    )

def test_assetguardian__sell_campaign_tier_does_not_double_subtract_already_filled_on_iter21_s1(tmp_path, monkeypatch):
    """[ITER21 #1] assetguardian._sell_campaign_tier does not double-subtract already_filled on partial fill."""
    # 1. assetguardian._sell_campaign_tier does not double-subtract already_filled on partial fill
    state_file = tmp_path / "ag_state.json"
    monkeypatch.setattr(ag, "STATE", ag.AssetGuardianState(str(state_file), fail_closed=False))
    monkeypatch.setattr(ag, "_p3b_threshold_multipliers", lambda *a, **k: (1.0, 1.0, None))
    completed_earlier = [t for t, _ in ag.SELL_TIERS[:-1]]
    last_thr, last_alloc = ag.SELL_TIERS[-1]
    tier_target = 10.0 * last_alloc
    partial_fill = tier_target * 0.6
    rem_expected = tier_target - partial_fill
    total_filled = (10.0 - tier_target) + partial_fill
    ag._save_symbol_campaign(
        "TAOUSDC",
        "sell",
        {
            "trough_price": 100.0,
            "trough_ts": 1000.0,
            "initial_qty": 10.0,
            "created_at": 1000.0,
            "completed_tiers": completed_earlier,
            "filled_qty_by_tier": {ag._tier_key(last_thr): partial_fill},
            "total_filled_qty": total_filled,
            "attempts_by_tier": {},
            "terminal_orders": [],
        },
    )

    tier_res, _, growth = ag._sell_campaign_tier(
        "TAOUSDC",
        current_price=100.0 * (1.0 + last_thr / 100.0 + 0.05),
        minimum_row={"price": 100.0, "timestamp": 1000.0},
        free_qty=rem_expected,
        provider=SimpleNamespace(name="binance", open_orders=lambda symbol: []),
    )
    assert tier_res is not None
    thr_out, alloc_out, rem_qty = tier_res
    assert thr_out == pytest.approx(last_thr)
    assert alloc_out == pytest.approx(last_alloc)
    # Must return the remaining 40% of tier_target, NOT 0.0 (min(rem_expected, tier_target) - partial_fill)!
    assert rem_qty == pytest.approx(rem_expected)

def test_assetguardian__p3b_threshold_multipliers_unions_caller_bypass_guards_with_venue_bypass_guards_iter23_s1(tmp_path, monkeypatch):
    """[ITER23 #1] assetguardian._p3b_threshold_multipliers unions caller bypass_guards with venue bypass_guards & check_external_order_guards & check_llm_order_guards honor intelligence_guards_mode='off'."""
    # 1. assetguardian._p3b_threshold_multipliers unions caller bypass_guards with venue bypass_guards
    orig_margins = dict(order_guard._load_margins())
    monkeypatch.setattr(
        order_guard,
        "_load_margins",
        lambda: {**orig_margins, "binance_bypass_guards": {"GLLM2"}},
    )
    defensive_assessment = {
        "recommended_action": "DEFENSIVE",
        "market_bias": "BEARISH",
        "risk_level": "HIGH",
        "confidence": 0.9,
        "ts": time.time(),
    }
    sell_mult, buy_mult, eff_a = ag._p3b_threshold_multipliers(
        assessment=defensive_assessment,
        bypass_guards={"G_PROF"},
    )
    assert sell_mult == pytest.approx(1.0)
    assert buy_mult == pytest.approx(1.0)
    assert eff_a is None


    # 2. check_external_order_guards & check_llm_order_guards honor intelligence_guards_mode='off'
    off_margins = {**orig_margins, "intelligence_guards_mode": "off"}
    assert check_external_order_guards("BTCUSDC", "BUY", 85000.0, margins=off_margins) == (
        True,
        "intelligence_guards_off",
        1.0,
    )
    assert check_llm_order_guards("binance", "BTCUSDC", "BUY", 85000.0, margins=off_margins) == (
        True,
        "intelligence_guards_off",
        1.0,
    )

    # resolve_cached_gaussian_horizon rejects non-finite / negative values
    t_cache = tmp_path / "t_cache.json"
    t_cache.write_text(
        json.dumps({"BTCUSDC": {"p90_d": -5.0, "median_d": float("nan"), "T": float("inf")}}),
        encoding="utf-8",
    )
    monkeypatch.setattr("intelligence.internal.state.survival.T_CACHE_FILE", str(t_cache))
    assert resolve_cached_gaussian_horizon("BTCUSDC") == (None, None, None)

def test_assetguardian__buy_remaining_cash__record_terminal_buy_clamp_negative_corrupted_prior_spent_fee_iter63_s4(monkeypatch):
    """[ITER63 #4] assetguardian._buy_remaining_cash & _record_terminal_buy clamp negative corrupted prior_spent / fee."""
    # 4. assetguardian._buy_remaining_cash & _record_terminal_buy clamp negative corrupted prior_spent / fee
    rem, target, prior = ag._buy_remaining_cash(
        {"initial_cash": 1000.0, "spent_quote_by_tier": {"5": -500.0}},
        5.0,
        0.2,
    )
    assert prior == 0.0
    assert rem == pytest.approx(target)

    saved = {}
    monkeypatch.setattr(ag, "_save_symbol_campaign", lambda sym, kind, st: saved.update(st))
    updated = ag._record_terminal_buy(
        "BTCUSDC",
        {
            "initial_cash": 1000.0,
            "pending": {"threshold": 5.0, "allocation": 0.2, "order_id": "1", "client_order_id": "c1"},
            "spent_quote_by_tier": {"5": -100.0},
            "total_spent_quote": -200.0,
        },
        SimpleNamespace(status="canceled", filled_qty=1.0, cost=50.0, fee=-20.0),
    )
    assert updated["spent_quote_by_tier"]["5"] == pytest.approx(50.0)
    assert updated["total_spent_quote"] == pytest.approx(50.0)

def test_assetguardian_boolean_flags__safe_tier_attempt_and__reuse_unaccepted_attempt_iter157_s4(monkeypatch, tmp_path):
    """[ITER157 #4] assetguardian boolean flags, _safe_tier_attempt, and _reuse_unaccepted_attempt."""
    # 4. assetguardian boolean flags, _safe_tier_attempt, and _reuse_unaccepted_attempt
    st = ag.AssetGuardianState(path=str(tmp_path / "ag.json"), fail_closed="false")
    assert st.fail_closed is False
    st_nan = ag.AssetGuardianState(path=str(tmp_path / "ag2.json"), fail_closed=float("nan"))
    assert st_nan.fail_closed is True

    assert ag._safe_tier_attempt({"3": "2.0"}, "3") == 3
    assert ag._safe_tier_attempt({"3": True}, "3") == 1

    saved_states = []
    monkeypatch.setattr(ag, "_save_symbol_campaign", lambda sym, kind, s: saved_states.append(dict(s)))
    res_bool = ag._reuse_unaccepted_attempt(
        "BTCUSDC",
        "buy",
        {"attempts_by_tier": {"3": 2}},
        {"threshold": 3.0, "attempt": True},
    )
    assert res_bool["attempts_by_tier"]["3"] == 2
    assert saved_states == []

    res_float_str = ag._reuse_unaccepted_attempt(
        "BTCUSDC",
        "buy",
        {"attempts_by_tier": "corrupt"},
        {"threshold": 3.0, "attempt": "2.0"},
    )
    assert res_float_str["attempts_by_tier"] == {"3": 1}

    # _record_terminal_buy tolerates corrupted non-dict spent_quote_by_tier / non-list terminal_orders
    term_state = ag._record_terminal_buy(
        "BTCUSDC",
        {
            "pending": {"threshold": 3.0, "allocation": 0.25, "order_id": "11", "client_order_id": "cid-11"},
            "spent_quote_by_tier": "corrupt",
            "terminal_orders": "corrupt",
            "initial_cash": 1000.0,
        },
        OrderStatus("closed", 0.01, 50.0, 0.05),
    )
    assert term_state["spent_quote_by_tier"] == {"3": 50.05}
    assert isinstance(term_state["terminal_orders"], list) and len(term_state["terminal_orders"]) == 1

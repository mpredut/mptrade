#!/usr/bin/env python3
"""Tests for the Kraken trailing_stop (no real API, no money)."""
import ast
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import signal
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from botcore import parse_dotenv
from cacheaccount import CacheTradeManager
from cachepaths import cache_path
import kraken.trailing_stop as ts_mod
from kraken.trailing_stop import KrakenTrailing
import order_guard
from providers.base import MarketDataProvider
from providers.hyperliquid_provider import HyperliquidProvider
from providers.kraken_provider import KrakenProvider
from providers.order_ohlc_utils import (
    build_open_order_entry,
    format_strategy_execution_mode,
    parse_ohlc_interval,
    parse_ohlc_interval_minutes,
    resolve_ohlc_time_window_ms,
)
from strategies import spot_engine as _xstock_strategy
from tests.conftest import TempDirTestCase, TrailingTestBase, load_venue_modules
import trailing_core
from trailing_core import should_sell

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_KRAKEN_DIR = os.path.join(_ROOT_DIR, "kraken")
xw = load_venue_modules("kraken", "kraken_xstock_watch")


class FakeK:
    def __init__(
        self,
        price=60.0,
        total=25.0,
        held=0.0,
        *,
        prices=None,
        balances=None,
        fail_balance=False,
        open_orders_fn=None,
        query_orders_fn=None,
    ):
        self.price = price
        self.total = total
        self.held = held
        self.orders = []
        self.prices = prices
        self.balances = balances
        self.fail_balance = fail_balance
        self.balance_calls = 0
        self.cancelled = []
        self._open_orders_fn = open_orders_fn
        self._query_orders_fn = query_orders_fn

    def _private(self, method, data=None):
        if method == "BalanceEx":
            self.balance_calls += 1
            if self.fail_balance:
                return {}
            if self.balances is not None:
                return self.balances
            return {"HYPE": {"balance": str(self.total), "hold_trade": str(self.held)}}
        if method == "TradesHistory":
            return {"trades": {}}
        return {}

    def last_price(self, pair):
        if self.prices is not None:
            return self.prices.get(pair, 0.0)
        return self.price

    def add_order(self, pair, side, volume, price=None, ordertype="limit", validate=False, cl_ord_id=None):
        self.orders.append(
            {
                "side": side,
                "volume": volume,
                "price": price,
                "ordertype": ordertype,
                "cl_ord_id": cl_ord_id,
            }
        )
        return {"txid": ["X"]}

    def query_orders(self, txids):
        if self._query_orders_fn is not None:
            return self._query_orders_fn(txids)
        order = self.orders[-1]
        px = order["price"] if order["price"] is not None else self.price
        return {
            str(txids): {
                "status": "closed",
                "vol_exec": str(order["volume"]),
                "cost": str(order["volume"] * px),
                "fee": "0",
            }
        }

    def open_orders(self):
        if self._open_orders_fn is not None:
            return self._open_orders_fn(self)
        return {}

    def closed_orders(self):
        return {}

    def pair_info(self, pair):
        return {"pair_decimals": 2, "lot_decimals": 8, "ordermin": "0.01"}

    def cancel_order(self, txid):
        self.cancelled.append(txid)
        self.held = 0.0
        return {"count": 1}


class _KrakenTrailingBase(TrailingTestBase):
    TRAIL_MODULE = ts_mod
    TRAIL_DEFAULTS = {"HYPE": 18.0, "TAO": 22.0, "ADA": 25.0}
    ENV_OVERRIDES = {"KRAKEN_LIVE_ORDERS": "true"}
    SUPPRESS_MODULE_NOTIFY = True

    def ts(self, client, enabled=True, min_profit_pct=0.0):
        return KrakenTrailing(client, log=lambda *a: None, enabled=enabled, state_file=self.sf,
                              min_profit_pct=min_profit_pct)


class TestTrailingKraken(_KrakenTrailingBase):
    def test_should_sell(self):
        self.assertTrue(should_sell(82, 100, 18))
        self.assertFalse(should_sell(83, 100, 18))
        self.assertFalse(should_sell(50, 0, 18))

    def test_a_rise_updates_the_peak_and_does_not_sell(self):
        c = FakeK(60.0)
        self.ts(c).check_once()
        c.price = 65.0
        ts = self.ts(c); ts.check_once()
        self.assertEqual(c.orders, [])
        self.assertEqual(json.load(open(self.sf))["HYPE"]["peak"], 65.0)

    def test_a_crash_beyond_18pct_sells_the_free_balance(self):
        c = FakeK(60.0, total=25.0, held=3.38)        # 21.62 free
        ts = self.ts(c)
        ts.check_once()                                # peak 60
        c.price = 49.0                                 # -18.3% (HYPE threshold 18%)
        ts.check_once()
        self.assertEqual(len(c.orders), 1)
        self.assertEqual(c.orders[0]["side"], "sell")
        self.assertEqual(c.orders[0]["ordertype"], "market")
        self.assertAlmostEqual(c.orders[0]["volume"], 21.62, places=2)   # only the free balance, not 25

    def test_trailing_market_false_places_limit_order(self):
        c = FakeK(60.0, total=25.0, held=3.38)
        ts = KrakenTrailing(c, log=lambda *a: None, enabled=True, state_file=self.sf,
                            trailing_market=False)
        ts.check_once()
        c.price = 49.0
        ts.check_once()
        self.assertEqual(len(c.orders), 1)
        self.assertEqual(c.orders[0]["ordertype"], "limit")
        self.assertAlmostEqual(c.orders[0]["price"], 49.0 * 0.995, places=4)

    def test_a_small_fall_does_not_sell(self):
        c = FakeK(60.0)
        ts = self.ts(c); ts.check_once()
        c.price = 56.0                                 # -6.7% < 18%
        ts.check_once()
        self.assertEqual(c.orders, [])

    def test_a_dry_run_does_not_sell(self):
        c = FakeK(60.0)
        ts = self.ts(c, enabled=False); ts.check_once()
        c.price = 48.0
        ts.check_once()
        self.assertEqual(c.orders, [])

    def test_live_switch_flip_blocks_the_final_trailing_submit(self):
        c = FakeK(60.0)
        trailing = self.ts(c)
        trailing.check_once()
        c.price = 48.0
        os.environ["KRAKEN_LIVE_ORDERS"] = "false"

        trailing.check_once()

        self.assertEqual(c.orders, [])

    def test_the_peak_survives_a_restart(self):
        c = FakeK(65.0)
        self.ts(c).check_once()
        c.price = 53.0                                 # -18.5% from 65
        self.ts(c).check_once()
        self.assertEqual(len(c.orders), 1)

    def test_sub_notional_ignored(self):
        c = FakeK(60.0, total=0.1)                     # 0.1*60 = $6 < $10
        ts = self.ts(c); ts.check_once()
        c.price = 40.0
        ts.check_once()
        self.assertEqual(c.orders, [])

    def test_without_a_free_balance_it_emits_a_heartbeat(self):
        messages = []
        c = FakeK(60.0, total=3.38, held=3.38)
        ts = KrakenTrailing(c, log=messages.append, enabled=True, state_file=self.sf)

        ts.check_once()

        self.assertEqual(c.orders, [])
        self.assertEqual(messages, ["  [TRAIL-K] heartbeat"])


class TestMinProfitKraken(_KrakenTrailingBase):
    """Require minimum profit before Kraken trailing activates."""

    def test_warming_up_does_not_sell_below_the_threshold(self):
        c = FakeK(60.0)
        ts = self.ts(c, min_profit_pct=5.0)
        ts.check_once()                    # initial=60, active at 63.0, stop at 49.2 (-18%)
        c.price = 55.0                     # normal dip of -8.3% (<18% stop): warmup suppresses sell
        ts.check_once()
        self.assertEqual(c.orders, [], "it does not sell on normal dips before reaching the profit threshold")
        c.price = 48.0                     # crash of -20% (exceeds 18% stop): extreme crash protection triggers
        ts.check_once()
        self.assertEqual(len(c.orders), 1, "extreme crash protection triggers even before warmup")
        self.assertEqual(c.orders[0]["side"], "sell")

    def test_active_past_the_threshold_sells(self):
        c = FakeK(60.0)
        ts = self.ts(c, min_profit_pct=5.0)
        ts.check_once()                    # initial=60
        c.price = 64.0; ts.check_once()   # +6.7% > 5% -> trailing active, peak=64
        c.price = 52.0; ts.check_once()   # -18.8% from peak 64 (HYPE threshold 18%)
        self.assertEqual(len(c.orders), 1, "it sells once the profit threshold is passed")
        self.assertEqual(c.orders[0]["side"], "sell")


class TestRebuyGuardsKraken(_KrakenTrailingBase):
    """Verify Kraken re-buy uses bypass_guards={'math_reference'} and defers on guard refusal."""

    def test_rebuy_executes_when_guards_allow(self):
        c = FakeK(60.0, total=25.0, held=0.0)
        ts = self.ts(c)
        rb_state = {"qty": 10.0, "sell_price": 40.0, "low": 40.0}

        with patch("order_guard.check_intelligence_guards", return_value=(True, "ok", 1.0)) as mock_guard:
            res = ts.execute_rebuy("HYPE", "HYPE", "HYPEUSD", 10.0, 42.0, rb_state, persist=lambda _: None)
            mock_guard.assert_called_once()
            self.assertEqual(mock_guard.call_args.kwargs.get("bypass_guards"), {"math_reference"})
            self.assertEqual(len(c.orders), 1)
            self.assertEqual(c.orders[0]["side"], "buy")
            self.assertEqual(res.outcome, "active")
            self.assertIsNotNone(res.intent.get("order_id"))

    def test_rebuy_defers_when_guards_block_cascade(self):
        c = FakeK(60.0, total=25.0, held=0.0)
        ts = self.ts(c)
        rb_state = {"qty": 10.0, "sell_price": 40.0, "low": 40.0}

        with patch("order_guard.check_intelligence_guards",
                   return_value=(False, "liquidation_cascade (OI -6.2%, taker 0.72)", 0.0)):
            persisted = []
            res = ts.execute_rebuy("HYPE", "HYPE", "HYPEUSD", 10.0, 42.0, rb_state, persist=persisted.append)
            self.assertEqual(len(c.orders), 0, "No buy order should be sent during cascade")
            self.assertEqual(res.outcome, "refused")
            self.assertEqual(persisted, [None], "Pending order cleared so rebuy intent persists")

    def test_rebuy_downscales_quantity_when_intelligence_guard_suggests_scale(self):
        c = FakeK(60.0, total=25.0, held=0.0)
        ts = self.ts(c)
        rb_state = {"qty": 10.0, "sell_price": 40.0, "low": 40.0}

        with patch("order_guard.check_intelligence_guards",
                   return_value=(True, "elevated_threat_downscale", 0.5)):
            res = ts.execute_rebuy("HYPE", "HYPE", "HYPEUSD", 10.0, 42.0, rb_state, persist=lambda _: None)
            self.assertEqual(len(c.orders), 1)
            self.assertEqual(c.orders[0]["side"], "buy")
            self.assertAlmostEqual(c.orders[0]["volume"], 5.0)
            self.assertAlmostEqual(res.intent.get("requested_qty"), 5.0)

    def test_rebuy_below_sell_price_bypasses_all_guards(self):
        c = FakeK(60.0, total=25.0, held=0.0)
        ts = self.ts(c)
        rb_state = {"qty": 10.0, "sell_price": 45.0, "low": 40.0}

        with patch("order_guard.check_intelligence_guards",
                   return_value=(False, "would_block_if_consulted", 0.0)) as mock_guard:
            res = ts.execute_rebuy("HYPE", "HYPE", "HYPEUSD", 10.0, 42.0, rb_state, persist=lambda _: None)
            mock_guard.assert_not_called()
            self.assertEqual(len(c.orders), 1)
            self.assertEqual(c.orders[0]["side"], "buy")
            self.assertAlmostEqual(c.orders[0]["volume"], 10.0)
            self.assertEqual(res.outcome, "active")


    def test_multi_asset_trailing_kraken(self):
        """Verify trailing protection operates on multiple configured assets (e.g. TAO, ADA)."""
        balances = {
            "HYPE": {"balance": "10.0", "hold_trade": "0.0"},
            "TAO": {"balance": "5.0", "hold_trade": "0.0"},
            "ADA": {"balance": "1000.0", "hold_trade": "0.0"},
        }
        prices = {"HYPEUSD": 50.0, "TAOUSD": 200.0, "ADAUSD": 0.50}
        c = FakeK(price=50.0, prices=prices, balances=balances)
        with patch.dict(ts_mod.PAIR_FOR, {"HYPE": "HYPEUSD", "TAO": "TAOUSD", "ADA": "ADAUSD"}, clear=False):
            ts = self.ts(c)
            ts.check_once()  # tracks initial peaks: HYPE=50, TAO=200, ADA=0.50

            # Crash TAO by -25% (exceeds 22% trail) while HYPE and ADA stay normal
            c.prices["TAOUSD"] = 150.0  # -25% from 200
            ts.check_once()

        self.assertEqual(len(c.orders), 1)
        self.assertEqual(c.orders[0]["side"], "sell")
        self.assertEqual(c.orders[0]["volume"], 5.0)

    def test_emergency_unlock_cancels_resting_sell_and_executes_stop(self):
        """When all balance is locked in a resting SELL order and price crashes past stop,
        total_qty + unlock_qty cancel the resting order (including slash pair aliases) and sell."""
        c = FakeK(
            price=60.0,
            total=20.0,
            held=0.0,
            open_orders_fn=lambda cli: (
                {"OPEN-TP-1": {"descr": {"type": "sell", "pair": "HYPE/USD", "price": "75.0"}}}
                if cli.held > 0
                else {}
            ),
        )
        ts = self.ts(c)
        ts.check_once()  # peak=60.0 while free=20.0

        # Lock all 20.0 HYPE in a resting TP order, then crash price to 48.0 (-20% > 18%)
        c.held = 20.0
        c.price = 48.0
        ts.check_once()

        self.assertEqual(c.cancelled, ["OPEN-TP-1"])
        self.assertEqual(len(c.orders), 1)
        self.assertEqual(c.orders[0]["side"], "sell")
        self.assertAlmostEqual(c.orders[0]["volume"], 20.0)

    def test_execute_sell_metadata_is_not_double_nested(self):
        c = FakeK(60.0, total=10.0, held=0.0)
        ts = self.ts(c)
        res = ts.execute_sell("HYPE", "HYPE", "HYPEUSD", 10.0, 48.0, 60.0, 18.0, persist=lambda _: None)
        self.assertTrue(res.intent.get("market"))
        self.assertNotIn("metadata", res.intent)

    def test_cachemanager_fetch_rest_into_refreshes_empty_configured_pairs_and_matches_aliases(self):
        fake_ledger_cli = SimpleNamespace(
            _private=lambda method, data=None, fresh=False: {
                "ledger": {
                    "L1": {"refid": "R1", "asset": "HYPE.F", "amount": "5.0", "type": "trade", "time": 1700000000.0},
                    "L2": {"refid": "R1", "asset": "ZUSD", "amount": "-250.0", "type": "trade", "time": 1700000000.0},
                }
            }
        )

        cfg = KrakenProvider.resolve_cache_daemon_config()
        pairs = ["HYPEUSD", "TAOUSD"]
        lookback_sec = cfg["lookback_sec"]
        cache = {"items": {"TAOUSD": [{"id": "OLD"}]}, "fetchtime": {}}
        by_sym = KrakenProvider.fetch_ledger_fills_by_symbol(
            fake_ledger_cli, pairs=pairs, lookback_sec=lookback_sec
        )
        CacheTradeManager.merge_venue_fills_into_dict(cache, by_sym, is_snapshot=True)
        self.assertEqual(len(cache["items"]["HYPEUSD"]), 1)
        self.assertEqual(cache["items"]["HYPEUSD"][0]["symbol"], "HYPEUSD")
        self.assertEqual(cache["items"]["TAOUSD"], [])
        self.assertIn("HYPEUSD", cache["fetchtime"])
        self.assertIn("TAOUSD", cache["fetchtime"])

        # WS incremental merge preserves existing WS fills while adding missing REST fills
        ws_cache = {
            "items": {
                "HYPEUSD": [{"id": "WS-1", "symbol": "HYPEUSD", "time": 1700000005000, "price": 52.0, "qty": 1.0}],
            },
            "fetchtime": {},
        }
        ws_by_sym = KrakenProvider.fetch_ledger_fills_by_symbol(
            fake_ledger_cli, pairs=pairs, lookback_sec=lookback_sec
        )
        CacheTradeManager.merge_venue_fills_into_dict(ws_cache, ws_by_sym, is_snapshot=False)
        self.assertEqual([x["id"] for x in ws_cache["items"]["HYPEUSD"]], ["R1", "WS-1"])
        CacheTradeManager.touch_venue_ws_fetchtimes_in_dict(
            ws_cache, now_ms=1700000010000, symbols=pairs, create_missing=True
        )
        self.assertEqual(ws_cache["fetchtime"]["HYPEUSD"], 1700000010000)
        self.assertEqual(ws_cache["fetchtime"]["TAOUSD"], 1700000010000)

    def test_empty_or_failed_balance_ex_does_not_wipe_peak_and_caches_per_tick(self):
        c = FakeK(price=65.0, total=25.0, held=0.0)
        ts = self.ts(c)
        ts.check_once()
        self.assertEqual(c.balance_calls, 1)
        self.assertEqual(json.load(open(self.sf))["HYPE"]["peak"], 65.0)

        # Transient empty BalanceEx must skip the tick and NOT wipe peak
        c.fail_balance = True
        ts.check_once()
        self.assertEqual(json.load(open(self.sf))["HYPE"]["peak"], 65.0)

    def test_canceled_zero_fill_order_increments_retry_attempt_and_client_order_id(self):
        c = FakeK(
            60.0,
            total=25.0,
            held=0.0,
            query_orders_fn=lambda txids: {
                str(txids): {
                    "status": "canceled",
                    "vol_exec": "0",
                    "cost": "0",
                    "fee": "0",
                }
            },
        )
        ts = self.ts(c)
        ts.check_once()  # peak=60.0
        c.price = 48.0   # -20% > 18% -> first SELL attempt
        ts.check_once()
        self.assertEqual(len(c.orders), 1)
        first_cid = c.orders[0]["cl_ord_id"]

        # Next tick: reconciles as canceled with 0 fill; next tick retries with a fresh client_order_id
        ts.check_once()
        ts.check_once()
        self.assertEqual(len(c.orders), 2)
        second_cid = c.orders[1]["cl_ord_id"]
        self.assertNotEqual(first_cid, second_cid)

    def test_trend_value_resolves_cross_venue_symbol_variants(self):
        trend_path = os.path.join(self.tmp, "trend.json")
        prev_cache = ts_mod.CACHE_TREND
        try:
            with open(trend_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "TAOUSD": {
                            "ts": time.time() - 3600.0,  # stale exact entry must not shadow fresh variant
                            "gradient_recent": 0.99,
                        },
                        "TAOUSDC": {
                            "ts": time.time(),
                            "gradient_recent": -0.42,
                        },
                    },
                    f,
                )
            ts_mod.CACHE_TREND = trend_path
            c = FakeK(60.0)
            ts = self.ts(c)
            self.assertAlmostEqual(ts.trend("TAOUSD"), -0.42)
            self.assertAlmostEqual(ts.trend("TAO/USD"), -0.42)
        finally:
            ts_mod.CACHE_TREND = prev_cache

    def test_free_balance_fails_closed_on_corrupt_hold_trade_and_unlock_qty_handles_null_descr(self):
        c = FakeK(60.0, total=25.0, held=0.0)
        ts = self.ts(c)
        ts._tick_balances = {"HYPE": {"balance": "25.0", "hold_trade": "corrupt"}}
        self.assertEqual(ts.free_qty("HYPE"), 0.0)
        ts._tick_balances = {"HYPE": {"balance": "25.0", "hold_trade": "-5.0"}}
        self.assertEqual(ts.free_qty("HYPE"), 0.0)
        ts._tick_balances = {"HYPE": {"balance": "25.0", "hold_trade": True}}
        self.assertEqual(ts.free_qty("HYPE"), 0.0)

        cancelled = []
        c.open_orders = lambda: {
            "TX-NULL-DESCR": {"descr": None, "pair": "HYPEUSD", "type": "sell"},
            "TX-NON-DICT": "malformed",
        }
        c.cancel_order = lambda txid: cancelled.append(txid) or {"count": 1}
        self.assertTrue(ts.unlock_qty("HYPEUSD"))
        self.assertEqual(cancelled, ["TX-NULL-DESCR"])

    def test_sell_trend_defer_up_to_7m_then_forces_sell(self):
        c = FakeK(60.0, total=25.0, held=0.0)
        logs = []
        ts = KrakenTrailing(c, log=logs.append, enabled=True, state_file=self.sf)
        ts.trend = lambda _pair: 1.0  # short trend UP
        with patch("trailing_core.time.time", return_value=2000.0):
            ts.check_once()  # peak=60.0
            c.price = 48.0   # -20% > 18% stop
            ts.check_once()  # t=2000s (0s < 420s): deferred by UP trend
        self.assertEqual(c.orders, [])
        self.assertTrue(any("instant trend is UP" in m for m in logs))

        # After 420s (7 minutes) at t=2420s, timer expires and forces emergency SELL
        with patch("trailing_core.time.time", return_value=2420.0):
            ts.check_once()
        self.assertEqual(len(c.orders), 1)
        self.assertEqual(c.orders[0]["side"], "sell")
        self.assertTrue(any("sell trend deferral expired" in m for m in logs))

    def test_rebuy_clamps_limit_below_sell_price_and_enforces_daily_limit_above_sell_price(self):
        c = FakeK(60.0, total=25.0, held=0.0)
        ts = self.ts(c)
        ts.check_once()  # peak=60.0
        c.price = 48.0   # -20% > 18% -> SELL @ 48.0
        ts.check_once()
        c.total = 0.0
        ts.check_once()  # terminal SELL fill confirmed -> rebuy armed with sell_price=48.0, low=48.0

        # Dip to 47.0, then bounce to 47.80 (+1.7% > 1.2% bounce, and 47.80 < 48.0 sell_price,
        # while 47.80 * 1.005 = 48.039 >= 48.0 sell_price -> limit must clamp to 47.80!)
        c.price = 47.0
        ts.check_once()
        c.price = 47.80
        ts.check_once()
        buy_orders = [o for o in c.orders if o["side"] == "buy"]
        self.assertEqual(len(buy_orders), 1)
        self.assertAlmostEqual(buy_orders[0]["price"], 47.80)

        # Now test rebuy above sell_price (price=49.0 >= sell_price=48.0) when daily_limit_guard blocks GO1
        self.tearDown(); self.setUp()
        c2 = FakeK(60.0, total=25.0, held=0.0)
        ts2 = self.ts(c2)
        ts2.check_once()
        c2.price = 48.0
        ts2.check_once()
        c2.total = 0.0
        ts2.check_once()  # rebuy armed with sell_price=48.0
        c2.price = 49.0   # +2.08% > 1.2%, above sell_price=48.0
        ts2.executor.get_orders = lambda *_a, **_k: []
        with patch("kraken.trailing_stop.order_guard.daily_limit_guard", return_value=(False, "daily_limit_exceeded")):
            ts2.check_once()
        self.assertEqual([o for o in c2.orders if o["side"] == "buy"], [])

    def test_cross_venue_shared_cost_basis_cli_dispatch_and_strategy_dry_run_gates(self):
        # 1. KrakenTrailing.cost_basis delegates to KrakenProvider.position_cost_basis / compute_fills_cost_basis
        tmp_cache_path = os.path.join(self.tmp, "kraken_trades.json")
        now_ms = int(time.time() * 1000)
        with open(tmp_cache_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "items": {
                        "HYPEUSD": [
                            {
                                "id": "T1",
                                "ordertxid": "O1",
                                "symbol": "HYPEUSD",
                                "side": "BUY",
                                "price": 40.0,
                                "qty": 10.0,
                                "fee": 0.0,
                                "feeAsset": "USD",
                                "time": now_ms - 5000,
                            },
                            {
                                "id": "T2",
                                "ordertxid": "O2",
                                "symbol": "HYPEUSD",
                                "side": "BUY",
                                "price": 50.0,
                                "qty": 15.0,
                                "fee": 0.0,
                                "feeAsset": "USD",
                                "time": now_ms - 2000,
                            },
                        ]
                    },
                    "fetchtime": {"HYPEUSD": now_ms},
                },
                f,
            )
        with patch("providers.kraken_provider._KRAKEN_CACHE_FILE", tmp_cache_path):
            c = FakeK(60.0, total=25.0, held=5.0)  # total balance = 25.0 (20 free + 5 held)
            ts = self.ts(c)
            basis = ts.cost_basis("HYPEUSD")
            # 10 @ 40 + 15 @ 50 = 400 + 750 = 1150 / 25 = 46.0
            self.assertAlmostEqual(basis, 46.0)

        # 2. Shared TrailingAdapterBase print_status and dispatch_cli
        c = FakeK(60.0, total=25.0, held=0.0)
        ts = self.ts(c)
        ts.check_once()
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = ts.dispatch_cli(SimpleNamespace(status=True, once=False))
        self.assertEqual(rc, 0)
        self.assertIn("HYPE: peak=60.0", buf.getvalue())
        self.assertIn("ENABLED=True", buf.getvalue())

        # 3. Shared MarketDataProvider.resolve_strategy_dry_run
        self.assertTrue(MarketDataProvider.resolve_strategy_dry_run(True, True, True))
        self.assertTrue(MarketDataProvider.resolve_strategy_dry_run(False, False, True))
        self.assertTrue(MarketDataProvider.resolve_strategy_dry_run(False, True, False))
        self.assertFalse(MarketDataProvider.resolve_strategy_dry_run(False, True, True))

        with patch.dict(os.environ, {"HL_LIVE_ORDERS": "true", "KRAKEN_LIVE_ORDERS": "true"}):
            self.assertFalse(HyperliquidProvider.strategy_dry_run(False, True))
            self.assertFalse(KrakenProvider.strategy_dry_run(False, True))
        with patch.dict(os.environ, {"HL_LIVE_ORDERS": "false", "KRAKEN_LIVE_ORDERS": "false"}):
            self.assertTrue(HyperliquidProvider.strategy_dry_run(False, True))
            self.assertTrue(KrakenProvider.strategy_dry_run(False, True))

        # 4. AuditedStrategyExecutor passthroughs & SSOT cache_path resolution
        c.pair_info = lambda _p: {"pair_decimals": 2, "lot_decimals": 4, "ordermin": "0.01", "base": "HYPE"}
        self.assertAlmostEqual(ts.executor.min_order_qty("HYPEUSD"), 0.01)
        self.assertAlmostEqual(ts.executor.round_quantity("HYPEUSD", 1.23459), 1.2345)
        self.assertAlmostEqual(ts.executor.round_price("HYPEUSD", 48.129), 48.13)
        hl_Live_ssot = HyperliquidProvider(token="HYPE", live_enabled=True)
        hl_Live_ssot._force_ssot_cache = True
        hl_cache_file = hl_Live_ssot._fills_cache_file()
        self.assertEqual(hl_cache_file, cache_path("cache_trade_hyperliquid.json"))

        # 5. Shared OHLC interval/window helpers, clamp_orderbook_limit & resolve_venue_instrument_spec
        self.assertEqual(parse_ohlc_interval(" 1H "), (60, "1h"))
        self.assertEqual(parse_ohlc_interval(" 1M "), (43200, "1M"))
        self.assertEqual(parse_ohlc_interval_minutes("240"), 240)
        _, s_ms, e_ms, to_f = resolve_ohlc_time_window_ms(end_ms=100_000_000, lookback_hours=1, timeout=15)
        self.assertEqual(e_ms - s_ms, 3600 * 1000)
        self.assertEqual(to_f, 15.0)
        self.assertIsNotNone(MarketDataProvider.resolve_venue_instrument_spec("kraken", "HYPEUSD"))
        self.assertIsNotNone(MarketDataProvider.resolve_venue_instrument_spec("binance", "ADAUSDC"))
        self.assertIsNone(MarketDataProvider.resolve_venue_instrument_spec("kraken", "TAOUSD"))
        oo_rec = build_open_order_entry(
            {"cl_ord_id": "abc123", "status": "open", "time": 1700000000},
            order_id="O1",
            side="BUY",
            price=25.0,
            orig_qty=2.0,
            executed_qty=0.5,
        )
        self.assertEqual(oo_rec["orderId"], "O1")
        self.assertEqual(oo_rec["clientOrderId"], "abc123")
        self.assertEqual(oo_rec["status"], "OPEN")
        self.assertEqual(oo_rec["time"], 1700000000000)
        self.assertEqual(format_strategy_execution_mode(True, True), "OBSERVE ONLY (AI managed)")
        self.assertEqual(format_strategy_execution_mode(False, True), "PAPER (no money)")
        self.assertEqual(format_strategy_execution_mode(False, False), "\u26a0 REAL \u2014 REAL MONEY")


# --- xStock Watcher & Strategy Adoption (from test_xstock.py) ---------------

class FakeKrakenXStock:
    def __init__(self, bal=None, pairs=None, price=200.0):
        self.bal = bal or {}
        self.pairs = pairs or {}
        self.price = price

    def balance(self):
        return self.bal

    def asset_pairs(self):
        return self.pairs

    def last_price(self, pair):
        return self.price

    def pair_info(self, pair):
        return {"base": "TSTX", "pair_decimals": 2, "lot_decimals": 6, "ordermin": "0.01"}

    def pair_precision(self, pair):
        info = self.pair_info(pair)
        return SimpleNamespace(
            price_decimals=info["pair_decimals"],
            volume_decimals=info["lot_decimals"],
            order_min=float(info["ordermin"]),
            base_asset=info["base"],
        )

    def free_balance(self, asset):
        return float(self.bal.get(asset, 0.0) or 0.0)


class _XStockBase(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self._env = dict(os.environ)
        path = os.path.join(self.tmp, "xstock_state.json")
        self.state_path = path
        os.environ["XSTOCK_STATE_FILE"] = path
        os.environ.update(parse_dotenv(os.path.join(_KRAKEN_DIR, "config.env")))
        self.alerts: list[str] = []
        self._xw_notify = xw.notify
        self._st_notify = _xstock_strategy.notify
        xw.notify = lambda **kw: self.alerts.append(kw.get("title", ""))
        _xstock_strategy.notify = lambda **kw: self.alerts.append(kw.get("title", ""))

    def tearDown(self):
        xw.notify = self._xw_notify
        _xstock_strategy.notify = self._st_notify
        os.environ.clear()
        os.environ.update(self._env)
        super().tearDown()


class TestWatcherPersistence(_XStockBase):
    def test_allocation_alerts_once_across_a_restart(self):
        c = FakeKrakenXStock(bal={"ZUSD": "100"})
        st = xw._load_state()
        xw.check_balance(c, st, "SPCX", False)
        self.assertEqual(len(self.alerts), 0)
        c.bal = {"ZUSD": "100", "SPCXX": "37.5"}
        xw.check_balance(c, st, "SPCX", False)
        self.assertEqual(len(self.alerts), 1)
        self.assertEqual(st["allocated"]["asset"], "SPCXX")
        xw._save_state(st)
        st = xw._load_state()
        xw.check_balance(c, st, "SPCX", False)
        self.assertEqual(len(self.alerts), 1, "a restart must not duplicate the alert")
        self.assertEqual(st["allocated"]["qty"], 37.5, "the allocation must be known after a restart")

    def test_a_new_unmatched_asset_gives_an_informational_alert(self):
        c = FakeKrakenXStock(bal={"ZUSD": "100"})
        st = xw._load_state()
        xw.check_balance(c, st, "SPCX", False)
        c.bal = {"ZUSD": "100", "RANDOMCOIN": "5"}
        xw.check_balance(c, st, "SPCX", False)
        self.assertEqual(len(self.alerts), 1)
        self.assertIsNone(st["allocated"], "a non-matching asset is not an allocation")

    def test_the_pair_prefers_the_quote_currency_and_does_not_duplicate(self):
        pairs = {"SPCXXEUR": {"wsname": "SPCXx/EUR", "base": "SPCXX"},
                 "SPCXXUSD": {"wsname": "SPCXx/USD", "base": "SPCXX"}}
        c = FakeKrakenXStock(pairs=pairs)
        st = xw._load_state()
        xw.check_pairs(c, st, "SPCX", False, quote="USD")
        self.assertEqual(st["pair"], "SPCXXUSD", "the USD quote must be preferred")
        self.assertEqual(len(self.alerts), 1)
        xw._save_state(st)
        st = xw._load_state()
        xw.check_pairs(c, st, "SPCX", False, quote="USD")
        self.assertEqual(len(self.alerts), 1, "the listing alert is not duplicated")

    def test_the_tp_and_sl_level_alerts_fire_once(self):
        c = FakeKrakenXStock(price=250.0)
        st = xw._load_state()
        self.assertIn("alerted_tp2", st)
        self.assertFalse(st["alerted_tp2"])
        st["allocated"] = {"asset": "SPCXX", "qty": 37.5}
        st["pair"] = "SPCXXUSD"
        xw.check_levels(c, st, 200.0, 20.0, 15.0, "", False)
        xw.check_levels(c, st, 200.0, 20.0, 15.0, "", False)
        self.assertEqual(len(self.alerts), 1, "the TP alert fires only once")
        c.price = 160.0
        xw.check_levels(c, st, 200.0, 20.0, 15.0, "", False)
        self.assertEqual(len(self.alerts), 2, "and the SL alert fires only once")

    def test_load_state_sanitizes_corrupt_allocated_and_bot_pid(self):
        xw._save_state({
            "known_assets": {"ZUSD": 100.0},
            "allocated": {"asset": True, "qty": "corrupt"},
            "bot_pid": True,
        })
        st = xw._load_state()
        self.assertIsNone(st["allocated"])
        self.assertIsNone(st["bot_pid"])

        xw._save_state({
            "known_assets": {"ZUSD": 100.0},
            "allocated": {"asset": " SPCXX ", "qty": "37.5", "ts": -5},
            "bot_pid": 4321.0,
        })
        st2 = xw._load_state()
        self.assertEqual(st2["allocated"]["asset"], "SPCXX")
        self.assertEqual(st2["allocated"]["qty"], 37.5)
        self.assertEqual(st2["allocated"]["ts"], 0.0)
        self.assertEqual(st2["bot_pid"], 4321)


class TestAutoStart(_XStockBase):
    def setUp(self):
        super().setUp()
        self._script, self._log = xw.BOT_SCRIPT, xw.BOT_LOG
        self.stub = os.path.join(self.tmp, "bot_stub.py")
        with open(self.stub, "w", encoding="utf-8") as f:
            f.write("import time\ntime.sleep(120)\n")
        xw.BOT_SCRIPT = self.stub
        xw.BOT_LOG = self.stub + ".log"
        os.environ["XSTOCK_AUTOSTART"] = "true"
        self.st = {"allocated": {"asset": "SPCXX", "qty": 37.5}, "pair": "SPCXXUSD",
                   "bot_pid": None, "alerted_need_price": False}

    def tearDown(self):
        if self.st.get("bot_pid"):
            try:
                os.kill(int(self.st["bot_pid"]), signal.SIGKILL)
                time.sleep(0.2)
                xw._bot_alive(self.st["bot_pid"])
            except (OSError, ValueError, TypeError):
                pass
        xw.BOT_SCRIPT, xw.BOT_LOG = self._script, self._log
        super().tearDown()

    def test_it_starts_and_is_idempotent(self):
        xw.maybe_start_bot(self.st, 200.0, False)
        pid = self.st["bot_pid"]
        self.assertTrue(xw._bot_alive(pid))
        self.assertEqual(len(self.alerts), 1)
        xw.maybe_start_bot(self.st, 200.0, False)
        self.assertEqual(self.st["bot_pid"], pid, "it does not start a second bot")
        self.assertEqual(len(self.alerts), 1)

    def test_the_watchdog_restarts_a_dead_or_zombie_bot(self):
        xw.maybe_start_bot(self.st, 200.0, False)
        pid = self.st["bot_pid"]
        os.kill(pid, signal.SIGKILL)
        time.sleep(0.3)
        self.assertFalse(xw._bot_alive(pid), "the zombie must be reaped and declared dead")
        xw.maybe_start_bot(self.st, 200.0, False)
        self.assertNotEqual(self.st["bot_pid"], pid)
        self.assertTrue(xw._bot_alive(self.st["bot_pid"]))
        self.assertIn("RESTARTED", self.alerts[-1])

    def test_it_does_not_start_without_a_price_but_asks_for_it_once(self):
        xw.maybe_start_bot(self.st, 0.0, False)
        xw.maybe_start_bot(self.st, 0.0, False)
        self.assertIsNone(self.st["bot_pid"])
        self.assertEqual(len(self.alerts), 1)
        self.assertIn("XSTOCK_ALLOC_PRICE", self.alerts[0])

    def test_autostart_disabled_does_not_start(self):
        os.environ["XSTOCK_AUTOSTART"] = "false"
        xw.maybe_start_bot(self.st, 200.0, False)
        self.assertIsNone(self.st["bot_pid"])
        self.assertEqual(len(self.alerts), 0)

    def test_it_does_not_start_without_an_allocation_or_a_pair(self):
        xw.maybe_start_bot({"allocated": None, "pair": "X", "bot_pid": None,
                            "alerted_need_price": False}, 200.0, False)
        xw.maybe_start_bot({"allocated": {"asset": "A", "qty": 1}, "pair": None,
                            "bot_pid": None, "alerted_need_price": False}, 200.0, False)
        self.assertEqual(len(self.alerts), 0)

    def test_the_paper_flag_is_passed_to_the_bot(self):
        os.environ["XSTOCK_BOT_PAPER"] = "true"
        with open(self.stub, "w") as f:
            f.write("import sys,time\nopen(sys.argv[0]+'.argv','w').write(' '.join(sys.argv[1:]))\ntime.sleep(120)\n")
        xw.maybe_start_bot(self.st, 200.0, False)
        time.sleep(0.6)
        argv = open(self.stub + ".argv").read()
        os.remove(self.stub + ".argv")
        self.assertIn("--paper", argv)
        self.assertIn("--pair SPCXXUSD", argv)


class TestAdoption(_XStockBase):
    PAIR = "TSTXUSD"

    def setUp(self):
        super().setUp()
        os.environ.update(parse_dotenv(os.path.join(_KRAKEN_DIR, "config.env")))
        os.environ["STRAT_ADOPT_COST"] = "200"
        sp = _xstock_strategy.state_path_for(self.PAIR)
        if os.path.exists(sp):
            os.remove(sp)

    def tearDown(self):
        sp = _xstock_strategy.state_path_for(self.PAIR)
        if os.path.exists(sp):
            os.remove(sp)
        super().tearDown()

    def _strategy(self, client):
        return _xstock_strategy.Strategy(client, self.PAIR, _xstock_strategy.StratParams.from_env(), dry_run=True, desktop=False)

    def test_adopts_from_the_balance_and_places_a_tp(self):
        s = self._strategy(FakeKrakenXStock(bal={"TSTX": "37.5"}))
        s.step(200.0)
        self.assertTrue(s.s["adopted"])
        self.assertAlmostEqual(s.s["qty"], 37.5, delta=0.001)
        self.assertLess(s.s["qty"], 37.5 + 1e-12, "it never sells more than the ledger")
        self.assertAlmostEqual(s.s["cost"] / s.s["qty"], 200.0)
        sells = [o for o in s.s["orders"] if o["side"] == "sell" and o["kind"] == "TP"]
        self.assertEqual(
            len(sells), 0,
            "configured trend-hold waits for the TP threshold before trailing",
        )

    def test_waits_for_the_allocation_without_buying(self):
        c = FakeKrakenXStock(bal={"ZUSD": "100"})
        s = self._strategy(c)
        s.step(100.0)
        self.assertEqual(s.s["orders"], [], "it does NOT buy a new entry while waiting")
        c.bal = {"ZUSD": "100", "TSTX": "37.5"}
        s.step(200.0)
        self.assertTrue(s.s["adopted"])

    def test_it_does_not_adopt_over_a_cycle_in_progress(self):
        with open(_xstock_strategy.state_path_for(self.PAIR), "w") as f:
            json.dump({"cycle": 3, "qty": 2.46, "cost": 150.0, "spent": 150.0,
                       "dca_buys": 1, "orders": []}, f)
        s = self._strategy(FakeKrakenXStock(bal={"TSTX": "99"}))
        s._maybe_adopt()
        self.assertFalse(s.s.get("adopted", False))
        self.assertEqual(s.s["qty"], 2.46, "the existing position remains untouched")

    def test_it_does_not_re_adopt_after_a_restart(self):
        s = self._strategy(FakeKrakenXStock(bal={"TSTX": "37.5"}))
        s._maybe_adopt()
        q1 = s.s["qty"]
        s._save()
        s2 = self._strategy(FakeKrakenXStock(bal={"TSTX": "99999"}))
        s2._maybe_adopt()
        self.assertEqual(s2.s["qty"], q1, "a restart does not re-adopt or duplicate")

    def test_the_allocation_does_not_consume_the_dca_cap(self):
        s = self._strategy(FakeKrakenXStock(bal={"TSTX": "37.5"}))
        s._maybe_adopt()
        self.assertEqual(s.s["spent"], 0.0, "the cap remains available for additional funds")

# ============================================================================
# Domain Regressions & Input Hygiene (Targeted Validation)
# ============================================================================


def test_krakentrailing__submit_lifecycle_order_normalizes_bypass_guards_all_all_iter7_s3(monkeypatch):
    """[ITER7 #3] KrakenTrailing._submit_lifecycle_order normalizes bypass_guards='ALL' / {'ALL'}."""
    # 3. KrakenTrailing._submit_lifecycle_order normalizes bypass_guards="ALL" / {"ALL"}
    eng = object.__new__(KrakenTrailing)
    eng.executor = MagicMock()
    eng.order_lifecycle = MagicMock()
    eng.order_lifecycle.submit.return_value = MagicMock(outcome="accepted")
    guard_called = []
    monkeypatch.setattr(
        order_guard,
        "check_intelligence_guards",
        lambda *a, **k: guard_called.append(1) or (False, "should_be_skipped", 1.0),
    )
    res = eng._submit_lifecycle_order(
        intent={"intent_id": "id1", "client_order_id": "cid1"},
        persist=lambda x: None, pair="SOLUSD", side="BUY",
        qty=1.0, price=100.0, kind="TRAIL", bypass_guards="ALL", market=False,
    )
    assert res.outcome == "accepted"
    assert guard_called == []

def test_trailing_core_should_sell_krakentrailing__trend_value_previous_recovery_iter21_s3(tmp_path, monkeypatch):
    """[ITER21 #3] trailing_core.should_sell & KrakenTrailing._trend_value .previous recovery."""
    # 3. trailing_core.should_sell & KrakenTrailing._trend_value .previous recovery
    assert trailing_core.should_sell(0.0, 100.0, 15.0) is False
    assert trailing_core.should_sell(-5.0, 100.0, 15.0) is False
    assert trailing_core.should_sell(float("nan"), 100.0, 15.0) is False
    assert trailing_core.should_sell(80.0, 100.0, 15.0) is True

    trend_file = tmp_path / "cache_short_trend.json"
    trend_prev = tmp_path / "cache_short_trend.json.previous"
    trend_file.write_text("{corrupt", encoding="utf-8")
    trend_prev.write_text(
        json.dumps({"HYPEUSDC": {"ts": time.time() * 1000.0, "gradient_recent": 0.42}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(ts_mod, "CACHE_TREND", str(trend_file))
    kt_inst = object.__new__(ts_mod.KrakenTrailing)
    assert kt_inst.trend("HYPE/USD") == pytest.approx(0.42)

def test_kraken_trailing_stop_krakentrailing_reconcile_pending_execute_rebuy_nan_bool_safety_iter52_s3(tmp_path):
    """[ITER52 #3] kraken/trailing_stop: KrakenTrailing.reconcile_pending & execute_rebuy NaN/bool safety."""
    # 3. kraken/trailing_stop: KrakenTrailing.reconcile_pending & execute_rebuy NaN/bool safety
    kt = ts_mod.KrakenTrailing(
        client=SimpleNamespace(),
        log=lambda _m: None,
        enabled=False,
        state_file=str(tmp_path / "kt_state.json"),
    )
    kt.order_lifecycle = SimpleNamespace(
        reconcile=lambda pending, persist: SimpleNamespace(
            outcome="terminal",
            status=SimpleNamespace(status="closed", filled_qty=float("nan"), cost=float("nan"), fee=0.0),
            intent=pending,
        ),
        new_intent=lambda **kw: kw,
    )
    st_dict = {"retry_counts": {}}
    persist_fn = lambda _val: None
    persist_fn.st = st_dict
    pending_obj = {"symbol": "HYPEUSD", "action": "SELL", "submitted_qty": float("nan"), "attempt": True}
    res = kt.reconcile_pending(pending_obj, persist_fn)
    assert res.outcome == "terminal"
    assert st_dict["retry_counts"]["SELL"] == 1

    captured_intent = {}
    kt._submit_lifecycle_order = lambda intent, *a, **kw: (
        captured_intent.update(intent) or SimpleNamespace(outcome="refused", intent=intent)
    )
    kt.execute_rebuy(
        "HYPE", "HYPE", "HYPEUSD", 1.0, 25.0,
        {"low": float("nan"), "sell_price": True},
        persist_fn,
    )
    assert captured_intent["metadata"]["anchor"] == 25.0
    assert captured_intent["metadata"]["sell_price"] == 0.0

def test_krakentrailing__sync_retry_count_from_state__order_intent_iter57_s3():
    """[ITER57 #3] KrakenTrailing._sync_retry_count_from_state & _order_intent."""
    # 3. KrakenTrailing._sync_retry_count_from_state & _order_intent
    adapter = object.__new__(KrakenTrailing)
    adapter._retry_counts = {}
    adapter.order_lifecycle = MagicMock()
    adapter.order_lifecycle.new_intent.side_effect = lambda **kw: kw
    dummy_persist = type("P", (), {"st": {"retry_counts": {"SELL": True}}})()
    adapter._sync_retry_count_from_state("HYPEUSD", "SELL", dummy_persist)
    assert adapter._retry_counts.get(("HYPEUSD", "SELL"), 0) == 0
    intent = adapter._order_intent("SELL", "HYPE", "HYPE", "HYPEUSD", 1.5, 20.0, anchor=None)
    assert ":20.00000000" in intent["intent_id"]

def test_krakentrailing_string_enabled_trailing_market_and_reconcile_pending_safety_iter117_s2(tmp_path):
    """[ITER117 #2] KrakenTrailing string enabled / trailing_market and reconcile_pending safety."""
    # 2. KrakenTrailing string enabled / trailing_market and reconcile_pending safety
    kt = KrakenTrailing(
        client=object(),
        enabled="false",
        trailing_market="0",
        state_file=str(tmp_path / "kt_state.json"),
    )
    assert kt.enabled is False
    assert kt.trailing_market is False
    assert kt.reconcile_pending(None, lambda _p: None) is None
    assert kt.reconcile_pending("not-a-dict", lambda _p: None) is None
    kt.order_lifecycle = SimpleNamespace(
        reconcile=lambda _p, persist: SimpleNamespace(outcome="absent", status=None)
    )
    kt.reconcile_pending({"symbol": " XXBTZUSD ", "action": " sell "}, lambda _p: None)
    assert kt._retry_counts.get(("XXBTZUSD", "SELL")) == 1


def test_trailing_core_is_venue_agnostic_ast():
    """trailing_core.py must contain zero venue SDK imports or hardcoded venue string branches."""
    src = Path(trailing_core.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)

    docstring_nodes = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                docstring_nodes.add(node.body[0].value)

    forbidden_venues = ("binance", "kraken", "hyperliquid", "t212", "212trading")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0].lower()
                assert not any(v in root for v in forbidden_venues), f"Forbidden import {alias.name}"
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0].lower()
            assert not any(v in root for v in forbidden_venues), f"Forbidden import from {node.module}"
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node not in docstring_nodes:
            low = node.value.lower()
            assert not any(v in low for v in forbidden_venues), (
                f"trailing_core.py line {node.lineno} contains hardcoded venue literal {node.value!r}"
            )


def test_kraken_tokenized_asset_routing_and_spv_pair_matching(monkeypatch):
    from cachepaths import is_kraken_tokenized_asset, kraken_pair_matches
    from kraken.kraken_client import KrakenClient

    assert is_kraken_tokenized_asset("ANTxUSD") is True
    assert is_kraken_tokenized_asset("SPCXx/USD") is True
    assert is_kraken_tokenized_asset("SPCXSPVUSD") is True
    assert is_kraken_tokenized_asset("HYPEUSD") is False
    assert is_kraken_tokenized_asset("XBTUSD") is False
    assert is_kraken_tokenized_asset("AVAXUSD") is False

    assert kraken_pair_matches("SPCXxUSD", "SPCXSPV/USD") is True
    assert kraken_pair_matches("SPCXXUSD", "SPCXSPV/USD") is True
    assert kraken_pair_matches("SPCXXUSD", "SPCXxUSD") is True
    assert kraken_pair_matches("ANTXUSD", "ANTSPV/USD") is True
    assert kraken_pair_matches("ANTXUSD", "ANTxUSD") is True
    assert kraken_pair_matches("ANTxUSD", "ANTSPV/USD") is True
    assert kraken_pair_matches("AVAXUSD", "AVAUSD") is False
    assert kraken_pair_matches("AVAUSD", "AVAXUSD") is False

    calls = []
    cli = KrakenClient("key", "c2VjcmV0")
    monkeypatch.setattr(
        cli,
        "_public",
        lambda method, params=None, fresh=False: calls.append(("pub", method, dict(params or {})))
        or (
            {"ANTxUSD": {"c": ["95.25", "1.0"], "pair_decimals": 2, "lot_decimals": 8, "ordermin": "0.05", "base": "ANTx"}}
            if method != "OHLC"
            else {"ANTxUSD": [[1700000000, "90", "96", "89", "95", "93", "10", 5], [1700000300, "95", "96", "94", "95.5", "95", "2", 1]], "last": 1700000300}
        ),
    )
    monkeypatch.setattr(
        cli,
        "_private",
        lambda method, data=None, fresh=False: calls.append(("priv", method, dict(data or {}))) or {"txid": ["OTOK-1"]},
    )

    assert cli.last_price("ANTxUSD") == pytest.approx(95.25)
    assert calls[-1] == ("pub", "Ticker", {"pair": "ANTxUSD", "asset_class": "tokenized_asset"})

    info = cli.pair_info("ANTxUSD")
    assert info["base"] == "ANTx"
    assert calls[-1] == ("pub", "AssetPairs", {"pair": "ANTxUSD", "aclass_base": "tokenized_asset"})

    assert cli.ohlc_closes("ANTxUSD", 5) == [95.0]
    assert calls[-1] == ("pub", "OHLC", {"pair": "ANTxUSD", "interval": 5, "asset_class": "tokenized_asset"})

    cli.add_order("ANTxUSD", "sell", 2.5, price=100.0)
    assert calls[-1][0:2] == ("priv", "AddOrder")
    assert calls[-1][2]["pair"] == "ANTxUSD"
    assert calls[-1][2]["asset_class"] == "tokenized_asset"

    cli.add_order("HYPEUSD", "sell", 1.0, price=70.0)
    assert "asset_class" not in calls[-1][2]


def test_xstock_watch_ledger_alloc_price_and_regex_target_sync():
    import re
    cfg = parse_dotenv(os.path.join(_KRAKEN_DIR, "config.env"))
    # Live config disables pending ANTx IPO target while preserving active SPCXxUSD supervision.
    assert cfg.get("XSTOCK_REGEX", "") == ""
    assert "SPCXxUSD" in cfg.get("XSTOCK_ACTIVE_PAIRS", "")

    rx = r"^(?:ANTx|ANTSPV)(?:[./]|USD|EUR|$)"
    for hit in ("ANTx.T", "ANTx", "ANTxUSD", "ANTx/USD", "ANTSPVUSD"):
        assert re.search(rx, hit, re.I), f"expected {hit} to match {rx}"
    for miss in ("MANTRA", "MANTRAUSD", "ANTOx", "ANTOxUSD", "ANTOSPV", "SPCXx.T"):
        assert not re.search(rx, miss, re.I), f"expected {miss} NOT to match {rx}"

    st = {
        "known_assets": {"SPCXx.T": 15.74, "HYPE": 26.32},
        "allocated": {"asset": "SPCXx.T", "qty": 4.2786},
        "pair": "SPCXxUSD",
        "alerted_pair": True,
        "alerted_tp": True,
        "alerted_tp2": False,
        "alerted_sl": True,
        "bot_pid": None,
        "alerted_need_price": False,
    }
    assert xw.sync_target_regex(st, rx) is True
    assert st["allocated"] is None
    assert st["pair"] is None
    assert st["alerted_tp"] is False
    assert st["known_assets"] == {"SPCXx.T": 15.74, "HYPE": 26.32}

    # Empty XSTOCK_REGEX clears stale ANTx target/bot_pid state while preserving known_assets.
    st_ant = {
        "known_assets": {"ANTx.T": 0.001, "SPCXx.T": 1e-6},
        "allocated": {"asset": "ANTx.T", "qty": 0.531, "alloc_price": 87.1499},
        "pair": "ANTxUSD",
        "alerted_pair": True,
        "alerted_tp": True,
        "alerted_tp2": True,
        "alerted_sl": False,
        "bot_pid": 86652,
        "alerted_need_price": False,
    }
    assert xw.sync_target_regex(st_ant, "") is True
    assert st_ant["allocated"] is None
    assert st_ant["pair"] is None
    assert st_ant["bot_pid"] is None
    assert st_ant["known_assets"] == {"ANTx.T": 0.001, "SPCXx.T": 1e-6}
    assert xw.sync_target_regex(st_ant, "") is False

    # Reservation-only ledger (before allocation settles) must return None
    pre_cli = SimpleNamespace(
        _private=lambda method, params=None: {
            "ledger": {
                "L1": {"type": "transfer", "subtype": "krakenlaunchallocation", "asset": "ZUSD", "amount": "2764.4600", "balance": "2764.4601", "refid": "R1", "time": 1794800000},
                "L2": {"type": "transfer", "subtype": "krakenlaunchallocation", "asset": "ZUSD", "amount": "-2764.4600", "balance": "0.0055", "refid": "R1", "time": 1794800000},
            }
        }
    )
    assert xw.resolve_ledger_alloc_price(pre_cli, "ANTx.T", 35.0) is None

    # Settled allocation ledger (reproduces exact SPCX 606.4915 / 4.2786 = 141.75)
    settled_cli = SimpleNamespace(
        _private=lambda method, params=None: {
            "ledger": {
                "L4": {"type": "transfer", "subtype": "krakenlaunchallocation", "asset": "ZUSD", "amount": "-606.4915", "balance": "0.0001", "refid": "R_DEBIT", "time": 1794901000},
                "L3": {"type": "transfer", "subtype": "krakenlaunchallocation", "asset": "ANTx", "amount": "4.278600", "balance": "4.278600", "refid": "R_TOK", "time": 1794900500},
                "L2": {"type": "transfer", "subtype": "krakenlaunchallocation", "asset": "ZUSD", "amount": "2157.9685", "balance": "2157.9740", "refid": "R_REFUND", "time": 1794900000},
                "L1": {"type": "transfer", "subtype": "krakenlaunchallocation", "asset": "ZUSD", "amount": "-2157.9685", "balance": "606.4916", "refid": "R_REFUND", "time": 1794900000},
            }
        }
    )
    assert xw.resolve_ledger_alloc_price(settled_cli, "ANTx.T", 4.2786) == pytest.approx(141.75)


def test_spot_engine_hybrid_tranche_plus_trailing_and_zero_reentry(tmp_path):
    pair = "TSTXUSD"
    sp = _xstock_strategy.state_path_for(pair, str(tmp_path))
    if os.path.exists(sp):
        os.remove(sp)
    cfg = parse_dotenv(os.path.join(_KRAKEN_DIR, "config.env"))
    env_merged = dict(cfg)
    for src, dst in (
        ("XSTOCK_BOT_ENTRY", "STRAT_ENTRY"),
        ("XSTOCK_BOT_DCA", "STRAT_DCA"),
        ("XSTOCK_BOT_MAX_DCA_BUYS", "STRAT_MAX_DCA_BUYS"),
        ("XSTOCK_BOT_DCA_DROP_PCT", "STRAT_DCA_DROP_PCT"),
        ("XSTOCK_BOT_TP_PCT", "STRAT_TAKEPROFIT_PCT"),
        ("XSTOCK_BOT_TP_TRANCHES", "STRAT_TP_TRANCHES"),
        ("XSTOCK_BOT_TP_TREND_HOLD", "STRAT_TP_TREND_HOLD"),
        ("XSTOCK_BOT_TP_TRAIL_PCT", "STRAT_TP_TRAIL_PCT"),
        ("XSTOCK_BOT_TP_TRAIL_ADAPTIVE", "STRAT_TP_TRAIL_ADAPTIVE"),
        ("XSTOCK_BOT_TP_MDHTP_ENABLED", "STRAT_TP_MDHTP_ENABLED"),
        ("XSTOCK_BOT_TP_DYNAMIC_FLAT", "STRAT_TP_DYNAMIC_FLAT"),
        ("XSTOCK_BOT_IPO_TP_TRANCHES", "STRAT_IPO_TP_TRANCHES"),
        ("XSTOCK_BOT_IPO_TRAIL_PCT", "STRAT_IPO_TRAIL_PCT"),
        ("XSTOCK_BOT_IPO_REENTRY_DROP_PCT", "STRAT_IPO_REENTRY_DROP_PCT"),
        ("XSTOCK_BOT_SL_PCT", "STRAT_STOP_LOSS_PCT"),
        ("XSTOCK_BOT_TREND_OVERLAY", "STRAT_TREND_OVERLAY"),
        ("XSTOCK_BOT_MAX_BUDGET", "STRAT_MAX_BUDGET"),
        ("XSTOCK_BOT_CHECK_MINUTES", "STRAT_CHECK_MINUTES"),
    ):
        env_merged[dst] = cfg[src]
    assert cfg["XSTOCK_BOT_ENTRY"] == "0"
    assert cfg["XSTOCK_BOT_MAX_DCA_BUYS"] == "0"
    env_merged["STRAT_ENTRY"] = "500"
    env_merged["STRAT_ADOPT_COST"] = "80.0"

    params = _xstock_strategy.StratParams.from_env(env_merged)
    client = FakeKrakenXStock(bal={"TSTX": "30.0", "ZUSD": "3000.0"})
    client._ohlc_bars = [95.0, 96.0, 97.0]  # initially < 20 bars (young IPO!)
    client.ohlc_closes = lambda _p, _itv: list(client._ohlc_bars)
    strat = _xstock_strategy.Strategy(
        client, pair, params, dry_run=True, desktop=False, state_dir=str(tmp_path)
    )
    strat.notify = lambda **kw: None
    sim_clock = [1000.0]
    strat._clock = lambda: sim_clock[0]

    def _tick(px: float, ts: float) -> None:
        sim_clock[0] = ts
        strat.reconcile(px)
        strat.step(px, timestamp=ts)

    # 1. Phase 1 adopts 30 TSTX @ 80 and places ONLY Tranche 0 (50% = 15.0 TSTX @ +10% = 88.00), holding 50% off-book
    _tick(82.0, 1000.0)
    assert strat.s["adopted"] is True
    assert strat.s["ipo_matured"] is False
    assert strat.s["qty"] == pytest.approx(30.0, abs=1e-4)
    sells = [o for o in strat.s["orders"] if o["side"] == "sell"]
    assert len(sells) == 1
    assert sells[0]["vol"] == pytest.approx(15.0, abs=1e-4)
    assert sells[0]["price"] == pytest.approx(88.00)

    # 2. Opening dip to 68 (-15%) does NOT trigger stop-loss or DCA buy during Phase 1
    _tick(68.0, 1015.0)
    assert strat.s["qty"] == pytest.approx(30.0, abs=1e-4)
    assert not any(o["side"] == "buy" for o in strat.s["orders"])

    # 3. Modest pop to 94 (+17.5%) fills Tranche 0 (15.0 @ 88.00) and arms trailing (>= +16% = 92.80) on remaining 15.0 TSTX
    _tick(94.0, 1030.0)
    assert strat.s["qty"] == pytest.approx(15.0, abs=1e-4)
    assert [o for o in strat.s["orders"] if o["side"] == "sell"] == []
    assert strat.s["trail_peak"] == pytest.approx(94.0)

    # 4. Rally continues to 100 (+25%) -> peak rises to 100, IPO 3.5% trail_stop = 100 * 0.965 = 96.50
    #    A normal -2.0% wiggle (98.00) in 5m does NOT trigger crypto Fast-Profit-Guard (1.0%) during Phase 1!
    _tick(100.0, 1045.0)
    assert strat.s["qty"] == pytest.approx(15.0, abs=1e-4)
    assert strat.s["trail_peak"] == pytest.approx(100.0)
    assert strat.s["trail_stop"] == pytest.approx(96.50)
    _tick(98.00, 1052.0)
    assert strat.s["qty"] == pytest.approx(15.0, abs=1e-4)
    assert [o for o in strat.s["orders"] if o["side"] == "sell"] == []

    # 5. Pullback > 3.5% to 96.20 triggers trailing market exit of the remaining 15.0 TSTX
    _tick(96.20, 1060.0)
    _tick(96.00, 1075.0)
    assert strat.s["qty"] == pytest.approx(0.0, abs=1e-6)
    assert strat.s["ipo_peak"] == pytest.approx(100.0)

    # 6. Post-IPO Maturation Gate blocks premature re-entry at 85 (< 20% drop from 100) and when < 20 4h bars exist
    _tick(85.0, 1090.0)
    assert strat.s["ipo_matured"] is False
    assert strat.s["orders"] == []
    _tick(78.0, 1105.0)  # >= 20% drop (<= 80), but still only 3 OHLC bars (< 20)!
    assert strat.s["ipo_matured"] is False
    assert strat.s["orders"] == []

    # 7. Once >= 20 4h bars exist AND price bounces +0.6% off trough 78.0 -> 78.50, Phase 2 unlocks automatically!
    client._ohlc_bars = [78.0 + (i % 2) * 0.2 for i in range(25)]
    _tick(78.50, 200_000.0)
    assert strat.s["ipo_matured"] is True
    buys = [o for o in strat.s["orders"] if o["side"] == "buy" and o["kind"] == "ENTRY"]
    assert len(buys) == 1
    assert buys[0]["amount"] == pytest.approx(500.0)


    # 8. In observe_only=True (AI-managed mode), step() still adopts at STRAT_ADOPT_COST without overwriting entry_price with market price
    obs_strat = _xstock_strategy.Strategy(
        FakeKrakenXStock(bal={"TSTX": "30.0", "ZUSD": "3000.0"}),
        pair,
        params,
        dry_run=False,
        desktop=False,
        observe_only=True,
        initial_state=_xstock_strategy._new_state(),
        state_dir=str(tmp_path),
    )
    obs_strat.notify = lambda **kw: None
    obs_strat.step(115.0, timestamp=1000.0)
    assert obs_strat.s["adopted"] is True
    assert obs_strat.s["entry_price"] == pytest.approx(80.0)
    assert obs_strat.s["orders"] == []


def test_xstock_watch_incremental_trade_archive_sync(tmp_path, monkeypatch):
    from state_io import meta_file_path, read_json_dict, read_jsonl_dicts

    monkeypatch.setenv("CACHEDB_DIR", str(tmp_path))
    monkeypatch.setenv("XSTOCK_ARCHIVE_PAIRS", "SPCXxUSD")
    xw_fresh = load_venue_modules("kraken", "kraken_xstock_watch")
    monkeypatch.setattr(
        xw_fresh,
        "cache_path",
        lambda fn: str(tmp_path / "prices" / os.path.basename(fn)),
    )

    seen_calls = []

    def _fake_trades_page(pair, since="0"):
        seen_calls.append((pair, since))
        if pair == "SPCXxUSD" and since == "0":
            return [["142.50", "1.5", 1794990000.0, "b", "l", ""]], "1794990000000000000"
        if pair == "SPCXxUSD" and since == "1794990000000000000":
            return [["143.10", "2.0", 1794990060.0, "b", "m", ""]], "1794990060000000000"
        if pair == "ANTxUSD" and since == "0":
            return [["82.00", "5.0", 1795070000.0, "b", "m", ""]], "1795070000000000000"
        return [], since

    cli = SimpleNamespace(trades_page=_fake_trades_page)

    # Cycle 1 before ANTxUSD lists: syncs SPCXxUSD from 0
    xw_fresh.sync_configured_archives(cli, {"pair": None})
    spcx_file = tmp_path / "prices" / "cache_365_price_SPCXxUSD.jsonl"
    assert spcx_file.exists()
    assert len(read_jsonl_dicts(str(spcx_file))) == 1

    # Cycle 2 after ANTxUSD lists: incrementally appends SPCXxUSD from cursor AND bootstraps ANTxUSD from 0
    xw_fresh.sync_configured_archives(cli, {"pair": "ANTxUSD"})
    ant_file = tmp_path / "prices" / "cache_365_price_ANTxUSD.jsonl"
    assert ant_file.exists()
    spcx_rows = read_jsonl_dicts(str(spcx_file))
    ant_rows = read_jsonl_dicts(str(ant_file))
    assert len(spcx_rows) == 2
    assert spcx_rows[-1]["i"][1] == pytest.approx(143.10)
    assert len(ant_rows) == 1
    assert ant_rows[0]["s"] == "ANTxUSD"
    assert ant_rows[0]["i"][1] == pytest.approx(82.00)
    spcx_meta = read_json_dict(meta_file_path(str(spcx_file)))
    assert spcx_meta["kraken_trades_since_ns"] == "1794990060000000000"
    assert ("SPCXxUSD", "1794990000000000000") in seen_calls

    # 2. ensure_active_xstock_bots honors instruments.conf enabled = no (does not spawn disabled xStock bots)
    (tmp_path / "instruments.conf").write_text(
        "[KRAKEN_SPCX]\nprovider = kraken\nsymbol = SPCXxUSD\nbase = SPCXx\nquote = USD\n"
        "enabled = no\nisolation = own_ledger\nmarket_hours = 24x7\n"
        "role.mt = no\nrole.tradeall_fire = no\nrole.kalman_primary = no\n"
        "role.trailing = no\nrole.assetguardian = no\nrole.archive = no\n"
        "role.rtrade = no\nrole.force_sell = no\n",
        encoding="utf-8",
    )
    spawned_pairs = []
    monkeypatch.setenv("XSTOCK_AUTOSTART", "true")
    monkeypatch.setenv("XSTOCK_ACTIVE_PAIRS", "SPCXxUSD:141.75")
    monkeypatch.setattr(xw_fresh, "_spawn_xstock_bot", lambda pair, *a, **kw: spawned_pairs.append(pair))
    xw_fresh.ensure_active_xstock_bots(workspace_dir=str(tmp_path))
    assert spawned_pairs == []


def test_declarative_xstock_role_gating_uppercase_resolution_and_archive_ohlc(tmp_path):
    from instrument_registry import (
        discover_tradable_specs,
        is_xstock_trading_enabled,
        load_registry,
        resolve_instrument_spec,
        resolve_registered_symbol_case,
        symbols_for,
    )
    from kraken.kraken_client import _resolve_kraken_pair
    from kraken.market_data import load_archive_ohlc_candles
    from providers.price_poller import KrakenPriceStream
    from state_io import atomic_write_jsonl

    # 1. Live instruments.conf enables SPCXxUSD for AI trading (ai.tradable=yes), while ANTxUSD is price/archive-only (ai.tradable=no)
    assert resolve_registered_symbol_case("SPCXXUSD", "kraken") == "SPCXxUSD"
    assert resolve_registered_symbol_case("ANTXUSD", "kraken") == "ANTxUSD"
    assert _resolve_kraken_pair("SPCXXUSD") == ("SPCXxUSD", True)
    assert _resolve_kraken_pair("ANTXUSD") == ("ANTxUSD", True)
    assert is_xstock_trading_enabled("SPCXxUSD") is True
    assert is_xstock_trading_enabled("ANTxUSD") is False
    assert is_xstock_trading_enabled("AAPLxUSD") is False
    assert symbols_for("kraken", "mt") == ["HYPEUSD"]
    live_reg = load_registry()
    assert resolve_instrument_spec("SPCXxUSD", "kraken", live_reg) is not None
    assert resolve_instrument_spec("ANTxUSD", "kraken", live_reg) is None
    assert resolve_instrument_spec("AAPLxUSD", "kraken", live_reg) is None
    crypto_tradable = discover_tradable_specs(asset_scope="crypto")
    assert "KRAKEN_SPCX" in crypto_tradable
    assert "KRAKEN_ANT" not in crypto_tradable

    # 2. Setting ai.tradable=no and role.mt=no declaratively locks an xStock out of execution
    locked_conf = tmp_path / "instruments_locked.conf"
    locked_conf.write_text(
        "[KRAKEN_SPCX]\n"
        "provider = kraken\n"
        "symbol = SPCXxUSD\n"
        "base = SPCXx\n"
        "quote = USD\n"
        "enabled = yes\n"
        "isolation = own_ledger\n"
        "market_hours = 24x7\n"
        "role.mt = no\n"
        "role.tradeall_fire = no\n"
        "role.kalman_primary = no\n"
        "role.trailing = no\n"
        "role.assetguardian = no\n"
        "role.archive = yes\n"
        "role.rtrade = no\n"
        "role.force_sell = no\n",
        encoding="utf-8",
    )
    assert is_xstock_trading_enabled("SPCXxUSD", path=str(locked_conf)) is False
    assert resolve_instrument_spec("SPCXxUSD", "kraken", load_registry(str(locked_conf))) is None

    # 3. Flipping role.mt = yes or ai.tradable = yes unlocks both registry and AI trading scope
    custom_conf = tmp_path / "instruments.conf"
    custom_conf.write_text(
        "[KRAKEN_SPCX]\n"
        "provider = kraken\n"
        "symbol = SPCXxUSD\n"
        "base = SPCXx\n"
        "quote = USD\n"
        "enabled = yes\n"
        "isolation = own_ledger\n"
        "market_hours = 24x7\n"
        "role.mt = yes\n"
        "role.tradeall_fire = no\n"
        "role.kalman_primary = no\n"
        "role.trailing = no\n"
        "role.assetguardian = no\n"
        "role.archive = yes\n"
        "role.rtrade = no\n"
        "role.force_sell = no\n",
        encoding="utf-8",
    )
    assert is_xstock_trading_enabled("SPCXxUSD", path=str(custom_conf)) is True
    unlocked_reg = load_registry(str(custom_conf))
    resolved = resolve_instrument_spec("SPCXxUSD", "kraken", unlocked_reg)
    assert resolved is not None and resolved.symbol == "SPCXxUSD"
    # Raw .T balance tokens remain non-tradable even when pair is unlocked
    assert is_xstock_trading_enabled("SPCXx.T", path=str(custom_conf)) is False
    assert resolve_instrument_spec("SPCXx.T", "kraken", unlocked_reg) is None

    # 3. KrakenPriceStream skips tokenized_asset pairs on v1 WS so crypto WS stays clean
    ks = KrakenPriceStream(["HYPEUSD", "SPCXxUSD"])
    assert ks._ws_pairs == ["HYPE/USD"]

    # 4. load_archive_ohlc_candles reconstructs completed OHLC bars offline from JSONL tick archive
    arch_file = tmp_path / "cache_365_price_SPCXxUSD.jsonl"
    base_ms = 1_781_283_000_000
    atomic_write_jsonl(
        str(arch_file),
        [
            {"s": "SPCXxUSD", "i": [base_ms + 10_000, 140.0, 1.0]},
            {"s": "SPCXxUSD", "i": [base_ms + 90_000, 145.5, 2.0]},
            {"s": "SPCXxUSD", "i": [base_ms + 110_000, 139.2, 1.0]},
            {"s": "SPCXxUSD", "i": [base_ms + 115_000, 143.0, 1.0]},
            {"s": "SPCXxUSD", "i": [base_ms + 360_000, 144.0, 1.0]},  # incomplete tail bar dropped
        ],
    )
    bars = load_archive_ohlc_candles("SPCXXUSD", 5, drop_incomplete=True, archive_path=str(arch_file))
    assert len(bars) == 1
    assert (bars[0]["open"], bars[0]["high"], bars[0]["low"], bars[0]["close"]) == (140.0, 145.5, 139.2, 143.0)


def test_xstock_audit_fixes_bbo_clamp_gap_tranche_post_only_and_ipo_env_isolation(tmp_path, monkeypatch):
    from kraken.kraken_client import KrakenClient
    from strategies.spot_engine import Strategy, StratParams

    # 1. KrakenClient.last_price clamps stale last-trade c[0] to live BBO mid and handles pre-trade BBO quotes
    cli = KrakenClient("key", "c2VjcmV0")
    monkeypatch.setattr(cli, "ticker", lambda _p: {"c": ["146.87", "0.3"], "b": ["147.18", "17"], "a": ["147.25", "17"]})
    assert cli.last_price("SPCXxUSD") == pytest.approx(147.215)
    monkeypatch.setattr(cli, "ticker", lambda _p: {"b": ["80.00", "50"], "a": ["80.20", "50"]})
    assert cli.last_price("ANTxUSD") == pytest.approx(80.10)

    # 2. _build_xstock_bot_env isolates Phase-1 IPO env vars to ipo_mode=True and clears them for matured active pairs
    monkeypatch.setenv("XSTOCK_BOT_IPO_TP_TRANCHES", "10:50,16:50")
    monkeypatch.setenv("XSTOCK_BOT_IPO_TRAIL_PCT", "3.5")
    monkeypatch.setenv("XSTOCK_BOT_IPO_REENTRY_DROP_PCT", "20")
    ipo_env = xw._build_xstock_bot_env(80.0, symbol_label="ANTx", ipo_mode=True)
    matured_env = xw._build_xstock_bot_env(141.75, symbol_label="SPCXxUSD", ipo_mode=False)
    assert ipo_env["STRAT_IPO_TP_TRANCHES"] == "10:50,16:50"
    assert ipo_env["STRAT_IPO_TRAIL_PCT"] == "3.5"
    assert ipo_env["STRAT_IPO_REENTRY_DROP_PCT"] == "20"
    assert matured_env["STRAT_IPO_TP_TRANCHES"] == ""
    assert matured_env["STRAT_IPO_TRAIL_PCT"] == "0"
    assert matured_env["STRAT_IPO_REENTRY_DROP_PCT"] == "0"

    # 3. Live Phase-1 IPO gap-up TP tranche disables post_only (prevents Kraken Post-Only cancel loop)
    submitted = []

    class FakeGapExecutor:
        def pair_precision(self, _pair):
            return SimpleNamespace(price_decimals=2, volume_decimals=6, order_min=0.01, base_asset="ANTx")

        def free_balance(self, _asset):
            return 20.0

        def submit_order(self, pair, side, qty, price=None, *, market=False, kind=None, client_order_id=None, post_only=True):
            submitted.append((pair, side, qty, price, market, post_only))
            return f"OGAP-{len(submitted)}"

        def open_orders(self, _pair):
            return []

    base_kw = dict(
        currency="USD",
        entry_amount=500.0,
        entry_discount_pct=0.0,
        dca_amount=450.0,
        dca_drop_pct=3.0,
        takeprofit_pct=9.0,
        check_minutes=0.25,
        max_budget=2750.0,
        max_dca_buys=5,
        enable_takeprofit=True,
        order_ttl_min=10.0,
        stop_loss_pct=0.0,
        trend_overlay=False,
        dca_trend_brake=False,
    )
    p_ipo = StratParams(
        **base_kw,
        adopt_cost=80.0,
        tp_trend_hold=True,
        ipo_tp_tranches=((10.0, 50.0), (16.0, 50.0)),
        ipo_trail_pct=3.5,
        ipo_reentry_drop_pct=20.0,
    )
    st_gap = Strategy(FakeGapExecutor(), "ANTxUSD", p_ipo, state_dir=str(tmp_path / "gap"), dry_run=False)
    st_gap.step(96.0)  # Opens +20% above $80 alloc (> +10% Tranche 0 target $88)
    assert len(submitted) == 1
    assert submitted[0][1] == "sell"
    assert submitted[0][3] == pytest.approx(88.0)
    assert submitted[0][4] is False  # limit, NOT market (must not trigger full pending_exit)
    assert submitted[0][5] is False  # post_only=False so it fills immediately against live bids

    # Below target ($84 < $88), resting maker tranche keeps default post_only=True
    submitted.clear()
    st_rest = Strategy(FakeGapExecutor(), "ANTxUSD", p_ipo, state_dir=str(tmp_path / "rest"), dry_run=False)
    st_rest.step(84.0)
    assert len(submitted) == 1
    assert submitted[0][3] == pytest.approx(88.0)
    assert submitted[0][5] is True

    # 4. resolve_ledger_alloc_price finds paged-out reservation via type="transfer" filter
    class FakePagedLedger:
        def _private(self, method, params=None):
            if (params or {}).get("type") == "transfer":
                return {
                    "ledger": {
                        "R1": {"amount": "2764.4600", "asset": "ZUSD", "balance": "2764.4601", "refid": "TR1", "time": 1794817263.0, "subtype": "krakenlaunchallocation"},
                        "R2": {"amount": "-2764.4600", "asset": "ZUSD", "balance": "0.0055", "refid": "TR1", "time": 1794817263.0, "subtype": "krakenlaunchallocation"},
                        "F1": {"amount": "1564.4600", "asset": "ZUSD", "balance": "2364.3257", "refid": "TR2", "time": 1795100000.0, "subtype": "tokenizedipoallocation"},
                        "F2": {"amount": "-1564.4600", "asset": "ZUSD", "balance": "1200.0000", "refid": "TR2", "time": 1795100000.0, "subtype": "tokenizedipoallocation"},
                        "T1": {"amount": "15.000000", "asset": "ANTx", "balance": "15.000000", "refid": "TR3", "time": 1795100100.0, "subtype": "tokenizedipoallocation"},
                    }
                }
            return {"ledger": {f"FL{i}": {"amount": "0.0", "asset": "EURC", "subtype": "flexlineinterestcharge"} for i in range(50)}}

    assert xw.resolve_ledger_alloc_price(FakePagedLedger(), "ANTx.T", 15.0) == pytest.approx(80.0)

    # 5. Live (observe_only=False) flat Strategy reconciles external AI BUY when ledger_reconcile_required is set
    class FakeFlatBuyExec(FakeGapExecutor):
        def free_balance(self, _asset):
            return 10.0

        def get_orders(self, _pair, _side, _lookback):
            return [{"price": 144.0, "timestamp": 1795100000000}]

    st_flat = Strategy(
        FakeFlatBuyExec(), "SPCXxUSD",
        StratParams(**base_kw, adopt_cost=0.0),
        state_dir=str(tmp_path / "flat"), dry_run=False,
    )
    st_flat.s["ledger_reconcile_required"] = True
    st_flat.step(144.5)
    assert st_flat.s["qty"] == pytest.approx(10.0)
    assert st_flat.s["entry_price"] == pytest.approx(144.0)
    assert st_flat.s["ledger_reconcile_required"] is False


def test_antx_ipo_post_exit_dust_reconcile_zero_known_assets_and_sell_sweep(tmp_path, monkeypatch):
    from intelligence.sentiment.collectors.portfolio_collector import KrakenXstocksCollector
    from intelligence.sentiment.evaluators.tca_risk_evaluator import resolve_venue_fee_and_limits
    from providers.quantity import clamp_notional_to_wallet_balance
    from strategies.spot_engine import Strategy, StratParams

    # 1. Sub-ordermin + sub-$1 untradable dust after external AI exit closes IPO Cycle 1 cleanly
    class FakeDustReconcileExec:
        def pair_precision(self, _pair):
            return SimpleNamespace(price_decimals=2, volume_decimals=8, order_min=0.06, base_asset="ANTx")

        def free_balance(self, asset):
            return 0.001387 if asset == "ANTx" else 1000.0

        def open_orders(self, _pair):
            return []

        def ohlc_closes_with_timestamps(self, _pair, _itv):
            return [97.0, 98.2], (1795110000.0, 1795113600.0)

    p_ant = StratParams(
        currency="USD",
        entry_amount=500.0,
        entry_discount_pct=0.8,
        dca_amount=450.0,
        dca_drop_pct=3.0,
        check_minutes=0.25,
        takeprofit_pct=9.0,
        max_budget=2750.0,
        max_dca_buys=0,
        enable_takeprofit=True,
        order_ttl_min=10.0,
        adopt_cost=87.1499,
        ipo_tp_tranches=((10.0, 50.0), (16.0, 50.0)),
        ipo_trail_pct=3.5,
        ipo_reentry_drop_pct=20.0,
    )
    st_ant = Strategy(FakeDustReconcileExec(), "ANTxUSD", p_ant, state_dir=str(tmp_path / "ant"), dry_run=False)
    st_ant.s.update(
        qty=0.2655009,
        cost=0.2655009 * 87.1499,
        entry_price=87.1499,
        last_buy_price=87.1499,
        adopted=True,
        ipo_matured=False,
        ipo_peak=99.195,
        ledger_reconcile_required=True,
    )
    st_ant.step(98.205)
    assert st_ant.s["qty"] == 0.0
    assert st_ant.s["cost"] == 0.0
    assert st_ant.s["adopted"] is True
    assert st_ant.s["ipo_matured"] is False
    assert st_ant.s["ipo_peak"] == pytest.approx(99.195)
    assert st_ant.s["ledger_reconcile_required"] is False

    # 2. kraken_xstock_watch.check_balance drops sold-out known_assets and persists cleanly across reload
    monkeypatch.setenv("XSTOCK_STATE_FILE", str(tmp_path / "xw_state.json"))
    fake_kr = FakeKrakenXStock(bal={"ZUSD": "100", "ANTx.T": "0.531"})
    xw_st = xw._load_state()
    xw.check_balance(fake_kr, xw_st, r"^ANTx", False)
    assert xw_st["known_assets"]["ANTx.T"] == pytest.approx(0.531)
    fake_kr.bal = {"ZUSD": "152", "ANTx.T": "0.00000000"}
    xw.check_balance(fake_kr, xw_st, r"^ANTx", False)
    assert "ANTx.T" not in xw_st["known_assets"]
    xw._save_state(xw_st)
    reloaded_xw = xw._load_state()
    assert "ANTx.T" not in reloaded_xw["known_assets"]

    # 3. KrakenXstocksCollector drops <$1 priced dust while keeping pre-IPO unlisted holdings; XSTOCK_ZERO_FEE caps max_buy at 0
    krk_dir = tmp_path / "krk_cfg"
    krk_dir.mkdir(parents=True, exist_ok=True)
    (krk_dir / "config.env").write_text(
        "XSTOCK_YAHOO=ANT\nXSTOCK_ACTIVE_PAIRS=SPCXxUSD:141.75\nXSTOCK_REGEX=^ANTx\n",
        encoding="utf-8",
    )
    (krk_dir / "xstock_state.json").write_text(
        json.dumps(
            {
                "known_assets": {"SPCXx.T": 10.22512, "ANTx.T": 0.001387},
                "allocated": {"asset": "ANTx.T", "qty": 0.531, "ts": 1795105777.9, "alloc_price": 87.1499},
                "pair": "ANTxUSD",
            }
        ),
        encoding="utf-8",
    )
    xs_dust = KrakenXstocksCollector().collect(
        str(tmp_path),
        kraken_dir=str(krk_dir),
        prices_and_trends={"SPCXxUSD": {"price": 137.68}, "ANTxUSD": {"price": 98.20}},
    )
    assert xs_dust["holdings"] == {"SPCXx.T": 10.22512}
    assert xs_dust["allocated"] is None
    assert "holdings_underlying_map" not in xs_dust

    xs_pre_ipo = KrakenXstocksCollector().collect(
        str(tmp_path),
        kraken_dir=str(krk_dir),
        prices_and_trends={"SPCXxUSD": {"price": 137.68}},
    )
    assert xs_pre_ipo["holdings"]["ANTx.T"] == pytest.approx(0.001387)
    assert xs_pre_ipo["allocated"]["qty"] == pytest.approx(0.531)

    ws_cyc = tmp_path / "ws_cyc"
    (ws_cyc / "kraken").mkdir(parents=True, exist_ok=True)
    (ws_cyc / "kraken" / "xstock_state.json").write_text((krk_dir / "xstock_state.json").read_text(encoding="utf-8"), encoding="utf-8")
    from intelligence.sentiment.collectors.portfolio_collector import harvest_venue_cycles
    krk_cycs, _, _ = harvest_venue_cycles(
        str(ws_cyc),
        {"ANTxUSD": {"price": 98.20}},
        {},
        read_json_fn=lambda p: json.loads(open(p, encoding="utf-8").read()),
    )
    assert "ANTxUSD" not in krk_cycs

    fake_xstock_inst = SimpleNamespace(params={"fee_tier": "XSTOCK_ZERO_FEE", "maker_fee_bps": 0.0, "taker_fee_bps": 0.0})
    fee_limits = resolve_venue_fee_and_limits(
        "kraken", "ANTxUSD", fake_xstock_inst, price=98.20, free_quote=3800.0, locked_quote_usd=500.0, free_base=0.2655
    )
    assert fee_limits["effective_max_buy_usd"] == 0.0
    assert "replace_max_buy_usd" not in fee_limits
    assert fee_limits["effective_max_sell_usd"] == pytest.approx(round(0.2655 * 98.20, 2))
    fee_limits_no_inst = resolve_venue_fee_and_limits(
        "kraken", "SPCXxUSD", None, price=137.50, free_quote=3368.0, locked_quote_usd=450.0, free_base=0.50
    )
    assert (fee_limits_no_inst["fee_tier"], fee_limits_no_inst["maker_fee_bps"], fee_limits_no_inst["taker_fee_bps"]) == (
        "XSTOCK_ZERO_FEE", 0.0, 0.0
    )
    assert fee_limits_no_inst["effective_max_buy_usd"] == 0.0
    assert "replace_max_buy_usd" not in fee_limits_no_inst

    # Sub-min_notional and sub-min_qty dust clamps symmetrically to 0.0 on BUY and SELL
    ant_dust_limits = resolve_venue_fee_and_limits(
        "kraken", "ANTxUSD", None, price=95.72, free_quote=3368.0, free_base=0.001387
    )
    assert ant_dust_limits["effective_max_buy_usd"] == 0.0
    assert ant_dust_limits["effective_max_sell_usd"] == 0.0
    hl_dust_buy = resolve_venue_fee_and_limits(
        "hyperliquid", "HYPEUSDC", None, price=67.0, free_quote=1.53, free_base=11.0
    )
    assert hl_dust_buy["effective_max_buy_usd"] == 0.0
    assert hl_dust_buy["effective_max_sell_usd"] == pytest.approx(737.0)
    fb_lim_sell = resolve_venue_fee_and_limits(
        "kraken",
        "HYPEUSD",
        None,
        price=67.0,
        free_quote=500.0,
        free_base=0.05,
        fallback_limits={"exchange_min_notional_usd": 10.0, "min_qty": 0.08},
    )
    assert fb_lim_sell["effective_max_sell_usd"] == 0.0

    from providers.kraken_provider import KrakenProvider

    class _FakeKpCli:
        def cached_pair_info(self, _cands):
            return {
                "fees_maker": [[0, 0.16]],
                "fees": [[0, 0.26]],
                "ordermin": "0.06",
                "costmin": "1.0",
                "lot_decimals": 6,
                "pair_decimals": 2,
            }

    tok_meta = KrakenProvider.extract_cached_fee_and_limits(SimpleNamespace(_cli=_FakeKpCli()), "ANTxUSD")
    assert (tok_meta["fee_tier"], tok_meta["maker_fee_bps"], tok_meta["taker_fee_bps"]) == ("XSTOCK_ZERO_FEE", 0.0, 0.0)
    assert tok_meta["min_qty"] == pytest.approx(0.06)

    # 4. clamp_notional_to_wallet_balance sweeps near-full SELL LIMIT notional up to avail_usd to prevent <$10 dust
    ant_inst = SimpleNamespace(
        symbol="ANTxUSD",
        base="ANTx",
        quote="USD",
        provider_name="kraken",
        _provider=SimpleNamespace(name="Kraken", free_balance=lambda _a: 0.2655009),
    )
    ok_sweep, swept_usd, sweep_reason = clamp_notional_to_wallet_balance(
        ant_inst, "SELL", 98.48, 26.01, min_notional_usd=10.0
    )
    assert ok_sweep is True
    assert swept_usd == pytest.approx(0.2655009 * 98.48)
    assert "Swept full ANTx balance" in sweep_reason

    # 5. Kraken EU sell-only xStock BUY guard (preflight + live place_order/submit_order) and entry_amount=0 gate skip
    from providers.strategy_executor import SubmissionRefused

    add_calls = []
    kp_live = KrakenProvider(
        client=SimpleNamespace(
            add_order=lambda *a, **kw: add_calls.append((a, kw)) or {"txid": ["T1"]},
            pair_info=lambda _p: {"ordermin": "0.06", "lot_decimals": 6, "pair_decimals": 2, "base": "ANTx"},
        ),
        live_enabled=True,
    )
    with pytest.raises(SubmissionRefused, match="sell-only"):
        kp_live.preflight_order("ANTxUSD", "BUY", 1.0)
    with pytest.raises(SubmissionRefused, match="sell-only"):
        kp_live.place_order("SPCXxUSD", "BUY", 137.0, 1.0)
    with pytest.raises(SubmissionRefused, match="sell-only"):
        kp_live.submit_order("ANTxUSD", "BUY", 1.0, price=90.0)
    assert add_calls == []

    p_sell_only = StratParams(**{**p_ant.__dict__, "entry_amount": 0.0})
    st_sell_only = Strategy(FakeDustReconcileExec(), "ANTxUSD", p_sell_only, state_dir=str(tmp_path / "ant_so"), dry_run=False)
    st_sell_only.s.update(qty=0.0, cost=0.0, adopted=True, ipo_matured=False, ipo_peak=99.195)
    st_sell_only.step(75.0)
    assert st_sell_only.s["ipo_matured"] is False
    assert st_sell_only.s["orders"] == []

    (krk_dir / "config.env").write_text(
        "XSTOCK_YAHOO=ANT\nXSTOCK_ACTIVE_PAIRS=SPCXxUSD:141.75\nXSTOCK_REGEX=^ANTx\n"
        "XSTOCK_BOT_TP_PCT=9.0\nXSTOCK_BOT_TP_TRAIL_PCT=3.0\nXSTOCK_BOT_DCA_DROP_PCT=3.0\nXSTOCK_BOT_MAX_DCA_BUYS=0\n",
        encoding="utf-8",
    )
    xs_no_dca = KrakenXstocksCollector().collect(
        str(tmp_path),
        kraken_dir=str(krk_dir),
        prices_and_trends={"SPCXxUSD": {"price": 137.68}, "ANTxUSD": {"price": 98.20}},
    )
    assert "TP +9% / Trail 3%" in xs_no_dca["active_swing_directive"]
    assert "DCA -" not in xs_no_dca["active_swing_directive"]


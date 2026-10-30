#!/usr/bin/env python3
"""Tests for trailing_stop (no real API, no money)."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from binance_api.trailing_stop import TrailingStop, should_sell  # noqa: E402
import binance_api.trailing_stop as binance_ts_mod  # noqa: E402
from providers.strategy_executor import OrderStatus  # noqa: E402


class FakeApi:
    def __init__(self, price, free=5.0, asset="TAO"):
        self.price = price
        self.free = free
        self.asset = asset
    def get_account_assets_balances(self):
        return [{"asset": self.asset, "free": str(self.free), "locked": "0"}]
    def get_current_price(self, symbol):
        return self.price
    def split_symbol(self, symbol):
        return (symbol.replace("USDC", "").replace("USDT", ""), "USDC")


class FakePo:
    # 30 Jul: TrailingStop now uses the single guarded proxy (.place(symbol, side,...)),
    # not place_safe_order. The fake exposes both (the new place plus the legacy place_safe_order)
    # so it stays robust; execute_sell/rebuy call .place().
    def __init__(self):
        self.orders = []
        self.result = {"orderId": 1}
        self.status = "closed"
        self.by_client_id = {}
    def place(self, symbol, side, price, qty, force=False, **kw):
        bypasses = kw.get("bypass_guards") or set()
        order = {"side": side, "symbol": symbol, "price": price,
                            "qty": qty, "force": force,
                            "client_order_id": kw.get("client_order_id"),
                            "bypass_guards": bypasses}
        self.orders.append(order)
        order_res = dict(self.result) if isinstance(self.result, dict) else self.result
        if isinstance(order_res, dict) and "orderId" in order_res:
            order_res["orderId"] = len(self.orders)
        if order_res and order["client_order_id"]:
            self.by_client_id[order["client_order_id"]] = order_res
        return order_res
    def order_by_client_id(self, symbol, client_order_id, *, provider_name=None):
        return self.by_client_id.get(client_order_id)
    def order_status(self, symbol, order_id, *, provider_name=None):
        order = next(o for o in self.orders
                     if str(self.by_client_id.get(o["client_order_id"], {}).get("orderId"))
                     == str(order_id))
        filled = order["qty"] if self.status == "closed" else 0.0
        return OrderStatus(self.status, filled, filled * order["price"], 0.0)
    def cancel_order(self, symbol, order_id, *, provider_name=None):
        self.status = "canceled"
    def place_safe_order(self, side, symbol, price, qty, force=False, **kw):
        return self.place(symbol, side, price, qty, force=force, **kw)


class FakeSym:
    symbols = ["TAOUSDC"]


class Base(unittest.TestCase):
    def setUp(self):
        fd, self.sf = tempfile.mkstemp(suffix=".json"); os.close(fd); os.remove(self.sf)
        self.po = FakePo()
        os.environ.pop("TRAILING_ENABLED", None)
        self._orig_trail_pct = dict(binance_ts_mod.TRAIL_PCT)
        binance_ts_mod.TRAIL_PCT["TAOUSDC"] = 22.0
        binance_ts_mod.TRAIL_PCT["BTCUSDC"] = 13.0
        binance_ts_mod.TRAIL_PCT["ARBUSDC"] = 26.0

    def tearDown(self):
        binance_ts_mod.TRAIL_PCT.clear()
        binance_ts_mod.TRAIL_PCT.update(self._orig_trail_pct)
        for p in (self.sf, self.sf + ".tmp"):
            if os.path.exists(p):
                os.remove(p)
    def ts(self, api, enabled=True, frac=1.0, min_profit_pct=0.0):
        return TrailingStop(api, self.po, FakeSym(), log=lambda *a: None,
                            enabled=enabled, sell_fraction=frac, state_file=self.sf,
                            min_profit_pct=min_profit_pct)


class TestLogica(unittest.TestCase):
    def test_should_sell(self):
        self.assertTrue(should_sell(90, 100, 10))      # exact -10%
        self.assertTrue(should_sell(89, 100, 10))
        self.assertFalse(should_sell(91, 100, 10))     # only -9%
        self.assertFalse(should_sell(100, 100, 10))
        self.assertFalse(should_sell(50, 0, 10))       # no peak


class TestTrailing(Base):
    def test_basic_state_and_price_movements(self):
        with self.subTest(msg="an_empty_balance_snapshot_does_not_change_the_state"):
            api = FakeApi(250.0)
            api.get_account_assets_balances = lambda: []
            ts = self.ts(api)
            ts.check_once()
            self.assertFalse(os.path.exists(self.sf))
            self.assertEqual(self.po.orders, [])

        with self.subTest(msg="a_non_finite_price_does_not_change_the_state"):
            self.tearDown(); self.setUp()
            api = FakeApi(float("nan"))
            self.ts(api).check_once()
            import json
            self.assertEqual(json.load(open(self.sf)), {})

        with self.subTest(msg="a_rise_does_not_sell_and_updates_the_peak"):
            self.tearDown(); self.setUp()
            api = FakeApi(250.0)
            ts = self.ts(api)
            ts.check_once()
            api.price = 260.0
            ts.check_once()
            self.assertEqual(self.po.orders, [])
            import json
            self.assertEqual(json.load(open(self.sf))["TAOUSDC"]["peak"], 260.0)

        with self.subTest(msg="falling_below_the_threshold_sells_with_force"):
            self.tearDown(); self.setUp()
            api = FakeApi(250.0)
            ts = self.ts(api)
            ts.check_once()                                # peak 250
            api.price = 190.0                              # -24% from 250 (TAO threshold 22%)
            ts.check_once()
            self.assertEqual(len(self.po.orders), 1)
            self.assertEqual(self.po.orders[0]["side"], "SELL")
            self.assertTrue(self.po.orders[0]["force"], "force=True is required so it bypasses the weight")
            self.assertEqual(
                self.po.orders[0]["bypass_guards"],
                {"all"},
                "the protective exit must bypass all guards explicitly",
            )

        with self.subTest(msg="a_small_fall_does_not_sell"):
            self.tearDown(); self.setUp()
            api = FakeApi(250.0)
            ts = self.ts(api)
            ts.check_once()
            api.price = 240.0                              # -4% < 22%
            ts.check_once()
            self.assertEqual(self.po.orders, [])

        with self.subTest(msg="below_min_notional_ignored"):
            self.tearDown(); self.setUp()
            api = FakeApi(250.0, free=0.01)                # 0.01*250 = $2.5 < $11
            ts = self.ts(api)
            ts.check_once()
            api.price = 200.0
            ts.check_once()
            self.assertEqual(self.po.orders, [])

    def test_refusals_and_execution_variations(self):
        with self.subTest(msg="typed_filter_refusal_does_not_enter_order_recovery"):
            class RefusingPo(FakePo):
                def place(self, symbol, side, price, qty, force=False, **kwargs):
                    kwargs["_outcome_context"].update(
                        state="refused", reason="below_min_notional")
                    return None

            self.po = RefusingPo()
            ts = self.ts(FakeApi(190.0))
            persisted = []
            result = ts.execute_sell(
                "TAOUSDC", "TAO", "TAOUSDC", 0.2, 190.0,
                250.0, 22.0, persisted.append)

            self.assertEqual(result.outcome, "refused")
            self.assertEqual(
                result.intent["submit_status"], "refused_before_submit")
            self.assertIsNone(persisted[-1])

        with self.subTest(msg="a_refused_sell_keeps_the_peak_and_does_not_arm_a_rebuy"):
            self.tearDown(); self.setUp()
            api = FakeApi(250.0)
            ts = self.ts(api)
            ts.check_once()
            self.po.result = None
            api.price = 190.0
            ts.check_once()
            import json
            state = json.load(open(self.sf))["TAOUSDC"]
            self.assertEqual(state["peak"], 250.0)
            self.assertNotIn("rebuy", state)

        with self.subTest(msg="a_refused_rebuy_stays_for_a_retry"):
            self.tearDown(); self.setUp()
            api = FakeApi(250.0)
            ts = self.ts(api)
            ts.check_once()
            api.price = 190.0
            ts.check_once()
            self.po.result = None
            api.free = 0.0
            api.price = 193.0
            ts.check_once()
            import json
            self.assertIn("rebuy", json.load(open(self.sf))["TAOUSDC"])

        with self.subTest(msg="a_dry_run_does_not_sell"):
            self.tearDown(); self.setUp()
            api = FakeApi(250.0)
            ts = self.ts(api, enabled=False)
            ts.check_once()
            api.price = 190.0
            ts.check_once()
            self.assertEqual(self.po.orders, [], "dry run: it only logs, it does not place orders")

        with self.subTest(msg="the_peak_survives_a_restart"):
            self.tearDown(); self.setUp()
            api = FakeApi(260.0)
            self.ts(api).check_once()                      # peak 260, instance 1
            api.price = 200.0                              # -23% from 260 (threshold 22%)
            self.ts(api).check_once()                      # instance 2 (a restart) — it reads the peak
            self.assertEqual(len(self.po.orders), 1, "peak 260 survives restart")

        with self.subTest(msg="a_partial_sale"):
            self.tearDown(); self.setUp()
            api = FakeApi(250.0, free=4.0)
            ts = self.ts(api, frac=0.5)
            ts.check_once()
            api.price = 190.0
            ts.check_once()
            self.assertAlmostEqual(self.po.orders[0]["qty"], 2.0)   # 50% of 4

        with self.subTest(msg="it_re_arms_after_a_sale"):
            self.tearDown(); self.setUp()
            api = FakeApi(250.0)
            ts = self.ts(api)
            ts.check_once()
            api.price = 190.0; ts.check_once()             # it sells, the peak resets to 190
            ts.check_once()                                # terminal status confirmed
            import json
            self.assertEqual(json.load(open(self.sf))["TAOUSDC"]["peak"], 190.0)


class TestPerCoinRebuy(Base):
    """Re-buy is now per-coin (registry trailing.rebuy), not one global switch."""

    def _sell_then_confirm(self, ts, api):
        ts.check_once()                                # arm + track (min_profit=0)
        api.price = 190.0; ts.check_once()             # trailing sells
        ts.check_once()                                # terminal fill confirmed
        import json
        return json.load(open(self.sf))["TAOUSDC"]

    def test_rebuy_configurations(self):
        with self.subTest(msg="rebuy_disabled_per_coin_does_not_arm_a_rebuy"):
            api = FakeApi(250.0)
            ts = self.ts(api)
            ts.rebuy_enabled_for = lambda symbol: False    # per-coin OFF (overrides registry)
            st = self._sell_then_confirm(ts, api)
            self.assertNotIn("rebuy", st, "rebuy must NOT arm when disabled for the coin")
            api.price = 210.0; ts.check_once()             # a recovery must not re-buy
            self.assertTrue(all(o["side"] != "BUY" for o in self.po.orders))

        with self.subTest(msg="rebuy_enabled_per_coin_arms_a_rebuy"):
            self.tearDown(); self.setUp()
            api = FakeApi(250.0)
            ts = self.ts(api)
            ts.rebuy_enabled_for = lambda symbol: True     # per-coin ON
            st = self._sell_then_confirm(ts, api)
            self.assertIn("rebuy", st, "rebuy must arm when enabled for the coin")

        with self.subTest(msg="core_rebuy_for_falls_back_to_global_without_adapter_method"):
            self.tearDown(); self.setUp()
            from trailing_core import TrailingCore
            class Bare:  # no rebuy_enabled_for attribute
                pass
            core = TrailingCore(Bare(), log=lambda *_: None, enabled=True, state_file=self.sf,
                                min_notional=11.0, rebuy_enabled=True, rebuy_bounce_pct=1.2,
                                rebuy_skip_if_trend_down=True, sell_skip_if_trend_up=False,
                                sell_fraction=1.0, item_isolation=True, min_profit_pct=0.0)
            self.assertTrue(core._rebuy_for("XUSD"))
            core.rebuy_enabled = False
            self.assertFalse(core._rebuy_for("XUSD"))

        with self.subTest(msg="declared_policy_failure_never_falls_back_to_global_true"):
            self.tearDown(); self.setUp()
            from unittest.mock import Mock
            ts = self.ts(FakeApi(250.0))
            ts.rebuy_enabled_for = Mock(side_effect=RuntimeError("unavailable"))
            self.assertFalse(ts.core._rebuy_for("TAOUSDC"))
            ts.rebuy_enabled_for = Mock(return_value=None)
            self.assertFalse(ts.core._rebuy_for("TAOUSDC"))

    def test_auto_mode_and_trend(self):
        with self.subTest(msg="auto_mode_delegates_to_the_long_term_trend"):
            import binance_api.trailing_stop as m
            api = FakeApi(250.0); ts = self.ts(api)
            saved = dict(m.REBUY_MODE_BY_SYMBOL)
            m.REBUY_MODE_BY_SYMBOL["TAOUSDC"] = "auto"
            try:
                ts._long_trend_up = lambda s: True
                self.assertTrue(ts.rebuy_enabled_for("TAOUSDC"))
                ts._long_trend_up = lambda s: False
                self.assertFalse(ts.rebuy_enabled_for("TAOUSDC"))
            finally:
                m.REBUY_MODE_BY_SYMBOL.clear(); m.REBUY_MODE_BY_SYMBOL.update(saved)

        with self.subTest(msg="long_trend_up_uses_the_configured_sma_and_fails_closed"):
            self.tearDown(); self.setUp()
            import binance_api.trailing_stop as m
            from types import SimpleNamespace
            api = FakeApi(250.0); ts = self.ts(api)
            day = int(m.time.time() // 86400)
            kl = lambda n: [[d * 86400000, 0, 0, 0, "100.0", 0,
                             (d + 1) * 86400000 - 1] for d in range(day - n + 1, day + 1)]
            ts.api.client = SimpleNamespace(get_klines=lambda **kw: kl(m.REBUY_TREND_DAYS + 1))
            self.assertTrue(ts._long_trend_up("TAOUSDC"))          # price 250 > SMA 100 -> up
            api.price = 90.0  # The cached average must not freeze yesterday's verdict.
            self.assertFalse(ts._long_trend_up("TAOUSDC"))         # price 90 < SMA 100 -> down
            ts._long_trend_cache.clear()
            ts.api.client = SimpleNamespace(get_klines=lambda **kw: kl(4))   # <N -> fail closed
            self.assertFalse(ts._long_trend_up("TAOUSDC"))

        with self.subTest(msg="auto_keeps_recovery_intent_while_down_then_buys_after_recovery"):
            self.tearDown(); self.setUp()
            import binance_api.trailing_stop as m
            from unittest.mock import patch
            api = FakeApi(250.0); ts = self.ts(api)
            ts.core.sell_skip_if_trend_up = False
            with patch.dict(m.REBUY_MODE_BY_SYMBOL, TAOUSDC="auto"):
                ts._long_trend_up = lambda symbol: False
                ts.trend = lambda pair: 1.0
                state = self._sell_then_confirm(ts, api)
                self.assertIn("rebuy", state)
                api.free = 0.0
                api.price = 150.0; ts.check_once()
                self.assertEqual(ts._load()["TAOUSDC"]["rebuy"]["low"], 150.0)
                self.assertFalse(any(o["side"] == "BUY" for o in self.po.orders))
                ts._long_trend_up = lambda symbol: True
                api.price = 160.0; ts.check_once()
                self.assertEqual(len([o for o in self.po.orders if o["side"] == "BUY"]), 1)

        with self.subTest(msg="auto_rejects_stale_gapped_and_nonfinite_candles"):
            self.tearDown(); self.setUp()
            import binance_api.trailing_stop as m
            from types import SimpleNamespace
            from unittest.mock import patch
            day = 20000
            good = [[d * 86400000, 0, 0, 0, "100", 0, (d + 1) * 86400000 - 1]
                    for d in range(day - 3, day + 1)]
            bad_sets = [good[:-2], good[1:] + [good[-1]],
                        [good[0], good[0], good[2], good[3]]]
            bad_close = [row[:] for row in good]; bad_close[1][4] = "nan"
            bad_sets.append(bad_close)
            with patch.object(m, "REBUY_TREND_DAYS", 3), patch.object(m.time, "time", return_value=day * 86400 + 60):
                for candles in bad_sets:
                    ts = self.ts(FakeApi(250.0))
                    ts.api.client = SimpleNamespace(get_klines=lambda **kw: candles)
                    self.assertFalse(ts._long_trend_up("TAOUSDC"))
                ts.api.client = SimpleNamespace(get_klines=lambda **kw: good)
                self.assertTrue(ts._long_trend_up("TAOUSDC"))
                ts.api.price = float("inf")
                self.assertFalse(ts._long_trend_up("TAOUSDC"))
                ts.api.price = 250.0
                ts.api.client = SimpleNamespace(get_klines=lambda **kw: good[:-1])
                with patch.object(m.time, "time", return_value=(day + 1) * 86400 + 60):
                    self.assertFalse(ts._long_trend_up("TAOUSDC"))

    def test_rebuy_execution(self):
        with self.subTest(msg="below_minimum_recovery_is_not_treated_as_a_completed_buy"):
            ts = self.ts(FakeApi(100.0))
            state = {"TAOUSDC": {"peak": 100.0, "rebuy": {"qty": 0.01, "low": 90.0, "sell_price": 100.0}}}
            ts.core._handle_rebuy("TAOUSDC", "TAO", "TAOUSDC", state["TAOUSDC"], 100.0, state)
            self.assertIn("rebuy", state["TAOUSDC"])
            self.assertEqual(self.po.orders, [])

        with self.subTest(msg="trend_value_uses_get_fresh_snapshot_not_stale_get_snapshot"):
            from unittest.mock import MagicMock, patch
            ts = self.ts(FakeApi(100.0))
            fake_mgr = MagicMock()
            fake_mgr.get_fresh_snapshot.return_value = None
            fake_mgr.get_snapshot.return_value = {"gradient_recent": -0.5}
            with patch("cacheManager.get_short_trend_manager", return_value=fake_mgr):
                self.assertEqual(ts._trend_value("TAOUSDC"), 0.0)
                fake_mgr.get_fresh_snapshot.assert_called_once_with("TAOUSDC")
                fake_mgr.get_snapshot.assert_not_called()
                fake_mgr.get_fresh_snapshot.return_value = {"gradient_recent": -0.25}
                self.assertEqual(ts._trend_value("TAOUSDC"), -0.25)

        with self.subTest(msg="rebuy_below_sell_price_bypasses_all_guards_and_above_consults_guards"):
            self.tearDown(); self.setUp()
            ts = self.ts(FakeApi(100.0))
            rb_state = {"qty": 1.0, "low": 90.0, "sell_price": 100.0}
            ts.execute_rebuy("TAOUSDC", "TAO", "TAOUSDC", 1.0, 95.0, rb_state, persist=lambda _: None)
            ts.execute_rebuy("TAOUSDC", "TAO", "TAOUSDC", 1.0, 100.0, rb_state, persist=lambda _: None)
            ts.execute_rebuy("TAOUSDC", "TAO", "TAOUSDC", 1.0, 105.0, rb_state, persist=lambda _: None)
            self.assertEqual(len(self.po.orders), 3)
            self.assertEqual(self.po.orders[0]["bypass_guards"], {"all"})
            self.assertEqual(self.po.orders[1]["bypass_guards"], {"math_reference"})
            self.assertEqual(self.po.orders[2]["bypass_guards"], {"math_reference"})


class TestPerCoin(Base):
    def test_per_coin_configurations(self):
        with self.subTest(msg="market_data_only_symbol_is_not_trailed_or_logged_as_managed"):
            from types import SimpleNamespace
            ts = TrailingStop(FakeApi(1), self.po,
                              SimpleNamespace(symbols=["TAOUSDC", "DATAONLYUSDC"]),
                              state_file=self.sf, log=lambda *_args: None)
            self.assertEqual([row[0] for row in ts.assets()], ["TAOUSDC"])
            def stop():
                raise KeyboardInterrupt
            ts.check_once = stop
            ts.run()

        with self.subTest(msg="new_arb_state_is_not_implicitly_preseeded_or_armed"):
            self.tearDown(); self.setUp()
            from types import SimpleNamespace
            ts = TrailingStop(FakeApi(0.17, free=100, asset="ARB"), self.po,
                              SimpleNamespace(symbols=["ARBUSDC"]), enabled=True,
                              state_file=self.sf, min_profit_pct=5, log=lambda *_args: None)
            ts.check_once()
            state = ts._load()["ARBUSDC"]
            self.assertAlmostEqual(state["warmup_at"], 0.17 * 1.05)
            self.assertEqual(self.po.orders, [])
            saved = {"ARBUSDC": {"peak": 0.174}}
            ts._save(saved)
            restarted = TrailingStop(ts.api, self.po, ts.sym, enabled=True,
                                     state_file=self.sf, min_profit_pct=5, log=lambda *_args: None)
            self.assertEqual(restarted._load(), saved)

        with self.subTest(msg="per_coin_threshold"):
            import instruments_config as ic
            self.tearDown(); self.setUp()
            ts = self.ts(FakeApi(1.0))
            expected_trails = ic.trail_pct_map()
            for sym in ("BTCUSDC", "TAOUSDC", "ARBUSDC"):
                self.assertEqual(self._orig_trail_pct[sym], expected_trails[sym])
            with self.assertRaises(KeyError):
                ts.trail_pct_for("XYZUSDC")  # No invented policy for an unconfigured asset.

        with self.subTest(msg="invalid_sell_fraction_fails_at_start"):
            self.tearDown(); self.setUp()
            with self.assertRaises(ValueError):
                self.ts(FakeApi(1.0), frac=1.01)
            with self.assertRaises(ValueError):
                self.ts(FakeApi(1.0), frac=float("nan"))


class TestMinProfit(Base):
    """The minimum profit threshold before the trailing activates."""

    def test_min_profit_behaviors(self):
        with self.subTest(msg="non_finite_provider_basis_does_not_make_warmup_unreachable"):
            for basis in (float("inf"), float("nan"), -1, True):
                with self.subTest(basis=basis):
                    self.tearDown(); self.setUp()
                    ts = self.ts(FakeApi(250), min_profit_pct=5)
                    ts.cost_basis = lambda _pair: basis
                    ts._save({})
                    ts.check_once()
                    self.assertAlmostEqual(ts._load()["TAOUSDC"]["warmup_at"], 262.5)

        with self.subTest(msg="cost_basis_uses_total_balance_and_preserves_existing_warmup"):
            self.tearDown(); self.setUp()
            from unittest.mock import Mock
            ts = self.ts(FakeApi(250), min_profit_pct=5)
            ts._balances = [{"asset": "TAO", "free": "3", "locked": "2"}]
            self.po.position_cost_basis = Mock(return_value=200)
            self.assertEqual(ts.cost_basis("TAOUSDC"), 200)
            self.assertEqual(self.po.position_cost_basis.call_args.args[1], 5)
            ts._save({"TAOUSDC": {"peak": 250, "warmup_at": 262.5}})
            self.po.position_cost_basis.reset_mock()
            ts.check_once()
            self.po.position_cost_basis.assert_not_called()
            self.assertEqual(ts._load()["TAOUSDC"]["warmup_at"], 262.5)

        with self.subTest(msg="warming_up_does_not_sell_on_normal_dips_but_sells_on_crash"):
            self.tearDown(); self.setUp()
            api = FakeApi(250.0)
            ts = self.ts(api, min_profit_pct=5.0)
            ts.check_once()                    # initial=250, active at 262.5, stop at 195.0 (-22%)
            api.price = 230.0                  # normal dip of -8% (<22% stop): warmup suppresses sell
            ts.check_once()
            self.assertEqual(self.po.orders, [], "it does not sell on normal dips before profit threshold is reached")
            api.price = 190.0                  # crash of -24% (exceeds 22% stop): extreme protection triggers regardless of warmup
            ts.check_once()
            self.assertEqual(len(self.po.orders), 1, "extreme crash protection triggers even before warmup")
            self.assertEqual(self.po.orders[0]["side"], "SELL")

        with self.subTest(msg="active_past_the_threshold_sells"):
            self.tearDown(); self.setUp()
            api = FakeApi(250.0)
            ts = self.ts(api, min_profit_pct=5.0)
            ts.check_once()                    # initial=250
            api.price = 263.0                  # +5.2% > 5% threshold -> trailing active
            ts.check_once()                    # peak=263
            api.price = 200.0                  # -23.9% from peak 263 (TAO threshold 22%)
            ts.check_once()
            self.assertEqual(len(self.po.orders), 1, "it sells once the profit threshold is passed")
            self.assertEqual(self.po.orders[0]["side"], "SELL")

        with self.subTest(msg="it_initially_resets_to_the_rebuy"):
            self.tearDown(); self.setUp()
            api = FakeApi(250.0)
            ts = self.ts(api, min_profit_pct=5.0)
            ts.check_once()                    # initial=250, peak=250
            api.price = 263.0; ts.check_once() # passes threshold -> active
            api.price = 200.0; ts.check_once() # a crash of -23.9% -> it sells; it arms the rebuy
            self.assertEqual(len(self.po.orders), 1)
            ts.check_once()                    # it confirms the SELL fill and arms the re-buy
            # simulate the rebuy: the price rises 1.2% from 200 -> 202.4
            api.price = 199.0; ts.check_once() # low=199
            api.price = 201.5; ts.check_once() # +1.26% from 199 -> re-buy; initial=201.5
            ts.check_once()                    # it confirms the REBUY fill and sets the warmup
            # now the trailing is inactive until 201.5*1.05=211.6
            api.price = 180.0; ts.check_once() # a crash from 201.5 but below the activation threshold
            # the orders: 1 sell + 1 re-buy; the third does NOT execute (warming up)
            sells = [o for o in self.po.orders if o["side"] == "SELL"]
            self.assertEqual(len(sells), 1, "the second crash does not trigger a sell (warming up after the rebuy)")

        with self.subTest(msg="warmup_arms_from_real_cost_basis"):
            self.tearDown(); self.setUp()
            class PoWithBasis(FakePo):
                def position_cost_basis(self, symbol, held_qty, since_s, **kwargs):
                    return 200.0 if held_qty == 5 else None
            self.po = PoWithBasis()
            api = FakeApi(250.0)
            ts = self.ts(api, min_profit_pct=5.0)
            ts.check_once()                    # is_new: cost_basis=200 -> warmup 210 < 250 -> ARMED, peak=250
            api.price = 190.0                  # -24% from peak 250, below the 22% TAO stop (195)
            ts.check_once()
            self.assertEqual(len(self.po.orders), 1, "armed via real cost basis -> the crash sells")
            self.assertEqual(self.po.orders[0]["side"], "SELL")

        with self.subTest(msg="resting_orders_unlocked_when_free_balance_low"):
            self.tearDown(); self.setUp()
            api = FakeApi(250.0)
            ts = self.ts(api, min_profit_pct=0.0)
            # Simulate initial tick with free balance
            ts.check_once()
            # Now simulate coins getting locked in a TP order (free=0)
            api.free = 0.0
            api.price = 190.0  # Crash of 24% (exceeds 22% TAO stop)

            unlocked = []
            def fake_unlock(pair):
                unlocked.append(pair)
                # Unlocking restores free balance
                api.free = 5.0
                ts._balances = api.get_account_assets_balances()
                return True

            ts.unlock_qty = fake_unlock
            ts.check_once()
            self.assertEqual(unlocked, ["TAOUSDC"])
            self.assertEqual(len(self.po.orders), 1, "unlocked coins allowed crash sell to execute")
            self.assertEqual(self.po.orders[0]["side"], "SELL")

        with self.subTest(msg="resting_orders_not_unlocked_when_price_healthy"):
            self.tearDown(); self.setUp()
            api = FakeApi(250.0)
            ts = self.ts(api, min_profit_pct=0.0)
            ts.check_once()
            # Coins locked in resting order, but price is healthy at 250 (not in crash)
            api.free = 0.0
            api.price = 250.0

            unlocked = []
            def fake_unlock(pair):
                unlocked.append(pair)
                return True

            ts.unlock_qty = fake_unlock
            ts.check_once()
        with self.subTest(msg="resting_orders_not_unlocked_when_dry_run"):
            self.tearDown(); self.setUp()
            api = FakeApi(250.0)
            ts = self.ts(api, enabled=False, min_profit_pct=0.0)
            ts.check_once()
            api.free = 0.0
            api.price = 190.0  # Crash of 24% (exceeds 22% TAO stop)

            unlocked = []
            def fake_unlock(pair):
                unlocked.append(pair)
                return True

            ts.unlock_qty = fake_unlock
            ts.check_once()
            self.assertEqual(unlocked, [], "resting orders must NOT be cancelled in dry-run mode")
            self.assertEqual(self.po.orders, [], "no orders placed in dry run")

        with self.subTest(msg="resting_orders_unlocked_when_partial_free_available"):
            self.tearDown(); self.setUp()
            api = FakeApi(250.0)
            ts = self.ts(api, min_profit_pct=0.0)
            ts.check_once()
            # Partial free: 0.2 free, 4.8 locked
            api.free = 0.2
            api.price = 190.0  # Crash

            unlocked = []
            def fake_unlock(pair):
                unlocked.append(pair)
                api.free = 5.0
                ts._balances = api.get_account_assets_balances()
                return True

            ts.unlock_qty = fake_unlock
            ts.check_once()
            self.assertEqual(unlocked, ["TAOUSDC"], "resting orders must be unlocked even if partial free balance exists")
            self.assertEqual(len(self.po.orders), 1)
            self.assertEqual(self.po.orders[0]["qty"], 5.0)

        with self.subTest(msg="partial_fill_retains_peak_until_flat"):
            self.tearDown(); self.setUp()
            api = FakeApi(250.0)
            ts = self.ts(api, min_profit_pct=0.0)
            ts.check_once()  # establishes peak 250.0
            api.price = 190.0

            # First tick: only 0.2 is available to sell
            api.free = 0.2
            # Total holdings = 5.0 (0.2 free + 4.8 locked in orders)
            ts.total_qty = lambda asset: 5.0
            ts.check_once()
            self.assertEqual(len(self.po.orders), 1)
            self.assertEqual(self.po.orders[0]["qty"], 0.2)

            # Confirm terminal fill of the 0.2 order
            ts.check_once()
            import json
            st = json.load(open(self.sf))["TAOUSDC"]
            self.assertEqual(st["peak"], 250.0, "peak must NOT be reset to 190 while position remains incomplete")
            self.assertEqual(st["exit_sold_qty"], 0.2)
            self.assertNotIn("rebuy", st, "rebuy must not be armed until exit is completed")

            # Second tick: remaining 4.8 is unlocked and sold at 190.0
            api.free = 4.8
            ts.total_qty = lambda asset: 4.8
            ts.check_once()
            self.assertEqual(len(self.po.orders), 2)
            self.assertEqual(self.po.orders[1]["qty"], 4.8)

            # Confirm terminal fill of the 4.8 order
            ts.total_qty = lambda asset: 0.0
            api.free = 0.0
            ts.check_once()
            st = json.load(open(self.sf))["TAOUSDC"]
            self.assertEqual(st["peak"], 190.0, "peak is now reset after completing the targeted position")
            self.assertIn("rebuy", st, "rebuy is armed once position is flat")
        with self.subTest(msg="new_position_does_not_inherit_stale_peak_from_unheld_period"):
            self.tearDown(); self.setUp()
            api = FakeApi(100.0, free=0.0)
            ts = self.ts(api, min_profit_pct=5.0)
            ts.check_once()  # Tick 1: flat at 100

            api.price = 200.0  # Climbs to 200 while holding 0
            ts.check_once()

            # Now a new position is bought at 150
            api.price = 150.0
            api.free = 1.0
            ts.check_once()

            self.assertEqual(self.po.orders, [], "new position must not be sold using stale peak from unheld period")
            import json
            st = json.load(open(self.sf))["TAOUSDC"]
            self.assertEqual(st["peak"], 150.0, "peak must re-anchor to new position entry, not unheld 200 high")

        with self.subTest(msg="partial_exit_state_retained_on_bounce_above_stop"):
            self.tearDown(); self.setUp()
            api = FakeApi(250.0, free=5.0)
            ts = self.ts(api, min_profit_pct=0.0)
            ts.check_once()  # Establishes peak 250.0
            api.price = 190.0

            # Step 1: partial exit of 0.2 out of 5.0 (4.8 locked)
            api.free = 0.2
            ts.total_qty = lambda asset: 5.0
            ts.check_once()
            self.assertEqual(len(self.po.orders), 1)
            self.assertEqual(self.po.orders[0]["qty"], 0.2)

            # Confirm 0.2 fill
            ts.check_once()
            import json
            st = json.load(open(self.sf))["TAOUSDC"]
            self.assertEqual(st["exit_sold_qty"], 0.2)

            # Step 2: price bounces to 240 (above stop 195), free is 0, 4.8 still held
            api.price = 240.0
            api.free = 0.0
            ts.total_qty = lambda asset: 4.8
            ts.check_once()

            # Verify partial exit state is NOT wiped by the bounce
            st = json.load(open(self.sf))["TAOUSDC"]
            self.assertEqual(st.get("target_exit_qty"), 5.0, "target_exit_qty must survive rebound while position held")
            self.assertEqual(st.get("exit_sold_qty"), 0.2, "exit_sold_qty must survive rebound while position held")

            # Step 3: price falls back to 190, 4.8 is unlocked and sold
            api.price = 190.0
            api.free = 4.8
            ts.total_qty = lambda asset: 4.8
            ts.check_once()
            self.assertEqual(len(self.po.orders), 2)
            self.assertEqual(self.po.orders[1]["qty"], 4.8, "remaining 4.8 target must be sold")

            # Confirm fill of 4.8
            ts.total_qty = lambda asset: 0.0
            api.free = 0.0
            ts.check_once()
            st = json.load(open(self.sf))["TAOUSDC"]
            self.assertIn("rebuy", st, "rebuy must be armed once position is completed")
            self.assertEqual(st["rebuy"]["qty"], 5.0, "rebuy must track full 5.0 units")

        with self.subTest(msg="rebuy_resets_peak_and_does_not_immediately_resell"):
            self.tearDown(); self.setUp()
            api = FakeApi(200.0, free=2.0)
            ts = self.ts(api, min_profit_pct=5.0)
            ts.rebuy_enabled_for = lambda symbol: True
            ts.trend = lambda _pair: 0.0
            ts.check_once()  # Peak = 200.0
            api.price = 150.0  # Crash below 156.0 (-22% of 200) -> triggers exit SELL
            ts.check_once()
            self.assertEqual(len(self.po.orders), 1)
            self.assertEqual(self.po.orders[0]["side"], "SELL")
            ts.check_once()  # Reconcile SELL terminal -> arms rebuy at low=150.0

            api.price = 140.0  # Post-sale low drops to 140.0
            ts.check_once()
            api.price = 142.0  # Bounces +1.43% (> 1.2%) -> submits REBUY
            ts.check_once()
            self.assertEqual(len(self.po.orders), 2)
            self.assertEqual(self.po.orders[1]["side"], "BUY")

            ts.check_once()  # Reconcile REBUY terminal -> resets peak to 142.0
            import json
            st = json.load(open(self.sf))["TAOUSDC"]
            self.assertEqual(st["peak"], 142.0)

            # Next tick at 142.0 must NOT trigger a false crash sell from the old 200.0 peak!
            ts.check_once()
            self.assertEqual(len(self.po.orders), 2, "must not immediately crash-sell newly rebought coins")

        with self.subTest(msg="warmup_completes_even_when_coins_locked_in_tp_order"):
            self.tearDown(); self.setUp()
            api = FakeApi(200.0, free=2.0)
            ts = self.ts(api, min_profit_pct=5.0)
            ts.total_qty = lambda asset: 2.0
            # Tick 1 at 200.0: sets warmup_at = 210.0 (+5%)
            ts.check_once()
            import json
            st = json.load(open(self.sf))["TAOUSDC"]
            self.assertAlmostEqual(st["warmup_at"], 210.0)

            # Tick 2 at 212.0 (>= 210.0), but free=0.0 because coins are locked in a resting TP order
            api.price = 212.0
            api.free = 0.0
            ts.check_once()
            st = json.load(open(self.sf))["TAOUSDC"]
            self.assertNotIn("warmup_at", st, "warmup_at must be cleared once price crosses threshold even if coins are locked in TP")
            self.assertEqual(st["peak"], 212.0)

        with self.subTest(msg="warmup_at_prevents_sub_threshold_peak_ratcheting"):
            self.tearDown(); self.setUp()
            api = FakeApi(250.0, free=2.0)
            ts = self.ts(api, min_profit_pct=5.0)
            ts.check_once()  # peak=250.0, warmup_at=262.5
            api.price = 260.0  # +4.0% (< +5.0% warmup_at): trailing not active yet
            ts.check_once()
            import json
            st = json.load(open(self.sf))["TAOUSDC"]
            self.assertEqual(st["peak"], 250.0, "peak must not ratchet up before warmup_at is reached")
            api.price = 200.0  # -20% from 250 (< 22% TAO trail), though -23.1% from 260
            ts.check_once()
            self.assertEqual(self.po.orders, [], "must not sell based on unactivated sub-warmup high")

        with self.subTest(msg="fee_or_step_adjusted_rebuy_completes_cleanly"):
            self.tearDown(); self.setUp()
            api = FakeApi(200.0, free=5.0)
            ts = self.ts(api, min_profit_pct=5.0)
            ts.rebuy_enabled_for = lambda symbol: True
            ts.trend = lambda _pair: 0.0
            ts.check_once()  # peak=200
            api.price = 150.0
            ts.check_once()  # SELL 5.0
            api.free = 0.0
            ts.check_once()  # SELL terminal -> arms rebuy qty=5.0, is_flat=True

            # Simulate fee-capped / lot-step rounded fill on REBUY (4.995 filled out of 5.0)
            orig_order_status = self.po.order_status
            self.po.order_status = lambda symbol, order_id, **kw: OrderStatus("closed", 4.995, 4.995 * 152.0, 0.0)
            api.price = 152.0
            ts.check_once()  # submits REBUY
            api.free = 4.995
            ts.check_once()  # reconciles REBUY terminal
            self.po.order_status = orig_order_status

            import json
            st = json.load(open(self.sf))["TAOUSDC"]
            self.assertNotIn("rebuy", st, "closed fee-adjusted rebuy must clear rebuy state")
            self.assertFalse(st.get("is_flat", False), "is_flat must be reset to False after rebuy fill")
            self.assertAlmostEqual(st["peak"], 152.0)

        with self.subTest(msg="dry_run_does_not_mutate_rebuy_or_warmup_state"):
            self.tearDown(); self.setUp()
            api = FakeApi(200.0, free=2.0)
            ts = self.ts(api, enabled=False, min_profit_pct=5.0)
            ts.rebuy_enabled_for = lambda symbol: True
            ts.trend = lambda _pair: 0.0
            seeded = {
                "TAOUSDC": {
                    "peak": 200.0,
                    "warmup_at": 210.0,
                    "rebuy": {"qty": 2.0, "sell_price": 155.0, "low": 150.0},
                }
            }
            ts._save(seeded)
            api.price = 153.0  # triggers dry-run rebuy (+2% > 1.2%) and dry-run sell (-23.5% > 22%)
            ts.check_once()
            st = ts._load()["TAOUSDC"]
            self.assertIn("rebuy", st, "dry-run must not pop armed rebuy state")
            self.assertIn("warmup_at", st, "dry-run must not pop warmup_at state")
            self.assertNotIn("target_exit_qty", st, "dry-run must not persist target_exit_qty")

        with self.subTest(msg="unlock_qty_preserves_balances_on_failure_and_skips_when_unlocked"):
            self.tearDown(); self.setUp()
            api = FakeApi(200.0, free=2.0)
            cancelled = []
            api.cancel_open_orders = lambda side, pair: cancelled.append((side, pair))
            ts = self.ts(api)
            ts.begin_tick()
            # Locked is 0 in FakeApi -> unlock_qty must skip cancel_open_orders
            self.assertFalse(ts.unlock_qty("TAOUSDC"))
            self.assertEqual(cancelled, [])

            # Now simulate locked > 0, but subsequent balance refresh returns []
            ts._balances = [{"asset": "TAO", "free": "0.0", "locked": "2.0"}]
            api.get_account_assets_balances = lambda: []
            self.assertFalse(ts.unlock_qty("TAOUSDC"))
            self.assertEqual(len(cancelled), 1)
            self.assertEqual(len(ts._balances), 1, "transient empty balance refresh must not wipe _balances")

        with self.subTest(msg="retry_after_zero_fill_terminal_generates_distinct_client_order_id"):
            self.tearDown(); self.setUp()
            api = FakeApi(250.0, free=2.0)
            ts = self.ts(api)
            ts.check_once()  # peak = 250.0
            api.price = 190.0
            ts.check_once()  # SELL attempt 1
            cid1 = self.po.orders[-1]["client_order_id"]

            # Simulate canceled order with 0 fill
            self.po.status = "canceled"
            ts.check_once()  # reconciles terminal with 0 fill

            # Next tick at same price/qty/anchor submits attempt 2 with distinct client_order_id
            self.po.status = "closed"
            ts.check_once()
            cid2 = self.po.orders[-1]["client_order_id"]
            self.assertNotEqual(cid1, cid2)

        with self.subTest(msg="sell_skip_if_trend_up_defers_sell_when_enabled"):
            self.tearDown(); self.setUp()
            from unittest.mock import patch
            api = FakeApi(250.0, free=2.0)
            logs = []
            ts = TrailingStop(api, self.po, FakeSym(), log=logs.append,
                              enabled=True, state_file=self.sf)
            ts.core.sell_skip_if_trend_up = True
            ts.core.sell_trend_defer_max_sec = 420.0
            ts.trend = lambda _pair: 1.0
            with patch("trailing_core.time.time", return_value=1000.0):
                ts.check_once()  # peak = 250.0
                api.price = 190.0
                ts.check_once()  # t=1000s (0s elapsed < 420s): deferred by UP trend
            self.assertEqual(self.po.orders, [])
            self.assertTrue(any("instant trend is UP" in m for m in logs))

            # At t=1200s (200s elapsed < 420s): still deferred
            with patch("trailing_core.time.time", return_value=1200.0):
                ts.check_once()
            self.assertEqual(self.po.orders, [])

            # At t=1421s (421s elapsed >= 420s = 7m): timer expires -> forces emergency SELL!
            with patch("trailing_core.time.time", return_value=1421.0):
                ts.check_once()
            self.assertEqual(len(self.po.orders), 1)
            self.assertEqual(self.po.orders[0]["side"], "SELL")
            self.assertTrue(any("sell trend deferral expired" in m for m in logs))

        with self.subTest(msg="fractional_exit_sub_notional_remainder_completes_without_infinite_unlock_loop"):
            self.tearDown(); self.setUp()
            api = FakeApi(250.0, free=5.01)
            ts = self.ts(api, min_profit_pct=0.0)
            ts.core.sell_fraction = 0.5
            ts.rebuy_enabled_for = lambda symbol: True
            ts.trend = lambda _pair: 0.0
            ts.check_once()  # peak = 250.0

            # Crash to 190.0: target_exit_qty = 5.01 * 0.5 = 2.505, venue rounds fill to 2.500
            # Remaining 2.51 in account ($476.90 >> min_notional $10, so is_flat=False),
            # but remaining_target = 0.005 ($0.95 < min_notional $10).
            orig_order_status = self.po.order_status
            self.po.order_status = lambda symbol, order_id, **kw: OrderStatus("closed", 2.500, 2.500 * 190.0, 0.0)
            api.price = 190.0
            ts.check_once()  # submits SELL
            api.free = 2.51
            ts.check_once()  # reconciles SELL terminal -> must complete!
            self.po.order_status = orig_order_status

            import json
            st = json.load(open(self.sf))["TAOUSDC"]
            self.assertNotIn("target_exit_qty", st)
            self.assertNotIn("exit_sold_qty", st)
            self.assertEqual(st["peak"], 190.0)
            self.assertIn("rebuy", st)

            # Next tick at 190.0 must NOT call unlock_qty in a loop
            unlock_calls = []
            ts.unlock_qty = lambda pair: unlock_calls.append(pair) or False
            ts.check_once()
            self.assertEqual(unlock_calls, [])

        with self.subTest(msg="new_peak_clears_stale_partial_exit_tracking"):
            self.tearDown(); self.setUp()
            api = FakeApi(250.0, free=5.0)
            ts = self.ts(api, min_profit_pct=0.0)
            seeded = {
                "TAOUSDC": {
                    "peak": 250.0,
                    "target_exit_qty": 2.5,
                    "exit_sold_qty": 1.0,
                    "exit_sold_cost": 190.0,
                }
            }
            ts._save(seeded)
            api.price = 260.0  # New peak > 250.0
            ts.check_once()
            st = ts._load()["TAOUSDC"]
            self.assertEqual(st["peak"], 260.0)
            self.assertNotIn("target_exit_qty", st)
            self.assertNotIn("exit_sold_qty", st)
            self.assertNotIn("exit_sold_cost", st)

        with self.subTest(msg="retry_counts_persisted_across_adapter_restarts"):
            self.tearDown(); self.setUp()
            api = FakeApi(250.0, free=2.0)
            ts1 = self.ts(api)
            ts1.check_once()  # peak = 250.0
            api.price = 190.0
            ts1.check_once()  # SELL attempt 1
            cid1 = self.po.orders[-1]["client_order_id"]

            # Reconcile canceled 0-fill order
            self.po.status = "canceled"
            ts1.check_once()

            # Simulate --once / daemon restart with a brand new TrailingStop instance
            self.po.status = "closed"
            ts2 = self.ts(api)
            ts2.check_once()
            cid2 = self.po.orders[-1]["client_order_id"]
            self.assertNotEqual(cid1, cid2, "restarted adapter must not reuse canceled client_order_id")

        with self.subTest(msg="corrupted_peak_or_warmup_and_market_data_hardening"):
            self.tearDown(); self.setUp()
            import importlib.util
            from trailing_core import should_sell
            from kraken.market_data import pair_precision

            hl_md_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hyperliquid", "market_data.py")
            hl_spec = importlib.util.spec_from_file_location("hl_md_test", hl_md_path)
            hl_md = importlib.util.module_from_spec(hl_spec)
            hl_spec.loader.exec_module(hl_md)

            self.assertFalse(should_sell(True, 100.0, 10.0))
            self.assertFalse(should_sell(80.0, True, 10.0))
            self.assertFalse(should_sell(80.0, 100.0, True))

            self.assertEqual(pair_precision({"pair_decimals": None, "lot_decimals": None, "ordermin": -5}), (2, 8, 0.0))
            self.assertEqual(
                hl_md.normalize_closed_candles(
                    [
                        {"t": 1000, "T": 2000, "o": "nan", "h": 10, "l": 9, "c": 10},
                        {"t": 2000, "T": 3000, "o": 10, "h": 11, "l": 9, "c": 10.5},
                    ],
                    now_ms=5000,
                ),
                [{"timestamp": 2, "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.5}],
            )

            api = FakeApi(250.0, free=2.0)
            ts = self.ts(api)
            ts._save({"TAOUSDC": {"peak": None, "warmup_at": None}})
            ts.check_once()
            st = ts._load()["TAOUSDC"]
            self.assertEqual(st["peak"], 250.0)
            self.assertNotIn("warmup_at", st)

        with self.subTest(msg="expired_sell_trend_deferral_persists_if_order_canceled_zero_fill"):
            self.tearDown(); self.setUp()
            from unittest.mock import patch
            api = FakeApi(250.0, free=2.0)
            ts = self.ts(api)
            ts._trend_value = lambda _s: 1.0
            with patch("trailing_core.time.time", return_value=1000.0):
                ts.check_once()  # peak = 250.0
                api.price = 190.0  # -24% > 22%
                ts.check_once()  # t=1000s: deferred by UP trend
            self.assertEqual(self.po.orders, [])

            # At t=1420s (420s elapsed), deferral expires -> SELL attempt 1
            with patch("trailing_core.time.time", return_value=1420.0):
                ts.check_once()
            self.assertEqual(len(self.po.orders), 1)

            # Suppose SELL attempt 1 is canceled with 0 fill; next tick must retry immediately
            # at t=1425s without starting a new 420s deferral timer!
            self.po.status = "canceled"
            with patch("trailing_core.time.time", return_value=1425.0):
                ts.check_once()  # reconciles canceled 0-fill
                self.po.status = "closed"
                ts.check_once()  # retries immediately because sell_defer_started_at=1000.0 is preserved
            self.assertEqual(len(self.po.orders), 2)



if __name__ == "__main__":
    unittest.main(verbosity=2)


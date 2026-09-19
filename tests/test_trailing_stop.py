#!/usr/bin/env python3
"""Tests for trailing_stop (no real API, no money)."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from binance_api.trailing_stop import TrailingStop, should_sell  # noqa: E402
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
        order = {"side": side, "symbol": symbol, "price": price,
                            "qty": qty, "force": force,
                            "client_order_id": kw.get("client_order_id"),
                            "bypass_profit_guard": bool(
                                kw.get("bypass_profit_guard", False)
                            )}
        self.orders.append(order)
        if self.result and order["client_order_id"]:
            self.by_client_id[order["client_order_id"]] = self.result
        return self.result
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
    def tearDown(self):
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
            ts.check_once()                                # varf 250
            api.price = 190.0                              # -24% de la 250 (prag TAO 22%)
            ts.check_once()
            self.assertEqual(len(self.po.orders), 1)
            self.assertEqual(self.po.orders[0]["side"], "SELL")
            self.assertTrue(self.po.orders[0]["force"], "force=True is required so it bypasses the weight")
            self.assertTrue(
                self.po.orders[0]["bypass_profit_guard"],
                "the protective exit must bypass the profit guard explicitly",
            )

        with self.subTest(msg="a_small_fall_does_not_sell"):
            self.tearDown(); self.setUp()
            api = FakeApi(250.0)
            ts = self.ts(api)
            ts.check_once()
            api.price = 240.0                              # -4% < 22%
            ts.check_once()
            self.assertEqual(self.po.orders, [])

        with self.subTest(msg="sub_notional_minim_ignora"):
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
            self.ts(api).check_once()                      # varf 260, instanta 1
            api.price = 200.0                              # -23% de la 260 (prag 22%)
            self.ts(api).check_once()                      # instance 2 (a restart) — it reads the peak
            self.assertEqual(len(self.po.orders), 1, "varful 260 supravietuieste restartului")

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
            ts.check_once()                                # status terminal confirmat
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


class TestPerMoneda(Base):
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

        with self.subTest(msg="prag_diferentiat"):
            self.tearDown(); self.setUp()
            ts = self.ts(FakeApi(1.0))
            self.assertEqual(ts.trail_pct_for("BTCUSDC"), 20.0)
            self.assertEqual(ts.trail_pct_for("TAOUSDC"), 22.0)
            self.assertEqual(ts.trail_pct_for("ARBUSDC"), 26.0)
            with self.assertRaises(KeyError):
                ts.trail_pct_for("XYZUSDC")  # No invented policy for an unconfigured asset.

        with self.subTest(msg="sell_fraction_invalid_esueaza_la_start"):
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

        with self.subTest(msg="warming_up_does_not_sell_below_the_threshold"):
            self.tearDown(); self.setUp()
            api = FakeApi(250.0)
            ts = self.ts(api, min_profit_pct=5.0)
            ts.check_once()                    # initial=250, activ la 262.5
            api.price = 190.0                  # a crash of -24% but below the activation threshold
            ts.check_once()
            self.assertEqual(self.po.orders, [], "it does not sell before the profit threshold is reached")

        with self.subTest(msg="active_past_the_threshold_sells"):
            self.tearDown(); self.setUp()
            api = FakeApi(250.0)
            ts = self.ts(api, min_profit_pct=5.0)
            ts.check_once()                    # initial=250
            api.price = 263.0                  # +5.2% > 5% prag -> trailing activ
            ts.check_once()                    # peak=263
            api.price = 200.0                  # -23.9% de la peak 263 (prag TAO 22%)
            ts.check_once()
            self.assertEqual(len(self.po.orders), 1, "it sells once the profit threshold is passed")
            self.assertEqual(self.po.orders[0]["side"], "SELL")

        with self.subTest(msg="it_initially_resets_to_the_rebuy"):
            self.tearDown(); self.setUp()
            api = FakeApi(250.0)
            ts = self.ts(api, min_profit_pct=5.0)
            ts.check_once()                    # initial=250, peak=250
            api.price = 263.0; ts.check_once() # trece de prag -> activ
            api.price = 200.0; ts.check_once() # a crash of -23.9% -> it sells; it arms the rebuy
            self.assertEqual(len(self.po.orders), 1)
            ts.check_once()                    # it confirms the SELL fill and arms the re-buy
            # simulate the rebuy: the price rises 1.2% from 200 -> 202.4
            api.price = 199.0; ts.check_once() # low=199
            api.price = 201.5; ts.check_once() # +1.26% de la 199 -> re-buy; initial=201.5
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


if __name__ == "__main__":
    unittest.main(verbosity=2)

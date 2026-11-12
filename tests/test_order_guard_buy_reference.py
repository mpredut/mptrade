"""The per-venue switch for the historical BUY reference in order_guard.profit_guard.

With the reference on, a BUY must sit below the lowest sell of the window (minus the
margin); after a rise that anchor blocked every automated Binance buy for two weeks.
Switching it off for a venue must leave the SELL reference untouched."""

import order_guard
import unittest
from external_order_guard import check_external_order_guards
from instrument import Instrument
from llm_order_guard import check_geopolitical_order_guards, check_llm_order_guards
from providers import order_id_context as oic
from providers.market_api import BinanceProvider, MarketApi
from qlimit_order_guard import check_qlimit2_order_guards
from tests.conftest import DummySpotExecutionProvider, make_test_margins
from types import SimpleNamespace
from unittest.mock import patch


def _Provider(name, last_fill=216.28, *, free_bal=10000.0):
    return SimpleNamespace(
        name=name,
        last_opposite_fill=lambda _symbol, _order_type: last_fill,
        free_balance=lambda _asset: free_bal,
        validate_symbol=lambda _symbol: None,
        guards_internally=lambda: False,
    )


def _OrdersProvider(orders=(), *, last_fill=None, side_filter=None, name="binance", orders_by_side=None):
    return DummySpotExecutionProvider(
        name, orders=orders, last_fill=last_fill, side_filter=side_filter, orders_by_side=orders_by_side, free_bal=10000.0
    )


_margins = lambda **overrides: make_test_margins(strip_venue_refs=True, **overrides)


class BuyReferenceSwitchTest(unittest.TestCase):
    def test_buy_reference_off_lets_a_buy_above_old_sells_through(self):
        with patch.object(order_guard, "_MARGINS", _margins(binance_buy_reference="off")):
            provider = _Provider("binance")
            # TAO on 24 Sep: lowest sell of the window 216.28, price ~293.
            self.assertTrue(order_guard.profit_guard(
                provider, "TAOUSDC", "BUY", 293.0, 1.15, window_ref=216.28))

    def test_sell_reference_stays_active_when_buy_reference_is_off(self):
        with patch.object(order_guard, "_MARGINS", _margins(binance_buy_reference="off")):
            provider = _Provider("binance")
            # Highest buy 276.0: a SELL below it (plus margin) is still refused.
            self.assertFalse(order_guard.profit_guard(
                provider, "TAOUSDC", "SELL", 270.0, 1.15, window_ref=276.0))
            self.assertTrue(order_guard.profit_guard(
                provider, "TAOUSDC", "SELL", 290.0, 1.15, window_ref=276.0))

    def test_other_venues_keep_the_buy_reference(self):
        with patch.object(order_guard, "_MARGINS", _margins(binance_buy_reference="off")):
            self.assertFalse(order_guard.profit_guard(
                _Provider("kraken"), "HYPEUSD", "BUY", 93.0, 1.15, window_ref=56.66))
            self.assertNotEqual(order_guard.get_profit_guard_mode_cfg("kraken", "BUY"), "off")
            self.assertEqual(order_guard.get_profit_guard_mode_cfg("binance", "BUY"), "off")

    def test_unreadable_value_raises_value_error(self):
        with patch.object(order_guard, "_MARGINS", _margins(binance_buy_reference="maybe")):
            with self.assertRaises(ValueError):
                order_guard.get_profit_guard_mode_cfg("binance", "BUY")

    def test_repository_config_sets_binance_buy_reference_dynamic(self):
        with patch.object(order_guard, "_MARGINS", None):
            self.assertEqual(order_guard.get_profit_guard_mode_cfg("binance", "BUY"), "dynamic")
            self.assertEqual(order_guard.get_profit_guard_mode_cfg("kraken", "BUY"), "strict")

    def test_dynamic_mode_bypasses_in_bull_trend_and_enforces_in_bear_or_flat(self):
        with patch.object(order_guard, "_MARGINS", _margins(binance_buy_reference="dynamic")):
            provider = _Provider("binance")
            # In confirmed BULL trend: BUY above old sell reference is permitted
            with patch.object(order_guard, "_symbol_trend", return_value="bull"):
                self.assertTrue(order_guard.profit_guard(
                    provider, "TAOUSDC", "BUY", 293.0, 1.15, window_ref=216.28))
            # In BEAR trend: BUY above old sell reference is blocked for defense
            with patch.object(order_guard, "_symbol_trend", return_value="bear"):
                self.assertFalse(order_guard.profit_guard(
                    provider, "TAOUSDC", "BUY", 293.0, 1.15, window_ref=216.28))
            # In FLAT/UNKNOWN trend: defensive guard is maintained
            with patch.object(order_guard, "_symbol_trend", return_value="unknown"):
                self.assertFalse(order_guard.profit_guard(
                    provider, "TAOUSDC", "BUY", 293.0, 1.15, window_ref=216.28))

    def test_provider_as_string_and_case_insensitivity(self):
        with patch.object(order_guard, "_MARGINS", _margins(binance_buy_reference="off")):
            # Both string name and Provider object with various casings must work
            self.assertTrue(order_guard.profit_guard("binance", "TAOUSDC", "BUY", 293.0, 1.15, window_ref=216.28))
            self.assertTrue(order_guard.profit_guard("Binance", "TAOUSDC", "BUY", 293.0, 1.15, window_ref=216.28))
            self.assertTrue(order_guard.profit_guard(_Provider("Binance"), "TAOUSDC", "BUY", 293.0, 1.15, window_ref=216.28))
            self.assertEqual(order_guard.get_profit_guard_mode_cfg("Binance", "BUY"), "off")

    def test_missing_default_buy_reference_in_margins_raises_key_error(self):
        with patch.object(order_guard, "_MARGINS", {"binance": 1.15}):
            with self.assertRaises(KeyError):
                order_guard.get_profit_guard_mode_cfg("binance", "BUY")
            with self.assertRaises(KeyError):
                order_guard.get_profit_guard_mode_cfg("kraken", "BUY")

    def test_real_binance_provider_integration(self):
        with patch.object(order_guard, "_MARGINS", None):
            provider = BinanceProvider()
            self.assertEqual(order_guard.get_profit_guard_mode_cfg(provider.name, "BUY"), "dynamic")
            # Keep the real provider/config routing, but isolate its history I/O.
            with patch.object(provider, "get_orders", return_value=[]) as history, \
                    patch.object(order_guard, "_symbol_trend", return_value="bull"):
                self.assertTrue(order_guard.profit_guard(provider, "TAOUSDC", "BUY", 293.0, 1.15, window_ref=216.28))
                history.assert_called_with(
                    "TAOUSDC", "SELL", order_guard.get_dynamic_window_sec_cfg("TAOUSDC"))
                history.return_value = [{"price": 216.28}]
                self.assertFalse(order_guard.profit_guard(
                    provider, "TAOUSDC", "BUY", 293.0, 1.15, window_ref=216.28))
            # SELL below historical buy reference is still blocked
            self.assertFalse(order_guard.profit_guard(provider, "TAOUSDC", "SELL", 270.0, 1.15, window_ref=276.0))

    def test_instrument_pipeline_respects_buy_reference_switch(self):
        p = DummySpotExecutionProvider("binance", free_bal=10000.0, requires_permit=True)
        p.placed = []
        p.get_current_price = lambda symbol: 293.0
        p.profit_guard_window_ref = lambda symbol, side, safeback_override=None: 216.28 if side == "BUY" else 276.0
        p.order_status = lambda symbol, order_id: {"orderId": order_id, "status": "FILLED"}
        p.place_order = lambda symbol, side, price, qty, **kwargs: (
            p.placed.append({"side": side, "qty": qty, "price": price}) or {"orderId": "123", "status": "FILLED"}
        )
        inst = Instrument(name="TAO", symbol="TAOUSDC", provider="binance", base="TAO", quote="USDC", api=MarketApi([p]))

        # Exercise reference routing without live trend discovery on a cold cache.
        with patch.object(order_guard, "_symbol_trend", return_value="unknown"):
            # With binance_buy_reference = off: BUY at 293 (above past sell 216.28) is accepted
            with patch.object(order_guard, "_MARGINS", _margins(binance_buy_reference="off")):
                res = inst.place("BUY", 293.0, 1.0, motivation="test_buy", cache_permit=object(), smart=False, wait_for_trend=False, caller_owns_retry=True)
                self.assertIsNotNone(res)
                self.assertEqual(len(p.placed), 1)

            # With binance_buy_reference = strict: BUY at 293 (above past sell 216.28) is refused by profit_guard
            p.placed.clear()
            with patch.object(order_guard, "_MARGINS", _margins(binance_buy_reference="strict")):
                res = inst.place("BUY", 293.0, 1.0, motivation="test_buy", cache_permit=object(), smart=False, wait_for_trend=False, caller_owns_retry=True)
                self.assertIsNone(res)
                self.assertEqual(len(p.placed), 0)

    def test_dynamic_window_scaling_by_trend(self):
        # Favorable: 8h (BUY in bull / SELL in bear), Flat: 24h, Adverse: 72h (BUY in bear / SELL in bull)
        with patch.object(order_guard, "_symbol_trend", return_value="bull"):
            self.assertEqual(order_guard.get_dynamic_window_sec_cfg("TAOUSDC", order_type="BUY"), 8.0 * 3600.0)
            self.assertEqual(order_guard.get_dynamic_window_sec_cfg("TAOUSDC", order_type="SELL"), 72.0 * 3600.0)
            self.assertEqual(order_guard.get_window_cfg("binance", "TAOUSDC", "BUY"), 8.0 * 3600.0)

        with patch.object(order_guard, "_symbol_trend", return_value="bear"):
            self.assertEqual(order_guard.get_dynamic_window_sec_cfg("TAOUSDC", order_type="BUY"), 72.0 * 3600.0)
            self.assertEqual(order_guard.get_dynamic_window_sec_cfg("TAOUSDC", order_type="SELL"), 8.0 * 3600.0)
            self.assertEqual(order_guard.get_window_cfg("binance", "TAOUSDC", "BUY"), 72.0 * 3600.0)

        with patch.object(order_guard, "_symbol_trend", return_value="flat"):
            self.assertEqual(order_guard.get_dynamic_window_sec_cfg("TAOUSDC", order_type="BUY"), 24.0 * 3600.0)
            self.assertEqual(order_guard.get_dynamic_window_sec_cfg("TAOUSDC", order_type="SELL"), 24.0 * 3600.0)
            self.assertEqual(order_guard.get_window_cfg("binance", "TAOUSDC", "BUY"), 24.0 * 3600.0)

        with patch.object(order_guard, "_symbol_trend", return_value="unknown"):
            self.assertEqual(order_guard.get_dynamic_window_sec_cfg("TAOUSDC", order_type="BUY"), 24.0 * 3600.0)
            self.assertEqual(order_guard.get_dynamic_window_sec_cfg("TAOUSDC", order_type="SELL"), 24.0 * 3600.0)

    def test_dynamic_window_bounds_clamping(self):
        # Configured extreme values are safely clamped into user-defined bands
        with patch.object(order_guard, "_MARGINS", _margins(dynamic_window_favorable_h=1.0, dynamic_window_adverse_h=500.0)):
            with patch.object(order_guard, "_symbol_trend", return_value="bull"):
                self.assertEqual(order_guard.get_dynamic_window_sec_cfg("TAOUSDC", order_type="BUY"), 4.0 * 3600.0)  # clamped to min 4h
                self.assertEqual(order_guard.get_dynamic_window_sec_cfg("TAOUSDC", order_type="SELL"), 168.0 * 3600.0)  # clamped to max 168h
            with patch.object(order_guard, "_symbol_trend", return_value="bear"):
                self.assertEqual(order_guard.get_dynamic_window_sec_cfg("TAOUSDC", order_type="BUY"), 168.0 * 3600.0)  # clamped to max 168h (7d)
                self.assertEqual(order_guard.get_dynamic_window_sec_cfg("TAOUSDC", order_type="SELL"), 4.0 * 3600.0)  # clamped to min 4h

    def test_dynamic_window_protects_against_recent_top_and_permits_pullback(self):
        # Recent sell at 290.0 within 8h bull window
        provider = _OrdersProvider([{"price": 290.0, "qty": 1.0, "timestamp": 1000.0}])
        with patch.object(order_guard, "_MARGINS", _margins(binance_buy_reference="dynamic")):
            with patch.object(order_guard, "_symbol_trend", return_value="bull"):
                # BUY at 293.0 is higher than recent sell 290.0 -> BLOCKED (protects from buying top)
                self.assertFalse(order_guard.profit_guard(
                    provider, "TAOUSDC", "BUY", 293.0, 1.15, window_ref=290.0))
                # BUY at 285.0 is below recent sell 290.0 (+1.72% diff) -> ALLOWED (pullback re-entry)
                self.assertTrue(order_guard.profit_guard(
                    provider, "TAOUSDC", "BUY", 285.0, 1.15, window_ref=290.0))

    def test_dynamic_window_ancient_sell_expires_in_bull_trend(self):
        provider = _OrdersProvider(last_fill=216.28)
        with patch.object(order_guard, "_MARGINS", _margins(binance_buy_reference="dynamic")):
            with patch.object(order_guard, "_symbol_trend", return_value="bull"):
                # window_ref is None because no orders in 8h -> permitted without 14-day lockup
                self.assertTrue(order_guard.profit_guard(
                    provider, "TAOUSDC", "BUY", 293.0, 1.15, window_ref=None))
                # Even if legacy caller passed ancient window_ref 216.28, get_orders shows 0 recent fills in 8h
                self.assertTrue(order_guard.profit_guard(
                    provider, "TAOUSDC", "BUY", 293.0, 1.15, window_ref=216.28))

    def test_dynamic_window_bear_market_holds_long_defensive_memory(self):
        provider = _OrdersProvider([{"price": 300.0, "qty": 1.0, "timestamp": 1000.0}])
        with patch.object(order_guard, "_MARGINS", _margins(binance_buy_reference="dynamic")):
            with patch.object(order_guard, "_symbol_trend", return_value="bear"):
                # BUY at 298.0 against 300.0 sell is only 0.67% diff < 1.15% -> BLOCKED
                self.assertFalse(order_guard.profit_guard(
                    provider, "TAOUSDC", "BUY", 298.0, 1.15, window_ref=300.0))

    def test_dynamic_window_ancient_sell_expires_in_flat_trend(self):
        provider = _OrdersProvider(last_fill=216.28)
        with patch.object(order_guard, "_MARGINS", _margins(binance_buy_reference="dynamic")):
            with patch.object(order_guard, "_symbol_trend", return_value="flat"):
                # In flat trend (24h window), 10-day-old anchor 216.28 is outside 24h -> BUY at 298 is permitted!
                self.assertTrue(order_guard.profit_guard(
                    provider, "TAOUSDC", "BUY", 298.0, 1.15, window_ref=216.28))

    def test_sell_orders_never_use_short_dynamic_window(self):
        # On SELL with static mode ('strict'), get_window_cfg returns full venue window (e.g. 336h kraken)
        with patch.object(order_guard, "_symbol_trend", return_value="bull"):
            kraken_sell_win = order_guard.get_window_cfg("kraken", "HYPEUSD", "SELL")
            self.assertEqual(kraken_sell_win, 336.0 * 3600.0)

        # On SELL, profit guard NEVER allows selling below buy price + margin
        provider = _Provider("binance")
        self.assertFalse(order_guard.profit_guard(
            provider, "TAOUSDC", "SELL", 270.0, 1.15, window_ref=276.0))

    def test_math_rejection_short_circuits_intelligence_guards(self):
        """Mathematical failure must return False without executing intelligence or macro shadow checks."""
        provider = _Provider("binance", last_fill=216.28)

        with patch.object(order_guard, "_MARGINS", _margins(binance_buy_reference="strict")):
            with patch.object(order_guard, "check_intelligence_guards", return_value=(True, "ok", 1.0)) as mock_intel:
                # BUY at 293 with sell anchor 216.28 -> mathematically unprofitable (diff = -35% < 1.15%)
                allowed = order_guard.profit_guard(
                    provider, "TAOUSDC", "BUY", 293.0, 1.15, window_ref=216.28
                )
                self.assertFalse(allowed)
                self.assertFalse(mock_intel.called, "Intelligence guards should not be evaluated on mathematically rejected orders")

    def test_profit_guard_skips_intelligence_when_qty_none_or_non_positive(self):
        """Mathematically valid orders must skip heavy intelligence evaluation when qty is None or <= 0."""
        provider = _Provider("binance", last_fill=300.0)

        with patch.object(order_guard, "_MARGINS", _margins(binance_buy_reference="strict")):
            with patch.object(order_guard, "check_intelligence_guards", return_value=(True, "ok", 1.0)) as mock_intel:
                # Mathematically profitable: BUY 280 vs sell anchor 300 (diff 6.67% >= 1.15%)
                allowed_none = order_guard.profit_guard(
                    provider, "TAOUSDC", "BUY", 280.0, 1.15, window_ref=300.0, qty=None
                )
                self.assertTrue(allowed_none)
                self.assertFalse(mock_intel.called, "Intelligence guards must be skipped when qty is None")

                mock_intel.reset_mock()
                allowed_zero = order_guard.profit_guard(
                    provider, "TAOUSDC", "BUY", 280.0, 1.15, window_ref=300.0, qty=0.0
                )
                self.assertTrue(allowed_zero)
                self.assertFalse(mock_intel.called, "Intelligence guards must be skipped when qty is 0.0")

                mock_intel.reset_mock()
                allowed_positive = order_guard.profit_guard(
                    provider, "TAOUSDC", "BUY", 280.0, 1.15, window_ref=300.0, qty=1.5
                )
                self.assertTrue(allowed_positive)
                self.assertTrue(mock_intel.called, "Intelligence guards must be evaluated when qty is positive")

    def test_instrument_preflight_refuses_zero_balance(self):
        """Instrument.place must refuse BUY orders at Stage 0 if provider free balance is zero or empty."""
        provider = _Provider("binance", last_fill=300.0, free_bal=0.0)
        api = MarketApi([provider])
        inst = Instrument("TAO", "TAOUSDC", provider="binance", quote="USDC", api=api)
        order = inst.place("BUY", 280.0, 1.0, caller_owns_retry=True)
        self.assertIsNone(order)

    def test_active_cycle_window_ref_ignores_closed_cycles(self):
        """Closed higher-price BUY cycles (including <=2% spot fee dust) must not poison
        the active cycle's SELL reference price."""
        provider = _OrdersProvider(
            name="kraken",
            orders_by_side={
                "BUY": [
                    {"price": 90.0, "qty": 10.0, "timestamp": 3000.0},
                    {"price": 120.0, "qty": 10.0, "timestamp": 1000.0},
                ],
                "SELL": [
                    # Sold 99.9% of the 120.0 lot (0.1% spot fee dust remaining)
                    {"price": 130.0, "qty": 9.99, "timestamp": 2000.0},
                ],
            },
        )
        ref = order_guard.window_reference(provider, "HYPEUSD", "SELL", 14 * 86400)
        self.assertEqual(ref, 90.0)
        with patch.object(order_guard, "_MARGINS", _margins()):
            # Selling at 100.0 (+11.1% above active 90.0 BUY) is permitted
            self.assertTrue(
                order_guard.profit_guard(provider, "HYPEUSD", "SELL", 100.0, 1.15, window_ref=ref)
            )

    def test_active_cycle_window_ref_consumes_harvested_dip_lot_via_lifo(self):
        """When a recent lower DCA tranche is harvested by a SELL while older higher BUY
        tranches remain open, LIFO matching must consume the harvested dip lot rather than
        the oldest higher lot so subsequent SELLs do not reference the already-sold dip price."""
        opp_orders = [
            {"price": 240.5, "qty": 1.2515, "timestamp": 1000.0},
            {"price": 237.1, "qty": 2.1056, "timestamp": 2000.0},
            {"price": 213.2, "qty": 1.7715, "timestamp": 3000.0},
        ]
        same_orders = [
            # Harvested the 213.2 dip lot (1.7715) plus 0.6788 of the 237.1 lot
            {"price": 224.7, "qty": 2.4503, "timestamp": 4000.0},
        ]
        ref_hybrid = order_guard.active_cycle_window_ref(
            opp_orders, same_orders, "SELL", reference_mode="hybrid"
        )
        ref_last = order_guard.active_cycle_window_ref(
            opp_orders, same_orders, "SELL", reference_mode="last"
        )
        self.assertEqual(ref_hybrid, 237.1)
        self.assertEqual(ref_last, 237.1)
        provider = _Provider("binance", last_fill=213.2)
        with patch.object(order_guard, "_MARGINS", _margins()):
            # Selling at 226.2 (above harvested 213.2, but below remaining 237.1 lot) must be blocked
            self.assertFalse(
                order_guard.profit_guard(
                    provider, "TAOUSDC", "SELL", 226.2, 1.15, window_ref=ref_hybrid, reference_mode="hybrid"
                )
            )
            self.assertTrue(
                order_guard.profit_guard(
                    provider, "TAOUSDC", "SELL", 240.0, 1.15, window_ref=ref_hybrid, reference_mode="hybrid"
                )
            )

    def test_sell_falls_back_to_last_opposite_fill_when_window_empty(self):
        """Even on venues with window_h > 0, both BUY and SELL orders in static modes
        with no opposite fill inside the window fall back symmetrically to last_opposite_fill."""
        provider = _Provider("kraken", last_fill=200.0)
        with patch.object(order_guard, "_MARGINS", _margins(kraken_window_h=336.0)):
            # Selling at 190.0 below last_fill 200.0 must be blocked even when window_ref is None
            self.assertFalse(
                order_guard.profit_guard(provider, "HYPEUSD", "SELL", 190.0, 1.15, window_ref=None)
            )
            self.assertTrue(
                order_guard.profit_guard(provider, "HYPEUSD", "SELL", 210.0, 1.15, window_ref=None)
            )
            # Symmetrically, buying at 210.0 above last_fill 200.0 in 'strict' mode is blocked when window_ref is None
            self.assertFalse(
                order_guard.profit_guard(provider, "HYPEUSD", "BUY", 210.0, 1.15, window_ref=None)
            )
            self.assertTrue(
                order_guard.profit_guard(provider, "HYPEUSD", "BUY", 190.0, 1.15, window_ref=None)
            )

    def test_dynamic_window_recomputes_ref_from_recent_fills_ignoring_older_14d_anchor(self):
        """When caller passes a 14d static window_ref (e.g. 216.28) that is older than dyn_window_s,
        and a newer SELL at 300.0 exists inside dyn_window_s, dynamic mode must use min(recent_prices)=300.0."""
        # Inside the 8h/24h dynamic window, only the 300.0 sell exists
        provider = _OrdersProvider([{"price": 300.0, "qty": 1.0, "timestamp": 5000.0}])
        with patch.object(order_guard, "_MARGINS", _margins(binance_buy_reference="dynamic")):
            with patch.object(order_guard, "_symbol_trend", return_value="flat"):
                # Caller passes 14-day min sell window_ref=216.28, but inside dynamic window min sell is 300.0.
                # Buying at 280.0 (+6.67% below 300.0 >= 1.15%) must be PERMITTED!
                self.assertTrue(
                    order_guard.profit_guard(provider, "TAOUSDC", "BUY", 280.0, 1.15, window_ref=216.28)
                )
                # Buying at 299.0 (+0.33% below 300.0 < 1.15%) must still be BLOCKED!
                self.assertFalse(
                    order_guard.profit_guard(provider, "TAOUSDC", "BUY", 299.0, 1.15, window_ref=216.28)
                )

    def test_sell_reference_difficulty_tiers_hierarchy(self):
        """Verify all 4 SELL difficulty levels (strict, average, last, hybrid) and off."""
        _MultiBuyProvider = lambda orders: _OrdersProvider(orders, side_filter="BUY")

        # Scenario 1: DCA down (bought 2 TAO @ 300.0 ten days ago, then 2 TAO @ 270.0 recently)
        # - strict (max)   = 300.0 -> requires >= 303.49
        # - average (VWAP) = 285.0 -> requires >= 288.32
        # - last           = 270.0 -> requires >= 273.15
        # - hybrid         = min(285.0, 270.0) = 270.0 -> requires >= 273.15
        dca_down_orders = [
            {"price": 300.0, "qty": 2.0, "timestamp": 1000.0},
            {"price": 270.0, "qty": 2.0, "timestamp": 2000.0},
        ]
        p_down = _MultiBuyProvider(dca_down_orders)

        self.assertAlmostEqual(
            order_guard.window_reference(p_down, "TAOUSDC", "SELL", 14 * 86400, reference_mode="strict"),
            300.0,
        )
        self.assertAlmostEqual(
            order_guard.window_reference(p_down, "TAOUSDC", "SELL", 14 * 86400, reference_mode="average"),
            285.0,
        )
        self.assertAlmostEqual(
            order_guard.window_reference(p_down, "TAOUSDC", "SELL", 14 * 86400, reference_mode="last"),
            270.0,
        )
        self.assertAlmostEqual(
            order_guard.window_reference(p_down, "TAOUSDC", "SELL", 14 * 86400, reference_mode="hybrid"),
            270.0,
        )
        self.assertIsNone(
            order_guard.window_reference(p_down, "TAOUSDC", "SELL", 14 * 86400, reference_mode="off")
        )

        # Selling at 281.1 (yesterday's TAO peak):
        # - Blocked under 'strict' (300.0) and 'average' (285.0)
        # - Permitted under 'last' (270.0) and 'hybrid' (270.0)
        # - Selling at 268.0 (in loss vs both average and last) is still BLOCKED even under 'hybrid'!
        with patch.object(order_guard, "_MARGINS", _margins()):
            self.assertFalse(
                order_guard.profit_guard(p_down, "TAOUSDC", "SELL", 281.1, 1.15, reference_mode="strict")
            )
            self.assertFalse(
                order_guard.profit_guard(p_down, "TAOUSDC", "SELL", 281.1, 1.15, reference_mode="average")
            )
            self.assertTrue(
                order_guard.profit_guard(p_down, "TAOUSDC", "SELL", 281.1, 1.15, reference_mode="last")
            )
            self.assertTrue(
                order_guard.profit_guard(p_down, "TAOUSDC", "SELL", 281.1, 1.15, reference_mode="hybrid")
            )
            self.assertFalse(
                order_guard.profit_guard(p_down, "TAOUSDC", "SELL", 268.0, 1.15, reference_mode="hybrid")
            )

        # Scenario 2: Bought main position low (3 TAO @ 260.0) and added a small top-up (1 TAO @ 280.0)
        # - average (VWAP) = (3*260 + 1*280)/4 = 265.0
        # - last           = 280.0
        # - hybrid         = min(265.0, 280.0) = 265.0
        dca_up_orders = [
            {"price": 260.0, "qty": 3.0, "timestamp": 1000.0},
            {"price": 280.0, "qty": 1.0, "timestamp": 2000.0},
        ]
        p_up = _MultiBuyProvider(dca_up_orders)
        self.assertAlmostEqual(
            order_guard.window_reference(p_up, "TAOUSDC", "SELL", 14 * 86400, reference_mode="average"),
            265.0,
        )
        self.assertAlmostEqual(
            order_guard.window_reference(p_up, "TAOUSDC", "SELL", 14 * 86400, reference_mode="last"),
            280.0,
        )
        self.assertAlmostEqual(
            order_guard.window_reference(p_up, "TAOUSDC", "SELL", 14 * 86400, reference_mode="hybrid"),
            265.0,
        )

    def test_buy_reference_difficulty_tiers_symmetry(self):
        """Verify BUY difficulty levels (strict, average, last, hybrid, off) are 100% symmetric with SELL."""
        _MultiSellProvider = lambda orders: _OrdersProvider(orders, side_filter="SELL")

        # Scenario 1: Scaled out on the way up (sold 2 TAO @ 270.0 earlier, then 2 TAO @ 300.0 recently)
        # - strict (min)   = 270.0 -> requires BUY <= ~266.89
        # - average (VWAP) = 285.0 -> requires BUY <= ~281.72
        # - last           = 300.0 -> requires BUY <= ~296.55
        # - hybrid         = max(285.0, 300.0) = 300.0 -> requires BUY <= ~296.55
        scale_up_sells = [
            {"price": 270.0, "qty": 2.0, "timestamp": 1000.0},
            {"price": 300.0, "qty": 2.0, "timestamp": 2000.0},
        ]
        p_up = _MultiSellProvider(scale_up_sells)

        self.assertAlmostEqual(
            order_guard.window_reference(p_up, "TAOUSDC", "BUY", 14 * 86400, reference_mode="strict"),
            270.0,
        )
        self.assertAlmostEqual(
            order_guard.window_reference(p_up, "TAOUSDC", "BUY", 14 * 86400, reference_mode="average"),
            285.0,
        )
        self.assertAlmostEqual(
            order_guard.window_reference(p_up, "TAOUSDC", "BUY", 14 * 86400, reference_mode="last"),
            300.0,
        )
        self.assertAlmostEqual(
            order_guard.window_reference(p_up, "TAOUSDC", "BUY", 14 * 86400, reference_mode="hybrid"),
            300.0,
        )
        self.assertIsNone(
            order_guard.window_reference(p_up, "TAOUSDC", "BUY", 14 * 86400, reference_mode="off")
        )

        # Rebuying at 290.0 (-3.33% below last SELL 300.0, > 1.15% profit margin):
        # - Blocked under 'strict' (270.0) and 'average' (285.0)
        # - Permitted under 'last' (300.0) and 'hybrid' (300.0)
        # - Rebuying at 302.0 (in loss vs both average and last) is still BLOCKED even under 'hybrid'!
        with patch.object(order_guard, "_MARGINS", _margins()):
            self.assertFalse(
                order_guard.profit_guard(p_up, "TAOUSDC", "BUY", 290.0, 1.15, reference_mode="strict")
            )
            self.assertFalse(
                order_guard.profit_guard(p_up, "TAOUSDC", "BUY", 290.0, 1.15, reference_mode="average")
            )
            self.assertTrue(
                order_guard.profit_guard(p_up, "TAOUSDC", "BUY", 290.0, 1.15, reference_mode="last")
            )
            self.assertTrue(
                order_guard.profit_guard(p_up, "TAOUSDC", "BUY", 290.0, 1.15, reference_mode="hybrid")
            )
            self.assertFalse(
                order_guard.profit_guard(p_up, "TAOUSDC", "BUY", 302.0, 1.15, reference_mode="hybrid")
            )

        # Scenario 2: Sold main position high (3 TAO @ 310.0) and a small tranche lower (1 TAO @ 290.0)
        # - average (VWAP) = (3*310 + 1*290)/4 = 305.0
        # - last           = 290.0
        # - hybrid         = max(305.0, 290.0) = 305.0
        scale_down_sells = [
            {"price": 310.0, "qty": 3.0, "timestamp": 1000.0},
            {"price": 290.0, "qty": 1.0, "timestamp": 2000.0},
        ]
        p_down = _MultiSellProvider(scale_down_sells)
        self.assertAlmostEqual(
            order_guard.window_reference(p_down, "TAOUSDC", "BUY", 14 * 86400, reference_mode="average"),
            305.0,
        )
        self.assertAlmostEqual(
            order_guard.window_reference(p_down, "TAOUSDC", "BUY", 14 * 86400, reference_mode="last"),
            290.0,
        )
        self.assertAlmostEqual(
            order_guard.window_reference(p_down, "TAOUSDC", "BUY", 14 * 86400, reference_mode="hybrid"),
            305.0,
        )

    def test_per_module_and_per_provider_reference_mode_resolution(self):
        """Verify per-module (assetguardian, monitortrades, tradeall, rtrade) and per-provider strict hierarchy."""
        custom_margins = _margins(
            default_buy_reference="strict",
            default_sell_reference="strict",
            binance_buy_reference="dynamic",
            binance_sell_reference="average",
            kraken_buy_reference="strict",
            kraken_sell_reference="strict",
            assetguardian_buy_reference="venue",
            assetguardian_sell_reference="hybrid",
            monitortrades_sell_reference="last",
            tradeall_buy_reference="average",
            rtrade_sell_reference="off",
        )
        with patch.object(order_guard, "_MARGINS", custom_margins):
            # Provider-level resolution
            self.assertEqual(order_guard.get_profit_guard_mode_cfg("binance", "BUY"), "dynamic")
            self.assertEqual(order_guard.get_profit_guard_mode_cfg("binance", "SELL"), "average")
            self.assertEqual(order_guard.get_profit_guard_mode_cfg("kraken", "BUY"), "strict")
            self.assertEqual(order_guard.get_profit_guard_mode_cfg("kraken", "SELL"), "strict")

            # Module-level override vs 'venue' inheritance
            self.assertEqual(
                order_guard.get_profit_guard_mode_cfg("binance", "BUY", module="assetguardian"),
                "dynamic",  # inherits binance_buy_reference via 'venue'
            )
            self.assertEqual(
                order_guard.get_profit_guard_mode_cfg("kraken", "BUY", module="assetguardian"),
                "strict",   # inherits kraken_buy_reference via 'venue'
            )
            self.assertEqual(
                order_guard.get_profit_guard_mode_cfg("binance", "SELL", module="assetguardian"),
                "hybrid",   # overrides binance_sell_reference ('average') with 'hybrid'
            )
            self.assertEqual(
                order_guard.get_profit_guard_mode_cfg("binance", "SELL", module="monitortrades"),
                "last",
            )
            self.assertEqual(
                order_guard.get_profit_guard_mode_cfg("binance", "BUY", module="tradeall"),
                "average",
            )
            self.assertEqual(
                order_guard.get_profit_guard_mode_cfg("binance", "SELL", module="rtrade"),
                "off",
            )

            # Missing module key raises KeyError (fail fast)
            with self.assertRaises(KeyError):
                order_guard.get_profit_guard_mode_cfg("binance", "BUY", module="rtrade")

            # Explicit reference_mode argument must be a valid mode
            self.assertEqual(
                order_guard.get_profit_guard_mode_cfg(
                    "binance", "SELL", module="assetguardian", reference_mode="strict"
                ),
                "strict",
            )
            with self.assertRaises(ValueError):
                order_guard.get_profit_guard_mode_cfg(
                    "binance", "SELL", module="assetguardian", reference_mode="invalid_mode"
                )

    def test_validate_guard_mode_cfg_and_margins_validation_fail_fast(self):
        """Verify invalid Pillar 1-4 guard modes or discrete policies fail fast instead of silently passing."""
        for valid_mode in ("off", "shadow", "enforce"):
            self.assertEqual(order_guard.validate_guard_mode_cfg(valid_mode, "llm_guard_mode"), valid_mode)

        for bad_mode in ("on", "strict", "true", "1", "enforec", 1, 0):
            with self.assertRaises(ValueError):
                order_guard.validate_guard_mode_cfg(bad_mode, "llm_guard_mode")

        base_full = dict(order_guard._load_margins())
        order_guard._validate_margins_cfg(base_full, "order_guard.conf")

        bad_guard_cfg = dict(base_full, llm_guard_mode="on")
        with self.assertRaises(ValueError):
            order_guard._validate_margins_cfg(bad_guard_cfg, "order_guard.conf")

        bad_policy_cfg = dict(base_full, weibull_exhaustion_policy="ignore")
        with self.assertRaises(ValueError):
            order_guard._validate_margins_cfg(bad_policy_cfg, "order_guard.conf")

        bad_ref_cfg = dict(base_full, binance_sell_reference="1")
        with self.assertRaises(ValueError):
            order_guard._validate_margins_cfg(bad_ref_cfg, "order_guard.conf")

        with self.assertRaises(ValueError):
            check_qlimit2_order_guards("BTCUSDC", "BUY", 80000.0, margins=dict(base_full, internal_guard_mode="on"))
        with self.assertRaises(ValueError):
            check_external_order_guards("BTCUSDC", "BUY", 80000.0, margins=dict(base_full, whale_guard_mode="strict"))
        with self.assertRaises(ValueError):
            check_llm_order_guards("binance", "BTCUSDC", "BUY", 80000.0, qty=1.0, margins=dict(base_full, llm_guard_mode="true"))
        with self.assertRaises(ValueError):
            check_geopolitical_order_guards("BTCUSDC", "BUY", margins=dict(base_full, geopolitical_guard_mode="1"))

    def test_repository_config_sets_tradeall_symmetric_hybrid_reference(self):
        """Verify live order_guard.conf sets symmetric hybrid BUY/SELL reference for tradeall."""
        with patch.object(order_guard, "_MARGINS", None):
            self.assertEqual(order_guard.get_profit_guard_mode_cfg("binance", "BUY", module="tradeall"), "hybrid")
            self.assertEqual(order_guard.get_profit_guard_mode_cfg("binance", "SELL", module="tradeall"), "hybrid")
            self.assertEqual(order_guard.get_profit_guard_mode_cfg("binance", "BUY", module="monitortrades"), "dynamic")
            self.assertEqual(order_guard.get_profit_guard_mode_cfg("binance", "SELL", module="monitortrades"), "hybrid")

    def test_resolve_caller_module_preserves_lineage_across_outbox_and_repricer(self):
        """Ensure outbox retries (TA_OR) and monitororder repricing (MO_TA_) retain module identity."""
        with patch.object(oic, "get_process_name", return_value="ORW"):
            self.assertIsNone(order_guard.resolve_caller_module(None))
            self.assertEqual(order_guard.resolve_caller_module(None, prefix="TA_OR"), "tradeall")
            self.assertEqual(order_guard.resolve_caller_module(None, client_order_id="TA_OR_abc123"), "tradeall")
            self.assertEqual(order_guard.resolve_caller_module(None, client_order_id="MO_TA_abc123"), "tradeall")
            self.assertEqual(order_guard.resolve_caller_module(None, prefix="AG_OR"), "assetguardian")
            self.assertEqual(order_guard.resolve_caller_module(None, prefix="MT_OR"), "monitortrades")
            self.assertEqual(order_guard.resolve_caller_module(None, prefix="RT_OR"), "rtrade")

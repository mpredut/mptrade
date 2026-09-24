"""The per-venue switch for the historical BUY reference in order_guard.profit_guard.

With the reference on, a BUY must sit below the lowest sell of the window (minus the
margin); after a rise that anchor blocked every automated Binance buy for two weeks.
Switching it off for a venue must leave the SELL reference untouched."""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import order_guard  # noqa: E402


class _Provider:
    def __init__(self, name, last_fill=216.28):
        self.name = name
        self._last_fill = last_fill

    def last_opposite_fill(self, _symbol, _order_type):
        return self._last_fill


def _margins(**overrides):
    base = {
        "default": 1.15, "binance": 1.15, "default_window_h": 0.0,
        "default_max_daily_trades": 25, "default_safeback_sec": 14 * 24 * 3600 + 60,
        "default_recent_transaction_sec": 180, "default_buy_reference": 1.0,
    }
    base.update(overrides)
    return base


class BuyReferenceSwitchTest(unittest.TestCase):
    def test_buy_reference_off_lets_a_buy_above_old_sells_through(self):
        with mock.patch.object(order_guard, "_MARGINS", _margins(binance_buy_reference=0.0)):
            provider = _Provider("binance")
            # TAO on 24 Sep: lowest sell of the window 216.28, price ~293.
            self.assertTrue(order_guard.profit_guard(
                provider, "TAOUSDC", "BUY", 293.0, 1.15, window_ref=216.28))

    def test_sell_reference_stays_active_when_buy_reference_is_off(self):
        with mock.patch.object(order_guard, "_MARGINS", _margins(binance_buy_reference=0.0)):
            provider = _Provider("binance")
            # Highest buy 276.0: a SELL below it (plus margin) is still refused.
            self.assertFalse(order_guard.profit_guard(
                provider, "TAOUSDC", "SELL", 270.0, 1.15, window_ref=276.0))
            self.assertTrue(order_guard.profit_guard(
                provider, "TAOUSDC", "SELL", 290.0, 1.15, window_ref=276.0))

    def test_other_venues_keep_the_buy_reference(self):
        with mock.patch.object(order_guard, "_MARGINS", _margins(binance_buy_reference=0.0)):
            self.assertFalse(order_guard.profit_guard(
                _Provider("kraken"), "HYPEUSD", "BUY", 93.0, 1.15, window_ref=56.66))
            self.assertTrue(order_guard.buy_reference_enabled("kraken"))
            self.assertFalse(order_guard.buy_reference_enabled("binance"))

    def test_unreadable_value_keeps_the_conservative_guard(self):
        with mock.patch.object(order_guard, "_MARGINS", _margins(binance_buy_reference="maybe")):
            self.assertTrue(order_guard.buy_reference_enabled("binance"))

    def test_repository_config_sets_binance_buy_reference_dynamic(self):
        with mock.patch.object(order_guard, "_MARGINS", None):
            self.assertEqual(order_guard.buy_reference_mode("binance"), "dynamic")
            self.assertEqual(order_guard.buy_reference_mode("kraken"), "on")
            self.assertTrue(order_guard.buy_reference_enabled("binance"))
            self.assertTrue(order_guard.buy_reference_enabled("kraken"))

    def test_dynamic_mode_bypasses_in_bull_trend_and_enforces_in_bear_or_flat(self):
        with mock.patch.object(order_guard, "_MARGINS", _margins(binance_buy_reference="dynamic")):
            provider = _Provider("binance")
            # In confirmed BULL trend: BUY above old sell reference is permitted
            with mock.patch.object(order_guard, "_symbol_trend", return_value="bull"):
                self.assertTrue(order_guard.profit_guard(
                    provider, "TAOUSDC", "BUY", 293.0, 1.15, window_ref=216.28))
            # In BEAR trend: BUY above old sell reference is blocked for defense
            with mock.patch.object(order_guard, "_symbol_trend", return_value="bear"):
                self.assertFalse(order_guard.profit_guard(
                    provider, "TAOUSDC", "BUY", 293.0, 1.15, window_ref=216.28))
            # In FLAT/UNKNOWN trend: defensive guard is maintained
            with mock.patch.object(order_guard, "_symbol_trend", return_value="unknown"):
                self.assertFalse(order_guard.profit_guard(
                    provider, "TAOUSDC", "BUY", 293.0, 1.15, window_ref=216.28))

    def test_provider_as_string_and_case_insensitivity(self):
        with mock.patch.object(order_guard, "_MARGINS", _margins(binance_buy_reference=0.0)):
            # Both string name and Provider object with various casings must work
            self.assertTrue(order_guard.profit_guard("binance", "TAOUSDC", "BUY", 293.0, 1.15, window_ref=216.28))
            self.assertTrue(order_guard.profit_guard("Binance", "TAOUSDC", "BUY", 293.0, 1.15, window_ref=216.28))
            self.assertTrue(order_guard.profit_guard(_Provider("Binance"), "TAOUSDC", "BUY", 293.0, 1.15, window_ref=216.28))
            self.assertFalse(order_guard.buy_reference_enabled("Binance"))
            self.assertFalse(order_guard.buy_reference_enabled("binance"))
            self.assertEqual(order_guard.buy_reference_mode("Binance"), "off")

    def test_missing_default_buy_reference_in_margins_does_not_raise(self):
        # Even if a test mock or custom dict omits default_buy_reference, no KeyError is thrown
        with mock.patch.object(order_guard, "_MARGINS", {"binance": 1.15}):
            self.assertTrue(order_guard.buy_reference_enabled("binance"))
            self.assertTrue(order_guard.buy_reference_enabled("kraken"))

    def test_real_binance_provider_integration(self):
        with mock.patch.object(order_guard, "_MARGINS", None):
            from providers.market_api import BinanceProvider
            provider = BinanceProvider()
            self.assertEqual(order_guard.buy_reference_mode(provider.name), "dynamic")
            # In BULL trend, BUY above historical sell reference is permitted on Binance
            with mock.patch.object(order_guard, "_symbol_trend", return_value="bull"):
                self.assertTrue(order_guard.profit_guard(provider, "TAOUSDC", "BUY", 293.0, 1.15, window_ref=216.28))
            # SELL below historical buy reference is still blocked
            self.assertFalse(order_guard.profit_guard(provider, "TAOUSDC", "SELL", 270.0, 1.15, window_ref=276.0))

    def test_instrument_pipeline_respects_buy_reference_switch(self):
        from instrument import Instrument
        from providers.market_api import MarketApi, MarketDataProvider

        class _MockBinanceProvider(MarketDataProvider):
            def __init__(self):
                self._name = "binance"
                self.placed = []

            @property
            def name(self):
                return self._name

            def guards_internally(self):
                return False

            def get_current_price(self, symbol):
                return 293.0

            def free_balance(self, asset):
                return 10000.0

            def get_orders(self, symbol, side, since_s):
                return []

            def profit_guard_window_ref(self, symbol, side, safeback_override=None):
                return 216.28 if side == "BUY" else 276.0

            def supports_symbol(self, symbol):
                return True

            def order_status(self, symbol, order_id):
                return {"orderId": order_id, "status": "FILLED"}

            def place_order(self, symbol, side, price, qty, **kwargs):
                self.placed.append({"side": side, "qty": qty, "price": price})
                return {"orderId": "123", "status": "FILLED"}

        p = _MockBinanceProvider()
        inst = Instrument(name="TAO", symbol="TAOUSDC", provider="binance", base="TAO", quote="USDC", api=MarketApi([p]))

        # With binance_buy_reference = 0: BUY at 293 (above past sell 216.28) is accepted
        with mock.patch.object(order_guard, "_MARGINS", _margins(binance_buy_reference=0.0)):
            res = inst.place("BUY", 293.0, 1.0, motivation="test_buy", cache_permit=object(), smart=False, wait_for_trend=False)
            self.assertIsNotNone(res)
            self.assertEqual(len(p.placed), 1)

        # With binance_buy_reference = 1: BUY at 293 (above past sell 216.28) is refused by profit_guard
        p.placed.clear()
        with mock.patch.object(order_guard, "_MARGINS", _margins(binance_buy_reference=1.0)):
            res = inst.place("BUY", 293.0, 1.0, motivation="test_buy", cache_permit=object(), smart=False, wait_for_trend=False)
            self.assertIsNone(res)
            self.assertEqual(len(p.placed), 0)

    def test_dynamic_window_scaling_by_trend(self):
        # Bull: 8h (within 4h-12h), Flat: 24h (within 12h-48h), Bear: 72h (within 48h-168h)
        with mock.patch.object(order_guard, "_symbol_trend", return_value="bull"):
            self.assertEqual(order_guard.dynamic_buy_window_sec("TAOUSDC"), 8.0 * 3600.0)
            self.assertEqual(order_guard.window_for("binance", "TAOUSDC", "BUY"), 8.0 * 3600.0)

        with mock.patch.object(order_guard, "_symbol_trend", return_value="bear"):
            self.assertEqual(order_guard.dynamic_buy_window_sec("TAOUSDC"), 72.0 * 3600.0)
            self.assertEqual(order_guard.window_for("binance", "TAOUSDC", "BUY"), 72.0 * 3600.0)

        with mock.patch.object(order_guard, "_symbol_trend", return_value="flat"):
            self.assertEqual(order_guard.dynamic_buy_window_sec("TAOUSDC"), 24.0 * 3600.0)
            self.assertEqual(order_guard.window_for("binance", "TAOUSDC", "BUY"), 24.0 * 3600.0)

        with mock.patch.object(order_guard, "_symbol_trend", return_value="unknown"):
            self.assertEqual(order_guard.dynamic_buy_window_sec("TAOUSDC"), 24.0 * 3600.0)

    def test_dynamic_window_bounds_clamping(self):
        # Configured extreme values are safely clamped into user-defined bands
        with mock.patch.object(order_guard, "_MARGINS", _margins(buy_window_bull_h=1.0, buy_window_bear_h=500.0)):
            with mock.patch.object(order_guard, "_symbol_trend", return_value="bull"):
                self.assertEqual(order_guard.dynamic_buy_window_sec("TAOUSDC"), 4.0 * 3600.0)  # clamped to min 4h
            with mock.patch.object(order_guard, "_symbol_trend", return_value="bear"):
                self.assertEqual(order_guard.dynamic_buy_window_sec("TAOUSDC"), 168.0 * 3600.0)  # clamped to max 168h (7d)

    def test_dynamic_window_protects_against_recent_top_and_permits_pullback(self):
        class _OrderProvider:
            def __init__(self, orders):
                self.name = "binance"
                self._orders = orders

            def get_orders(self, symbol, side, since_s):
                return self._orders

            def last_opposite_fill(self, symbol, side):
                return self._orders[0]["price"] if self._orders else None

        # Recent sell at 290.0 within 8h bull window
        provider = _OrderProvider([{"price": 290.0, "qty": 1.0, "timestamp": 1000.0}])
        with mock.patch.object(order_guard, "_MARGINS", _margins(binance_buy_reference="dynamic")):
            with mock.patch.object(order_guard, "_symbol_trend", return_value="bull"):
                # BUY at 293.0 is higher than recent sell 290.0 -> BLOCKED (protects from buying top)
                self.assertFalse(order_guard.profit_guard(
                    provider, "TAOUSDC", "BUY", 293.0, 1.15, window_ref=290.0))
                # BUY at 285.0 is below recent sell 290.0 (+1.72% diff) -> ALLOWED (pullback re-entry)
                self.assertTrue(order_guard.profit_guard(
                    provider, "TAOUSDC", "BUY", 285.0, 1.15, window_ref=290.0))

    def test_dynamic_window_ancient_sell_expires_in_bull_trend(self):
        class _OrderProviderNoRecent:
            def __init__(self):
                self.name = "binance"

            def get_orders(self, symbol, side, since_s):
                # No fills in the last 8h
                return []

            def last_opposite_fill(self, symbol, side):
                # Ancient fill from 14 days ago
                return 216.28

        provider = _OrderProviderNoRecent()
        with mock.patch.object(order_guard, "_MARGINS", _margins(binance_buy_reference="dynamic")):
            with mock.patch.object(order_guard, "_symbol_trend", return_value="bull"):
                # window_ref is None because no orders in 8h -> permitted without 14-day lockup
                self.assertTrue(order_guard.profit_guard(
                    provider, "TAOUSDC", "BUY", 293.0, 1.15, window_ref=None))
                # Even if legacy caller passed ancient window_ref 216.28, get_orders shows 0 recent fills in 8h
                self.assertTrue(order_guard.profit_guard(
                    provider, "TAOUSDC", "BUY", 293.0, 1.15, window_ref=216.28))

    def test_dynamic_window_bear_market_holds_long_defensive_memory(self):
        class _OrderProviderBear:
            def __init__(self):
                self.name = "binance"

            def get_orders(self, symbol, side, since_s):
                # Sell from 48h ago (inside the 72h bear window)
                return [{"price": 300.0, "qty": 1.0, "timestamp": 1000.0}]

            def last_opposite_fill(self, symbol, side):
                return 300.0

        provider = _OrderProviderBear()
        with mock.patch.object(order_guard, "_MARGINS", _margins(binance_buy_reference="dynamic")):
            with mock.patch.object(order_guard, "_symbol_trend", return_value="bear"):
                # BUY at 298.0 against 300.0 sell is only 0.67% diff < 1.15% -> BLOCKED
                self.assertFalse(order_guard.profit_guard(
                    provider, "TAOUSDC", "BUY", 298.0, 1.15, window_ref=300.0))

    def test_sell_orders_never_use_short_dynamic_window(self):
        # On SELL, window_for never uses dynamic short window; returns full venue window (e.g. 336h kraken)
        with mock.patch.object(order_guard, "_symbol_trend", return_value="bull"):
            kraken_sell_win = order_guard.window_for("kraken", "HYPEUSD", "SELL")
            self.assertEqual(kraken_sell_win, 336.0 * 3600.0)

        # On SELL, profit guard NEVER allows selling below buy price + margin
        provider = _Provider("binance")
        self.assertFalse(order_guard.profit_guard(
            provider, "TAOUSDC", "SELL", 270.0, 1.15, window_ref=276.0))


if __name__ == "__main__":
    unittest.main()


"""Characterization and verification tests for order_guard consuming MarketRegimeService.

Validates that order_guard:
- Consumes the shared MarketRegimeDecision instead of duplicating parallel heuristics.
- Respects freshness bounds, timestamps, candle continuity, and provider fallback.
- Preserves the exact dynamic BUY lookback windows:
    * Bull: 4h - 12h (default 8h)
    * Bear: 48h - 168h (default 72h)
    * Sideways / Flat / Unknown: 12h - 48h (default 24h)
- Safely maps 'sideways' regime to 'flat' for backward compatibility.
"""
import os
import sys
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import order_guard
from market_regime import MarketRegimeDecision, MarketRegimeService, ClosedPriceSeries


class _MockVenueProvider:
    """Mock venue provider with optional OHLC capabilities."""

    def __init__(self, name="binance", closes=None, timestamps=None, last_closed_at=None):
        self.name = name
        self._closes = closes or ()
        self._timestamps = timestamps or ()
        self._last_closed_at = last_closed_at

    def ohlc_series(self, symbol, interval_min):
        return ClosedPriceSeries(
            closes=tuple(self._closes),
            interval_min=interval_min,
            last_closed_at=self._last_closed_at,
            timestamps=tuple(self._timestamps),
        )

    def last_opposite_fill(self, symbol, order_type):
        return 216.28


class OrderGuardMarketRegimeCharacterizationTest(unittest.TestCase):
    def setUp(self):
        self.now = time.time()
        order_guard._DEFAULT_REGIME_SERVICE._cache.clear()

    def test_fresh_bull_snapshot_produces_bull_regime_and_8h_window(self):
        snap = {
            "gradient_recent": 0.5,
            "epsilon": 0.1,
            "ts": self.now - 10.0,  # 10s old -> fresh
        }
        mock_mgr = mock.MagicMock()
        mock_mgr.fresh_snapshot.return_value = snap

        with mock.patch("cacheManager.get_short_trend_manager", return_value=mock_mgr):
            decision = order_guard.symbol_regime("BTCUSDT", now=self.now)
            self.assertEqual(decision.regime, "bull")
            self.assertTrue(decision.fresh)
            self.assertEqual(decision.source, "snapshot")

            trend = order_guard._symbol_trend("BTCUSDT", now=self.now)
            self.assertEqual(trend, "bull")

            window_s = order_guard.dynamic_buy_window_sec("BTCUSDT", now=self.now)
            self.assertEqual(window_s, 8.0 * 3600.0)

    def test_fresh_bear_snapshot_produces_bear_regime_and_72h_window(self):
        snap = {
            "gradient_recent": -0.6,
            "epsilon": 0.1,
            "ts": self.now - 10.0,
        }
        mock_mgr = mock.MagicMock()
        mock_mgr.fresh_snapshot.return_value = snap

        with mock.patch("cacheManager.get_short_trend_manager", return_value=mock_mgr):
            decision = order_guard.symbol_regime("BTCUSDT", now=self.now)
            self.assertEqual(decision.regime, "bear")
            self.assertTrue(decision.fresh)

            trend = order_guard._symbol_trend("BTCUSDT", now=self.now)
            self.assertEqual(trend, "bear")

            window_s = order_guard.dynamic_buy_window_sec("BTCUSDT", now=self.now)
            self.assertEqual(window_s, 72.0 * 3600.0)

    def test_weak_signal_produces_sideways_regime_mapped_to_flat_and_24h_window(self):
        # abs(gradient) / epsilon = 0.15 / 0.1 = 1.5 <= strength_threshold 2.0 -> sideways
        snap = {
            "gradient_recent": 0.15,
            "epsilon": 0.1,
            "ts": self.now - 10.0,
        }
        mock_mgr = mock.MagicMock()
        mock_mgr.fresh_snapshot.return_value = snap

        with mock.patch("cacheManager.get_short_trend_manager", return_value=mock_mgr):
            decision = order_guard.symbol_regime("BTCUSDT", now=self.now)
            self.assertEqual(decision.regime, "sideways")
            self.assertTrue(decision.fresh)

            trend = order_guard._symbol_trend("BTCUSDT", now=self.now)
            self.assertEqual(trend, "flat")

            window_s = order_guard.dynamic_buy_window_sec("BTCUSDT", now=self.now)
            self.assertEqual(window_s, 24.0 * 3600.0)

    def test_stale_snapshot_rejected_and_falls_back_to_provider_ohlc(self):
        # Snapshot is 300s old (> default max age 120s) -> stale
        stale_snap = {
            "gradient_recent": 0.5,
            "epsilon": 0.1,
            "ts": self.now - 300.0,
        }
        mock_mgr = mock.MagicMock()
        mock_mgr.fresh_snapshot.return_value = stale_snap

        # Provider provides 15 rising 1m closes up to self.now
        rising_closes = tuple(100.0 + i * 2.0 for i in range(20))
        timestamps = tuple(self.now - (19 - i) * 60.0 for i in range(20))
        provider = _MockVenueProvider(
            "binance",
            closes=rising_closes,
            timestamps=timestamps,
            last_closed_at=self.now,
        )

        with mock.patch("cacheManager.get_short_trend_manager", return_value=mock_mgr):
            # When fallback allowed: falls back to provider OHLC
            decision = order_guard.symbol_regime("BTCUSDT", provider=provider, now=self.now, allow_fallback=True)
            self.assertTrue(decision.fresh)
            self.assertEqual(decision.regime, "bull")
            self.assertTrue(decision.fallback_used)
            self.assertTrue(decision.source.startswith("ohlc:"))

            # Window uses the verified fallback bull regime
            window_s = order_guard.dynamic_buy_window_sec("BTCUSDT", provider=provider, now=self.now)
            self.assertEqual(window_s, 8.0 * 3600.0)

    def test_stale_snapshot_without_fallback_becomes_unknown_and_uses_24h_window(self):
        stale_snap = {
            "gradient_recent": 0.5,
            "epsilon": 0.1,
            "ts": self.now - 300.0,
        }
        mock_mgr = mock.MagicMock()
        mock_mgr.fresh_snapshot.return_value = stale_snap

        with mock.patch("cacheManager.get_short_trend_manager", return_value=mock_mgr):
            decision = order_guard.symbol_regime("BTCUSDT", provider=None, now=self.now, allow_fallback=False)
            self.assertFalse(decision.fresh)
            self.assertEqual(decision.regime, "unknown")

            trend = order_guard._symbol_trend("BTCUSDT", provider=None, now=self.now)
            self.assertEqual(trend, "unknown")

            window_s = order_guard.dynamic_buy_window_sec("BTCUSDT", provider=None, now=self.now)
            self.assertEqual(window_s, 24.0 * 3600.0)

    def test_ohlc_candle_gap_fails_closed_to_unknown(self):
        stale_snap = {
            "gradient_recent": 0.5,
            "epsilon": 0.1,
            "ts": self.now - 300.0,
        }
        mock_mgr = mock.MagicMock()
        mock_mgr.fresh_snapshot.return_value = stale_snap

        # Gap between candles (e.g. 500s interval on 1m candles)
        rising_closes = (100.0, 102.0, 104.0, 106.0)
        gap_timestamps = (self.now - 1000.0, self.now - 940.0, self.now - 400.0, self.now)
        provider = _MockVenueProvider(
            "binance",
            closes=rising_closes,
            timestamps=gap_timestamps,
            last_closed_at=self.now,
        )

        with mock.patch("cacheManager.get_short_trend_manager", return_value=mock_mgr):
            decision = order_guard.symbol_regime("BTCUSDT", provider=provider, now=self.now, allow_fallback=True)
            self.assertFalse(decision.fresh)
            self.assertEqual(decision.regime, "unknown")

    def test_legacy_growth_coefficient_snapshot_is_normalized(self):
        # Legacy snapshot containing only growth_coefficient instead of gradient_recent
        legacy_snap = {
            "growth_coefficient": 0.6,
            "epsilon": 0.1,
            "ts": self.now - 5.0,
        }
        mock_mgr = mock.MagicMock()
        mock_mgr.fresh_snapshot.return_value = legacy_snap

        with mock.patch("cacheManager.get_short_trend_manager", return_value=mock_mgr):
            decision = order_guard.symbol_regime("BTCUSDT", now=self.now)
            self.assertEqual(decision.regime, "bull")
            self.assertTrue(decision.fresh)
            self.assertEqual(decision.gradient, 0.6)

    def test_provider_with_direct_market_regime_is_delegated(self):
        class _DirectRegimeProvider:
            def market_regime(self, symbol, allow_fallback=True):
                return MarketRegimeDecision(
                    regime="bear",
                    gradient=-0.8,
                    epsilon=0.1,
                    strength=8.0,
                    fresh=True,
                    reason="custom_delegation",
                )

        provider = _DirectRegimeProvider()
        decision = order_guard.symbol_regime("BTCUSDT", provider=provider)
        self.assertEqual(decision.regime, "bear")
        self.assertEqual(decision.reason, "custom_delegation")
        self.assertEqual(order_guard._symbol_trend("BTCUSDT", provider=provider), "bear")
        self.assertEqual(order_guard.dynamic_buy_window_sec("BTCUSDT", provider=provider), 72.0 * 3600.0)

    def test_resolution_error_does_not_bypass_freshness_validation(self):
        class _BrokenService(MarketRegimeService):
            def resolve_with_evidence(self, *_args, **_kwargs):
                raise RuntimeError("resolution failed")

        snap = {
            "gradient_recent": 0.5,
            "epsilon": 0.1,
            "ts": self.now - 10.0,
        }
        mock_mgr = mock.MagicMock()
        mock_mgr.fresh_snapshot.return_value = snap

        with mock.patch(
            "cacheManager.get_short_trend_manager",
            return_value=mock_mgr,
        ):
            decision = order_guard.symbol_regime(
                "BTCUSDT", regime_service=_BrokenService(), now=self.now,
            )

        self.assertEqual(decision.regime, "unknown")
        self.assertFalse(decision.fresh)
        self.assertEqual(decision.reason, "regime_resolution_failed")

    def test_missing_symbol_or_empty_cache_fails_safely_to_unknown(self):
        decision = order_guard.symbol_regime("", now=self.now)
        self.assertEqual(decision.regime, "unknown")
        self.assertEqual(decision.reason, "missing_symbol")

        with mock.patch("cacheManager.get_short_trend_manager", side_effect=Exception("Cache error")):
            decision = order_guard.symbol_regime("BTCUSDT", provider=None, now=self.now, allow_fallback=False)
            self.assertEqual(decision.regime, "unknown")
            self.assertEqual(order_guard.dynamic_buy_window_sec("BTCUSDT", provider=None), 24.0 * 3600.0)

    def test_profit_guard_dynamic_mode_respects_bull_and_bear_regimes(self):
        bull_snap = {
            "gradient_recent": 0.5,
            "epsilon": 0.1,
            "ts": self.now - 10.0,
        }
        bear_snap = {
            "gradient_recent": -0.6,
            "epsilon": 0.1,
            "ts": self.now - 10.0,
        }

        class _MockExecutionProvider:
            def __init__(self, name="binance"):
                self.name = name

            def last_opposite_fill(self, symbol, order_type):
                return 216.28

        provider = _MockExecutionProvider()
        mock_mgr = mock.MagicMock()

        margins = {
            "default": 1.15,
            "binance": 1.15,
            "default_window_h": 0.0,
            "binance_buy_reference": "dynamic",
            "buy_window_mode": "dynamic",
        }

        with mock.patch.object(order_guard, "_MARGINS", margins):
            # In confirmed BULL trend: BUY at 293 above past sell 216.28 is allowed
            mock_mgr.fresh_snapshot.return_value = bull_snap
            with mock.patch("cacheManager.get_short_trend_manager", return_value=mock_mgr):
                allowed = order_guard.profit_guard(
                    provider, "TAOUSDC", "BUY", 293.0, 1.15, window_ref=216.28
                )
                self.assertTrue(allowed)

            # In BEAR trend: BUY at 293 above past sell 216.28 is BLOCKED for capital defense
            mock_mgr.fresh_snapshot.return_value = bear_snap
            with mock.patch("cacheManager.get_short_trend_manager", return_value=mock_mgr):
                allowed = order_guard.profit_guard(
                    provider, "TAOUSDC", "BUY", 293.0, 1.15, window_ref=216.28
                )
                self.assertFalse(allowed)

    def test_profit_guard_reuses_one_regime_for_window_and_policy(self):
        class _Provider:
            name = "binance"

            def get_orders(self, _symbol, _side, _window_s):
                return []

        margins = {
            "default": 1.15,
            "default_buy_reference": "dynamic",
            "buy_window_mode": "dynamic",
        }
        with mock.patch.object(order_guard, "_MARGINS", margins), \
                mock.patch.object(
                    order_guard,
                    "_symbol_trend",
                    side_effect=["bull", "bear"],
                ) as trend:
            allowed = order_guard.profit_guard(
                _Provider(),
                "TAOUSDC",
                "BUY",
                293.0,
                1.15,
                window_ref=216.28,
            )

        self.assertTrue(allowed)
        trend.assert_called_once_with("TAOUSDC", provider=mock.ANY)

    def test_injected_resolvers_isolate_order_guard_from_globals(self):
        called = {"snapshot": 0, "provider": 0}

        def mock_snapshot(symbol, now=None):
            called["snapshot"] += 1
            return {
                "gradient_recent": 0.45,
                "epsilon": 0.05,
                "ts": self.now - 5.0,
            }

        def mock_provider(provider):
            called["provider"] += 1
            return mock.MagicMock(name="mock_provider")

        decision = order_guard.symbol_regime(
            "BTCUSDT",
            provider="mock_venue",
            now=self.now,
            snapshot_resolver=mock_snapshot,
            provider_resolver=mock_provider,
        )
        self.assertEqual(decision.regime, "bull")
        self.assertEqual(called["snapshot"], 1)
        self.assertEqual(called["provider"], 1)

    def test_symbol_regime_context_bundles_decision_and_properties(self):
        mock_mgr = mock.MagicMock()
        mock_mgr.fresh_snapshot.return_value = {
            "gradient_recent": -0.6,
            "epsilon": 0.1,
            "ts": self.now - 10.0,
        }
        with mock.patch("cacheManager.get_short_trend_manager", return_value=mock_mgr):
            ctx = order_guard.symbol_regime_context("BTCUSDT", now=self.now)

        self.assertEqual(ctx.regime, "bear")
        self.assertEqual(ctx.resolved_trend, "bear")
        self.assertTrue(ctx.fresh)
        self.assertAlmostEqual(ctx.strength, 6.0)

    def test_profit_guard_uses_passed_regime_context_without_trend_lookup(self):
        class _Provider:
            name = "binance"

            def get_orders(self, _symbol, _side, _window_s):
                return []

        margins = {
            "default": 1.15,
            "default_buy_reference": "dynamic",
            "buy_window_mode": "dynamic",
        }
        mock_decision = order_guard.MarketRegimeDecision(
            regime="bull", gradient=0.5, epsilon=0.1, strength=5.0,
            fresh=True, reason="directional_signal",
        )
        ctx = order_guard.MarketRegimeContext.from_decision(mock_decision, evaluated_at=self.now)

        with mock.patch.object(order_guard, "_MARGINS", margins), \
                mock.patch.object(order_guard, "_symbol_trend") as mock_trend:
            allowed = order_guard.profit_guard(
                _Provider(),
                "TAOUSDC",
                "BUY",
                293.0,
                1.15,
                window_ref=216.28,
                regime_context=ctx,
            )

        self.assertTrue(allowed)
        mock_trend.assert_not_called()


if __name__ == "__main__":
    unittest.main()


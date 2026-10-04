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
import builtins
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

    @staticmethod
    def _snapshot_resolver(snapshot):
        """Return an explicit resolver for one deterministic snapshot."""
        return lambda _symbol, now=None: snapshot

    def test_fresh_bull_snapshot_produces_bull_regime_and_8h_window(self):
        snap = {
            "gradient_recent": 0.5,
            "epsilon": 0.1,
            "ts": self.now - 10.0,  # 10s old -> fresh
        }
        resolver = self._snapshot_resolver(snap)
        decision = order_guard.symbol_regime(
            "BTCUSDT", now=self.now, snapshot_resolver=resolver,
        )
        self.assertEqual(decision.regime, "bull")
        self.assertTrue(decision.fresh)
        self.assertEqual(decision.source, "snapshot")

        trend = order_guard._symbol_trend(
            "BTCUSDT", now=self.now, snapshot_resolver=resolver,
        )
        self.assertEqual(trend, "bull")

        window_s = order_guard.dynamic_buy_window_sec(
            "BTCUSDT", now=self.now, snapshot_resolver=resolver,
        )
        self.assertEqual(window_s, 8.0 * 3600.0)

    def test_fresh_bear_snapshot_produces_bear_regime_and_72h_window(self):
        snap = {
            "gradient_recent": -0.6,
            "epsilon": 0.1,
            "ts": self.now - 10.0,
        }
        resolver = self._snapshot_resolver(snap)
        decision = order_guard.symbol_regime(
            "BTCUSDT", now=self.now, snapshot_resolver=resolver,
        )
        self.assertEqual(decision.regime, "bear")
        self.assertTrue(decision.fresh)

        trend = order_guard._symbol_trend(
            "BTCUSDT", now=self.now, snapshot_resolver=resolver,
        )
        self.assertEqual(trend, "bear")

        window_s = order_guard.dynamic_buy_window_sec(
            "BTCUSDT", now=self.now, snapshot_resolver=resolver,
        )
        self.assertEqual(window_s, 72.0 * 3600.0)

    def test_weak_signal_produces_sideways_regime_mapped_to_flat_and_24h_window(self):
        # abs(gradient) / epsilon = 0.15 / 0.1 = 1.5 <= strength_threshold 2.0 -> sideways
        snap = {
            "gradient_recent": 0.15,
            "epsilon": 0.1,
            "ts": self.now - 10.0,
        }
        resolver = self._snapshot_resolver(snap)
        decision = order_guard.symbol_regime(
            "BTCUSDT", now=self.now, snapshot_resolver=resolver,
        )
        self.assertEqual(decision.regime, "sideways")
        self.assertTrue(decision.fresh)

        trend = order_guard._symbol_trend(
            "BTCUSDT", now=self.now, snapshot_resolver=resolver,
        )
        self.assertEqual(trend, "flat")

        window_s = order_guard.dynamic_buy_window_sec(
            "BTCUSDT", now=self.now, snapshot_resolver=resolver,
        )
        self.assertEqual(window_s, 24.0 * 3600.0)

    def test_stale_snapshot_rejected_and_falls_back_to_provider_ohlc(self):
        # Snapshot is 300s old (> default max age 120s) -> stale
        stale_snap = {
            "gradient_recent": 0.5,
            "epsilon": 0.1,
            "ts": self.now - 300.0,
        }
        resolver = self._snapshot_resolver(stale_snap)

        # Provider provides 15 rising 1m closes up to self.now
        rising_closes = tuple(100.0 + i * 2.0 for i in range(20))
        timestamps = tuple(self.now - (19 - i) * 60.0 for i in range(20))
        provider = _MockVenueProvider(
            "binance",
            closes=rising_closes,
            timestamps=timestamps,
            last_closed_at=self.now,
        )

        # When fallback is allowed, the stale snapshot falls back to provider OHLC.
        decision = order_guard.symbol_regime(
            "BTCUSDT", provider=provider, now=self.now, allow_fallback=True,
            snapshot_resolver=resolver,
        )
        self.assertTrue(decision.fresh)
        self.assertEqual(decision.regime, "bull")
        self.assertTrue(decision.fallback_used)
        self.assertTrue(decision.source.startswith("ohlc:"))

        # The window uses the verified fallback bull regime.
        window_s = order_guard.dynamic_buy_window_sec(
            "BTCUSDT", provider=provider, now=self.now,
            snapshot_resolver=resolver,
        )
        self.assertEqual(window_s, 8.0 * 3600.0)

    def test_stale_snapshot_without_fallback_becomes_unknown_and_uses_24h_window(self):
        stale_snap = {
            "gradient_recent": 0.5,
            "epsilon": 0.1,
            "ts": self.now - 300.0,
        }
        resolver = self._snapshot_resolver(stale_snap)
        decision = order_guard.symbol_regime(
            "BTCUSDT", provider=None, now=self.now, allow_fallback=False,
            snapshot_resolver=resolver,
        )
        self.assertFalse(decision.fresh)
        self.assertEqual(decision.regime, "unknown")

        trend = order_guard._symbol_trend(
            "BTCUSDT", provider=None, now=self.now,
            snapshot_resolver=resolver,
        )
        self.assertEqual(trend, "unknown")

        window_s = order_guard.dynamic_buy_window_sec(
            "BTCUSDT", provider=None, now=self.now,
            snapshot_resolver=resolver,
        )
        self.assertEqual(window_s, 24.0 * 3600.0)

    def test_ohlc_candle_gap_fails_closed_to_unknown(self):
        stale_snap = {
            "gradient_recent": 0.5,
            "epsilon": 0.1,
            "ts": self.now - 300.0,
        }
        resolver = self._snapshot_resolver(stale_snap)

        # Gap between candles (e.g. 500s interval on 1m candles)
        rising_closes = (100.0, 102.0, 104.0, 106.0)
        gap_timestamps = (self.now - 1000.0, self.now - 940.0, self.now - 400.0, self.now)
        provider = _MockVenueProvider(
            "binance",
            closes=rising_closes,
            timestamps=gap_timestamps,
            last_closed_at=self.now,
        )

        decision = order_guard.symbol_regime(
            "BTCUSDT", provider=provider, now=self.now, allow_fallback=True,
            snapshot_resolver=resolver,
        )
        self.assertFalse(decision.fresh)
        self.assertEqual(decision.regime, "unknown")

    def test_legacy_growth_coefficient_snapshot_is_normalized(self):
        # Legacy snapshot containing only growth_coefficient instead of gradient_recent
        legacy_snap = {
            "growth_coefficient": 0.6,
            "epsilon": 0.1,
            "ts": self.now - 5.0,
        }
        decision = order_guard.symbol_regime(
            "BTCUSDT", now=self.now,
            snapshot_resolver=self._snapshot_resolver(legacy_snap),
        )
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
        decision = order_guard.symbol_regime(
            "BTCUSDT", regime_service=_BrokenService(), now=self.now,
            snapshot_resolver=self._snapshot_resolver(snap),
        )

        self.assertEqual(decision.regime, "unknown")
        self.assertFalse(decision.fresh)
        self.assertEqual(decision.reason, "regime_resolution_failed")

    def test_missing_symbol_or_missing_sources_fails_safely_without_hidden_imports(self):
        decision = order_guard.symbol_regime("", now=self.now)
        self.assertEqual(decision.regime, "unknown")
        self.assertEqual(decision.reason, "missing_symbol")

        real_import = builtins.__import__

        def reject_hidden_market_sources(name, *args, **kwargs):
            if name in {"cacheManager", "market_api", "providers.market_api"}:
                raise AssertionError(f"hidden market source imported: {name}")
            return real_import(name, *args, **kwargs)

        def failing_snapshot_resolver(_symbol, now=None):
            raise RuntimeError("snapshot unavailable")

        cases = (
            ("no_sources", None),
            ("snapshot_error", failing_snapshot_resolver),
        )
        with mock.patch("builtins.__import__", side_effect=reject_hidden_market_sources):
            for case, resolver in cases:
                with self.subTest(case=case):
                    decision = order_guard.symbol_regime(
                        "BTCUSDT",
                        provider=None,
                        now=self.now,
                        allow_fallback=False,
                        snapshot_resolver=resolver,
                    )
                    self.assertEqual(decision.regime, "unknown")
            window_s = order_guard.dynamic_buy_window_sec(
                "BTCUSDT", provider=None, now=self.now,
            )

        self.assertFalse(hasattr(order_guard, "cacheManager"))
        self.assertFalse(hasattr(order_guard, "MarketApi"))
        self.assertEqual(window_s, 24.0 * 3600.0)

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

        margins = {
            "default": 1.15,
            "binance": 1.15,
            "default_window_h": 0.0,
            "binance_buy_reference": "dynamic",
            "buy_window_mode": "dynamic",
        }

        cases = (
            ("bull", bull_snap, True),
            ("bear", bear_snap, False),
        )
        with mock.patch.object(order_guard, "_MARGINS", margins):
            for regime, snapshot, expected in cases:
                with self.subTest(regime=regime):
                    context = order_guard.symbol_regime_context(
                        "TAOUSDC",
                        provider=provider,
                        now=self.now,
                        snapshot_resolver=self._snapshot_resolver(snapshot),
                    )
                    allowed = order_guard.profit_guard(
                        provider,
                        "TAOUSDC",
                        "BUY",
                        293.0,
                        1.15,
                        window_ref=216.28,
                        regime_context=context,
                    )
                    self.assertEqual(allowed, expected)

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
        calls = 0

        def snapshot_resolver(_symbol, now=None):
            nonlocal calls
            calls += 1
            return {
                "gradient_recent": 0.5,
                "epsilon": 0.1,
                "ts": self.now - 10.0,
            }

        provider = _Provider()
        context = order_guard.symbol_regime_context(
            "TAOUSDC",
            provider=provider,
            now=self.now,
            snapshot_resolver=snapshot_resolver,
        )
        with mock.patch.object(order_guard, "_MARGINS", margins), \
                mock.patch.object(order_guard, "_symbol_trend") as trend:
            window_s = order_guard.window_for(
                provider, "TAOUSDC", "BUY", regime_context=context,
            )
            allowed = order_guard.profit_guard(
                provider,
                "TAOUSDC",
                "BUY",
                293.0,
                1.15,
                window_ref=216.28,
                regime_context=context,
            )

        self.assertEqual(calls, 1)
        self.assertEqual(window_s, 8.0 * 3600.0)
        self.assertTrue(allowed)
        trend.assert_not_called()

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
        snapshot = {
            "gradient_recent": -0.6,
            "epsilon": 0.1,
            "ts": self.now - 10.0,
        }
        ctx = order_guard.symbol_regime_context(
            "BTCUSDT",
            now=self.now,
            snapshot_resolver=self._snapshot_resolver(snapshot),
        )

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

    def test_intelligence_guards_shadow_and_enforce_modes(self):
        # 1. In shadow mode, an exhausted trend or surge still allows the order
        margins_shadow = {
            "default": 1.15,
            "intelligence_guards_mode": "shadow",
            "weibull_exhaustion_policy": "veto",
        }
        with mock.patch.object(order_guard, "_MARGINS", margins_shadow):
            # Trend duration 15 days (> P90)
            ok, reason, scale = order_guard.check_intelligence_guards(
                "binance", "BTCUSDC", "BUY", 85000.0, trend_duration_seconds=15 * 86400
            )
            self.assertTrue(ok)
            self.assertEqual(scale, 1.0)

        # 2. In enforce mode with veto policy, exhausted trend blocks the order
        margins_enforce_veto = {
            "default": 1.15,
            "intelligence_guards_mode": "enforce",
            "weibull_exhaustion_policy": "veto",
        }
        with mock.patch.object(order_guard, "_MARGINS", margins_enforce_veto):
            ok, reason, scale = order_guard.check_intelligence_guards(
                "binance", "BTCUSDC", "BUY", 85000.0, trend_duration_seconds=15 * 86400
            )
            self.assertFalse(ok)
            self.assertIn("trend_exhausted", reason)

        # 3. In enforce mode with downscale policy, exhausted trend suggests scale
        margins_enforce_scale = {
            "default": 1.15,
            "intelligence_guards_mode": "enforce",
            "weibull_exhaustion_policy": "downscale",
            "weibull_exhausted_scale": 0.25,
        }
        with mock.patch.object(order_guard, "_MARGINS", margins_enforce_scale):
            ok, reason, scale = order_guard.check_intelligence_guards(
                "binance", "BTCUSDC", "BUY", 85000.0, trend_duration_seconds=15 * 86400
            )
            self.assertTrue(ok)
            self.assertEqual(scale, 0.25)
            self.assertIn("trend_exhausted", reason)

    def test_market_regime_context_validation_and_benchmarks(self):
        dec = order_guard.MarketRegimeDecision(
            regime="bull", gradient=0.5, epsilon=0.1, strength=5.0,
            fresh=True, reason="signal",
        )
        # 1. Matching symbol and provider
        ctx = order_guard.MarketRegimeContext.from_decision(
            dec,
            evaluated_at=self.now,
            symbol="ETHUSDT",
            provider="binance",
            trend_duration_seconds=3600.0,
            benchmark_symbol="BTCUSDT",
        )
        self.assertEqual(ctx.symbol, "ETHUSDT")
        self.assertEqual(ctx.provider, "binance")
        self.assertEqual(ctx.trend_duration_seconds, 3600.0)
        self.assertEqual(ctx.benchmark_symbol, "BTCUSDT")

        # Valid for matching symbol
        self.assertTrue(ctx.is_valid_for("ETHUSDT", provider="binance", max_age_seconds=60.0, now=self.now + 10.0))
        # Case insensitive
        self.assertTrue(ctx.is_valid_for("ethusdt", provider="BINANCE", max_age_seconds=60.0, now=self.now + 10.0))
        # Invalid for mismatched symbol
        self.assertFalse(ctx.is_valid_for("SOLUSDT", provider="binance", max_age_seconds=60.0, now=self.now + 10.0))
        # Invalid for mismatched provider
        self.assertFalse(ctx.is_valid_for("ETHUSDT", provider="kraken", max_age_seconds=60.0, now=self.now + 10.0))
        # Invalid when stale (> max_age_seconds)
        self.assertFalse(ctx.is_valid_for("ETHUSDT", provider="binance", max_age_seconds=60.0, now=self.now + 65.0))

        # 2. Generic context (symbol=None, provider=None) backwards compatible
        generic_ctx = order_guard.MarketRegimeContext.from_decision(dec, evaluated_at=self.now)
        self.assertTrue(generic_ctx.is_valid_for("ANY_SYMBOL", provider="any_provider", max_age_seconds=60.0, now=self.now + 10.0))
        self.assertFalse(generic_ctx.is_valid_for("ANY_SYMBOL", max_age_seconds=60.0, now=self.now + 65.0))

    def test_profit_guard_rejects_mismatched_or_stale_context(self):
        class _Provider:
            name = "binance"
            def get_orders(self, _symbol, _side, _window_s):
                return []

        provider = _Provider()
        margins = {
            "default": 1.15,
            "default_buy_reference": "dynamic",
            "buy_window_mode": "dynamic",
            "regime_context_max_age_sec": 60.0,
        }
        dec = order_guard.MarketRegimeDecision(
            regime="bull", gradient=0.5, epsilon=0.1, strength=5.0,
            fresh=True, reason="signal",
        )
        # Context built for BTCUSDC, but order placed for TAOUSDC
        btc_ctx = order_guard.MarketRegimeContext.from_decision(
            dec, evaluated_at=self.now, symbol="BTCUSDC", provider="binance"
        )

        with mock.patch.object(order_guard, "_MARGINS", margins), \
                mock.patch.object(order_guard, "_symbol_trend", return_value="bear") as mock_trend:
            allowed = order_guard.profit_guard(
                provider,
                "TAOUSDC",
                "BUY",
                293.0,
                1.15,
                window_ref=216.28,
                regime_context=btc_ctx,
            )
            # Mismatch should bypass btc_ctx and call _symbol_trend for TAOUSDC!
            mock_trend.assert_called_with("TAOUSDC", provider=provider)
            self.assertTrue(allowed)

    def test_check_intelligence_guards_does_not_reuse_cached_decision_across_symbols(self):
        dec = order_guard.MarketRegimeDecision(
            regime="bull", gradient=0.5, epsilon=0.1, strength=5.0,
            fresh=True, reason="signal",
        )
        ctx = order_guard.MarketRegimeContext.from_decision(
            dec, evaluated_at=self.now, symbol="BTCUSDT", provider="binance"
        )
        # Pre-seed memoized decision for BTCUSDT
        object.__setattr__(ctx, "_intelligence_decision", (False, "btc_blocked", 0.0))

        # Check for BTCUSDT: reuses cached memoized decision
        ok_btc, reason_btc, _ = order_guard.check_intelligence_guards(
            "binance", "BTCUSDT", "BUY", 80000.0, regime_context=ctx
        )
        self.assertFalse(ok_btc)
        self.assertEqual(reason_btc, "btc_blocked")

        # Check for ETHUSDT: mismatched symbol MUST NOT reuse BTC's memoized decision
        ok_eth, reason_eth, _ = order_guard.check_intelligence_guards(
            "binance", "ETHUSDT", "BUY", 3000.0, regime_context=ctx
        )
        self.assertTrue(ok_eth)
        self.assertEqual(reason_eth, "ok")


if __name__ == "__main__":
    unittest.main()



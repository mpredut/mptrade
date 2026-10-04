import unittest
import time
from unittest.mock import Mock, patch

from market_regime import ClosedPriceSeries, MarketRegimeService
from providers.base import MarketDataProvider
from providers.market_api import MarketApi
from providers.strategy_executor import (
    OrderReconciliationCapabilities,
    OrderStatus,
    ProviderError,
)


class FakeProvider(MarketDataProvider):
    name = "Fake"

    def supports_symbol(self, symbol):
        return symbol == "ABCUSD"

    def get_current_price(self, symbol):
        return 10.0

    def reconciliation_capabilities(self):
        return OrderReconciliationCapabilities(
            status_by_order_id=True,
            cancel_by_order_id=True,
        )

    def order_status(self, symbol, order_id):
        return OrderStatus("closed", 2.0, 20.0, 0.02)

    def cancel_order(self, symbol, order_id):
        self.canceled = (symbol, order_id)

    def preflight_order(self, symbol, side, qty, price=None, *, market=False,
                        kind=None):
        self.preflighted = (symbol, side, qty, price, market, kind)

    def order_filter_refusal(
            self, symbol, side, price, qty, *, market=False,
            enforce_business_minimum=True):
        self.filter_checked = (
            symbol, side, price, qty, market, enforce_business_minimum)
        return getattr(self, "filter_reason", None)

    def get_trades(self, symbol, since_s):
        return [
            {"side": "BUY", "price": 9, "qty": 2, "timestamp": 1},
            {"side": "SELL", "price": 11, "qty": 2,
             "timestamp": time.time() * 1000},
        ]

    def ohlc_closes(self, symbol, interval_min):
        return list(range(10, 50))

    def ohlc_series(self, symbol, interval_min):
        closes = tuple(self.ohlc_closes(symbol, interval_min))
        observed_at = getattr(self, "series_now", time.time())
        step = int(interval_min) * 60
        timestamps = tuple(
            observed_at - (len(closes) - index - 1) * step
            for index in range(len(closes))
        )
        return ClosedPriceSeries(
            closes, int(interval_min), observed_at, timestamps)


class MarketApiLifecycleTest(unittest.TestCase):
    def setUp(self):
        self.provider = FakeProvider()
        self.api = MarketApi([self.provider])

    def test_status_properties_are_provider_neutral(self):
        status = self.api.order_status("ABCUSD", "7")
        self.assertTrue(status.terminal)
        self.assertTrue(status.fully_filled)
        self.assertTrue(status.has_fill)
        self.assertFalse(status.partially_filled)

        partial_cancel = OrderStatus("canceled", 0.5, 5.0, 0.01)
        self.assertTrue(partial_cancel.terminal)
        self.assertFalse(partial_cancel.fully_filled)
        self.assertTrue(partial_cancel.partially_filled)

        normalized = OrderStatus("open", "0.5", "5", "0.01")
        self.assertEqual(normalized.filled_qty, 0.5)
        self.assertTrue(normalized.has_fill)
        rebate = OrderStatus("closed", 1, 10, -0.01)
        self.assertEqual(rebate.fee, -0.01)

    def test_cancel_and_latest_fill_route_through_common_facade(self):
        self.api.cancel_order("ABCUSD", "8")
        self.assertEqual(self.provider.canceled, ("ABCUSD", "8"))
        self.assertEqual(
            self.api.latest_fill_price("ABCUSD", "SELL", 60), 11.0)

    def test_preflight_routes_through_common_facade(self):
        self.api.preflight_order(
            "ABCUSD", "SELL", 2.0, 11.0,
            market=False, kind="replacement")
        self.assertEqual(
            self.provider.preflighted,
            ("ABCUSD", "SELL", 2.0, 11.0, False, "replacement"))

    def test_order_filter_refusal_honors_explicit_provider(self):
        second = FakeProvider()
        second.name = "Second"
        second.filter_reason = "below_min_notional"
        api = MarketApi([self.provider, second])

        refusal = api.order_filter_refusal(
            "ABCUSD",
            "SELL",
            11.0,
            2.0,
            market=False,
            enforce_business_minimum=True,
            provider_name="Second",
        )

        self.assertEqual(refusal, "below_min_notional")
        self.assertFalse(hasattr(self.provider, "filter_checked"))
        self.assertEqual(
            second.filter_checked,
            ("ABCUSD", "SELL", 11.0, 2.0, False, True),
        )

    def test_place_honors_explicit_provider_for_overlapping_symbol(self):
        second = FakeProvider()
        second.name = "Second"
        api = MarketApi([self.provider, second])
        with patch(
                "instrument.Instrument.place",
                lambda instrument, *args, **kwargs: instrument.provider_name):
            selected = api.place(
                "ABCUSD", "BUY", 10.0, 2.0,
                provider_name="Second")

        self.assertEqual(selected, "Second")

    def test_current_price_explicit_provider_does_not_pollute_implicit_route(self):
        implicit = FakeProvider()
        implicit.name = "Binance"
        explicit = FakeProvider()
        explicit.name = "Kraken"
        explicit.get_current_price = lambda symbol: 85.49
        api = MarketApi([implicit, explicit])

        self.assertEqual(
            api.get_current_price("HYPEUSD", provider_name="Kraken"),
            85.49,
        )
        self.assertEqual(api.provider_name_for("HYPEUSD"), "Binance")

    def test_unsupported_reconciliation_operations_fail_closed(self):
        capabilities = self.api.reconciliation_capabilities("ABCUSD")
        self.assertTrue(capabilities.status_by_order_id)
        self.assertTrue(capabilities.cancel_by_order_id)
        self.assertFalse(capabilities.lookup_by_client_order_id)
        self.assertFalse(capabilities.list_open_orders)

        with self.assertRaisesRegex(ProviderError, "order_by_client_id is unsupported"):
            self.api.order_by_client_id("ABCUSD", "CID-1")
        with self.assertRaisesRegex(ProviderError, "open_orders is unsupported"):
            self.api.open_orders("ABCUSD")

    def test_tracked_lifecycle_factory_keeps_place_synchronous_contract_separate(self):
        lifecycle = self.api.tracked_order_lifecycle(
            provider_name="Fake", max_age_seconds=60,
            retry_on_lookup_error=True)
        self.assertIs(lifecycle.market_api, self.api)
        self.assertEqual(lifecycle.provider_name, "Fake")
        self.assertEqual(lifecycle.max_age_seconds, 60)
        self.assertTrue(lifecycle.retry_on_lookup_error)

    def test_context_uses_injected_snapshot_once_and_preserves_freshness(self):
        now = 10_000_000.0
        snapshot = {"gradient_recent": -0.6, "epsilon": 0.1, "ts": now - 5.0}
        resolver = Mock(return_value=snapshot)
        api = MarketApi([self.provider], regime_snapshot_resolver=resolver)
        with patch.object(self.provider, "ohlc_series") as ohlc:
            context = api.market_regime_context("ABCUSD", now=now)
        self.assertEqual(context.regime, "bear")
        self.assertTrue(context.fresh)
        self.assertEqual(context.resolved_trend, "bear")
        self.assertEqual(context.evaluated_at, now)
        resolver.assert_called_once_with("ABCUSD", now=now)
        ohlc.assert_not_called()

        for label, timestamp in (("stale", now - 300.0), ("future", now + 60.0),
                                 ("missing", None)):
            with self.subTest(label):
                resolver.return_value = dict(snapshot, ts=timestamp)
                context = api.market_regime_context(
                    "ABCUSD", now=now, allow_fallback=False)
                self.assertFalse(context.fresh)
                self.assertEqual(context.regime, "unknown")

    def test_context_snapshot_controls_and_missing_source_fallback(self):
        now = 10_000_000.0
        self.provider.series_now = now
        snapshot = {"gradient_recent": -0.6, "epsilon": 0.1, "ts": now}
        resolver = Mock(side_effect=RuntimeError("snapshot unavailable"))
        api = MarketApi([self.provider], regime_snapshot_resolver=resolver)
        with patch.object(self.provider, "ohlc_series",
                          wraps=self.provider.ohlc_series) as ohlc:
            context = api.market_regime_context("ABCUSD", now=now)
        self.assertEqual(context.regime, "bull")
        resolver.assert_called_once_with("ABCUSD", now=now)
        ohlc.assert_called_once_with("ABCUSD", 1)

        for label, options in (
            ("explicit_snapshot", {"snapshot": snapshot}),
            ("disabled_runtime_snapshot", {"use_runtime_snapshot": False}),
        ):
            with self.subTest(label):
                resolver.reset_mock()
                context = api.market_regime_context("ABCUSD", now=now, **options)
                self.assertEqual(context.regime,
                                 "bear" if "snapshot" in options else "bull")
                resolver.assert_not_called()

    def test_context_routes_fallback_to_explicit_venue_and_injected_service(self):
        now = 10_000_000.0
        second = FakeProvider()
        second.name = "Second"
        second.series_now = now
        second.ohlc_closes = lambda symbol, interval: list(range(50, 10, -1))
        service = MarketRegimeService()
        api = MarketApi([self.provider, second], regime_service=service)
        with patch.object(self.provider, "ohlc_series") as wrong_venue, \
                patch.object(service, "resolve_with_evidence",
                             wraps=service.resolve_with_evidence) as resolve, \
                patch("providers.market_api._runtime_market_regime_snapshot") as hidden:
            context = api.market_regime_context(
                "ABCUSD", provider_name="Second", now=now)
        self.assertEqual(context.regime, "bear")
        self.assertTrue(context.fresh)
        self.assertEqual(context.source, "ohlc:1m")
        self.assertIs(resolve.call_args.args[0], second)
        resolve.assert_called_once()
        wrong_venue.assert_not_called()
        hidden.assert_not_called()

    def test_runtime_snapshot_adapter_is_only_wired_at_application_boundary(self):
        from providers.market_api import api, _runtime_market_regime_snapshot

        self.assertIs(api._regime_snapshot_resolver, _runtime_market_regime_snapshot)
        manager = Mock()
        snapshot = {"gradient_recent": 0.6, "epsilon": 0.1, "ts": 1000.0}
        manager.fresh_snapshot.return_value = snapshot
        with patch("cacheManager.get_short_trend_manager", return_value=manager):
            self.assertIs(_runtime_market_regime_snapshot("ABCUSD", now=1001.0),
                          snapshot)
        manager.fresh_snapshot.assert_called_once_with("ABCUSD", now=1001.0)

    def test_market_regime_routes_to_provider_ohlc(self):
        decision = self.api.market_regime(
            "ABCUSD", interval_min=1, window_seconds=300)
        self.assertEqual(decision.regime, "bull")
        self.assertEqual(decision.horizon, "short")
        self.assertEqual(decision.source, "ohlc:1m")

    def test_plain_market_regime_can_reject_stale_ohlc(self):
        self.provider.series_now = time.time() - 120
        decision = self.api.market_regime(
            "ABCUSD",
            allow_fallback=False,
            ohlc_max_age_seconds={1: 30},
        )

        self.assertFalse(decision.fresh)
        self.assertIn("stale_source", decision.reason)

    def test_composite_market_regime_supports_multiple_crypto_benchmarks(self):
        decision = self.api.composite_market_regime(
            "ABCUSD", benchmarks=("BTCUSD", "ETHUSD"))
        self.assertTrue(decision.actionable)
        self.assertEqual(decision.regime, "bull")
        self.assertEqual(len(decision.components), 6)

    def test_benchmarks_route_independently_and_single_string_is_normalized(self):
        asset = FakeProvider()
        asset.name = "Asset"
        benchmark = FakeProvider()
        benchmark.name = "Benchmark"
        benchmark.supports_symbol = lambda symbol: symbol == "BTCUSD"
        api = MarketApi([asset, benchmark])

        bundle = api.market_regime_bundle("ABCUSD", benchmarks="BTCUSD")

        self.assertEqual(
            {item.provider for item in bundle.asset_short.evidence},
            {"Asset"},
        )
        benchmark_evidence = bundle.benchmarks[0][1].evidence
        self.assertEqual(
            {item.provider for item in benchmark_evidence},
            {"Benchmark"},
        )
        self.assertEqual(len(bundle.composite.components), 4)

    def test_asset_does_not_reinforce_itself_as_a_benchmark(self):
        bundle = self.api.market_regime_bundle(
            "ABCUSD", benchmarks=("abcusd",)
        )

        self.assertEqual(len(bundle.composite.components), 2)

    def test_timestamp_unknown_bundle_is_not_actionable(self):
        provider = FakeProvider()
        provider.ohlc_series = None
        api = MarketApi([provider])
        bundle = api.market_regime_bundle("ABCUSD")

        self.assertFalse(bundle.composite.actionable)
        self.assertIsNone(bundle.asset_short.primary)
        self.assertEqual(
            bundle.asset_short.evidence[0].temporal_state,
            "unknown",
        )
        self.assertTrue(api.market_regime("ABCUSD").fresh)

    def test_regime_bundle_alternates_are_observational_only(self):
        def build_api():
            provider = FakeProvider()
            provider.ohlc_calls = []
            provider.series_now = 10_000_000

            def closes(_symbol, interval):
                provider.ohlc_calls.append(interval)
                return list(range(10, 50))

            provider.ohlc_closes = closes
            return provider, MarketApi([provider])

        snapshot = {
            "gradient_recent": 0.6,
            "epsilon": 0.1,
            "ts": 10_000_000,
        }
        primary_provider, primary_api = build_api()
        primary = primary_api.market_regime_bundle(
            "ABCUSD",
            snapshot=snapshot,
            snapshot_max_age_seconds=30,
            now=10_000_000,
        )
        alternate_provider, alternate_api = build_api()
        alternate = alternate_api.market_regime_bundle(
            "ABCUSD",
            snapshot=snapshot,
            snapshot_max_age_seconds=30,
            include_alternates=True,
            now=10_000_000,
        )

        self.assertEqual(primary_provider.ohlc_calls, [240])
        self.assertEqual(
            alternate_provider.ohlc_calls,
            [1, 5, 240, 1440],
        )
        self.assertEqual(primary.composite, alternate.composite)
        self.assertEqual(len(primary.evidence), 2)
        self.assertEqual(len(alternate.evidence), 5)
        self.assertEqual(len(alternate.composite.components), 2)

    def test_latest_fill_rejects_nonfinite_inputs(self):
        with self.assertRaises(ValueError):
            self.api.latest_fill_price("ABCUSD", "BUY", float("nan"))
        with self.assertRaises(ValueError):
            self.api.latest_fill_price(
                "ABCUSD", "BUY", 60, min_notional=20, max_notional=10)

    def test_invalid_provider_status_is_normalized_to_provider_error(self):
        self.provider.order_status = lambda _symbol, _order_id: OrderStatus(
            "venue-specific", 0, 0, 0)
        with self.assertRaises(ProviderError):
            self.api.order_status("ABCUSD", "9")


if __name__ == "__main__":
    unittest.main()

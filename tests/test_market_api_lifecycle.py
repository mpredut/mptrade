import ast
import botcore
import importlib
import market_regime
import order_guard
import pytest
import time
import unittest
from cachepaths import (
    cache_relpath,
    crypto_quote_priority_rank,
    ensure_quote_suffix,
    has_crypto_quote_suffix,
    is_cachedb_relpath,
    is_eur_pair,
    is_fiat_quote_pair,
    is_fiat_usd_pair,
    is_primary_stream_venue,
    is_readonly_xstock_symbol,
    is_usd_compatible_pair,
    is_usd_stablecoin_pair,
    is_usdc_pair,
    matches_asset_scope,
    resolve_kraken_lookup_symbol,
    same_canonical_base_asset,
    strip_equity_suffix,
    symbol_quote_asset,
    uses_cycle_state_files,
)
from collections import defaultdict
from market_regime import ClosedPriceSeries, MarketRegimeService
from notify_engine.ntfydispatcher import _resolve_provider_label
from offline.backtests import datasets as ds, evaluation as ev, execution as ex, metrics as mt
from pathlib import Path
from providers import market_api as ma, ohlc_utils, order_ohlc_utils, strategy_executor as sexec
from providers.base import ClosedPriceSeries as BaseClosedPriceSeries, MarketDataProvider
from providers.binance_provider import BinanceProvider
from providers.hyperliquid_provider import HyperliquidProvider
from providers.intent_execution import resolve_intent_topic_tag
from providers.kraken_provider import KrakenProvider
from providers.market_api import MarketApi, _runtime_market_regime_snapshot, api
from providers.ohlc_utils import (
    ClosedPriceSeries as OhlcClosedPriceSeries,
    candles_to_historical_points,
    extract_trading_quote_pairs,
    extract_yahoo_chart_latest_price,
    extract_yahoo_chart_result_and_quote,
    series_to_historical_points,
)
from providers.broker_history import (
    extract_broker_order_ticker,
    normalize_broker_history_items,
)
from providers.order_ohlc_utils import parse_nonneg_order_field
from providers.ssot_bridge import ProviderSsotBridgeMixin
from providers.venue_metadata import ProviderVenueMetadataMixin
from providers import ws_stream
from providers.replay_provider import ReplayMarketDataProvider
from providers.strategy_executor import (
    OrderReconciliationCapabilities,
    OrderStatus,
    ProviderError,
    StrategyExecutor,
    SubmissionRefused,
)
from verify_tools.ssot_ast_audit import (
    audit_repo_ssot_ast,
    audit_venue_string_branches_ast,
)
from providers.t212_provider import T212Provider
from strategies import replay_execution, replay_metrics
from trailing_core import apply_trailing_asset_env_overrides
from unittest.mock import Mock, patch


class FakeProvider(MarketDataProvider):
    name = "Fake"

    def __init__(
        self,
        prov_name: str = "Fake",
        *,
        supports="ABCUSD",
        price: float = 10.0,
        trades=None,
        place_fn=None,
        priority: int | None = None,
        ohlc_points=None,
        asset_scope: str | None = None,
    ):
        self.name = prov_name
        self._supports = supports
        self._price = price
        self._trades = trades
        self._place_fn = place_fn
        self._ohlc_points = ohlc_points
        if priority is not None:
            self._HISTORICAL_OHLC_PRIORITY = priority
        if asset_scope is not None:
            self._ASSET_SCOPE = asset_scope

    def supports_symbol(self, symbol):
        if isinstance(self._supports, bool):
            return self._supports
        return symbol == self._supports

    def _fetch_current_price(self, symbol):
        return self._price

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
        if self._trades is not None:
            return list(self._trades)
        return [
            {"side": "BUY", "price": 9, "qty": 2, "timestamp": 1},
            {"side": "SELL", "price": 11, "qty": 2,
             "timestamp": time.time() * 1000},
        ]

    def place_order(self, symbol, side, price, qty, force=False, **kwargs):
        if self._place_fn is not None:
            return self._place_fn(symbol, side, price, qty, force=force, **kwargs)
        return {"orderId": "ORD-777", "symbol": symbol, "side": side, "price": price, "qty": qty, "force": force}

    def fetch_historical_ohlc_points(self, symbol: str, days: int, interval: str = "1h"):
        return self._ohlc_points

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

        partial_cancel = OrderStatus("canceled", 0.5, 5.0, 0.01)
        self.assertTrue(partial_cancel.terminal)
        self.assertFalse(partial_cancel.fully_filled)
        self.assertTrue(partial_cancel.has_fill)

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
        self.assertIs(api._regime_snapshot_resolver, _runtime_market_regime_snapshot)
        manager = Mock()
        snapshot = {"gradient_recent": 0.6, "epsilon": 0.1, "ts": 1000.0}
        manager.get_fresh_snapshot.return_value = snapshot
        with patch("cacheManager.get_short_trend_manager", return_value=manager):
            self.assertIs(_runtime_market_regime_snapshot("ABCUSD", now=1001.0),
                          snapshot)
        manager.get_fresh_snapshot.assert_called_once_with("ABCUSD", now=1001.0)

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


# ============================================================================
# Domain Regressions & Input Hygiene (Targeted Validation)
# ============================================================================


def test_marketapi_provider_by_name_handles_non_string_safely_iter86_s4():
    """[ITER86 #4] MarketApi.provider_by_name handles non-string safely."""
    # 4. MarketApi.provider_by_name handles non-string safely
    assert api.provider_by_name(True) is None
    assert api.provider_by_name(123) is None
    assert api.provider_by_name("binance") is not None

def test_marketapi_rejects_bool_none_blank_order_id_and_client_order_id_iter112_s3():
    """[ITER112 #3] MarketApi rejects bool/None/blank order_id and client_order_id before str() coercion."""
    # 3. MarketApi rejects bool/None/blank order_id and client_order_id before str() coercion
    with pytest.raises(ProviderError):
        ma.api.order_by_client_id("BTCUSDC", True)
    with pytest.raises(ProviderError):
        ma.api.order_status("BTCUSDC", True)
    with pytest.raises(ProviderError):
        ma.api.order_status("BTCUSDC", "   ")
    with pytest.raises(ProviderError):
        ma.api.cancel_order("BTCUSDC", None)

def test_marketapi_latest_fill_price_and_market_regime_context_raise_valueerror_on_none_iter112_s4():
    """[ITER112 #4] MarketApi.latest_fill_price and market_regime_context raise ValueError on None/non-numeric."""
    # 4. MarketApi.latest_fill_price and market_regime_context raise ValueError on None/non-numeric
    with pytest.raises(ValueError):
        ma.api.latest_fill_price("BTCUSDC", "BUY", None)
    with pytest.raises(ValueError):
        ma.api.latest_fill_price("BTCUSDC", "BUY", 60.0, min_notional="bad")
    with pytest.raises(ValueError):
        ma.api.market_regime_context("BTCUSDC", now="bad")

def test_market_api__parse_bool_flag_and_require_int_order_id_iter170_s1():
    """[ITER170 #1] market_api._parse_bool_flag and require_int_order_id."""
    # 1. market_api._parse_bool_flag and require_int_order_id
    assert ma._parse_bool_flag("y", False) is True
    assert ma._parse_bool_flag("n", True) is False
    assert ma._parse_bool_flag(2, False) is True
    assert ma.require_int_order_id("+12345.0", "ctx") == 12345
    with pytest.raises(sexec.ProviderError):
        ma.require_int_order_id(frozenset({1}), "ctx")

def test_marketapi_latest_fill_price_with_bid_ask_aliases_and_marketapi_iter170_s2():
    """[ITER170 #2] MarketApi.latest_fill_price with BID/ASK aliases and MarketApi.place_order with positional force."""
    # 2. MarketApi.latest_fill_price with BID/ASK aliases and MarketApi.place_order with positional force
    captured = {}

    def _place(symbol, side, price, qty, force=False, **kwargs):
        captured["force"] = force
        return {"orderId": "111", "force": force}

    facade = ma.MarketApi([
        FakeProvider(
            "Dummy",
            supports=True,
            price=100.0,
            trades=[{"side": "BUY", "price": 105.0, "qty": 1.0, "timestamp": int(time.time() * 1000)}],
            place_fn=_place,
        )
    ])
    assert facade.latest_fill_price("BTCUSDC", "BID", 3600) == pytest.approx(105.0)
    res = facade.place_order("BTCUSDC", "BUY", 100.0, 1.0, True)
    assert res["force"] is True and captured["force"] is True


def test_declarative_venue_attributes_and_unified_routing(monkeypatch):
    """Verify venue-agnostic MarketDataProvider capabilities, SSOT cold-start client detection, and unified MarketApi routing."""

    assert not hasattr(MarketDataProvider, "_REAL_CLIENT_CLASS_NAMES")
    assert not hasattr(MarketApi, "_provider_routed_or_default")
    assert KrakenProvider._CLIENT_CLASS_NAME == "KrakenClient"
    assert HyperliquidProvider._CLIENT_CLASS_NAME == "HLClient"
    assert T212Provider._CLIENT_CLASS_NAME == "T212Client"
    assert HyperliquidProvider().default_quote_asset() == "USDC"
    assert KrakenProvider().default_quote_asset() is None

    monkeypatch.setattr(botcore._RUNTIME_SERVICES, "allow_shared_disk_cache", True)
    # Fresh Kraken/T212 providers with uninitialized _cli must allow SSOT cache (must not confuse bound _client method with mock).
    assert KrakenProvider()._allow_ssot_price_cache() is True
    assert T212Provider()._allow_ssot_price_cache() is True
    assert HyperliquidProvider()._allow_ssot_price_cache() is True
    # Injected mock clients disable shared disk SSOT cache unless explicitly enabled.
    assert KrakenProvider(client=object())._allow_ssot_price_cache() is False
    assert T212Provider(client=object())._allow_ssot_price_cache() is False
    assert HyperliquidProvider(client=object())._allow_ssot_price_cache() is False

    hl = HyperliquidProvider(token="HYPE", live_enabled=False)
    hl_api = MarketApi([hl])
    with pytest.raises(ProviderError, match="unsupported"):
        hl_api.get_price_history("BTCUSDC", 1.0)
    with pytest.raises(ProviderError, match="unsupported"):
        hl_api.get_orders("BTCUSDC", "BUY", 60.0)

    captured_quotes = []
    monkeypatch.setattr(
        "instrument.Instrument.place",
        lambda inst, *args, **kwargs: captured_quotes.append((inst.base, inst.quote)) or "ok",
    )
    assert hl_api.place("HYPE", "BUY", 40.0, 1.0) == "ok"
    assert captured_quotes[-1][1] == "USDC"


def test_provider_dag_purity_and_submit_order_unification(monkeypatch):
    """Enforce strict downward DAG across providers/ and verify unified submit_order, ClosedPriceSeries SSOT, and declarative intent tags."""

    root = Path(__file__).resolve().parents[1]
    forbidden_base_imports = {
        "order_guard",
        "guard_hierarchy",
        "llm_order_guard",
        "order_lifecycle",
        "order_retry",
        "order_retry_worker",
        "instrument",
        "strategies",
        "market_regime",
        "intelligence",
        "orchestratorTrade",
        "orchestratorOS",
        "cacheManager",
        "binance_api",
        "kraken",
        "hyperliquid",
        "212trading",
    }
    for base_mod_file in (
        "base.py",
        "ws_stream.py",
        "ssot_bridge.py",
        "venue_metadata.py",
        "ohlc_utils.py",
        "price_poller.py",
    ):
        base_src = (root / "providers" / base_mod_file).read_text(encoding="utf-8")
        base_tree = ast.parse(base_src, filename=f"providers/{base_mod_file}")
        imported_roots = set()
        for node in ast.walk(base_tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_roots.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported_roots.add(node.module.split(".")[0])
        assert not (imported_roots & forbidden_base_imports), (
            f"DAG violation in providers/{base_mod_file}: {imported_roots & forbidden_base_imports}"
        )

    non_facade_modules = (
        "base.py",
        "ws_stream.py",
        "ssot_bridge.py",
        "venue_metadata.py",
        "venue_ws_protocols.py",
        "broker_history.py",
        "price_poller.py",
        "binance_provider.py",
        "hyperliquid_provider.py",
        "kraken_provider.py",
        "t212_provider.py",
        "replay_provider.py",
        "order_id_context.py",
        "ohlc_utils.py",
        "order_ohlc_utils.py",
        "quantity.py",
        "strategy_executor.py",
        "execution_audit.py",
        "intent_state.py",
        "intent_execution.py",
        "intent_reconciliation.py",
        "intent_telemetry.py",
    )
    forbidden_upward_layers = {
        "cacheInterface",
        "cacheaccount",
        "cacheprice",
        "cachetrend",
        "cacheManager",
        "market_regime",
        "order_guard",
        "guard_hierarchy",
        "llm_order_guard",
        "order_lifecycle",
        "order_retry",
        "order_retry_worker",
        "instrument",
        "strategies",
        "intelligence",
        "orchestratorTrade",
        "orchestratorOS",
    }
    for mod_file in non_facade_modules:
        mod_src = (root / "providers" / mod_file).read_text(encoding="utf-8")
        mod_tree = ast.parse(mod_src, filename=f"providers/{mod_file}")
        mod_roots = set()
        for node in ast.walk(mod_tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    mod_roots.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mod_roots.add(node.module.split(".")[0])
        assert not (mod_roots & forbidden_upward_layers), (
            f"Upward DAG violation in providers/{mod_file}: {mod_roots & forbidden_upward_layers}"
        )

    binance_transport_modules = (
        "bapi.py",
        "bapi_allorders.py",
        "bapi_client.py",
        "bapi_placeorder.py",
        "bapi_trades.py",
        "bapi_ws.py",
        "bapi_ws_bridge.py",
        "bapi_ws_normalize.py",
        "market_data.py",
    )
    venue_transport_modules = (
        *(f"binance_api/{m}" for m in binance_transport_modules),
        "212trading/t212_client.py",
        "212trading/market_data.py",
        "kraken/kraken_client.py",
        "kraken/market_data.py",
        "hyperliquid/hl_client.py",
        "hyperliquid/market_data.py",
    )
    forbidden_transport_layers = forbidden_upward_layers | {"offline"}
    forbidden_transport_provider_targets = {
        "providers.market_api",
        "providers.binance_provider",
        "providers.kraken_provider",
        "providers.hyperliquid_provider",
        "providers.t212_provider",
        "providers.replay_provider",
    }
    for vmod_rel in venue_transport_modules:
        bmod_src = (root / vmod_rel).read_text(encoding="utf-8")
        bmod_tree = ast.parse(bmod_src, filename=vmod_rel)
        bmod_roots = set()
        bmod_full_imports = set()
        for node in ast.walk(bmod_tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    bmod_roots.add(alias.name.split(".")[0])
                    bmod_full_imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                bmod_roots.add(node.module.split(".")[0])
                bmod_full_imports.add(node.module)
                for alias in node.names:
                    bmod_full_imports.add(f"{node.module}.{alias.name}")
        assert not (bmod_roots & forbidden_transport_layers), (
            f"Upward DAG violation in {vmod_rel}: {bmod_roots & forbidden_transport_layers}"
        )
        assert not (bmod_full_imports & forbidden_transport_provider_targets), (
            f"Transport-to-adapter DAG violation in {vmod_rel}: {bmod_full_imports & forbidden_transport_provider_targets}"
        )

    facade_src = (root / "providers" / "market_api.py").read_text(encoding="utf-8")
    facade_tree = ast.parse(facade_src, filename="providers/market_api.py")
    facade_roots = set()
    for node in ast.walk(facade_tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                facade_roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            facade_roots.add(node.module.split(".")[0])
    forbidden_facade_upward = forbidden_upward_layers - {"market_regime"}
    assert not (facade_roots & forbidden_facade_upward), (
        f"Upward DAG violation in providers/market_api.py: {facade_roots & forbidden_facade_upward}"
    )

    gateway_modules = (
        "instrument.py",
        "order_guard.py",
        "order_lifecycle.py",
        "order_retry.py",
        "order_retry_worker.py",
    )
    forbidden_venue_sdks = {"binance_api", "binance", "kraken", "hyperliquid", "212trading", "hl_client"}
    for gw_file in gateway_modules:
        gw_src = (root / gw_file).read_text(encoding="utf-8")
        gw_tree = ast.parse(gw_src, filename=gw_file)
        gw_roots = set()
        for node in ast.walk(gw_tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    gw_roots.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                gw_roots.add(node.module.split(".")[0])
        assert not (gw_roots & forbidden_venue_sdks), (
            f"Venue SDK leak in Order Gateway module {gw_file}: {gw_roots & forbidden_venue_sdks}"
        )

    # Enforce 0 module-level SCCs and 0 top-level package SCCs across the entire repository

    venv_roots = {
        d.name
        for d in root.iterdir()
        if d.is_dir() and (d.name in ("venv", ".venv", "myenv") or (d / "pyvenv.cfg").exists())
    }
    excluded_top_dirs = {"tests", "offline", "verify_tools"} | venv_roots
    all_mods = {}
    for p in sorted(root.rglob("*.py")):
        rel = p.relative_to(root)
        if any(
            part.startswith(".") or part in ("__pycache__", "site-packages")
            for part in rel.parts
        ) or rel.parts[0] in excluded_top_dirs:
            continue
        mod_name = ".".join(rel.parts).removesuffix(".py").removesuffix(".__init__")
        all_mods[mod_name] = rel

    def _subpkg_id(mod: str) -> str:
        parts = mod.split(".")
        return ".".join(parts[:2]) if len(parts) >= 2 else parts[0]

    pkg_edges = defaultdict(set)
    subpkg_edges = defaultdict(set)
    mod_edges = defaultdict(set)
    offline_imports = []
    for mod_name, rel in sorted(all_mods.items()):
        p = root / rel
        src_top = rel.parts[0].removesuffix(".py")
        src_subpkg = _subpkg_id(mod_name)
        pkg_parts = rel.parts[:-1]
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.Import):
                for a in node.names:
                    targets.append(a.name)
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:
                    targets.append(node.module)
                    for a in node.names:
                        sub = f"{node.module}.{a.name}"
                        if sub in all_mods:
                            targets.append(sub)
                elif node.level > 0:
                    base_parts = list(pkg_parts[: len(pkg_parts) - (node.level - 1)])
                    if node.module:
                        base_parts.extend(node.module.split("."))
                        rel_mod = ".".join(base_parts)
                        targets.append(rel_mod)
                    else:
                        rel_mod = ".".join(base_parts)
                    for a in node.names:
                        sub = f"{rel_mod}.{a.name}" if rel_mod else a.name
                        if sub in all_mods:
                            targets.append(sub)
            for tgt in targets:
                if tgt.startswith(("hyperliquid.info", "hyperliquid.exchange", "hyperliquid.utils")):
                    continue
                t_top = tgt.split(".")[0]
                if t_top == "offline":
                    offline_imports.append(f"{rel}:{getattr(node, 'lineno', 0)} -> {tgt}")
                if (
                    t_top != src_top
                    and t_top not in excluded_top_dirs
                    and ((root / f"{t_top}.py").exists() or (root / t_top).is_dir())
                ):
                    pkg_edges[src_top].add(t_top)
                parts = tgt.split(".")
                for i in range(len(parts), 0, -1):
                    cand = ".".join(parts[:i])
                    if cand in all_mods:
                        if cand != mod_name:
                            mod_edges[mod_name].add(cand)
                        t_subpkg = _subpkg_id(cand)
                        if t_subpkg != src_subpkg:
                            subpkg_edges[src_subpkg].add(t_subpkg)
                        break

    assert not offline_imports, f"Upward/sideways offline.* imports in production modules: {offline_imports}"

    def _find_sccs(graph):
        index = 0
        indices = {}
        lowlink = {}
        stack = []
        on_stack = set()
        sccs = []

        def _strongconnect(v):
            nonlocal index
            indices[v] = index
            lowlink[v] = index
            index += 1
            stack.append(v)
            on_stack.add(v)
            for w in sorted(graph.get(v, ())):
                if w not in indices:
                    _strongconnect(w)
                    lowlink[v] = min(lowlink[v], lowlink[w])
                elif w in on_stack:
                    lowlink[v] = min(lowlink[v], indices[w])
            if lowlink[v] == indices[v]:
                scc = []
                while True:
                    w = stack.pop()
                    on_stack.remove(w)
                    scc.append(w)
                    if w == v:
                        break
                if len(scc) > 1:
                    sccs.append(sorted(scc))

        for v in sorted(graph):
            if v not in indices:
                _strongconnect(v)
        return sccs

    assert _find_sccs(pkg_edges) == [], f"Top-level package SCC cycle detected: {_find_sccs(pkg_edges)}"
    assert _find_sccs(subpkg_edges) == [], f"Subpackage-level SCC cycle detected: {_find_sccs(subpkg_edges)}"
    assert _find_sccs(mod_edges) == [], f"Module-level SCC cycle detected: {_find_sccs(mod_edges)}"


    assert T212Provider().supports_order_validity() is True
    assert KrakenProvider().supports_order_validity() is False
    assert KrakenProvider().defers_preflight_quantity() is True
    assert HyperliquidProvider().defers_preflight_quantity() is True
    assert T212Provider().defers_preflight_quantity() is False

    assert OhlcClosedPriceSeries is market_regime.ClosedPriceSeries
    assert BaseClosedPriceSeries is OhlcClosedPriceSeries
    assert MarketDataProvider.resolve_venue_intent_executor_tag("t212") == "[auto_sharetrade_executor]"
    assert MarketDataProvider.resolve_venue_intent_executor_tag("binance") == "[auto_cryptotrade_executor]"
    assert resolve_intent_topic_tag({}, "t212") == "[auto_sharetrade_executor]"
    assert resolve_intent_topic_tag({}, "kraken") == "[auto_cryptotrade_executor]"

    dummy = FakeProvider("DummyExec", supports="TESTUSDC", price=50.0)
    facade = MarketApi([dummy])
    assert isinstance(dummy, StrategyExecutor)
    assert isinstance(facade, StrategyExecutor)
    assert facade.submit_order("TESTUSDC", "buy", 2.0, price=49.5) == "ORD-777"
    with pytest.raises(SubmissionRefused):
        facade.submit_order("TESTUSDC", "HOLD", 2.0, price=49.5)

    # Verify order_guard bound hooks remain monkeypatch-transparent
    monkeypatch.setattr(order_guard, "window_reference", lambda *a, **kw: 98.75)
    monkeypatch.setattr(order_guard, "get_window_cfg", lambda *a, **kw: 3600.0)
    assert dummy.profit_guard_window_ref("TESTUSDC", "BUY", 120.0) == pytest.approx(98.75)

    replay = ReplayMarketDataProvider({"BTCUSDC": [(1000.0, 100.0), (1060.0, 102.0)]})
    replay.advance("BTCUSDC")
    assert isinstance(replay, StrategyExecutor)
    replay_oid = replay.submit_order("BTCUSDC", "BUY", 1.0, price=100.0)
    assert replay_oid.startswith("replay-")


def test_replay_execution_metrics_and_config_ssot_identity(monkeypatch, tmp_path):
    """Verify canonical SSOT identity across botcore, order_ohlc_utils, strategies.replay_*, and offline.backtests.*."""

    assert ex.ExecutionModel is replay_execution.ExecutionModel
    assert ex.FeeModel is replay_execution.FeeModel
    assert ex.split_order_fill is replay_execution.split_order_fill
    assert ex.choose_intrabar_scenario is replay_execution.choose_intrabar_scenario
    assert ex.validate_ohlc_bars is ohlc_utils.validate_ohlc_bars
    assert ds.align_previous_values is ohlc_utils.align_previous_values
    assert ohlc_utils.validate_ohlc_bars is order_ohlc_utils.validate_ohlc_bars
    assert ohlc_utils.align_previous_values is order_ohlc_utils.align_previous_values
    assert ohlc_utils.candle_interval is sexec.candle_interval
    assert ohlc_utils.ProviderError is sexec.ProviderError
    assert mt.calculate_performance_metrics is replay_metrics.calculate_performance_metrics
    assert mt.round_metric is replay_metrics.round_metric
    assert mt.buy_hold_pct is replay_metrics.buy_hold_pct
    assert ev.load_layered_env_config is botcore.load_layered_env_config

    saved_botcore_dict = dict(botcore.__dict__)
    try:
        importlib.reload(botcore)
        assert ev.load_layered_env_config is botcore.load_layered_env_config
    finally:
        botcore.__dict__.update(saved_botcore_dict)
        botcore._sync_reexport_modules()

    assert _resolve_provider_label("xstock_watch", "") == "Kraken"
    assert _resolve_provider_label("hl_bot", "") == "Hyperliquid"
    assert _resolve_provider_label("", "trading212") == "T212"
    assert _resolve_provider_label("rtrade", "") == "Binance"

    assert botcore.is_simulated_provider_name("Replay") is True
    assert botcore.is_simulated_provider_name("paper_sim") is True
    assert botcore.is_simulated_provider_name("binance") is False
    assert botcore.is_simulated_provider_name(True) is False
    assert botcore.secret_env("SECRET_HASH_KEY", env={"SECRET_HASH_KEY": " abc#123 "}) == "abc#123"
    assert botcore.is_env_defined("K", env={"K": ""}) is True
    assert botcore.is_env_defined("K", env={}) is False

    (tmp_path / "runtime.env").write_text("T212_ORDER_VALIDITY=DAY\n", encoding="utf-8")
    assert botcore.folder_or_env_value(tmp_path, "T212_ORDER_VALIDITY", env={}) == "DAY"
    assert (
        botcore.folder_or_env_value(
            tmp_path, "T212_ORDER_VALIDITY", respect_empty_env=True, env={"T212_ORDER_VALIDITY": ""}
        )
        is None
    )
    assert botcore.any_bool_env("A", "B", env={"A": "0", "B": "yes"}) is True
    assert botcore.is_replay_or_backtest_env(env={"REPLAY_MODE": "1"}) is True
    assert botcore.is_gemini_llm_disabled(env={"DISABLE_GEMINI_LLM": "true"}) is True
    assert botcore.is_gemini_llm_disabled(env={}) is False
    assert botcore.is_offline_disk_cache_enabled(env={"MPTRADE_OFFLINE_DISK_CACHE": "1"}) is True
    assert botcore.is_offline_disk_cache_enabled(env={}) is False
    assert botcore.is_auto_start_websockets_enabled(env={}) is True
    assert botcore.is_auto_start_websockets_enabled(env={"BINANCE_AUTO_START_WEBSOCKETS": "0"}) is False
    monkeypatch.setenv("CUSTOM_NONROOT_ENV_TEST", "42")
    assert botcore.has_explicit_nonroot_env("CUSTOM_NONROOT_ENV_TEST") is True


def test_core_facade_and_gateway_zero_venue_string_branches_ast(monkeypatch):
    """Lock Open-Closed Principle: all non-venue-adapter production modules across the repository contain zero venue name comparisons."""

    root = Path(__file__).resolve().parents[1]
    violations = audit_venue_string_branches_ast(root)
    assert violations == [], f"Forbidden venue string branch in venue-agnostic core module: {violations}"

    assert is_primary_stream_venue("binance") is True
    assert is_primary_stream_venue("", default_primary=True) is True
    assert is_primary_stream_venue("", default_primary=False) is False
    assert is_primary_stream_venue("kraken") is False
    assert uses_cycle_state_files("binance") is False
    assert uses_cycle_state_files("kraken") is True
    assert uses_cycle_state_files("hyperliquid") is True
    assert uses_cycle_state_files("t212") is True
    assert MarketDataProvider.canonical_venues() == ("binance", "hyperliquid", "kraken", "t212")
    assert MarketDataProvider.merges_audit_terminal_orders("hyperliquid") is True
    assert MarketDataProvider.merges_audit_terminal_orders("t212") is True
    assert MarketDataProvider.merges_audit_terminal_orders("binance") is False
    assert MarketDataProvider.merges_audit_terminal_orders("kraken") is False
    assert MarketDataProvider.uses_bare_base_order_symbols("hyperliquid") is True
    assert MarketDataProvider.uses_bare_base_order_symbols("binance") is False
    assert MarketDataProvider.uses_shared_trailing_map("binance") is True
    assert MarketDataProvider.uses_shared_trailing_map("kraken") is True
    assert MarketDataProvider.uses_shared_trailing_map("hyperliquid") is False
    assert MarketDataProvider.uses_shared_trailing_map("t212") is False
    assert MarketDataProvider.resolve_venue_default_quote("hyperliquid") == "USDC"
    assert MarketDataProvider.resolve_venue_default_quote("t212") == "USD"
    assert MarketDataProvider.resolve_venue_position_price_decimals("t212") == 4
    assert MarketDataProvider.resolve_venue_position_price_decimals("binance") == 6
    assert MarketDataProvider.uses_shared_registry_symbols("binance") is True
    assert MarketDataProvider.uses_shared_registry_symbols("") is True
    assert MarketDataProvider.uses_shared_registry_symbols("kraken") is False
    assert MarketDataProvider.uses_shared_registry_symbols("hyperliquid") is False
    assert MarketDataProvider.uses_shared_registry_symbols("t212") is False
    assert MarketDataProvider.has_native_l1_derivatives("hyperliquid") is True
    assert MarketDataProvider.has_native_l1_derivatives("binance") is False
    assert MarketDataProvider.is_exact_telemetry_symbol_venue("hyperliquid", quote_isolated=False) is True
    assert MarketDataProvider.is_exact_telemetry_symbol_venue("t212", quote_isolated=False) is True
    assert MarketDataProvider.is_exact_telemetry_symbol_venue("binance", quote_isolated=True, multi_venue_base=False) is False
    assert MarketDataProvider.is_exact_telemetry_symbol_venue("binance", quote_isolated=True, multi_venue_base=True) is True
    assert MarketDataProvider.is_exact_telemetry_symbol_venue("kraken", quote_isolated=True, multi_venue_base=False) is True
    assert is_readonly_xstock_symbol("SPCXx.T") is True
    assert is_readonly_xstock_symbol("SPCXx/USD") is True
    assert is_readonly_xstock_symbol("SPCXxUSD") is True
    assert is_readonly_xstock_symbol("NVDAX.TUSD") is True
    assert is_readonly_xstock_symbol("XBTUSD") is False
    assert is_readonly_xstock_symbol("NVDA_US_EQ") is False
    assert matches_asset_scope("NVDA_US_EQ", "t212", "equities") is True
    assert matches_asset_scope("NVDA_US_EQ", "binance", "equities") is False
    assert matches_asset_scope("BTCUSDC", "t212", "equities") is False
    assert matches_asset_scope("BTCUSDC", "binance", "crypto") is True
    assert matches_asset_scope("NVDA", "t212", "crypto") is False
    assert candles_to_historical_points([{"timestamp": 1_700_000_000_000, "close": "42.5"}, {"timestamp": 0, "close": -1}]) == [
        (1_700_000_000.0, 42.5)
    ]
    assert strip_equity_suffix("NVDA_US_EQ") == "NVDA"
    assert strip_equity_suffix("BRK/B_US_EQ") == "BRK/B"
    assert strip_equity_suffix("AAPL_EQ") == "AAPL"
    assert strip_equity_suffix("VOD_L_EQ", base_only=True) == "VOD"
    assert strip_equity_suffix("BRK/B_US_EQ", base_only=True) == "BRK/B"
    assert same_canonical_base_asset("XBTUSD", "BTCUSDC") is True
    assert same_canonical_base_asset("SPCX_US_EQ", "SPCX") is True
    assert same_canonical_base_asset("ETHUSDC", "BTCUSDC") is False
    assert symbol_quote_asset("XBTEUR") == "EUR"
    assert symbol_quote_asset("EURUSDC") == "USDC"
    assert symbol_quote_asset("BTCUSDC") == "USDC"
    assert resolve_kraken_lookup_symbol("BTCUSDC", quote="EUR") == "XBTEUR"
    assert resolve_kraken_lookup_symbol("DOGEUSDC", quote="EUR") == "XDGEUR"
    assert extract_broker_order_ticker({"instrument": {"ticker": "NVDA_US_EQ"}}) == "NVDA_US_EQ"
    assert normalize_broker_history_items([]) == {}
    assert issubclass(MarketDataProvider, ProviderSsotBridgeMixin)
    assert issubclass(MarketDataProvider, ProviderVenueMetadataMixin)
    assert issubclass(ws_stream.PublicPriceStreamBase, ws_stream.AsyncReconnectStreamBase)
    from providers import price_poller
    assert issubclass(price_poller.VenuePriceStreamBase, ws_stream.PublicPriceStreamBase)
    assert price_poller.HyperliquidPriceStream.__module__ == "providers.price_poller"
    assert price_poller.KrakenPriceStream.__module__ == "providers.price_poller"
    assert price_poller.start_nonbinance_trend_poller.__module__ == "providers.price_poller"
    sample_series = ClosedPriceSeries((101.5, 102.0), 60, 1700003600, (1700000000, 1700003600))
    assert series_to_historical_points(sample_series) == [(1700000000.0, 101.5), (1700003600.0, 102.0)]
    all_t, usdc_p, p2b = extract_trading_quote_pairs(
        {
            "symbols": [
                {"symbol": "TAOUSDC", "baseAsset": "TAO", "quoteAsset": "USDC", "status": "TRADING"},
                {"symbol": "TAOBTC", "baseAsset": "TAO", "quoteAsset": "BTC", "status": "TRADING"},
                {"symbol": "DEADUSDC", "baseAsset": "DEAD", "quoteAsset": "USDC", "status": "BREAK"},
            ]
        },
        quote="USDC",
        strict=True,
    )
    assert all_t == {"TAOUSDC", "TAOBTC"}
    assert usdc_p == {"TAOUSDC"}
    assert p2b == {"TAOUSDC": "TAO"}
    yh_payload = {
        "chart": {
            "result": [
                {
                    "meta": {"regularMarketPrice": 119.0},
                    "timestamp": [1700000000, 1700000300],
                    "indicators": {"quote": [{"close": [120.123456, None]}]},
                }
            ]
        }
    }
    res_y, _, ts_y, q_y = extract_yahoo_chart_result_and_quote(yh_payload)
    assert res_y is not None and ts_y == [1700000000, 1700000300] and q_y["close"][0] == 120.123456
    assert extract_yahoo_chart_latest_price(yh_payload, decimals=4) == 120.1235
    assert extract_yahoo_chart_latest_price({"chart": {"result": [{"meta": {"regularMarketPrice": 88.5}}]}}) == 88.5
    for cls in (BinanceProvider, KrakenProvider, HyperliquidProvider, T212Provider):
        assert "reconciliation_capabilities" not in cls.__dict__
        assert "fetch_ohlc_candles" not in cls.__dict__
        assert isinstance(cls._OHLC_MARKET_DATA_SPEC, tuple) and len(cls._OHLC_MARKET_DATA_SPEC) == 3

    facade = MarketApi([
        FakeProvider("PrimaryCrypto", priority=1, supports=True, ohlc_points=[(1.0, 100.0)], asset_scope="crypto"),
        FakeProvider(
            "SecondaryCrypto",
            priority=2,
            supports=False,
            ohlc_points=[(float(i), 100.0 + i) for i in range(60)],
            asset_scope="crypto",
        ),
    ])
    monkeypatch.setattr("botcore.is_live_network_backfill_enabled", lambda: False)
    assert len(facade.fetch_historical_ohlc_points("BTCUSDC", 7, interval="1h")) == 1
    monkeypatch.setattr("botcore.is_live_network_backfill_enabled", lambda: True)
    pts = facade.fetch_historical_ohlc_points("BTCUSDC", 7, interval="1h")
    assert len(pts) == 60


def test_repo_wide_ssot_primitives_and_ast_purity(tmp_path, monkeypatch):
    """Lock repo-wide SSOT AST boundaries (zero raw json/ConfigParser/urlopen/Request/os.environ outside canonical SSOT modules) and verify quote/env/HTTP helpers."""
    root = Path(__file__).resolve().parents[1]
    ssot_violations = audit_repo_ssot_ast(root)
    assert ssot_violations == [], f"SSOT boundary violation(s) detected: {ssot_violations}"

    assert parse_nonneg_order_field(None) == 0.0
    assert parse_nonneg_order_field("") is None
    assert parse_nonneg_order_field("", allow_empty_str=True) == 0.0
    assert parse_nonneg_order_field("   ", allow_empty_str=True) == 0.0
    assert parse_nonneg_order_field("", allow_empty_str="false") is None
    assert parse_nonneg_order_field("12.5") == 12.5
    for bad_field in (True, False, -0.01, "nan", "inf", "abc"):
        assert parse_nonneg_order_field(bad_field, allow_empty_str=True) is None, bad_field

    # Fiat-USD vs USD-stablecoin quote disambiguation (including false TUSD/BUSD stems)
    for fiat_sym in ("XBTUSD", "DOTUSD", "ARBUSD", "XAUTUSD", "TAOUSD", "TAO/USD"):
        assert is_fiat_usd_pair(fiat_sym) is True, fiat_sym
        assert is_usd_stablecoin_pair(fiat_sym) is False, fiat_sym
    for stable_sym in ("BTCUSDC", "BTCUSDT", "BTCTUSD", "BTCBUSD", "BTCFDUSD", "SOL/USDC"):
        assert is_usd_stablecoin_pair(stable_sym) is True, stable_sym
        assert is_fiat_usd_pair(stable_sym) is False, stable_sym
    for non_crypto_usd in ("SPCX_US_EQ", "AAPL_EQ", "SPCXUSEQ", "SPCXx.T", "BTC", "", None, True):
        assert is_fiat_usd_pair(non_crypto_usd) is False, non_crypto_usd
        assert is_usd_stablecoin_pair(non_crypto_usd) is False, non_crypto_usd

    assert has_crypto_quote_suffix("BTCUSDC") is True
    assert has_crypto_quote_suffix("XBTUSD") is True
    assert has_crypto_quote_suffix("ETHBTC") is True
    assert has_crypto_quote_suffix("SPCX_US_EQ") is False
    assert has_crypto_quote_suffix("NVDA") is False

    assert cache_relpath("order_retry_queue.jsonl") == "cachedb/order_retry_queue.jsonl"
    assert cache_relpath("cachedb/trailing_state.json") == "cachedb/trailing_state.json"
    assert is_cachedb_relpath("cachedb/trailing_state.json") is True
    assert is_cachedb_relpath("kraken/trailing_state.json") is False

    # Declarative MarketApi crypto OHLC provider routing & venue inference
    assert api.resolve_crypto_ohlc_provider("HYPEUSDC").name == "Hyperliquid"
    assert api.resolve_crypto_ohlc_provider("TAOUSDC").name == "Binance"
    assert api.resolve_crypto_ohlc_provider("BTCTUSD").name == "Binance"
    assert api.resolve_crypto_ohlc_provider("XBTUSD").name == "Kraken"
    assert api.resolve_crypto_ohlc_provider("DOTUSD").name == "Kraken"
    assert api.resolve_crypto_ohlc_provider("ARBUSD").name == "Kraken"
    assert api.resolve_crypto_ohlc_provider("XBTEUR").name == "Kraken"
    assert api.resolve_crypto_ohlc_provider("XXBTZUSD").name == "Kraken"
    assert api.resolve_crypto_ohlc_provider("SPCX_US_EQ") is None

    assert api.infer_symbol_venue("HYPEUSDC") == "hyperliquid"
    assert api.infer_symbol_venue("BTCUSDC") == "binance"
    assert api.infer_symbol_venue("BTCTUSD") == "binance"
    assert api.infer_symbol_venue("XBTUSD") == "kraken"
    assert api.infer_symbol_venue("XBTEUR") == "kraken"
    assert api.infer_symbol_venue("XXBTZUSD") == "kraken"
    assert api.infer_symbol_venue("NVDA_US_EQ") == "t212"
    assert api.infer_symbol_venue("kraken:SOLUSDC") == "kraken"
    assert api.infer_symbol_venue("SOLUSDC", explicit_venue="kraken") == "kraken"

    assert is_usdc_pair("BTCUSDC") is True
    assert is_usdc_pair("SOL/USDC") is True
    assert is_usdc_pair("BTCUSDT") is False
    assert is_usdc_pair("NVDA_US_EQ") is False
    assert is_eur_pair("XBTEUR") is True
    assert is_eur_pair("BTCEUR") is True
    assert is_eur_pair("EURUSD") is False
    assert is_eur_pair("SAP_DE_EQ") is False
    assert is_fiat_quote_pair("XBTUSD") is True
    assert is_fiat_quote_pair("XBTEUR") is True
    assert is_fiat_quote_pair("BTCTUSD") is False
    assert is_fiat_quote_pair("BTCUSDC") is False
    assert is_fiat_quote_pair("SPCXx.T") is False
    assert is_fiat_quote_pair("ETH.S") is False
    assert is_usd_compatible_pair("BTCUSDC") is True
    assert is_usd_compatible_pair("XBTUSD") is True
    assert is_usd_compatible_pair("XBTEUR") is False
    assert ensure_quote_suffix("BTC", "USDC") == "BTCUSDC"
    assert ensure_quote_suffix("BTCUSDC", "USDC") == "BTCUSDC"
    assert ensure_quote_suffix("sol/usdc", "USDC") == "SOLUSDC"
    assert ensure_quote_suffix("null", "USDC") == ""
    _null_prov_mkt = type("M", (), {"provider_name_for": staticmethod(lambda _s: "null")})()
    assert ma.resolve_symbol_provider_name("BTCUSDC", market_api=_null_prov_mkt, fallback="Binance") == "Binance"
    assert ma.resolve_symbol_provider_name("BTCUSDC", market_api=_null_prov_mkt, fallback=None) is None
    assert crypto_quote_priority_rank("BTCUSDC") == 0
    assert crypto_quote_priority_rank("BTCUSDT") == 1
    assert crypto_quote_priority_rank("XBTUSD") == 2
    assert crypto_quote_priority_rank("XBTEUR") == 3
    assert crypto_quote_priority_rank("ETHBTC") == 4

    # New botcore & trailing_core SSOT helpers
    cp = botcore.empty_ini_config(inline_comments=True)
    cp.read_string("[sec]\nk = val # comment\n")
    assert cp.get("sec", "k") == "val"

    captured_reqs = []

    class _DummyResp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    assert (
        botcore.http_post_status(
            "https://ntfy.sh/test_topic",
            "hello",
            headers={"Title": "Test"},
            timeout=5.0,
            urlopen_fn=lambda req, timeout=0: (captured_reqs.append((req, timeout)) or _DummyResp()),
        )
        == 200
    )
    assert len(captured_reqs) == 1
    assert captured_reqs[0][0].full_url == "https://ntfy.sh/test_topic"
    assert captured_reqs[0][0].data == b"hello"
    assert captured_reqs[0][1] == 5.0

    f_low = tmp_path / "low.env"
    f_mid = tmp_path / "mid.env"
    f_high = tmp_path / "high.env"
    f_low.write_text("A=low\nB=low\nC=low\n", encoding="utf-8")
    f_mid.write_text("B=mid\nC=mid\n", encoding="utf-8")
    f_high.write_text("C=high\n", encoding="utf-8")
    monkeypatch.setenv("T212_CUSTOM_TEST", "from_env")
    layered = botcore.load_layered_env_config(f_low, f_mid, f_high, override_prefixes=("T212_",))
    assert (layered["A"], layered["B"], layered["C"], layered["T212_CUSTOM_TEST"]) == ("low", "mid", "high", "from_env")
    layered_all = botcore.load_layered_env_config(f_low, override_all_env=True)
    assert layered_all["T212_CUSTOM_TEST"] == "from_env"

    monkeypatch.delenv("BINANCE_AUTO_START_WEBSOCKETS", raising=False)
    botcore.disable_auto_start_websockets_default()
    assert botcore.optional_env("BINANCE_AUTO_START_WEBSOCKETS") == "0"
    assert botcore.set_process_env("TEST_SET_PROC_ENV", 123) == "123"
    cloned = botcore.clone_process_env({"TEST_CLONE_EXTRA": "yes", "TEST_SET_PROC_ENV": None})
    assert cloned["TEST_CLONE_EXTRA"] == "yes" and "TEST_SET_PROC_ENV" not in cloned

    pair_for = {"HYPE": "HYPEUSD"}
    trail_pct = {"HYPE": 15.0}
    apply_trailing_asset_env_overrides(
        "KRAKEN_TRAILING_",
        pair_for,
        trail_pct,
        12.5,
        env={
            "KRAKEN_TRAILING_DEFAULT_PCT": "12.5",
            "KRAKEN_TRAILING_SOL_PAIR": "SOLUSD",
            "KRAKEN_TRAILING_HYPE_PCT": "18.0",
        },
    )
    assert pair_for == {"HYPE": "HYPEUSD", "SOL": "SOLUSD"}
    assert trail_pct == {"HYPE": 18.0, "SOL": 12.5}


def test_intent_reconciliation_modularization_and_lazy_compat(tmp_path):
    from pathlib import Path
    import providers.execution_audit as ea
    import providers.intent_execution as ie
    import providers.intent_reconciliation as ir
    import providers.intent_state as is_mod
    import providers.intent_telemetry as it

    root = Path(__file__).resolve().parent.parent
    ea_tree = ast.parse((root / "providers" / "execution_audit.py").read_text(encoding="utf-8"))
    ea_imports = {
        node.module
        for node in ast.walk(ea_tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "intent_execution" not in ea_imports
    assert "intent_reconciliation" not in ea_imports
    assert "intent_telemetry" not in ea_imports

    ir_tree = ast.parse((root / "providers" / "intent_reconciliation.py").read_text(encoding="utf-8"))
    ir_imported_mods = {
        alias.name
        for node in ast.walk(ir_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "glob" not in ir_imported_mods

    it_tree = ast.parse((root / "providers" / "intent_telemetry.py").read_text(encoding="utf-8"))
    it_imported_mods = {
        alias.name
        for node in ast.walk(it_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module
        for node in ast.walk(it_tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert it_imported_mods.isdisjoint(
        {"os", "glob", "state_io", "intent_reconciliation", "providers.intent_reconciliation"}
    )

    assert ir.reconcile_intents_from_execution_audit.__module__ == "providers.intent_reconciliation"
    assert it.summarize_recent_ai_intents.__module__ == "providers.intent_telemetry"
    assert it.build_last_cycle_execution_status.__module__ == "providers.intent_telemetry"
    assert it._resolve_intent_execution_details.__module__ == "providers.intent_telemetry"
    assert ea.reconcile_intents_from_execution_audit is ir.reconcile_intents_from_execution_audit
    assert ea.summarize_recent_ai_intents is it.summarize_recent_ai_intents
    assert ir.summarize_recent_ai_intents is it.summarize_recent_ai_intents
    assert ea.build_last_cycle_execution_status is it.build_last_cycle_execution_status
    assert ir.build_last_cycle_execution_status is it.build_last_cycle_execution_status
    assert ea.cancel_existing_ai_open_orders is ie.cancel_existing_ai_open_orders
    assert ea.normalize_trading_decisions is is_mod.normalize_trading_decisions

    for fname in (
        "execution_audit_2026-11-10.jsonl",
        "execution_audit_2026-11-11.jsonl",
        "execution_audit_2026-11-12.jsonl",
        "execution_audit_2026-11-13.jsonl",
        "execution_audit_custom.jsonl",
    ):
        (tmp_path / fname).write_text("", encoding="utf-8")

    assert ea.iter_audit_jsonl_paths("") == []
    bounded = [
        Path(p).name
        for p in ea.iter_audit_jsonl_paths(str(tmp_path), min_day="2026-11-11", max_day="2026-11-12")
    ]
    assert bounded == [
        "execution_audit_2026-11-11.jsonl",
        "execution_audit_2026-11-12.jsonl",
        "execution_audit_custom.jsonl",
    ]
    limited_rev = [
        Path(p).name
        for p in ea.iter_audit_jsonl_paths(str(tmp_path), reverse=True, unbounded_limit=2)
    ]
    assert limited_rev == [
        "execution_audit_custom.jsonl",
        "execution_audit_2026-11-13.jsonl",
    ]


def test_market_monitor_polymorphic_registries_and_soc_split():
    from market_monitor import newcoinsdiscovery as ncd, priceplatforms as pp, pricefetcher as pf

    assert set(pp.PricePlatformFactory._PLATFORM_REGISTRY) == {
        "binance",
        "hyperliquid",
        "coinmarketcap",
        "yahoo",
    }
    assert issubclass(pp.BinancePricePlatform, pp.CryptoPricePlatformBase)
    assert issubclass(pp.HyperliquidPricePlatform, pp.CryptoPricePlatformBase)
    assert issubclass(pp.CoinMarketCapPricePlatform, pp.CryptoPricePlatformBase)
    assert issubclass(pp.StockYahooPricePlatform, pp.EquityPricePlatformBase)
    assert pp.StockYahooPricePlatform.IS_EQUITY_PLATFORM is True
    assert pp.BinancePricePlatform.IS_EQUITY_PLATFORM is False
    assert pf.PricePlatformFactory is pp.PricePlatformFactory
    assert pf.StockYahooPricePlatform is pp.StockYahooPricePlatform

    assert set(ncd.NewCoinsFactory._SOURCE_REGISTRY) == {
        "coinmarketcap",
        "coingecko",
        "binance",
        "dexscreener",
    }
    assert issubclass(ncd.CoinMarketCapSource, ncd.PricedNewCoinsSource)
    assert issubclass(ncd.CoinGeckoSource, ncd.InformationalNewCoinsSource)
    assert ncd.NewCoinsFactory.registered_source_has_price("coinmarketcap") is True
    assert ncd.NewCoinsFactory.registered_source_has_price("CoinMarketCap") is True
    assert ncd.NewCoinsFactory.registered_source_has_price("binance") is False
    assert ncd.NewCoinsFactory.registered_source_has_price("dexscreener") is False


def test_ohlc_utils_and_ssot_ohlc_24h_bridge_helpers(tmp_path):
    from cachepaths import cache_ohlc_24h_filename
    from providers.ohlc_utils import (
        compute_ols_slope_pct,
        extract_kraken_ohlc_rows,
        extract_yahoo_intraday_chart_bars,
        require_ohlc_symbol,
    )
    from providers.ssot_bridge import ProviderSsotBridgeMixin
    from state_io import save_json_state

    assert cache_ohlc_24h_filename("nvda_us_eq") == "cache_ohlc_24h_NVDA_US_EQ.json"
    assert cache_ohlc_24h_filename(True) == "cache_ohlc_24h_.json"
    assert require_ohlc_symbol(" btcusdc ", upper=True) == "BTCUSDC"
    for bad_sym in (None, True, False, "", "   "):
        with pytest.raises(ValueError):
            require_ohlc_symbol(bad_sym, label="pair")

    kr_rows = extract_kraken_ohlc_rows(
        {"XXBTZUSD": [[100, 1, 2, 1, 1.5], [200, 1.5, 2.5, 1.2, 2.0]], "last": 200},
        drop_incomplete=True,
    )
    assert kr_rows == [[100, 1, 2, 1, 1.5]]
    assert compute_ols_slope_pct([100.0, 101.0, 102.0, 103.0, 104.0, 105.0], bars=6) > 0.0

    meta, y_bars = extract_yahoo_intraday_chart_bars(
        {
            "chart": {
                "result": [
                    {
                        "meta": {"regularMarketPrice": 152.0},
                        "timestamp": [1700000000, 1700000300],
                        "indicators": {"quote": [{"close": [150.5, 152.0], "volume": [1000, 1200]}]},
                    }
                ]
            }
        }
    )
    assert meta == {"regularMarketPrice": 152.0}
    assert y_bars == [(1700000000, 150.5, 1000.0), (1700000300, 152.0, 1200.0)]

    ohlc_file = tmp_path / cache_ohlc_24h_filename("NVDA_US_EQ")
    save_json_state(
        str(ohlc_file),
        {"ts": time.time(), "symbol": "NVDA_US_EQ", "points": [[1700000000000, 150.5, 100.0], [1700000300000, 152.0, 120.0]]},
    )
    assert ProviderSsotBridgeMixin.read_ssot_ohlc_24h_closes(
        ("NVDA_US_EQ",), max_age_sec=300.0, cache_dir_override=str(tmp_path)
    ) == [150.5, 152.0]


def test_venue_ws_protocols_three_tier_soc_and_provider_inheritance():
    from pathlib import Path
    from providers.hyperliquid_provider import HyperliquidProvider
    from providers.kraken_provider import KrakenProvider
    from providers.venue_ws_protocols import (
        HyperliquidWsMixin,
        KrakenWsLedgerMixin,
        VenueWsProtocolInterface,
    )

    root = Path(__file__).resolve().parent.parent
    vwp_tree = ast.parse((root / "providers" / "venue_ws_protocols.py").read_text(encoding="utf-8"))
    vwp_imports = {
        node.module
        for node in ast.walk(vwp_tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "base" not in vwp_imports
    assert "providers.base" not in vwp_imports
    assert "kraken_provider" not in vwp_imports
    assert "hyperliquid_provider" not in vwp_imports

    assert issubclass(KrakenWsLedgerMixin, VenueWsProtocolInterface)
    assert issubclass(HyperliquidWsMixin, VenueWsProtocolInterface)
    assert issubclass(KrakenProvider, KrakenWsLedgerMixin)
    assert issubclass(HyperliquidProvider, HyperliquidWsMixin)

    assert KrakenProvider.ledger_to_fills.__qualname__ == "KrakenWsLedgerMixin.ledger_to_fills"
    assert KrakenProvider.extract_ws_own_trades.__qualname__ == "KrakenWsLedgerMixin.extract_ws_own_trades"
    assert HyperliquidProvider.extract_ws_all_mids_prices.__qualname__ == "HyperliquidWsMixin.extract_ws_all_mids_prices"
    assert HyperliquidProvider.extract_ws_user_fills.__qualname__ == "HyperliquidWsMixin.extract_ws_user_fills"

    assert '"name":"ticker"' in KrakenProvider.build_ws_ticker_subscription(["XBT/USD"]).replace(" ", "")
    assert '"type":"allMids"' in HyperliquidProvider.build_ws_all_mids_subscription().replace(" ", "")
    assert HyperliquidProvider.extract_ws_all_mids_prices(
        {"channel": "allMids", "data": {"mids": {"@107": "35.25"}}},
        ["HYPEUSDC"],
    ) == {"HYPEUSDC": 35.25}


def test_order_id_context_and_broker_history_soc_split():
    from providers import broker_history, order_id_context as oic, order_ohlc_utils as oou

    root = Path(__file__).resolve().parent.parent
    for mod_name in ("order_id_context.py", "broker_history.py"):
        mod_tree = ast.parse((root / "providers" / mod_name).read_text(encoding="utf-8"))
        mod_imports = {
            node.module
            for node in ast.walk(mod_tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        assert "order_ohlc_utils" not in mod_imports
        assert "providers.order_ohlc_utils" not in mod_imports
        assert "base" not in mod_imports
        assert "providers.base" not in mod_imports

    # Canonical SSOT module ownership & identity
    for fn_name in (
        "extract_client_order_id",
        "resolve_order_actor",
        "resolve_order_origin",
        "format_actor_tag",
        "resolve_venue_provider_class",
        "derive_venue_128bit_client_order_id",
        "all_venue_native_client_order_ids",
        "format_venue_intent_client_order_id",
    ):
        fn = getattr(oic, fn_name)
        assert fn.__module__ == "providers.order_id_context"
        assert getattr(oou, fn_name) is fn

    for fn_name in (
        "extract_broker_order_ticker",
        "iter_broker_history_order_fills",
        "extract_broker_history_matches",
        "aggregate_broker_history_fills",
        "normalize_broker_history_items",
    ):
        fn = getattr(broker_history, fn_name)
        assert fn.__module__ == "providers.broker_history"
        assert getattr(oou, fn_name) is fn

    # Unified BOT_CID_SPEC_TABLE invariants (owner vs telemetry module disambiguation)
    assert oic.resolve_order_owner("MO_SD_123") == "spot_dca"
    assert oic.resolve_order_owner("SD_OR_123") == "spot_dca"
    assert oic.resolve_order_owner("SD_123") == "spot_dca"
    assert oic.resolve_order_owner("CW_1") == "watchdogfor_cacheandconfig"
    assert oic.resolve_order_owner("CW") == "unknown"
    assert oic.resolve_order_owner("and_xyz") == "manual"

    assert oic.resolve_order_origin({"clientOrderId": "MO_SD_EXIT_12345678"}) == ("BOT", "trailing", "EXIT")
    assert oic.resolve_order_origin({"clientOrderId": "SD_OR_TRAIL_12345678"}) == ("BOT", "trailing", "TRAIL")
    assert oic.resolve_order_origin({"clientOrderId": "TRL_PEAK_99999999"}) == ("BOT", "trailing", "PEAK")
    assert oic.resolve_order_origin({"clientOrderId": "TO_GRID_12345678"}) == ("BOT", "tradeall", "GRID")
    assert oic.resolve_order_origin({"clientOrderId": "AGS999"}) == ("BOT", "assetguardian", "SL_PROTECT")
    assert oic.resolve_order_origin({"clientOrderId": "AGB999"}) == ("BOT", "assetguardian", "BUY_TIER")
    assert oic.resolve_order_origin({"clientOrderId": "and_abc"}) == ("MANUAL", "manual", None)
    assert oic.format_actor_tag("MANUAL", "manual", None) == "MANUAL"
    assert oic.format_actor_tag("BOT", "trailing", "EXIT") == "BOT(module=trailing, rule=EXIT)"


def test_t212_broker_lifecycle_soc_and_ssot_boundaries():
    from types import SimpleNamespace
    from strategies.base_spot_mixin import BaseSpotMixin
    from tests.conftest import load_venue_modules

    t212_bl, t212_strat, t212_om = load_venue_modules(
        "212trading", "t212_broker_lifecycle", "strategy", "order_manager"
    )
    assert issubclass(t212_bl.T212BrokerLifecycleMixin, BaseSpotMixin)
    assert issubclass(t212_strat.Strategy, t212_bl.T212BrokerLifecycleMixin)
    assert t212_strat._new_state is t212_bl._new_state
    assert t212_strat._sanitize_t212_state is t212_bl._sanitize_t212_state
    assert t212_om.t212_portfolio_position is t212_bl.t212_portfolio_position
    assert t212_om.t212_active_order_matches is t212_bl.t212_active_order_matches

    root = Path(__file__).resolve().parent.parent
    bl_tree = ast.parse((root / "212trading" / "t212_broker_lifecycle.py").read_text(encoding="utf-8"))
    bl_imports = {
        alias.name
        for node in ast.walk(bl_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module
        for node in ast.walk(bl_tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert bl_imports.isdisjoint({"strategy", "market_data", "t212_bot", "replay", "backtest", "order_manager"})

    strat_tree = ast.parse((root / "212trading" / "strategy.py").read_text(encoding="utf-8"))
    strat_imported_names = {
        alias.name
        for node in ast.walk(strat_tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert strat_imported_names.isdisjoint(
        {"ProviderError", "OrderNotFound", "restore_cancelled_orders", "extract_broker_order_ticker", "extract_orig_qty"}
    )

    cli_none = SimpleNamespace(get_portfolio=lambda: None)
    assert t212_bl.t212_portfolio_position(cli_none, "NVDA_US_EQ") is None
    cli_pos = SimpleNamespace(get_portfolio=lambda: [{"ticker": "NVDA_US_EQ", "quantity": 2.5}])
    assert t212_bl.t212_portfolio_position(cli_pos, "NVDA_US_EQ") == (2.5, 0.0)
    assert t212_bl.t212_portfolio_position(cli_pos, "SPCX_US_EQ") == (0.0, 0.0)

    sell_ord = {"id": "s-1", "instrument": {"ticker": "NVDA_US_EQ"}, "quantity": -1.25, "limitPrice": 150.0}
    assert t212_bl.t212_active_order_matches(
        sell_ord, target_ticker="NVDA_US_EQ", expected_side="SELL", expected_qty=1.25, expected_limit=150.0
    )
    assert not t212_bl.t212_active_order_matches(
        sell_ord, target_ticker="NVDA_US_EQ", expected_side="BUY", expected_qty=1.25, expected_limit=150.0
    )
    assert t212_bl.t212_active_order_matches(
        {"id": "m-1", "ticker": "NVDA_US_EQ", "quantity": -1.25},
        target_ticker="NVDA_US_EQ",
        expected_side="SELL",
        expected_qty=1.25,
        is_market=True,
    )


def test_pass12_provider_for_transient_supports_error_never_poisons_route() -> None:
    """Pass #12 invariant & cross-check: transient supports_symbol() errors must not permanently memoize fallback default."""
    default_prov = FakeProvider("Binance", supports="BTCUSDC", price=60000.0)
    flaky_calls = [0]

    class _FlakyProvider(FakeProvider):
        def supports_symbol(self, symbol):
            flaky_calls[0] += 1
            if flaky_calls[0] == 1:
                raise RuntimeError("transient venue catalog error")
            return symbol == "HYPEUSD"

    target_prov = _FlakyProvider("Kraken", supports="HYPEUSD", price=63.25)
    mkt = MarketApi([default_prov, target_prov])

    # 1. Pure invariant: transient exception falls back to default WITHOUT poisoning _route
    assert mkt._provider_for("HYPEUSD") is default_prov
    assert "HYPEUSD" not in mkt._route

    # 2. Dependent cross-check: once transient error clears, get_current_price routes to target_prov and caches it
    assert mkt.get_current_price("HYPEUSD") == pytest.approx(63.25)
    assert mkt._route.get("HYPEUSD") is target_prov

    # 3. Control invariant: clean non-match (no exception) still caches fallback default
    assert mkt._provider_for("UNKNOWNPAIR") is default_prov
    assert mkt._route.get("UNKNOWNPAIR") is default_prov


def test_pass24_tracked_lifecycle_heals_far_future_created_at_for_max_age_cancel():
    """[Pass 24] Far-future pending created_at is healed to now so max_age_seconds cancels stuck orders cleanly."""
    from order_lifecycle import TrackedOrderLifecycle
    from providers.strategy_executor import OrderStatus

    clock = [1_750_000_000.0]
    cancelled: list[str] = []
    persisted: list[dict | None] = []

    active_status = OrderStatus("open", 0.0, 0.0, 0.0)
    cancelled_status = OrderStatus("canceled", 0.0, 0.0, 0.0)

    class _FakeApi:
        def order_status(self, _sym, oid, provider_name=None):
            return cancelled_status if oid in cancelled else active_status

        def cancel_order(self, _sym, oid, provider_name=None):
            cancelled.append(oid)

    lc = TrackedOrderLifecycle(
        _FakeApi(),
        max_age_seconds=180.0,
        clock=lambda: clock[0],
        provider_name="TestProv",
    )

    # 1. Pure invariant: far-future created_at (> now + 60s) is healed in-place to now and persisted
    pending = {
        "intent_id": "INT-1",
        "client_order_id": "CID-1",
        "order_id": "OID-1",
        "symbol": "BTCUSDC",
        "side": "BUY",
        "requested_qty": 0.1,
        "created_at": clock[0] + 7200.0,
    }
    res1 = lc.reconcile(pending, persist=lambda p: persisted.append(dict(p) if isinstance(p, dict) else p))
    assert res1.outcome == "active"
    assert res1.intent["created_at"] == pytest.approx(clock[0])
    assert cancelled == []

    # 2. Dependent cross-check: advancing clock past max_age_seconds cancels the order (instead of waiting 2 hours)
    clock[0] += 181.0
    res2 = lc.reconcile(res1.intent, persist=lambda p: persisted.append(dict(p) if isinstance(p, dict) else p))
    assert cancelled == ["OID-1"]
    assert res2.outcome == "terminal"




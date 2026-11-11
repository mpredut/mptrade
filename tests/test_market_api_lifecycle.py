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
        from providers.market_api import api, _runtime_market_regime_snapshot

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


if __name__ == "__main__":
    unittest.main()

# ============================================================================
# Domain Regressions & Input Hygiene (Targeted Validation)
# ============================================================================


def test_marketapi_provider_by_name_handles_non_string_safely_iter86_s4():
    """[ITER86 #4] MarketApi.provider_by_name handles non-string safely."""
    from providers.market_api import api
    # 4. MarketApi.provider_by_name handles non-string safely
    assert api.provider_by_name(True) is None
    assert api.provider_by_name(123) is None
    assert api.provider_by_name("binance") is not None

def test_marketapi_rejects_bool_none_blank_order_id_and_client_order_id_iter112_s3():
    """[ITER112 #3] MarketApi rejects bool/None/blank order_id and client_order_id before str() coercion."""
    import pytest
    import providers.market_api as ma
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
    import pytest
    import providers.market_api as ma
    # 4. MarketApi.latest_fill_price and market_regime_context raise ValueError on None/non-numeric
    with pytest.raises(ValueError):
        ma.api.latest_fill_price("BTCUSDC", "BUY", None)
    with pytest.raises(ValueError):
        ma.api.latest_fill_price("BTCUSDC", "BUY", 60.0, min_notional="bad")
    with pytest.raises(ValueError):
        ma.api.market_regime_context("BTCUSDC", now="bad")

def test_market_api__parse_bool_flag_and_require_int_order_id_iter170_s1():
    """[ITER170 #1] market_api._parse_bool_flag and require_int_order_id."""
    import pytest
    from providers import market_api as mapi
    from providers import strategy_executor as sexec
    # 1. market_api._parse_bool_flag and require_int_order_id
    assert mapi._parse_bool_flag("y", False) is True
    assert mapi._parse_bool_flag("n", True) is False
    assert mapi._parse_bool_flag(2, False) is True
    assert mapi.require_int_order_id("+12345.0", "ctx") == 12345
    with pytest.raises(sexec.ProviderError):
        mapi.require_int_order_id(frozenset({1}), "ctx")

def test_marketapi_latest_fill_price_with_bid_ask_aliases_and_marketapi_iter170_s2():
    """[ITER170 #2] MarketApi.latest_fill_price with BID/ASK aliases and MarketApi.place_order with positional force."""
    import pytest
    from providers import market_api as mapi
    # 2. MarketApi.latest_fill_price with BID/ASK aliases and MarketApi.place_order with positional force
    captured = {}

    class _DummyProvider(mapi.MarketDataProvider):
        @property
        def name(self):
            return "Dummy"

        def get_current_price(self, symbol):
            return 100.0

        def supports_symbol(self, symbol):
            return True

        def get_trades(self, symbol, since_s):
            return [{"side": "BUY", "price": 105.0, "qty": 1.0, "timestamp": int(time.time() * 1000)}]

        def place_order(self, symbol, side, price, qty, force=False, **kwargs):
            captured["force"] = force
            return {"orderId": "111", "force": force}

    facade = mapi.MarketApi([_DummyProvider()])
    assert facade.latest_fill_price("BTCUSDC", "BID", 3600) == pytest.approx(105.0)
    res = facade.place_order("BTCUSDC", "BUY", 100.0, 1.0, True)
    assert res["force"] is True and captured["force"] is True


def test_declarative_venue_attributes_and_unified_routing(monkeypatch):
    """Verify venue-agnostic MarketDataProvider capabilities, SSOT cold-start client detection, and unified MarketApi routing."""
    import pytest
    import botcore
    from providers.hyperliquid_provider import HyperliquidProvider
    from providers.kraken_provider import KrakenProvider
    from providers.t212_provider import T212Provider

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
    import ast
    from pathlib import Path
    import pytest
    import market_regime
    import order_guard
    from providers.base import ClosedPriceSeries as BaseClosedPriceSeries
    from providers.intent_execution import resolve_intent_topic_tag
    from providers.order_ohlc_utils import ClosedPriceSeries as OhlcClosedPriceSeries
    from providers.replay_provider import ReplayMarketDataProvider
    from providers.strategy_executor import StrategyExecutor, SubmissionRefused

    root = Path(__file__).resolve().parents[1]
    base_src = (root / "providers" / "base.py").read_text(encoding="utf-8")
    base_tree = ast.parse(base_src, filename="providers/base.py")
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
    imported_roots = set()
    for node in ast.walk(base_tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert not (imported_roots & forbidden_base_imports), f"DAG violation in providers/base.py: {imported_roots & forbidden_base_imports}"

    non_facade_modules = (
        "base.py",
        "binance_provider.py",
        "hyperliquid_provider.py",
        "kraken_provider.py",
        "t212_provider.py",
        "replay_provider.py",
        "order_id_context.py",
        "order_ohlc_utils.py",
        "quantity.py",
        "strategy_executor.py",
        "execution_audit.py",
        "intent_state.py",
        "intent_execution.py",
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
    for vmod_rel in venue_transport_modules:
        bmod_src = (root / vmod_rel).read_text(encoding="utf-8")
        bmod_tree = ast.parse(bmod_src, filename=vmod_rel)
        bmod_roots = set()
        for node in ast.walk(bmod_tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    bmod_roots.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                bmod_roots.add(node.module.split(".")[0])
        assert not (bmod_roots & forbidden_transport_layers), (
            f"Upward DAG violation in {vmod_rel}: {bmod_roots & forbidden_transport_layers}"
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
    from collections import defaultdict

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

    pkg_edges = defaultdict(set)
    mod_edges = defaultdict(set)
    offline_imports = []
    for mod_name, rel in sorted(all_mods.items()):
        p = root / rel
        src_top = rel.parts[0].removesuffix(".py")
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
    assert _find_sccs(mod_edges) == [], f"Module-level SCC cycle detected: {_find_sccs(mod_edges)}"

    from providers.hyperliquid_provider import HyperliquidProvider
    from providers.kraken_provider import KrakenProvider
    from providers.t212_provider import T212Provider

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

    class _BaseSubmitDummy(MarketDataProvider):
        @property
        def name(self) -> str:
            return "DummyExec"

        def supports_symbol(self, symbol: str) -> bool:
            return symbol == "TESTUSDC"

        def _fetch_current_price(self, symbol: str):
            return 50.0

        def place_order(self, symbol: str, side: str, price: float, qty: float, force: bool = False, **kwargs):
            return {"orderId": "ORD-777", "symbol": symbol, "side": side, "price": price, "qty": qty, "force": force}

    dummy = _BaseSubmitDummy()
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


def test_replay_execution_metrics_and_config_ssot_identity():
    """Verify canonical SSOT identity across botcore, order_ohlc_utils, strategies.replay_*, and offline.backtests.*."""
    import importlib
    import botcore
    import offline.backtests.datasets as ds
    import offline.backtests.evaluation as ev
    import offline.backtests.execution as ex
    import offline.backtests.metrics as mt
    from notify_engine.server import _resolve_provider_label
    from providers import order_ohlc_utils
    from strategies import replay_execution, replay_metrics

    assert ex.ExecutionModel is replay_execution.ExecutionModel
    assert ex.FeeModel is replay_execution.FeeModel
    assert ex.split_order_fill is replay_execution.split_order_fill
    assert ex.choose_intrabar_scenario is replay_execution.choose_intrabar_scenario
    assert ex.validate_ohlc_bars is order_ohlc_utils.validate_ohlc_bars
    assert ds.align_previous_values is order_ohlc_utils.align_previous_values
    assert mt.calculate_performance_metrics is replay_metrics.calculate_performance_metrics
    assert mt.round_metric is replay_metrics.round_metric
    assert mt.buy_hold_pct is replay_metrics.buy_hold_pct
    assert ev.load_layered_env_config is botcore.load_layered_env_config

    importlib.reload(botcore)
    assert ev.load_layered_env_config is botcore.load_layered_env_config

    assert _resolve_provider_label("xstock_watch", "") == "Kraken"
    assert _resolve_provider_label("hl_bot", "") == "Hyperliquid"
    assert _resolve_provider_label("", "trading212") == "T212"
    assert _resolve_provider_label("rtrade", "") == "Binance"



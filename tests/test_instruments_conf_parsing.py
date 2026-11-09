"""Regression coverage for the authoritative instrument registry (instruments.conf)."""
import configparser
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

from instrument import load_for, load_instruments, load_tradable_instruments  # noqa: E402
import instrument_registry as registry  # noqa: E402
from instrument_registry import ROLES  # noqa: E402
import instrument_registry as ic  # noqa: E402


NUMERIC_MT_KEYS = [
    "gain", "lost", "maxage_days", "hardtp", "hardtp_fraction",
    "hardtp_cooldown_h", "buy_budget", "max_budget",
]


class _Provider:
    name = "fake"

    @staticmethod
    def validate_symbol(symbol):
        if not symbol:
            raise ValueError("empty symbol")


class _Api:
    @staticmethod
    def provider_by_name(name):
        return _Provider() if name.casefold() == "fake" else None


def section(name="BINANCE_TEST", **overrides):
    values = dict(provider="binance", symbol="TESTUSDC", base="TEST", quote="USDC",
                  enabled="yes", isolation="own_ledger", market_hours="24x7")
    values.update({f"role.{role}": "no" for role in registry.ROLES})
    values.update(overrides)
    return f"[{name}]\n" + "\n".join(f"{key} = {value}" for key, value in values.items()) + "\n"


class TestInstrumentRegistry(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.registry = Path(self._temp_dir.name) / "instruments.conf"

    def tearDown(self):
        self._temp_dir.cleanup()

    def _write(self, content):
        self.registry.write_text(content, encoding="utf-8")
        return str(self.registry)

    def _valid_section(self, *, name="ONE", symbol="BTCUSD", extra=""):
        return f"""[{name}]
provider = fake
symbol = {symbol}
base = BTC
quote = USD
enabled = yes
isolation = own_ledger
market_hours = 24x7
{chr(10).join('role.' + role + ' = no' for role in sorted(ROLES))}
{extra}
"""

    def test_missing_registry_fails_closed(self):
        missing = str(Path(self._temp_dir.name) / "missing.conf")
        with self.assertRaises(FileNotFoundError):
            load_instruments(missing, api=_Api())

    def test_missing_core_field_fails_closed(self):
        path = self._write(self._valid_section().replace('quote = USD\n', ""))
        with self.assertRaisesRegex(ValueError, "missing required 'quote'"):
            load_instruments(path, api=_Api())

    def test_invalid_core_enums_fail_closed(self):
        cases = {
            "enabled": self._valid_section().replace("enabled = yes", "enabled = perhaps"),
            "isolation": self._valid_section().replace(
                "isolation = own_ledger", "isolation = shared"),
            "market_hours": self._valid_section().replace(
                "market_hours = 24x7", "market_hours = weekends"),
        }
        for name, content in cases.items():
            with self.subTest(field=name):
                path = self._write(content)
                with self.assertRaises(ValueError):
                    load_instruments(path, api=_Api())

    def test_duplicate_provider_symbol_fails_closed(self):
        path = self._write(
            self._valid_section(name="ONE", symbol="BTCUSD")
            + self._valid_section(name="TWO", symbol="btcusd"))
        with self.assertRaisesRegex(ValueError, "duplicate provider/symbol"):
            load_instruments(path, api=_Api())

    def test_inline_comments_and_false_boolean_parse_without_defaults(self):
        path = self._write(
            self._valid_section(extra='enabled = off # intentionally disabled\n')
            .replace('enabled = yes\n', ""))
        instrument = load_instruments(path, api=_Api())["ONE"]
        self.assertFalse(instrument.enabled)
        self.assertEqual(instrument.isolation, "own_ledger")


class TestCurrentRegistry(unittest.TestCase):
    def test_all_enabled_mt_instruments_parse_numeric_params(self):
        instruments = load_instruments()
        raw_sections = _raw_param_presence()
        for name, inst in instruments.items():
            if not inst.enabled:
                continue
            for key in NUMERIC_MT_KEYS:
                if f"mt.{key}" not in raw_sections.get(name, set()):
                    continue
                value = inst.param("mt", key, None, float)
                self.assertIsNotNone(
                    value,
                    f"[{name}] mt.{key} is present but cannot be parsed as a float")

    def test_budget_regressions(self):
        instruments = load_instruments()
        cases = {
            "BINANCE_BTC": (250.0, 3500.0),
            "BINANCE_TAO": (250.0, 3500.0),
            "KRAKEN_HYPE": (200.0, 5000.0),
        }
        for name, (buy_budget, max_budget) in cases.items():
            with self.subTest(instrument=name):
                inst = instruments[name]
                self.assertEqual(inst.param("mt", "buy_budget", None, float), buy_budget)
                self.assertEqual(inst.param("mt", "max_budget", None, float), max_budget)


class TestInstrumentsSingleSource(unittest.TestCase):
    def test_binance_symbols_match_pre_refactor(self):
        self.assertEqual(ic.binance_symbols(), ["BTCUSDC", "TAOUSDC", "ARBUSDC"])

    def test_trail_pct_map_matches_pre_refactor(self):
        expected = {
            item.symbol: float(item.number("trailing.pct"))
            for item in registry.select_instruments("binance", "trailing").values()
        }
        self.assertEqual(set(expected), {"BTCUSDC", "TAOUSDC", "ARBUSDC"})
        self.assertEqual(ic.trail_pct_map(), expected)

    def test_tradeall_allowlist_matches_pre_refactor(self):
        self.assertEqual(ic.tradeall_trade_symbols(), {"BTCUSDC", "TAOUSDC", "ARBUSDC"})

    def test_arb_is_trend_tracked_and_traded(self):
        self.assertIn("ARBUSDC", ic.binance_symbols())
        self.assertIn("ARBUSDC", ic.trail_pct_map())
        self.assertIn("ARBUSDC", ic.tradeall_trade_symbols())

    def test_trail_and_trade_are_subsets_of_symbols(self):
        syms = set(ic.binance_symbols())
        self.assertLessEqual(set(ic.trail_pct_map()), syms)
        self.assertLessEqual(ic.tradeall_trade_symbols(), syms)

    def test_missing_registry_fails_loud(self):
        with self.assertRaises(FileNotFoundError):
            ic.binance_symbols(path="/nonexistent/instruments.conf")

    def test_registry_import_is_light(self):
        code = (
            "import sys, instrument_registry as instruments_config; "
            "assert 'providers.market_api' not in sys.modules, "
            "'instruments_config dragged in the provider stack'; "
            "print('ok')"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=str(ROOT),
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_symbols_facade_constants_and_dynamic_lookup(self):
        import instrument_registry as sym
        self.assertEqual(sym.symbols, ["BTCUSDC", "TAOUSDC", "ARBUSDC"])
        self.assertEqual(sym.symbols_for("binance", "force_sell"), ["BTCUSDC", "TAOUSDC"])
        with self.assertRaises(AttributeError):
            getattr(sym, "nonexistentsymbol")

    def test_ownership_inventory_symbols_fallback(self):
        from verify_tools.ownership_inventory import _python_symbols
        syms, tao = _python_symbols(ROOT)
        self.assertEqual(syms, ["BTCUSDC", "TAOUSDC", "ARBUSDC"])
        self.assertEqual(tao, "TAOUSDC")


def test_current_execution_policy_is_preserved(tmp_path):
    assert registry.symbols_for("binance") == ["BTCUSDC", "TAOUSDC", "ARBUSDC"]
    expected = {
        "tradeall_fire": {"BTCUSDC", "TAOUSDC", "ARBUSDC"}, "kalman_primary": {"BTCUSDC"},
        "mt": {"BTCUSDC", "TAOUSDC", "ARBUSDC"}, "assetguardian": {"BTCUSDC", "TAOUSDC", "ARBUSDC"},
        "archive": {"BTCUSDC", "TAOUSDC", "ARBUSDC"}, "rtrade": {"TAOUSDC"},
        "force_sell": {"BTCUSDC", "TAOUSDC"},
        "trailing": {"BTCUSDC", "TAOUSDC", "ARBUSDC"},
    }
    for role, symbols in expected.items():
        assert set(registry.symbols_for("binance", role)) == symbols
    modes = {item.symbol: item.setting("tradeall.kalman_mode") for item in
             registry.select_instruments("binance", "tradeall_fire").values()}
    assert modes == {"BTCUSDC": "strict", "TAOUSDC": "strict", "ARBUSDC": "strict"}
    trails = {item.symbol: item.number("trailing.pct") for item in
              registry.select_instruments("binance", "trailing").values()}
    assert set(trails) == {"BTCUSDC", "TAOUSDC", "ARBUSDC"}
    assert trails == ic.trail_pct_map()
    assert all(5.0 <= pct <= 50.0 for pct in trails.values())
    rebuys = {item.symbol: item.rebuy_mode() for item in
              registry.select_instruments("binance", "trailing").values()}
    assert rebuys == {"BTCUSDC": "on", "TAOUSDC": "on", "ARBUSDC": "auto"}
    assert set(registry.select_instruments(role="mt")) == {
        "BINANCE_BTC", "BINANCE_TAO", "BINANCE_ARB", "KRAKEN_HYPE"}
    all_syms, nb_syms, nb_providers = registry.tracked_price_instruments()
    assert all_syms == [
        "BTCUSDC", "TAOUSDC", "ARBUSDC",
        "HYPEUSDC", "HYPEUSD",
        "SPCX_US_EQ", "NVDA_US_EQ", "RGNT_US_EQ",
        "TAOUSD", "ADAUSD",
    ]
    assert nb_syms == [
        "HYPEUSDC", "HYPEUSD",
        "SPCX_US_EQ", "NVDA_US_EQ", "RGNT_US_EQ",
        "TAOUSD", "ADAUSD",
    ]
    assert nb_providers == {
        "HYPEUSDC": "hyperliquid",
        "HYPEUSD": "kraken",
        "SPCX_US_EQ": "t212",
        "NVDA_US_EQ": "t212",
        "RGNT_US_EQ": "t212",
        "TAOUSD": "kraken",
        "ADAUSD": "kraken",
    }
    arch_syms, arch_nb, arch_prov = registry.tracked_price_instruments(role="archive")
    assert arch_syms == all_syms
    assert arch_nb == nb_syms
    assert arch_prov == nb_providers
    krk_trails = {
        item.base: (item.symbol, item.number("trailing.pct"), item.rebuy_mode())
        for item in registry.select_instruments("kraken", "trailing").values()
    }
    assert krk_trails == {
        "HYPE": ("HYPEUSD", 28.0, "on"),
        "TAO": ("TAOUSD", 32.0, "on"),
        "ADA": ("ADAUSD", 35.0, "on"),
    }
    assert registry.is_equity_symbol("SPCX_US_EQ") is True
    assert registry.is_equity_symbol("NVDA_US_EQ") is True
    assert registry.is_equity_symbol("AAPL_EQ") is True
    assert registry.is_equity_symbol("BTCUSDC") is False
    assert registry.is_equity_symbol("TAOUSD") is False
    assert registry.is_equity_symbol(None) is False
    assert registry.is_equity_symbol(True) is False
    custom_eq_conf = tmp_path / "instruments.conf"
    custom_eq_conf.write_text(
        section(
            "T212_AAPL",
            provider="t212",
            symbol="AAPL",
            base="AAPL",
            quote="USD",
            isolation="dedicated",
            market_hours="rth",
        ),
        encoding="utf-8",
    )
    assert registry.is_equity_symbol("AAPL", path=custom_eq_conf) is True


def test_invalid_policy_fails_before_selection(tmp_path):
    overrides = [
        {"role.trailing": "perhaps"}, {"role.trailng": "yes"},
        {"role.trailing": "yes"}, {"role.trailing": "yes", "trailing.pct": "nan"},
        {"role.trailing": "yes", "trailing.pct": "100"},
        {"role.trailing": "yes", "trailing.pct": "13"},
        {"role.trailing": "yes", "trailing.pct": "13", "trailing.rebuy": "maybe"},
        {"role.tradeall_fire": "yes"},
        {"role.tradeall_fire": "yes", "tradeall.kalman_mode": "typo"},
        {"role.kalman_primary": "yes"}, {"symbol": "testusdc"}, {"quote": "USD"},
        {"trail.enabled": "yes"}, {"trail.pct": "13"}, {"tradeall.trade": "yes"},
    ]
    for override in overrides:
        path = tmp_path / "instruments.conf"
        path.write_text(section(**override), encoding="utf-8")
        with pytest.raises(ValueError):
            registry.symbols_for("binance", path=path)


def test_missing_role_is_not_an_implicit_opt_out(tmp_path):
    path = tmp_path / "instruments.conf"
    path.write_text(section().replace("role.trailing = no\n", ""), encoding="utf-8")
    with pytest.raises(ValueError, match="role.trailing"):
        registry.load_registry(path)


def test_disabled_instrument_is_excluded_from_every_consumer(tmp_path):
    path = tmp_path / "instruments.conf"
    path.write_text(section(enabled="no", **{"role.trailing": "yes", "trailing.pct": "13",
                                             "trailing.rebuy": "yes"}),
                    encoding="utf-8")
    assert registry.symbols_for("binance", path=path) == []
    for role in registry.ROLES:
        assert registry.symbols_for("binance", role, path=path) == []
    assert len(registry.select_instruments(only_enabled=False, path=path)) == 1


def test_market_data_only_does_not_enable_any_order_consumer(tmp_path):
    path = tmp_path / "instruments.conf"
    path.write_text(section(), encoding="utf-8")
    assert registry.symbols_for("binance", path=path) == ["TESTUSDC"]
    for role in registry.ROLES:
        assert registry.symbols_for("binance", role, path=path) == []


def test_single_pair_consumer_rejects_ambiguous_or_empty_selection(tmp_path):
    path = tmp_path / "instruments.conf"
    first = section(**{"role.rtrade": "yes"})
    second = section("BINANCE_OTHER", symbol="OTHERUSDC", base="OTHER", **{"role.rtrade": "yes"})
    for content in (section(), first + second):
        path.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError, match="exactly one"):
            registry.single_symbol_for("binance", "rtrade", path=path)
    path.write_text(first, encoding="utf-8")
    assert registry.single_symbol_for("binance", "rtrade", path=path) == "TESTUSDC"


def test_one_new_section_reaches_real_consumers_without_source_edits(tmp_path):
    path = tmp_path / "instruments.conf"
    path.write_text((ROOT / "instruments.conf").read_text() + section(**{
        "role.tradeall_fire": "yes", "tradeall.kalman_mode": "off",
        "role.kalman_primary": "yes", "role.trailing": "yes", "trailing.pct": "18",
        "trailing.rebuy": "yes",
        "role.assetguardian": "yes", "role.archive": "yes", "role.force_sell": "yes",
        "role.mt": "yes", "mt.gain": "9", "mt.lost": "5", "mt.maxage_days": "10",
    }), encoding="utf-8")
    code = textwrap.dedent('''
        import sys
        from pathlib import Path
        import instrument_registry as registry
        registry.DEFAULT_PATH = Path(sys.argv[1])
        assert not any(name in sys.modules for name in ("keys", "providers", "binance.client"))
        import tradeall, assetguardian
        from binance_api import trailing_stop, bapi_client
        from instrument import load_for
        assert "TESTUSDC" in registry.symbols
        assert "TESTUSDC" in registry.symbols_for("binance", "force_sell")
        assert "TESTUSDC" in tradeall.TRADEALL_FIRE_SYMBOLS
        assert "TESTUSDC" in tradeall.KALMAN_PRIMARY_SYMBOLS
        assert tradeall._kalman_gate_blocks("TESTUSDC", "BUY") == (False, "off", None)
        assert "TESTUSDC" in assetguardian.TRACKED_SYMBOLS
        assert "ARBUSDC" in tradeall.TRADEALL_FIRE_SYMBOLS
        assert "ARBUSDC" in assetguardian.TRACKED_SYMBOLS
        assert "TESTUSDC" in registry.symbols_for("binance", "archive")
        assert "BINANCE_TEST" in load_for("mt")
        assert trailing_stop.TRAIL_PCT["TESTUSDC"] == 18
        assert trailing_stop.TRAILING_INSTRUMENTS["TESTUSDC"].base == "TEST"
        assert bapi_client._client is None
    ''')
    result = subprocess.run([sys.executable, "-c", code, str(path)], cwd=ROOT,
                            env=dict(os.environ, BINANCE_API_KEY="", BINANCE_API_SECRET="",
                                     BINANCE_API_KEY_WS="", BINANCE_AUTO_START_WEBSOCKETS="0",
                                     TRADEALL_FIRE_SYMBOLS="ARBUSDC", AG_SYMBOLS="ARBUSDC"),
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_old_symbol_environment_lists_are_not_loaded_anywhere():
    for filename, keys in {
        "config.env": ["TRADEALL_FIRE_SYMBOLS=", "KALMAN_PRIMARY_SYMBOLS=",
                       "KALMAN_GATE_MODE=", "AG_SYMBOLS=", "LONGTREND_NONBINANCE="],
    }.items():
        content = (ROOT / filename).read_text()
        assert not any(key in content for key in keys)


def test_registry_refactor_preserves_existing_mt_values():
    cp = configparser.ConfigParser()
    cp.read(ROOT / "instruments.conf")
    assert cp["BINANCE_BTC"]["mt.buy_budget"] == "250"
    assert cp["BINANCE_TAO"]["mt.max_budget"] == "3500"
    assert cp["KRAKEN_HYPE"]["mt.max_budget"] == "5000"


def _raw_param_presence():
    """Read the registry independently to identify declared mt parameters."""
    path = ROOT / "instruments.conf"
    cp = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    cp.read(path)
    return {
        section_name: {key for key in cp[section_name] if key.startswith("mt.")}
        for section_name in cp.sections()
    }


def test_resolve_tradable_instrument_filtering_and_variants():
    from types import SimpleNamespace

    btc_usd = SimpleNamespace(symbol="BTC/USD", provider_name="kraken", base="BTC", quote="USD", enabled=True)
    btc_eur = SimpleNamespace(symbol="BTC/EUR", provider_name="kraken", base="BTC", quote="EUR", enabled=True)
    hype_usdc = SimpleNamespace(symbol="HYPEUSDC", provider_name="hyperliquid", base="HYPE", quote="USDC", enabled=True)
    disabled_eth = SimpleNamespace(symbol="ETHUSDC", provider_name="binance", base="ETH", quote="USDC", enabled=False)

    inst_map = {
        "KRAKEN_BTC": btc_usd,
        "KRAKEN_BTC_EUR": btc_eur,
        "HYPERLIQUID_HYPE": hype_usdc,
        "BINANCE_ETH": disabled_eth,
    }

    # Rejects junk test symbols and empty/bool
    assert ic.resolve_tradable_instrument("", instruments_map=inst_map) is None
    assert ic.resolve_tradable_instrument(None, instruments_map=inst_map) is None
    assert ic.resolve_tradable_instrument(False, instruments_map=inst_map) is None
    assert ic.resolve_tradable_instrument("BTCUSDC", venue=True, instruments_map=inst_map) is None
    assert ic.resolve_tradable_instrument("BTCX.T", instruments_map=inst_map) is None
    assert ic.resolve_tradable_instrument("BTC.T", instruments_map=inst_map) is None

    # Disabled instrument not returned
    assert ic.resolve_tradable_instrument("ETHUSDC", instruments_map=inst_map) is None

    # Exact symbol and variant matching
    assert ic.resolve_tradable_instrument("BTC/USD", venue="kraken", instruments_map=inst_map) is btc_usd
    assert ic.resolve_tradable_instrument("BTCUSD", venue="kraken", instruments_map=inst_map) is btc_usd
    assert ic.resolve_tradable_instrument("BTCUSDT", venue="kraken", instruments_map=inst_map) is btc_usd
    assert ic.resolve_tradable_instrument("BTC", venue="kraken", instruments_map=inst_map) is btc_usd

    # Quote isolation: BTC/EUR should resolve to btc_eur, not btc_usd
    assert ic.resolve_tradable_instrument("BTC/EUR", venue="kraken", instruments_map=inst_map) is btc_eur
    assert ic.resolve_tradable_instrument("BTCEUR", venue="kraken", instruments_map=inst_map) is btc_eur

    # Venue filtering
    assert ic.resolve_tradable_instrument("HYPEUSDC", venue="hyperliquid", instruments_map=inst_map) is hype_usdc
    assert ic.resolve_tradable_instrument("HYPEUSDC", venue="kraken", instruments_map=inst_map) is None


def test_load_tradable_instruments_duplicate_kraken_base(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "instrument_registry.discover_kraken_cache_pairs",
        lambda ws: [("BTC/USD", "BTC", "USD"), ("BTC/EUR", "BTC", "EUR")]
    )
    monkeypatch.setattr("instrument_registry.load_registry", lambda path: {})
    specs = ic.discover_tradable_specs(str(tmp_path), asset_scope="crypto")
    assert "KRAKEN_BTC" in specs
    assert "KRAKEN_BTC/EUR" in specs
    res = load_tradable_instruments(str(tmp_path), asset_scope="crypto")
    assert "KRAKEN_BTC" in res
    assert "KRAKEN_BTC/EUR" in res


def test_provider_ssot_cachemanager_coupling_and_telemetry_resolution(tmp_path):
    import json
    import time
    from providers.base import live_price_fetch_context
    from providers.execution_audit import AuditedStrategyExecutor
    from providers.hyperliquid_provider import HyperliquidProvider
    from providers.kraken_provider import KrakenProvider
    from intelligence.external.collectors.base_collector import (
        _has_hyperliquid_instrument,
        _resolve_configured_venue,
    )

    now = time.time()
    price_cache = tmp_path / "cache_current_price.json"
    price_cache.write_text(
        json.dumps({
            "timestamp": now,
            "items": {
                "HYPEUSDC": {"symbol": "HYPEUSDC", "price": 41.25, "timestamp": now},
                "HYPEUSD": {"symbol": "HYPEUSD", "price": 41.10, "timestamp": now},
                "ADAUSD": {"symbol": "ADAUSD", "price": 0.725, "timestamp": now},
            },
        }),
        encoding="utf-8",
    )

    trend_cache = tmp_path / "cache_short_trend.json"
    trend_cache.write_text(
        json.dumps({
            "timestamp": now,
            "items": {
                "HYPEUSDC": {"symbol": "HYPEUSDC", "trend": 0.42, "timestamp": now},
                "HYPEUSD": {"symbol": "HYPEUSD", "trend": -0.18, "timestamp": now},
                "ADAUSD": {"symbol": "ADAUSD", "trend": 0.11, "timestamp": now},
            },
        }),
        encoding="utf-8",
    )

    hist_points = [
        {"timestamp": int((now - (11 - i) * 300) * 1000), "price": 40.0 + i * 0.1}
        for i in range(12)
    ]
    (tmp_path / "cache_24_price_HYPEUSDC.json").write_text(
        json.dumps({"timestamp": now, "items": hist_points}),
        encoding="utf-8",
    )

    class _BoomClient:
        def spot_mid(self, pair):
            raise AssertionError("should have used SSOT cache")

        def last_price(self, pair):
            raise AssertionError("should have used SSOT cache")

    hl_prov = HyperliquidProvider(token="HYPE", live_enabled=False)
    hl_prov._client = _BoomClient()
    hl_prov._spot_pair = "@107"
    hl_prov._ssot_price_cache_path = str(price_cache)
    hl_prov._ssot_trend_cache_path = str(trend_cache)
    hl_prov._ssot_24price_cache_dir = str(tmp_path)

    kr_prov = KrakenProvider(client=_BoomClient())
    kr_prov._ssot_price_cache_path = str(price_cache)
    kr_prov._ssot_trend_cache_path = str(trend_cache)
    kr_prov._ssot_24price_cache_dir = str(tmp_path)

    # Venue-isolated SSOT reads: HL gets HYPEUSDC (41.25), Kraken gets HYPEUSD (41.10) & ADAUSD (0.725)
    assert hl_prov.get_current_price("HYPE") == pytest.approx(41.25)
    assert hl_prov.get_current_price("HYPEUSDC") == pytest.approx(41.25)
    assert kr_prov.get_current_price("HYPEUSD") == pytest.approx(41.10)
    assert kr_prov.get_current_price("ADAUSD") == pytest.approx(0.725)

    # 24h history read from cache_24_price_HYPEUSDC.json
    hist = hl_prov.get_price_history("HYPEUSDC", 2.0)
    assert hist is not None and len(hist) == 12

    # Trend slope delegation through StrategyExecutor and AuditedStrategyExecutor
    hl_exec = AuditedStrategyExecutor(hl_prov, venue="hyperliquid")
    kr_exec = AuditedStrategyExecutor(kr_prov, venue="kraken")
    assert hl_exec.trend("HYPEUSDC") == pytest.approx(0.42)
    assert kr_exec.trend("HYPEUSD") == pytest.approx(-0.18)
    assert kr_exec.trend("ADAUSD") == pytest.approx(0.11)

    # Inside live_price_fetch_context (used by cacheManager & cachetrend pollers), SSOT cache is bypassed
    with live_price_fetch_context():
        assert hl_prov.get_ssot_cached_price("HYPEUSDC") is None
        assert kr_prov.get_ssot_cached_price("HYPEUSD") is None

    # Telemetry collectors resolve venues via SSOT registry
    assert _resolve_configured_venue("HYPEUSDC") == "hyperliquid"
    assert _resolve_configured_venue("HYPEUSD") == "kraken"
    assert _resolve_configured_venue("ADAUSD") == "kraken"
    assert _resolve_configured_venue("TAOUSDC") == "binance"
    assert _resolve_configured_venue("SUIUSD") == "binance"
    assert _has_hyperliquid_instrument("HYPEUSDC") is True
    assert _has_hyperliquid_instrument("HYPEUSD") is True
    assert _has_hyperliquid_instrument("ADAUSD") is False

    # SSOT orderbook payload normalization and collector reference price coupling
    from providers.base import MarketDataProvider, resolve_ssot_price_from_cache
    from intelligence.external.collectors.orderbook_depth import OrderbookDepthCollector

    assert resolve_ssot_price_from_cache("HYPEUSDC", cache_dir=str(tmp_path), now_ts=now) == pytest.approx(41.25)
    assert resolve_ssot_price_from_cache("ADAUSD", cache_dir=str(tmp_path), now_ts=now) == pytest.approx(0.725)

    hl_book = MarketDataProvider.normalize_orderbook_payload({
        "levels": [[{"px": "41.20", "sz": "10"}], [{"px": "41.30", "sz": "12"}]]
    })
    assert hl_book == {"bids": [(41.2, 10.0)], "asks": [(41.3, 12.0)]}

    kr_book = MarketDataProvider.normalize_orderbook_payload({
        "result": {"ADAUSD": {"bids": [["0.724", "1000", 123]], "asks": [["0.726", "1200", 123]]}}
    })
    assert kr_book == {"bids": [(0.724, 1000.0)], "asks": [(0.726, 1200.0)]}

    ob_col = OrderbookDepthCollector(cache_dir=str(tmp_path))
    assert ob_col._resolve_ssot_price("HYPEUSDC", now=now) == pytest.approx(41.25)

    # Shared SSOT fills cache for Hyperliquid (cache_trade_hyperliquid.json)
    now_ms = int(now * 1000)
    hl_fills_cache = tmp_path / "cache_trade_hyperliquid.json"
    hl_prov._ssot_fills_cache_path = str(hl_fills_cache)
    hl_prov.write_ssot_fills_cache("HYPEUSDC", str(hl_fills_cache), [
        {"coin": "@107", "oid": 991, "side": "B", "px": "40.5", "sz": "2.0", "time": now_ms - 5000},
        {"coin": "@107", "oid": 991, "side": "B", "px": "41.5", "sz": "2.0", "time": now_ms - 4000},
    ])
    cached_orders = hl_prov.get_orders("HYPEUSDC", "BUY", 3600.0)
    assert len(cached_orders) == 1
    assert cached_orders[0]["orderId"] == "991"
    assert cached_orders[0]["price"] == pytest.approx(41.0)
    assert cached_orders[0]["qty"] == pytest.approx(4.0)
    hl_prov.invalidate_ssot_fills_cache("HYPEUSDC", str(hl_fills_cache))
    assert hl_prov._fills_from_cache("HYPEUSDC") is None

    # _parse_ssot_price_point must never treat timestamp element [0] as price when [1] is None/0
    assert MarketDataProvider._parse_ssot_price_point([1700000000000, None], fallback_ts=1700000000000) is None
    assert MarketDataProvider._parse_ssot_price_point([1700000000000, 0.0], fallback_ts=1700000000000) is None
    assert MarketDataProvider._parse_ssot_price_point([[1700000000000, 42.5]], fallback_ts=1700000000000) == (1700000000000, 42.5)

    # _ssot_symbol_candidates rejects bool and maps Kraken XXBTZUSD -> BTCUSD / XBTUSD
    assert kr_prov._ssot_symbol_candidates(True) == ()  # type: ignore[arg-type]
    assert "BTCUSD" in kr_prov._ssot_symbol_candidates("XXBTZUSD")
    assert "XBTUSD" in kr_prov._ssot_symbol_candidates("XXBTZUSD")

    # _resolve_configured_venue & _is_multi_venue_base inspect dict-based load_registry() without AttributeError
    from types import SimpleNamespace
    import intelligence.external.collectors.base_collector as obd_mod
    import instrument_registry as ir_mod
    orig_load = ir_mod.load_registry
    try:
        ir_mod.load_registry = lambda path=None: {
            "KRAKEN_SOL": SimpleNamespace(symbol="SOLUSD", base="SOL", quote="USD", provider="kraken", enabled=True),
            "HL_SOL": SimpleNamespace(symbol="SOLUSDC", base="SOL", quote="USDC", provider="hyperliquid", enabled=False),
        }
        assert obd_mod._resolve_configured_venue("SOLUSD") == "kraken"
        assert obd_mod._resolve_configured_venue("SOLUSDC") == "hyperliquid"
        assert obd_mod._is_multi_venue_base("SOLUSD") is True
        assert obd_mod._has_hyperliquid_instrument("SOLUSD") is True
    finally:
        ir_mod.load_registry = orig_load


def test_registry_execution_independence_and_layered_binding(tmp_path):
    import ast
    import json
    from cacheprice import resolve_cached_symbol_price
    from instrument import Instrument
    from instrument_registry import InstrumentSpec

    # 1. AST hygiene: instrument_registry has zero imports of execution/cache/intelligence layers and zero reload()
    reg_src = (ROOT / "instrument_registry.py").read_text(encoding="utf-8")
    tree = ast.parse(reg_src)
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert imported_roots.isdisjoint({"instrument", "providers", "state_io", "intelligence", "cacheprice", "binance_api"})
    assert "reload(" not in reg_src
    assert "load_env_file_defaults" not in reg_src
    assert "._provider" not in reg_src and '"_provider"' not in reg_src
    assert "_live_enabled" not in reg_src
    assert "_order_validity" not in reg_src

    # 2. InstrumentSpec.with_execution_policy and InstrumentSpec.param are pure value transformations
    spec = InstrumentSpec(
        name="T212_NVDA",
        provider="t212",
        symbol="NVDA_US_EQ",
        base="NVDA",
        quote="USD",
        enabled=True,
        isolation="own_ledger",
        market_hours="rth",
        params={"mt.buy_budget": "300"},
    ).with_execution_policy(live_enabled=True, order_validity="DAY")
    assert spec.param("mt", "buy_budget", None, float) == pytest.approx(300.0)
    assert spec.param("execution", "live_enabled") == "true"
    assert spec.param("execution", "order_validity") == "DAY"

    # 3. Instrument.from_spec binds provider and forwards per-order order_validity without mutating provider state
    captured_kwargs = {}

    class _FakeT212Provider(_Provider):
        name = "t212"

        @staticmethod
        def guards_internally():
            return True

        @staticmethod
        def is_execution_enabled():
            return True

        @staticmethod
        def place_order(symbol, side, price, qty, **kwargs):
            captured_kwargs.update(kwargs)
            return {"orderId": "t212_1", "status": "FILLED"}

    inst = Instrument.from_spec(spec, api=type("_T212Api", (), {"provider_by_name": staticmethod(lambda _n: _FakeT212Provider())})())
    assert inst.spec is spec
    res = inst.place("BUY", 120.0, 2.0, market=True)
    assert res["orderId"] == "t212_1"
    assert captured_kwargs.get("order_validity") == "DAY"
    assert captured_kwargs.get("live_enabled") is True

    # 4. cacheprice.resolve_cached_symbol_price resolves orderbook fallback without intelligence collector imports
    telem_dir = tmp_path / "telemetry"
    telem_dir.mkdir(parents=True, exist_ok=True)
    (telem_dir / "orderbook_depth_SOLUSDC.json").write_text(
        json.dumps({"symbol": "SOLUSDC", "best_bid": 149.5, "best_ask": 150.5}),
        encoding="utf-8",
    )
    assert resolve_cached_symbol_price("SOLUSDC", venue="binance", base="SOL", quote="USDC", cache_dir=str(tmp_path)) == pytest.approx(150.0)


if __name__ == "__main__":
    unittest.main()

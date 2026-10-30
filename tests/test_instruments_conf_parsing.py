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

import instrument_registry as registry  # noqa: E402
from instrument_registry import ROLES  # noqa: E402
import instruments_config as ic  # noqa: E402


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
            ic.load_instruments(missing, api=_Api())

    def test_missing_core_field_fails_closed(self):
        path = self._write(self._valid_section().replace('quote = USD\n', ""))
        with self.assertRaisesRegex(ValueError, "missing required 'quote'"):
            ic.load_instruments(path, api=_Api())

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
                    ic.load_instruments(path, api=_Api())

    def test_duplicate_provider_symbol_fails_closed(self):
        path = self._write(
            self._valid_section(name="ONE", symbol="BTCUSD")
            + self._valid_section(name="TWO", symbol="btcusd"))
        with self.assertRaisesRegex(ValueError, "duplicate provider/symbol"):
            ic.load_instruments(path, api=_Api())

    def test_inline_comments_and_false_boolean_parse_without_defaults(self):
        path = self._write(
            self._valid_section(extra='enabled = off # intentionally disabled\n')
            .replace('enabled = yes\n', ""))
        instrument = ic.load_instruments(path, api=_Api())["ONE"]
        self.assertFalse(instrument.enabled)
        self.assertEqual(instrument.isolation, "own_ledger")


class TestCurrentRegistry(unittest.TestCase):
    def test_all_enabled_mt_instruments_parse_numeric_params(self):
        instruments = ic.load_instruments()
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
        instruments = ic.load_instruments()
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
            "import sys, instruments_config; "
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
        import symbols as sym
        self.assertEqual(sym.btcsymbol, "BTCUSDC")
        self.assertEqual(sym.taosymbol, "TAOUSDC")
        self.assertEqual(sym.arbsymbol, "ARBUSDC")
        self.assertEqual(sym.hypesymbol, "HYPEUSDC")
        self.assertEqual(sym.symbols, ["BTCUSDC", "TAOUSDC", "ARBUSDC"])
        self.assertEqual(sym.forcesellsymbol, ["BTCUSDC", "TAOUSDC"])
        self.assertEqual(getattr(sym, "btcsymbol"), "BTCUSDC")
        with self.assertRaises(AttributeError):
            getattr(sym, "nonexistentsymbol")

    def test_ownership_inventory_symbols_fallback(self):
        from verify_tools.ownership_inventory import _python_symbols
        syms, tao = _python_symbols(ROOT)
        self.assertEqual(syms, ["BTCUSDC", "TAOUSDC", "ARBUSDC"])
        self.assertEqual(tao, "TAOUSDC")


def test_current_execution_policy_is_preserved():
    assert registry.symbols_for("binance") == ["BTCUSDC", "TAOUSDC", "ARBUSDC"]
    expected = {
        "tradeall_fire": {"BTCUSDC", "TAOUSDC", "ARBUSDC"}, "kalman_primary": {"BTCUSDC"},
        "mt": {"BTCUSDC", "TAOUSDC", "ARBUSDC"}, "assetguardian": {"BTCUSDC", "TAOUSDC"},
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
    assert all_syms == ["BTCUSDC", "TAOUSDC", "ARBUSDC", "HYPEUSD"]
    assert nb_syms == ["HYPEUSD"]
    assert nb_providers == {"HYPEUSD": "kraken"}
    arch_syms, arch_nb, arch_prov = registry.tracked_price_instruments(role="archive")
    assert arch_syms == ["BTCUSDC", "TAOUSDC", "ARBUSDC", "HYPEUSD"]
    assert arch_nb == ["HYPEUSD"]
    assert arch_prov == {"HYPEUSD": "kraken"}


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
        import symbols, tradeall, assetguardian
        from binance_api import trailing_stop, bapi_client
        from instruments_config import load_for
        assert "TESTUSDC" in symbols.symbols
        assert "TESTUSDC" in symbols.forcesellsymbol
        assert "TESTUSDC" in tradeall.TRADEALL_FIRE_SYMBOLS
        assert "TESTUSDC" in tradeall.KALMAN_PRIMARY_SYMBOLS
        assert tradeall._kalman_gate_blocks("TESTUSDC", "BUY") == (False, "off", None)
        assert "TESTUSDC" in assetguardian.TRACKED_SYMBOLS
        assert "ARBUSDC" in tradeall.TRADEALL_FIRE_SYMBOLS
        assert "ARBUSDC" not in assetguardian.TRACKED_SYMBOLS
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
        "instruments_config.discover_kraken_cache_pairs",
        lambda ws: [("BTC/USD", "BTC", "USD"), ("BTC/EUR", "BTC", "EUR")]
    )
    monkeypatch.setattr("instruments_config.load_registry", lambda path: {})
    res = ic.load_tradable_instruments(str(tmp_path), asset_scope="crypto")
    assert "KRAKEN_BTC" in res
    assert "KRAKEN_BTC/EUR" in res


if __name__ == "__main__":
    unittest.main()

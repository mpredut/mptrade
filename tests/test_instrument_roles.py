"""Coin onboarding changes one registry section, never a provider or trading test balance."""
import configparser
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

import instrument_registry as registry

ROOT = Path(__file__).resolve().parents[1]


def section(name="BINANCE_TEST", **overrides):
    values = dict(provider="binance", symbol="TESTUSDC", base="TEST", quote="USDC",
                  enabled="yes", isolation="own_ledger", market_hours="24x7")
    values.update({f"role.{role}": "no" for role in registry.ROLES})
    values.update(overrides)
    return f"[{name}]\n" + "\n".join(f"{key} = {value}" for key, value in values.items()) + "\n"


def test_current_execution_policy_is_preserved():
    assert registry.symbols_for("binance") == ["BTCUSDC", "TAOUSDC", "ARBUSDC"]
    expected = {
        "tradeall_fire": {"BTCUSDC", "TAOUSDC"}, "kalman_primary": {"BTCUSDC"},
        "mt": {"BTCUSDC", "TAOUSDC"}, "assetguardian": {"BTCUSDC", "TAOUSDC"},
        "archive": {"BTCUSDC", "TAOUSDC"}, "rtrade": {"TAOUSDC"},
        "force_sell": {"BTCUSDC", "TAOUSDC"},
        "trailing": {"BTCUSDC", "TAOUSDC", "ARBUSDC"},
    }
    for role, symbols in expected.items():
        assert set(registry.symbols_for("binance", role)) == symbols
    modes = {item.symbol: item.setting("tradeall.kalman_mode") for item in
             registry.select_instruments("binance", "tradeall_fire").values()}
    assert modes == {"BTCUSDC": "strict", "TAOUSDC": "permissive"}
    trails = {item.symbol: item.number("trailing.pct") for item in
              registry.select_instruments("binance", "trailing").values()}
    assert trails == {"BTCUSDC": 20, "TAOUSDC": 22, "ARBUSDC": 26}
    rebuys = {item.symbol: item.rebuy_mode() for item in
              registry.select_instruments("binance", "trailing").values()}
    assert rebuys == {"BTCUSDC": "on", "TAOUSDC": "on", "ARBUSDC": "auto"}
    assert set(registry.select_instruments(role="mt")) == {
        "BINANCE_BTC", "BINANCE_TAO", "KRAKEN_HYPE"}


@pytest.mark.parametrize("override", [
    {"role.trailing": "perhaps"}, {"role.trailng": "yes"},
    {"role.trailing": "yes"}, {"role.trailing": "yes", "trailing.pct": "nan"},
    {"role.trailing": "yes", "trailing.pct": "100"},
    {"role.trailing": "yes", "trailing.pct": "13"},   # valid pct but missing trailing.rebuy
    {"role.trailing": "yes", "trailing.pct": "13", "trailing.rebuy": "maybe"},
    {"role.tradeall_fire": "yes"},
    {"role.tradeall_fire": "yes", "tradeall.kalman_mode": "typo"},
    {"role.kalman_primary": "yes"}, {"symbol": "testusdc"}, {"quote": "USD"},
    {"trail.enabled": "yes"}, {"trail.pct": "13"}, {"tradeall.trade": "yes"},
])
def test_invalid_policy_fails_before_selection(tmp_path, override):
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
        assert "ARBUSDC" not in tradeall.TRADEALL_FIRE_SYMBOLS
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
    # A stale service environment must not override registry ownership silently.
    for filename, keys in {
        "config.env": ["TRADEALL_FIRE_SYMBOLS=", "KALMAN_PRIMARY_SYMBOLS=",
                       "KALMAN_GATE_MODE=", "AG_SYMBOLS="],
    }.items():
        content = (ROOT / filename).read_text()
        assert not any(key in content for key in keys)


def test_registry_refactor_preserves_existing_mt_values():
    cp = configparser.ConfigParser()
    cp.read(ROOT / "instruments.conf")
    assert cp["BINANCE_BTC"]["mt.buy_budget"] == "250"
    assert cp["BINANCE_TAO"]["mt.max_budget"] == "3500"
    assert cp["KRAKEN_HYPE"]["mt.max_budget"] == "5000"

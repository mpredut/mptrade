"""config.env is sourced by root shell daemons (pia_supervisor.sh, vpn_watchdog.sh) under `set -u` with no HOME.

Any `$VAR` reference in it silently kills those daemons (unbound variable with stderr discarded), so the file
must be sourceable from an empty environment. Use `~/...` for user paths (expanded by expand_config_path).
"""
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_config_env_sources_under_set_u_with_empty_environment():
    proc = subprocess.run(
        ["/bin/bash", "-c", "set -u; set -a; . ./config.env && echo SOURCE_OK"],
        cwd=ROOT, env={}, capture_output=True, text=True, timeout=20,
    )
    assert proc.returncode == 0 and "SOURCE_OK" in proc.stdout, proc.stderr


def test_config_module_parses_falsy_trade_enabled_fail_closed(tmp_path, monkeypatch):
    import config

    env_file = tmp_path / "config.env"
    txt_file = tmp_path / "config.txt"
    monkeypatch.setattr(config, "CONFIG_ENV_PATH", str(env_file))
    monkeypatch.setattr(config, "CONFIG_TXT_PATH", str(txt_file))
    monkeypatch.delenv("TRADE_ENABLED", raising=False)

    for raw in ("0", "false", '"false"', "'0'", "no", "off"):
        env_file.write_text(f"export TRADE_ENABLED={raw}\n", encoding="utf-8")
        config.load_config()
        assert config.is_trade_enabled() is False, f"Expected False for {raw}"

    env_file.write_text("TRADE_ENABLED=1\n", encoding="utf-8")
    txt_file.write_text("trade_enabled=0 # override\n", encoding="utf-8")
    config.load_config()
    assert config.is_trade_enabled() is False

    config.config_cache["trade_enabled"] = "0"
    assert config.is_trade_enabled() is False

    config.start_config_watcher()
    assert config.watcher_thread is not None and config.watcher_thread.is_alive()
    config.stop_config_watcher()
    assert config.watcher_thread is None


def test_shell_launchers_source_env_common_correctly():
    with open(os.path.join(ROOT, "kraken", "spcx_launch.sh"), encoding="utf-8") as fh:
        spcx_text = fh.read()
    assert 'source "$HERE/../orchestratorOS/lib/env_common.sh"\n\nSTRAT_ENTRY=800' in spcx_text

    with open(
        os.path.join(ROOT, "hyperliquid", "archive", "delta_neutral", "dn_close.sh"),
        encoding="utf-8",
    ) as fh:
        dn_text = fh.read()
    assert 'source "$HERE/../../../orchestratorOS/lib/env_common.sh"' in dn_text


def test_env_common_env_get_works_under_set_u_without_home():
    proc = subprocess.run(
        [
            "/bin/bash",
            "-c",
            "set -u; source ./orchestratorOS/lib/env_common.sh && env_get PIA_VPN_IF wgpia0 && echo ENV_GET_OK",
        ],
        cwd=ROOT,
        env={"PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert proc.returncode == 0 and "ENV_GET_OK" in proc.stdout, proc.stderr

    with open(os.path.join(ROOT, "orchestratorOS", "livecheck", "pia_supervisor.sh"), encoding="utf-8") as fh:
        pia_text = fh.read()
    assert pia_text.index('. "$ROOT/config.env"') < pia_text.index('. "$ROOT/.env"')
    assert 'if [ "$current_reg" = "$target_region" ] && vpn_healthy; then' in pia_text

    with open(os.path.join(ROOT, "orchestratorOS", "livecheck", "vpn_watchdog.sh"), encoding="utf-8") as fh:
        wd_text = fh.read()
    assert "pia.service enabled but inactive/failed while PIA is healthy -> starting it" in wd_text
    assert 'orch_state=$(systemctl show -p ActiveState --value python_orchestrator.service' in wd_text
    assert '[ "$orch_state" != "deactivating" ]' in wd_text


def test_watchdog_and_gateway_hardening(tmp_path):
    import pytest
    from orchestratorOS.lib import watchdog_common as wc
    from orchestratorOS.livecheck import os_resources_watchdog as rw
    from orchestratorTrade.autonomous_intent_gateway import AutonomousIntentGateway

    bad_state = tmp_path / "bad_state.json"
    bad_state.write_text("[1, 2, 3]", encoding="utf-8")
    assert wc.load_state(bad_state) == {}

    st = {"restart_history": {"bot.py": [9999999999.0]}}
    assert rw._restart_allowed(st, "bot.py", 1000.0) is True
    assert st["restart_history"]["bot.py"] == []

    with pytest.raises(ValueError):
        AutonomousIntentGateway(min_confidence=float("nan"))
    with pytest.raises(ValueError):
        AutonomousIntentGateway(min_confidence=1.5)
    with pytest.raises(ValueError):
        AutonomousIntentGateway(min_confidence=True)
    with pytest.raises(ValueError):
        AutonomousIntentGateway(max_notional_usd=True)


def test_botcore_env_helpers_handle_zero_booleans_and_non_finite_values():
    import pytest
    import botcore
    from rtrade_pair_store import RTradePairStore
    from strategies.rtrade_pair import OrderSnapshot, OrderTicket, PairPolicy, anchored_exit_price, quote_prices

    assert botcore.float_env("K", {"K": 0}) == 0.0
    assert botcore.float_env("K", {"K": 1.5}) == 1.5
    assert botcore.float_env("K", {"K": "2.5 # comment"}) == 2.5
    assert botcore.float_env("K", {"K": True}) is None
    assert botcore.float_env("K", {"K": "nan"}) is None
    assert botcore.float_env("K", {"K": "inf"}) is None

    assert botcore.required_env("K", {"K": 0}) == "0"
    assert botcore.defined_env("K", {"K": 0}) == "0"
    assert botcore.required_float_env("K", {"K": 0}) == 0.0
    assert botcore.required_int_env("K", {"K": 0}) == 0
    with pytest.raises(ValueError):
        botcore.required_float_env("K", {"K": True})
    with pytest.raises(ValueError):
        RTradePairStore._finite_positive(True, "qty")
    with pytest.raises(ValueError):
        RTradePairStore._finite_positive(False, "qty")
    with pytest.raises(ValueError):
        OrderTicket(order_id="1", side="BUY", price=True, qty=1.0)
    with pytest.raises(ValueError):
        OrderSnapshot(status="open", filled_qty=True)
    with pytest.raises(ValueError):
        PairPolicy(adjustment_fraction=True)
    with pytest.raises(ValueError):
        quote_prices(float("nan"), 0.01)
    with pytest.raises(ValueError):
        anchored_exit_price("SELL", 100.0, float("nan"), 0.01, 0.01)


def test_shadow_and_outcome_log_hardening(tmp_path, monkeypatch):
    import json
    import pytest
    import order_outcomes_log
    import shadow_runtime
    import shadow_signals

    monkeypatch.setenv("TEST_SHADOW_FLOAT", "nan")
    assert shadow_signals._f_env("TEST_SHADOW_FLOAT", 15.0) == 15.0
    monkeypatch.setenv("TEST_SHADOW_FLOAT", "-1")
    assert shadow_signals._f_env("TEST_SHADOW_FLOAT", 15.0) == 15.0

    tracker = shadow_signals.KalmanStateTracker(state_path=str(tmp_path / "kalman.json"))
    tracker._last_state_write = 9999999999.0
    tracker._last_fed["BTCUSDC"] = 9999999999.0
    tracker._last_vol_time["BTCUSDC"] = 9999999999.0
    tracker.update("BTCUSDC", 1000.0, 50000.0, 0.01, [50000.0], 1.0)
    assert tracker._last_fed["BTCUSDC"] == 1000.0
    assert tracker._last_state_write == 1000.0

    with pytest.raises(ValueError):
        shadow_runtime.require_shadow_interval(True, 1.0, "test")

    outcomes_file = tmp_path / "outcomes.jsonl"
    monkeypatch.setattr(order_outcomes_log, "_OUTCOMES_PATH", str(outcomes_file))
    order_outcomes_log.log_order_outcome(
        venue="binance",
        symbol="BTCUSDC",
        side="BUY",
        qty=0.01,
        price=50000.0,
        decision="ALLOWED",
        source="test",
        regime_context={"regime": "BULL", "strength": "invalid_float"},
    )
    rows = [json.loads(line) for line in outcomes_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["regime"] == "BULL"
    assert rows[0]["regime_strength"] is None




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

    from types import SimpleNamespace
    monkeypatch.setattr(order_outcomes_log, "ORDER_OUTCOMES_LOG_DIR", str(tmp_path))
    order_outcomes_log.log_order_outcome(
        symbol="BTCUSDC",
        side="BUY",
        price=50000.0,
        qty=0.01,
        outcome="accepted",
        refuse_reason="",
        motivation="test",
        caller="test_suite",
        regime_context=SimpleNamespace(regime="BULL", strength="invalid_float"),
    )
    log_files = list(tmp_path.glob("order_outcomes_*.log"))
    assert len(log_files) == 1
    lines = log_files[0].read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert "|BULL||" in lines[0]


def test_position_exit_manager_and_audit_hardening(tmp_path, monkeypatch):
    from decimal import Decimal
    import pytest
    import active_intents
    import binance_cache_health
    from liquidation_signal import LiquidationSignal
    from providers import binance_filters, execution_audit
    from strategies.position_exit_manager import (
        PositionExitManager,
        PositionSnapshot,
        ReentrySnapshot,
    )

    monkeypatch.setenv("EXECUTION_AUDIT_CACHE_MAX", "5000 # inline comment")
    assert execution_audit._parse_cache_max() == 5000
    monkeypatch.setenv("EXECUTION_AUDIT_CACHE_MAX", "bad_int")
    assert execution_audit._parse_cache_max() == 10000

    with pytest.raises(binance_filters.BinanceFilterError):
        binance_filters.decimal_places(Decimal("NaN"))
    with pytest.raises(ValueError):
        binance_cache_health._positive_finite(True, "test")

    summary = active_intents._build_summary(
        intents=[{"symbol": "BTCUSDC", "side": "BUY", "requested_qty": "bad", "executed_qty": None, "requested_price": "nan"}],
        exposures=[],
    )
    assert summary["pending_notional"]["total_buy"] == 0.0

    mgr = PositionExitManager()
    mgr.bind_state({
        "sl_episode_state": {"BTCUSDC": {"started_at": 1000.0, "buy_price": 100.0, "disaster_floor": "0.0"}},
        "sl_bounce_state": {"BTCUSDC": {"started_at": 1000.0, "buy_price": 100.0, "lowest_price": None, "peak_bounce_price": None}},
        "tp_ride_state": {"ETHUSDC": {"started_at": 1000.0, "buy_price": 100.0, "peak_price": None, "peak_gain": None}},
        "reentry_state": {"BTCUSDC": {"trough": None, "peak": None, "peak_bounce": None, "started_at": 1000.0}},
    })
    sl_dec = mgr.evaluate_stop_loss(
        PositionSnapshot(symbol="BTCUSDC", current_price=94.0, avg_buy_price=100.0, current_time_s=1010.0, trend_up=True)
    )
    assert sl_dec.should_exit is False

    tp_dec = mgr.evaluate_take_profit(
        PositionSnapshot(symbol="ETHUSDC", current_price=108.0, avg_buy_price=100.0, current_time_s=1010.0, trend_up=True)
    )
    assert tp_dec.should_exit is False

    re_dec = mgr.evaluate_reentry(
        ReentrySnapshot(symbol="BTCUSDC", current_price=95.0, last_sell_price=100.0, last_exit_kind="STOP", current_time_s=1010.0)
    )
    assert re_dec.should_enter is False

    sig = LiquidationSignal(symbols=["BTCUSDT"])
    sig._ingest("[]")
    sig._ingest('{"topic":"allLiquidation.BTCUSDT","data":[{"s":"BTCUSDT","S":"Unknown","v":"1","p":"50000"}]}')
    sig._ingest('{"topic":"allLiquidation.BTCUSDT","data":[{"s":"BTCUSDT","S":"Buy","v":"-1","p":"-50000"}]}')
    assert sig.get_signal("BTCUSDT")["long_count"] == 0
    assert sig.get_signal("BTCUSDT")["short_count"] == 0


def test_alertnotifiers_and_hl_hardening(tmp_path, monkeypatch):
    import importlib.util
    import json
    import math
    from datetime import datetime, timezone
    from pathlib import Path
    from notify_engine import alertnotifiers

    hl_dir = Path(__file__).resolve().parent.parent / "hyperliquid"
    monkeypatch.syspath_prepend(str(hl_dir))

    def _load_hl_mod(name: str):
        spec = importlib.util.spec_from_file_location(f"_hl_{name}", str(hl_dir / f"{name}.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    hl_client = _load_hl_mod("hl_client")
    hl_signals = _load_hl_mod("signals")
    hl_pa = _load_hl_mod("price_analysis")

    state_file = tmp_path / "notif_state.json"
    monkeypatch.setenv("NOTIFICATION_STATE_FILE", str(state_file))
    monkeypatch.setenv("NOTIFICATION_GATE_SECONDS", "300")

    now_ts = 1700000000.0
    today = datetime.fromtimestamp(now_ts, timezone.utc).date().isoformat()
    # Corrupted / future clock-skewed state
    state_file.write_text(json.dumps({
        "date_utc": today,
        "channels": {
            "ntfy": {
                "sent": "bad",
                "last": {"fp1": now_ts + 9999.0, "fp2": "not_a_float"},
                "last_gate_ts": now_ts + 3600.0,
                "pending_bulk": ["not_a_dict"],
            }
        },
    }), encoding="utf-8")

    allowed, reason, _, payload = alertnotifiers.pass_common_gate(
        "ntfy", "Test Alert", "Body", now=now_ts
    )
    assert allowed is True
    assert reason == "reserved"
    assert payload is not None

    msg = alertnotifiers.AlertNotifier.format_new_coin_message(
        {"symbol": "ABC", "price": "1.23456"}
    )
    assert "Price: $1.2346" in msg

    assert math.isnan(hl_client._round_px(float("nan"), 2))

    sig_file = tmp_path / "signal.json"
    sig_file.write_text(json.dumps({"trend": "up", "confidence": 5.0, "ts": now_ts + 999999999.0}), encoding="utf-8")
    monkeypatch.setenv("SIGNAL_MAX_AGE_MIN", "30")
    res_future = hl_signals._file_signal(str(sig_file))
    assert res_future["trend"] == "neutral"

    sig_file.write_text(json.dumps({"trend": "up", "confidence": 5.0}), encoding="utf-8")
    res_clamped = hl_signals._file_signal(str(sig_file))
    assert res_clamped["confidence"] == 1.0

    monkeypatch.setenv("ANALYSIS_WINDOW_H", "2")
    monkeypatch.setenv("ANALYSIS_STEP_H", "1")
    class _DummyClient:
        def candles(self, coin, interval, lookback_hours):
            return [{"t": "bad", "c": "10.0"}, {"t": 1000, "c": float("nan")}]
        def mid(self, coin):
            return float("nan")
    assert hl_pa.analyze(_DummyClient(), "HYPE") is None


def test_trailing_and_shadow_hardening(tmp_path, monkeypatch):
    import pytest
    from binance_api import market_data as b_md
    from binance_api import trailing_stop as b_ts
    from kraken import shadow_live

    with pytest.raises(ValueError):
        b_md.parse_interval_minutes(True)
    with pytest.raises(ValueError):
        b_ts._finite(True, name="test_bool")

    monkeypatch.setattr(shadow_live, "LOG_DIR", str(tmp_path))
    anchor_file = tmp_path / "HYPEUSD_240m.anchor"
    anchor_file.write_text("corrupted_int", encoding="utf-8")
    assert shadow_live._get_anchor("HYPEUSD", 240, 1700000000) == 1700000000

    hist_file = tmp_path / "HYPEUSD_240m.ohlc.json"
    hist_file.write_text('[[1700000000, 10, 12, 9, 11], ["bad"]]', encoding="utf-8")
    loaded = shadow_live._load_history("HYPEUSD", 240)
    assert loaded == [(1700000000, 10.0, 12.0, 9.0, 11.0)]


def test_market_monitor_and_pricewindow_hardening():
    import time
    import pytest
    import tradeall_price_archiver
    from market_monitor import alerts_config
    from market_monitor.pricechecker import PriceChecker
    from pricewindow import IncrementalTrendBuffer, PriceWindow

    with pytest.raises(ValueError):
        alerts_config._pair("-5 / 10")
    with pytest.raises(ValueError):
        alerts_config._pair("nan / 10")
    with pytest.raises(ValueError):
        tradeall_price_archiver._positive_finite(True, "test")

    checker = PriceChecker(cachePriceAll=None)
    checker._last_alert_time["BTC_up"] = time.time() + 7200.0
    assert checker._should_send_alert("BTC", "up") is True

    buf = IncrementalTrendBuffer()
    buf.sync_entries([[1000, 50.0], [2000, 51.0]])
    buf.sync_entries([["bad_ts", "bad_px"], [2000, 51.0], [3000, 52.0]])
    assert len(buf.prices) == 2

    pw = PriceWindow("BTCUSDC", 0)
    assert pw.window_size == 1


def test_cooldown_and_internal_state_hardening(tmp_path):
    import numpy as np
    from lock.cooldown import Cooldown
    from intelligence.internal.state.volatility import RollingVolatilityTracker, adaptive_thresholds
    from intelligence.internal.state.persistence import classify_hurst_regime
    from intelligence.internal.state.survival import hybrid_T

    cd = Cooldown("test", base_dir=str(tmp_path))
    cd._write({"BTCUSDC": {"timestamp": None}, "ETHUSDC": "corrupted", "SOLUSDC": {"timestamp": 9999999999.0}})
    assert cd.reserve("BTCUSDC", ttl=60)[0] is True
    assert cd.reserve("SOLUSDC", ttl=60)[0] is True
    assert cd.last_age("ETHUSDC") is None
    cd.update("ETHUSDC", order_id=1)

    tracker = RollingVolatilityTracker(window_size=25)
    for i in range(22):
        tracker.add_price(100.0 + i * 0.1, sample_rate_sec=60.0)
    assert tracker.add_price(float("nan"), sample_rate_sec=60.0) is None
    assert tracker.add_price(True, sample_rate_sec=60.0) is None
    vol = tracker.add_price(103.0, sample_rate_sec=60.0)
    assert vol is not None and np.isfinite(vol)

    assert adaptive_thresholds(float("nan")) == (None, None)
    assert adaptive_thresholds(True) == (None, None)

    assert classify_hurst_regime(float("nan")) == "unknown"
    assert classify_hurst_regime(True) == "unknown"

    durs = [24.0, 48.0, float("nan"), -5.0, 72.0]
    t_res = hybrid_T(durs)
    assert isinstance(t_res, dict) and isinstance(t_res["T"], int) and t_res["n"] == 3


def test_triggers_guards_and_collectors_hardening(tmp_path):
    import json
    import time
    import numpy as np
    from intelligence.internal.guards.guard_decision import BrakeAction, GuardDecision
    from intelligence.internal.guards.trend_significance_guard import TrendSignificanceGuard
    from intelligence.internal.triggers.gradient_trigger import LinearGradientTrigger
    from intelligence.internal.triggers.kalman_trigger import KalmanTrend
    from intelligence.internal.triggers.mean_reversion_trigger import MeanReversionTrigger
    from intelligence.internal.triggers.trigger_event import TriggerAction, TriggerEvent, TriggerSide
    from intelligence.sentiment.collectors.fear_greed_collector import FearGreedCollector
    from intelligence.sentiment.collectors.market_breadth_collector import MarketBreadthCollector
    from intelligence.macro.news_feed_collector import NewsFeedCollector

    dec_nan = GuardDecision.downscale("test_guard", float("nan"), "nan_scale")
    assert dec_nan.allowed is False
    assert dec_nan.suggested_scale == 0.0
    assert dec_nan.brake_action == BrakeAction.HARD_VETO

    tsg = TrendSignificanceGuard(max_p_value=0.05)
    ev = TriggerEvent(action=TriggerAction.ENTRY, side=TriggerSide.BUY, source="t", symbol="BTCUSDC", price=100.0, strength=1.0)
    assert tsg.evaluate(ev, np.array([100.0 + i for i in range(12)])).allowed is True

    lgt = LinearGradientTrigger(strength_threshold=2.0)
    assert lgt.evaluate("BTCUSDC", float("nan"), 3.0, 1.0) == ("unknown", None)
    assert lgt.last_state == "neutral"

    kt = KalmanTrend()
    assert kt.update(None, 100.0, 1.0)["vel"] == 0.0
    assert kt.update(1000.0, True, 1.0)["vel"] == 0.0

    mrt = MeanReversionTrigger(mode="rsi", rsi_period=3)
    for i in range(6):
        res_ev = mrt.update("BTCUSDC", 1000.0 + i, 100.0)
    assert res_ev.metadata["rsi"] == 50.0
    assert res_ev.side == TriggerSide.HOLD

    ev_bool = TriggerEvent(action=TriggerAction.ENTRY, side=TriggerSide.BUY, source="t", symbol="BTCUSDC", price=True, strength=1.0)
    assert ev_bool.is_executable() is False

    assert FearGreedCollector.parse_payload(None) is None
    assert FearGreedCollector.parse_payload({"data": [{"value": None}]}) is None

    tickers = [{"symbol": f"COIN{i}USDT", "quoteVolume": "10000000", "priceChangePercent": "2.0"} for i in range(6)]
    tickers.append({"symbol": "BADUSDT", "quoteVolume": "inf", "priceChangePercent": "nan"})
    mb_snap = MarketBreadthCollector.parse_tickers(tickers)
    assert mb_snap is not None and mb_snap.total_count == 6

    hist_file = tmp_path / "news_feed_collect.json"
    now = time.time()
    hist_file.write_text(
        json.dumps([
            {"first_seen_ts": "corrupted", "title": "Bad"},
            {"first_seen_ts": now - 10, "title": "Valid Headline", "matched_keywords": None, "is_high_severity": True},
        ]),
        encoding="utf-8",
    )
    nfc = NewsFeedCollector(history_file=str(hist_file))
    assert len(nfc.get_pool_snapshot(now_ts=now).headlines) == 1


def test_agy_store_and_order_persistence_hardening(tmp_path, monkeypatch):
    import accepted_order_persistence as aop
    from lock import trade_cooldown as tc
    from orchestratorOS.agy.summary_store import _drop_summaries, parse_db_timestamp, resolve_project_ids, write_annotation_title, read_annotation_title
    from orchestratorOS.agy.topics import condense_decision, load_topic_registry, save_topic_registry

    assert parse_db_timestamp(None) is None
    assert parse_db_timestamp(12345) is None
    assert parse_db_timestamp("2026-10-26 04:00:00.12Z") is not None

    cli_dir = tmp_path / "store" / "antigravity-cli"
    proj_dir = tmp_path / "store" / "config" / "projects"
    proj_dir.mkdir(parents=True)
    (proj_dir / "bad.json").write_text("[1, 2, 3]", encoding="utf-8")
    (proj_dir / "good.json").write_text('{"id": "p1", "name": "mptrade"}', encoding="utf-8")
    assert "p1" in resolve_project_ids(["mptrade"], [str(cli_dir)])

    write_annotation_title(str(cli_dir), "cid1", 'Test "Title"')
    assert read_annotation_title(str(cli_dir), "cid1") == 'Test "Title"'
    _drop_summaries(str(cli_dir), set())

    from state_io import stage_previous_generation
    reg_file = tmp_path / "topics.json"
    save_topic_registry({"macro_sentiment": {"conversation_id": "c1"}}, str(reg_file))
    stage_previous_generation(str(reg_file))
    reg_file.unlink()
    reg_file.write_text("{corrupted", encoding="utf-8")
    loaded_reg = load_topic_registry(str(reg_file))
    assert loaded_reg.get("macro_sentiment", {}).get("conversation_id") == "c1"
    assert len(condense_decision("A very long decision string here", max_len=2)) == 4

    monkeypatch.setattr(tc, "STATE_FILE", str(tmp_path / "tc.json"))
    monkeypatch.setattr(tc, "LOCK_FILE", str(tmp_path / "tc.lock"))
    tc._cooldown()._write({"BTCUSDC": {"timestamp": "bad"}})
    assert "no order recorded" in tc.describe_last_trade("BTCUSDC")

    class _DummyRetry:
        @staticmethod
        def order_id_from_response(o):
            return "ord1"
        @staticmethod
        def load_all():
            return ["not_a_dict", {"id": "c1", "place_kwargs": "bad"}]

    assert aop._accepted_is_durable(_DummyRetry, "not_a_dict", {}) is False
    assert aop._accepted_is_durable(_DummyRetry, {"id": "c1", "place_kwargs": "bad"}, {}) is False



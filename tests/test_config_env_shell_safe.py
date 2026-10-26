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


def test_order_id_context_and_pricefetcher_hardening(tmp_path):
    from binance_api import bapi_trades
    from binance_api import order_id_context as oic
    from market_monitor import pricefetcher as pf

    assert oic._base36(-5) == "0"
    cid = oic.create_client_order_id("VERY_LONG_PROCESS_NAME_EXCEEDING_THIRTY_SIX_CHARS")
    assert len(cid) <= oic.CLIENT_ORDER_ID_MAX_LEN

    assert pf.is_valid_symbol_for_monitoring(123) is False
    cmc = pf.CoinMarketCapPricePlatform(api_key=None)
    assert cmc._extract_usd_price("BTC", {"data": None}) is None
    assert cmc._extract_usd_price("BTC", {"data": {"BTC": {"quote": None}}}) is None
    assert cmc._extract_usd_price("BTC", {"data": {"BTC": {"quote": {"USD": None}}}}) is None

    bad_file = tmp_path / "trades.json"
    bad_file.write_text('{"not": "a_list"}', encoding="utf-8")
    bapi_trades.load_trades_from_file(str(bad_file))
    assert bapi_trades.trade_cache == []


def test_price_analysis_market_regime_and_monitororder_hardening(monkeypatch):
    from market_regime import MarketRegimeContext, MarketRegimeDecision, MarketRegimeEvaluator, MarketRegimeService
    import monitororder as mo
    import priceAnalysis as pa
    from strategies.position_exit_manager import PositionExitConfig, PositionExitManager, PositionSnapshot

    # 1. PositionExitManager preserves t=0.0 episode/bounce/peak timestamps across simulation ticks
    mgr = PositionExitManager(
        config=PositionExitConfig(
            lost_threshold=0.05,
            sl_trend_defer_max_sec=100.0,
            tp_stagnation_max_sec=100.0,
            allow_external_cache=False,
        )
    )
    snap0 = PositionSnapshot(symbol="BTCUSDC", current_price=94.0, avg_buy_price=100.0, current_time_s=0.0, trend_up=True)
    mgr.evaluate_stop_loss(snap0)
    assert mgr.sl_episode_state["BTCUSDC"]["started_at"] == 0.0
    assert mgr.sl_bounce_state["BTCUSDC"]["started_at"] == 0.0
    snap50 = PositionSnapshot(symbol="BTCUSDC", current_price=94.0, avg_buy_price=100.0, current_time_s=50.0, trend_up=True)
    mgr.evaluate_stop_loss(snap50)
    assert mgr.sl_episode_state["BTCUSDC"]["started_at"] == 0.0
    assert mgr.sl_bounce_state["BTCUSDC"]["started_at"] == 0.0

    # 2. MarketRegimeEvaluator & MarketRegimeService reject bool inputs
    ev = MarketRegimeEvaluator()
    assert ev.evaluate({"gradient_recent": True, "epsilon": 0.01}).fresh is False
    svc = MarketRegimeService()
    assert svc.evaluate_closes([100.0, 101.0, True]).fresh is False
    ctx = MarketRegimeContext(
        decision=MarketRegimeDecision("bull", 0.01, 0.001, 10.0, True, "ok"),
        resolved_trend="bull",
        evaluated_at=1000.0,
    )
    assert ctx.is_valid_for(now=True) is False
    assert ctx.is_valid_for(now="bad") is False

    # 3. priceAnalysis.detect_long_term_trend filters non-finite points
    assert pa.detect_long_term_trend([1.0, 2.0, float("nan")], [100.0, float("nan"), 102.0]) is None

    # 4. monitororder skips invalid/zero/bool current_price without raising
    monkeypatch.setattr(mo.api, "get_open_orders", lambda side, sym: {"1": {"price": 100.0, "quantity": 1.0}})
    monkeypatch.setattr(mo.api, "get_current_price", lambda sym: 0.0)
    mo.monitor_open_orders_by_type("BTCUSDC", "SELL")


def test_qlimit_tradeall_and_order_retry_hardening(monkeypatch):
    from types import SimpleNamespace
    import order_retry
    import qlimit_order_guard as qog
    import tradeall as ta

    # 1. order_retry._client_order_id length <= 36 and bool rejection in valid_record / price_gate_ok
    cid = order_retry._client_order_id("a" * 32, 123, prefix="VERY_LONG_PREFIX_EXCEEDING_LIMIT_36")
    assert len(cid) <= 36
    assert order_retry.valid_record({"id": "1", "symbol": "BTCUSDC", "side": "BUY", "qty": True, "created_ts": 1000.0}) is False
    assert order_retry.price_gate_ok({"side": "BUY", "requested_price": 85000.0}, True) is False
    assert order_retry.price_gate_ok({"side": "BUY", "requested_price": True}, 1.0) is False

    # 2. tradeall._kalman_gate_blocks normalizes lowercase action and fails closed on NaN/None age for BUY
    monkeypatch.setitem(ta.KALMAN_MODES, "BTCUSDC", "strict")
    monkeypatch.setattr(ta, "_kalman_ref", SimpleNamespace(current_trend=lambda sym: (1, float("nan"))))
    blocked_buy, _, _ = ta._kalman_gate_blocks("BTCUSDC", "buy")
    blocked_sell, _, _ = ta._kalman_gate_blocks("BTCUSDC", "sell")
    assert blocked_buy is True
    assert blocked_sell is False

    # 3. tradeall.track_and_place_order rejects bool/NaN prices without canceling live orders
    assert ta.track_and_place_order("BUY", "BTCUSDC", 1, True, 85000.0, order_ids=["live1"]) == []

    # 4. qlimit_order_guard falls back to _resolve_trend_duration when trend_duration_seconds is NaN/bool
    called = []
    monkeypatch.setattr(qog, "_resolve_trend_duration", lambda sym: called.append(sym) or 0.0)
    qog.check_qlimit2_order_guards(
        "BTCUSDC",
        "BUY",
        85000.0,
        price_history=[],
        trend_duration_seconds=float("nan"),
        margins={
            "intelligence_guards_mode": "shadow",
            "internal_guard_mode": "shadow",
            "parabolic_surge_pct": 10.0,
            "parabolic_pullback_pct": 2.0,
            "weibull_exhaustion_policy": "downscale",
            "weibull_exhausted_scale": 0.5,
            "weibull_p90_days": 14.0,
        },
    )
    assert called == ["BTCUSDC"]


def test_assetguardian_rtrade_and_spot_rules_hardening(monkeypatch):
    from types import SimpleNamespace
    import assetguardian as ag
    import rtrade as rt
    from strategies import spot_rules as sr

    # 1. assetguardian._finite_float rejects bool and _completed_tier_values / _safe_tier_attempt handle malformed state
    assert ag._finite_float(True, positive=True) is None
    assert ag._completed_tier_values({"completed_tiers": [True, 2.5, "bad"]}) == {2.5}
    assert ag._completed_tier_values("corrupted") == set()
    assert ag._safe_tier_attempt({"2.5": "bad"}, "2.5") == 1
    assert ag._safe_tier_attempt({"2.5": True}, "2.5") == 1

    # 2. rtrade._followup_exit_price and _whale_flow_blocks_side handle NaN/None safely
    monkeypatch.setattr(rt.api, "get_current_price", lambda sym: float("nan"))
    sell_px = rt._followup_exit_price("TAOUSDC", "SELL", 200.0)
    assert sell_px > 200.0

    monkeypatch.setattr(rt, "_is_gi_whale_enforced", lambda bg=None: True)
    import external_order_guard as eog
    monkeypatch.setattr(
        eog,
        "get_whale_collector",
        lambda: SimpleNamespace(
            fetch=lambda sym, allow_network=False: SimpleNamespace(
                taker_buy_sell_ratio=None,
                open_interest_1h_change_pct=None,
                divergence_regime="aggressive_shorting",
            )
        ),
    )
    blocked, reason = rt._whale_flow_blocks_side("TAOUSDC", "BUY")
    assert blocked is True
    assert "aggressive_shorting_divergence" in reason

    # 3. spot_rules reentry barriers fail closed on NaN price and reject bool/None in exhaustion checks
    assert sr.reentry_stop_blocked(float("nan"), 100.0, 2.0, 0.05) is True
    assert sr.reentry_drop_blocked(float("nan"), 100.0, 2.0, 0.05) is True
    assert sr.reentry_hybrid_blocked(float("nan"), 100.0, 105.0, True, 2.0, 1.0)[0] is True
    assert sr.dynamic_surge_gain_pct(True, fallback_gain_pct=25.0) == 25.0
    assert sr.dynamic_flat_tp_pct(True, base_tp_pct=5.0) == 5.0
    assert sr.dynamic_trend_trail_pct(float("nan"), base_trail_pct=8.0) == 8.0
    assert sr.check_fast_profit_reversal(None, 100.0, 5.0, 2.0, [], 60.0, 1.0) == (False, 0.0, 0.0)
    assert sr.check_parabolic_surge_exhaustion(None, 100.0, 130.0, 25.0, 2.0) == (False, "")
    assert sr.check_slow_grind_exhaustion(None, 100.0, 1000.0, 900000.0, 7.0, 15.0, 120.0, [], 900.0, 1.5, 2.5) == (False, "")


def test_external_order_guard_rtrade_pair_and_monitortrades_hardening(monkeypatch):
    import pytest
    import external_order_guard as eog
    import monitortrades as mt
    from strategies.rtrade_pair import OrderSnapshot, OrderTicket, PairCoordinator, PairPolicy
    from strategies.position_exit_manager import ExitAction, ExitDecision, PositionSnapshot

    # 1. external_order_guard.check_external_order_guards fails fast on empty margins and works with explicit keys
    with pytest.raises(KeyError):
        eog.check_external_order_guards("BTCUSDC", "BUY", 85000.0, margins={})
    ok, reason, scale = eog.check_external_order_guards(
        "BTCUSDC",
        "BUY",
        85000.0,
        margins={
            "shadow_notify": False,
            "whale_guard_mode": "off",
            "orderbook_wall_guard_mode": "off",
            "funding_guard_mode": "off",
        },
    )
    assert (ok, reason, scale) == (True, "ok", 1.0)

    # 2. PairCoordinator rejects bool qty/mid/checkpoint fields and derives fallback cost from ticket.price
    policy = PairPolicy(adjustment_fraction=0.01)
    with pytest.raises(ValueError):
        PairCoordinator(None, qty=True, policy=policy)
    pc = PairCoordinator(None, qty=1.0, policy=policy)
    with pytest.raises(ValueError):
        pc.start(mid=True)
    with pytest.raises(ValueError):
        PairCoordinator.from_state(
            None,
            policy,
            {
                "pair_id": "p1",
                "qty": 1.0,
                "start_side": "BUY",
                "phase": "quoting",
                "tickets": [],
                "hard_stop_revision": -1,
            },
        )
    pc.tickets = [OrderTicket(order_id="o1", side="BUY", price=100.0, qty=2.0)]
    pc.snapshots = {"o1": OrderSnapshot(status="closed", filled_qty=2.0, cost=0.0, fee=0.0)}
    assert pc._side_totals("BUY") == (2.0, 200.0, 0.0)

    # 3. monitortrades._log_tp_hold_state and _log_sl_hold_state handle None values in state dicts safely
    snap = PositionSnapshot(symbol="BTCUSDC", current_price=108.0, avg_buy_price=100.0, current_time_s=1000.0)
    monkeypatch.setitem(mt._TP_RIDE_STATE, "BTCUSDC", {"peak_price": None, "last_peak_time": None, "peak_gain": None})
    mt._log_tp_hold_state("BTCUSDC", snap, ExitDecision(ExitAction.HOLD, False, "riding"), 0.07)
    monkeypatch.setitem(mt._SL_BOUNCE_STATE, "BTCUSDC", {"peak_bounce_price": None, "started_at": None})
    mt._log_sl_hold_state("BTCUSDC", snap, ExitDecision(ExitAction.HOLD, False, "bounce", metadata={"disaster_floor": None}), 0.05)

    # 4. order_rollback.extract_order_restore_params and restore_cancelled_orders reject NaN/bool/inf
    import order_rollback as orb

    nan_params = orb.extract_order_restore_params({"side": "SELL", "price": float("nan"), "qty": 2.0, "amount": True})
    assert nan_params["price"] == 0.0
    assert nan_params["amount"] == 0.0
    called = []
    assert orb.restore_cancelled_orders(
        [{"side": "SELL", "price": float("nan"), "qty": 2.0}, {"side": "SELL", "price": 100.0, "qty": 2.0, "amount": True}],
        restore_fn=lambda p: called.append(p) or True,
    ) == 1
    assert len(called) == 1 and called[0]["amount"] == 200.0


def test_spot_engine_trailing_core_and_pricewindow_hardening():
    import pricewindow as pw
    import trailing_core as tc
    from providers.strategy_executor import PairPrecision
    from strategies import spot_engine as se

    # 1. spot_engine._qty_for, _dust_safe_qty, _place, and step reject NaN/bool/<=0 safely
    class _DummyClient:
        def pair_precision(self, _pair):
            return PairPrecision(2, 4, 0.01, "BTC")

        def free_balance(self, _asset):
            return 1000.0

    params = se.StratParams(
        currency="EUR",
        entry_amount=100.0,
        entry_discount_pct=1.0,
        dca_amount=100.0,
        dca_drop_pct=2.0,
        check_minutes=1.0,
        takeprofit_pct=3.0,
        max_budget=500.0,
        max_dca_buys=3,
        enable_takeprofit=True,
        order_ttl_min=60.0,
    )
    strat = se.Strategy(_DummyClient(), "BTCEUR", params, dry_run=True, initial_state=se._new_state(), replay_mode=True)
    assert strat._qty_for(float("nan"), 100.0) == 0.0
    assert strat._qty_for(100.0, float("inf")) == 0.0
    assert strat._dust_safe_qty(float("nan")) == 0.0
    assert strat._dust_safe_qty(True) == 0.0
    assert strat._place("buy", float("nan"), 100.0, kind="ENTRY") is False
    assert strat._place("buy", 1.0, 0.0, kind="ENTRY") is False
    strat.step(float("nan"))
    strat.step(0.0)
    assert len(strat._shadow_prices) == 0

    # 2. trailing_core handles free_qty=None/NaN and corrupted rebuy low/qty safely
    class _DummyAdapter:
        def free_qty(self, _asset):
            return None

        def price(self, _pair):
            return 100.0

        def trend(self, _pair):
            return 0.0

        def log_hold(self, *_a):
            pass

    core = tc.TrailingCore(
        _DummyAdapter(),
        log=lambda _m: None,
        enabled=False,
        state_file="dummy.json",
        min_notional=10.0,
        rebuy_enabled=True,
        rebuy_bounce_pct=2.0,
        rebuy_skip_if_trend_down=False,
    )
    state = {"BTC": {"peak": 110.0, "rebuy": {"low": None, "qty": float("nan")}}}
    core._process("BTC", "BTC", "BTCEUR", 5.0, state)
    assert "rebuy" not in state["BTC"]

    # 3. pricewindow.calculate_acceleration handles non-finite window_sec and sample_rate_sec
    w = pw.PriceWindow("BTCUSDC", 20, sample_rate_sec=1.0, initial_prices=[100.0, 101.0, 102.0, 103.0])
    assert w.calculate_acceleration(window_sec=float("inf")) != (float("nan"), float("nan"))
    assert w.calculate_acceleration(window_sec=True) != (float("nan"), float("nan"))


def test_shadow_signals_utils_and_volatility_hardening():
    import math
    import shadow_signals as ss
    import utils as u
    from intelligence.internal.state import volatility as vol

    # 1. shadow_signals.KalmanStateTracker.current_trend and update handle None/NaN/bool safely
    tracker = ss.KalmanStateTracker(state_path=None, compute_vol=False)
    tracker._state["BTCUSDC"] = {"kalman_trend": 1, "ts": None}
    assert tracker.current_trend("BTCUSDC") == (None, 1e18)
    out = tracker.update("BTCUSDC", 100.0, 85000.0, float("nan"), [], float("nan"))
    assert math.isfinite(out["kalman_vel"]) and math.isfinite(out["kalman_vel_std"])
    out_bad = tracker.update("BTCUSDC", 200.0, float("nan"), 1.0, [], 1.0)
    assert out_bad["kalman_trend"] == out["kalman_trend"]

    # 2. utils.gaussian_weights_from_idx handles non-finite T/idx and negative idx
    t_arr, w_arr = u.gaussian_weights_from_idx(float("inf"), 0)
    assert len(t_arr) == 0 and len(w_arr) == 0
    t_arr2, w_arr2 = u.gaussian_weights_from_idx(10, -3)
    assert len(t_arr2) == 10 and abs(float(w_arr2.sum()) - 1.0) < 1e-9

    # 3. volatility.calculate_volatility_1h and adaptive_thresholds handle non-finite/bool args
    assert vol.calculate_volatility_1h([100.0] * 25, float("nan")) is None
    assert vol.calculate_volatility_1h([100.0] * 25, True) is None
    assert vol.adaptive_thresholds(2.5, k_reentry=float("nan"), k_dca=1.0) == (None, None)
    assert vol.adaptive_thresholds(2.5, k_reentry=True, k_dca=1.0) == (None, None)


def test_intelligence_cli_macro_analyzer_and_instrument_hardening(tmp_path):
    import json
    import math
    from intelligence import cli as icli
    from intelligence.macro_analyzer import MacroAnalyzer
    from instrument import Instrument

    # 1. intelligence.cli._format_age and get_daemon_status handle None/NaN/bool safely
    assert icli._format_age(None) == "unknown"
    assert icli._format_age(float("nan")) == "unknown"
    assert icli._format_age(True) == "unknown"
    assert icli._format_age(-5.0) == "now"

    hb = tmp_path / "globaltelemetry_collector.heartbeat"
    hb.write_text(json.dumps({"pid": None, "ts": None}), encoding="utf-8")
    status = icli.get_daemon_status(str(tmp_path))
    assert status["running"] is False
    assert status["pid"] == 0
    assert status["age_seconds"] is None
    assert status["age_formatted"] == "unknown"

    # 2. MacroAnalyzer._resolve_now handles NaN/bool/None safely
    now_res = MacroAnalyzer._resolve_now(float("nan"))
    assert math.isfinite(now_res) and now_res > 0
    now_bool = MacroAnalyzer._resolve_now(True)
    assert math.isfinite(now_bool) and now_bool > 1000.0

    # 3. Instrument.min_qty handles bool/NaN/negative safely
    class _DummyProvider:
        name = "dummy"
        raw_min = float("nan")

        def validate_symbol(self, sym):
            pass

        def min_order_qty(self, sym):
            return self.raw_min

    class _DummyApi:
        def __init__(self, p):
            self._p = p

        def provider_by_name(self, name):
            return self._p

    prov = _DummyProvider()
    inst = Instrument("test", "BTCUSDC", "dummy", api=_DummyApi(prov))
    assert inst.min_qty() == 0.0
    prov.raw_min = True
    assert inst.min_qty() == 0.0
    prov.raw_min = -5.0
    assert inst.min_qty() == 0.0
    prov.raw_min = 0.001
    assert inst.min_qty() == 0.001


def test_persistence_survival_and_triggers_hardening():
    from types import SimpleNamespace
    import numpy as np
    from intelligence.internal.state import persistence as pers
    from intelligence.internal.state import survival as surv
    from intelligence.internal.triggers.gradient_trigger import LinearGradientTrigger
    from intelligence.external.triggers.whale_accumulation_trigger import WhaleAccumulationTrigger
    from intelligence.sentiment.triggers.sentiment_contrarian_trigger import SentimentContrarianTrigger

    # 1. persistence.calculate_mann_kendall and calculate_hurst_exponent handle None/mixed sequences safely
    assert pers.calculate_mann_kendall([100.0, None, True, 102.0]) == (0, 0.0, 1.0)
    assert pers.calculate_hurst_exponent([100.0, None, True]) is None

    # 2. survival.block_slopes and hybrid_T handle zero/negative step_h and non-finite prior_T/k safely
    bt, bs = surv.block_slopes(np.array([0.0, 3600.0]), np.array([100.0, 101.0]), 24, 0)
    assert len(bt) == 0 and len(bs) == 0
    ht = surv.hybrid_T([24.0, None, float("nan"), 48.0], prior_T=float("nan"), k=-2.0)
    assert ht["T"] >= 4 and ht["n"] == 2

    # 3. LinearGradientTrigger sanitizes ts
    gt = LinearGradientTrigger(strength_threshold=2.0)
    regime, ev = gt.evaluate("BTCUSDC", 85000.0, 5.0, 1.0, ts=float("nan"))
    assert regime == "bull" and ev is not None and ev.ts == 0.0

    # 4. WhaleAccumulationTrigger validates price, now, and snapshot fields
    wt = WhaleAccumulationTrigger()
    bad_snap = SimpleNamespace(
        top_traders_long_pct=0.70,
        taker_buy_sell_ratio=1.5,
        open_interest_1h_change_pct=2.0,
        divergence_regime="accumulation",
    )
    assert wt.evaluate("BTCUSDC", float("nan"), snapshot=bad_snap) is None
    assert wt.evaluate("BTCUSDC", 0.0, snapshot=bad_snap) is None
    ev_w = wt.evaluate("BTCUSDC", 85000.0, snapshot=bad_snap, now=float("nan"))
    assert ev_w is not None and ev_w.ts > 0.0

    # 5. SentimentContrarianTrigger handles None/malformed historical_values
    st = SentimentContrarianTrigger()
    fg_snap = SimpleNamespace(
        value=15,
        classification="Extreme Fear",
        historical_values=None,
        trend_7d_change=-5,
    )
    sig, ev_s = st.evaluate("BTCUSDC", 85000.0, fg_snap)
    assert sig == "CONTRARIAN_BUY" and ev_s is not None












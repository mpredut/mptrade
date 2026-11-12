"""config.env is sourced by root shell daemons (pia_supervisor.sh, vpn_watchdog.sh) under `set -u` with no HOME.

Any `$VAR` reference in it silently kills those daemons (unbound variable with stderr discarded), so the file
must be sourceable from an empty environment. Use `~/...` for user paths (expanded by expand_config_path).
"""
from datetime import datetime, timezone
from decimal import Decimal
import json
import math
import os
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import account_cache_health
import active_intents
import assetguardian as ag
from binance_api import market_data as b_md, trailing_stop as b_ts
import botcore
from botcore import resolve_now_ts
import cachetrend as ct
import external_order_guard as eog
from instrument import Instrument
from intelligence import cli as icli
from intelligence.composite import CompositeMarketIntelligence
from intelligence.external.collectors import bybit_liquidations_collector as ls
from intelligence.external.collectors.whale_positioning_collector import WhalePositioningCollector
from intelligence.external.guards import smart_stop_loss_guard as sslg
from intelligence.external.guards.cascade_guard import LiquidationCascadeGuard
from intelligence.external.guards.funding_crowding_guard import FundingCrowdingGuard
from intelligence.external.guards.orderbook_wall_guard import OrderbookWallGuard
from intelligence.external.guards.whale_divergence_guard import WhaleDivergenceGuard
from intelligence.external.triggers.liquidation_trigger import LiquidationCapitulationTrigger
from intelligence.internal.guards.exhaustion_guard import WeibullExhaustionGuard, weibull_survival_and_hazard
from intelligence.internal.guards.guard_decision import BrakeAction, GuardDecision
from intelligence.internal.guards.noise_guard import NoiseFloorGuard
from intelligence.internal.guards.parabolic_guard import ParabolicSurgeGuard
from intelligence.internal.guards.trend_significance_guard import TrendSignificanceGuard
from intelligence.internal.state import persistence as pers, survival as surv, volatility as vol
from intelligence.internal.state.persistence import classify_hurst_regime
from intelligence.internal.state.survival import hybrid_T
from intelligence.internal.state.volatility import adaptive_thresholds
from intelligence.internal.triggers.gradient_trigger import LinearGradientTrigger
from intelligence.internal.triggers.kalman_trigger import KalmanTrend
from intelligence.internal.triggers.mean_reversion_trigger import MeanReversionTrigger
from intelligence.internal.triggers.trigger_event import TriggerAction, TriggerEvent, TriggerSide
from intelligence.macro.geopolitical_analyzer import GeopoliticalThreatAnalyzer
from intelligence.macro.geopolitical_guard import GeopoliticalShockGuard
from intelligence.macro.news_feed_collector import NewsFeedCollector
from intelligence.macro_analyzer import MacroAnalyzer
from intelligence.sentiment.collectors.fear_greed_collector import FearGreedCollector
from intelligence.sentiment.collectors.market_breadth_collector import MarketBreadthCollector
from intelligence.sentiment.gemini_client import GeminiClient
from intelligence.sentiment.guards.extreme_greed_guard import ExtremeGreedGuard
from intelligence.sentiment.guards.high_stake_guard import HighStakeGuard, collect_pretrade_telemetry
from intelligence.sentiment.guards.panic_washout_guard import PanicWashoutGuard
from intelligence.sentiment.guards.post_stop_reentry_guard import PostStopReentryGuard
from intelligence.sentiment.sentiment_advisor import SentimentAdvisor
from intelligence.sentiment.triggers.market_breadth_trigger import MarketBreadthTrigger
from intelligence.sentiment.triggers.sentiment_contrarian_trigger import SentimentContrarianTrigger
from kraken import shadow_live
from market_monitor import alerts_config, new_coins_discovery as ncd, pricefetcher as pf
from market_monitor.pricechecker import PriceChecker
from market_regime import MarketRegimeContext, MarketRegimeDecision, MarketRegimeEvaluator, MarketRegimeService
import monitororder as mo
import monitortrades as mt
from notify_engine import alertnotifiers
from offline.backtests import (
    datasets as bds,
    evaluation as bev,
    execution as bex,
    execution_calibration as bec,
    financial_benchmark as bfb,
    metrics as bm,
    walk_forward as bwf,
)
from orchestratorOS.agy import topics as top_mod
from orchestratorOS.agy.summary_store import (
    _drop_summaries,
    parse_db_timestamp,
    read_annotation_title,
    resolve_project_ids,
    write_annotation_title,
)
from orchestratorOS.agy.topics import condense_decision, load_topic_registry, save_topic_registry
from orchestratorOS.lib import watchdog_common as wc
from orchestratorOS.lib.technical_repair_ssot import select_patch_pytest_targets
from orchestratorOS.livecheck import os_resources_watchdog as orw
from orchestratorTrade.autonomous_intent_gateway import AutonomousIntentGateway
import order_lifecycle as ol
import order_retry
import priceAnalysis as pa
from pricewindow import IncrementalTrendBuffer, PriceWindow
from providers import execution_audit, order_id_context as oic, quantity as binance_filters
from providers.strategy_executor import PairPrecision
import qlimit_order_guard as qog
import rtrade as rt
from rtrade_pair_store import RTradePairStore
import shadow_signals
from state_io import stage_previous_generation
from strategies import spot_engine as se, spot_rules as sr
from strategies.position_exit_manager import (
    ExitAction,
    ExitDecision,
    PositionExitConfig,
    PositionExitManager,
    PositionSnapshot,
    ReentrySnapshot,
)
from strategies.rtrade_pair import OrderSnapshot, OrderTicket, PairCoordinator, PairPolicy, anchored_exit_price, quote_prices
from tests.conftest import (
    DummyGeminiClient,
    DummyTrailingAdapter,
    SingleProviderApi,
    load_venue_modules,
    make_spot_test_params,
)
import tradeall as ta
import trailing_core as tc
import utils as u
from verify_tools import portfolio_snapshot as ps, watch_trade_decisions as wtd
from verify_tools.gen_alert_thresholds import thresholds_from_prices
from visuals import tradeall_observe as tobs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_config_env_sources_under_set_u_with_empty_environment():
    proc = subprocess.run(
        ["/bin/bash", "-c", "set -u; set -a; . ./config.env && echo SOURCE_OK"],
        cwd=ROOT, env={}, capture_output=True, text=True, timeout=20,
    )
    assert proc.returncode == 0 and "SOURCE_OK" in proc.stdout, proc.stderr


def test_config_module_parses_falsy_trade_enabled_fail_closed(tmp_path, monkeypatch):

    env_file = tmp_path / "config.env"
    txt_file = tmp_path / "config.txt"
    monkeypatch.setattr(botcore, "CONFIG_ENV_PATH", str(env_file))
    monkeypatch.setattr(botcore, "CONFIG_TXT_PATH", str(txt_file))
    monkeypatch.delenv("TRADE_ENABLED", raising=False)

    for raw in ("0", "false", '"false"', "'0'", "no", "off"):
        env_file.write_text(f"export TRADE_ENABLED={raw}\n", encoding="utf-8")
        botcore.load_config()
        assert botcore.is_trade_enabled() is False, f"Expected False for {raw}"

    env_file.write_text("TRADE_ENABLED=1\n", encoding="utf-8")
    txt_file.write_text("trade_enabled=0 # override\n", encoding="utf-8")
    botcore.load_config()
    assert botcore.is_trade_enabled() is False

    botcore.config_cache["trade_enabled"] = "0"
    assert botcore.is_trade_enabled() is False


def test_shell_launchers_source_env_common_correctly():
    with open(os.path.join(ROOT, "kraken", "spcx_launch.sh"), encoding="utf-8") as fh:
        spcx_text = fh.read()
    assert 'source "$HERE/../orchestratorOS/lib/env_common.sh"\n\nSTRAT_ENTRY=800' in spcx_text


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

    with open(os.path.join(ROOT, "orchestratorOS", "lib", "env_common.sh"), encoding="utf-8") as fh:
        env_common_text = fh.read()
    assert env_common_text.index('. "$ROOT/config.env"') < env_common_text.index('. "$ROOT/.env"')

    with open(os.path.join(ROOT, "orchestratorOS", "livecheck", "pia_supervisor.sh"), encoding="utf-8") as fh:
        pia_text = fh.read()
    assert "load_root_env_sh --with-secrets" in pia_text
    assert 'if [ "$current_reg" = "$target_region" ] && vpn_healthy; then' in pia_text

    with open(os.path.join(ROOT, "orchestratorOS", "livecheck", "vpn_watchdog.sh"), encoding="utf-8") as fh:
        wd_text = fh.read()
    assert "pia.service enabled but inactive/failed while PIA is healthy -> starting it" in wd_text
    assert 'orch_state=$(systemctl show -p ActiveState --value python_orchestrator.service' in wd_text
    assert '[ "$orch_state" != "deactivating" ]' in wd_text


def test_watchdog_and_gateway_hardening(tmp_path):
    bad_state = tmp_path / "bad_state.json"
    bad_state.write_text("[1, 2, 3]", encoding="utf-8")
    assert wc.load_state(bad_state) == {}

    st = {"restart_history": {"bot.py": [9999999999.0]}}
    assert orw._restart_allowed(st, "bot.py", 1000.0) is True
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
    monkeypatch.setenv("TEST_SHADOW_FLOAT", "nan")
    assert shadow_signals.pos_float_env("TEST_SHADOW_FLOAT", 15.0) == 15.0
    monkeypatch.setenv("TEST_SHADOW_FLOAT", "-1")
    assert shadow_signals.pos_float_env("TEST_SHADOW_FLOAT", 15.0) == 15.0

    tracker = shadow_signals.KalmanStateTracker(state_path=str(tmp_path / "kalman.json"))
    tracker._last_state_write = 9999999999.0
    tracker._last_fed["BTCUSDC"] = 9999999999.0
    tracker._last_vol_time["BTCUSDC"] = 9999999999.0
    tracker.update("BTCUSDC", 1000.0, 50000.0, 0.01, [50000.0], 1.0)
    assert tracker._last_fed["BTCUSDC"] == 1000.0
    assert tracker._last_state_write == 1000.0

    with pytest.raises(ValueError):
        botcore.require_shadow_interval(True, 1.0, "test")

    monkeypatch.setattr(ol, "ORDER_OUTCOMES_LOG_DIR", str(tmp_path))
    ol.log_order_outcome(
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
    monkeypatch.setenv("EXECUTION_AUDIT_CACHE_MAX", "5000 # inline comment")
    assert execution_audit._parse_cache_max() == 5000
    monkeypatch.setenv("EXECUTION_AUDIT_CACHE_MAX", "bad_int")
    assert execution_audit._parse_cache_max() == 10000

    with pytest.raises(binance_filters.BinanceFilterError):
        binance_filters.decimal_places(Decimal("NaN"))
    with pytest.raises(ValueError):
        account_cache_health.require_pos_float(True, "test")

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

    sig = ls.BybitLiquidationCollector(symbols=["BTCUSDT"])
    sig._ingest("[]")
    sig._ingest('{"topic":"allLiquidation.BTCUSDT","data":[{"s":"BTCUSDT","S":"Unknown","v":"1","p":"50000"}]}')
    sig._ingest('{"topic":"allLiquidation.BTCUSDT","data":[{"s":"BTCUSDT","S":"Buy","v":"-1","p":"-50000"}]}')
    assert sig.get_signal("BTCUSDT")["long_count"] == 0
    assert sig.get_signal("BTCUSDT")["short_count"] == 0


def test_alertnotifiers_and_hl_hardening(tmp_path, monkeypatch):
    hl_client, hl_signals = load_venue_modules("hyperliquid", "hl_client", "signals")
    hl_pa = hl_signals

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
    large_oid = (1 << 53) + 7
    assert hl_client._exact_int_in_range(large_oid, 1) == large_oid
    assert hl_client._exact_int_in_range(str(large_oid), 1) == large_oid

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
    dummy_client = SimpleNamespace(
        candles=lambda coin, interval, lookback_hours: [{"t": "bad", "c": "10.0"}, {"t": 1000, "c": float("nan")}],
        mid=lambda coin: float("nan"),
    )
    assert hl_pa.analyze(dummy_client, "HYPE") is None


def test_trailing_and_shadow_hardening(tmp_path, monkeypatch):
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
    with pytest.raises(ValueError):
        alerts_config._pair("-5 / 10")
    with pytest.raises(ValueError):
        alerts_config._pair("nan / 10")
    with pytest.raises(ValueError):
        botcore.require_pos_float(True, "test")

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
    assert adaptive_thresholds(float("nan")) == (None, None)
    assert adaptive_thresholds(True) == (None, None)

    assert classify_hurst_regime(float("nan")) == "unknown"
    assert classify_hurst_regime(True) == "unknown"

    durs = [24.0, 48.0, float("nan"), -5.0, 72.0]
    t_res = hybrid_T(durs)
    assert isinstance(t_res, dict) and isinstance(t_res["T"], int) and t_res["n"] == 3


def test_triggers_guards_and_collectors_hardening(tmp_path):
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
    assert parse_db_timestamp(None) is None
    assert parse_db_timestamp(12345) is None
    assert parse_db_timestamp("2026-10-26 04:00:00.12Z") is not None

    cli_dir = tmp_path / "store" / "antigravity-cli"
    proj_dir = tmp_path / "store" / "config" / "projects"
    proj_dir.mkdir(parents=True, exist_ok=True)
    (proj_dir / "bad.json").write_text("[1, 2, 3]", encoding="utf-8")
    (proj_dir / "good.json").write_text('{"id": "p1", "name": "mptrade"}', encoding="utf-8")
    assert "p1" in resolve_project_ids(["mptrade"], [str(cli_dir)])

    write_annotation_title(str(cli_dir), "cid1", 'Test "Title"')
    assert read_annotation_title(str(cli_dir), "cid1") == 'Test "Title"'
    _drop_summaries(str(cli_dir), set())

    reg_file = tmp_path / "topics.json"
    save_topic_registry({"macro_sentiment": {"conversation_id": "c1"}}, str(reg_file))
    stage_previous_generation(str(reg_file))
    reg_file.unlink()
    reg_file.write_text("{corrupted", encoding="utf-8")
    loaded_reg = load_topic_registry(str(reg_file))
    assert loaded_reg.get("macro_sentiment", {}).get("conversation_id") == "c1"
    assert len(condense_decision("A very long decision string here", max_len=2)) == 4

    dummy_retry = SimpleNamespace(load_all=lambda: ["not_a_dict", {"id": "c1", "place_kwargs": "bad"}])
    assert ol._accepted_is_durable(dummy_retry, "not_a_dict", {"orderId": "ord1"}) is False
    assert ol._accepted_is_durable(dummy_retry, {"id": "c1", "place_kwargs": "bad"}, {"orderId": "ord1"}) is False


def test_order_id_context_and_pricefetcher_hardening(tmp_path):
    assert oic._base36(-5) == "0"
    cid = oic.create_client_order_id("VERY_LONG_PROCESS_NAME_EXCEEDING_THIRTY_SIX_CHARS")
    assert len(cid) <= oic.CLIENT_ORDER_ID_MAX_LEN

    assert pf.is_valid_symbol_for_monitoring(123) is False
    cmc = pf.CoinMarketCapPricePlatform(api_key=None)
    assert cmc._extract_usd_price("BTC", {"data": None}) is None
    assert cmc._extract_usd_price("BTC", {"data": {"BTC": {"quote": None}}}) is None
    assert cmc._extract_usd_price("BTC", {"data": {"BTC": {"quote": {"USD": None}}}}) is None


def test_price_analysis_market_regime_and_monitororder_hardening(monkeypatch):
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
    monkeypatch.setattr(qog._og, "_resolve_trend_duration", lambda sym: called.append(sym) or 0.0)
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

    # 3. spot_rules reject bool/None in exhaustion checks
    assert sr.dynamic_surge_gain_pct(True, fallback_gain_pct=25.0) == 25.0
    assert sr.dynamic_flat_tp_pct(True, base_tp_pct=5.0) == 5.0
    assert sr.dynamic_trend_trail_pct(float("nan"), base_trail_pct=8.0) == 8.0
    assert sr.check_fast_profit_reversal(None, 100.0, 5.0, 2.0, [], 60.0, 1.0) == (False, 0.0, 0.0)
    assert sr.check_parabolic_surge_exhaustion(None, 100.0, 130.0, 25.0, 2.0) == (False, "")
    assert sr.check_slow_grind_exhaustion(None, 100.0, 1000.0, 900000.0, 7.0, 15.0, 120.0, [], 900.0, 1.5, 2.5) == (False, "")


def test_external_order_guard_rtrade_pair_and_monitortrades_hardening(monkeypatch):
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
    nan_params = ol.extract_order_restore_params({"side": "SELL", "price": float("nan"), "qty": 2.0, "amount": True})
    assert nan_params["price"] == 0.0
    assert nan_params["amount"] == 0.0
    called = []
    assert ol.restore_cancelled_orders(
        [{"side": "SELL", "price": float("nan"), "qty": 2.0}, {"side": "SELL", "price": 100.0, "qty": 2.0, "amount": True}],
        restore_fn=lambda p: called.append(p) or True,
    ) == 1
    assert len(called) == 1 and called[0]["amount"] == 200.0


def test_spot_engine_trailing_core_and_pricewindow_hardening():
    # 1. spot_engine._qty_for, _dust_safe_qty, _place, and step reject NaN/bool/<=0 safely
    dummy_client = SimpleNamespace(
        pair_precision=lambda _pair: PairPrecision(2, 4, 0.01, "BTC"),
        free_balance=lambda _asset: 1000.0,
    )
    params = make_spot_test_params(currency="EUR")
    strat = se.Strategy(dummy_client, "BTCEUR", params, dry_run=True, initial_state=se._new_state(), replay_mode=True)
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
    core = tc.TrailingCore(
        DummyTrailingAdapter(free=None, px=100.0),
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
    w = PriceWindow("BTCUSDC", 20, sample_rate_sec=1.0, initial_prices=[100.0, 101.0, 102.0, 103.0])
    assert w.calculate_acceleration(window_sec=float("inf")) != (float("nan"), float("nan"))
    assert w.calculate_acceleration(window_sec=True) != (float("nan"), float("nan"))


def test_shadow_signals_utils_and_volatility_hardening():
    # 1. shadow_signals.KalmanStateTracker.current_trend and update handle None/NaN/bool safely
    tracker = shadow_signals.KalmanStateTracker(state_path=None, compute_vol=False)
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

    # 2. resolve_now_ts handles NaN/bool/None safely
    now_res = resolve_now_ts(float("nan"))
    assert math.isfinite(now_res) and now_res > 0
    now_bool = resolve_now_ts(True)
    assert math.isfinite(now_bool) and now_bool > 1000.0

    # 3. Instrument.min_qty handles bool/NaN/negative safely
    prov = SimpleNamespace(name="dummy", raw_min=float("nan"), validate_symbol=lambda sym: None)
    prov.min_order_qty = lambda sym: prov.raw_min
    inst = Instrument("test", "BTCUSDC", "dummy", api=SingleProviderApi(prov))
    assert inst.min_qty() == 0.0
    prov.raw_min = True
    assert inst.min_qty() == 0.0
    prov.raw_min = -5.0
    assert inst.min_qty() == 0.0
    prov.raw_min = 0.001
    assert inst.min_qty() == 0.001


def test_persistence_survival_and_triggers_hardening():
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

    # 4. SentimentContrarianTrigger handles None/malformed historical_values
    st = SentimentContrarianTrigger()
    fg_snap = SimpleNamespace(
        value=15,
        classification="Extreme Fear",
        historical_values=None,
        trend_7d_change=-5,
    )
    sig, ev_s = st.evaluate("BTCUSDC", 85000.0, fg_snap)
    assert sig == "CONTRARIAN_BUY" and ev_s is not None


def test_collectors_and_gemini_client_hardening():
    # 1. WhalePositioningCollector.fetch handles price_1h_change_pct=None/NaN/bool without TypeError
    w_col = WhalePositioningCollector(cache_dir=None)
    assert w_col.fetch("BTCUSDC", price_1h_change_pct=None, allow_network=False, now=0.0) is None
    assert w_col.fetch("BTCUSDC", price_1h_change_pct=float("nan"), allow_network=False, now=True) is None

    # 2. NewsFeedCollector.parse_rss_xml sanitizes non-finite/bool/<=0 now_ts
    xml_data = b"<rss><channel><item><title>Major offensive reported</title><source>Reuters</source></item></channel></rss>"
    snap_rss = NewsFeedCollector.parse_rss_xml(xml_data, now_ts=float("nan"))
    assert snap_rss is not None and math.isfinite(snap_rss.ts) and snap_rss.ts > 1000.0
    snap_rss2 = NewsFeedCollector.parse_rss_xml(xml_data, now_ts=True)
    assert snap_rss2 is not None and math.isfinite(snap_rss2.ts) and snap_rss2.ts > 1000.0

    # 3. GeminiClient.query_text sanitizes timeout_sec and cache_ttl_sec
    received = {}

    def _runner(prompt, model, timeout):
        received["timeout"] = timeout
        return '{"ok": true}'

    gc = GeminiClient(model="test-model", custom_runner=_runner, project_id="test-proj")
    out = gc.query_text("hello", timeout_sec=float("nan"), cache_ttl_sec=float("nan"))
    assert out == '{"ok": true}'
    assert received["timeout"] == 15.0


def test_geopolitical_sentiment_advisor_and_high_stake_hardening(tmp_path):
    dummy_gemini = DummyGeminiClient(
        {"market_bias": "NEUTRAL", "risk_level": "LOW", "confidence": 0.8, "summary": "ok"}
    )

    # 1. GeopoliticalThreatAnalyzer sanitizes non-finite TTLs
    ga = GeopoliticalThreatAnalyzer(
        gemini_client=dummy_gemini,
        cache_ttl_sec=float("nan"),
        fallback_retry_ttl_sec=True,
        state_file=str(tmp_path / "geo.json"),
    )
    assert math.isfinite(ga.cache_ttl_sec) and ga.cache_ttl_sec > 0
    assert math.isfinite(ga.fallback_retry_ttl_sec) and ga.fallback_retry_ttl_sec > 0

    # 2. SentimentAdvisor.review handles float/None trend_7d_change and advance_ratio without ValueError/TypeError
    sa = SentimentAdvisor(
        gemini_client=dummy_gemini,
        cache_ttl_sec=float("nan"),
        cache_file=str(tmp_path / "adv.json"),
    )
    fg = SimpleNamespace(value=45.0, classification=None, trend_7d_change=3.5)
    mb = SimpleNamespace(advance_ratio=None, median_change_pct=float("nan"), regime=None)
    res_sa = sa.review(fear_greed=fg, breadth=mb, force_refresh=True)
    assert res_sa is not None and res_sa.market_bias == "NEUTRAL"

    # 3. collect_pretrade_telemetry and HighStakeGuard/PostStopReentryGuard handle non-finite/bool params
    tel = collect_pretrade_telemetry("BTCUSDC", dur_sec=float("inf"), cachedb_dir=str(tmp_path))
    assert "Trend Duration" not in tel
    hsg = HighStakeGuard(gemini_client=dummy_gemini, min_notional_eur=float("nan"), timeout_sec=True)
    assert math.isfinite(hsg.min_notional_eur) and hsg.min_notional_eur > 0
    assert hsg.timeout_sec == 120.0
    psg = PostStopReentryGuard(gemini_client=dummy_gemini, timeout_sec=float("nan"), cache_ttl_sec=float("inf"))
    assert psg.timeout_sec == 120.0 and psg.cache_ttl_sec == 0.0


def test_smart_sl_exhaustion_and_resources_watchdog_hardening(monkeypatch):
    # 1. smart_stop_loss_guard handles bool/NaN/None in _format_price, resolve_now_ts, and on_recovered
    assert sslg._format_price(True) == ""
    assert resolve_now_ts(0.0, allow_zero=True) == 0.0
    assert resolve_now_ts(float("nan"), allow_zero=True) > 1_000_000_000.0
    assert resolve_now_ts(True, allow_zero=True) > 1_000_000_000.0
    guard = sslg.SmartStopLossGuard(mode="enforce")
    monkeypatch.setitem(
        sslg._GRACE_TRACKER,
        "BTCUSDC",
        {"start_ts": 1000.0, "initial_loss_pct": None, "provider": "sim", "buy_price": 85000.0},
    )
    assert guard.on_recovered("BTCUSDC", None, None, now=1010.0, provider="sim", buy_price=85000.0) is True

    # 2. weibull_survival_and_hazard and WeibullExhaustionGuard handle non-finite/bool inputs safely
    s, h, k, lam = weibull_survival_and_hazard(float("nan"), 3.0, 7.0)
    assert s == 1.0 and h == 0.0 and k > 0 and lam > 0
    s2, h2, _, _ = weibull_survival_and_hazard(True, 3.0, 7.0)
    assert s2 == 1.0 and h2 == 0.0
    wg = WeibullExhaustionGuard(default_p90_days=float("nan"), policy=None, exhausted_scale=True)
    assert wg.default_p90_days == 7.0 and wg.policy == "downscale" and wg.exhausted_scale == 0.25

    # 3. os_resources_watchdog handles corrupted state dicts, None high/normal counts, and bool/NaN timestamps
    state_corrupt = {"restart_history": None}
    assert orw._restart_allowed(state_corrupt, "bot.py", 1000.0) is True
    assert orw._usage("not_dict", {"ticks": 100}, 10, 20, 4) is None
    monkeypatch.setattr(
        orw.wc,
        "load_state",
        lambda _f: {
            "total_ticks": "bad",
            "processes": "corrupted",
            "tracked": {"100:1": {"high": None, "normal": None, "alerted": False, "last_alert": float("nan")}},
        },
    )
    monkeypatch.setattr(orw.wc, "save_state", lambda _f, _s: None)
    monkeypatch.setattr(orw.wc, "send_ntfy", lambda _t, _m: True)
    assert orw.check_once(
        now=1000.0,
        total_ticks=500,
        processes={"100:1": {"pid": 100, "ticks": 10, "rss_mb": 10.0, "command": "python3 bot.py"}},
        cpu_count=4,
    ) is False


def test_liquidation_kalman_mean_reversion_and_breadth_triggers_hardening():
    # 1. LiquidationCapitulationTrigger handles None/bool fields on summary and <=0 min_notional_usd
    lt = LiquidationCapitulationTrigger(min_notional_usd=0.0, ratio_threshold=float("nan"), window_sec=True)
    assert lt.min_notional_usd == 100_000.0 and lt.ratio_threshold == 0.75 and lt.window_sec == 300.0
    bad_summary = SimpleNamespace(
        total_usd=None,
        long_liq_usd=200_000.0,
        short_liq_usd=0.0,
        capitulation_ratio=0.9,
        squeeze_ratio=0.1,
    )
    assert lt.evaluate("BTCUSDC", 85000.0, summary=bad_summary) is None

    # 2. NoiseFloorGuard and TrendSignificanceGuard handle non-finite/bool args and None trigger
    ng = NoiseFloorGuard(min_strength_ratio=float("nan"))
    assert ng.min_strength_ratio == 1.0
    tsg = TrendSignificanceGuard(max_p_value=True, min_hurst=float("nan"))
    assert tsg.max_p_value == 0.05 and tsg.min_hurst is None
    assert tsg.evaluate(None, [100.0] * 10).allowed is True

    # 3. KalmanTrend and MeanReversionTrigger sanitize constructor args and timestamp
    kt = KalmanTrend(qr=float("nan"), gap_reset_sec=True)
    assert kt.qr > 0.0 and kt.gap_reset_sec > 0.0
    mrt = MeanReversionTrigger(rsi_period=True, bb_period=float("nan"), rsi_low=None)
    assert mrt.rsi_period == 14 and mrt.bb_period == 20 and mrt.rsi_low == 30.0
    ev_mr = mrt.update("BTCUSDC", float("nan"), 85000.0)
    assert ev_mr.ts == 0.0

    # 4. MarketBreadthTrigger handles string-encoded ratios and missing regime attribute
    mbt = MarketBreadthTrigger()
    snap_mb = SimpleNamespace(advance_ratio="0.10", median_change_pct="-5.0")
    sig_mb, ev_mb = mbt.evaluate("BTCUSDC", 85000.0, snap_mb, ts=float("nan"))
    assert sig_mb == "BREADTH_WASHOUT_BUY" and ev_mb is not None and ev_mb.metadata["regime"] == "NEUTRAL"

    # 5. FearGreedCollector and MarketBreadthCollector sanitize TTLs
    fgc = FearGreedCollector(cache_ttl_sec=float("nan"))
    assert fgc.cache_ttl_sec == 900.0
    mbc = MarketBreadthCollector(cache_ttl_sec=True, min_volume_usd=float("nan"))
    assert mbc.cache_ttl_sec == 60.0 and mbc.min_volume_usd == 5_000_000.0


def test_news_feed_history_macro_analyzer_and_pricechecker_hardening(tmp_path):
    # 1. NewsFeedCollector._history alias and MacroAnalyzer.assess_geopolitical_threat with news_feed_collect.json
    now_ts = time.time()
    nf_path = tmp_path / "news_feed_collect.json"
    nf_path.write_text(
        json.dumps(
            [
                {
                    "title": "Minor market update",
                    "source": "Test",
                    "pub_date": "",
                    "url": "",
                    "matched_keywords": [],
                    "is_high_severity": False,
                    "first_seen_ts": now_ts - 10.0,
                }
            ]
        ),
        encoding="utf-8",
    )
    nc = NewsFeedCollector(history_file=str(nf_path), cache_ttl_sec=float("nan"), rolling_window_sec=True)
    assert nc.cache_ttl_sec == 60.0 and nc.rolling_window_sec == 14400.0
    assert len(nc._aggregated_pool) == 1

    ma = MacroAnalyzer(cache_dir=str(tmp_path))
    ma.shadow_notify = False
    assert ma.assess_geopolitical_threat(force=True, now=now_ts) is True

    # 2. GeopoliticalShockGuard handles missing summary/risk_score and non-finite downscale_factor
    gsg = GeopoliticalShockGuard(downscale_factor=float("nan"))
    assert gsg.downscale_factor == 0.50
    dec_g = gsg.check("BTCUSDC", "BUY", SimpleNamespace(threat_level="ELEVATED", recommended_brake="DOWNSCALE_50"))
    assert dec_g.suggested_scale == 0.50 and dec_g.metadata["risk_score"] == 0.0

    # 3. PriceChecker handles bool/NaN hours, current_price, and interval_seconds
    dummy_cache = SimpleNamespace(
        original_symbols=["BTC"],
        get_latest_price=lambda _sym: True,
        get_price_history=lambda _sym, limit=1000: [
            {"timestamp": True, "price": 100.0},
            {"timestamp": now_ts * 1000, "price": True},
        ],
    )
    pc = PriceChecker(dummy_cache)
    assert pc._get_price_history_last_hours("BTC", float("nan")) == []
    assert pc._get_price_history_last_hours("BTC", True) == []
    assert pc._get_price_history_last_hours("BTC", 24.0) == []
    assert pc._calculate_24h_stats("BTC")["has_data"] is False
    with pytest.raises(ValueError):
        pc.start_monitoring(interval_seconds=True)
    with pytest.raises(ValueError):
        pc.start_monitoring(interval_seconds=float("nan"))


def test_external_sentiment_parabolic_and_composite_guards_hardening():
    pg = ParabolicSurgeGuard(surge_threshold_pct=float("nan"), pullback_required_pct=True, window_seconds=0.0, vol_multiplier=None)
    assert pg.surge_threshold_pct == 4.0 and pg.pullback_required_pct == 1.5 and pg.window_seconds == 7200.0 and pg.vol_multiplier == 3.0

    cg = LiquidationCascadeGuard(max_active_cascade_usd=float("nan"), active_window_sec=True)
    assert cg.max_active_cascade_usd == 500_000.0 and cg.active_window_sec == 60.0

    fcg = FundingCrowdingGuard(max_long_funding_rate=0.0, min_short_funding_rate=1.0, crowding_policy="invalid", crowded_scale=float("nan"))
    assert fcg.max_long_funding_rate == 0.0005 and fcg.min_short_funding_rate == -0.0005 and fcg.crowding_policy == "downscale" and fcg.crowded_scale == 0.50

    owg = OrderbookWallGuard(min_buy_imbalance=float("nan"), max_sell_imbalance=2.0, whale_wall_usd_limit=0.0)
    assert owg.min_buy_imbalance == 0.25 and owg.max_sell_imbalance == 0.75 and owg.whale_wall_usd_limit == 1_000_000.0

    wdg = WhaleDivergenceGuard(severe_cascade_oi_pct=float("nan"), min_taker_ratio=0.0, taker_downscale_threshold=True, taker_downscaled_scale=float("inf"))
    assert wdg.severe_cascade_oi_pct == -5.0 and wdg.min_taker_ratio == 0.75 and wdg.taker_downscale_threshold == 0.90 and wdg.taker_downscaled_scale == 0.50

    egg = ExtremeGreedGuard(downscale_threshold=float("nan"), hard_veto_threshold=True, downscale_factor=0.0)
    assert egg.downscale_threshold == 80 and egg.hard_veto_threshold == 90 and egg.downscale_factor == 0.35

    pwg = PanicWashoutGuard(severe_panic_fng_threshold=True, panic_advance_threshold=2.0, downscale_factor=float("nan"))
    assert pwg.severe_panic_fng_threshold == 12 and pwg.panic_advance_threshold == 0.10 and pwg.downscale_factor == 0.50

    cmi = CompositeMarketIntelligence()
    ev = cmi.evaluate(
        "BTCUSDC",
        85000.0,
        fear_greed_snapshot=SimpleNamespace(),
        market_breadth_snapshot=SimpleNamespace(),
        geopolitical_assessment=SimpleNamespace(),
    )
    assert ev.metrics["fear_greed"] is None and ev.metrics["market_regime"] is None and ev.metrics["geopolitical_threat"] is None


def test_new_coins_discovery_and_pricefetcher_hardening():
    assert ncd.is_trackable_symbol(123) is False
    assert ncd.is_trackable_symbol(True) is False
    assert ncd.is_trackable_symbol("NEWCOIN", name=None, slug=None) is True
    assert ncd._safe_days_back(float("nan")) == float(ncd.NEW_COINS_AGE_DAYS)
    assert ncd._safe_days_back(True) == float(ncd.NEW_COINS_AGE_DAYS)

    dummy_factory = SimpleNamespace(
        get_all_new_coins=lambda _d: {
            "coinmarketcap": [
                {"symbol": "NEWCOIN", "name": "New", "added_at": pd.NaT, "price": "0.1234", "volume_24h": "bad", "change_24h": None}
            ]
        },
        get_available_sources=lambda: ["coinmarketcap"],
        refresh_all=lambda: None,
    )
    dummy_pm = SimpleNamespace(
        original_symbols=[],
        active_symbols=set(),
        max_monitored_symbols=10,
        symbol_added_time={"NEWCOIN": True},
        add_symbol=lambda _s, preferred_source=None: True,
    )
    mon = ncd.NewCoinsMonitor(price_monitor=dummy_pm, factory=dummy_factory)
    assert "$0.123400" in mon.get_report()
    assert mon.add_new_coin_to_watchlist(mon.all_new_coins["coinmarketcap"][0]) is True
    assert mon.should_keep_monitoring("NEWCOIN") is True
    with pytest.raises(ValueError):
        mon.start_monitoring(interval_seconds=True)
    with pytest.raises(ValueError):
        mon.start_monitoring(interval_seconds=float("nan"))

    assert pf.safe_pos_int(float("nan"), default=pf.MAX_MONITORED_SYMBOLS) == pf.MAX_MONITORED_SYMBOLS
    assert pf.safe_pos_int(True, default=pf.MAX_MONITORED_SYMBOLS) == pf.MAX_MONITORED_SYMBOLS
    assert pf.safe_pos_int(12, default=pf.MAX_MONITORED_SYMBOLS) == 12
    yahoo = pf.StockYahooPricePlatform(tickers=["NVDA"])
    assert yahoo.supports_symbol(None) is False
    assert yahoo.supports_symbol("NVDA") is True


def test_liquidation_signal_cachetrend_and_active_intents_hardening(tmp_path):
    sig = ls.BybitLiquidationCollector(symbols=["", "BTCUSDT"], retention_sec=float("nan"))
    assert sig._symbols == ["BTCUSDT"] and sig._retention == ls.MAX_RETENTION_SEC
    sig._ingest(json.dumps({"topic": "allLiquidation.BTCUSDT", "data": [{"s": "BTCUSDT", "S": "Buy", "v": True, "p": "75000"}]}))
    assert sig.get_signal("BTCUSDT", window_sec=float("nan"))["long_liq_usd"] == 0.0
    sig._ingest(json.dumps({"topic": "allLiquidation.BTCUSDT", "data": [{"s": "BTCUSDT", "S": "Buy", "v": "1.0", "p": "75000"}]}))
    out = sig.get_signal("BTCUSDT", window_sec=True)
    assert out["window_sec"] == 300.0 and out["long_liq_usd"] == 75000.0
    assert math.isfinite(sig.capitulation_ratio("BTCUSDT", window_sec=float("nan"), baseline_sec=True))

    mgr = ct.CacheShortTrendManager(
        ["BTCUSDC"],
        filename=str(tmp_path / "st.json"),
        window_seconds=[True, float("nan"), -5.0],
        sync_ts=float("nan"),
    )
    assert len(mgr.window_seconds) == 2 and all(w > 0 for w in mgr.window_seconds)

    summary_txt = active_intents.format_summary(
        {
            "intents": None,
            "summary": {
                "pending_notional": {"total_buy": "bad", "total_sell": None, "by_symbol": {"BTC": {"buy": "12.5", "sell": True}}},
                "net_exposures": {"total_cost": float("nan"), "positions": [{"symbol": "BTC", "qty": 1.0, "cost": "100.5", "venue": "binance"}]},
            },
        }
    )
    assert "$100.50" in summary_txt and "BUY $12.50" in summary_txt


def test_log_configure_and_verify_tools_hardening():
    assert thresholds_from_prices([], window=20, pct=85) is None
    assert thresholds_from_prices([100.0, 200.0], window=True, pct=85) is None
    assert thresholds_from_prices([100.0, 200.0], window=1, pct=float("nan")) is None
    res_t = thresholds_from_prices([None, True, float("nan"), -5.0, 100.0, 110.0, 95.0], window=1, pct=85)
    assert res_t is not None and len(res_t) == 4 and all(v >= 3.0 for v in res_t[:2])

    assert ps.safe_float(True, 0.0) == 0.0
    assert ps.safe_float(float("nan"), 0.0) == 0.0
    assert ps.safe_float(float("inf"), 1.5) == 1.5
    assert ps.safe_float("42.5", 0.0) == 42.5


def test_tradeall_observe_topics_and_watch_hardening(tmp_path):
    txt = tobs.format_state_text(
        {
            "current_price": 85000.0,
            "final_trend": 1,
            "gradient_recent": "bad",
            "slope_small": float("nan"),
            "slope_big": True,
            "epsilon": None,
            "trend_fast": -1,
            "gradient_recent_fast": "bad",
            "kalman_trend": 0,
            "kalman_vel": "bad",
            "kalman_vel_std": float("inf"),
            "vol_1h_pct": "bad",
        },
        "HEADER",
    )
    assert "HEADER" in txt and "grad rec:  +0.0000" in txt and "vol 1h:" not in txt

    with pytest.raises(ValueError):
        top_mod.prepare_topic_session("macro_geopolitical", cli_base=str(tmp_path), compact_steps=True)
    with pytest.raises(ValueError):
        top_mod.prepare_topic_session("macro_geopolitical", cli_base=str(tmp_path), compact_steps="bad")
    assert len(top_mod.condense_decision('{"decision": "HOLD", "summary": "ok"}', max_len=True)) > 10
    assert len(top_mod.condense_decision('{"decision": "HOLD", "summary": "ok"}', max_len=float("nan"))) > 10

    assert wtd.safe_float(True, 0.0) == 0.0
    assert wtd.safe_float(float("nan"), 0.0) == 0.0
    assert wtd.safe_float("12.5", 0.0) == 12.5


def test_backtest_metrics_walk_forward_and_execution_hardening():
    with pytest.raises(ValueError):
        bm.calculate_performance_metrics([1.0], initial_capital=True)
    with pytest.raises(ValueError):
        bm.calculate_performance_metrics([100.0, 105.0], initial_capital=100.0, periods_per_year=True)

    valid_res = {
        "return_pct": 1.5,
        "max_drawdown_pct": 0.5,
        "cycles": 1,
        "fills": 2,
        "initial_capital": 1000.0,
        "account_currency": "USD",
        "net_pnl": 15.0,
        "realized": 15.0,
        "total": 15.0,
        "fees": 0.5,
    }
    bm.validate_replay_metrics(valid_res)
    with pytest.raises(AssertionError):
        bm.validate_replay_metrics({**valid_res, "cycles": True})
    with pytest.raises(AssertionError):
        bm.validate_replay_metrics({**valid_res, "return_pct": False})

    with pytest.raises(ValueError):
        bwf.walk_forward_splits(100, train_size=True, validation_size=10, test_size=10)
    with pytest.raises(ValueError):
        bwf.walk_forward_splits(100, train_size=20, validation_size=10, test_size=10, step_size=True)

    with pytest.raises(ValueError):
        bex.FeeModel(limit_fee_pct=True, market_fee_pct=0.1)
    with pytest.raises(ValueError):
        bex.ExecutionModel(partial_fill_ratio=True)
    with pytest.raises(ValueError):
        bex.ExecutionModel(spread_bps=True)


def test_datasets_execution_calibration_and_benchmark_hardening():
    with pytest.raises(ValueError):
        bds.validate_dataset([{"timestamp": 1000, "open": 10.0, "high": float("nan"), "low": 9.0, "close": 10.0}])
    with pytest.raises(ValueError):
        bds.validate_dataset([{"timestamp": 1000, "open": True, "high": 10.0, "low": 9.0, "close": 10.0}])
    with pytest.raises(ValueError):
        bds.validate_dataset(
            [{"timestamp": 1000, "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0}],
            interval_minutes=True,
        )

    dist = bec.distribution([1.0, None, True, float("nan"), float("inf"), 3.0])
    assert dist["count"] == 2 and dist["mean"] == 2.0
    cal = bec.calibrate_execution_events(
        [
            {"intent_id": "i1", "event": "submit_requested", "ts": "bad", "qty": "bad", "price": float("nan")},
            {"intent_id": "i1", "event": "order_terminal", "ts": True, "filled_qty": "bad", "cost": float("nan"), "fee": None},
        ]
    )
    assert len(cal["orders"]) == 1 and cal["orders"][0]["filled_qty"] == 0.0

    assert bec.clean_finite_floats([1.0, None, True, float("nan"), 2.0]) == [1.0, 2.0]
    with pytest.raises(ValueError):
        bfb.aggregate_financial_windows([], initial_capital=True)

    with pytest.raises(ValueError):
        bev.automatic_window_sizes(True)
    assert bev._optional_min([None, True, float("nan"), 2.5, 1.5]) == 1.5


def test_backup_sync_common_modular_primitives(tmp_path):
    scripts = [
        os.path.join(ROOT, "orchestratorOS", "lib", "env_common.sh"),
        os.path.join(ROOT, "orchestratorOS", "lib", "os_notify.sh"),
        os.path.join(ROOT, "orchestratorOS", "lib", "backup_sync_common.sh"),
        os.path.join(ROOT, "orchestratorOS", "lib", "pia_common.sh"),
        os.path.join(ROOT, "orchestratorOS", "livecheck", "pia_supervisor.sh"),
        os.path.join(ROOT, "orchestratorOS", "livecheck", "vpn_watchdog.sh"),
        os.path.join(ROOT, "orchestratorOS", "livecheck", "wait_for_vpn.sh"),
        os.path.join(ROOT, "orchestratorOS", "livecheck", "deadman_switch.sh"),
        os.path.join(ROOT, "orchestratorOS", "livecheck", "ntfy_check.sh"),
        os.path.join(ROOT, "orchestratorOS", "admin", "manage_backups.sh"),
        os.path.join(ROOT, "orchestratorOS", "admin", "manage_agy.sh"),
        os.path.join(ROOT, "orchestratorOS", "admin", "manage_gitautodeploy.sh"),
        os.path.join(ROOT, "orchestratorOS", "admin", "manage_logs.sh"),
        os.path.join(ROOT, "orchestratorOS", "admin", "make_venv_portable.sh"),
        os.path.join(ROOT, "offline", "runners", "load_dev_backtest_env.sh"),
        os.path.join(ROOT, "offline", "runners", "sync_cache_to.sh"),
        os.path.join(ROOT, "offline", "runners", "trigger_dev_regression.sh"),
        os.path.join(ROOT, "offline", "runners", "check_dev_regression.sh"),
        os.path.join(ROOT, "offline", "runners", "run_dev_regression.sh"),
        os.path.join(ROOT, "offline", "runners", "run_backtest_cycle.sh"),
        os.path.join(ROOT, "offline", "runners", "run_backtest_parallel.sh"),
        os.path.join(ROOT, "offline", "runners", "run_financial_baseline.sh"),
        os.path.join(ROOT, "offline", "runners", "run_overnight_master_suite.sh"),
        os.path.join(ROOT, "offline", "research", "kraken_continuous_grid", "run_all.sh"),
        os.path.join(ROOT, "systemd", "pia_settings_mtu.sh"),
        os.path.join(ROOT, "systemd", "install_prod.sh"),
    ]
    for script in scripts:
        subprocess.run(["bash", "-n", script], check=True)

    fake_home = tmp_path / "home"
    src_dir = tmp_path / "src_backup"
    dst_dir = tmp_path / "dst_repo"
    (src_dir / "cachedb").mkdir(parents=True, exist_ok=True)
    (src_dir / ".ssh_backup").mkdir(parents=True, exist_ok=True)
    (src_dir / "logs").mkdir(parents=True, exist_ok=True)
    (src_dir / "cachedb" / "cache_365_price_BTCUSDC.jsonl").write_text('{"p":1}\n', encoding="utf-8")
    (src_dir / ".ssh_backup" / "id_ed25519").write_text("secret", encoding="utf-8")
    (src_dir / "logs" / "prod.log").write_text("log", encoding="utf-8")
    dst_dir.mkdir(parents=True, exist_ok=True)

    cmd = (
        f'export HOME="{fake_home}"; '
        f'export STORJ_ACCES_KEY="key1"; export STORJ_SECRET_KEY="sec1"; export STORJ_ENDPOINT="Yhttps://gateway.storjshare.io"; '
        f'source "{os.path.join(ROOT, "orchestratorOS", "lib", "backup_sync_common.sh")}"; '
        f'ensure_rclone_storj_config; '
        f'export STORJ_ACCESS_KEY="key2"; ensure_rclone_storj_config; '
        f'unpack_repo_payload "{src_dir}" "{dst_dir}" --exclude="logs"'
    )
    subprocess.run(["bash", "-c", cmd], check=True)

    rclone_conf = (fake_home / ".config" / "rclone" / "rclone.conf").read_text(encoding="utf-8")
    assert rclone_conf.count("[storj]") == 1
    assert "access_key_id = key2" in rclone_conf
    assert "endpoint = https://gateway.storjshare.io" in rclone_conf
    assert (dst_dir / "cachedb" / "cache_365_price_BTCUSDC.jsonl").is_file()
    assert not (dst_dir / ".ssh_backup").exists()
    assert not (dst_dir / "logs").exists()

    targets = select_patch_pytest_targets(
        os.path.join(ROOT, "orchestratorOS", "lib", "pia_common.sh"),
        ROOT,
    )
    assert "tests/test_config_env_shell_safe.py" in targets
    assert len(targets) <= 5


def _write_bash_stub(fake_bin, name: str, body: str) -> None:
    p = fake_bin / name
    p.write_text("#!/usr/bin/env bash\n" + body + "\n", encoding="utf-8")
    p.chmod(0o755)


def test_pia_common_ssot_and_bounded_wait_for_vpn(tmp_path):

    with open(os.path.join(ROOT, "config.env"), encoding="utf-8") as fh:
        cfg_text = fh.read()
    assert "\nPIA_RESTORE_INTERVAL=600\n" in cfg_text
    assert "\nAGY_VPN_WAIT_MAX_SEC=8\n" in cfg_text

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    state_file = tmp_path / "pia_supervisor.state"

    # 1. pia() strips CR and preserves piactl exit codes; supervisor heartbeat SSOT detects fresh vs stale
    _write_bash_stub(fake_bin, "piactl", 'if [ "${1:-}" = "failcmd" ]; then exit 7; fi; printf "Connected\\r\\n"')
    _write_bash_stub(fake_bin, "systemctl", 'if [ "${1:-}" = "show" ] && [ "${3:-}" = "ActiveState" ]; then echo active; exit 0; fi; echo 0')
    ssot_cmd = (
        f'export PATH="{fake_bin}:$PATH" PIA_SUPERVISOR_STATE_FILE="{state_file}" CONNECT_SETTLE=0; '
        f'source "{os.path.join(ROOT, "orchestratorOS", "lib", "env_common.sh")}"; '
        f'source "{os.path.join(ROOT, "orchestratorOS", "lib", "pia_common.sh")}"; '
        f'[ "$(pia get connectionstate)" = "Connected" ] && '
        f'! pia failcmd && '
        f'pia_write_supervisor_state healthy && '
        f'pia_supervisor_owning_tunnel && '
        f'printf "SUPERVISOR_TS=1\\nSUPERVISOR_PID=1\\nSUPERVISOR_MODE=healthy\\n" > "{state_file}" && '
        f'! pia_supervisor_owning_tunnel && echo SSOT_OK'
    )
    res = subprocess.run(["/bin/bash", "-c", ssot_cmd], cwd=ROOT, capture_output=True, text=True, timeout=15)
    assert res.returncode == 0 and "SSOT_OK" in res.stdout, res.stderr

    wait_script = os.path.join(ROOT, "orchestratorOS", "livecheck", "wait_for_vpn.sh")
    _write_bash_stub(fake_bin, "systemctl", "exit 0")

    # 2. Fast-path when VPN route is already UP (< 1.5s)
    _write_bash_stub(fake_bin, "piactl", 'echo Connected')
    _write_bash_stub(fake_bin, "ip", 'if [ "${1:-}" = "link" ]; then echo "3: wgpia0: <POINTOPOINT,NOARP,UP,LOWER_UP>"; else echo "1.1.1.1 dev wgpia0 src 10.0.0.2"; fi')
    t0 = time.monotonic()
    r_up = subprocess.run(
        ["/bin/bash", wait_script],
        cwd=ROOT,
        env={"PATH": f"{fake_bin}:/usr/bin:/bin", "AGY_VPN_WAIT_MAX_SEC": "8"},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert r_up.returncode == 0 and (time.monotonic() - t0) < 1.5

    # 3. Fast-path when PIA is intentionally Disconnected (ISP fallback) and default route is up
    _write_bash_stub(fake_bin, "piactl", 'echo Disconnected')
    _write_bash_stub(fake_bin, "ip", 'if [ "${1:-}" = "route" ]; then echo "1.1.1.1 via 192.168.0.1 dev ens18"; else exit 1; fi')
    t0 = time.monotonic()
    r_isp = subprocess.run(
        ["/bin/bash", wait_script],
        cwd=ROOT,
        env={"PATH": f"{fake_bin}:/usr/bin:/bin", "AGY_VPN_WAIT_MAX_SEC": "8"},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert r_isp.returncode == 0 and (time.monotonic() - t0) < 1.5

    # 4. Bounded wall-clock wait when PIA is stuck in Connecting -- exits 0 after AGY_VPN_WAIT_MAX_SEC
    _write_bash_stub(fake_bin, "piactl", 'echo Connecting')
    _write_bash_stub(fake_bin, "ip", 'if [ "${1:-}" = "route" ]; then echo "1.1.1.1 via 192.168.0.1 dev ens18"; else exit 1; fi')
    t0 = time.monotonic()
    r_cap = subprocess.run(
        ["/bin/bash", wait_script],
        cwd=ROOT,
        env={"PATH": f"{fake_bin}:/usr/bin:/bin", "AGY_VPN_WAIT_MAX_SEC": "2"},
        capture_output=True,
        text=True,
        timeout=10,
    )
    elapsed = time.monotonic() - t0
    assert r_cap.returncode == 0 and 1.5 <= elapsed < 4.5


def test_pia_audit_fixes_readonly_check_dip_priority_tunnel_stuck_and_notify_bypass(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    piactl_calls = tmp_path / "piactl_calls.log"

    # Alphabetical regions list where dedicated-belgium appears BEFORE dedicated-de-frankfurt
    _write_bash_stub(
        fake_bin,
        "piactl",
        f'printf "%s\\n" "$*" >> "{piactl_calls}"\n'
        'case "${1:-} ${2:-}" in\n'
        '  "get regions") printf "dedicated-belgium-212.102.35.157\\ndedicated-de-frankfurt-192.109.159.105\\nde-frankfurt\\n" ;;\n'
        '  "get connectionstate") echo Disconnected ;;\n'
        '  "get vpnip") echo Unknown ;;\n'
        '  "get region") echo dedicated-belgium-212.102.35.157 ;;\n'
        '  "get portforward") echo Inactive ;;\n'
        '  "-v ") echo 3.7.2 ;;\n'
        '  *) exit 1 ;;\n'
        'esac',
    )
    _write_bash_stub(fake_bin, "ip", "exit 1")
    _write_bash_stub(fake_bin, "curl", "exit 1")

    # 1. pia_preferred_dedicated_region selects Frankfurt DIP over Belgium DIP despite alphabetical sort
    dip_cmd = (
        f'export PATH="{fake_bin}:/usr/bin:/bin"; '
        f'source "{os.path.join(ROOT, "orchestratorOS", "lib", "env_common.sh")}"; '
        f'source "{os.path.join(ROOT, "orchestratorOS", "lib", "pia_common.sh")}"; '
        f'pia_preferred_dedicated_region'
    )
    r_dip = subprocess.run(["/bin/bash", "-c", dip_cmd], cwd=ROOT, capture_output=True, text=True, timeout=10)
    assert r_dip.returncode == 0
    assert r_dip.stdout.strip() == "dedicated-de-frankfurt-192.109.159.105"

    # 2. pia_supervisor.sh --check is strictly read-only (never calls dedicatedip add or login) and reports DISCONNECTED
    _write_bash_stub(
        fake_bin,
        "piactl",
        f'printf "%s\\n" "$*" >> "{piactl_calls}"\n'
        'case "${1:-} ${2:-}" in\n'
        '  "get regions") printf "de-frankfurt\\n" ;;\n'
        '  "get connectionstate") echo Disconnected ;;\n'
        '  "get vpnip") echo Unknown ;;\n'
        '  "get region") echo de-frankfurt ;;\n'
        '  "get portforward") echo Inactive ;;\n'
        '  *) exit 1 ;;\n'
        'esac',
    )
    piactl_calls.write_text("", encoding="utf-8")
    sup_script = os.path.join(ROOT, "orchestratorOS", "livecheck", "pia_supervisor.sh")
    r_sup = subprocess.run(
        ["/bin/bash", sup_script, "--check"],
        cwd=ROOT,
        env={
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "PIA_DIP_TOKEN_FRANKFURT": "FAKE_TOKEN",
            "PIA_USER": "u",
            "PIA_PASS": "p",
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert r_sup.returncode == 1
    assert "Region Role:      DISCONNECTED" in r_sup.stdout
    calls_text = piactl_calls.read_text(encoding="utf-8")
    assert "dedicatedip" not in calls_text and "login" not in calls_text

    # 3. manage_agy.sh is_tunnel_stuck handles 80-line file_watcher.go bursts without hair-triggering on 1-2 transient retries
    fake_home = tmp_path / "agy_home"
    agy_cli_dir = fake_home / ".gemini" / "antigravity-cli"
    agy_cli_dir.mkdir(parents=True)
    log_file = agy_cli_dir / "cli.log"
    _write_bash_stub(fake_bin, "agy", "echo 'Antigravity CLI 1.4.0'")
    _write_bash_stub(fake_bin, "systemctl", "exit 1")

    manage_agy = os.path.join(ROOT, "orchestratorOS", "admin", "manage_agy.sh")
    noise = "".join(f"E1110 20:20:01.{i:06d} 89 file_watcher.go:194] skipping empty or temp file:\n" for i in range(80))

    def check_stuck() -> int:
        snippet = (
            f'export HOME="{fake_home}" PATH="{fake_bin}:/usr/bin:/bin" XDG_RUNTIME_DIR="{tmp_path}"; '
            f'eval "$(sed \'/^_MANAGE_AGY_LOCKED=0/,$d\' "{manage_agy}")"; '
            f'is_tunnel_stuck'
        )
        return subprocess.run(["/bin/bash", "-c", snippet, manage_agy], cwd=ROOT, capture_output=True, text=True, timeout=10).returncode

    # Single startup failure without prior Connected -> not stuck yet
    log_file.write_text("W1110 20:00:01.000000 1 remote_control_v2.go:1] Connection status: Failed before open\n" + noise, encoding="utf-8")
    assert check_stuck() != 0

    # Connected -> inbound silence timeout -> 1st or 2nd transient Failed before open -> NOT stuck yet (allows built-in backoff recovery)
    log_file.write_text(
        "I1110 20:10:00.000000 1 remote_control_v2.go:1] Connection status: Connected\n"
        "W1110 20:16:41.000000 1 client.go:478] No inbound traffic from server for 1m0s; assuming dead tunnel and closing\n"
        "W1110 20:17:29.000000 1 remote_control_v2.go:1] Connection status: Failed before open\n"
        + noise
        + "W1110 20:18:19.000000 1 remote_control_v2.go:1] Connection status: Failed before open\n"
        + noise,
        encoding="utf-8",
    )
    assert check_stuck() != 0

    # Three consecutive failures diluted by 80 file_watcher.go lines -> genuinely stuck!
    log_file.write_text(
        log_file.read_text(encoding="utf-8")
        + "W1110 20:19:15.000000 1 remote_control_v2.go:1] Connection status: Failed before open\n"
        + noise,
        encoding="utf-8",
    )
    assert check_stuck() == 0

    # Recovered to Connected -> healthy!
    log_file.write_text(
        log_file.read_text(encoding="utf-8")
        + "I1110 20:20:10.000000 1 remote_control_v2.go:1] Connection status: Connected\n",
        encoding="utf-8",
    )
    assert check_stuck() != 0

    # 4. os_notify.sh bypasses active topic gate on mixed-case recovery titles ("PIA VPN: Recovered", "PIA VPN: Restored...")
    notify_out = tmp_path / "notify_out.txt"
    notify_cmd = f"""
set -euo pipefail
ROOT="{tmp_path}"
mkdir -p "$ROOT/logs"
env_get() {{ echo "${{2:-}}"; }}
source "{os.path.join(ROOT, "orchestratorOS", "lib", "os_notify.sh")}"
send_os_email() {{ :; }}
curl() {{
    local title=""
    while [ $# -gt 0 ]; do
        case "$1" in
            -H)
                case "$2" in Title:\\ *) title="${{2#Title: }}" ;; esac
                shift 2 ;;
            -d) shift 2 ;;
            *) shift ;;
        esac
    done
    printf '%s\\n' "$title" >> "{notify_out}"
    return 0
}}
export OS_NOTIFY_DEDUP_SECONDS=900 OS_NOTIFY_GATE_SECONDS=300 OS_NOTIFY_TOPIC_BULK_SECONDS=300
send_os_ntfy "PIA VPN: Frankfurt DIP Failed" "Falling back to Belgium" "high" "ntfy-error-test"
send_os_ntfy "PIA VPN: Restored to Primary DIP" "Frankfurt DIP back" "default" "ntfy-error-test"
"""
    subprocess.run(["/bin/bash", "-c", notify_cmd], cwd=ROOT, check=True, timeout=10)
    sent_titles = notify_out.read_text(encoding="utf-8").splitlines()
    assert "PIA VPN: Frankfurt DIP Failed" in sent_titles
    assert "PIA VPN: Restored to Primary DIP" in sent_titles







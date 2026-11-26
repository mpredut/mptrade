"""Unit tests for AutonomousIntentGateway in orchestratorTrade."""
from datetime import datetime, timezone
import json
import math
import os
import shutil
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from botcore import clean_scale
from intelligence.sentiment.collectors.portfolio_collector import (
    harvest_bot_runtime_states,
    harvest_tradable_positions_and_orders,
)
from intelligence.sentiment.evaluators.price_series_evaluator import apply_short_trend_item
from intelligence.sentiment.guards.high_stake_guard import collect_pretrade_telemetry
import order_retry
from orchestratorOS.agy.topics import condense_decision, topic_spec
from orchestratorOS.lib.technical_repair_ssot import LogEventSummary
from orchestratorOS.supervisor.autonomous_ai_reconciler import (
    AutonomousAIReconciler,
    TelemetrySensorSnapshot,
)
from orchestratorOS.supervisor.prompt_formatters import build_supervisor_prompt
from orchestratorTrade.autonomous_intent_gateway import (
    AUTONOMOUS_BYPASS_GUARDS,
    AutonomousIntentGateway,
)
from providers import order_id_context as rc
from providers.execution_audit import (
    _resolve_intent_execution_details,
    build_last_cycle_execution_status,
    cancel_existing_ai_open_orders,
    collect_known_ai_order_ids,
    durable_intents_lock,
    execute_enforce_intent_order,
    extract_intent_notional,
    find_duplicate_ai_open_limit_order,
    intent_venue_client_order_ids,
    load_durable_intents_state,
    normalize_trading_decisions,
    reconcile_intents_from_execution_audit,
    save_durable_intents_state,
    summarize_recent_ai_intents,
    upsert_durable_intent,
)
from providers.hyperliquid_provider import HyperliquidProvider
from providers.kraken_provider import KrakenProvider
from providers.order_id_context import REPRICEABLE_BOTS, resolve_order_owner
from providers.order_ohlc_utils import normalize_open_order


@pytest.fixture
def temp_workspace(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    cache = ws / "cachedb"
    cache.mkdir(parents=True, exist_ok=True)

    # Write a test order_guard.conf
    conf_path = ws / "order_guard.conf"
    with open(conf_path, "w", encoding="utf-8") as f:
        f.write("autonomous_trading_mode = shadow\n")
        f.write("autonomous_trading_max_notional_usd = 2000.0\n")
        f.write("autonomous_trading_min_notional_usd = 10.0\n")
        f.write("autonomous_trading_min_confidence = 0.85\n")
        f.write("autonomous_trading_notify = 1\n")

    return str(ws), str(cache)


def test_gateway_initialization(temp_workspace):
    ws, _ = temp_workspace
    gw = AutonomousIntentGateway(workspace_dir=ws, cache_dir="cachedb")
    assert gw.mode == "shadow"
    assert gw.max_notional_usd == 2000.0
    assert gw.min_notional_usd == 10.0
    assert gw.min_confidence == 0.85
    assert gw.notify_enabled is True
    assert gw.load_all_intents() == []


def test_gateway_fails_fast_without_usd_config_keys(tmp_path):
    ws = tmp_path / "legacy_ws"
    ws.mkdir()
    legacy_conf = ws / "order_guard.conf"
    legacy_conf.write_text(
        "autonomous_trading_mode = shadow\n"
        "autonomous_trading_max_notional_eur = 500.0\n"
        "autonomous_trading_min_notional_eur = 10.0\n"
        "autonomous_trading_min_confidence = 0.85\n"
        "autonomous_trading_notify = 1\n",
        encoding="utf-8",
    )
    with patch(
        "orchestratorTrade.autonomous_intent_gateway.load_key_value_conf",
        return_value={
            "autonomous_trading_mode": "shadow",
            "autonomous_trading_max_notional_eur": "500.0",
            "autonomous_trading_min_confidence": "0.85",
            "autonomous_trading_notify": "1",
        },
    ):
        with pytest.raises(KeyError, match="autonomous_trading_min_notional_usd"):
            AutonomousIntentGateway(workspace_dir=str(ws), cache_dir="cachedb")


def test_intent_persistence(temp_workspace):
    ws, _ = temp_workspace
    gw = AutonomousIntentGateway(workspace_dir=ws, cache_dir="cachedb")

    intents = [
        {
            "intent_id": "test_1",
            "timestamp": time.time(),
            "decision": "BUY",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "confidence": 0.90,
            "suggested_notional_usd": 500.0,
            "status": "PENDING",
            "thesis": "High breadth",
        }
    ]
    ok = gw.save_all_intents(intents)
    assert ok is True

    loaded = gw.load_all_intents()
    assert len(loaded) == 1
    assert loaded[0]["intent_id"] == "test_1"
    assert loaded[0]["symbol"] == "BTCUSDC"


def test_process_intent_low_confidence_rejected(temp_workspace):
    ws, _ = temp_workspace
    gw = AutonomousIntentGateway(workspace_dir=ws, cache_dir="cachedb")

    intent = {
        "intent_id": "low_conf_1",
        "decision": "BUY",
        "symbol": "BTCUSDC",
        "venue": "binance",
        "confidence": 0.70,  # Below 0.85
        "suggested_notional_usd": 200.0,
        "status": "PENDING",
    }
    processed = gw.process_intent(intent)
    assert processed["status"] == "REJECTED_LOW_CONFIDENCE"
    assert len(processed.get("history", [])) == 1


def test_process_intent_unknown_instrument_rejected(temp_workspace):
    ws, _ = temp_workspace
    gw = AutonomousIntentGateway(workspace_dir=ws, cache_dir="cachedb")

    intent = {
        "intent_id": "unknown_inst_1",
        "decision": "BUY",
        "symbol": "XYZNONEXISTENT",
        "venue": "binance",
        "confidence": 0.90,
        "suggested_notional_usd": 100.0,
        "status": "PENDING",
    }
    processed = gw.process_intent(intent, instruments_map={})
    assert processed["status"] == "REJECTED_UNKNOWN_INSTRUMENT"


def _make_mock_inst(
    symbol="BTCUSDC",
    *,
    base="BTC",
    quote="USDC",
    venue="binance",
    price=80000.0,
    free_balance=None,
    enabled=True,
    place_return=None,
):
    inst = MagicMock()
    inst.enabled = enabled
    inst.symbol = symbol
    inst.base = base
    inst.quote = quote
    inst.provider_name = venue
    inst._provider.name = venue
    inst._provider.get_current_price.return_value = price
    if free_balance is not None:
        inst._provider.free_balance.return_value = free_balance
    if place_return is not None:
        inst.place.return_value = place_return
    return inst


def test_process_intent_shadow_mode_observation(temp_workspace):
    ws, _ = temp_workspace
    gw = AutonomousIntentGateway(workspace_dir=ws, cache_dir="cachedb", mode="shadow", max_notional_usd=2000.0)

    mock_inst = _make_mock_inst()

    intent = {
        "intent_id": "shadow_buy_1",
        "decision": "BUY",
        "symbol": "BTCUSDC",
        "venue": "binance",
        "confidence": 0.92,
        "suggested_notional_usd": 2500.0,  # Capped at 2000.0 USD
        "status": "PENDING",
        "thesis": "Bull breakout with positive macro sentiment",
    }

    with patch("order_guard.daily_limit_guard", return_value=(True, None)), \
         patch.object(gw, "_dispatch_notify", return_value=True) as mock_notify:
        processed = gw.process_intent(intent, instruments_map={"BTC": mock_inst})

    mock_inst.place.assert_not_called()
    assert processed["status"] == "OBSERVED_SHADOW"
    assert processed["history"][-1]["notional_usd"] == 2000.0
    mock_notify.assert_called_once()
    assert "BTCUSDC" in mock_notify.call_args[1]["symbol"]
    assert mock_notify.call_args[1]["title"] == "SHADOW BUY BTC@80000.00 ~2000 $~ [binance:auto_cryptotrade_executor]"


@pytest.mark.parametrize("side", ["BUY", "SELL"])
def test_process_intent_symmetric_buy_and_sell_bypass_all_profit_guards(temp_workspace, side):
    ws, _ = temp_workspace
    gw = AutonomousIntentGateway(workspace_dir=ws, cache_dir="cachedb", mode="enforce")

    mock_inst = _make_mock_inst(
        price=79500.0,
        free_balance=2500.0 if side == "BUY" else 0.08,
        place_return={"orderId": f"ord_{side}", "status": "FILLED"},
    )

    intent = {
        "id": f"intent_sym_{side.lower()}",
        "decision": side,
        "symbol": "BTCUSDC",
        "venue": "binance",
        "confidence": 0.93,
        "suggested_notional_usd": 1200.0,
        "status": "PENDING",
        "thesis": f"Symmetric {side} decision including protective de-risking below entry",
    }

    #Neither profit_guard nor daily_limit_guard may ever block autonomous AI orders on BUY or SELL.
    with patch("order_guard.profit_guard", side_effect=AssertionError("profit_guard must not be called")), \
         patch("order_guard.daily_limit_guard", side_effect=AssertionError("daily_limit_guard must not be called")), \
         patch.object(gw, "_dispatch_notify", return_value=True) as mock_notify:
        processed = gw.process_intent(intent, instruments_map={"BTC": mock_inst})

    assert processed["status"] == "EXECUTED"
    assert processed["execution_payload"]["orderId"] == f"ord_{side}"
    mock_inst.place.assert_called_once()
    kwargs = mock_inst.place.call_args.kwargs
    assert kwargs["side"] == side
    assert kwargs["market"] is False
    assert kwargs["smart"] is False
    assert kwargs["price"] == pytest.approx(79500.0)
    assert processed["order_type"] == "LIMIT"
    assert processed["limit_price"] == pytest.approx(79500.0)
    assert "force" not in kwargs
    assert "is_market" not in kwargs
    assert kwargs["enforce_business_minimum"] is False
    assert kwargs["wait_for_trend"] is False
    assert kwargs["regime_context"] is None
    assert kwargs["intent_id"] == f"intent_sym_{side.lower()}"
    assert kwargs["client_order_id"] == f"AI_intent_sym_{side.lower()}"
    assert kwargs["bypass_guards"] == AUTONOMOUS_BYPASS_GUARDS
    assert {"all", "daily_limit", "profit_guard", "trend", "qlimit", "intelligence", "llm"} <= set(kwargs["bypass_guards"])
    assert mock_notify.call_args[1]["title"] == f"FILLED {side} BTC@79500.00 ~1200 $~ [binance:auto_cryptotrade_executor]"


def test_process_intent_clamps_to_available_wallet_balance_and_blocks_zero_balance(temp_workspace):
    ws, _ = temp_workspace
    gw = AutonomousIntentGateway(workspace_dir=ws, cache_dir="cachedb", mode="enforce")

    # 1. SELL $2000 on ARBUSDC when wallet holds 600 ARB @ $0.20 ($120) -> smoothly clamps to $120
    mock_arb = _make_mock_inst(
        "ARBUSDC",
        base="ARB",
        quote="USDC",
        price=0.20,
        free_balance=600.0,
        place_return={"orderId": "arb_sell_1"},
    )

    sell_intent = {
        "intent_id": "clamp_sell_1",
        "decision": "SELL",
        "symbol": "ARBUSDC",
        "venue": "binance",
        "confidence": 0.91,
        "suggested_notional_usd": 2000.0,
        "status": "PENDING",
    }
    with patch("order_guard.daily_limit_guard", return_value=(True, None)), \
         patch.object(gw, "_dispatch_notify", return_value=True):
        res_sell = gw.process_intent(sell_intent, instruments_map={"ARB": mock_arb})

    assert res_sell["status"] == "EXECUTED"
    assert mock_arb.place.call_args.kwargs["qty"] == pytest.approx(600.0)
    assert res_sell["history"][-1]["notional_usd"] == pytest.approx(120.0)

    # 2. BUY on Kraken HYPEUSD when free USD balance is $0.00 -> blocked cleanly before placement
    mock_krk = _make_mock_inst(
        "HYPEUSD",
        base="HYPE",
        quote="USD",
        venue="kraken",
        price=75.0,
        free_balance=0.0,
    )

    buy_zero = {
        "intent_id": "zero_usd_buy",
        "decision": "BUY",
        "symbol": "HYPEUSD",
        "venue": "kraken",
        "confidence": 0.92,
        "suggested_notional_usd": 500.0,
        "status": "PENDING",
    }
    with patch.object(gw, "_dispatch_notify", return_value=True):
        res_zero = gw.process_intent(buy_zero, instruments_map={"KRAKEN_HYPE": mock_krk})

    assert res_zero["status"] == "BLOCKED_BY_GUARD"
    assert "Insufficient free USD balance" in res_zero["history"][-1]["guard_reason"]
    mock_krk.place.assert_not_called()


def test_multi_tranche_same_side_scaling_has_zero_dedup_lock_and_previous_recovery(temp_workspace):
    ws, _ = temp_workspace
    gw = AutonomousIntentGateway(workspace_dir=ws, cache_dir="cachedb", mode="enforce")

    mock_inst = _make_mock_inst(
        free_balance=0.09,  # $7,200 USD BTC position
        place_return={"orderId": "tranche_ok"},
    )

    sell_dec = {
        "decision": "SELL",
        "symbol": "BTCUSDC",
        "venue": "binance",
        "confidence": 0.92,
        "suggested_notional_usd": 2000.0,
        "thesis": "Protective multi-tranche trim",
    }

    with patch.object(gw, "_dispatch_notify", return_value=True):
        first = gw.submit(sell_dec, source="auto_trade_executor", max_mode="enforce", instruments_map={"BTC": mock_inst})
        second = gw.submit(sell_dec, source="auto_trade_executor", max_mode="enforce", instruments_map={"BTC": mock_inst})
        assert first["status"] == "EXECUTED"
        assert second["status"] == "EXECUTED"
        assert second["intent_id"] != first["intent_id"]
        assert mock_inst.place.call_count == 2
        cid_1 = mock_inst.place.call_args_list[0].kwargs["client_order_id"]
        cid_2 = mock_inst.place.call_args_list[1].kwargs["client_order_id"]
        assert cid_1.startswith("AI_") and cid_2.startswith("AI_")
        assert cid_1 != cid_2

    # Verify .previous crash recovery on corrupted primary intents file
    shutil.copyfile(gw.intents_file, f"{gw.intents_file}.previous")
    with open(gw.intents_file, "w", encoding="utf-8") as f:
        f.write("{corrupted_json")
    recovered = gw.load_all_intents()
    assert len(recovered) == 2
    assert recovered[-1]["intent_id"] == second["intent_id"]


def test_process_all_pending_lifecycle(temp_workspace):
    ws, _ = temp_workspace
    gw = AutonomousIntentGateway(workspace_dir=ws, cache_dir="cachedb", mode="shadow")

    intents = [
        {
            "intent_id": "intent_old",
            "decision": "BUY",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "confidence": 0.90,
            "status": "EXECUTED",
        },
        {
            "intent_id": "intent_pending",
            "decision": "BUY",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "confidence": 0.91,
            "suggested_notional_usd": 450.0,
            "status": "PENDING",
        },
    ]
    gw.save_all_intents(intents)

    mock_inst = _make_mock_inst()

    with patch("instrument.load_instruments", return_value={"BTC": mock_inst}), \
         patch("order_guard.daily_limit_guard", return_value=(True, None)), \
         patch.object(gw, "_dispatch_notify", return_value=True):
        processed = gw.process_all_pending()

    assert len(processed) == 1
    assert processed[0]["intent_id"] == "intent_pending"
    assert processed[0]["status"] == "OBSERVED_SHADOW"

    reloaded = gw.load_all_intents()
    assert len(reloaded) == 2
    assert reloaded[0]["status"] == "EXECUTED"
    assert reloaded[1]["status"] == "OBSERVED_SHADOW"


def test_resolve_instrument_strict_venue_isolation_and_disabled_filtering(temp_workspace):
    ws, _ = temp_workspace
    gw = AutonomousIntentGateway(workspace_dir=ws, cache_dir="cachedb")

    mock_binance = _make_mock_inst("BTCUSDC", venue="binance")
    mock_kraken = _make_mock_inst("ETHUSD", base="ETH", quote="USD", venue="kraken")
    mock_disabled = _make_mock_inst("HYPEUSDC", base="HYPE", venue="hyperliquid", enabled=False)

    inst_map = {
        "BTC_BINANCE": mock_binance,
        "ETH_KRAKEN": mock_kraken,
        "HL_DISABLED": mock_disabled,
    }

    assert gw.resolve_instrument("BTCUSDC", venue="kraken", instruments_map=inst_map) is None
    assert gw.resolve_instrument("BTCUSDC", venue="binance", instruments_map=inst_map) is mock_binance
    assert gw.resolve_instrument("HYPEUSDC", venue="hyperliquid", instruments_map=inst_map) is None


def test_load_tradable_instruments_includes_kraken_cache_pairs(temp_workspace):
    ws, _ = temp_workspace
    krk_dir = os.path.join(ws, "kraken")
    os.makedirs(krk_dir, exist_ok=True)
    with open(os.path.join(krk_dir, "config.env"), "w", encoding="utf-8") as f:
        f.write("KRAKEN_CACHE_PAIRS=HYPEUSD,TAOUSD,ADAUSD\n")

    gw = AutonomousIntentGateway(workspace_dir=ws, cache_dir="cachedb")
    mock_btc = MagicMock(enabled=True, symbol="BTCUSDC", provider_name="binance")
    mock_hype = MagicMock(enabled=True, symbol="HYPEUSD", provider_name="kraken")
    mock_hl = MagicMock(enabled=False, symbol="HYPEUSDC", provider_name="hyperliquid")

    with patch(
        "instrument.load_instruments",
        return_value={"BINANCE_BTC": mock_btc, "KRAKEN_HYPE": mock_hype, "HL_HYPE": mock_hl},
    ):
        tradable = gw.load_tradable_instruments()

    assert "BINANCE_BTC" in tradable
    assert "KRAKEN_HYPE" in tradable
    assert "HL_HYPE" not in tradable
    assert "KRAKEN_TAO" in tradable and tradable["KRAKEN_TAO"].symbol == "TAOUSD"
    assert "KRAKEN_ADA" in tradable and tradable["KRAKEN_ADA"].symbol == "ADAUSD"


def test_process_intent_mode_off_rejected(temp_workspace):
    ws, _ = temp_workspace
    gw = AutonomousIntentGateway(workspace_dir=ws, cache_dir="cachedb", mode="off")

    mock_inst = _make_mock_inst()

    intent = {
        "intent_id": "mode_off_1",
        "decision": "BUY",
        "symbol": "BTCUSDC",
        "venue": "binance",
        "confidence": 0.95,
        "suggested_notional_usd": 200.0,
        "status": "PENDING",
    }

    with patch("order_guard.daily_limit_guard", return_value=(True, None)):
        processed = gw.process_intent(intent, instruments_map={"BTC": mock_inst})

    assert processed["status"] == "REJECTED_GATEWAY_DISABLED"
    mock_inst.place.assert_not_called()


def test_submit_is_the_single_validated_entry_point(temp_workspace):
    ws, _ = temp_workspace
    gw = AutonomousIntentGateway(workspace_dir=ws, cache_dir="cachedb", mode="enforce")
    seen_modes = []

    def fake_process(intent, instruments_map=None, mode=None):
        seen_modes.append(mode)
        intent["status"] = "OBSERVED_SHADOW" if mode == "shadow" else "EXECUTED"
        return intent

    gw.process_intent = fake_process
    assert gw.submit({"decision": "HOLD", "confidence": 0.99}, source="t", max_mode="enforce") is None
    assert gw.submit({"decision": "BUY", "symbol": "TAOUSDC", "venue": "binance", "confidence": 0.5},
                     source="t", max_mode="enforce") is None
    assert gw.submit({"decision": "BUY", "symbol": "TAOUSDC", "confidence": 0.95}, source="t",
                     max_mode="enforce") is None   # venue is mandatory

    buy = {"decision": "BUY", "symbol": "TAOUSDC", "venue": "binance", "confidence": 0.95,
           "suggested_notional_usd": 9999.0}
    first = gw.submit(buy, source="auto_trade_executor", max_mode="shadow")
    assert seen_modes == ["shadow"]                      # producer mode bounds an enforce gateway
    assert first["status"] == "OBSERVED_SHADOW" and first["created_by"] == "auto_trade_executor"
    assert first["suggested_notional_usd"] == gw.max_notional_usd   # clamped to 2000.0 USD

    second = gw.submit(buy, source="auto_trade_executor", max_mode="enforce")
    assert second["intent_id"] != first["intent_id"] and seen_modes == ["shadow", "enforce"]
    assert len(gw.load_all_intents()) == 2

    sell = gw.submit(dict(buy, decision="SELL"), source="auto_trade_executor", max_mode="enforce")
    assert sell["status"] == "EXECUTED" and seen_modes == ["shadow", "enforce", "enforce"]


def test_reconcile_failed_intent_from_execution_audit_and_capture_outcome_reason(temp_workspace):
    ws, _ = temp_workspace
    gw = AutonomousIntentGateway(workspace_dir=ws, cache_dir="cachedb", mode="enforce")

    mock_inst = _make_mock_inst("TAOUSDC", base="TAO", quote="USDC", price=251.0, free_balance=4.70)

    def place_deferred(**kwargs):
        ctx = kwargs.get("_outcome_context")
        if isinstance(ctx, dict):
            ctx["category"] = "deferred"
            ctx["reason"] = "retry_enqueued"
        return None

    mock_inst.place.side_effect = place_deferred

    intent = {
        "intent_id": "intent_1792941034_sell_taousdc",
        "decision": "SELL",
        "symbol": "TAOUSDC",
        "venue": "binance",
        "confidence": 0.89,
        "suggested_notional_usd": 376.5,
        "status": "PENDING",
    }
    with patch.object(gw, "_dispatch_notify", return_value=True):
        res = gw.process_intent(intent, instruments_map={"TAO": mock_inst})

    assert res["status"] == "FAILED"
    assert res["history"][-1]["reason"] == "retry_enqueued"
    assert res["history"][-1]["category"] == "deferred"
    gw.save_all_intents([res])

    # Now write execution_audit record showing order_retry_worker filled the order
    audit_dir = os.path.join(ws, "logger", "execution_audit")
    os.makedirs(audit_dir, exist_ok=True)
    audit_file = os.path.join(audit_dir, "execution_audit_2026-10-25.jsonl")
    with open(audit_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "event": "order_terminal",
            "intent_id": "intent_1792941034_sell_taousdc",
            "order_id": "718325868",
            "symbol": "TAOUSDC",
            "side": "sell",
            "filled_qty": 1.5,
            "cost": 376.71,
            "status": "closed",
            "venue_status": "FILLED",
            "ts": 1792941051.2,
        }) + "\n")

    reconciled = gw.load_all_intents()
    assert len(reconciled) == 1
    assert reconciled[0]["status"] == "EXECUTED"
    assert reconciled[0]["execution_payload"]["orderId"] == "718325868"
    assert reconciled[0]["execution_payload"]["reconciled_from_audit"] is True
    assert reconciled[0]["history"][-1]["action"] == "RECONCILED_EXECUTED"


def test_resolve_instrument_and_live_price_support_cross_venue_quote_variants(temp_workspace):
    ws, cache_dir = temp_workspace
    gw = AutonomousIntentGateway(workspace_dir=ws, cache_dir="cachedb", mode="shadow")

    kraken_tao = _make_mock_inst("TAOUSD", base="TAO", quote="USD", venue="kraken", price=None)

    instruments_map = {"KRAKEN_TAO": kraken_tao}
    assert gw.resolve_instrument("TAOUSDC", venue="kraken", instruments_map=instruments_map) is kraken_tao
    assert gw.resolve_instrument("TAO/USDT", venue="kraken", instruments_map=instruments_map) is kraken_tao

    price_cache = os.path.join(cache_dir, "cache_current_price.json")
    with open(price_cache, "w", encoding="utf-8") as f:
        json.dump({"items": {"TAOUSDC": [[1700000000, 345.67]]}}, f)

    assert gw._get_live_price(kraken_tao) == pytest.approx(345.67)


def test_load_tradable_instruments_discovers_hyperliquid_and_rth_gated_t212(temp_workspace):
    ws, _ = temp_workspace
    hl_dir = os.path.join(ws, "hyperliquid")
    os.makedirs(hl_dir, exist_ok=True)
    with open(os.path.join(hl_dir, "config.env"), "w", encoding="utf-8") as f:
        f.write("HL_MAINNET=true\nHL_LIVE_ORDERS=true\nHL_SPOT_TOKEN=HYPE\n")

    t212_dir = os.path.join(ws, "212trading")
    os.makedirs(t212_dir, exist_ok=True)
    with open(os.path.join(t212_dir, "runtime.env"), "w", encoding="utf-8") as f:
        f.write("T212_LIVE_ORDERS=true\nT212_ORDER_VALIDITY=DAY\n")
    with open(os.path.join(t212_dir, "config.spcx.env"), "w", encoding="utf-8") as f:
        f.write("T212_TICKER=SPCX_US_EQ\n")
    with open(os.path.join(t212_dir, "config.nvda.env"), "w", encoding="utf-8") as f:
        f.write("STRAT_PAIR=NVDA_US_EQ\n")

    gw = AutonomousIntentGateway(workspace_dir=ws, cache_dir="cachedb", mode="shadow")

    # 1. Outside US RTH (Saturday 14:00 America/New_York): Hyperliquid is active, T212 is excluded
    sat_ts = datetime(2026, 10, 24, 14, 0, tzinfo=ZoneInfo("America/New_York")).timestamp()
    closed_map = gw.load_tradable_instruments(now_ts=sat_ts)
    assert "HYPERLIQUID_HYPE" in closed_map
    assert "T212_SPCX" not in closed_map
    assert "T212_NVDA" not in closed_map

    # 2. During US RTH (Monday 10:30 America/New_York): Both Hyperliquid and T212 are active in 'all' scope,
    #    while 'crypto' scope isolates crypto and 'equities' scope isolates Trading212
    mon_rth_ts = datetime(2026, 10, 26, 10, 30, tzinfo=ZoneInfo("America/New_York")).timestamp()
    open_map = gw.load_tradable_instruments(now_ts=mon_rth_ts)
    assert "HYPERLIQUID_HYPE" in open_map
    assert "T212_SPCX" in open_map
    assert "T212_NVDA" in open_map

    crypto_map = gw.load_tradable_instruments(now_ts=mon_rth_ts, asset_scope="crypto")
    assert "HYPERLIQUID_HYPE" in crypto_map
    assert "T212_SPCX" not in crypto_map

    eq_map = gw.load_tradable_instruments(now_ts=mon_rth_ts, asset_scope="equities")
    assert "T212_SPCX" in eq_map
    assert "T212_NVDA" in eq_map
    assert "HYPERLIQUID_HYPE" not in eq_map

    # 3. Resolve symbol variants within their respective venues
    hl_inst = gw.resolve_instrument("HYPEUSDC", venue="hyperliquid", instruments_map=open_map)
    assert hl_inst is open_map["HYPERLIQUID_HYPE"]
    assert gw.resolve_instrument("HYPE", venue="hyperliquid", instruments_map=open_map) is hl_inst

    spcx_inst = gw.resolve_instrument("SPCX_US_EQ", venue="t212", instruments_map=open_map)
    assert spcx_inst is open_map["T212_SPCX"]
    assert gw.resolve_instrument("SPCX", venue="t212", instruments_map=open_map) is spcx_inst
    assert gw.resolve_instrument("NVDA", venue="t212", instruments_map=open_map) is open_map["T212_NVDA"]

    # 4. Hyperliquid BUY uses 0.95 wallet clamp buffer to accommodate 1.05x IOC market limit price
    hl_inst._provider.free_balance = MagicMock(return_value=100.0)
    ok, clamped_usd, _ = gw._clamp_notional_to_wallet_balance(hl_inst, "BUY", 40.0, 100.0)
    assert ok is True
    assert clamped_usd == pytest.approx(95.0)

    # 5. Trading212 SELL reserves $6.0 USD fractional-share dust buffer when sub-2-decimal residual exists
    spcx_inst._provider.free_balance = MagicMock(return_value=5.15296546)
    ok_t212, clamped_t212, _ = gw._clamp_notional_to_wallet_balance(spcx_inst, "SELL", 150.0, 1000.0)
    assert ok_t212 is True
    # round(5.15296546 - 6.0 / 150.0, 2) = round(5.11296546, 2) = 5.11 shares * $150 = $766.50
    assert clamped_t212 == pytest.approx(5.11 * 150.0)


def test_reconcile_intents_ignores_canceled_zero_fill_and_supports_90_900_short_trend(temp_workspace):
    ws, cache_dir = temp_workspace
    gw = AutonomousIntentGateway(workspace_dir=ws, cache_dir="cachedb", mode="enforce")

    intents = [
        {"intent_id": "id_canceled", "symbol": "BTCUSDC", "decision": "BUY", "status": "FAILED"},
        {"intent_id": "id_partial_retry", "symbol": "ETHUSDC", "decision": "BUY", "status": "FAILED"},
    ]
    audit_dir = os.path.join(ws, "logger", "execution_audit")
    os.makedirs(audit_dir, exist_ok=True)
    audit_file = os.path.join(audit_dir, "execution_audit_2026-10-28.jsonl")
    with open(audit_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "event": "submit_accepted",
            "intent_id": "id_canceled",
            "order_id": "ord_101",
            "symbol": "BTCUSDC",
            "side": "buy",
            "qty": 0.01,
            "ts": 1793000001.0,
        }) + "\n")
        f.write(json.dumps({
            "event": "order_terminal",
            "intent_id": "id_canceled",
            "order_id": "ord_101",
            "symbol": "BTCUSDC",
            "side": "buy",
            "filled_qty": 0.0,
            "status": "canceled",
            "venue_status": "CANCELED",
            "ts": 1793000005.0,
        }) + "\n")
        f.write(json.dumps({
            "event": "order_terminal_retry",
            "intent_id": "id_partial_retry",
            "order_id": "ord_202",
            "symbol": "ETHUSDC",
            "side": "buy",
            "filled_qty": 0.25,
            "cost": 650.0,
            "status": "canceled",
            "venue_status": "EXPIRED",
            "ts": 1793000010.0,
        }) + "\n")

    assert gw.reconcile_intents(intents) is True
    assert intents[0]["status"] == "FAILED"
    assert intents[1]["status"] == "EXECUTED"
    assert intents[1]["execution_payload"]["executedQty"] == "0.25"

    # Verify 90s/900s published window structure populates 3.7m fast slope from top-level tinfo
    tinfo_90_900 = {
        "current_price": 412.5,
        "final_trend": 1,
        "slope_full": 0.0021,
        "growth_coefficient": 0.42,
        "dynamic": {
            "90": {"slope_full": 0.0035, "final_trend": 1},
            "900": {"slope_full": 0.0014, "final_trend": 1},
        },
    }
    entry = {}
    assert apply_short_trend_item(entry, tinfo_90_900) is True
    assert entry["slope_3_7m"] == pytest.approx(0.0021)
    assert entry["slope_15m"] == pytest.approx(0.0014)

    with open(os.path.join(cache_dir, "cache_short_trend.json"), "w", encoding="utf-8") as f:
        json.dump({"items": {"TAOUSDC": tinfo_90_900}}, f)
    telem = collect_pretrade_telemetry("TAOUSDC", cachedb_dir=cache_dir)
    assert "3.7m=+0.00210" in telem.get("Dynamic Multi-Window Slopes", "")
    assert "15m=+0.00140" in telem.get("Dynamic Multi-Window Slopes", "")


def test_notional_confidence_t212_floor_and_account_cache_retry_hardening(temp_workspace):
    ws, _ = temp_workspace
    gw = AutonomousIntentGateway(workspace_dir=ws, cache_dir="cachedb", mode="enforce")

    # 1. extract_intent_notional parses currency strings, aliases, and rejects <= 0
    assert extract_intent_notional({"suggested_notional_usd": "$1,250.50"}) == pytest.approx(1250.50)
    assert extract_intent_notional({"notional": "800 USD"}) == pytest.approx(800.0)
    assert extract_intent_notional({"value_usd": "$450 USDC"}) == pytest.approx(450.0)
    assert extract_intent_notional({"suggested_notional_usd": "-50"}, default=25.0) == pytest.approx(25.0)

    # 2. clean_scale normalizes percentage strings and 5..100 numbers for intent confidence
    assert clean_scale("91%", 0.0, allow_percent=True) == pytest.approx(0.91)
    assert clean_scale(92, 0.0, allow_percent=True) == pytest.approx(0.92)
    assert clean_scale(0.88, 0.0, allow_percent=True) == pytest.approx(0.88)
    assert clean_scale(True, default=0.0, allow_percent=True) == 0.0

    # 3. _clamp_notional_to_wallet_balance:
    #    - rejects non-positive/NaN price on SELL
    #    - floors T212 2-decimal shares (never rounds up above wallet balance)
    #    - prevents partial T212 SELL from leaving < $6.0 unsellable remainder
    t212_inst = MagicMock(symbol="NVDA_US_EQ", base="NVDA_US_EQ", quote="USD", provider_name="t212")
    t212_inst._provider.free_balance.return_value = 1.00
    ok_nan, _, reason_nan = gw._clamp_notional_to_wallet_balance(t212_inst, "SELL", float("nan"), 50.0)
    assert ok_nan is False and "Invalid live price" in reason_nan

    # Partial T212 SELL: 1.004 shares @ $100 = $100.40; selling $96 would leave $4.40 (< $6 min), so it clamps to 0.94 shares ($94)
    t212_inst._provider.free_balance.return_value = 1.004
    ok_part, usd_part, _ = gw._clamp_notional_to_wallet_balance(t212_inst, "SELL", 100.0, 96.0)
    assert ok_part is True
    assert usd_part == pytest.approx(94.0)

    # 4. process_intent retries once on transient account_cache_not_fresh and records clamped_notional_usd
    bin_inst = MagicMock(symbol="ARBUSDC", base="ARB", quote="USDC", provider_name="binance")
    bin_inst._provider.name = "binance"
    bin_inst._provider.get_current_price.return_value = 0.20
    bin_inst._provider.free_balance.return_value = 500.0  # $100 worth of ARB
    call_n = {"n": 0}

    def place_retry_once(**kwargs):
        call_n["n"] += 1
        ctx = kwargs.get("_outcome_context")
        if call_n["n"] == 1:
            if isinstance(ctx, dict):
                ctx["category"] = "refused_pre_submit"
                ctx["reason"] = "account_cache_not_fresh"
            return None
        return {"orderId": "arb_ok_2", "cummulativeQuoteQty": "100.0"}

    bin_inst.place.side_effect = place_retry_once
    intent_retry = {
        "intent_id": "retry_fresh_1",
        "decision": "SELL",
        "symbol": "ARBUSDC",
        "venue": "binance",
        "confidence": "92%",
        "suggested_notional_usd": "$500",
        "status": "PENDING",
    }
    with patch.object(gw, "_dispatch_notify", return_value=True):
        res_retry = gw.process_intent(intent_retry, instruments_map={"ARBUSDC": bin_inst})
    assert res_retry["status"] == "EXECUTED"
    assert call_n["n"] == 2
    assert res_retry["clamped_notional_usd"] == pytest.approx(100.0)
    assert res_retry["confidence"] == pytest.approx(0.92)

    # 5. reconcile_intents records submit_accepted without fabricating fill, then computes fallback cummulativeQuoteQty on order_terminal
    audit_dir = os.path.join(ws, "logger", "execution_audit")
    os.makedirs(audit_dir, exist_ok=True)
    audit_path = os.path.join(audit_dir, "execution_audit_2026-10-29.jsonl")
    with open(audit_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "event": "submit_accepted",
            "intent_id": "id_calc_cost",
            "order_id": "ord_909",
            "symbol": "TAOUSDC",
            "side": "sell",
            "qty": 2.0,
            "price": 250.5,
            "ts": 1793100001.0,
        }) + "\n")
    pending_list = [{"intent_id": "id_calc_cost", "symbol": "TAOUSDC", "decision": "SELL", "status": "PENDING_EXECUTION"}]
    assert gw.reconcile_intents(pending_list) is True
    assert pending_list[0]["status"] == "EXECUTED"
    assert pending_list[0]["execution_payload"]["status"] == "NEW"
    assert pending_list[0]["execution_payload"]["origQty"] == "2.0"
    assert pending_list[0]["execution_payload"]["executedQty"] == ""
    assert pending_list[0]["execution_payload"]["cummulativeQuoteQty"] == ""
    assert "terminal_ts" not in pending_list[0]["execution_payload"]

    with open(audit_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "event": "order_terminal",
            "intent_id": "id_calc_cost",
            "order_id": "ord_909",
            "symbol": "TAOUSDC",
            "side": "sell",
            "filled_qty": 2.0,
            "price": 250.5,
            "venue_status": "FILLED",
            "status": "closed",
            "ts": 1793100005.0,
        }) + "\n")
    assert gw.reconcile_intents(pending_list) is True
    assert pending_list[0]["execution_payload"]["status"] == "FILLED"
    assert pending_list[0]["execution_payload"]["executedQty"] == "2.0"
    assert pending_list[0]["execution_payload"]["cummulativeQuoteQty"] == "501.0"
    assert pending_list[0]["execution_payload"]["terminal_ts"] == pytest.approx(1793100005.0)


def test_passive_limit_orders_order_replacement_and_contrarian_dip_prompt(temp_workspace):
    ws, _ = temp_workspace
    gw = AutonomousIntentGateway(workspace_dir=ws, cache_dir="cachedb", mode="enforce")

    # 1. AI_ clientOrderId prefix resolves to ai_executor (never repriced by monitororder)
    assert resolve_order_owner("AI_OR_998877") == "ai_executor"
    assert resolve_order_owner("AI_123456") == "ai_executor"
    assert "ai_executor" not in REPRICEABLE_BOTS

    # 2. Passive BUY LIMIT order with replace_open_orders=True cancels prior AI_ open BUY orders
    #    and calls inst.place(..., price=79850.0, market=False, smart=False)
    btc_inst = MagicMock(symbol="BTCUSDC", base="BTC", quote="USDC", provider_name="binance")
    btc_inst._provider.name = "binance"
    btc_inst._provider.get_current_price.return_value = 80500.0
    btc_inst._provider.free_balance.return_value = 5000.0
    btc_inst.open_orders.return_value = [
        {"orderId": "ai_old_buy", "clientOrderId": "AI_OR_111", "side": "BUY", "price": "79200.0"},
        {"orderId": "ai_sell_keep", "clientOrderId": "AI_OR_222", "side": "SELL", "price": "83000.0"},
        {"orderId": "bot_buy_keep", "clientOrderId": "TO_333", "side": "BUY", "price": "79000.0"},
    ]
    btc_inst.place.return_value = {"orderId": "ai_new_limit_1", "status": "NEW", "price": "79850.0", "origQty": "0.00501"}

    with patch.object(gw, "_dispatch_notify", return_value=True):
        rec = gw.submit({
            "decision": "BUY",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "order_type": "LIMIT",
            "limit_price": 79850.0,
            "replace_open_orders": True,
            "confidence": 0.91,
            "suggested_notional_usd": 400.0,
            "thesis": "Passive buy limit at 79,850 support bid wall during geopolitical dip",
        }, source="auto_cryptotrade_executor", max_mode="enforce", instruments_map={"BTCUSDC": btc_inst})

    assert rec["status"] == "EXECUTED"
    assert rec["order_type"] == "LIMIT"
    assert rec["limit_price"] == pytest.approx(79850.0)
    assert rec["replace_open_orders"] is True
    assert rec["canceled_prior_open_orders"] == ["ai_old_buy"]
    btc_inst.cancel_order.assert_called_once_with("ai_old_buy")
    place_kwargs = btc_inst.place.call_args.kwargs
    assert place_kwargs["market"] is False
    assert place_kwargs["smart"] is False
    assert place_kwargs["price"] == pytest.approx(79850.0)
    assert place_kwargs["qty"] == pytest.approx(400.0 / 79850.0, rel=1e-4)

    # 3. Crossing limit price (BUY limit_price > live_price) clamps safely to live_price
    with patch.object(gw, "_dispatch_notify", return_value=True):
        rec_clamp = gw.submit({
            "decision": "BUY",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "order_type": "LIMIT",
            "limit_price": 81000.0,
            "confidence": 0.90,
            "suggested_notional_usd": 250.0,
        }, source="auto_cryptotrade_executor", max_mode="enforce", instruments_map={"BTCUSDC": btc_inst})
    assert rec_clamp["status"] == "EXECUTED"
    assert rec_clamp["limit_price"] == pytest.approx(80500.0)

    # 4. Out-of-bounds or missing limit_price (< 0.5x, > 2.0x, or None when order_type="LIMIT")
    #    falls back to a safe passive price (0.995 * live_price for BUY, 1.005 * live_price for SELL)
    #    so orders are never left unsent.
    with patch.object(gw, "_dispatch_notify", return_value=True):
        rec_bad = gw.submit({
            "decision": "BUY",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "order_type": "LIMIT",
            "limit_price": 10000.0,
            "confidence": 0.92,
            "suggested_notional_usd": 300.0,
        }, source="auto_cryptotrade_executor", max_mode="enforce", instruments_map={"BTCUSDC": btc_inst})
        rec_missing_px = gw.submit({
            "decision": "BUY",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "order_type": "LIMIT",
            "limit_price": None,
            "confidence": 0.92,
            "suggested_notional_usd": 300.0,
        }, source="auto_cryptotrade_executor", max_mode="enforce", instruments_map={"BTCUSDC": btc_inst})
    assert rec_bad["status"] == "EXECUTED"
    assert rec_bad["limit_price"] == pytest.approx(80500.0 * 0.995)
    assert rec_missing_px["status"] == "EXECUTED"
    assert rec_missing_px["limit_price"] == pytest.approx(80500.0 * 0.995)

    # 5. normalize_trading_decisions, summarize_recent_ai_intents, condense_decision, and prompt_formatters
    norm = normalize_trading_decisions({
        "market_trading_decisions": [
            {
                "decision": "BUY",
                "symbol": "BTCUSDC",
                "venue": "binance",
                "order_type": "limit",
                "limit_price": "79850",
                "replace_open_orders": True,
                "confidence": 0.91,
                "suggested_notional_usd": 400.0,
            }
        ]
    })
    assert norm[0]["order_type"] == "LIMIT"
    assert norm[0]["limit_price"] == pytest.approx(79850.0)
    assert norm[0]["replace_open_orders"] is True

    recent_summary = summarize_recent_ai_intents(
        gw.load_all_intents(),
        "crypto",
        1793306600.0,
        extract_intent_notional,
    )
    assert any(r.get("order_type") == "LIMIT" and r.get("limit_price") == pytest.approx(79850.0) for r in recent_summary)

    condensed = condense_decision(json.dumps({
        "summary": "Resting passive limit bid at BTC support.",
        "market_trading_decisions": norm,
    }))
    assert "BUY(LIMIT@79850.0):BTCUSDC@0.91" in condensed

    # Verify prompt contains contrarian dip accumulation & dual execution mode instructions
    snap = TelemetrySensorSnapshot(
        geopolitical_threat={
            "threat_level": "CRITICAL_SHOCK",
            "risk_score": 0.96,
            "recommended_brake": "HARD_VETO_NEW_BUYS",
            "summary": "Headline shock",
        },
        prices_and_trends={"BTCUSDC": {"venue": "binance", "price": 80500.0}},
        portfolio_summary={
            "balances": {"USDC": 7900.0},
            "tradable_positions": [],
            "recent_ai_intents": recent_summary,
        },
    )

    prompt = build_supervisor_prompt(
        sensors=snap,
        logs=LogEventSummary(0, 0, 0),
        spec=topic_spec("auto_cryptotrade_executor"),
        min_conf=0.85,
        min_usd=10.0,
        max_usd=2000.0,
    )
    assert "HARD_VETO_NEW_BUYS" not in prompt
    assert "level=CRITICAL_SHOCK" not in prompt
    assert "brake=" not in prompt
    assert "risk_score=0.96 | summary=Headline shock" in prompt
    assert "CONTRARIAN DIP ACCUMULATION (SCALE-IN)" in prompt
    assert "SPOT MAKER 'LIMIT' EXECUTION IN THE ORDERBOOK" in prompt
    assert "do NOT sell minutes after a bot fill unless structure genuinely breaks down" in prompt
    assert "right at live BBO (`best_bid` for BUY / `best_ask` for SELL)" in prompt
    assert '"order_type": "LIMIT"' in prompt
    assert '"decision": "BUY" | "SELL" | "CANCEL" | "HOLD"' in prompt
    assert "BUY LIMIT" in prompt and "limit_px=79850.0" in prompt

    # 6. When a bot's open order locks wallet liquidity so free_balance is 0, replace_open_orders=True
    #    automatically cancels the locking bot order and re-clamps wallet balance so the AI can repair/reverse the trade
    #    using a near-market spot LIMIT order at live_price.
    locked_inst = MagicMock(symbol="BTCUSDC", base="BTC", quote="USDC", provider_name="binance")
    locked_inst._provider.name = "binance"
    locked_inst._provider.get_current_price.return_value = 80000.0
    free_state = {"BTC": 0.0, "USDC": 1000.0}
    locked_inst._provider.free_balance.side_effect = lambda asset: free_state.get(asset, 0.0)
    locked_inst.open_orders.return_value = [
        {"orderId": "bot_locking_sell_99", "clientOrderId": "MT_TP_99", "side": "SELL", "price": "84000.0", "origQty": "0.01"},
    ]

    def _unlock_on_cancel(oid):
        if oid == "bot_locking_sell_99":
            free_state["BTC"] = 0.01

    locked_inst.cancel_order.side_effect = _unlock_on_cancel
    locked_inst.place.return_value = {"orderId": "ai_repair_sell_1", "status": "FILLED", "price": "80000.0", "executedQty": "0.005"}

    with patch.object(gw, "_dispatch_notify", return_value=True):
        rec_unlock = gw.submit({
            "decision": "SELL",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "order_type": "MARKET",
            "replace_open_orders": True,
            "confidence": 0.93,
            "suggested_notional_usd": 400.0,
            "thesis": "Reverse unwanted bot buy and unlock bot-locked BTC",
        }, source="auto_cryptotrade_executor", max_mode="enforce", instruments_map={"BTCUSDC": locked_inst})

    assert rec_unlock["status"] == "EXECUTED"
    assert rec_unlock["order_type"] == "LIMIT"
    assert rec_unlock["limit_price"] == pytest.approx(80000.0)
    assert locked_inst.place.call_args.kwargs["market"] is False
    assert locked_inst.place.call_args.kwargs["smart"] is False
    assert locked_inst.place.call_args.kwargs["price"] == pytest.approx(80000.0)
    assert rec_unlock["canceled_bot_open_orders"] == ["bot_locking_sell_99"]
    assert "bot_locking_sell_99" in rec_unlock["canceled_prior_open_orders"]
    locked_inst.cancel_order.assert_called_once_with("bot_locking_sell_99")



def test_cancel_bot_open_orders_and_balance_unlock_via_replacement(temp_workspace):
    """Verify explicit cancel_bot_open_orders cancels bot orders and replace_open_orders unlocks balance when constrained."""
    ws, _ = temp_workspace
    gw = AutonomousIntentGateway(workspace_dir=ws, cache_dir="cachedb", mode="enforce")

    btc_inst = MagicMock(symbol="BTCUSDC", base="BTC", quote="USDC", provider_name="binance")
    btc_inst._provider.name = "binance"
    btc_inst._provider.get_current_price.return_value = 80000.0

    # Case 1: cancel_bot_open_orders=True on BUY cancels only same-side BUY bot orders, preserving protective SELL orders
    btc_inst._provider.free_balance.return_value = 5000.0
    btc_inst.open_orders.return_value = [
        {"orderId": "bot_buy_1", "clientOrderId": "TO_1", "side": "BUY", "price": "79000.0"},
        {"orderId": "bot_sell_1", "clientOrderId": "TO_2", "side": "SELL", "price": "81000.0"},
    ]
    btc_inst.place.return_value = {"orderId": "ai_order_1", "status": "FILLED"}

    with patch.object(gw, "_dispatch_notify", return_value=True):
        rec1 = gw.submit({
            "decision": "BUY",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "confidence": 0.92,
            "cancel_bot_open_orders": True,
            "suggested_notional_usd": 300.0,
        }, source="auto_cryptotrade_executor", max_mode="enforce", instruments_map={"BTCUSDC": btc_inst})

    assert rec1["status"] == "EXECUTED"
    assert rec1["canceled_prior_open_orders"] == ["bot_buy_1"]
    btc_inst.cancel_order.assert_called_once_with("bot_buy_1")

    # Case 2: replace_open_orders=True automatically cancels bot orders when initial wallet balance is constrained
    btc_inst.cancel_order.reset_mock()
    btc_inst._provider.free_balance.side_effect = [0.0, 500.0]  # First check returns 0.0, after cancel returns 500.0
    btc_inst.open_orders.return_value = [
        {"orderId": "bot_locking_buy", "clientOrderId": "TO_3", "side": "BUY", "price": "79000.0"},
        {"orderId": "bot_trailing_sell", "clientOrderId": "SD_0123456789abcdef0123456789abcdef", "side": "SELL", "price": "85000.0"},
    ]

    with patch.object(gw, "_dispatch_notify", return_value=True):
        rec2 = gw.submit({
            "decision": "BUY",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "confidence": 0.92,
            "replace_open_orders": True,
            "suggested_notional_usd": 400.0,
        }, source="auto_cryptotrade_executor", max_mode="enforce", instruments_map={"BTCUSDC": btc_inst})

    assert rec2["status"] == "EXECUTED"
    assert "bot_locking_buy" in rec2["canceled_prior_open_orders"]
    assert rec2["canceled_bot_open_orders"] == ["bot_locking_buy"]
    btc_inst.cancel_order.assert_called_with("bot_locking_buy")
    # The opposite-side protective SELL (trailing stop) frees no quote balance and must not be canceled
    assert "bot_trailing_sell" not in rec2["canceled_prior_open_orders"]

    # Case 3: When canceling a prior AI order alone unlocks sufficient balance, same-side bot orders are preserved
    # and canceled_bot_open_orders is not populated.
    btc_inst.cancel_order.reset_mock()
    free_usdc = {"val": 0.0}
    btc_inst._provider.free_balance.side_effect = lambda _a: free_usdc["val"]
    btc_inst.open_orders.return_value = [
        {"orderId": "ai_prior_buy_9", "clientOrderId": "AI_prior_9", "side": "BUY", "price": "79100.0"},
        {"orderId": "bot_sibling_buy_9", "clientOrderId": "TO_sibling_9", "side": "BUY", "price": "78500.0"},
    ]
    btc_inst.cancel_order.side_effect = lambda oid: free_usdc.update(val=500.0) if oid == "ai_prior_buy_9" else None

    with patch.object(gw, "_dispatch_notify", return_value=True):
        rec3 = gw.submit({
            "decision": "BUY",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "confidence": 0.92,
            "replace_open_orders": True,
            "suggested_notional_usd": 400.0,
        }, source="auto_cryptotrade_executor", max_mode="enforce", instruments_map={"BTCUSDC": btc_inst})

    assert rec3["status"] == "EXECUTED"
    assert rec3["canceled_prior_open_orders"] == ["ai_prior_buy_9"]
    assert "canceled_bot_open_orders" not in rec3
    btc_inst.cancel_order.assert_called_once_with("ai_prior_buy_9")


def test_execution_details_resting_limit_ignores_prior_recent_orders_and_collect_ai_ids():
    """Verify unfilled resting LIMIT orders and non-executed intents do not inherit prior fills from recent_orders."""

    now_ts = 1_760_000_100.0
    intents = [
        {
            "intent_id": "rest_lim_1",
            "timestamp": now_ts - 10.0,
            "decision": "BUY",
            "symbol": "HYPEUSD",
            "venue": "kraken",
            "order_type": "LIMIT",
            "limit_price": 42.0,
            "reference_price": 42.5,
            "suggested_notional_usd": 420.0,
            "status": "EXECUTED",
            "execution_payload": {"ordertxid": "KRK_LIM_1", "status": "OPEN", "vol": "10.0", "vol_exec": "0.0"},
        },
        {
            "intent_id": "blocked_1",
            "timestamp": now_ts - 5.0,
            "decision": "BUY",
            "symbol": "HYPEUSD",
            "venue": "kraken",
            "order_type": "MARKET",
            "reference_price": 42.5,
            "suggested_notional_usd": 420.0,
            "status": "BLOCKED_BY_GUARD",
            "execution_payload": None,
            "history": [{"ts": now_ts - 5.0, "action": "BLOCKED_BY_GUARD", "payload": {"oid": "HIST_OID_9"}}],
        },
    ]
    recent_orders = {
        "kraken:HYPEUSD": {
            "last_buy": {"price": 42.4, "qty": 5.0, "ts_ms": int((now_ts - 30.0) * 1000), "actor": "BOT"}
        }
    }
    open_orders = [
        {"ordertxid": "KRK_LIM_1", "venue": "kraken", "symbol": "HYPEUSD", "descr": {"type": "buy", "price": "42.0"}, "vol": "10.0", "vol_exec": "0.0"}
    ]

    last_st = build_last_cycle_execution_status(
        intents, "crypto", now_ts, extract_intent_notional, recent_orders=recent_orders, open_orders=open_orders
    )
    assert isinstance(last_st, list) and len(last_st) == 2
    assert last_st[0]["status"] == "OPEN_RESTING"
    assert last_st[0]["fill_px"] is None
    assert last_st[0]["filled_qty"] is None
    assert last_st[0]["slippage_bps"] is None
    assert last_st[1]["status"] == "BLOCKED_BY_GUARD"
    assert last_st[1]["fill_px"] is None
    assert last_st[1]["filled_qty"] is None

    summary = summarize_recent_ai_intents(
        intents, "crypto", now_ts, extract_intent_notional, recent_orders=recent_orders, open_orders=open_orders
    )
    assert "fill_px" not in summary[0]
    assert "filled_qty" not in summary[0]
    assert "fill_px" not in summary[1]

    known_ids = collect_known_ai_order_ids(intents, {"INIT_1"})
    assert {"INIT_1", "KRK_LIM_1", "HIST_OID_9"} <= known_ids
    # Venue-native AI client IDs (Binance AI_, Kraken UUID hex, HL cloid) are also known for each intent
    for it in intents:
        assert intent_venue_client_order_ids(it) <= known_ids

    # When the resting LIMIT order later fills (no longer in open_orders) and recent_orders
    # records the fill price, fill_px updates to the actual fill price rather than staying
    # locked to the initial NEW placement limit price.
    filled_intents = [
        {
            "intent_id": "filled_lim_1",
            "timestamp": now_ts - 20.0,
            "decision": "BUY",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "order_type": "LIMIT",
            "limit_price": 78500.0,
            "reference_price": 78500.0,
            "suggested_notional_usd": 392.5,
            "status": "EXECUTED",
            "execution_payload": {
                "orderId": "BIN_LIM_99",
                "status": "NEW",
                "price": "78500.0",
                "origQty": "0.005",
                "executedQty": "0.0",
            },
        }
    ]
    recent_after_fill = {
        "binance:BTCUSDC": {
            "last_buy": {"price": 78420.0, "qty": 0.005, "ts_ms": int((now_ts - 5.0) * 1000), "actor": "AI"}
        }
    }
    filled_st = build_last_cycle_execution_status(
        filled_intents, "crypto", now_ts, extract_intent_notional, recent_orders=recent_after_fill, open_orders=[]
    )
    assert isinstance(filled_st, dict)
    assert filled_st["status"] == "FILLED"
    assert filled_st["fill_px"] == pytest.approx(78420.0)
    assert filled_st["filled_qty"] == pytest.approx(0.005)
    assert filled_st["slippage_bps"] == pytest.approx(((78420.0 - 78500.0) / 78500.0) * 10000.0, abs=0.05)









def test_ai_order_identity_is_isolated_from_monitororder_repricing_and_bot_intent_cids():
    """AI orders are never repriced by monitororder, and bot/repriced orders are never adopted as AI orders."""
    # 1. AI_ cids resolve to ai_executor and are protected from monitororder repricing
    assert rc.resolve_order_owner("AI_20261102-BUY-BTCUSDC_ab12cd_7f9e1a2") == "ai_executor"
    assert rc.is_order_repriceable_by_monitororder("AI_20261102-BUY-BTCUSDC_ab12cd_7f9e1a2") is False
    # monitororder replacement cids (tradeall/manual) stay repriceable BOT lineage, never AI
    assert rc.resolve_order_owner("MO_TA_OR_MainThread_1A2B3C4D") == "tradeall"
    assert rc.resolve_order_owner("MO_MN_OR_MainThread_1A2B3C4D") == "manual"

    # 2. Gateway replace_open_orders never cancels Binance trailing_stop SD_<hex> intent orders as "AI" orders
    trailing_cid = "SD_0123456789abcdef0123456789abcdef"
    inst = MagicMock()
    inst.symbol = "BTCUSDC"
    inst.open_orders.return_value = [
        {"orderId": 101, "side": "SELL", "clientOrderId": trailing_cid, "price": "80000"},
        {"orderId": 102, "side": "SELL", "clientOrderId": "MO_TA_OR_MainThread_1A2B3C4D", "price": "80100"},
        {"orderId": 103, "side": "SELL", "clientOrderId": "AI_20261102-SELL-BTCUSDC_ab12cd_7f9e1a2", "price": "80200"},
    ]
    canceled = cancel_existing_ai_open_orders(inst, "SELL", intents=[])
    assert canceled == ["103"]
    inst.cancel_order.assert_called_once_with("103")
    # A venue-refused cancel (False) is not reported as canceled
    inst.cancel_order.reset_mock()
    inst.cancel_order.return_value = False
    assert cancel_existing_ai_open_orders(inst, "SELL", intents=[]) == []

    # 3. A response-lost AI LIMIT intent (no order_id) is never matched to a repriced BOT order on the same side
    now_ts = time.time()
    intent = {
        "intent_id": "intent-lost-1",
        "timestamp": now_ts - 5000.0,
        "venue": "binance",
        "symbol": "BTCUSDC",
        "decision": "BUY",
        "order_type": "LIMIT",
        "limit_price": 78000.0,
        "status": "EXECUTED",
        "execution_payload": None,
        "history": [],
    }
    bot_open = [{
        "venue": "binance", "symbol": "BTCUSDC", "side": "BUY", "order_id": "555", "orderId": "555",
        "actor": "BOT", "module": "tradeall", "clientOrderId": "MO_TA_OR_MainThread_1A2B3C4D", "price": 77900.0,
    }]
    details = _resolve_intent_execution_details(intent, now_ts, extract_intent_notional, open_orders=bot_open)
    assert details.get("exec_status") == "CANCELED_OR_EXPIRED"
    ai_open = [dict(bot_open[0], actor="AI", clientOrderId="AI_intent-lost-1", module=None)]
    details_ai = _resolve_intent_execution_details(intent, now_ts, extract_intent_notional, open_orders=ai_open)
    assert details_ai.get("exec_status") == "OPEN_RESTING"

    # 4. Kraken / Hyperliquid hash the AI_ client ID into 128-bit form: response-lost intents still match
    #    their own resting order, open orders are labeled AI, and another AI intent's order is not adopted.
    for venue_name, native_cid in (
        ("kraken", KrakenProvider.derive_client_order_id("AI_intent-lost-1")),
        ("hyperliquid", HyperliquidProvider.derive_client_order_id("AI_intent-lost-1")),
    ):
        v_intent = dict(intent, venue=venue_name, symbol="BTCUSD")
        raw_oo = {"orderId": "V1", "side": "BUY", "price": "77900", "clientOrderId": native_cid}
        known = collect_known_ai_order_ids([v_intent])
        norm_oo = normalize_open_order(raw_oo, venue=venue_name, symbol="BTCUSD", known_ai_order_ids=known)
        assert norm_oo["actor"] == "AI"
        v_details = _resolve_intent_execution_details(v_intent, now_ts, extract_intent_notional, open_orders=[norm_oo])
        assert v_details.get("exec_status") == "OPEN_RESTING"
        other_oo = dict(norm_oo, clientOrderId=KrakenProvider.derive_client_order_id("AI_other-intent-2"))
        other_details = _resolve_intent_execution_details(v_intent, now_ts, extract_intent_notional, open_orders=[other_oo])
        assert other_details.get("exec_status") == "CANCELED_OR_EXPIRED"


def test_execution_details_finds_own_ai_fill_behind_later_bot_fill_and_matches_by_order_id():
    """A later same-side BOT fill (e.g. monitororder-repriced tradeall order) must not hide the intent's own fill."""

    now_ts = time.time()
    intent = {
        "intent_id": "lim-hidden-fill-1",
        "timestamp": now_ts - 5000.0,
        "venue": "binance",
        "symbol": "BTCUSDC",
        "decision": "BUY",
        "order_type": "LIMIT",
        "limit_price": 78000.0,
        "status": "EXECUTED",
        "execution_payload": {"orderId": "AI_OID_1", "status": "NEW", "price": "78000.0", "origQty": "0.004", "executedQty": "0.0"},
        "history": [],
    }
    ai_fill = {"price": 77990.0, "qty": 0.004, "ts_ms": int((now_ts - 4000.0) * 1000), "actor": "AI", "order_id": "AI_OID_1"}
    other_ai_fill = {"price": 77500.0, "qty": 0.01, "ts_ms": int((now_ts - 3000.0) * 1000), "actor": "AI", "order_id": "AI_OID_2"}
    bot_fill = {"price": 77800.0, "qty": 0.02, "ts_ms": int((now_ts - 60.0) * 1000), "actor": "BOT", "module": "tradeall", "order_id": "MO_1"}
    recent = {"binance:BTCUSDC": {"last_buy": bot_fill, "recent_buys": [ai_fill, other_ai_fill, bot_fill]}}
    details = _resolve_intent_execution_details(intent, now_ts, extract_intent_notional, recent_orders=recent, open_orders=[])
    assert details["exec_status"] == "FILLED"
    assert details["fill_px"] == pytest.approx(77990.0)
    assert details["filled_qty"] == pytest.approx(0.004)


def test_reconcile_from_audit_scans_only_days_since_oldest_pending_intent(tmp_path):
    """Audit reconciliation skips UTC-day files older than the oldest pending intent (minus 1 day) and caches files."""
    audit_dir = tmp_path / "execution_audit"
    audit_dir.mkdir()
    intent_ts = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc).timestamp()
    fill_rec = {
        "event": "order_terminal", "intent_id": "intent_old_skip", "order_id": "X1", "venue": "binance",
        "symbol": "BTCUSDC", "side": "buy", "status": "filled", "venue_status": "FILLED",
        "filled_qty": 0.01, "cost": 800.0, "ts": intent_ts + 60.0,
    }
    # A record in a file 5 days before the intent cannot belong to it and must not be scanned
    (audit_dir / "execution_audit_2026-10-15.jsonl").write_text(json.dumps(fill_rec) + "\n", encoding="utf-8")
    intents = [{
        "intent_id": "intent_old_skip", "timestamp": intent_ts, "decision": "BUY", "symbol": "BTCUSDC",
        "venue": "binance", "status": "FAILED", "execution_payload": None, "history": [],
    }]
    assert reconcile_intents_from_execution_audit(intents, str(audit_dir)) is False
    assert intents[0]["status"] == "FAILED"

    # The same record in the intent's own UTC day is reconciled
    (audit_dir / "execution_audit_2026-10-20.jsonl").write_text(json.dumps(fill_rec) + "\n", encoding="utf-8")
    assert reconcile_intents_from_execution_audit(intents, str(audit_dir)) is True
    assert intents[0]["status"] == "EXECUTED"
    assert intents[0]["execution_payload"]["status"] == "FILLED"

    # Abandoned pre-submit PENDING_EXECUTION (>10m old, zero audit events) self-heals to FAILED
    today_day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    (audit_dir / f"execution_audit_{today_day}.jsonl").write_text("", encoding="utf-8")
    stale_pre_submit = {
        "intent_id": "intent_abandoned_pre_submit",
        "timestamp": time.time() - 900.0,
        "decision": "SELL",
        "symbol": "BTCUSDC",
        "venue": "binance",
        "status": "PENDING_EXECUTION",
        "execution_payload": None,
        "history": [],
    }
    fresh_pre_submit = {
        "intent_id": "intent_fresh_pre_submit",
        "timestamp": time.time() - 30.0,
        "decision": "SELL",
        "symbol": "BTCUSDC",
        "venue": "binance",
        "status": "PENDING_EXECUTION",
        "execution_payload": None,
        "history": [],
    }
    mixed_pending = [stale_pre_submit, fresh_pre_submit]
    assert reconcile_intents_from_execution_audit(mixed_pending, str(audit_dir)) is True
    assert stale_pre_submit["status"] == "FAILED"
    assert stale_pre_submit["history"][-1]["action"] == "RECONCILED_ABANDONED_PRE_SUBMIT"
    assert fresh_pre_submit["status"] == "PENDING_EXECUTION"


def test_durable_intent_upserts_from_concurrent_executors_never_drop_intents(tmp_path):
    """Overlapping executors merging intents by id under the cross-process lock never lose each other's records."""
    intents_file = str(tmp_path / "autonomous_trade_intents_state.json")
    save_durable_intents_state(intents_file, [{"intent_id": "existing_1", "status": "EXECUTED"}])

    # Executor A took a stale snapshot before executor B appended its intent; A must merge, not overwrite.
    stale_snapshot = load_durable_intents_state(intents_file)
    assert upsert_durable_intent(intents_file, {"intent_id": "share_B", "status": "PENDING"})
    crypto_a = {"intent_id": "crypto_A", "status": "PENDING"}
    stale_snapshot.append(crypto_a)
    assert upsert_durable_intent(intents_file, crypto_a)
    crypto_a_done = dict(crypto_a, status="EXECUTED")
    assert upsert_durable_intent(intents_file, crypto_a_done)
    ids = [i["intent_id"] for i in load_durable_intents_state(intents_file)]
    assert ids == ["existing_1", "share_B", "crypto_A"]
    assert load_durable_intents_state(intents_file)[-1]["status"] == "EXECUTED"

    # The lock is reentrant within a thread (load/save nested inside upsert) and exclusive across threads.
    order: list[str] = []
    entered = threading.Event()

    def _holder():
        with durable_intents_lock(intents_file):
            with durable_intents_lock(intents_file):
                entered.set()
                time.sleep(0.3)
                order.append("holder_done")

    t = threading.Thread(target=_holder)
    t.start()
    entered.wait(2.0)
    upsert_durable_intent(intents_file, {"intent_id": "late_C", "status": "PENDING"})
    order.append("upsert_done")
    t.join(2.0)
    assert order == ["holder_done", "upsert_done"]
    assert [i["intent_id"] for i in load_durable_intents_state(intents_file)][-1] == "late_C"


def test_duplicate_open_ai_limit_order_is_skipped_while_ladder_at_distinct_price_executes(temp_workspace):
    """Restated AI LIMIT order at the same price & notional (replace_open_orders=False) is skipped, while a ladder order at a distinct price executes."""
    ws, _ = temp_workspace
    gw = AutonomousIntentGateway(workspace_dir=ws, cache_dir="cachedb", mode="enforce")

    hl_inst = MagicMock(symbol="HYPEUSDC", base="HYPE", quote="USDC", provider_name="hyperliquid")
    hl_inst._provider.name = "hyperliquid"
    hl_inst._provider.get_current_price.return_value = 86.50
    hl_inst._provider.free_balance.return_value = 25.0
    hl_inst.open_orders.return_value = [
        {
            "oid": 592714535208,
            "side": "A",
            "limitPx": "88.45",
            "sz": "3.95",
            "origSz": "3.95",
            "cloid": "0x2708f6bd78122f4b86a9ba9643d508d8",
        }
    ]
    hl_inst.place.return_value = {"orderId": "hl_new_1", "status": "NEW"}

    # Seed the existing intent so its venue order ID / cloid is recognized as AI
    gw.save_all_intents([
        {
            "intent_id": "intent_1793644299_sell_hypeusdc",
            "decision": "SELL",
            "symbol": "HYPEUSDC",
            "venue": "hyperliquid",
            "order_type": "LIMIT",
            "limit_price": 88.45,
            "suggested_notional_usd": 350.0,
            "status": "EXECUTED",
            "execution_payload": {"orderId": "592714535208", "status": "NEW", "price": "88.45", "origQty": "3.95"},
            "history": [],
        }
    ])

    with patch.object(gw, "_dispatch_notify", return_value=True):
        dup_rec = gw.submit({
            "decision": "SELL",
            "symbol": "HYPEUSDC",
            "venue": "hyperliquid",
            "order_type": "LIMIT",
            "limit_price": 88.45,
            "replace_open_orders": False,
            "confidence": 0.90,
            "suggested_notional_usd": 350.0,
            "thesis": "Existing profit target order; capture upside as price rebounds.",
        }, source="auto_cryptotrade_executor", max_mode="enforce", instruments_map={"HYPEUSDC": hl_inst})

        assert dup_rec["status"] == "SKIPPED_DUPLICATE_OPEN_ORDER"
        assert dup_rec["duplicate_of_order_id"] == "592714535208"
        hl_inst.place.assert_not_called()

        # When replace_open_orders=False, a restated AI LIMIT order at the exact same price (<= 1 bps)
        # but different notional (e.g. $200 vs $350 as free balance shrinks) is ALSO skipped rather than stacked
        dup_same_px_diff_notional = gw.submit({
            "decision": "SELL",
            "symbol": "HYPEUSDC",
            "venue": "hyperliquid",
            "order_type": "LIMIT",
            "limit_price": 88.45,
            "replace_open_orders": False,
            "confidence": 0.90,
            "suggested_notional_usd": 200.0,
            "thesis": "Fallback LLM restating same price level with smaller notional and replace_open_orders=False.",
        }, source="auto_cryptotrade_executor", max_mode="enforce", instruments_map={"HYPEUSDC": hl_inst})

        assert dup_same_px_diff_notional["status"] == "SKIPPED_DUPLICATE_OPEN_ORDER"
        assert dup_same_px_diff_notional["duplicate_of_order_id"] == "592714535208"
        hl_inst.place.assert_not_called()

        # Even when replace_open_orders=True is passed, an identical resting AI limit order is preserved
        # while any stale AI open order at a different price is still canceled
        ai_cloid = HyperliquidProvider.derive_client_order_id("AI_intent_1793644299_sell_hypeusdc")
        hl_inst.open_orders.return_value = [
            {
                "oid": 592714000111,
                "side": "A",
                "limitPx": "90.10",
                "sz": "3.00",
                "origSz": "3.00",
                "cloid": ai_cloid,
            },
            {
                "oid": 592714535208,
                "side": "A",
                "limitPx": "88.45",
                "sz": "3.95",
                "origSz": "3.95",
                "cloid": ai_cloid,
            },
        ]
        dup_replace_rec = gw.submit({
            "decision": "SELL",
            "symbol": "HYPEUSDC",
            "venue": "hyperliquid",
            "order_type": "LIMIT",
            "limit_price": 88.45,
            "replace_open_orders": True,
            "confidence": 0.90,
            "suggested_notional_usd": 350.0,
            "thesis": "Restated limit order with replace_open_orders=True.",
        }, source="auto_cryptotrade_executor", max_mode="enforce", instruments_map={"HYPEUSDC": hl_inst})

        assert dup_replace_rec["status"] == "SKIPPED_DUPLICATE_OPEN_ORDER"
        assert dup_replace_rec["duplicate_of_order_id"] == "592714535208"
        assert dup_replace_rec.get("canceled_prior_open_orders") == ["592714000111"]
        hl_inst.cancel_order.assert_called_once_with("592714000111")
        hl_inst.place.assert_not_called()

        # Multi-tranche scale-out ladder at a distinct price (e.g. $92.00 vs $88.45) executes normally
        ladder_rec = gw.submit({
            "decision": "SELL",
            "symbol": "HYPEUSDC",
            "venue": "hyperliquid",
            "order_type": "LIMIT",
            "limit_price": 92.00,
            "replace_open_orders": False,
            "confidence": 0.90,
            "suggested_notional_usd": 350.0,
            "thesis": "Second scale-out tranche at $92.00 resistance.",
        }, source="auto_cryptotrade_executor", max_mode="enforce", instruments_map={"HYPEUSDC": hl_inst})

        assert ladder_rec["status"] == "EXECUTED"
        hl_inst.place.assert_called_once()

    # Lot-size rounded small order ($10.00 requested -> 0.11 * $84.00 = $9.24 resting) is also recognized as duplicate
    hl_inst.open_orders.return_value = [
        {
            "oid": 592714999999,
            "side": "B",
            "limitPx": "84.00",
            "sz": "0.11",
            "origSz": "0.11",
            "cloid": "0x2708f6bd78122f4b86a9ba9643d508d8",
        }
    ]
    assert find_duplicate_ai_open_limit_order(
        hl_inst, "BUY", 84.00, 10.00, known_ai_order_ids={"592714999999"}
    ) == "592714999999"

    # Superseded canceled intent on quote-compatible alias (HYPEUSDC vs HYPEUSD) is deduplicated in summarize_recent_ai_intents
    alias_intents = [
        {
            "intent_id": "krk_old_canceled",
            "timestamp": 1793660000.0,
            "decision": "BUY",
            "symbol": "HYPEUSDC",
            "venue": "kraken",
            "order_type": "LIMIT",
            "limit_price": 80.0,
            "suggested_notional_usd": 100.0,
            "status": "EXECUTED",
            "execution_payload": {"ordertxid": "KRK_OLD_CANC", "status": "CANCELED", "vol_exec": "0.0"},
        },
        {
            "intent_id": "krk_new_resting",
            "timestamp": 1793660300.0,
            "decision": "BUY",
            "symbol": "HYPEUSD",
            "venue": "kraken",
            "order_type": "LIMIT",
            "limit_price": 81.0,
            "suggested_notional_usd": 100.0,
            "status": "EXECUTED",
            "execution_payload": {"ordertxid": "KRK_NEW_OPEN", "status": "OPEN", "vol_exec": "0.0"},
        },
    ]
    sum_alias = summarize_recent_ai_intents(
        alias_intents, "crypto", 1793660400.0, extract_intent_notional,
        open_orders=[{"venue": "kraken", "symbol": "HYPEUSD", "side": "BUY", "order_id": "KRK_NEW_OPEN", "price": 81.0}],
    )
    assert [x.get("order_id") for x in sum_alias] == ["KRK_NEW_OPEN"]



def test_reconcile_canceled_prior_open_orders_and_carried_resting_multi_tranche(tmp_path):

    now_ts = 1793660000.0
    intents = [
        {
            "intent_id": "intent_old_replaced",
            "timestamp": now_ts - 3600.0,
            "decision": "BUY",
            "symbol": "HYPEUSDC",
            "venue": "hyperliquid",
            "order_type": "LIMIT",
            "limit_price": 41.0,
            "reference_price": 41.2,
            "suggested_notional_usd": 120.0,
            "status": "EXECUTED",
            "execution_payload": {"orderId": "HL_OLD_1", "status": "NEW", "msg": "resting", "executedQty": "0.0"},
        },
        {
            "intent_id": "intent_new_replacer",
            "timestamp": now_ts - 1800.0,
            "decision": "BUY",
            "symbol": "HYPEUSDC",
            "venue": "hyperliquid",
            "order_type": "LIMIT",
            "limit_price": 41.5,
            "reference_price": 41.6,
            "suggested_notional_usd": 120.0,
            "status": "EXECUTED",
            "canceled_prior_open_orders": ["HL_OLD_1"],
            "execution_payload": {"orderId": "HL_OPEN_2", "status": "NEW", "price": "41.5", "executedQty": "0.0"},
        },
        {
            "intent_id": "intent_blocked_latest",
            "timestamp": now_ts - 600.0,
            "decision": "BUY",
            "symbol": "HYPEUSDC",
            "venue": "hyperliquid",
            "order_type": "LIMIT",
            "limit_price": 41.6,
            "reference_price": 41.7,
            "suggested_notional_usd": 120.0,
            "status": "BLOCKED_BY_GUARD",
            "execution_payload": {"guard": "G_QLIMIT1"},
        },
        {
            "intent_id": "intent_arb_tranche_1",
            "timestamp": now_ts - 1200.0,
            "decision": "SELL",
            "symbol": "ARBUSDC",
            "venue": "binance",
            "order_type": "LIMIT",
            "limit_price": 0.1628,
            "reference_price": 0.1610,
            "suggested_notional_usd": 120.0,
            "status": "EXECUTED",
            "execution_payload": {"orderId": "562350381", "status": "NEW", "price": "0.1628", "executedQty": "0.0"},
        },
        {
            "intent_id": "intent_arb_tranche_2",
            "timestamp": now_ts - 10.0,
            "decision": "SELL",
            "symbol": "ARBUSDC",
            "venue": "binance",
            "order_type": "LIMIT",
            "limit_price": 0.1598,
            "reference_price": 0.1585,
            "suggested_notional_usd": 85.0,
            "status": "EXECUTED",
            "execution_payload": {"orderId": "562352955", "status": "NEW", "price": "0.1598", "executedQty": "0.0"},
        },
    ]

    changed = reconcile_intents_from_execution_audit(intents, str(tmp_path))
    assert changed is True
    assert intents[0]["execution_payload"]["status"] == "CANCELED"
    assert intents[0]["execution_payload"]["reconciled_terminal"] is True
    assert "msg" not in intents[0]["execution_payload"]

    open_orders = [
        {"venue": "hyperliquid", "symbol": "HYPEUSDC", "side": "BUY", "order_id": "HL_OPEN_2", "price": 41.5, "by": "AI"},
        {"venue": "binance", "symbol": "ARBUSDC", "side": "SELL", "order_id": "562350381", "price": 0.1628},
        {"venue": "binance", "symbol": "ARBUSDC", "side": "SELL", "order_id": "562352955", "price": 0.1598},
    ]
    last_st = build_last_cycle_execution_status(
        intents, "crypto", now_ts, extract_intent_notional, recent_orders={}, open_orders=open_orders
    )
    assert isinstance(last_st, list)
    assert [x.get("order_id") for x in last_st] == ["HL_OPEN_2", "562350381", "562352955"]
    assert all(x["status"] == "OPEN_RESTING" for x in last_st)

    # Partial fill in audit does not prematurely lock reconciled_terminal=True before final FILLED audit arrives,
    # and a non-fill cancel audit does not overwrite a fill audit on the same intent.
    audit_file = tmp_path / "execution_audit_2026-11-03.jsonl"
    audit_file.write_text(
        json.dumps({
            "ts": now_ts - 5.0,
            "event": "order_terminal",
            "intent_id": "intent_arb_tranche_1",
            "order_id": "562350381",
            "venue_status": "PARTIALLY_FILLED",
            "status": "partially_filled",
            "filled_qty": "250.0",
            "price": "0.1628",
            "cost": "40.70",
        }) + "\n",
        encoding="utf-8",
    )
    assert reconcile_intents_from_execution_audit(intents, str(tmp_path)) is True
    assert intents[3]["execution_payload"]["status"] == "PARTIALLY_FILLED"
    assert intents[3]["execution_payload"]["executedQty"] == "250.0"
    assert "reconciled_terminal" not in intents[3]["execution_payload"]
    # Second pass without new audit events does not report a spurious state change
    assert reconcile_intents_from_execution_audit(intents, str(tmp_path)) is False

    # Subsequent FILLED event (plus a trailing non-fill cancel event) transitions intent to terminal FILLED
    audit_file.write_text(
        audit_file.read_text(encoding="utf-8")
        + json.dumps({
            "ts": now_ts - 2.0,
            "event": "order_terminal",
            "intent_id": "intent_arb_tranche_1",
            "order_id": "562350381",
            "venue_status": "FILLED",
            "status": "closed",
            "filled_qty": "737.1",
            "price": "0.1628",
            "cost": "120.0",
        }) + "\n"
        + json.dumps({
            "ts": now_ts - 1.0,
            "event": "order_terminal_retry",
            "intent_id": "intent_arb_tranche_1",
            "order_id": "562350381",
            "venue_status": "CANCELED",
            "status": "canceled",
            "filled_qty": "0.0",
        }) + "\n",
        encoding="utf-8",
    )
    assert reconcile_intents_from_execution_audit(intents, str(tmp_path)) is True
    assert intents[3]["execution_payload"]["status"] == "FILLED"
    assert intents[3]["execution_payload"]["executedQty"] == "737.1"
    assert intents[3]["execution_payload"]["reconciled_terminal"] is True


def test_enforce_intent_order_cleans_up_refused_retry_record_when_second_attempt_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_a, **_kw: None)
    q_file = tmp_path / "order_retry_queue.jsonl"
    l_file = tmp_path / "order_retry_queue.lock"
    monkeypatch.setattr(order_retry, "QUEUE_FILE", str(q_file))
    monkeypatch.setattr(order_retry, "LOCK_FILE", str(l_file))

    q_file.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {"id": "refused_attempt0_rec", "intent_id": "intent_1793650000_buy_btcusdc", "submission_state": "refused"},
                {"id": "other_rec", "intent_id": "intent_other", "submission_state": "refused"},
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    calls = []

    def _dummy_place(**kwargs):
        calls.append(kwargs)
        ctx = kwargs.get("_outcome_context")
        if len(calls) == 1:
            if isinstance(ctx, dict):
                ctx.update({"state": "refused", "reason": "account_cache_not_fresh"})
            return None
        if isinstance(ctx, dict):
            ctx.update({"state": "accepted", "accepted": True})
        return {"orderId": "99887766", "status": "FILLED"}

    intent = {
        "intent_id": "intent_1793650000_buy_btcusdc",
        "symbol": "BTCUSDC",
        "decision": "BUY",
        "status": "PENDING",
    }
    execute_enforce_intent_order(
        SimpleNamespace(symbol="BTCUSDC", provider_name="binance", place=_dummy_place),
        intent,
        intent_id="intent_1793650000_buy_btcusdc",
        decision="BUY",
        target_qty=0.01,
        price=70000.0,
        effective_usd=700.0,
        guard_allowed=True,
        guard_reason="",
        thesis="Retest support",
        trade_tag="[AI]",
        now_ts=1793650000.0,
        bypass_guards=set(),
        is_limit=False,
        dispatch_notify_fn=None,
        log=None,
    )

    assert len(calls) == 2
    assert intent["status"] == "EXECUTED"
    remaining = order_retry.load_all()
    assert [r["id"] for r in remaining] == ["other_rec"]


def test_harvest_bot_runtime_states_excludes_executed_and_rejected_ai_intents(tmp_path):

    now_ts = time.time()
    intents_file = tmp_path / "autonomous_trade_intents_state.json"
    intents_file.write_text(
        json.dumps([
            {"intent_id": "intent_done_1", "status": "EXECUTED"},
            {"intent_id": "intent_rej_2", "status": "REJECTED_LOW_CONFIDENCE"},
            {"intent_id": "intent_dup_3", "status": "SKIPPED_DUPLICATE_OPEN_ORDER"},
            {"intent_id": "intent_pending_4", "status": "PENDING"},
        ]),
        encoding="utf-8",
    )
    retry_q = tmp_path / "order_retry_queue.jsonl"
    retry_q.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {
                    "id": "r1",
                    "intent_id": "intent_done_1",
                    "symbol": "BTCUSDC",
                    "side": "BUY",
                    "qty": 0.01,
                    "lifecycle": "submit_pending",
                    "created_ts": now_ts - 10,
                },
                {
                    "id": "r2",
                    "intent_id": "intent_rej_2",
                    "symbol": "ETHUSDC",
                    "side": "BUY",
                    "qty": 0.1,
                    "lifecycle": "submit_pending",
                    "created_ts": now_ts - 10,
                },
                {
                    "id": "r3",
                    "intent_id": "intent_dup_3",
                    "symbol": "SOLUSDC",
                    "side": "SELL",
                    "qty": 1.0,
                    "lifecycle": "submit_pending",
                    "created_ts": now_ts - 10,
                },
                {
                    "id": "r4",
                    "intent_id": "intent_pending_4",
                    "symbol": "TAOUSDC",
                    "side": "BUY",
                    "qty": 0.5,
                    "lifecycle": "submit_pending",
                    "created_ts": now_ts - 10,
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    res = harvest_bot_runtime_states(str(tmp_path), now_ts=now_ts)
    active_q = res.get("active_retry_queue") or []
    assert [x.get("symbol") for x in active_q] == ["TAOUSDC"]


def _make_tao_inst_with_orders(*, free_fn, price: float, open_orders_fn, cancel_fn=None, params=None):
    prov = SimpleNamespace(
        free_balance=free_fn,
        get_current_price=lambda _symbol: price,
    )
    return SimpleNamespace(
        symbol="TAOUSDC",
        base="TAO",
        quote="USDC",
        provider_name="binance",
        params=params or {},
        _provider=prov,
        open_orders=open_orders_fn,
        cancel_order=cancel_fn or (lambda _oid: False),
    )


def test_process_intent_replace_open_orders_unlocks_bot_order_when_free_remainder_clamped(temp_workspace, monkeypatch):
    ws, _ = temp_workspace
    gw = AutonomousIntentGateway(workspace_dir=ws, cache_dir="cachedb", mode="enforce")

    locked_state = {"locked_shares": 1.0}

    def _cancel_order(oid):
        if str(oid) == "bot_tp_1":
            locked_state["locked_shares"] = 0.0
            return True
        return False

    inst = _make_tao_inst_with_orders(
        free_fn=lambda asset: (1.24 - locked_state["locked_shares"]) if asset == "TAO" else 500.0,
        price=100.0,
        open_orders_fn=lambda: (
            [{"orderId": "bot_tp_1", "symbol": "TAOUSDC", "side": "SELL", "price": 108.0, "origQty": 1.0, "executedQty": 0.0}]
            if locked_state["locked_shares"] > 0
            else []
        ),
        cancel_fn=_cancel_order,
    )
    monkeypatch.setattr(gw, "resolve_instrument", lambda *_a, **_kw: inst)

    executed_calls = []

    def _fake_exec(inst_arg, intent_arg, **kwargs):
        executed_calls.append(kwargs)
        intent_arg["status"] = "EXECUTED"
        return intent_arg

    monkeypatch.setattr("orchestratorTrade.autonomous_intent_gateway.execute_enforce_intent_order", _fake_exec)

    intent = {
        "intent_id": "intent_replace_bot_tp",
        "symbol": "TAOUSDC",
        "venue": "binance",
        "decision": "SELL",
        "confidence": 0.90,
        "reference_price": 100.0,
        "suggested_notional_usd": 124.0,
        "replace_open_orders": True,
        "status": "PENDING",
    }
    out = gw.process_intent(intent, instruments_map={"BINANCE_TAO": inst})
    assert out["status"] == "EXECUTED"
    assert out.get("canceled_bot_open_orders") == ["bot_tp_1"]
    assert len(executed_calls) == 1
    assert executed_calls[0]["effective_usd"] == pytest.approx(124.0)


def test_harvest_tradable_positions_preserves_trailing_peak_for_100pct_locked_sell_and_updates_bcache(tmp_path):
    bcache_file = tmp_path / "venue_balances_state.json"
    bcache_file.write_text("{}", encoding="utf-8")

    inst = _make_tao_inst_with_orders(
        free_fn=lambda asset: 0.0 if str(asset).upper() == "TAO" else 50.0,
        price=400.0,
        params={"trailing.pct": 1.5},
        open_orders_fn=lambda: [
            {
                "orderId": "sell_lock_1",
                "symbol": "TAOUSDC",
                "side": "SELL",
                "price": 420.0,
                "origQty": 1.5,
                "executedQty": 0.0,
                "status": "NEW",
            }
        ],
    )

    prices_and_trends = {"TAOUSDC": {"price": 400.0, "trend": 1, "growth_coefficient": 0.1}}
    trailing_by_venue = {"binance": {"TAOUSDC": {"peak": 440.0}}}
    positions, open_orders, cash_map, _ = harvest_tradable_positions_and_orders(
        {"BINANCE_TAO": inst},
        prices_and_trends,
        {},
        trailing_by_venue,
        known_ai_order_ids=set(),
        gateway_max_usd=250.0,
        cache_dir=str(tmp_path),
    )
    assert len(positions) == 1
    pos = positions[0]
    assert pos["free_base"] == 0.0
    assert pos["locked_base_qty"] == pytest.approx(1.5)
    assert pos["trailing_peak"] == pytest.approx(440.0)
    assert pos["dd_from_peak_pct"] == pytest.approx(-9.09)

    from state_stores.portfolio_positions_state import PortfolioPositionsStateStore

    saved_positions = PortfolioPositionsStateStore.load_dict(str(tmp_path))
    cached_pos = saved_positions["pos:binance:TAOUSDC"]
    assert cached_pos["locked_base_qty"] == pytest.approx(1.5)
    assert cached_pos["trailing_peak"] == pytest.approx(440.0)

    # Verify harvest_bot_runtime_states does NOT classify a 100%-locked SELL position (value_usd=0, locked_base_usd=600) as dust
    ag_path = tmp_path / "assetguardian_state.json"
    ag_path.write_text(
        json.dumps({"symbols": {"TAOUSDC": {"sell": {"pending": False, "reason": "guard_watch"}}}}),
        encoding="utf-8",
    )
    runtime_states = harvest_bot_runtime_states(
        str(tmp_path),
        lambda p: json.loads(open(p, encoding="utf-8").read()) if os.path.exists(p) else None,
        tradable_positions=positions,
    )
    assert "sell" in runtime_states["assetguardian_state"]["TAOUSDC"]

    # Verify allow_empty_inst_recovery preserves step_size/exchange_min_notional_usd when open orders remain,
    # and strips stale locked_* / ghost trailing_peak when open_orders in cached_bals becomes empty
    cached_pos["step_size"] = 0.001
    cached_pos["exchange_min_notional_usd"] = 5.0
    PortfolioPositionsStateStore.upsert_locked(
        PortfolioPositionsStateStore.resolve_path(str(tmp_path)), saved_positions
    )

    rec_positions_with_oo, rec_oo, _, _ = harvest_tradable_positions_and_orders(
        {},
        {"TAOUSDC": {"price": 410.0, "trend": 1, "growth_coefficient": 0.1}},
        {},
        {},
        known_ai_order_ids=set(),
        gateway_max_usd=250.0,
        cache_dir=str(tmp_path),
        allow_empty_inst_recovery=True,
    )
    assert len(rec_positions_with_oo) == 1
    assert rec_positions_with_oo[0]["locked_base_qty"] == pytest.approx(1.5)
    assert rec_positions_with_oo[0]["step_size"] == pytest.approx(0.001)
    assert rec_positions_with_oo[0]["trailing_peak"] == pytest.approx(440.0)

    # Now clear open_orders in cache_order_open_binance.json and recover again -> stale locked_* and ghost trailing_peak (since free_base=0) are stripped
    open_file = tmp_path / "cache_order_open_binance.json"
    open_file.write_text(json.dumps({"items": {"TAOUSDC": []}, "fetchtime": {"TAOUSDC": 1700000000000}, "verified_ts": 1700000000.0}), encoding="utf-8")
    rec_positions_no_oo, _, _, _ = harvest_tradable_positions_and_orders(
        {},
        {"TAOUSDC": {"price": 410.0, "trend": 1, "growth_coefficient": 0.1}},
        {},
        {},
        known_ai_order_ids=set(),
        gateway_max_usd=250.0,
        cache_dir=str(tmp_path),
        allow_empty_inst_recovery=True,
    )
    assert len(rec_positions_no_oo) == 1
    assert "locked_base_qty" not in rec_positions_no_oo[0]
    assert "locked_base_usd" not in rec_positions_no_oo[0]
    assert "replace_max_sell_usd" not in rec_positions_no_oo[0]
    assert "trailing_peak" not in rec_positions_no_oo[0]

# ============================================================================
# Domain Regressions & Input Hygiene (Targeted Validation)
# ============================================================================


def test_autonomousaireconciler_synthesize_and_diagnose_handles_none_bool_nan_in_prices_and_trends_iter56_s4(tmp_path):
    """[ITER56 #4] AutonomousAIReconciler.synthesize_and_diagnose handles None/bool/NaN in prices_and_trends, microstructure, and portfolio_summary."""
    # 4. AutonomousAIReconciler.synthesize_and_diagnose handles None/bool/NaN in prices_and_trends, microstructure, and portfolio_summary
    (tmp_path / "order_guard.conf").write_text("supervisor_mode = shadow\nsupervisor_notify = 0\n", encoding="utf-8")
    (tmp_path / "config.env").write_text("NTFY_TOPIC_GUARD=test-guard\n", encoding="utf-8")
    mock_gemini = MagicMock()
    mock_gemini.query_json.return_value = {"summary": "OK", "market_trading_decisions": []}
    rec = AutonomousAIReconciler(workspace_dir=str(tmp_path), gemini_auditor=mock_gemini)
    snap = TelemetrySensorSnapshot(
        prices_and_trends={
            "BTCUSDC": {
                "price": float("nan"),
                "trend": None,
                "growth_coefficient": None,
                "chg_15m_pct": None,
                "chg_30m_pct": float("nan"),
                "chg_1h_pct": True,
                "chg_3h_pct": None,
                "low_24h": None,
                "high_24h": float("nan"),
                "chg_7d_pct": None,
                "slope_15m": None,
                "slope_3_7m": float("nan"),
                "slope_2_5h": True,
                "trend_3_7m": None,
                "trend_15m": float("nan"),
                "trend_2_5h": True,
                "gradient_recent_fast": True,
                "epsilon": float("nan"),
                "kalman_vel": True,
            }
        },
        microstructure={
            "BTCUSDC": {
                "bid_wall_px": float("nan"),
                "ask_wall_px": True,
            }
        },
        portfolio_summary={
            "total_value_usdc": float("nan"),
            "venue_quote_cash": {"binance:USDC": {"venue": "binance", "asset": "USDC", "free": float("nan")}},
            "recent_orders": {"BTCUSDC": {"last_buy": {"ts_ms": "nan"}, "last_sell": None}},
        },
    )
    logs = LogEventSummary(total_scanned_files=0, error_count=0, warning_count=0)
    res = rec.synthesize_and_diagnose(snap, logs, topic_key="auto_cryptotrade_executor")
    assert res["summary"] == "OK"

def test_autonomousintentgateway__get_live_price_rejects_boolean_true_prices_from_provider_iter65_s3(tmp_path):
    """[ITER65 #3] AutonomousIntentGateway._get_live_price rejects boolean True prices from provider and cache."""
    # 3. AutonomousIntentGateway._get_live_price rejects boolean True prices from provider and cache
    ws = tmp_path / "gw_ws"
    ws.mkdir()
    cache = ws / "cachedb"
    cache.mkdir()
    (ws / "order_guard.conf").write_text(
        "autonomous_trading_mode = shadow\n"
        "autonomous_trading_max_notional_usd = 2000.0\n"
        "autonomous_trading_min_notional_usd = 10.0\n"
        "autonomous_trading_min_confidence = 0.85\n"
        "autonomous_trading_notify = 0\n",
        encoding="utf-8",
    )
    (cache / "cache_current_price.json").write_text(
        json.dumps({"items": {"BTCUSDC": [[1000, True]]}}),
        encoding="utf-8",
    )
    gw = AutonomousIntentGateway(workspace_dir=str(ws), cache_dir="cachedb")
    inst = SimpleNamespace(
        symbol="BTCUSDC",
        base="BTC",
        _provider=SimpleNamespace(get_current_price=lambda _s: True),
    )
    assert math.isnan(gw._get_live_price(inst))

def test_autonomousintentgateway_resolve_instrument_rejects_empty_boolean_symbol_venue_and_iter120_s5(monkeypatch, tmp_path):
    """[ITER120 #5] AutonomousIntentGateway.resolve_instrument rejects empty/boolean symbol/venue and process_intent normalizes mode."""
    # 5. AutonomousIntentGateway.resolve_instrument rejects empty/boolean symbol/venue and process_intent normalizes mode
    gw = AutonomousIntentGateway(
        workspace_dir=str(tmp_path),
        mode="shadow",
        max_notional_usd=100.0,
        min_confidence=0.80,
        notify_enabled=False,
    )
    fake_inst = SimpleNamespace(symbol="BTCUSDC", provider_name="binance", base="BTC", quote="USDC", enabled=True)
    inst_map = {"BINANCE_BTC": fake_inst}
    assert gw.resolve_instrument("", venue="binance", instruments_map=inst_map) is None
    assert gw.resolve_instrument(True, venue="binance", instruments_map=inst_map) is None
    assert gw.resolve_instrument("BTCUSDC", venue=True, instruments_map=inst_map) is None
    assert gw.resolve_instrument("BTCUSDC", venue="binance", instruments_map=inst_map) is fake_inst

    monkeypatch.setattr(gw, "_get_live_price", lambda _i: 50000.0)
    out = gw.process_intent(
        {
            "intent_id": True,
            "decision": "BUY",
            "symbol": "BTCUSDC",
            "venue": None,
            "confidence": 0.90,
            "suggested_notional_usd": 50.0,
        },
        instruments_map=inst_map,
        mode=" SHADOW ",
    )
    assert out["status"] == "OBSERVED_SHADOW"


def test_execution_contract_persistence_shadow_cap_client_id_and_strict_identity_notional(temp_workspace, monkeypatch):
    """Verify gateway execution contract: fail-closed initial save, persisted shadow ceiling on resume,
    client_order_id reconciliation before resubmit, explicit positive notional, and zero prefix collisions."""
    ws, _ = temp_workspace
    gw = AutonomousIntentGateway(workspace_dir=ws, cache_dir="cachedb", mode="enforce", notify_enabled=False)

    ethfi_inst = _make_mock_inst("ETHFIUSDC", base="ETHFI", quote="USDC", venue="binance", price=2.50, free_balance=1000.0)
    btc_inst = _make_mock_inst(
        "BTCUSDC",
        base="BTC",
        quote="USDC",
        venue="binance",
        price=80000.0,
        free_balance=1000.0,
        place_return={"orderId": "ord_btc_1", "status": "FILLED"},
    )
    inst_map = {"BINANCE_ETHFI": ethfi_inst, "BINANCE_BTC": btc_inst}

    # 1. Prefix collision blocked: 'ETH' never resolves to 'ETHFIUSDC', and no executable intent is created
    assert gw.resolve_instrument("ETH", venue="binance", instruments_map=inst_map) is None
    assert (
        gw.submit(
            {"decision": "BUY", "symbol": "ETH", "venue": "binance", "confidence": 0.95, "suggested_notional_usd": 100.0},
            source="auto_cryptotrade_executor",
            max_mode="enforce",
            instruments_map=inst_map,
        )
        is None
    )
    ethfi_inst.place.assert_not_called()
    assert gw.load_all_intents() == []

    # Ambiguous multi-candidate base match is also rejected rather than picking first
    btc_dup = _make_mock_inst("BTCUSDT", base="BTC", quote="USDT", venue="binance", price=80000.0, free_balance=1000.0)
    assert gw.resolve_instrument("BTC", venue="binance", instruments_map={"B1": btc_inst, "B2": btc_dup}) is None

    # 2. Missing, invalid, zero, or negative notional USD is rejected (never defaults to $2,000 or $10)
    for bad_dec in (
        {"decision": "BUY", "symbol": "BTCUSDC", "venue": "binance", "confidence": 0.95},
        {"decision": "BUY", "symbol": "BTCUSDC", "venue": "binance", "confidence": 0.95, "suggested_notional_usd": "invalid"},
        {"decision": "BUY", "symbol": "BTCUSDC", "venue": "binance", "confidence": 0.95, "suggested_notional_usd": 0},
        {"decision": "BUY", "symbol": "BTCUSDC", "venue": "binance", "confidence": 0.95, "suggested_notional_usd": -50.0},
    ):
        assert gw.submit(bad_dec, source="auto_cryptotrade_executor", max_mode="enforce", instruments_map=inst_map) is None
    assert gw.load_all_intents() == []

    bad_preflight = gw.process_intent(
        {"intent_id": "missing_notional_1", "decision": "BUY", "symbol": "BTCUSDC", "venue": "binance", "confidence": 0.95},
        instruments_map=inst_map,
    )
    assert bad_preflight["status"] == "REJECTED_INVALID_NOTIONAL"
    btc_inst.place.assert_not_called()

    # 3. Initial save returning False aborts before process_intent / venue submission
    with patch("orchestratorTrade.autonomous_intent_gateway.upsert_durable_intent", return_value=False):
        aborted = gw.submit(
            {"decision": "BUY", "symbol": "BTC", "venue": "binance", "confidence": 0.95, "suggested_notional_usd": 150.0},
            source="auto_cryptotrade_executor",
            max_mode="enforce",
            instruments_map=inst_map,
        )
    assert aborted is None
    btc_inst.place.assert_not_called()

    # 4. Interruption between initial save and processing preserves max_mode='shadow' when resumed in an enforce gateway
    def _boom(*_a, **_kw):
        raise RuntimeError("simulated interruption before execution")

    with patch.object(gw, "process_intent", side_effect=_boom):
        with pytest.raises(RuntimeError, match="simulated interruption"):
            gw.submit(
                {"decision": "BUY", "symbol": "BTC", "venue": "binance", "confidence": 0.95, "suggested_notional_usd": 150.0},
                source="auto_cryptotrade_executor",
                max_mode="shadow",
                instruments_map=inst_map,
            )

    queued = gw.load_all_intents()
    assert len(queued) == 1
    assert queued[0]["symbol"] == "BTCUSDC"
    assert queued[0]["max_mode"] == "shadow"
    assert queued[0]["status"] == "PENDING_EXECUTION"

    gw_enforce_resume = AutonomousIntentGateway(workspace_dir=ws, cache_dir="cachedb", mode="enforce", notify_enabled=False)
    monkeypatch.setattr(gw_enforce_resume, "load_tradable_instruments", lambda **_kw: inst_map)
    resumed_shadow = gw_enforce_resume.process_all_pending()
    assert len(resumed_shadow) == 1
    assert resumed_shadow[0]["status"] == "OBSERVED_SHADOW"
    btc_inst.place.assert_not_called()

    # 5. Resuming a claimed enforce intent reconciles an open order carrying AI_<intent_id> without resubmitting
    claimed_id = "intent_1794000001_buy_btcusdc"
    gw.save_all_intents([
        {
            "intent_id": claimed_id,
            "timestamp": 1794000001.0,
            "decision": "BUY",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "confidence": 0.95,
            "suggested_notional_usd": 160.0,
            "order_type": "LIMIT",
            "limit_price": 79000.0,
            "max_mode": "enforce",
            "status": "PENDING_EXECUTION",
            "history": [],
        }
    ])
    btc_inst.open_orders.return_value = [
        {
            "orderId": "99887766",
            "clientOrderId": f"AI_{claimed_id}",
            "side": "BUY",
            "price": "79000.0",
            "origQty": "0.00202",
            "status": "NEW",
        }
    ]
    resumed_enforce = gw_enforce_resume.process_all_pending()
    assert len(resumed_enforce) == 1
    assert resumed_enforce[0]["status"] == "EXECUTED"
    assert resumed_enforce[0]["execution_payload"]["orderId"] == "99887766"
    assert resumed_enforce[0]["history"][-1]["action"] == "RECONCILED_CLIENT_ORDER_ID"
    btc_inst.place.assert_not_called()


def test_audit_reconciliation_explicit_accepted_unknown_partial_terminal_and_pending_recovery(temp_workspace):
    import json
    from providers.intent_reconciliation import _resolve_intent_execution_details, reconcile_intents_from_execution_audit
    from providers.intent_state import extract_intent_notional

    ws, _ = temp_workspace
    gw = AutonomousIntentGateway(workspace_dir=ws, cache_dir="cachedb", mode="enforce")
    audit_dir = os.path.join(ws, "logger", "execution_audit")
    os.makedirs(audit_dir, exist_ok=True)
    audit_file = os.path.join(audit_dir, "execution_audit_2026-11-12.jsonl")

    # 1. submit_accepted on MARKET order recovers both FAILED and PENDING as accepted (NEW, empty executedQty, no terminal_ts)
    with open(audit_file, "w", encoding="utf-8") as f:
        for iid, oid in (("mkt_failed", "ord_f1"), ("mkt_pending", "ord_p1")):
            f.write(json.dumps({
                "event": "submit_accepted",
                "intent_id": iid,
                "order_id": oid,
                "symbol": "BTCUSDC",
                "venue": "binance",
                "side": "buy",
                "qty": 0.01,
                "price": 80000.0,
                "market": True,
                "ts": 1794500001.0,
            }) + "\n")
        f.write(json.dumps({
            "event": "submit_unknown",
            "intent_id": "mkt_unknown",
            "symbol": "ETHUSDC",
            "venue": "binance",
            "side": "buy",
            "qty": 0.5,
            "client_order_id": "AI_mkt_unknown",
            "failure_reason": "timeout",
            "ts": 1794500002.0,
        }) + "\n")

    intents = [
        {"intent_id": "mkt_failed", "symbol": "BTCUSDC", "venue": "binance", "decision": "BUY", "order_type": "MARKET", "status": "FAILED"},
        {"intent_id": "mkt_pending", "symbol": "BTCUSDC", "venue": "binance", "decision": "BUY", "order_type": "MARKET", "status": "PENDING"},
        {"intent_id": "mkt_unknown", "symbol": "ETHUSDC", "venue": "binance", "decision": "BUY", "order_type": "MARKET", "status": "PENDING"},
    ]
    assert reconcile_intents_from_execution_audit(intents, audit_dir) is True
    # Idempotence when no new audit records arrive
    assert reconcile_intents_from_execution_audit(intents, audit_dir) is False

    for rec_item in (intents[0], intents[1]):
        assert rec_item["status"] == "EXECUTED"
        assert rec_item["history"][-1]["action"] == "RECONCILED_ACCEPTED"
        ep = rec_item["execution_payload"]
        assert ep["status"] == "NEW"
        assert ep["origQty"] == "0.01"
        assert ep["executedQty"] == ""
        assert ep["cummulativeQuoteQty"] == ""
        assert ep["accepted_at"] == pytest.approx(1794500001.0)
        assert "filled_at" not in ep
        assert "terminal_ts" not in ep
        assert "reconciled_terminal" not in ep
        det = _resolve_intent_execution_details(rec_item, 1794500010.0, extract_intent_notional)
        assert det["exec_status"] == "ACCEPTED"
        assert det["filled_qty"] is None
        assert det["fill_px"] is None

    # Unknown submission prevents PENDING re-execution and models UNKNOWN explicitly
    assert intents[2]["status"] == "SUBMIT_UNKNOWN"
    assert intents[2]["history"][-1]["action"] == "RECONCILED_UNKNOWN"
    assert intents[2]["execution_payload"]["status"] == "UNKNOWN"
    assert intents[2]["execution_payload"]["submission_state"] == "unknown"
    intents[2]["execution_payload"]["origQty"] = "7753.6"
    det_unk = _resolve_intent_execution_details(intents[2], 1794500010.0, extract_intent_notional)
    assert det_unk["exec_status"] == "SUBMIT_UNKNOWN"
    assert det_unk["filled_qty"] is None

    gw.save_all_intents(intents)
    mock_btc = _make_mock_inst("BTCUSDC", base="BTC", quote="USDC", price=80000.0, free_balance=5000.0)
    with patch.object(gw, "load_tradable_instruments", return_value={"BTCUSDC": mock_btc}):
        assert gw.process_all_pending() == []
    mock_btc.place.assert_not_called()

    # 2. Partial fill (order_status) advances mkt_failed without setting terminal_ts; mkt_unknown resolves via order_recovered + terminal EXPIRED
    with open(audit_file, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "event": "order_status",
            "intent_id": "mkt_failed",
            "order_id": "ord_f1",
            "symbol": "BTCUSDC",
            "venue_status": "PARTIALLY_FILLED",
            "status": "partially_filled",
            "filled_qty": 0.004,
            "cost": 320.0,
            "ts": 1794500004.0,
        }) + "\n")
        f.write(json.dumps({
            "event": "order_recovered",
            "intent_id": "mkt_unknown",
            "order_id": "ord_u1",
            "symbol": "ETHUSDC",
            "ts": 1794500005.0,
        }) + "\n")
        f.write(json.dumps({
            "event": "order_terminal",
            "intent_id": "mkt_unknown",
            "order_id": "ord_u1",
            "symbol": "ETHUSDC",
            "venue_status": "EXPIRED",
            "status": "expired",
            "filled_qty": 0.0,
            "ts": 1794500006.0,
        }) + "\n")

    assert reconcile_intents_from_execution_audit(intents, audit_dir) is True
    ep_part = intents[0]["execution_payload"]
    assert ep_part["status"] == "PARTIALLY_FILLED"
    assert ep_part["executedQty"] == "0.004"
    assert ep_part["cummulativeQuoteQty"] == "320.0"
    assert "terminal_ts" not in ep_part
    assert "reconciled_terminal" not in ep_part

    assert intents[2]["status"] == "FAILED"
    assert intents[2]["history"][-1]["action"] == "RECONCILED_TERMINAL"
    assert intents[2]["execution_payload"]["status"] == "EXPIRED"
    assert intents[2]["execution_payload"]["terminal_ts"] == pytest.approx(1794500006.0)
    assert intents[2]["execution_payload"]["reconciled_terminal"] is True

    # 3. Terminal FILLED completes mkt_failed with terminal_ts and reconciled_terminal=True
    with open(audit_file, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "event": "order_terminal",
            "intent_id": "mkt_failed",
            "order_id": "ord_f1",
            "symbol": "BTCUSDC",
            "venue_status": "FILLED",
            "status": "closed",
            "filled_qty": 0.01,
            "cost": 800.5,
            "ts": 1794500008.0,
        }) + "\n")

    assert reconcile_intents_from_execution_audit(intents, audit_dir) is True
    ep_done = intents[0]["execution_payload"]
    assert ep_done["status"] == "FILLED"
    assert ep_done["executedQty"] == "0.01"
    assert ep_done["cummulativeQuoteQty"] == "800.5"
    assert ep_done["filled_at"] == pytest.approx(1794500008.0)
    assert ep_done["terminal_ts"] == pytest.approx(1794500008.0)
    assert ep_done["reconciled_terminal"] is True
    det_done = _resolve_intent_execution_details(intents[0], 1794500015.0, extract_intent_notional)
    assert det_done["exec_status"] == "FILLED"
    assert det_done["filled_qty"] == pytest.approx(0.01)
    assert det_done["fill_px"] == pytest.approx(80050.0)


def test_explicit_cancel_order_ids_and_abort_replacement_on_cancel_failure(temp_workspace):
    """Explicit cancel_order_ids limits cancellation to listed orders, and cancel refusal/exception aborts replacement."""
    ws, _ = temp_workspace
    gw = AutonomousIntentGateway(workspace_dir=ws, cache_dir="cachedb", mode="enforce")

    btc_inst = MagicMock(symbol="BTCUSDC", base="BTC", quote="USDC", provider_name="binance")
    btc_inst._provider.name = "binance"
    btc_inst._provider.get_current_price.return_value = 80000.0
    btc_inst._provider.free_balance.return_value = 5000.0

    # 1. Explicit cancel_order_ids cancels ONLY the identified order (keeping other BUY ladder & protective SELL intact)
    btc_inst.open_orders.return_value = [
        {"orderId": "bot_buy_target", "clientOrderId": "TO_1", "side": "BUY", "price": "79000.0"},
        {"orderId": "bot_buy_keep", "clientOrderId": "TO_2", "side": "BUY", "price": "77500.0"},
        {"orderId": "bot_sell_sl", "clientOrderId": "SD_0123456789abcdef0123456789abcdef", "side": "SELL", "price": "84000.0"},
    ]
    btc_inst.cancel_order.return_value = True
    btc_inst.place.return_value = {"orderId": "ai_new_1", "status": "NEW", "price": "79500.0"}

    norm = normalize_trading_decisions({
        "market_trading_decisions": [
            {
                "decision": "BUY",
                "symbol": "BTCUSDC",
                "venue": "binance",
                "order_type": "LIMIT",
                "limit_price": 79500.0,
                "cancel_bot_open_orders": True,
                "cancel_order_ids": ["bot_buy_target"],
                "confidence": 0.92,
                "suggested_notional_usd": 300.0,
            }
        ]
    })
    assert norm[0]["cancel_order_ids"] == ["bot_buy_target"]

    with patch.object(gw, "_dispatch_notify", return_value=True):
        rec_targeted = gw.submit(
            norm[0],
            source="auto_cryptotrade_executor",
            max_mode="enforce",
            instruments_map={"BTCUSDC": btc_inst},
        )
    assert rec_targeted["status"] == "EXECUTED"
    assert rec_targeted["cancel_order_ids"] == ["bot_buy_target"]
    assert rec_targeted["canceled_prior_open_orders"] == ["bot_buy_target"]
    btc_inst.cancel_order.assert_called_once_with("bot_buy_target")
    btc_inst.place.assert_called_once()

    # 2. Venue refuses cancellation (returns False) -> gateway aborts replacement and does NOT submit new order
    btc_inst.cancel_order.reset_mock()
    btc_inst.place.reset_mock()
    btc_inst.open_orders.return_value = [
        {"orderId": "ai_old_buy", "clientOrderId": "AI_old_1", "side": "BUY", "price": "78500.0"},
    ]
    btc_inst.cancel_order.return_value = False

    with patch.object(gw, "_dispatch_notify", return_value=True):
        rec_refused = gw.submit({
            "decision": "BUY",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "order_type": "LIMIT",
            "limit_price": 79400.0,
            "replace_open_orders": True,
            "confidence": 0.93,
            "suggested_notional_usd": 350.0,
        }, source="auto_cryptotrade_executor", max_mode="enforce", instruments_map={"BTCUSDC": btc_inst})

    assert rec_refused["status"] == "FAILED"
    assert rec_refused["failed_cancel_order_ids"] == ["ai_old_buy"]
    assert rec_refused["history"][-1]["action"] == "CANCEL_PRIOR_ORDER_FAILED"
    btc_inst.cancel_order.assert_called_once_with("ai_old_buy")
    btc_inst.place.assert_not_called()

    # 3. Venue raises exception during balance-unlock cancel -> gateway aborts replacement without submitting
    btc_inst.cancel_order.reset_mock()
    btc_inst.place.reset_mock()
    btc_inst._provider.free_balance.return_value = 0.0
    btc_inst.open_orders.return_value = [
        {"orderId": "bot_lock_buy", "clientOrderId": "TO_9", "side": "BUY", "price": "78000.0"},
    ]
    btc_inst.cancel_order.side_effect = RuntimeError("venue cancel timeout")

    with patch.object(gw, "_dispatch_notify", return_value=True):
        rec_exc = gw.submit({
            "decision": "BUY",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "order_type": "MARKET",
            "replace_open_orders": True,
            "confidence": 0.94,
            "suggested_notional_usd": 350.0,
        }, source="auto_cryptotrade_executor", max_mode="enforce", instruments_map={"BTCUSDC": btc_inst})

    assert rec_exc["status"] == "FAILED"
    assert rec_exc["failed_cancel_order_ids"] == ["bot_lock_buy"]
    btc_inst.place.assert_not_called()

    # 4. Targeted replacement order already raced to FILLED (absent from open_orders, order_status='closed') -> abort double-buy
    btc_inst.cancel_order.reset_mock()
    btc_inst.cancel_order.side_effect = None
    btc_inst.cancel_order.return_value = True
    btc_inst.place.reset_mock()
    btc_inst._provider.free_balance.return_value = 5000.0
    btc_inst.open_orders.return_value = []
    btc_inst.order_status.return_value = SimpleNamespace(status="closed", filled_qty=0.005)

    with patch.object(gw, "_dispatch_notify", return_value=True):
        rec_raced = gw.submit({
            "decision": "BUY",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "order_type": "LIMIT",
            "limit_price": 79600.0,
            "replace_open_orders": True,
            "cancel_order_ids": ["ai_race_filled"],
            "confidence": 0.94,
            "suggested_notional_usd": 400.0,
        }, source="auto_cryptotrade_executor", max_mode="enforce", instruments_map={"BTCUSDC": btc_inst})

    assert rec_raced["status"] == "FAILED"
    assert rec_raced["failed_cancel_order_ids"] == ["ai_race_filled"]
    btc_inst.place.assert_not_called()

    # 4b. Targeted bot order via cancel_bot_open_orders + cancel_order_ids (without replace_open_orders) also aborts when already FILLED
    with patch.object(gw, "_dispatch_notify", return_value=True):
        rec_bot_raced = gw.submit({
            "decision": "BUY",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "order_type": "LIMIT",
            "limit_price": 79600.0,
            "cancel_bot_open_orders": True,
            "cancel_order_ids": ["bot_race_filled"],
            "confidence": 0.94,
            "suggested_notional_usd": 400.0,
        }, source="auto_cryptotrade_executor", max_mode="enforce", instruments_map={"BTCUSDC": btc_inst})

    assert rec_bot_raced["status"] == "FAILED"
    assert rec_bot_raced["failed_cancel_order_ids"] == ["bot_race_filled"]
    btc_inst.place.assert_not_called()

    # 5. Targeted replacement order already canceled with zero fill -> safe to submit replacement; also supports clientOrderId match
    btc_inst.place.reset_mock()
    btc_inst.cancel_order.reset_mock()
    btc_inst.order_status.return_value = SimpleNamespace(status="canceled", filled_qty=0.0)
    btc_inst.open_orders.return_value = [
        {"orderId": "venue_77", "clientOrderId": "TO_CLIENT_77", "side": "BUY", "price": "78800.0"},
    ]

    with patch.object(gw, "_dispatch_notify", return_value=True):
        rec_ok = gw.submit({
            "decision": "BUY",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "order_type": "LIMIT",
            "limit_price": 79650.0,
            "replace_open_orders": True,
            "cancel_order_ids": ["ai_already_canceled", "TO_CLIENT_77"],
            "confidence": 0.95,
            "suggested_notional_usd": 400.0,
        }, source="auto_cryptotrade_executor", max_mode="enforce", instruments_map={"BTCUSDC": btc_inst})

    assert rec_ok["status"] == "EXECUTED"
    assert rec_ok["canceled_prior_open_orders"] == ["venue_77"]
    btc_inst.cancel_order.assert_called_once_with("venue_77")
    btc_inst.place.assert_called_once()

    # 6. Opposite-side bot BUY order that already filled does NOT abort AI SELL replacement,
    #    while a same-side raced fill aborts BEFORE canceling other open orders.
    btc_inst.place.reset_mock()
    btc_inst.cancel_order.reset_mock()
    btc_inst.order_status.side_effect = lambda oid: (
        SimpleNamespace(status="closed", filled_qty=0.5942, side="BUY")
        if oid == "bot_buy_filled"
        else SimpleNamespace(status="closed", filled_qty=0.01, side="SELL")
    )
    btc_inst.open_orders.return_value = [
        {"orderId": "ai_sell_open", "clientOrderId": "AI_sell_1", "side": "SELL", "price": "82000.0"},
    ]
    with patch.object(gw, "_dispatch_notify", return_value=True):
        rec_opp_filled = gw.submit({
            "decision": "SELL",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "order_type": "LIMIT",
            "limit_price": 81500.0,
            "replace_open_orders": True,
            "cancel_bot_open_orders": True,
            "cancel_order_ids": ["bot_buy_filled", "ai_sell_open"],
            "confidence": 0.96,
            "suggested_notional_usd": 400.0,
        }, source="auto_cryptotrade_executor", max_mode="enforce", instruments_map={"BTCUSDC": btc_inst})
    assert rec_opp_filled["status"] == "EXECUTED"
    assert rec_opp_filled["canceled_prior_open_orders"] == ["ai_sell_open"]
    btc_inst.cancel_order.assert_called_once_with("ai_sell_open")
    btc_inst.place.assert_called_once()

    btc_inst.place.reset_mock()
    btc_inst.cancel_order.reset_mock()
    with patch.object(gw, "_dispatch_notify", return_value=True):
        rec_same_filled = gw.submit({
            "decision": "SELL",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "order_type": "LIMIT",
            "limit_price": 81500.0,
            "replace_open_orders": True,
            "cancel_bot_open_orders": True,
            "cancel_order_ids": ["bot_sell_filled", "ai_sell_open"],
            "confidence": 0.96,
            "suggested_notional_usd": 400.0,
        }, source="auto_cryptotrade_executor", max_mode="enforce", instruments_map={"BTCUSDC": btc_inst})
    assert rec_same_filled["status"] == "FAILED"
    assert rec_same_filled["failed_cancel_order_ids"] == ["bot_sell_filled"]
    btc_inst.cancel_order.assert_not_called()
    btc_inst.place.assert_not_called()

    # 7. When monitororder reprices a targeted bot order (old ID canceled with 0 fill, new MO_ order open with
    #    explicit parent-child link), cancel_existing_ai_open_orders cancels the linked replacement and unlocks balance.
    btc_inst.place.reset_mock()
    btc_inst.cancel_order.reset_mock()
    btc_inst.order_status.side_effect = None
    btc_inst.order_status.return_value = SimpleNamespace(status="canceled", filled_qty=0.0, side="SELL")
    btc_inst.open_orders.return_value = [
        {
            "orderId": "10863524639",
            "clientOrderId": "MO_TA_OR_4a2f467af57547fb972a2277_0",
            "replaces_order_id": "10863498235",
            "side": "SELL",
            "price": "64802.1",
        },
    ]
    with patch.object(gw, "_dispatch_notify", return_value=True):
        rec_repriced = gw.submit({
            "decision": "SELL",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "order_type": "LIMIT",
            "limit_price": 65550.0,
            "replace_open_orders": True,
            "cancel_bot_open_orders": True,
            "cancel_order_ids": ["10863498235"],
            "confidence": 0.95,
            "suggested_notional_usd": 400.0,
        }, source="auto_cryptotrade_executor", max_mode="enforce", instruments_map={"BTCUSDC": btc_inst})
    assert rec_repriced["status"] == "EXECUTED"
    assert rec_repriced["canceled_prior_open_orders"] == ["10863524639"]
    btc_inst.cancel_order.assert_called_once_with("10863524639")
    btc_inst.place.assert_called_once()

    # 8. HOLD decision with cancel_bot_open_orders=True / cancel_order_ids cancels targeted bot open orders
    #    while preserving untargeted AI resting orders and placing no new order.
    btc_inst.place.reset_mock()
    btc_inst.cancel_order.reset_mock()
    btc_inst.open_orders.return_value = [
        {"orderId": "725781222", "clientOrderId": "AI_intent_1794905565_sell_taousdc", "side": "SELL", "price": "196.4"},
        {"orderId": "725793121", "clientOrderId": "RT_BUY_1", "side": "BUY", "price": "189.3"},
        {"orderId": "725793123", "clientOrderId": "RT_SELL_1", "side": "SELL", "price": "191.7"},
    ]
    with patch.object(gw, "_dispatch_notify", return_value=True):
        rec_hold_cancel = gw.submit({
            "decision": "HOLD",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "cancel_bot_open_orders": True,
            "cancel_order_ids": ["725793121", "725793123"],
            "confidence": 0.95,
        }, source="auto_cryptotrade_executor", max_mode="enforce", instruments_map={"BTCUSDC": btc_inst})
    assert rec_hold_cancel is not None
    assert rec_hold_cancel["decision"] == "HOLD_CANCEL"
    assert rec_hold_cancel["status"] == "CANCELED"
    assert rec_hold_cancel["requested_order_ids"] == ["725793121", "725793123"]
    assert rec_hold_cancel["canceled_order_ids"] == ["725793121", "725793123"]
    assert rec_hold_cancel["failed_order_ids"] == []
    assert [c.args[0] for c in btc_inst.cancel_order.call_args_list] == ["725793121", "725793123"]
    btc_inst.place.assert_not_called()

    # 9. Rejected BUY/SELL decisions (low confidence or invalid notional), low-confidence HOLD decisions,
    #    HOLD without explicit cancel_order_ids, HOLD with an already-canceled target order, and BUY with
    #    missing target order alongside an opposite-side MO_ SELL order must NEVER cancel protective SELL orders.
    btc_inst.cancel_order.reset_mock()
    btc_inst.open_orders.return_value = [
        {"orderId": "bot_buy_1", "clientOrderId": "RT_BUY_1", "side": "BUY", "price": "189.3"},
        {"orderId": "725793123", "clientOrderId": "RT_SELL_1", "side": "SELL", "price": "191.7"},
        {"orderId": "mo_sell_tp", "clientOrderId": "MO_AG_SELL_1", "side": "SELL", "price": "195.0"},
    ]
    btc_inst.order_status.return_value = SimpleNamespace(status="canceled", filled_qty=0.0, side="BUY")
    for non_canceling_dec in (
        # Rejected BUY due to confidence < min_confidence
        {
            "decision": "BUY",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "cancel_bot_open_orders": True,
            "cancel_order_ids": ["bot_buy_1"],
            "confidence": 0.20,
            "suggested_notional_usd": 300.0,
        },
        # Rejected BUY due to invalid notional
        {
            "decision": "BUY",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "cancel_bot_open_orders": True,
            "cancel_order_ids": ["bot_buy_1"],
            "confidence": 0.95,
            "suggested_notional_usd": 0.0,
        },
        # Rejected HOLD due to confidence < min_confidence
        {
            "decision": "HOLD",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "cancel_bot_open_orders": True,
            "cancel_order_ids": ["bot_buy_1"],
            "confidence": 0.20,
        },
        # HOLD without explicit cancel_order_ids
        {
            "decision": "HOLD",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "cancel_bot_open_orders": True,
            "confidence": 0.95,
        },
        # HOLD targeting an order ("725793121") already absent/canceled must not wipe untargeted RT_SELL_1 / MO_AG_SELL_1
    ):
        assert gw.submit(
            non_canceling_dec,
            source="auto_cryptotrade_executor",
            max_mode="enforce",
            instruments_map={"BTCUSDC": btc_inst},
        ) is None
    absent_hold_res = gw.submit(
        {
            "decision": "HOLD",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "cancel_bot_open_orders": True,
            "cancel_order_ids": ["725793121"],
            "confidence": 0.95,
        },
        source="auto_cryptotrade_executor",
        max_mode="enforce",
        instruments_map={"BTCUSDC": btc_inst},
    )
    assert absent_hold_res is not None
    assert absent_hold_res["status"] == "CANCEL_NOOP"
    assert absent_hold_res["canceled_order_ids"] == []
    assert absent_hold_res["failed_order_ids"] == []
    btc_inst.cancel_order.assert_not_called()

    # BUY with missing targeted BUY order ("stale_bot_buy_1") must NOT cancel opposite-side MO_AG_SELL_1
    with patch.object(gw, "_dispatch_notify", return_value=True):
        rec_buy_preserve_mo_sell = gw.submit({
            "decision": "BUY",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "order_type": "LIMIT",
            "limit_price": 190.0,
            "replace_open_orders": True,
            "cancel_bot_open_orders": True,
            "cancel_order_ids": ["stale_bot_buy_1"],
            "confidence": 0.95,
            "suggested_notional_usd": 300.0,
        }, source="auto_cryptotrade_executor", max_mode="enforce", instruments_map={"BTCUSDC": btc_inst})
    assert rec_buy_preserve_mo_sell["status"] == "EXECUTED"
    btc_inst.cancel_order.assert_not_called()


def test_reconcile_resumed_intent_adopts_already_filled_order_via_client_id_without_duplicate_place(temp_workspace, monkeypatch):
    """Resumed enforce intents adopt orders that already filled on the venue (absent from open_orders) via order_by_client_id, while 0-fill canceled orders still execute."""
    ws, _ = temp_workspace
    gw = AutonomousIntentGateway(workspace_dir=ws, cache_dir="cachedb", mode="enforce", notify_enabled=False)
    btc_inst = _make_mock_inst(
        "BTCUSDC",
        base="BTC",
        quote="USDC",
        venue="binance",
        price=80000.0,
        free_balance=1000.0,
        place_return={"orderId": "ord_retry_ok", "status": "FILLED"},
    )
    inst_map = {"BINANCE_BTC": btc_inst}
    monkeypatch.setattr(gw, "load_tradable_instruments", lambda **_kw: inst_map)

    # 1. Filled order absent from open_orders() is recovered via _provider.order_by_client_id -> no duplicate place()
    filled_id = "intent_1795000001_buy_btcusdc"
    gw.save_all_intents([
        {
            "intent_id": filled_id,
            "timestamp": 1795000001.0,
            "decision": "BUY",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "confidence": 0.95,
            "suggested_notional_usd": 160.0,
            "order_type": "MARKET",
            "max_mode": "enforce",
            "status": "PENDING_EXECUTION",
            "history": [],
        }
    ])
    btc_inst.open_orders.return_value = []
    btc_inst._provider.order_by_client_id = MagicMock(
        return_value={
            "orderId": "fill_771",
            "clientOrderId": f"AI_{filled_id}",
            "status": "FILLED",
            "executedQty": "0.002",
        }
    )
    resumed_filled = gw.process_all_pending()
    assert len(resumed_filled) == 1
    assert resumed_filled[0]["status"] == "EXECUTED"
    assert resumed_filled[0]["execution_payload"]["orderId"] == "fill_771"
    assert resumed_filled[0]["history"][-1]["action"] == "RECONCILED_CLIENT_ORDER_ID"
    btc_inst._provider.order_by_client_id.assert_called_once_with("BTCUSDC", f"AI_{filled_id}")
    btc_inst.place.assert_not_called()

    # 2. Canceled order with 0 fill is NOT falsely adopted as EXECUTED -> proceeds to place()
    canceled_id = "intent_1795000002_buy_btcusdc"
    gw.save_all_intents([
        {
            "intent_id": canceled_id,
            "timestamp": 1795000002.0,
            "decision": "BUY",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "confidence": 0.95,
            "suggested_notional_usd": 160.0,
            "order_type": "LIMIT",
            "limit_price": 79500.0,
            "max_mode": "enforce",
            "status": "PENDING_EXECUTION",
            "history": [],
        }
    ])
    btc_inst.order_status.return_value = SimpleNamespace(status="canceled", filled_qty=0.0)
    btc_inst._provider.order_by_client_id = MagicMock(
        return_value={
            "orderId": "canc_0",
            "clientOrderId": f"AI_{canceled_id}",
            "status": "CANCELED",
            "executedQty": "0.0",
        }
    )
    resumed_canceled = gw.process_all_pending()
    assert len(resumed_canceled) == 1
    assert resumed_canceled[0]["status"] == "EXECUTED"
    assert resumed_canceled[0]["execution_payload"]["orderId"] == "ord_retry_ok"
    btc_inst.place.assert_called_once()


def test_atomic_rollback_restores_canceled_open_orders_when_replacement_fails_or_is_blocked(temp_workspace):
    """When prior same-side open orders are canceled to unlock credit/balance and the replacement placement
    is subsequently blocked or refused, the canceled order is atomically restored via restore_cancelled_instrument_orders."""
    ws, _ = temp_workspace
    gw = AutonomousIntentGateway(workspace_dir=ws, cache_dir="cachedb", mode="enforce", notify_enabled=False)

    btc_inst = MagicMock(symbol="BTCUSDC", base="BTC", quote="USDC", provider_name="binance")
    btc_inst._provider.name = "binance"
    btc_inst._provider.get_current_price.return_value = 80000.0
    btc_inst._provider.free_balance.return_value = 500.0
    btc_inst.open_orders.return_value = [
        {
            "orderId": "ai_prior_buy_1",
            "clientOrderId": "AI_prior_1",
            "side": "BUY",
            "price": "78500.0",
            "origQty": "0.005",
            "executedQty": "0.0",
        }
    ]
    btc_inst.cancel_order.return_value = True

    place_calls = []

    def _place_side_effect(**kwargs):
        place_calls.append(kwargs)
        if len(place_calls) == 1:
            # Replacement order refused on venue
            ctx = kwargs.get("_outcome_context")
            if isinstance(ctx, dict):
                ctx["state"] = "refused"
                ctx["reason"] = "exchange_filter_failure"
            return None
        # Second call is atomic rollback restoring the canceled order
        return {"orderId": "restored_buy_1", "status": "NEW"}

    btc_inst.place.side_effect = _place_side_effect

    res = gw.submit(
        {
            "decision": "BUY",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "order_type": "LIMIT",
            "limit_price": 79200.0,
            "replace_open_orders": True,
            "confidence": 0.94,
            "suggested_notional_usd": 400.0,
        },
        source="auto_cryptotrade_executor",
        max_mode="enforce",
        instruments_map={"BTCUSDC": btc_inst},
    )

    assert res["status"] == "FAILED"
    assert res["canceled_prior_open_orders"] == ["ai_prior_buy_1"]
    assert res["restored_prior_open_orders"] == 1
    assert "restored_buy_1" in res.get("restored_ai_order_ids", [])
    assert len(place_calls) == 2
    rollback_call = place_calls[1]
    assert rollback_call["side"] == "BUY"
    assert rollback_call["price"] == pytest.approx(78500.0)
    assert rollback_call["qty"] == pytest.approx(0.005)
    assert rollback_call["market"] is False
    assert rollback_call["caller_owns_retry"] is True
    assert rollback_call["smart"] is False
    assert str(rollback_call.get("client_order_id") or "").startswith("AI_")
    # Restored AI order is tracked in known_ai_order_ids so even without clientOrderId on the venue open order
    # it is recognized as actor="AI" on the next cycle.
    known_after_rollback = collect_known_ai_order_ids(gw.load_all_intents())
    assert "restored_buy_1" in known_after_rollback
    assert rc.resolve_order_actor({"orderId": "restored_buy_1", "side": "BUY", "price": "78500.0"}, known_after_rollback) == "AI"

    # 2. Verify zero double-cancel when initial AI cancel pass is followed by balance-unlock pass on static open_orders
    btc_inst.cancel_order.reset_mock()
    place_calls.clear()
    btc_inst._provider.free_balance.side_effect = [100.0, 500.0, 500.0]
    btc_inst.open_orders.return_value = [
        {
            "orderId": "ai_prior_buy_1",
            "clientOrderId": "AI_prior_1",
            "side": "BUY",
            "price": "78500.0",
            "origQty": "0.002",
            "executedQty": "0.0",
        },
        {
            "orderId": "bot_prior_buy_2",
            "clientOrderId": "TO_prior_2",
            "side": "BUY",
            "price": "78000.0",
            "origQty": "0.003",
            "executedQty": "0.0",
        },
    ]
    res_no_dup = gw.submit(
        {
            "decision": "BUY",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "order_type": "LIMIT",
            "limit_price": 79200.0,
            "replace_open_orders": True,
            "confidence": 0.94,
            "suggested_notional_usd": 400.0,
        },
        source="auto_cryptotrade_executor",
        max_mode="enforce",
        instruments_map={"BTCUSDC": btc_inst},
    )
    assert [c.args[0] for c in btc_inst.cancel_order.call_args_list] == ["ai_prior_buy_1", "bot_prior_buy_2"]
    assert res_no_dup["restored_prior_open_orders"] == 2

    # 3. Verify ambiguous submission (state='unknown' / submit_ambiguous) does NOT restore canceled orders on top of in-flight order
    place_calls.clear()

    def _place_ambiguous(**kwargs):
        place_calls.append(kwargs)
        ctx = kwargs.get("_outcome_context")
        if isinstance(ctx, dict):
            ctx["state"] = "unknown"
            ctx["reason"] = "submit_ambiguous"
        return None

    btc_inst.place.side_effect = _place_ambiguous
    res_ambig = gw.submit(
        {
            "decision": "BUY",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "order_type": "LIMIT",
            "limit_price": 79200.0,
            "replace_open_orders": True,
            "confidence": 0.94,
            "suggested_notional_usd": 400.0,
        },
        source="auto_cryptotrade_executor",
        max_mode="enforce",
        instruments_map={"BTCUSDC": btc_inst},
    )
    assert res_ambig["status"] == "FAILED"
    assert "restored_prior_open_orders" not in res_ambig
    assert len(place_calls) == 1

    # 4. Verify manual (web_*/MN_*) and trailing-stop (TS_*/TRL_*) orders preserve canonical MN_ / SD_ prefixes on rollback
    import order_lifecycle as ol_mod

    assert rc.resolve_priority_bot_owner("web_abc123") == "manual"
    assert rc.resolve_priority_bot_owner("TS_trail_99") == "trailing_stop"
    assert rc.resolve_owner_cid_prefix("manual") == "MN"
    assert rc.resolve_owner_cid_prefix("trailing_stop") == "SD"
    restored_kwargs = []
    dummy_inst = MagicMock()
    dummy_inst.place.side_effect = lambda **kw: restored_kwargs.append(kw) or {"orderId": f"r_{len(restored_kwargs)}"}
    cnt_restored = ol_mod.restore_cancelled_instrument_orders(
        dummy_inst,
        [
            {"orderId": "m1", "clientOrderId": "web_abc123", "side": "BUY", "price": "75000.0", "origQty": "0.004"},
            {"orderId": "t1", "clientOrderId": "TS_trail_99", "side": "SELL", "price": "85000.0", "origQty": "0.004"},
        ],
        bypass_guards={"all"},
        reason="test_manual_and_ts_rollback",
    )
    assert cnt_restored == 2
    assert str(restored_kwargs[0].get("client_order_id") or "").startswith("MN_")
    assert str(restored_kwargs[1].get("client_order_id") or "").startswith("SD_")

    # 5. extract_executed_qty never reports origQty on UNKNOWN/FAILED payloads and prompt_sections ages stale SUBMIT_UNKNOWN & omits entry_px=None
    from orchestratorOS.supervisor.prompt_sections import _format_portfolio_section, _format_venue_cycle_line
    from providers.order_ohlc_utils import extract_executed_qty

    assert extract_executed_qty({"status": "UNKNOWN", "submission_state": "unknown", "executedQty": "", "origQty": "7753.6"}) == 0.0
    assert extract_executed_qty({"status": "FAILED", "origQty": "12.5"}) == 0.0

    rgnt_cyc_line = _format_venue_cycle_line(
        "t212",
        "RGNT_US_EQ",
        {"qty": 3.93, "cost_usd": 5.64, "avg_cost_px": 1.434575, "entry_price": None, "last_buy_price": 1.43, "last_sell_price": 1.5, "dca_buys": 0, "dd_pct": -15.65},
    )
    assert "entry_px=None" not in rgnt_cyc_line

    sec3c_lines = "\n".join(
        _format_portfolio_section(
            {
                "recent_ai_intents": [
                    {"decision": "SELL", "symbol": "ARBUSDC", "venue": "binance", "status": "SUBMIT_UNKNOWN", "venue_status": "UNKNOWN", "age_h": 59.5},
                    {"decision": "SELL", "symbol": "ARBUSDC", "venue": "binance", "status": "EXECUTED", "venue_status": "OPEN", "order_id": "566749020", "age_h": 0.8},
                ]
            },
            "crypto",
        )
    )
    assert "SUBMIT_UNKNOWN" not in sec3c_lines
    assert "RESTING_IN_BOOK" in sec3c_lines


def test_proposal4c_asymmetric_ai_circuit_breaker_and_replace_churn_cap(tmp_path):
    """Verify 4C standalone asymmetric by=AI circuit-breaker (24h BUY count/notional + 1h micro-replace cap; SELL & HIGH escape)."""
    from botcore import load_key_value_conf
    from orchestratorTrade.autonomous_intent_gateway import ROOT_DIR as GW_ROOT_DIR
    import providers.intent_state as intent_state_mod
    from providers.ai_circuit_breaker import (
        evaluate_ai_intent_circuit_breaker,
        parse_intent_circuit_breaker_config,
    )

    # 0. Strict acyclic decoupling: circuit-breaker SSOT lives in providers.ai_circuit_breaker with zero reverse shims
    assert not hasattr(intent_state_mod, "evaluate_ai_intent_circuit_breaker")
    assert not hasattr(intent_state_mod, "parse_intent_circuit_breaker_config")

    # 1. Versioned order_guard.conf exposes the calibrated circuit-breaker and survival knobs
    root_cfg = load_key_value_conf(GW_ROOT_DIR, GW_ROOT_DIR)
    cb_cfg = parse_intent_circuit_breaker_config(root_cfg)
    assert cb_cfg["max_daily_buys_per_symbol"] == 12
    assert cb_cfg["max_hourly_replaces_per_symbol"] == 6
    assert cb_cfg["replace_min_price_delta_pct"] == pytest.approx(0.25)
    assert cb_cfg["max_daily_buy_notional_usd"] == pytest.approx(20000.0)
    assert cb_cfg["flip_cooldown_min"] == pytest.approx(30.0)
    assert cb_cfg["flip_min_edge_pct"] == pytest.approx(0.60)
    assert cb_cfg["target_ev_fee_mult"] == pytest.approx(2.0)

    # 2. Pure unit tests on evaluate_ai_intent_circuit_breaker
    now_s = 1_795_000_000.0
    prior_12_buys = [
        {
            "intent_id": f"buy_{k}",
            "timestamp": now_s - 600.0 * (k + 1),
            "decision": "BUY",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "status": "EXECUTED",
            "clamped_notional_usd": 500.0,
            "limit_price": 90000.0,
            "execution_payload": {"orderId": f"ord_{k}", "status": "FILLED", "executedQty": "0.005"},
        }
        for k in range(12)
    ]
    # Relaxed defaults comfortably allow a 7th BUY (6 prior fills + canceled unfilled order must NOT count)
    prior_with_canceled = [
        *prior_12_buys[:6],
        {
            "intent_id": "canceled_buy",
            "timestamp": now_s - 500.0,
            "decision": "BUY",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "status": "EXECUTED",
            "clamped_notional_usd": 1500.0,
            "limit_price": 89900.0,
            "execution_payload": {"orderId": "ord_canceled", "status": "CANCELED", "executedQty": "0.0"},
        },
    ]
    ok_7th, usd_7th, rsn_7th, m_7th = evaluate_ai_intent_circuit_breaker(
        {"intent_id": "new_7", "decision": "BUY", "urgency": "NORMAL"},
        prior_with_canceled,
        decision="BUY",
        symbol="BTCUSDC",
        venue="binance",
        price=90000.0,
        effective_usd=500.0,
        min_notional_usd=10.0,
        now_ts=now_s,
    )
    assert ok_7th is True and usd_7th == pytest.approx(500.0) and rsn_7th == ""
    assert m_7th["daily_sym_buys"] == 6 and m_7th["daily_buy_notional_usd"] == pytest.approx(3000.0)

    # 13th BUY on same symbol blocked by per-symbol 24h BUY cap (default 12)
    ok_13, _, rsn_13, _ = evaluate_ai_intent_circuit_breaker(
        {"intent_id": "new_13", "decision": "BUY", "urgency": "NORMAL"},
        prior_12_buys,
        decision="BUY",
        symbol="BTCUSDC",
        venue="binance",
        price=90000.0,
        effective_usd=500.0,
        min_notional_usd=10.0,
        now_ts=now_s,
    )
    assert ok_13 is False and "ai_daily_buy_count_cap" in rsn_13

    # Asymmetric safety: protective SELL on same symbol is NEVER blocked by BUY cap
    ok_sell, usd_sell, rsn_sell, _ = evaluate_ai_intent_circuit_breaker(
        {"intent_id": "prot_sell", "decision": "SELL", "urgency": "NORMAL"},
        prior_12_buys,
        decision="SELL",
        symbol="BTCUSDC",
        venue="binance",
        price=89500.0,
        effective_usd=1500.0,
        min_notional_usd=10.0,
        now_ts=now_s,
    )
    assert ok_sell is True and usd_sell == pytest.approx(1500.0) and rsn_sell == ""

    # Relaxed default $20k notional cap allows $6,500 daily BUY without clamping
    prior_6500 = [
        {
            "intent_id": f"big_{k}",
            "timestamp": now_s - 1000.0 * (k + 1),
            "decision": "BUY",
            "symbol": f"SYM{k}USDC",
            "venue": "binance",
            "status": "EXECUTED",
            "clamped_notional_usd": 2000.0,
            "execution_payload": {"orderId": f"big_ord_{k}", "status": "FILLED", "executedQty": "1.0"},
        }
        for k in range(3)
    ]
    ok_rel, usd_rel, rsn_rel, _ = evaluate_ai_intent_circuit_breaker(
        {"intent_id": "relaxed_buy", "decision": "BUY", "urgency": "NORMAL"},
        prior_6500,
        decision="BUY",
        symbol="TAOUSDC",
        venue="binance",
        price=300.0,
        effective_usd=500.0,
        min_notional_usd=10.0,
        now_ts=now_s,
    )
    assert ok_rel is True and usd_rel == pytest.approx(500.0) and rsn_rel == ""

    # Global 24h BUY notional cap smoothly clamps remaining headroom, then blocks when exhausted
    prior_5400 = [
        {
            "intent_id": f"big_{k}",
            "timestamp": now_s - 1000.0 * (k + 1),
            "decision": "BUY",
            "symbol": f"SYM{k}USDC",
            "venue": "binance",
            "status": "EXECUTED",
            "clamped_notional_usd": 1800.0,
            "execution_payload": {"orderId": f"big_ord_{k}", "status": "FILLED", "executedQty": "1.0"},
        }
        for k in range(3)
    ]
    ok_clamp, usd_clamp, rsn_clamp, _ = evaluate_ai_intent_circuit_breaker(
        {"intent_id": "clamp_buy", "decision": "BUY", "urgency": "NORMAL"},
        prior_5400,
        decision="BUY",
        symbol="TAOUSDC",
        venue="binance",
        price=300.0,
        effective_usd=1000.0,
        min_notional_usd=10.0,
        now_ts=now_s,
        max_daily_buy_notional_usd=6000.0,
    )
    assert ok_clamp is True and usd_clamp == pytest.approx(600.0) and "ai_daily_buy_notional_clamped" in rsn_clamp

    prior_6000 = [*prior_5400, {"intent_id": "big_4", "timestamp": now_s - 300.0, "decision": "BUY", "symbol": "TAOUSDC", "venue": "binance", "status": "EXECUTED", "clamped_notional_usd": 600.0}]
    ok_full, _, rsn_full, _ = evaluate_ai_intent_circuit_breaker(
        {"intent_id": "over_buy", "decision": "BUY", "urgency": "NORMAL"},
        prior_6000,
        decision="BUY",
        symbol="ARBUSDC",
        venue="binance",
        price=1.0,
        effective_usd=200.0,
        min_notional_usd=10.0,
        now_ts=now_s,
        max_daily_buy_notional_usd=6000.0,
    )
    assert ok_full is False and "ai_daily_buy_notional_cap" in rsn_full

    # 3. End-to-end gateway test for 1h micro-replace churn cap (MUST NOT cancel resting order when blocked!)
    (tmp_path / "order_guard.conf").write_text(
        "autonomous_trading_mode = enforce\n"
        "autonomous_trading_min_notional_usd = 10.0\n"
        "autonomous_trading_max_notional_usd = 2000.0\n"
        "autonomous_trading_min_confidence = 0.85\n"
        "autonomous_trading_notify = 0\n"
        "autonomous_trading_max_daily_buys_per_symbol = 6\n"
        "autonomous_trading_max_hourly_replaces_per_symbol = 3\n"
        "autonomous_trading_replace_min_price_delta_pct = 0.50\n"
        "autonomous_trading_max_daily_buy_notional_usd = 6000.0\n",
        encoding="utf-8",
    )
    gw = AutonomousIntentGateway(workspace_dir=str(tmp_path))
    recent_replaces = [
        {
            "intent_id": f"rep_{k}",
            "timestamp": time.time() - 300.0 * (3 - k),
            "decision": "BUY",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "status": "EXECUTED",
            "order_type": "LIMIT",
            "limit_price": 90000.0 + k * 5.0,
            "replace_open_orders": True,
            "canceled_prior_open_orders": [f"old_ord_{k}"],
            "clamped_notional_usd": 300.0,
            "execution_payload": {"orderId": f"rep_ord_{k}", "status": "CANCELED" if k < 2 else "NEW", "executedQty": "0.0"},
        }
        for k in range(3)
    ]
    gw.save_all_intents(recent_replaces)

    btc_inst = MagicMock(symbol="BTCUSDC", provider_name="binance", base="BTC", quote="USDC", params={})
    btc_inst._provider.name = "binance"
    btc_inst._provider.get_current_price.return_value = 90500.0
    btc_inst._provider.free_balance.return_value = 2000.0
    btc_inst.open_orders.return_value = [
        {"orderId": "rep_ord_2", "clientOrderId": "AI_rep_2", "side": "BUY", "price": "90010.0", "origQty": "0.0033", "executedQty": "0.0"}
    ]
    btc_inst.cancel_order.return_value = True
    btc_inst.place.return_value = {"orderId": "new_placed_9", "status": "NEW", "executedQty": "0.0"}

    # 4th micro-replace (+0.05% delta < 0.50%, NORMAL urgency) -> BLOCKED without canceling rep_ord_2!
    rec_micro = gw.submit(
        {
            "decision": "BUY",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "order_type": "LIMIT",
            "limit_price": 90050.0,
            "replace_open_orders": True,
            "confidence": 0.92,
            "suggested_notional_usd": 300.0,
            "urgency": "NORMAL",
        },
        source="auto_cryptotrade_executor",
        max_mode="enforce",
        instruments_map={"BTCUSDC": btc_inst},
    )
    assert rec_micro["status"] == "REJECTED_AI_CIRCUIT_BREAKER"
    assert "ai_hourly_replace_churn_cap" in rec_micro["circuit_breaker_reason"]
    btc_inst.cancel_order.assert_not_called()
    btc_inst.place.assert_not_called()

    # Shadow mode reports guard_allowed=False with the same circuit-breaker reason
    rec_shadow = gw.submit(
        {
            "decision": "BUY",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "order_type": "LIMIT",
            "limit_price": 90050.0,
            "replace_open_orders": True,
            "confidence": 0.92,
            "suggested_notional_usd": 300.0,
            "urgency": "NORMAL",
        },
        source="auto_cryptotrade_executor",
        max_mode="shadow",
        instruments_map={"BTCUSDC": btc_inst},
    )
    assert rec_shadow["status"] == "OBSERVED_SHADOW"
    assert rec_shadow["history"][-1]["guard_allowed"] is False
    assert "ai_hourly_replace_churn_cap" in rec_shadow["history"][-1]["guard_reason"]

    # Same 4th replace with material price move (>= 0.50%) -> ALLOWED!
    rec_moved = gw.submit(
        {
            "decision": "BUY",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "order_type": "LIMIT",
            "limit_price": 89400.0,
            "replace_open_orders": True,
            "confidence": 0.92,
            "suggested_notional_usd": 300.0,
            "urgency": "NORMAL",
        },
        source="auto_cryptotrade_executor",
        max_mode="enforce",
        instruments_map={"BTCUSDC": btc_inst},
    )
    assert rec_moved["status"] == "EXECUTED"
    btc_inst.cancel_order.assert_called_once_with("rep_ord_2")
    btc_inst.place.assert_called_once()

    # Urgency=HIGH on BUY no longer bypasses the hourly micro-replace cap (bypass reserved for defensive SELL)
    btc_inst.cancel_order.reset_mock()
    btc_inst.place.reset_mock()
    btc_inst.open_orders.return_value = [
        {"orderId": "new_placed_9", "clientOrderId": "AI_rep_9", "side": "BUY", "price": "89400.0", "origQty": "0.0033", "executedQty": "0.0"}
    ]
    rec_high = gw.submit(
        {
            "decision": "BUY",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "order_type": "LIMIT",
            "limit_price": 89450.0,
            "replace_open_orders": True,
            "confidence": 0.92,
            "suggested_notional_usd": 300.0,
            "urgency": "HIGH",
        },
        source="auto_cryptotrade_executor",
        max_mode="enforce",
        instruments_map={"BTCUSDC": btc_inst},
    )
    assert rec_high["status"] == "REJECTED_AI_CIRCUIT_BREAKER"
    assert "ai_hourly_replace_churn_cap" in rec_high["circuit_breaker_reason"]
    btc_inst.place.assert_not_called()


def test_ai_ping_pong_guard_and_scorecard_circuit_breaker(tmp_path):
    """Verify deterministic anti-ping-pong flip protection, 6h whipsaw cap, and 48h AI decision scorecard BUY shadow cooldown."""
    from providers.ai_survival_guard import (
        evaluate_ai_decision_scorecard,
        evaluate_ai_ping_pong_guard,
        format_ai_decision_scorecard_line,
    )

    now_s = 1_795_100_000.0

    # 1. Rapid opposite-side flip within 30m without edge (< +0.60%) -> BLOCKED
    recent_buy_fill = [
        {
            "intent_id": "ai_b1",
            "timestamp": now_s - 600.0,  # 10m ago
            "decision": "BUY",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "status": "EXECUTED",
            "limit_price": 70000.0,
            "clamped_notional_usd": 1000.0,
            "execution_payload": {"orderId": "o_b1", "status": "FILLED", "executedQty": "0.014", "avg_fill_price": 70000.0},
        }
    ]
    ok_pp, rsn_pp, m_pp = evaluate_ai_ping_pong_guard(
        {"intent_id": "cur_s1", "decision": "SELL", "urgency": "NORMAL"},
        recent_buy_fill,
        decision="SELL",
        symbol="BTCUSDC",
        venue="binance",
        price=70100.0,  # +0.14% < 0.60% min edge
        now_ts=now_s,
    )
    assert ok_pp is False and "ai_ping_pong_flip_cooldown" in rsn_pp
    assert m_pp["flip_ref_actor"] == "AI"

    # Calibrated take-profit >= 0.60% passes immediately (e.g. +0.65% on low-fee venues)
    ok_tp, _, _ = evaluate_ai_ping_pong_guard(
        {"intent_id": "cur_s2", "decision": "SELL", "urgency": "NORMAL"},
        recent_buy_fill,
        decision="SELL",
        symbol="BTCUSDC",
        venue="binance",
        price=70455.0,  # +0.65% >= 0.60%
        now_ts=now_s,
    )
    assert ok_tp is True

    # Formulaic emergency SELL drop (<= -1.50% = -2.5 * 0.60%) passes immediately
    ok_emg, _, _ = evaluate_ai_ping_pong_guard(
        {"intent_id": "cur_s3", "decision": "SELL", "urgency": "NORMAL"},
        recent_buy_fill,
        decision="SELL",
        symbol="BTCUSDC",
        venue="binance",
        price=68880.0,  # -1.60% <= -1.50%
        now_ts=now_s,
    )
    assert ok_emg is True

    # After calibrated 30m cooldown elapses (31m), small-edge flip is allowed cleanly
    ok_cd_exp, _, _ = evaluate_ai_ping_pong_guard(
        {"intent_id": "cur_s_cd", "decision": "SELL", "urgency": "NORMAL"},
        recent_buy_fill,
        decision="SELL",
        symbol="BTCUSDC",
        venue="binance",
        price=70100.0,
        now_ts=now_s + 1260.0,  # 31m after fill
    )
    assert ok_cd_exp is True

    # Pre-committed invalidation_px breach bypasses flip cooldown even on small drop (-0.50%)
    plan_buy_fill = [
        {
            **recent_buy_fill[0],
            "invalidation_px": 69700.0,
            "target_px": 72500.0,
        }
    ]
    ok_inv, _, m_inv = evaluate_ai_ping_pong_guard(
        {"intent_id": "cur_s4", "decision": "SELL", "urgency": "NORMAL"},
        plan_buy_fill,
        decision="SELL",
        symbol="BTCUSDC",
        venue="binance",
        price=69650.0,
        spot_price=69650.0,
        now_ts=now_s,
    )
    assert ok_inv is True and m_inv.get("plan_triggered") == "INVALIDATED"

    # Frozen path-invalidated / target-hit plan status bypasses flip cooldown even after spot bounces inside [inv, tgt]
    for frozen_st in ("INVALIDATED", "TARGET_HIT"):
        frozen_plan_buy = [{**plan_buy_fill[0], "resolved_plan_status": frozen_st}]
        ok_frz, _, m_frz = evaluate_ai_ping_pong_guard(
            {"intent_id": f"cur_frz_{frozen_st}", "decision": "SELL", "urgency": "NORMAL"},
            frozen_plan_buy,
            decision="SELL",
            symbol="BTCUSDC",
            venue="binance",
            price=70100.0,
            spot_price=70100.0,
            now_ts=now_s,
        )
        assert ok_frz is True and m_frz.get("plan_triggered") == frozen_st and m_frz.get("flip_bypassed_by_plan") is True

    # Unfilled resting LIMIT order does NOT trigger flip cooldown (including status-less Binance/Kraken/HL ACKs)
    unfilled_buy = [
        {
            **recent_buy_fill[0],
            "execution_payload": {"orderId": "o_rest", "status": "NEW", "executedQty": "0.0"},
        },
        {
            "intent_id": "ai_b_ack",
            "timestamp": now_s - 300.0,
            "decision": "BUY",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "order_type": "LIMIT",
            "limit_price": 70000.0,
            "status": "EXECUTED",
            "execution_payload": {"symbol": "BTCUSDC", "orderId": 567222023, "transactTime": 1795099700000},
        },
        {
            "intent_id": "ai_kr_ack",
            "timestamp": now_s - 5000.0,
            "decision": "BUY",
            "symbol": "SPCXxUSD",
            "venue": "kraken",
            "order_type": "LIMIT",
            "limit_price": 10.0,
            "clamped_notional_usd": 500.0,
            "status": "EXECUTED",
            "execution_payload": {"txid": ["OCKOPE-6YBJP-3IKD27"], "descr": {"order": "buy 50 SPCXxUSD @ limit 10.0"}},
        },
        {
            "intent_id": "ai_hl_rest",
            "timestamp": now_s - 4000.0,
            "decision": "BUY",
            "symbol": "HYPEUSDC",
            "venue": "hyperliquid",
            "order_type": "LIMIT",
            "limit_price": 65.0,
            "clamped_notional_usd": 500.0,
            "status": "EXECUTED",
            "execution_payload": {"orderId": 611083964170, "ok": True, "msg": "resting"},
        },
    ]
    ok_rest, _, _ = evaluate_ai_ping_pong_guard(
        {"intent_id": "cur_s5", "decision": "SELL", "urgency": "NORMAL"},
        unfilled_buy,
        decision="SELL",
        symbol="BTCUSDC",
        venue="binance",
        price=70050.0,
        now_ts=now_s,
    )
    assert ok_rest is True
    sc_unfilled = evaluate_ai_decision_scorecard(
        unfilled_buy,
        now_ts=now_s,
        live_prices={"BTCUSDC": 65000.0, "SPCXxUSD": 8.0, "HYPEUSDC": 50.0},
    )
    assert sc_unfilled["count"] == 0 and sc_unfilled["buy_shadow_active"] is False

    # Reconciled passive LIMIT order placed 3h ago that filled 5m ago enforces cooldown from filled_at
    late_fill_buy = [
        {
            "intent_id": "ai_late_fill",
            "timestamp": now_s - 10800.0,  # staged 3h ago
            "decision": "BUY",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "order_type": "LIMIT",
            "limit_price": 70000.0,
            "status": "EXECUTED",
            "execution_payload": {
                "orderId": "o_late",
                "status": "FILLED",
                "executedQty": "0.01",
                "price": 70000.0,
                "filled_at": now_s - 300.0,  # filled 5m ago
                "reconciled_terminal": True,
            },
        }
    ]
    ok_late, rsn_late, m_late = evaluate_ai_ping_pong_guard(
        {"intent_id": "cur_s_late", "decision": "SELL", "urgency": "NORMAL"},
        late_fill_buy,
        decision="SELL",
        symbol="BTCUSDC",
        venue="binance",
        price=70050.0,
        now_ts=now_s,
    )
    assert ok_late is False and "ai_ping_pong_flip_cooldown" in rsn_late
    assert m_late["flip_elapsed_min"] == pytest.approx(5.0)

    # Fresh by=BOT fill within 30m also blocks unprofitable reversal
    ok_bot, rsn_bot, m_bot = evaluate_ai_ping_pong_guard(
        {"intent_id": "cur_b_rev", "decision": "BUY", "urgency": "NORMAL"},
        [],
        decision="BUY",
        symbol="BTCUSDC",
        venue="binance",
        price=70020.0,
        now_ts=now_s,
        recent_bot_fill={"ts": now_s - 420.0, "price": 70000.0, "side": "SELL", "actor": "BOT"},
    )
    assert ok_bot is False and "by=BOT" in rsn_bot and m_bot["flip_ref_actor"] == "BOT"

    # 6-hour whipsaw cap (2 flips already happened: BUY -> SELL -> BUY -> 3rd reversal SELL blocked even at +1.2%)
    whipsaw_hist = [
        {
            "intent_id": "w1",
            "timestamp": now_s - 14400.0,
            "decision": "BUY",
            "symbol": "TAOUSDC",
            "venue": "binance",
            "status": "EXECUTED",
            "execution_payload": {"orderId": "ow1", "status": "FILLED", "executedQty": "2.0", "avg_fill_price": 200.0},
        },
        {
            "intent_id": "w2",
            "timestamp": now_s - 10800.0,
            "decision": "SELL",
            "symbol": "TAOUSDC",
            "venue": "binance",
            "status": "EXECUTED",
            "execution_payload": {"orderId": "ow2", "status": "FILLED", "executedQty": "2.0", "avg_fill_price": 205.0},
        },
        {
            "intent_id": "w3",
            "timestamp": now_s - 7200.0,
            "decision": "BUY",
            "symbol": "TAOUSDC",
            "venue": "binance",
            "status": "EXECUTED",
            "execution_payload": {"orderId": "ow3", "status": "FILLED", "executedQty": "2.0", "avg_fill_price": 201.0},
        },
    ]
    ok_ws, rsn_ws, _ = evaluate_ai_ping_pong_guard(
        {"intent_id": "w4", "decision": "SELL", "urgency": "NORMAL"},
        whipsaw_hist,
        decision="SELL",
        symbol="TAOUSDC",
        venue="binance",
        price=204.0,  # +1.49% favorable, >45m elapsed, but 3rd flip in 6h -> BLOCKED
        now_ts=now_s,
    )
    assert ok_ws is False and "ai_ping_pong_whipsaw_cap" in rsn_ws

    # 2. AI Decision Scorecard: standalone profit-taking SELL never penalized
    standalone_sell = [
        {
            "intent_id": "ss1",
            "timestamp": now_s - 3600.0,
            "decision": "SELL",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "status": "EXECUTED",
            "clamped_notional_usd": 1000.0,
            "execution_payload": {"orderId": "oss1", "status": "FILLED", "executedQty": "0.015", "avg_fill_price": 70000.0},
        }
    ]
    sc_empty = evaluate_ai_decision_scorecard(standalone_sell, now_ts=now_s, live_prices={"BTCUSDC": 75000.0})
    assert sc_empty["count"] == 0 and sc_empty["buy_shadow_active"] is False

    # 3 consecutive LOSS outcomes (closed BUY->SELL loss + bad SELL->BUY reload + matured losing BUY MTM) -> BUY shadow cooldown!
    losing_history = [
        # Loss 1: BUY @ 210 -> SELL @ 205 (-2.38%)
        {
            "intent_id": "l1_b",
            "timestamp": now_s - 10800.0,
            "decision": "BUY",
            "symbol": "TAOUSDC",
            "venue": "binance",
            "status": "EXECUTED",
            "clamped_notional_usd": 1000.0,
            "execution_payload": {"orderId": "ol1b", "status": "FILLED", "executedQty": "4.76", "avg_fill_price": 210.0},
        },
        {
            "intent_id": "l1_s",
            "timestamp": now_s - 9000.0,
            "decision": "SELL",
            "symbol": "TAOUSDC",
            "venue": "binance",
            "status": "EXECUTED",
            "clamped_notional_usd": 1000.0,
            "execution_payload": {"orderId": "ol1s", "status": "FILLED", "executedQty": "4.76", "avg_fill_price": 205.0},
        },
        # Loss 2: unmatched SELL @ 0.15 -> higher reload BUY @ 0.156 (-4.0%)
        {
            "intent_id": "l2_s",
            "timestamp": now_s - 7200.0,
            "decision": "SELL",
            "symbol": "ARBUSDC",
            "venue": "binance",
            "status": "EXECUTED",
            "clamped_notional_usd": 500.0,
            "execution_payload": {"orderId": "ol2s", "status": "FILLED", "executedQty": "3333", "avg_fill_price": 0.15},
        },
        {
            "intent_id": "l2_b",
            "timestamp": now_s - 5400.0,
            "decision": "BUY",
            "symbol": "ARBUSDC",
            "venue": "binance",
            "status": "EXECUTED",
            "clamped_notional_usd": 500.0,
            "execution_payload": {"orderId": "ol2b", "status": "FILLED", "executedQty": "3205", "avg_fill_price": 0.156},
        },
        # Loss 3: matured open BUY @ 80.0 now trading at 77.5 (-3.125%)
        {
            "intent_id": "l3_b",
            "timestamp": now_s - 3600.0,
            "decision": "BUY",
            "symbol": "HYPEUSDC",
            "venue": "hyperliquid",
            "status": "EXECUTED",
            "clamped_notional_usd": 800.0,
            "execution_payload": {"orderId": "ol3b", "status": "FILLED", "executedQty": "10.0", "avg_fill_price": 80.0},
        },
    ]
    sc_bad = evaluate_ai_decision_scorecard(
        losing_history,
        now_ts=now_s,
        live_prices={"hyperliquid:HYPEUSDC": 77.5},
        window_trades=5,
        max_loss_streak=3,
        cooldown_hours=4.0,
    )
    assert sc_bad["count"] == 3 and sc_bad["losses"] == 3 and sc_bad["loss_streak"] == 3
    assert sc_bad["buy_shadow_active"] is True
    assert sc_bad["status"] == "BUY_SHADOW_COOLDOWN"
    assert sc_bad["cooldown_remaining_h"] > 2.5
    sc_line = format_ai_decision_scorecard_line(sc_bad)
    assert "0W/0BE/3L" in sc_line and "BUY_SHADOW_COOLDOWN" in sc_line

    # Cooldown auto-expires once > 4h elapsed from latest loss
    sc_expired = evaluate_ai_decision_scorecard(
        losing_history,
        now_ts=now_s + 4.5 * 3600.0,
        live_prices={"hyperliquid:HYPEUSDC": 77.5},
        window_trades=5,
        max_loss_streak=3,
        cooldown_hours=4.0,
    )
    assert sc_expired["buy_shadow_active"] is False

    # 3. End-to-end gateway test: scorecard downgrades BUY to OBSERVED_SHADOW while allowing protective SELL live
    (tmp_path / "order_guard.conf").write_text(
        "autonomous_trading_mode = enforce\n"
        "autonomous_trading_min_notional_usd = 10.0\n"
        "autonomous_trading_max_notional_usd = 2000.0\n"
        "autonomous_trading_min_confidence = 0.85\n"
        "autonomous_trading_notify = 0\n"
        "autonomous_trading_flip_cooldown_min = 45\n"
        "autonomous_trading_flip_min_edge_pct = 0.90\n"
        "autonomous_trading_max_flips_6h = 2\n"
        "autonomous_trading_scorecard_window_trades = 5\n"
        "autonomous_trading_scorecard_max_loss_streak = 3\n"
        "autonomous_trading_scorecard_cooldown_hours = 4.0\n",
        encoding="utf-8",
    )
    gw = AutonomousIntentGateway(workspace_dir=str(tmp_path))
    live_now = time.time()
    shifted = []
    for it in losing_history:
        dt = now_s - it["timestamp"]
        shifted.append({**it, "timestamp": live_now - dt})
    gw.save_all_intents(shifted)
    (tmp_path / "cachedb" / "prices").mkdir(parents=True, exist_ok=True)
    (tmp_path / "cachedb" / "prices" / "cache_current_price.json").write_text(
        json.dumps({"items": {"HYPEUSDC": 77.5, "BTCUSDC": 70000.0}}),
        encoding="utf-8",
    )

    btc_inst = MagicMock(symbol="BTCUSDC", provider_name="binance", base="BTC", quote="USDC", params={})
    btc_inst._provider.name = "binance"
    btc_inst._provider.get_current_price.return_value = 70000.0
    btc_inst._provider.free_balance.side_effect = lambda a: 3000.0 if a == "USDC" else 0.1
    btc_inst.open_orders.return_value = []
    btc_inst.place.return_value = {"orderId": "sell_live_1", "status": "NEW", "executedQty": "0.0"}

    # New BUY during 3L streak -> downgraded to OBSERVED_SHADOW without touching exchange!
    rec_buy_cd = gw.submit(
        {
            "decision": "BUY",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "order_type": "LIMIT",
            "limit_price": 69800.0,
            "confidence": 0.95,
            "suggested_notional_usd": 500.0,
            "urgency": "HIGH",
        },
        source="auto_cryptotrade_executor",
        max_mode="enforce",
        instruments_map={"BTCUSDC": btc_inst},
    )
    assert rec_buy_cd["status"] == "OBSERVED_SHADOW"
    assert "ai_scorecard_buy_shadow_cooldown" in rec_buy_cd.get("scorecard_downgraded_to_shadow", "")
    btc_inst.place.assert_not_called()

    # Protective SELL during the same 3L streak -> still executes live!
    rec_sell_ok = gw.submit(
        {
            "decision": "SELL",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "order_type": "LIMIT",
            "limit_price": 70050.0,
            "confidence": 0.92,
            "suggested_notional_usd": 500.0,
            "urgency": "NORMAL",
        },
        source="auto_cryptotrade_executor",
        max_mode="enforce",
        instruments_map={"BTCUSDC": btc_inst},
    )
    assert rec_sell_ok["status"] == "EXECUTED"
    btc_inst.place.assert_called_once()

    # 4. Regression checks from verification pass:
    # (a) Closed stale AI BUY plan does NOT bypass a fresh by=BOT BUY flip cooldown
    closed_plan_hist = [
        {
            "intent_id": "cp_b1",
            "timestamp": now_s - 86400.0,
            "decision": "BUY",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "status": "EXECUTED",
            "invalidation_px": 68000.0,
            "execution_payload": {"orderId": "ocp1", "status": "FILLED", "executedQty": "0.01", "avg_fill_price": 70000.0},
        },
        {
            "intent_id": "cp_s1",
            "timestamp": now_s - 43200.0,
            "decision": "SELL",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "status": "EXECUTED",
            "execution_payload": {"orderId": "ocp2", "status": "FILLED", "executedQty": "0.01", "avg_fill_price": 68000.0},
        },
    ]
    ok_stale_plan, rsn_stale_plan, _ = evaluate_ai_ping_pong_guard(
        {"intent_id": "cp_s2", "decision": "SELL", "urgency": "NORMAL"},
        closed_plan_hist,
        decision="SELL",
        symbol="BTCUSDC",
        venue="binance",
        price=67500.0,
        spot_price=67500.0,
        now_ts=now_s,
        recent_bot_fill={"ts": now_s - 300.0, "price": 67550.0, "side": "BUY", "actor": "BOT"},
    )
    assert ok_stale_plan is False and "by=BOT" in rsn_stale_plan

    # (b) 6h whipsaw cap MUST still allow emergency crash SELL even when last AI BUY is > 45m old
    ok_ws_crash, _, _ = evaluate_ai_ping_pong_guard(
        {"intent_id": "w_crash", "decision": "SELL", "urgency": "NORMAL"},
        whipsaw_hist,  # last BUY @ 201.0 was 2h ago (> 45m flip cooldown)
        decision="SELL",
        symbol="TAOUSDC",
        venue="binance",
        price=190.0,  # -5.47% crash <= -2.25% emergency drop
        now_ts=now_s,
    )
    assert ok_ws_crash is True

    # (c) LIFO matching: harvesting a profitable dip BUY (75k -> 68k -> SELL 70k) scores latest outcome as WIN
    lifo_hist = [
        {
            "intent_id": "lf1",
            "timestamp": now_s - 10800.0,
            "decision": "BUY",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "status": "EXECUTED",
            "clamped_notional_usd": 500.0,
            "execution_payload": {"orderId": "olf1", "status": "FILLED", "executedQty": "0.0067", "avg_fill_price": 75000.0},
        },
        {
            "intent_id": "lf2",
            "timestamp": now_s - 7200.0,
            "decision": "BUY",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "status": "EXECUTED",
            "clamped_notional_usd": 500.0,
            "execution_payload": {"orderId": "olf2", "status": "FILLED", "executedQty": "0.0073", "avg_fill_price": 68000.0},
        },
        {
            "intent_id": "lf3",
            "timestamp": now_s - 1800.0,
            "decision": "SELL",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "status": "EXECUTED",
            "clamped_notional_usd": 515.0,
            "execution_payload": {"orderId": "olf3", "status": "FILLED", "executedQty": "0.0073", "avg_fill_price": 70000.0},
        },
    ]
    sc_lifo = evaluate_ai_decision_scorecard(lifo_hist, now_ts=now_s, live_prices={"BTCUSDC": 70000.0})
    assert sc_lifo["wins"] == 1 and sc_lifo["losses"] == 1 and sc_lifo["loss_streak"] == 0

    # (d) Gateway scorecard parses production [[ts_ms, price]] cache_current_price format & isolates crypto vs equities scope
    (tmp_path / "cachedb" / "prices" / "cache_current_price.json").write_text(
        json.dumps({"items": {"HYPEUSDC": [[int(live_now * 1000), 77.5]], "BTCUSDC": [[int(live_now * 1000), 70000.0]]}}),
        encoding="utf-8",
    )
    loaded_px = gw._load_live_prices_for_scorecard("BTCUSDC", "binance", 70000.0)
    assert loaded_px["HYPEUSDC"] == pytest.approx(77.5)

    # 3 T212 equity losses must NOT downgrade a Binance crypto BUY (asset-scope isolation)
    eq_losses = [
        {
            "intent_id": f"eq_l{k}",
            "timestamp": live_now - (3600.0 * (4 - k)),
            "decision": "BUY",
            "symbol": "NVDA_US_EQ",
            "venue": "t212",
            "status": "EXECUTED",
            "clamped_notional_usd": 300.0,
            "execution_payload": {"orderId": f"oeq{k}", "status": "FILLED", "executedQty": "1.5", "avg_fill_price": 200.0},
        }
        for k in range(3)
    ]
    gw.save_all_intents(eq_losses)
    (tmp_path / "cachedb" / "prices" / "cache_current_price.json").write_text(
        json.dumps({"items": {"NVDA_US_EQ": [[int(live_now * 1000), 185.0]], "BTCUSDC": [[int(live_now * 1000), 70000.0]]}}),
        encoding="utf-8",
    )
    btc_inst.place.reset_mock()
    btc_inst.place.return_value = {"orderId": "btc_buy_isolated", "status": "NEW", "executedQty": "0.0"}
    rec_crypto_ok = gw.submit(
        {
            "decision": "BUY",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "order_type": "LIMIT",
            "limit_price": 69800.0,
            "confidence": 0.92,
            "suggested_notional_usd": 500.0,
            "urgency": "NORMAL",
        },
        source="auto_cryptotrade_executor",
        max_mode="enforce",
        instruments_map={"BTCUSDC": btc_inst},
    )
    assert rec_crypto_ok["status"] == "EXECUTED"
    btc_inst.place.assert_called_once()

    # (e) _resolve_recent_bot_opposite_fill ignores synthetic ID-less fallbacks (e.g. T212 averagePrice) and non-fills
    fake_t212_inst = MagicMock(symbol="NVDA_US_EQ", provider_name="t212")
    fake_t212_inst._provider.get_orders.return_value = [
        {"side": "BUY", "price": 190.0, "qty": 1.0, "timestamp": int((live_now - 900.0) * 1000)},
        {"orderId": "bot_canc_1", "side": "BUY", "price": 190.0, "status": "CANCELED", "timestamp": int((live_now - 300.0) * 1000)},
    ]
    assert AutonomousIntentGateway._resolve_recent_bot_opposite_fill(fake_t212_inst, "SELL", live_now, []) is None
    fake_t212_inst._provider.get_orders.return_value.extend([
        {"orderId": "ai_known_1", "side": "BUY", "price": 192.0, "status": "FILLED", "timestamp": int((live_now - 120.0) * 1000)},
        {"orderId": "bot_fill_1", "side": "BUY", "price": 191.0, "status": "FILLED", "timestamp": int((live_now - 240.0) * 1000)},
    ])
    known_ai_intents = [{"intent_id": "ai_k1", "execution_payload": {"orderId": "ai_known_1"}}]
    bot_hit = AutonomousIntentGateway._resolve_recent_bot_opposite_fill(fake_t212_inst, "SELL", live_now, known_ai_intents)
    assert bot_hit is not None and bot_hit["actor"] == "BOT" and bot_hit["price"] == pytest.approx(191.0)

    # (f) Real Binance MARKET fills (price='0.00000000' + fills/cummulativeQuoteQty), partial fills, and reconciled LIMIT fill timestamps
    from providers.ai_survival_guard import (
        _extract_intent_exec_price,
        _extract_intent_exec_ts,
        _extract_intent_notional_usd,
    )
    from intelligence.sentiment.collectors.guard_probe_collector import probe_deterministic_guards

    binance_mkt_buy = {
        "intent_id": "bn_mkt_1",
        "timestamp": live_now - 7200.0,
        "decision": "BUY",
        "symbol": "BTCUSDC",
        "venue": "binance",
        "status": "EXECUTED",
        "clamped_notional_usd": 800.0,
        "execution_payload": {
            "orderId": 9001,
            "status": "FILLED",
            "price": "0.00000000",
            "executedQty": "0.005",
            "cummulativeQuoteQty": "400.0",
            "transactTime": int((live_now - 7190.0) * 1000),
            "fills": [{"price": "80000.00000000", "qty": "0.005"}],
        },
    }
    assert _extract_intent_exec_price(binance_mkt_buy) == pytest.approx(80000.0)
    assert _extract_intent_exec_ts(binance_mkt_buy) == pytest.approx(live_now - 7190.0)
    assert _extract_intent_notional_usd(binance_mkt_buy) == pytest.approx(400.0)

    # Placement executed_at must NOT shadow a later resting LIMIT fill_at / RECONCILED_EXECUTED history ts
    reconciled_limit = {
        "intent_id": "lim_rec_1",
        "timestamp": live_now - 10800.0,
        "executed_at": live_now - 10795.0,
        "decision": "BUY",
        "symbol": "TAOUSDC",
        "venue": "binance",
        "order_type": "LIMIT",
        "status": "EXECUTED",
        "execution_payload": {"orderId": 9002, "status": "FILLED", "executedQty": "2.0", "filled_at": live_now - 600.0},
    }
    assert _extract_intent_exec_ts(reconciled_limit) == pytest.approx(live_now - 600.0)

    hist_reconciled = {
        "intent_id": "hist_rec_1",
        "timestamp": live_now - 10800.0,
        "decision": "SELL",
        "symbol": "TAOUSDC",
        "venue": "binance",
        "status": "EXECUTED",
        "execution_payload": {"orderId": 9003, "status": "FILLED", "executedQty": "2.0", "cummulativeQuoteQty": "420.0"},
        "history": [{"ts": live_now - 900.0, "action": "RECONCILED_EXECUTED"}],
    }
    assert _extract_intent_exec_price(hist_reconciled) == pytest.approx(210.0)
    assert _extract_intent_exec_ts(hist_reconciled) == pytest.approx(live_now - 900.0)

    # (g) probe_deterministic_guards honors intelligence_guards_mode='off', venue reference='off', and stale micro timestamps
    stale_micro = {
        "ob_imbalance": 0.10,
        "ask_wall_usd": 2_000_000.0,
        "ask_wall_px": 100.5,
        "_ob_raw_ts": live_now - 86400.0 * 3,
        "taker_buy_sell_ratio": 0.50,
        "_whale_ts": live_now - 86400.0 * 3,
    }
    p_stale = probe_deterministic_guards(
        "BTCUSDC", "BUY", 100.0, venue="binance", microstructure_entry=stale_micro, now_ts=live_now, allow_disk_cache=False
    )
    assert p_stale["flags"] == []

    fresh_micro = {
        "ob_imbalance": 0.10,
        "ask_wall_usd": 2_000_000.0,
        "ask_wall_px": 100.5,
        "_ob_raw_ts": live_now - 30.0,
    }
    p_master_off = probe_deterministic_guards(
        "BTCUSDC",
        "BUY",
        100.0,
        venue="binance",
        margins={"intelligence_guards_mode": "off", "default": 1.15},
        microstructure_entry=fresh_micro,
        opposite_fill_px=100.2,
        now_ts=live_now,
        allow_disk_cache=False,
    )
    assert [f["guard"] for f in p_master_off["flags"]] == ["G_PROF"]

    p_ref_off = probe_deterministic_guards(
        "BTCUSDC",
        "BUY",
        100.0,
        venue="binance",
        margins={"intelligence_guards_mode": "off", "binance_buy_reference": "off", "default": 1.15},
        opposite_fill_px=100.2,
        now_ts=live_now,
        allow_disk_cache=False,
    )
    assert p_ref_off["flags"] == []

    # (h) Open BUY_MTM calibration: normal intraday DCA dip (-0.50% above invalidation) NEVER triggers false BUY cooldown,
    # whereas breaching invalidation_px (even at 20m) or maturing (>= 60m) below -1.50% DOES score LOSS.
    dca_dip_open = [
        {
            "intent_id": f"dca_{k}",
            "timestamp": now_s - (2700.0 if k == 0 else 5400.0 + k * 600.0),  # 45m (immature) + 90m/100m (matured)
            "decision": "BUY",
            "symbol": f"SYMDCA{k}USDC",
            "venue": "binance",
            "status": "EXECUTED",
            "clamped_notional_usd": 500.0,
            "invalidation_px": 97.0,
            "target_px": 106.0,
            "execution_payload": {"orderId": f"odca_{k}", "status": "FILLED", "executedQty": "5.0", "avg_fill_price": 100.0},
        }
        for k in range(3)
    ]
    sc_dca = evaluate_ai_decision_scorecard(
        dca_dip_open,
        now_ts=now_s,
        live_prices={"SYMDCA0USDC": 99.5, "SYMDCA1USDC": 99.5, "SYMDCA2USDC": 99.5},  # -0.50% dip, above 97.0 invalidation
    )
    assert sc_dca["count"] == 2  # 45m tranche still immature (< 60m); 90m & 100m tranches matured as SCRATCH
    assert sc_dca["losses"] == 0 and sc_dca["scratches"] == 2 and sc_dca["buy_shadow_active"] is False

    # Breaching invalidation_px scores LOSS immediately even before 60m maturity (20m old, -0.50% vs 69700 invalidation)
    inv_open = [
        {
            "intent_id": "inv_fast",
            "timestamp": now_s - 1200.0,  # 20m ago
            "decision": "BUY",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "status": "EXECUTED",
            "clamped_notional_usd": 500.0,
            "invalidation_px": 69700.0,
            "target_px": 73000.0,
            "execution_payload": {"orderId": "oinv_fast", "status": "FILLED", "executedQty": "0.007", "avg_fill_price": 70000.0},
        }
    ]
    sc_inv = evaluate_ai_decision_scorecard(inv_open, now_ts=now_s, live_prices={"BTCUSDC": 69650.0})
    assert sc_inv["count"] == 1 and sc_inv["losses"] == 1 and sc_inv["loss_streak"] == 1
    # Frozen path-invalidated open BUY stays LOSS before 60m maturity even if spot bounces above invalidation_px
    inv_open_frozen = [{**inv_open[0], "resolved_plan_status": "INVALIDATED"}]
    sc_inv_frz = evaluate_ai_decision_scorecard(inv_open_frozen, now_ts=now_s, live_prices={"BTCUSDC": 69850.0})
    assert sc_inv_frz["count"] == 1 and sc_inv_frz["losses"] == 1 and sc_inv_frz["loss_streak"] == 1

    # Closed realized BUY->SELL round-trip at -0.50% STILL scores LOSS (< -0.35% closed threshold)
    closed_small_loss = [
        {
            "intent_id": "csl_b",
            "timestamp": now_s - 3600.0,
            "decision": "BUY",
            "symbol": "TAOUSDC",
            "venue": "binance",
            "status": "EXECUTED",
            "clamped_notional_usd": 500.0,
            "execution_payload": {"orderId": "ocsl_b", "status": "FILLED", "executedQty": "2.5", "avg_fill_price": 200.0},
        },
        {
            "intent_id": "csl_s",
            "timestamp": now_s - 1800.0,
            "decision": "SELL",
            "symbol": "TAOUSDC",
            "venue": "binance",
            "status": "EXECUTED",
            "clamped_notional_usd": 497.5,
            "execution_payload": {"orderId": "ocsl_s", "status": "FILLED", "executedQty": "2.5", "avg_fill_price": 199.0},
        },
    ]
    sc_csl = evaluate_ai_decision_scorecard(closed_small_loss, now_ts=now_s)
    assert sc_csl["count"] == 1 and sc_csl["losses"] == 1

    # Zero-fill CANCELED_OR_EXPIRED / FAILED intents must not consume daily BUY budget, and circuit-breaker rejections are terminal
    from providers.ai_circuit_breaker import evaluate_ai_intent_circuit_breaker
    from providers.order_status_ssot import TERMINAL_ACTIVE_INTENT_STATUSES

    assert "rejected_ai_circuit_breaker" in TERMINAL_ACTIVE_INTENT_STATUSES
    zero_fill_non_counted = [
        {
            "intent_id": "zf_1",
            "timestamp": now_s - 600.0,
            "decision": "BUY",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "status": "EXECUTED",
            "clamped_notional_usd": 5000.0,
            "execution_payload": {"orderId": "ozf1", "status": "CANCELED_OR_EXPIRED", "executedQty": "0.0"},
        },
        {
            "intent_id": "zf_2",
            "timestamp": now_s - 500.0,
            "decision": "BUY",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "status": "EXECUTED",
            "clamped_notional_usd": 5000.0,
            "execution_payload": {"orderId": "ozf2", "status": "FAILED", "executedQty": "0.0"},
        },
    ]
    ok_zf, usd_zf, _, m_zf = evaluate_ai_intent_circuit_breaker(
        {"intent_id": "zf_new", "decision": "BUY"},
        zero_fill_non_counted,
        decision="BUY",
        symbol="BTCUSDC",
        venue="binance",
        price=70000.0,
        effective_usd=1000.0,
        min_notional_usd=10.0,
        now_ts=now_s,
        max_daily_buy_notional_usd=5000.0,
    )
    assert ok_zf is True and usd_zf == pytest.approx(1000.0) and m_zf["daily_buy_notional_usd"] == 0.0


def test_fee_aware_ping_pong_and_4b1_negative_ev_target_guard(tmp_path):
    """Verify venue-fee-aware ping-pong thresholds and 4B-1 negative-EV target_px filtering."""
    from providers.ai_survival_guard import (
        evaluate_ai_ping_pong_guard,
        evaluate_ai_target_ev_guard,
        resolve_venue_flip_min_edge_pct,
        resolve_venue_round_trip_fee_bps,
    )

    # 1. Venue SSOT round-trip fees and fee-aware flip minimum edge scaling
    assert resolve_venue_round_trip_fee_bps("kraken", "SPCXxUSD", order_type="LIMIT") == pytest.approx(0.0)
    assert resolve_venue_round_trip_fee_bps("hyperliquid", "HYPEUSDC", order_type="LIMIT") == pytest.approx(8.0)
    assert resolve_venue_round_trip_fee_bps("hyperliquid", "HYPEUSDC", order_type="MARKET") == pytest.approx(14.0)
    assert resolve_venue_round_trip_fee_bps("binance", "BTCUSDC", order_type="LIMIT") == pytest.approx(20.0)
    assert resolve_venue_round_trip_fee_bps("t212", "NVDA_US_EQ", order_type="LIMIT") == pytest.approx(30.0)
    assert resolve_venue_round_trip_fee_bps("kraken", "TAOUSD", order_type="LIMIT") == pytest.approx(32.0)
    assert resolve_venue_round_trip_fee_bps("kraken", "TAOUSD", order_type="MARKET") == pytest.approx(52.0)

    assert resolve_venue_flip_min_edge_pct("kraken", "SPCXxUSD", order_type="LIMIT") == (pytest.approx(0.20), pytest.approx(0.0))
    assert resolve_venue_flip_min_edge_pct("hyperliquid", "HYPEUSDC", order_type="LIMIT") == (pytest.approx(0.36), pytest.approx(8.0))
    assert resolve_venue_flip_min_edge_pct("binance", "BTCUSDC", order_type="LIMIT") == (pytest.approx(0.60), pytest.approx(20.0))
    assert resolve_venue_flip_min_edge_pct("t212", "NVDA_US_EQ", order_type="LIMIT") == (pytest.approx(0.80), pytest.approx(30.0))
    assert resolve_venue_flip_min_edge_pct("kraken", "TAOUSD", order_type="LIMIT") == (pytest.approx(0.84), pytest.approx(32.0))
    assert resolve_venue_flip_min_edge_pct("kraken", "TAOUSD", order_type="MARKET") == (pytest.approx(1.24), pytest.approx(52.0))

    # 2. Ping-pong respects per-venue fee tiers: +0.45% passes on HL LIMIT (0.36%), +0.65% blocks on Kraken Crypto LIMIT (0.84%), +0.25% passes on xStock (0.20%)
    now_s = 1_795_200_000.0
    hl_buy = [{
        "intent_id": "hl_b1",
        "timestamp": now_s - 600.0,
        "decision": "BUY",
        "symbol": "HYPEUSDC",
        "venue": "hyperliquid",
        "order_type": "LIMIT",
        "status": "EXECUTED",
        "execution_payload": {"orderId": "ohl1", "status": "FILLED", "executedQty": "10.0", "avg_fill_price": 50.0},
    }]
    ok_hl, _, m_hl = evaluate_ai_ping_pong_guard(
        {"intent_id": "hl_s1", "decision": "SELL", "order_type": "LIMIT", "urgency": "NORMAL"},
        hl_buy,
        decision="SELL",
        symbol="HYPEUSDC",
        venue="hyperliquid",
        price=50.225,  # +0.45% >= 0.36%
        now_ts=now_s,
    )
    assert ok_hl is True and m_hl["effective_flip_min_edge_pct"] == pytest.approx(0.36)

    kr_buy = [{
        "intent_id": "kr_b1",
        "timestamp": now_s - 600.0,
        "decision": "BUY",
        "symbol": "TAOUSD",
        "venue": "kraken",
        "order_type": "LIMIT",
        "status": "EXECUTED",
        "execution_payload": {"orderId": "okr1", "status": "FILLED", "executedQty": "2.0", "avg_fill_price": 200.0},
    }]
    ok_kr, rsn_kr, m_kr = evaluate_ai_ping_pong_guard(
        {"intent_id": "kr_s1", "decision": "SELL", "order_type": "LIMIT", "urgency": "NORMAL"},
        kr_buy,
        decision="SELL",
        symbol="TAOUSD",
        venue="kraken",
        price=201.30,  # +0.65% < 0.84% Kraken Crypto LIMIT edge
        now_ts=now_s,
    )
    assert ok_kr is False and "ai_ping_pong_flip_cooldown" in rsn_kr
    assert m_kr["effective_flip_min_edge_pct"] == pytest.approx(0.84)

    xs_buy = [{
        "intent_id": "xs_b1",
        "timestamp": now_s - 600.0,
        "decision": "BUY",
        "symbol": "SPCXxUSD",
        "venue": "kraken",
        "order_type": "LIMIT",
        "status": "EXECUTED",
        "execution_payload": {"orderId": "oxs1", "status": "FILLED", "executedQty": "50.0", "avg_fill_price": 10.0},
    }]
    ok_xs, _, m_xs = evaluate_ai_ping_pong_guard(
        {"intent_id": "xs_s1", "decision": "SELL", "order_type": "LIMIT", "urgency": "NORMAL"},
        xs_buy,
        decision="SELL",
        symbol="SPCXxUSD",
        venue="kraken",
        price=10.025,  # +0.25% >= 0.20% xStock zero-fee edge
        now_ts=now_s,
    )
    assert ok_xs is True and m_xs["venue_rt_fee_bps"] == pytest.approx(0.0)

    # 3. 4B-1 Negative-EV target_px guard unit tests
    ok_ev_bad, rsn_ev_bad, m_ev_bad = evaluate_ai_target_ev_guard(
        {"decision": "BUY", "order_type": "LIMIT", "urgency": "NORMAL", "target_px": 200.6, "invalidation_px": 196.0},
        decision="BUY",
        symbol="TAOUSD",
        venue="kraken",
        price=200.0,  # +0.30% < 0.64% (2x 32 bps)
    )
    assert ok_ev_bad is False and "ai_negative_ev_target" in rsn_ev_bad
    assert m_ev_bad["min_target_ev_pct"] == pytest.approx(0.64)

    # Valid Kraken Crypto target (+1.50% >= 0.64%) passes
    ok_ev_good, _, _ = evaluate_ai_target_ev_guard(
        {"decision": "BUY", "order_type": "LIMIT", "urgency": "NORMAL", "target_px": 203.0, "invalidation_px": 196.0},
        decision="BUY",
        symbol="TAOUSD",
        venue="kraken",
        price=200.0,
    )
    assert ok_ev_good is True

    # Urgency=HIGH still enforces 4B-1 on BUY entries (no breakout bypass), while defensive SELL exits pass
    ok_ev_high_buy, rsn_ev_high_buy, _ = evaluate_ai_target_ev_guard(
        {"decision": "BUY", "order_type": "LIMIT", "urgency": "HIGH", "target_px": 200.4},
        decision="BUY",
        symbol="TAOUSD",
        venue="kraken",
        price=200.0,
    )
    assert ok_ev_high_buy is False and "ai_negative_ev_target" in rsn_ev_high_buy
    ok_ev_high_sell, _, _ = evaluate_ai_target_ev_guard(
        {"decision": "SELL", "order_type": "LIMIT", "urgency": "HIGH", "target_px": 199.6},
        decision="SELL",
        symbol="TAOUSD",
        venue="kraken",
        price=200.0,
    )
    assert ok_ev_high_sell is True
    ok_ev_notgt, _, _ = evaluate_ai_target_ev_guard(
        {"decision": "SELL", "order_type": "LIMIT", "urgency": "NORMAL"},
        decision="SELL",
        symbol="TAOUSD",
        venue="kraken",
        price=200.0,
    )
    assert ok_ev_notgt is True

    # Hyperliquid LIMIT (+0.30% >= 0.16%) passes while xStock below 0.10% floor blocks
    ok_ev_hl, _, _ = evaluate_ai_target_ev_guard(
        {"decision": "BUY", "order_type": "LIMIT", "urgency": "NORMAL", "target_px": 50.15},
        decision="BUY",
        symbol="HYPEUSDC",
        venue="hyperliquid",
        price=50.0,
    )
    assert ok_ev_hl is True
    ok_ev_xs_bad, _, _ = evaluate_ai_target_ev_guard(
        {"decision": "BUY", "order_type": "LIMIT", "urgency": "NORMAL", "target_px": 10.005},
        decision="BUY",
        symbol="SPCXxUSD",
        venue="kraken",
        price=10.0,
    )
    assert ok_ev_xs_bad is False

    # 4. End-to-end gateway enforcement: blocks negative-EV target_px without placing/canceling orders
    (tmp_path / "order_guard.conf").write_text(
        "autonomous_trading_mode = enforce\n"
        "autonomous_trading_min_notional_usd = 10.0\n"
        "autonomous_trading_max_notional_usd = 2000.0\n"
        "autonomous_trading_min_confidence = 0.85\n"
        "autonomous_trading_notify = 0\n"
        "autonomous_trading_target_ev_fee_mult = 2.0\n",
        encoding="utf-8",
    )
    gw = AutonomousIntentGateway(workspace_dir=str(tmp_path))
    tao_inst = MagicMock(symbol="TAOUSD", provider_name="kraken", base="TAO", quote="USD", params={})
    tao_inst._provider.name = "kraken"
    tao_inst._provider.get_current_price.return_value = 200.0
    tao_inst._provider.free_balance.return_value = 2000.0
    tao_inst.open_orders.return_value = []
    tao_inst.place.return_value = {"orderId": "kr_ok_1", "status": "NEW", "executedQty": "0.0"}

    rec_bad = gw.submit(
        {
            "decision": "BUY",
            "symbol": "TAOUSD",
            "venue": "kraken",
            "order_type": "LIMIT",
            "limit_price": 200.0,
            "target_px": 200.5,  # +0.25% < 0.64% Kraken LIMIT RT EV floor
            "invalidation_px": 195.0,
            "confidence": 0.92,
            "suggested_notional_usd": 300.0,
            "urgency": "NORMAL",
        },
        source="auto_cryptotrade_executor",
        max_mode="enforce",
        instruments_map={"TAOUSD": tao_inst},
    )
    assert rec_bad["status"] == "REJECTED_AI_CIRCUIT_BREAKER"
    assert "ai_negative_ev_target" in rec_bad["circuit_breaker_reason"]
    tao_inst.place.assert_not_called()

    rec_ok = gw.submit(
        {
            "decision": "BUY",
            "symbol": "TAOUSD",
            "venue": "kraken",
            "order_type": "LIMIT",
            "limit_price": 200.0,
            "target_px": 204.0,  # +2.00% >= 0.64%
            "invalidation_px": 195.0,
            "confidence": 0.92,
            "suggested_notional_usd": 300.0,
            "urgency": "NORMAL",
        },
        source="auto_cryptotrade_executor",
        max_mode="enforce",
        instruments_map={"TAOUSD": tao_inst},
    )
    assert rec_ok["status"] == "EXECUTED"
    tao_inst.place.assert_called_once()


def test_passive_limit_fill_reports_zero_limit_slippage_and_separate_arrival_shortfall():
    """Verify passive LIMIT fills measure slippage_bps vs limit_price while preserving arrival_shortfall_bps vs reference_price."""
    from providers.intent_telemetry import build_last_cycle_execution_status, summarize_recent_ai_intents

    now_s = 1_795_000_000.0
    intents = [
        {
            "intent_id": "lim_sell_arb_1",
            "timestamp": now_s - 4200.0,
            "decision": "SELL",
            "symbol": "ARBUSDC",
            "venue": "binance",
            "order_type": "LIMIT",
            "limit_price": 0.1201,
            "reference_price": 0.1194,
            "suggested_notional_usd": 240.2,
            "status": "EXECUTED",
            "execution_payload": {
                "orderId": "ARB_LIM_1",
                "status": "FILLED",
                "price": "0.1201",
                "avgPrice": "0.1201",
                "origQty": "2000.0",
                "executedQty": "2000.0",
                "filled_ts": now_s - 30.0,
            },
        }
    ]
    st = build_last_cycle_execution_status(intents, "crypto", now_s, lambda d, default=None: d.get("suggested_notional_usd", default))
    assert isinstance(st, dict)
    assert st["status"] == "FILLED"
    assert st["fill_px"] == pytest.approx(0.1201)
    assert st["limit_px"] == pytest.approx(0.1201)
    assert st["slippage_bps"] == 0.0
    assert st["arrival_shortfall_bps"] == pytest.approx(-58.63, abs=0.05)

    summ = summarize_recent_ai_intents(intents, "crypto", now_s, lambda d, default=None: d.get("suggested_notional_usd", default))
    assert len(summ) == 1
    assert summ[0]["slippage_bps"] == 0.0
    assert summ[0]["arrival_shortfall_bps"] == pytest.approx(-58.63, abs=0.05)


def test_etapa0_gateway_survival_guards_use_persisted_live_venue_fees(tmp_path):
    """Verify Q3: AutonomousIntentGateway feeds live persisted venue fees into 4B-1 EV and anti-ping-pong guards."""
    now_s = time.time()
    (tmp_path / "order_guard.conf").write_text(
        "autonomous_trading_mode = enforce\n"
        "autonomous_trading_min_notional_usd = 10.0\n"
        "autonomous_trading_max_notional_usd = 2000.0\n"
        "autonomous_trading_min_confidence = 0.85\n"
        "autonomous_trading_notify = 0\n"
        "autonomous_trading_target_ev_fee_mult = 2.0\n"
        "autonomous_trading_flip_cooldown_min = 60.0\n"
        "autonomous_trading_flip_min_edge_pct = 0.60\n",
        encoding="utf-8",
    )
    fee_state_path = tmp_path / "cachedb" / "state" / "venue_fee_maintenance_state.json"
    fee_state_path.parent.mkdir(parents=True, exist_ok=True)
    fee_payload = {
        "schema_version": 1,
        "ts": now_s - 60.0,
        "interval_sec": 14400.0,
        "state_max_age_sec": 86400.0,
        "venues": {
            "kraken": {
                "ts": now_s - 60.0,
                "fee_spec": {"fee_tier": "LIVE_VOL", "maker_fee_bps": 22.0, "taker_fee_bps": 38.0},
            },
            "binance": {
                "ts": now_s - 60.0,
                "fee_spec": {"fee_tier": "VIP0", "maker_fee_bps": 10.0, "taker_fee_bps": 10.0},
                "by_symbol": {
                    "BTCUSDC": {"fee_tier": "VIP0_BNB", "maker_fee_bps": 7.5, "taker_fee_bps": 7.12},
                },
            },
        },
    }
    fee_state_path.write_text(json.dumps(fee_payload), encoding="utf-8")

    gw = AutonomousIntentGateway(workspace_dir=str(tmp_path))

    tao_inst = MagicMock(symbol="TAOUSD", provider_name="kraken", base="TAO", quote="USD", params={})
    tao_inst._provider.name = "kraken"
    tao_inst._provider.get_current_price.return_value = 200.0
    tao_inst._provider.free_balance.return_value = 2000.0
    tao_inst.open_orders.return_value = []
    tao_inst.place.return_value = {"orderId": "kr_ok_1", "status": "NEW", "executedQty": "0.0"}

    btc_inst = MagicMock(symbol="BTCUSDC", provider_name="binance", base="BTC", quote="USDC", params={})
    btc_inst._provider.name = "binance"
    btc_inst._provider.get_current_price.return_value = 62000.0
    btc_inst._provider.free_balance.return_value = 5000.0
    btc_inst.open_orders.return_value = []
    btc_inst.place.return_value = {"orderId": "bn_ok_1", "status": "NEW", "executedQty": "0.0"}

    # 1. Kraken live maker=22bps (44bps RT -> 0.88% EV floor) blocks +0.75% target that static 32bps (0.64%) would allow
    rec_kr_block = gw.submit(
        {
            "decision": "BUY",
            "symbol": "TAOUSD",
            "venue": "kraken",
            "order_type": "LIMIT",
            "limit_price": 200.0,
            "target_px": 201.5,  # +0.75% < 0.88%
            "invalidation_px": 195.0,
            "confidence": 0.92,
            "suggested_notional_usd": 300.0,
            "urgency": "NORMAL",
        },
        source="auto_cryptotrade_executor",
        max_mode="enforce",
        instruments_map={"TAOUSD": tao_inst},
    )
    assert rec_kr_block["status"] == "REJECTED_AI_CIRCUIT_BREAKER"
    assert "ai_negative_ev_target" in rec_kr_block["circuit_breaker_reason"]
    assert "rt_fee=44bps" in rec_kr_block["circuit_breaker_reason"]
    tao_inst.place.assert_not_called()

    # 2. Binance per-symbol BTCUSDC live maker=7.5bps (15bps RT -> 0.30% EV floor) allows +0.35% target (static 20bps -> 0.40%)
    rec_bn_ok = gw.submit(
        {
            "decision": "BUY",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "order_type": "LIMIT",
            "limit_price": 62000.0,
            "target_px": 62217.0,  # +0.35% >= 0.30%
            "invalidation_px": 61800.0,
            "confidence": 0.92,
            "suggested_notional_usd": 300.0,
            "urgency": "NORMAL",
        },
        source="auto_cryptotrade_executor",
        max_mode="enforce",
        instruments_map={"BTCUSDC": btc_inst},
    )
    assert rec_bn_ok["status"] == "EXECUTED"
    btc_inst.place.assert_called_once()

    # 3. Anti-ping-pong guard uses live Kraken 44bps RT -> flip min edge 1.08% (blocks +0.90% flip that static 0.84% would pass)
    gw.save_all_intents([
        {
            "intent_id": "prior_kr_buy_1",
            "timestamp": now_s - 600.0,
            "decision": "BUY",
            "symbol": "TAOUSD",
            "venue": "kraken",
            "order_type": "LIMIT",
            "limit_price": 200.0,
            "suggested_notional_usd": 300.0,
            "status": "EXECUTED",
            "execution_payload": {
                "orderId": "kr_buy_1",
                "status": "FILLED",
                "executedQty": "1.5",
                "avgPrice": "200.0",
                "filled_ts": now_s - 500.0,
            },
        }
    ])
    tao_inst._provider.get_current_price.return_value = 201.8
    rec_pp_block = gw.submit(
        {
            "decision": "SELL",
            "symbol": "TAOUSD",
            "venue": "kraken",
            "order_type": "LIMIT",
            "limit_price": 201.8,  # +0.90% < live 1.08% flip edge (static 32bps would be 0.84%)
            "confidence": 0.92,
            "suggested_notional_usd": 300.0,
            "urgency": "NORMAL",
        },
        source="auto_cryptotrade_executor",
        max_mode="enforce",
        instruments_map={"TAOUSD": tao_inst},
    )
    assert rec_pp_block["status"] == "REJECTED_AI_CIRCUIT_BREAKER"
    assert "ai_ping_pong" in rec_pp_block["circuit_breaker_reason"]
    tao_inst.place.assert_not_called()

    # 4. Stale fee state (>state_max_age_sec) falls back cleanly to static defaults (Kraken 32bps -> 0.64% allows +0.75%)
    gw.save_all_intents([])
    fee_payload["ts"] = now_s - 10 * 86400.0
    fee_payload["venues"]["kraken"]["ts"] = now_s - 10 * 86400.0
    fee_state_path.write_text(json.dumps(fee_payload), encoding="utf-8")
    tao_inst._provider.get_current_price.return_value = 200.0
    rec_stale_fallback = gw.submit(
        {
            "decision": "BUY",
            "symbol": "TAOUSD",
            "venue": "kraken",
            "order_type": "LIMIT",
            "limit_price": 200.0,
            "target_px": 201.5,  # +0.75% >= static 0.64%
            "invalidation_px": 198.0,
            "confidence": 0.92,
            "suggested_notional_usd": 300.0,
            "urgency": "NORMAL",
        },
        source="auto_cryptotrade_executor",
        max_mode="enforce",
        instruments_map={"TAOUSD": tao_inst},
    )
    assert rec_stale_fallback["status"] == "EXECUTED"
    tao_inst.place.assert_called_once()


def test_ping_pong_guard_allows_same_side_continuation_after_already_completed_flip():
    """Verify BUY(t=0) -> SELL(t=10m, valid flip) -> second same-side SELL(t=20m) is NOT falsely blocked as BUY->SELL flip."""
    from providers.ai_survival_guard import evaluate_ai_ping_pong_guard

    now_s = 1_795_000_000.0
    prior_intents = [
        {
            "intent_id": "b1",
            "timestamp": now_s - 1200.0,  # 20m ago
            "decision": "BUY",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "status": "EXECUTED",
            "limit_price": 70000.0,
            "execution_payload": {"orderId": "o_b1", "status": "FILLED", "executedQty": "0.01", "avg_fill_price": 70000.0},
        },
        {
            "intent_id": "s1",
            "timestamp": now_s - 600.0,  # 10m ago (valid profit-take flip)
            "decision": "SELL",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "status": "EXECUTED",
            "limit_price": 71000.0,
            "execution_payload": {"orderId": "o_s1", "status": "FILLED", "executedQty": "0.005", "avg_fill_price": 71000.0},
        },
    ]
    # Second same-side SELL at $70,200 (fav vs older BUY is only +0.29% < +0.60%, but latest side is already SELL!)
    allowed, reason, _ = evaluate_ai_ping_pong_guard(
        {"intent_id": "s2", "decision": "SELL", "symbol": "BTCUSDC", "venue": "binance", "limit_price": 70200.0},
        prior_intents,
        decision="SELL",
        symbol="BTCUSDC",
        venue="binance",
        price=70200.0,
        now_ts=now_s,
    )
    assert allowed is True, f"Same-side continuation SELL was falsely blocked: {reason}"


def test_pre_cancel_submit_cache_gate_and_rollback_stale_cache_retry(temp_workspace, monkeypatch):
    """Verify replace_open_orders never cancels live resting orders when account_cache_not_fresh, and atomic rollback retries transient stale cache."""
    import order_lifecycle as ol_mod
    import providers.intent_execution as ie_mod

    monkeypatch.setattr(ie_mod, "get_stale_cache_retry_delay_sec", lambda _d=None: 0.0)
    ws, _ = temp_workspace
    gw = AutonomousIntentGateway(workspace_dir=ws, cache_dir="cachedb", mode="enforce", notify_enabled=False)

    canceled_calls: list[str] = []
    place_calls: list[dict] = []
    fake_prov = SimpleNamespace(
        name="binance",
        get_current_price=lambda _s: 188.0,
        free_balance=lambda _a: 3.15,
        prepare_order_state=lambda: None,
    )
    tao_inst = SimpleNamespace(
        symbol="TAOUSDC",
        base="TAO",
        quote="USDC",
        provider_name="binance",
        _provider=fake_prov,
        _provider_bool_cap=lambda cap, **_kw: cap == "requires_cache_permit",
        open_orders=lambda: [
            {
                "orderId": "727100712",
                "clientOrderId": "AI_prior_tao_buy",
                "side": "BUY",
                "price": "188.80",
                "origQty": "3.1493",
                "executedQty": "0.0",
            }
        ],
        cancel_order=lambda oid: canceled_calls.append(oid) or True,
        place=lambda **kw: place_calls.append(kw) or {"orderId": "new_1"},
    )

    # BUY with replace_open_orders=True must fail closed before canceling prior live BUY orders
    res = gw.submit(
        {
            "decision": "BUY",
            "symbol": "TAOUSDC",
            "venue": "binance",
            "order_type": "LIMIT",
            "limit_price": 188.40,
            "replace_open_orders": True,
            "confidence": 0.94,
            "suggested_notional_usd": 593.0,
        },
        source="auto_cryptotrade_executor",
        max_mode="enforce",
        instruments_map={"TAOUSDC": tao_inst},
    )
    assert res["status"] == "FAILED"
    assert res["history"][-1]["reason"] == "account_cache_not_fresh"
    assert canceled_calls == []
    assert place_calls == []

    # SELL exits fail OPEN on stale account cache so protective exits are never trapped
    assert ie_mod.ensure_submit_account_cache_ready(tao_inst, side="SELL") == (True, "")

    # Atomic rollback retries transient account_cache_not_fresh and restores the canceled order
    rb_attempts = []

    def _flaky_rb_place(**kw):
        rb_attempts.append(kw)
        if len(rb_attempts) < 3:
            if isinstance(ctx := kw.get("_outcome_context"), dict):
                ctx["state"] = "refused"
                ctx["reason"] = "account_cache_not_fresh"
            return None
        return {"orderId": "restored_tao_727", "status": "NEW"}

    rb_inst = SimpleNamespace(place=_flaky_rb_place)
    restored_out: list[dict] = []
    cnt = ol_mod.restore_cancelled_instrument_orders(
        rb_inst,
        [{"orderId": "727100712", "clientOrderId": "AI_prior_tao_sell", "side": "SELL", "price": "188.80", "origQty": "3.1493"}],
        bypass_guards={"all"},
        reason="test_stale_cache_rollback",
        restored_orders_out=restored_out,
        retry_delay_sec=0.0,
        sleep_fn=lambda _d: None,
    )
    assert cnt == 1
    assert len(rb_attempts) == 3
    assert restored_out[0]["order_id"] == "restored_tao_727"
    assert restored_out[0]["actor"] == "AI"


def test_tokenized_equity_longterm_sell_survival_guard_and_gateway(tmp_path):
    """Verify SELL-only tokenized equity (SPCXxUSD) long-term holding survival guard blocks premature/panic exits and allows high-conviction multi-day profit exits."""
    from providers.ai_survival_guard import evaluate_ai_tokenized_equity_sell_guard

    now_s = 1_795_530_000.0
    cost_px = 134.844

    # 1. Blocks routine SELL at a loss or small profit (+4.9% < +6.0%), including HIGH urgency with shallow drop (< 8%)
    ok_loss_norm, rsn_loss_norm, _ = evaluate_ai_tokenized_equity_sell_guard(
        {"decision": "SELL", "urgency": "NORMAL", "target_px": 115.0},
        [],
        decision="SELL",
        symbol="SPCXxUSD",
        venue="kraken",
        price=130.0,
        confidence=0.96,
        now_ts=now_s,
        cost_basis_px=cost_px,
    )
    assert ok_loss_norm is False and "ai_xstock_longterm_min_profit" in rsn_loss_norm

    ok_loss_shallow, rsn_loss_shallow, _ = evaluate_ai_tokenized_equity_sell_guard(
        {"decision": "SELL", "urgency": "HIGH", "target_px": 122.0},  # -6.15% < 8.0% crash floor
        [],
        decision="SELL",
        symbol="SPCXxUSD",
        venue="kraken",
        price=130.0,
        confidence=0.96,
        now_ts=now_s,
        cost_basis_px=cost_px,
    )
    assert ok_loss_shallow is False and "ai_xstock_longterm_min_profit" in rsn_loss_shallow

    ok_small, rsn_small, _ = evaluate_ai_tokenized_equity_sell_guard(
        {"decision": "SELL", "urgency": "HIGH", "target_px": 131.0},  # -7.42% < 8.0% crash floor
        [],
        decision="SELL",
        symbol="SPCXxUSD",
        venue="kraken",
        price=141.5,  # +4.9% < +6.0%
        confidence=0.95,
        now_ts=now_s,
        cost_basis_px=cost_px,
    )
    assert ok_small is False and "ai_xstock_longterm_min_profit" in rsn_small

    # 1b. Allows imminent-crash capital-preservation SELL even at a loss ONLY when urgency=HIGH, conf>=0.92, target drop>=8%, and cooldown clear
    ok_crash_lowconf, rsn_crash_lowconf, _ = evaluate_ai_tokenized_equity_sell_guard(
        {"decision": "SELL", "urgency": "HIGH", "target_px": 115.0},  # -11.54% >= 8.0%
        [],
        decision="SELL",
        symbol="SPCXxUSD",
        venue="kraken",
        price=130.0,
        confidence=0.90,  # < 0.92 crash floor
        now_ts=now_s,
        cost_basis_px=cost_px,
    )
    assert ok_crash_lowconf is False and "ai_xstock_longterm_min_profit" in rsn_crash_lowconf

    ok_crash, rsn_crash, m_crash = evaluate_ai_tokenized_equity_sell_guard(
        {"decision": "SELL", "urgency": "HIGH", "target_px": 115.0},  # -11.54% >= 8.0%
        [],
        decision="SELL",
        symbol="SPCXxUSD",
        venue="kraken",
        price=130.0,
        confidence=0.94,
        now_ts=now_s,
        cost_basis_px=cost_px,
    )
    assert ok_crash is True and rsn_crash == ""
    assert m_crash.get("xstock_crash_protection_exit") is True
    assert m_crash["xstock_target_drop_pct"] >= 8.0

    # 2. Fail-closed when active cost basis is unavailable
    ok_nocost, rsn_nocost, _ = evaluate_ai_tokenized_equity_sell_guard(
        {"decision": "SELL", "urgency": "NORMAL", "target_px": 140.0},
        [],
        decision="SELL",
        symbol="SPCXxUSD",
        venue="kraken",
        price=160.0,
        confidence=0.95,
        now_ts=now_s,
        cost_basis_px=None,
    )
    assert ok_nocost is False and "ai_xstock_longterm_missing_cost_basis" in rsn_nocost

    # 3. Blocks when profit >= +6% but target_px is missing or drop < 4% (insufficient to clear heavy manual rebuy commission)
    ok_notgt, rsn_notgt, _ = evaluate_ai_tokenized_equity_sell_guard(
        {"decision": "SELL", "urgency": "NORMAL"},
        [],
        decision="SELL",
        symbol="SPCXxUSD",
        venue="kraken",
        price=145.0,  # +7.53%
        confidence=0.95,
        now_ts=now_s,
        cost_basis_px=cost_px,
    )
    assert ok_notgt is False and "ai_xstock_longterm_missing_target_px" in rsn_notgt

    ok_shallow, rsn_shallow, _ = evaluate_ai_tokenized_equity_sell_guard(
        {"decision": "SELL", "urgency": "NORMAL", "target_px": 141.0},  # -2.76% < 4.0%
        [],
        decision="SELL",
        symbol="SPCXxUSD",
        venue="kraken",
        price=145.0,
        confidence=0.95,
        now_ts=now_s,
        cost_basis_px=cost_px,
    )
    assert ok_shallow is False and "ai_xstock_longterm_min_target_drop" in rsn_shallow

    # 4. Blocks low confidence (< 0.90) and repeat SELL within 24h cooldown
    ok_conf, rsn_conf, _ = evaluate_ai_tokenized_equity_sell_guard(
        {"decision": "SELL", "urgency": "NORMAL", "target_px": 138.0},
        [],
        decision="SELL",
        symbol="SPCXxUSD",
        venue="kraken",
        price=145.0,
        confidence=0.87,
        now_ts=now_s,
        cost_basis_px=cost_px,
    )
    assert ok_conf is False and "ai_xstock_longterm_low_confidence" in rsn_conf

    prior_sell = [{
        "intent_id": "xs_prev_sell",
        "timestamp": now_s - 6 * 3600.0,
        "decision": "SELL",
        "symbol": "SPCXxUSD",
        "venue": "kraken",
        "order_type": "LIMIT",
        "status": "EXECUTED",
        "execution_payload": {"orderId": "ok1", "status": "FILLED", "executedQty": "5.0", "avg_fill_price": 144.5},
    }]
    ok_cd, rsn_cd, _ = evaluate_ai_tokenized_equity_sell_guard(
        {"intent_id": "xs_new_sell", "decision": "SELL", "urgency": "NORMAL", "target_px": 138.0},
        prior_sell,
        decision="SELL",
        symbol="SPCXxUSD",
        venue="kraken",
        price=145.0,
        confidence=0.92,
        now_ts=now_s,
        cost_basis_px=cost_px,
    )
    assert ok_cd is False and "ai_xstock_longterm_sell_cooldown" in rsn_cd

    # 5. Allows SELL when profit > +6%, target drop >= 4%, confidence >= 0.90, and cooldown clear
    ok_pass, rsn_pass, m_pass = evaluate_ai_tokenized_equity_sell_guard(
        {"decision": "SELL", "urgency": "NORMAL", "target_px": 138.0},  # -4.83% drop
        [],
        decision="SELL",
        symbol="SPCXxUSD",
        venue="kraken",
        price=145.0,  # +7.53% profit (> 6%)
        confidence=0.91,
        now_ts=now_s,
        cost_basis_px=cost_px,
    )
    assert ok_pass is True and rsn_pass == ""
    assert m_pass["xstock_profit_pct"] > 6.0
    assert m_pass["xstock_target_drop_pct"] > 4.0

    # 6. Never affects regular crypto or RTH equities
    ok_btc, _, _ = evaluate_ai_tokenized_equity_sell_guard(
        {"decision": "SELL", "urgency": "NORMAL"},
        [],
        decision="SELL",
        symbol="BTCUSDC",
        venue="binance",
        price=60000.0,
        confidence=0.86,
        now_ts=now_s,
        cost_basis_px=70000.0,
    )
    assert ok_btc is True

    # 7. End-to-end AutonomousIntentGateway blocks premature SPCXxUSD SELL using kraken/.state_SPCXxUSD.json
    krk_dir = tmp_path / "kraken"
    krk_dir.mkdir(parents=True, exist_ok=True)
    (krk_dir / ".state_SPCXxUSD.json").write_text(
        json.dumps({"qty": 24.47, "cost": 3300.0, "entry_price": 134.844}),
        encoding="utf-8",
    )
    (tmp_path / "order_guard.conf").write_text(
        "autonomous_trading_mode = enforce\n"
        "autonomous_trading_min_notional_usd = 10.0\n"
        "autonomous_trading_max_notional_usd = 2000.0\n"
        "autonomous_trading_min_confidence = 0.85\n"
        "autonomous_trading_notify = 0\n"
        "autonomous_trading_xstock_sell_min_profit_pct = 6.0\n"
        "autonomous_trading_xstock_sell_min_target_drop_pct = 4.0\n"
        "autonomous_trading_xstock_sell_min_confidence = 0.90\n"
        "autonomous_trading_xstock_sell_cooldown_hours = 24.0\n",
        encoding="utf-8",
    )
    gw = AutonomousIntentGateway(workspace_dir=str(tmp_path))
    place_calls = []
    spcx_inst = MagicMock(symbol="SPCXxUSD", provider_name="kraken", base="SPCXx", quote="USD", params={})
    spcx_inst._provider.name = "kraken"
    spcx_inst._provider.get_current_price.return_value = 140.0
    spcx_inst._provider.free_balance.return_value = 24.47
    spcx_inst.open_orders.return_value = []
    spcx_inst.place.side_effect = lambda **kw: place_calls.append(kw) or {"orderId": "xs_ok_1", "status": "NEW"}

    res_blk = gw.submit(
        {
            "decision": "SELL",
            "symbol": "SPCXxUSD",
            "venue": "kraken",
            "order_type": "LIMIT",
            "limit_price": 140.0,  # +3.8% < +12%
            "target_px": 130.0,
            "confidence": 0.95,
            "urgency": "HIGH",
            "suggested_notional_usd": 1000.0,
        },
        source="auto_cryptotrade_executor",
        max_mode="enforce",
        instruments_map={"SPCXxUSD": spcx_inst},
    )
    assert res_blk["status"] == "REJECTED_AI_CIRCUIT_BREAKER"
    assert "ai_xstock_longterm_min_profit" in str(res_blk["history"][-1].get("reason") or "")
    assert place_calls == []

    # 8. End-to-end AutonomousIntentGateway permits imminent-crash SELL below cost basis on urgency=HIGH + conf>=0.92 + drop>=8%
    spcx_inst._provider.get_current_price.return_value = 127.0
    res_crash = gw.submit(
        {
            "decision": "SELL",
            "symbol": "SPCXxUSD",
            "venue": "kraken",
            "order_type": "LIMIT",
            "limit_price": 127.0,  # -5.8% loss, but imminent crash target_px=114 (-10.2% >= 8.0%)
            "target_px": 114.0,
            "confidence": 0.94,
            "urgency": "HIGH",
            "suggested_notional_usd": 1000.0,
        },
        source="auto_cryptotrade_executor",
        max_mode="enforce",
        instruments_map={"SPCXxUSD": spcx_inst},
    )
    assert res_crash["status"] == "EXECUTED"
    assert len(place_calls) == 1


def test_gateway_stale_cache_abort_rolls_back_prior_canceled_orders(temp_workspace):
    """If Step-2 balance-unlock aborts on stale account cache, already-canceled Step-1 AI orders must be restored."""
    gw = AutonomousIntentGateway(workspace_dir=str(temp_workspace))
    inst = MagicMock()
    inst.symbol = "BTCUSDC"
    inst.provider_name = "binance"
    intent = {"intent_id": "intent_stale_rollback", "symbol": "BTCUSDC", "venue": "binance", "history": []}
    prior_snaps = [
        {
            "order_id": "prior-ai-1",
            "client_order_id": "AI_intent_old",
            "side": "BUY",
            "price": 69000.0,
            "qty": 0.01,
            "actor": "AI",
        }
    ]
    rollback_calls = []
    with (
        patch(
            "orchestratorTrade.autonomous_intent_gateway.ensure_submit_account_cache_ready",
            return_value=(False, "account_cache_stale:binance"),
        ),
        patch.object(
            gw,
            "_rollback_canceled_orders",
            side_effect=lambda i, it, snaps, **kw: rollback_calls.append((list(snaps), kw.get("reason"))),
        ),
    ):
        ok, cids = gw._cancel_prior_orders_or_abort(
            inst,
            intent,
            intent_id="intent_stale_rollback",
            decision="BUY",
            now_ts=1_795_500_000.0,
            canceled_order_snapshots=prior_snaps,
        )
    assert ok is False and cids == []
    assert intent["status"] == "FAILED"
    assert len(rollback_calls) == 1
    assert rollback_calls[0][0][0]["order_id"] == "prior-ai-1"
    assert prior_snaps == []


def test_circuit_breaker_and_reconciliation_handle_compound_terminal_and_orphan_records(tmp_path):
    """Circuit breaker ignores compound non-fill zero-fill statuses; reconciliation correlates iid=None orphan rows."""
    from providers.ai_circuit_breaker import evaluate_ai_intent_circuit_breaker
    from providers.order_id_context import derive_ai_intent_client_order_id, derive_venue_128bit_client_order_id
    from providers.order_ledger import OrderEventType, OrderLedgerWriter

    now_s = 1_795_500_000.0
    # 1. Compound terminal non-fill status (MINTRADENTLREJECTED) with 0 fill must NOT count toward daily buy cap
    rejected_intents = [
        {
            "intent_id": f"intent_rej_{i}",
            "timestamp": now_s - 60.0 * (i + 1),
            "symbol": "HYPEUSDC",
            "venue": "hyperliquid",
            "decision": "BUY",
            "status": "EXECUTED",
            "suggested_notional_usd": 500.0,
            "execution_payload": {"orderId": f"hl_{i}", "status": "MINTRADENTLREJECTED", "executedQty": "0"},
        }
        for i in range(5)
    ]
    ok_cb, _adj_usd, reason_cb, metrics_cb = evaluate_ai_intent_circuit_breaker(
        {"intent_id": "intent_cur", "symbol": "HYPEUSDC", "venue": "hyperliquid", "decision": "BUY"},
        rejected_intents,
        symbol="HYPEUSDC",
        venue="hyperliquid",
        decision="BUY",
        effective_usd=500.0,
        min_notional_usd=10.0,
        price=50.0,
        now_ts=now_s,
        max_daily_buys_per_symbol=2,
        max_daily_buy_notional_usd=1000.0,
    )
    assert ok_cb is True, f"unexpected block: {reason_cb}"
    assert metrics_cb["daily_sym_buys"] == 0

    # 2. Orphan-adopted ledger record with iid=None reconciles via 128-bit Hyperliquid cloid AND via order_id
    cdir = tmp_path / "cachedb"
    audit_dir = tmp_path / "logs" / "execution_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    writer = OrderLedgerWriter(cachedb_dir=str(cdir))
    iid_cloid = "intent_1795500000_hl01"
    hl_cloid = derive_venue_128bit_client_order_id("hyperliquid", derive_ai_intent_client_order_id(iid_cloid))
    writer.record_event(
        OrderEventType.RECONCILED_FILLED.value,
        venue="hyperliquid",
        symbol="HYPEUSDC",
        ts=now_s - 10.0,
        owner="ai_executor",
        side="BUY",
        qty=2.0,
        fq=2.0,
        px=50.0,
        cid=hl_cloid,
        vid="777111",
        vst="FILLED",
        iid=None,
    )
    iid_oid = "intent_1795500000_bn02"
    writer.record_event(
        OrderEventType.STATUS.value,
        venue="hyperliquid",
        symbol="HYPEUSDC",
        ts=now_s - 5.0,
        owner="ai_executor",
        side="BUY",
        qty=1.0,
        px=50.0,
        vid="777222",
        vst="MINTRADENTLREJECTED",
        iid=None,
    )
    intents = [
        {
            "intent_id": iid_cloid,
            "timestamp": now_s - 30.0,
            "venue": "hyperliquid",
            "symbol": "HYPEUSDC",
            "decision": "BUY",
            "status": "SUBMIT_UNKNOWN",
            "history": [],
        },
        {
            "intent_id": iid_oid,
            "timestamp": now_s - 20.0,
            "venue": "hyperliquid",
            "symbol": "HYPEUSDC",
            "decision": "BUY",
            "status": "EXECUTED",
            "execution_payload": {"orderId": "777222", "status": "NEW"},
            "history": [],
        },
    ]
    changed = reconcile_intents_from_execution_audit(intents, str(audit_dir), cachedb_dir=str(cdir))
    assert changed is True
    assert intents[0]["status"] == "EXECUTED"
    assert intents[0]["execution_payload"]["status"] == "FILLED"
    assert intents[1]["execution_payload"]["status"] == "MINTRADENTLREJECTED"
    assert intents[1]["execution_payload"].get("reconciled_terminal") is True


def test_min_target_rr_guard_explicit_cancel_decision_horizon_and_prompt_parity(temp_workspace):
    """Verify:
    1. evaluate_ai_target_ev_guard + gateway block poor R:R plans (e.g. chased REPLACE dropping BTC R:R from 0.99 -> 0.24)
       on urgency=NORMAL while allowing healthy R:R (>= 0.50) and urgency=HIGH exits.
    2. Explicit decision='CANCEL' with cancel_order_ids cancels listed AI or BOT orders directly without replace_open_orders=True,
       while rejecting CANCEL without cancel_order_ids or below min_confidence.
    3. Optional horizon_h parses numeric & compact unit strings ('12h', '2d', '90m') and persists on emitted intents.
    4. Full & compact continuation prompts stay aligned on LIMIT-only execution, AI survival breakers, R:R replace rule, and CANCEL.
    """
    from botcore import topic_spec
    from intelligence.macro_sentiment_ssot import TelemetrySensorSnapshot
    from orchestratorOS.lib.technical_repair_ssot import LogEventSummary
    from orchestratorOS.supervisor.prompt_formatters import build_supervisor_prompt
    from providers.ai_survival_guard import evaluate_ai_target_ev_guard
    from providers.intent_state import extract_intent_horizon_hours, normalize_trading_decisions

    # 1. Pure R:R check on BTC chase-replace example ($59,680 -> $60,397.68 with same target $60,650 / inv $58,700)
    ok_orig, rsn_orig, m_orig = evaluate_ai_target_ev_guard(
        {"decision": "BUY", "order_type": "LIMIT", "urgency": "NORMAL", "target_px": 60650.0, "invalidation_px": 58700.0},
        decision="BUY",
        symbol="BTCUSDC",
        venue="binance",
        price=59680.0,
        target_ev_fee_mult=2.0,
        min_target_rr=0.50,
    )
    assert ok_orig is True and rsn_orig == ""
    assert m_orig["reward_risk_ratio"] == pytest.approx(0.99, abs=0.02)

    ok_chased, rsn_chased, m_chased = evaluate_ai_target_ev_guard(
        {"decision": "BUY", "order_type": "LIMIT", "urgency": "NORMAL", "target_px": 60650.0, "invalidation_px": 58700.0},
        decision="BUY",
        symbol="BTCUSDC",
        venue="binance",
        price=60397.68,
        target_ev_fee_mult=2.0,
        min_target_rr=0.50,
    )
    assert ok_chased is False
    assert "ai_poor_rr_plan" in rsn_chased
    assert m_chased["reward_risk_ratio"] < 0.25

    # BUY with urgency=HIGH still enforces EV and R:R floors (momentum/breakout entries cannot bypass risk checks)
    ok_high_buy_rr, rsn_high_buy_rr, m_high_buy_rr = evaluate_ai_target_ev_guard(
        {"decision": "BUY", "order_type": "LIMIT", "urgency": "HIGH", "target_px": 60850.0, "invalidation_px": 58700.0},
        decision="BUY",
        symbol="BTCUSDC",
        venue="binance",
        price=60577.10,
        target_ev_fee_mult=2.0,
        min_target_rr=0.50,
    )
    assert ok_high_buy_rr is False
    assert "ai_poor_rr_plan" in rsn_high_buy_rr
    assert m_high_buy_rr["reward_risk_ratio"] < 0.20

    # urgency=HIGH bypasses R:R floor strictly for defensive SELL exits
    ok_high, _, _ = evaluate_ai_target_ev_guard(
        {"decision": "SELL", "order_type": "LIMIT", "urgency": "HIGH", "target_px": 60000.0, "invalidation_px": 62000.0},
        decision="SELL",
        symbol="BTCUSDC",
        venue="binance",
        price=60300.0,
        target_ev_fee_mult=2.0,
        min_target_rr=0.50,
    )
    assert ok_high is True

    # 2. Gateway enforcement for R:R, horizon_h persistence, and explicit CANCEL for AI + BOT orders
    ws, _ = temp_workspace
    gw = AutonomousIntentGateway(workspace_dir=ws, cache_dir="cachedb", mode="enforce", notify_enabled=False)
    btc_inst = _make_mock_inst("BTCUSDC", venue="binance", price=60500.0, free_balance={"USDC": 5000.0, "BTC": 0.1})
    btc_inst.open_orders.return_value = [
        {"orderId": "ai_ord_1", "clientOrderId": "AI_intent_1795000001_buy_btcusdc", "side": "BUY", "price": "59680.0"},
        {"orderId": "bot_ord_2", "clientOrderId": "RT_BUY_2", "side": "BUY", "price": "59500.0"},
        {"orderId": "ai_keep_3", "clientOrderId": "AI_intent_1795000002_sell_btcusdc", "side": "SELL", "price": "62000.0"},
    ]

    with patch.object(gw, "_dispatch_notify", return_value=True):
        rec_poor_rr = gw.submit(
            {
                "decision": "BUY",
                "symbol": "BTCUSDC",
                "venue": "binance",
                "order_type": "LIMIT",
                "limit_price": 60397.68,
                "target_px": 60650.0,
                "invalidation_px": 58700.0,
                "horizon_h": "12h",
                "confidence": 0.90,
                "suggested_notional_usd": 500.0,
                "urgency": "NORMAL",
                "thesis": "Chased replace",
            },
            source="auto_cryptotrade_executor",
            max_mode="enforce",
            instruments_map={"BTCUSDC": btc_inst},
        )
    assert rec_poor_rr["status"] == "REJECTED_AI_CIRCUIT_BREAKER"
    assert "ai_poor_rr_plan" in rec_poor_rr["circuit_breaker_reason"]
    assert rec_poor_rr["horizon_h"] == pytest.approx(12.0)

    # Explicit CANCEL cancels both AI and BOT targeted IDs without requiring replace_open_orders=True
    # and returns structured per-ID confirmation without polluting autonomous_trade_intents.json
    intents_before = len(gw.load_all_intents())
    btc_inst.cancel_order.reset_mock()
    btc_inst.place.reset_mock()
    res_cancel = gw.submit(
        {
            "decision": "CANCEL",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "cancel_order_ids": ["ai_ord_1", "bot_ord_2"],
            "confidence": 0.90,
            "thesis": "Cancel stale ladder legs",
        },
        source="auto_cryptotrade_executor",
        max_mode="enforce",
        instruments_map={"BTCUSDC": btc_inst},
    )
    assert res_cancel is not None
    assert res_cancel["decision"] == "CANCEL"
    assert res_cancel["status"] == "CANCELED"
    assert res_cancel["requested_order_ids"] == ["ai_ord_1", "bot_ord_2"]
    assert res_cancel["canceled_order_ids"] == ["ai_ord_1", "bot_ord_2"]
    assert res_cancel["failed_order_ids"] == []
    assert len(gw.load_all_intents()) == intents_before
    assert [c.args[0] for c in btc_inst.cancel_order.call_args_list] == ["ai_ord_1", "bot_ord_2"]
    btc_inst.place.assert_not_called()

    # Partial and failed CANCEL outcomes propagate per-ID status through submit_supervisor_trading_decisions
    from orchestratorOS.lib.technical_repair_ssot import submit_supervisor_trading_decisions

    btc_inst.cancel_order.reset_mock()
    btc_inst.cancel_order.side_effect = lambda oid: False if oid == "bot_ord_2" else True
    btc_inst.open_orders.return_value = [
        {"orderId": "ai_ord_1", "clientOrderId": "AI_intent_1795000001_buy_btcusdc", "side": "BUY", "price": "59680.0"},
        {"orderId": "bot_ord_2", "clientOrderId": "RT_BUY_2", "side": "BUY", "price": "59500.0"},
    ]
    res_partial = gw.submit(
        {
            "decision": "CANCEL",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "cancel_order_ids": ["ai_ord_1", "bot_ord_2"],
            "confidence": 0.92,
        },
        source="auto_cryptotrade_executor",
        max_mode="enforce",
        instruments_map={"BTCUSDC": btc_inst},
    )
    assert res_partial["status"] == "CANCEL_PARTIAL"
    assert res_partial["canceled_order_ids"] == ["ai_ord_1"]
    assert res_partial["failed_order_ids"] == ["bot_ord_2"]

    btc_inst.cancel_order.reset_mock()
    res_failed = gw.submit(
        {
            "decision": "CANCEL",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "cancel_order_ids": ["bot_ord_2"],
            "confidence": 0.92,
        },
        source="auto_cryptotrade_executor",
        max_mode="enforce",
        instruments_map={"BTCUSDC": btc_inst},
    )
    assert res_failed["status"] == "CANCEL_FAILED"
    assert res_failed["canceled_order_ids"] == []
    assert res_failed["failed_order_ids"] == ["bot_ord_2"]

    res_shadow_cancel = gw.submit(
        {
            "decision": "CANCEL",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "cancel_order_ids": ["ai_ord_1"],
            "confidence": 0.92,
        },
        source="auto_cryptotrade_executor",
        max_mode="shadow",
        instruments_map={"BTCUSDC": btc_inst},
    )
    assert res_shadow_cancel["status"] == "OBSERVED_SHADOW"
    assert res_shadow_cancel["requested_order_ids"] == ["ai_ord_1"]

    with patch.object(gw, "submit", return_value=res_partial):
        _, _, emitted_rep, _, any_exec_rep = submit_supervisor_trading_decisions(
            {
                "market_trading_decisions": [
                    {
                        "decision": "CANCEL",
                        "symbol": "BTCUSDC",
                        "venue": "binance",
                        "cancel_order_ids": ["ai_ord_1", "bot_ord_2"],
                        "confidence": 0.92,
                    }
                ]
            },
            topic_spec("auto_cryptotrade_executor"),
            topic_key="auto_cryptotrade_executor",
            mode="enforce",
            now_ts=time.time(),
            intent_gateway=gw,
            is_rth_open_fn=lambda _: True,
        )
    assert len(emitted_rep) == 1 and emitted_rep[0]["status"] == "CANCEL_PARTIAL"
    assert any_exec_rep is True
    with patch.object(gw, "submit", return_value=res_failed):
        _, _, emitted_fail, _, any_exec_fail = submit_supervisor_trading_decisions(
            {
                "market_trading_decisions": [
                    {
                        "decision": "CANCEL",
                        "symbol": "BTCUSDC",
                        "venue": "binance",
                        "cancel_order_ids": ["bot_ord_2"],
                        "confidence": 0.92,
                    }
                ]
            },
            topic_spec("auto_cryptotrade_executor"),
            topic_key="auto_cryptotrade_executor",
            mode="enforce",
            now_ts=time.time(),
            intent_gateway=gw,
            is_rth_open_fn=lambda _: True,
        )
    assert len(emitted_fail) == 1 and emitted_fail[0]["status"] == "CANCEL_FAILED"
    assert any_exec_fail is False
    btc_inst.cancel_order.side_effect = None

    # CANCEL without explicit cancel_order_ids or low confidence is a safe no-op
    btc_inst.cancel_order.reset_mock()
    for bad_cancel in (
        {"decision": "CANCEL", "symbol": "BTCUSDC", "venue": "binance", "confidence": 0.95},
        {"decision": "CANCEL", "symbol": "BTCUSDC", "venue": "binance", "cancel_order_ids": ["ai_keep_3"], "confidence": 0.50},
    ):
        assert gw.submit(bad_cancel, source="auto_cryptotrade_executor", max_mode="enforce", instruments_map={"BTCUSDC": btc_inst}) is None
    btc_inst.cancel_order.assert_not_called()

    # 3. horizon_h parsing & normalization
    assert extract_intent_horizon_hours({"horizon_h": 24}) == pytest.approx(24.0)
    assert extract_intent_horizon_hours({"horizon_h": "90m"}) == pytest.approx(1.5)
    assert extract_intent_horizon_hours({"horizon": "2d"}) == pytest.approx(48.0)
    assert extract_intent_horizon_hours({"horizon_h": -5}) is None
    assert extract_intent_horizon_hours({"horizon_h": 9999}) is None
    norm = normalize_trading_decisions(
        {"market_trading_decisions": [{"decision": "BUY", "symbol": "BTCUSDC", "venue": "binance", "horizon_h": "6h"}]}
    )
    assert norm[0]["horizon_h"] == pytest.approx(6.0)

    # 4. Full & compact prompts surface LIMIT-only execution, AI survival breakers, R:R rule, CANCEL, and horizon_h
    empty_logs = LogEventSummary(0, 0, 0)
    full_p = build_supervisor_prompt(TelemetrySensorSnapshot(), empty_logs, topic_spec("auto_cryptotrade_executor"), compact_rules=False)
    comp_p = build_supervisor_prompt(TelemetrySensorSnapshot(), empty_logs, topic_spec("auto_cryptotrade_executor"), compact_rules=True)
    assert "min R:R >= 0.5x" in full_p
    assert "always re-evaluate R:R from the new `limit_price`" in full_p
    assert '"horizon_h": null' in full_p and '"horizon_h": null' in comp_p
    assert "AI survival breakers active:" in comp_p
    assert "RR>=0.5x" in comp_p
    assert "re-evaluate R:R from new `limit_price` on replace" in comp_p
    assert "zero deterministic guards," not in comp_p
    assert "only on poor R:R" not in full_p and "only on poor R:R" not in comp_p
    assert "instead of chasing into poor R:R" in full_p and "instead of chasing into poor R:R" in comp_p
    assert 'near-BBO maker `LIMIT` (`urgency:"HIGH"' not in full_p
    assert 'near-BBO maker `LIMIT` (`urgency:"HIGH"' not in comp_p
    assert '`urgency:"NORMAL"`' in full_p and '`urgency:"NORMAL"`' in comp_p


def test_simultaneous_buys_share_cycle_venue_quote_cash_budget(tmp_path):
    """Simultaneous BUYs on the same (venue, quote) within one cycle share a cumulative quote cap without double-deducting live balance refreshes."""
    from orchestratorOS.lib.technical_repair_ssot import submit_supervisor_trading_decisions

    ws = tmp_path / "ws"
    (ws / "cachedb").mkdir(parents=True)
    (ws / "order_guard.conf").write_text(
        "autonomous_trading_mode = enforce\n"
        "autonomous_trading_min_notional_usd = 10.0\n"
        "autonomous_trading_max_notional_usd = 2000.0\n"
        "autonomous_trading_min_confidence = 0.85\n"
        "autonomous_trading_notify = 0\n",
        encoding="utf-8",
    )
    gw = AutonomousIntentGateway(workspace_dir=str(ws), cache_dir="cachedb", mode="enforce")

    def _make_inst(venue: str, sym: str, base: str, quote: str, px: float, bal_box: dict):
        inst = MagicMock()
        inst.symbol = sym
        inst.provider_name = venue
        inst.base = base
        inst.quote = quote
        inst._provider = MagicMock()
        inst._provider.name = venue
        inst._provider.free_balance.side_effect = lambda asset: bal_box.get(str(asset).upper(), 0.0)
        inst._provider.get_open_orders.return_value = []
        inst._provider.get_orders.return_value = []
        inst.open_orders.return_value = []
        inst.cancel_order.return_value = True
        inst.get_price.return_value = px

        def _place(**kwargs):
            if bal_box.get("_live_deduct"):
                bal_box[quote] = max(0.0, bal_box[quote] - float(kwargs.get("qty", 0.0)) * float(kwargs.get("price", px)))
            return {"orderId": f"oid_{sym}_{int(kwargs.get('price', px) * 100)}", "status": "NEW"}

        inst.place.side_effect = _place
        return inst

    # 1. Stale balance cache ($2000 stays static across back-to-back calls):
    #    BUY #1 BTC $1500 executes, BUY #2 ARB $1500 clamps to ~$499 (500 * 0.998), BUY #3 ADA $500 is blocked;
    #    meanwhile Hyperliquid BUY $1200 uses its own independent venue quote cash unaffected by Binance spend.
    bn_bal_stale = {"USDC": 2000.0, "BTC": 0.0, "ARB": 0.0, "ADA": 0.0}
    hl_bal_stale = {"USDC": 1500.0, "HYPE": 0.0}
    btc_inst = _make_inst("binance", "BTCUSDC", "BTC", "USDC", 60000.0, bn_bal_stale)
    arb_inst = _make_inst("binance", "ARBUSDC", "ARB", "USDC", 0.11, bn_bal_stale)
    ada_inst = _make_inst("binance", "ADAUSDC", "ADA", "USDC", 0.15, bn_bal_stale)
    hype_inst = _make_inst("hyperliquid", "HYPEUSDC", "HYPE", "USDC", 60.0, hl_bal_stale)
    inst_map = {
        ("binance", "BTCUSDC"): btc_inst,
        ("binance", "ARBUSDC"): arb_inst,
        ("binance", "ADAUSDC"): ada_inst,
        ("hyperliquid", "HYPEUSDC"): hype_inst,
    }
    gw.resolve_instrument = lambda sym, venue=None, instruments_map=None: inst_map[(str(venue).lower(), str(sym).upper())]
    gw._get_live_price = lambda inst: inst.get_price()

    plan_stale = {
        "market_trading_decisions": [
            {"decision": "BUY", "venue": "binance", "symbol": "BTCUSDC", "order_type": "LIMIT", "limit_price": 59900.0, "target_px": 62000.0, "invalidation_px": 58500.0, "confidence": 0.92, "suggested_notional_usd": 1500.0, "thesis": "BTC dip"},
            {"decision": "BUY", "venue": "binance", "symbol": "ARBUSDC", "order_type": "LIMIT", "limit_price": 0.1095, "target_px": 0.1160, "invalidation_px": 0.1050, "confidence": 0.91, "suggested_notional_usd": 1500.0, "thesis": "ARB dip"},
            {"decision": "BUY", "venue": "binance", "symbol": "ADAUSDC", "order_type": "LIMIT", "limit_price": 0.1490, "target_px": 0.1580, "invalidation_px": 0.1440, "confidence": 0.90, "suggested_notional_usd": 500.0, "thesis": "ADA dip"},
            {"decision": "BUY", "venue": "hyperliquid", "symbol": "HYPEUSDC", "order_type": "LIMIT", "limit_price": 59.5, "target_px": 63.0, "invalidation_px": 57.5, "confidence": 0.92, "suggested_notional_usd": 1200.0, "thesis": "HYPE dip"},
        ]
    }
    _, _, emitted_stale, _, any_exec = submit_supervisor_trading_decisions(
        plan_stale,
        topic_spec("auto_cryptotrade_executor"),
        topic_key="auto_cryptotrade_executor",
        mode="enforce",
        now_ts=time.time(),
        intent_gateway=gw,
        is_rth_open_fn=lambda _: True,
    )
    assert any_exec is True
    assert len(emitted_stale) == 4
    by_sym_stale = {e["symbol"]: e for e in emitted_stale}
    assert by_sym_stale["BTCUSDC"]["status"] == "EXECUTED" and by_sym_stale["BTCUSDC"]["clamped_notional_usd"] == pytest.approx(1500.0, abs=0.05)
    assert by_sym_stale["ARBUSDC"]["status"] == "EXECUTED" and by_sym_stale["ARBUSDC"]["clamped_notional_usd"] == pytest.approx(499.0, abs=1.5)
    assert by_sym_stale["ADAUSDC"]["status"] == "BLOCKED_BY_GUARD"
    assert by_sym_stale["HYPEUSDC"]["status"] == "EXECUTED" and by_sym_stale["HYPEUSDC"]["clamped_notional_usd"] == pytest.approx(1200.0, abs=0.05)
    ada_inst.place.assert_not_called()

    # 2. Live balance refresh between BUY #1 and BUY #2 must NOT double-deduct:
    #    $2000 -> live drops to $500 after BUY #1 -> BUY #2 still gets ~$499 (not $0).
    gw2 = AutonomousIntentGateway(workspace_dir=str(ws), cache_dir="cachedb", mode="enforce")
    (ws / "cachedb" / "autonomous_trade_intents.json").unlink(missing_ok=True)
    bn_bal_live = {"USDC": 2000.0, "BTC": 0.0, "ARB": 0.0, "_live_deduct": True}
    btc_live = _make_inst("binance", "BTCUSDC", "BTC", "USDC", 60000.0, bn_bal_live)
    arb_live = _make_inst("binance", "ARBUSDC", "ARB", "USDC", 0.11, bn_bal_live)
    live_map = {("binance", "BTCUSDC"): btc_live, ("binance", "ARBUSDC"): arb_live}
    gw2.resolve_instrument = lambda sym, venue=None, instruments_map=None: live_map[(str(venue).lower(), str(sym).upper())]
    gw2._get_live_price = lambda inst: inst.get_price()

    _, _, emitted_live, _, _ = submit_supervisor_trading_decisions(
        {"market_trading_decisions": plan_stale["market_trading_decisions"][:2]},
        topic_spec("auto_cryptotrade_executor"),
        topic_key="auto_cryptotrade_executor",
        mode="enforce",
        now_ts=time.time(),
        intent_gateway=gw2,
        is_rth_open_fn=lambda _: True,
    )
    assert emitted_live[0]["status"] == "EXECUTED" and emitted_live[0]["clamped_notional_usd"] == pytest.approx(1500.0, abs=0.05)
    assert emitted_live[1]["status"] == "EXECUTED" and emitted_live[1]["clamped_notional_usd"] == pytest.approx(499.0, abs=1.5)

    # 3. Replacing a resting BUY order in BUY #2 credits its freed quote notional cleanly; shadow mode enforces identical cap.
    (ws / "cachedb" / "autonomous_trade_intents.json").unlink(missing_ok=True)
    bn_bal_rep = {"USDC": 2000.0, "BTC": 0.0, "ARB": 0.0}
    btc_rep = _make_inst("binance", "BTCUSDC", "BTC", "USDC", 60000.0, bn_bal_rep)
    arb_rep = _make_inst("binance", "ARBUSDC", "ARB", "USDC", 0.11, bn_bal_rep)
    arb_rep.open_orders.return_value = [
        {"orderId": "old_arb_1", "clientOrderId": "AI_intent_1_buy_arbusdc", "side": "BUY", "price": "0.10", "origQty": "9000", "executedQty": "0"}
    ]
    rep_map = {("binance", "BTCUSDC"): btc_rep, ("binance", "ARBUSDC"): arb_rep}
    gw2.resolve_instrument = lambda sym, venue=None, instruments_map=None: rep_map[(str(venue).lower(), str(sym).upper())]
    plan_rep = {
        "market_trading_decisions": [
            plan_stale["market_trading_decisions"][0],
            {**plan_stale["market_trading_decisions"][1], "replace_open_orders": True, "suggested_notional_usd": 1200.0},
        ]
    }
    _, _, emitted_rep, _, _ = submit_supervisor_trading_decisions(
        plan_rep,
        topic_spec("auto_cryptotrade_executor"),
        topic_key="auto_cryptotrade_executor",
        mode="enforce",
        now_ts=time.time(),
        intent_gateway=gw2,
        is_rth_open_fn=lambda _: True,
    )
    assert emitted_rep[0]["status"] == "EXECUTED" and emitted_rep[0]["clamped_notional_usd"] == pytest.approx(1500.0, abs=0.05)
    # Remaining $500 + $900 canceled ARB BUY = $1400 cap -> full $1200 replacement fits cleanly!
    assert emitted_rep[1]["status"] == "EXECUTED" and emitted_rep[1]["clamped_notional_usd"] == pytest.approx(1200.0, abs=0.05)

    # 4. Shadow mode also clamps simultaneous BUYs sharing venue cash.
    (ws / "cachedb" / "autonomous_trade_intents.json").unlink(missing_ok=True)
    arb_rep.open_orders.return_value = []
    _, _, emitted_sh, _, _ = submit_supervisor_trading_decisions(
        {"market_trading_decisions": plan_stale["market_trading_decisions"][:2]},
        topic_spec("auto_cryptotrade_executor"),
        topic_key="auto_cryptotrade_executor",
        mode="shadow",
        now_ts=time.time(),
        intent_gateway=gw2,
        is_rth_open_fn=lambda _: True,
    )
    assert emitted_sh[0]["status"] == "OBSERVED_SHADOW" and emitted_sh[0]["clamped_notional_usd"] == pytest.approx(1500.0, abs=0.05)
    assert emitted_sh[1]["status"] == "OBSERVED_SHADOW" and emitted_sh[1]["clamped_notional_usd"] == pytest.approx(499.0, abs=1.5)

    # 5. Ambiguous BUY submission conservatively reserves quote spend (fail-closed) so BUY #2 cannot overspend.
    (ws / "cachedb" / "autonomous_trade_intents.json").unlink(missing_ok=True)
    bn_bal_amb = {"USDC": 2000.0, "BTC": 0.0, "ARB": 0.0}
    btc_amb = _make_inst("binance", "BTCUSDC", "BTC", "USDC", 60000.0, bn_bal_amb)
    arb_amb = _make_inst("binance", "ARBUSDC", "ARB", "USDC", 0.11, bn_bal_amb)

    def _ambiguous_place(**kwargs):
        ctx = kwargs.get("_outcome_context")
        if isinstance(ctx, dict):
            ctx.update({"state": "unknown", "reason": "submit_ambiguous"})
        return None

    btc_amb.place.side_effect = _ambiguous_place
    amb_map = {("binance", "BTCUSDC"): btc_amb, ("binance", "ARBUSDC"): arb_amb}
    gw2.resolve_instrument = lambda sym, venue=None, instruments_map=None: amb_map[(str(venue).lower(), str(sym).upper())]
    _, _, emitted_amb, _, _ = submit_supervisor_trading_decisions(
        {"market_trading_decisions": plan_stale["market_trading_decisions"][:2]},
        topic_spec("auto_cryptotrade_executor"),
        topic_key="auto_cryptotrade_executor",
        mode="enforce",
        now_ts=time.time(),
        intent_gateway=gw2,
        is_rth_open_fn=lambda _: True,
    )
    by_amb = {e["symbol"]: e for e in emitted_amb}
    assert by_amb["BTCUSDC"]["history"][-1].get("submission_state") == "unknown"
    assert by_amb["ARBUSDC"]["status"] == "EXECUTED" and by_amb["ARBUSDC"]["clamped_notional_usd"] == pytest.approx(499.0, abs=1.5)

    # 6. Standalone CANCEL runs BEFORE same-cycle BUYs in submit_supervisor_trading_decisions (unlocking locked quote
    #    on the exchange first), AND mid-scope BUY -> standalone CANCEL -> BUY also credits freed BUY quote cleanly.
    (ws / "cachedb" / "autonomous_trade_intents.json").unlink(missing_ok=True)
    bn_bal_can = {"USDC": 2000.0, "BTC": 0.0, "ARB": 0.0, "ADA": 0.0}
    btc_can = _make_inst("binance", "BTCUSDC", "BTC", "USDC", 60000.0, bn_bal_can)
    ada_can = _make_inst("binance", "ADAUSDC", "ADA", "USDC", 0.15, bn_bal_can)
    arb_can = _make_inst("binance", "ARBUSDC", "ARB", "USDC", 0.11, bn_bal_can)
    ada_can.open_orders.return_value = [
        {"orderId": "ada_rest_1", "clientOrderId": "AI_intent_9_buy_adausdc", "side": "BUY", "price": "0.14", "origQty": "6000", "executedQty": "0"}
    ]

    def _unlock_cancel(oid):
        if oid == "ada_rest_1":
            bn_bal_can["USDC"] += 840.0
        return True

    ada_can.cancel_order.side_effect = _unlock_cancel
    can_map = {
        ("binance", "BTCUSDC"): btc_can,
        ("binance", "ADAUSDC"): ada_can,
        ("binance", "ARBUSDC"): arb_can,
    }
    gw2.resolve_instrument = lambda sym, venue=None, instruments_map=None: can_map[(str(venue).lower(), str(sym).upper())]
    plan_can = {
        "market_trading_decisions": [
            {**plan_stale["market_trading_decisions"][0], "confidence": 0.95},
            {"decision": "CANCEL", "venue": "binance", "symbol": "ADAUSDC", "cancel_order_ids": ["ada_rest_1"], "confidence": 0.93},
            {**plan_stale["market_trading_decisions"][1], "confidence": 0.90, "suggested_notional_usd": 1200.0},
        ]
    }
    _, _, emitted_can, _, _ = submit_supervisor_trading_decisions(
        plan_can,
        topic_spec("auto_cryptotrade_executor"),
        topic_key="auto_cryptotrade_executor",
        mode="enforce",
        now_ts=time.time(),
        intent_gateway=gw2,
        is_rth_open_fn=lambda _: True,
    )
    assert emitted_can[0]["decision"] == "CANCEL" and emitted_can[0]["status"] == "CANCELED"
    by_can = {e["symbol"]: e for e in emitted_can}
    assert by_can["BTCUSDC"]["status"] == "EXECUTED" and by_can["BTCUSDC"]["clamped_notional_usd"] == pytest.approx(1500.0, abs=0.05)
    # Unlocked $840 + initial $2000 = $2840 -> $1500 BTC + $1200 ARB both fit cleanly!
    assert by_can["ARBUSDC"]["status"] == "EXECUTED" and by_can["ARBUSDC"]["clamped_notional_usd"] == pytest.approx(1200.0, abs=0.05)

    # 6b. Direct cycle_shared_cash_scope() BUY -> standalone CANCEL -> BUY also credits freed quote into cycle_caps.
    (ws / "cachedb" / "autonomous_trade_intents.json").unlink(missing_ok=True)
    bn_bal_can["USDC"] = 2000.0
    ada_can.cancel_order.side_effect = None
    ada_can.cancel_order.return_value = True
    with gw2.cycle_shared_cash_scope():
        r_b1 = gw2.submit(plan_can["market_trading_decisions"][0], source="auto_cryptotrade_executor", max_mode="enforce")
        r_c1 = gw2.submit(plan_can["market_trading_decisions"][1], source="auto_cryptotrade_executor", max_mode="enforce")
        r_b2 = gw2.submit(plan_can["market_trading_decisions"][2], source="auto_cryptotrade_executor", max_mode="enforce")
    assert r_b1["status"] == "EXECUTED" and r_b1["clamped_notional_usd"] == pytest.approx(1500.0, abs=0.05)
    assert r_c1["status"] == "CANCELED"
    assert r_b2["status"] == "EXECUTED" and r_b2["clamped_notional_usd"] == pytest.approx(1200.0, abs=0.05)

    # 7. Deferred BUY (queued to order_retry_worker) reserves quote spend so BUY #2 cannot steal its cash in the same cycle.
    (ws / "cachedb" / "autonomous_trade_intents.json").unlink(missing_ok=True)
    bn_bal_def = {"USDC": 2000.0, "BTC": 0.0, "ARB": 0.0}
    btc_def = _make_inst("binance", "BTCUSDC", "BTC", "USDC", 60000.0, bn_bal_def)
    arb_def = _make_inst("binance", "ARBUSDC", "ARB", "USDC", 0.11, bn_bal_def)

    def _deferred_place(**kwargs):
        ctx = kwargs.get("_outcome_context")
        if isinstance(ctx, dict):
            ctx.update({"category": "deferred", "reason": "queued_for_retry"})
        return None

    btc_def.place.side_effect = _deferred_place
    def_map = {("binance", "BTCUSDC"): btc_def, ("binance", "ARBUSDC"): arb_def}
    gw2.resolve_instrument = lambda sym, venue=None, instruments_map=None: def_map[(str(venue).lower(), str(sym).upper())]
    _, _, emitted_def, _, _ = submit_supervisor_trading_decisions(
        {"market_trading_decisions": plan_stale["market_trading_decisions"][:2]},
        topic_spec("auto_cryptotrade_executor"),
        topic_key="auto_cryptotrade_executor",
        mode="enforce",
        now_ts=time.time(),
        intent_gateway=gw2,
        is_rth_open_fn=lambda _: True,
    )
    by_def = {e["symbol"]: e for e in emitted_def}
    assert by_def["BTCUSDC"]["history"][-1].get("category") == "deferred"
    assert by_def["ARBUSDC"]["status"] == "EXECUTED" and by_def["ARBUSDC"]["clamped_notional_usd"] == pytest.approx(499.0, abs=1.5)

    # 8. normalize_trading_decisions preserves same-symbol standalone CANCEL directives alongside BUY/SELL and
    #    deduplicates overlapping cancel_order_ids + plain HOLD cleanly.
    same_sym_decs = normalize_trading_decisions({
        "market_trading_decisions": [
            {"venue": "binance", "symbol": "BTCUSDC", "decision": "BUY", "confidence": 0.91, "suggested_notional_usd": 500.0},
            {"venue": "binance", "symbol": "BTCUSDC", "decision": "CANCEL", "cancel_order_ids": ["ord-opp-1"], "confidence": 0.90},
            {"venue": "binance", "symbol": "BTCUSDC", "decision": "CANCEL", "cancel_order_ids": ["ord-opp-1", "ord-opp-2"], "confidence": 0.89},
            {"venue": "binance", "symbol": "BTCUSDC", "decision": "HOLD", "confidence": 0.95},
        ]
    })
    assert [(d["decision"], d.get("cancel_order_ids")) for d in same_sym_decs] == [
        ("BUY", None),
        ("CANCEL", ["ord-opp-1"]),
        ("CANCEL", ["ord-opp-2"]),
    ]
    # Redundant standalone CANCEL targeting an order_id already covered by BUY is deduplicated cleanly
    overlap_decs = normalize_trading_decisions({
        "market_trading_decisions": [
            {"venue": "binance", "symbol": "BTCUSDC", "decision": "BUY", "cancel_order_ids": ["ord-1"], "confidence": 0.92, "suggested_notional_usd": 500.0},
            {"venue": "binance", "symbol": "BTCUSDC", "decision": "CANCEL", "cancel_order_ids": ["ord-1"], "confidence": 0.90},
        ]
    })
    assert len(overlap_decs) == 1 and overlap_decs[0]["decision"] == "BUY"

    # 9. Standalone CANCEL marks matching unfilled EXECUTED AI intents as terminal CANCELED in durable state so
    #    daily BUY circuit-breaker stops counting them and prune_reconciled_terminal_open_orders drops stale book ghosts.
    from providers.ai_circuit_breaker import evaluate_ai_intent_circuit_breaker
    from providers.intent_state import prune_reconciled_terminal_open_orders

    (ws / "cachedb" / "autonomous_trade_intents.json").unlink(missing_ok=True)
    now_sc = time.time()
    gw2.save_all_intents([
        {
            "intent_id": "ai_buy_to_cancel",
            "timestamp": now_sc - 120.0,
            "decision": "BUY",
            "symbol": "ADAUSDC",
            "venue": "binance",
            "order_type": "LIMIT",
            "limit_price": 0.14,
            "status": "EXECUTED",
            "clamped_notional_usd": 840.0,
            "execution_payload": {"orderId": "ada_rest_1", "status": "NEW", "executedQty": "0.0"},
            "history": [],
        }
    ])
    ada_can.open_orders.return_value = [
        {"orderId": "ada_rest_1", "clientOrderId": "AI_ai_buy_to_cancel", "side": "BUY", "price": "0.14", "origQty": "6000", "executedQty": "0"}
    ]
    gw2.resolve_instrument = lambda sym, venue=None, instruments_map=None: can_map[(str(venue).lower(), str(sym).upper())]
    res_sc = gw2.submit(
        {"decision": "CANCEL", "venue": "binance", "symbol": "ADAUSDC", "cancel_order_ids": ["ada_rest_1"], "confidence": 0.92},
        source="auto_cryptotrade_executor",
        max_mode="enforce",
    )
    assert res_sc is not None and res_sc["status"] == "CANCELED"
    persisted_after = gw2.load_all_intents()
    assert persisted_after[0]["execution_payload"]["status"] == "CANCELED"
    assert persisted_after[0]["execution_payload"]["reconciled_terminal"] is True
    _, _, _, cb_after = evaluate_ai_intent_circuit_breaker(
        {"intent_id": "next_buy", "decision": "BUY"},
        persisted_after,
        decision="BUY",
        symbol="ADAUSDC",
        venue="binance",
        price=0.14,
        effective_usd=500.0,
        min_notional_usd=10.0,
        now_ts=now_sc,
    )
    assert cb_after["daily_sym_buys"] == 0 and cb_after["daily_buy_notional_usd"] == 0.0
    assert prune_reconciled_terminal_open_orders([{"venue": "binance", "order_id": "ada_rest_1"}], persisted_after) == []











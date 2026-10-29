"""Unit tests for AutonomousIntentGateway in orchestratorTrade."""
from __future__ import annotations

import os
import time
from unittest.mock import MagicMock, patch
import pytest

from orchestratorTrade.autonomous_intent_gateway import (
    AUTONOMOUS_BYPASS_GUARDS,
    AutonomousIntentGateway,
)


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

    pending = gw.get_pending_intents()
    assert len(pending) == 1
    assert pending[0]["symbol"] == "BTCUSDC"


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


def test_process_intent_shadow_mode_observation(temp_workspace):
    ws, _ = temp_workspace
    gw = AutonomousIntentGateway(workspace_dir=ws, cache_dir="cachedb", mode="shadow", max_notional_usd=2000.0)

    mock_inst = MagicMock()
    mock_inst.symbol = "BTCUSDC"
    mock_inst.provider_name = "binance"
    mock_inst._provider.name = "binance"
    mock_inst._provider.get_current_price.return_value = 80000.0

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
    assert "$2000" in mock_notify.call_args[1]["title"]


@pytest.mark.parametrize("side", ["BUY", "SELL"])
def test_process_intent_symmetric_buy_and_sell_bypass_all_profit_guards(temp_workspace, side):
    ws, _ = temp_workspace
    gw = AutonomousIntentGateway(workspace_dir=ws, cache_dir="cachedb", mode="enforce")

    mock_inst = MagicMock()
    mock_inst.symbol = "BTCUSDC"
    mock_inst.base = "BTC"
    mock_inst.quote = "USDC"
    mock_inst.provider_name = "binance"
    mock_inst._provider.name = "binance"
    mock_inst._provider.get_current_price.return_value = 79500.0
    mock_inst._provider.free_balance.return_value = 2500.0 if side == "BUY" else 0.08
    mock_inst.place.return_value = {"orderId": f"ord_{side}", "status": "FILLED"}

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
    assert kwargs["market"] is True
    assert "force" not in kwargs
    assert "is_market" not in kwargs
    assert kwargs["enforce_business_minimum"] is False
    assert kwargs["wait_for_trend"] is False
    assert kwargs["regime_context"] is None
    assert kwargs["intent_id"] == f"intent_sym_{side.lower()}"
    assert kwargs["client_order_id"] == f"AI_intent_sym_{side.lower()}"
    assert kwargs["bypass_guards"] == AUTONOMOUS_BYPASS_GUARDS
    assert {"all", "daily_limit", "profit_guard", "trend", "qlimit", "intelligence", "llm"} <= set(kwargs["bypass_guards"])
    assert "$1200" in mock_notify.call_args[1]["title"]


def test_process_intent_clamps_to_available_wallet_balance_and_blocks_zero_balance(temp_workspace):
    ws, _ = temp_workspace
    gw = AutonomousIntentGateway(workspace_dir=ws, cache_dir="cachedb", mode="enforce")

    # 1. SELL $2000 on ARBUSDC when wallet holds 600 ARB @ $0.20 ($120) -> smoothly clamps to $120
    mock_arb = MagicMock()
    mock_arb.symbol = "ARBUSDC"
    mock_arb.base = "ARB"
    mock_arb.quote = "USDC"
    mock_arb.provider_name = "binance"
    mock_arb._provider.name = "binance"
    mock_arb._provider.get_current_price.return_value = 0.20
    mock_arb._provider.free_balance.return_value = 600.0
    mock_arb.place.return_value = {"orderId": "arb_sell_1"}

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
    mock_krk = MagicMock()
    mock_krk.symbol = "HYPEUSD"
    mock_krk.base = "HYPE"
    mock_krk.quote = "USD"
    mock_krk.provider_name = "kraken"
    mock_krk._provider.name = "kraken"
    mock_krk._provider.get_current_price.return_value = 75.0
    mock_krk._provider.free_balance.return_value = 0.0

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

    mock_inst = MagicMock()
    mock_inst.symbol = "BTCUSDC"
    mock_inst.base = "BTC"
    mock_inst.quote = "USDC"
    mock_inst.provider_name = "binance"
    mock_inst._provider.name = "binance"
    mock_inst._provider.get_current_price.return_value = 80000.0
    mock_inst._provider.free_balance.return_value = 0.09  # $7,200 USD BTC position
    mock_inst.place.return_value = {"orderId": "tranche_ok"}

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

    # Verify .previous crash recovery on corrupted primary intents file
    import shutil
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

    mock_inst = MagicMock()
    mock_inst.enabled = True
    mock_inst.symbol = "BTCUSDC"
    mock_inst.provider_name = "binance"
    mock_inst._provider.name = "binance"
    mock_inst._provider.get_current_price.return_value = 80000.0

    with patch("instruments_config.load_instruments", return_value={"BTC": mock_inst}), \
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

    mock_binance = MagicMock()
    mock_binance.enabled = True
    mock_binance.symbol = "BTCUSDC"
    mock_binance.provider_name = "binance"

    mock_kraken = MagicMock()
    mock_kraken.enabled = True
    mock_kraken.symbol = "ETHUSD"
    mock_kraken.provider_name = "kraken"

    mock_disabled = MagicMock()
    mock_disabled.enabled = False
    mock_disabled.symbol = "HYPEUSDC"
    mock_disabled.provider_name = "hyperliquid"

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
        "instruments_config.load_instruments",
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

    mock_inst = MagicMock()
    mock_inst.symbol = "BTCUSDC"
    mock_inst.provider_name = "binance"
    mock_inst._provider.name = "binance"
    mock_inst._provider.get_current_price.return_value = 80000.0

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
    import json
    ws, _ = temp_workspace
    gw = AutonomousIntentGateway(workspace_dir=ws, cache_dir="cachedb", mode="enforce")

    mock_inst = MagicMock()
    mock_inst.symbol = "TAOUSDC"
    mock_inst.base = "TAO"
    mock_inst.quote = "USDC"
    mock_inst.provider_name = "binance"
    mock_inst._provider.name = "binance"
    mock_inst._provider.get_current_price.return_value = 251.0
    mock_inst._provider.free_balance.return_value = 4.70

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
    import json
    ws, cache_dir = temp_workspace
    gw = AutonomousIntentGateway(workspace_dir=ws, cache_dir="cachedb", mode="shadow")

    kraken_tao = MagicMock()
    kraken_tao.symbol = "TAOUSD"
    kraken_tao.base = "TAO"
    kraken_tao.quote = "USD"
    kraken_tao.provider_name = "kraken"
    kraken_tao.enabled = True
    kraken_tao._provider.get_current_price.return_value = None

    instruments_map = {"KRAKEN_TAO": kraken_tao}
    assert gw.resolve_instrument("TAOUSDC", venue="kraken", instruments_map=instruments_map) is kraken_tao
    assert gw.resolve_instrument("TAO/USDT", venue="kraken", instruments_map=instruments_map) is kraken_tao

    price_cache = os.path.join(cache_dir, "cache_current_price.json")
    with open(price_cache, "w", encoding="utf-8") as f:
        json.dump({"items": {"TAOUSDC": [[1700000000, 345.67]]}}, f)

    assert gw._get_live_price(kraken_tao) == pytest.approx(345.67)


def test_load_tradable_instruments_discovers_hyperliquid_and_rth_gated_t212(temp_workspace):
    from datetime import datetime
    from zoneinfo import ZoneInfo

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
    import json
    from intelligence.sentiment.guards.high_stake_guard import collect_pretrade_telemetry
    from orchestratorOS.supervisor.autonomous_ai_reconciler import _apply_short_trend_item

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
    assert _apply_short_trend_item(entry, tinfo_90_900) is True
    assert entry["slope_3_7m"] == pytest.approx(0.0021)
    assert entry["slope_15m"] == pytest.approx(0.0014)

    with open(os.path.join(cache_dir, "cache_short_trend.json"), "w", encoding="utf-8") as f:
        json.dump({"items": {"TAOUSDC": tinfo_90_900}}, f)
    telem = collect_pretrade_telemetry("TAOUSDC", cachedb_dir=cache_dir)
    assert "3.7m=+0.00210" in telem.get("Dynamic Multi-Window Slopes", "")
    assert "15m=+0.00140" in telem.get("Dynamic Multi-Window Slopes", "")


def test_notional_confidence_t212_floor_and_account_cache_retry_hardening(temp_workspace):
    import json
    from orchestratorTrade.autonomous_intent_gateway import (
        _extract_intent_notional,
        _parse_intent_confidence,
    )

    ws, _ = temp_workspace
    gw = AutonomousIntentGateway(workspace_dir=ws, cache_dir="cachedb", mode="enforce")

    # 1. _extract_intent_notional parses currency strings, aliases, and rejects <= 0
    assert _extract_intent_notional({"suggested_notional_usd": "$1,250.50"}) == pytest.approx(1250.50)
    assert _extract_intent_notional({"notional": "800 USD"}) == pytest.approx(800.0)
    assert _extract_intent_notional({"value_usd": "$450 USDC"}) == pytest.approx(450.0)
    assert _extract_intent_notional({"suggested_notional_usd": "-50"}, default=25.0) == pytest.approx(25.0)

    # 2. _parse_intent_confidence normalizes percentage strings and 1..100 numbers
    assert _parse_intent_confidence("91%") == pytest.approx(0.91)
    assert _parse_intent_confidence(92) == pytest.approx(0.92)
    assert _parse_intent_confidence(0.88) == pytest.approx(0.88)
    assert _parse_intent_confidence(True, default=0.0) == 0.0

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

    # 5. reconcile_intents computes fallback cummulativeQuoteQty from qty * price on submit_accepted
    audit_dir = os.path.join(ws, "logger", "execution_audit")
    os.makedirs(audit_dir, exist_ok=True)
    with open(os.path.join(audit_dir, "execution_audit_2026-10-29.jsonl"), "w", encoding="utf-8") as f:
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
    assert pending_list[0]["execution_payload"]["cummulativeQuoteQty"] == "501.0"


def test_passive_limit_orders_order_replacement_and_contrarian_dip_prompt(temp_workspace):
    import json
    from binance_api.order_id_context import REPRICEABLE_BOTS, resolve_order_owner
    from orchestratorOS.agy.topics import condense_decision
    from orchestratorOS.supervisor.prompt_formatters import build_supervisor_prompt
    from providers.execution_audit import normalize_trading_decisions, summarize_recent_ai_intents

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

    # 4. Out-of-bounds or missing limit_price (< 0.5x, > 2.0x, or None when order_type="LIMIT") is rejected
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
    assert rec_bad["status"] == "REJECTED_INVALID_LIMIT_PRICE"
    assert rec_missing_px["status"] == "REJECTED_INVALID_LIMIT_PRICE"

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

    from providers.execution_audit import extract_intent_notional
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

    from orchestratorOS.agy.topics import topic_spec
    from orchestratorOS.lib.technical_repair_ssot import LogEventSummary
    from orchestratorOS.supervisor.autonomous_ai_reconciler import TelemetrySensorSnapshot

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
    assert "advisory volatility signal for legacy bots only" in prompt
    assert "CONTRARIAN DIP ACCUMULATION & ADVISORY STATUS OF MACRO/GEOPOLITICAL LABELS" in prompt
    assert "DUAL EXECUTION MODE — IMMEDIATE 'MARKET' ORDERS OR PASSIVE 'LIMIT' ORDERS IN THE ORDERBOOK" in prompt
    assert '"order_type": "MARKET" | "LIMIT"' in prompt
    assert "'order_type': 'LIMIT'" in prompt and "79850.0" in prompt







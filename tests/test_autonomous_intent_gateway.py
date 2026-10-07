"""Unit tests for AutonomousIntentGateway in orchestratorTrade."""
from __future__ import annotations

import json
import os
import time
from unittest.mock import MagicMock, patch
import pytest

from orchestratorTrade.autonomous_intent_gateway import AutonomousIntentGateway, TradeIntentRecord


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
        f.write("autonomous_trading_max_notional_eur = 300.0\n")
        f.write("autonomous_trading_min_confidence = 0.85\n")
        f.write("autonomous_trading_notify = 1\n")

    return str(ws), str(cache)


def test_gateway_initialization(temp_workspace):
    ws, cache = temp_workspace
    gw = AutonomousIntentGateway(workspace_dir=ws, cache_dir="cachedb")
    assert gw.mode == "shadow"
    assert gw.max_notional_eur == 300.0
    assert gw.min_confidence == 0.85
    assert gw.notify_enabled is True
    assert gw.load_all_intents() == []


def test_intent_persistence(temp_workspace):
    ws, cache = temp_workspace
    gw = AutonomousIntentGateway(workspace_dir=ws, cache_dir="cachedb")

    intents = [
        {
            "intent_id": "test_1",
            "timestamp": time.time(),
            "decision": "BUY",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "confidence": 0.90,
            "suggested_notional_eur": 250.0,
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
    ws, cache = temp_workspace
    gw = AutonomousIntentGateway(workspace_dir=ws, cache_dir="cachedb")

    intent = {
        "intent_id": "low_conf_1",
        "decision": "BUY",
        "symbol": "BTCUSDC",
        "venue": "binance",
        "confidence": 0.70,  # Below 0.85
        "suggested_notional_eur": 200.0,
        "status": "PENDING",
    }
    processed = gw.process_intent(intent)
    assert processed["status"] == "REJECTED_LOW_CONFIDENCE"
    assert len(processed.get("history", [])) == 1


def test_process_intent_unknown_instrument_rejected(temp_workspace):
    ws, cache = temp_workspace
    gw = AutonomousIntentGateway(workspace_dir=ws, cache_dir="cachedb")

    intent = {
        "intent_id": "unknown_inst_1",
        "decision": "BUY",
        "symbol": "XYZNONEXISTENT",
        "venue": "binance",
        "confidence": 0.90,
        "suggested_notional_eur": 100.0,
        "status": "PENDING",
    }
    # Pass empty instruments_map
    processed = gw.process_intent(intent, instruments_map={})
    assert processed["status"] == "REJECTED_UNKNOWN_INSTRUMENT"


def test_process_intent_shadow_mode_observation(temp_workspace):
    ws, cache = temp_workspace
    gw = AutonomousIntentGateway(workspace_dir=ws, cache_dir="cachedb", mode="shadow")

    # Mock Instrument
    mock_inst = MagicMock()
    mock_inst.symbol = "BTCUSDC"
    mock_inst.provider_name = "binance"
    mock_inst._provider.get_current_price.return_value = 80000.0
    mock_inst._call_profit_guard_window_ref.return_value = None

    intent = {
        "intent_id": "shadow_buy_1",
        "decision": "BUY",
        "symbol": "BTCUSDC",
        "venue": "binance",
        "confidence": 0.92,
        "suggested_notional_eur": 500.0,  # Capped at 300.0
        "status": "PENDING",
        "thesis": "Bull breakout with positive macro sentiment",
    }

    with patch("order_guard.profit_guard", return_value=True), \
         patch.object(gw, "_dispatch_notify", return_value=True) as mock_notify:
        processed = gw.process_intent(intent, instruments_map={"BTC": mock_inst})

    # Under shadow mode, order must NOT be placed on provider
    mock_inst.place.assert_not_called()
    assert processed["status"] == "OBSERVED_SHADOW"
    mock_notify.assert_called_once()
    assert "BTCUSDC" in mock_notify.call_args[1]["symbol"]
    assert "€300" in mock_notify.call_args[1]["title"]  # Capped notional


def test_process_intent_enforce_mode_execution(temp_workspace):
    ws, cache = temp_workspace
    gw = AutonomousIntentGateway(workspace_dir=ws, cache_dir="cachedb", mode="enforce")

    # Mock Instrument
    mock_inst = MagicMock()
    mock_inst.symbol = "BTCUSDC"
    mock_inst.provider_name = "binance"
    mock_inst._provider.get_current_price.return_value = 80000.0
    mock_inst._call_profit_guard_window_ref.return_value = None
    mock_inst.place.return_value = {"orderId": "123456", "status": "FILLED"}

    intent = {
        "intent_id": "enforce_buy_1",
        "decision": "BUY",
        "symbol": "BTCUSDC",
        "venue": "binance",
        "confidence": 0.89,
        "suggested_notional_eur": 200.0,
        "status": "PENDING",
        "thesis": "Strong momentum",
    }

    with patch("order_guard.profit_guard", return_value=True), \
         patch.object(gw, "_dispatch_notify", return_value=True) as mock_notify:
        processed = gw.process_intent(intent, instruments_map={"BTC": mock_inst})

    # Under enforce mode, Instrument.place must be invoked
    mock_inst.place.assert_called_once()
    assert processed["status"] == "EXECUTED"
    assert processed["execution_payload"]["orderId"] == "123456"
    mock_notify.assert_called_once()
    assert "EXECUTED" in mock_notify.call_args[1]["title"]


def test_process_intent_enforce_mode_guard_blocked(temp_workspace):
    ws, cache = temp_workspace
    gw = AutonomousIntentGateway(workspace_dir=ws, cache_dir="cachedb", mode="enforce")

    # Mock Instrument
    mock_inst = MagicMock()
    mock_inst.symbol = "BTCUSDC"
    mock_inst.provider_name = "binance"
    mock_inst._provider.get_current_price.return_value = 80000.0
    mock_inst._call_profit_guard_window_ref.return_value = None

    intent = {
        "intent_id": "guard_blocked_1",
        "decision": "BUY",
        "symbol": "BTCUSDC",
        "venue": "binance",
        "confidence": 0.90,
        "suggested_notional_eur": 200.0,
        "status": "PENDING",
        "thesis": "Attempting buy above last sell",
    }

    with patch("order_guard.profit_guard", return_value=False), \
         patch.object(gw, "_dispatch_notify", return_value=True) as mock_notify:
        processed = gw.process_intent(intent, instruments_map={"BTC": mock_inst})

    # Guard blocked, Instrument.place must NOT be called
    mock_inst.place.assert_not_called()
    assert processed["status"] == "BLOCKED_BY_GUARD"
    mock_notify.assert_called_once()
    assert "BLOCKED" in mock_notify.call_args[1]["title"]


def test_process_all_pending_lifecycle(temp_workspace):
    ws, cache = temp_workspace
    gw = AutonomousIntentGateway(workspace_dir=ws, cache_dir="cachedb", mode="shadow")

    # Seed 2 intents: 1 pending, 1 already executed
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
            "suggested_notional_eur": 150.0,
            "status": "PENDING",
        },
    ]
    gw.save_all_intents(intents)

    mock_inst = MagicMock()
    mock_inst.symbol = "BTCUSDC"
    mock_inst.provider_name = "binance"
    mock_inst._provider.get_current_price.return_value = 80000.0

    with patch("instruments_config.load_instruments", return_value={"BTC": mock_inst}), \
         patch("order_guard.profit_guard", return_value=True), \
         patch.object(gw, "_dispatch_notify", return_value=True):
        processed = gw.process_all_pending()

    assert len(processed) == 1
    assert processed[0]["intent_id"] == "intent_pending"
    assert processed[0]["status"] == "OBSERVED_SHADOW"

    # Verify state saved to disk
    reloaded = gw.load_all_intents()
    assert len(reloaded) == 2
    assert reloaded[0]["status"] == "EXECUTED"
    assert reloaded[1]["status"] == "OBSERVED_SHADOW"

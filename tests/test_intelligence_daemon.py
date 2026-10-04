"""Tests for Intelligence Telemetry Daemon and CLI utilities."""

from __future__ import annotations

import json
import os
import time
from unittest.mock import MagicMock, patch

import pytest

from intelligence.cli import get_daemon_status, inspect_symbol_intelligence
from intelligence.daemon import IntelligenceTelemetryDaemon
from intelligence.external.collectors.orderbook_depth import OrderbookSnapshot
from intelligence.external.collectors.derivatives_telemetry import DerivativesTelemetry
from intelligence.external.collectors.whale_positioning import WhalePositioningSnapshot
from intelligence.macro.geopolitical_analyzer import GeopoliticalThreatAssessment


@pytest.fixture
def temp_cache_dir(tmp_path):
    cache_dir = tmp_path / "cachedb"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir)


def test_daemon_initialization(temp_cache_dir):
    daemon = IntelligenceTelemetryDaemon(
        symbols=["BTCUSDC", "TAOUSDC"],
        cache_dir=temp_cache_dir,
        orderbook_interval_sec=10.0,
        derivatives_interval_sec=20.0,
        whale_interval_sec=30.0,
        macro_interval_sec=60.0,
    )
    assert daemon.symbols == ["BTCUSDC", "TAOUSDC"]
    assert daemon.orderbook_interval_sec == 10.0
    assert daemon.derivatives_interval_sec == 20.0
    assert daemon.whale_interval_sec == 30.0
    assert daemon.macro_interval_sec == 60.0


def test_daemon_run_cycle_mocked(temp_cache_dir):
    daemon = IntelligenceTelemetryDaemon(
        symbols=["BTCUSDC"],
        cache_dir=temp_cache_dir,
    )

    mock_ob_snap = OrderbookSnapshot(
        symbol="BTCUSDC",
        mid_price=60000.0,
        bid_depth_usd=100000.0,
        ask_depth_usd=80000.0,
        imbalance_ratio=0.55,
        largest_bid_wall_usd=20000.0,
        largest_bid_wall_price=59900.0,
        largest_ask_wall_usd=15000.0,
        largest_ask_wall_price=60100.0,
        ts=time.time(),
    )
    mock_deriv = DerivativesTelemetry(
        symbol="BTCUSDC",
        funding_rate=0.0001,
        predicted_funding_rate=0.0001,
        open_interest=5000.0,
        open_interest_usd=300000000.0,
        ts=time.time(),
    )
    mock_whale = WhalePositioningSnapshot(
        symbol="BTCUSDC",
        top_traders_long_ratio=1.5,
        top_traders_long_pct=0.6,
        taker_buy_sell_ratio=1.2,
        taker_buy_vol_usd=500000.0,
        taker_sell_vol_usd=400000.0,
        open_interest_usd=300000000.0,
        open_interest_1h_change_pct=1.5,
        divergence_regime="accumulation",
        ts=time.time(),
    )
    mock_geo = GeopoliticalThreatAssessment(
        threat_level="NORMAL",
        risk_score=0.1,
        summary="Calm",
        recommended_brake="NONE",
        headlines_analyzed=5,
        ts=time.time(),
    )

    with patch.object(daemon.orderbook_collector, "fetch", return_value=mock_ob_snap), \
         patch.object(daemon.derivatives_collector, "fetch", return_value=mock_deriv), \
         patch.object(daemon.whale_collector, "fetch", return_value=mock_whale), \
         patch.object(daemon.news_collector, "fetch", return_value=MagicMock()), \
         patch.object(daemon.geopolitical_analyzer, "assess", return_value=mock_geo):

        res = daemon.run_cycle(force=True)

        assert res["orderbook_updated"] == 1
        assert res["derivatives_updated"] == 1
        assert res["whale_updated"] == 1
        assert res["macro_updated"] is True

        # Verify heartbeat file was created
        hb_path = os.path.join(temp_cache_dir, "intelligence_daemon.heartbeat")
        assert os.path.exists(hb_path)
        with open(hb_path, "r", encoding="utf-8") as f:
            hb = json.load(f)
        assert hb["status"] == "alive"
        assert hb["symbols"] == ["BTCUSDC"]
        assert hb["last_run"]["orderbook_updated"] == 1


def test_daemon_heartbeat_inspection(temp_cache_dir):
    hb_path = os.path.join(temp_cache_dir, "intelligence_daemon.heartbeat")
    with open(hb_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "pid": os.getpid(),
                "status": "alive",
                "ts": time.time(),
                "symbols": ["BTCUSDC"],
                "intervals": {"orderbook": 15.0},
            },
            f,
        )

    status = get_daemon_status(temp_cache_dir)
    assert status["running"] is True
    assert status["pid"] == os.getpid()
    assert status["status"] == "alive"
    assert status["symbols"] == ["BTCUSDC"]


def test_inspect_symbol_intelligence(temp_cache_dir):
    # Inject synthetic snapshots
    ob_path = os.path.join(temp_cache_dir, "orderbook_depth_BTCUSDC.json")
    with open(ob_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "symbol": "BTCUSDC",
                "mid_price": 65000.0,
                "bid_depth_usd": 500000.0,
                "ask_depth_usd": 400000.0,
                "imbalance_ratio": 0.556,
                "largest_bid_wall_usd": 50000.0,
                "largest_bid_wall_price": 64900.0,
                "largest_ask_wall_usd": 30000.0,
                "largest_ask_wall_price": 65100.0,
                "ts": time.time(),
            },
            f,
        )

    with patch("order_guard.check_intelligence_guards", return_value=(True, "ok", 1.0)):
        report = inspect_symbol_intelligence("BTCUSDC", cache_dir=temp_cache_dir)
    assert report["symbol"] == "BTCUSDC"
    assert report["pillar2_external"]["orderbook"]["mid_price"] == 65000.0
    assert report["pillar2_external"]["orderbook"]["imbalance_ratio"] == 0.556
    assert "buy_standard" in report["guard_preview"]

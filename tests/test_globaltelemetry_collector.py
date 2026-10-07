"""Unit tests for GlobalTelemetryCollector."""

from __future__ import annotations

import json
import os
import time
from unittest.mock import MagicMock, patch

import pytest

from intelligence.globaltelemetry_collector import GlobalTelemetryCollector
from intelligence.external.collectors.orderbook_depth import OrderbookSnapshot
from intelligence.external.collectors.derivatives_telemetry import DerivativesTelemetry
from intelligence.external.collectors.whale_positioning import WhalePositioningSnapshot
from intelligence.sentiment.collectors.fear_greed_collector import FearGreedSnapshot
from intelligence.sentiment.collectors.market_breadth_collector import MarketBreadthSnapshot


@pytest.fixture
def temp_cache_dir(tmp_path):
    cache_dir = tmp_path / "cachedb"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir)


def test_collector_initialization(temp_cache_dir):
    collector = GlobalTelemetryCollector(
        symbols=["BTCUSDC", "TAOUSDC"],
        cache_dir=temp_cache_dir,
        orderbook_interval_sec=10.0,
        derivatives_interval_sec=20.0,
        whale_interval_sec=30.0,
        news_interval_sec=60.0,
        fear_greed_interval_sec=120.0,
        breadth_interval_sec=90.0,
    )
    assert collector.symbols == ["BTCUSDC", "TAOUSDC"]
    assert collector.orderbook_interval_sec == 10.0
    assert collector.derivatives_interval_sec == 20.0
    assert collector.whale_interval_sec == 30.0
    assert collector.news_interval_sec == 60.0
    assert collector.fear_greed_interval_sec == 120.0
    assert collector.breadth_interval_sec == 90.0


def test_collector_run_cycle_mocked(temp_cache_dir):
    collector = GlobalTelemetryCollector(
        symbols=["BTCUSDC"],
        cache_dir=temp_cache_dir,
    )

    mock_ob = OrderbookSnapshot(
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
    mock_fg = FearGreedSnapshot(
        value=65,
        classification="Greed",
        timestamp=time.time(),
        historical_values=(65, 60),
        trend_7d_change=5,
        is_extreme_fear=False,
        is_extreme_greed=False,
    )
    mock_breadth = MarketBreadthSnapshot(
        advance_ratio=0.62,
        advancing_count=217,
        declining_count=133,
        total_count=350,
        median_change_pct=1.45,
        mean_change_pct=1.20,
        dispersion_std=0.85,
        regime="BULLISH_BREADTH",
        top_gainers=(),
        top_losers=(),
        ts=time.time(),
    )

    with patch.object(collector.orderbook_collector, "fetch", return_value=mock_ob), \
         patch.object(collector.derivatives_collector, "fetch", return_value=mock_deriv), \
         patch.object(collector.whale_collector, "fetch", return_value=mock_whale), \
         patch.object(collector.news_collector, "fetch", return_value=MagicMock()), \
         patch.object(collector.fear_greed_collector, "fetch", return_value=mock_fg), \
         patch.object(collector.breadth_collector, "fetch", return_value=mock_breadth):

        res = collector.run_cycle(force=True)

        assert res["orderbook_updated"] == 1
        assert res["derivatives_updated"] == 1
        assert res["whale_updated"] == 1
        assert res["news_updated"] is True
        assert res["fear_greed_updated"] is True
        assert res["breadth_updated"] is True

        # Heartbeat verification
        hb_path = os.path.join(temp_cache_dir, "globaltelemetry_collector.heartbeat")
        assert os.path.exists(hb_path)
        with open(hb_path, "r", encoding="utf-8") as f:
            hb = json.load(f)
        assert hb["status"] == "alive"
        assert hb["symbols"] == ["BTCUSDC"]

        # Cache file verifications
        fg_path = os.path.join(temp_cache_dir, "fear_greed_cache.json")
        assert os.path.exists(fg_path)
        with open(fg_path, "r", encoding="utf-8") as f:
            fg_data = json.load(f)
        assert fg_data["value"] == 65
        assert fg_data["sentiment"] == "Greed"

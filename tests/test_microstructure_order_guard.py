"""Unit tests for microstructure_order_guard.py (Pillar 2)."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from microstructure_order_guard import (
    check_microstructure_order_guards,
    get_whale_collector,
    get_orderbook_collector,
    get_derivatives_collector,
)
from intelligence.external.collectors.orderbook_depth import OrderbookSnapshot
from intelligence.external.collectors.derivatives_telemetry import DerivativesTelemetry
from intelligence.external.collectors.whale_positioning import WhalePositioningSnapshot


def test_microstructure_non_buy():
    allowed, reason, scale = check_microstructure_order_guards(
        symbol="BTCUSDC",
        side="SELL",
        price=60000.0,
    )
    assert allowed is True
    assert reason == "not_buy_side"
    assert scale == 1.0


def test_microstructure_orderbook_wall_veto():
    wall_snap = OrderbookSnapshot(
        symbol="BTCUSDC",
        mid_price=60000.0,
        bid_depth_usd=200_000.0,
        ask_depth_usd=2_500_000.0,
        imbalance_ratio=0.074,
        largest_bid_wall_usd=50_000.0,
        largest_bid_wall_price=59900.0,
        largest_ask_wall_usd=2_000_000.0,
        largest_ask_wall_price=60100.0,
        ts=time.time(),
    )

    margins = {
        "orderbook_wall_guard_mode": "enforce",
        "whale_guard_mode": "off",
        "funding_guard_mode": "off",
        "min_buy_imbalance": 0.25,
        "whale_wall_usd_limit": 1_000_000.0,
    }

    with patch.object(get_orderbook_collector(), "fetch", return_value=wall_snap):
        allowed, reason, scale = check_microstructure_order_guards(
            symbol="BTCUSDC",
            side="BUY",
            price=60000.0,
            margins=margins,
        )
        assert allowed is False
        assert "ask_wall" in reason or "imbalance" in reason
        assert scale == 0.0


def test_microstructure_funding_crowding_downscale():
    crowded_snap = DerivativesTelemetry(
        symbol="BTCUSDC",
        funding_rate=0.0008,
        predicted_funding_rate=0.0008,
        open_interest=50000.0,
        open_interest_usd=4_250_000_000.0,
        ts=time.time(),
    )

    margins = {
        "funding_guard_mode": "enforce",
        "whale_guard_mode": "off",
        "orderbook_wall_guard_mode": "off",
        "funding_max_long_rate": 0.0005,
        "funding_crowding_policy": "downscale",
        "funding_crowded_scale": 0.50,
    }

    with patch.object(get_derivatives_collector(), "fetch", return_value=crowded_snap):
        allowed, reason, scale = check_microstructure_order_guards(
            symbol="BTCUSDC",
            side="BUY",
            price=60000.0,
            margins=margins,
        )
        assert allowed is True
        assert "long_crowding" in reason
        assert scale == 0.50

"""Unit tests for modular order guard pillars:
- Pillar 1: internal_order_guard
- Pillar 2: external_order_guard
- Pillar 3: intelligence_order_guard
- Pillar 4: geopolitical_order_guard
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from internal_order_guard import check_internal_order_guards
from external_order_guard import (
    check_external_order_guards,
    get_whale_collector,
    get_orderbook_collector,
    get_derivatives_collector,
)
from intelligence_order_guard import check_intelligence_order_guards
from geopolitical_order_guard import check_geopolitical_order_guards
from intelligence.macro.geopolitical_analyzer import (
    GeopoliticalThreatAssessment,
    GeopoliticalThreatAnalyzer,
)
from intelligence.external.collectors.orderbook_depth import OrderbookSnapshot
from intelligence.external.collectors.derivatives_telemetry import DerivativesTelemetry
from intelligence.external.collectors.whale_positioning import WhalePositioningSnapshot


# =====================================================================
# Pillar 1: Internal Order Guard Tests
# =====================================================================

def test_internal_guard_non_buy_order_passes():
    allowed, reason, scale = check_internal_order_guards(
        symbol="BTCUSDC",
        side="SELL",
        price=60000.0,
    )
    assert allowed is True
    assert reason == "not_buy_side"
    assert scale == 1.0


def test_internal_order_guards_off():
    allowed, reason, scale = check_internal_order_guards(
        symbol="BTCUSDC",
        side="BUY",
        price=60000.0,
        margins={"intelligence_guards_mode": "off"},
    )
    assert allowed is True
    assert reason == "intelligence_guards_off"
    assert scale == 1.0


def test_internal_parabolic_surge_guard():
    # Synthetic vertical parabolic spike: 50k -> 65k (30% surge)
    history = [
        (100.0, 50000.0),
        (200.0, 52000.0),
        (300.0, 65000.0),
    ]
    margins = {
        "intelligence_guards_mode": "enforce",
        "parabolic_surge_pct": 10.0,
        "parabolic_pullback_pct": 2.0,
    }

    allowed, reason, scale = check_internal_order_guards(
        symbol="BTCUSDC",
        side="BUY",
        price=65000.0,
        price_history=history,
        margins=margins,
    )
    assert allowed is False
    assert "parabolic_surge" in reason
    assert scale == 0.0


# =====================================================================
# Pillar 2: External Order Guard Tests
# =====================================================================

def test_external_microstructure_non_buy():
    allowed, reason, scale = check_external_order_guards(
        symbol="BTCUSDC",
        side="SELL",
        price=60000.0,
    )
    assert allowed is True
    assert reason == "not_buy_side"
    assert scale == 1.0


def test_external_microstructure_orderbook_wall_veto():
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
        allowed, reason, scale = check_external_order_guards(
            symbol="BTCUSDC",
            side="BUY",
            price=60000.0,
            margins=margins,
        )
        assert allowed is False
        assert "ask_wall" in reason or "imbalance" in reason
        assert scale == 0.0


def test_external_microstructure_funding_crowding_downscale():
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
        allowed, reason, scale = check_external_order_guards(
            symbol="BTCUSDC",
            side="BUY",
            price=60000.0,
            margins=margins,
        )
        assert allowed is True
        assert "long_crowding" in reason
        assert scale == 0.50


# =====================================================================
# Pillar 3: Intelligence Order Guard Composite Tests
# =====================================================================

def test_intelligence_non_buy_order_passes_immediately():
    allowed, reason, scale = check_intelligence_order_guards(
        provider="binance",
        symbol="BTCUSDC",
        order_type="SELL",
        price=60000.0,
    )
    assert allowed is True
    assert reason == "not_buy_side"
    assert scale == 1.0


def test_intelligence_geopolitical_veto_in_enforce_mode():
    mock_geo = GeopoliticalThreatAssessment(
        threat_level="CRITICAL_SHOCK",
        risk_score=0.95,
        summary="Severe regional conflict escalating.",
        recommended_brake="HARD_VETO_NEW_BUYS",
        headlines_analyzed=10,
        ts=time.time(),
    )

    margins = {
        "geopolitical_guard_mode": "enforce",
        "whale_guard_mode": "off",
        "orderbook_wall_guard_mode": "off",
        "funding_guard_mode": "off",
        "gemini_guard_mode": "off",
    }

    with patch.object(GeopoliticalThreatAnalyzer, "_load_from_disk", lambda self: setattr(self, "_cached_assessment", mock_geo)):
        allowed, reason, scale = check_intelligence_order_guards(
            provider="binance",
            symbol="BTCUSDC",
            order_type="BUY",
            price=60000.0,
            margins=margins,
        )
        assert allowed is False
        assert "veto_critical_geopolitical_shock" in reason
        assert scale == 0.0


def test_intelligence_geopolitical_downscale_in_enforce_mode():
    mock_geo = GeopoliticalThreatAssessment(
        threat_level="ELEVATED",
        risk_score=0.55,
        summary="Tensions rising at strategic energy corridor.",
        recommended_brake="DOWNSCALE_50",
        headlines_analyzed=8,
        ts=time.time(),
    )

    margins = {
        "geopolitical_guard_mode": "enforce",
        "whale_guard_mode": "off",
        "orderbook_wall_guard_mode": "off",
        "funding_guard_mode": "off",
        "gemini_guard_mode": "off",
    }

    with patch.object(GeopoliticalThreatAnalyzer, "_load_from_disk", lambda self: setattr(self, "_cached_assessment", mock_geo)):
        allowed, reason, scale = check_intelligence_order_guards(
            provider="binance",
            symbol="BTCUSDC",
            order_type="BUY",
            price=60000.0,
            margins=margins,
        )
        assert allowed is True
        assert "downscale_elevated_geopolitical_risk" in reason
        assert scale == 0.50


def test_collector_getters_reexported_from_external():
    assert get_whale_collector() is not None
    assert get_orderbook_collector() is not None
    assert get_derivatives_collector() is not None


# =====================================================================
# Pillar 4: Geopolitical Order Guard Tests
# =====================================================================

def test_geopolitical_non_buy():
    allowed, reason, scale = check_geopolitical_order_guards(
        symbol="BTCUSDC",
        side="SELL",
    )
    assert allowed is True
    assert reason == "not_buy_side"
    assert scale == 1.0


def test_geopolitical_guard_off():
    allowed, reason, scale = check_geopolitical_order_guards(
        symbol="BTCUSDC",
        side="BUY",
        margins={"geopolitical_guard_mode": "off"},
    )
    assert allowed is True
    assert reason == "geopolitical_guard_off"
    assert scale == 1.0


def test_geopolitical_shock_veto_enforce():
    mock_geo = GeopoliticalThreatAssessment(
        threat_level="CRITICAL_SHOCK",
        risk_score=0.95,
        summary="Major military conflict escalation.",
        recommended_brake="HARD_VETO_NEW_BUYS",
        headlines_analyzed=12,
        ts=time.time(),
    )

    with patch.object(GeopoliticalThreatAnalyzer, "_load_from_disk", lambda self: setattr(self, "_cached_assessment", mock_geo)):
        allowed, reason, scale = check_geopolitical_order_guards(
            symbol="BTCUSDC",
            side="BUY",
            margins={"geopolitical_guard_mode": "enforce"},
        )
        assert allowed is False
        assert "veto_critical_geopolitical_shock" in reason
        assert scale == 0.0


def test_geopolitical_shock_downscale_enforce():
    mock_geo = GeopoliticalThreatAssessment(
        threat_level="ELEVATED",
        risk_score=0.60,
        summary="Rising tensions in key maritime trade route.",
        recommended_brake="DOWNSCALE_50",
        headlines_analyzed=7,
        ts=time.time(),
    )

    with patch.object(GeopoliticalThreatAnalyzer, "_load_from_disk", lambda self: setattr(self, "_cached_assessment", mock_geo)):
        allowed, reason, scale = check_geopolitical_order_guards(
            symbol="BTCUSDC",
            side="BUY",
            margins={"geopolitical_guard_mode": "enforce"},
        )
        assert allowed is True
        assert "downscale_elevated_geopolitical_risk" in reason
        assert scale == 0.50

"""Unit tests for macro_order_guard.py."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from macro_order_guard import check_macro_order_guards, get_whale_collector, get_orderbook_collector, get_derivatives_collector
from intelligence.macro.geopolitical_analyzer import GeopoliticalThreatAssessment
from intelligence.external.collectors.whale_positioning import WhalePositioningSnapshot
from intelligence.external.collectors.orderbook_depth import OrderbookSnapshot
from intelligence.external.collectors.derivatives_telemetry import DerivativesTelemetry


def test_non_buy_order_passes_immediately():
    allowed, reason, scale = check_macro_order_guards(
        provider="binance",
        symbol="BTCUSDC",
        order_type="SELL",
        price=60000.0,
    )
    assert allowed is True
    assert reason == "not_buy_side"
    assert scale == 1.0


def test_geopolitical_veto_in_enforce_mode():
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

    from intelligence.macro.geopolitical_analyzer import GeopoliticalThreatAnalyzer
    with patch.object(GeopoliticalThreatAnalyzer, "_load_from_disk", lambda self: setattr(self, "_cached_assessment", mock_geo)):
        allowed, reason, scale = check_macro_order_guards(
            provider="binance",
            symbol="BTCUSDC",
            order_type="BUY",
            price=60000.0,
            margins=margins,
        )
        assert allowed is False
        assert "veto_critical_geopolitical_shock" in reason
        assert scale == 0.0


def test_geopolitical_downscale_in_enforce_mode():
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

    from intelligence.macro.geopolitical_analyzer import GeopoliticalThreatAnalyzer
    with patch.object(GeopoliticalThreatAnalyzer, "_load_from_disk", lambda self: setattr(self, "_cached_assessment", mock_geo)):
        allowed, reason, scale = check_macro_order_guards(
            provider="binance",
            symbol="BTCUSDC",
            order_type="BUY",
            price=60000.0,
            margins=margins,
        )
        assert allowed is True
        assert "downscale_elevated_geopolitical_risk" in reason
        assert scale == 0.50


def test_collector_getters():
    wc = get_whale_collector()
    oc = get_orderbook_collector()
    dc = get_derivatives_collector()

    assert wc is not None
    assert oc is not None
    assert dc is not None

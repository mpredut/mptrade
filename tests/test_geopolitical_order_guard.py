"""Unit tests for geopolitical_order_guard.py (Pillar 4)."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from geopolitical_order_guard import check_geopolitical_order_guards
from intelligence.macro.geopolitical_analyzer import GeopoliticalThreatAssessment


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

    from intelligence.macro.geopolitical_analyzer import GeopoliticalThreatAnalyzer
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

    from intelligence.macro.geopolitical_analyzer import GeopoliticalThreatAnalyzer
    with patch.object(GeopoliticalThreatAnalyzer, "_load_from_disk", lambda self: setattr(self, "_cached_assessment", mock_geo)):
        allowed, reason, scale = check_geopolitical_order_guards(
            symbol="BTCUSDC",
            side="BUY",
            margins={"geopolitical_guard_mode": "enforce"},
        )
        assert allowed is True
        assert "downscale_elevated_geopolitical_risk" in reason
        assert scale == 0.50

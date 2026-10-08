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

def test_external_microstructure_sell_and_stop_loss_exempt():
    # Small sell order (< 1000 EUR) passes with 0 ms overhead
    allowed, reason, scale = check_external_order_guards(
        symbol="BTCUSDC",
        side="SELL",
        price=60000.0,
        qty=0.01, # 600 EUR < 1000 EUR
    )
    assert allowed is True
    assert reason == "small_sell_exempt"
    assert scale == 1.0

    # Stop-loss order is explicitly exempt
    allowed, reason, scale = check_external_order_guards(
        symbol="BTCUSDC",
        side="SELL",
        price=60000.0,
        notional_eur=5000.0,
        is_stop_loss=True,
    )
    assert allowed is True
    assert reason == "stop_loss_exempt"
    assert scale == 1.0


def test_external_microstructure_large_sell_whale_support_defer():
    # Large sell order (5000 EUR >= 1000 EUR) facing massive $2M whale buy wall
    wall_snap = OrderbookSnapshot(
        symbol="BTCUSDC",
        mid_price=60000.0,
        bid_depth_usd=2_500_000.0,
        ask_depth_usd=200_000.0,
        imbalance_ratio=0.926,
        largest_bid_wall_usd=2_000_000.0,
        largest_bid_wall_price=59900.0,
        largest_ask_wall_usd=50_000.0,
        largest_ask_wall_price=60100.0,
        ts=time.time(),
    )

    margins = {
        "orderbook_wall_guard_mode": "enforce",
        "whale_guard_mode": "off",
        "funding_guard_mode": "off",
        "whale_wall_usd_limit": 1_000_000.0,
        "max_sell_imbalance": 0.75,
        "sell_guard_min_notional_eur": 1000.0,
    }

    with patch.object(get_orderbook_collector(), "fetch", return_value=wall_snap):
        allowed, reason, scale = check_external_order_guards(
            symbol="BTCUSDC",
            side="SELL",
            price=60000.0,
            notional_eur=5000.0,
            margins=margins,
        )
        assert allowed is False
        assert "whale_buy_wall_supporting" in reason or "bid_supported" in reason
        assert scale == 0.0


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

def test_intelligence_sell_and_stop_loss_exempt():
    # Small sell order (< 1000 EUR)
    allowed, reason, scale = check_intelligence_order_guards(
        provider="binance",
        symbol="BTCUSDC",
        order_type="SELL",
        price=60000.0,
        qty=0.01,
    )
    assert allowed is True
    assert reason == "small_sell_exempt"
    assert scale == 1.0

    # Stop-loss order
    allowed, reason, scale = check_intelligence_order_guards(
        provider="binance",
        symbol="BTCUSDC",
        order_type="SELL",
        price=60000.0,
        notional_eur=5000.0,
        is_stop_loss=True,
    )
    assert allowed is True
    assert reason == "stop_loss_exempt"
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


# =====================================================================
# Pillar Bypass & Re-Buy Guard Tests
# =====================================================================

def test_profit_guard_rebuy_bypasses_math_reference_but_enforces_microstructure():
    """Verify that RE-BUY (bypass_guards={'math_reference'}) skips mathematical comparison
    against historical 14-day sales, but actively blocks on microstructure ask walls / cascades."""
    import order_guard
    from intelligence.external.collectors.orderbook_depth import OrderbookDepthCollector

    class _MockProvider:
        name = "binance"
        def guards_internally(self):
            return False
        def get_orders(self, symbol, side, since_s):
            # Previous sell was at 200, current price is 250 (math reference would block a BUY at 250)
            return [{"price": 200.0, "qty": 1.0, "timestamp": time.time() * 1000 - 3600 * 1000}]

    p = _MockProvider()

    # With normal profit_guard (no bypass): blocked by math reference
    assert not order_guard.profit_guard(
        p, "TAOUSDC", "BUY", 250.0, 1.15, window_ref=200.0, qty=1.0,
    )

    # With bypass_guards={"math_reference"}: skips math reference, but checks microstructure
    # Simulate a severe ask wall / orderbook imbalance (seller domination)
    ob_collector = OrderbookDepthCollector()
    ob_collector._cache["TAOUSDC"] = OrderbookSnapshot(
        symbol="TAOUSDC",
        mid_price=250.0,
        bid_depth_usd=200_000.0,
        ask_depth_usd=2_500_000.0,
        imbalance_ratio=0.074,
        largest_bid_wall_usd=50_000.0,
        largest_bid_wall_price=249.0,
        largest_ask_wall_usd=2_000_000.0,
        largest_ask_wall_price=251.0,
        ts=time.time(),
    )

    with patch("external_order_guard.get_orderbook_collector", return_value=ob_collector), \
         patch("order_guard.get_orderbook_collector", return_value=ob_collector), \
         patch.dict(order_guard._load_margins(), {
             "external_guard_mode": "enforce",
             "microstructure_guard_mode": "enforce",
             "orderbook_wall_guard_mode": "enforce",
             "max_sell_imbalance": 0.75,
         }):
        # Blocked by microstructure despite math reference being bypassed!
        allowed = order_guard.profit_guard(
            p, "TAOUSDC", "BUY", 250.0, 1.15, window_ref=200.0, qty=1.0,
            bypass_guards={"math_reference"},
        )
        assert allowed is False

        # If Pillar 2 & Pillar 4 are ALSO bypassed, it goes through
        allowed_p2_bypassed = order_guard.profit_guard(
            p, "TAOUSDC", "BUY", 250.0, 1.15, window_ref=200.0, qty=1.0,
            bypass_guards={"math_reference", "p2", "p4"},
        )
        assert allowed_p2_bypassed is True


def test_profit_guard_all_bypasses_everything():
    """Verify emergency stop-loss / 'all' bypasses both math reference and all pillars."""
    import order_guard

    class _MockProvider:
        name = "binance"
        def guards_internally(self):
            return False

    allowed = order_guard.profit_guard(
        _MockProvider(), "BTCUSDC", "SELL", 50000.0, 1.15, window_ref=60000.0, qty=1.0,
        bypass_guards={"all"},
    )
    assert allowed is True


def test_whale_divergence_liquidation_cascade_blocks():
    """Verify that severe liquidation cascade (OI 1h change <= -5% with taker sell dominance)
    actively vetoes BUY entries to prevent catching falling knives."""
    from intelligence.external.guards.whale_divergence_guard import WhaleDivergenceGuard
    from intelligence.external.collectors.whale_positioning import WhalePositioningSnapshot

    guard = WhaleDivergenceGuard(
        block_liquidation_cascade=True,
        severe_cascade_oi_pct=-5.0,
        min_taker_ratio=0.65,
    )

    # Normal market pullback: OI drops -2%, taker ratio 0.95 -> Allowed
    normal_snap = WhalePositioningSnapshot(
        symbol="TAOUSDC",
        top_traders_long_ratio=1.1,
        top_traders_long_pct=0.55,
        open_interest_usd=100_000_000.0,
        open_interest_1h_change_pct=-2.0,
        taker_buy_sell_ratio=0.95,
        taker_buy_vol_usd=48_000_000.0,
        taker_sell_vol_usd=50_000_000.0,
        divergence_regime="neutral",
        ts=time.time(),
    )
    dec_normal = guard.check("TAOUSDC", "BUY", snapshot=normal_snap)
    assert dec_normal.allowed is True

    # Severe liquidation cascade: OI dumps -8.5% in 1 hour with taker buy/sell ratio at 0.72 -> BLOCKED!
    cascade_snap = WhalePositioningSnapshot(
        symbol="TAOUSDC",
        top_traders_long_ratio=0.8,
        top_traders_long_pct=0.45,
        open_interest_usd=100_000_000.0,
        open_interest_1h_change_pct=-8.5,
        taker_buy_sell_ratio=0.72,
        taker_buy_vol_usd=36_000_000.0,
        taker_sell_vol_usd=50_000_000.0,
        divergence_regime="long_liquidation",
        ts=time.time(),
    )
    dec_cascade = guard.check("TAOUSDC", "BUY", snapshot=cascade_snap)
    assert dec_cascade.allowed is False
    assert "liquidation_cascade_active" in dec_cascade.reason


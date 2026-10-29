"""Unit tests for modular order guard pillars:
- Pillar 1 (G_QLIMIT2): qlimit_order_guard
- Pillar 2 (GI_OBOOK / GI_WHALE / GI_FUNDING): external_order_guard
- Pillar 3 & 4 (GLLM1 / GLLM2 / GLLM3): llm_order_guard
"""

from __future__ import annotations

import time
from unittest.mock import patch

import order_guard
from qlimit_order_guard import check_qlimit2_order_guards as check_internal_order_guards
from external_order_guard import (
    check_external_order_guards,
    get_whale_collector,
    get_orderbook_collector,
    get_derivatives_collector,
)
from llm_order_guard import (
    check_geopolitical_order_guards,
    check_llm_order_guards as check_intelligence_order_guards,
)
from intelligence.macro.geopolitical_analyzer import (
    GeopoliticalThreatAssessment,
    GeopoliticalThreatAnalyzer,
)
from intelligence.external.collectors.orderbook_depth import OrderbookSnapshot
from intelligence.external.collectors.derivatives_telemetry import DerivativesTelemetry
from intelligence.external.collectors.whale_positioning import WhalePositioningSnapshot

_BASE_MARGINS = dict(order_guard._load_margins())


def _margins(**overrides):
    return {**_BASE_MARGINS, **overrides}


# =====================================================================
# Pillar 1: Internal Order Guard Tests
# =====================================================================

def test_internal_guard_non_buy_order_passes():
    # Stop-loss / bypass_guards={'all'} order is exempt
    allowed, reason, scale = check_internal_order_guards(
        symbol="BTCUSDC",
        side="SELL",
        price=60000.0,
        bypass_guards={"all"},
    )
    assert allowed is True
    assert reason == "stop_loss_exempt"
    assert scale == 1.0

    # Unsupported side passes
    allowed_hold, reason_hold, scale_hold = check_internal_order_guards(
        symbol="BTCUSDC",
        side="HOLD",
        price=60000.0,
    )
    assert allowed_hold is True
    assert reason_hold == "not_supported_side"
    assert scale_hold == 1.0


def test_internal_order_guards_off():
    allowed, reason, scale = check_internal_order_guards(
        symbol="BTCUSDC",
        side="BUY",
        price=60000.0,
        margins=_margins(internal_guard_mode="off", intelligence_guards_mode="off"),
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
    margins = _margins(
        internal_guard_mode="enforce",
        intelligence_guards_mode="enforce",
        parabolic_surge_pct=10.0,
        parabolic_pullback_pct=2.0,
    )

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
    # Unsupported side passes with 0 ms overhead
    allowed, reason, scale = check_external_order_guards(
        symbol="BTCUSDC",
        side="HOLD",
        price=60000.0,
        qty=0.01,
    )
    assert allowed is True
    assert reason == "not_supported_side"
    assert scale == 1.0

    # Stop-loss / bypass_guards={'all'} order is explicitly exempt
    allowed, reason, scale = check_external_order_guards(
        symbol="BTCUSDC",
        side="SELL",
        price=60000.0,
        notional_eur=5000.0,
        bypass_guards={"all"},
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

    margins = _margins(
        orderbook_wall_guard_mode="enforce",
        whale_guard_mode="off",
        funding_guard_mode="off",
        whale_wall_usd_limit=1_000_000.0,
        max_sell_imbalance=0.75,
        high_stake_min_notional_usd=700.0,
    )

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

    margins = _margins(
        orderbook_wall_guard_mode="enforce",
        whale_guard_mode="off",
        funding_guard_mode="off",
        min_buy_imbalance=0.25,
        whale_wall_usd_limit=1_000_000.0,
    )

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

    margins = _margins(
        funding_guard_mode="enforce",
        whale_guard_mode="off",
        orderbook_wall_guard_mode="off",
        funding_max_long_rate=0.0005,
        funding_crowding_policy="downscale",
        funding_crowded_scale=0.50,
    )

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
    # Small sell order (< high_stake_min_notional_usd = 700 USD) skips GLLM1 symmetrically (like small BUY) and returns 'ok'
    allowed, reason, scale = check_intelligence_order_guards(
        provider="binance",
        symbol="BTCUSDC",
        order_type="SELL",
        price=60000.0,
        qty=0.01,
    )
    assert allowed is True
    assert reason == "ok"
    assert scale == 1.0

    # Stop-loss / bypass_guards={'all'} order
    allowed, reason, scale = check_intelligence_order_guards(
        provider="binance",
        symbol="BTCUSDC",
        order_type="SELL",
        price=60000.0,
        notional_eur=5000.0,
        bypass_guards={"all"},
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

    margins = _margins(
        geopolitical_guard_mode="enforce",
        whale_guard_mode="off",
        orderbook_wall_guard_mode="off",
        funding_guard_mode="off",
        llm_guard_mode="off",
    )

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

    margins = _margins(
        geopolitical_guard_mode="enforce",
        whale_guard_mode="off",
        orderbook_wall_guard_mode="off",
        funding_guard_mode="off",
        llm_guard_mode="off",
    )

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
        margins=_margins(geopolitical_guard_mode="off"),
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
            margins=_margins(geopolitical_guard_mode="enforce"),
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
            margins=_margins(geopolitical_guard_mode="enforce"),
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
    ob_collector.record_snapshot(OrderbookSnapshot(
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
    ))

    with patch("external_order_guard.get_orderbook_collector", return_value=ob_collector), \
         patch("order_guard.get_orderbook_collector", return_value=ob_collector), \
         patch.dict(order_guard._load_margins(), {
             "external_guard_mode": "enforce",
             "orderbook_wall_guard_mode": "enforce",
             "max_sell_imbalance": 0.75,
         }):
        # Blocked by microstructure despite math reference being bypassed!
        allowed = order_guard.profit_guard(
            p, "TAOUSDC", "BUY", 250.0, 1.15, window_ref=200.0, qty=1.0,
            bypass_guards={"math_reference"},
        )
        assert allowed is False

        # If GI (microstructure) & GLLM3 (geopolitical) are ALSO bypassed, it goes through
        allowed_p2_bypassed = order_guard.profit_guard(
            p, "TAOUSDC", "BUY", 250.0, 1.15, window_ref=200.0, qty=1.0,
            bypass_guards={"math_reference", "gi", "GLLM3"},
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


def test_granular_bypass_p3_p4_avoids_unbound_local_error():
    from order_guard import check_intelligence_guards
    # Passing GI and GLLM bypass skips external and LLM guards and must NOT raise UnboundLocalError
    allowed, reason, scale = check_intelligence_guards(
        provider="binance",
        symbol="BTCUSDC",
        order_type="BUY",
        price=60000.0,
        qty=0.01,
        bypass_guards={"gi", "GLLM1", "GLLM3"},
    )
    assert allowed is True
    assert scale == 1.0


def test_internal_guard_mode_off_does_not_disable_other_pillars(monkeypatch):
    from order_guard import check_intelligence_guards
    # internal_guard_mode=off should not return 'intelligence_guards_off' globally
    import order_guard
    monkeypatch.setattr(
        order_guard,
        "_load_margins",
        lambda: _margins(internal_guard_mode="off", intelligence_guards_mode="enforce"),
    )
    allowed, reason, scale = check_intelligence_guards(
        provider="binance",
        symbol="BTCUSDC",
        order_type="BUY",
        price=60000.0,
        qty=0.01,
        bypass_guards={"G_QLIMIT2", "GLLM1", "GLLM3"},
    )
    # With intelligence_guards_mode=enforce, global check does NOT short-circuit
    assert reason != "intelligence_guards_off"


def test_pillars_fail_fast_on_missing_config_keys():
    import pytest
    from intelligence.sentiment.guards.post_stop_reentry_guard import get_post_stop_reentry_guard

    with pytest.raises(KeyError):
        check_internal_order_guards("BTCUSDC", "BUY", 60000.0, margins={})

    with pytest.raises(KeyError):
        check_external_order_guards("BTCUSDC", "BUY", 60000.0, margins={})

    with pytest.raises(KeyError):
        check_intelligence_order_guards("binance", "BTCUSDC", "BUY", 60000.0, margins={})

    with pytest.raises(KeyError):
        check_geopolitical_order_guards("BTCUSDC", "BUY", margins={})

    with pytest.raises(KeyError):
        get_post_stop_reentry_guard(margins={}, gemini_client=object())


def test_weibull_does_not_overlap_gaussian_horizon_and_computes_weibull_survival(monkeypatch):
    """Verify WeibullExhaustionGuard (G_QLIMIT2) uses true Weibull survival math and does not overlap
    with G_QLIMIT1's Gaussian Zone 1 & Zone 2 (0 .. 1.4*T)."""
    import qlimit_order_guard
    from intelligence.internal.guards.exhaustion_guard import WeibullExhaustionGuard

    guard = WeibullExhaustionGuard(default_p90_days=7.0, policy="downscale", exhausted_scale=0.25)

    # On TAOUSDC: T=8d -> gaussian_horizon_days = 11.2d. At 10.3d (inside Gauss Zone 2), Weibull must NOT downscale!
    dec_in_zone2 = guard.check(
        "TAOUSDC",
        "SELL",
        trend_duration_seconds=10.3 * 86400,
        median_days=3.0,
        trend_direction="bear",
        gaussian_horizon_days=11.2,
    )
    assert dec_in_zone2.allowed is True
    assert dec_in_zone2.suggested_scale == 1.0
    assert "trend_healthy" in dec_in_zone2.reason
    assert "weibull_survival" in dec_in_zone2.metadata

    # Beyond 11.2d (e.g. 12.5d, Zone 3 tail), Weibull downscales to 0.25 with S_weibull < 0.10
    dec_zone3 = guard.check(
        "TAOUSDC",
        "SELL",
        trend_duration_seconds=12.5 * 86400,
        median_days=3.0,
        trend_direction="bear",
        gaussian_horizon_days=11.2,
    )
    assert dec_zone3.allowed is True
    assert dec_zone3.suggested_scale == 0.25
    assert "trend_exhausted" in dec_zone3.reason
    assert dec_zone3.metadata["weibull_survival"] < 0.10

    # Verify check_qlimit2_order_guards resolves cached Gaussian horizon when G_QLIMIT1 is active
    monkeypatch.setattr(qlimit_order_guard, "_resolve_trend_duration", lambda _s: 10.3 * 86400)
    monkeypatch.setattr(qlimit_order_guard, "_resolve_trend_direction", lambda _s, **_k: "bear")
    monkeypatch.setattr(qlimit_order_guard, "resolve_cached_gaussian_horizon", lambda _s, **_k: (6.6, 3.0, 11.2))
    m = _margins(
        internal_guard_mode="enforce",
        weibull_exhaustion_policy="downscale",
        weibull_exhausted_scale=0.25,
        weibull_p90_days=7.0,
        high_stake_min_notional_usd=700.0,
    )
    ok, reason, scale = qlimit_order_guard.check_qlimit2_order_guards(
        "TAOUSDC", "SELL", 250.0, qty=4.0, notional_eur=1000.0, price_history=[], margins=m
    )
    assert ok is True
    assert scale == 1.0
    assert reason == "ok"


def test_weibull_uses_symbol_p90_days_when_qlimit1_is_bypassed(monkeypatch):
    """When G_QLIMIT1 is bypassed, G_QLIMIT2 must use the symbol's empirical p90_days (e.g. 6.7d)
    from resolve_cached_gaussian_horizon rather than falling back to default_p90_days (10.0d)."""
    import qlimit_order_guard

    m = dict(order_guard._load_margins())
    m["internal_guard_mode"] = "enforce"
    m["weibull_exhaustion_policy"] = "downscale"
    m["weibull_exhausted_scale"] = "0.25"
    m["weibull_p90_days"] = "10.0"

    # Symbol has empirical p90_d = 6.7d, median_d = 2.4d, gauss_horizon = 9.38d
    monkeypatch.setattr(
        qlimit_order_guard,
        "resolve_cached_gaussian_horizon",
        lambda sym: (6.7, 2.4, 9.38),
    )
    # Duration = 8.0 days (> p90_d=6.7d, < default_p90=10.0d, < gauss_horizon=9.38d)
    monkeypatch.setattr(
        qlimit_order_guard,
        "_resolve_trend_duration",
        lambda sym: 8.0 * 86400.0,
    )
    monkeypatch.setattr(
        qlimit_order_guard,
        "_resolve_trend_direction",
        lambda sym, regime_context=None: "bull",
    )

    # 1. With G_QLIMIT1 active, limit_p90 = max(6.7, 9.38) = 9.38d -> 8.0d is allowed at 1.0 scale
    ok_active, _, scale_active = qlimit_order_guard.check_qlimit2_order_guards(
        "BTCUSDC", "BUY", 65000.0, qty=0.01, notional_eur=650.0, price_history=[], margins=m
    )
    assert ok_active is True
    assert scale_active == 1.0

    # 2. With G_QLIMIT1 bypassed, limit_p90 = 6.7d (empirical p90_d) -> 8.0d > 6.7d is downscaled to 0.25
    ok_bp, reason_bp, scale_bp = qlimit_order_guard.check_qlimit2_order_guards(
        "BTCUSDC",
        "BUY",
        65000.0,
        qty=0.01,
        notional_eur=650.0,
        price_history=[],
        margins=m,
        bypass_guards={"G_QLIMIT1"},
    )
    assert ok_bp is True
    assert scale_bp == 0.25
    assert "P90=6.7d" in reason_bp


def test_normalize_bypassed_guards_accepts_frozenset():
    """Verify normalize_bypassed_guards handles frozenset inputs (e.g. AUTONOMOUS_BYPASS_GUARDS)."""
    from guard_hierarchy import GUARD_REGISTRY, normalize_bypassed_guards
    from orchestratorTrade.autonomous_intent_gateway import AUTONOMOUS_BYPASS_GUARDS

    assert isinstance(AUTONOMOUS_BYPASS_GUARDS, frozenset)
    resolved = normalize_bypassed_guards(bypass_guards=AUTONOMOUS_BYPASS_GUARDS)
    assert "all" in resolved
    assert GUARD_REGISTRY["GO1"].is_bypassed(resolved) is True
    assert GUARD_REGISTRY["G_PROF"].is_bypassed(resolved) is True
    assert GUARD_REGISTRY["G_QLIMIT1"].is_bypassed(resolved) is True
    assert GUARD_REGISTRY["GLLM1"].is_bypassed(resolved) is True


def test_gllm1_bypass_does_not_leak_to_gllm2_or_gllm3():
    """Bug 4 regression: bypassing GLLM1 individually must NOT bypass GLLM2 or GLLM3 via category collision,
    even when normalize_bypassed_guards is called multiple times (idempotent), whereas bypassing
    category 'llm' or 'gllm' bypasses all three."""
    from guard_hierarchy import GUARD_REGISTRY, normalize_bypassed_guards

    only_gllm1 = normalize_bypassed_guards(bypass_guards={"GLLM1"})
    assert "llm" in only_gllm1
    assert "gllm3" not in only_gllm1
    assert GUARD_REGISTRY["GLLM1"].is_bypassed(only_gllm1) is True
    assert GUARD_REGISTRY["GLLM2"].is_bypassed(only_gllm1) is False
    assert GUARD_REGISTRY["GLLM3"].is_bypassed(only_gllm1) is False

    # Re-normalizing must be idempotent and still not leak into GLLM2/GLLM3
    renormalized = normalize_bypassed_guards(bypass_guards=only_gllm1)
    assert renormalized == only_gllm1
    assert GUARD_REGISTRY["GLLM3"].is_bypassed(renormalized) is False

    all_llm = normalize_bypassed_guards(bypass_guards={"llm"})
    assert GUARD_REGISTRY["GLLM1"].is_bypassed(all_llm) is True
    assert GUARD_REGISTRY["GLLM2"].is_bypassed(all_llm) is True
    assert GUARD_REGISTRY["GLLM3"].is_bypassed(all_llm) is True


def test_batch38b_summary_store_visible_dir_topics_intent_gateway_and_geo_guard(tmp_path, monkeypatch):
    """Verify summary_store visible_conversations_dir handling, topics.get_topic_registry_path, AutonomousIntentGateway string wallet balance & status normalization, and GeopoliticalShockGuard BID/LONG/percentage downscale."""
    import sqlite3
    from types import SimpleNamespace
    from orchestratorOS.agy import summary_store, topics
    from orchestratorTrade.autonomous_intent_gateway import AutonomousIntentGateway
    from intelligence.macro.geopolitical_guard import GeopoliticalShockGuard

    # 1. summary_store respects visible_conversations_dir for antigravity-cli store layout
    gem_dir = tmp_path / ".gemini"
    cli_base = gem_dir / "antigravity-cli"
    vis_dir = gem_dir / "antigravity" / "conversations"
    (cli_base / "conversations").mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    db_path = cli_base / "conversation_summaries.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE conversation_summaries (conversation_id TEXT PRIMARY KEY, title TEXT, step_count INTEGER, status TEXT)")
    conn.execute("INSERT INTO conversation_summaries VALUES ('cid_vis', 'Visible Thread', 5, 'CASCADE_RUN_STATUS_IDLE')")
    conn.execute("INSERT INTO conversation_summaries VALUES ('cid_orphan', 'Orphan Thread', 1, 'CASCADE_RUN_STATUS_IDLE')")
    conn.commit()
    conn.close()

    vis_db = vis_dir / "cid_vis.db"
    vis_db.write_bytes(b"sqlite_dummy_data")
    assert summary_store.get_thread_step_count("cid_vis", str(cli_base)) == (True, 5, "CASCADE_RUN_STATUS_IDLE")
    removed = summary_store.clean_orphan_summaries(str(cli_base), dry_run=False)
    assert removed == 1
    assert summary_store.read_summary_row(str(cli_base), "cid_vis") is not None
    assert summary_store.read_summary_row(str(cli_base), "cid_orphan") is None

    cnt, freed = summary_store.delete_threads({"cid_vis"}, str(cli_base), dry_run=False)
    assert cnt == 1 and freed == len(b"sqlite_dummy_data")
    assert not vis_db.exists()
    assert summary_store.read_summary_row(str(cli_base), "cid_vis") is None

    # 2. topics.get_topic_registry_path respects MPTRADE_CACHEDB_DIR when LLM_TOPIC_REGISTRY_PATH is unset
    monkeypatch.delenv("LLM_TOPIC_REGISTRY_PATH", raising=False)
    monkeypatch.setenv("MPTRADE_CACHEDB_DIR", str(tmp_path))
    assert topics.get_topic_registry_path() == str(tmp_path / "llm_topic_threads.json")

    # 3. AutonomousIntentGateway._clamp_notional_to_wallet_balance parses numeric string balances and normalizes lowercase status
    gw = AutonomousIntentGateway(workspace_dir=str(tmp_path), mode="shadow", max_notional_usd=500.0, min_confidence=0.80, notify_enabled=False)
    inst_zero = SimpleNamespace(
        symbol="BTCUSDC",
        base="BTC",
        quote="USDC",
        provider_name="binance",
        _provider=SimpleNamespace(free_balance=lambda asset: "0.0"),
    )
    ok_bal, clamped_usd, reason_bal = gw._clamp_notional_to_wallet_balance(inst_zero, "BUY", 60000.0, 200.0)
    assert ok_bal is False and clamped_usd == 0.0 and "Insufficient" in reason_bal

    inst_neg = SimpleNamespace(
        symbol="BTCUSDC",
        base="BTC",
        quote="USDC",
        provider_name="binance",
        _provider=SimpleNamespace(free_balance=lambda asset: -1.0),
    )
    ok_neg, clamped_neg, _ = gw._clamp_notional_to_wallet_balance(inst_neg, "BUY", 60000.0, 200.0)
    assert ok_neg is True and clamped_neg == 200.0

    gw.save_all_intents([{"intent_id": "i1", "status": " pending ", "decision": "HOLD"}])
    assert len(gw.get_pending_intents()) == 1

    # 4. GeopoliticalShockGuard normalizes percentage downscale_factor and checks BID/LONG
    gsg = GeopoliticalShockGuard(downscale_factor=40.0)
    assert gsg.downscale_factor == 0.40
    elev_assess = GeopoliticalThreatAssessment(
        threat_level="ELEVATED",
        risk_score=0.6,
        summary="Tensions",
        recommended_brake="DOWNSCALE_50",
        headlines_analyzed=3,
        ts=1000.0,
    )
    dec_long = gsg.check("BTCUSDC", "LONG", elev_assess)
    assert dec_long.allowed is True and dec_long.suggested_scale == 0.40


def test_batch39_composite_daemon_and_cli_hardening(tmp_path, monkeypatch):
    """Verify CompositeMarketIntelligence side alias & percentage scale normalization, IntelligenceTelemetryDaemon update_macro now propagation, and intelligence.cli local cache_T_trend lookup."""
    import json
    from types import SimpleNamespace
    from intelligence.composite import CompositeMarketIntelligence
    from intelligence.daemon import IntelligenceTelemetryDaemon
    from intelligence.cli import inspect_symbol_intelligence
    from intelligence.internal.guards.guard_decision import BrakeAction, GuardDecision

    # 1. CompositeMarketIntelligence.evaluate_guards normalizes LONG/BID/SHORT/ASK and percentage scale
    comp = CompositeMarketIntelligence()
    elev_assess = GeopoliticalThreatAssessment(
        threat_level="ELEVATED",
        risk_score=0.6,
        summary="Tensions",
        recommended_brake="DOWNSCALE_50",
        headlines_analyzed=3,
        ts=1000.0,
    )
    dec_long = comp.evaluate_guards("BTCUSDC", "LONG", 60000.0, geopolitical_assessment=elev_assess)
    assert dec_long.allowed is True and dec_long.suggested_scale == 0.50

    monkeypatch.setattr(
        comp.geopolitical_guard,
        "check",
        lambda sym, side, assess: SimpleNamespace(
            guard_name="GeopoliticalShockGuard",
            allowed=True,
            brake_action=BrakeAction.DOWNSCALE_QTY,
            suggested_scale=35.0,
            reason="pct_scale",
            metadata={},
        ),
    )
    dec_pct = comp.evaluate_guards("BTCUSDC", "BID", 60000.0, geopolitical_assessment=elev_assess)
    assert dec_pct.allowed is True and dec_pct.suggested_scale == 0.35

    # 2. IntelligenceTelemetryDaemon.run_cycle propagates now to update_macro -> news_collector.fetch & geopolitical_analyzer.assess
    daemon = IntelligenceTelemetryDaemon(symbols=["BTCUSDC"], cache_dir=str(tmp_path), skip_external=True, skip_macro=False)
    captured_now = {}

    def _fake_fetch(*, force_refresh=False, now_ts=None):
        captured_now["fetch_now"] = now_ts
        return SimpleNamespace(headlines=(), has_critical_shock_keywords=False)

    def _fake_assess(snap, *, force_refresh=False, now_ts=None):
        captured_now["assess_now"] = now_ts
        return elev_assess

    monkeypatch.setattr(daemon.news_collector, "fetch", _fake_fetch)
    monkeypatch.setattr(daemon.geopolitical_analyzer, "assess", _fake_assess)
    res = daemon.run_cycle(force=True, now=12345.0)
    assert res["macro_updated"] is True
    assert captured_now["fetch_now"] == 12345.0
    assert captured_now["assess_now"] == 12345.0

    # 3. inspect_symbol_intelligence reads cache_T_trend.json from cache_dir before falling back to estimate_T
    (tmp_path / "cache_T_trend.json").write_text(
        json.dumps({"BTCUSDC": {"T": 11, "T_emp": 12.5, "median_d": 4.2, "p90_d": 9.1, "n": 42, "ts": 1000.0}}),
        encoding="utf-8",
    )
    monkeypatch.setattr("order_guard.check_intelligence_guards", lambda *a, **kw: (True, "ok", 1.0))
    rep = inspect_symbol_intelligence("BTC/USDC", cache_dir=str(tmp_path))
    assert rep["pillar1_internal"]["trend_survival"]["T"] == 11
    assert rep["pillar1_internal"]["trend_survival"]["P90_days"] == 9.1
    assert rep["pillar1_internal"]["trend_survival"]["samples"] == 42


def test_batch40_market_monitor_and_notify_engine_hardening(tmp_path, monkeypatch):
    """Verify alerts_config case-insensitive keys, new_coins_discovery timestamp/string coercion, pricefetcher CMC .previous recovery, and alertnotifiers delivery state .previous recovery."""
    import json
    import time
    from unittest.mock import MagicMock
    from market_monitor.alerts_config import load_config, resolve
    from market_monitor.new_coins_discovery import (
        BinanceNewListingsSource,
        CoinMarketCapSource,
        DexScreenerSource,
        NewCoinsFactory,
        NewCoinsMonitor,
    )
    from market_monitor.pricefetcher import CoinMarketCapPricePlatform
    from notify_engine.alertnotifiers import _load_delivery_state, _save_delivery_state

    # 1. alerts_config.load_config normalizes mixed-case keys & resolve handles lowercase per_coin keys and float strings
    conf_file = tmp_path / "alerts.conf"
    conf_file.write_text(
        "\n".join([
            "WATCH = BTC, TAO",
            "Sources = coinmarketcap",
            "Discover_New_Coins = ON",
            "DEFAULT = 4.5 / 8.0",
            "NEW_COIN = 11.0 / 22.0",
            "btc = 3.0 / 6.0",
            "Cooldown_Minutes = 30",
            "Lookback_Hours = 24",
            "Max_Monitored = 15",
            "Max_New_Coins = 5",
            "New_Coins_Scan_Seconds = 1800",
            "Price_Scan_Seconds = 60",
        ]),
        encoding="utf-8",
    )
    cfg = load_config(str(conf_file))
    assert cfg["discover_new_coins"] is True
    assert cfg["watch"] == ["BTC", "TAO"]
    assert cfg["alert_config"]["default"] == {"up_percent": 4.5, "down_percent": 8.0}
    assert cfg["alert_config"]["dynamic"] == {"up_percent": 11.0, "down_percent": 22.0}
    assert resolve({"per_coin": {"btc": {"up_percent": 3.0, "down_percent": 6.0}}}, "BTCUSDC", "0.0") == {
        "up_percent": 3.0,
        "down_percent": 6.0,
    }
    assert resolve(cfg["alert_config"], "SOLUSDC", "1.0") == {"up_percent": 11.0, "down_percent": 22.0}

    # 2. new_coins_discovery sources handle numeric timestamps, ISO strings, and booleans without crashing
    now_ts = time.time()
    cmc_src = CoinMarketCapSource.__new__(CoinMarketCapSource)
    cmc_src._cache = [
        {"symbol": "CMCA", "added_at": now_ts},
        {"symbol": "CMCB", "added_at": True},
    ]
    assert [c["symbol"] for c in cmc_src.get_new_coins(days_back=7)] == ["CMCA"]

    bin_src = BinanceNewListingsSource.__new__(BinanceNewListingsSource)
    bin_src._new_listings = [
        {"symbol": "BINA", "added_at": now_ts * 1000.0},
        {"symbol": "BINB", "added_at": "2099-01-01T00:00:00Z"},
        {"symbol": "BINC", "added_at": False},
    ]
    assert [c["symbol"] for c in bin_src.get_new_coins(days_back=7)] == ["BINA", "BINB"]

    dex_src = DexScreenerSource.__new__(DexScreenerSource)
    dex_src._cache = [
        {"symbol": "DEXA", "added_at": now_ts},
        {"symbol": "DEXB", "added_at": "2099-01-01"},
    ]
    assert [c["symbol"] for c in dex_src.get_new_coins(days_back=7)] == ["DEXA", "DEXB"]

    factory = MagicMock(spec=NewCoinsFactory)
    factory.get_all_new_coins.return_value = {
        "coinmarketcap": [{"symbol": "  newx  ", "name": "NewX", "slug": "newx"}, "not_a_dict"],
        "broken": None,
    }
    mon = NewCoinsMonitor(factory=factory)
    assert "NEWX" in mon.all_symbols

    # 3. CoinMarketCapPricePlatform recovers corrupted cmc_listings_cache.json from .previous
    cmc_plat = CoinMarketCapPricePlatform(api_key=None)
    cmc_cache = tmp_path / "cmc_listings_cache.json"
    cmc_plat._cache_file = str(cmc_cache)
    cmc_cache.write_text("{corrupt", encoding="utf-8")
    (tmp_path / "cmc_listings_cache.json.previous").write_text(
        json.dumps({"tao": {"id": 1, "name": "Bittensor", "slug": "bittensor"}}),
        encoding="utf-8",
    )
    assert cmc_plat._load_symbols_from_disk_cache(max_age_s=86400) is True
    assert "TAO" in cmc_plat._supported_symbols

    # 4. alertnotifiers._load_delivery_state recovers from .previous when primary is corrupted
    deliv_path = tmp_path / "notification_delivery_state.json"
    _save_delivery_state(deliv_path, {"schema_version": 1, "date_utc": "2026-10-27", "channels": {"ntfy": {"sent": 7}}})
    (tmp_path / "notification_delivery_state.json.previous").write_text(
        deliv_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    deliv_path.write_text("{corrupt", encoding="utf-8")
    recovered = _load_delivery_state(deliv_path, "2026-10-27")
    assert recovered["channels"]["ntfy"]["sent"] == 7


def test_batch41_verify_tools_drawprices_and_tradeall_backtest_hardening(tmp_path, monkeypatch):
    import json
    from unittest.mock import patch, MagicMock
    from verify_tools import portfolio_snapshot, kraken_pnl_backfill
    from visuals import drawprices
    from offline.backtests import tradeall as bt_tradeall

    # 1. portfolio_snapshot._ntfy respects DISABLE_EXTERNAL_NOTIFICATIONS and strips CR/LF in title
    monkeypatch.setenv("NTFY_TOPIC", "https://ntfy.sh/custom_topic")
    monkeypatch.setenv("DISABLE_EXTERNAL_NOTIFICATIONS", "1")
    with patch("urllib.request.urlopen") as mock_urlopen:
        portfolio_snapshot._ntfy("Title\r\nInjected: 1", "Body")
        mock_urlopen.assert_not_called()

    monkeypatch.setenv("DISABLE_EXTERNAL_NOTIFICATIONS", "0")
    captured_req = {}

    def _fake_urlopen(req, timeout=10):
        captured_req["url"] = req.full_url
        captured_req["title"] = req.headers.get("Title")
        resp = MagicMock()
        resp.__enter__.return_value = resp
        return resp

    with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        portfolio_snapshot._ntfy("Title\r\nLine2", "Body")
        assert captured_req["url"] == "https://ntfy.sh/custom_topic"
        assert "\r" not in captured_req["title"] and "\n" not in captured_req["title"]

    # 2. kraken_pnl_backfill.fetch_all_trades falls back to len(all_trades) when count is omitted
    fake_kc = MagicMock()
    fake_kc._private.side_effect = [
        {"trades": {"T1": {"time": 100.0}, "T2": {"time": 101.0}}},
        {"trades": {}},
    ]
    trades, total = kraken_pnl_backfill.fetch_all_trades(fake_kc)
    assert len(trades) == 2
    assert total == 2

    # 3. drawprices.forecast_lstm sanitizes None/bool/NaN entries without crashing
    with patch.object(drawprices, "try_import_tf", return_value=(True, MagicMock(), MagicMock(), MagicMock(), MagicMock())):
        preds, err = drawprices.forecast_lstm([100.0, None, True, float("nan")], horizon=3, window=5)
        assert preds is None
        assert "too short" in err

    # 4. offline/backtests/tradeall.py load_ticks_cache24 recovers from .previous when primary is corrupt
    cache_dir = tmp_path / "cache24"
    cache_dir.mkdir(parents=True, exist_ok=True)
    primary_cache = cache_dir / "cache_24_price_BTCUSDC.json"
    prev_cache = cache_dir / "cache_24_price_BTCUSDC.json.previous"
    primary_cache.write_text("{corrupt", encoding="utf-8")
    prev_cache.write_text(
        json.dumps({"items": {"BTCUSDC": [[1700000000000, 65000.0], [1700000060000, 65100.0]]}}),
        encoding="utf-8",
    )
    ticks = list(bt_tradeall.load_ticks_cache24("BTCUSDC", 0.0, 2000000000.0, filename=str(primary_cache)))
    assert len(ticks) == 2
    assert ticks[0][1] == 65000.0


def test_batch42_mailer_server_pricechecker_and_execution_calibration_hardening(tmp_path, monkeypatch):
    from unittest.mock import patch, MagicMock
    from notify_engine import mailer
    from notify_engine.server import NotificationServer
    from market_monitor.pricechecker import PriceAlert, PriceChecker
    from offline.backtests.execution_calibration import calibrate_execution_events

    # 1. mailer.send_email strips CR/LF from subject before building MIMEText headers
    monkeypatch.setenv("DISABLE_EXTERNAL_NOTIFICATIONS", "0")
    monkeypatch.setenv("BACKTEST_MODE", "0")
    monkeypatch.setenv("REPLAY_MODE", "0")
    sent_msgs = []
    with (
        patch.object(mailer, "_config", return_value={"server": "smtp.example.com", "port": 587, "username": "u@e.com", "password": "pw", "to": "a@e.com"}),
        patch.object(mailer, "_smtp_send", side_effect=lambda msg, cfg: sent_msgs.append(msg)),
    ):
        assert mailer.send_email("Alert\r\nBcc: injected@e.com", "Body text", dedup=False) is True
        assert len(sent_msgs) == 1
        assert "\r" not in sent_msgs[0]["Subject"] and "\n" not in sent_msgs[0]["Subject"]
        assert "Alert Bcc: injected@e.com" == sent_msgs[0]["Subject"]
        # Ensure msg.as_string() does not raise HeaderWriteError
        assert "Subject: Alert Bcc: injected@e.com" in sent_msgs[0].as_string()

    # 2. NotificationServer._enqueue_payload preserves __orchestrator_intent__ even if data has that key
    qfile = tmp_path / "queue.jsonl"
    monkeypatch.setenv("NOTIFICATION_QUEUE_FILE", str(qfile))
    srv = NotificationServer()
    srv._enqueue_payload("ntfy_webhook", {"__orchestrator_intent__": None, "title": "Test"})
    import json
    enqueued = json.loads(qfile.read_text(encoding="utf-8").strip())
    assert enqueued["__orchestrator_intent__"] == "ntfy_webhook"

    # 3. PriceAlert and PriceChecker handle bools and case-insensitive per_coin/symbol_added_time
    pa = PriceAlert(symbol=True, alert_type="RISE", current_price=10.0, reference_price=9.0, percent_change=11.1, threshold=5.0, url=False)
    assert pa.symbol == "N/A"
    assert pa.alert_type == "up"
    assert pa.url == ""

    fake_cache = MagicMock()
    fake_cache.symbol_added_time = {"newcoin": 1700000000.0}
    checker = PriceChecker(fake_cache, config={"per_coin": {"tao": {"up_percent": 6.5, "down_percent": 9.5}}})
    assert checker._is_dynamic_symbol("NEWCOIN") is True
    assert checker._get_thresholds_for_symbol("TAOUSDC") == {"up_percent": 6.5, "down_percent": 9.5}

    # 4. calibrate_execution_events normalizes whitespace/alias sides, ms timestamps, and ignores bool intent_id
    res = calibrate_execution_events([
        {"intent_id": True, "event": "submit_requested", "ts": 1000.0},
        {"intent_id": "id-1", "event": "submit_requested", "ts": 1700000000000, "side": " BUY ", "qty": 2.0, "price": 100.0, "market": True, "reference_price": 100.0},
        {"intent_id": "id-1", "event": "submit_accepted", "ts": 1700000001000},
        {"intent_id": "id-1", "event": "order_terminal", "ts": 1700000003500, "filled_qty": 2.0, "cost": 202.0, "fee": 0.2},
    ])
    assert len(res["orders"]) == 1
    ord0 = res["orders"][0]
    import pytest
    assert ord0["first_fill_latency_s"] == pytest.approx(2.5)
    assert ord0["limit_fill_deviation_bps"] == pytest.approx(100.0)
    assert ord0["market_execution_shortfall_bps"] == pytest.approx(100.0)


def test_batch43_offline_runners_hardening(tmp_path):
    import json
    import pytest
    from offline.runners import (
        fetch_hyperliquid_candles as fhc,
        calibrate_execution_audit as cea,
        kraken_financial_benchmark as kfb,
    )

    # 1. fetch_hyperliquid_candles.resolve_spot_pair strips whitespace and rejects bools
    meta = {
        "tokens": [{"name": " hype ", "index": 150}, {"name": " usdc ", "index": 0}],
        "universe": [{"name": " @107 ", "tokens": [150, 0]}],
    }
    assert fhc.resolve_spot_pair(" HYPE ", " USDC ", meta) == "@107"
    with pytest.raises(ValueError):
        fhc.resolve_spot_pair(True, "USDC", meta)

    # 2. fetch_hyperliquid_candles.validate_continuity rejects non-sequences and bool timestamps
    with pytest.raises(ValueError):
        fhc.validate_continuity(None, 60)
    with pytest.raises(ValueError):
        fhc.validate_continuity([{"timestamp": True}, {"timestamp": 3600}], 60)

    # 3. calibrate_execution_audit._load normalizes venue whitespace and ignores bool venue
    audit_file = tmp_path / "execution_audit_test.jsonl"
    audit_file.write_text(
        "\n".join([
            json.dumps({"venue": " Kraken ", "intent_id": "i1"}),
            json.dumps({"venue": True, "intent_id": "i2"}),
        ]),
        encoding="utf-8",
    )
    loaded = cea._load([audit_file], " kraken ")
    assert len(loaded) == 1
    assert loaded[0]["intent_id"] == "i1"

    # 4. kraken_financial_benchmark.markdown_report handles malformed total_test_bars / interval_minutes
    md = kfb.markdown_report({
        "candidate_name": "test_cand",
        "scenarios": {
            "central": {
                "aggregate": {
                    "mean_return_pct": 1.2,
                    "mean_net_pnl_usd_per_window": 12.0,
                    "sum_reset_net_pnl_usd": 120.0,
                    "worst_return_pct": -0.5,
                    "worst_max_drawdown_pct": 2.0,
                    "mean_buy_hold_return_pct": 0.8,
                    "worst_cvar_95_pct": -1.1,
                    "mean_exposure_pct": 40.0,
                    "positive_windows": 8,
                    "window_count": 10,
                    "total_test_bars": "90.0",
                    "regimes": {},
                }
            }
        },
        "walk_forward": {"interval_minutes": True},
    })
    assert "The 90 TEST bars mean 15 days" in md


def test_batch44_verify_tools_and_order_outcomes_log_hardening(tmp_path, monkeypatch):
    import enum
    import subprocess
    import order_outcomes_log as ool
    import verify_tools.ownership_inventory as oi

    # 1. ownership_inventory._truthy strips inline comments & accepts "1.0"
    assert oi._truthy("true # inline comment", default=False) is True
    assert oi._truthy("1.0", default=False) is True
    assert oi._truthy(" # comment only ", default=True) is True
    assert oi._truthy(False, default=True) is False

    # 2. ownership_inventory._commands handles subprocess errors cleanly
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **kw: (_ for _ in ()).throw(subprocess.TimeoutExpired(cmd="ps", timeout=10)),
    )
    assert oi._commands() == []

    # 3. order_outcomes_log unwraps Enum values in regime_context while still rejecting primitives
    class _RegimeEnum(enum.Enum):
        BULL = "BULL_TREND"

    monkeypatch.setattr(ool, "ORDER_OUTCOMES_LOG_DIR", str(tmp_path))
    ool.log_order_outcome(
        symbol="TAOUSDC",
        side="BUY",
        price=350.0,
        qty=1.0,
        outcome="accepted",
        refuse_reason="",
        motivation="test",
        caller="unit_test",
        regime_context={
            "regime": _RegimeEnum.BULL,
            "strength": 0.75,
            "source": "cache",
            "fresh": True,
            "fallback_used": False,
            "reason": 12345,  # non-string primitive should stay ""
        },
    )
    log_files = list(tmp_path.glob("order_outcomes_*.log"))
    assert len(log_files) == 1
    row = log_files[0].read_text(encoding="utf-8").strip().split("|")
    assert row[9] == "BULL_TREND"
    assert row[10] == "0.7500"
    assert row[11] == "cache"
    assert row[12] == "True"
    assert row[13] == "False"
    assert row[14] == ""


def test_batch45_offline_backtests_and_runners_hardening(monkeypatch):
    import pytest
    import offline.backtests.datasets as ds
    import offline.backtests.evaluation as ev
    import offline.runners.kraken_financial_compare as kfc
    import offline.runners.walk_forward as wf

    # 1. align_previous_values rejects inf timestamp with ValueError instead of OverflowError
    with pytest.raises(ValueError, match="non-finite"):
        ds.align_previous_values([1700000000], [{"timestamp": float("inf"), "close": 1.0}])
    assert ds.align_previous_values(["1700000060.0"], [{"timestamp": "1700000000.0", "close": 1.25}]) == [1.25]

    # 2. evaluate_walk_forward handles float-string / bool ambiguous_bars safely
    records = [
        {"timestamp": 1700000000 + i * 60, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5}
        for i in range(6)
    ]
    res = ev.evaluate_walk_forward(
        records,
        lambda ohlc, warm, ctx: {
            "return_pct": 1.0,
            "max_drawdown_pct": 0.5,
            "cycles": 1,
            "fills": 2,
            "ambiguous_bars": "2.0",
        },
        train_size=2,
        validation_size=2,
        test_size=2,
        step_size=2,
    )
    assert res["aggregate_test"]["total_ambiguous_bars"] == 2

    # 3. kraken_financial_compare.markdown_report handles non-numeric median_calmar_improvement_ratio
    md = kfc.markdown_report({
        "candidates": {
            "live": {
                "scenarios": {
                    "central": {
                        "mean_return_pct": 1.0,
                        "worst_return_pct": -0.5,
                    }
                },
                "promotion_gate": {
                    "return_gate": {
                        "scenarios": {
                            "central": {
                                "passed": False,
                                "mean_return_delta_pp": 0.0,
                                "worst_return_delta_pp": 0.0,
                                "worst_drawdown_delta_pp": 0.0,
                                "wins_ties_losses": {"wins": 0, "ties": 10, "losses": 0},
                            }
                        }
                    },
                    "defensive_gate": {
                        "scenarios": {
                            "central": {
                                "passed": False,
                                "median_calmar_improvement_ratio": "n/a",
                                "drawdown_wins_ties_losses": {"wins": 0, "ties": 10, "losses": 0},
                            }
                        }
                    },
                },
            }
        },
        "promotable": [],
    })
    assert "| n/a |" in md

    # 4. walk_forward.dispatch accepts uppercase/whitespace --venue and --mode
    captured = {}
    monkeypatch.setattr(wf, "run_baseline", lambda v, a: captured.update(venue=v, args=a) or 0)
    assert wf.dispatch(["--venue", " KRAKEN ", "--mode", " BASELINE ", "--intervals", "240"]) == 0
    assert captured["venue"] == "kraken"


def test_batch46_offline_research_tapes_and_ranges_hardening(tmp_path, monkeypatch):
    import json
    import offline.research.backtest_ranges as br
    import offline.research.chop_weight_backtest as cwb
    import offline.research.comprehensive_strategy_sweep as css
    import offline.research.intelligence_backtest as ib

    # 1. intelligence_backtest._precompute_intelligence_tape initializes trend_start_ts to first valid t
    tape = ib._precompute_intelligence_tape(
        "BTCUSDC",
        [(True, 100.0), ("1700000000", "50000.0"), (1700000300.0, 50100.0)],
        step_seconds=300,
        parabolic_surge_pct=3.5,
    )
    assert len(tape) == 2
    assert tape[0]["trend_duration_sec"] == 0.0
    assert tape[1]["trend_duration_sec"] == 300.0

    # 2. chop_weight_backtest._precompute_chop_bar_tape sanitizes btc_sampled and first bar ts
    chop_tape = cwb._precompute_chop_bar_tape(
        "TAOUSDC",
        [(None, 100.0), ("1700000000", "350.0"), (1700000900.0, 352.0)],
        btc_sampled=[(True, 50000.0), ("1700000000", "60000.0")],
        step_seconds=900,
    )
    assert len(chop_tape) == 2
    assert chop_tape[0]["trend_start_ts"] == 1700000000.0
    assert chop_tape[0]["btc_trend_start_ts"] == 1700000000.0

    # 3. comprehensive_strategy_sweep.run_live_forward_shadow accepts string OHLC values
    shadow_json = tmp_path / "shadow.ohlc.json"
    shadow_json.write_text(
        json.dumps([
            [2000000000, "25.0", "25.5", "24.8", "25.2"],
            [2000014400, "25.2", "26.0", "25.0", "25.8"],
        ]),
        encoding="utf-8",
    )
    monkeypatch.setattr(css, "SHADOW_OHLC_JSON", shadow_json)
    res_shadow = css.run_live_forward_shadow({"00_LIVE_CURRENT": {}})
    assert "00_LIVE_CURRENT" in res_shadow

    # 4. backtest_ranges.scan_all skips non-path items and normalizes Path objects
    conf_file = tmp_path / "sample.conf"
    conf_file.write_text("# BACKTEST: 1.0, 2.0\nmt.gain = 1.5\n", encoding="utf-8")
    scanned = br.scan_all([conf_file, {"unhashable": 1}, None])
    assert str(conf_file) in scanned
    assert scanned[str(conf_file)] == {"mt.gain": ["1.0", "2.0"]}


def test_batch47_surge_sweep_grids_and_ml_forecast_hardening():
    import numpy as np
    import offline.research.comprehensive_surge_sweep.run_deep_surge_sweep as dss
    import offline.research.ml_forecast.forecast as mlf
    import offline.research.ml_forecast.vol_chronos as vc

    # 1. _win_move_at_ts safely handles None, bool, and string timestamps/windows
    assert dss._win_move_at_ts([100.0, 110.0], 0, None, 4.0) is None
    assert dss._win_move_at_ts([100.0, 110.0], 0, True, 4.0) is None
    assert dss._win_move_at_ts([100.0, 110.0], 0, "14400.0", "4.0") == 10.0

    # 2. forecast._to_price_array replaces bool/None/invalid entries with NaN
    arr = mlf._to_price_array([100.0, True, None, "bad", "105.5"])
    assert len(arr) == 5
    assert arr[0] == 100.0
    assert np.isnan(arr[1])
    assert np.isnan(arr[2])
    assert np.isnan(arr[3])
    assert arr[4] == 105.5

    # 3. vol_chronos.realized_vol_series sanitizes bool/None/string entries
    rv = vc.realized_vol_series([100.0, True, 102.0, 104.0], win=2)
    assert len(rv) == 4
    assert np.isnan(rv[2])  # because index 1 (True) was sanitized to NaN


def test_batch48_offline_runners_backtests_and_legacy_tools_hardening(tmp_path, monkeypatch):
    import json
    import pytest
    import offline.backtests.comparison as cmp_mod
    import offline.backtests.promotion as prom_mod
    import offline.legacy_tools.clean_order_cache as coc_mod
    import offline.legacy_tools.convertor as conv_mod
    import offline.runners.fetch_hyperliquid_candles as fhc_mod

    # 1. fetch_hyperliquid_candles.validate_continuity handles float strings and rejects NaN/inf/non-integer timestamps
    fhc_mod.validate_continuity([{"timestamp": "1700000000.0"}, {"timestamp": "1700003600.0"}], 60)
    with pytest.raises(ValueError, match="invalid timestamp"):
        fhc_mod.validate_continuity([{"timestamp": float("nan")}, {"timestamp": 1700003600}], 60)
    with pytest.raises(ValueError, match="invalid timestamp"):
        fhc_mod.validate_continuity([{"timestamp": 1700000000.5}, {"timestamp": 1700003600.5}], 60)

    # 2. comparison.relative_to_baseline ignores unhashable window keys instead of raising TypeError
    rel = cmp_mod.relative_to_baseline(
        [{"key": ["unhashable"], "return_pct": 5.0}, {"key": "w1", "return_pct": 3.0}],
        [{"key": {"bad": 1}, "return_pct": 1.0}, {"key": "w1", "return_pct": 1.0}],
    )
    assert rel["wins_vs_live"] == 1
    assert rel["mean_return_delta_pct"] == pytest.approx(2.0)

    # 3. promotion._extract_scenario_windows rejects unhashable or bool window keys with ValueError
    with pytest.raises(ValueError, match="different windows"):
        prom_mod._extract_scenario_windows(
            "central",
            {"assumptions": {}, "windows": [{"key": ["unhashable"], "return_pct": 1.0}]},
            {"assumptions": {}, "windows": [{"key": ["unhashable"], "return_pct": 2.0}]},
        )

    # 4. convertor.convert_price_cache handles list-rooted JSON and non-dict fetchtime
    in_path = tmp_path / "in_cache.json"
    out_path = tmp_path / "out_cache.json"
    in_path.write_text(json.dumps([[1700000000, 50000.0]]), encoding="utf-8")
    conv_mod.convert_price_cache(str(in_path), str(out_path), "BTCUSDC")
    converted = json.loads(out_path.read_text(encoding="utf-8"))
    assert converted["items"]["BTCUSDC"] == [[1700000000, 50000.0]]
    assert converted["fetchtime"] == {}

    # 5. clean_order_cache preserves non-list entries in items when cleaning
    order_file = tmp_path / "cache_order.json"
    order_file.write_text(
        json.dumps({
            "items": {
                "BTCUSDC": [
                    {"status": "FILLED", "price": "50000.0"},
                    {"status": "CANCELED", "price": "0.0"},
                ],
                "_meta": {"version": 1},
            }
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(coc_mod, "ORDER_FILE", str(order_file))
    monkeypatch.setattr("sys.argv", ["clean_order_cache.py"])
    coc_mod.main()
    cleaned_data = json.loads(order_file.read_text(encoding="utf-8"))
    assert len(cleaned_data["items"]["BTCUSDC"]) == 1
    assert cleaned_data["items"]["_meta"] == {"version": 1}


def test_batch49_cmc_platform_and_news_feed_deadline_hardening(tmp_path, monkeypatch):
    import market_monitor.pricefetcher as pf_mod
    from intelligence.macro.news_feed_collector import NewsFeedCollector
    import intelligence.macro_sentiment_ssot as ssot

    # 1. Explicit api_key=None or api_key="" does not fall back to os.environ["CMC_API_KEY"]
    monkeypatch.setenv("CMC_API_KEY", "env-cmc-secret-key")
    p_none = pf_mod.CoinMarketCapPricePlatform(api_key=None)
    assert p_none.api_key is None
    assert p_none._supported_symbols == set()

    p_empty = pf_mod.CoinMarketCapPricePlatform(api_key="   ")
    assert p_empty.api_key is None
    assert p_empty._supported_symbols == set()

    # 2. NewsFeedCollector.fetch stops iterating remaining feeds when deadline is exhausted after parsing a feed
    collector = NewsFeedCollector()
    monkeypatch.setattr(collector, "get_active_feed_urls", lambda: ["https://feed1.example/rss", "https://feed2.example/rss"])
    calls = []

    class _DummyResp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b"<rss><channel><item><title>BTC surges</title></item></channel></rss>"

    t_vals = iter([100.0, 100.1, 110.8])
    monkeypatch.setattr("intelligence.macro.news_feed_collector.time.monotonic", lambda: next(t_vals, 111.0))

    def _fake_urlopen(req, timeout=None):
        calls.append((req.full_url, timeout))
        return _DummyResp()

    monkeypatch.setattr("intelligence.macro.news_feed_collector.urllib.request.urlopen", _fake_urlopen)
    res = collector.fetch(force_refresh=True)
    assert len(calls) == 1
    assert calls[0][0] == "https://feed1.example/rss"
    assert calls[0][1] <= 4.0
    assert res is not None and res.total_fetched == 1

    # 3. Dynamic lexicon stat cache avoids redundant disk parses and invalidates on update
    ssot.update_dynamic_lexicon(["hormuz blockade"], cache_dir=str(tmp_path), now_ts=1700000000.0)
    matched1 = ssot.match_high_severity_keywords(
        "Oil tankers halt after Hormuz blockade",
        cache_dir=str(tmp_path),
        now_ts=1700000010.0,
    )
    assert "hormuz blockade" in matched1
    ssot.update_dynamic_lexicon(["suez freeze"], cache_dir=str(tmp_path), now_ts=1700000010.0)
    matched2 = ssot.match_high_severity_keywords(
        "Shipping delays after Suez freeze",
        cache_dir=str(tmp_path),
        now_ts=1700000010.0,
    )
    assert "suez freeze" in matched2


def test_batch51_verify_tools_hardening(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import verify_tools.translation_audit as ta_mod
    import verify_tools.watch_trade_decisions as wtd_mod

    # 1. translation_audit.scan_file handles missing/unreadable files cleanly without raising
    hits, is_rt = ta_mod.scan_file(str(tmp_path / "nonexistent.py"), 2)
    assert hits == [] and is_rt is False

    # 2. watch_trade_decisions.main handles specs where spec.number("mt.gain") returns None (e.g. ARBUSDC)
    class _DummyMgr:
        def get_snapshot(self, sym):
            return {"current_price": 1.5, "growth_coefficient": 0.01}

    class _DummySpec:
        symbol = "ARBUSDC"

        def number(self, key):
            return None

    monkeypatch.setattr(wtd_mod, "SYMBOLS", ["ARBUSDC"])
    monkeypatch.setattr(wtd_mod.cm, "get_short_trend_manager", lambda: _DummyMgr())
    monkeypatch.setattr(wtd_mod, "load_registry", lambda: {"BINANCE_ARB": _DummySpec()})
    monkeypatch.setattr(wtd_mod, "position", lambda sym: (1.4, 10.0, 0, 1.4))
    monkeypatch.setattr(wtd_mod, "free_qty", lambda sym: 10.0)
    wtd_mod.main()


def test_batch52_macro_analyzer_and_daemon_sigterm_exit_and_agy_store_fix(tmp_path, monkeypatch):
    import signal
    from intelligence.daemon import IntelligenceTelemetryDaemon
    from intelligence.macro_analyzer import MacroAnalyzer

    # 1. MacroAnalyzer.start() exits immediately without sleeping when SIGTERM handler raises SystemExit(0)
    ma = MacroAnalyzer(cache_dir=str(tmp_path), symbols=["BTCUSDC"])
    sleep_calls = []
    monkeypatch.setattr("intelligence.macro_analyzer.time.sleep", lambda s: sleep_calls.append(s))
    registered_handlers = {}
    monkeypatch.setattr("intelligence.macro_analyzer.signal.signal", lambda sig, fn: registered_handlers.setdefault(sig, fn))

    def _ma_cycle(force=False):
        registered_handlers[signal.SIGTERM](signal.SIGTERM, None)

    monkeypatch.setattr(ma, "run_cycle", _ma_cycle)
    ma.start(poll_interval_sec=5.0)
    assert ma.running is False
    assert sleep_calls == []

    # 2. IntelligenceTelemetryDaemon.start() exits immediately without sleeping when SIGTERM handler raises SystemExit(0)
    itd = IntelligenceTelemetryDaemon(cache_dir=str(tmp_path), symbols=["BTCUSDC"], skip_macro=True, skip_external=True)
    itd_handlers = {}
    monkeypatch.setattr("intelligence.daemon.time.sleep", lambda s: sleep_calls.append(s))
    monkeypatch.setattr("intelligence.daemon.signal.signal", lambda sig, fn: itd_handlers.setdefault(sig, fn))

    def _itd_cycle(force=False):
        itd_handlers[signal.SIGTERM](signal.SIGTERM, None)

    monkeypatch.setattr(itd, "run_cycle", _itd_cycle)
    itd.start(poll_interval_sec=5.0)
    assert itd.running is False
    assert sleep_calls == []


def test_batch53_short_trend_benchmark_and_mt_backtest_hardening(tmp_path, monkeypatch):
    import numpy as np
    import offline.research.short_trend_estimator_benchmark as st_bench
    from offline.research.monitortrades_backtest.backtest_wait_and_regime import (
        compute_exact_time_window_signals,
        evaluate_regime_900s,
        simulate_should_wait,
    )
    from offline.research.monitortrades_backtest.sweep_mt_models import compute_window_bounds

    # 1. short_trend_estimator_benchmark: empty/non-matching JSONL does not raise ValueError on np.diff or write poisoned .npz
    jsonl_file = tmp_path / "cache_365_price_TESTCOIN.jsonl"
    jsonl_file.write_text('{"s": "OTHERCOIN", "i": [1700000000000, 100.0]}\n', encoding="utf-8")
    monkeypatch.setattr(st_bench, "D", str(tmp_path))
    monkeypatch.setattr(st_bench, "resolve_cachedb_filepath", lambda p: p)
    ts_empty, px_empty = st_bench.load_series("TESTCOIN")
    assert len(ts_empty) == 0 and len(px_empty) == 0
    assert not (tmp_path / "cache_365_price_TESTCOIN.npz").exists()

    # 2. evaluate(sym) on empty/short series returns empty dict cleanly; conv_steps/fwd_index/trigger_indices handle empty arrays
    assert len(st_bench.conv_steps(np.empty(0), np.ones(5))) == 0
    idx_e, ok_e = st_bench.fwd_index(np.empty(0), 60.0)
    assert len(idx_e) == 0 and len(ok_e) == 0
    assert st_bench.trigger_indices(np.empty(0), np.empty(0), 0, 10, 120.0, 20.0, -1, 600.0) == []
    sym_out, n_out, res_out = st_bench.evaluate("TESTCOIN")
    assert sym_out == "TESTCOIN" and n_out == 0 and res_out == {}

    # 3. compute_exact_time_window_signals with negative win_sec and compute_window_bounds with non-positive win_sec
    ts_arr = np.array([1000.0, 1001.0, 1002.0, 1003.0], dtype=np.float64)
    px_arr = np.array([100.0, 101.0, 102.0, 103.0], dtype=np.float64)
    sig = compute_exact_time_window_signals(ts_arr, px_arr, -10.0, "BTCUSDC")
    assert len(sig["gc"]) == 4
    a_arr, sr_arr = compute_window_bounds(ts_arr, win_sec=-5.0)
    assert len(a_arr) == 4 and len(sr_arr) == 4

    # 4. simulate_should_wait and evaluate_regime_900s guard zero price without ZeroDivisionError
    px_zero = np.array([0.0, 100.0, 101.0, 102.0], dtype=np.float64)
    mask = np.array([True, False, False, False], dtype=bool)
    wait_res = simulate_should_wait(ts_arr, px_zero, mask, mask, stride=1)
    assert np.isfinite(wait_res["buy_bps"]) and np.isfinite(wait_res["sell_bps"])
    reg_res, _, _ = evaluate_regime_900s(ts_arr, px_zero, sig, stride=1)
    assert len(reg_res) == 6


def test_batch54_apply_proposals_and_scheduled_pilot_hardening(monkeypatch):
    import subprocess
    import offline.runners.apply_proposals as ap_mod
    from offline.research.monitortrades_backtest import scheduled_pilot as sp_mod

    # 1. apply_proposals._read_proposals handles subprocess.TimeoutExpired / OSError cleanly
    def _raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="git", timeout=15)

    monkeypatch.setattr(ap_mod.subprocess, "run", _raise_timeout)
    assert ap_mod._read_proposals() == []

    # 2. apply_proposals.main skips proposals whose live _current_value raises ValueError/KeyError
    #    and continues to subsequent valid proposals
    monkeypatch.setattr("sys.argv", ["apply_proposals.py", "--dry-run"])
    monkeypatch.setattr(
        ap_mod,
        "_read_proposals",
        lambda: [
            {
                "full_key": "BINANCE_BAD.mt.gain",
                "section": "BINANCE_BAD",
                "key": "mt.gain",
                "symbol": "BADUSDC",
                "winner_value": 6.0,
                "dev_commit": "abc1234",
            },
            {
                "full_key": "BINANCE_OK.mt.gain",
                "section": "BINANCE_OK",
                "key": "mt.gain",
                "symbol": "OKUSDC",
                "winner_value": 6.0,
                "dev_commit": "abc1234",
            },
        ],
    )
    audit_records = []
    monkeypatch.setattr(sp_mod, "_append_audit", lambda rec: audit_records.append(dict(rec)))
    monkeypatch.setattr(sp_mod, "is_rate_limited", lambda fk: (False, None))

    def _fake_current_val(sec, key):
        if sec == "BINANCE_BAD":
            raise ValueError("Non-finite config value")
        return 4.0

    monkeypatch.setattr(sp_mod, "_current_value", _fake_current_val)
    ap_mod.main()
    assert len(audit_records) == 1
    assert audit_records[0]["full_key"] == "BINANCE_OK.mt.gain"
    assert audit_records[0]["proposed_new_value"] == 5.0

    # 3. scheduled_pilot.evaluate_key ignores non-numeric grid annotation tokens cleanly
    monkeypatch.setattr(
        sp_mod,
        "scan_backtest_ranges",
        lambda _path: {"BINANCE_OK.mt.hardtp": ["bad_token", "nan", "8.0"]},
    )
    monkeypatch.setattr(sp_mod, "_current_value", lambda s, k: 6.0)
    monkeypatch.setattr(sp_mod.rb, "SYMBOLS", {"OKUSDC": {"params": {"mt.hardtp": "bad"}}})
    monkeypatch.setattr(sp_mod, "_split_series", lambda sym: ([1] * 120, [1] * 120))
    monkeypatch.setattr(
        sp_mod,
        "_run_one",
        lambda sym, base, params, series: {"net": 100.0 if params.get("mt.hardtp") == "8.0" else 50.0, "buy_hold": 0.0},
    )
    res_eval = sp_mod.evaluate_key("BINANCE_OK.mt.hardtp", "OKUSDC", "OK", "mt.hardtp", dry_run=True, propose=True)
    assert res_eval["action"] == "proposed"
    assert res_eval["winner_value"] == 8.0


def test_batch55_undefined_globals_and_shadow_live_hardening(tmp_path, monkeypatch):
    import importlib.util
    import os
    from types import SimpleNamespace
    import cachetrend as ct
    from kraken import shadow_live as kr_sl

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hl_sig_spec = importlib.util.spec_from_file_location(
        "hl_signals_test", os.path.join(root_dir, "hyperliquid", "signals.py")
    )
    hl_signals = importlib.util.module_from_spec(hl_sig_spec)
    hl_sig_spec.loader.exec_module(hl_signals)

    # 1. CacheShortTrendManager.load_state resolves meta_file_path without NameError
    stm = ct.CacheShortTrendManager(
        ["BTCUSDC"],
        filename=str(tmp_path / "cache_short_trend.json"),
    )
    assert stm.filename.endswith("cache_short_trend.json")

    # 2. hyperliquid.signals._builtin_trend resolves safe_nonneg_float without NameError
    monkeypatch.setenv("SIGNAL_FAST_H", "2")
    monkeypatch.setenv("SIGNAL_SLOW_H", "4")
    monkeypatch.setenv("SIGNAL_BAND_PCT", "1.0")
    fake_hl_client = SimpleNamespace(
        candles=lambda coin, interval, lookback_hours=0: [{"c": "100.0"}, {"c": "101.0"}, {"c": "102.0"}, {"c": "105.0"}]
    )
    sig_res = hl_signals._builtin_trend(fake_hl_client, "BTC")
    assert sig_res["source"] == "builtin"
    assert sig_res["trend"] in ("up", "down", "neutral")

    # 3. kraken.shadow_live._eval_block resolves replay module `rp` without NameError
    fake_param = SimpleNamespace(effective_max_budget=lambda: 1000.0)
    monkeypatch.setattr(kr_sl, "_variants", lambda interval: {"current": fake_param})
    monkeypatch.setattr(
        kr_sl,
        "_run_one",
        lambda ohlc, params, interval, fee_pct, include_decision_trace=False: {
            "net": 25.0,
            "total": 30.0,
            "max_drawdown_pct": 1.5,
            "cycles": 2,
            "open_qty": 0.0,
            "decision_trace": [],
        },
    )
    rows = kr_sl._eval_block([(100.0, 101.0, 99.0, 100.5), (100.5, 102.0, 100.0, 101.5)], 240, 0.25)
    assert "current" in rows
    assert rows["current"]["net_pct"] == 2.5


def test_batch56_intelligence_backtest_full_composite_and_research_hardening(monkeypatch):
    from types import SimpleNamespace
    import numpy as np
    import offline.research.intelligence_backtest as ib_mod
    from offline.research.chop_weight_backtest import _downsample_price_series
    from offline.research.ml_forecast.vol_chronos import realized_vol_series

    # 1. _downsample_price_series ignores NaN timestamp without poisoning subsequent bars
    ds = _downsample_price_series(
        [(float("nan"), 100.0), (1000.0, -5.0), (1000.0, 100.0), (1300.0, 102.0)],
        step_seconds=300,
    )
    assert ds == [(1000.0, 100.0), (1300.0, 102.0)]

    # 2. realized_vol_series handles empty and 1-element arrays without ValueError
    assert len(realized_vol_series(np.empty(0), 24)) == 0
    rv1 = realized_vol_series(np.array([100.0]), 24)
    assert len(rv1) == 1 and np.isnan(rv1[0])

    # 3. run_intelligence_backtest FULL_COMPOSITE exercises GeopoliticalThreatAssessment & FearGreedSnapshot constructors cleanly
    fake_entry = SimpleNamespace(
        action=SimpleNamespace(value="ENTRY"),
        side=SimpleNamespace(value="BUY"),
    )
    fake_exit = SimpleNamespace(
        action=SimpleNamespace(value="EXIT"),
        side=SimpleNamespace(value="SELL"),
    )
    fake_tape = [
        # Bar 0: drop_30m >= 3.5 triggers GeopoliticalThreatAssessment veto in FULL_COMPOSITE
        {
            "t": 1000.0,
            "p": 100.0,
            "n_roll": 30,
            "k_out": {"vel": 1.0, "vel_std": 0.1},
            "k_event": None,
            "mr_event": fake_entry,
            "p_dec": None,
            "current_trend": "BEAR",
            "trend_duration_sec": 600.0,
            "recent_high_20": 105.0,
            "local_high_48": 105.0,
            "drop_15m": 1.0,
            "drop_30m": 4.0,
            "p_1h_pct": -4.0,
            "p_4h_pct": -5.0,
        },
        # Bar 1: n_roll >= 48, p >= local_high_48 * 0.998, p_1h_pct >= 1.5 triggers FearGreedSnapshot downscale in FULL_COMPOSITE
        {
            "t": 1300.0,
            "p": 110.0,
            "n_roll": 50,
            "k_out": {"vel": 1.0, "vel_std": 0.1},
            "k_event": fake_entry,
            "mr_event": None,
            "p_dec": None,
            "current_trend": "BULL",
            "trend_duration_sec": 600.0,
            "recent_high_20": 110.0,
            "local_high_48": 110.0,
            "drop_15m": 0.0,
            "drop_30m": 0.0,
            "p_1h_pct": 1.8,
            "p_4h_pct": 2.0,
        },
        # Bar 2: exit
        {
            "t": 1600.0,
            "p": 112.0,
            "n_roll": 51,
            "k_out": {"vel": 0.0, "vel_std": 0.1},
            "k_event": fake_exit,
            "mr_event": None,
            "p_dec": None,
            "current_trend": "FLAT",
            "trend_duration_sec": 0.0,
            "recent_high_20": 112.0,
            "local_high_48": 112.0,
            "drop_15m": 0.0,
            "drop_30m": 0.0,
            "p_1h_pct": 0.5,
            "p_4h_pct": 2.5,
        },
    ]
    monkeypatch.setattr(ib_mod, "load_price_series", lambda path, sym: [(1000.0, 100.0), (1300.0, 110.0), (1600.0, 112.0)])
    monkeypatch.setattr(ib_mod, "_precompute_intelligence_tape", lambda sym, sampled, step_sec, surge_pct: fake_tape)
    res_ib = ib_mod.run_intelligence_backtest("BTCUSDC", cache_path="/dummy/path", step_seconds=300)
    full_stats = res_ib["FULL_COMPOSITE"]
    assert full_stats.geopolitical_vetoes == 1
    assert full_stats.sentiment_downscales == 1






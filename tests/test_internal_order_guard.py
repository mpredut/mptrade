"""Unit tests for internal_order_guard.py (Pillar 1)."""

from __future__ import annotations

import pytest

from internal_order_guard import check_internal_order_guards


def test_non_buy_order_passes():
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

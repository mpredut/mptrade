"""Inventory-reconciled acquisition cost; no exchange requests or real state writes."""
import math
from types import SimpleNamespace
from threading import RLock
from unittest.mock import Mock

import pytest

from providers.quantity import remaining_average_cost


def fill(i, side="BUY", qty=1, price=100, **values):
    return dict(id=i, timestamp=i, side=side, qty=qty, price=price,
                base_fee=0, quote_fee=0, **values)


def test_closed_cycle_does_not_pollute_the_new_position():
    rows = [fill(1, price=10), fill(2, "SELL", price=12), fill(3, price=100)]
    assert remaining_average_cost(rows, 1) == 100


def test_partial_sale_then_new_buy_uses_remaining_average_cost():
    rows = [fill(1, qty=2, price=20), fill(2, "SELL", price=30), fill(3)]
    assert remaining_average_cost(rows, 2) == 60


def test_ambiguous_invalid_or_unreconciled_history_is_unknown():
    cases = [
        ([], 1), ([fill(1)], 2), ([fill(1, "SELL"), fill(2)], 1),
        ([fill(1), fill(1)], 2), ([fill(2), fill(1)], 2),
        ([fill(1), fill(2, "SELL", qty=2)], 1),
        ([fill(1, price=float("inf"))], 1), ([fill(1, qty=float("nan"))], 1),
        ([fill(1, qty=-1)], 1), ([fill(1, qty=True)], 1),
        ([fill(1, "UNKNOWN")], 1), ([fill(1)], float("nan")), ([fill(1)], True),
    ]
    for rows, held in cases:
        assert remaining_average_cost(rows, held) is None


def test_base_and_quote_fees_are_accounted_for_without_guessing():
    row = fill(1)
    row.update(base_fee=0.001, quote_fee=0.1)
    assert math.isclose(remaining_average_cost([row], 0.999), 100.1 / 0.999)
    row.pop("base_fee")
    assert remaining_average_cost([row], 1) is None


def test_negative_fee_is_unknown():
    row = fill(1)
    row["quote_fee"] = -0.1
    assert remaining_average_cost([row], 1) is None


@pytest.fixture
def binance_history(monkeypatch):
    import binance_cache_health as health
    import cacheManager as cm
    from providers.market_api import api

    rows = []
    manager = SimpleNamespace(cache={"ARBUSDC": rows}, lock=RLock(),
                              _is_valid_trade=cm.CacheTradeManager._is_valid_trade)
    current = Mock(return_value=SimpleNamespace(order_cache_version="o", trade_cache_version="t"))
    ensure = Mock()
    get_cache = Mock(return_value=manager)
    monkeypatch.setattr(health, "require_fresh_account_cache", current)
    monkeypatch.setattr(cm, "ensure_account_cache_readers", ensure)
    monkeypatch.setattr(cm, "get_cache_manager", get_cache)
    return api, rows, current, ensure, get_cache


def native(i, side="BUY", qty=1, price=100, **extra):
    import time
    return dict(symbol="ARBUSDC", id=i, orderId=i, time=int(time.time()*1000)-1000+i,
                isBuyer=side == "BUY", qty=str(qty), price=str(price),
                commission="0", commissionAsset="USDC", **extra)


def test_binance_uses_immutable_fills_and_current_durable_versions(binance_history):
    api, rows, current, ensure, get_cache = binance_history
    # The cache can receive REST overlap out of order; venue time/ID orders the fills.
    rows.extend([native(3), native(1, price=10), native(2, "SELL", price=12)])
    assert api.position_cost_basis("ARBUSDC", 1, 3600, provider_name="binance") == 100
    current.assert_called_once()
    ensure.assert_called_once_with(current.return_value)
    get_cache.assert_called_once_with("Trade", start_sync=False)
    assert api.position_cost_basis("ARBUSDC", 2, 3600, provider_name="binance") is None


def test_binance_rejects_stale_history_before_reading(binance_history):
    import binance_cache_health as health
    api, rows, current, ensure, get_cache = binance_history
    rows.append(native(1))
    current.side_effect = health.AccountCacheNotReady("stale")
    with pytest.raises(health.AccountCacheNotReady):
        api.position_cost_basis("ARBUSDC", 1, 3600, provider_name="binance")
    ensure.assert_not_called()
    get_cache.assert_not_called()


def test_binance_base_fee_matches_actual_inventory(binance_history):
    api, rows, *_ = binance_history
    row = native(1)
    row.update(commission="0.001", commissionAsset="ARB")
    rows.append(row)
    assert math.isclose(api.position_cost_basis("ARBUSDC", 0.999, 3600), 100 / 0.999)


def test_provider_without_inventory_cost_support_reports_unknown():
    from providers.market_api import api
    assert api.position_cost_basis("HYPEUSD", 1, 3600, provider_name="kraken") is None


def test_binance_missing_commission_returns_none_instead_of_keyerror(binance_history):
    api, rows, *_ = binance_history
    row = native(1)
    row.pop("commission")
    row.pop("commissionAsset")
    rows.append(row)
    assert api.position_cost_basis("ARBUSDC", 1, 3600, provider_name="binance") is None


def test_extract_order_fee_canonical_across_venues():
    from providers.base import extract_order_fee

    # Binance REST trade shape
    info = extract_order_fee(
        {"commission": "0.02", "commissionAsset": "ARB"},
        base_asset="ARB", quote_asset="USDC", fill_price=1.5,
    )
    assert info is not None
    assert info["fee"] == 0.02
    assert info["fee_asset"] == "ARB"
    assert info["base_fee"] == 0.02
    assert math.isclose(info["fee_in_quote"], 0.03)

    # Binance WS executionReport shape (n / N)
    info = extract_order_fee(
        {"n": "0.05", "N": "USDC"},
        base_asset="ARB", quote_asset="USDC", fill_price=1.5,
    )
    assert info is not None
    assert info["fee"] == 0.05
    assert info["fee_asset"] == "USDC"
    assert info["quote_fee"] == 0.05
    assert math.isclose(info["fee_in_quote"], 0.05)

    # Hyperliquid fill shape (fee / feeToken)
    info = extract_order_fee(
        {"fee": "0.01", "feeToken": "HYPE"},
        base_asset="HYPE", quote_asset="USDC", fill_price=25.0,
    )
    assert info is not None
    assert info["fee"] == 0.01
    assert info["fee_asset"] == "HYPE"
    assert math.isclose(info["fee_in_quote"], 0.25)

    # Kraken fill shape (fee without explicit asset defaults to quote unless require_explicit_asset=True)
    info = extract_order_fee(
        {"fee": "0.12"},
        base_asset="HYPE", quote_asset="USD", fill_price=25.0,
    )
    assert info is not None
    assert info["fee"] == 0.12
    assert info["fee_asset"] == "USD"
    assert math.isclose(info["fee_in_quote"], 0.12)
    assert extract_order_fee(
        {"fee": "0.12"},
        base_asset="HYPE", quote_asset="USD", fill_price=25.0,
        require_explicit_asset=True,
    ) is None


def test_ws_execution_report_preserves_commission_and_zero_price_fallback(monkeypatch):
    import cacheManager as cm

    persisted = []

    class DummyTradeCache:
        _is_valid_trade = staticmethod(cm.CacheTradeManager._is_valid_trade)

        def _persist_items(self, symbol, items):
            persisted.extend(items)

    monkeypatch.setattr(cm, "get_cache_manager", lambda name: DummyTradeCache())
    cm._append_trade_from_execution_report({
        "x": "TRADE",
        "X": "PARTIALLY_FILLED",
        "s": "ARBUSDC",
        "S": "BUY",
        "i": 1001,
        "t": 5001,
        "L": "0.00000000",
        "p": "1.25000000",
        "l": "0.00000000",
        "q": "10.00000000",
        "n": "0.01000000",
        "N": "ARB",
        "T": 1700000000000,
    })
    assert len(persisted) == 1
    assert persisted[0]["price"] == "1.25000000"
    assert persisted[0]["qty"] == "10.00000000"
    assert persisted[0]["commission"] == "0.01000000"
    assert persisted[0]["commissionAsset"] == "ARB"


"""Provider contract conformance, StrategyExecutor metadata, and Binance order filter tests."""
from decimal import Decimal
import os
import sys
import unittest

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

from providers.binance_filters import (  # noqa: E402
    BinanceFilterError,
    BinanceOrderRules,
    binance_filter_refusal_reason,
    decimal_places,
)
from providers.hyperliquid_provider import HyperliquidProvider  # noqa: E402
from providers.kraken_provider import KrakenProvider  # noqa: E402
from providers.market_api import BinanceProvider  # noqa: E402
from providers.strategy_executor import (  # noqa: E402
    OrderReconciliationCapabilities,
    PairPrecision,
    ProviderError,
    StrategyExecutor,
    candle_interval,
)
from providers.t212_provider import T212Provider  # noqa: E402

CONTRACT_METHODS = (
    "get_current_price", "submit_order", "order_status", "cancel_order",
    "pair_precision", "free_balance", "ohlc_closes",
    "reconciliation_capabilities",
)

EXPECTED_RECONCILIATION = {
    "kraken": OrderReconciliationCapabilities(False, True, True, True),
    "hyperliquid": OrderReconciliationCapabilities(True, True, True, True),
    "binance": OrderReconciliationCapabilities(
        True, True, True, True, 90 * 24 * 60 * 60),
    "trading212": OrderReconciliationCapabilities(False, True, True, False),
}


def _providers():
    return [
        ("kraken", KrakenProvider()),
        ("hyperliquid", HyperliquidProvider(token="HYPE")),
        ("binance", BinanceProvider()),
        ("trading212", T212Provider()),
    ]


class ProviderContractConformanceTest(unittest.TestCase):
    def test_every_provider_satisfies_StrategyExecutor(self):
        for name, prov in _providers():
            with self.subTest(provider=name):
                self.assertIsInstance(prov, StrategyExecutor,
                                      f"{name} does NOT satisfy the StrategyExecutor contract")

    def test_every_contract_method_exists_and_is_callable(self):
        for name, prov in _providers():
            for meth in CONTRACT_METHODS:
                with self.subTest(provider=name, method=meth):
                    self.assertTrue(callable(getattr(prov, meth, None)),
                                    f"{name}.{meth} is missing or not callable")

    def test_reconciliation_capabilities_are_declared_explicitly(self):
        for name, provider in _providers():
            with self.subTest(provider=name):
                self.assertEqual(
                    provider.reconciliation_capabilities(),
                    EXPECTED_RECONCILIATION[name],
                )


def test_pair_precision_rejects_invalid_venue_metadata():
    invalid_cases = [
        {"price_decimals": -1, "volume_decimals": 2, "order_min": 0.01},
        {"price_decimals": 2, "volume_decimals": -1, "order_min": 0.01},
        {"price_decimals": 1.5, "volume_decimals": 2, "order_min": 0.01},
        {"price_decimals": 2, "volume_decimals": 2, "order_min": -0.01},
        {"price_decimals": 2, "volume_decimals": 2, "order_min": float("nan")},
    ]
    for kwargs in invalid_cases:
        with pytest.raises(ValueError):
            PairPrecision(**kwargs)


def test_pair_precision_normalizes_valid_metadata_once():
    assert PairPrecision(2.0, 6, "0.01", " HYPE ") == PairPrecision(
        2, 6, 0.01, "HYPE"
    )


def test_candle_interval_maps_only_supported_horizons():
    cases = [(1, "1m"), (5, "5m"), (15, "15m"), (60, "1h"), (240, "4h"), (1440, "1d")]
    for minutes, expected in cases:
        assert candle_interval(minutes) == expected


def test_candle_interval_rejects_unknown_horizons():
    unknown_cases = [None, "bad", 0, 30, 60.5]
    for minutes in unknown_cases:
        with pytest.raises(ProviderError):
            candle_interval(minutes)


def _binance_filter_info():
    return {"baseAsset": "TAO", "quoteAsset": "USDC", "filters": [
        {"filterType": "PRICE_FILTER", "tickSize": "0.05", "minPrice": "0.05", "maxPrice": "10000"},
        {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.010", "maxQty": "100"},
        {"filterType": "MARKET_LOT_SIZE", "stepSize": "0.01", "minQty": "0.10", "maxQty": "50"},
        {"filterType": "NOTIONAL", "minNotional": "10", "maxNotional": "100000",
         "applyMinToMarket": True, "applyMaxToMarket": True},
    ]}


def test_limit_normalization_floors_to_exchange_steps():
    rules = BinanceOrderRules.from_symbol_info(_binance_filter_info())
    qty, price = rules.normalize(quantity="1.2349", price="101.079")
    assert qty == Decimal("1.234")
    assert price == Decimal("101.05")
    assert decimal_places(rules.tick_size) == 2
    assert decimal_places(rules.lot_step) == 3


def test_market_uses_market_lot_and_reference_notional():
    rules = BinanceOrderRules.from_symbol_info(_binance_filter_info())
    qty, price = rules.normalize(quantity="1.239", market=True, reference_price="100")
    assert qty == Decimal("1.23")
    assert price is None


def test_filter_refusal_reason_is_stable_for_retry_classification():
    assert binance_filter_refusal_reason(
        BinanceFilterError("notional 5 is below minimum 10")) == "below_min_notional"
    assert binance_filter_refusal_reason(
        BinanceFilterError("quantity 0 violates [1, unbounded]")) == (
            "binance_filter_refused:quantity 0 violates [1, unbounded]")
    assert binance_filter_refusal_reason(
        BinanceFilterError("invalid reference_price: None")) == (
            "binance_filter_unavailable:invalid reference_price: None")


def test_exchange_and_business_minima_are_distinct_and_fail_closed():
    rules = BinanceOrderRules.from_symbol_info(_binance_filter_info())
    with pytest.raises(BinanceFilterError, match="below minimum 100"):
        rules.normalize(quantity="0.5", price="100", business_min_notional="100")
    with pytest.raises(BinanceFilterError, match="PRICE_FILTER"):
        BinanceOrderRules.from_symbol_info({"baseAsset": "TAO", "quoteAsset": "USDC", "filters": []})


if __name__ == "__main__":
    unittest.main(verbosity=2)

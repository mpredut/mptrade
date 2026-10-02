"""Phase 5 provider unification — A SINGLE GUARD: EVERY venue provider satisfies the
StrategyExecutor. If someone adds a method to the contract but forgets a provider (or
changes a signature), this test fails -> base v2 never gets to crash on that venue.

Parametrised over kraken / hyperliquid / binance / Trading212. Instantiated without network (the clients are
lazy); we check ONLY that the interface exists and is callable, not the behaviour (that lives in
the per-provider test_*_provider_executor.py modules)."""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

from providers.strategy_executor import (  # noqa: E402
    OrderReconciliationCapabilities,
    StrategyExecutor,
)
from providers.kraken_provider import KrakenProvider  # noqa: E402
from providers.hyperliquid_provider import HyperliquidProvider  # noqa: E402
from providers.market_api import BinanceProvider  # noqa: E402
from providers.t212_provider import T212Provider  # noqa: E402

# Methods required by the venue-agnostic strategies/spot_engine.py engine.
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


if __name__ == "__main__":
    unittest.main(verbosity=2)

#!/usr/bin/env python3
"""Tests for Hyperliquid trailing_stop and dynamic multi-venue trailing symmetry (no real API, no money)."""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import active_intents
from botcore import ClosedPriceSeries
from intelligence.sentiment.collectors.portfolio_collector import VenueTrailingMapsCollector
from providers.strategy_executor import OrderReconciliationCapabilities, OrderStatus, PairPrecision
from tests.conftest import TrailingTestBase, load_venue_modules
from trailing_core import StrategyExecutorTrailingAdapter
from verify_tools import ownership_inventory as oi

hl_ts_mod = load_venue_modules("hyperliquid", "trailing_stop")
HyperliquidTrailing = hl_ts_mod.HyperliquidTrailing


class FakeHLProvider:
    """In-memory StrategyExecutor stub mirroring HyperliquidProvider's public contract."""

    def __init__(
        self,
        *,
        token: str = "HYPE",
        price: float = 50.0,
        total: float = 20.0,
        held: float = 0.0,
        open_sell_orders: list[dict] | None = None,
        daily_closes: list[float] | None = None,
        live_enabled: bool = True,
        fail_balances: bool = False,
    ):
        self.name = "Hyperliquid"
        self.token = token
        self.price = price
        self.total = total
        self.held = held
        self.open_sell_orders = list(open_sell_orders or [])
        self.daily_closes = list(daily_closes) if daily_closes is not None else [40.0] * 120
        self.live_enabled = live_enabled
        self.fail_balances = fail_balances
        self.orders: list[dict] = []
        self.cancelled: list[tuple[str, str]] = []
        self.balance_calls = 0

    def execution_enabled(self) -> bool:
        return self.live_enabled

    def reconciliation_capabilities(self):
        return OrderReconciliationCapabilities(
            lookup_by_client_order_id=True,
            status_by_order_id=True,
            cancel_by_order_id=True,
            list_open_orders=True,
            replace_by_order_id=True,
        )

    def all_free_and_locked_balances(self, assets=None):
        self.balance_calls += 1
        if self.fail_balances:
            return None
        free = max(self.total - self.held, 0.0)
        return {self.token: (free, self.held), "USDC": (1000.0, 0.0)}

    def free_balance(self, asset: str):
        if asset.upper() == self.token:
            return max(self.total - self.held, 0.0)
        return 1000.0 if asset.upper() == "USDC" else 0.0

    def get_current_price(self, symbol: str):
        return self.price

    def pair_precision(self, symbol: str):
        return PairPrecision(price_decimals=3, volume_decimals=2, order_min=0.01, base_asset=self.token)

    def round_quantity(self, symbol: str, qty: float) -> float:
        return round(float(qty), 2)

    def min_order_qty(self, symbol: str) -> float:
        return 0.01

    def open_orders(self, symbol: str):
        return list(self.open_sell_orders)

    def cancel_order(self, symbol: str, order_id: str) -> None:
        self.cancelled.append((symbol, str(order_id)))
        self.open_sell_orders = [o for o in self.open_sell_orders if str(o.get("orderId")) != str(order_id)]
        self.held = 0.0

    def ohlc_series(self, symbol: str, interval_min: int):
        return ClosedPriceSeries(tuple(self.daily_closes), interval_min)

    def position_cost_basis(self, symbol: str, held_qty: float, since_s: float):
        return None

    def get_orders(self, symbol: str, side=None, lookback=None, **_kwargs):
        return []

    def submit_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        price=None,
        *,
        market: bool = False,
        kind=None,
        client_order_id=None,
        post_only: bool = True,
    ) -> str:
        if not self.live_enabled:
            from providers.ohlc_utils import ProviderError

            raise ProviderError("HL_LIVE_ORDERS=false")
        oid = str(1000 + len(self.orders) + 1)
        self.orders.append(
            {
                "order_id": oid,
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "price": price,
                "market": market,
                "kind": kind,
                "client_order_id": client_order_id,
                "post_only": post_only,
            }
        )
        return oid

    def order_status(self, symbol: str, order_id: str) -> OrderStatus:
        order = next(o for o in self.orders if o["order_id"] == str(order_id))
        px = order["price"] if order["price"] is not None else self.price
        return OrderStatus("closed", order["qty"], order["qty"] * px, order["qty"] * px * 0.0007, venue_status="FILLED")

    def order_by_client_id(self, symbol: str, client_order_id: str):
        for o in self.orders:
            if o.get("client_order_id") == client_order_id:
                return {"orderId": o["order_id"], "status": "FILLED"}
        return None


class TestHyperliquidTrailing(TrailingTestBase):
    TRAIL_MODULE = hl_ts_mod
    TRAIL_DEFAULTS = {"HYPE": 24.0}
    ENV_OVERRIDES = {"HL_LIVE_ORDERS": "true"}
    SUPPRESS_MODULE_NOTIFY = True

    def setUp(self):
        super().setUp()
        self._orig_pair_for = dict(hl_ts_mod.PAIR_FOR)
        self._orig_rebuy_mode = dict(hl_ts_mod.REBUY_MODE_BY_ASSET)
        self._orig_quote_for = dict(hl_ts_mod.QUOTE_FOR)
        hl_ts_mod.PAIR_FOR["HYPE"] = "HYPEUSDC"
        hl_ts_mod.REBUY_MODE_BY_ASSET["HYPE"] = "on"
        hl_ts_mod.QUOTE_FOR["HYPE"] = "USDC"

    def tearDown(self):
        hl_ts_mod.PAIR_FOR.clear()
        hl_ts_mod.PAIR_FOR.update(self._orig_pair_for)
        hl_ts_mod.REBUY_MODE_BY_ASSET.clear()
        hl_ts_mod.REBUY_MODE_BY_ASSET.update(self._orig_rebuy_mode)
        hl_ts_mod.QUOTE_FOR.clear()
        hl_ts_mod.QUOTE_FOR.update(self._orig_quote_for)
        super().tearDown()

    def _adapter(self, prov: FakeHLProvider, *, enabled=True, min_profit_pct=0.0, trailing_market=True, rebuy_trend_days=100):
        return HyperliquidTrailing(
            provider=prov,
            log=lambda *a: None,
            enabled=enabled,
            state_file=self.sf,
            min_profit_pct=min_profit_pct,
            trailing_market=trailing_market,
            rebuy_trend_days=rebuy_trend_days,
        )

    def test_inherits_shared_strategy_executor_adapter_and_defaults_dry_run(self):
        self.assertTrue(issubclass(HyperliquidTrailing, StrategyExecutorTrailingAdapter))
        prov = FakeHLProvider(price=50.0, total=20.0)
        ts = self._adapter(prov, enabled=None)
        # hyperliquid/trailing.conf ships with HL_TRAILING_ENABLED=false by default
        self.assertFalse(ts.enabled)
        ts.check_once()
        prov.price = 35.0
        ts.check_once()
        self.assertEqual(prov.orders, [])

    def test_crash_sells_free_balance_only_and_emits_usdc_fill_notification(self):
        notified = []
        hl_ts_mod.notify = lambda **kw: notified.append(kw)
        prov = FakeHLProvider(price=50.0, total=20.0, held=4.0)
        ts = self._adapter(prov, enabled=True)
        ts.check_once()  # peak=50.0
        prov.price = 37.0  # -26% >= 24%
        ts.check_once()
        self.assertEqual(len(prov.orders), 1)
        self.assertEqual(prov.orders[0]["side"], "sell")
        self.assertTrue(prov.orders[0]["market"])
        self.assertFalse(prov.orders[0]["post_only"])
        self.assertAlmostEqual(prov.orders[0]["qty"], 16.0, places=2)
        # Next tick reconciles terminal fill and emits trade notification with dynamic USDC quote
        prov.total = 4.0
        ts.check_once()
        self.assertEqual(len(notified), 1)
        self.assertIn("USDC", str(notified[0].get("body", "")) + str(notified[0].get("title", "")))
        self.assertEqual(notified[0].get("source"), "hyperliquid-trail")

    def test_unlock_qty_cancels_open_sell_orders_and_releases_balance(self):
        prov = FakeHLProvider(
            price=50.0,
            total=20.0,
            held=20.0,
            open_sell_orders=[
                {"orderId": "901", "side": "SELL", "price": 60.0, "qty": 20.0},
                {"orderId": "902", "side": "BUY", "price": 40.0, "qty": 5.0},
            ],
        )
        ts = self._adapter(prov, enabled=True)
        ts.check_once()  # peak=50.0 (tracks total_qty=20 even when held)
        prov.price = 37.0  # -26% -> unlocks resting SELL and sells full 20.0
        ts.check_once()
        self.assertEqual(prov.cancelled, [("HYPEUSDC", "901")])
        self.assertEqual(len(prov.orders), 1)
        self.assertAlmostEqual(prov.orders[0]["qty"], 20.0, places=2)

    def test_rebuy_auto_uses_daily_sma_gate_and_intelligence_guard_downscale(self):
        hl_ts_mod.REBUY_MODE_BY_ASSET["HYPE"] = "auto"
        prov_bear = FakeHLProvider(price=35.0, daily_closes=[50.0] * 120)
        ts_bear = self._adapter(prov_bear, enabled=True, rebuy_trend_days=100)
        self.assertFalse(ts_bear.rebuy_enabled_for("HYPE"))
        self.assertTrue(ts_bear.rebuy_configured_for("HYPE"))

        prov_bull = FakeHLProvider(price=60.0, daily_closes=[45.0] * 120)
        ts_bull = self._adapter(prov_bull, enabled=True, rebuy_trend_days=100)
        self.assertTrue(ts_bull.rebuy_enabled_for("HYPE"))

        rb_state = {"qty": 10.0, "sell_price": 55.0, "low": 55.0}
        with patch("order_guard.check_intelligence_guards", return_value=(True, "downscale", 0.5)):
            res = ts_bull.execute_rebuy("HYPE", "HYPE", "HYPEUSDC", 10.0, 58.0, rb_state, persist=lambda _: None)
            self.assertEqual(res.outcome, "active")
            self.assertAlmostEqual(prov_bull.orders[0]["qty"], 5.0)


def test_dynamic_trailing_venue_derivation_across_intents_telemetry_and_ownership(tmp_path: Path):
    """active_intents, VenueTrailingMapsCollector, and ownership_inventory derive trailing venues dynamically."""
    bots_dir = tmp_path / "statedb" / "bots"
    bots_dir.mkdir(parents=True, exist_ok=True)
    cdb_dir = tmp_path / "cachedb"
    cdb_dir.mkdir(parents=True, exist_ok=True)

    # 1. active_intents discovers trailing_state_hyperliquid.json and any future venue dynamically
    (bots_dir / "trailing_state_hyperliquid.json").write_text(
        json.dumps(
            {
                "HYPE": {
                    "peak": 52.0,
                    "pending_order": {
                        "intent_id": "trailing:hyperliquid:HYPE:SELL:10:39:52",
                        "client_order_id": "0x" + "ab" * 16,
                        "symbol": "HYPEUSDC",
                        "side": "SELL",
                        "requested_qty": 10.0,
                        "requested_price": 39.0,
                        "status": "pending",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    (bots_dir / "trailing_state_bybit.json").write_text(
        json.dumps(
            {
                "SOL": {
                    "peak": 180.0,
                    "pending_order": {
                        "intent_id": "trailing:bybit:SOL:SELL:5:130:180",
                        "symbol": "SOLUSDT",
                        "side": "SELL",
                        "requested_qty": 5.0,
                        "requested_price": 130.0,
                        "status": "pending",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    idx = active_intents.build_active_intent_index(str(tmp_path))
    by_owner = {row["owner"]: row for row in idx["intents"]}
    assert "hyperliquid_trailing" in by_owner
    assert by_owner["hyperliquid_trailing"]["venue"] == "hyperliquid"
    assert by_owner["hyperliquid_trailing"]["symbol"] == "HYPEUSDC"
    assert "bybit_trailing" in by_owner
    assert by_owner["bybit_trailing"]["venue"] == "bybit"

    # 2. VenueTrailingMapsCollector dynamically discovers Hyperliquid & Bybit without hardcoded venue maps
    t_map, by_venue = VenueTrailingMapsCollector(start_sync=False).collect(
        str(cdb_dir), workspace_dir=str(tmp_path)
    )
    assert "hyperliquid" in by_venue
    assert by_venue["hyperliquid"]["HYPEUSDC"]["peak"] == pytest.approx(52.0)
    assert t_map["HYPEUSDC"]["peak"] == pytest.approx(52.0)
    assert "bybit" in by_venue
    assert by_venue["bybit"]["SOL"]["peak"] == pytest.approx(180.0)

    # 3. ownership_inventory derives hyperliquid-trailing dynamically from instruments.conf + trailing.conf
    (tmp_path / "procs.conf").write_text(
        "hyperliquid/trailing_stop.py|$ROOT|python3 hyperliquid/trailing_stop.py|HL-trailing|cachedb/HL-trailing.heartbeat|900|bot\n",
        encoding="utf-8",
    )
    (tmp_path / "instruments.conf").write_text(
        "[HYPERLIQUID_HYPE]\n"
        "provider=hyperliquid\n"
        "symbol=HYPEUSDC\n"
        "base=HYPE\n"
        "quote=USDC\n"
        "enabled=yes\n"
        "role.mt=no\n"
        "role.tradeall_fire=yes\n"
        "role.assetguardian=yes\n"
        "role.trailing=yes\n"
        "trailing.pct=24.0\n"
        "trailing.rebuy=auto\n",
        encoding="utf-8",
    )
    hl_dir = tmp_path / "hyperliquid"
    hl_dir.mkdir(parents=True, exist_ok=True)
    (hl_dir / "config.env").write_text("HL_SPOT_TOKEN=HYPE\nHL_LIVE_ORDERS=true\n", encoding="utf-8")
    (hl_dir / "trailing.conf").write_text("HL_TRAILING_ENABLED=true\n", encoding="utf-8")

    owners = oi.build_inventory(tmp_path, commands=["python3 hyperliquid/trailing_stop.py"])
    hl_tr = next(o for o in owners if o.owner_id == "hyperliquid-trailing")
    assert hl_tr.venue == "hyperliquid"
    assert hl_tr.symbol == "HYPEUSDC"
    assert hl_tr.role == "protective"
    assert hl_tr.configured is True
    assert hl_tr.live_enabled is True
    assert hl_tr.running is True

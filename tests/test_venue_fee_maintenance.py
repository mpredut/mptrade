"""Unit tests for 3-tier polymorphic venue fee maintenance, BNB hysteresis, and TCA fee resolution."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from binance_api import trailing_stop as b_ts
import botcore
from intelligence.sentiment.evaluators.tca_risk_evaluator import resolve_venue_fee_and_limits
from providers.execution_audit import ExecutionAudit
from providers.venue_fee_maintenance import (
    BinanceFeeMaintainer,
    HyperliquidFeeMaintainer,
    KrakenFeeMaintainer,
    T212FeeMaintainer,
    VENUE_FEE_MAINTAINER_REGISTRY,
    VenueFeeMaintainerBase,
    VenueFeeMaintainerInterface,
    load_fee_maintenance_config,
    run_venue_fee_maintenance,
)


def _sample_cfg() -> dict:
    return {
        "FEE_MAINTENANCE_ENABLED": "true",
        "FEE_MAINTENANCE_INTERVAL_SEC": 14400,
        "BINANCE_BNB_RESERVE_MIN_USDC": 100.0,
        "BINANCE_BNB_RESERVE_TARGET_USDC": 200.0,
        "BINANCE_BNB_MIN_FREE_USDC": 300.0,
        "BINANCE_BNB_ORDER_STALE_SEC": 1800,
        "KRAKEN_IDLE_USDC_ALERT_MIN": 25.0,
        "KRAKEN_IDLE_USDC_ALERT_COOLDOWN_SEC": 86400,
    }


def _bnb_symbol_info() -> dict:
    return {
        "symbol": "BNBUSDC",
        "baseAsset": "BNB",
        "quoteAsset": "USDC",
        "filters": [
            {"filterType": "PRICE_FILTER", "minPrice": "0.10", "maxPrice": "100000.00", "tickSize": "0.10"},
            {"filterType": "LOT_SIZE", "minQty": "0.001", "maxQty": "9000.000", "stepSize": "0.001"},
            {"filterType": "NOTIONAL", "minNotional": "5.00", "applyMinToMarket": True},
        ],
    }


def test_polymorphic_hierarchy_and_config_validation():
    assert set(VENUE_FEE_MAINTAINER_REGISTRY.keys()) >= {"binance", "kraken", "hyperliquid", "t212"}
    for cls in VENUE_FEE_MAINTAINER_REGISTRY.values():
        assert issubclass(cls, VenueFeeMaintainerBase)
        assert issubclass(cls, VenueFeeMaintainerInterface)

    cfg = load_fee_maintenance_config(_sample_cfg())
    assert cfg["interval_sec"] == 14400.0
    assert cfg["binance_bnb_reserve_min_usdc"] == 100.0
    assert cfg["binance_bnb_reserve_target_usdc"] == 200.0

    bad = dict(_sample_cfg(), BINANCE_BNB_RESERVE_MIN_USDC=250.0)
    with pytest.raises(ValueError, match="strictly below"):
        load_fee_maintenance_config(bad)


def test_binance_zero_bnb_enables_burn_and_places_limit_maker_to_200_target(tmp_path, monkeypatch):
    monkeypatch.setattr(botcore, "is_trade_enabled", lambda: True)
    cli = MagicMock()
    cli.get_bnb_burn_spot_margin.return_value = {"spotBNBBurn": False, "interestBNBBurn": False}
    cli.toggle_bnb_burn_spot_margin.return_value = {"spotBNBBurn": True, "interestBNBBurn": False}
    cli.get_account.return_value = {
        "balances": [
            {"asset": "BNB", "free": "0.0", "locked": "0.0"},
            {"asset": "USDC", "free": "2973.38", "locked": "0.0"},
        ]
    }
    cli.get_orderbook_ticker.return_value = {"symbol": "BNBUSDC", "bidPrice": "625.00", "askPrice": "625.10"}
    cli.get_open_orders.return_value = []
    cli.get_symbol_info.return_value = _bnb_symbol_info()
    cli.get_trade_fee.return_value = [{"symbol": "BTCUSDC", "makerCommission": "0.001", "takerCommission": "0.001"}]

    submitted_calls = []

    def _fake_submit(order_type, symbol, qty, **kwargs):
        submitted_calls.append((order_type, symbol, qty, kwargs))
        return {"orderId": 987654, "price": str(kwargs["price"]), "origQty": str(qty), "status": "NEW"}

    bapi_mod = SimpleNamespace(client=cli, _cancel_and_resolve=MagicMock(return_value=True))
    place_mod = SimpleNamespace(_submit_binance_order=_fake_submit)
    audit = ExecutionAudit(directory=str(tmp_path / "audit"))
    maintainer = BinanceFeeMaintainer(client=cli, bapi_mod=bapi_mod, placeorder_mod=place_mod, audit=audit)

    cfg = load_fee_maintenance_config(_sample_cfg())
    res = maintainer.maintain(1_770_000_000.0, {}, cfg, dry_run=False)

    cli.toggle_bnb_burn_spot_margin.assert_called_once_with(spotBNBBurn=True)
    assert res["spot_bnb_burn"] is True
    assert res["topup_action"] == "placed_limit_maker"
    assert res["has_pending_bnb_order"] is True
    assert res["placed_order_id"] == "987654"
    assert len(submitted_calls) == 1
    side, sym, qty, kw = submitted_calls[0]
    assert (side, sym) == ("BUY", "BNBUSDC")
    assert qty == pytest.approx(0.32, rel=1e-6)  # 200 USDC / 625.00 = 0.320 BNB
    assert kw["price"] == pytest.approx(625.0)
    assert kw["post_only"] is True
    assert kw["client_order_id"] == "FEE_BNB_1770000000"
    # Until BNB order fills, held_bnb_usdc == 0 so VIP0 (10 bps) is reported truthfully.
    assert res["fee_spec"] == {"fee_tier": "VIP0", "maker_fee_bps": 10.0, "taker_fee_bps": 10.0}
    kinds = [e["kind"] for e in res["events"]]
    assert "ops_notice" in kinds and "order_placed" in kinds


def test_binance_hysteresis_band_skips_buy_and_reports_vip0_bnb_discount(tmp_path, monkeypatch):
    monkeypatch.setattr(botcore, "is_trade_enabled", lambda: True)
    cli = MagicMock()
    cli.get_bnb_burn_spot_margin.return_value = {"spotBNBBurn": True}
    # 0.20 BNB @ 600 = 120 USDC (between 100 min and 200 target -> must NOT buy)
    cli.get_account.return_value = {
        "balances": [
            {"asset": "BNB", "free": "0.20", "locked": "0.0"},
            {"asset": "USDC", "free": "1500.0", "locked": "0.0"},
        ]
    }
    cli.get_orderbook_ticker.return_value = {"symbol": "BNBUSDC", "bidPrice": "600.00", "askPrice": "600.10"}
    cli.get_open_orders.return_value = []
    cli.get_trade_fee.return_value = [{"symbol": "BTCUSDC", "makerCommission": "0.001", "takerCommission": "0.001"}]

    place_mod = SimpleNamespace(_submit_binance_order=MagicMock())
    bapi_mod = SimpleNamespace(client=cli, _cancel_and_resolve=MagicMock())
    maintainer = BinanceFeeMaintainer(
        client=cli, bapi_mod=bapi_mod, placeorder_mod=place_mod, audit=ExecutionAudit(directory=str(tmp_path / "audit"))
    )

    res = maintainer.maintain(1_770_000_500.0, {}, load_fee_maintenance_config(_sample_cfg()))
    place_mod._submit_binance_order.assert_not_called()
    assert res["topup_action"] == "none"
    assert res["held_bnb_usdc"] == 120.0
    assert res["fee_spec"] == {"fee_tier": "VIP0_BNB", "maker_fee_bps": 7.5, "taker_fee_bps": 7.5}


def test_binance_cancels_stale_fee_order_and_reanchors_at_new_bid(tmp_path, monkeypatch):
    monkeypatch.setattr(botcore, "is_trade_enabled", lambda: True)
    now_ts = 1_770_010_000.0
    cli = MagicMock()
    cli.get_bnb_burn_spot_margin.return_value = {"spotBNBBurn": True}
    cli.get_account.return_value = {
        "balances": [
            {"asset": "BNB", "free": "0.0", "locked": "0.0"},
            {"asset": "USDC", "free": "900.0", "locked": "0.0"},
        ]
    }
    cli.get_orderbook_ticker.return_value = {"symbol": "BNBUSDC", "bidPrice": "640.00", "askPrice": "640.10"}
    cli.get_open_orders.return_value = [
        {
            "orderId": 111222,
            "clientOrderId": "FEE_BNB_1770007000",
            "side": "BUY",
            "origQty": "0.320",
            "executedQty": "0.0",
            "price": "620.00",
            "time": int((now_ts - 2400.0) * 1000),
        }
    ]
    cli.get_symbol_info.return_value = _bnb_symbol_info()
    cli.get_trade_fee.return_value = [{"symbol": "BTCUSDC", "makerCommission": "0.001", "takerCommission": "0.001"}]

    cancel_mock = MagicMock(return_value=True)
    submit_mock = MagicMock(return_value={"orderId": 333444, "price": "640.0", "origQty": "0.312", "status": "NEW"})
    maintainer = BinanceFeeMaintainer(
        client=cli,
        bapi_mod=SimpleNamespace(client=cli, _cancel_and_resolve=cancel_mock),
        placeorder_mod=SimpleNamespace(_submit_binance_order=submit_mock),
        audit=ExecutionAudit(directory=str(tmp_path / "audit")),
    )

    res = maintainer.maintain(now_ts, {}, load_fee_maintenance_config(_sample_cfg()))
    cancel_mock.assert_called_once_with("BNBUSDC", "111222")
    submit_mock.assert_called_once()
    assert res["canceled_stale_ids"] == ["111222"]
    assert res["placed_order_id"] == "333444"


def test_4h_schedule_gate_and_tca_evaluator_cache_dir_override(tmp_path, monkeypatch):
    monkeypatch.setattr(botcore, "is_trade_enabled", lambda: True)
    now_ts = 1_770_020_000.0

    # Fake maintainers for deterministic full-cycle orchestration
    bn_m = MagicMock(spec=VenueFeeMaintainerInterface)
    bn_m.maintain.return_value = {
        "ts": now_ts,
        "fee_spec": {"fee_tier": "VIP0_BNB", "maker_fee_bps": 7.5, "taker_fee_bps": 7.5},
        "held_bnb_usdc": 185.0,
        "has_pending_bnb_order": False,
        "retry_soon": False,
        "topup_action": "none",
        "events": [],
    }
    kr_p = MagicMock()
    kr_cli = MagicMock()
    kr_cli._private.return_value = {
        "volume": "14174.04",
        "fees": {"HYPEUSD": {"fee": "0.3800"}, "TAOUSD": {"fee": "0.3800"}},
        "fees_maker": {"HYPEUSD": {"fee": "0.2200"}, "TAOUSD": {"fee": "0.2200"}},
    }
    kr_p._client.return_value = kr_cli
    kr_p.usd_cash_breakdown.return_value = {"ZUSD": 1799.35, "USDC": 42.50}
    kr_m = KrakenFeeMaintainer(provider=kr_p)

    hl_p = MagicMock()
    hl_cli = SimpleNamespace(
        info=SimpleNamespace(
            user_fees=lambda _addr: {
                "userSpotAddRate": "0.0004",
                "userSpotCrossRate": "0.0007",
                "activeReferralDiscount": "0.0",
                "activeStakingDiscount": "0.0",
            }
        )
    )
    hl_p._hl.return_value = hl_cli
    hl_p._env_account_address.return_value = "0xabc123"
    hl_m = HyperliquidFeeMaintainer(provider=hl_p)
    t212_m = T212FeeMaintainer()

    events_seen = []
    r1 = run_venue_fee_maintenance(
        cache_dir=str(tmp_path),
        cfg_map=_sample_cfg(),
        maintainers={"binance": bn_m, "kraken": kr_m, "hyperliquid": hl_m, "t212": t212_m},
        event_handler=events_seen.append,
        now_ts=now_ts,
    )
    assert r1["errors"] == {}, r1["errors"]
    assert r1["status"] == "ok"
    assert len(events_seen) == 1
    assert "42.50 USDC" in events_seen[0]["body"]

    # Within 4h (e.g. +3600s), run_venue_fee_maintenance skips execution ("not_due")
    r2 = run_venue_fee_maintenance(
        cache_dir=str(tmp_path),
        cfg_map=_sample_cfg(),
        maintainers={"binance": bn_m, "kraken": kr_m, "hyperliquid": hl_m, "t212": t212_m},
        now_ts=now_ts + 3600.0,
    )
    assert r2["status"] == "not_due"
    assert bn_m.maintain.call_count == 1

    # After 4h (+14401s), runs again
    r3 = run_venue_fee_maintenance(
        cache_dir=str(tmp_path),
        cfg_map=_sample_cfg(),
        maintainers={"binance": bn_m, "kraken": kr_m, "hyperliquid": hl_m, "t212": t212_m},
        now_ts=now_ts + 14401.0,
    )
    assert r3["status"] == "ok"
    assert bn_m.maintain.call_count == 2

    # Without cache_dir, static _DEFAULT_FEE_SPEC remains unchanged; with cache_dir, live fees apply!
    with patch("providers.venue_fee_maintenance.resolve_now_ts", return_value=now_ts + 14410.0):
        static_bn = resolve_venue_fee_and_limits("binance", "BTCUSDC", None, price=90000.0)
        assert (static_bn["fee_tier"], static_bn["maker_fee_bps"], static_bn["taker_fee_bps"]) == ("VIP0", 10.0, 10.0)

        live_bn = resolve_venue_fee_and_limits("binance", "BTCUSDC", None, price=90000.0, cache_dir=str(tmp_path))
        assert (live_bn["fee_tier"], live_bn["maker_fee_bps"], live_bn["taker_fee_bps"]) == ("VIP0_BNB", 7.5, 7.5)

        live_kr = resolve_venue_fee_and_limits("kraken", "HYPEUSD", None, price=40.0, cache_dir=str(tmp_path))
        assert (live_kr["fee_tier"], live_kr["maker_fee_bps"], live_kr["taker_fee_bps"]) == ("PRO_LIVE", 22.0, 38.0)


def test_binance_trailing_stop_rebuy_uses_passive_maker_while_sell_keeps_market(tmp_path):
    calls = []
    fake_po = SimpleNamespace(
        place=lambda pair, side, price, qty, **kw: calls.append((pair, side, kw)) or {"orderId": "101", "status": "NEW"}
    )
    ts = b_ts.TrailingStop(
        api=MagicMock(),
        po=fake_po,
        sym=SimpleNamespace(symbols=["BTCUSDC"]),
        enabled=True,
        state_file=str(tmp_path / "trailing_state.json"),
    )
    ts._submit_lifecycle_order(
        {
            "intent_id": "INT_REBUY_1",
            "client_order_id": "TR_REBUY_1",
            "symbol": "BTCUSDC",
            "side": "BUY",
            "requested_qty": 0.01,
            "requested_price": 90000.0,
        },
        lambda *_a, **_kw: None,
        "BTCUSDC",
        "BUY",
        0.01,
        90000.0,
        kind="TRAIL_REBUY",
        market=False,
    )
    ts._submit_lifecycle_order(
        {
            "intent_id": "INT_SELL_1",
            "client_order_id": "TR_SELL_1",
            "symbol": "BTCUSDC",
            "side": "SELL",
            "requested_qty": 0.01,
            "requested_price": 90000.0,
        },
        lambda *_a, **_kw: None,
        "BTCUSDC",
        "SELL",
        0.01,
        90000.0,
        kind="TRAIL",
        market=True,
    )
    assert calls[0][2]["force"] is False
    assert calls[1][2]["force"] is True

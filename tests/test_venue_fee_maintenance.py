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
    build_persisted_venue_fee_resolver,
    load_fee_maintenance_config,
    read_persisted_venue_fee_schedule,
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

    # Without fee_spec_resolver, static _DEFAULT_FEE_SPEC remains unchanged; with injected fee_spec_resolver, live fees apply!
    with patch("providers.venue_fee_maintenance.resolve_now_ts", return_value=now_ts + 14410.0):
        static_bn = resolve_venue_fee_and_limits("binance", "BTCUSDC", None, price=90000.0)
        assert (static_bn["fee_tier"], static_bn["maker_fee_bps"], static_bn["taker_fee_bps"]) == ("VIP0", 10.0, 10.0)

        with patch(
            "providers.venue_fee_maintenance.read_persisted_venue_fee_schedule",
            wraps=read_persisted_venue_fee_schedule,
        ) as spy_read_sched:
            fee_resolver = build_persisted_venue_fee_resolver(tmp_path)
            assert fee_resolver is not None
            live_bn = resolve_venue_fee_and_limits("binance", "BTCUSDC", None, price=90000.0, fee_spec_resolver=fee_resolver)
            assert (live_bn["fee_tier"], live_bn["maker_fee_bps"], live_bn["taker_fee_bps"]) == ("VIP0_BNB", 7.5, 7.5)

            live_kr = resolve_venue_fee_and_limits("kraken", "HYPEUSD", None, price=40.0, fee_spec_resolver=fee_resolver)
            assert (live_kr["fee_tier"], live_kr["maker_fee_bps"], live_kr["taker_fee_bps"]) == ("PRO_LIVE", 22.0, 38.0)

            xstock_kr = resolve_venue_fee_and_limits("kraken", "SPCXxUSD", None, price=150.0, fee_spec_resolver=fee_resolver)
            assert (xstock_kr["fee_tier"], xstock_kr["maker_fee_bps"], xstock_kr["taker_fee_bps"]) == ("XSTOCK_ZERO_FEE", 0.0, 0.0)

            # Preloaded once in collector (zero hot-loop disk reads) and isolated against caller dict mutation
            assert spy_read_sched.call_count == 1
            mut_spec = fee_resolver("binance", "BTCUSDC")
            mut_spec["maker_fee_bps"] = 999.0
            assert fee_resolver("binance", "BTCUSDC")["maker_fee_bps"] == 7.5
    kr_pair_arg = kr_cli._private.call_args[0][1]["pair"]
    assert "SPCXXUSD" not in kr_pair_arg and "ANTXUSD" not in kr_pair_arg
    assert "HYPEUSD" in kr_pair_arg


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


def test_binance_per_symbol_fee_schedule_and_live_maker_cross_venue_arb(tmp_path, monkeypatch):
    """Verify BinanceFeeMaintainer stores by_symbol schedules and compute_cross_venue_arb_metrics uses live maker fees."""
    from intelligence.sentiment.evaluators.microstructure_evaluator import compute_cross_venue_arb_metrics
    from providers.venue_fee_maintenance import read_persisted_venue_fee_spec
    from state_io import save_json_state

    monkeypatch.setattr(botcore, "is_trade_enabled", lambda: True)
    now_ts = 1_770_050_000.0
    cli = MagicMock()
    cli.get_bnb_burn_spot_margin.return_value = {"spotBNBBurn": True}
    cli.get_account.return_value = {
        "balances": [
            {"asset": "BNB", "free": "0.30", "locked": "0.0"},
            {"asset": "USDC", "free": "1500.0", "locked": "0.0"},
        ]
    }
    cli.get_orderbook_ticker.return_value = {"symbol": "BNBUSDC", "bidPrice": "600.00", "askPrice": "600.10"}
    cli.get_open_orders.return_value = []
    cli.get_trade_fee.return_value = [
        {"symbol": "BTCUSDC", "makerCommission": "0.001", "takerCommission": "0.00095"},
        {"symbol": "TAOUSDC", "makerCommission": "0.001", "takerCommission": "0.001"},
        {"symbol": "ARBUSDC", "makerCommission": "0.0", "takerCommission": "0.00095"},
    ]

    maintainer = BinanceFeeMaintainer(
        client=cli,
        bapi_mod=SimpleNamespace(client=cli, _cancel_and_resolve=MagicMock()),
        placeorder_mod=SimpleNamespace(_submit_binance_order=MagicMock()),
        audit=ExecutionAudit(directory=str(tmp_path / "audit")),
    )
    res = maintainer.maintain(now_ts, {}, load_fee_maintenance_config(_sample_cfg()))
    assert res["fee_spec"] == {"fee_tier": "VIP0_BNB", "maker_fee_bps": 7.5, "taker_fee_bps": 7.12}
    assert res["by_symbol"]["BTCUSDC"] == {"fee_tier": "VIP0_BNB", "maker_fee_bps": 7.5, "taker_fee_bps": 7.12}
    assert res["by_symbol"]["TAOUSDC"] == {"fee_tier": "VIP0_BNB", "maker_fee_bps": 7.5, "taker_fee_bps": 7.5}
    assert res["by_symbol"]["ARBUSDC"] == {"fee_tier": "VIP0_BNB", "maker_fee_bps": 0.0, "taker_fee_bps": 7.12}

    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    save_json_state(
        str(state_dir / "venue_fee_maintenance_state.json"),
        {
            "schema_version": 1,
            "ts": now_ts,
            "interval_sec": 14400.0,
            "state_max_age_sec": 86400.0,
            "venues": {
                "binance": res,
                "kraken": {
                    "ts": now_ts,
                    "fee_spec": {"fee_tier": "PRO_LIVE", "maker_fee_bps": 22.0, "taker_fee_bps": 38.0},
                    "by_symbol": {
                        "TAOUSD": {"fee_tier": "PRO_LIVE", "maker_fee_bps": 22.0, "taker_fee_bps": 38.0}
                    },
                },
            },
        },
        indent=2,
    )
    assert read_persisted_venue_fee_spec("binance", "TAOUSDC", cache_dir=str(tmp_path), now_ts=now_ts)["taker_fee_bps"] == 7.5
    assert read_persisted_venue_fee_spec("binance", "ARBUSDC", cache_dir=str(tmp_path), now_ts=now_ts)["maker_fee_bps"] == 0.0

    # Cross-venue arb uses live persisted maker fees (Binance 7.5 vs Kraken 22.0) instead of static defaults (10 vs 16)
    m_bn = {"ob_mid_price": 200.0, "spread_bps": 4.0}
    m_kr = {"ob_mid_price": 200.0, "spread_bps": 10.0}
    compute_cross_venue_arb_metrics(
        "TAOUSDC",
        m_bn,
        {"price": 200.0},
        {"TAOUSDC": m_bn, "TAOUSD": m_kr},
        {"TAOUSDC": {"price": 200.0}, "TAOUSD": {"price": 200.0}},
        fee_spec_resolver=build_persisted_venue_fee_resolver(str(tmp_path), now_ts=now_ts),
        now_ts=now_ts,
    )
    assert m_bn["arb_fee_basis"] == "maker"
    assert m_bn["net_buy_edge_vs_peer_bps"] == pytest.approx(22.0 - 7.5, abs=0.1)
    assert m_bn["net_sell_edge_vs_peer_bps"] == pytest.approx(22.0 - 7.5, abs=0.1)
    # Taker edge uses live persisted taker fees (38 + 5) - (7.5 + 2) = +33.5 bps
    assert m_bn["net_buy_taker_edge_vs_peer_bps"] == pytest.approx(33.5, abs=0.1)
    assert m_bn["best_buy_venue"] == "binance:TAOUSDC"


def test_persisted_symbol_limits_override_fallback_ladder_and_align_tca_front_run(tmp_path, monkeypatch):
    """Verify BinanceFeeMaintainer persists exact exchange filters into cache_fee_schedule.json,
    resolve_venue_fee_and_limits consumes them (while keeping T212 static), BinanceProvider exposes
    cached rules, and OpenLimitOrdersTcaEvaluator snaps front-run prices to real tick_size."""
    from intelligence.sentiment.evaluators.tca_risk_evaluator import enrich_open_limit_orders_tca
    from providers.binance_provider import BinanceProvider
    from providers.quantity import BinanceOrderRules
    from state_stores.cache_fee_schedule import CacheFeeScheduleStore

    monkeypatch.setattr(botcore, "is_trade_enabled", lambda: True)
    now_ts = 1_770_060_000.0
    tao_info = {
        "symbol": "TAOUSDC",
        "baseAsset": "TAO",
        "quoteAsset": "USDC",
        "filters": [
            {"filterType": "PRICE_FILTER", "minPrice": "0.1", "maxPrice": "100000.0", "tickSize": "0.10"},
            {"filterType": "LOT_SIZE", "minQty": "0.0001", "maxQty": "9000.0", "stepSize": "0.0001"},
            {"filterType": "NOTIONAL", "minNotional": "5.00", "applyMinToMarket": True},
        ],
    }
    btc_info = {
        "symbol": "BTCUSDC",
        "baseAsset": "BTC",
        "quoteAsset": "USDC",
        "filters": [
            {"filterType": "PRICE_FILTER", "minPrice": "0.01", "maxPrice": "1000000.0", "tickSize": "0.01"},
            {"filterType": "LOT_SIZE", "minQty": "0.00001", "maxQty": "9000.0", "stepSize": "0.00001"},
            {"filterType": "NOTIONAL", "minNotional": "5.00", "applyMinToMarket": True},
        ],
    }
    cli = MagicMock()
    cli.get_bnb_burn_spot_margin.return_value = {"spotBNBBurn": True}
    cli.get_account.return_value = {
        "balances": [
            {"asset": "BNB", "free": "0.30", "locked": "0.0"},
            {"asset": "USDC", "free": "2000.0", "locked": "0.0"},
        ]
    }
    cli.get_orderbook_ticker.return_value = {"symbol": "BNBUSDC", "bidPrice": "600.00", "askPrice": "600.10"}
    cli.get_open_orders.return_value = []
    cli.get_trade_fee.return_value = [
        {"symbol": "TAOUSDC", "makerCommission": "0.001", "takerCommission": "0.00095"},
        {"symbol": "BTCUSDC", "makerCommission": "0.001", "takerCommission": "0.00095"},
    ]
    cli.get_exchange_info.return_value = {"symbols": [tao_info, btc_info]}

    maintainer = BinanceFeeMaintainer(
        client=cli,
        bapi_mod=SimpleNamespace(client=cli, _cancel_and_resolve=MagicMock()),
        placeorder_mod=SimpleNamespace(_submit_binance_order=MagicMock()),
        audit=ExecutionAudit(directory=str(tmp_path / "audit")),
    )
    res = maintainer.maintain(now_ts, {}, load_fee_maintenance_config(_sample_cfg()))
    assert res["symbol_limits"]["TAOUSDC"] == {
        "exchange_min_notional_usd": 5.0,
        "min_qty": 0.0001,
        "step_size": 0.0001,
        "tick_size": 0.1,
    }
    assert res["symbol_limits"]["BTCUSDC"]["step_size"] == 0.00001

    sched_path = CacheFeeScheduleStore.resolve_schedule_path(str(tmp_path))
    CacheFeeScheduleStore.save_schedule(
        sched_path,
        {
            "schema_version": 1,
            "ts": now_ts,
            "interval_sec": 14400.0,
            "state_max_age_sec": 86400.0,
            "venues": {
                "binance": res,
                "t212": {
                    "ts": now_ts,
                    "fee_spec": {"fee_tier": "INVEST_RTH", "maker_fee_bps": 15.0, "taker_fee_bps": 15.0},
                    "symbol_limits": {"NVDA_US_EQ": {"min_qty": 0.001, "step_size": 0.001, "tick_size": 0.001}},
                },
            },
        },
        indent=2,
    )
    resolver = build_persisted_venue_fee_resolver(str(tmp_path), now_ts=now_ts)
    tao_lim = resolve_venue_fee_and_limits("binance", "TAOUSDC", None, price=183.5, fee_spec_resolver=resolver)
    assert tao_lim["step_size"] == 0.0001
    assert tao_lim["min_qty"] == 0.0001
    assert tao_lim["tick_size"] == 0.1

    # Static-precision T212 stays protected at 0.01/0.01/0.01 even if stale limits exist
    nvda_lim = resolve_venue_fee_and_limits("t212", "NVDA_US_EQ", None, price=150.0, fee_spec_resolver=resolver)
    assert (nvda_lim["min_qty"], nvda_lim["step_size"], nvda_lim["tick_size"]) == (0.01, 0.01, 0.01)

    # BinanceProvider.extract_cached_fee_and_limits reads cached BinanceOrderRules without REST
    bp = BinanceProvider(live_enabled=False)
    bp._cache_symbol_rules("TAOUSDC", BinanceOrderRules.from_symbol_info(tao_info))
    assert BinanceProvider.extract_cached_fee_and_limits(bp, "TAOUSDC")["tick_size"] == 0.1

    # OpenLimitOrdersTcaEvaluator snaps front-run prices to TAOUSDC's real 0.10 tick (never 185.39)
    orders = [{"venue": "binance", "symbol": "TAOUSDC", "side": "SELL", "price": 186.0, "qty": 2.0}]
    enrich_open_limit_orders_tca(
        orders,
        {"TAOUSDC": {"price": 183.5, "vol_1h_pct": 1.2}},
        {
            "TAOUSDC": {
                "best_bid": 183.4,
                "best_ask": 183.5,
                "ob_mid_price": 183.45,
                "spread_bps": 5.45,
                "ask_depth_usd": 100000.0,
                "ask_wall_px": 185.4,
                "ask_wall_usd": 40000.0,
                "_ob_raw_ts": now_ts - 5.0,
            }
        },
        now_ts=now_ts,
        tradable_positions=[{"venue": "binance", "symbol": "TAOUSDC", "tick_size": 0.1}],
    )
    assert orders[0]["wall_queue_pos"] == "BEHIND_ASK_WALL"
    assert orders[0]["recommended_front_run_px"] == 185.3


def test_fee_maintenance_partial_error_shortens_retry_and_rejects_corrupt_venue_ts(tmp_path):
    """Partial-venue failures retry at stale_sec instead of waiting 4h, and corrupt per-venue ts=0 is never resurrected by top-level ts."""
    from providers.venue_fee_maintenance import read_persisted_venue_fee_spec
    from state_stores.cache_fee_schedule import CacheFeeScheduleStore

    now_ts = 1_770_200_000.0
    fail_once = {"fail": True}

    class _FlakyMaintainer(VenueFeeMaintainerInterface):
        venue = "binance"

        def maintain(self, now_ts, prev_venue_state, cfg, *, dry_run=False):
            if fail_once["fail"]:
                raise RuntimeError("VPN timeout")
            return {
                "ts": round(now_ts, 3),
                "fee_spec": {"fee_tier": "VIP0_BNB", "maker_fee_bps": 7.5, "taker_fee_bps": 7.5},
                "has_pending_bnb_order": False,
                "retry_soon": False,
                "events": [],
            }

    maintainers = {"binance": _FlakyMaintainer(), "t212": T212FeeMaintainer()}
    r1 = run_venue_fee_maintenance(
        venues=["binance", "t212"],
        cache_dir=str(tmp_path),
        cfg_map=_sample_cfg(),
        maintainers=maintainers,
        now_ts=now_ts,
    )
    assert r1["status"] == "partial_error"
    assert "binance" in r1["errors"]

    # Within retry cooldown (+600s < 1800s) -> not_due (prevents spam)
    r_wait = run_venue_fee_maintenance(
        venues=["binance", "t212"],
        cache_dir=str(tmp_path),
        cfg_map=_sample_cfg(),
        maintainers=maintainers,
        now_ts=now_ts + 600.0,
    )
    assert r_wait["status"] == "not_due"

    # After shortened retry interval (+1801s < 14400s) -> retries and recovers instead of waiting 4 hours
    fail_once["fail"] = False
    r_rec = run_venue_fee_maintenance(
        venues=["binance", "t212"],
        cache_dir=str(tmp_path),
        cfg_map=_sample_cfg(),
        maintainers=maintainers,
        now_ts=now_ts + 1801.0,
    )
    assert r_rec["status"] == "ok"
    assert r_rec["errors"] == {}

    # Once recovered, normal 4h cadence resumes (+1801s after recovery is not_due)
    r_after = run_venue_fee_maintenance(
        venues=["binance", "t212"],
        cache_dir=str(tmp_path),
        cfg_map=_sample_cfg(),
        maintainers=maintainers,
        now_ts=now_ts + 3602.0,
    )
    assert r_after["status"] == "not_due"

    # Corrupt per-venue ts=0 MUST NOT fall back to fresh top-level payload["ts"]
    sched_path = CacheFeeScheduleStore.resolve_schedule_path(str(tmp_path))
    CacheFeeScheduleStore.save_schedule(
        sched_path,
        {
            "schema_version": 1,
            "ts": now_ts + 3600.0,
            "interval_sec": 14400.0,
            "state_max_age_sec": 86400.0,
            "venues": {
                "binance": {
                    "ts": 0,
                    "fee_spec": {"fee_tier": "VIP0_BNB", "maker_fee_bps": 7.5, "taker_fee_bps": 7.5},
                }
            },
        },
        indent=2,
    )
    assert read_persisted_venue_fee_spec("binance", "BTCUSDC", cache_dir=str(tmp_path), now_ts=now_ts + 3605.0) == {}


def test_maintain_venue_fees_cli_nonzero_exit_when_all_venues_fail_with_empty_venues(monkeypatch):
    """Pure invariant + cross-check: maintain_venue_fees.main returns exit 1 when errors is non-empty and venues is empty."""
    from orchestratorOS.admin import maintain_venue_fees as mvf

    monkeypatch.setattr(mvf.botcore, "load_env_stack", lambda: None)

    # All venues failed before populating result["venues"] -> must exit 1 (never 0)
    monkeypatch.setattr(
        mvf,
        "run_venue_fee_maintenance",
        lambda **_kw: {"status": "error", "venues": {}, "errors": {"binance": "timeout"}},
    )
    assert mvf.main(["--json"]) == 1

    # Partial failure (1 error out of 2 venues populated) -> exits 0
    monkeypatch.setattr(
        mvf,
        "run_venue_fee_maintenance",
        lambda **_kw: {
            "status": "partial_error",
            "venues": {"binance": {"fee_spec": {}}, "t212": {"fee_spec": {}}},
            "errors": {"binance": "timeout"},
        },
    )
    assert mvf.main(["--json"]) == 0

    # All populated venues errored -> exits 1
    monkeypatch.setattr(
        mvf,
        "run_venue_fee_maintenance",
        lambda **_kw: {
            "status": "error",
            "venues": {"binance": {}},
            "errors": {"binance": "timeout"},
        },
    )
    assert mvf.main(["--json"]) == 1




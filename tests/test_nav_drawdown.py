"""Unit and integration tests for Unified NAV history persistence, cashflow-adjusted drawdown, and asymmetric guard."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from unittest.mock import MagicMock
import pytest

from botcore import load_key_value_conf
from cachepaths import NAV_HISTORY_STATE_FILENAME, resolve_cache_subdomain
from orchestratorOS.supervisor.prompt_sections import _format_portfolio_section
from orchestratorTrade.autonomous_intent_gateway import AutonomousIntentGateway, ROOT_DIR as GW_ROOT_DIR
from providers.ai_attribution import format_ai_attribution_report
from providers.ai_attribution_store import load_and_evaluate_ai_attribution
from providers.ai_circuit_breaker import parse_intent_circuit_breaker_config
from providers.nav_drawdown import (
    append_nav_history_point,
    build_nav_history_point,
    evaluate_daily_nav_drawdown_guard,
    evaluate_nav_drawdown_metrics,
    format_nav_drawdown_summary_line,
)
from providers.nav_history_store import (
    load_and_evaluate_nav_drawdown,
    load_nav_history_state,
    record_nav_history_snapshot,
)


def test_nav_drawdown_config_and_cashflow_immune_metrics():
    """Verify cachepaths registration, order_guard.conf defaults, and cashflow-adjusted NAV drawdown math."""
    assert resolve_cache_subdomain(NAV_HISTORY_STATE_FILENAME) == "portfolio"

    root_cfg = load_key_value_conf(GW_ROOT_DIR, GW_ROOT_DIR)
    cb_cfg = parse_intent_circuit_breaker_config(root_cfg)
    assert cb_cfg["daily_nav_dd_mode"] == "shadow"
    assert cb_cfg["daily_nav_dd_limit_pct"] == pytest.approx(5.0)

    # Multi-venue NAV point respects free_base, locked_base_qty, free_cash, and live prices across all 4 venues
    vb = {
        "cash:binance:USDC": {"venue": "binance", "asset": "USDC", "free": 3000.0},
        "pos:binance:BTCUSDC": {"venue": "binance", "symbol": "BTCUSDC", "base": "BTC", "free_base": 0.1, "price": 70000.0, "value_usd": 7000.0},
        "cash:kraken:USD": {"venue": "kraken", "asset": "USD", "free": 2000.0},
        "pos:kraken:TAOUSD": {"venue": "kraken", "symbol": "TAOUSD", "base": "TAO", "free_base": 1.0, "locked_base_qty": 4.0, "price": 200.0, "value_usd": 200.0, "locked_base_usd": 800.0},
        "cash:t212:USD": {"venue": "t212", "asset": "USD", "free": 1000.0},
        "pos:t212:NVDA_US_EQ": {"venue": "t212", "symbol": "NVDA_US_EQ", "base": "NVDA_US_EQ", "free_base": 10.0, "price": 150.0, "value_usd": 1500.0},
    }
    pt = build_nav_history_point(vb, now_ts=1_795_400_000.0)
    assert pt is not None
    assert pt["nav_usd"] == pytest.approx(15500.0)  # 10000 bn + 3000 kr + 2500 t212
    assert pt["invested_usd"] == pytest.approx(9500.0)
    assert pt["free_cash_usd"] == pytest.approx(6000.0)
    assert set(pt["venues_usd"].keys()) == {"binance", "kraken", "t212"}

    # Build a UTC day sequence with:
    # 1) Opening NAV $20,000 ($14k invested + $6k cash)
    # 2) External withdrawal -$2,000 cash -> NAV $18,000 ($14k invested + $4k cash): MUST NOT trigger DD!
    # 3) External deposit +$4,000 cash -> NAV $22,000 ($14k invested + $8k cash): MUST NOT fake +22% gain!
    # 4) Temporary venue outage dropping t212 -> NAV $19,500: MUST NOT fake -11% crash!
    # 5) Real market crash -6.0% on invested positions (cash unchanged) -> MUST score exact -6.0% DD!
    day0_utc = datetime(2026, 11, 20, 0, 10, tzinfo=timezone.utc).timestamp()
    hist: list[dict] = []
    hist, ok1 = append_nav_history_point(
        hist,
        {"ts": day0_utc, "nav_usd": 20000.0, "invested_usd": 14000.0, "free_cash_usd": 6000.0, "venues_usd": {"binance": 12000.0, "kraken": 5500.0, "t212": 2500.0}},
    )
    assert ok1 is True

    # Sub-10m duplicate is throttled
    hist, ok_dup = append_nav_history_point(
        hist,
        {"ts": day0_utc + 120.0, "nav_usd": 19950.0, "invested_usd": 13950.0, "free_cash_usd": 6000.0, "venues_usd": {"binance": 11950.0, "kraken": 5500.0, "t212": 2500.0}},
    )
    assert ok_dup is False and len(hist) == 1

    # Confirmed withdrawal -$2,000 cash (-10% NAV)
    hist, ok_wd = append_nav_history_point(
        hist,
        {
            "ts": day0_utc + 3600.0,
            "nav_usd": 18000.0,
            "invested_usd": 14000.0,
            "free_cash_usd": 4000.0,
            "venues_usd": {"binance": 10000.0, "kraken": 5500.0, "t212": 2500.0},
            "confirmed_cashflow_usd": -2000.0,
        },
    )
    assert ok_wd is True and hist[-1].get("external_cashflow") is True
    m_after_wd = evaluate_nav_drawdown_metrics(hist, now_ts=day0_utc + 3600.0)
    assert m_after_wd["daily_nav_drawdown_pct"] == pytest.approx(0.0)
    assert m_after_wd["daily_nav_return_pct"] == pytest.approx(0.0)
    assert m_after_wd["external_cashflow_events"] == 1

    # Confirmed deposit +$4,000 cash (+22.2% NAV)
    hist, _ = append_nav_history_point(
        hist,
        {
            "ts": day0_utc + 7200.0,
            "nav_usd": 22000.0,
            "invested_usd": 14000.0,
            "free_cash_usd": 8000.0,
            "venues_usd": {"binance": 14000.0, "kraken": 5500.0, "t212": 2500.0},
            "confirmed_cashflow_usd": 4000.0,
        },
    )
    assert hist[-1].get("external_cashflow") is True
    m_after_dep = evaluate_nav_drawdown_metrics(hist, now_ts=day0_utc + 7200.0)
    assert m_after_dep["daily_nav_return_pct"] == pytest.approx(0.0)

    # Venue outage (t212 missing from venues_usd)
    hist, _ = append_nav_history_point(
        hist,
        {"ts": day0_utc + 10800.0, "nav_usd": 19500.0, "invested_usd": 12500.0, "free_cash_usd": 7000.0, "venues_usd": {"binance": 14000.0, "kraken": 5500.0}},
    )
    assert hist[-1].get("venue_set_changed") is True
    m_after_out = evaluate_nav_drawdown_metrics(hist, now_ts=day0_utc + 10800.0)
    assert m_after_out["daily_nav_drawdown_pct"] == pytest.approx(0.0)

    # Real market crash -6.0% on invested positions (same venues, cash unchanged!)
    hist, _ = append_nav_history_point(
        hist,
        {"ts": day0_utc + 14400.0, "nav_usd": 18330.0, "invested_usd": 11330.0, "free_cash_usd": 7000.0, "venues_usd": {"binance": 13160.0, "kraken": 5170.0}},
    )
    assert not hist[-1].get("external_cashflow") and not hist[-1].get("venue_set_changed")
    m_crash = evaluate_nav_drawdown_metrics(hist, now_ts=day0_utc + 14400.0)
    assert m_crash["daily_nav_drawdown_pct"] == pytest.approx(-6.0, abs=0.05)
    assert m_crash["daily_nav_return_pct"] == pytest.approx(-6.0, abs=0.05)
    assert m_crash["max_drawdown_7d_pct"] == pytest.approx(-6.0, abs=0.05)

    # Guard evaluation: BUY blocked at -6.0% <= -5.0%; SELL always allowed; mode='off' always allowed
    ok_b, rsn_b, gm_b = evaluate_daily_nav_drawdown_guard(m_crash, decision="BUY", mode="enforce", limit_pct=5.0)
    assert ok_b is False and "ai_daily_nav_drawdown_guard" in rsn_b and gm_b["daily_nav_drawdown_pct"] == pytest.approx(-6.0, abs=0.05)
    ok_s, _, _ = evaluate_daily_nav_drawdown_guard(m_crash, decision="SELL", mode="enforce", limit_pct=5.0)
    assert ok_s is True
    ok_off, _, _ = evaluate_daily_nav_drawdown_guard(m_crash, decision="BUY", mode="off", limit_pct=5.0)
    assert ok_off is True


def test_nav_drawdown_multiday_sharpe_and_retention():
    """Verify multi-day UTC close Sharpe calculation, 7d/30d MaxDD, and 90-day retention trimming."""
    base_ts = datetime(2026, 11, 10, 23, 0, tzinfo=timezone.utc).timestamp()
    # 5 daily closes with a real -8% trough on day 3 and partial recovery on day 5
    daily_navs = [10000.0, 10200.0, 9384.0, 9700.0, 10100.0]
    hist: list[dict] = []
    # Add one ancient point > 90 days old that must be pruned
    hist, _ = append_nav_history_point(
        hist,
        {"ts": base_ts - 100 * 86400.0, "nav_usd": 5000.0, "invested_usd": 4000.0, "free_cash_usd": 1000.0},
    )
    for idx, nav_v in enumerate(daily_navs):
        hist, _ = append_nav_history_point(
            hist,
            {
                "ts": base_ts + idx * 86400.0,
                "nav_usd": nav_v,
                "invested_usd": nav_v - 2000.0,
                "free_cash_usd": 2000.0,
                "venues_usd": {"binance": nav_v},
            },
        )
    assert len(hist) == 5  # ancient 100d point trimmed
    m = evaluate_nav_drawdown_metrics(hist, now_ts=base_ts + 4 * 86400.0)
    assert m["max_drawdown_7d_pct"] == pytest.approx(-8.0, abs=0.05)
    assert m["max_drawdown_30d_pct"] == pytest.approx(-8.0, abs=0.05)
    assert m["daily_sharpe_30d"] is not None
    summary = format_nav_drawdown_summary_line(m)
    assert "Unified NAV=$10,100.00" in summary and "7d_maxdd=-8.00%" in summary


def test_nav_drawdown_gateway_shadow_vs_enforce_attribution_and_section3c(tmp_path):
    """End-to-end test: disk snapshot RMW, gateway shadow vs enforce asymmetric guard, attribution report, and Section 3C."""
    now_s = datetime.now(timezone.utc).timestamp()
    # Ensure both snapshots fall within the same UTC day
    utc_midnight = datetime.fromtimestamp(now_s, timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    t_open = max(utc_midnight + 60.0, now_s - 3600.0)
    t_crash = max(t_open + 700.0, now_s - 60.0)

    (tmp_path / "cachedb" / "state").mkdir(parents=True, exist_ok=True)
    (tmp_path / "cachedb" / "prices").mkdir(parents=True, exist_ok=True)

    # Record opening NAV = $10,000 ($8,000 BTC + $2,000 USDC), then crash to $9,400 ($7,400 BTC + $2,000 USDC = -6.0% DD)
    vb_open = {
        "cash:binance:USDC": {"venue": "binance", "asset": "USDC", "free": 2000.0},
        "pos:binance:BTCUSDC": {"venue": "binance", "symbol": "BTCUSDC", "base": "BTC", "free_base": 0.1, "price": 80000.0, "value_usd": 8000.0},
    }
    vb_crash = {
        "cash:binance:USDC": {"venue": "binance", "asset": "USDC", "free": 2000.0},
        "pos:binance:BTCUSDC": {"venue": "binance", "symbol": "BTCUSDC", "base": "BTC", "free_base": 0.1, "price": 74000.0, "value_usd": 7400.0},
    }
    (tmp_path / "cachedb" / "state" / "venue_balances_state.json").write_text(json.dumps(vb_crash), encoding="utf-8")
    (tmp_path / "cachedb" / "prices" / "cache_current_price.json").write_text(
        json.dumps({"items": {"BTCUSDC": 74000.0}}), encoding="utf-8"
    )

    record_nav_history_snapshot(str(tmp_path), venue_balances_state=vb_open, live_prices={"BTCUSDC": 80000.0}, now_ts=t_open)
    m_disk = record_nav_history_snapshot(str(tmp_path), venue_balances_state=vb_crash, live_prices={"BTCUSDC": 74000.0}, now_ts=t_crash)
    assert len(load_nav_history_state(str(tmp_path))) == 2
    assert m_disk["daily_nav_drawdown_pct"] == pytest.approx(-6.0, abs=0.05)

    # 1. Gateway in daily_nav_dd_mode = shadow: logs warning & stamps intent, still executes BUY live
    (tmp_path / "order_guard.conf").write_text(
        "autonomous_trading_mode = enforce\n"
        "autonomous_trading_min_notional_usd = 10.0\n"
        "autonomous_trading_max_notional_usd = 2000.0\n"
        "autonomous_trading_min_confidence = 0.85\n"
        "autonomous_trading_notify = 0\n"
        "autonomous_trading_daily_nav_dd_mode = shadow\n"
        "autonomous_trading_daily_nav_dd_limit_pct = 5.0\n",
        encoding="utf-8",
    )
    gw_shadow = AutonomousIntentGateway(workspace_dir=str(tmp_path))
    btc_inst = MagicMock(symbol="BTCUSDC", provider_name="binance", base="BTC", quote="USDC", params={})
    btc_inst._provider.name = "binance"
    btc_inst._provider.get_current_price.return_value = 74000.0
    btc_inst._provider.free_balance.side_effect = lambda a: 2000.0 if a == "USDC" else 0.1
    btc_inst.open_orders.return_value = []
    btc_inst.place.return_value = {"orderId": "ord_nav_shadow", "status": "NEW", "executedQty": "0.0"}

    rec_sh = gw_shadow.submit(
        {
            "decision": "BUY",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "order_type": "LIMIT",
            "limit_price": 73800.0,
            "confidence": 0.92,
            "suggested_notional_usd": 300.0,
            "urgency": "NORMAL",
        },
        source="auto_cryptotrade_executor",
        max_mode="enforce",
        instruments_map={"BTCUSDC": btc_inst},
    )
    assert rec_sh["status"] == "EXECUTED"
    assert "ai_daily_nav_drawdown_guard" in rec_sh.get("daily_nav_dd_shadow_warning", "")
    btc_inst.place.assert_called_once()

    # 2. Gateway in daily_nav_dd_mode = enforce: downgrades BUY to OBSERVED_SHADOW while allowing protective SELL live!
    (tmp_path / "order_guard.conf").write_text(
        "autonomous_trading_mode = enforce\n"
        "autonomous_trading_min_notional_usd = 10.0\n"
        "autonomous_trading_max_notional_usd = 2000.0\n"
        "autonomous_trading_min_confidence = 0.85\n"
        "autonomous_trading_notify = 0\n"
        "autonomous_trading_daily_nav_dd_mode = enforce\n"
        "autonomous_trading_daily_nav_dd_limit_pct = 5.0\n",
        encoding="utf-8",
    )
    gw_enf = AutonomousIntentGateway(workspace_dir=str(tmp_path))
    gw_enf.save_all_intents([])
    btc_inst.place.reset_mock()
    btc_inst.place.return_value = {"orderId": "ord_nav_sell", "status": "NEW", "executedQty": "0.0"}

    rec_buy_enf = gw_enf.submit(
        {
            "decision": "BUY",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "order_type": "LIMIT",
            "limit_price": 73800.0,
            "confidence": 0.95,
            "suggested_notional_usd": 300.0,
            "urgency": "HIGH",
        },
        source="auto_cryptotrade_executor",
        max_mode="enforce",
        instruments_map={"BTCUSDC": btc_inst},
    )
    assert rec_buy_enf["status"] == "OBSERVED_SHADOW"
    assert "ai_daily_nav_drawdown_guard" in rec_buy_enf.get("daily_nav_dd_downgraded_to_shadow", "")
    btc_inst.place.assert_not_called()

    rec_sell_enf = gw_enf.submit(
        {
            "decision": "SELL",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "order_type": "LIMIT",
            "limit_price": 74100.0,
            "confidence": 0.92,
            "suggested_notional_usd": 300.0,
            "urgency": "NORMAL",
        },
        source="auto_cryptotrade_executor",
        max_mode="enforce",
        instruments_map={"BTCUSDC": btc_inst},
    )
    assert rec_sell_enf["status"] == "EXECUTED"
    btc_inst.place.assert_called_once()

    # 3. Attribution loader/formatter surfaces Unified NAV & Drawdown Curve
    attr = load_and_evaluate_ai_attribution(str(tmp_path), lookback_days=7.0, now_ts=t_crash + 10.0)
    assert attr["unified_nav_drawdown"]["daily_nav_drawdown_pct"] == pytest.approx(-6.0, abs=0.05)
    rep_txt = format_ai_attribution_report(attr)
    assert "-- Unified NAV & Drawdown Curve --" in rep_txt
    assert "day_dd=-6.00%" in rep_txt

    # 4. Section 3C renders single raw numeric token nav_dd_day=-6.00% without rigid labels
    sec3c = "\n".join(
        _format_portfolio_section(
            {
                "portfolio_risk_engine": {
                    "unified_nav_usd": 9400.0,
                    "invested_exposure_usd": 7400.0,
                    "global_free_cash_usd": 2000.0,
                    "global_free_cash_pct": 21.3,
                    "intraday_var_95_usd": 350.0,
                    "intraday_var_95_pct": 3.7,
                    "daily_nav_drawdown_pct": -6.0,
                }
            },
            "crypto",
        )
    )
    assert "nav_dd_day=-6.00%" in sec3c


def test_nav_drawdown_small_venue_outage_below_cashflow_threshold():
    """Verify temporary outage of a small venue (<4% of NAV) is flagged as venue_set_changed and never pollutes daily DD."""
    day0_utc = datetime(2026, 11, 20, 0, 0, 0, tzinfo=timezone.utc).timestamp()
    hist: list[dict] = []
    hist, _ = append_nav_history_point(
        hist,
        {
            "ts": day0_utc + 3600.0,
            "nav_usd": 20000.0,
            "invested_usd": 14000.0,
            "free_cash_usd": 6000.0,
            "venues_usd": {"binance": 14000.0, "kraken": 5500.0, "hyperliquid": 500.0},
        },
    )
    # Hyperliquid ($500 = 2.5% < 4.0% cashflow jump threshold) temporarily disappears from balances snapshot
    hist, _ = append_nav_history_point(
        hist,
        {
            "ts": day0_utc + 7200.0,
            "nav_usd": 19500.0,
            "invested_usd": 13500.0,
            "free_cash_usd": 6000.0,
            "venues_usd": {"binance": 14000.0, "kraken": 5500.0},
        },
    )
    assert hist[-1].get("venue_set_changed") is True
    m_out = evaluate_nav_drawdown_metrics(hist, now_ts=day0_utc + 7200.0)
    assert m_out["daily_nav_drawdown_pct"] == pytest.approx(0.0)
    assert m_out["external_cashflow_events"] == 1


def test_nav_drawdown_resets_daily_dd_after_utc_midnight_when_history_is_from_prior_day():
    """Verify yesterday's -6% daily drawdown resets to 0.0% on the new UTC day before today's first sample while preserving rolling 24h DD."""
    day0_utc = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc).timestamp()
    day1_00_05_utc = datetime(2026, 11, 20, 0, 5, 0, tzinfo=timezone.utc).timestamp()
    hist: list[dict] = []
    hist, _ = append_nav_history_point(
        hist,
        {"ts": day0_utc + 12 * 3600.0, "nav_usd": 20000.0, "invested_usd": 15000.0, "free_cash_usd": 5000.0, "venues_usd": {"binance": 20000.0}},
    )
    hist, _ = append_nav_history_point(
        hist,
        {"ts": day0_utc + 23 * 3600.0, "nav_usd": 18800.0, "invested_usd": 13800.0, "free_cash_usd": 5000.0, "venues_usd": {"binance": 18800.0}},
    )
    m_next_day = evaluate_nav_drawdown_metrics(hist, now_ts=day1_00_05_utc)
    assert m_next_day["utc_day"] == "2026-11-20"
    assert m_next_day["daily_nav_drawdown_pct"] == pytest.approx(0.0)
    assert m_next_day["daily_nav_return_pct"] == pytest.approx(0.0)
    assert m_next_day["rolling_24h_drawdown_pct"] == pytest.approx(-6.0, abs=0.05)


def test_nav_drawdown_scoped_projection_and_live_point_isolate_crypto_from_equities():
    """Verify scope='crypto' and scope='equities' project multi-venue NAV history and live points independently."""
    day0_utc = datetime(2026, 11, 20, 0, 0, 0, tzinfo=timezone.utc).timestamp()
    hist: list[dict] = []
    hist, _ = append_nav_history_point(
        hist,
        {
            "ts": day0_utc + 3600.0,
            "nav_usd": 20000.0,
            "invested_usd": 14000.0,
            "free_cash_usd": 6000.0,
            "venues_usd": {"binance": 10000.0, "kraken": 8000.0, "t212": 2000.0},
        },
    )
    # T212 drops 10% (-$200) while crypto stays flat ($18,000)
    hist, _ = append_nav_history_point(
        hist,
        {
            "ts": day0_utc + 7200.0,
            "nav_usd": 19800.0,
            "invested_usd": 13800.0,
            "free_cash_usd": 6000.0,
            "venues_usd": {"binance": 10000.0, "kraken": 8000.0, "t212": 1800.0},
        },
    )
    # Live crypto mark-to-market drops -1.0% ($18,000 -> $17,820) with scoped venues_usd
    live_crypto_pt = {
        "ts": day0_utc + 7500.0,
        "nav_usd": 17820.0,
        "invested_usd": 11820.0,
        "free_cash_usd": 6000.0,
        "venues_usd": {"binance": 9820.0, "kraken": 8000.0},
    }
    m_crypto = evaluate_nav_drawdown_metrics(
        hist, now_ts=day0_utc + 7500.0, current_point=live_crypto_pt, scope="crypto"
    )
    assert m_crypto["current_nav_usd"] == pytest.approx(17820.0)
    assert m_crypto["daily_nav_drawdown_pct"] == pytest.approx(-1.0, abs=0.02)
    assert set(m_crypto["venues_usd"].keys()) == {"binance", "kraken"}

    m_eq = evaluate_nav_drawdown_metrics(hist, now_ts=day0_utc + 7500.0, scope="equities")
    assert m_eq["current_nav_usd"] == pytest.approx(1800.0)
    assert m_eq["daily_nav_drawdown_pct"] == pytest.approx(-10.0, abs=0.05)
    assert set(m_eq["venues_usd"].keys()) == {"t212"}


def test_compute_venue_nav_breakdown_prefers_fresh_quote_variant_over_stale_retired_pair():
    """Verify retired stale TAOUSD (2d old) is ignored when fresh TAOUSDC exists in cache_current_price."""
    from providers.portfolio_valuation import compute_venue_nav_breakdown

    now_s = 1_795_631_300.0
    nav = compute_venue_nav_breakdown(
        venue_balance_caches={
            "binance": {
                "venue": "binance",
                "items": {
                    "USDC": {"free": "1000.0", "locked": "0.0"},
                    "TAO": {"free": "4.0", "locked": "0.0"},
                },
            }
        },
        live_prices={
            "items": {
                "TAOUSD": [[int((now_s - 172_800.0) * 1000), 191.50]],
                "TAOUSDC": [[int((now_s - 2.0) * 1000), 173.70]],
            }
        },
        scope="crypto",
    )
    assert nav["venues_usd"]["binance"] == pytest.approx(1000.0 + round(4.0 * 173.70, 2), abs=0.05)
    assert nav["portfolio_total_usd"] == pytest.approx(1000.0 + round(4.0 * 173.70, 2), abs=0.05)


def test_macro_snapshot_nav_dd_includes_live_bnb_reserve_parity(tmp_path):
    """Verify PortfolioAndVenueCyclesCollector uses live BNB qty and _finalize_snapshot matches nav_history BNB parity."""
    from intelligence.macro_sentiment_ssot import TelemetrySensorSnapshot
    from intelligence.sentiment.collectors.macro_telemetry_collector import (
        MarketAndPortfolioSnapshotOrchestrator,
        PortfolioAndVenueCyclesCollector,
    )
    from providers.portfolio_state_store import resolve_venue_balance_cache_path
    from providers.venue_fee_maintenance import resolve_fee_schedule_cache_path
    from state_io import atomic_write_json
    from state_stores.nav_history_state import NavHistoryStateStore

    now_s = datetime(2026, 11, 25, 18, 0, 0, tzinfo=timezone.utc).timestamp()
    cdir = tmp_path / "cachedb"
    atomic_write_json(
        resolve_fee_schedule_cache_path(str(cdir)),
        {"venues": {"binance": {"held_bnb_usdc": 111.13, "bnb_price_usdc": 540.0}}},
    )
    atomic_write_json(
        resolve_venue_balance_cache_path(str(cdir), "binance"),
        {"venue": "binance", "items": {"BNB": {"free": "0.20", "locked": "0.0"}}},
    )
    ps: dict = {}
    PortfolioAndVenueCyclesCollector(start_sync=False).collect(
        str(cdir), workspace_dir=str(tmp_path), portfolio_summary=ps, now_ts=now_s
    )
    # 0.20 BNB * $540 = $108.00 (live balance cache beats stale 4h $111.13 fee-schedule snapshot)
    assert ps["binance_bnb_fee_reserve_usdc"] == pytest.approx(108.0)

    # Seed nav_history_state.json including $108 BNB inside Binance ($10,108 total)
    atomic_write_json(
        NavHistoryStateStore.resolve_workspace_path(str(tmp_path)),
        {
            "points": [
                {
                    "ts": now_s - 3600.0,
                    "nav_usd": 10108.0,
                    "invested_usd": 5000.0,
                    "free_cash_usd": 5000.0,
                    "venues_usd": {"binance": 10108.0},
                },
                {
                    "ts": now_s - 1800.0,
                    "nav_usd": 10108.0,
                    "invested_usd": 5000.0,
                    "free_cash_usd": 5000.0,
                    "venues_usd": {"binance": 10108.0},
                },
            ]
        },
    )
    snap = TelemetrySensorSnapshot(
        portfolio_summary={
            "binance_bnb_fee_reserve_usdc": 108.0,
            "tradable_positions": [
                {"venue": "binance", "symbol": "BTCUSDC", "free_base": 0.1, "price": 50000.0, "value_usd": 5000.0}
            ],
            "venue_quote_cash": {"binance:USDC": {"venue": "binance", "asset": "USDC", "free": 5000.0}},
        },
        prices_and_trends={"BTCUSDC": {"price": 50000.0, "vol_1h_pct": 1.0}},
    )
    MarketAndPortfolioSnapshotOrchestrator._apply_macro_scope_and_risk_overlays(
        snap,
        scope="crypto",
        now_ts=now_s,
        inst_by_sym={},
        t212_cycles=None,
        workspace_dir=str(tmp_path),
    )
    risk_eng = snap.portfolio_summary["portfolio_risk_engine"]
    # Without BNB parity this would show -1.07%; with BNB parity it is 0.0%.
    assert risk_eng["daily_nav_drawdown_pct"] == pytest.approx(0.0, abs=0.02)


def test_nav_drawdown_large_deposit_with_dip_preserves_index_and_guard():
    """Large confirmed deposit + market dip must use start-of-period base so index never collapses to 0."""
    day0 = datetime(2026, 11, 20, 1, 0, tzinfo=timezone.utc).timestamp()
    day1 = datetime(2026, 11, 21, 1, 0, tzinfo=timezone.utc).timestamp()
    hist: list[dict] = []
    hist, _ = append_nav_history_point(
        hist,
        {"ts": day0, "nav_usd": 5000.0, "invested_usd": 4000.0, "free_cash_usd": 1000.0, "venues_usd": {"binance": 5000.0}},
    )
    # Deposit +$50,000 (base becomes $55,000) while market dips -10% -> NAV = $49,500.
    # End-of-period subtraction would give (49500 - 50000)/5000 = 0.0, permanently locking cur_idx=0.
    hist, _ = append_nav_history_point(
        hist,
        {
            "ts": day0 + 3600.0,
            "nav_usd": 49500.0,
            "invested_usd": 44500.0,
            "free_cash_usd": 5000.0,
            "venues_usd": {"binance": 49500.0},
            "confirmed_cashflow_usd": 50000.0,
        },
    )
    m_dip = evaluate_nav_drawdown_metrics(hist, now_ts=day0 + 3600.0)
    assert m_dip["daily_nav_drawdown_pct"] == pytest.approx(-10.0, abs=0.05)
    assert m_dip["daily_nav_return_pct"] == pytest.approx(-10.0, abs=0.05)
    ok_day0, rsn_day0, _ = evaluate_daily_nav_drawdown_guard(m_dip, decision="BUY", mode="enforce", limit_pct=5.0)
    assert ok_day0 is False and "ai_daily_nav_drawdown_guard" in rsn_day0

    # Next UTC day: real -8% crash must STILL be measured and blocked (proving cur_idx > 0).
    hist, _ = append_nav_history_point(
        hist,
        {"ts": day1, "nav_usd": 49500.0, "invested_usd": 44500.0, "free_cash_usd": 5000.0, "venues_usd": {"binance": 49500.0}},
    )
    hist, _ = append_nav_history_point(
        hist,
        {"ts": day1 + 3600.0, "nav_usd": 45540.0, "invested_usd": 40540.0, "free_cash_usd": 5000.0, "venues_usd": {"binance": 45540.0}},
    )
    m_day1 = evaluate_nav_drawdown_metrics(hist, now_ts=day1 + 3600.0)
    assert m_day1["daily_nav_drawdown_pct"] == pytest.approx(-8.0, abs=0.05)
    ok_day1, rsn_day1, _ = evaluate_daily_nav_drawdown_guard(m_day1, decision="BUY", mode="enforce", limit_pct=5.0)
    assert ok_day1 is False and "ai_daily_nav_drawdown_guard" in rsn_day1






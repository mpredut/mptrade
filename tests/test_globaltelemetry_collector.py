"""Unit tests for GlobalTelemetryCollector."""

from __future__ import annotations

import json
import os
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from intelligence.globaltelemetry_collector import GlobalTelemetryCollector
from intelligence.external.collectors.orderbook_depth_collector import OrderbookDepthCollector, OrderbookSnapshot
from intelligence.external.collectors.derivatives_telemetry_collector import DerivativesTelemetry, DerivativesTelemetryCollector
from intelligence.external.collectors.whale_positioning_collector import WhalePositioningCollector, WhalePositioningSnapshot
from intelligence.sentiment.collectors.fear_greed_collector import FearGreedCollector, FearGreedSnapshot
from intelligence.sentiment.collectors.market_breadth_collector import MarketBreadthCollector, MarketBreadthSnapshot
from cacheaccount import CacheAssetValueManager, CacheOrderManager, CacheTradeManager
from state_io import save_json_state
from intelligence.external.collectors.base_collector import TelemetryCacheCollectorInterface, TelemetryCollectorFactory
from intelligence.sentiment.guards.high_stake_guard import HighStakeGuard
from cachepaths import resolve_cachedb_filepath
from cacheprice import Cache24PriceManager, Cache365PriceManager, CacheCurrentPriceManager, CacheSparsePriceManager
from cachetrend import CacheLongTrendManager, CacheShortTrendManager
from intelligence.internal.guards.guard_decision import OrderGuardEvaluatorBase, OrderGuardEvaluatorFactory, OrderGuardEvaluatorInterface
from intelligence.macro.news_feed_collector import NewsFeedCollector
from intelligence.sentiment.collectors.macro_telemetry_collector import MarketAndPortfolioSnapshotOrchestrator
from intelligence.sentiment.collectors.pretrade_telemetry_collector import (
    PretradeEvaluationContext,
    PretradePillarCollectorBase,
    PretradeTelemetryOrchestrator,
    PretradeTelemetrySnapshot,
    collect_pretrade_telemetry,
)
from intelligence.sentiment.evaluators.microstructure_evaluator import CrossVenueSiblingArbEvaluator, LiquidationClusterEvaluator, MicrostructureCycleDeltaEvaluator, MicrostructureEvaluatorBase, OrderbookCvdFlowEvaluator
from intelligence.sentiment.evaluators.price_series_evaluator import Horizon24hPriceEvaluator, PriceSeriesEvaluatorBase, ShortTrendAndKalmanEvaluator, Structural7dWaveEvaluator, TelemetryEvaluatorInterface, VolumeProfileEvaluator, resolve_asset_session_policy
from intelligence.sentiment.guards.post_stop_reentry_guard import PostStopReentryGuard


class _CustomAuditStage(PretradePillarCollectorBase):
    def evaluate(self, ctx: PretradeEvaluationContext) -> None:
        ctx.telemetry["Custom Polymorphic Pillar"] = f"{ctx.symbol}:{ctx.session_policy.cls_name}"


def test_collector_initialization(temp_cache_dir):
    collector = GlobalTelemetryCollector(
        symbols=["BTCUSDC", "TAOUSDC"],
        cache_dir=temp_cache_dir,
        orderbook_interval_sec=10.0,
        derivatives_interval_sec=20.0,
        whale_interval_sec=30.0,
        news_interval_sec=60.0,
        fear_greed_interval_sec=120.0,
        breadth_interval_sec=90.0,
    )
    assert collector.symbols == ["BTCUSDC", "TAOUSDC"]
    assert collector.orderbook_interval_sec == 10.0
    assert collector.orderbook_collector.cache_ttl_sec == 8.0
    assert collector.orderbook_collector.disk_ttl_sec == 8.0
    assert collector.derivatives_interval_sec == 20.0
    assert collector.derivatives_collector.cache_ttl_sec == 16.0
    assert collector.derivatives_collector.disk_ttl_sec == 16.0
    assert collector.whale_interval_sec == 30.0
    assert collector.whale_collector.cache_ttl_sec == 24.0
    assert collector.whale_collector.disk_ttl_sec == 24.0
    assert collector.news_interval_sec == 60.0
    assert collector.fear_greed_interval_sec == 120.0
    assert collector.breadth_interval_sec == 90.0


def test_collector_run_cycle_mocked(temp_cache_dir):
    collector = GlobalTelemetryCollector(
        symbols=["BTCUSDC"],
        cache_dir=temp_cache_dir,
    )

    mock_ob = OrderbookSnapshot(
        symbol="BTCUSDC",
        mid_price=60000.0,
        bid_depth_usd=100000.0,
        ask_depth_usd=80000.0,
        imbalance_ratio=0.55,
        largest_bid_wall_usd=20000.0,
        largest_bid_wall_price=59900.0,
        largest_ask_wall_usd=15000.0,
        largest_ask_wall_price=60100.0,
        ts=time.time(),
    )
    mock_deriv = DerivativesTelemetry(
        symbol="BTCUSDC",
        funding_rate=0.0001,
        predicted_funding_rate=0.0001,
        open_interest=5000.0,
        open_interest_usd=300000000.0,
        ts=time.time(),
    )
    mock_whale = WhalePositioningSnapshot(
        symbol="BTCUSDC",
        top_traders_long_ratio=1.5,
        top_traders_long_pct=0.6,
        taker_buy_sell_ratio=1.2,
        taker_buy_vol_usd=500000.0,
        taker_sell_vol_usd=400000.0,
        open_interest_usd=300000000.0,
        open_interest_1h_change_pct=1.5,
        divergence_regime="accumulation",
        ts=time.time(),
    )
    mock_fg = FearGreedSnapshot(
        value=65,
        classification="Greed",
        timestamp=time.time(),
        historical_values=(65, 60),
        trend_7d_change=5,
        is_extreme_fear=False,
        is_extreme_greed=False,
    )
    mock_breadth = MarketBreadthSnapshot(
        advance_ratio=0.62,
        advancing_count=217,
        declining_count=133,
        total_count=350,
        median_change_pct=1.45,
        mean_change_pct=1.20,
        dispersion_std=0.85,
        regime="BULLISH_BREADTH",
        top_gainers=(),
        top_losers=(),
        ts=time.time(),
    )

    with patch.object(collector.orderbook_collector, "fetch", return_value=mock_ob), \
         patch.object(collector.derivatives_collector, "fetch", return_value=mock_deriv), \
         patch.object(collector.whale_collector, "fetch", return_value=mock_whale), \
         patch.object(collector.news_collector, "fetch", return_value=MagicMock()), \
         patch.object(collector.fear_greed_collector, "fetch", return_value=mock_fg), \
         patch.object(collector.breadth_collector, "fetch", return_value=mock_breadth), \
         patch.object(collector, "update_macro_benchmarks", return_value=True), \
         patch.object(collector, "update_pretrade_caches", return_value=True):

        res = collector.run_cycle(force=True)

        assert res["orderbook_updated"] == 1
        assert res["derivatives_updated"] == 1
        assert res["whale_updated"] == 1
        assert res["news_updated"] is True
        assert res["fear_greed_updated"] is True
        assert res["breadth_updated"] is True
        assert res["macro_benchmarks_updated"] is True
        assert res["pretrade_caches_updated"] is True

        # Heartbeat verification
        hb_path = os.path.join(temp_cache_dir, "globaltelemetry_collector.heartbeat")
        assert os.path.exists(hb_path)
        with open(hb_path, "r", encoding="utf-8") as f:
            hb = json.load(f)
        assert hb["status"] == "alive"
        assert hb["symbols"] == ["BTCUSDC"]

        # Cache file verifications
        fg_path = os.path.join(temp_cache_dir, "fear_greed_collect.json")
        assert os.path.exists(fg_path)
        with open(fg_path, "r", encoding="utf-8") as f:
            fg_data = json.load(f)
        assert fg_data["value"] == 65
        assert fg_data["sentiment"] == "Greed"


def test_collector_bounded_step_timeout_still_writes_heartbeat(temp_cache_dir):
    collector = GlobalTelemetryCollector(
        symbols=["BTCUSDC"],
        cache_dir=temp_cache_dir,
        skip_external=True,
        skip_news=False,
    )
    hb_path = os.path.join(temp_cache_dir, "globaltelemetry_collector.heartbeat")

    def slow_step():
        time.sleep(0.5)
        return True

    ok = collector._run_step_bounded(slow_step, timeout_sec=0.05, label="hung_dns_test")
    assert ok is False

    collector.write_heartbeat({"ts": time.time(), "symbols": ["BTCUSDC"]})
    assert os.path.exists(hb_path)


def test_bounded_step_skips_overlapping_thread_and_backs_off(temp_cache_dir):
    import threading

    collector = GlobalTelemetryCollector(
        symbols=["HYPEUSD"],
        cache_dir=temp_cache_dir,
        orderbook_interval_sec=60.0,
        derivatives_interval_sec=60.0,
        whale_interval_sec=60.0,
    )

    gate = threading.Event()
    calls = 0

    def blocked_step():
        nonlocal calls
        calls += 1
        gate.wait(timeout=2.0)
        return True

    try:
        assert collector._run_step_bounded(blocked_step, timeout_sec=0.05, label="overlap_test") is False
        # While the first timed-out thread is still alive, a second call with the same label must not spawn a duplicate thread
        assert collector._run_step_bounded(blocked_step, timeout_sec=0.05, label="overlap_test") is False
        assert calls == 1
    finally:
        gate.set()

    # Verify failed step applies retry backoff instead of retrying 1s later
    with patch.object(collector, "update_orderbook", return_value=False) as mock_ob, \
         patch.object(collector, "update_derivatives", return_value=True), \
         patch.object(collector, "update_whale", return_value=True), \
         patch.object(collector, "update_news", return_value=True), \
         patch.object(collector, "update_fear_greed", return_value=True), \
         patch.object(collector, "update_market_breadth", return_value=True), \
         patch.object(collector, "update_macro_benchmarks", return_value=True):
        collector.run_cycle(force=False, now=1000.0)
        assert mock_ob.call_count == 1
        # 1 second later (now=1001.0), orderbook must still be in backoff (15s for 60s interval)
        collector.run_cycle(force=False, now=1001.0)
        assert mock_ob.call_count == 1
        # 16 seconds later (now=1016.0), retry backoff has elapsed and it retries
        collector.run_cycle(force=False, now=1016.0)
        assert mock_ob.call_count == 2

    # Verify OrderbookDepthCollector remembers the working URL per symbol after fallback
    ob_col = OrderbookDepthCollector(cache_dir=temp_cache_dir, cache_ttl_sec=0.0, disk_ttl_sec=0.0)
    urls_called = []

    def fake_fetch_json(url, label, lookup_sym, **kwargs):
        urls_called.append(url)
        if "api.binance.com" in url:
            raise RuntimeError("HTTP 400 Invalid symbol")
        return {
            "result": {
                "SUIUSD": {
                    "bids": [["20.0", "10.0"]],
                    "asks": [["20.1", "10.0"]],
                }
            }
        }

    with patch.object(ob_col, "_fetch_json_dict", side_effect=fake_fetch_json):
        s1 = ob_col.fetch("SUIUSD", allow_network=True, now=1000.0)
        assert s1 is not None
        assert len(urls_called) == 3  # SUIUSDC spot, SUIUSDT spot, then Kraken SUIUSD
        assert "api.kraken.com" in urls_called[-1]

        urls_called.clear()
        s2 = ob_col.fetch("SUIUSD", allow_network=True, now=1010.0)
        assert s2 is not None
        assert len(urls_called) == 1  # Goes straight to preferred Kraken URL on subsequent fetch!
        assert "api.kraken.com" in urls_called[0]


def test_three_tier_polymorphic_telemetry_evaluators_and_collectors(temp_cache_dir):
    from intelligence.sentiment.collectors.microstructure_collector import (
        CompositeSymbolPriceSeriesLoader,
        EquityYahoo5dChartCollector,
        Fleet24hAndMicrostructureEnricher,
        MicrostructureCacheCollectorBase,
        PriceSeriesCacheCollectorBase,
        SymbolMicrostructureSnapshotCollector,
    )
    from intelligence.sentiment.evaluators.price_series_evaluator import AssetSessionPolicy, Crypto24x7SessionPolicy, EquityRthSessionPolicy

    for abstract_cls in (
        TelemetryEvaluatorInterface,
        PriceSeriesEvaluatorBase,
        MicrostructureEvaluatorBase,
        AssetSessionPolicy,
        TelemetryCacheCollectorInterface,
        PriceSeriesCacheCollectorBase,
        MicrostructureCacheCollectorBase,
        PretradePillarCollectorBase,
    ):
        with pytest.raises(TypeError):
            abstract_cls()

    crypto_pol = resolve_asset_session_policy("BTCUSDC")
    eq_pol = resolve_asset_session_policy("NVDA_US_EQ")
    assert isinstance(crypto_pol, Crypto24x7SessionPolicy)
    assert isinstance(eq_pol, EquityRthSessionPolicy)
    assert crypto_pol.is_equity is False and eq_pol.is_equity is True

    series = [[1_700_000_000_000 + i * 3600_000, 100.0 + i * 0.5, 1000.0] for i in range(25)]
    vp_ev = VolumeProfileEvaluator()
    h24_ev = Horizon24hPriceEvaluator()
    s7d_ev = Structural7dWaveEvaluator()
    stk_ev = ShortTrendAndKalmanEvaluator()
    assert isinstance(vp_ev, PriceSeriesEvaluatorBase)
    assert "poc_px" in vp_ev.evaluate(series)
    m24 = h24_ev.evaluate(series)
    assert m24.get("chg_24h_pct") == 12.0 and "poc_px" in m24 and "bounce_24h_low_pct" in m24
    m7d = s7d_ev.evaluate([100.0, 105.0, 98.0, 112.0], cur_px=112.0)
    assert m7d.get("high_7d") == 112.0
    st_out = stk_ev.evaluate(series, live_px=112.5)
    assert st_out.get("price") == 112.5

    cvd_ev = OrderbookCvdFlowEvaluator()
    liq_ev = LiquidationClusterEvaluator()
    arb_ev = CrossVenueSiblingArbEvaluator()
    cd_ev = MicrostructureCycleDeltaEvaluator()
    assert isinstance(cvd_ev, MicrostructureEvaluatorBase)
    m_sample = {"ob_imbalance": 0.62, "taker_buy_sell_ratio": 1.25, "cvd_15m_usd": 500_000.0, "oi_usd": 10_000_000.0}
    cvd_ev.evaluate(m_sample, price_info={"price": 112.0, "chg_15m_pct": 0.4})
    assert "cvd_regime" in m_sample
    liq_res = liq_ev.evaluate({"price": 112.0, "low_24h": 100.0, "high_24h": 115.0}, micro_info=m_sample)
    assert "long_liq_px" in liq_res
    assert isinstance(arb_ev, MicrostructureEvaluatorBase)
    assert isinstance(cd_ev, MicrostructureEvaluatorBase)

    loader = TelemetryCollectorFactory.get_price_series_loader()
    assert isinstance(loader, CompositeSymbolPriceSeriesLoader)
    assert loader.price_24_cls is Cache24PriceManager
    assert loader.price_365_cls is Cache365PriceManager
    assert loader.price_sparse_cls is CacheSparsePriceManager
    assert loader.current_price_cls is CacheCurrentPriceManager
    assert Cache365PriceManager.disk_filename_for_symbol("BTCUSDC") == "cache_365_price_BTCUSDC.jsonl"
    assert Cache24PriceManager.disk_filename_for_symbol("BTCUSDC") == "cache_24_price_BTCUSDC.json"
    assert CacheSparsePriceManager.disk_filename_for_symbol("BTCUSDC") == "cache_sparse_price_BTCUSDC.jsonl"
    assert CacheCurrentPriceManager.disk_filename_for_symbol() == "cache_current_price.json"
    from cacheprice import CachePriceManagerBase

    assert CacheShortTrendManager.disk_filename_for_symbol() == "cache_short_trend.json"
    assert CacheShortTrendManager.live_window_relpath_for_symbol("BTCUSDC").replace("\\", "/") == "24trend/live_BTCUSDC.json"
    assert CacheLongTrendManager.disk_filename_for_symbol() == "cache_long_trend.json"
    assert CacheTradeManager.disk_filename_for_symbol() == "cache_trade_binance.jsonl"
    assert CacheOrderManager.disk_filename_for_symbol() == "cache_order_binance.json"
    assert CacheAssetValueManager.disk_filename_for_symbol() == "cache_asset_value.jsonl"
    assert CachePriceManagerBase.extract_jsonl_series_point({"s": "BTCUSDC", "i": [1700000000000, 95000.5]}, {"BTCUSDC"}) == [
        1700000000000,
        95000.5,
    ]
    assert CachePriceManagerBase.extract_jsonl_series_point({"s": "ETHUSDC", "i": [1700000000000, 3000.0]}, {"BTCUSDC"}) is None
    assert isinstance(TelemetryCollectorFactory.get_collector(SymbolMicrostructureSnapshotCollector), SymbolMicrostructureSnapshotCollector)
    assert isinstance(TelemetryCollectorFactory.get_fleet_enricher(), Fleet24hAndMicrostructureEnricher)
    assert isinstance(TelemetryCollectorFactory.get_yahoo_chart_collector(), EquityYahoo5dChartCollector)
    assert isinstance(TelemetryCollectorFactory.get_collector(PretradeTelemetryOrchestrator), PretradeTelemetryOrchestrator)

    custom_tel = collect_pretrade_telemetry("BTCUSDC", cachedb_dir=temp_cache_dir, stages=(_CustomAuditStage(),))
    assert custom_tel == {"Custom Polymorphic Pillar": "BTCUSDC:Crypto24x7SessionPolicy"}


def test_runtime_services_dependency_injection_and_zero_pytest_branches():
    import subprocess
    import botcore

    slept = []
    botcore.configure_runtime_services(
        clock_fn=lambda: 1_750_000_000.0,
        sleep_fn=lambda s: slept.append(s),
        stale_cache_retry_delay_sec=0.125,
        enforce_rth_gate=True,
        allow_live_network_backfill=True,
        allow_unmocked_llm_queries=True,
        allow_shared_disk_cache=True,
    )
    assert botcore.resolve_now_ts() == 1_750_000_000.0
    assert botcore.resolve_now_ms() == 1_750_000_000_000
    assert botcore.get_stale_cache_retry_delay_sec() == 0.125
    assert botcore.is_rth_gate_enabled() is True
    assert botcore.is_live_network_backfill_enabled() is True
    assert botcore.is_unmocked_llm_query_allowed() is True
    assert botcore.is_shared_disk_cache_enabled() is True

    res = subprocess.run(
        [
            "git",
            "grep",
            "-nE",
            r"PYTEST_CURRENT_TEST|\"pytest\" in sys\.modules|\"unittest\" in sys\.modules|\"unittest\" not in sys\.modules|\"PYTEST\"|\"UNITTE\"",
            "--",
            "*.py",
            ":!tests",
            ":!offline",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.stdout.strip() == "", f"Unexpected test-detection branches in app modules:\n{res.stdout}"


def test_pretrade_telemetry_read_write_separation_and_explicit_snapshot_contract(temp_cache_dir, monkeypatch):
    from intelligence.internal.guards.guard_decision import BrakeAction, fetch_guard_snapshot
    from intelligence.sentiment.collectors import microstructure_collector as mc_mod
    from intelligence.sentiment.collectors import pretrade_telemetry_collector as ptc_mod
    from intelligence.sentiment.collectors.pretrade_telemetry_collector import read_pretrade_telemetry_snapshot, refresh_pretrade_telemetry_caches
    from intelligence.sentiment.guards.high_stake_guard import _HIGH_STAKE_DECISION_CACHE
    from intelligence.sentiment.guards.post_stop_reentry_guard import clear_post_stop_reentry_cache

    _HIGH_STAKE_DECISION_CACHE.clear()
    clear_post_stop_reentry_cache()

    now_ts = 1_750_000_000.0
    ob_payload = {
        "symbol": "NVDA_US_EQ",
        "mid_price": 135.50,
        "best_bid": 135.48,
        "best_ask": 135.52,
        "bid_depth_usd": 45000.0,
        "ask_depth_usd": 40000.0,
        "imbalance_ratio": 0.56,
        "largest_bid_wall_usd": 12000.0,
        "largest_bid_wall_price": 135.00,
        "largest_ask_wall_usd": 10000.0,
        "largest_ask_wall_price": 136.00,
        "spread_bps": 2.95,
        "ts": now_ts - 25.0,
    }
    with open(os.path.join(temp_cache_dir, "orderbook_depth_NVDA_US_EQ.json"), "w", encoding="utf-8") as f:
        json.dump(ob_payload, f)
    with open(os.path.join(temp_cache_dir, "venue_balances_state.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "pos:t212:NVDA_US_EQ": {
                    "symbol": "NVDA_US_EQ",
                    "free": 5.0,
                    "price": 135.50,
                    "value_usd": 677.50,
                    "ts": now_ts - 600.0,
                }
            },
            f,
        )

    # Enable live network backfill flag to prove read-only consumer path still never triggers network/writes
    monkeypatch.setattr(ptc_mod, "is_live_network_backfill_enabled", lambda: True)
    monkeypatch.setattr(mc_mod, "is_live_network_backfill_enabled", lambda: True)
    monkeypatch.setattr(ptc_mod, "get_cachedb_dir", lambda: temp_cache_dir)

    network_calls: list[str] = []
    monkeypatch.setattr(
        ptc_mod,
        "ensure_fresh_symbol_microstructure",
        lambda *a, **kw: network_calls.append("microstructure"),
    )
    monkeypatch.setattr(
        ptc_mod,
        "_refresh_stale_t212_cached_positions",
        lambda *a, **kw: network_calls.append("t212_refresh") or True,
    )
    monkeypatch.setattr(
        mc_mod.EquityYahoo5dChartCollector,
        "collect",
        lambda self, *a, **kw: network_calls.append("yahoo_chart") or ([], [], [], False, False, 0.0, 0),
    )

    orchestrator = PretradeTelemetryOrchestrator()
    assert orchestrator.start_sync is False

    # 1. Read-only snapshot extraction has explicit identity, source, and age and triggers zero network calls
    snap = read_pretrade_telemetry_snapshot("NVDA_US_EQ", cachedb_dir=temp_cache_dir, now_ts=now_ts)
    assert isinstance(snap, PretradeTelemetrySnapshot)
    assert snap.symbol == "NVDA_US_EQ"
    assert snap.canonical_symbol == "NVDAUSEQ"
    assert snap.asset_scope == "EQUITY"
    assert snap.source == "cachedb"
    assert snap.pillar_sources.get("orderbook") in ("cachedb", "memory_cache")
    assert snap.ts == pytest.approx(now_ts - 25.0)
    assert snap.age_sec == pytest.approx(25.0)
    assert snap.is_fresh is True
    assert snap.is_available is True
    assert "Orderbook Depth Imbalance" in snap
    assert network_calls == []

    # 2. HighStakeGuard and PostStopReentryGuard accept explicit snapshot or read-only snapshot without network I/O
    prompts_seen: list[str] = []
    monkeypatch.setattr(
        "intelligence.sentiment.guards.high_stake_guard.query_topic_json",
        lambda client, topic, prompt, **kw: prompts_seen.append(prompt)
        or {"decision": "APPROVED", "suggested_scale": 1.0, "reason": "ok"},
    )
    hs_guard = HighStakeGuard(gemini_client=object(), min_notional_eur=500.0, cache_ttl_sec=0.0)
    dec_hs = hs_guard.check("NVDA_US_EQ", "BUY", price=135.50, qty=10.0, snapshot=snap, now=now_ts)
    assert dec_hs.allowed is True
    assert dec_hs.metadata["telemetry_symbol"] == "NVDAUSEQ"
    assert dec_hs.metadata["telemetry_source"] == "cachedb"
    assert dec_hs.metadata["telemetry_age_sec"] == pytest.approx(25.0)
    assert dec_hs.metadata["telemetry_fresh"] is True
    assert any("Telemetry Snapshot: symbol=NVDAUSEQ" in p for p in prompts_seen)
    assert network_calls == []

    # 3. Unavailable/stale telemetry preserves fallback_action and shadow/enforce policies without blocking on LLM
    prompts_seen.clear()
    unavail_snap = PretradeTelemetrySnapshot.unavailable("BTCUSDC", reason="telemetry_stale", now_ts=now_ts)
    for fb_action, mode_val, exp_allowed, exp_brake, exp_scale in (
        ("veto", "enforce", False, BrakeAction.HARD_VETO, 0.0),
        ("downscale", "enforce", True, BrakeAction.DOWNSCALE_QTY, 0.5),
        ("allow", "enforce", True, BrakeAction.NONE, 1.0),
        ("veto", "shadow", True, BrakeAction.NONE, 1.0),
    ):
        g_hs = HighStakeGuard(
            gemini_client=object(),
            min_notional_eur=500.0,
            fallback_action=fb_action,
            mode=mode_val,
            cache_ttl_sec=0.0,
        )
        d_hs = g_hs.check("BTCUSDC", "BUY", price=60000.0, qty=0.02, snapshot=unavail_snap, now=now_ts)
        assert d_hs.allowed is exp_allowed
        assert d_hs.brake_action == exp_brake
        assert d_hs.suggested_scale == pytest.approx(exp_scale)
        assert d_hs.metadata["telemetry_available"] is False
        assert d_hs.metadata["telemetry_unavailable_reason"] == "telemetry_stale"
        if mode_val == "shadow":
            assert d_hs.shadow_decision is not None
            assert d_hs.shadow_decision.allowed is False
            assert d_hs.shadow_decision.brake_action == BrakeAction.HARD_VETO

        g_ps = PostStopReentryGuard(
            gemini_client=object(),
            fallback_action=fb_action,
            mode=mode_val,
            notify_enabled=False,
            cache_ttl_sec=0.0,
        )
        d_ps = g_ps.check(
            "BTCUSDC",
            current_price=60500.0,
            last_sell_price=60000.0,
            trough=59000.0,
            bounce_pct=0.025,
            snapshot=unavail_snap,
            now=now_ts,
        )
        assert d_ps.allowed is exp_allowed
        assert d_ps.brake_action == exp_brake
    assert prompts_seen == []

    # 4. fetch_guard_snapshot defaults to allow_network=False for all Pillar 2 guards
    seen_fetch_kwargs: list[dict] = []
    fetch_guard_snapshot(
        SimpleNamespace(fetch=lambda symbol, **kwargs: seen_fetch_kwargs.append(kwargs) or {"symbol": symbol}),
        "BTCUSDC",
        now=now_ts,
    )
    assert seen_fetch_kwargs == [{"allow_network": False, "now": now_ts}]

    # 5. Producer refresh_pretrade_telemetry_caches / GlobalTelemetryCollector.update_pretrade_caches performs refresh
    status = refresh_pretrade_telemetry_caches("NVDA_US_EQ", cachedb_dir=temp_cache_dir, now_ts=now_ts)
    assert status["microstructure_refreshed"] is True
    assert status["t212_positions_refreshed"] is True
    assert "microstructure" in network_calls
    assert "t212_refresh" in network_calls
    assert "yahoo_chart" in network_calls


def test_portfolio_tca_and_macro_3tier_polymorphic_hierarchy_and_cache_contracts(temp_cache_dir: str):
    """Verify 3-tier polymorphic collectors/evaluators across portfolio, TCA/risk, and macro telemetry and cache contract reuse."""
    from intelligence.sentiment.collectors.macro_telemetry_collector import InitialPricesAndTrendsCollector, MacroAndNewsSensorsCollector, MacroBenchmarksSnapshotCollector, MacroTelemetryCollectorBase, PortfolioAndVenueCyclesCollector
    from intelligence.sentiment.collectors.portfolio_collector import (
        BotRuntimeStatesCollector,
        KrakenXstocksCollector,
        PortfolioCacheCollectorBase,
        RecentOrdersAndPositionEnricherCollector,
        SymbolBestRecentFillsCollector,
        TradablePositionsAndOrdersCollector,
        VenueCyclesCollector,
        VenueTrailingMapsCollector,
        _resolve_order_cache_files,
    )
    from intelligence.sentiment.evaluators.macro_telemetry_evaluator import (
        AssetScopeFilterEvaluator,
        CrossAssetRegimeEvaluator,
        DeribitOptionsSurfaceEvaluator,
        EquityMacroRegimeEvaluator,
        FuturesPremarketLeadEvaluator,
        InstitutionalPostureDivergenceEvaluator,
        LiveVenueEquityReconciliationEvaluator,
        MacroTelemetryEvaluatorBase,
    )
    from intelligence.sentiment.evaluators.portfolio_evaluator import (
        PortfolioEvaluatorBase,
        PositionCostBasisEnrichmentEvaluator,
        SymbolRecentFillsEvaluator,
        VenueCycleReconciliationEvaluator,
        VenueCycleStateEvaluator,
    )
    from intelligence.sentiment.evaluators.tca_risk_evaluator import (
        CrossMarketContextEvaluator,
        EnterprisePortfolioRiskEvaluator,
        OpenLimitOrdersTcaEvaluator,
        TcaRiskEvaluatorBase,
        VenueFeeAndLimitsEvaluator,
    )

    # 1. Level-2 abstract base classes enforce non-instantiation and inherit Level-1 contracts
    for l2_col_cls in (PortfolioCacheCollectorBase, MacroTelemetryCollectorBase):
        assert issubclass(l2_col_cls, TelemetryCacheCollectorInterface)
        with pytest.raises(TypeError):
            l2_col_cls()

    for l2_eval_cls in (PortfolioEvaluatorBase, TcaRiskEvaluatorBase, MacroTelemetryEvaluatorBase):
        assert issubclass(l2_eval_cls, TelemetryEvaluatorInterface)
        with pytest.raises(TypeError):
            l2_eval_cls()

    # 2. All Level-3 portfolio & macro collectors are registered in TelemetryCollectorFactory
    expected_collectors = (
        ("portfolio_trailing_maps", VenueTrailingMapsCollector, PortfolioCacheCollectorBase),
        ("portfolio_venue_cycles", VenueCyclesCollector, PortfolioCacheCollectorBase),
        ("portfolio_kraken_xstocks", KrakenXstocksCollector, PortfolioCacheCollectorBase),
        ("portfolio_tradable_positions", TradablePositionsAndOrdersCollector, PortfolioCacheCollectorBase),
        ("portfolio_symbol_recent_fills", SymbolBestRecentFillsCollector, PortfolioCacheCollectorBase),
        ("portfolio_recent_orders", RecentOrdersAndPositionEnricherCollector, PortfolioCacheCollectorBase),
        ("portfolio_bot_runtime_states", BotRuntimeStatesCollector, PortfolioCacheCollectorBase),
        ("macro_benchmarks_snapshot", MacroBenchmarksSnapshotCollector, MacroTelemetryCollectorBase),
        ("macro_and_news_sensors", MacroAndNewsSensorsCollector, MacroTelemetryCollectorBase),
        ("macro_initial_prices_trends", InitialPricesAndTrendsCollector, MacroTelemetryCollectorBase),
        ("macro_portfolio_venue_cycles", PortfolioAndVenueCyclesCollector, MacroTelemetryCollectorBase),
        ("macro_market_portfolio_snapshot", MarketAndPortfolioSnapshotOrchestrator, MacroTelemetryCollectorBase),
    )
    for col_id, col_cls, base_cls in expected_collectors:
        assert issubclass(col_cls, base_cls)
        inst = TelemetryCollectorFactory.create(col_id)
        assert isinstance(inst, col_cls)
        assert TelemetryCollectorFactory.get_collector(col_cls) is inst

    # 3. Cache hierarchy filename & schema contracts are reused
    from cacheaccount import CacheAccountManagerBase
    from order_guard import (
        CompositeProfitGuardEvaluator,
        IntelligenceMacroSentimentGuardEvaluator,
        MathProfitReferenceGuardEvaluator,
    )

    assert (CacheOrderManager.DEFAULT_FILENAME, "binance") in _resolve_order_cache_files()
    assert _resolve_order_cache_files() == CacheAccountManagerBase.venue_order_and_trade_cache_contracts()
    from instrument_registry import venue_for_symbol

    assert venue_for_symbol("BTCUSDC") == "binance"
    assert venue_for_symbol("TAOUSD") == "kraken"
    assert venue_for_symbol("HYPEUSDC") == "hyperliquid"
    assert venue_for_symbol("NVDA_US_EQ") == "t212"
    assert CacheTradeManager.venue_for_symbol("TAOUSD") == "kraken"
    assert CacheTradeManager.venue_for_symbol("HYPEUSDC") == "hyperliquid"

    for expected_venue, expected_fname, expected_fmt in (
        ("binance", "cache_trade_binance.jsonl", "jsonl"),
        ("kraken", "cache_trade_kraken.json", "json"),
        ("hyperliquid", "cache_trade_hyperliquid.json", "json"),
        ("t212", "cache_trade_t212.json", "json"),
    ):
        assert issubclass(CacheTradeManager, CacheAccountManagerBase)
        assert CacheTradeManager.disk_filename(expected_venue) == expected_fname
        trd_inst = CacheTradeManager(
            venue=expected_venue,
            filename=os.path.join(temp_cache_dir, expected_fname),
            read_only=True,
        )
        assert trd_inst.venue == expected_venue
        assert trd_inst.stream_format == expected_fmt
        assert trd_inst.append_persist is (expected_fmt == "jsonl")

    kraken_trd_path = os.path.join(temp_cache_dir, CacheTradeManager.disk_filename("kraken"))

    save_json_state(
        kraken_trd_path,
        {
            "items": {
                "TAOUSD": [
                    {"id": "tx-1", "orderId": "k-1", "symbol": "TAOUSD", "isBuyer": False, "price": 412.5, "qty": 1.2, "time": 1700000100000},
                    {"id": "tx-2", "orderId": "k-2", "symbol": "TAOUSD", "isBuyer": True, "price": 405.0, "qty": 1.2, "time": 1700000200000},
                ]
            },
            "fetchtime": {"TAOUSD": 1700000200},
        },
        indent=2,
    )
    sanitized = CacheTradeManager.load_sanitized_trade_snapshot(kraken_trd_path, venue="kraken")
    assert len(sanitized["items"].get("TAOUSD", [])) == 2
    assert sanitized["fetchtime"].get("TAOUSD") == 1700000200
    k_mgr = CacheTradeManager(filename=kraken_trd_path, venue="kraken", stream_format="json", read_only=True)
    assert k_mgr.get_last_opposite_fill_price("TAOUSD", "BUY") == pytest.approx(412.5)
    assert k_mgr.get_last_opposite_fill_price("TAOUSD", "SELL") == pytest.approx(405.0)

    for col_id, col_cls in (
        ("sentiment_fear_greed", FearGreedCollector),
        ("sentiment_market_breadth", MarketBreadthCollector),
    ):
        assert issubclass(col_cls, TelemetryCacheCollectorInterface)
        assert isinstance(TelemetryCollectorFactory.create(col_id), col_cls)

    for og_id, og_cls in (
        ("math_profit_reference", MathProfitReferenceGuardEvaluator),
        ("intelligence_guards", IntelligenceMacroSentimentGuardEvaluator),
        ("profit_guard", CompositeProfitGuardEvaluator),
    ):
        assert issubclass(og_cls, OrderGuardEvaluatorBase)
        assert issubclass(og_cls, OrderGuardEvaluatorInterface)
        og_inst = OrderGuardEvaluatorFactory.create(og_id)
        assert isinstance(og_inst, og_cls)
        assert OrderGuardEvaluatorFactory.get_evaluator(og_cls) is og_inst
    assert CacheCurrentPriceManager.DEFAULT_FILENAME == "cache_current_price.json"
    assert CacheShortTrendManager.DEFAULT_FILENAME == "cache_short_trend.json"
    assert CacheShortTrendManager.KALMAN_TREND_FILENAME == "cache_kalman_trend.json"
    assert CacheShortTrendManager.LIVE_TREND_SUBDIR == "24trend"
    assert CacheLongTrendManager.DEFAULT_FILENAME == "cache_long_trend.json"
    assert CacheLongTrendManager.SOURCE_ANALYSIS_FILENAME == "priceanalysis.json"
    assert CacheLongTrendManager.T_TREND_FILENAME == "cache_T_trend.json"
    assert Cache24PriceManager.disk_filename_for_symbol("BTCUSDC") == "cache_24_price_BTCUSDC.json"
    assert Cache365PriceManager.disk_filename_for_symbol("BTCUSDC") == "cache_365_price_BTCUSDC.jsonl"
    assert CacheSparsePriceManager.disk_filename_for_symbol("BTCUSDC") == "cache_sparse_price_BTCUSDC.jsonl"

    av_rec = {
        "t": 1700000000,
        "i": {
            "total_value_usdc": 12500.5,
            "venues_usd": {"binance": 12500.5, "kraken": 3500.0},
            "portfolio_total_usd": 16000.5,
        },
    }
    av_summary = CacheAssetValueManager.extract_asset_value_summary(av_rec, now_ts=1700000300.0)
    assert av_summary == {
        "total_value_usdc": 12500.5,
        "venues_value_usd": {"binance": 12500.5, "kraken": 3500.0},
        "portfolio_total_usd": 16000.5,
        "total_value_usdc_age_min": 5.0,
    }

    # 4. Level-3 evaluators inherit from Level-2 bases, register in TelemetryEvaluatorFactory, and evaluate statelessly
    from intelligence.sentiment.evaluators.microstructure_evaluator import CrossVenueTrendsEnricherEvaluator, SiblingSymbolTelemetrySyncEvaluator
    from intelligence.sentiment.evaluators.price_series_evaluator import TelemetryEvaluatorFactory

    for eval_id, eval_cls, l2_base in (
        ("price_volume_profile", VolumeProfileEvaluator, PriceSeriesEvaluatorBase),
        ("price_horizon_24h", Horizon24hPriceEvaluator, PriceSeriesEvaluatorBase),
        ("price_structural_7d_wave", Structural7dWaveEvaluator, PriceSeriesEvaluatorBase),
        ("price_short_trend_kalman", ShortTrendAndKalmanEvaluator, PriceSeriesEvaluatorBase),
        ("micro_orderbook_cvd_flow", OrderbookCvdFlowEvaluator, MicrostructureEvaluatorBase),
        ("micro_liquidation_clusters", LiquidationClusterEvaluator, MicrostructureEvaluatorBase),
        ("micro_cross_venue_sibling_arb", CrossVenueSiblingArbEvaluator, MicrostructureEvaluatorBase),
        ("micro_cycle_delta", MicrostructureCycleDeltaEvaluator, MicrostructureEvaluatorBase),
        ("micro_cross_venue_trends", CrossVenueTrendsEnricherEvaluator, MicrostructureEvaluatorBase),
        ("micro_sibling_symbol_sync", SiblingSymbolTelemetrySyncEvaluator, MicrostructureEvaluatorBase),
        ("portfolio_venue_cycle_state", VenueCycleStateEvaluator, PortfolioEvaluatorBase),
        ("portfolio_symbol_recent_fills", SymbolRecentFillsEvaluator, PortfolioEvaluatorBase),
        ("portfolio_venue_cycle_reconciliation", VenueCycleReconciliationEvaluator, PortfolioEvaluatorBase),
        ("portfolio_position_cost_basis", PositionCostBasisEnrichmentEvaluator, PortfolioEvaluatorBase),
        ("tca_venue_fee_and_limits", VenueFeeAndLimitsEvaluator, TcaRiskEvaluatorBase),
        ("tca_open_limit_orders", OpenLimitOrdersTcaEvaluator, TcaRiskEvaluatorBase),
        ("tca_enterprise_portfolio_risk", EnterprisePortfolioRiskEvaluator, TcaRiskEvaluatorBase),
        ("tca_cross_market_context", CrossMarketContextEvaluator, TcaRiskEvaluatorBase),
        ("macro_futures_premarket_lead", FuturesPremarketLeadEvaluator, MacroTelemetryEvaluatorBase),
        ("macro_equity_regime", EquityMacroRegimeEvaluator, MacroTelemetryEvaluatorBase),
        ("macro_cross_asset_regime", CrossAssetRegimeEvaluator, MacroTelemetryEvaluatorBase),
        ("macro_deribit_options_surface", DeribitOptionsSurfaceEvaluator, MacroTelemetryEvaluatorBase),
        ("macro_institutional_posture_divergence", InstitutionalPostureDivergenceEvaluator, MacroTelemetryEvaluatorBase),
        ("macro_asset_scope_filter", AssetScopeFilterEvaluator, MacroTelemetryEvaluatorBase),
        ("macro_live_venue_equity_reconciliation", LiveVenueEquityReconciliationEvaluator, MacroTelemetryEvaluatorBase),
    ):
        assert issubclass(eval_cls, l2_base)
        ev = TelemetryEvaluatorFactory.create(eval_id)
        assert isinstance(ev, eval_cls)
        assert isinstance(ev, TelemetryEvaluatorInterface)
        assert TelemetryEvaluatorFactory.get_evaluator(eval_cls) is ev


def test_external_collectors_and_survival_producer_consumer_decoupling(
    temp_cache_dir: str,
    monkeypatch: pytest.MonkeyPatch,
):
    """Verify Step 2 Producer-Consumer decoupling across external symbol collectors, survival estimation, and cycle deltas."""
    import external_order_guard as eog
    from intelligence.internal.state import survival as surv_mod
    from intelligence.sentiment.evaluators.microstructure_evaluator import (
        enrich_cross_venue_trends,
        update_microstructure_cycle_deltas,
    )

    monkeypatch.setenv("MPTRADE_ENABLE_LIVE_BACKFILL", "1")
    monkeypatch.setattr(eog, "_whale_collector", None)
    monkeypatch.setattr(eog, "_orderbook_collector", None)
    monkeypatch.setattr(eog, "_derivatives_collector", None)

    # 1. External order guard initializes collectors as read-only consumers (start_sync=False)
    wh_reader, ob_reader, dt_reader = eog.get_whale_collector(), eog.get_orderbook_collector(), eog.get_derivatives_collector()
    for col in (wh_reader, ob_reader, dt_reader):
        assert col.start_sync is False

    # 2. GlobalTelemetryCollector (ExternalTelemetryMixin) initializes collectors as producers (start_sync=True)
    daemon = GlobalTelemetryCollector(symbols=["BTCUSDC"], cache_dir=temp_cache_dir)
    for col in (daemon.orderbook_collector, daemon.derivatives_collector, daemon.whale_collector):
        assert col.start_sync is True

    # 3. Read-only consumer collectors never trigger HTTP fetches or sibling disk writes on cache miss
    http_calls: list[str] = []
    monkeypatch.setattr(
        OrderbookDepthCollector,
        "_fetch_json_dict",
        classmethod(lambda cls, url, *a, **kw: http_calls.append(url) or {}),
    )
    ob_consumer = OrderbookDepthCollector(cache_dir=temp_cache_dir, start_sync=False)
    assert ob_consumer.fetch("UNLISTEDUSDC", now=1700000100.0) is None
    assert ob_consumer.read_snapshot("UNLISTEDUSDC", now=1700000100.0) is None
    assert http_calls == []

    # 4. estimate_remaining_trend_hours and enrich_cross_venue_trends do not trigger live kline fetches by default
    kline_calls: list[str] = []
    monkeypatch.setattr(
        surv_mod,
        "fetch_klines",
        surv_mod._ORIG_FETCH_KLINES,
    )
    monkeypatch.setattr(
        surv_mod,
        "get_cached_t_entry",
        lambda sym: None,
    )
    rem_h = surv_mod.estimate_remaining_trend_hours(48.0, symbol="UNLISTEDUSDC")
    assert rem_h == pytest.approx(24.0)

    pt_map = {"UNLISTEDUSDC": {"price": 10.0, "long_trend_hours": 24.0}}
    enrich_cross_venue_trends(pt_map, {}, {}, {}, {}, {}, {})
    assert "weibull_median_d" not in pt_map["UNLISTEDUSDC"]
    assert kline_calls == []

    # 5. update_microstructure_cycle_deltas with persist=False computes cycle_delta in-memory without writing disk
    mhist_path = str(resolve_cachedb_filepath(os.path.join(temp_cache_dir, "microstructure_cycle_history.json")))
    if os.path.exists(mhist_path):
        os.remove(mhist_path)
    micro_map = {"BTCUSDC": {"ob_imbalance": 0.62, "cvd_15m_usd": 120000.0}}
    update_microstructure_cycle_deltas(temp_cache_dir, micro_map, now_ts=1700000100.0, persist=False)
    assert not os.path.exists(mhist_path)
    update_microstructure_cycle_deltas(temp_cache_dir, micro_map, now_ts=1700000100.0, persist=True)
    assert os.path.exists(str(resolve_cachedb_filepath(os.path.join(temp_cache_dir, "microstructure_cycle_history.json"))))


def test_read_market_and_portfolio_snapshot_read_only_consumer(
    temp_cache_dir: str,
    monkeypatch: pytest.MonkeyPatch,
):
    """Verify Step 3: MarketAndPortfolioSnapshotOrchestrator.read_snapshot performs 0 live venue/network calls and 0 disk writes while stamping snapshot identity and age."""
    from orchestratorOS.supervisor.autonomous_ai_reconciler import TelemetrySensorSnapshot

    monkeypatch.setenv("MPTRADE_ENABLE_LIVE_BACKFILL", "1")
    now_ts = 1700000500.0

    # Seed cachedb files including venue_balances_state.json and stale market_breadth
    save_json_state(
        os.path.join(temp_cache_dir, "cache_current_price.json"),
        {"ts": now_ts - 15.0, "items": {"BTCUSDC": {"price": 65000.0, "ts": now_ts - 15.0}}},
        indent=2,
    )
    save_json_state(
        os.path.join(temp_cache_dir, "cache_short_trend.json"),
        {"ts": now_ts - 15.0, "items": {"BTCUSDC": {"trend": 1, "growth_coefficient": 0.012}}},
        indent=2,
    )
    save_json_state(
        os.path.join(temp_cache_dir, "market_breadth_collect.json"),
        {
            "advance_ratio": 0.55,
            "median_change_pct": 1.2,
            "mean_change_pct": 1.4,
            "dispersion_std": 2.1,
            "regime": "BULLISH_BROAD",
            "total_symbols": 10,
            "total_count": 10,
            "advancing": 6,
            "declining": 4,
            "ts": now_ts - 7200.0,  # Stale (>30m), would normally trigger live MarketBreadthCollector refresh
        },
        indent=2,
    )
    bals_path = os.path.join(temp_cache_dir, "venue_balances_state.json")
    save_json_state(
        bals_path,
        {
            "binance:BTC": 0.25,
            "binance:USDC": 1500.0,
            "cash:binance:USDC": {"venue": "binance", "asset": "USDC", "free": 1500.0},
            "open_orders:binance:BTCUSDC": [
                {"orderId": "ord-1", "symbol": "BTCUSDC", "side": "SELL", "price": 68000.0, "origQty": 0.1}
            ],
        },
        indent=2,
    )

    # Create dummy instruments whose live provider methods fail if invoked
    def _fail_live(msg: str):
        raise AssertionError(msg)

    failing_provider = SimpleNamespace(
        free_balance=lambda asset: _fail_live(f"live free_balance({asset}) must not be called in read_snapshot mode"),
        get_current_price=lambda symbol: _fail_live(f"live get_current_price({symbol}) must not be called in read_snapshot mode"),
        usd_cash_breakdown=lambda: _fail_live("live usd_cash_breakdown() must not be called in read_snapshot mode"),
        get_orders=lambda *args, **kwargs: _fail_live("live get_orders() must not be called in read_snapshot mode"),
    )
    dummy_inst = SimpleNamespace(
        symbol="BTCUSDC",
        provider_name="binance",
        base="BTC",
        quote="USDC",
        params={"trailing.pct": 2.5},
        _provider=failing_provider,
        open_orders=lambda: _fail_live("live open_orders() must not be called in read_snapshot mode"),
    )
    dummy_hl_inst = SimpleNamespace(
        symbol="SOLUSDC",
        provider_name="hyperliquid",
        base="SOL",
        quote="USDC",
        params={},
        _provider=failing_provider,
        open_orders=lambda: _fail_live("live open_orders() must not be called in read_snapshot mode"),
    )
    dummy_gateway = SimpleNamespace(
        max_notional_usd=2000.0,
        load_all_intents=lambda: [],
        load_tradable_instruments=lambda **_kwargs: {"binance:BTCUSDC": dummy_inst, "hyperliquid:SOLUSDC": dummy_hl_inst},
    )

    # Ensure instruments.conf exists in workspace so load_tradable_instruments is exercised
    with open(os.path.join(temp_cache_dir, "instruments.conf"), "w", encoding="utf-8") as fh:
        fh.write("BTCUSDC\n")

    mhist_path = str(resolve_cachedb_filepath(os.path.join(temp_cache_dir, "microstructure_cycle_history.json")))
    if os.path.exists(mhist_path):
        os.remove(mhist_path)
    bals_mtime_before = os.path.getmtime(bals_path)

    snap = TelemetrySensorSnapshot()
    out = MarketAndPortfolioSnapshotOrchestrator(start_sync=False).read_snapshot(
        temp_cache_dir,
        "*",
        snapshot=snap,
        workspace_dir=temp_cache_dir,
        logs_dirs=[],
        intent_gateway=dummy_gateway,
        scope="crypto",
        now_ts=now_ts,
    )

    assert out is snap
    assert snap.source == "cachedb_read_only"
    assert snap.asset_scope == "crypto"
    assert snap.snapshot_id.startswith("market_portfolio:crypto:")
    assert snap.ts == pytest.approx(now_ts - 15.0)
    assert snap.evaluated_at_ts == pytest.approx(now_ts)
    assert snap.age_sec == pytest.approx(15.0)

    # Verify positions and open orders were reconstructed from cached venue_balances_state.json without live calls
    positions = snap.portfolio_summary.get("tradable_positions") or []
    btc_pos = next((p for p in positions if p.get("symbol") == "BTCUSDC"), None)
    assert btc_pos is not None
    assert btc_pos["free_base"] == pytest.approx(0.25)
    assert btc_pos["free_quote"] == pytest.approx(1500.0)
    open_orders = snap.portfolio_summary.get("open_orders") or []
    assert len(open_orders) == 1
    assert open_orders[0]["order_id"] == "ord-1"

    # Verify 0 disk mutations occurred during read-only snapshot population
    assert not os.path.exists(mhist_path)
    assert os.path.getmtime(bals_path) == pytest.approx(bals_mtime_before)

    # Verify FearGreedCollector, MarketBreadthCollector, and NewsFeedCollector start_sync=False / read_snapshot()

    fg_ro = FearGreedCollector(start_sync=False)
    mb_ro = MarketBreadthCollector(start_sync=False)
    nf_ro = NewsFeedCollector(history_file=os.path.join(temp_cache_dir, "news_feed_collect.json"), start_sync=False)
    assert fg_ro.start_sync is False
    assert mb_ro.start_sync is False
    assert nf_ro.start_sync is False
    assert fg_ro.read_snapshot(now_ts=now_ts) is None
    assert mb_ro.read_snapshot(now_ts=now_ts) is None
    nf_snap = nf_ro.read_snapshot(now_ts=now_ts)
    assert nf_snap is not None and nf_snap.total_fetched == 0


def test_telemetry_collector_factory_and_topic_agents_start_sync_contract(temp_cache_dir):
    from intelligence.macro.geopolitical_analyzer import GeopoliticalThreatAnalyzer
    from intelligence.macro_analyzer import MacroAnalyzer
    from intelligence.sentiment.sentiment_advisor import SentimentAdvisor

    # 1. TelemetryCollectorFactory start_sync consumer vs producer separation
    mb_consumer = TelemetryCollectorFactory.create("macro_benchmarks_snapshot", start_sync=False)
    mb_producer = TelemetryCollectorFactory.create("macro_benchmarks_snapshot", start_sync=True)
    assert mb_consumer is not mb_producer
    assert mb_consumer.start_sync is False
    assert mb_producer.start_sync is True

    # MacroBenchmarksSnapshotCollector.read_snapshot reads from disk with 0 network calls
    save_json_state(
        os.path.join(temp_cache_dir, "macro_benchmarks_collect.json"),
        {"ts": 1_750_000_000.0, "equity_regime": "RISK_ON", "items": {"SPY": {"price": 550.0}}},
    )
    cached_mb = mb_consumer.read_snapshot(temp_cache_dir)
    assert isinstance(cached_mb, dict)
    assert cached_mb["equity_regime"] == "RISK_ON"

    # EquityYahoo5dChartCollector.read_snapshot returns empty tuple with 0 network calls
    yc_consumer = TelemetryCollectorFactory.get_yahoo_chart_collector(start_sync=False)
    assert yc_consumer.start_sync is False
    assert yc_consumer.read_snapshot(temp_cache_dir, "NVDA_US_EQ") == ([], [], [], False, False, 0.0, 0)

    # 2. CachedTopicAssessmentAgent (GeopoliticalThreatAnalyzer & SentimentAdvisor) consumer vs producer
    geo_path = os.path.join(temp_cache_dir, "geopolitical_eval.json")
    adv_path = os.path.join(temp_cache_dir, "sentiment_eval.json")
    save_json_state(
        geo_path,
        {
            "threat_level": "ELEVATED",
            "risk_score": 0.65,
            "summary": "Cached geopolitical warning.",
            "recommended_brake": "DOWNSCALE_50",
            "headlines_analyzed": 3,
            "ts": 1_750_000_000.0,
        },
    )
    save_json_state(
        adv_path,
        {
            "market_bias": "CAUTION",
            "risk_level": "HIGH",
            "confidence": 0.8,
            "summary": "Cached advisor caution.",
            "key_risks": ["liquidity"],
            "recommended_action": "REDUCE",
            "ts": 1_750_000_000.0,
        },
    )

    geo_consumer = GeopoliticalThreatAnalyzer(state_file=geo_path, start_sync=False)
    adv_consumer = SentimentAdvisor(cache_file=adv_path, start_sync=False)
    assert geo_consumer.start_sync is False
    assert adv_consumer.start_sync is False
    assert geo_consumer.assess(None, force_refresh=True).threat_level == "ELEVATED"
    assert adv_consumer.review(force_refresh=True).market_bias == "CAUTION"

    daemon = MacroAnalyzer(cache_dir=temp_cache_dir)
    assert daemon.geopolitical_analyzer.start_sync is True
    assert daemon.sentiment_advisor.start_sync is True

    # 3. Border cases: PretradeTelemetrySnapshot age recomputation on later guard check + corrupted explicit telemetry

    snap_t0 = PretradeTelemetrySnapshot.from_raw(
        {"Orderbook Depth Imbalance": "0.58 (58% bids vs 42% asks)"},
        symbol="BTCUSDC",
        now_ts=1_750_000_000.0,
        ts=1_750_000_000.0,
    )
    assert snap_t0.age_sec == 0.0
    snap_t1 = PretradeTelemetrySnapshot.from_raw(snap_t0, symbol="BTCUSDC", now_ts=1_750_000_125.0)
    assert snap_t1.age_sec == 125.0

    guard = HighStakeGuard(
        gemini_client=MagicMock(),
        min_notional_eur=500.0,
        fallback_action="veto",
        mode="enforce",
        cache_ttl_sec=0.0,
    )
    dec_corrupt = guard.check("BTCUSDC", "BUY", 65000.0, 0.02, notional_eur=1300.0, telemetry="corrupted", now=1_750_000_125.0)
    assert dec_corrupt.allowed is False
    assert dec_corrupt.metadata["telemetry_available"] is False

    # MarketAndPortfolioSnapshotOrchestrator.read_snapshot with snapshot=None and tick-only cache_current_price.json
    save_json_state(
        os.path.join(temp_cache_dir, "cache_current_price.json"),
        {"items": {"BTCUSDC": [[1_750_000_050_000, 65100.0]]}},
    )
    orch = TelemetryCollectorFactory.create("macro_market_portfolio_snapshot", start_sync=False)
    auto_snap = orch.read_snapshot(temp_cache_dir, now_ts=1_750_000_100.0)
    assert auto_snap.source == "cachedb_read_only"
    assert auto_snap.ts == 1_750_000_050.0
    assert auto_snap.age_sec == 50.0

    # 4. Cross-venue CacheTradeManager.get_last_opposite_fill_price & equity order_guard._read_cached_price_history
    import order_guard

    krk_tm = CacheTradeManager(
        venue="kraken",
        filename=os.path.join(temp_cache_dir, "cache_trade_kraken.json"),
        read_only=True,
    )
    krk_tm._persist_items(
        "XBTUSD",
        [{"id": "KRK-XBT-1", "symbol": "XBTUSD", "isBuyer": True, "price": 67890.0, "time": 1_750_000_000_000}],
    )
    assert krk_tm.get_last_opposite_fill_price("BTCUSD", "SELL") == pytest.approx(67890.0)

    t212_tm = CacheTradeManager(
        venue="t212",
        filename=os.path.join(temp_cache_dir, "cache_trade_t212.json"),
        read_only=True,
    )
    t212_tm._persist_items(
        "NVDA_US_EQ",
        [{"id": "T212-NVDA-1", "symbol": "NVDA_US_EQ", "side": "BUY", "price": 138.5, "time": 1_750_000_000_000}],
    )
    assert t212_tm.get_last_opposite_fill_price("NVDA", "SELL") == pytest.approx(138.5)
    assert t212_tm.get_last_opposite_fill_price("NVDA_US_EQ", "SELL") == pytest.approx(138.5)

    p24_eq_dir = os.path.join(temp_cache_dir, "prices")
    os.makedirs(p24_eq_dir, exist_ok=True)
    save_json_state(
        os.path.join(p24_eq_dir, "cache_24_price_NVDA_US_EQ.json"),
        {"items": {"NVDA_US_EQ": [[1_749_999_000_000, 137.0], [1_750_000_000_000, 138.5]]}},
    )
    eq_hist = order_guard._read_cached_price_history("NVDA_US_EQ", window_seconds=3600.0, now=1_750_000_010.0)
    assert eq_hist is not None and len(eq_hist) == 2
    assert eq_hist[-1][1] == pytest.approx(138.5)
    assert CacheTradeManager.venue_for_symbol("XXBTZUSD") == "kraken"
    assert CacheTradeManager.venue_for_symbol("XDGUSD") == "kraken"


def test_unified_external_macro_collectors_four_pillar_guards_and_venue_runtime_paths(
    temp_cache_dir: str,
    monkeypatch: pytest.MonkeyPatch,
):
    """Verify Areas 1, 2, and 3:
    1. External (_DiskBackedSymbolCollector subclasses) & Macro (NewsFeedCollector) collectors inherit from
       TelemetryCacheCollectorInterface and register in TelemetryCollectorFactory.
    2. All 4 pillars of guards inherit from OrderGuardEvaluatorBase and register in OrderGuardEvaluatorFactory.
    3. Parameterized runtime state path resolution in cachepaths.py for trailing_state.json and .state_<SYM>.json.
    """
    from cachepaths import (
        TRAILING_STATE_FILENAME,
        iter_venue_cycle_state_files,
        resolve_venue_runtime_dir,
        resolve_venue_trailing_state_path,
    )
    from external_order_guard import ExternalOrderGuardsEvaluator
    from intelligence.composite import CompositeMarketIntelligence
    from intelligence.external.collectors.base_collector import _DiskBackedSymbolCollector
    from intelligence.external.collectors.bybit_liquidations_collector import BybitLiquidationCollector
    from intelligence.external.guards.cascade_guard import LiquidationCascadeGuard
    from intelligence.external.guards.funding_crowding_guard import FundingCrowdingGuard
    from intelligence.external.guards.orderbook_wall_guard import OrderbookWallGuard
    from intelligence.external.guards.smart_stop_loss_guard import SmartStopLossGuard
    from intelligence.external.guards.whale_divergence_guard import WhaleDivergenceGuard
    from intelligence.internal.guards.exhaustion_guard import WeibullExhaustionGuard
    from intelligence.internal.guards.noise_guard import NoiseFloorGuard
    from intelligence.internal.guards.parabolic_guard import ParabolicSurgeGuard
    from intelligence.internal.guards.trend_significance_guard import TrendSignificanceGuard
    from intelligence.macro.geopolitical_guard import GeopoliticalShockGuard
    from intelligence.sentiment.guards.extreme_greed_guard import ExtremeGreedGuard
    from intelligence.sentiment.guards.high_stake_guard import PreTradeLLMGuardBase
    from intelligence.sentiment.guards.panic_washout_guard import PanicWashoutGuard
    from llm_order_guard import LLMOrderGuardsEvaluator
    from qlimit_order_guard import QLimit2OrderGuardEvaluator

    # Area 1: External & Macro Collectors under TelemetryCacheCollectorInterface & TelemetryCollectorFactory
    assert issubclass(_DiskBackedSymbolCollector, TelemetryCacheCollectorInterface)
    with pytest.raises(TypeError):
        _DiskBackedSymbolCollector(5.0, 5.0, temp_cache_dir, 60.0, 60.0)

    for col_id, col_cls in (
        ("external_orderbook_depth", OrderbookDepthCollector),
        ("external_derivatives_telemetry", DerivativesTelemetryCollector),
        ("external_whale_positioning", WhalePositioningCollector),
        ("external_bybit_liquidations", BybitLiquidationCollector),
        ("macro_news_feed", NewsFeedCollector),
    ):
        assert issubclass(col_cls, TelemetryCacheCollectorInterface)
        consumer = TelemetryCollectorFactory.create(col_id, start_sync=False)
        producer = TelemetryCollectorFactory.create(col_id, start_sync=True)
        assert isinstance(consumer, col_cls)
        assert isinstance(producer, col_cls)
        assert consumer is not producer
        assert consumer.start_sync is False
        assert producer.start_sync is True

    bybit_col = TelemetryCollectorFactory.create("external_bybit_liquidations", start_sync=False)
    bybit_col.record_event("BTCUSDT", "Sell", 75000.0, ts=1_750_000_000.0)
    liq_snap = bybit_col.read_snapshot(temp_cache_dir, "BTCUSDT", window_sec=300.0, now=1_750_000_010.0)
    assert liq_snap.symbol == "BTCUSDT" and liq_snap.short_liq_usd == pytest.approx(75000.0)

    ob_col = OrderbookDepthCollector(cache_dir=temp_cache_dir, start_sync=False)
    assert ob_col.read_snapshot("UNLISTEDUSDC", now=1_750_000_000.0) is None
    assert ob_col.read_snapshot(temp_cache_dir, "UNLISTEDUSDC", now=1_750_000_000.0) is None
    assert ob_col.collect(temp_cache_dir, "UNLISTEDUSDC", allow_network=False, now=1_750_000_000.0) is None

    nf_col = NewsFeedCollector(history_file=os.path.join(temp_cache_dir, "news_feed_collect.json"), start_sync=False)
    nf_snap = nf_col.collect(temp_cache_dir, "*", allow_network=False, now_ts=1_750_000_000.0)
    assert nf_snap is not None and nf_snap.total_fetched == 0

    # Area 2: 4 Pillars of Guards under OrderGuardEvaluatorBase & OrderGuardEvaluatorFactory
    assert issubclass(PreTradeLLMGuardBase, OrderGuardEvaluatorBase)
    with pytest.raises(TypeError):
        PreTradeLLMGuardBase()

    expected_guards = (
        ("internal_noise_floor", NoiseFloorGuard),
        ("internal_weibull_exhaustion", WeibullExhaustionGuard),
        ("internal_parabolic_surge", ParabolicSurgeGuard),
        ("internal_trend_significance", TrendSignificanceGuard),
        ("qlimit2_order_guards", QLimit2OrderGuardEvaluator),
        ("external_liquidation_cascade", LiquidationCascadeGuard),
        ("external_funding_crowding", FundingCrowdingGuard),
        ("external_orderbook_wall", OrderbookWallGuard),
        ("external_whale_divergence", WhaleDivergenceGuard),
        ("external_smart_stop_loss", SmartStopLossGuard),
        ("external_order_guards", ExternalOrderGuardsEvaluator),
        ("sentiment_extreme_greed", ExtremeGreedGuard),
        ("sentiment_panic_washout", PanicWashoutGuard),
        ("sentiment_high_stake_llm", HighStakeGuard),
        ("sentiment_post_stop_reentry_llm", PostStopReentryGuard),
        ("llm_order_guards", LLMOrderGuardsEvaluator),
        ("macro_geopolitical_shock", GeopoliticalShockGuard),
        ("composite_market_intelligence", CompositeMarketIntelligence),
    )
    for guard_id, guard_cls in expected_guards:
        assert issubclass(guard_cls, OrderGuardEvaluatorBase)
        assert issubclass(guard_cls, OrderGuardEvaluatorInterface)
        g_inst = OrderGuardEvaluatorFactory.create(guard_id)
        assert isinstance(g_inst, guard_cls)
        assert OrderGuardEvaluatorFactory.get_evaluator(guard_cls) is g_inst

    # Verify polymorphic evaluate() <-> check() delegation
    nf_guard = OrderGuardEvaluatorFactory.create("internal_noise_floor", min_strength_ratio=2.0)
    dec_eval = nf_guard.evaluate("BTCUSDC", "BUY", 0.001, 0.002)
    dec_check = nf_guard.check("BTCUSDC", "BUY", 0.001, 0.002)
    assert dec_eval.allowed is False and dec_check.allowed is False

    ts_guard = OrderGuardEvaluatorFactory.create("internal_trend_significance")
    assert ts_guard.check(None, [100.0, 101.0]).allowed is True

    comp_guard = OrderGuardEvaluatorFactory.create("composite_market_intelligence")
    assert comp_guard.check("BTCUSDC", "BUY", 70000.0, gradient=0.0005, epsilon=0.001).allowed is False

    # Area 3: Parameterized Runtime State Path Resolution in cachepaths.py
    ws_dir = os.path.join(temp_cache_dir, "ws")
    for v_name, sub_dir in (("kraken", "kraken"), ("hyperliquid", "hyperliquid"), ("t212", "212trading")):
        os.makedirs(os.path.join(ws_dir, sub_dir), exist_ok=True)
    assert resolve_venue_runtime_dir("binance", cache_dir="cachedb") == "cachedb"
    assert resolve_venue_runtime_dir("binance", cache_dir=temp_cache_dir) == temp_cache_dir
    assert resolve_venue_runtime_dir("kraken", workspace_dir=ws_dir) == os.path.join(ws_dir, "kraken")
    assert resolve_venue_runtime_dir("hyperliquid", workspace_dir=ws_dir) == os.path.join(ws_dir, "hyperliquid")
    assert resolve_venue_runtime_dir("t212", workspace_dir=ws_dir) == os.path.join(ws_dir, "212trading")

    assert resolve_venue_trailing_state_path("binance", cache_dir="cachedb") == os.path.join(
        "cachedb", TRAILING_STATE_FILENAME
    )
    assert resolve_venue_trailing_state_path("binance", cache_dir=temp_cache_dir) == os.path.join(
        temp_cache_dir, TRAILING_STATE_FILENAME
    )
    assert resolve_venue_trailing_state_path("kraken", workspace_dir=ws_dir) == os.path.join(
        ws_dir, "kraken", TRAILING_STATE_FILENAME
    )
    assert resolve_venue_trailing_state_path(
        "t212", workspace_dir=ws_dir, filename="trailing_peaks.json"
    ) == os.path.join(ws_dir, "212trading", "trailing_peaks.json")


    tm_str_flag = CacheTradeManager(
        60,
        [],
        os.path.join(temp_cache_dir, "str_flag_trade.jsonl"),
        read_only="false",
        data_sync_mode="producer",
    )
    assert tm_str_flag.read_only is False
    assert tm_str_flag.data_sync_mode == "producer"

    krk_st = os.path.join(ws_dir, "kraken", ".state_TAOUSD.json")
    hl_st = os.path.join(ws_dir, "hyperliquid", ".state_HYPE_LONG.json")

    for st_p in (
        krk_st,
        hl_st,
        os.path.join(ws_dir, "hyperliquid", ".state_HYPE.json"),
    ):
        with open(st_p, "w", encoding="utf-8") as fh:
            fh.write("{}")
    assert iter_venue_cycle_state_files("kraken", workspace_dir=ws_dir) == [krk_st]
    hl_files = iter_venue_cycle_state_files("hyperliquid", workspace_dir=ws_dir)
    assert hl_files[0] == hl_st

    from intelligence.sentiment.evaluators.price_series_evaluator import (
        _apply_24h_series_enrichment,
        recalculate_live_price_geometry,
    )

    rgnt_entry = {
        "price": 1.50,
        "low_7d": 1.16,
        "high_7d": 1.60,
        "swing_low_px": 1.27,
        "swing_high_px": 1.60,
        "major_swing_low_px": 1.16,
        "rally_from_swing_low_pct": 18.11,
        "rally_from_major_low_pct": 29.31,
        "recent_swings_7d": ["+37.9%", "-20.6%", "+18.1%"],
    }
    _apply_24h_series_enrichment(
        "RGNT_US_EQ",
        rgnt_entry,
        {"low_24h": 1.15, "high_24h": 1.55},
        [[1000, 1.15], [2000, 1.50]],
        None,
    )
    assert rgnt_entry["low_7d"] == pytest.approx(1.15)
    assert rgnt_entry["swing_low_px"] == pytest.approx(1.27)
    assert rgnt_entry["major_swing_low_px"] == pytest.approx(1.15)
    recalculate_live_price_geometry(rgnt_entry, 1.50)
    assert rgnt_entry["swing_low_px"] == pytest.approx(1.27)
    assert rgnt_entry["major_swing_low_px"] == pytest.approx(1.15)
    assert rgnt_entry["rally_from_major_low_pct"] == pytest.approx(30.43, abs=0.02)

    rgnt_breakdown = {
        "price": 1.15,
        "low_7d": 1.16,
        "high_7d": 1.60,
        "swing_low_px": 1.27,
        "swing_high_px": 1.60,
        "major_swing_low_px": 1.16,
        "rally_from_swing_low_pct": 18.11,
        "rally_from_major_low_pct": 29.31,
        "recent_swings_7d": ["+37.9%", "-20.6%", "+18.1%"],
    }
    _apply_24h_series_enrichment(
        "RGNT_US_EQ",
        rgnt_breakdown,
        {"low_24h": 1.15, "high_24h": 1.55},
        [[1000, 1.50], [2000, 1.15]],
        None,
    )
    assert rgnt_breakdown["swing_low_px"] == pytest.approx(1.15)
    assert "major_swing_low_px" not in rgnt_breakdown
    assert "rally_from_major_low_pct" not in rgnt_breakdown


def test_bare_equity_symbol_matches_suffixed_equity_in_pretrade_telemetry(tmp_path):
    """Bare equity ticker ('NVDA') must resolve EquityRthSessionPolicy and match 'NVDA_US_EQ' without crypto BTC relative strength."""

    assert resolve_asset_session_policy("NVDA").is_equity is True
    assert resolve_asset_session_policy("NVDA_US_EQ").is_equity is True

    now_ms = int(time.time() * 1000)
    pts_24h = [[now_ms - (24 - i) * 3600 * 1000, 170.0 + i * 0.25, 1000.0 + i * 10.0] for i in range(25)]
    btc_24h = [[now_ms - (24 - i) * 3600 * 1000, 70000.0 + i * 100.0, 500.0] for i in range(25)]
    (tmp_path / "cache_24_price_NVDA_US_EQ.json").write_text(
        json.dumps({"items": {"NVDA_US_EQ": pts_24h}}), encoding="utf-8"
    )
    (tmp_path / "cache_ohlc_24h_NVDA_US_EQ.json").write_text(
        json.dumps({"ts": now_ms / 1000.0, "symbol": "NVDA_US_EQ", "points": pts_24h}), encoding="utf-8"
    )
    (tmp_path / "cache_24_price_BTCUSDC.json").write_text(
        json.dumps({"items": {"BTCUSDC": btc_24h}}), encoding="utf-8"
    )

    t_bare = collect_pretrade_telemetry("NVDA", cachedb_dir=str(tmp_path), allow_network=False)
    t_eq = collect_pretrade_telemetry("NVDA_US_EQ", cachedb_dir=str(tmp_path), allow_network=False)
    assert "Relative Strength vs BTC (24h Benchmark)" not in t_bare
    assert "Relative Strength vs BTC (24h Benchmark)" not in t_eq
    assert t_bare.get("24h Multi-Horizon Price Evolution") == t_eq.get("24h Multi-Horizon Price Evolution")
    assert t_bare.get("24h Volume Profile (POC / Value Area)") == t_eq.get("24h Volume Profile (POC / Value Area)")









"""Unit tests for GlobalTelemetryCollector."""

from __future__ import annotations

import json
import os
import time
from unittest.mock import MagicMock, patch

import pytest

from intelligence.globaltelemetry_collector import GlobalTelemetryCollector
from intelligence.external.collectors.orderbook_depth import OrderbookSnapshot
from intelligence.external.collectors.derivatives_telemetry import DerivativesTelemetry
from intelligence.external.collectors.whale_positioning import WhalePositioningSnapshot
from intelligence.sentiment.collectors.fear_greed_collector import FearGreedSnapshot
from intelligence.sentiment.collectors.market_breadth_collector import MarketBreadthSnapshot


@pytest.fixture
def temp_cache_dir(tmp_path):
    cache_dir = tmp_path / "cachedb"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir)


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
         patch.object(collector.breadth_collector, "fetch", return_value=mock_breadth):

        res = collector.run_cycle(force=True)

        assert res["orderbook_updated"] == 1
        assert res["derivatives_updated"] == 1
        assert res["whale_updated"] == 1
        assert res["news_updated"] is True
        assert res["fear_greed_updated"] is True
        assert res["breadth_updated"] is True

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
    from intelligence.external.collectors.orderbook_depth import OrderbookDepthCollector

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
    import pytest
    from cacheprice import (
        Cache24PriceManager,
        Cache365PriceManager,
        CacheCurrentPriceManager,
        CacheSparsePriceManager,
    )
    from intelligence.sentiment.collectors.microstructure_collector import (
        CompositeSymbolPriceSeriesLoader,
        EquityYahoo5dChartCollector,
        Fleet24hAndMicrostructureEnricher,
        MicrostructureCacheCollectorBase,
        PriceSeriesCacheCollectorBase,
        SymbolMicrostructureSnapshotCollector,
        TelemetryCacheCollectorInterface,
        TelemetryCollectorFactory,
    )
    from intelligence.sentiment.collectors.pretrade_telemetry_collector import (
        PretradeEvaluationContext,
        PretradePillarCollectorBase,
        PretradeTelemetryOrchestrator,
        collect_pretrade_telemetry,
    )
    from intelligence.sentiment.evaluators.microstructure_evaluator import (
        CrossVenueSiblingArbEvaluator,
        LiquidationClusterEvaluator,
        MicrostructureCycleDeltaEvaluator,
        MicrostructureEvaluatorBase,
        OrderbookCvdFlowEvaluator,
    )
    from intelligence.sentiment.evaluators.price_series_evaluator import (
        AssetSessionPolicy,
        Crypto24x7SessionPolicy,
        EquityRthSessionPolicy,
        Horizon24hPriceEvaluator,
        PriceSeriesEvaluatorBase,
        ShortTrendAndKalmanEvaluator,
        Structural7dWaveEvaluator,
        TelemetryEvaluatorInterface,
        VolumeProfileEvaluator,
        resolve_asset_session_policy,
    )

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
    assert crypto_pol.supports_cross_venue_sibling_arb("BTCUSDC") is True
    assert eq_pol.supports_cross_venue_sibling_arb("NVDA_US_EQ") is False

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
    from cacheaccount import CacheAssetValueManager, CacheOrderManager, CacheTradeManager
    from cacheprice import CachePriceManagerBase
    from cachetrend import CacheLongTrendManager, CacheShortTrendManager

    assert CacheShortTrendManager.disk_filename_for_symbol() == "cache_short_trend.json"
    assert CacheShortTrendManager.live_window_relpath_for_symbol("BTCUSDC").replace("\\", "/") == "24trend/live_BTCUSDC.json"
    assert CacheLongTrendManager.disk_filename_for_symbol() == "cache_long_trend.json"
    assert CacheTradeManager.disk_filename_for_symbol() == "cache_trade.jsonl"
    assert CacheOrderManager.disk_filename_for_symbol() == "cache_order.json"
    assert CacheAssetValueManager.disk_filename_for_symbol() == "cache_asset_value.jsonl"
    assert CachePriceManagerBase.extract_jsonl_series_point({"s": "BTCUSDC", "i": [1700000000000, 95000.5]}, {"BTCUSDC"}) == [
        1700000000000,
        95000.5,
    ]
    assert CachePriceManagerBase.extract_jsonl_series_point({"s": "ETHUSDC", "i": [1700000000000, 3000.0]}, {"BTCUSDC"}) is None
    assert isinstance(TelemetryCollectorFactory.get_microstructure_collector(), SymbolMicrostructureSnapshotCollector)
    assert isinstance(TelemetryCollectorFactory.get_fleet_enricher(), Fleet24hAndMicrostructureEnricher)
    assert isinstance(TelemetryCollectorFactory.get_yahoo_chart_collector(), EquityYahoo5dChartCollector)
    assert isinstance(TelemetryCollectorFactory.get_collector(PretradeTelemetryOrchestrator), PretradeTelemetryOrchestrator)

    class _CustomAuditStage(PretradePillarCollectorBase):
        def evaluate(self, ctx: PretradeEvaluationContext) -> None:
            ctx.telemetry["Custom Polymorphic Pillar"] = f"{ctx.symbol}:{ctx.session_policy.cls_name}"

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
    botcore.runtime_sleep(0.5)
    assert slept == [0.5]

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
    from intelligence.sentiment.collectors.pretrade_telemetry_collector import (
        PretradeTelemetryOrchestrator,
        PretradeTelemetrySnapshot,
        read_pretrade_telemetry_snapshot,
        refresh_pretrade_telemetry_caches,
    )
    from intelligence.sentiment.guards.high_stake_guard import (
        _HIGH_STAKE_DECISION_CACHE,
        HighStakeGuard,
    )
    from intelligence.sentiment.guards.post_stop_reentry_guard import (
        PostStopReentryGuard,
        clear_post_stop_reentry_cache,
    )

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
    assert orchestrator.is_read_only_consumer() is True

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

    class _DummyCollector:
        def fetch(self, symbol: str, **kwargs):
            seen_fetch_kwargs.append(kwargs)
            return {"symbol": symbol}

    fetch_guard_snapshot(_DummyCollector(), "BTCUSDC", now=now_ts)
    assert seen_fetch_kwargs == [{"allow_network": False, "now": now_ts}]

    # 5. Producer refresh_pretrade_telemetry_caches / GlobalTelemetryCollector.update_pretrade_caches performs refresh
    status = refresh_pretrade_telemetry_caches("NVDA_US_EQ", cachedb_dir=temp_cache_dir, now_ts=now_ts)
    assert status["microstructure_refreshed"] is True
    assert status["t212_positions_refreshed"] is True
    assert "microstructure" in network_calls
    assert "t212_refresh" in network_calls
    assert "yahoo_chart" in network_calls






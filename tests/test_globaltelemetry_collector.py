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
    from intelligence.sentiment.collectors.microstructure_collector import (
        Archive365AndSparsePriceCollector,
        CompositeSymbolPriceSeriesLoader,
        Dense24hPriceSeriesCollector,
        EquityYahoo5dChartCollector,
        Fleet24hAndMicrostructureEnricher,
        LiveCurrentPriceOverlayCollector,
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

    assert isinstance(TelemetryCollectorFactory.get_price_series_loader(), CompositeSymbolPriceSeriesLoader)
    assert isinstance(TelemetryCollectorFactory.get_microstructure_collector(), SymbolMicrostructureSnapshotCollector)
    assert isinstance(TelemetryCollectorFactory.get_fleet_enricher(), Fleet24hAndMicrostructureEnricher)
    assert isinstance(TelemetryCollectorFactory.get_yahoo_chart_collector(), EquityYahoo5dChartCollector)
    assert isinstance(TelemetryCollectorFactory.get_collector(PretradeTelemetryOrchestrator), PretradeTelemetryOrchestrator)
    assert issubclass(Dense24hPriceSeriesCollector, PriceSeriesCacheCollectorBase)
    assert issubclass(Archive365AndSparsePriceCollector, PriceSeriesCacheCollectorBase)
    assert issubclass(LiveCurrentPriceOverlayCollector, PriceSeriesCacheCollectorBase)

    class _CustomAuditStage(PretradePillarCollectorBase):
        def evaluate(self, ctx: PretradeEvaluationContext) -> None:
            ctx.telemetry["Custom Polymorphic Pillar"] = f"{ctx.symbol}:{ctx.session_policy.cls_name}"

    custom_tel = collect_pretrade_telemetry("BTCUSDC", cachedb_dir=temp_cache_dir, stages=(_CustomAuditStage(),))
    assert custom_tel == {"Custom Polymorphic Pillar": "BTCUSDC:Crypto24x7SessionPolicy"}




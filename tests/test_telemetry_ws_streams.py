"""Unit tests for shared JsonPublicWsStreamBase, venue WS telemetry parsers, and TelemetryWsHub integration."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import time
import pytest

from intelligence.external.collectors.derivatives_telemetry_collector import DerivativesTelemetryCollector
from intelligence.external.collectors.orderbook_depth_collector import OrderbookDepthCollector
from intelligence.external.collectors.telemetry_ws_streams import (
    BinanceFuturesMarkPriceStream,
    BinanceSpotKline5mStream,
    HyperliquidAssetCtxStream,
    TelemetryWsHub,
)
from intelligence.globaltelemetry_collector import GlobalTelemetryCollector
from providers.binance_provider import BinanceProvider
from providers.venue_ws_protocols import BinancePublicWsMixin, HyperliquidWsMixin
from state_io import encode_json_payload, read_json_dict


def test_binance_and_hyperliquid_ws_protocol_parsers_and_url_builders():
    """Verify stateless URL builders, subscribe payloads, and payload parsers in providers/venue_ws_protocols.py."""
    mp_url = BinancePublicWsMixin.build_ws_mark_price_url(["BTCUSDT", "btcusdt", "ETHUSDT"], interval_sec=1)
    assert mp_url == "wss://fstream.binance.com/market/stream?streams=btcusdt@markPrice@1s/ethusdt@markPrice@1s"

    kl_url = BinancePublicWsMixin.build_ws_spot_kline_5m_url(["BTCUSDC", "BTCUSDT", "btcusdc"])
    assert kl_url == "wss://stream.binance.com:9443/stream?streams=btcusdc@kline_5m/btcusdt@kline_5m"

    mp_env = {
        "stream": "btcusdt@markPrice@1s",
        "data": {
            "e": "markPriceUpdate",
            "s": "BTCUSDT",
            "p": "60500.25",
            "i": "60480.00",
            "r": "-0.000125",
            "T": 1_760_006_400_000,
        },
    }
    parsed_mp = BinancePublicWsMixin.extract_ws_mark_price(mp_env)
    assert parsed_mp is not None
    assert parsed_mp["symbol"] == "BTCUSDT"
    assert parsed_mp["mark_price"] == pytest.approx(60500.25)
    assert parsed_mp["index_price"] == pytest.approx(60480.00)
    assert parsed_mp["funding_rate"] == pytest.approx(-0.000125)
    assert BinancePublicWsMixin.extract_ws_mark_price({"e": "other"}) is None

    kl_env = {
        "stream": "btcusdc@kline_5m",
        "data": {
            "e": "kline",
            "s": "BTCUSDC",
            "k": {"t": 1_760_000_000_000, "s": "BTCUSDC", "i": "5m", "q": "100000.0", "Q": "120000.0"},
        },
    }
    # Buy quote volume must be clamped safely to total quote volume
    assert BinancePublicWsMixin.extract_ws_kline_5m_flow(kl_env) == (
        "BTCUSDC",
        1_760_000_000_000,
        100000.0,
        100000.0,
    )
    # Non-5m intervals are ignored
    kl_1m = {"e": "kline", "s": "BTCUSDC", "k": {"t": 1_760_000_000_000, "i": "1m", "q": "10", "Q": "5"}}
    assert BinancePublicWsMixin.extract_ws_kline_5m_flow(kl_1m) is None

    subs = HyperliquidWsMixin.build_ws_active_asset_ctx_subscriptions(["HYPEUSDC", "HYPE"])
    assert len(subs) == 1
    assert '"activeAssetCtx"' in subs[0] and '"HYPE"' in subs[0]
    assert '"ping"' in HyperliquidWsMixin.build_ws_ping_payload()

    hl_env = {
        "channel": "activeAssetCtx",
        "data": {
            "coin": "HYPE",
            "ctx": {
                "funding": "0.000015",
                "openInterest": "250000.5",
                "markPx": "61.25",
                "oraclePx": "61.20",
            },
        },
    }
    hl_parsed = HyperliquidWsMixin.extract_ws_active_asset_ctx(hl_env)
    assert hl_parsed is not None
    coin, ctx = hl_parsed
    assert coin == "HYPE"
    assert ctx["funding"] == pytest.approx(0.000015)
    assert ctx["openInterest"] == pytest.approx(250000.5)
    assert ctx["markPx"] == pytest.approx(61.25)
    assert ctx["oraclePx"] == pytest.approx(61.20)
    assert HyperliquidWsMixin.extract_ws_active_asset_ctx({"channel": "pong"}) is None


def test_binance_spot_kline_5m_stream_seed_parity_gap_detection_and_disconnect():
    """Verify exact CVD parity with REST extractor, non-regressing seed, VPN gap detection, and disconnect fail-closed."""
    stream = BinanceSpotKline5mStream(["BTCUSDC", "ARBUSDC"])
    t0_ms = 1_760_000_100_000  # aligned 5m bucket base
    t1_ms = t0_ms + 300_000
    t2_ms = t1_ms + 300_000
    now_s = (t2_ms + 45_000) / 1000.0

    # REST payload with 3 contiguous 5m bars
    rest_rows = [
        [t0_ms, "1", "1", "1", "1", "10", t0_ms + 299_999, "100000.0", 50, "6", "60000.0", "0"],
        [t1_ms, "1", "1", "1", "1", "10", t1_ms + 299_999, "200000.0", 80, "4", "80000.0", "0"],
        [t2_ms, "1", "1", "1", "1", "10", t2_ms + 299_999, "150000.0", 60, "5", "90000.0", "0"],
    ]
    rest_cvd, rest_vol = BinanceProvider.extract_spot_flow_15m_usd(rest_rows)

    # Seed from REST before WS frame arrives -> still needs live WS activity to return from RAM
    assert stream.seed_from_rest_klines("BTCUSDC", rest_rows, now=now_s) == 3
    assert stream.get_spot_flow_15m("BTCUSDC", now=now_s) is None

    # Feed live WS update for newest bar (same volumes -> bit-for-bit parity with REST)
    ws_frame = encode_json_payload({
        "stream": "btcusdc@kline_5m",
        "data": {"e": "kline", "s": "BTCUSDC", "k": {"t": t2_ms, "s": "BTCUSDC", "i": "5m", "q": "150000.0", "Q": "90000.0"}},
    })
    assert stream.ingest_raw_message(ws_frame, now_ts=now_s) is True
    assert stream.get_spot_flow_15m("BTCUSDC", now=now_s + 5.0) == (rest_cvd, rest_vol)

    # Newer WS update increases live bucket volume; re-seeding with stale REST must NOT regress volume
    ws_frame_newer = encode_json_payload({
        "stream": "btcusdc@kline_5m",
        "data": {"e": "kline", "s": "BTCUSDC", "k": {"t": t2_ms, "s": "BTCUSDC", "i": "5m", "q": "180000.0", "Q": "120000.0"}},
    })
    stream.ingest_raw_message(ws_frame_newer, now_ts=now_s + 6.0)
    stream.seed_from_rest_klines("BTCUSDC", rest_rows, now=now_s + 7.0)
    # Total vol = 100k + 200k + 180k = 480k; buy = 60k + 80k + 120k = 260k; cvd = 520k - 480k = +40k
    assert stream.get_spot_flow_15m("BTCUSDC", now=now_s + 8.0) == (40000.0, 480000.0)

    # Quiet symbol (ARBUSDC) seeded with 3 contiguous bars remains valid while combined socket stays fresh via BTCUSDC
    stream._connected = True
    assert stream.seed_from_rest_klines("ARBUSDC", rest_rows, now=now_s - 40.0) == 3
    assert stream.get_spot_flow_15m("ARBUSDC", now=now_s + 10.0) == (rest_cvd, rest_vol)

    # VPN gap detection: if a 5m bucket is skipped (t2 -> t2 + 10m), get_spot_flow_15m fails closed -> REST re-seed
    t_gap_ms = t2_ms + 600_000
    gap_frame = encode_json_payload({
        "stream": "btcusdc@kline_5m",
        "data": {"e": "kline", "s": "BTCUSDC", "k": {"t": t_gap_ms, "s": "BTCUSDC", "i": "5m", "q": "50000.0", "Q": "25000.0"}},
    })
    stream.ingest_raw_message(gap_frame, now_ts=(t_gap_ms + 10_000) / 1000.0)
    assert stream.get_spot_flow_15m("BTCUSDC", now=(t_gap_ms + 15_000) / 1000.0) is None

    # Disconnect fail-closed: _on_session_end immediately disables WS reads until reconnected
    stream._on_session_end(RuntimeError("vpn_reset"))
    assert stream.get_spot_flow_15m("ARBUSDC", now=now_s + 12.0) is None


def test_collectors_use_ws_hub_fast_path_and_transparently_fallback_to_rest(tmp_path, monkeypatch):
    """Verify DerivativesTelemetryCollector, OrderbookDepthCollector, and GlobalTelemetryCollector heartbeat integration."""
    hub = TelemetryWsHub(["BTCUSDC", "HYPEUSDC", "SPCX_US_EQ", "SPCXxUSD"])
    assert hub.mark_price_stream is not None
    assert hub.hl_asset_ctx_stream is not None
    assert hub.binance_spot_kline_stream is not None

    t0_ms = 1_760_010_000_000
    t1_ms = t0_ms + 300_000
    t2_ms = t1_ms + 300_000
    now_s = (t2_ms + 30_000) / 1000.0

    # 1. DerivativesTelemetryCollector: Binance uses WS markPrice (skips premiumIndex REST, fetches openInterest only)
    hub.mark_price_stream.ingest_raw_message(
        {
            "stream": "btcusdt@markPrice@1s",
            "data": {"e": "markPriceUpdate", "s": "BTCUSDT", "p": "61000.0", "i": "60969.5", "r": "0.0001"},
        },
        now_ts=now_s,
    )
    deriv_col = DerivativesTelemetryCollector(cache_dir=str(tmp_path), start_sync=True)
    deriv_col.ws_hub = hub
    deriv_urls = []

    def _fake_deriv_fetch(url, label, lookup_sym, **_kw):
        deriv_urls.append(label)
        if label == "openInterest":
            return {"openInterest": "1000.0"}
        if label == "premiumIndex":
            return {"lastFundingRate": "0.0002", "nextFundingRate": "0.0002", "markPrice": "61100.0", "indexPrice": "61050.0"}
        return {}

    monkeypatch.setattr(deriv_col, "_fetch_json_dict", _fake_deriv_fetch)
    snap_ws = deriv_col.fetch("BTCUSDC", allow_network=True, now=now_s + 2.0)
    assert snap_ws is not None
    assert deriv_urls == ["openInterest"]
    assert snap_ws.mark_price == pytest.approx(61000.0)
    assert snap_ws.funding_rate == pytest.approx(0.0001)
    assert snap_ws.open_interest_usd == pytest.approx(61_000_000.0)

    # Disconnect -> transparently falls back to REST premiumIndex + openInterest once TTL expires
    hub.mark_price_stream._on_session_end(RuntimeError("closed"))
    deriv_urls.clear()
    snap_rest = deriv_col.fetch("BTCUSDC", allow_network=True, now=now_s + 180.0)
    assert snap_rest is not None
    assert deriv_urls == ["premiumIndex", "openInterest"]
    assert snap_rest.funding_rate == pytest.approx(0.0002)

    # 2. DerivativesTelemetryCollector: Hyperliquid uses WS activeAssetCtx with ZERO REST calls
    hub.hl_asset_ctx_stream.ingest_raw_message(
        {
            "channel": "activeAssetCtx",
            "data": {
                "coin": "HYPE",
                "ctx": {"funding": "0.00002", "openInterest": "500000.0", "markPx": "60.0", "oraclePx": "59.94"},
            },
        },
        now_ts=now_s,
    )
    hl_rest_called = [False]
    monkeypatch.setattr(deriv_col, "_fetch_hyperliquid_asset_ctx", lambda *_a, **_kw: hl_rest_called.__setitem__(0, True) or None)
    hl_snap = deriv_col._fetch_hyperliquid_derivatives("HYPEUSDC", now_s + 3.0)
    assert hl_snap is not None
    assert hl_rest_called[0] is False
    assert hl_snap.open_interest_usd == pytest.approx(30_000_000.0)
    assert hl_snap.funding_rate == pytest.approx(0.00002)

    # 3. OrderbookDepthCollector: seeds BinanceSpotKline5mStream on 1st REST call, then serves 0-REST WS flow
    ob_col = OrderbookDepthCollector(cache_dir=str(tmp_path), start_sync=True)
    ob_col.ws_hub = hub
    kline_rest_calls = []

    def _fake_cvd_payload(url, _body, timeout=3.5):
        kline_rest_calls.append(url)
        return [
            [t0_ms, "1", "1", "1", "1", "10", t0_ms + 299_999, "100000.0", 50, "6", "60000.0", "0"],
            [t1_ms, "1", "1", "1", "1", "10", t1_ms + 299_999, "100000.0", 50, "6", "60000.0", "0"],
            [t2_ms, "1", "1", "1", "1", "10", t2_ms + 299_999, "100000.0", 50, "6", "60000.0", "0"],
        ]

    monkeypatch.setattr(ob_col, "_fetch_spot_cvd_payload", _fake_cvd_payload)
    # 1st call: WS not yet seeded -> hits REST for BTCUSDC + BTCUSDT and seeds both
    cvd_1 = ob_col._resolve_real_spot_cvd_15m("BTCUSDC", venue="binance", now=now_s)
    assert cvd_1 == pytest.approx(60000.0)
    assert len(kline_rest_calls) == 2

    # Push WS updates for both BTCUSDC and BTCUSDT -> 2nd call uses 0 REST calls and reflects live WS delta
    hub.binance_spot_kline_stream.ingest_raw_message(
        {"stream": "btcusdc@kline_5m", "data": {"e": "kline", "s": "BTCUSDC", "k": {"t": t2_ms, "s": "BTCUSDC", "i": "5m", "q": "150000.0", "Q": "125000.0"}}},
        now_ts=now_s + 10.0,
    )
    hub.binance_spot_kline_stream.ingest_raw_message(
        {"stream": "btcusdt@kline_5m", "data": {"e": "kline", "s": "BTCUSDT", "k": {"t": t2_ms, "s": "BTCUSDT", "i": "5m", "q": "200000.0", "Q": "150000.0"}}},
        now_ts=now_s + 10.0,
    )
    kline_rest_calls.clear()
    cvd_2 = ob_col._resolve_real_spot_cvd_15m("BTCUSDC", venue="binance", now=now_s + 15.0)
    assert kline_rest_calls == []
    # BTCUSDC bars: (100k,60k)+(100k,60k)+(150k,125k) => tot=350k, buy=245k => cvd = 490k - 350k = +140k
    assert cvd_2 == pytest.approx(140000.0)
    assert ob_col._agg_spot_cvd_memo[("binance", "BTCUSDC")][1] == pytest.approx(140000.0 + 140000.0)

    # 4. GlobalTelemetryCollector heartbeat includes compact ws_streams status when hub attached
    gtc = GlobalTelemetryCollector(symbols=["BTCUSDC", "HYPEUSDC"], cache_dir=str(tmp_path), skip_news=True, skip_external=True)
    gtc.telemetry_ws_hub = hub
    gtc.write_heartbeat({"ts": now_s})
    hb_doc = read_json_dict(gtc.heartbeat_path)
    assert isinstance(hb_doc, dict)
    assert "ws_streams" in hb_doc
    assert "binance_mark_price" in hb_doc["ws_streams"]
    assert "hyperliquid_asset_ctx" in hb_doc["ws_streams"]
    assert "binance_spot_kline_5m" in hb_doc["ws_streams"]


def test_telemetry_ws_streams_concurrent_stress():
    """Verify thread-safety across 8 concurrent writers/readers on all 3 telemetry WS stream classes."""
    mp_st = BinanceFuturesMarkPriceStream(["BTCUSDT", "ETHUSDT"])
    hl_st = HyperliquidAssetCtxStream(["HYPE"])
    kl_st = BinanceSpotKline5mStream(["BTCUSDC", "ETHUSDC"])
    t0_ms = 1_760_020_000_000
    t1_ms = t0_ms + 300_000
    t2_ms = t1_ms + 300_000
    base_now = (t2_ms + 20_000) / 1000.0
    errs = []

    def _worker(i: int) -> None:
        try:
            for j in range(30):
                now_v = base_now + j * 0.1
                mp_st.ingest_raw_message(
                    {"e": "markPriceUpdate", "s": "BTCUSDT", "p": str(60000 + i + j), "i": "60000", "r": "0.0001"},
                    now_ts=now_v,
                )
                mp_st.get_mark_price("BTCUSDT", now=now_v)
                hl_st.ingest_raw_message(
                    {"channel": "activeAssetCtx", "data": {"coin": "HYPE", "ctx": {"funding": "0.00001", "openInterest": "100", "markPx": str(60 + j * 0.01), "oraclePx": "60.0"}}},
                    now_ts=now_v,
                )
                hl_st.get_asset_ctx("HYPEUSDC", now=now_v)
                kl_st.seed_from_rest_klines(
                    "BTCUSDC",
                    [
                        [t0_ms, "1", "1", "1", "1", "1", t0_ms + 299_999, "1000", 1, "1", "600", "0"],
                        [t1_ms, "1", "1", "1", "1", "1", t1_ms + 299_999, "1000", 1, "1", "600", "0"],
                    ],
                    now=now_v,
                )
                kl_st.ingest_raw_message(
                    {"e": "kline", "s": "BTCUSDC", "k": {"t": t2_ms, "s": "BTCUSDC", "i": "5m", "q": str(1000 + j * 10), "Q": str(600 + j * 5)}},
                    now_ts=now_v,
                )
                kl_st.get_spot_flow_15m("BTCUSDC", now=now_v)
        except Exception as exc:
            errs.append(exc)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(_worker, range(8)))
    assert errs == []


def test_ws_pong_does_not_mask_staleness_silent_socket_reconnects_and_whale_uses_ws_hub(tmp_path, monkeypatch):
    """Verify non-data frames (pong) do not refresh last_message_ts, silent sockets break for reconnect, and WhalePositioningCollector uses ws_hub."""
    import asyncio
    from intelligence.external.collectors.whale_positioning_collector import WhalePositioningCollector
    import providers.ws_stream as ws_mod

    # 1. Pong / control frames must NOT update last_message_ts
    hl_st = HyperliquidAssetCtxStream(["HYPE"])
    t0 = 1_760_030_000.0
    assert hl_st.ingest_raw_message({"channel": "pong"}, now_ts=t0) is False
    assert hl_st.last_message_ts() == 0.0
    assert hl_st.ingest_raw_message(
        {"channel": "activeAssetCtx", "data": {"coin": "HYPE", "ctx": {"funding": "0.00001", "openInterest": "2000", "markPx": "60.0", "oraclePx": "60.0"}}},
        now_ts=t0 + 1.0,
    ) is True
    assert hl_st.last_message_ts() == pytest.approx(t0 + 1.0)
    assert hl_st.ingest_raw_message({"channel": "pong"}, now_ts=t0 + 50.0) is False
    assert hl_st.last_message_ts() == pytest.approx(t0 + 1.0)

    # 2. Silent zombie socket breaks cleanly out of _connect_and_run once stale_reconnect_sec elapses
    class _SilentFakeWs:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_a):
            return False

        async def send(self, _msg):
            return None

        async def recv(self):
            await asyncio.sleep(0.05)
            raise asyncio.TimeoutError()

    class _FakeWsLib:
        class exceptions:
            ConnectionClosed = RuntimeError

        @staticmethod
        def connect(*_a, **_kw):
            return _SilentFakeWs()

    monkeypatch.setattr(ws_mod, "_websockets", _FakeWsLib)
    monkeypatch.setattr(ws_mod, "WS_RECV_TIMEOUT", 0.02)
    silent_st = BinanceFuturesMarkPriceStream(["BTCUSDT"], stale_reconnect_sec=0.08)
    t_start = time.monotonic()
    asyncio.run(silent_st._connect_and_run())
    elapsed = time.monotonic() - t_start
    assert 0.06 <= elapsed < 2.0

    # 3. WhalePositioningCollector uses HyperliquidAssetCtxStream from ws_hub with 0 REST metaAndAssetCtxs calls
    hub = TelemetryWsHub(["HYPEUSDC"])
    hub.hl_asset_ctx_stream.ingest_raw_message(
        {"channel": "activeAssetCtx", "data": {"coin": "HYPE", "ctx": {"funding": "0.000025", "openInterest": "100000.0", "markPx": "60.0", "oraclePx": "60.0"}}},
        now_ts=t0,
    )
    wh_col = WhalePositioningCollector(cache_dir=str(tmp_path), start_sync=True)
    wh_col.ws_hub = hub
    monkeypatch.setattr(
        wh_col,
        "_fetch_json_dict",
        lambda url, label, *_a, **_kw: {"result_list": [{"px": "60.0", "sz": "10.0", "side": "B", "time": int(t0 * 1000)}]} if label == "recentTrades" else {},
    )
    hl_rest_called = [False]
    monkeypatch.setattr(wh_col, "_fetch_hyperliquid_asset_ctx", lambda *_a, **_kw: hl_rest_called.__setitem__(0, True) or None)
    snap = wh_col._fetch_hyperliquid_whale_fallback("HYPEUSDC", 0.0, t0 + 2.0, None)
    assert snap is not None
    assert hl_rest_called[0] is False
    assert snap.open_interest_usd == pytest.approx(6_000_000.0)
    assert snap.funding_rate == pytest.approx(0.000025)


def test_public_price_stream_stale_reconnect_watchdog(monkeypatch):
    """Verify BinancePriceStream, HyperliquidPriceStream, and KrakenPriceStream expose silent-socket stale reconnect."""
    import asyncio
    from binance_api import bapi_ws
    from providers import price_poller as pp_mod

    bn = bapi_ws.BinancePriceStream(["BTCUSDC"])
    hl = pp_mod.HyperliquidPriceStream(["HYPEUSDC"])
    kr = pp_mod.KrakenPriceStream(["HYPEUSD"])
    assert bn._stale_reconnect_sec == 120.0
    assert hl._stale_reconnect_sec == 120.0
    assert kr._stale_reconnect_sec == 300.0

    # Silent recv loop exits quickly once stale_reconnect_sec elapses without price frames
    class _SilentWs:
        async def recv(self):
            await asyncio.sleep(5.0)
            return "{}"

    bn_fast = bapi_ws.BinancePriceStream(["BTCUSDC"], stale_reconnect_sec=0.06)
    monkeypatch.setattr(bapi_ws, "WS_RECV_TIMEOUT", 0.02)
    t0 = time.monotonic()
    asyncio.run(bn_fast._recv_loop(_SilentWs()))
    elapsed = time.monotonic() - t0
    assert 0.05 <= elapsed < 2.0

    # Receiving a price tick resets the silent-socket timer
    bn_fast._notify_subscribers("BTCUSDC", [[int(time.time() * 1000), 60000.0]])
    assert bn_fast._is_price_stream_stale(t0) is False


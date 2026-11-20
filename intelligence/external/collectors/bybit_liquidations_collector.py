"""Multi-venue forced liquidation WebSocket stream collectors (Bybit, Binance Futures, OKX).

Streams public USDT-perpetual forced liquidations in real time over zero-credential public endpoints
through a 3-tier polymorphic hierarchy:
- Level 1: ``TelemetryCacheCollectorInterface``
- Level 2: ``LiquidationCollectorBase``
- Level 3: ``BybitLiquidationCollector``, ``BinanceFuturesLiquidationCollector``, ``OkxLiquidationCollector``
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple, Type

import websockets

from botcore import (
    _parse_bool_flag,
    async_interruptible_sleep,
    first_pos_float,
    is_worker_thread_alive,
    normalize_epoch_sec,
    normalize_side,
    parse_symbol_list,
    resolve_now_ts,
    safe_float,
    safe_nonneg_float,
    safe_pos_float,
)
from cachepaths import cache_symbol_variants, canonical_base_asset, clean_symbol_key
from instrument_registry import casefold_get
from intelligence.external.collectors.base_collector import (
    TelemetryCacheCollectorInterface,
    TelemetryCollectorFactory,
)
from intelligence.macro_sentiment_ssot import (
    fetch_public_json,
    get_binance_futures_liquidations_ws_url,
    get_bybit_liquidations_ws_url,
    get_okx_public_ws_url,
    get_okx_swap_instruments_url,
)
from providers.ws_stream import run_ws_event_loop
from state_io import encode_json_payload, parse_json_payload

logger = logging.getLogger("intelligence.external.bybit_liquidations")

BYBIT_WS_URL = get_bybit_liquidations_ws_url()
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

MAX_RETENTION_SEC = 3600.0
WS_RECV_TIMEOUT = 5.0
WS_APP_PING_SEC = 20.0
WS_PING_INTERVAL = 20
WS_PING_TIMEOUT = 10
WS_CLOSE_TIMEOUT = 2
WS_RETRY_INITIAL = 1.0
WS_RETRY_MAX = 60.0
STABLE_SESSION_SEC = 60.0

LIQUIDATION_COLLECTOR_REGISTRY: Dict[str, Type["LiquidationCollectorBase"]] = {}


@dataclass(frozen=True)
class LiquidationSummary:
    """Aggregated liquidation metrics for a time window."""

    symbol: str
    window_sec: float
    long_liq_usd: float
    short_liq_usd: float
    net_usd: float
    capitulation_ratio: float
    squeeze_ratio: float
    events_count: int
    last_event_ts: float

    @property
    def total_usd(self) -> float:
        return self.long_liq_usd + self.short_liq_usd


def _normalize_liq_position_side(raw_side: Any) -> Optional[str]:
    """Normalize a liquidated position side into canonical ``LONG`` or ``SHORT``."""
    s_u = str(raw_side or "").strip().upper()
    if s_u in ("LONG", "SHORT"):
        return s_u
    norm_s = normalize_side(raw_side)
    if norm_s == "BUY":
        return "LONG"
    if norm_s == "SELL":
        return "SHORT"
    return None


class LiquidationCollectorBase(TelemetryCacheCollectorInterface, ABC):
    """Level 2 abstract base class for thread-safe rolling WebSocket forced-liquidation collectors."""

    cls_name = "LiquidationCollectorBase"
    VENUE_SOURCE: str = ""
    _THREAD_NAME: str = "LiquidationWS"

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        retention_sec: float = MAX_RETENTION_SEC,
        *,
        start_sync: bool = False,
    ) -> None:
        super().__init__(start_sync=start_sync)
        self._symbols = parse_symbol_list(symbols) or list(DEFAULT_SYMBOLS)
        self._tracked_bases = {
            b for s in self._symbols if (b := canonical_base_asset(s))
        }
        self._retention = safe_pos_float(retention_sec, default=MAX_RETENTION_SEC) or MAX_RETENTION_SEC
        # symbol -> deque[(ts, "LONG"|"SHORT", notional_usd)]
        self._events: Dict[str, Deque[Tuple[float, str, float]]] = {}
        self._lock = threading.Lock()
        self._start_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._retry = WS_RETRY_INITIAL

    @abstractmethod
    def _ws_url(self) -> str:
        """Return the public WebSocket endpoint URL for this venue."""

    def _subscribe_payload(self) -> Optional[Any]:
        """Return optional subscribe message (dict or str) sent immediately upon WS connect."""
        return None

    def _app_ping_payload(self) -> Optional[Any]:
        """Return optional application-level heartbeat payload (dict or str) sent every ``WS_APP_PING_SEC``."""
        return None

    def _before_session(self) -> None:
        """Optional hook executed before opening a WebSocket session (e.g., loading contract multipliers)."""

    @abstractmethod
    def _ingest(self, raw: str) -> None:
        """Parse one raw WebSocket frame and call ``record_event()`` for matching liquidation events."""

    def start(self, name: Optional[str] = None, daemon: bool = True) -> "LiquidationCollectorBase":
        daemon_b = _parse_bool_flag(daemon, True)
        thread_name = str(name).strip() if isinstance(name, str) and name.strip() else self._THREAD_NAME
        with self._start_lock:
            if self._thread and self._thread.is_alive():
                return self
            self._stop.clear()
            self._thread = threading.Thread(target=self._worker, name=thread_name, daemon=daemon_b)
            self._thread.start()
        logger.info("%s started for %s", self.cls_name, ",".join(self._symbols))
        return self

    def stop(self, timeout: float = 8.0) -> None:
        self._stop.set()
        t_sec = safe_nonneg_float(timeout, default=8.0)
        if (
            self._thread
            and self._thread.is_alive()
            and self._thread is not threading.current_thread()
        ):
            self._thread.join(timeout=t_sec)

    is_running = property(is_worker_thread_alive)

    def _worker(self) -> None:
        try:
            if getattr(asyncio.run, "__module__", "") == "asyncio.runners":
                run_ws_event_loop(self._run_with_reconnect())
            else:
                asyncio.run(self._run_with_reconnect())
        except Exception as e:
            logger.exception("%s worker crash: %s", self.cls_name, e)
        finally:
            logger.info("%s worker exited", self.cls_name)

    async def _run_with_reconnect(self) -> None:
        while not self._stop.is_set():
            t0 = time.time()
            try:
                await self._session()
            except Exception as e:
                logger.warning("%s WS error: %s", self.cls_name, e)
            if self._stop.is_set():
                break
            if time.time() - t0 >= STABLE_SESSION_SEC:
                self._retry = WS_RETRY_INITIAL
            logger.info("%s reconnecting in %.1fs", self.cls_name, self._retry)
            await self._sleep(self._retry)
            self._retry = min(self._retry * 2, WS_RETRY_MAX)

    async def _session(self) -> None:
        self._before_session()
        async with websockets.connect(
            self._ws_url(),
            ping_interval=WS_PING_INTERVAL,
            ping_timeout=WS_PING_TIMEOUT,
            close_timeout=WS_CLOSE_TIMEOUT,
        ) as ws:
            sub_msg = self._subscribe_payload()
            if sub_msg is not None:
                await ws.send(encode_json_payload(sub_msg) if isinstance(sub_msg, (dict, list)) else str(sub_msg))
            ping_msg = self._app_ping_payload()
            last_ping = time.time()
            while not self._stop.is_set():
                now = time.time()
                if now < last_ping:
                    last_ping = now
                if ping_msg is not None and now - last_ping >= WS_APP_PING_SEC:
                    await ws.send(encode_json_payload(ping_msg) if isinstance(ping_msg, (dict, list)) else str(ping_msg))
                    last_ping = now
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=WS_RECV_TIMEOUT)
                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed:
                    return
                self._ingest(raw)

    async def _sleep(self, delay: float, step: float = 0.2) -> None:
        await async_interruptible_sleep(self._stop, delay, step, sleep_fn=asyncio.sleep)

    def record_event(self, symbol: str, side: str, notional_usd: float, ts: Optional[float] = None) -> None:
        """Inject event directly (useful for tests and synthetic feeds)."""
        notional = safe_pos_float(notional_usd, default=None)
        if notional is None or not isinstance(symbol, str):
            return
        sym_norm = clean_symbol_key(symbol)
        if not sym_norm:
            return
        side_norm = _normalize_liq_position_side(side)
        if side_norm is None:
            return
        now = normalize_epoch_sec(ts) if ts is not None else time.time()
        if now is None:
            return
        with self._lock:
            dq = self._events.setdefault(sym_norm, deque())
            dq.append((now, side_norm, notional))
            cutoff = now - self._retention
            while dq and (safe_float(dq[0][0], default=None) is None or dq[0][0] < cutoff):
                dq.popleft()

    def _lookup_symbol_events(self, symbol: str) -> List[Tuple[float, str, float]]:
        symbol_str = str(symbol or "").strip().upper()
        sym_clean = clean_symbol_key(symbol_str)
        with self._lock:
            items = list(
                self._events.get(sym_clean)
                or casefold_get(self._events, sym_clean)
                or casefold_get(self._events, symbol_str)
                or ()
            )
            if not items and (sym_clean or symbol_str):
                for cand in cache_symbol_variants(symbol_str or sym_clean, include_original=True, quotes=("USDT", "USDC", "USD")):
                    cand_k = clean_symbol_key(cand)
                    cand_items = (
                        self._events.get(cand_k)
                        or casefold_get(self._events, cand_k)
                        or casefold_get(self._events, cand)
                    )
                    if cand_items:
                        items = list(cand_items)
                        break
        return items

    def _aggregate_window(
        self,
        items: List[Tuple[float, str, float]],
        cutoff: float,
        max_ts: Optional[float] = None,
    ) -> Tuple[float, float, int, int, float]:
        long_usd = short_usd = last_ts = 0.0
        long_n = short_n = 0
        for row in items:
            if not isinstance(row, (list, tuple)) or len(row) < 3:
                continue
            ts, side, notional = row[0], row[1], row[2]
            ts_f = normalize_epoch_sec(ts, allow_zero=True)
            if ts_f is None or ts_f < cutoff or (max_ts is not None and ts_f > max_ts):
                continue
            notional_f = safe_pos_float(notional, default=None)
            if notional_f is None:
                continue
            side_norm = _normalize_liq_position_side(side)
            if side_norm not in ("LONG", "SHORT"):
                continue
            last_ts = max(last_ts, ts_f)
            if side_norm == "LONG":
                long_usd += notional_f
                long_n += 1
            else:
                short_usd += notional_f
                short_n += 1
        return long_usd, short_usd, long_n, short_n, last_ts

    def get_summary(
        self,
        symbol: str,
        window_sec: float = 300.0,
        now: Optional[float] = None,
    ) -> LiquidationSummary:
        """Aggregate liquidation metrics for symbol over window_sec."""
        items = self._lookup_symbol_events(symbol)
        symbol = clean_symbol_key(symbol)
        now_ts = resolve_now_ts(now)
        win_sec = safe_pos_float(window_sec, default=300.0) or 300.0

        long_usd, short_usd, long_n, short_n, last_ts = self._aggregate_window(
            items, now_ts - win_sec, max_ts=now_ts + 5.0
        )
        total = long_usd + short_usd
        return LiquidationSummary(
            symbol=symbol,
            window_sec=win_sec,
            long_liq_usd=long_usd,
            short_liq_usd=short_usd,
            net_usd=short_usd - long_usd,
            capitulation_ratio=long_usd / total if total > 0 else 0.5,
            squeeze_ratio=short_usd / total if total > 0 else 0.5,
            events_count=long_n + short_n,
            last_event_ts=last_ts,
        )

    def get_signal(
        self,
        symbol: str,
        window_sec: float = 300.0,
        now: Optional[float] = None,
    ) -> Dict[str, float]:
        """Aggregate liquidations for ``symbol`` over the last ``window_sec`` as a dict."""
        symbol_str = str(symbol or "").strip().upper()
        eff_win = safe_pos_float(window_sec, 300.0)
        now_ts = resolve_now_ts(now)
        items = self._lookup_symbol_events(symbol_str)
        long_usd, short_usd, long_n, short_n, last_ts = self._aggregate_window(items, now_ts - eff_win, max_ts=now_ts + 5.0)
        return {
            "symbol": symbol_str,
            "window_sec": eff_win,
            "long_liq_usd": long_usd,
            "short_liq_usd": short_usd,
            "long_count": long_n,
            "short_count": short_n,
            "net_usd": short_usd - long_usd,
            "age_sec": max(0.0, now_ts - last_ts) if last_ts else math.inf,
        }

    def capitulation_ratio(
        self,
        symbol: str,
        window_sec: float = 120.0,
        baseline_sec: float = 3600.0,
        now: Optional[float] = None,
    ) -> float:
        """Intensity of recent LONG-liquidation burst relative to baseline rate."""
        eff_win = safe_pos_float(window_sec, 120.0)
        eff_base = max(safe_pos_float(baseline_sec, 3600.0), eff_win)
        now_ts = resolve_now_ts(now)
        recent = self.get_signal(symbol, eff_win, now=now_ts)["long_liq_usd"]
        base = self.get_signal(symbol, eff_base, now=now_ts)["long_liq_usd"]
        recent_rate = recent / max(eff_win, 1.0)
        base_rate = base / max(eff_base, 1.0)
        return recent_rate / base_rate if base_rate > 0 else 0.0

    def collect(
        self,
        symbol_or_cache_dir: str = "cachedb",
        symbol: Optional[str] = None,
        *,
        window_sec: float = 300.0,
        now: Optional[float] = None,
        **kwargs: Any,
    ) -> LiquidationSummary:
        """Polymorphic ``TelemetryCacheCollectorInterface`` entrypoint delegating to ``get_summary()``."""
        eff_sym = (
            str(symbol)
            if (symbol is not None and str(symbol).strip())
            else str(symbol_or_cache_dir)
        )
        eff_now = now if now is not None else kwargs.get("now_ts")
        return self.get_summary(eff_sym, window_sec=window_sec, now=eff_now)

    def read_snapshot(
        self,
        symbol_or_cache_dir: str = "cachedb",
        symbol: Optional[str] = None,
        *,
        window_sec: float = 300.0,
        now: Optional[float] = None,
        **kwargs: Any,
    ) -> LiquidationSummary:
        """Read-only consumer accessor returning aggregated liquidation summary from in-memory window."""
        return self.collect(symbol_or_cache_dir, symbol, window_sec=window_sec, now=now, **kwargs)


class BybitLiquidationCollector(LiquidationCollectorBase):
    """Level 3 collector tracking rolling forced liquidations from Bybit Linear WS."""

    cls_name = "BybitLiquidationCollector"
    COLLECTOR_ID = "external_bybit_liquidations"
    VENUE_SOURCE = "bybit_ws"
    _THREAD_NAME = "BybitLiquidationWS"

    def _ws_url(self) -> str:
        return BYBIT_WS_URL

    def _subscribe_payload(self) -> Optional[Any]:
        return {"op": "subscribe", "args": [f"allLiquidation.{s}" for s in self._symbols]}

    def _app_ping_payload(self) -> Optional[Any]:
        return {"op": "ping"}

    def _ingest(self, raw: str) -> None:
        msg = parse_json_payload(raw, default=None) if isinstance(raw, (str, bytes, bytearray)) else None
        if not isinstance(msg, dict):
            return
        topic = str(msg.get("topic") or "")
        if not topic.startswith("allLiquidation."):
            return
        now = time.time()
        data_items = msg.get("data")
        if not isinstance(data_items, list):
            return
        for item in data_items:
            if not isinstance(item, dict):
                continue
            v = safe_pos_float(item.get("v"), default=None)
            p = safe_pos_float(item.get("p"), default=None)
            if v is not None and p is not None:
                notional = safe_pos_float(v * p, default=None)
                if notional is not None:
                    self.record_event(item.get("s"), item.get("S"), notional, ts=now)


class BinanceFuturesLiquidationCollector(LiquidationCollectorBase):
    """Level 3 collector tracking rolling forced liquidations from Binance Futures ``!forceOrder@arr`` WS."""

    cls_name = "BinanceFuturesLiquidationCollector"
    COLLECTOR_ID = "external_binance_futures_liquidations"
    VENUE_SOURCE = "binance_futures_ws"
    _THREAD_NAME = "BinanceFuturesLiquidationWS"

    def _ws_url(self) -> str:
        return get_binance_futures_liquidations_ws_url()

    def _ingest(self, raw: str) -> None:
        parsed = parse_json_payload(raw, default=None) if isinstance(raw, (str, bytes, bytearray)) else None
        if isinstance(parsed, dict) and isinstance(parsed.get("data"), (dict, list)):
            parsed = parsed["data"]
        events = parsed if isinstance(parsed, list) else ([parsed] if isinstance(parsed, dict) else [])
        for msg in events:
            if not isinstance(msg, dict):
                continue
            if str(msg.get("e") or "") != "forceOrder":
                continue
            order = msg.get("o")
            if not isinstance(order, dict):
                continue
            raw_sym = str(order.get("s") or "").strip().upper()
            base = canonical_base_asset(raw_sym)
            if not base or (self._tracked_bases and base not in self._tracked_bases):
                continue
            # Binance Futures forceOrder reports the forced liquidation order side:
            # SELL order liquidates a LONG position; BUY order liquidates a SHORT position.
            ord_side = normalize_side(order.get("S"))
            liq_side = "LONG" if ord_side == "SELL" else ("SHORT" if ord_side == "BUY" else None)
            if liq_side is None:
                continue
            qty = first_pos_float(order.get("z"), order.get("l"), order.get("q"))
            px = first_pos_float(order.get("ap"), order.get("p"))
            if qty is None or px is None:
                continue
            notional = safe_pos_float(qty * px, default=None)
            if notional is None:
                continue
            ev_ts = normalize_epoch_sec(order.get("T") or msg.get("E")) or time.time()
            self.record_event(f"{base}USDT", liq_side, notional, ts=ev_ts)


class OkxLiquidationCollector(LiquidationCollectorBase):
    """Level 3 collector tracking rolling forced liquidations from OKX public SWAP WS."""

    cls_name = "OkxLiquidationCollector"
    COLLECTOR_ID = "external_okx_liquidations"
    VENUE_SOURCE = "okx_ws"
    _THREAD_NAME = "OkxLiquidationWS"

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        retention_sec: float = MAX_RETENTION_SEC,
        *,
        start_sync: bool = False,
        ct_val_by_inst: Optional[Dict[str, float]] = None,
    ) -> None:
        super().__init__(symbols=symbols, retention_sec=retention_sec, start_sync=start_sync)
        self._ct_val_by_inst: Dict[str, float] = {
            str(k).strip().upper(): v
            for k, raw_v in (ct_val_by_inst or {}).items()
            if (v := safe_pos_float(raw_v, default=None)) is not None
        }

    def _ws_url(self) -> str:
        return get_okx_public_ws_url()

    def _subscribe_payload(self) -> Optional[Any]:
        return {"op": "subscribe", "args": [{"channel": "liquidation-orders", "instType": "SWAP"}]}

    def _app_ping_payload(self) -> Optional[Any]:
        return "ping"

    def _before_session(self) -> None:
        if self._ct_val_by_inst:
            return
        try:
            payload = fetch_public_json(
                get_okx_swap_instruments_url(),
                timeout=4.0,
                user_agent="mptrade-liquidations/1.0",
            )
            rows = payload.get("data") if isinstance(payload, dict) else None
            if not isinstance(rows, list):
                return
            loaded: Dict[str, float] = {}
            for row in rows:
                if not isinstance(row, dict):
                    continue
                inst_id = str(row.get("instId") or "").strip().upper()
                if not inst_id.endswith("-USDT-SWAP"):
                    continue
                ct_val = safe_pos_float(row.get("ctVal"), default=None)
                if ct_val is not None:
                    loaded[inst_id] = ct_val
            if loaded:
                self._ct_val_by_inst = loaded
        except Exception as exc:
            logger.debug("Could not load OKX SWAP contract multipliers: %s", exc)

    def _ingest(self, raw: str) -> None:
        if isinstance(raw, str) and raw.strip().lower() == "pong":
            return
        msg = parse_json_payload(raw, default=None) if isinstance(raw, (str, bytes, bytearray)) else None
        if not isinstance(msg, dict):
            return
        arg = msg.get("arg")
        if not isinstance(arg, dict) or str(arg.get("channel") or "") != "liquidation-orders":
            return
        data_items = msg.get("data")
        if not isinstance(data_items, list):
            return
        for item in data_items:
            if not isinstance(item, dict):
                continue
            inst_id = str(item.get("instId") or "").strip().upper()
            if not inst_id.endswith("-USDT-SWAP"):
                continue
            base = inst_id.split("-", 1)[0].strip().upper()
            if not base or (self._tracked_bases and base not in self._tracked_bases):
                continue
            ct_val = safe_pos_float(self._ct_val_by_inst.get(inst_id), default=None)
            if ct_val is None:
                continue
            details = item.get("details")
            if not isinstance(details, list):
                continue
            for det in details:
                if not isinstance(det, dict):
                    continue
                sz = safe_pos_float(det.get("sz"), default=None)
                bk_px = safe_pos_float(det.get("bkPx"), default=None)
                if sz is None or bk_px is None:
                    continue
                pos_side = str(det.get("posSide") or "").strip().lower()
                ord_side = normalize_side(det.get("side"))
                if pos_side == "long":
                    liq_side = "LONG"
                elif pos_side == "short":
                    liq_side = "SHORT"
                elif ord_side == "SELL":
                    liq_side = "LONG"
                elif ord_side == "BUY":
                    liq_side = "SHORT"
                else:
                    continue
                notional = safe_pos_float(sz * ct_val * bk_px, default=None)
                if notional is None:
                    continue
                ev_ts = normalize_epoch_sec(det.get("ts")) or time.time()
                self.record_event(f"{base}USDT", liq_side, notional, ts=ev_ts)


for _liq_cls in (
    BybitLiquidationCollector,
    BinanceFuturesLiquidationCollector,
    OkxLiquidationCollector,
):
    TelemetryCollectorFactory.register(_liq_cls.COLLECTOR_ID, _liq_cls)
    LIQUIDATION_COLLECTOR_REGISTRY[_liq_cls.VENUE_SOURCE] = _liq_cls

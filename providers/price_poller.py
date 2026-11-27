"""Public venue price WebSocket streams and bounded-wait fetch helper."""

from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
import builtins
import concurrent.futures as _futures
from dataclasses import dataclass
import time
from typing import Any, Callable, Optional
import websockets as _websockets

from botcore import (
    PRICE_FALLBACK_HTTP_DEADLINE_SEC,
    _parse_bool_flag,
    clean_scalar_str,
    clean_upper_scalar,
    resolve_now_ms,
    safe_pos_float,
    safe_pos_int,
)
from cachepaths import (
    clean_symbol_key,
    normalize_venue_name,
)
from state_io import parse_json_payload
from .base import MarketDataProvider
from .ssot_bridge import live_price_fetch_context
from .ws_stream import (
    PublicPriceStreamBase,
    WS_CLOSE_TIMEOUT,
    WS_PING_INTERVAL,
    WS_PING_TIMEOUT,
    WS_RECV_TIMEOUT,
)

__all__ = [
    "HyperliquidPriceStream",
    "KrakenPriceStream",
    "VenuePriceStreamBase",
    "VenuePriceStreamSpec",
    "_VenuePriceStreamBase",
    "_fetch_price_with_deadline",
    "fetch_price_with_deadline",
]


@dataclass(frozen=True)
class VenuePriceStreamSpec:
    """Declarative registration spec for one venue's public WebSocket price stream."""

    venue: str
    ws_thread_name: str
    builder: Callable[..., Any]
    owned: bool = True
    default_ws_getter: Optional[Callable[[], Any]] = None
    request_stop_hook: Optional[Callable[[], None]] = None
    stop_hook: Optional[Callable[[float], None]] = None


class VenuePriceStreamBase(PublicPriceStreamBase, ABC):
    """Shared public WebSocket price stream base for multi-venue streams."""

    MIN_SAME_PRICE_EMIT_MS = 1000
    _stream_init_hooks: list[Callable[..., None]] = []
    _VENUE_STREAM_SPECS: dict[str, VenuePriceStreamSpec] = {}

    @classmethod
    def register_venue_stream(
        cls,
        venue: str,
        ws_thread_name: str,
        builder: Callable[..., Any],
        *,
        owned: bool = True,
        default_ws_getter: Optional[Callable[[], Any]] = None,
        request_stop_hook: Optional[Callable[[], None]] = None,
        stop_hook: Optional[Callable[[float], None]] = None,
    ) -> None:
        """Declaratively register a public WebSocket stream builder and lifecycle hooks for ``venue``."""
        v_key = normalize_venue_name(venue, default="")
        if v_key:
            cls._VENUE_STREAM_SPECS[v_key] = VenuePriceStreamSpec(
                venue=v_key,
                ws_thread_name=str(ws_thread_name),
                builder=builder,
                owned=_parse_bool_flag(owned, default=True),
                default_ws_getter=default_ws_getter,
                request_stop_hook=request_stop_hook,
                stop_hook=stop_hook,
            )

    @classmethod
    def get_registered_venue_stream_spec(
        cls, venue: object
    ) -> Optional[VenuePriceStreamSpec]:
        """Return the registered ``VenuePriceStreamSpec`` for ``venue``, or ``None``."""
        v_key = normalize_venue_name(venue, default="")
        return cls._VENUE_STREAM_SPECS.get(v_key) if v_key else None

    @classmethod
    def iter_registered_venue_stream_specs(cls) -> tuple[VenuePriceStreamSpec, ...]:
        """Return all registered ``VenuePriceStreamSpec`` entries in registration order."""
        return tuple(cls._VENUE_STREAM_SPECS.values())

    @classmethod
    def iter_registered_venue_streams(
        cls,
    ) -> tuple[tuple[str, str, Callable[..., Any]], ...]:
        """Return ``(venue_key, ws_thread_name, builder)`` for all registered venue price streams."""
        return tuple(
            (spec.venue, spec.ws_thread_name, spec.builder)
            for spec in cls._VENUE_STREAM_SPECS.values()
        )

    _VENUE_NAME: str = ""

    def __init__(
        self,
        symbols=None,
        *,
        ws_url: Optional[str] = None,
        stale_reconnect_sec: float = 0.0,
    ):
        prov_cls = (
            MarketDataProvider.resolve_venue_provider_class(self._VENUE_NAME)
            if self._VENUE_NAME
            else None
        )
        super().__init__(symbols=symbols, stale_reconnect_sec=stale_reconnect_sec)
        self._prov_cls = prov_cls
        self.ws_url = clean_scalar_str(ws_url) or getattr(prov_cls, "PUBLIC_WS_URL", "")
        self._last_emit_ts_ms: dict[str, int] = {}
        self._last_emit_price: dict[str, float] = {}
        self._extra_subscribe_builders: list[Callable[[], tuple[str, ...]]] = []
        self._channel_handlers: dict[str, Callable[[object, int], None]] = {}
        self._tick_listeners: list[Callable[[int], None]] = []

    @classmethod
    def register_stream_init_hook(cls, hook: Callable[..., None]) -> None:
        if "_stream_init_hooks" not in cls.__dict__:
            cls._stream_init_hooks = []
        if hook not in cls._stream_init_hooks:
            cls._stream_init_hooks.append(hook)

    def register_channel_handler(
        self,
        channel: str,
        handler: Callable[[object, int], None],
        *,
        subscribe_builder: Optional[Callable[[], tuple[str, ...]]] = None,
        on_tick: Optional[Callable[[int], None]] = None,
    ) -> None:
        self._channel_handlers[str(channel)] = handler
        if callable(subscribe_builder):
            self._extra_subscribe_builders.append(subscribe_builder)
        if callable(on_tick):
            self._tick_listeners.append(on_tick)

    def remove_symbol(self, symbol: str) -> bool:
        norm_s = self._norm_ws_symbol(symbol)
        if not super().remove_symbol(symbol):
            return False
        if norm_s:
            with self._lock:
                self._last_emit_ts_ms.pop(norm_s, None)
                self._last_emit_price.pop(norm_s, None)
        return True

    def _notify_subscribers(
        self, symbol: str, items, *, update_staleness: bool = True
    ) -> None:
        ts_ms = None
        px = None
        if (
            isinstance(items, list)
            and items
            and isinstance(items[-1], (list, tuple))
            and len(items[-1]) >= 2
        ):
            ts_ms = safe_pos_int(items[-1][0], default=None)
            px = safe_pos_float(items[-1][1], default=None)
        now_m = time.monotonic()
        throttled = False
        with self._lock:
            if update_staleness:
                self._last_price_mono = now_m
                self._last_price_mono_by_symbol[symbol] = now_m
            subscribers = list(self._subscribers)
            if ts_ms is not None and px is not None:
                prev_ts = self._last_emit_ts_ms.get(symbol)
                prev_px = self._last_emit_price.get(symbol)
                if (
                    prev_ts is not None
                    and prev_px == px
                    and 0 <= (ts_ms - prev_ts) < self.MIN_SAME_PRICE_EMIT_MS
                ):
                    throttled = True
                else:
                    self._last_emit_ts_ms[symbol] = ts_ms
                    self._last_emit_price[symbol] = px
        if throttled:
            now_s = time.time()
            for sub in subscribers:
                ws_map = getattr(sub, "_ws_stream_event_ts_per_symbol", None)
                if isinstance(ws_map, dict):
                    ws_map[symbol] = now_s
                mark_fn = getattr(sub, "mark_symbol_ws_event", None)
                if callable(mark_fn):
                    mark_fn(symbol, now_s)
                else:
                    last_map = getattr(sub, "_ws_last_event_ts_per_symbol", None)
                    if isinstance(last_map, dict):
                        last_map[symbol] = now_s
            return
        for sub in subscribers:
            try:
                sub.on_items_update(symbol, items)
            except Exception as exc:
                builtins.print(
                    f"[{self.__class__.__name__}] subscriber error ({symbol}): {exc}"
                )

    def _has_active_ws_targets(self) -> bool:
        """Return True when this stream has at least one WebSocket-subscribable target (caller holds ``self._lock``)."""
        return bool(self._symbols)

    async def _connect_and_run(self) -> None:
        while not self._stop_event.is_set():
            with self._lock:
                has_targets = self._has_active_ws_targets()
            if has_targets:
                break
            await self._interruptible_sleep(1.0)
        with self._lock:
            has_targets = self._has_active_ws_targets()
        if not has_targets or self._stop_event.is_set():
            return
        async with _websockets.connect(
            self.ws_url,
            ping_interval=WS_PING_INTERVAL,
            ping_timeout=WS_PING_TIMEOUT,
            close_timeout=WS_CLOSE_TIMEOUT,
        ) as ws:
            sub_msg = self._build_subscribe_payload()
            if sub_msg:
                await ws.send(sub_msg)
            for extra_msg in self._build_extra_subscribe_payloads():
                if extra_msg:
                    await ws.send(extra_msg)
            session_start = time.monotonic()
            while not self._stop_event.is_set():
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=WS_RECV_TIMEOUT)
                except asyncio.TimeoutError:
                    raw = None
                except _websockets.exceptions.ConnectionClosed:
                    break
                except Exception as exc:
                    builtins.print(
                        f"[{self.__class__.__name__}] recv error ({exc}); reconnecting..."
                    )
                    break
                if raw is not None:
                    try:
                        self._process_message(raw)
                    except Exception:
                        pass
                if self._check_data_staleness_or_reconnect(session_start):
                    break

    @abstractmethod
    def _build_subscribe_payload(self) -> Optional[str]:
        """Build the venue-specific WebSocket subscription message."""

    def _build_extra_subscribe_payloads(self) -> tuple[str, ...]:
        payloads: list[str] = []
        for builder in self._extra_subscribe_builders:
            built = builder()
            if built:
                payloads.extend(p for p in built if p)
        return tuple(payloads)

    @abstractmethod
    def _process_message(self, raw: str, *, now_ms: Optional[int] = None) -> None:
        """Process one incoming venue WebSocket message."""


_VenuePriceStreamBase = VenuePriceStreamBase


class HyperliquidPriceStream(VenuePriceStreamBase):
    """Real-time Hyperliquid WebSocket price stream delegating protocol logic to ``HyperliquidProvider``."""

    _VENUE_NAME = "hyperliquid"
    _SPOT_KEY_RETRY_SEC = 60.0

    def __init__(
        self,
        symbols=None,
        *,
        market_api=None,
        ws_url: Optional[str] = None,
        stale_reconnect_sec: float = 120.0,
        **extension_kwargs,
    ):
        super().__init__(symbols=symbols, ws_url=ws_url, stale_reconnect_sec=stale_reconnect_sec)
        self.market_api = market_api
        self._resolved_spot_keys: dict[str, tuple[str, ...]] = {}
        self._resolved_spot_retry_mono: dict[str, float] = {}
        for hook in getattr(self, "_stream_init_hooks", ()):
            hook(self, **extension_kwargs)

    def remove_symbol(self, symbol: str) -> bool:
        norm_s = self._norm_ws_symbol(symbol)
        removed = super().remove_symbol(symbol)
        if removed and norm_s:
            with self._lock:
                self._resolved_spot_keys.pop(norm_s, None)
                self._resolved_spot_retry_mono.pop(norm_s, None)
        return removed

    def _candidate_mids_keys(self, symbol: str) -> tuple[str, ...]:
        now_m = time.monotonic()
        with self._lock:
            cached = self._resolved_spot_keys.get(symbol)
            retry_at = self._resolved_spot_retry_mono.get(symbol, 0.0)
        if cached and (retry_at <= 0.0 or now_m < retry_at):
            return cached
        prov_cls = self._prov_cls
        resolved = (
            prov_cls.resolve_ws_spot_keys(symbol, market_api=self.market_api)
            if prov_cls is not None
            else (symbol,)
        )
        has_explicit_spot_idx = any(
            isinstance(k, str) and k.startswith("@") for k in resolved
        )
        with self._lock:
            self._resolved_spot_keys[symbol] = resolved
            if has_explicit_spot_idx:
                self._resolved_spot_retry_mono.pop(symbol, None)
            else:
                self._resolved_spot_retry_mono[symbol] = now_m + self._SPOT_KEY_RETRY_SEC
        return resolved

    def _build_subscribe_payload(self) -> str:
        prov_cls = self._prov_cls
        return prov_cls.build_ws_all_mids_subscription() if prov_cls is not None else ""

    def _process_message(self, raw: str, *, now_ms: Optional[int] = None) -> None:
        try:
            envelope = parse_json_payload(raw) if isinstance(raw, str) else raw
            if not isinstance(envelope, dict):
                return
            ts_ms = resolve_now_ms(now_ms)
            channel = envelope.get("channel")
            handler = self._channel_handlers.get(str(channel or ""))
            if handler is not None:
                handler(envelope.get("data"), ts_ms=ts_ms)
                return
            if (prov_cls := self._prov_cls) is None:
                return
            with self._lock:
                symbols = self._symbols
            prices = prov_cls.extract_ws_all_mids_prices(
                envelope, symbols, candidate_mids_fn=self._candidate_mids_keys
            )
            if prices is None:
                return
            for s_sym, px in prices.items():
                with self._lock:
                    self._prices[s_sym] = px
                self._notify_subscribers(s_sym, [[ts_ms, px]])
            for tick_fn in self._tick_listeners:
                tick_fn(ts_ms)
        except Exception:
            return


class KrakenPriceStream(VenuePriceStreamBase):
    """Real-time Kraken public WebSocket stream delegating protocol logic to ``KrakenProvider``."""

    _VENUE_NAME = "kraken"

    def __init__(
        self,
        symbols=None,
        *,
        ws_url: Optional[str] = None,
        stale_reconnect_sec: float = 300.0,
    ):
        super().__init__(symbols=symbols, ws_url=ws_url, stale_reconnect_sec=stale_reconnect_sec)
        self._ws_pairs: list[str] = []
        self._pair_to_symbol: dict[str, str] = {}
        self._last_real_tick_mono_by_symbol: dict[str, float] = {}
        self._keepalive_max_age_sec: float = (
            min(60.0, self._stale_reconnect_sec * 0.5)
            if self._stale_reconnect_sec > 0.0
            else 60.0
        )
        for s_sym in self._symbols:
            self._register_symbol_pair(s_sym)

    def _has_active_ws_targets(self) -> bool:
        return bool(self._ws_pairs)

    def _active_staleness_symbols(self) -> set[str]:
        """Only enforce WS staleness on pairs actually subscribed to Kraken's public ticker channel."""
        return set(self._pair_to_symbol.values())

    def _register_symbol_pair(self, s_sym: str) -> None:
        prov_cls = self._prov_cls
        if prov_cls is not None:
            prov_cls.register_ws_ticker_symbol_pair(
                s_sym, self._ws_pairs, self._pair_to_symbol
            )

    def add_symbol(self, symbol: str) -> bool:
        norm_s = self._norm_ws_symbol(symbol)
        if not super().add_symbol(symbol) or not norm_s:
            return False
        with self._lock:
            self._register_symbol_pair(norm_s)
        return True

    def remove_symbol(self, symbol: str) -> bool:
        norm_s = self._norm_ws_symbol(symbol)
        if not super().remove_symbol(symbol):
            return False
        with self._lock:
            if norm_s:
                self._last_real_tick_mono_by_symbol.pop(norm_s, None)
            self._ws_pairs.clear()
            self._pair_to_symbol.clear()
            for s_sym in self._symbols:
                self._register_symbol_pair(s_sym)
        return True

    def _build_subscribe_payload(self) -> str:
        prov_cls = self._prov_cls
        if prov_cls is None:
            return ""
        with self._lock:
            pairs = list(self._ws_pairs)
        return prov_cls.build_ws_ticker_subscription(pairs)

    def _resolve_symbol(self, pair_label: str) -> Optional[str]:
        if (raw_u := clean_upper_scalar(pair_label)) is None:
            return None
        with self._lock:
            return self._pair_to_symbol.get(raw_u) or self._pair_to_symbol.get(
                clean_symbol_key(raw_u)
            )

    def _process_message(self, raw: str, *, now_ms: Optional[int] = None) -> None:
        try:
            if (prov_cls := self._prov_cls) is None:
                return
            msg = parse_json_payload(raw) if isinstance(raw, str) else raw
            ts_ms = resolve_now_ms(now_ms)
            kind, s_sym, price = prov_cls.parse_ws_public_ticker_message(
                msg, resolve_symbol_fn=self._resolve_symbol
            )
            if kind == "keepalive":
                now_m = time.monotonic()
                with self._lock:
                    known_prices = [
                        (sym_key, px)
                        for sym_key, px in self._prices.items()
                        if (last_m := self._last_real_tick_mono_by_symbol.get(sym_key)) is not None
                        and (now_m - last_m) <= self._keepalive_max_age_sec
                    ]
                for sym_key, px in known_prices:
                    self._notify_subscribers(sym_key, [[ts_ms, px]], update_staleness=False)
                return
            if kind == "tick" and s_sym and price is not None:
                now_m = time.monotonic()
                with self._lock:
                    self._prices[s_sym] = price
                    self._last_real_tick_mono_by_symbol[s_sym] = now_m
                self._notify_subscribers(s_sym, [[ts_ms, price]])
        except Exception:
            return


VenuePriceStreamBase.register_venue_stream(
    "hyperliquid",
    "HyperliquidPriceWS",
    lambda syms, cpm=None: HyperliquidPriceStream(
        syms, market_api=getattr(cpm, "market_api", None)
    ),
)
VenuePriceStreamBase.register_venue_stream(
    "kraken",
    "KrakenPriceWS",
    lambda syms, cpm=None: KrakenPriceStream(syms),
)


def fetch_price_with_deadline(
    market_api,
    symbol,
    pool,
    deadline_sec=PRICE_FALLBACK_HTTP_DEADLINE_SEC,
    provider_name=None,
    prefer_public_extended=False,
):
    """Bound the poller's wait while the provider owns the network-call timeout."""
    kwargs = {"symbol": symbol}
    if provider_name is not None and not isinstance(provider_name, bool):
        if isinstance(provider_name, str):
            if (pn := clean_scalar_str(provider_name)) is not None:
                kwargs["provider_name"] = pn
        else:
            kwargs["provider_name"] = provider_name
    use_ext = _parse_bool_flag(prefer_public_extended, False)

    def _poll():
        with live_price_fetch_context():
            if use_ext:
                try:
                    return market_api.get_current_price(**kwargs, prefer_public_extended=True)
                except TypeError:
                    pass
            return market_api.get_current_price(**kwargs)

    fut = pool.submit(_poll)
    eff_deadline = (
        safe_pos_float(deadline_sec, default=PRICE_FALLBACK_HTTP_DEADLINE_SEC)
        or PRICE_FALLBACK_HTTP_DEADLINE_SEC
    )
    try:
        return fut.result(timeout=eff_deadline)
    except _futures.TimeoutError:
        fut.cancel()
        raise


_fetch_price_with_deadline = fetch_price_with_deadline

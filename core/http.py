"""core/http.py -- Canonical HTTP transport, requests gateway, and Ed25519 signing helpers."""
from __future__ import annotations

import base64
import concurrent.futures as _futures
from dataclasses import dataclass
import importlib
import re
import sys
import threading
from typing import Any, Callable, Optional, Sequence
import urllib.error
import urllib.parse
import urllib.request

from cachepaths import exact_int
from core.lock import log as _default_log
from core.math_coercion import _UNSET, require_pos_float
import state_io as _state_io

HTTP_TIMEOUT = 25
ACCOUNT_REST_HARD_TIMEOUT_SEC: float = round(HTTP_TIMEOUT * 2.0, 3)

_RATE_LIMIT_ERROR_TOKENS: tuple[str, ...] = (
    "429",
    "-1003",
    "rate limit",
    "too many requests",
    "too much request weight",
    "temporary lockout",
)


def is_rate_limit_error(exc_or_msg: object) -> bool:
    """Return True when ``exc_or_msg`` indicates an exchange/HTTP rate-limit or lockout error."""
    txt = str(exc_or_msg or "").lower()
    return any(tok in txt for tok in _RATE_LIMIT_ERROR_TOKENS)


def compute_rest_error_backoff_sec(
    exc_or_msg: object,
    streak: int,
    *,
    base_sec: float,
    max_sec: float,
    rate_limit_base_sec: float = 15.0,
    rate_limit_max_sec: float = 120.0,
    factor: float = 2.0,
    rate_limit_factor: float = 2.0,
    max_exp: int = 5,
    rate_limit_max_exp: int = 3,
) -> float:
    """Compute bounded exponential backoff (seconds) for a failed REST sync attempt."""
    eff_streak = max(1, exact_int(streak, default=1) or 1)
    if is_rate_limit_error(exc_or_msg):
        return min(rate_limit_max_sec, rate_limit_base_sec * (rate_limit_factor ** min(eff_streak - 1, rate_limit_max_exp)))
    return min(max_sec, base_sec * (factor ** min(eff_streak - 1, max_exp)))


def shutdown_daemon_executor(pool: Optional[_futures.ThreadPoolExecutor]) -> None:
    """Shut down a daemon ``ThreadPoolExecutor`` without blocking interpreter exit."""
    if pool is None:
        return
    pool.shutdown(wait=False, cancel_futures=True)
    try:
        for th in list(getattr(pool, "_threads", ())):
            _futures.thread._threads_queues.pop(th, None)
    except Exception:
        pass


@dataclass(slots=True)
class InFlightRestSlot:
    """Single per-key in-flight REST request tracked without blocking poll loops."""

    future: _futures.Future
    submit_mono: float
    submit_wall_ts: float
    meta: Any = None


class NonBlockingRestSlotTracker:
    """Thread-safe SSOT tracker for non-blocking per-key REST request slots.

    Guarantees at most one active REST request per key (symbol or venue) without blocking
    poll loops on a short deadline: loops revisit slots non-blockingly, harvest completed
    results or real socket errors, and evict/reset genuinely wedged requests only after
    ``hard_timeout_sec``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._slots: dict[str, InFlightRestSlot] = {}
        self._pool_lock = threading.Lock()
        self._pool: Optional[_futures.ThreadPoolExecutor] = None

    @property
    def slots(self) -> dict[str, InFlightRestSlot]:
        return self._slots

    @property
    def pool(self) -> Optional[_futures.ThreadPoolExecutor]:
        return self._pool

    def ensure_pool(
        self, *, max_workers: int, thread_name_prefix: str
    ) -> _futures.ThreadPoolExecutor:
        """Lazily create and return an owned daemon ``ThreadPoolExecutor``."""
        with self._pool_lock:
            if self._pool is None:
                self._pool = _futures.ThreadPoolExecutor(
                    max_workers=max(1, max_workers),
                    thread_name_prefix=thread_name_prefix,
                )
            return self._pool

    def shutdown_pool(self) -> None:
        """Clear active slots and tear down the owned daemon pool without blocking exit."""
        self.clear()
        with self._pool_lock:
            pool = self._pool
            self._pool = None
        shutdown_daemon_executor(pool)

    def submit_batch_and_wait(
        self,
        items: Sequence[tuple[str, Callable[[], Any], Any]],
        *,
        pool: _futures.ThreadPoolExecutor,
        now_mono: float,
        now_wall_ts: float,
        micro_wait_sec: float,
        on_done: Optional[Callable[[_futures.Future], None]] = None,
        return_when: str = _futures.ALL_COMPLETED,
    ) -> list[str]:
        """Submit ``(key, fn, meta)`` entries to ``pool``, wait up to ``micro_wait_sec``, and return completed keys."""
        submitted_futs: dict[_futures.Future, str] = {}
        for key, fn, meta in items:
            slot = self.submit(
                key,
                pool,
                fn,
                now_mono=now_mono,
                now_wall_ts=now_wall_ts,
                meta=meta,
                on_done=on_done,
            )
            if slot is not None:
                submitted_futs[slot.future] = key
        if not submitted_futs:
            return []
        done_now, _ = _futures.wait(
            submitted_futs.keys(),
            timeout=max(0.0, micro_wait_sec),
            return_when=return_when,
        )
        return [submitted_futs[f] for f in done_now]

    def active_symbols(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._slots.keys())

    def has_active(self) -> bool:
        with self._lock:
            return bool(self._slots)

    def has_done(self) -> bool:
        with self._lock:
            return any(slot.future.done() for slot in self._slots.values())

    def next_hard_timeout_remaining(
        self, *, now_mono: float, hard_timeout_sec: float
    ) -> Optional[float]:
        with self._lock:
            if not self._slots:
                return None
            return min(
                max(0.0, (slot.submit_mono + hard_timeout_sec) - now_mono)
                for slot in self._slots.values()
            )

    def inspect_and_harvest(
        self,
        key: str,
        *,
        now_mono: float,
        hard_timeout_sec: float,
    ) -> tuple[str, Any, Optional[InFlightRestSlot]]:
        """Inspect ``key``'s slot without blocking.

        Returns ``(state, payload_or_exc, slot)`` where ``state`` is one of:
        ``IDLE``, ``IN_FLIGHT``, ``HARD_TIMEOUT``, ``DONE_OK``, ``DONE_ERR``, ``DONE_CANCELLED``.
        """
        with self._lock:
            slot = self._slots.get(key)
            if slot is None:
                return "IDLE", None, None
            if not slot.future.done():
                if (now_mono - slot.submit_mono) >= hard_timeout_sec:
                    self._slots.pop(key, None)
                    slot.future.cancel()
                    return "HARD_TIMEOUT", None, slot
                return "IN_FLIGHT", None, slot
            self._slots.pop(key, None)
        if slot.future.cancelled():
            return "DONE_CANCELLED", None, slot
        try:
            return "DONE_OK", slot.future.result(), slot
        except Exception as exc:  # noqa: BLE001
            return "DONE_ERR", exc, slot

    def submit(
        self,
        key: str,
        pool: _futures.ThreadPoolExecutor,
        fn: Callable[[], Any],
        *,
        now_mono: float,
        now_wall_ts: float,
        meta: Any = None,
        on_done: Optional[Callable[[_futures.Future], None]] = None,
    ) -> Optional[InFlightRestSlot]:
        """Submit ``fn`` to ``pool`` for ``key`` if idle; returns the new slot or ``None`` if active."""
        with self._lock:
            existing = self._slots.get(key)
            if existing is not None and not existing.future.done():
                return None
            try:
                fut = pool.submit(fn)
            except RuntimeError:
                return None
            slot = InFlightRestSlot(
                future=fut,
                submit_mono=now_mono,
                submit_wall_ts=now_wall_ts,
                meta=meta,
            )
            self._slots[key] = slot
        if callable(on_done):
            fut.add_done_callback(on_done)
        return slot

    def submit_thread(
        self,
        key: str,
        fn: Callable[[], Any],
        *,
        thread_name: str,
        now_mono: float,
        now_wall_ts: float,
        thread_cls: Any = threading.Thread,
        meta: Any = None,
        on_done: Optional[Callable[[_futures.Future], None]] = None,
    ) -> Optional[InFlightRestSlot]:
        """Run ``fn`` on a single daemon thread tracked as an ``InFlightRestSlot`` for ``key``."""
        fut: _futures.Future = _futures.Future()
        slot = InFlightRestSlot(
            future=fut,
            submit_mono=now_mono,
            submit_wall_ts=now_wall_ts,
            meta=meta,
        )
        with self._lock:
            existing = self._slots.get(key)
            if existing is not None and not existing.future.done():
                return None
            self._slots[key] = slot

        def _run() -> None:
            try:
                res = fn()
            except BaseException as exc:  # noqa: BLE001
                if not fut.done():
                    fut.set_exception(exc)
            else:
                if not fut.done():
                    fut.set_result(res)

        try:
            thread_cls(target=_run, name=thread_name, daemon=True).start()
        except (RuntimeError, OSError):
            with self._lock:
                if self._slots.get(key) is slot:
                    self._slots.pop(key, None)
            fut.cancel()
            return None
        if callable(on_done):
            fut.add_done_callback(on_done)
        return slot

    def clear(self) -> None:
        with self._lock:
            for slot in self._slots.values():
                slot.future.cancel()
            self._slots.clear()


def _eff_log(msg: str) -> None:
    bc = sys.modules.get("botcore")
    fn = bc.__dict__.get("log") if bc is not None and "log" in bc.__dict__ else _default_log
    if callable(fn):
        fn(msg)


def http_request(
    method: str,
    url: str,
    headers: dict | None = None,
    payload: dict | None = None,
    *,
    form: dict | None = None,
    body: bytes | None = None,
    timeout: float | None = None,
) -> tuple[int, bytes]:
    """Shared standard-library HTTP transport with a ``(status, body)`` contract."""
    if sum(x is not None for x in (payload, form, body)) > 1:
        raise ValueError("payload, form, and body are mutually exclusive")
    timeout_f = HTTP_TIMEOUT if timeout is None else require_pos_float(timeout, name="timeout")

    request_headers = dict(headers) if isinstance(headers, dict) else {}
    data = None
    if payload is not None:
        data = _state_io.encode_json_payload(payload, ensure_ascii=False).encode("utf-8")
        request_headers.setdefault("Content-Type", "application/json")
    elif form is not None:
        data = urllib.parse.urlencode(form).encode()
        request_headers.setdefault("Content-Type", "application/x-www-form-urlencoded")
    elif body is not None:
        data = bytes(body)

    if not method or isinstance(method, bool) or not isinstance(method, str) or not method.strip():
        return 0, b""
    if not url or isinstance(url, bool) or not isinstance(url, str) or not url.strip():
        return 0, b""
    verb = method.strip().upper()
    try:
        req = urllib.request.Request(url.strip(), data=data, headers=request_headers, method=verb)
        with urllib.request.urlopen(req, timeout=timeout_f) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except Exception as e:  # noqa: BLE001
        _eff_log(f"  ! network error {verb}: {e}")
        return 0, b""


def http_get(url: str, headers: dict | None = None) -> tuple[int, bytes]:
    return http_request("GET", url, headers=headers)


def http_post_json(url: str, payload: dict, headers: dict | None = None) -> tuple[int, bytes]:
    return http_request("POST", url, headers=headers, payload=payload)


def http_post_form(url: str, data: dict, headers: dict | None = None) -> tuple[int, bytes]:
    return http_request("POST", url, headers=headers, form=data)


def _validate_http_url(url: object) -> str:
    if isinstance(url, bool) or not isinstance(url, str) or not url.strip():
        raise ValueError(f"invalid HTTP URL: {url!r}")
    u = url.strip()
    if not u.lower().startswith(("http://", "https://")):
        raise ValueError(f"unsupported non-HTTP URL scheme: {url!r}")
    return u


def _urlopen_bytes(req: urllib.request.Request, timeout: float) -> bytes:
    timeout_f = require_pos_float(timeout, name="timeout")
    with urllib.request.urlopen(req, timeout=timeout_f) as resp:
        return resp.read()


def _urlopen_json(req: urllib.request.Request, timeout: float) -> object:
    raw = _urlopen_bytes(req, timeout)
    return _state_io.parse_json_payload(raw)


def _build_public_get_request(
    url: str, *, user_agent: str = "mptrade-intelligence/1.0", headers: dict | None = None
) -> urllib.request.Request:
    clean_url = _validate_http_url(url)
    req_headers = {"User-Agent": user_agent, **(headers if isinstance(headers, dict) else {})}
    return urllib.request.Request(clean_url, headers=req_headers)


def fetch_public_bytes(
    url: str, *, timeout: float = 5.0, user_agent: str = "mptrade-intelligence/1.0", headers: dict | None = None
) -> bytes:
    """Fetch raw bytes from a public HTTP GET endpoint."""
    return _urlopen_bytes(_build_public_get_request(url, user_agent=user_agent, headers=headers), timeout)


def fetch_public_text(
    url: str,
    *,
    timeout: float = 5.0,
    user_agent: str = "mptrade-intelligence/1.0",
    headers: dict | None = None,
    encoding: str = "utf-8",
    errors: str = "replace",
) -> str:
    """Fetch and decode a text/HTML/CSV payload from a public HTTP GET endpoint."""
    return fetch_public_bytes(url, timeout=timeout, user_agent=user_agent, headers=headers).decode(
        encoding, errors=errors
    )


def fetch_public_json(
    url: str, *, timeout: float = 5.0, user_agent: str = "mptrade-intelligence/1.0", headers: dict | None = None
) -> object:
    """Fetch and decode a JSON payload from a public HTTP GET endpoint."""
    return _urlopen_json(_build_public_get_request(url, user_agent=user_agent, headers=headers), timeout)


def fetch_json_post(
    url: str, payload: dict | bytes | None = None, *, timeout: float = 5.0, headers: dict | None = None
) -> object:
    """Send an HTTP POST request with a JSON body and decode the JSON response."""
    clean_url = _validate_http_url(url)
    req_headers = {"Content-Type": "application/json", **(headers if isinstance(headers, dict) else {})}
    if payload is None:
        data = b"{}"
    elif isinstance(payload, (bytes, bytearray)):
        data = bytes(payload)
    else:
        data = _state_io.encode_json_payload(payload, ensure_ascii=False).encode("utf-8")
    return _urlopen_json(urllib.request.Request(clean_url, data=data, headers=req_headers), timeout)


def _validate_url_and_timeout(url: str, timeout: float) -> tuple[str, float]:
    return _validate_http_url(url), require_pos_float(timeout, name="timeout")


def http_post_status(
    url: str,
    body: str | bytes = b"",
    *,
    headers: dict | None = None,
    timeout: float = 10.0,
    urlopen_fn: object = None,
) -> int:
    """Send an HTTP POST request and return its integer status code (raising on transport error)."""
    clean_url, timeout_f = _validate_url_and_timeout(url, timeout)
    data = body.encode("utf-8") if isinstance(body, str) else bytes(body)
    req_headers = dict(headers) if isinstance(headers, dict) else {}
    req = urllib.request.Request(clean_url, data=data, headers=req_headers, method="POST")
    opener = urlopen_fn if callable(urlopen_fn) else urllib.request.urlopen
    with opener(req, timeout=timeout_f) as resp:
        status = (
            getattr(resp, "status", None)
            or getattr(resp, "code", None)
            or (resp.getcode() if callable(getattr(resp, "getcode", None)) else None)
        )
        return exact_int(status, default=0) or 0


def _resolve_requests_module(requests_mod: object = None) -> object:
    return requests_mod if requests_mod is not None else importlib.import_module("requests")


def _dispatch_requests_call(
    method: str,
    url: str,
    *,
    headers: dict | None = None,
    timeout: float = 10.0,
    requests_mod: object = None,
    extra_kw: dict | None = None,
    **kwargs,
):
    clean_url, timeout_f = _validate_url_and_timeout(url, timeout)
    call_kw = dict(kwargs)
    if headers is not None:
        call_kw["headers"] = headers
    if extra_kw:
        call_kw.update(extra_kw)
    return getattr(_resolve_requests_module(requests_mod), method)(
        clean_url, timeout=timeout_f, **call_kw
    )


def _decode_requests_json(resp: object) -> object:
    resp.raise_for_status()
    return resp.json()


def http_requests_get(
    url: str,
    *,
    headers: dict | None = None,
    params: dict | None = None,
    timeout: float = 10.0,
    requests_mod: object = None,
    **kwargs,
):
    """Execute a ``requests.get`` call through the canonical botcore HTTP gateway."""
    return _dispatch_requests_call(
        "get",
        url,
        headers=headers,
        timeout=timeout,
        requests_mod=requests_mod,
        extra_kw={"params": params} if params is not None else None,
        **kwargs,
    )


def http_requests_post(
    url: str,
    *,
    data: object = _UNSET,
    json_payload: object = _UNSET,
    headers: dict | None = None,
    timeout: float = 10.0,
    requests_mod: object = None,
    **kwargs,
):
    """Execute a ``requests.post`` call through the canonical botcore HTTP gateway."""
    extra: dict = {}
    if data is not _UNSET:
        extra["data"] = data
    if json_payload is not _UNSET:
        extra["json"] = json_payload
    return _dispatch_requests_call(
        "post",
        url,
        headers=headers,
        timeout=timeout,
        requests_mod=requests_mod,
        extra_kw=extra,
        **kwargs,
    )


def fetch_requests_json_get(
    url: str,
    *,
    headers: dict | None = None,
    params: dict | None = None,
    timeout: float = 10.0,
    requests_mod: object = None,
    **kwargs,
) -> object:
    """Perform an HTTP GET via ``http_requests_get``, enforce 2xx status, and return decoded JSON."""
    return _decode_requests_json(
        http_requests_get(
            url, headers=headers, params=params, timeout=timeout, requests_mod=requests_mod, **kwargs
        )
    )


def fetch_requests_json_post(
    url: str,
    payload: object,
    *,
    headers: dict | None = None,
    timeout: float = 10.0,
    requests_mod: object = None,
    **kwargs,
) -> object:
    """Perform an HTTP POST with JSON payload via ``http_requests_post``, enforce 2xx status, and return decoded JSON."""
    return _decode_requests_json(
        http_requests_post(
            url, json_payload=payload, headers=headers, timeout=timeout, requests_mod=requests_mod, **kwargs
        )
    )


_PEM_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN PRIVATE KEY-----(.*?)-----END PRIVATE KEY-----", re.DOTALL
)


def _load_ed25519_signing_key():
    try:
        import nacl.signing

        pem_data = _state_io.read_text_file("keys/ed25519_private.pem")
        b64match = _PEM_PRIVATE_KEY_RE.search(pem_data)
        if not b64match:
            raise ValueError("Invalid PEM")
        der_bytes = base64.b64decode(b64match.group(1).strip())
        if len(der_bytes) < 32:
            raise ValueError("DER too short")
        return nacl.signing.SigningKey(der_bytes[-32:])
    except Exception as e:
        print(f"Error loading the Ed25519 key: {e}")
        return None


def _sign_ed25519(signing_key, payload: str) -> str:
    if signing_key is None or payload is None or isinstance(payload, bool):
        raise ValueError("signing_key and payload are required")
    raw_bytes = payload if isinstance(payload, bytes) else str(payload).encode("utf-8")
    return base64.b64encode(signing_key.sign(raw_bytes).signature).decode("ascii")

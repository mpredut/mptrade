####Binance
import time
import threading

from binance.client import Client
from botcore import exact_pos_int, safe_nonneg_float
from providers.http_resilience import mount_connect_retry

####MYLIB
_client = None

# Keep the local timestamp slightly behind server time so jitter cannot trigger Binance
# -1021 timestamp-ahead errors while remaining well within the receive window.
TIME_SAFETY_MARGIN_MS = 1000
TIME_RESYNC_INTERVAL_SEC = 5 * 60

# Use a 30s receive window (Binance allows up to 60s) so that urllib3 GET retries after a
# 10s socket timeout + backoff do not expire the originally signed timestamp (default 5s).
REQUEST_RECVWINDOW_MS = 30 * 1000

# Resilience against brief DNS/network failures: a REST call without a timeout may hang
# indefinitely, blocking cacheManager updates until the cache is stale and the watchdog
# restarts the process. A short timeout fails quickly so callers can handle the exception
# and continue their loops. Apply REQUEST_TIMEOUT_SEC to every request.
REQUEST_TIMEOUT_SEC = 10

_resync_started = False
_resync_thread = None
_resync_stop_event = threading.Event()
_resync_lock = threading.Lock()
_client_lock = threading.Lock()


def _install_retry(cl):
    """Retry transient DNS/connection/5xx/429 failures on idempotent methods only.

    Never retry POST order placement: if the request succeeds but its response is lost,
    retrying could duplicate a real-money order. urllib3's idempotent-method default
    (kept by the shared helper) excludes POST. Together with the timeout, a transient
    fault fails quickly and GET requests retry internally instead of hanging.
    """
    if not mount_connect_retry(cl.session, attempts=3, backoff=1.0, read=3,
                               status_forcelist=[429, 500, 502, 503, 504],
                               raise_on_status=True):
        print("[bapi_client] _install_retry: adapter mount failed (timeout still guards)")


def sync_time(safety_margin_ms=TIME_SAFETY_MARGIN_MS):
    """Synchronize ``client.timestamp_offset`` with Binance server time.

    Correct local clock skew, commonly observed in WSL, which causes
    APIError(-1021): 'Timestamp for this request was 1000ms ahead of server's time'.
    The unsigned public endpoint does not itself depend on the timestamp."""
    if _client is None:
        return None
    try:
        margin_ms = int(safe_nonneg_float(safety_margin_ms, default=TIME_SAFETY_MARGIN_MS))
        t0_ms = int(time.time() * 1000)
        srv_resp = _client.get_server_time()
        raw_srv = srv_resp.get("serverTime") if isinstance(srv_resp, dict) else None
        server_ms = exact_pos_int(raw_srv, default=None)
        if server_ms is None:
            raise ValueError("invalid serverTime in Binance response")
        t1_ms = int(time.time() * 1000)
        local_ms = (t0_ms + t1_ms) // 2
        _client.timestamp_offset = server_ms - local_ms - margin_ms
        return _client.timestamp_offset
    except Exception as e:
        print(f"[bapi_client] sync_time failed: {e}")
        return None


def _install_recvwindow_retry(cl):
    """Resynchronize server time and retry once when a signed request fails with -1021.

    A -1021 error means Binance rejected the request at the pre-execution timestamp gate
    before matching or mutating any account state, so re-signing with a fresh timestamp
    after ``sync_time()`` is safe and prevents transient clock drift or slow network
    retries from dropping balance/order polls.
    """
    orig_request = getattr(cl, "_request", None)
    if not callable(orig_request) or getattr(cl, "_recvwindow_retry_installed", False):
        return

    def _sanitize_exc(exc):
        msg = getattr(exc, "message", None)
        if isinstance(msg, str) and ("\n" in msg or "\r" in msg):
            exc.message = " ".join(msg.split())

    def _request_with_recvwindow_retry(method, uri, signed, force_params=False, **kwargs):
        saved_data = dict(kwargs["data"]) if isinstance(kwargs.get("data"), dict) else None
        saved_kwargs = dict(kwargs)
        try:
            return orig_request(method, uri, signed, force_params=force_params, **kwargs)
        except Exception as exc:
            is_recvwindow = bool(signed and getattr(exc, "code", None) == -1021)
            is_idempotent_5xx = (
                str(method).lower() == "get"
                and getattr(exc, "status_code", None) in (500, 502, 503, 504)
            )
            if not (is_recvwindow or is_idempotent_5xx):
                _sanitize_exc(exc)
                raise
            if is_recvwindow:
                sync_time()
            retry_kwargs = dict(saved_kwargs)
            if saved_data is not None:
                clean_data = dict(saved_data)
                clean_data.pop("timestamp", None)
                clean_data.pop("signature", None)
                retry_kwargs["data"] = clean_data
            try:
                return orig_request(method, uri, signed, force_params=force_params, **retry_kwargs)
            except Exception as retry_exc:
                _sanitize_exc(retry_exc)
                raise

    cl._request = _request_with_recvwindow_retry
    cl._recvwindow_retry_installed = True


def _start_periodic_resync():
    """Periodically resynchronize from a daemon thread to compensate for WSL drift."""
    global _resync_started, _resync_thread
    with _resync_lock:
        if _resync_thread is not None and _resync_thread.is_alive():
            return
        _resync_stop_event.clear()
        _resync_started = True

        def loop():
            while not _resync_stop_event.wait(TIME_RESYNC_INTERVAL_SEC):
                sync_time()

        _resync_thread = threading.Thread(
            target=loop,
            name="BinanceTimeResync",
            daemon=True,
        )
        _resync_thread.start()


def stop_periodic_resync(timeout=2.0):
    """Stop the resynchronization worker and wait for it to terminate.

    Repeated calls are safe, allowing tests, batch processes, and controlled shutdowns
    to leave no Binance activity after execution ends.
    """
    global _resync_started, _resync_thread
    with _resync_lock:
        thread = _resync_thread
        _resync_stop_event.set()

    if thread is not None and thread is not threading.current_thread():
        join_timeout = safe_nonneg_float(timeout, default=2.0)
        thread.join(timeout=join_timeout)

    with _resync_lock:
        if _resync_thread is thread and (thread is None or not thread.is_alive()):
            _resync_thread = None
            _resync_started = False

    return thread is None or not thread.is_alive()


def getClient():
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is None:
            from keys.apikeys import api_key, api_secret
            if not api_key or not api_secret:
                raise ValueError("BINANCE_API_KEY and BINANCE_API_SECRET are required for the REST client")
            cl = Client(api_key, api_secret, requests_params={"timeout": REQUEST_TIMEOUT_SEC})
            cl.REQUEST_RECVWINDOW = REQUEST_RECVWINDOW_MS
            _install_retry(cl)             # Retry transient GET failures, never POST/order placement.
            _install_recvwindow_retry(cl)  # Re-sync and re-sign once on -1021 recvWindow rejection.
            _client = cl
            sync_time()                    # Initial server-time alignment.
            _start_periodic_resync()       # Maintain alignment over time.
    return _client


class _LazyClientProxy:
    """Preserve the ``client.*`` API while constructing the client on first real use."""
    def __getattr__(self, name):
        return getattr(getClient(), name)

    def __setattr__(self, name, value):
        setattr(getClient(), name, value)


client = _LazyClientProxy()

####Binance
import time
import threading

from binance.client import Client

####MYLIB
_client = None

# Keep the local timestamp slightly behind server time so jitter cannot trigger Binance
# -1021 timestamp-ahead errors while remaining well within the five-second receive window.
TIME_SAFETY_MARGIN_MS = 1000
TIME_RESYNC_INTERVAL_SEC = 5 * 60

# Resilience against brief DNS/network failures: a REST call without a timeout may hang
# indefinitely, blocking cacheManager updates until the cache is stale and the watchdog
# restarts the process. A short timeout fails quickly so callers can handle the exception
# and continue their loops. Apply REQUEST_TIMEOUT_SEC to every request.
REQUEST_TIMEOUT_SEC = 10

_resync_started = False
_resync_thread = None
_resync_stop_event = threading.Event()
_resync_lock = threading.Lock()


def _install_retry(cl):
    """Retry transient DNS/connection/5xx/429 failures on idempotent methods only.

    Never retry POST order placement: if the request succeeds but its response is lost,
    retrying could duplicate a real-money order. urllib3's idempotent-method default
    (kept by the shared helper) excludes POST. Together with the timeout, a transient
    fault fails quickly and GET requests retry internally instead of hanging.
    """
    from providers.http_resilience import mount_connect_retry
    if not mount_connect_retry(cl.session, attempts=3, backoff=1.0, read=3,
                               status_forcelist=[429, 502, 503, 504],
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
        server_ms = _client.get_server_time()["serverTime"]
        local_ms = int(time.time() * 1000)
        _client.timestamp_offset = server_ms - local_ms - safety_margin_ms
        return _client.timestamp_offset
    except Exception as e:
        print(f"[bapi_client] sync_time failed: {e}")
        return None


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
        thread.join(timeout=timeout)

    with _resync_lock:
        if _resync_thread is thread and (thread is None or not thread.is_alive()):
            _resync_thread = None
            _resync_started = False

    return thread is None or not thread.is_alive()


def getClient():
    global _client
    if _client is None:
        from keys.apikeys import api_key, api_secret
        if not api_key or not api_secret:
            raise ValueError("BINANCE_API_KEY and BINANCE_API_SECRET are required for the REST client")
        _client = Client(api_key, api_secret, requests_params={"timeout": REQUEST_TIMEOUT_SEC})
        _install_retry(_client)      # Retry transient GET failures, never POST/order placement.
        sync_time()                 # Initial server-time alignment.
        _start_periodic_resync()    # Maintain alignment over time.
    return _client


class _LazyClientProxy:
    """Preserve the ``client.*`` API while constructing the client on first real use."""
    def __getattr__(self, name):
        return getattr(getClient(), name)

    def __setattr__(self, name, value):
        setattr(getClient(), name, value)


client = _LazyClientProxy()

####Binance
import math
import time
import threading
from urllib.parse import urlencode
import weakref

import requests
from binance.client import Client
from botcore import _parse_bool_flag, exact_pos_int, safe_nonneg_float
import credentials as _credentials
from providers.ws_stream import mount_connect_retry

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

_resync_thread = None
_resync_stop_event = threading.Event()
_resync_lock = threading.Lock()
_client_lock = threading.Lock()


def sync_time(safety_margin_ms=TIME_SAFETY_MARGIN_MS):
    """Synchronize ``client.timestamp_offset`` with Binance server time.

    Correct local clock skew, commonly observed in WSL, which causes
    APIError(-1021): 'Timestamp for this request was 1000ms ahead of server's time'.
    The unsigned public endpoint does not itself depend on the timestamp."""
    if _client is None:
        return None
    try:
        margin_ms = math.trunc(safe_nonneg_float(safety_margin_ms, default=TIME_SAFETY_MARGIN_MS) or 0.0)
        t0_ms = math.trunc(time.time() * 1000)
        srv_resp = _client.get_server_time()
        raw_srv = srv_resp.get("serverTime") if isinstance(srv_resp, dict) else None
        server_ms = exact_pos_int(raw_srv, default=None)
        if server_ms is None:
            raise ValueError("invalid serverTime in Binance response")
        t1_ms = math.trunc(time.time() * 1000)
        local_ms = (t0_ms + t1_ms) // 2
        _client.timestamp_offset = server_ms - local_ms - margin_ms
        return _client.timestamp_offset
    except Exception as e:
        print("[bapi_client] sync_time failed:", e)
        return None


def _install_threadsafe_request(cl):
    """Eliminate python-binance's shared ``self.response`` and ``self.session`` races across threads.

    ``python-binance``'s stock ``Client._request`` assigns ``self.response = ...`` on the shared
    client instance and then calls ``self._handle_response(self.response)``. Under concurrent
    requests (e.g., ``CPFallbackFetch`` price fallback workers or ``BinanceTimeResync``), one
    thread can overwrite ``self.response`` before another thread parses it, returning another
    symbol's payload (such as TAOUSDC's price on an ARBUSDC ticker call). Each worker thread also
    gets its own ``requests.Session`` cloned with identical headers and retry policy.
    """
    if getattr(cl, "_threadsafe_request_installed", False):
        return
    if not callable(getattr(cl, "_get_request_kwargs", None)) or not callable(
        getattr(cl, "_handle_response", None)
    ):
        return

    owner_tid = threading.get_ident()
    tls = threading.local()
    sessions_lock = threading.Lock()
    thread_sessions: dict[int, requests.Session] = {}

    def _close_and_forget_session(sess_id: int, sess_obj: requests.Session) -> None:
        with sessions_lock:
            thread_sessions.pop(sess_id, None)
        try:
            sess_obj.close()
        except Exception:
            pass

    def _get_thread_session():
        base_sess = getattr(cl, "session", None)
        if not isinstance(base_sess, requests.Session) or threading.get_ident() == owner_tid:
            return base_sess
        base_id = id(base_sess)
        local_sess = getattr(tls, "session", None)
        if local_sess is not None and getattr(tls, "base_session_id", None) == base_id:
            return local_sess
        new_sess = requests.Session()
        new_sess.headers.update(base_sess.headers)
        mount_connect_retry(
            new_sess,
            attempts=3,
            backoff=1.0,
            read=3,
            status_forcelist=[429, 500, 502, 503, 504],
            raise_on_status=True,
        )
        tls.session = new_sess
        tls.base_session_id = base_id
        sess_id = id(new_sess)
        with sessions_lock:
            thread_sessions[sess_id] = new_sess
        try:
            weakref.finalize(
                threading.current_thread(), _close_and_forget_session, sess_id, new_sess
            )
        except TypeError:
            pass
        return new_sess

    def _threadsafe_request(method, uri, signed, force_params=False, **kwargs):
        headers = {}
        method_s = str(method).lower()
        if method_s in ("post", "put", "delete"):
            headers["Content-Type"] = "application/x-www-form-urlencoded"

        req_in = dict(kwargs)
        if "data" in req_in:
            raw_data = req_in["data"]
            if isinstance(raw_data, dict):
                copied_data = dict(raw_data)
                if "headers" in copied_data:
                    extra_headers = copied_data.pop("headers")
                    if isinstance(extra_headers, dict):
                        headers.update(extra_headers)
                req_in["data"] = copied_data

        req_kwargs = cl._get_request_kwargs(method_s, signed, force_params, **req_in)
        data = req_kwargs.pop("data", None)

        if signed and getattr(cl, "PRIVATE_KEY", None) and data:
            dict_data = Client.convert_to_dict(data)
            signature = dict_data.pop("signature", None)
            if signature:
                data = f"{urlencode(dict_data)}&signature={signature}"

        session = _get_thread_session()
        resp = getattr(session, method_s)(uri, headers=headers, data=data, **req_kwargs)
        tls.response = resp

        if getattr(cl, "verbose", False) and hasattr(cl, "logger"):
            cl.logger.debug(
                "\nRequest: %s %s\nRequestHeaders: %s\nRequestBody: %s\nResponse: %s\nResponseHeaders: %s\nResponseBody: %s",
                method_s.upper(),
                uri,
                headers,
                data,
                getattr(resp, "status_code", None),
                dict(getattr(resp, "headers", {}) or {}),
                resp.text[:1000] if getattr(resp, "text", None) else None,
            )

        return cl._handle_response(resp)

    orig_close = getattr(cl, "close_connection", None)

    def _close_all_connections():
        with sessions_lock:
            to_close = list(thread_sessions.values())
            thread_sessions.clear()
        for sess in to_close:
            try:
                sess.close()
            except Exception:
                pass
        if callable(orig_close):
            return orig_close()
        return None

    cl._tls = tls
    cl._request = _threadsafe_request
    cl.close_connection = _close_all_connections
    cl._threadsafe_request_installed = True


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
                if isinstance(msg := getattr(exc, "message", None), str) and ("\n" in msg or "\r" in msg):
                    exc.message = " ".join(msg.split())
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
                if isinstance(msg := getattr(retry_exc, "message", None), str) and ("\n" in msg or "\r" in msg):
                    retry_exc.message = " ".join(msg.split())
                raise

    cl._request = _request_with_recvwindow_retry
    cl._recvwindow_retry_installed = True


def _install_post_only_limit_maker(cl):
    """Route limit orders on ``cl`` through ``ORDER_TYPE_LIMIT_MAKER`` (Post-Only) by default."""
    create_fn = getattr(cl, "create_order", None)
    if getattr(cl, "_post_only_limit_installed", False) or not callable(create_fn):
        return

    def _order_limit_maker(timeInForce=Client.TIME_IN_FORCE_GTC, **params):
        post_only = _parse_bool_flag(params.pop("post_only", True), True)
        if post_only and not params.get("icebergQty"):
            params["type"] = getattr(cl, "ORDER_TYPE_LIMIT_MAKER", "LIMIT_MAKER")
            params.pop("timeInForce", None)
        else:
            params.update({"type": getattr(cl, "ORDER_TYPE_LIMIT", "LIMIT"), "timeInForce": timeInForce})
        return create_fn(**params)

    cl.order_limit = _order_limit_maker
    cl._post_only_limit_installed = True


def _start_periodic_resync():
    """Periodically resynchronize from a daemon thread to compensate for WSL drift."""
    global _resync_thread
    with _resync_lock:
        if _resync_thread is not None and _resync_thread.is_alive():
            return
        _resync_stop_event.clear()

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
    global _resync_thread
    with _resync_lock:
        thread = _resync_thread
        _resync_stop_event.set()

    if thread is not None and thread is not threading.current_thread():
        join_timeout = safe_nonneg_float(timeout, default=2.0)
        thread.join(timeout=join_timeout)

    with _resync_lock:
        if _resync_thread is thread and (thread is None or not thread.is_alive()):
            _resync_thread = None

    return thread is None or not thread.is_alive()


def getClient():
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is None:
            api_key = _credentials.api_key
            api_secret = _credentials.api_secret
            if not api_key or not api_secret:
                raise ValueError("BINANCE_API_KEY and BINANCE_API_SECRET are required for the REST client")
            try:
                cl = Client(
                    api_key,
                    api_secret,
                    requests_params={"timeout": REQUEST_TIMEOUT_SEC},
                    ping=False,
                )
            except TypeError:
                cl = Client(api_key, api_secret, requests_params={"timeout": REQUEST_TIMEOUT_SEC})
            cl.REQUEST_RECVWINDOW = REQUEST_RECVWINDOW_MS
            # Retry transient DNS/connection/5xx/429 failures on idempotent methods only (never POST/order placement).
            if not mount_connect_retry(cl.session, attempts=3, backoff=1.0, read=3,
                                       status_forcelist=[429, 500, 502, 503, 504],
                                       raise_on_status=True):
                print("[bapi_client] _install_retry: adapter mount failed (timeout still guards)")
            _install_threadsafe_request(cl)  # Isolate per-thread HTTP sessions & local response objects.
            _install_recvwindow_retry(cl)  # Re-sync and re-sign once on -1021 recvWindow rejection.
            _install_post_only_limit_maker(cl)  # Default limit orders to LIMIT_MAKER (Post-Only).
            _client = cl
            sync_time()                    # Initial server-time alignment.
            _start_periodic_resync()       # Maintain alignment over time.
    return _client


class _LazyClientProxy:
    """Preserve the ``client.*`` API while constructing the client on first real use."""
    def __getattr__(self, name):
        try:
            target = getClient()
        except ValueError as exc:
            if hasattr(Client, name):
                def _deferred_call(*args, **kwargs):
                    return getattr(getClient(), name)(*args, **kwargs)
                _deferred_call._is_lazy_client_stub = True
                return _deferred_call
            raise AttributeError(str(exc)) from exc
        return getattr(target, name)

    def __setattr__(self, name, value):
        if getattr(value, "__dict__", {}).get("_is_lazy_client_stub") is True:
            self.__dict__.pop(name, None)
            if _client is not None:
                _client.__dict__.pop(name, None)
        else:
            object.__setattr__(self, name, value)
            if _client is not None:
                setattr(_client, name, value)

    def __delattr__(self, name):
        found = False
        if name in self.__dict__:
            object.__delattr__(self, name)
            found = True
        if _client is not None and (name in _client.__dict__ or hasattr(type(_client), name)):
            if name in _client.__dict__:
                delattr(_client, name)
            found = True
        if not found and not hasattr(Client, name):
            raise AttributeError(name)


client = _LazyClientProxy()

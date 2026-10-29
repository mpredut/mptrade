"""Order-safe HTTP retry mounting, shared across provider clients.

DNS/connection blips (e.g. "Failed to resolve <host>" during a VPN or resolver
hiccup) happen BEFORE a request is sent, so retrying a *connect* failure can never
re-send a POST that already reached the venue and placed a real-money order. That
connect-only profile is therefore safe for any HTTP method and is the default here.

Read/status retries cover transient read errors and retryable status codes on
GET-class (idempotent) requests only: urllib3's default ``allowed_methods`` excludes
POST, so an order POST is never replayed. NEVER widen allowed_methods to include POST.

Used by the Hyperliquid SDK session (all requests are POST -> connect-only) and the
python-binance client session (GET reads also retry on read/status, POST orders stay
connect-only). Kraken uses its own ``http_post_form`` layer rather than a
``requests.Session`` and is not wired here; its DNS is covered by the systemd-resolved
cache (see systemd/DNS_RESILIENCE.md).
"""
from __future__ import annotations

import math

from botcore import _parse_bool_flag, safe_nonneg_float, safe_nonneg_int, safe_pos_int


def mount_connect_retry(session, *, attempts: int = 3, backoff: float = 0.4,
                        read: int = 0, status_forcelist=None,
                        raise_on_status: bool = False) -> bool:
    """Mount a Retry adapter on ``session`` for https:// and http:// (best-effort).

    ``attempts`` connect retries are always mounted -- the order-safe DNS/connect
    layer. Pass ``read`` > 0 and/or ``status_forcelist`` to also retry idempotent
    GET-class requests on read errors / those status codes; a POST stays connect-only
    through urllib3's method default.

    Returns True when mounted, False if the adapter could not be built (the caller's
    request timeout then remains the only guard). Never raises, so it cannot block
    client startup.
    """
    try:
        if session is None or isinstance(status_forcelist, (bool, str, bytes)):
            return False
        att_i = safe_nonneg_int(attempts, default=None)
        read_i = safe_nonneg_int(read, default=None)
        bo_f = safe_nonneg_float(backoff, default=None)
        if att_i is None or read_i is None or bo_f is None:
            return False
        forcelist = None
        if status_forcelist:
            forcelist = []
            for code in status_forcelist:
                c_i = safe_pos_int(code, default=None)
                if c_i is None:
                    return False
                forcelist.append(c_i)
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        retry = Retry(
            total=att_i,
            connect=att_i,
            read=read_i,
            status=att_i if forcelist else 0,
            redirect=0,
            backoff_factor=bo_f,
            status_forcelist=forcelist if forcelist else None,
            raise_on_status=_parse_bool_flag(raise_on_status, default=False),
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return True
    except Exception:  # noqa: BLE001 -- retries are best-effort; the timeout still guards
        return False

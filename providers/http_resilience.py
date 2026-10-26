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
        if (
            isinstance(attempts, bool)
            or isinstance(read, bool)
            or isinstance(backoff, bool)
            or isinstance(status_forcelist, bool)
            or not isinstance(attempts, (int, float))
            or not isinstance(read, (int, float))
            or not isinstance(backoff, (int, float))
        ):
            return False
        att_f = float(attempts)
        read_f = float(read)
        bo_f = float(backoff)
        if not (math.isfinite(att_f) and math.isfinite(read_f) and math.isfinite(bo_f)):
            return False
        att_i = int(att_f)
        read_i = int(read_f)
        if att_i < 0 or read_i < 0 or bo_f < 0.0:
            return False
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        retry = Retry(
            total=att_i,
            connect=att_i,
            read=read_i,
            status=att_i if status_forcelist else 0,
            redirect=0,
            backoff_factor=bo_f,
            status_forcelist=list(status_forcelist) if status_forcelist else None,
            raise_on_status=bool(raise_on_status),
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return True
    except Exception:  # noqa: BLE001 -- retries are best-effort; the timeout still guards
        return False

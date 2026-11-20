"""core/http.py -- Canonical HTTP transport, requests gateway, and Ed25519 signing helpers."""
from __future__ import annotations

import base64
import importlib
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

from cachepaths import exact_int
from core.lock import log as _default_log
from core.math_coercion import _UNSET, require_pos_float
import state_io as _state_io

HTTP_TIMEOUT = 25


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

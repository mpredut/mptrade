#!/usr/bin/env python3
"""
hl_client.py — Hyperliquid client built on the official SDK.

Reads (unsigned Info API): prices, positions, open orders, and balances.
Trading (Exchange API, EIP-712 agent-wallet signature): place and cancel orders.

Model: long-only perpetuals; the HYPE perpetual is liquid. At low 1x leverage it
behaves approximately like spot with a distant liquidation price. "buy" opens or
increases a long position, while "TP" reduces it.

Run with the Python interpreter from the SDK virtual environment:
    python hl_bot.py
"""

from __future__ import annotations

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _HERE not in sys.path:
    sys.path.append(_HERE)

_common_mod = sys.modules.get("common")
_common_file = getattr(_common_mod, "__file__", None) if _common_mod is not None else None
if _common_mod is None or (isinstance(_common_file, str) and os.path.dirname(os.path.abspath(_common_file)) != _HERE):
    import importlib.util as _ilu
    _c_spec = _ilu.spec_from_file_location("common", os.path.join(_HERE, "common.py"))
    _common_mod = _ilu.module_from_spec(_c_spec)
    sys.modules["common"] = _common_mod
    _c_spec.loader.exec_module(_common_mod)
log = _common_mod.log
safe_float = _common_mod.safe_float
safe_pos_float = _common_mod.safe_pos_float
safe_nonneg_float = _common_mod.safe_nonneg_float

try:
    import eth_account
    from hyperliquid.info import Info
    from hyperliquid.exchange import Exchange
    from hyperliquid.utils import constants
    _SDK_OK = True
except Exception as _e:  # noqa: BLE001
    _SDK_OK = False
    _SDK_ERR = str(_e)


class HLError(Exception):
    pass


INFO_TIMEOUT_SECONDS = 10.0
EXCHANGE_TIMEOUT_SECONDS = 30.0


def _exact_int_in_range(val: object, lo: int = 0, hi: int | None = None) -> int | None:
    f = safe_float(val, default=None)
    if f is None or not f.is_integer():
        return None
    i = int(f)
    if i < lo or (hi is not None and i > hi):
        return None
    return i


def _round_px(px: float, sz_decimals: int, is_perp: bool = True) -> float:
    """Round an HL price to 5 significant figures and at most (6/8 - szDecimals) decimals."""
    if isinstance(px, bool) or isinstance(sz_decimals, bool):
        raise ValueError("px and sz_decimals must not be boolean")
    try:
        px_raw = float(px)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("px and sz_decimals must be finite numbers") from exc
    dec_f = safe_float(sz_decimals, default=None)
    if dec_f is None:
        raise ValueError("px and sz_decimals must be finite numbers")
    dec_i = _exact_int_in_range(dec_f, 0)
    if dec_i is None:
        raise ValueError("sz_decimals must be a non-negative integer")
    px_f = safe_pos_float(px_raw, default=None)
    if px_f is None:
        return px_raw
    max_dec = (6 if is_perp else 8) - dec_i
    px = float(f"{px_f:.5g}")           # five significant figures
    return round(px, max(max_dec, 0))


def _parse_order_response(res: object, *, include_avg_px: bool = False) -> tuple[bool, int | None, str]:
    """Extract (ok, oid, status_message) from a Hyperliquid Exchange.order payload."""
    if not isinstance(res, dict) or res.get("status") != "ok":
        return False, None, str(res)
    resp_obj = res.get("response") if isinstance(res.get("response"), dict) else {}
    data_obj = resp_obj.get("data") if isinstance(resp_obj.get("data"), dict) else {}
    statuses = data_obj.get("statuses") if isinstance(data_obj.get("statuses"), list) else []
    if not statuses or not isinstance(statuses[0], dict):
        return False, None, str(res)
    st = statuses[0]
    if "resting" in st and isinstance(st["resting"], dict):
        return True, st["resting"].get("oid"), "resting"
    if "filled" in st and isinstance(st["filled"], dict):
        filled = st["filled"]
        msg = (
            f"filled {filled.get('totalSz')} @ {filled.get('avgPx')}"
            if include_avg_px
            else f"filled {filled.get('totalSz')}"
        )
        return True, filled.get("oid"), msg
    if "error" in st:
        return False, None, str(st["error"])
    return True, None, str(st)


def _force_timeout(api_obj, seconds: float = 30.0) -> None:
    """Force a default timeout on the API object's requests session.
    The HL SDK does not set a read timeout, so a request on a dead keep-alive
    connection could otherwise block the process indefinitely without an error."""
    try:
        # The current Hyperliquid SDK explicitly calls
        # ``session.post(..., timeout=self.timeout)``. Its default is None, so a
        # simple setdefault had no effect because the key already existed and
        # requests remained deadline-free. Set the SDK attribute as well, and
        # have the wrapper explicitly replace timeout=None.
        if getattr(api_obj, "timeout", None) is None:
            api_obj.timeout = seconds
        sess = api_obj.session
        orig = sess.request

        def _req(*a, **kw):
            if kw.get("timeout") is None:
                kw["timeout"] = seconds
            return orig(*a, **kw)

        sess.request = _req

        # Retry ONLY connection-establishment failures (connect), notably the
        # intermittent 'Failed to resolve api.hyperliquid.xyz' during VPN/resolver
        # blips: those happen BEFORE the request is sent, so a retry cannot double-
        # submit an order. Shared order-safe helper with its connect-only default
        # (read/status stay 0, so a POST that may already have reached the exchange is
        # never replayed). Shrinks the "blind" ticks the systemd-resolved cache alone
        # cannot fully prevent.
        from providers.http_resilience import mount_connect_retry
        mount_connect_retry(sess)
    except Exception:  # noqa: BLE001 — do not block startup if SDK internals change
        pass


class HLClient:
    def __init__(self, secret_key: str | None = None, account_address: str | None = None,
                 mainnet: bool = True):
        if not _SDK_OK:
            raise HLError(f"Hyperliquid SDK unavailable: {_SDK_ERR} "
                          f"(run it with the python from .venv)")
        self.base = constants.MAINNET_API_URL if mainnet else constants.TESTNET_API_URL
        # Info fetches metadata in its constructor, so the timeout must be passed
        # during construction. Applying it afterwards leaves startup able to hang.
        self.info = Info(
            self.base, skip_ws=True, timeout=INFO_TIMEOUT_SECONDS,
        )
        _force_timeout(self.info, seconds=INFO_TIMEOUT_SECONDS)
        self.address = account_address
        self.exchange = None
        if secret_key:
            wallet = eth_account.Account.from_key(secret_key)
            self.address = account_address or wallet.address
            # Exchange also builds an Info client internally during construction.
            self.exchange = Exchange(
                wallet,
                self.base,
                account_address=self.address,
                timeout=EXCHANGE_TIMEOUT_SECONDS,
            )
            _force_timeout(self.exchange, seconds=EXCHANGE_TIMEOUT_SECONDS)
        self._meta_cache: dict[str, dict] = {}

    @classmethod
    def public(cls, *, account_address: str | None = None,
               mainnet: bool = True) -> "HLClient":
        """Build an unsigned client; account reads may use a public address."""
        return cls(secret_key=None, account_address=account_address, mainnet=mainnet)

    # ----- metadata / prices --------------------------------------------------
    def _meta(self) -> dict:
        if not self._meta_cache:
            raw_meta = self.info.meta()
            universe = raw_meta.get("universe") if isinstance(raw_meta, dict) else None
            for a in (universe if isinstance(universe, list) else ()):
                if isinstance(a, dict) and a.get("name"):
                    self._meta_cache[a["name"]] = a
        return self._meta_cache

    def sz_decimals(self, coin: str) -> int:
        if isinstance(coin, bool) or not isinstance(coin, str) or not coin.strip():
            raise HLError(f"Missing size precision metadata for {coin}")
        coin = coin.strip()
        meta = self._meta().get(coin)
        dec_i = _exact_int_in_range(meta.get("szDecimals") if isinstance(meta, dict) else None, 0, 18)
        if dec_i is not None:
            return dec_i
        try:
            spot_meta = self.info.spot_meta() if hasattr(self.info, "spot_meta") else {}
            for token in (spot_meta or {}).get("tokens", []):
                if isinstance(token, dict) and token.get("name") == coin and "szDecimals" in token:
                    tdec_i = _exact_int_in_range(token["szDecimals"], 0, 18)
                    if tdec_i is not None:
                        return tdec_i
        except Exception:  # noqa: BLE001
            pass
        raise HLError(f"Missing size precision metadata for {coin}")

    def max_leverage(self, coin: str) -> int:
        if isinstance(coin, bool) or not isinstance(coin, str) or not coin.strip():
            raise HLError(f"Missing leverage metadata for {coin}")
        coin = coin.strip()
        meta = self._meta().get(coin)
        lev_i = _exact_int_in_range(meta.get("maxLeverage") if isinstance(meta, dict) else None, 1)
        if lev_i is not None:
            return lev_i
        raise HLError(f"Missing leverage metadata for {coin}")

    def coin_listed(self, coin: str) -> bool:
        if isinstance(coin, bool) or not isinstance(coin, str) or not coin.strip():
            return False
        return coin.strip() in self._meta()

    def _mid_for_key(self, key: str, label: str) -> float | None:
        if isinstance(key, bool) or not isinstance(key, str) or not key.strip():
            return None
        key = key.strip()
        try:
            mids = self.info.all_mids()
            if not isinstance(mids, dict):
                return None
            return safe_pos_float(mids.get(key), None)
        except Exception as e:  # noqa: BLE001
            log(f"  ! {label}({key}) failed: {e}")
            return None

    def mid(self, coin: str) -> float | None:
        return self._mid_for_key(coin, "mid")

    # ----- account (read-only, address only) ----------------------------------
    def _user_state(self) -> dict:
        if not self.address:
            raise HLError("HL_ACCOUNT_ADDRESS missing")
        st = self.info.user_state(self.address)
        if not isinstance(st, dict):
            raise HLError(f"Invalid user_state payload: {type(st).__name__}")
        return st

    def position_strict(self, coin: str) -> tuple[float, float]:
        """Return position data while propagating API errors.
        Use when code must distinguish no position (0) from unknown state, such as
        delta-neutral logic where a false zero could open a duplicate leg."""
        if isinstance(coin, bool) or not isinstance(coin, str) or not coin.strip():
            return 0.0, 0.0
        coin = coin.strip()
        for ap in (self._user_state().get("assetPositions") or []):
            if not isinstance(ap, dict):
                continue
            p = ap.get("position")
            if isinstance(p, dict) and p.get("coin") == coin:
                raw_szi = p.get("szi")
                raw_entry = p.get("entryPx")
                if isinstance(raw_szi, bool) or isinstance(raw_entry, bool):
                    raise HLError(f"Invalid boolean position state for {coin}")
                szi = safe_float(raw_szi or 0.0, default=None)
                entry = safe_nonneg_float(raw_entry or 0.0, default=None)
                if szi is None or entry is None:
                    raise HLError(f"Invalid numeric position state for {coin}")
                return szi, entry
        return 0.0, 0.0

    def position(self, coin: str) -> tuple[float, float]:
        """Return (szi, entryPx) for a coin, where szi>0 is long.
        Return (0, 0) when no position exists or after a logged API error."""
        try:
            return self.position_strict(coin)
        except HLError:
            raise
        except Exception as e:  # noqa: BLE001
            log(f"  ! position({coin}) failed: {e}")
        return 0.0, 0.0

    def withdrawable(self) -> float:
        try:
            return safe_nonneg_float(self._user_state().get("withdrawable"), 0.0)
        except Exception:  # noqa: BLE001
            return 0.0

    def open_orders_strict(self, coin: str | None = None) -> list[dict]:
        """Return open orders while propagating API errors."""
        if not self.address:
            raise HLError("HL_ACCOUNT_ADDRESS missing")
        coin_norm = None
        if coin is not None:
            if isinstance(coin, bool) or not isinstance(coin, str) or not coin.strip():
                return []
            coin_norm = coin.strip()
        oo = self.info.open_orders(self.address)
        if not isinstance(oo, list):
            raise HLError(f"Invalid open_orders payload: {type(oo).__name__}")
        return [o for o in oo if isinstance(o, dict) and (coin_norm is None or o.get("coin") == coin_norm)]

    def open_orders(self, coin: str | None = None) -> list[dict]:
        if not self.address:
            return []
        try:
            return self.open_orders_strict(coin)
        except Exception as e:  # noqa: BLE001
            log(f"  ! open_orders failed: {e}")
            return []

    # ----- SPOT + funding reads -----------------------------------------------
    def resolve_spot_pair(self, token: str) -> str | None:
        """Resolve a TOKEN/USDC spot pair (@index) from spotMeta for any token,
        such as HYPE -> @107, USOL -> @156, or PURR -> PURR/USDC."""
        if isinstance(token, bool) or not isinstance(token, str) or not token.strip():
            return None
        token = token.strip()
        try:
            m = self.info.spot_meta()
            if not isinstance(m, dict):
                return None
            tokens = {
                t.get("name"): t.get("index")
                for t in (m.get("tokens") or [])
                if isinstance(t, dict)
            }
            ti, usdc = tokens.get(token), tokens.get("USDC")
            if ti is None or usdc is None:
                return None
            for u in (m.get("universe") or []):
                if isinstance(u, dict) and u.get("tokens") == [ti, usdc]:
                    return u.get("name")
        except Exception as e:  # noqa: BLE001
            log(f"  ! resolve_spot_pair({token}) failed: {e}")
        return None

    def spot_mid(self, pair: str) -> float | None:
        """Return the spot price for an @index pair, e.g. @107 = HYPE/USDC."""
        return self._mid_for_key(pair, "spot_mid")

    def spot_balance_strict(self, token: str) -> float:
        """Return spot balance while propagating API errors; see position_strict."""
        if not self.address:
            raise HLError("HL_ACCOUNT_ADDRESS missing")
        if isinstance(token, bool) or not isinstance(token, str) or not token.strip():
            return 0.0
        token = token.strip()
        st = self.info.spot_user_state(self.address)
        if not isinstance(st, dict):
            raise HLError(f"Invalid spot_user_state payload: {type(st).__name__}")
        for b in (st.get("balances") or []):
            if isinstance(b, dict) and b.get("coin") == token:
                raw_tot = b.get("total")
                if isinstance(raw_tot, bool):
                    raise HLError(f"Invalid boolean spot balance for {token}")
                tot = safe_nonneg_float(raw_tot or 0.0, default=None)
                if tot is None:
                    raise HLError(f"Invalid numeric spot balance for {token}")
                return tot
        return 0.0

    def spot_balance(self, token: str) -> float:
        """Return the held quantity of a spot token such as HYPE or USDC.
        Return 0.0 when absent or after a logged API error."""
        try:
            return self.spot_balance_strict(token)
        except Exception as e:  # noqa: BLE001
            log(f"  ! spot_balance({token}) failed: {e}")
        return 0.0

    def funding_rate(self, coin: str) -> float | None:
        """Return the current hourly perpetual funding rate; positive means longs pay shorts."""
        if isinstance(coin, bool) or not isinstance(coin, str) or not coin.strip():
            return None
        coin = coin.strip()
        try:
            meta, ctxs = self.info.meta_and_asset_ctxs()
            universe = (meta.get("universe") or []) if isinstance(meta, dict) else []
            if not isinstance(ctxs, list):
                return None
            for i, a in enumerate(universe):
                if isinstance(a, dict) and a.get("name") == coin and i < len(ctxs) and isinstance(ctxs[i], dict):
                    return safe_float(ctxs[i].get("funding"), None)
        except Exception as e:  # noqa: BLE001
            log(f"  ! funding_rate({coin}) failed: {e}")
        return None

    def position_full(self, coin: str) -> dict | None:
        """Return the complete perpetual position including szi, entryPx, liquidationPx, and PnL."""
        if isinstance(coin, bool) or not isinstance(coin, str) or not coin.strip():
            return None
        coin = coin.strip()
        try:
            for ap in (self._user_state().get("assetPositions") or []):
                if not isinstance(ap, dict):
                    continue
                p = ap.get("position")
                if isinstance(p, dict) and p.get("coin") == coin:
                    return p
        except Exception as e:  # noqa: BLE001
            log(f"  ! position_full failed: {e}")
        return None

    def margin_summary(self) -> dict:
        """Return perpetual account value, used margin, and withdrawable balance."""
        try:
            st = self._user_state()
            ms = st.get("marginSummary", {}) if isinstance(st, dict) else {}
            if not isinstance(ms, dict):
                ms = {}
            return {
                "accountValue": safe_float(ms.get("accountValue"), 0.0),
                "totalMarginUsed": safe_float(ms.get("totalMarginUsed"), 0.0),
                "withdrawable": safe_float(st.get("withdrawable") if isinstance(st, dict) else 0.0, 0.0),
            }
        except Exception:  # noqa: BLE001
            return {}

    def funding_history(self, start_ms: int) -> list[dict]:
        """Return actual funding payments received or paid since start_ms."""
        if not self.address:
            return []
        start_f = safe_nonneg_float(start_ms, default=None)
        if start_f is None:
            return []
        try:
            res = self.info.user_funding_history(self.address, int(start_f))
            return [x for x in res if isinstance(x, dict)] if isinstance(res, list) else []
        except Exception as e:  # noqa: BLE001
            log(f"  ! funding_history failed: {e}")
            return []

    def candles(self, coin: str, interval: str = "1h", lookback_hours: int = 60) -> list[dict]:
        """Return OHLCV candles for trend indicators."""
        if (
            not coin or isinstance(coin, bool) or not isinstance(coin, str) or not coin.strip()
            or not interval or isinstance(interval, bool) or not isinstance(interval, str) or not interval.strip()
        ):
            return []
        lh_f = safe_pos_float(lookback_hours, default=None)
        if lh_f is None:
            return []
        coin = coin.strip()
        interval = interval.strip()
        end = int(time.time() * 1000)
        start = max(0, int(end - lh_f * 3600 * 1000))
        try:
            res = self.info.candles_snapshot(coin, interval, start, end)
            return [x for x in res if isinstance(x, dict)] if isinstance(res, list) else []
        except Exception as e:  # noqa: BLE001
            log(f"  ! candles({coin}) failed: {e}")
            return []

    def spot_order(self, pair: str, is_buy: bool, sz: float, px: float,
                   sz_decimals: int = 2,
                   cloid: str | None = None,
                   is_market: bool = False,
                   tif: str | None = None) -> tuple[bool, int | None, str]:
        """Place a spot limit or market IOC order; pair is the @index name such as @107."""
        if not self.exchange:
            raise HLError("No agent wallet (HL_SECRET_KEY)")
        if isinstance(pair, bool) or not isinstance(pair, str) or not pair.strip():
            return False, None, "invalid pair"
        pair = pair.strip()
        if not isinstance(is_buy, bool):
            return False, None, "invalid non-boolean is_buy"
        if not isinstance(is_market, bool):
            return False, None, "invalid non-boolean is_market"
        if tif is not None and (isinstance(tif, bool) or not isinstance(tif, str) or not tif.strip()):
            return False, None, "invalid tif"
        if isinstance(sz, bool) or isinstance(px, bool) or isinstance(sz_decimals, bool):
            return False, None, "invalid boolean order parameter"
        sz_f = safe_float(sz, default=None)
        px_f = safe_float(px, default=None)
        dec_raw = _exact_int_in_range(sz_decimals, 0)
        if sz_f is None or px_f is None or dec_raw is None:
            return False, None, "invalid numeric order parameter"
        dec_i = min(dec_raw, 18)
        if sz_f <= 0 or px_f <= 0:
            return False, None, "invalid non-positive sz or px"
        sz = round(sz_f, dec_i)
        px = _round_px(px_f, dec_i, is_perp=False)
        if sz <= 0 or px <= 0:
            return False, None, "rounded sz or px is non-positive"
        order_tif = tif or ("Ioc" if is_market else "Gtc")
        try:
            kwargs = {}
            if cloid is not None:
                # The official SDK requires a Cloid object rather than a raw hex string.
                from hyperliquid.utils.types import Cloid
                kwargs["cloid"] = Cloid.from_str(cloid)
            res = self.exchange.order(
                pair, is_buy, sz, px, {"limit": {"tif": order_tif}}, **kwargs,
            )
        except Exception as e:  # noqa: BLE001
            return False, None, str(e)
        return _parse_order_response(res, include_avg_px=False)

    # ----- signed trading -----------------------------------------------------
    def set_leverage(self, coin: str, leverage: int) -> None:
        if not self.exchange or not coin or isinstance(coin, bool) or not isinstance(coin, str) or not coin.strip():
            return
        lev_i = _exact_int_in_range(leverage, 1)
        if lev_i is None:
            return
        try:
            self.exchange.update_leverage(lev_i, coin.strip(), is_cross=True)
            log(f"  [HL] leverage {coin.strip()} set to {lev_i}x")
        except Exception as e:  # noqa: BLE001
            log(f"  ! set_leverage failed: {e}")

    def place_limit(self, coin: str, is_buy: bool, sz: float, px: float,
                    reduce_only: bool = False) -> tuple[bool, int | None, str]:
        """Place a GTC LIMIT order and return (ok, oid, message)."""
        if not self.exchange:
            raise HLError("No agent wallet (HL_SECRET_KEY) — orders cannot be placed")
        if isinstance(coin, bool) or not isinstance(coin, str) or not coin.strip():
            return False, None, "invalid coin"
        coin = coin.strip()
        if not isinstance(is_buy, bool) or not isinstance(reduce_only, bool):
            return False, None, "invalid non-boolean is_buy or reduce_only"
        if isinstance(sz, bool) or isinstance(px, bool):
            return False, None, "invalid boolean order parameter"
        sz_f = safe_float(sz, default=None)
        px_f = safe_float(px, default=None)
        if sz_f is None or px_f is None:
            return False, None, "invalid numeric order parameter"
        if sz_f <= 0 or px_f <= 0:
            return False, None, "invalid non-positive sz or px"
        dec_i = self.sz_decimals(coin)
        sz = round(sz_f, dec_i)
        px = _round_px(px_f, dec_i)
        if sz <= 0 or px <= 0:
            return False, None, "rounded sz or px is non-positive"
        try:
            res = self.exchange.order(coin, is_buy, sz, px,
                                      {"limit": {"tif": "Gtc"}}, reduce_only=reduce_only)
        except Exception as e:  # noqa: BLE001
            return False, None, str(e)
        return _parse_order_response(res, include_avg_px=True)

    def cancel(self, coin: str, oid: int) -> bool:
        if (
            not self.exchange
            or isinstance(coin, bool)
            or not isinstance(coin, str)
            or not coin.strip()
            or oid is None
            or isinstance(oid, bool)
        ):
            return False
        coin = coin.strip()
        if isinstance(oid, str):
            s_oid = oid.strip()
            if s_oid.endswith(".0") and s_oid[:-2].lstrip("+").isdigit():
                s_oid = s_oid[:-2]
            if not s_oid.lstrip("+").isdigit():
                return False
            oid = s_oid
        oid_i = _exact_int_in_range(oid, 1)
        if oid_i is None:
            return False
        try:
            res = self.exchange.cancel(coin, oid_i)
            if not isinstance(res, dict) or res.get("status") != "ok":
                return False
            resp_obj = res.get("response") if isinstance(res.get("response"), dict) else {}
            data_obj = resp_obj.get("data") if isinstance(resp_obj.get("data"), dict) else {}
            statuses = data_obj.get("statuses") if isinstance(data_obj.get("statuses"), list) else []
            if not statuses:
                return False
            st = statuses[0]
            if st == "success":
                return True
            if isinstance(st, dict) and "error" in st:
                log(f"  ! cancel oid={oid} rejected: {st['error']}")
                return False
            if isinstance(st, str) and "error" in st.lower():
                log(f"  ! cancel oid={oid} rejected: {st}")
                return False
            return True
        except Exception as e:  # noqa: BLE001
            log(f"  ! cancel failed: {e}")
            return False

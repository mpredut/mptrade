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

import math
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    from .common import log
except ImportError:
    from common import log

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


def _round_px(px: float, sz_decimals: int, is_perp: bool = True) -> float:
    """Round an HL price to 5 significant figures and at most (6/8 - szDecimals) decimals."""
    if isinstance(px, bool) or isinstance(sz_decimals, bool):
        raise ValueError("px and sz_decimals must not be boolean")
    try:
        px = float(px)
        dec_f = float(sz_decimals)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("px and sz_decimals must be finite numbers") from exc
    if not math.isfinite(dec_f) or not dec_f.is_integer() or int(dec_f) < 0:
        raise ValueError("sz_decimals must be a non-negative integer")
    if not math.isfinite(px) or px <= 0:
        return px
    max_dec = (6 if is_perp else 8) - int(dec_f)
    px = float(f"{px:.5g}")           # five significant figures
    return round(px, max(max_dec, 0))


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
        meta = self._meta().get(coin)
        raw_dec = meta.get("szDecimals") if isinstance(meta, dict) else None
        if raw_dec is not None and not isinstance(raw_dec, bool):
            try:
                dec_f = float(raw_dec)
                if math.isfinite(dec_f) and dec_f.is_integer() and 0 <= int(dec_f) <= 18:
                    return int(dec_f)
            except (TypeError, ValueError, OverflowError):
                pass
        try:
            spot_meta = self.info.spot_meta() if hasattr(self.info, "spot_meta") else {}
            for token in (spot_meta or {}).get("tokens", []):
                if isinstance(token, dict) and token.get("name") == coin and "szDecimals" in token:
                    raw_tdec = token["szDecimals"]
                    if raw_tdec is not None and not isinstance(raw_tdec, bool):
                        tdec_f = float(raw_tdec)
                        if math.isfinite(tdec_f) and tdec_f.is_integer() and 0 <= int(tdec_f) <= 18:
                            return int(tdec_f)
        except Exception:  # noqa: BLE001
            pass
        raise HLError(f"Missing size precision metadata for {coin}")

    def max_leverage(self, coin: str) -> int:
        meta = self._meta().get(coin)
        raw_lev = meta.get("maxLeverage") if isinstance(meta, dict) else None
        if raw_lev is not None and not isinstance(raw_lev, bool):
            try:
                lev_f = float(raw_lev)
                if math.isfinite(lev_f) and lev_f.is_integer() and int(lev_f) > 0:
                    return int(lev_f)
            except (TypeError, ValueError, OverflowError):
                pass
        raise HLError(f"Missing leverage metadata for {coin}")

    def coin_listed(self, coin: str) -> bool:
        return coin in self._meta()

    def mid(self, coin: str) -> float | None:
        try:
            mids = self.info.all_mids()
            if not isinstance(mids, dict):
                return None
            v = mids.get(coin)
            if v is None or isinstance(v, bool):
                return None
            px = float(v)
            return px if math.isfinite(px) and px > 0.0 else None
        except Exception as e:  # noqa: BLE001
            log(f"  ! mid({coin}) failed: {e}")
            return None

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
        for ap in (self._user_state().get("assetPositions") or []):
            if not isinstance(ap, dict):
                continue
            p = ap.get("position")
            if isinstance(p, dict) and p.get("coin") == coin:
                raw_szi = p.get("szi")
                raw_entry = p.get("entryPx")
                if isinstance(raw_szi, bool) or isinstance(raw_entry, bool):
                    raise HLError(f"Invalid boolean position state for {coin}")
                try:
                    szi = float(raw_szi or 0.0)
                    entry = float(raw_entry or 0.0)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise HLError(f"Invalid numeric position state for {coin}") from exc
                if not (math.isfinite(szi) and math.isfinite(entry)) or entry < 0.0:
                    raise HLError(f"Invalid non-finite position state for {coin}")
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
            raw = self._user_state().get("withdrawable")
            if raw is None or isinstance(raw, bool):
                return 0.0
            val = float(raw)
            return val if math.isfinite(val) and val >= 0.0 else 0.0
        except Exception:  # noqa: BLE001
            return 0.0

    def open_orders_strict(self, coin: str | None = None) -> list[dict]:
        """Return open orders while propagating API errors."""
        if not self.address:
            raise HLError("HL_ACCOUNT_ADDRESS missing")
        oo = self.info.open_orders(self.address)
        if not isinstance(oo, list):
            raise HLError(f"Invalid open_orders payload: {type(oo).__name__}")
        return [o for o in oo if isinstance(o, dict) and (coin is None or o.get("coin") == coin)]

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
        try:
            mids = self.info.all_mids()
            if not isinstance(mids, dict):
                return None
            v = mids.get(pair)
            if v is None or isinstance(v, bool):
                return None
            px = float(v)
            return px if math.isfinite(px) and px > 0.0 else None
        except Exception as e:  # noqa: BLE001
            log(f"  ! spot_mid({pair}) failed: {e}")
            return None

    def spot_balance_strict(self, token: str) -> float:
        """Return spot balance while propagating API errors; see position_strict."""
        if not self.address:
            raise HLError("HL_ACCOUNT_ADDRESS missing")
        st = self.info.spot_user_state(self.address)
        if not isinstance(st, dict):
            raise HLError(f"Invalid spot_user_state payload: {type(st).__name__}")
        for b in (st.get("balances") or []):
            if isinstance(b, dict) and b.get("coin") == token:
                raw_tot = b.get("total")
                if isinstance(raw_tot, bool):
                    raise HLError(f"Invalid boolean spot balance for {token}")
                try:
                    tot = float(raw_tot or 0.0)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise HLError(f"Invalid numeric spot balance for {token}") from exc
                if not math.isfinite(tot) or tot < 0.0:
                    raise HLError(f"Invalid non-finite spot balance for {token}")
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
        try:
            meta, ctxs = self.info.meta_and_asset_ctxs()
            universe = (meta.get("universe") or []) if isinstance(meta, dict) else []
            if not isinstance(ctxs, list):
                return None
            for i, a in enumerate(universe):
                if isinstance(a, dict) and a.get("name") == coin and i < len(ctxs) and isinstance(ctxs[i], dict):
                    raw_f = ctxs[i].get("funding")
                    if isinstance(raw_f, bool):
                        return None
                    rate = float(raw_f or 0.0)
                    return rate if math.isfinite(rate) else None
        except Exception as e:  # noqa: BLE001
            log(f"  ! funding_rate({coin}) failed: {e}")
        return None

    def position_full(self, coin: str) -> dict | None:
        """Return the complete perpetual position including szi, entryPx, liquidationPx, and PnL."""
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

            def _num(raw) -> float:
                if raw is None or isinstance(raw, bool):
                    return 0.0
                try:
                    val = float(raw)
                except (TypeError, ValueError, OverflowError):
                    return 0.0
                return val if math.isfinite(val) else 0.0

            return {
                "accountValue": _num(ms.get("accountValue")),
                "totalMarginUsed": _num(ms.get("totalMarginUsed")),
                "withdrawable": _num(st.get("withdrawable") if isinstance(st, dict) else 0.0),
            }
        except Exception:  # noqa: BLE001
            return {}

    def funding_history(self, start_ms: int) -> list[dict]:
        """Return actual funding payments received or paid since start_ms."""
        if not self.address or start_ms is None or isinstance(start_ms, bool):
            return []
        try:
            start_f = float(start_ms)
            if not math.isfinite(start_f) or start_f < 0:
                return []
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
            or lookback_hours is None or isinstance(lookback_hours, bool)
        ):
            return []
        coin = coin.strip()
        interval = interval.strip()
        try:
            lh_f = float(lookback_hours)
        except (TypeError, ValueError, OverflowError):
            return []
        if not math.isfinite(lh_f) or lh_f <= 0:
            return []
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
        if isinstance(sz, bool) or isinstance(px, bool) or isinstance(sz_decimals, bool):
            return False, None, "invalid boolean order parameter"
        try:
            sz_f = float(sz)
            px_f = float(px)
            dec_f = float(sz_decimals)
            if not math.isfinite(dec_f) or not dec_f.is_integer() or int(dec_f) < 0:
                return False, None, "invalid numeric order parameter"
            dec_i = max(0, min(int(dec_f), 18))
        except (TypeError, ValueError, OverflowError):
            return False, None, "invalid numeric order parameter"
        if not (math.isfinite(sz_f) and sz_f > 0 and math.isfinite(px_f) and px_f > 0):
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
            return True, st["filled"].get("oid"), f"filled {st['filled'].get('totalSz')}"
        if "error" in st:
            return False, None, str(st["error"])
        return True, None, str(st)

    # ----- signed trading -----------------------------------------------------
    def set_leverage(self, coin: str, leverage: int) -> None:
        if not self.exchange:
            return
        if not coin or isinstance(coin, bool) or not isinstance(coin, str) or not coin.strip():
            return
        if leverage is None or isinstance(leverage, bool):
            return
        try:
            lev_f = float(leverage)
            if not math.isfinite(lev_f) or not lev_f.is_integer() or int(lev_f) <= 0:
                return
            lev_i = int(lev_f)
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
        try:
            sz_f = float(sz)
            px_f = float(px)
        except (TypeError, ValueError, OverflowError):
            return False, None, "invalid numeric order parameter"
        if not (math.isfinite(sz_f) and sz_f > 0 and math.isfinite(px_f) and px_f > 0):
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
            return True, st["filled"].get("oid"), f"filled {st['filled'].get('totalSz')} @ {st['filled'].get('avgPx')}"
        if "error" in st:
            return False, None, str(st["error"])
        return True, None, str(st)

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
        try:
            oid_f = float(oid) if isinstance(oid, (int, float)) else float(int(str(oid).strip()))
            if not math.isfinite(oid_f) or not oid_f.is_integer() or int(oid_f) <= 0:
                return False
            oid_i = int(oid_f)
        except (TypeError, ValueError, OverflowError):
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

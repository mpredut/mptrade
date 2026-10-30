#!/usr/bin/env python3
"""
strategy.py — Hyperliquid perpetual DCA and take-profit engine for LONG or SHORT positions.

Generalized by direction (HL_DIRECTION = long | short):
  * LONG: enter by buying below market, DCA as price FALLS, and take profit as price RISES.
  * SHORT: enter by selling above market, DCA as price RISES, and take profit as price FALLS.

Sign convention: sign = +1 (long) / -1 (short).
  open_px  = price * (1 - sign * discount)       (entry/DCA)
  tp_px    = avg   * (1 + sign * takeprofit)      (reduce-only close)
  DCA when : sign * (price - last_open) <= -drop  (price moved AGAINST the position)
  profit   = sign * (price - avg) * qty

Position and average price come from clearinghouseState (signed szi and entryPx).
HL fees are small enough for a tight take profit. HL_LEVERAGE=1 approximates spot.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from botcore import ensure_local_module_priority  # noqa: E402
ensure_local_module_priority(_HERE, "common", "notify", "hl_client", "market_data", "signals")

from common import (
    _parse_bool_flag,
    log, are_close, load_env_stack, required_env, required_float_env,
    required_int_env, required_bool_env, safe_float, safe_int,
    safe_nonneg_float, safe_pos_float, safe_pos_int,
)
from notify import notify
from providers.base import _clean_single_order_id
from state_io import load_json_state, save_json_state
from strategies.state_store import safe_state_token as _safe_state_token
from hl_client import HLClient
from market_data import get_price
from signals import _normalize_signal_trend, get_signal

load_env_stack(os.path.join(_HERE, ".env"))
HL_FEE_PCT = required_float_env("HL_FEE_PCT")


def state_path_for(coin: str, direction: str) -> str:
    safe = _safe_state_token(coin, "unknown")
    dir_safe = _safe_state_token(direction, "long", lower=True)
    return os.path.join(_HERE, f".state_{safe}_{dir_safe}.json")


@dataclass
class StratParams:
    currency: str = "USDC"
    direction: str = "long"            # "long" | "short"
    entry_amount: float = 50.0
    entry_discount_pct: float = 0.2
    dca_amount: float = 30.0
    dca_drop_pct: float = 2.0
    check_minutes: float = 60.0
    takeprofit_pct: float = 1.0
    max_budget: float = 500.0
    max_dca_buys: int = 10
    enable_takeprofit: bool = True
    order_ttl_min: float = 60.0
    signal_gate: bool = False        # enter only when the trend/prediction favors the direction
    stop_loss_pct: float = 0.0       # close all when unrealized loss exceeds this percentage; 0 disables
    reentry_tolerance_pct: float = 0.0  # deterministically treat near-threshold as DCA hit via are_close
                                  # 0 preserves the exact legacy behavior; see kraken/212trading
    fee_pct: float = 0.045
    sz_decimals: int = 4

    @classmethod
    def from_env(cls, env: dict | None = None) -> "StratParams":
        mode = required_env("STRATEGY_MODE", env).strip().lower()
        if mode not in {"avg_tp", "dca_only"}:
            raise ValueError(f"Invalid STRATEGY_MODE: {mode!r}")
        direction = required_env("HL_DIRECTION", env).strip().lower()
        if direction not in ("long", "short"):
            raise ValueError(f"Invalid HL_DIRECTION: {direction!r}")
        return cls(
            currency           = required_env("STRAT_CURRENCY", env).strip().upper(),
            direction          = direction,
            entry_amount       = required_float_env("STRAT_ENTRY", env),
            entry_discount_pct = required_float_env("STRAT_ENTRY_DISCOUNT_PCT", env),
            dca_amount         = required_float_env("STRAT_DCA", env),
            dca_drop_pct       = required_float_env("STRAT_DCA_DROP_PCT", env),
            check_minutes      = required_float_env("STRAT_CHECK_MINUTES", env),
            takeprofit_pct     = required_float_env("STRAT_TAKEPROFIT_PCT", env),
            max_budget         = required_float_env("STRAT_MAX_BUDGET", env),
            max_dca_buys       = required_int_env("STRAT_MAX_DCA_BUYS", env),
            enable_takeprofit  = (mode != "dca_only"),
            order_ttl_min      = required_float_env("STRAT_ORDER_TTL_MIN", env),
            signal_gate        = required_bool_env("HL_SIGNAL_GATE", env),
            stop_loss_pct      = required_float_env("STRAT_STOP_LOSS_PCT", env),
            reentry_tolerance_pct = required_float_env("STRAT_REENTRY_TOLERANCE_PCT", env),
            fee_pct            = required_float_env("HL_FEE_PCT", env),
        )

    @classmethod
    def from_dict(cls, d: dict, direction: str | None = None) -> "StratParams":
        if not isinstance(d, dict):
            raise ValueError("StratParams.from_dict requires a dict")
        raw_dir = direction if direction is not None else d.get("direction", "long")
        if isinstance(raw_dir, bool):
            raise ValueError(f"Invalid HL_DIRECTION: {raw_dir!r}")
        dir_val = str(raw_dir or "long").strip().lower()
        if dir_val not in ("long", "short"):
            raise ValueError(f"Invalid HL_DIRECTION: {dir_val!r}")

        def _f(keys: tuple[str, ...], default: float) -> float:
            raw = default
            for k in keys:
                if k in d and d[k] is not None:
                    raw = d[k]
                    break
            val = safe_float(raw, default=None)
            if val is None:
                raise ValueError(f"invalid float for {keys[0]}: {raw!r}")
            return val

        def _i(keys: tuple[str, ...], default: int) -> int:
            val_f = _f(keys, float(default))
            if not val_f.is_integer():
                raise ValueError(f"non-integer value for {keys[0]}: {val_f!r}")
            return int(val_f)

        def _b(key: str, default: bool) -> bool:
            raw = d.get(key, default)
            if raw is None:
                return default
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                if raw not in (0, 1):
                    raise ValueError(f"invalid numeric boolean for {key}: {raw!r}")
                return bool(raw)
            parsed = _parse_bool_flag(raw, default=None)
            if parsed is None:
                raise ValueError(f"invalid boolean value for {key}: {raw!r}")
            return parsed

        raw_ccy = d.get("currency", "USDC")
        ccy_val = (
            str(raw_ccy).strip().upper()
            if raw_ccy is not None and not isinstance(raw_ccy, bool) and str(raw_ccy).strip()
            else "USDC"
        )
        sz_dec = _i(("sz_decimals",), 4)
        if not (0 <= sz_dec <= 18):
            raise ValueError(f"sz_decimals out of range [0, 18]: {sz_dec}")
        return cls(
            currency=ccy_val,
            direction=dir_val,
            entry_amount=_f(("entry", "entry_amount"), 50.0),
            entry_discount_pct=_f(("disc", "entry_discount_pct"), 0.2),
            dca_amount=_f(("dca", "dca_amount"), 30.0),
            dca_drop_pct=_f(("drop", "dca_drop_pct"), 2.0),
            check_minutes=_f(("check_minutes",), 60.0),
            takeprofit_pct=_f(("tp", "takeprofit_pct"), 1.0),
            max_budget=_f(("budget", "max_budget"), 500.0),
            max_dca_buys=_i(("maxdca", "max_dca_buys"), 10),
            enable_takeprofit=_b("enable_takeprofit", True),
            order_ttl_min=_f(("order_ttl_min",), 60.0),
            signal_gate=_b("signal_gate", False),
            stop_loss_pct=_f(("sl", "stop_loss_pct"), 0.0),
            reentry_tolerance_pct=_f(("reentry_tolerance_pct",), 0.0),
            fee_pct=_f(("fee", "fee_pct"), HL_FEE_PCT),
            sz_decimals=sz_dec,
        )


def _new_state() -> dict:
    return {
        "cycle": 1, "cycles_completed": 0, "wins": 0, "qty": 0.0, "cost": 0.0, "spent": 0.0, "dca_buys": 0,
        "entry_price": None, "last_open_price": None,
        "realized_gross": 0.0, "realized_net": 0.0, "fees_total": 0.0,
        "orders": [],   # {oid, role(open/close), side, sz, px, amount, kind, ts}
    }


_clean_oid = _clean_single_order_id


def _normalize_stored_oid(raw_oid: object) -> int | str | None:
    clean_oid = _clean_oid(raw_oid)
    if clean_oid is None:
        return None
    if isinstance(raw_oid, (int, float)) and not isinstance(raw_oid, bool) and clean_oid.isdigit():
        return int(clean_oid)
    return clean_oid


def _is_live_order_oid(oid: object) -> bool:
    return oid is not None and not str(oid).startswith("PAPER")


def _sanitize_state(raw: dict) -> dict:
    merged = _new_state()
    if isinstance(raw, dict):
        merged.update(raw)
    for k in ("qty", "cost", "spent", "fees_total"):
        merged[k] = safe_nonneg_float(merged.get(k, 0.0), 0.0)
    for k in ("realized_gross", "realized_net"):
        merged[k] = safe_float(merged.get(k, 0.0), 0.0)
    for k, default_i in (("cycle", 1), ("cycles_completed", 0), ("wins", 0), ("dca_buys", 0)):
        vf = safe_float(merged.get(k, default_i), None)
        merged[k] = max(default_i if k == "cycle" else 0, int(vf)) if vf is not None else default_i
    for k in ("entry_price", "last_open_price"):
        merged[k] = safe_pos_float(merged.get(k), None)
    raw_orders = merged.get("orders")
    clean_orders = []
    if isinstance(raw_orders, list):
        for o in raw_orders:
            if not isinstance(o, dict) or o.get("role") not in ("open", "close"):
                continue
            stored_oid = _normalize_stored_oid(o.get("oid"))
            if stored_oid is None:
                continue
            px_f = safe_pos_float(o.get("px"), 0.0)
            sz_f = safe_pos_float(o.get("sz"), 0.0)
            if px_f <= 0.0 or sz_f <= 0.0:
                continue
            o_copy = dict(o)
            o_copy["oid"] = stored_oid
            o_copy["px"] = px_f
            o_copy["sz"] = sz_f
            o_copy["amount"] = safe_nonneg_float(o_copy.get("amount", 0.0), 0.0)
            if "ts" in o_copy and o_copy.get("ts") is not None:
                o_copy["ts"] = safe_nonneg_float(o_copy.get("ts"), 0.0)
            if "dca_counted" in o_copy:
                o_copy["dca_counted"] = _parse_bool_flag(o_copy.get("dca_counted"), False)
            clean_orders.append(o_copy)
    merged["orders"] = clean_orders
    return merged


class Strategy:
    def __init__(self, client: HLClient | None, coin: str, params: StratParams,
                 dry_run: bool = True, desktop: bool = False, leverage: int = 1,
                 initial_state: dict | None = None, replay_mode: bool = False):
        self.client = client
        self.coin = coin
        self.p = params
        self.ccy = params.currency
        self.dry_run = dry_run
        self.desktop = desktop
        self.leverage = safe_pos_int(leverage, default=1)
        self.replay_mode = replay_mode
        dir_norm = str(getattr(params, "direction", "long") or "long").strip().lower()
        if dir_norm not in ("long", "short"):
            raise ValueError(f"Invalid Strategy direction: {params.direction!r}")
        self.p.direction = dir_norm
        self.sign = 1 if dir_norm == "long" else -1
        self.open_side = "buy" if self.sign > 0 else "sell"     # open or increase the position
        self.close_side = "sell" if self.sign > 0 else "buy"    # reduce-only close
        self.state_file = state_path_for(coin, dir_norm)
        if initial_state is not None:
            self.s = _sanitize_state(initial_state)
        elif replay_mode:
            self.s = _new_state()
        else:
            self.s = self._load()
        if replay_mode:
            self._save = lambda: None
        self._paper_seq = 0
        default_sz_dec = 4 if replay_mode else 2
        dec_i = safe_int(getattr(params, "sz_decimals", default_sz_dec), default=None)
        self.sz_dec = max(0, min(18, dec_i)) if dec_i is not None else default_sz_dec
        if client is not None:
            try:
                c_dec_i = safe_int(client.sz_decimals(coin), default=None)
                if c_dec_i is not None:
                    self.sz_dec = max(0, min(18, c_dec_i))
            except Exception:
                pass

    # -- persistence -----------------------------------------------------------
    def _load(self) -> dict:
        st = load_json_state(
            self.state_file, default_factory=dict, fail_closed=not self.dry_run,
            label="Hyperliquid strategy",
        )
        merged = _sanitize_state(st)
        if st:
            log(f"  [STRAT] state loaded (cycle {merged.get('cycle')}, qty {merged.get('qty')})")
        return merged

    def _save(self) -> None:
        try:
            save_json_state(self.state_file, self.s, indent=2)
        except OSError as e:
            log(f"  ! [STRAT] cannot save: {e}")

    # -- helpers ---------------------------------------------------------------
    def _avg(self) -> float | None:
        return self.s["cost"] / self.s["qty"] if self.s["qty"] > 1e-12 else None

    def _sz_for(self, amount: float, price: float) -> float:
        return round(amount / price, self.sz_dec) if price > 0 else 0.0

    def _open_pending(self) -> bool:
        return any(o["role"] == "open" for o in self.s["orders"])

    def _close_order(self) -> dict | None:
        return next((o for o in self.s["orders"] if o["role"] == "close"), None)

    def _remove(self, o: dict) -> None:
        if o in self.s["orders"]:
            self.s["orders"].remove(o)

    # -- placement -------------------------------------------------------------
    def _place(self, role: str, sz: float, px: float, kind: str, amount: float = 0.0) -> None:
        sz_f = safe_pos_float(sz, default=None)
        px_f = safe_pos_float(px, default=None)
        if sz_f is None or px_f is None:
            if not self.replay_mode:
                log("  ! [STRAT] invalid sz/px — sar")
            return
        sz = round(sz_f, self.sz_dec)
        px = px_f
        if sz <= 0:
            if not self.replay_mode:
                log("  ! [STRAT] sz 0 — sar")
            return
        side = self.open_side if role == "open" else self.close_side
        reduce_only = (role == "close")
        if self.dry_run or self.replay_mode:
            self._paper_seq += 1
            if not self.replay_mode:
                log(f"  [STRAT] [PAPER] {role.upper()}({side}) {kind} {sz} @ {px:.4f} {self.ccy}")
            self.s["orders"].append({"oid": f"PAPER-{self._paper_seq}", "role": role, "side": side,
                                     "sz": sz, "px": px, "amount": amount, "kind": kind, "ts": time.time()})
            return
        ok, oid, msg = self.client.place_limit(self.coin, side == "buy", sz, px, reduce_only=reduce_only)
        stored_oid = _normalize_stored_oid(oid)
        if ok and stored_oid is not None:
            log(f"  [STRAT] {role.upper()}({side}) {kind} placed oid={stored_oid} {sz} @ {px:.4f} ({msg})")
            self.s["orders"].append({"oid": stored_oid, "role": role, "side": side, "sz": sz, "px": px,
                                     "amount": amount, "kind": kind, "ts": time.time()})
        else:
            log(f"  ! [STRAT] {role} {kind} failed: {msg}")

    def _cancel_close(self) -> bool:
        o = self._close_order()
        if not o:
            return True
        oid = o.get("oid")
        if not self.dry_run and not self.replay_mode and _is_live_order_oid(oid):
            if self.client.cancel(self.coin, oid) is False:
                if not self.replay_mode:
                    log(f"  ! [STRAT] failed to cancel close {oid} — keeping tracked")
                return False
        self._remove(o)
        if not self.replay_mode:
            log(f"  [STRAT] cancelled close {oid}")
        return True

    # -- reconciliation --------------------------------------------------------
    def reconcile(self, price: float) -> bool:
        price_f = safe_pos_float(price, default=None) if isinstance(price, (int, float)) else None
        if price_f is None:
            return False
        if self.dry_run:
            self._reconcile_paper(price_f)
            return True
        return self._reconcile_real(price_f)

    def _reconcile_paper(self, price: float) -> None:
        for role in ("open", "close"):
            for o in [x for x in self.s["orders"] if x["role"] == role]:
                if o not in self.s["orders"]:
                    continue
                # Paper opens and STOP orders fill immediately; TP closes fill when signed price reaches target.
                if role == "open" or o.get("kind") == "STOP" or self.sign * (price - o["px"]) >= 0:
                    self._remove(o)
                    if role == "open":
                        self._apply_open(o["sz"], o["px"], o.get("amount", 0.0), o.get("kind"))
                    else:
                        fill_px = price if o.get("kind") == "STOP" else o["px"]
                        self._apply_close(self._avg() or fill_px, fill_px, o["sz"])

    def _reconcile_real(self, price: float) -> bool:
        pos_fn = getattr(self.client, "position_strict", None) or self.client.position
        try:
            szi_raw, entry_raw = pos_fn(self.coin)
        except Exception as e:
            log(f"  [STRAT] pozitie indisponibila ({e})")
            return False
        szi = safe_float(szi_raw, default=None)
        entry = safe_nonneg_float(entry_raw, default=None)
        if szi is None or entry is None:
            log(f"  ! [STRAT] invalid position state on {self.coin} (szi={szi_raw}, entry={entry_raw}); skipping step")
            return False

        if abs(szi) > 1e-9 and szi * self.sign < 0:
            log(f"  ! [STRAT] opposite-direction position detected on {self.coin} (szi={szi}); skipping step")
            return False

        qty_now = abs(szi)
        orders_fn = getattr(self.client, "open_orders_strict", None) or self.client.open_orders
        orders_ok = True
        try:
            active = {
                clean_id
                for o in (orders_fn(self.coin) or [])
                if isinstance(o, dict) and (clean_id := _clean_oid(o.get("oid"))) is not None
            }
        except Exception as e:
            orders_ok = False
            log(f"  [STRAT] open_orders unavailable ({e}); preserving tracked order state")
            active = {
                clean_id
                for o in self.s["orders"]
                if isinstance(o, dict) and (clean_id := _clean_oid(o.get("oid"))) is not None
            }

        prev = self.s["qty"]
        prev_cost = safe_nonneg_float(self.s.get("cost", 0.0), default=0.0)

        if qty_now > prev + 1e-9:        # position opened or increased
            fq = qty_now - prev
            filled_open = next(
                (o for o in self.s["orders"] if o.get("role") == "open" and _clean_oid(o.get("oid")) not in active),
                None,
            ) or next(
                (o for o in self.s["orders"] if o.get("role") == "open"),
                None,
            )
            filled_open_px = safe_pos_float(filled_open.get("px"), default=None) if filled_open else None
            if prev > 1e-9:
                marginal_px = ((qty_now * entry - prev_cost) / fq) if (entry > 0 and prev_cost > 0) else 0.0
                fp = marginal_px if marginal_px > 0 else (filled_open_px or price)
            else:
                fp = entry if entry > 0 else (filled_open_px or price)
            open_kind = filled_open.get("kind") if filled_open else None
            if open_kind == "ENTRY":
                count_dca = False
            elif filled_open is not None:
                count_dca = not _parse_bool_flag(filled_open.get("dca_counted"), False)
                filled_open["dca_counted"] = True
            else:
                count_dca = None
            self._apply_open(
                fq, fp, round(fq * fp, 2),
                open_kind,
                real_qty=qty_now, real_avg=entry if entry > 0 else fp,
                count_dca=count_dca,
            )
        elif qty_now < prev - 1e-9:      # position reduced by take profit
            filled_close = next(
                (o for o in self.s["orders"] if o.get("role") == "close" and _clean_oid(o.get("oid")) not in active),
                None,
            )
            filled_close_px = (
                safe_pos_float(filled_close.get("px"), default=None)
                if filled_close and filled_close.get("kind") != "STOP"
                else None
            )
            close_px = filled_close_px or price
            self._apply_close(self._avg() or entry or close_px, close_px, prev - qty_now,
                              real_qty=qty_now, real_avg=entry)
        else:
            if qty_now <= 1e-12:
                self.s["qty"] = 0.0
                self.s["cost"] = 0.0
                self.s["spent"] = 0.0
            else:
                self.s["qty"] = qty_now
                if entry > 0:
                    self.s["cost"] = qty_now * entry

        for o in list(self.s["orders"]):
            if not _is_live_order_oid(o.get("oid")): continue
            if _clean_oid(o.get("oid")) not in active:
                self._remove(o)
            elif orders_ok and self._open_is_stale(o, price):
                log(f"  [STRAT] open {o.get('oid')} unfilled, the price ran away — cancelling and re-placing")
                if self.client.cancel(self.coin, o["oid"]) is not False:
                    self._remove(o)
        return True

    def _open_is_stale(self, order: dict, price: float, elapsed_min: float | None = None) -> bool:
        """Return whether an unfilled open limit order has expired and price ran away."""
        if not isinstance(order, dict) or order.get("role") != "open":
            return False
        ord_px = safe_pos_float(order.get("px"), default=None)
        if ord_px is None:
            return False
        if elapsed_min is None:
            ts_f = safe_float(order.get("ts"), default=0.0)
            elapsed_min = (time.time() - ts_f) / 60.0
        el_f = safe_float(elapsed_min, default=None)
        if el_f is None or el_f <= self.p.order_ttl_min:
            return False
        return self.sign * (price - ord_px) > 0.003 * ord_px

    def _apply_open(self, fq, fp, amount, kind, real_qty=None, real_avg=None, *, count_dca=None):
        is_dca = (self.s["qty"] > 1e-9) if count_dca is None else bool(count_dca)
        self.s["last_open_price"] = fp
        if self.s["entry_price"] is None: self.s["entry_price"] = fp
        if is_dca: self.s["dca_buys"] += 1
        if real_qty is not None:
            eff_avg = real_avg if (real_avg is not None and real_avg > 0) else fp
            self.s["qty"] = real_qty; self.s["cost"] = real_qty * eff_avg
            self.s["spent"] = round(real_qty * eff_avg, 2)
        else:
            self.s["qty"] += fq; self.s["cost"] += fq * fp
            amt_f = safe_pos_float(amount, default=0.0)
            self.s["spent"] += amt_f if amt_f > 0.0 else fq * fp
        avg = self._avg()
        tag = "[PAPER] " if self.dry_run else ""
        if not self.replay_mode:
            log(f"  [STRAT] {tag}OPEN {self.p.direction.upper()} {fq:.6f} @ {fp:.4f} "
                f"({'DCA' if is_dca else 'ENTRY'}) qty={self.s['qty']:.6f} avg={avg:.4f}")
            notify(title=f"{tag}{self.coin} OPEN {self.p.direction} {fq:.2f}@{fp:.2f}",
                   body=f"{'DCA' if is_dca else 'ENTRY'} | q{self.s['qty']:.2f} a{avg:.2f}",
                   source="hyperliquid", price=fp, desktop=self.desktop)
        self._cancel_close()   # average changed; replace the take-profit order

    def _apply_close(self, avg, price, sz, real_qty=None, real_avg=None):
        gross = self.sign * (price - avg) * sz
        fee_pct = getattr(self.p, "fee_pct", HL_FEE_PCT)
        fee = (fee_pct / 100.0) * (avg * sz + price * sz)
        net = gross - fee
        self.s["realized_gross"] += gross; self.s["realized_net"] += net; self.s["fees_total"] += fee
        if gross > 0:
            self.s["wins"] = self.s.get("wins", 0) + 1
        tag = "[PAPER] " if self.dry_run else ""
        if not self.replay_mode:
            log(f"  [STRAT] {tag}CLOSE {sz:.6f} @ {price:.4f}  brut={gross:+.4f} fee={fee:.4f} net={net:+.4f}")
            notify(title=f"{tag}{self.coin} CLOSE {sz:.2f}@{price:.2f} N{net:+.2f}",
                   body=f"a{avg:.2f} · br{gross:+.2f} fee{fee:.2f} N{net:+.2f} | Ntot{self.s['realized_net']:+.2f}",
                   source="hyperliquid", price=price, desktop=self.desktop)
        if real_qty is not None:
            eff_avg = safe_pos_float(real_avg, default=avg)
            self.s["qty"] = real_qty
            self.s["cost"] = max(0.0, real_qty * (eff_avg or 0.0))
            if self.s["qty"] > 1e-12:
                self.s["spent"] = round(self.s["cost"], 2)
        else:
            prev_qty = self.s["qty"]
            self.s["qty"] -= sz
            self.s["cost"] = max(0.0, self.s["cost"] - sz * avg)
            if self.s["qty"] > 1e-12 and prev_qty > 0:
                self.s["spent"] = max(0.0, self.s["spent"] * (self.s["qty"] / prev_qty))
        if self.s["qty"] <= 1e-12:
            self.s["qty"] = 0.0
            self.s["cost"] = 0.0
            self.s["spent"] = 0.0
            self.s["cycles_completed"] = self.s.get("cycles_completed", 0) + 1
            remaining_orders = []
            for o in list(self.s.get("orders", [])):
                if not self.dry_run and not self.replay_mode and self.client and _is_live_order_oid(o.get("oid")):
                    try:
                        if not self.client.cancel(self.coin, o["oid"]):
                            remaining_orders.append(o)
                    except Exception as e:
                        log(f"  ! [STRAT] failed to cancel leftover order {o.get('oid')}: {e}")
                        remaining_orders.append(o)
            keep = (
                self.s["realized_gross"],
                self.s["realized_net"],
                self.s["fees_total"],
                self.s.get("cycle", 1) + 1,
                self.s["cycles_completed"],
                self.s.get("wins", 0),
            )
            self.s = _new_state()
            self.s["orders"] = remaining_orders
            (
                self.s["realized_gross"],
                self.s["realized_net"],
                self.s["fees_total"],
                self.s["cycle"],
                self.s["cycles_completed"],
                self.s["wins"],
            ) = keep
            if not self.replay_mode:
                log(f"  [STRAT] === ciclu inchis, reincep (ciclu {self.s['cycle']}) ===")

    def _check_stop_loss(self, price: float) -> bool:
        """Close the entire position when unrealized loss exceeds the configured threshold."""
        if self.p.stop_loss_pct <= 0:
            return False
        avg = self._avg()
        if not avg:
            return False
        loss_pct = -self.sign * (price - avg) / avg * 100   # positive while losing
        if loss_pct >= self.p.stop_loss_pct - 1e-9:
            if not self.replay_mode:
                log(f"  🛑 [STRAT] STOP-LOSS: loss {loss_pct:.2f}% >= {self.p.stop_loss_pct}% — CLOSING everything (cutting the loss)")
            # Cancel every pending order, including DCA, so the position cannot refill.
            close_cancel_failed = False
            for o in list(self.s["orders"]):
                oid = o.get("oid")
                if not self.dry_run and not self.replay_mode and _is_live_order_oid(oid):
                    if self.client.cancel(self.coin, oid) is False:
                        log(f"  ! [STRAT] failed to cancel {oid} before SL — keeping tracked")
                        if o.get("role") == "close":
                            close_cancel_failed = True
                        continue
                self._remove(o)
            if self.replay_mode:
                self._apply_close(avg, price, self.s["qty"])
                return True
            if close_cancel_failed:
                return True
            agg = price * (1 - self.sign * 0.005)           # aggressive price for reliable fill
            self._place("close", self.s["qty"], agg, "STOP")
            if not self.replay_mode:
                notify(title=f"🛑 SL {self.coin} -{loss_pct:.1f}%",
                       body=f"loss {loss_pct:.1f}% >=threshold{self.p.stop_loss_pct}% — position closed",
                       source="hyperliquid", price=price, desktop=self.desktop)
            return True
        return False

    # -- decision --------------------------------------------------------------
    def step(self, price: float, signal: str | None = None) -> None:
        if (safe_pos_float(price, default=None) if isinstance(price, (int, float)) else None) is None:
            if not self.replay_mode:
                log("  ! [STRAT] invalid price — deferring step")
            return
        held = self.s["qty"]
        d = self.p.entry_discount_pct / 100
        if held <= 1e-12:
            if self._open_pending(): return
            # SIGNAL GATE: do not enter against the trend or prediction.
            if self.p.signal_gate or (signal is not None and signal != "__any__"):
                if signal is not None:
                    trend = signal
                else:
                    sig = get_signal(self.client, self.coin)
                    trend = sig.get("trend") if isinstance(sig, dict) else sig
                if trend != "__any__":
                    trend = _normalize_signal_trend(trend)
                want = "up" if self.sign > 0 else "down"   # long requires up; short requires down
                if trend != "__any__" and trend != want:
                    if not self.replay_mode:
                        log(f"  [STRAT] semnal={trend} != {want} for {self.p.direction} — NOT entering (waiting for a favourable trend)")
                    return
                if not self.replay_mode:
                    log(f"  [STRAT] semnal={trend} favorabil pt {self.p.direction} — intru")
            if self.s["spent"] + self.p.entry_amount > self.p.max_budget:
                if not self.replay_mode:
                    log(f"  [STRAT] plafon {self.p.max_budget} {self.ccy} reached")
                return
            px = price * (1 - self.sign * d)
            self._place("open", self._sz_for(self.p.entry_amount, px), px, "ENTRY", self.p.entry_amount)
            return
        # STOP-LOSS: cap losses when the position moves too far against us, preventing runaway DCA.
        if self._check_stop_loss(price):
            return
        avg = self._avg()
        if self.p.enable_takeprofit and avg:
            target = avg * (1 + self.sign * self.p.takeprofit_pct/100)
            if target > 0:
                o = self._close_order()
                if o is None:
                    self._place("close", held, target, "TP")
                elif abs(o["px"]-target)/target > 0.001 or abs(o["sz"]-held) > 1e-9:
                    if self._cancel_close():
                        self._place("close", held, target, "TP")
        # DCA after price moves drop% AGAINST the position. Compare PRICE with a PRICE
        # threshold, not a percentage with a percentage: are_close is relative to the
        # magnitude of compared values and is correct only between two prices, matching
        # kraken/212strategy.py. The tolerance treats near-threshold as reached;
        # STRAT_REENTRY_TOLERANCE_PCT=0 preserves exact legacy behavior.
        lop = self.s["last_open_price"]
        moved = self.sign * (price - lop) / lop if lop else 0   # logging only
        prag_dca = lop * (1 - self.sign * self.p.dca_drop_pct / 100) if lop else None
        dca_hit = prag_dca is not None and (
            self.sign * (prag_dca - price) >= 0
            or (self.p.reentry_tolerance_pct > 0
                and are_close(price, prag_dca, self.p.reentry_tolerance_pct)))
        if (self.s["dca_buys"] < self.p.max_dca_buys and self.s["last_open_price"]
                and dca_hit
                and self.s["spent"] + self.p.dca_amount <= self.p.max_budget
                and not self._open_pending()):
            px = price * (1 - self.sign * d)
            if not self.replay_mode:
                log(f"  [STRAT] the price moved against us ({moved*100:.2f}%) — DCA")
            self._place("open", self._sz_for(self.p.dca_amount, px), px, "DCA", self.p.dca_amount)

    # -- loop ------------------------------------------------------------------
    def run(self) -> None:
        mode = "avg_tp" if self.p.enable_takeprofit else "dca_only"
        if not self.dry_run and self.client is not None and self.client.exchange:
            self.client.set_leverage(self.coin, self.leverage)
        fee_hl = getattr(self.p, "fee_pct", HL_FEE_PCT)
        log("  === STRATEGIE HYPERLIQUID PORNITA ===")
        log(f"      coin       : {self.coin} perp  DIRECTIE={self.p.direction.upper()} levier {self.leverage}x  {'[PAPER]' if self.dry_run else '⚠ REAL'}")
        log(f"      mod        : {mode}")
        log(f"      intrare    : {self.p.entry_amount} {self.ccy} @ market{'-' if self.sign>0 else '+'}{self.p.entry_discount_pct}%")
        log(f"      DCA        : {self.p.dca_amount} {self.ccy} la {self.p.dca_drop_pct}% contra (max {self.p.max_dca_buys})")
        log(f"      take-profit: {self.p.takeprofit_pct}% in favoare" if self.p.enable_takeprofit else "      TP: off")
        log(f"      PLAFON     : {self.p.max_budget} {self.ccy} / ciclu  |  fee HL ~{fee_hl}%/leg")
        sleep_sec = max(1.0, safe_pos_float(self.p.check_minutes, default=1.0) * 60.0)
        try:
            while True:
                try:
                    price = get_price(self.client, self.coin)
                    if price is None:
                        log("  [STRAT] price unavailable"); time.sleep(sleep_sec); continue
                    if self.reconcile(price) is False:
                        time.sleep(sleep_sec); continue
                    self.step(price); self._save()
                    avg = self._avg()
                    pos = f"qty={self.s['qty']:.6f} avg={avg:.4f}" if avg else "qty=0 (waiting for an entry)"
                    log(f"  [STRAT] price={price:.4f} [{self.p.direction}]  {pos}  "
                        f"NET={self.s['realized_net']:+.2f} (brut {self.s['realized_gross']:+.2f}, fee {self.s['fees_total']:.2f}) {self.ccy}  ord={len(self.s['orders'])}")
                except KeyboardInterrupt:
                    raise
                except Exception as e:  # noqa: BLE001
                    log(f"  ! [STRAT] cycle failed ({e.__class__.__name__}: {e}) — retrying on next check")
                    self._save()
                time.sleep(sleep_sec)
        except KeyboardInterrupt:
            log("  [STRAT] stopped manually."); self._save()

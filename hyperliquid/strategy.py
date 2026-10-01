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

import json
import os
import time
from dataclasses import dataclass

from common import (
    log, now_str, are_close, required_env, required_float_env,
    required_int_env, required_bool_env,
)
from notify import notify
from state_io import atomic_write_json, load_json_state
from hl_client import HLClient, HLError
from market_data import get_price
from signals import get_signal

_HERE = os.path.dirname(os.path.abspath(__file__))


def state_path_for(coin: str, direction: str) -> str:
    safe = "".join(c for c in coin if c.isalnum() or c in "._-")
    return os.path.join(_HERE, f".state_{safe}_{direction}.json")


@dataclass
class StratParams:
    currency: str
    direction: str            # "long" | "short"
    entry_amount: float
    entry_discount_pct: float
    dca_amount: float
    dca_drop_pct: float
    check_minutes: float
    takeprofit_pct: float
    max_budget: float
    max_dca_buys: int
    enable_takeprofit: bool
    order_ttl_min: float
    signal_gate: bool        # enter only when the trend/prediction favors the direction
    stop_loss_pct: float     # close all when unrealized loss exceeds this percentage; 0 disables
    reentry_tolerance_pct: float  # deterministically treat near-threshold as DCA hit via are_close
                                  # 0 preserves the exact legacy behavior; see kraken/212trading

    @classmethod
    def from_env(cls) -> "StratParams":
        mode = required_env("STRATEGY_MODE").lower()
        if mode not in {"avg_tp", "dca_only"}:
            raise ValueError(f"Invalid STRATEGY_MODE: {mode!r}")
        direction = required_env("HL_DIRECTION").lower()
        if direction not in ("long", "short"):
            raise ValueError(f"Invalid HL_DIRECTION: {direction!r}")
        return cls(
            currency           = required_env("STRAT_CURRENCY").upper(),
            direction          = direction,
            entry_amount       = required_float_env("STRAT_ENTRY"),
            entry_discount_pct = required_float_env("STRAT_ENTRY_DISCOUNT_PCT"),
            dca_amount         = required_float_env("STRAT_DCA"),
            dca_drop_pct       = required_float_env("STRAT_DCA_DROP_PCT"),
            check_minutes      = required_float_env("STRAT_CHECK_MINUTES"),
            takeprofit_pct     = required_float_env("STRAT_TAKEPROFIT_PCT"),
            max_budget         = required_float_env("STRAT_MAX_BUDGET"),
            max_dca_buys       = required_int_env("STRAT_MAX_DCA_BUYS"),
            enable_takeprofit  = (mode != "dca_only"),
            order_ttl_min      = required_float_env("STRAT_ORDER_TTL_MIN"),
            signal_gate        = required_bool_env("HL_SIGNAL_GATE"),
            stop_loss_pct      = float(os.environ.get("STRAT_STOP_LOSS_PCT", "0.0")),
            reentry_tolerance_pct = required_float_env("STRAT_REENTRY_TOLERANCE_PCT"),
        )


def _new_state() -> dict:
    return {
        "cycle": 1, "qty": 0.0, "cost": 0.0, "spent": 0.0, "dca_buys": 0,
        "entry_price": None, "last_open_price": None,
        "realized_gross": 0.0, "realized_net": 0.0, "fees_total": 0.0,
        "orders": [],   # {oid, role(open/close), side, sz, px, amount, kind, ts}
    }


class Strategy:
    def __init__(self, client: HLClient, coin: str, params: StratParams,
                 dry_run: bool = True, desktop: bool = False, leverage: int = 1):
        self.client = client
        self.coin = coin
        self.p = params
        self.ccy = params.currency
        self.dry_run = dry_run
        self.desktop = desktop
        self.leverage = leverage
        self.sign = 1 if params.direction == "long" else -1
        self.open_side = "buy" if self.sign > 0 else "sell"     # open or increase the position
        self.close_side = "sell" if self.sign > 0 else "buy"    # reduce-only close
        self.state_file = state_path_for(coin, params.direction)
        self.s = self._load()
        self._paper_seq = 0
        self.sz_dec = 2
        try:
            self.sz_dec = client.sz_decimals(coin)
        except HLError:
            pass

    # -- persistence -----------------------------------------------------------
    def _load(self) -> dict:
        st = load_json_state(
            self.state_file, default_factory=dict, fail_closed=not self.dry_run,
            label="Hyperliquid strategy",
        )
        merged = _new_state()
        merged.update(st)
        if st:
            log(f"  [STRAT] state loaded (cycle {merged.get('cycle')}, qty {merged.get('qty')})")
        return merged

    def _save(self) -> None:
        try:
            atomic_write_json(self.state_file, self.s, indent=2)
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
        sz = round(sz, self.sz_dec)
        if sz <= 0:
            log("  ! [STRAT] sz 0 — sar"); return
        side = self.open_side if role == "open" else self.close_side
        reduce_only = (role == "close")
        if self.dry_run:
            self._paper_seq += 1
            log(f"  [STRAT] [PAPER] {role.upper()}({side}) {kind} {sz} @ {px:.4f} {self.ccy}")
            self.s["orders"].append({"oid": f"PAPER-{self._paper_seq}", "role": role, "side": side,
                                     "sz": sz, "px": px, "amount": amount, "kind": kind, "ts": time.time()})
            return
        ok, oid, msg = self.client.place_limit(self.coin, side == "buy", sz, px, reduce_only=reduce_only)
        if ok:
            log(f"  [STRAT] {role.upper()}({side}) {kind} placed oid={oid} {sz} @ {px:.4f} ({msg})")
            self.s["orders"].append({"oid": oid, "role": role, "side": side, "sz": sz, "px": px,
                                     "amount": amount, "kind": kind, "ts": time.time()})
        else:
            log(f"  ! [STRAT] {role} {kind} failed: {msg}")

    def _cancel_close(self) -> None:
        o = self._close_order()
        if not o:
            return
        if not self.dry_run and not str(o["oid"]).startswith("PAPER"):
            self.client.cancel(self.coin, o["oid"])
        self._remove(o)
        log(f"  [STRAT] cancelled close {o['oid']}")

    # -- reconciliation --------------------------------------------------------
    def reconcile(self, price: float) -> None:
        if self.dry_run:
            self._reconcile_paper(price)
        else:
            self._reconcile_real(price)

    def _reconcile_paper(self, price: float) -> None:
        for role in ("open", "close"):
            for o in [x for x in self.s["orders"] if x["role"] == role]:
                if o not in self.s["orders"]:
                    continue
                # Paper opens fill immediately; closes fill when signed price reaches target.
                if role == "open" or self.sign * (price - o["px"]) >= 0:
                    self._remove(o)
                    if role == "open":
                        self._apply_open(o["sz"], o["px"], o.get("amount", 0.0), o.get("kind"))
                    else:
                        self._apply_close(self._avg() or o["px"], o["px"], o["sz"])

    def _reconcile_real(self, price: float) -> None:
        try:
            szi, entry = self.client.position(self.coin)
        except HLError as e:
            log(f"  [STRAT] pozitie indisponibila ({e})"); return
        qty_now = abs(szi)
        active = {o.get("oid") for o in self.client.open_orders(self.coin)}
        prev = self.s["qty"]

        if qty_now > prev + 1e-9:        # position opened or increased
            fp = entry if entry > 0 else price
            self._apply_open(qty_now - prev, fp, round((qty_now - prev) * fp, 2), None,
                             real_qty=qty_now, real_avg=entry)
        elif qty_now < prev - 1e-9:      # position reduced by take profit
            self._apply_close(self._avg() or entry or price, price, prev - qty_now,
                              real_qty=qty_now, real_avg=entry)
        else:
            self.s["qty"] = qty_now
            if qty_now > 1e-12: self.s["cost"] = qty_now * entry

        for o in list(self.s["orders"]):
            if str(o["oid"]).startswith("PAPER"): continue
            if o["oid"] not in active:
                self._remove(o)
            elif o["role"] == "open" and (time.time()-o.get("ts",0))/60 > self.p.order_ttl_min \
                    and self.sign*(price - o["px"]) > 0.003*o["px"]:
                log(f"  [STRAT] open {o['oid']} unfilled, the price ran away — cancelling and re-placing")
                self.client.cancel(self.coin, o["oid"]); self._remove(o)

    def _apply_open(self, fq, fp, amount, kind, real_qty=None, real_avg=None):
        is_dca = self.s["qty"] > 1e-9
        self.s["last_open_price"] = fp
        if self.s["entry_price"] is None: self.s["entry_price"] = fp
        if is_dca: self.s["dca_buys"] += 1
        if real_qty is not None:
            self.s["qty"] = real_qty; self.s["cost"] = real_qty * real_avg
            self.s["spent"] = round(real_qty * real_avg, 2)
        else:
            self.s["qty"] += fq; self.s["cost"] += fq * fp
            self.s["spent"] += amount or fq * fp
        avg = self._avg()
        tag = "[PAPER] " if self.dry_run else ""
        log(f"  [STRAT] {tag}OPEN {self.p.direction.upper()} {fq:.6f} @ {fp:.4f} "
            f"({'DCA' if is_dca else 'ENTRY'}) qty={self.s['qty']:.6f} avg={avg:.4f}")
        notify(title=f"{tag}{self.coin} OPEN {self.p.direction} {fq:.2f}@{fp:.2f}",
               body=f"{'DCA' if is_dca else 'ENTRY'} | q{self.s['qty']:.2f} a{avg:.2f}",
               source="hyperliquid", price=fp, desktop=self.desktop)
        self._cancel_close()   # average changed; replace the take-profit order

    def _apply_close(self, avg, price, sz, real_qty=None, real_avg=None):
        gross = self.sign * (price - avg) * sz
        fee = (HL_FEE_PCT/100.0) * (avg*sz + price*sz)
        net = gross - fee
        self.s["realized_gross"] += gross; self.s["realized_net"] += net; self.s["fees_total"] += fee
        tag = "[PAPER] " if self.dry_run else ""
        log(f"  [STRAT] {tag}CLOSE {sz:.6f} @ {price:.4f}  brut={gross:+.4f} fee={fee:.4f} net={net:+.4f}")
        notify(title=f"{tag}{self.coin} CLOSE {sz:.2f}@{price:.2f} N{net:+.2f}",
               body=f"a{avg:.2f} · br{gross:+.2f} fee{fee:.2f} N{net:+.2f} | Ntot{self.s['realized_net']:+.2f}",
               source="hyperliquid", price=price, desktop=self.desktop)
        if real_qty is not None:
            self.s["qty"] = real_qty; self.s["cost"] = real_qty * (real_avg or 0)
        else:
            self.s["qty"] -= sz
        if self.s["qty"] <= 1e-12:
            keep = (self.s["realized_gross"], self.s["realized_net"], self.s["fees_total"], self.s.get("cycle",1)+1)
            self.s = _new_state()
            (self.s["realized_gross"], self.s["realized_net"], self.s["fees_total"], self.s["cycle"]) = keep
            log(f"  [STRAT] === ciclu inchis, reincep (ciclu {self.s['cycle']}) ===")

    def _check_stop_loss(self, price: float) -> bool:
        """Close the entire position when unrealized loss exceeds the configured threshold."""
        if self.p.stop_loss_pct <= 0:
            return False
        avg = self._avg()
        if not avg:
            return False
        loss_pct = -self.sign * (price - avg) / avg * 100   # positive while losing
        if loss_pct >= self.p.stop_loss_pct:
            log(f"  🛑 [STRAT] STOP-LOSS: loss {loss_pct:.2f}% >= {self.p.stop_loss_pct}% — CLOSING everything (cutting the loss)")
            # Cancel every pending order, including DCA, so the position cannot refill.
            for o in list(self.s["orders"]):
                if not self.dry_run and not str(o["oid"]).startswith("PAPER"):
                    self.client.cancel(self.coin, o["oid"])
                self._remove(o)
            agg = price * (1 - self.sign * 0.005)           # aggressive price for reliable fill
            self._place("close", self.s["qty"], agg, "STOP")
            notify(title=f"🛑 SL {self.coin} -{loss_pct:.1f}%",
                   body=f"loss {loss_pct:.1f}% >=threshold{self.p.stop_loss_pct}% — position closed",
                   source="hyperliquid", price=price, desktop=self.desktop)
            return True
        return False

    # -- decision --------------------------------------------------------------
    def step(self, price: float) -> None:
        held = self.s["qty"]
        d = self.p.entry_discount_pct / 100
        if held <= 1e-12:
            if self._open_pending(): return
            # SIGNAL GATE: do not enter against the trend or prediction.
            if self.p.signal_gate:
                sig = get_signal(self.client, self.coin)
                want = "up" if self.sign > 0 else "down"   # long requires up; short requires down
                if sig["trend"] != want:
                    log(f"  [STRAT] semnal={sig['trend']} (conf {sig['confidence']}, {sig['source']}) "
                        f"!= {want} for {self.p.direction} — NOT entering (waiting for a favourable trend)")
                    return
                log(f"  [STRAT] semnal={sig['trend']} favorabil pt {self.p.direction} — intru")
            if self.s["spent"] + self.p.entry_amount > self.p.max_budget:
                log(f"  [STRAT] plafon {self.p.max_budget} {self.ccy} reached"); return
            px = price * (1 - self.sign * d)
            self._place("open", self._sz_for(self.p.entry_amount, px), px, "ENTRY", self.p.entry_amount)
            return
        # STOP-LOSS: cap losses when the position moves too far against us, preventing runaway DCA.
        if self._check_stop_loss(price):
            return
        avg = self._avg()
        if self.p.enable_takeprofit and avg:
            target = avg * (1 + self.sign * self.p.takeprofit_pct/100)
            o = self._close_order()
            if o is None:
                self._place("close", held, target, "TP")
            elif abs(o["px"]-target)/target > 0.001 or abs(o["sz"]-held) > 1e-9:
                self._cancel_close(); self._place("close", held, target, "TP")
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
            log(f"  [STRAT] the price moved against us ({moved*100:.2f}%) — DCA")
            self._place("open", self._sz_for(self.p.dca_amount, px), px, "DCA", self.p.dca_amount)

    # -- loop ------------------------------------------------------------------
    def run(self) -> None:
        mode = "avg_tp" if self.p.enable_takeprofit else "dca_only"
        if not self.dry_run and self.client.exchange:
            self.client.set_leverage(self.coin, self.leverage)
        log("  === STRATEGIE HYPERLIQUID PORNITA ===")
        log(f"      coin       : {self.coin} perp  DIRECTIE={self.p.direction.upper()} levier {self.leverage}x  {'[PAPER]' if self.dry_run else '⚠ REAL'}")
        log(f"      mod        : {mode}")
        log(f"      intrare    : {self.p.entry_amount} {self.ccy} @ market{'-' if self.sign>0 else '+'}{self.p.entry_discount_pct}%")
        log(f"      DCA        : {self.p.dca_amount} {self.ccy} la {self.p.dca_drop_pct}% contra (max {self.p.max_dca_buys})")
        log(f"      take-profit: {self.p.takeprofit_pct}% in favoare" if self.p.enable_takeprofit else "      TP: off")
        log(f"      PLAFON     : {self.p.max_budget} {self.ccy} / ciclu  |  fee HL ~{HL_FEE_PCT}%/leg")
        try:
            while True:
                price = get_price(self.client, self.coin)
                if price is None:
                    log("  [STRAT] price unavailable"); time.sleep(self.p.check_minutes*60); continue
                self.reconcile(price); self.step(price); self._save()
                avg = self._avg()
                pos = f"qty={self.s['qty']:.6f} avg={avg:.4f}" if avg else "qty=0 (waiting for an entry)"
                log(f"  [STRAT] price={price:.4f} [{self.p.direction}]  {pos}  "
                    f"NET={self.s['realized_net']:+.2f} (brut {self.s['realized_gross']:+.2f}, fee {self.s['fees_total']:.2f}) {self.ccy}  ord={len(self.s['orders'])}")
                time.sleep(self.p.check_minutes*60)
        except KeyboardInterrupt:
            log("  [STRAT] stopped manually."); self._save()

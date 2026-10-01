#!/usr/bin/env python3
"""Market-neutral Hyperliquid funding-farming strategy.

Hold equal long spot and short perpetual legs for near-zero price delta. Spot
gains offset perpetual losses when HYPE rises, and vice versa when it falls. The
intended return is positive funding received by the short, less fees, basis drift,
and rebalancing cost. Funding can reverse and make the position pay rather than
receive; opening and closing also require four orders per cycle.

``legs`` reads both legs, prices, and funding. Decision logic opens, holds, or
closes based on funding, while _open/_rebalance/_close move toward target size.
"""

from __future__ import annotations

import fcntl
import json
import os
import sys
import time
from dataclasses import dataclass

_HL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_ROOT_DIR = os.path.dirname(_HL_DIR)
for _p in (_HL_DIR, _ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common import (
    log, now_str, required_env, defined_env, required_float_env,
    required_int_env, required_bool_env,
)
from notify import notify
from hl_client import HLClient, HLError
from state_io import atomic_write_json, load_json_state


_HERE = os.path.dirname(os.path.abspath(__file__))
MIN_ORDER_USD = 10.5          # Do not submit orders below Hyperliquid's ~$10 minimum.
_OLD_STATE = os.path.join(_HERE, ".state_dn.json")   # Legacy single-coin state name.


def state_path_for(coin: str) -> str:
    """Return a per-coin state path so multiple DN instances can run in parallel."""
    safe = "".join(c if c.isalnum() else "_" for c in coin)
    return os.path.join(_HERE, f".state_dn_{safe}.json")


@dataclass
class DNParams:
    coin: str            # Perpetual coin, for example HYPE.
    spot_pair: str       # Spot @index, for example @107.
    spot_token: str      # Spot token, for example HYPE.
    notional: float      # USDC per leg.
    entry_funding_hr: float  # Open when average funding reaches this value.
    exit_funding_hr: float   # Close below this average, providing hysteresis.
    funding_window_h: float  # Averaging window that rejects isolated noise.
    min_hold_h: float        # Minimum hold before closing.
    rebalance_pct: float # Delta tolerance as a percentage of size.
    check_minutes: float
    sz_decimals: int
    liq_alert_pct: float # Alert within this percentage of short liquidation.
    auto_protect: bool   # Automatically reduce near liquidation.
    reduce_pct: float    # Percentage removed from both legs per intervention.
    perp_leverage: int   # Low leverage leaves more margin and distant liquidation.
    allow_scale_up: bool # Scale a live position toward notional with collateral guard.
    fee_pct: float       # Configured fee estimate used for accounting.

    @classmethod
    def from_env(cls, client: HLClient | None = None) -> "DNParams":
        coin = required_env("HL_COIN")
        if client is None:
            raise ValueError("A Hyperliquid client is required to resolve size precision")
        szd = client.sz_decimals(coin)
        spot_token = required_env("HL_SPOT_TOKEN")
        spot_pair = defined_env("HL_SPOT_PAIR")
        if not spot_pair and client is not None:       # Empty resolves automatically from spotMeta.
            spot_pair = client.resolve_spot_pair(spot_token) or ""
            if spot_pair:
                log(f"  [DN] pereche spot rezolvata automat: {spot_token} -> {spot_pair}")
        if not spot_pair:
            raise ValueError(
                f"Unable to resolve HL_SPOT_PAIR for configured token {spot_token!r}"
            )
        return cls(
            coin        = coin,
            spot_pair   = spot_pair,
            spot_token  = spot_token,
            notional    = required_float_env("DN_NOTIONAL"),
            entry_funding_hr = required_float_env("DN_ENTRY_FUNDING_HR_PCT") / 100.0,
            exit_funding_hr  = required_float_env("DN_EXIT_FUNDING_HR_PCT") / 100.0,
            funding_window_h = required_float_env("DN_FUNDING_WINDOW_H"),
            min_hold_h       = required_float_env("DN_MIN_HOLD_H"),
            rebalance_pct  = required_float_env("DN_REBALANCE_PCT"),
            check_minutes  = required_float_env("DN_CHECK_MINUTES"),
            sz_decimals = szd,
            liq_alert_pct = required_float_env("DN_LIQ_ALERT_PCT"),
            auto_protect  = required_bool_env("DN_AUTO_PROTECT"),
            reduce_pct    = required_float_env("DN_REDUCE_PCT"),
            perp_leverage = required_int_env("DN_PERP_LEVERAGE"),
            allow_scale_up = required_bool_env("DN_ALLOW_SCALE_UP"),
            fee_pct      = required_float_env("HL_FEE_PCT"),
        )


def _new_state() -> dict:
    return {"status": "flat", "target_sz": 0.0, "fees_paid": 0.0,
            "funding_accrued": 0.0, "opened_at": None, "opened_ts": None, "liq_alerted": False,
            "funding_hist": [],   # [[timestamp, rate], ...] for window averaging.
            "orphan_count": 0,    # Consecutive one-leg ticks for glitch rejection.
            "gone_count": 0,      # Consecutive ticks with both legs absent.
            "drift_count": 0,     # Consecutive large-drift confirmations before trading.
            "order_fails": 0,     # Consecutive order failures; alert at three.
            "cooldown_until": 0,  # Do not reopen before this anti-thrash timestamp.
            "spot_qty": 0.0, "perp_szi": 0.0}   # Used only in paper mode.


class DeltaNeutral:
    def __init__(self, client: HLClient, params: DNParams, dry_run: bool = True, desktop: bool = False):
        self.client = client
        self.p = params
        self.dry_run = dry_run
        self.desktop = desktop
        self.state_file = state_path_for(params.coin)
        # Preserve state when migrating from the legacy single-coin filename.
        if not os.path.exists(self.state_file) and os.path.exists(_OLD_STATE):
            try:
                os.rename(_OLD_STATE, self.state_file)
                log(f"  [DN] stare migrata: .state_dn.json -> {os.path.basename(self.state_file)}")
            except OSError as e:
                log(f"  ! [DN] state migration failed: {e}")
        self.s = self._load()

    def _load(self) -> dict:
        loaded = load_json_state(
            self.state_file, default_factory=dict, fail_closed=not self.dry_run,
            label="Hyperliquid delta-neutral",
        )
        state = _new_state()
        state.update(loaded)
        return state

    def _save(self):
        try:
            atomic_write_json(self.state_file, self.s, indent=2)
        except OSError as e:
            log(f"  ! [DN] cannot save: {e}")

    def _acquire_lock(self) -> bool:
        """Lock state so a second same-coin instance refuses to start."""
        try:
            self._lock_fh = open(self.state_file + ".lock", "w")
            fcntl.flock(self._lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            try:
                self._lock_fh.close()
            except (OSError, AttributeError):
                pass
            return False

    # -- Read layer for both legs. ---------------------------------------------
    def legs(self) -> dict | None:
        """Read prices and quantities, returning None if any read fails.

        A false zero balance or position could make rebalancing duplicate one leg.
        """
        spot_px = self.client.spot_mid(self.p.spot_pair)
        perp_px = self.client.mid(self.p.coin)
        funding = self.client.funding_rate(self.p.coin)
        if spot_px is None or perp_px is None or funding is None:
            return None
        if self.dry_run:
            spot_qty = self.s["spot_qty"]; perp_szi = self.s["perp_szi"]
        else:
            try:
                spot_qty = self.client.spot_balance_strict(self.p.spot_token)
                perp_szi, _ = self.client.position_strict(self.p.coin)
                self.s["spot_qty"] = spot_qty   # Persist real live position rather than paper zero.
                self.s["perp_szi"] = perp_szi
            except Exception as e:  # noqa: BLE001
                log(f"  [DN] reading the account failed ({e}) — skipping this tick (no guessing)")
                return None
        return {"spot_px": spot_px, "perp_px": perp_px, "funding": funding,
                "spot_qty": spot_qty, "perp_szi": perp_szi}

    def _round(self, sz: float) -> float:
        return round(sz, self.p.sz_decimals)

    # -- Execution. ------------------------------------------------------------
    def _skip_dust(self, sz: float, px: float, what: str) -> bool:
        """Skip sub-minimum orders to avoid repeated venue rejections."""
        if sz * px < MIN_ORDER_USD:
            log(f"  [DN] {what} {sz} (~${sz*px:.2f}) below the HL minimum — skipping (dust)")
            return True
        return False

    def _record(self, ok: bool, sz: float, px: float):
        """Count consecutive failures and accrue fees only on success."""
        if ok:
            self.s["order_fails"] = 0
            self.s["fees_paid"] += (self.p.fee_pct/100) * sz * px
        else:
            self.s["order_fails"] = self.s.get("order_fails", 0) + 1
            if self.s["order_fails"] == 3:
                notify(title=f"⚠ DN {self.p.coin}: 3 consecutive failed orders",
                       body="Check the margin and collateral on Hyperliquid. "
                            "The bot keeps retrying (without duplicating anything).",
                       source="dn", desktop=self.desktop)

    def _buy_spot(self, sz: float, px: float):
        sz = self._round(sz)
        if sz <= 0 or self._skip_dust(sz, px, "BUY SPOT"): return
        if self.dry_run:
            self.s["spot_qty"] += sz; log(f"  [DN] [PAPER] BUY SPOT {sz} {self.p.spot_token} @ {px:.4f}")
            self._record(True, sz, px)
        else:
            ok, oid, msg = self.client.spot_order(self.p.spot_pair, True, sz, px * 1.001, self.p.sz_decimals)
            log(f"  [DN] BUY SPOT {sz} @ ~{px:.4f} -> ok={ok} {msg}")
            self._record(ok, sz, px)

    def _sell_spot(self, sz: float, px: float):
        sz = self._round(sz)
        if sz <= 0 or self._skip_dust(sz, px, "SELL SPOT"): return
        if self.dry_run:
            self.s["spot_qty"] -= sz; log(f"  [DN] [PAPER] SELL SPOT {sz} @ {px:.4f}")
            self._record(True, sz, px)
        else:
            ok, oid, msg = self.client.spot_order(self.p.spot_pair, False, sz, px * 0.999, self.p.sz_decimals)
            log(f"  [DN] SELL SPOT {sz} @ ~{px:.4f} -> ok={ok} {msg}")
            self._record(ok, sz, px)

    def _short_perp(self, sz: float, px: float):
        sz = self._round(sz)
        if sz <= 0 or self._skip_dust(sz, px, "SHORT PERP"): return
        if self.dry_run:
            self.s["perp_szi"] -= sz; log(f"  [DN] [PAPER] SHORT PERP {sz} {self.p.coin} @ {px:.4f}")
            self._record(True, sz, px)
        else:
            ok, oid, msg = self.client.place_limit(self.p.coin, False, sz, px * 0.999, reduce_only=False)
            log(f"  [DN] SHORT PERP {sz} @ ~{px:.4f} -> ok={ok} {msg}")
            self._record(ok, sz, px)

    def _cover_perp(self, sz: float, px: float):
        sz = self._round(sz)
        if sz <= 0 or self._skip_dust(sz, px, "COVER PERP"): return
        if self.dry_run:
            self.s["perp_szi"] += sz; log(f"  [DN] [PAPER] COVER PERP {sz} @ {px:.4f}")
            self._record(True, sz, px)
        else:
            ok, oid, msg = self.client.place_limit(self.p.coin, True, sz, px * 1.001, reduce_only=True)
            log(f"  [DN] COVER PERP {sz} @ ~{px:.4f} -> ok={ok} {msg}")
            self._record(ok, sz, px)

    def _cancel_open_orders(self):
        """Best-effort cancellation of remaining coin/pair orders after an incident."""
        if self.dry_run:
            return
        try:
            for o in self.client.open_orders():
                if o.get("coin") in (self.p.coin, self.p.spot_pair):
                    self.client.cancel(o.get("coin"), o.get("oid"))
                    log(f"  [DN] leftover order cancelled: {o.get('coin')} oid={o.get('oid')}")
        except Exception as e:  # noqa: BLE001
            log(f"  ! [DN] curatenia ordinelor failed ({e}) — continui")

    # -- High-level actions. ---------------------------------------------------
    def _open(self, L: dict):
        sz = self._round(self.p.notional / L["perp_px"])
        # Low short leverage leaves ample margin and distant liquidation.
        if not self.dry_run and self.client.exchange:
            self.client.set_leverage(self.p.coin, self.p.perp_leverage)
        log(f"  [DN] >>> DESCHID delta-neutral: long {sz} SPOT + short {sz} PERP {self.p.coin} "
            f"(~{self.p.notional} USDC/picior, levier {self.p.perp_leverage}x)")
        self._buy_spot(sz, L["spot_px"])
        self._short_perp(sz, L["perp_px"])
        self.s["status"] = "open"; self.s["target_sz"] = sz
        self.s["opened_at"] = now_str(); self.s["opened_ts"] = time.time()
        notify(title=f"DN {self.p.coin} DESCHIS",
               body=f"long {sz} spot + short {sz} perp | fund {L['funding']*100:.4f}%/h",
               source="dn", desktop=self.desktop)

    def _close(self, L: dict, reason: str):
        sz_spot = self._round(L["spot_qty"]); sz_perp = self._round(abs(L["perp_szi"]))
        log(f"  [DN] <<< INCHID delta-neutral ({reason}): vand {sz_spot} spot + acopar {sz_perp} perp")
        if sz_spot > 0: self._sell_spot(sz_spot, L["spot_px"])
        if sz_perp > 0: self._cover_perp(sz_perp, L["perp_px"])
        notify(title=f"DN {self.p.coin} INCHIS ({reason})",
               body=f"fund~{self.s['funding_accrued']:+.2f}$ fee~{self.s['fees_paid']:.2f}$",
               source="dn", desktop=self.desktop)
        keep_fund = self.s["funding_accrued"]; keep_fee = self.s["fees_paid"]
        self.s = _new_state(); self.s["funding_accrued"] = keep_fund; self.s["fees_paid"] = keep_fee

    def _rebalance(self, L: dict):
        """Move both legs to target size, correcting partial fills and drift.

        Drift above half the target is suspicious and requires confirmation on two
        consecutive ticks before trading.
        """
        tgt = self.s["target_sz"]; tol = tgt * self.p.rebalance_pct/100
        d_spot = tgt - L["spot_qty"]
        d_perp = tgt - abs(L["perp_szi"])
        if tgt > 0 and max(abs(d_spot), abs(d_perp)) > tgt * 0.5:
            self.s["drift_count"] = self.s.get("drift_count", 0) + 1
            if self.s["drift_count"] < 2:
                log("  [DN] large drift detected — waiting for confirmation on one more tick (anti-glitch)")
                return
        else:
            self.s["drift_count"] = 0
        if abs(d_spot) > tol:
            (self._buy_spot if d_spot > 0 else self._sell_spot)(abs(d_spot), L["spot_px"])
            log(f"  [DN] rebalans SPOT {d_spot:+.4f} (target {tgt})")
        if abs(d_perp) > tol:
            (self._short_perp if d_perp > 0 else self._cover_perp)(abs(d_perp), L["perp_px"])
            log(f"  [DN] rebalans PERP {d_perp:+.4f} (target {tgt})")
        self.s["drift_count"] = 0

    def _go_flat(self, reason: str, cooldown_s: float = 3600):
        """Mark flat with an anti-thrash cooldown while preserving P&L counters."""
        keep_fund, keep_fee = self.s["funding_accrued"], self.s["fees_paid"]
        self.s = _new_state()
        self.s["funding_accrued"], self.s["fees_paid"] = keep_fund, keep_fee
        self.s["cooldown_until"] = time.time() + cooldown_s
        log(f"  [DN] -> flat ({reason}); {cooldown_s/60:.0f} min cooldown before a new opening")

    def _check_legs_integrity(self, L: dict) -> bool:
        """Handle a missing leg from liquidation, manual action, or API glitch.

        Require two consecutive confirmations before any action. Return True after
        handling so the current tick stops.
        """
        tgt = self.s["target_sz"]
        if tgt <= 0:
            self._go_flat("tinta zero", cooldown_s=0)
            return True
        sq, pq = L["spot_qty"], abs(L["perp_szi"])
        spot_gone, perp_gone = sq < tgt * 0.1, pq < tgt * 0.1
        if spot_gone and perp_gone:
            self.s["gone_count"] = self.s.get("gone_count", 0) + 1
            if self.s["gone_count"] < 2:
                log("  [DN] both legs look gone — waiting for confirmation (anti-glitch)")
                return True
            log("  [DN] the position disappeared from the account (closed manually?)")
            self._cancel_open_orders()
            notify(title=f"DN {self.p.coin}: the position disappeared — moving to flat",
                   body=f"both legs gone (closed manually?) — clearing the orders, 1h cooldown",
                   source="dn", desktop=self.desktop)
            self._go_flat("both legs disappeared")
            return True
        if spot_gone != perp_gone:                    # Exactly one missing leg creates directional risk.
            self.s["orphan_count"] = self.s.get("orphan_count", 0) + 1
            if self.s["orphan_count"] < 2:
                log("  [DN] one leg looks gone — waiting for confirmation (anti-glitch)")
                return True
            what = ("the perp short (LIQUIDATED or closed manually)" if perp_gone
                    else "the spot leg (sold manually?)")
            log(f"  ⚠ [DN] {what} is gone — the position is NO LONGER neutral!")
            self._cancel_open_orders()
            if self.p.auto_protect:
                if perp_gone and sq > 0:
                    self._sell_spot(sq, L["spot_px"])
                if spot_gone and pq > 0:
                    self._cover_perp(pq, L["perp_px"])
                notify(title=f"🛡 DN {self.p.coin}: a leg is gone — closed the rest too",
                       body=f"{what} — closed the remaining leg to remove directional risk; 1h cooldown",
                       source="dn", desktop=self.desktop)
                self._go_flat("orphaned leg closed")
            else:
                notify(title=f"⚠ DN {self.p.coin}: a leg is gone — MANUAL INTERVENTION",
                       body=f"{what}, DN_AUTO_PROTECT=false: not acting alone — the remaining position is DIRECTIONAL!",
                       source="dn", desktop=self.desktop)
            return True
        self.s["orphan_count"] = 0
        self.s["gone_count"] = 0
        return False

    # -- Liquidation monitoring and protection for the perpetual short. --------
    def _check_liq(self, L: dict) -> bool:
        """Return True after reducing the position so this tick skips rebalancing."""
        pos = self.client.position_full(self.p.coin)
        if not pos:
            return False
        try:
            liq = float(pos.get("liquidationPx") or 0)
        except (TypeError, ValueError):
            liq = 0.0
        if liq <= 0:
            return False
        perp_px = L["perp_px"]
        dist_pct = (liq - perp_px) / perp_px * 100   # Short liquidation is above price.
        if 0 < dist_pct <= self.p.liq_alert_pct:
            if not self.s.get("liq_alerted"):
                self.s["liq_alerted"] = True
                log(f"  ⚠ [DN] SHORT close to LIQUIDATION! price={perp_px:.4f} liq={liq:.4f} ({dist_pct:.1f}% away)")
                notify(title=f"⚠ {self.p.coin}: the short is close to LIQUIDATION!",
                       body=f"p{perp_px:.2f} liq{liq:.2f} (dist {dist_pct:.1f}%)",
                       source="dn", desktop=self.desktop)
            # Reduce both legs to shrink the short and move liquidation farther away.
            if self.p.auto_protect:
                cut = self._round(abs(L["perp_szi"]) * self.p.reduce_pct / 100)
                if cut > 0:
                    log(f"  🛡 [DN] AUTO-PROTECT: reducing both legs by {cut} (de-risk, remain neutral)")
                    self._cover_perp(cut, perp_px)
                    self._sell_spot(cut, L["spot_px"])
                    self.s["target_sz"] = max(0.0, self.s["target_sz"] - cut)
                    notify(title=f"🛡 {self.p.coin}: I reduced the position pre-emptively",
                           body=f"near liquidation — reduced both legs by {cut}, remaining neutral",
                           source="dn", desktop=self.desktop)
                    return True
        elif dist_pct > self.p.liq_alert_pct * 1.5:
            self.s["liq_alerted"] = False   # Re-arm after returning to safety.
        return False

    # -- Main loop. ------------------------------------------------------------
    def run(self):
        if not self._acquire_lock():
            log(f"  ! [DN] ANOTHER INSTANCE is already running on {self.p.coin} — EXITING (anti-duplication)")
            notify(title=f"DN {self.p.coin}: instanta dubla refuzata",
                   body="Another dn_bot is already running on the same coin and state. "
                        "This instance stopped itself so it would not duplicate the orders.",
                   source="dn", desktop=self.desktop)
            return
        log("  === DELTA-NEUTRAL STARTED ===")
        log(f"      coin={self.p.coin} spot={self.p.spot_pair}  notional={self.p.notional} USDC/picior  {'[PAPER]' if self.dry_run else '⚠ REAL'}")
        log(f"      intrare funding>= {self.p.entry_funding_hr*100:.4f}%/ora  | iesire funding< {self.p.exit_funding_hr*100:.4f}%/ora")
        log(f"      mediere {self.p.funding_window_h}h  | tin minim {self.p.min_hold_h}h  | rebalans la {self.p.rebalance_pct}% delta")
        errors = 0
        while True:
            try:
                L = self.legs()
                if L is None:
                    log("  [DN] date indisponibile — reincerc")
                else:
                    self.tick(L)
                    errors = 0
                self._save()
            except KeyboardInterrupt:
                log("  [DN] stopped manually."); self._save(); return
            except Exception as e:  # noqa: BLE001 — Keep the autonomous bot alive.
                errors += 1
                log(f"  ! [DN] eroare neasteptata (#{errors} consecutiv): {e!r} — botul continua")
                if errors == 3:
                    notify(title=f"⚠ DN {self.p.coin}: erori repetate",
                           body=f"{e!r}\nThe bot keeps running and retries with backoff.",
                           source="dn", desktop=self.desktop)
                try:
                    self._save()
                except Exception:  # noqa: BLE001
                    pass
            # Exponential error backoff capped at five minutes; otherwise normal cadence.
            time.sleep(min(self.p.check_minutes * 60 * (2 ** min(errors, 3)), 300))

    def _maybe_scale_up(self, L: dict) -> None:
        """Scale a live position toward notional while remaining neutral.

        Raise target size so rebalancing adds both legs. The collateral guard limits
        spot purchases to free USDC and scales partially when necessary.
        """
        if not self.p.allow_scale_up:
            return
        cur_notional = self.s["target_sz"] * L["perp_px"]
        if cur_notional >= self.p.notional * 0.95:
            return                                        # Already at target.
        want = self._round(self.p.notional / L["perp_px"])
        add = self._round(want - self.s["target_sz"])
        if add <= 0:
            return
        if not self.dry_run:
            try:
                # Spot USDC funds the long while unified collateral covers short
                # margin. Use the spot balance because perpetual withdrawable may
                # report zero even when spot USDC exists.
                free = self.client.spot_balance("USDC")
            except Exception as e:  # noqa: BLE001
                log(f"  ! [DN] scale-up: cannot read the collateral ({e}) — deferred"); return
            if free < add * L["spot_px"]:                 # Scale partially when full size is unaffordable.
                aff = self._round((free * 0.95) / L["spot_px"])
                if aff <= 0:
                    log(f"  [DN] scale-up wanted but the collateral is insufficient (free ${free:.0f})")
                    return
                want = self._round(self.s["target_sz"] + aff)
        log(f"  [DN] ⬆ SCALE-UP towards ${self.p.notional:.0f}/leg: target {self.s['target_sz']} -> {want} "
            f"(~${want*L['perp_px']:.0f}/leg). _rebalance buys the difference.")
        self.s["target_sz"] = want
        notify(title=f"⬆ DN {self.p.coin}: cresc pozitia la ~${want*L['perp_px']:.0f}/picior",
               body=f"scale-up spre notional {self.p.notional}, raman neutru",
               source="dn", desktop=self.desktop)

    def tick(self, L: dict) -> None:
        """Run one testable decision step: open, hold, close, or rebalance."""
        # Adopt an existing account position after restart or state loss rather
        # than opening a duplicate.
        if not self.dry_run and self.s["status"] == "flat":
            sq, pq = abs(L["spot_qty"]), abs(L["perp_szi"])
            if sq > 1e-6 and pq > 1e-6:
                self.s["status"] = "open"
                self.s["target_sz"] = round((sq + pq) / 2, 6)
                if not self.s.get("opened_ts"):
                    self.s["opened_ts"] = time.time()
                log(f"  [DN] adopt pozitie existenta: spot {L['spot_qty']} / perp {L['perp_szi']} "
                    f"-> status=open, target={self.s['target_sz']}")
        fhr = L["funding"]
        delta = L["spot_qty"] + L["perp_szi"]       # Approximately zero when hedged.
        basis = (L["perp_px"] - L["spot_px"]) / L["spot_px"] * 100

        # Average funding across the window to ignore isolated readings and churn.
        now = time.time()
        hist = self.s.setdefault("funding_hist", [])
        hist.append([now, fhr])
        self.s["funding_hist"] = [x for x in hist if now - x[0] <= self.p.funding_window_h * 3600]
        avg_f = sum(x[1] for x in self.s["funding_hist"]) / len(self.s["funding_hist"])

        if self.s["status"] == "flat":
            if now < self.s.get("cooldown_until", 0):
                log(f"  [DN] in cooldown after an incident ({(self.s['cooldown_until']-now)/60:.0f} min left) — not reopening")
            elif avg_f >= self.p.entry_funding_hr:
                self._open(L)
            else:
                log(f"  [DN] average funding {avg_f*100:+.4f}%/h < the entry threshold — waiting (flat)")
        else:
            if self._check_legs_integrity(L):       # Liquidation, manual close, or glitch.
                return
            self.s["funding_accrued"] += fhr * abs(L["perp_szi"]) * L["perp_px"] * (self.p.check_minutes/60)
            reduced = self._check_liq(L)            # Alert and automatically reduce risk.
            if not reduced:
                self._maybe_scale_up(L)             # Raise target toward the new notional.
            held_h = (now - (self.s.get("opened_ts") or now)) / 3600
            # Close only below the average threshold after the minimum hold period.
            if avg_f < self.p.exit_funding_hr and held_h >= self.p.min_hold_h:
                self._close(L, f"average funding {avg_f*100:.4f}%/h below the threshold, held {held_h:.1f}h")
            elif avg_f < self.p.exit_funding_hr:
                log(f"  [DN] average funding is negative but held only {held_h:.1f}h < {self.p.min_hold_h}h "
                    f"— NOT closing (letting time work, avoiding churn)")
                if not reduced:
                    self._rebalance(L)
            elif not reduced:
                self._rebalance(L)

        log(f"  [DN] funding={fhr*100:+.4f}%/ora (mediu {avg_f*100:+.4f}, ~{avg_f*24*365*100:.0f}%/an)  "
            f"delta={delta:+.4f}  basis={basis:+.3f}%  status={self.s['status']}  "
            f"funding_acumulat~{self.s['funding_accrued']:+.4f}  fee~{self.s['fees_paid']:.4f} USDC")

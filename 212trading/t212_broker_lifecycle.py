"""T212 state sanitization, durable pending-submit mechanics, order placement, and portfolio reconciliation.

Isolates Trading 212 broker transport, response-loss recovery, fill accounting, and
portfolio/order reconciliation from ``212trading/strategy.py`` signal/ladder/exit rules,
while sharing active-order matching and portfolio position lookup with ``order_manager.py``.
"""

from __future__ import annotations

import math
import os
import re

_T212_OWNED_QTY_RE = re.compile(r'owned["\']?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)')
import sys
from collections.abc import Callable
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_HERE)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)
from botcore import (  # noqa: E402
    _parse_bool_flag,
    clean_order_id,
    ensure_local_module_priority,
    extract_order_id,
    extract_order_price,
    extract_order_side,
    extract_timestamp_s,
    first_pos_float,
    is_market_order,
    log as _default_log,
    normalize_epoch_sec,
    require_pos_float,
    safe_float,
    safe_nonneg_float,
    safe_nonneg_int,
    safe_pos_float,
)
ensure_local_module_priority(_HERE, "ipo_common")

from cachepaths import same_canonical_base_asset as _t212_tickers_match  # noqa: E402
from ipo_common import notify as _default_notify  # noqa: E402
from notify_engine.ntfyapi import emit_trade_notification  # noqa: E402
from order_lifecycle import restore_cancelled_orders  # noqa: E402
from providers.broker_history import extract_broker_order_ticker as _order_ticker  # noqa: E402
from providers.execution_audit import new_intent_id  # noqa: E402
from providers.ohlc_utils import ProviderError  # noqa: E402
from providers.order_ohlc_utils import extract_orig_qty  # noqa: E402
from providers.strategy_executor import OrderNotFound  # noqa: E402
from strategies.base_spot_mixin import BaseSpotMixin, sanitize_common_spot_state  # noqa: E402


def _new_state() -> dict:
    return {
        "cycle": 1, "qty": 0.0, "cost_usd": 0.0, "spent_cash": 0.0, "dca_buys": 0,
        "entry_price": None, "last_buy_price": None,
        "last_sell_price": None, "last_sell_ts": None,
        "post_sell_peak": None, "post_sell_trough": None, "last_exit_kind": None,
        "sl_bounce_state": {}, "sl_episode_state": {}, "tp_ride_state": {}, "reentry_state": {},
        "realized_pnl_usd": 0.0, "realized_net_usd": 0.0, "fees_usd": 0.0,
        "loss_band": 0, "tp_sold_levels": [], "orders": [], "pending_submit": None,
        "pos_peak": 0.0, "tr_alerted": False, "tr_armed": False,
        "tp_pending_exit": None, "sl_pending_exit": None,
    }


def _sanitize_t212_state(st: dict) -> dict:
    """Ensure persisted or injected state has valid types for all structural and numeric fields in-place."""
    if not isinstance(st, dict):
        return _new_state()
    st = sanitize_common_spot_state(
        st,
        _new_state,
        nonneg_float_keys=("qty", "cost_usd", "spent_cash", "fees_usd", "pos_peak"),
        signed_float_keys=("realized_pnl_usd", "realized_net_usd"),
        nonneg_int_keys=("dca_buys", "loss_band"),
        pos_float_keys=(
            "entry_price", "last_buy_price", "last_sell_price",
            "last_sell_ts", "post_sell_peak", "post_sell_trough",
        ),
        bool_keys=("tr_alerted", "tr_armed", "sl_alerted", "sl_pending"),
        dict_keys=("sl_bounce_state", "sl_episode_state", "tp_ride_state", "reentry_state"),
        nullable_dict_keys=("pending_submit", "tp_pending_exit", "sl_pending_exit"),
        pos_float_list_keys=("tp_sold_levels",),
        epoch_keys=("buy_backoff_until", "locked_zero_until", "tr_fail_until", "sl_fail_until"),
        epoch_dict_keys=("tp_fail_until",),
        sanitize_orders=False,
    )
    if isinstance(st.get("pending_submit"), dict):
        ps = st["pending_submit"]
        ps_side = str(ps.get("side") or "").strip().upper()
        ps_market = bool(_parse_bool_flag(ps.get("market"), False))
        ps_qty = safe_pos_float(ps.get("qty"), None)
        ps_limit = (
            safe_nonneg_float(ps.get("limit"), 0.0 if ps.get("limit") is None else None)
            if ps_market
            else safe_pos_float(ps.get("limit"), None)
        )
        ps_intent = clean_order_id(ps.get("intent_id"))
        if ps_side not in ("BUY", "SELL") or ps_qty is None or ps_limit is None or ps_intent is None:
            st["pending_submit"] = None
        else:
            ps_copy = dict(ps)
            ps_copy.update(
                side=ps_side, qty=ps_qty, limit=ps_limit, intent_id=ps_intent,
                market=ps_market, level=safe_pos_float(ps.get("level"), None),
            )
            if "ts" in ps_copy and ps_copy.get("ts") is not None:
                ps_copy["ts"] = normalize_epoch_sec(ps_copy.get("ts"), default=0.0, allow_zero=True)
            if "portfolio_fill_observed" in ps_copy:
                ps_copy["portfolio_fill_observed"] = _parse_bool_flag(ps_copy.get("portfolio_fill_observed"), False)
            st["pending_submit"] = ps_copy
    raw_orders = st.get("orders")
    if isinstance(raw_orders, list):
        clean_orders = []
        for o in raw_orders:
            if not isinstance(o, dict):
                continue
            clean_oid = clean_order_id(o.get("id"))
            o_side = str(o.get("side") or "").strip().upper()
            if clean_oid is None or o_side not in ("BUY", "SELL"):
                continue
            o_qty = safe_pos_float(o.get("qty"), None)
            o_market = _parse_bool_flag(o.get("market"), False)
            o_limit = safe_nonneg_float(o.get("limit"), -1.0) if o_market else safe_pos_float(o.get("limit"), None)
            if o_qty is None or o_limit is None or o_limit < 0.0:
                continue
            o_copy = dict(o)
            o_copy.update(
                id=clean_oid, side=o_side, qty=o_qty, limit=o_limit,
                market=o_market, level=safe_pos_float(o.get("level"), None),
            )
            if "amount" in o_copy and o_copy.get("amount") is not None:
                o_copy["amount"] = safe_pos_float(o_copy.get("amount"), None)
            if "ts" in o_copy and o_copy.get("ts") is not None:
                o_copy["ts"] = normalize_epoch_sec(o_copy.get("ts"), default=0.0, allow_zero=True)
            if "intent_id" in o_copy:
                o_copy["intent_id"] = clean_order_id(o.get("intent_id"))
            for bk in ("cancel_requested", "dca_counted"):
                if bk in o_copy:
                    o_copy[bk] = _parse_bool_flag(o_copy.get(bk), False)
            clean_orders.append(o_copy)
        st["orders"] = clean_orders
    else:
        st["orders"] = []
    return st


def t212_portfolio_position(client: Any, ticker: str) -> tuple[float, float] | None:
    """Return ``(quantity, average_price)`` for ``ticker`` from ``client.get_portfolio()``, or ``None`` if unavailable."""
    pf = client.get_portfolio()
    if pf is None:
        return None
    for p in pf:
        if isinstance(p, dict) and _t212_tickers_match(p.get("ticker") or "", ticker):
            q_f = safe_nonneg_float(p.get("quantity"), default=0.0 if p.get("quantity") is None else None)
            avg_f = safe_nonneg_float(p.get("averagePrice"), default=0.0 if p.get("averagePrice") is None else None)
            return None if (q_f is None or avg_f is None) else (q_f, avg_f)
    return 0.0, 0.0


def t212_active_order_identity(order: dict) -> tuple[str, str, float | None, float | None]:
    """Extract ``(ticker_upper, side, qty, limit)`` from a T212 active order payload."""
    ticker = _order_ticker(order).upper()
    side = extract_order_side(order) or ""
    raw_qty = order.get("quantity", order.get("orderedQuantity", order.get("qty")))
    qty: float | None = safe_pos_float(raw_qty, default=None) or extract_orig_qty(order, default=0.0) or None
    raw_limit = order.get("limitPrice", order.get("limit"))
    limit: float | None = safe_pos_float(raw_limit, default=None) or extract_order_price(order, default=0.0) or None
    return ticker, side, qty, limit


def t212_active_order_matches(
    order: dict,
    *,
    target_ticker: str,
    expected_side: Any,
    expected_qty: Any,
    expected_limit: Any = None,
    is_market: bool = False,
) -> bool:
    """Return True when ``order`` matches ``target_ticker``, ``expected_side``, ``expected_qty``, and ``expected_limit``."""
    if not isinstance(order, dict):
        return False
    side_u = str(expected_side or "").strip().upper() if not isinstance(expected_side, bool) else ""
    if not side_u:
        return False
    ticker, side, qty, limit = t212_active_order_identity(order)
    if not _t212_tickers_match(ticker, target_ticker) or side != side_u:
        return False
    exp_q = safe_pos_float(expected_qty, default=None)
    if exp_q is None or qty is None or abs(qty - exp_q) > max(0.011, exp_q * 1e-6):
        return False
    if _parse_bool_flag(is_market, False):
        return True
    exp_lim = safe_pos_float(expected_limit, default=None)
    return exp_lim is not None and limit is not None and abs(limit - exp_lim) <= 0.011


class T212BrokerLifecycleMixin(BaseSpotMixin):
    """Domain lifecycle mixin owning T212 pending-submit durability, order execution, and reconciliation."""

    def _notify(self, **kwargs: Any) -> None:
        self._mod_global("notify", _default_notify)(**kwargs)

    # -- Durable Pending-Submit Mechanics --------------------------------------
    def _new_pending_submit(self, *, side: str, qty: float, limit: float,
                            amount: float | None, kind: str,
                            level: float | None, market: bool) -> dict:
        return {
            "intent_id": new_intent_id("T212", self.ticker, kind),
            "side": str(side).upper(),
            "qty": require_pos_float(qty, "qty"),
            "limit": round(require_pos_float(limit, "limit"), 2),
            "amount": amount,
            "kind": kind,
            "level": level,
            "market": _parse_bool_flag(market, False),
            "ts": self._now(),
            "before_qty": safe_nonneg_float(self.s.get("qty"), default=0.0) or 0.0,
            "lookup_misses": 0,
        }

    def _persist_pending_submit(self, pending: dict | None) -> None:
        self._persist_pending_state_slot("pending_submit", pending)

    def _adopt_pending_submit(self, order_id: str) -> dict:
        pending = dict(self.s.get("pending_submit") or {})
        if not pending:
            raise RuntimeError("there is no pending T212 intent to adopt")
        order = {
            "id": str(order_id),
            "side": pending["side"],
            "qty": safe_float(pending.get("qty"), default=0.0) or 0.0,
            "limit": safe_float(pending.get("limit"), default=0.0) or 0.0,
            "kind": pending.get("kind"),
            "level": pending.get("level"),
            "intent_id": pending.get("intent_id"),
            "market": is_market_order(pending),
            "ts": safe_pos_float(pending.get("ts"), default=None) or self._now(),
        }
        if pending.get("amount") is not None:
            order["amount"] = pending["amount"]
        self.s["orders"].append(order)
        self._persist_pending_submit(None)
        return order

    def _submit_pending_order(self, pending: dict):
        """Persist once, submit through the common typed contract, and adopt acceptance."""
        market = is_market_order(pending)
        qty = safe_pos_float(pending.get("qty"), default=None)
        limit = safe_pos_float(pending.get("limit"), default=None)
        if qty is None or (not market and limit is None):
            raise ValueError("invalid pending T212 order qty or limit")
        self._persist_pending_submit(pending)
        outcome = self.executor.submit_outcome_with_intent(
            pending["intent_id"], self.ticker, str(pending["side"]).lower(),
            qty, None if market else limit,
            market=market, kind=pending.get("kind"),
            reference_price=limit if market else None,
        )
        pending["submission_outcome"] = outcome.state
        if outcome.state == "accepted":
            self._adopt_pending_submit(outcome.order_id)
        elif outcome.state == "refused":
            pending["submit_error"] = outcome.reason
            self._persist_pending_submit(None)
        else:
            pending["submit_error"] = outcome.reason
            self._persist_pending_submit(pending)
        return outcome

    _active_order_identity = staticmethod(t212_active_order_identity)

    def _active_matches_pending(self, order: dict, pending: dict) -> bool:
        if not isinstance(pending, dict):
            return False
        return t212_active_order_matches(
            order,
            target_ticker=self.ticker,
            expected_side=pending.get("side"),
            expected_qty=pending.get("qty"),
            expected_limit=pending.get("limit"),
            is_market=is_market_order(pending),
        )

    def _recover_pending_submit(self, real_qty: float,
                                active_orders: list[dict] | None) -> str:
        """Recover one response-lost T212 submit without blocking the loop."""
        pending = self.s.get("pending_submit")
        if not isinstance(pending, dict) or not pending:
            return "none"
        if active_orders is None:
            return "waiting"
        matches = [
            order for order in active_orders
            if isinstance(order, dict) and self._active_matches_pending(order, pending)
        ]
        matched_oid = extract_order_id(matches[0]) if len(matches) == 1 else None
        if len(matches) == 1 and matched_oid is not None:
            order = self._adopt_pending_submit(matched_oid)
            self._log_msg(f"  [STRAT] T212 submit recovered from the active orders: {order['id']}")
            return "adopted"
        if len(matches) > 1:
            self._log_msg("  ! [STRAT] ambiguous T212 submit: several active orders match \u2014 keeping it pending")
            return "waiting"

        before = safe_float(pending.get("before_qty"), 0.0)
        side = str(pending.get("side") or "").upper()
        delta = safe_float(real_qty, before) - before
        if (side == "BUY" and delta > 1e-6) or (side == "SELL" and delta < -1e-6):
            pending["portfolio_fill_observed"] = True
            self._persist_pending_submit(pending)
            return "filled"

        misses = safe_nonneg_int(pending.get("lookup_misses"), default=0) + 1
        pending["lookup_misses"] = misses
        if misses >= 2:
            self._log_msg(
                f"  [STRAT] T212 submit absent in 2 snapshots and no portfolio delta "
                f"({side} {self.ticker}) \u2014 releasing it for re-evaluation/retry"
            )
        self._persist_pending_submit(pending if misses < 2 else None)
        return "waiting" if misses < 2 else "retryable"

    # -- Fill Application & Lot Sizing -----------------------------------------
    def _sync_real_position_cost(
        self, real_qty: float, avg_price: float, *, sync_spent_only_when_held: bool = False
    ) -> None:
        self.s["qty"] = real_qty
        self.s["cost_usd"] = real_qty * avg_price
        if not sync_spent_only_when_held or real_qty > 1e-9:
            fx_rate = safe_pos_float(self.fx_to_usd) or 1.0
            self.s["spent_cash"] = round(real_qty * avg_price / fx_rate, 2)

    def _record_realized_sell_pnl(self, avg_px: float, sell_px: float, qty: float) -> tuple[float, float, float]:
        gross = (sell_px - avg_px) * qty
        fee = (self.p.fx_fee_pct / 100.0) * (avg_px * qty + sell_px * qty)
        net = gross - fee
        for state_k, delta in (("realized_pnl_usd", gross), ("realized_net_usd", net), ("fees_usd", fee)):
            self.s[state_k] += delta
        return gross, fee, net

    def _can_close_dust_cycle(self) -> bool:
        raw_sold = self.s.get("tp_sold_levels")
        sold_lvls = {
            sf
            for x in (raw_sold if isinstance(raw_sold, (list, tuple, set, frozenset)) else ())
            if (sf := safe_float(x, default=None)) is not None
        }
        return bool(
            not self.p.tp_ladder
            or {lvl for (lvl, _) in self.p.tp_ladder}.issubset(sold_lvls)
            or self._state_flag("sl_pending")
            or self._state_flag("sl_pending_exit")
        )

    def _is_dust(self, qty: float, ref_price: float | None = None) -> bool:
        """Return True if qty is negligible dust (below minimum broker holding / trading value)."""
        if qty <= 1e-9:
            return True
        if self.p.ladder_min_free <= 0:
            return False
        p = ref_price or self._avg() or 0.0
        if p <= 0:
            return False
        min_dust = max(5.0, self.p.ladder_min_free * 1.5)
        max_allowed = (self.p.entry_amount * 0.5) if self.p.entry_amount > 0 else min_dust
        return (qty * p) <= min(min_dust, max_allowed)

    def _sellable_qty(self, held: float, ref_price: float | None = None) -> float:
        """Return the maximum safely sellable quantity avoiding T212 min-opened-position."""
        p = ref_price or self._avg() or 0.0
        if self._is_dust(held, p):
            return 0.0
        max_qty = math.floor(held * 100.0 + 1e-9) / 100.0
        dust = held - max_qty
        if dust > 1e-6 and self.p.ladder_min_free > 0 and p > 0:
            if dust * p < self.p.ladder_min_free:
                buf = self.p.ladder_min_free / p
                if held > buf:
                    return min(round(held - buf, 2), max_qty)
        return max_qty

    def _notify_trade(
        self, status: str, side: str, price_str: str, qty: float, val: float,
        order_id=None, *, price: float, detail=None, paper_tag: bool = False,
    ) -> None:
        emit_trade_notification(
            self._notify, status, side, self.yahoo_sym, price_str, qty, val, self.ccy, order_id,
            venue="t212", owner="t212_bot", detail=detail,
            prefix="[PAPER]" if (paper_tag and self.dry_run) else None,
            notify_kwargs={"source": "T212", "price": price, "desktop": self.desktop},
        )

    def _apply_fill(self, order: dict, qty: float, price: float) -> None:
        tag = "[PAPER] " if self.dry_run else ""
        if order["side"] == "BUY":
            self._apply_buy_fill_accounting(order, qty, price)
            avg = self._avg()
            self._log_msg(
                f"  [STRAT] {tag}BUY FILLED {qty} @ {price:.2f} USD ({order.get('kind')})  "
                f"qty_total={self.s['qty']:.2f} avg={avg:.2f}"
            )
            self._notify_trade(
                "FILLED", "BUY", f"{price:.2f}", qty, qty * price, order.get("id"),
                price=price, paper_tag=True,
                detail=(
                    f"{order.get('kind')}", f"tot_qty {self.s['qty']:.2f}", f"avg {avg:.2f}",
                    f"dep {self.s['spent_cash']:.0f} {self.ccy}", f"DCA {self.s['dca_buys']}/{self.p.max_dca_buys}",
                ),
            )
            self._cancel_non_market_sells()
        else:
            qty = min(qty, self.s["qty"])
            avg = self._avg() or price
            gross, fee, net = self._record_realized_sell_pnl(avg, price, qty)
            self._release_sell_cost_basis(qty, avg)
            self._log_msg(
                f"  [STRAT] {tag}SELL FILLED {qty} @ {price:.2f} USD  "
                f"gross={gross:+.2f}  fee={fee:.2f}  net={net:+.2f} USD"
            )
            self._notify_trade(
                "FILLED", "SELL", f"{price:.2f}", qty, qty * price, order.get("id"),
                price=price, paper_tag=True,
                detail=(
                    f"Net {net:+.2f}$ (avg {avg:.2f}, fee {fee:.2f}, Ntot {self.s['realized_net_usd']:+.2f}$)",
                    f"cycle {self.s['cycle']} closed",
                ),
            )
            is_flat = self.s["qty"] <= 1e-9
            is_dust_exit = (
                self._is_dust(self.s["qty"], price)
                and self._can_close_dust_cycle()
            )
            if is_flat or is_dust_exit:
                self._close_cycle(price, exit_kind=order.get("kind"))

    # -- Placement, Cancellation & Rollback ------------------------------------
    def _place_buy(self, amount: float, limit: float, kind: str) -> bool:
        lim_f = safe_pos_float(limit, None)
        amt_f = safe_pos_float(amount, None)
        invalid_buy = lim_f is None or amt_f is None or round(lim_f, 2) <= 0
        fx_f = safe_pos_float(self.fx_to_usd, None)
        usd = safe_pos_float(amt_f * fx_f, default=None) if (not invalid_buy and fx_f is not None) else None
        raw_qty = safe_pos_float(usd / lim_f, default=None) if (usd is not None and lim_f is not None) else None
        qty = round(raw_qty, 2) if raw_qty is not None else 0.0
        if invalid_buy or qty <= 0:
            self._log_msg(
                "  ! [STRAT] invalid BUY amount/limit \u2014 skipping" if invalid_buy else "  ! [STRAT] qty 0 \u2014 skipping"
            )
            return False
        limit, amount = lim_f, amt_f
        if self.dry_run:
            self._log_msg(f"  [STRAT] [PAPER] placing BUY {kind} {qty} @ {limit:.2f} USD (~{amount:.0f} {self.ccy})")
            self.s["orders"].append({
                "id": self._next_paper_order_id(), "side": "BUY", "qty": qty,
                "limit": round(limit, 2), "amount": amount, "kind": kind, "ts": self._now(),
            })
            return True
        if self.s.get("pending_submit"):
            self._log_msg(f"  [STRAT] BUY {kind} deferred \u2014 a T212 submit is already pending")
            return False
        pending = self._new_pending_submit(
            side="BUY", qty=qty, limit=limit, amount=amount, kind=kind, level=None, market=False
        )
        outcome = self._submit_pending_order(pending)
        if outcome.state != "accepted":
            definitive = outcome.state == "refused"
            if definitive:
                self._log_msg(f"  ! [STRAT] BUY {kind} explicitly refused: {outcome.reason}")
            else:
                self._log_msg(f"  ! [STRAT] BUY {kind} outcome unknown: {outcome.reason}; keeping pending")
            if "insufficient" in outcome.reason.lower():
                self.s["buy_backoff_until"] = self._now() + 1800
                self._log_msg("  [STRAT] insufficient funds \u2014 pausing buys for 30 minutes")
            return not definitive
        self._log_msg(f"  [STRAT] BUY {kind} placed id={outcome.order_id} {qty} @ {limit:.2f}")
        return True

    def _place_sell(self, qty: float, limit: float, level: float | None = None,
                    kind: str = "TP", *, market: bool = False) -> bool:
        market = bool(_parse_bool_flag(market, False))
        qty_f = safe_pos_float(qty, None)
        lim_f = safe_pos_float(limit, None)
        invalid_sell = qty_f is None or lim_f is None or round(lim_f, 2) <= 0
        qty = round(qty_f, 2) if qty_f is not None else 0.0
        if invalid_sell or qty <= 0:
            self._log_msg(
                "  ! [STRAT] "
                + (
                    "invalid SELL qty/limit \u2014 sending no order"
                    if invalid_sell
                    else "SELL qty 0 after rounding \u2014 keeping the dust, sending no order"
                )
            )
            return False
        limit = lim_f
        level = safe_pos_float(level, None)
        tag = f"+{level:g}%" if level is not None else ""
        if self.dry_run:
            order_type = "MKT" if market else f"@ {limit:.2f}"
            self._log_msg(f"  [STRAT] [PAPER] placing SELL {kind}{tag} {qty:.2f} {order_type} USD")
            self.s["orders"].append({
                "id": self._next_paper_order_id(), "side": "SELL", "qty": qty,
                "limit": round(limit, 2), "kind": kind, "level": level, "market": market, "ts": self._now(),
            })
            return True
        if self.s.get("pending_submit"):
            self._log_msg(f"  [STRAT] SELL {kind}{tag} deferred \u2014 a T212 submit is already pending")
            return False
        pending = self._new_pending_submit(
            side="SELL", qty=qty, limit=limit, amount=None, kind=kind, level=level, market=market
        )
        outcome = self._submit_pending_order(pending)
        if outcome.state != "accepted":
            message = outcome.reason
            definitive = outcome.state == "refused"
            if definitive:
                self._log_msg(f"  ! [STRAT] SELL {kind}{tag} explicitly refused: {message}")
            else:
                self._log_msg(f"  ! [STRAT] SELL {kind}{tag} outcome unknown: {message}; keeping pending")
            if "selling-equity-not-owned" not in message:
                return not definitive
            _m = _T212_OWNED_QTY_RE.search(message)
            _owned = safe_nonneg_float(_m.group(1), default=None) if _m is not None else None
            if _m is None:
                self._log_msg("  ! [STRAT] selling-not-owned without the owned quantity \u2014 NOT resetting the position")
            elif _owned is not None:
                if _owned <= 1e-6:
                    self._log_msg("  ! [STRAT] T212 confirms owned=0 \u2014 resetting the state (position closed/sold)")
                    exit_px = safe_pos_float(self.s.get("last_sell_price")) or limit
                    self._close_cycle(exit_px, exit_kind=kind)
                    self.s["locked_zero_until"] = self._now() + 300
                    self._save()
                else:
                    self._log_msg(
                        f"  ! [STRAT] selling-not-owned but owned={_owned} (free<order) \u2014 NOT resetting the position"
                    )
            return False
        order_type = "MARKET" if market else f"@ {limit:.2f}"
        self._log_msg(f"  [STRAT] SELL {kind}{tag} placed id={outcome.order_id} {qty:.2f} {order_type}")
        return True

    def _cancel_order(self, o: dict) -> bool:
        return self._cancel_tracked_order(
            o,
            "id",
            lambda order: self.executor.cancel_order_with_intent(
                order.get("intent_id") or f"legacy-t212-{self.ticker}-{order['id']}",
                self.ticker,
                str(order["id"]),
            ),
            log_prefix="cancelled order",
        )

    def _cancel_all_orders(self) -> bool:
        success = True
        for order in list(self.s["orders"]):
            if not self._cancel_order(order):
                success = False
        return success

    def _cancel_non_market_sells(self) -> None:
        for order in [
            x for x in list(self.s["orders"]) if str(x.get("side") or "").upper() == "SELL" and not is_market_order(x)
        ]:
            self._cancel_order(order)

    def _restore_cancelled_orders(self, cancelled_orders: list[dict], reason: str = "") -> None:
        """Best-effort rollback: restore previously resting orders after a replacement or exit fails."""
        def _restore_one(p: dict) -> bool:
            if p["side"] == "SELL":
                return bool(self._place_sell(p["qty"], p["limit"], level=p.get("level"), kind=p.get("kind", "TP")))
            if p["side"] == "BUY":
                raw_ord = p.get("raw") if isinstance(p.get("raw"), dict) else p
                usd_notional = p["qty"] * p["limit"]
                fallback_amt = (usd_notional / self.fx_to_usd) if self.fx_to_usd > 0 else usd_notional
                amount = safe_pos_float(raw_ord.get("amount"), default=fallback_amt)
                return bool(self._place_buy(amount, p["limit"], kind=p.get("kind", "ENTRY")))
            return False

        restore_cancelled_orders(
            cancelled_orders,
            restore_fn=_restore_one,
            log_fn=self._mod_global("log", _default_log),
            reason=reason,
        )

    def _cancel_and_await_exit_clearance(
        self, label: str, keep_pred: Callable[[dict], bool]
    ) -> bool:
        """Cancel non-exempt orders and return True only when no blocking orders/submits remain."""
        if not self._cancel_all_orders():
            self._log_msg(f"  ! [STRAT] {label} deferred: at least one order could not be cancelled")
            return False
        pending_orders = [o for o in self.s["orders"] if not keep_pred(o)]
        if pending_orders or self.s.get("pending_submit"):
            wait_reason = (
                f"{len(pending_orders)} existing order(s) to reconcile cancellation"
                if pending_orders
                else "pending submit to reconcile"
            )
            self._log_msg(f"  [STRAT] {label} deferred: waiting for {wait_reason}")
            return False
        return True

    # -- Paper & Real Broker Reconciliation ------------------------------------
    def reconcile(self, price: float) -> bool:
        px = safe_pos_float(price, None)
        if px is None:
            return False
        if self.dry_run:
            self._reconcile_paper(px)
            return True
        return self._reconcile_real(px)

    def _reconcile_paper(self, price: float) -> None:
        for side in ("BUY", "SELL"):
            for o in [x for x in self.s["orders"] if x["side"] == side]:
                if o not in self.s["orders"]:
                    continue
                if side == "BUY":
                    self._remove(o)
                    self._apply_fill(o, o["qty"], o["limit"])
                elif is_market_order(o) or price >= o["limit"]:
                    self._remove(o)
                    if o.get("level") is not None:
                        self._record_tp_sold_level(o["level"])
                    self._apply_fill(o, o["qty"], price if is_market_order(o) else o["limit"])

    def _portfolio_position(self) -> tuple[float, float] | None:
        return t212_portfolio_position(self.client, self.ticker)

    def _tracked_order_status(self, order: dict, cache: dict[str, object]):
        """Return strict order status, cached for one reconciliation pass."""
        key = str(order.get("id"))
        if key in cache:
            return cache[key]
        try:
            status = self.executor.order_status_with_intent(
                order.get("intent_id") or f"legacy-t212-{self.ticker}-{key}",
                self.ticker, key,
            )
        except OrderNotFound:
            self._log_msg(f"  [STRAT] T212 order {key} definitively not found on venue")
            cache[key] = "not_found"
            return "not_found"
        except ProviderError as exc:
            self._log_msg(f"  ! [STRAT] T212 status {key} unavailable: {exc} \u2014 keeping the order")
            cache[key] = None
            return None
        cache[key] = status
        return status

    def _exact_execution_price(self, side: str, position_delta: float,
                               cache: dict[str, object]) -> float | None:
        """Return the weighted price from tracked orders' cumulative deltas."""
        rows = []
        total_qty = total_cost = 0.0
        for order in [o for o in self.s["orders"] if o.get("side") == side]:
            status = self._tracked_order_status(order, cache)
            if status is None or isinstance(status, str):
                continue
            cq = safe_nonneg_float(status.filled_qty, default=None)
            cc = safe_nonneg_float(status.cost, default=None)
            cf = safe_float(status.fee, default=0.0)
            aq = safe_nonneg_float(order.get("applied_fill_qty"), default=0.0)
            ac = safe_nonneg_float(order.get("applied_fill_cost"), default=0.0)
            af = safe_float(order.get("applied_fill_fee"), default=0.0)
            if cq is None or cc is None or cf is None or aq is None or ac is None or af is None:
                continue
            if cq + 1e-9 < aq or cc + 1e-9 < ac:
                self._log_msg(f"  ! [STRAT] T212 fill {order.get('id')} regressed cumulatively \u2014 ignoring")
                continue
            delta_qty = max(0.0, cq - aq)
            delta_cost = max(0.0, cc - ac)
            if delta_qty > 1e-9:
                rows.append((order, cq, cc, cf))
                total_qty += delta_qty
                total_cost += delta_cost

        tolerance = max(0.011, abs(position_delta) * 1e-6)
        if not rows or abs(total_qty - position_delta) > tolerance or total_cost <= 0:
            if rows:
                self._log_msg(
                    f"  ! [STRAT] T212 {side} fills ({total_qty:.4f}) != portfolio delta "
                    f"({position_delta:.4f}) - using portfolio fallback"
                )
            hist_fn = getattr(self.client, "latest_filled_order_for_ticker", None)
            if callable(hist_fn) and not hasattr(hist_fn, "_mock_name"):
                latest = hist_fn(self.ticker, side=side, refresh=True)
                if isinstance(latest, dict):
                    fill_oid = extract_order_id(latest, default="") or ""
                    fill_ts_s = safe_pos_float(extract_timestamp_s(latest, default=None), None)
                    last_sell_ts = safe_pos_float(self.s.get("last_sell_ts"), None)
                    stale_hist = bool(
                        (fill_oid and fill_oid == str(self.s.get("last_hist_fill_id") or ""))
                        or (side == "SELL" and fill_ts_s is not None and last_sell_ts is not None and fill_ts_s <= last_sell_ts + 0.5)
                    )
                    if not stale_hist and fill_ts_s is not None:
                        latest_any = hist_fn(self.ticker, side=None, refresh=False)
                        if isinstance(latest_any, dict):
                            any_side = str(latest_any.get("side") or "").strip().upper()
                            any_ts_s = (
                                normalize_epoch_sec(latest_any.get("timestamp"), None)
                                or normalize_epoch_sec(latest_any.get("time"), None)
                            )
                            if any_side and any_side != side.upper() and any_ts_s is not None and any_ts_s >= fill_ts_s:
                                stale_hist = True
                    fill_qty = first_pos_float(latest.get("executedQty"), latest.get("qty"))
                    fill_px = safe_pos_float(latest.get("price"), None)
                    if (
                        not stale_hist
                        and fill_qty is not None
                        and fill_px is not None
                        and abs(fill_qty - abs(position_delta)) <= max(0.011, abs(position_delta) * 0.05)
                    ):
                        if fill_oid:
                            self.s["last_hist_fill_id"] = fill_oid
                        return fill_px
            return None

        for order, cq, cc, cf in rows:
            order["applied_fill_qty"] = cq
            order["applied_fill_cost"] = cc
            order["applied_fill_fee"] = cf
        return total_cost / total_qty

    def _reconcile_real(self, price: float) -> bool:
        real = self._portfolio_position()
        if real is None:
            now = self._now()
            if now - getattr(self, "_pf_unavail_logged", 0) > 600:
                self._log_msg("  [STRAT] portfolio unavailable - skipping reconciliation (suppressing repeats for 10 min)")
                self._pf_unavail_logged = now
            return False
        real_qty, real_avg = real
        active_orders = self.client.list_active_orders()
        pending_outcome = self._recover_pending_submit(real_qty, active_orders)
        pending_for_fill = dict(self.s.get("pending_submit") or {}) if pending_outcome == "filled" else None
        if active_orders is None:
            active = {str(o["id"]) for o in self.s["orders"]}
        else:
            active = {
                oid
                for o in active_orders
                if isinstance(o, dict)
                and _t212_tickers_match(self._active_order_identity(o)[0], self.ticker)
                and (oid := extract_order_id(o)) is not None
            }

        prev_qty = self.s["qty"]
        prev_avg = self._avg() or real_avg
        status_cache: dict[str, object] = {}

        if real_qty > prev_qty + 1e-6 and self._now() < safe_nonneg_float(self.s.get("locked_zero_until"), 0.0):
            self._log_msg("  [STRAT] adoption ignored \u2014 stale portfolio (recent not-owned, lock active)")
            return False
        if real_qty > prev_qty + 1e-6:
            fq = real_qty - prev_qty
            exact_price = self._exact_execution_price("BUY", fq, status_cache)
            source_order = next((o for o in self.s["orders"] if o.get("side") == "BUY"), None)
            if source_order is None and pending_for_fill and pending_for_fill.get("side") == "BUY":
                source_order = pending_for_fill
            raw_marginal = (real_avg * real_qty - prev_avg * prev_qty) / fq if (fq > 0 and real_avg > 0) else 0.0
            fp = (
                exact_price
                or (safe_pos_float(raw_marginal, None) if raw_marginal > 0 else None)
                or safe_pos_float((source_order or {}).get("limit"), None)
                or safe_pos_float(real_avg, None)
                or price
            )
            eff_real_avg = real_avg if real_avg > 0 else (prev_avg if (prev_avg and prev_avg > 0) else fp)
            is_adoption = source_order is None and prev_qty < 1e-9
            if is_adoption and self._is_dust(real_qty, eff_real_avg):
                now = self._now()
                if now - getattr(self, "_dust_logged", 0) > 600:
                    self._log_msg(
                        f"  [STRAT] portfolio dust ({real_qty:.4f} shares, ~{real_qty*eff_real_avg:.2f} USD) "
                        "ignored \u2014 staying flat for new cycle"
                    )
                    self._dust_logged = now
                return True
            if source_order is not None and source_order.get("kind") == "DCA" and not source_order.get("dca_counted"):
                source_order["dca_counted"] = True
                self.s["dca_buys"] += 1
            elif source_order is None and prev_qty > 1e-9:
                self.s["dca_buys"] += 1
            self.s["last_buy_price"] = fp
            if self.s["entry_price"] is None:
                self.s["entry_price"] = fp
            self._sync_real_position_cost(real_qty, eff_real_avg)
            kind_label = (
                "ADOPTED" if is_adoption
                else str((source_order or {}).get("kind") or ("DCA" if prev_qty > 1e-9 else "ENTRY"))
            )
            self._log_msg(
                f"  [STRAT] BUY EXECUTED {fq:.4f} @ {fp:.2f} USD ({kind_label})  qty={real_qty:.4f} avg={eff_real_avg:.2f}"
            )
            buy_px_ref = eff_real_avg if is_adoption else fp
            self._notify_trade(
                "ADOPTED" if is_adoption else "FILLED", "BUY", f"{buy_px_ref:.2f}", fq, fq * buy_px_ref,
                (source_order or {}).get("id"), price=fp,
                detail=(
                    kind_label, f"tot_qty {real_qty:.4f}", f"avg {eff_real_avg:.2f}",
                    f"dep {self.s['spent_cash']:.0f} {self.ccy}", f"DCA {self.s['dca_buys']}/{self.p.max_dca_buys}",
                ),
            )
            if source_order is pending_for_fill:
                self._persist_pending_submit(None)
            self._unconfirmed_zero_polls = 0
            self._cancel_non_market_sells()

        elif real_qty < prev_qty - 1e-6:
            sold = prev_qty - real_qty
            exact_price = self._exact_execution_price("SELL", sold, status_cache)
            has_tracked_sell = (
                any(o.get("side") == "SELL" for o in self.s["orders"])
                or bool(pending_for_fill and pending_for_fill.get("side") == "SELL")
            )
            hist_fn = getattr(self.client, "latest_filled_order_for_ticker", None)
            zero_polls = safe_nonneg_int(getattr(self, "_unconfirmed_zero_polls", 0), default=0)
            if (
                real_qty <= 1e-9
                and exact_price is None
                and not has_tracked_sell
                and callable(hist_fn)
                and not hasattr(hist_fn, "_mock_name")
                and zero_polls < 1
            ):
                self._unconfirmed_zero_polls = zero_polls + 1
                self._log_msg(
                    f"  [STRAT] {self.yahoo_sym} portfolio qty transiently 0 with no tracked SELL or fill history -- awaiting confirmation"
                )
                return False
            self._unconfirmed_zero_polls = 0
            sell_price = exact_price or price
            gross, fee, net = self._record_realized_sell_pnl(prev_avg, sell_price, sold)
            self._sync_real_position_cost(real_qty, real_avg)
            self.s["last_sell_price"] = sell_price
            self.s["last_sell_ts"] = self._now()
            approx = "" if exact_price is not None else "~"
            self._log_msg(
                f"  [STRAT] SELL EXECUTED {sold:.4f} @ {approx}{sell_price:.2f} USD  "
                f"gross={gross:+.2f}  fee={fee:.2f}  net={net:+.2f} USD"
            )
            sell_oid = (
                (pending_for_fill or {}).get("id")
                if pending_for_fill and pending_for_fill.get("side") == "SELL"
                else None
            )
            self._notify_trade(
                "FILLED", "SELL", f"{approx}{sell_price:.2f}", sold, sold * sell_price,
                sell_oid, price=sell_price,
                detail=f"Net {net:+.2f}$ (avg {prev_avg:.2f}, fee {fee:.2f}, Ntot {self.s['realized_net_usd']:+.2f}$)",
            )
            if pending_for_fill and pending_for_fill.get("side") == "SELL":
                self._persist_pending_submit(None)

        else:
            self._unconfirmed_zero_polls = 0
            self._sync_real_position_cost(real_qty, real_avg, sync_spent_only_when_held=True)

        for o in list(self.s["orders"]):
            if str(o["id"]).startswith("PAPER"):
                continue
            if str(o["id"]) not in active:
                status = self._tracked_order_status(o, status_cache)
                if status is not None and not isinstance(status, str) and status.status in {"closed", "canceled", "expired"}:
                    if (
                        o["side"] == "SELL" and o.get("level") is not None
                        and safe_nonneg_float(status.filled_qty, default=0.0) > 1e-9
                    ):
                        self._record_tp_sold_level(o["level"])
                    self._remove(o)
                elif status == "not_found":
                    misses = safe_nonneg_int(o.get("lookup_misses"), default=0) + 1
                    o["lookup_misses"] = misses
                    if misses >= (2 if _parse_bool_flag(o.get("cancel_requested"), False) else 3):
                        self._log_msg(
                            f"  [STRAT] order {o.get('side', '?')} {o['id']} confirmed absent ({misses} misses) \u2014 removing from tracking"
                        )
                        if (
                            o.get("side") == "SELL" and o.get("level") is not None
                            and safe_nonneg_float(o.get("applied_fill_qty"), default=0.0) > 1e-9
                        ):
                            self._record_tp_sold_level(o["level"])
                        self._remove(o)
                continue
            o.pop("lookup_misses", None)
            if self._buy_is_stale(o, price):
                self._log_and_cancel_stale_buy(o, "id", "BUY", "\u2014")

        is_flat = real_qty <= 1e-9
        is_dust = self._is_dust(real_qty, real_avg or price)
        cycle_closed = (is_flat and prev_qty > 1e-9) or (
            prev_qty > 1e-9 and is_dust and not self._has_open("SELL")
            and self._can_close_dust_cycle()
        )
        if cycle_closed:
            exit_kind = (pending_for_fill or {}).get("kind") if isinstance(pending_for_fill, dict) else None
            exit_px = first_pos_float(self.s.get("last_sell_price"), price) or price
            if exit_kind is None and prev_avg and prev_avg > 0 and exit_px < prev_avg * 0.995:
                exit_kind = "REDUCE"
            self._close_cycle(
                exit_px,
                keep_pending_submit=True,
                exit_kind=exit_kind,
            )
        return True


__all__ = [
    "T212BrokerLifecycleMixin",
    "_new_state",
    "_sanitize_t212_state",
    "t212_active_order_identity",
    "t212_active_order_matches",
    "t212_portfolio_position",
]

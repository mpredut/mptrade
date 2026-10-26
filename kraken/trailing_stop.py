#!/usr/bin/env python3
"""
trailing_stop.py (Kraken) — crash circuit breaker for manual Kraken holdings.

Protects manually purchased HYPE (~$1.7k) against a sustained collapse without
capping upside (wide threshold) or causing tight-stop whipsaw (15%, not 7%). It
sells only the FREE balance (free = total - hold_trade), so it does not touch the
3.38 HYPE locked in the bot's take-profit order, maintaining clean separation.

Walk-forward analysis shows trailing does not generate alpha; its role is crash
protection. A wide ~15% threshold triggers only on a material decline.

Re-buy after a crash sale mirrors binance_api/trailing_stop.py: after a forced
crash sale, it buys back when price rebounds REBUY_BOUNCE_PCT% from the post-sale
low, confirming the fall has stopped before entry. Optional trend filters use the
HYPE cache_short_trend data. Configuration is in kraken/trailing.conf.

The trailing and re-buy state machine is in trailing_core.TrailingCore and shared
with Binance. This module is only the Kraken ADAPTER for provider-specific API,
logging, and notifications. See kraken/test_trailing_kraken.py.

  python3 trailing_stop.py        # loop (enabled in trailing.conf)
  python3 trailing_stop.py --once                              # one check
  python3 trailing_stop.py --status                            # peaks and thresholds
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)   # import the shared core from the repository root

from kraken_common import (load_dotenv, load_env_stack, log, required_bool_env,  # noqa: E402
                           required_env, required_float_env, single_instance)
from kraken_client import KrakenClient  # noqa: E402
from notify import notify  # noqa: E402

from trailing_core import TrailingCore, should_sell  # noqa: E402  (re-export should_sell for tests/compatibility)
from providers.execution_audit import (  # noqa: E402
    AuditedStrategyExecutor,
    intent_client_order_id,
)
from providers.base import extract_order_side  # noqa: E402
from providers.kraken_provider import KrakenProvider, _kraken_pair_matches  # noqa: E402
from order_retry import (  # noqa: E402
    StrategyExecutorLifecycleApi,
    TrackedOrderLifecycle,
)
from credentials import kraken_credentials  # noqa: E402
from cachepaths import cache_path  # noqa: E402

STATE_FILE = os.path.join(_HERE, "trailing_state.json")
CACHE_TREND = cache_path("cache_short_trend.json")


# Match the launch order used by every Kraken process: real environment, secrets,
# versioned policy. trailing.conf is then a dedicated policy layer that fills only
# trailing-specific gaps.
load_env_stack(os.path.join(_HERE, ".env"))
load_dotenv(os.path.join(_HERE, "trailing.conf"))

# Wide threshold per asset (crash circuit breaker, not a profit tool) and sale pair.
TRAIL_PCT = {"HYPE": required_float_env("KRAKEN_TRAILING_HYPE_PCT")}
PAIR_FOR = {"HYPE": required_env("KRAKEN_TRAILING_HYPE_PAIR")}
DEFAULT_TRAIL_PCT = required_float_env("KRAKEN_TRAILING_DEFAULT_PCT")
MIN_NOTIONAL_USD = required_float_env("KRAKEN_TRAILING_MIN_NOTIONAL_USD")
CHECK_SECONDS = required_float_env("KRAKEN_TRAILING_CHECK_SECONDS")

# Discover any additional configured assets (e.g. TAO, ADA)
for _env_k, _env_v in os.environ.items():
    if _env_k.startswith("KRAKEN_TRAILING_") and _env_k.endswith("_PAIR") and _env_k != "KRAKEN_TRAILING_HYPE_PAIR":
        _asset = _env_k[len("KRAKEN_TRAILING_"):-len("_PAIR")]
        _pct_k = f"KRAKEN_TRAILING_{_asset}_PCT"
        TRAIL_PCT[_asset] = required_float_env(_pct_k)
        PAIR_FOR[_asset] = required_env(_env_k)

# After a crash sale, re-buy when price rebounds BOUNCE% from the post-sale low.
REBUY_ENABLED = required_bool_env("KRAKEN_TRAILING_REBUY_ENABLED")
REBUY_BOUNCE_PCT = required_float_env("KRAKEN_TRAILING_REBUY_BOUNCE_PCT")
# Trend filters use HYPE cache_short_trend and act only on a CLEAR opposing signal.
# Neutral/unknown does not block, providing safe degradation. Skip re-buy on a clear
# downtrend. Crash sells are unfiltered by default for reliability; true enables anti-wick.
REBUY_SKIP_IF_TREND_DOWN = required_bool_env("KRAKEN_TRAILING_REBUY_SKIP_IF_TREND_DOWN")
# Minimum profit before trailing activates (0 means immediate activation as before).
# This prevents selling at a loss after a normal dip immediately following a purchase.
MIN_PROFIT_PCT = required_float_env("KRAKEN_TRAILING_MIN_PROFIT_PCT")
TRAILING_MARKET = required_bool_env("KRAKEN_TRAILING_MARKET")


class KrakenTrailing:
    """Kraken adapter for TrailingCore, providing free-balance, price,
    limit sell/buy, and trend APIs plus provider-specific logging and notifications."""

    def __init__(self, client: KrakenClient, log=log, enabled=None, state_file=STATE_FILE,
                 min_profit_pct=MIN_PROFIT_PCT, trailing_market: bool | None = None):
        self.client = client
        self.log = log
        self.enabled = required_bool_env("KRAKEN_TRAILING_ENABLED") if enabled is None else enabled
        self.trailing_market = TRAILING_MARKET if trailing_market is None else bool(trailing_market)
        self.state_file = state_file
        self._tick_balances: dict | None = None
        self._retry_counts: dict[tuple[str, str], int] = {}
        self.core = TrailingCore(
            self, log=log, enabled=self.enabled, state_file=state_file,
            min_notional=MIN_NOTIONAL_USD, rebuy_enabled=REBUY_ENABLED,
            rebuy_bounce_pct=REBUY_BOUNCE_PCT,
            rebuy_skip_if_trend_down=REBUY_SKIP_IF_TREND_DOWN,
            sell_fraction=1.0, item_isolation=False,
            min_profit_pct=min_profit_pct)
        self.executor = AuditedStrategyExecutor(
            KrakenProvider(client=client), venue="Kraken")
        self.order_lifecycle = TrackedOrderLifecycle(
            StrategyExecutorLifecycleApi(self.executor),
            provider_name="Kraken", venue="Kraken",
            missing_confirmations=2, retry_on_lookup_error=False,
            max_age_seconds=180,
        )

    # -- state delegated to the core; retained for --status and tests ----------
    def _load(self) -> dict:
        return self.core.load()

    def _save(self, st: dict):
        self.core.save(st)

    def trail_pct_for(self, asset: str) -> float:
        return TRAIL_PCT.get(asset, DEFAULT_TRAIL_PCT)

    def _balance_ex_map(self) -> dict | None:
        if self._tick_balances is not None:
            return self._tick_balances
        try:
            bx_map = self.client._private("BalanceEx")
        except Exception as e:  # noqa: BLE001
            self.log(f"  ! [TRAIL-K] BalanceEx failed ({e.__class__.__name__}: {e})")
            return None
        if not isinstance(bx_map, dict) or not bx_map:
            return None
        return bx_map

    def _balance_ex_entry(self, asset: str) -> dict | None:
        bx_map = self._balance_ex_map()
        if not isinstance(bx_map, dict):
            return None
        candidates = [asset, f"X{asset}", f"Z{asset}", f"{asset}.F"]
        if str(asset or "").upper() == "BTC":
            candidates.extend(["XBT", "XXBT", "XBT.F", "XXBT.F"])
        for key in candidates:
            entry = bx_map.get(key)
            if isinstance(entry, dict):
                return entry
        return None

    def _free(self, asset: str) -> float:
        """Return FREE balance (total minus order holds), excluding the bot position."""
        bx = self._balance_ex_entry(asset)
        if not bx:
            return 0.0
        raw_bal = bx.get("balance", 0)
        raw_hold = bx.get("hold_trade", 0)
        if isinstance(raw_bal, bool) or isinstance(raw_hold, bool):
            return 0.0
        try:
            bal_f = 0.0 if raw_bal is None else float(raw_bal)
            hold_f = 0.0 if raw_hold is None else float(raw_hold)
            free = bal_f - hold_f
            return max(0.0, free) if (math.isfinite(bal_f) and math.isfinite(hold_f) and math.isfinite(free)) else 0.0
        except (TypeError, ValueError, OverflowError):
            return 0.0

    # -- HYPE short trend from cache_short_trend.json for optional filters --
    def _trend_value(self, pair: str) -> float:
        """Return instant-trend slope (>0 up, <0 down, 0 neutral/unknown).
        Errors return 0, making trend filters a safe no-op."""
        try:
            import json
            from state_io import read_recovering_previous
            data = read_recovering_previous(CACHE_TREND, json.loads, default=None)
            if not isinstance(data, dict):
                return 0.0
            raw = str(pair or "").strip().upper()
            cleaned = raw.replace("/", "").replace("-", "").replace("_", "")
            base = cleaned
            for quote in ("USDC", "USDT", "FDUSD", "EUR", "USD"):
                if base.endswith(quote) and len(base) > len(quote):
                    base = base[: -len(quote)]
                    break
            snap = None
            for cand in (pair, raw, cleaned, f"{base}USDC", f"{base}USDT", f"{base}USD", base):
                if cand and isinstance(data.get(cand), dict):
                    snap = data[cand]
                    break
            if isinstance(snap, dict):
                raw_ts_val = snap.get("ts")
                if raw_ts_val is None or isinstance(raw_ts_val, bool):
                    return 0.0
                raw_ts = float(raw_ts_val)
                if not math.isfinite(raw_ts) or raw_ts <= 0:
                    return 0.0
                if raw_ts > 1e11:
                    raw_ts /= 1000.0
                age = time.time() - raw_ts
                if -5.0 <= age <= 15.0:
                    raw_grad = snap.get("gradient_recent", snap.get("slope_small", 0.0))
                    if raw_grad is None or isinstance(raw_grad, bool):
                        return 0.0
                    val = float(raw_grad)
                    return val if math.isfinite(val) else 0.0
        except Exception:
            pass
        return 0.0

    # == TrailingCore ADAPTER contract ========================================
    def assets(self):
        for asset, trail in TRAIL_PCT.items():
            yield (asset, asset, PAIR_FOR[asset], trail)

    def begin_tick(self) -> bool:
        self._tick_balances = None
        bx_map = self._balance_ex_map()
        if bx_map is None:
            self.log("  ! [TRAIL-K] empty or invalid BalanceEx snapshot — skipping the tick")
            return False
        self._tick_balances = bx_map
        return True

    def free_qty(self, asset: str) -> float:
        return self._free(asset)

    def total_qty(self, asset: str) -> float | None:
        bx_map = self._balance_ex_map()
        if bx_map is None:
            return None
        bx = self._balance_ex_entry(asset)
        if not bx:
            return 0.0
        raw_bal = bx.get("balance", 0)
        if raw_bal is None or isinstance(raw_bal, bool):
            return 0.0
        try:
            val = float(raw_bal)
            return val if math.isfinite(val) and val >= 0 else 0.0
        except (TypeError, ValueError, OverflowError):
            return 0.0

    def unlock_qty(self, pair: str) -> bool:
        """Cancel open SELL orders to release locked quantity for emergency stop."""
        try:
            open_ords = self.client.open_orders() or {}
            cancelled_any = False
            for txid, ord_info in open_ords.items():
                descr = ord_info.get("descr", {}) if isinstance(ord_info, dict) else {}
                ord_pair = str(descr.get("pair") or (ord_info.get("pair") if isinstance(ord_info, dict) else "") or "")
                if extract_order_side(ord_info) == "SELL" and (ord_pair == pair or _kraken_pair_matches(pair, ord_pair)):
                    self.client.cancel_order(txid)
                    cancelled_any = True
            if cancelled_any:
                self._tick_balances = None
                refreshed = self._balance_ex_map()
                if refreshed is not None:
                    self._tick_balances = refreshed
            return cancelled_any
        except Exception as e:  # noqa: BLE001
            self.log(f"  ! [TRAIL-K] failed to cancel open orders for {pair}: {e}")
            return False

    def price(self, pair: str):
        return self.client.last_price(pair)

    def trend(self, pair: str) -> float:
        return self._trend_value(pair)

    def _sync_retry_count_from_state(self, pair: str, action: str, persist) -> None:
        st = getattr(persist, "st", None)
        if not isinstance(st, dict):
            return
        rc = st.get("retry_counts")
        if isinstance(rc, dict) and action in rc:
            try:
                persisted = int(rc[action] or 0)
            except (TypeError, ValueError):
                persisted = 0
            retry_key = (pair, action)
            if persisted > int(self._retry_counts.get(retry_key, 0)):
                self._retry_counts[retry_key] = persisted

    def _order_intent(self, action, key, asset, pair, qty, price, **metadata):
        attempt = int(self._retry_counts.get((pair, action), 0)) + 1
        intent_id = (
            f"trailing:kraken:{key}:{action}:{qty:.8f}:{price:.8f}:"
            f"{metadata.get('anchor', price):.8f}"
        )
        if attempt > 1:
            intent_id = f"{intent_id}:a{attempt}"
        return self.order_lifecycle.new_intent(
            intent_id=intent_id,
            client_order_id=intent_client_order_id("Kraken", intent_id),
            symbol=pair, side="SELL" if action == "SELL" else "BUY",
            requested_qty=qty, requested_price=price,
            kind=f"TRAIL_{action}",
            attempt=attempt,
            metadata={"action": action, "asset": asset, **metadata},
        )

    def reconcile_pending(self, pending, persist):
        result = self.order_lifecycle.reconcile(pending, persist=persist)
        pair = str(pending.get("symbol") or "")
        action = str(pending.get("action") or (pending.get("metadata") or {}).get("action") or "").upper()
        retry_key = (pair, action)
        st = getattr(persist, "st", None)
        rc = st.get("retry_counts") if isinstance(st, dict) and isinstance(st.get("retry_counts"), dict) else {}
        raw_rc_act = rc.get(action, 0)
        raw_att = pending.get("attempt")
        try:
            rc_act = 0 if isinstance(raw_rc_act, bool) else int(raw_rc_act or 0)
        except (TypeError, ValueError, OverflowError):
            rc_act = 0
        try:
            att_val = 1 if isinstance(raw_att, bool) else int(raw_att or 1)
        except (TypeError, ValueError, OverflowError):
            att_val = 1
        try:
            attempt_base = max(
                int(self._retry_counts.get(retry_key, 0)),
                rc_act,
                max(0, att_val - 1),
            )
        except (TypeError, ValueError):
            attempt_base = int(self._retry_counts.get(retry_key, 0))
        if result.outcome in ("absent", "retryable"):
            new_count = attempt_base + 1
            self._retry_counts[retry_key] = new_count
            if isinstance(st, dict):
                st.setdefault("retry_counts", {})[action] = new_count
                persist(None)
        elif result.outcome == "terminal":
            raw_filled = getattr(result.status, "filled_qty", 0.0)
            raw_req = pending.get("submitted_qty") or pending.get("requested_qty") or 0.0
            try:
                filled_f = float(0.0 if (raw_filled is None or isinstance(raw_filled, bool)) else raw_filled)
                filled = filled_f if (math.isfinite(filled_f) and filled_f >= 0.0) else 0.0
            except (TypeError, ValueError, OverflowError):
                filled = 0.0
            try:
                req_f = float(0.0 if (raw_req is None or isinstance(raw_req, bool)) else raw_req)
                req_qty = req_f if (math.isfinite(req_f) and req_f >= 0.0) else 0.0
            except (TypeError, ValueError, OverflowError):
                req_qty = 0.0
            st_name = str(getattr(result.status, "status", "") or "").lower()
            if filled <= 0 or (st_name != "closed" and req_qty > 0 and filled < req_qty - 1e-6):
                new_count = attempt_base + 1
                self._retry_counts[retry_key] = new_count
                if isinstance(st, dict):
                    st.setdefault("retry_counts", {})[action] = new_count
            else:
                self._retry_counts.pop(retry_key, None)
                if isinstance(st, dict) and isinstance(st.get("retry_counts"), dict):
                    st["retry_counts"].pop(action, None)
                    if not st["retry_counts"]:
                        st.pop("retry_counts", None)
            if filled > 0:
                meta = result.intent.get("metadata") if isinstance(result.intent.get("metadata"), dict) else {}
                notify_action = str(result.intent.get("action") or meta.get("action") or "ORDER")
                notify_asset = result.intent.get("asset") or meta.get("asset") or result.intent.get("symbol")
                qty = filled
                raw_cost = getattr(result.status, "cost", 0.0)
                try:
                    cost_f = float(0.0 if (raw_cost is None or isinstance(raw_cost, bool)) else raw_cost)
                    cost = cost_f if (math.isfinite(cost_f) and cost_f >= 0.0) else 0.0
                except (TypeError, ValueError, OverflowError):
                    cost = 0.0
                raw_req_px = result.intent.get("requested_price")
                try:
                    req_px_f = float(0.0 if (raw_req_px is None or isinstance(raw_req_px, bool)) else raw_req_px)
                    req_px = req_px_f if (math.isfinite(req_px_f) and req_px_f >= 0.0) else 0.0
                except (TypeError, ValueError, OverflowError):
                    req_px = 0.0
                avg = cost / qty if cost > 0 and qty > 0 else req_px
                notify(
                    title=f"{'🛑' if notify_action == 'SELL' else '🟢'} TRAILING "
                          f"{notify_asset} {notify_action} FILLED {qty:.2f}@~{avg:.2f}",
                    body=f"status={result.status.status} fee={result.status.fee}",
                    source="kraken-trail", price=avg, desktop=False,
                )
        return result

    def _submit_lifecycle_order(self, intent, persist, pair, side, qty, price,
                                kind, bypass_guards=None, market=False):
        """Submit through StrategyExecutor with optional guard pre-checks."""
        if bypass_guards is None:
            bypass_guards = {"all"} if side.upper() == "SELL" else {"math_reference"}

        from guard_hierarchy import normalize_bypassed_guards
        from order_guard import check_intelligence_guards
        normalized_bypass = normalize_bypassed_guards(bypass_guards=bypass_guards)
        if "all" not in normalized_bypass:
            allowed, reason, suggested_scale = check_intelligence_guards(
                self.executor._executor, pair, side.upper(), price,
                qty=qty,
                bypass_guards=bypass_guards,
            )
            if not allowed:
                intent["refusal_reason"] = reason
                persist(None)
                from order_retry import TrackedOrderResult
                return TrackedOrderResult(outcome="refused", intent=intent, status=None)
            if (
                side.upper() == "BUY"
                and suggested_scale is not None
                and not isinstance(suggested_scale, bool)
                and math.isfinite(float(suggested_scale))
                and 0.0 < float(suggested_scale) < 1.0
            ):
                scaled_qty = float(qty) * float(suggested_scale)
                round_fn = getattr(self.executor._executor, "round_quantity", None)
                if callable(round_fn):
                    try:
                        rq = round_fn(pair, scaled_qty)
                        if rq is not None and not isinstance(rq, bool) and math.isfinite(float(rq)):
                            scaled_qty = float(rq)
                    except Exception:
                        pass
                min_qty_fn = getattr(self.executor._executor, "min_order_qty", None)
                raw_min_q = min_qty_fn(pair) if callable(min_qty_fn) else 0.0
                try:
                    min_q = 0.0 if raw_min_q is None or isinstance(raw_min_q, bool) else float(raw_min_q)
                except (TypeError, ValueError, OverflowError):
                    min_q = 0.0
                if not math.isfinite(min_q) or min_q < 0.0:
                    min_q = 0.0
                if not math.isfinite(scaled_qty) or scaled_qty <= 0 or (min_q > 0 and scaled_qty < min_q):
                    intent["refusal_reason"] = f"downscaled_below_min_qty:{reason}"
                    persist(None)
                    from order_retry import TrackedOrderResult
                    return TrackedOrderResult(outcome="refused", intent=intent, status=None)
                qty = scaled_qty
                intent["requested_qty"] = qty

        result = self.order_lifecycle.submit(
            intent, persist=persist,
            submit=lambda: self.executor.submit_order_with_intent(
                intent["intent_id"], pair, side.lower(), qty, price,
                market=market, kind=kind,
                client_order_id=intent["client_order_id"]),
        )
        if result.outcome == "refused":
            persist(None)
        return result

    def execute_sell(self, key, asset, pair, qty, price, peak, trail, persist):
        use_market = bool(getattr(self, "trailing_market", True))
        limit = price if use_market else (price * 0.995)
        self._sync_retry_count_from_state(pair, "SELL", persist)
        intent = self._order_intent(
            "SELL", key, asset, pair, qty, limit, anchor=peak, trail=trail,
            market=use_market)
        result = self._submit_lifecycle_order(
            intent, persist, pair, "SELL", qty, limit,
            kind="TRAIL_SELL", bypass_guards={"all"}, market=use_market)
        order_desc = "MARKET" if use_market else f"@ ~{limit:.4f}"
        if result.outcome == "refused":
            self.log(
                f"  🛑 [TRAIL-K] SELL REFUSED {qty} {asset} {order_desc} "
                f"({result.intent.get('refusal_reason')})"
            )
        else:
            self.log(
                f"  🛑 [TRAIL-K] SELL PENDING {qty} {asset} {order_desc} "
                f"(orderId={result.intent.get('order_id')})"
            )
        return result

    def execute_rebuy(self, key, asset, pair, qty, price, rb, persist):
        limit = price * 1.005
        self._sync_retry_count_from_state(pair, "REBUY", persist)
        raw_low = rb.get("low") if isinstance(rb, dict) else None
        try:
            low_f = float(raw_low) if raw_low is not None and not isinstance(raw_low, bool) else float(price)
            anchor_val = low_f if math.isfinite(low_f) and low_f > 0 else float(price)
        except (TypeError, ValueError, OverflowError):
            anchor_val = float(price)
        raw_sp = rb.get("sell_price") if isinstance(rb, dict) else None
        try:
            sp_f = float(raw_sp) if raw_sp is not None and not isinstance(raw_sp, bool) else 0.0
            sell_price_val = sp_f if math.isfinite(sp_f) and sp_f >= 0 else 0.0
        except (TypeError, ValueError, OverflowError):
            sell_price_val = 0.0
        intent = self._order_intent(
            "REBUY", key, asset, pair, qty, limit,
            anchor=anchor_val,
            sell_price=sell_price_val)
        result = self._submit_lifecycle_order(
            intent, persist, pair, "BUY", qty, limit,
            kind="TRAIL_REBUY", bypass_guards={"math_reference"})
        if result.outcome == "refused":
            self.log(
                f"  🟢 [TRAIL-K] RE-BUY REFUSED {qty} {asset} @ ~{limit:.4f} "
                f"({result.intent.get('refusal_reason')}) (not catching the knife)"
            )
        else:
            self.log(
                f"  🟢 [TRAIL-K] REBUY PENDING {qty} {asset} @ ~{limit:.4f} "
                f"(orderId={result.intent.get('order_id')})"
            )
        return result

    def log_dry_sell(self, key, asset, pair, qty, price, peak, trail) -> None:
        self.log(f"  🟡 [TRAIL-K][DRY] WOULD SELL {qty} {asset} @ ~{price:.4f} "
                 f"(peak {peak:.4f}, -{trail}%)  [KRAKEN_TRAILING_ENABLED=true for it to execute]")

    def log_dry_rebuy(self, key, asset, pair, qty, price, rb) -> None:
        raw_low = rb.get("low") if isinstance(rb, dict) else None
        try:
            low_f = float(raw_low) if raw_low is not None and not isinstance(raw_low, bool) else float(price)
            low_val = low_f if math.isfinite(low_f) and low_f > 0 else float(price)
        except (TypeError, ValueError, OverflowError):
            low_val = float(price)
        self.log(f"  🟡 [TRAIL-K][DRY] WOULD RE-BUY {qty} {asset} @ ~{price:.4f}  "
                 f"(a pullback from the low {low_val:.4f})  [KRAKEN_TRAILING_ENABLED=true for it to execute]")

    def log_hold(self, key, asset, pair, price, peak, stop_at, trail, free) -> None:
        self.log(f"  [TRAIL-K] {asset}: {price:.4f}  varf {peak:.4f}  "
                 f"sells below {stop_at:.4f} (-{trail}%)  (free {free:.4f})")

    def log_skip_rebuy_trend(self, asset) -> None:
        self.log(f"  [TRAIL-K] re-buy {asset} deferred — the instant trend is CLEARLY down (not catching the knife)")

    def log_skip_sell_trend(self, key, asset, pair, trail) -> None:
        self.log(f"  [TRAIL-K] {asset}: -{trail}% reached but the instant trend is UP — NOT selling (anti-wick)")

    def log_tick_error(self, e) -> None:
        self.log(f"  ! [TRAIL-K] cycle failed ({e.__class__.__name__}: {e}) — retrying")

    # -- one step / loop -------------------------------------------------------
    def check_once(self) -> None:
        self.core.check_once()
        # healthcheck.sh uses the log mtime as its heartbeat. When there is no free
        # balance to protect, the core intentionally exits without another message;
        # explicitly confirm that the process still completed the cycle successfully.
        self.log("  [TRAIL-K] heartbeat")

    def run(self):
        mode = "⚠ ACTIVE (it really sells)" if self.enabled else "DRY RUN (it only logs)"
        self.log(f"=== TRAILING STOP KRAKEN started — {mode} ===")
        self.log(f"    protejez: " + ", ".join(f"{a}={t}%" for a, t in TRAIL_PCT.items()) +
                 "  (FREE balance only, not the bot position)")
        self.log(f"    re-buy: {'ON' if REBUY_ENABLED else 'off'} (recul +{REBUY_BOUNCE_PCT}% de la minim)")
        while True:
            self.check_once()
            time.sleep(CHECK_SECONDS)


def main() -> int:
    ap = argparse.ArgumentParser(description="A trailing-stop circuit breaker on Kraken (with a re-buy).")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    if not args.status:
        single_instance("kraken_trailing")
    # A dedicated trailing key (KRAKEN_API_KEY_TRAIL) keeps its nonce separate from
    # _BOT (kraken_bot + xstock_watch). Fall back to _BOT when _TRAIL is absent.
    credentials = kraken_credentials("trail")
    client = KrakenClient(credentials.key, credentials.secret)
    ts = KrakenTrailing(client)
    if args.status:
        st = ts._load()
        for a, t in TRAIL_PCT.items():
            e = st.get(a, {}) if isinstance(st, dict) else {}
            if not isinstance(e, dict):
                e = {}
            peak_raw = e.get("peak")
            try:
                cand_peak = float(peak_raw) if peak_raw is not None and not isinstance(peak_raw, bool) else None
                peak = cand_peak if (cand_peak is not None and math.isfinite(cand_peak) and cand_peak > 0) else None
            except (TypeError, ValueError, OverflowError):
                peak = None
            rb = e.get("rebuy") if isinstance(e.get("rebuy"), dict) else None
            print(f"{a}: varf={peak} trailing={t}% " +
                  (f"sells below {peak*(1-t/100):.4f}" if peak else "(no peak yet)") +
                  (f"  | re-buy ARMAT (qty {rb.get('qty')}, min {rb.get('low')})" if rb else ""))
        print(f"ENABLED={ts.enabled}  REBUY={REBUY_ENABLED} (bounce {REBUY_BOUNCE_PCT}%)")
        return 0
    if args.once:
        ts.check_once()
        return 0
    ts.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())

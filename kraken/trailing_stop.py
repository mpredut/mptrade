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
HYPE cache_instant_trend data. Configuration is in kraken/trailing.conf.

The trailing and re-buy state machine is in trailing_core.TrailingCore and shared
with Binance. This module is only the Kraken ADAPTER for provider-specific API,
logging, and notifications. See kraken/test_trailing_kraken.py.

  python3 trailing_stop.py        # loop (enabled in trailing.conf)
  python3 trailing_stop.py --once                              # one check
  python3 trailing_stop.py --status                            # peaks and thresholds
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from kraken_common import (load_dotenv, load_env_stack, log, float_env, required_bool_env,
                           required_env, required_float_env, single_instance)
from kraken_client import KrakenClient, KrakenError
from notify import notify

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)   # import the shared core from the repository root

from trailing_core import TrailingCore, should_sell  # noqa: E402  (re-export should_sell for tests/compatibility)
from providers.execution_audit import (  # noqa: E402
    AuditedStrategyExecutor,
    intent_client_order_id,
)
from providers.kraken_provider import KrakenProvider  # noqa: E402
from order_retry import (  # noqa: E402
    StrategyExecutorLifecycleApi,
    TrackedOrderLifecycle,
)
from credentials import kraken_credentials  # noqa: E402

STATE_FILE = os.path.join(_HERE, "trailing_state.json")
CACHE_TREND = os.path.join(_ROOT, "cachedb", "cache_instant_trend.json")


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

# After a crash sale, re-buy when price rebounds BOUNCE% from the post-sale low.
REBUY_ENABLED = required_bool_env("KRAKEN_TRAILING_REBUY_ENABLED")
REBUY_BOUNCE_PCT = required_float_env("KRAKEN_TRAILING_REBUY_BOUNCE_PCT")
# Trend filters use HYPE cache_instant_trend and act only on a CLEAR opposing signal.
# Neutral/unknown does not block, providing safe degradation. Skip re-buy on a clear
# downtrend. Crash sells are unfiltered by default for reliability; true enables anti-wick.
REBUY_SKIP_IF_TREND_DOWN = required_bool_env("KRAKEN_TRAILING_REBUY_SKIP_IF_TREND_DOWN")
SELL_SKIP_IF_TREND_UP = required_bool_env("KRAKEN_TRAILING_SELL_SKIP_IF_TREND_UP")
# Minimum profit before trailing activates (0 means immediate activation as before).
# This prevents selling at a loss after a normal dip immediately following a purchase.
MIN_PROFIT_PCT = required_float_env("KRAKEN_TRAILING_MIN_PROFIT_PCT")


class KrakenTrailing:
    """Kraken adapter for TrailingCore, providing free-balance, price,
    limit sell/buy, and trend APIs plus provider-specific logging and notifications."""

    def __init__(self, client: KrakenClient, log=log, enabled=None, state_file=STATE_FILE,
                 min_profit_pct=MIN_PROFIT_PCT):
        self.client = client
        self.log = log
        self.enabled = required_bool_env("KRAKEN_TRAILING_ENABLED") if enabled is None else enabled
        self.state_file = state_file
        self.core = TrailingCore(
            self, log=log, enabled=self.enabled, state_file=state_file,
            min_notional=MIN_NOTIONAL_USD, rebuy_enabled=REBUY_ENABLED,
            rebuy_bounce_pct=REBUY_BOUNCE_PCT,
            rebuy_skip_if_trend_down=REBUY_SKIP_IF_TREND_DOWN,
            sell_skip_if_trend_up=SELL_SKIP_IF_TREND_UP,
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

    def _free(self, asset: str) -> float:
        """Return FREE balance (total minus order holds), excluding the bot position."""
        bx = self.client._private("BalanceEx").get(asset)
        if not bx:
            return 0.0
        try:
            return float(bx.get("balance", 0)) - float(bx.get("hold_trade", 0))
        except (TypeError, ValueError):
            return 0.0

    # -- HYPE instant trend from cache_instant_trend.json for optional filters --
    def _trend_value(self, pair: str) -> float:
        """Return instant-trend slope (>0 up, <0 down, 0 neutral/unknown).
        Errors return 0, making trend filters a safe no-op."""
        try:
            import json
            with open(CACHE_TREND) as f:
                snap = json.load(f).get(pair)
            if snap:
                return float(snap.get("gradient_recent", snap.get("slope_small", 0.0)) or 0.0)
        except Exception:
            pass
        return 0.0

    # == TrailingCore ADAPTER contract ========================================
    def assets(self):
        for asset, trail in TRAIL_PCT.items():
            yield (asset, asset, PAIR_FOR[asset], trail)

    def begin_tick(self) -> bool:
        return True   # Kraken reads balances per asset in free_qty, not in bulk

    def free_qty(self, asset: str) -> float:
        return self._free(asset)

    def price(self, pair: str):
        return self.client.last_price(pair)

    def trend(self, pair: str) -> float:
        return self._trend_value(pair)

    def _order_intent(self, action, key, asset, pair, qty, price, **metadata):
        intent_id = (
            f"trailing:kraken:{key}:{action}:{qty:.8f}:{price:.8f}:"
            f"{metadata.get('anchor', price):.8f}"
        )
        return self.order_lifecycle.new_intent(
            intent_id=intent_id,
            client_order_id=intent_client_order_id("Kraken", intent_id),
            symbol=pair, side="SELL" if action == "SELL" else "BUY",
            requested_qty=qty, requested_price=price,
            kind=f"TRAIL_{action}",
            metadata={"action": action, "asset": asset, **metadata},
        )

    def reconcile_pending(self, pending, persist):
        result = self.order_lifecycle.reconcile(pending, persist=persist)
        if result.outcome == "terminal" and result.status.filled_qty > 0:
            action = str(result.intent.get("action") or "ORDER")
            qty = result.status.filled_qty
            avg = (result.status.cost / qty
                   if result.status.cost > 0 else result.intent["requested_price"])
            notify(
                title=f"{'🛑' if action == 'SELL' else '🟢'} TRAILING "
                      f"{result.intent.get('asset')} {action} FILLED {qty:.2f}@~{avg:.2f}",
                body=f"status={result.status.status} fee={result.status.fee}",
                source="kraken-trail", price=avg, desktop=False,
            )
        return result

    def _submit_lifecycle_order(self, intent, persist, pair, side, qty, price,
                                kind, bypass_guards=None):
        """Submit through StrategyExecutor with optional guard pre-checks."""
        if bypass_guards is None:
            bypass_guards = {"all"} if side.upper() == "SELL" else {"math_reference"}

        from order_guard import check_intelligence_guards
        if not ("all" in bypass_guards or "*" in bypass_guards):
            allowed, reason, _ = check_intelligence_guards(
                self.executor._executor, pair, side.upper(), price,
                qty=qty,
                bypass_guards=bypass_guards,
            )
            if not allowed:
                intent["refusal_reason"] = reason
                persist(None)
                from order_retry import TrackedOrderResult
                return TrackedOrderResult(outcome="refused", intent=intent, status=None)

        result = self.order_lifecycle.submit(
            intent, persist=persist,
            submit=lambda: self.executor.submit_order_with_intent(
                intent["intent_id"], pair, side.lower(), qty, price,
                market=False, kind=kind,
                client_order_id=intent["client_order_id"]),
        )
        if result.outcome == "refused":
            persist(None)
        return result

    def execute_sell(self, key, asset, pair, qty, price, peak, trail, persist):
        limit = price * 0.995
        intent = self._order_intent(
            "SELL", key, asset, pair, qty, limit, anchor=peak, trail=trail)
        result = self._submit_lifecycle_order(
            intent, persist, pair, "SELL", qty, limit,
            kind="TRAIL_SELL", bypass_guards={"all"})
        if result.outcome == "refused":
            self.log(
                f"  🛑 [TRAIL-K] SELL REFUSED {qty} {asset} @ ~{limit:.4f} "
                f"({result.intent.get('refusal_reason')})"
            )
        else:
            self.log(
                f"  🛑 [TRAIL-K] SELL PENDING {qty} {asset} @ ~{limit:.4f} "
                f"(orderId={result.intent.get('order_id')})"
            )
        return result

    def execute_rebuy(self, key, asset, pair, qty, price, rb, persist):
        limit = price * 1.005
        intent = self._order_intent(
            "REBUY", key, asset, pair, qty, limit,
            anchor=float(rb.get("low") or price),
            sell_price=float(rb.get("sell_price") or 0.0))
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
        self.log(f"  🟡 [TRAIL-K][DRY] WOULD RE-BUY {qty} {asset} @ ~{price:.4f}  "
                 f"(a pullback from the low {rb['low']:.4f})  [KRAKEN_TRAILING_ENABLED=true for it to execute]")

    def log_hold(self, key, asset, pair, price, peak, stop_at, trail, free) -> None:
        self.log(f"  [TRAIL-K] {asset}: {price:.4f}  varf {peak:.4f}  "
                 f"sells below {stop_at:.4f} (-{trail}%)  (free {free:.4f})")

    def log_skip_rebuy_trend(self, asset) -> None:
        self.log(f"  [TRAIL-K] re-buy {asset} deferred — the instant trend is CLEARLY down (not catching the knife)")

    def log_skip_sell_trend(self, key, asset, pair, trail) -> None:
        self.log(f"  [TRAIL-K] {asset}: -{trail}% reached but the instant trend is UP — NOT selling (anti-wick)")

    def log_tick_error(self, e) -> None:
        self.log(f"  ! [TRAIL-K] cycle failed ({e.__class__.__name__}: {e}) — reincerc")

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
            e = st.get(a, {})
            peak = e.get("peak")
            rb = e.get("rebuy")
            print(f"{a}: varf={peak} trailing={t}% " +
                  (f"sells below {peak*(1-t/100):.4f}" if peak else "(no peak yet)") +
                  (f"  | re-buy ARMAT (qty {rb['qty']}, min {rb.get('low')})" if rb else ""))
        print(f"ENABLED={ts.enabled}  REBUY={REBUY_ENABLED} (bounce {REBUY_BOUNCE_PCT}%)")
        return 0
    if args.once:
        ts.check_once()
        return 0
    ts.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())

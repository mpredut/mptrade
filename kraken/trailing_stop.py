#!/usr/bin/env python3
"""
trailing_stop.py (Kraken) -- crash circuit breaker for manual Kraken holdings.

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
logging, and notifications. See tests/test_trailing_kraken.py.

  python3 trailing_stop.py        # loop (enabled in trailing.conf)
  python3 trailing_stop.py --once                              # one check
  python3 trailing_stop.py --status                            # peaks and thresholds
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)   # import the shared core from the repository root
if _HERE not in sys.path:
    sys.path.append(_HERE)

from botcore import (  # noqa: E402
    _parse_bool_flag,
    format_price_log,
    load_dotenv,
    load_env_stack,
    log,
    require_pos_float,
    required_bool_env,
    required_env,
    required_float_env,
    safe_float,
    safe_nonneg_float,
    safe_pos_float,
)
from kraken_common import build_kraken_client, notify  # noqa: E402
from kraken_client import KrakenClient  # noqa: E402

from trailing_core import (  # noqa: E402
    COST_BASIS_LOOKBACK_S,
    TrailingAdapterBase,
    extract_trailing_intent_action_asset,
    load_trailing_common_env,
    parse_trailing_cli_args,
    vwap_or_fallback,
)
from providers.execution_audit import AuditedStrategyExecutor  # noqa: E402
from providers.order_ohlc_utils import extract_order_side
from providers.kraken_provider import (  # noqa: E402
    KrakenProvider,
    _kraken_balance_asset_candidates,
    kraken_pair_matches,
    split_kraken_pair,
)
from order_lifecycle import (  # noqa: E402
    StrategyExecutorLifecycleApi,
    TrackedOrderLifecycle,
    TrackedOrderResult,
)
from credentials import kraken_credentials  # noqa: E402
from cachepaths import (  # noqa: E402
    CACHE_SHORT_TREND_FILENAME,
    cache_path,
    resolve_venue_trailing_state_path,
)
from guard_hierarchy import normalize_bypassed_guards  # noqa: E402
from instrument_registry import casefold_get, select_instruments  # noqa: E402
import order_guard  # noqa: E402

STATE_FILE = resolve_venue_trailing_state_path("kraken")
CACHE_TREND = cache_path(CACHE_SHORT_TREND_FILENAME)


# Match the launch order used by every Kraken process: real environment, secrets,
# versioned policy. trailing.conf is then a dedicated policy layer that fills only
# trailing-specific gaps.
load_env_stack(os.path.join(_HERE, ".env"))
load_dotenv(os.path.join(_HERE, "trailing.conf"))

# Wide threshold per asset (crash circuit breaker, not a profit tool) and sale pair.
# Primary SSOT is instruments.conf (provider=kraken, role.trailing=yes), symmetric with Binance.
TRAILING_INSTRUMENTS = {
    spec.base.strip().upper(): spec
    for spec in select_instruments("kraken", "trailing").values()
}
TRAIL_PCT = {
    asset: spec.number("trailing.pct")
    for asset, spec in TRAILING_INSTRUMENTS.items()
}
PAIR_FOR = {
    asset: spec.symbol.strip().upper()
    for asset, spec in TRAILING_INSTRUMENTS.items()
}
REBUY_MODE_BY_ASSET = {
    asset: spec.rebuy_mode()
    for asset, spec in TRAILING_INSTRUMENTS.items()
}
DEFAULT_TRAIL_PCT = required_float_env("KRAKEN_TRAILING_DEFAULT_PCT")
MIN_NOTIONAL_USD = required_float_env("KRAKEN_TRAILING_MIN_NOTIONAL_USD")
(
    CHECK_SECONDS,
    REBUY_ENABLED,
    REBUY_BOUNCE_PCT,
    REBUY_SKIP_IF_TREND_DOWN,
    SELL_SKIP_IF_TREND_UP,
    SELL_TREND_DEFER_MAX_SEC,
    MIN_PROFIT_PCT,
) = load_trailing_common_env("KRAKEN_TRAILING_")
TRAILING_MARKET = required_bool_env("KRAKEN_TRAILING_MARKET")

# Optional environment overrides (e.g. ad-hoc tests or KRAKEN_TRAILING_<ASSET>_PAIR/PCT)
for _env_k, _env_v in os.environ.items():
    if _env_k.startswith("KRAKEN_TRAILING_") and _env_k.endswith("_PAIR"):
        _asset = _env_k[len("KRAKEN_TRAILING_"):-len("_PAIR")].strip().upper()
        if _asset and str(_env_v or "").strip():
            PAIR_FOR[_asset] = required_env(_env_k, os.environ)
            TRAIL_PCT.setdefault(_asset, DEFAULT_TRAIL_PCT)
for _env_k in os.environ:
    if (
        _env_k.startswith("KRAKEN_TRAILING_")
        and _env_k.endswith("_PCT")
        and _env_k not in {
            "KRAKEN_TRAILING_DEFAULT_PCT",
            "KRAKEN_TRAILING_REBUY_BOUNCE_PCT",
            "KRAKEN_TRAILING_MIN_PROFIT_PCT",
        }
    ):
        _asset = _env_k[len("KRAKEN_TRAILING_"):-len("_PCT")].strip().upper()
        if _asset and _asset in PAIR_FOR:
            TRAIL_PCT[_asset] = required_float_env(_env_k, os.environ)


class KrakenTrailing(TrailingAdapterBase):
    """Kraken adapter for TrailingCore, providing free-balance, price,
    limit sell/buy, and trend APIs plus provider-specific logging and notifications."""

    _VENUE_SLUG = "kraken"
    _VENUE_LABEL = "Kraken"
    _HEARTBEAT_NAME = "Kraken-trailing"
    _LOG_TAG = "[TRAIL-K]"
    _ENABLED_ENV_VAR = "KRAKEN_TRAILING_ENABLED"
    _REBUY_MODE_MAP_ATTR = "REBUY_MODE_BY_ASSET"
    _LOOKUP_BY_BASE_ASSET = True
    _PREFER_ASSET_LABEL = True
    _SELL_LIMIT_SLIPPAGE_FACTOR = 0.995
    _REBUY_LIMIT_CUSHION_FACTOR = 1.005
    _INCLUDE_MARKET_IN_SELL_INTENT = True

    def __init__(self, client: KrakenClient, log=log, enabled=None, state_file=STATE_FILE,
                 min_profit_pct=MIN_PROFIT_PCT, trailing_market: bool | None = None):
        self.client = client
        self.trailing_market = _parse_bool_flag(
            trailing_market,
            TRAILING_MARKET if trailing_market is None else False,
        )
        self._tick_balances: dict | None = None
        self._init_adapter_core(
            log=log,
            enabled=enabled,
            default_enabled=required_bool_env("KRAKEN_TRAILING_ENABLED") if enabled is None else False,
            state_file=state_file,
            min_notional=MIN_NOTIONAL_USD,
            rebuy_enabled=REBUY_ENABLED,
            rebuy_bounce_pct=REBUY_BOUNCE_PCT,
            rebuy_skip_if_trend_down=REBUY_SKIP_IF_TREND_DOWN,
            sell_skip_if_trend_up=SELL_SKIP_IF_TREND_UP,
            sell_trend_defer_max_sec=SELL_TREND_DEFER_MAX_SEC,
            sell_fraction=1.0,
            item_isolation=False,
            min_profit_pct=safe_nonneg_float(
                min_profit_pct,
                default=safe_nonneg_float(MIN_PROFIT_PCT, default=0.0),
            ),
        )
        self.executor = AuditedStrategyExecutor(
            KrakenProvider(client=client), venue="Kraken")
        self.order_lifecycle = TrackedOrderLifecycle(
            StrategyExecutorLifecycleApi(self.executor),
            provider_name="Kraken", venue="Kraken",
            missing_confirmations=2, retry_on_lookup_error=False,
            max_age_seconds=180,
        )

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

    def _balance_ex_entries(self, asset: str) -> list[object]:
        if isinstance(asset, bool) or not isinstance(asset, str) or not asset.strip():
            return []
        bx_map = self._balance_ex_map()
        if not isinstance(bx_map, dict):
            return []
        cand_folded = {c.casefold() for c in _kraken_balance_asset_candidates(asset) if isinstance(c, str) and c}
        if not cand_folded:
            return []
        matched: list[object] = []
        for k, val in bx_map.items():
            if isinstance(k, str) and k.strip().casefold() in cand_folded:
                matched.append(val)
        return matched

    # == TrailingCore ADAPTER contract ========================================
    def assets(self):
        for asset, trail in TRAIL_PCT.items():
            pair = casefold_get(PAIR_FOR, asset)
            if pair:
                yield (asset, asset, pair, trail)

    def begin_tick(self) -> bool:
        self._tick_balances = None
        bx_map = self._balance_ex_map()
        if bx_map is None:
            self.log("  ! [TRAIL-K] empty or invalid BalanceEx snapshot -- skipping the tick")
            return False
        self._tick_balances = bx_map
        return True

    def free_qty(self, asset: str) -> float:
        """Return FREE balance (total minus order holds), excluding the bot position."""
        total_free = 0.0
        for bx in self._balance_ex_entries(asset):
            if not isinstance(bx, dict):
                return 0.0
            bal_f = safe_nonneg_float(bx.get("balance"), default=None)
            hold_f = safe_nonneg_float(bx.get("hold_trade", 0.0), default=None)
            if bal_f is None or hold_f is None:
                return 0.0
            total_free += safe_nonneg_float(bal_f - hold_f, default=0.0)
        return total_free

    def total_qty(self, asset: str) -> float | None:
        if self._balance_ex_map() is None:
            return None
        total_bal = 0.0
        for bx in self._balance_ex_entries(asset):
            if not isinstance(bx, dict):
                return None
            bal_f = safe_nonneg_float(bx.get("balance", 0), default=None)
            if bal_f is None:
                return None
            total_bal += bal_f
        return total_bal

    def cost_basis(self, pair: str):
        """Use reconciled BUY/SELL inventory for new-position warm-up when fresh SSOT fills are available."""
        if isinstance(pair, bool) or not isinstance(pair, str) or not pair.strip():
            return None
        try:
            pair_norm = pair.strip().upper()
            spec = self._lookup_symbol_entry(TRAILING_INSTRUMENTS, pair)
            asset = spec.base if spec is not None else split_kraken_pair(pair_norm)[0]
            if not asset:
                return None
            held = self.total_qty(asset)
            if held is None or held <= 0.0:
                return None
            return self.executor.position_cost_basis(pair_norm or pair, held, COST_BASIS_LOOKBACK_S)
        except Exception as e:  # noqa: BLE001
            return self._log_cost_basis_unavailable(pair, e)

    def unlock_qty(self, pair: str) -> bool:
        """Cancel open SELL orders to release locked quantity for emergency stop."""
        try:
            open_ords = self.client.open_orders() or {}
            if not isinstance(open_ords, dict):
                return False
            cancelled_any = False
            for txid, ord_info in open_ords.items():
                if not isinstance(ord_info, dict):
                    continue
                raw_descr = ord_info.get("descr")
                descr = raw_descr if isinstance(raw_descr, dict) else {}
                ord_pair = str(descr.get("pair") or ord_info.get("pair") or "")
                if extract_order_side(ord_info) == "SELL" and (ord_pair == pair or kraken_pair_matches(pair, ord_pair)):
                    try:
                        res = self.client.cancel_order(txid)
                    except Exception as cancel_err:  # noqa: BLE001
                        self.log(f"  ! [TRAIL-K] failed to cancel open order {txid} for {pair}: {cancel_err}")
                        continue
                    if res is False or (
                        isinstance(res, dict)
                        and "count" in res
                        and safe_pos_float(res.get("count"), default=0.0) <= 0.0
                    ):
                        continue
                    cancelled_any = True
            if cancelled_any:
                self._tick_balances = None
                try:
                    refreshed = self._balance_ex_map()
                except Exception:  # noqa: BLE001
                    refreshed = None
                if refreshed is not None:
                    self._tick_balances = refreshed
            return cancelled_any
        except Exception as e:  # noqa: BLE001
            self.log(f"  ! [TRAIL-K] failed to cancel open orders for {pair}: {e}")
            return False

    def price(self, pair: str):
        get_px = getattr(self.executor, "get_current_price", None)
        if callable(get_px):
            return get_px(pair)
        return self.client.last_price(pair)

    def reconcile_pending(self, pending, persist):
        result = super().reconcile_pending(pending, persist)
        if result is not None and result.outcome == "terminal":
            filled = safe_nonneg_float(getattr(result.status, "filled_qty", None), default=0.0)
            if filled > 0:
                act_norm, asset_norm = extract_trailing_intent_action_asset(result.intent)
                notify_action = act_norm or "ORDER"
                notify_asset = asset_norm or (result.intent.get("symbol") if isinstance(result.intent, dict) else None)
                qty = filled
                avg = vwap_or_fallback(
                    getattr(result.status, "cost", None),
                    qty,
                    result.intent.get("requested_price") if isinstance(result.intent, dict) else 0.0,
                )
                avg_str = format_price_log(avg) if 0.0 < avg < 1.0 else f"{avg:.2f}"
                notify(
                    title=f"{'[SELL]' if notify_action == 'SELL' else '[BUY]'} TRAILING "
                          f"{notify_asset} {notify_action} FILLED {qty:.2f}@~{avg_str}",
                    body=f"status={result.status.status} fee={result.status.fee}",
                    source="kraken-trail", price=avg, desktop=False,
                )
        return result

    def _submit_lifecycle_order(self, intent, persist, pair, side, qty, price,
                                kind="TRAIL", bypass_guards=None, market=False):
        """Submit through StrategyExecutor with optional guard pre-checks."""
        if bypass_guards is None:
            bypass_guards = {"all"} if side.upper() == "SELL" else {"math_reference"}

        normalized_bypass = normalize_bypassed_guards(bypass_guards=bypass_guards)
        if "all" not in normalized_bypass:
            def _refuse_intent(refusal_reason: object) -> TrackedOrderResult:
                intent["refusal_reason"] = refusal_reason
                persist(None)
                return TrackedOrderResult(outcome="refused", intent=intent, status=None)

            if callable(getattr(self.executor, "get_orders", None)):
                dl_ok, dl_reason = order_guard.unpack_daily_limit_result(
                    order_guard.daily_limit_guard(
                        self.executor, pair, side.upper(), bypass_guards=bypass_guards,
                    )
                )
                if not dl_ok:
                    return _refuse_intent(dl_reason)
            allowed, reason, suggested_scale = order_guard.check_intelligence_guards(
                self.executor, pair, side.upper(), price,
                qty=qty,
                bypass_guards=bypass_guards,
            )
            if not allowed:
                return _refuse_intent(reason)
            if side.upper() == "BUY" and suggested_scale is not None:
                scale_f = safe_float(suggested_scale, default=None)
                if scale_f is None or scale_f <= 0.0:
                    return _refuse_intent(f"downscaled_to_zero:{reason}")
                if scale_f < 1.0:
                    qty_f = require_pos_float(qty, "qty")
                    scaled_qty = qty_f * scale_f
                    round_fn = getattr(self.executor, "round_quantity", None)
                    if callable(round_fn):
                        try:
                            rq = safe_float(round_fn(pair, scaled_qty), default=None)
                            if rq is not None:
                                scaled_qty = rq
                        except Exception:
                            pass
                    min_qty_fn = getattr(self.executor, "min_order_qty", None)
                    raw_min_q = min_qty_fn(pair) if callable(min_qty_fn) else 0.0
                    min_q = safe_nonneg_float(raw_min_q, default=0.0)
                    sq_pos = safe_pos_float(scaled_qty, default=None)
                    if sq_pos is None or (min_q > 0 and sq_pos < min_q):
                        return _refuse_intent(f"downscaled_below_min_qty:{reason}")
                    price_f = safe_pos_float(price, default=0.0)
                    min_notional = safe_nonneg_float(getattr(self.core, "min_notional", 0.0), default=0.0)
                    if min_notional > 0.0 and price_f > 0.0 and sq_pos * price_f < min_notional:
                        return _refuse_intent(f"downscaled_below_min_notional:{reason}")
                    qty = sq_pos
                    intent["requested_qty"] = qty
                    intent["submitted_qty"] = qty

        return self._submit_and_clear_if_refused(
            intent,
            persist,
            lambda: self.executor.submit_order_with_intent(
                intent["intent_id"], pair, side.lower(), qty, price,
                market=market, kind=kind,
                client_order_id=intent["client_order_id"]),
        )

    def log_tick_error(self, e) -> None:
        self.log(f"  ! {self._LOG_TAG} cycle failed ({e.__class__.__name__}: {e}) -- retrying")

    def _log_sell_result(
        self,
        result,
        *,
        asset,
        pair,
        qty,
        price,
        order_price,
        peak,
        trail,
        use_market: bool,
    ) -> None:
        order_desc = "MARKET" if use_market else f"@ ~{safe_float(order_price, 0.0):.4f}"
        status_lbl, detail = self._extract_order_result_status(result)
        self.log(f"  [SELL] {self._LOG_TAG} SELL {status_lbl} {qty} {asset} {order_desc} ({detail})")

    def _log_rebuy_result(
        self,
        result,
        *,
        asset,
        pair,
        qty,
        price_ref: float,
        order_price,
    ) -> None:
        status_lbl, detail = self._extract_order_result_status(result)
        rebuy_lbl = "RE-BUY REFUSED" if status_lbl == "REFUSED" else "REBUY PENDING"
        suffix = " (not catching the knife)" if status_lbl == "REFUSED" else ""
        self.log(
            f"  [REBUY] {self._LOG_TAG} {rebuy_lbl} {qty} {asset} "
            f"@ ~{safe_float(order_price, 0.0):.4f} ({detail}){suffix}"
        )

    def print_status(self) -> None:
        state = self.core.load()
        trail_map = self._module_attr("TRAIL_PCT")
        rows = (
            list(trail_map.items())
            if isinstance(trail_map, dict)
            else [(s, tr) for s, _a, _p, tr in self.assets()]
        )
        for a, t in rows:
            e = casefold_get(state, a, {}) if isinstance(state, dict) else {}
            if not isinstance(e, dict):
                e = {}
            peak = safe_pos_float(e.get("peak"), default=None)
            t_f = safe_nonneg_float(t, default=0.0)
            rb = e.get("rebuy") if isinstance(e.get("rebuy"), dict) else None
            print(
                f"{a}: peak={peak} trailing={t}% "
                + (f"sells below {peak * (1 - t_f / 100):.4f}" if peak else "(no peak yet)")
                + (f"  | re-buy ARMED (qty {rb.get('qty')}, min {rb.get('low')})" if rb else "")
            )
        rebuy_en = self._module_attr("REBUY_ENABLED", self.core.rebuy_enabled)
        rebuy_bounce = self._module_attr("REBUY_BOUNCE_PCT", self.core.rebuy_bounce_pct)
        print(f"ENABLED={self.enabled}  REBUY={rebuy_en} (bounce {rebuy_bounce}%)")

    # -- one step / loop -------------------------------------------------------
    def check_once(self) -> None:
        super().check_once()
        # healthcheck.sh uses the log mtime as its heartbeat. When there is no free
        # balance to protect, the core intentionally exits without another message;
        # explicitly confirm that the process still completed the cycle successfully.
        self.log("  [TRAIL-K] heartbeat")

    def run(self):
        mode = "[ACTIVE] (it really sells)" if self.enabled else "DRY RUN (it only logs)"
        self._run_trailing_loop(
            CHECK_SECONDS,
            [
                "=== TRAILING STOP KRAKEN started -- " + mode + " ===",
                "    protecting: " + ", ".join(f"{a}={t}%" for a, t in TRAIL_PCT.items())
                + "  (FREE balance only, not the bot position)",
                f"    re-buy: {'ON' if REBUY_ENABLED else 'off'} (bounce +{REBUY_BOUNCE_PCT}% from trough)",
            ],
        )


def main() -> int:
    args = parse_trailing_cli_args("A trailing-stop circuit breaker on Kraken (with a re-buy).", "kraken_trailing")
    # A dedicated trailing key (KRAKEN_API_KEY_TRAIL) keeps its nonce separate from
    # _BOT (kraken_bot + xstock_watch). Fall back to _BOT when _TRAIL is absent.
    client = build_kraken_client(KrakenClient, kraken_credentials, "trail")
    ts = KrakenTrailing(client)
    return ts.dispatch_cli(args)


if __name__ == "__main__":
    sys.exit(main())

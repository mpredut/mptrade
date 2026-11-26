"""Autonomous AI Trading Intent Gateway.

Linear choke point and bridge between decoupled AI intent producers and the
unified MPTrade execution pipeline (Instrument.place & order_guard).

Operational Architecture:
1. Decoupled Ingestion:
   Reads high-conviction trade intents emitted into `cachedb/autonomous_trade_intents.json`.
2. Minimal Operational Mechanics (100% Strategic & Risk Judgment Delegated to Gemini):
   - Confidence threshold verification (`autonomous_trading_min_confidence`, e.g. >= 0.85).
   - Symmetric USD notional bounds (`[autonomous_trading_min_notional_usd, autonomous_trading_max_notional_usd]`, up to $2,000 USD) clamped to available spot wallet balance.
   - Symmetric bypass of ALL deterministic/strategy/daily-cap/LLM guards (`AUTONOMOUS_BYPASS_GUARDS` includes `"all"`, `"daily_limit"`, `"profit_guard"`, `"trend"`, `"qlimit"`, `"intelligence"`, `"llm"`) across both `BUY` and `SELL`.
3. Modes of Operation:
   - "shadow": Simulates order, validates operational preflight, logs simulated action, dispatches phone alerts via ntfy.
   - "enforce": Preflights operational safety, clamps to live spot wallet balance, and submits market order via canonical `Instrument.place()`.
4. Durable Intent Tracking:
   Updates intent lifecycle state: PENDING -> OBSERVED_SHADOW | EXECUTED | BLOCKED_BY_GUARD | REJECTED_*.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import sys
import time
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

logger = logging.getLogger("orchestratorTrade.autonomous_intent_gateway")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from botcore import (
    _parse_bool_flag,
    add_store_true_cli_args,
    clean_order_id,
    clean_scalar_str,
    clean_upper_scalar,
    disable_auto_start_websockets_default,
    extract_order_id,
    extract_order_price,
    extract_order_side,
    extract_timestamp_s,
    first_pos_float,
    format_notify_price,
    is_us_rth_open,
    load_key_value_conf,
    opposite_side,
    resolve_now_ts,
    safe_nonneg_float,
    safe_pos_float,
    split_csv,
)
from cachepaths import (
    AUTO_ORDER_INTENTS_STATE_FILENAME,
    CACHE_CURRENT_PRICE_FILENAME,
    VENUE_RUNTIME_DIR_MAP,
    cycle_state_filename,
    is_equity_venue_or_suffix,
    normalize_venue_name,
    resolve_cachedb_filepath,
)
from instrument import (
    load_tradable_instruments,
    resolve_instrument_live_price,
    resolve_tradable_instrument,
)
from instrument_registry import (
    discover_tradable_specs,
    resolve_instrument_spec,
    resolve_kraken_tokenized_pair,
    resolve_symbol_identity,
)
from intelligence.sentiment.collectors.guard_probe_collector import (
    probe_deterministic_guards,
)
import notify_engine.alertnotifiers as _alertnotifiers
from order_lifecycle import OrderPreemptionTransaction, restore_cancelled_instrument_orders
from providers.ai_circuit_breaker import (
    DEFAULT_FLIP_COOLDOWN_MIN,
    DEFAULT_FLIP_MIN_EDGE_PCT,
    DEFAULT_MAX_DAILY_BUY_NOTIONAL_USD,
    DEFAULT_MAX_DAILY_BUYS_PER_SYMBOL,
    DEFAULT_MAX_FLIPS_6H,
    DEFAULT_MAX_HOURLY_REPLACES_PER_SYMBOL,
    DEFAULT_MIN_TARGET_RR,
    DEFAULT_REPLACE_MIN_PRICE_DELTA_PCT,
    DEFAULT_SCORECARD_COOLDOWN_HOURS,
    DEFAULT_SCORECARD_MAX_LOSS_STREAK,
    DEFAULT_SCORECARD_WINDOW_TRADES,
    DEFAULT_TARGET_EV_FEE_MULT,
    DEFAULT_XSTOCK_SELL_COOLDOWN_HOURS,
    DEFAULT_XSTOCK_SELL_CRASH_MIN_CONFIDENCE,
    DEFAULT_XSTOCK_SELL_CRASH_MIN_TARGET_DROP_PCT,
    DEFAULT_XSTOCK_SELL_MIN_CONFIDENCE,
    DEFAULT_XSTOCK_SELL_MIN_PROFIT_PCT,
    DEFAULT_XSTOCK_SELL_MIN_TARGET_DROP_PCT,
    evaluate_ai_intent_circuit_breaker,
    parse_intent_circuit_breaker_config,
)
from providers.ai_survival_guard import (
    build_persisted_round_trip_fee_resolver,
    evaluate_ai_decision_scorecard,
    evaluate_ai_ping_pong_guard,
    evaluate_ai_target_ev_guard,
    evaluate_ai_tokenized_equity_sell_guard,
    extract_live_prices_from_cache_payload,
)
from providers.venue_metadata import ProviderVenueMetadataMixin
from providers.nav_drawdown import (
    DEFAULT_DAILY_NAV_DD_LIMIT_PCT,
    DEFAULT_DAILY_NAV_DD_MODE,
    evaluate_daily_nav_drawdown_guard,
)
from providers.nav_history_store import load_and_evaluate_nav_drawdown
from providers.intent_reconciliation import (
    reconcile_intents_from_execution_audit,
)
from providers.intent_execution import (
    cancel_existing_ai_open_orders,
    ensure_submit_account_cache_ready,
    execute_enforce_intent_order,
    find_duplicate_ai_open_limit_order,
    preflight_intent_and_resolve_price,
    reconcile_open_intent_by_client_id,
    record_shadow_intent_observation,
    resolve_intent_topic_tag,
)
from providers.intent_state import (
    _extract_intent_id,
    _intent_order_ids,
    append_intent_history,
    collect_known_ai_order_ids,
    extract_intent_cancel_order_ids,
    parse_intent_gateway_config,
    resolve_hold_cancel_target,
    validate_and_build_intent_dict,
)
from providers.intent_state_store import (
    durable_intents_lock,
    load_durable_intents_state,
    save_durable_intents_state,
    upsert_durable_intent,
)
from providers.intent_telemetry import (
    TERMINAL_TRADE_PLAN_STATUSES,
    extract_frozen_trade_plan_status,
)
from providers.order_id_context import extract_client_order_id, resolve_order_actor
from providers.order_ohlc_utils import extract_executed_qty, extract_remaining_qty
from providers.order_status_ssot import NON_FILL_ORDER_STATUSES
from providers.portfolio_state_store import resolve_kraken_trade_cache_position
from providers.quantity import clamp_notional_to_wallet_balance, resolve_assets
from providers.venue_fee_maintenance import read_persisted_venue_fee_spec
from state_io import cachedb_exists, read_json_dict

# Symmetric guard bypass set for autonomous AI orders across both BUY and SELL:
# Trade thesis, profit/loss protection, multi-tranche scaling, trend timing, microstructure, and
# macro risk are evaluated directly by the Gemini model on complete cachedb telemetry, so ALL
# order_guard gates (GO1 daily/cooldown, G_PROF profit reference, GT_* trends, G_QLIMIT*, GI_*, GLLM*)
# are bypassed symmetrically. Only confidence, USD bounds, real spot wallet balance clamping,
# and exchange lot/minNotional filters apply.
AUTONOMOUS_BYPASS_GUARDS: frozenset[str] = frozenset({
    "all",
    "daily_limit",
    "profit_guard",
    "trend",
    "qlimit",
    "intelligence",
    "llm",
})


MODE_RANK = {"off": 0, "shadow": 1, "enforce": 2}


def _normalize_max_mode(max_mode: Any) -> str:
    norm = "off" if isinstance(max_mode, bool) else str(max_mode or "shadow").strip().lower()
    return norm if norm in MODE_RANK else "off"


class AutonomousIntentGateway:
    """Gateway bridge consuming decoupled trade intents and executing through Instrument.place."""

    def __init__(
        self,
        workspace_dir: Optional[str] = None,
        cache_dir: str = "cachedb",
        mode: Optional[str] = None,
        max_notional_usd: Optional[float] = None,
        min_confidence: Optional[float] = None,
        notify_enabled: Optional[bool] = None,
    ) -> None:
        self.workspace_dir = workspace_dir or ROOT_DIR
        self.cache_dir = os.path.join(self.workspace_dir, cache_dir)
        self.intents_file = os.path.join(self.cache_dir, AUTO_ORDER_INTENTS_STATE_FILENAME)

        conf = load_key_value_conf(self.workspace_dir, ROOT_DIR)
        (
            self.mode,
            self.min_notional_usd,
            self.max_notional_usd,
            self.min_confidence,
            self.notify_enabled,
        ) = parse_intent_gateway_config(
            conf,
            mode=mode,
            max_notional_usd=max_notional_usd,
            min_confidence=min_confidence,
            notify_enabled=notify_enabled,
        )
        cb_cfg = parse_intent_circuit_breaker_config(conf)
        self.max_daily_buys_per_symbol: int = cb_cfg["max_daily_buys_per_symbol"]
        self.max_hourly_replaces_per_symbol: int = cb_cfg["max_hourly_replaces_per_symbol"]
        self.replace_min_price_delta_pct: float = cb_cfg["replace_min_price_delta_pct"]
        self.max_daily_buy_notional_usd: float = cb_cfg["max_daily_buy_notional_usd"]
        self.flip_cooldown_min: float = cb_cfg["flip_cooldown_min"]
        self.flip_min_edge_pct: float = cb_cfg["flip_min_edge_pct"]
        self.target_ev_fee_mult: float = cb_cfg["target_ev_fee_mult"]
        self.min_target_rr: float = cb_cfg["min_target_rr"]
        self.max_flips_6h: int = cb_cfg["max_flips_6h"]
        self.scorecard_window_trades: int = cb_cfg["scorecard_window_trades"]
        self.scorecard_max_loss_streak: int = cb_cfg["scorecard_max_loss_streak"]
        self.scorecard_cooldown_hours: float = cb_cfg["scorecard_cooldown_hours"]
        self.daily_nav_dd_mode: str = cb_cfg["daily_nav_dd_mode"]
        self.daily_nav_dd_limit_pct: float = cb_cfg["daily_nav_dd_limit_pct"]
        self.xstock_sell_min_profit_pct: float = cb_cfg["xstock_sell_min_profit_pct"]
        self.xstock_sell_min_target_drop_pct: float = cb_cfg["xstock_sell_min_target_drop_pct"]
        self.xstock_sell_min_confidence: float = cb_cfg["xstock_sell_min_confidence"]
        self.xstock_sell_cooldown_hours: float = cb_cfg["xstock_sell_cooldown_hours"]
        self.xstock_sell_crash_min_target_drop_pct: float = cb_cfg["xstock_sell_crash_min_target_drop_pct"]
        self.xstock_sell_crash_min_confidence: float = cb_cfg["xstock_sell_crash_min_confidence"]
        self._cycle_quote_caps: Optional[Dict[Tuple[str, str], float]] = None
        self._last_cycle_clamp_balance: Optional[Dict[str, Any]] = None
        self._active_intent_freed_buy_quote: float = 0.0

    @contextlib.contextmanager
    def cycle_shared_cash_scope(self) -> Iterator[None]:
        """Track per-(venue, quote) BUY quote commitments across simultaneous intents within one cycle."""
        prev_caps = getattr(self, "_cycle_quote_caps", None)
        prev_bal = getattr(self, "_last_cycle_clamp_balance", None)
        prev_freed = getattr(self, "_active_intent_freed_buy_quote", 0.0)
        self._cycle_quote_caps = {}
        self._last_cycle_clamp_balance = None
        self._active_intent_freed_buy_quote = 0.0
        try:
            yield
        finally:
            self._cycle_quote_caps = prev_caps
            self._last_cycle_clamp_balance = prev_bal
            self._active_intent_freed_buy_quote = prev_freed

    @staticmethod
    def _resolve_cycle_quote_key(inst: Any) -> Optional[Tuple[str, str]]:
        provider = getattr(inst, "_provider", None)
        raw_venue = getattr(inst, "provider_name", None) or (getattr(provider, "name", None) if provider is not None else "")
        venue_l = normalize_venue_name(str(raw_venue or ""))
        raw_sym = getattr(inst, "symbol", "")
        raw_base, raw_quote = getattr(inst, "base", None), getattr(inst, "quote", None)
        _, quote_asset = resolve_assets(
            str(raw_sym or ""),
            raw_base if isinstance(raw_base, str) else None,
            raw_quote if isinstance(raw_quote, str) else None,
        )
        return (venue_l, quote_asset.upper()) if (venue_l and quote_asset) else None

    @staticmethod
    def _sum_canceled_buy_quote(canceled_order_snapshots: List[Dict[str, Any]]) -> float:
        total = 0.0
        for snap in canceled_order_snapshots:
            if not isinstance(snap, dict) or extract_order_side(snap) != "BUY":
                continue
            px = extract_order_price(snap, default=0.0) or 0.0
            qty = extract_remaining_qty(snap)
            if px > 0.0 and qty is not None and qty > 0.0:
                total += px * qty
        return total

    def _commit_cycle_quote_spend(self, decision: str, committed_usd: float) -> None:
        cycle_caps = getattr(self, "_cycle_quote_caps", None)
        if not isinstance(cycle_caps, dict) or str(decision or "").strip().upper() != "BUY":
            return
        bal_info = getattr(self, "_last_cycle_clamp_balance", None)
        if not isinstance(bal_info, dict) or committed_usd <= 0.0:
            return
        ven = bal_info.get("venue")
        asset = bal_info.get("asset")
        pre_bal = safe_nonneg_float(bal_info.get("balance"), default=None)
        if ven and asset and pre_bal is not None:
            cycle_caps[(str(ven), str(asset).upper())] = max(0.0, pre_bal - committed_usd)

    def load_all_intents(self) -> List[Dict[str, Any]]:
        """Loads all intents from the durable JSON queue with `.previous` crash recovery."""
        return load_durable_intents_state(self.intents_file, reconcile_fn=self.reconcile_intents)

    def reconcile_intents(self, intents: List[Dict[str, Any]]) -> bool:
        """Reconciles FAILED/PENDING/PENDING_EXECUTION/SUBMIT_UNKNOWN and incomplete EXECUTED intents against OrderLedger (with legacy execution_audit fallback)."""
        audit_dir = os.path.join(self.workspace_dir, "logger", "execution_audit")
        cachedb_dir = os.path.join(self.workspace_dir, "cachedb")
        updated = reconcile_intents_from_execution_audit(intents, audit_dir, cachedb_dir=cachedb_dir)
        if updated:
            self.save_all_intents(intents)
        return updated

    def save_all_intents(self, intents: List[Dict[str, Any]]) -> bool:
        """Transactionally persists all intents to disk using `.previous` staging and atomic replace."""
        return save_durable_intents_state(self.intents_file, intents, log=logger)

    def persist_resolved_trade_plan_statuses(
        self,
        resolved_by_intent_id: Mapping[str, str],
        *,
        now_ts: Optional[float] = None,
    ) -> int:
        """Write-once locked persistence of terminal trade-plan statuses so 7d calibration stays immutable past 24h."""
        clean_updates = {
            iid: st_u
            for raw_id, raw_st in (resolved_by_intent_id.items() if isinstance(resolved_by_intent_id, Mapping) else ())
            if (iid := clean_scalar_str(raw_id))
            and (st_u := clean_upper_scalar(raw_st) or "") in TERMINAL_TRADE_PLAN_STATUSES
        }
        if not clean_updates:
            return 0
        eff_now = round(resolve_now_ts(now_ts), 3)
        updated = 0
        with durable_intents_lock(self.intents_file):
            intents = load_durable_intents_state(self.intents_file, read_only=False)
            for item in intents:
                if not isinstance(item, dict):
                    continue
                iid = _extract_intent_id(item)
                if not iid or iid not in clean_updates or extract_frozen_trade_plan_status(item):
                    continue
                item["resolved_plan_status"] = clean_updates[iid]
                item["resolved_plan_ts"] = eff_now
                updated += 1
            if updated > 0:
                self.save_all_intents(intents)
        return updated

    def _mark_canceled_ai_intents(self, canceled_ids: List[str], now_ts: float) -> int:
        """Mark unfilled EXECUTED AI intents matching standalone-canceled order IDs as terminal CANCELED."""
        clean_ids = {c for raw in canceled_ids if (c := clean_order_id(raw))}
        if not clean_ids:
            return 0
        eff_now = round(resolve_now_ts(now_ts), 3)
        updated = 0
        with durable_intents_lock(self.intents_file):
            intents = load_durable_intents_state(self.intents_file, read_only=False)
            for item in intents:
                if not isinstance(item, dict) or (clean_upper_scalar(item.get("status")) or "") != "EXECUTED":
                    continue
                ep = item.get("execution_payload")
                if not isinstance(ep, dict) or extract_executed_qty(ep, default=0.0) > 0.0:
                    continue
                if (clean_upper_scalar(ep.get("status")) or "") in ("FILLED", "CLOSED", "CANCELED", "CANCELLED"):
                    continue
                matched = _intent_order_ids(item) & clean_ids
                if not matched:
                    continue
                ep_new = dict(ep)
                ep_new["status"] = "CANCELED"
                ep_new.pop("msg", None)
                ep_new["terminal_ts"] = eff_now
                ep_new["reconciled_terminal"] = True
                item["execution_payload"] = ep_new
                append_intent_history(
                    item,
                    {"ts": eff_now, "action": "CANCELED_STANDALONE", "canceled_order_ids": sorted(matched)},
                )
                updated += 1
            if updated > 0:
                self.save_all_intents(intents)
        return updated

    def load_tradable_instruments(
        self,
        *,
        only_workspace_conf: bool = False,
        include_closed_rth: bool = False,
        now_ts: Optional[float] = None,
        asset_scope: str = "all",
    ) -> Dict[str, Any]:
        """Load enabled instruments filtered by asset_scope ('crypto' | 'equities' | 'all')."""
        return load_tradable_instruments(
            self.workspace_dir,
            only_workspace_conf=only_workspace_conf,
            include_closed_rth=include_closed_rth,
            now_ts=now_ts,
            asset_scope=asset_scope,
            is_rth_open_fn=is_us_rth_open,
        )

    def resolve_instrument(
        self,
        symbol: str,
        venue: Optional[str] = None,
        instruments_map: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """Resolves an enabled Instrument instance for the given symbol and venue."""
        if instruments_map is None:
            instruments_map = self.load_tradable_instruments()
        return resolve_tradable_instrument(
            symbol,
            venue=venue,
            instruments_map=instruments_map,
            workspace_dir=self.workspace_dir,
        )

    def _get_live_price(self, inst: Any) -> float:
        """Fetches live market price from provider or cache."""
        return resolve_instrument_live_price(inst, cache_dir=self.cache_dir)

    def _load_live_prices_for_scorecard(
        self,
        symbol: str,
        venue: str,
        spot_price: float,
    ) -> Dict[str, float]:
        """Collect local live spot prices for MTM scoring of open AI BUY intents (0 ms, no network)."""
        prices: Dict[str, float] = {}
        try:
            cp_path = resolve_cachedb_filepath(os.path.join(self.cache_dir, CACHE_CURRENT_PRICE_FILENAME))
            if cp_path and cachedb_exists(cp_path, include_previous=True):
                data = read_json_dict(cp_path, default=None, recover_previous=True)
                prices = extract_live_prices_from_cache_payload(data)
        except Exception:
            pass
        if (sp := safe_pos_float(spot_price, default=None)) is not None:
            prices[symbol] = sp
            if venue:
                prices[f"{venue}:{symbol}"] = sp
        return prices

    @staticmethod
    def _resolve_recent_bot_opposite_fill(
        inst: Any,
        decision: str,
        now_ts: float,
        all_intents: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Return latest recent by=BOT opposite-side fill from local provider order cache (mock-safe)."""
        opp = opposite_side(decision)
        if opp not in ("BUY", "SELL"):
            return None
        prov = getattr(inst, "_provider", None)
        get_orders_fn = getattr(prov, "get_orders", None) if prov is not None else None
        if not callable(get_orders_fn):
            return None
        try:
            raw_orders = get_orders_fn(inst.symbol, opp, 1800)
        except Exception:
            return None
        if not isinstance(raw_orders, list) or not raw_orders:
            return None
        inst_venue = getattr(inst, "provider_name", None)
        known_ai_oids = collect_known_ai_order_ids(all_intents, venue=inst_venue, symbol=inst.symbol)
        latest: Optional[Dict[str, Any]] = None
        for ord_item in raw_orders:
            if not isinstance(ord_item, dict):
                continue
            # Ignore synthetic fallback placeholders (e.g. T212 averagePrice position fallback with no order/client ID)
            if not extract_order_id(ord_item, default="") and not extract_client_order_id(ord_item, default=""):
                continue
            ord_st = clean_upper_scalar(ord_item.get("status")) or ""
            if ord_st in NON_FILL_ORDER_STATUSES:
                continue
            if resolve_order_actor(ord_item, known_ai_order_ids=known_ai_oids, venue=inst_venue, symbol=inst.symbol) != "BOT":
                continue
            ts_s = extract_timestamp_s(ord_item, default=0.0) or 0.0
            px_f = extract_order_price(ord_item, default=0.0) or 0.0
            if ts_s <= 0.0 or ts_s > now_ts + 60.0 or px_f <= 0.0:
                continue
            if latest is None or ts_s > latest["ts"]:
                latest = {"ts": ts_s, "price": px_f, "side": opp, "actor": "BOT"}
        return latest

    def _clamp_notional_to_wallet_balance(
        self,
        inst: Any,
        decision: str,
        price: float,
        desired_usd: float,
    ) -> Tuple[bool, float, str]:
        """Clamp desired USD notional to real spot wallet liquidity via unified SSOT in providers.quantity."""
        cycle_caps = getattr(self, "_cycle_quote_caps", None)
        if isinstance(cycle_caps, dict) and str(decision or "").strip().upper() == "BUY":
            max_q: Optional[float] = None
            if (q_key := self._resolve_cycle_quote_key(inst)) is not None and q_key in cycle_caps:
                freed_q = safe_nonneg_float(getattr(self, "_active_intent_freed_buy_quote", 0.0), default=0.0) or 0.0
                max_q = cycle_caps[q_key] + freed_q
            bal_info: Dict[str, Any] = {}
            res = clamp_notional_to_wallet_balance(
                inst,
                decision,
                price,
                desired_usd,
                min_notional_usd=self.min_notional_usd,
                max_quote_balance=max_q,
                effective_balance_out=bal_info,
            )
            self._last_cycle_clamp_balance = bal_info if bal_info else None
            return res
        return clamp_notional_to_wallet_balance(
            inst,
            decision,
            price,
            desired_usd,
            min_notional_usd=self.min_notional_usd,
        )

    def _resolve_tokenized_equity_cost_basis(self, venue: str, symbol: str) -> Optional[float]:
        """Resolve active unit cost basis for a tokenized equity from trade-cache fills, falling back to venue cycle state or config."""
        ven = normalize_venue_name(venue) or ""
        rel_dir = VENUE_RUNTIME_DIR_MAP.get(ven, ven)
        if not rel_dir:
            return None
        venue_dir = os.path.join(self.workspace_dir, rel_dir)
        ident = resolve_symbol_identity(symbol, venue=ven)
        wire_sym = ident.wire_symbol or ident.display_symbol or str(symbol or "").strip()
        cand_syms = tuple(dict.fromkeys((wire_sym, str(symbol or "").strip(), wire_sym.upper())))
        if resolve_kraken_tokenized_pair(wire_sym)[1] and self.cache_dir:
            for cand_sym in cand_syms:
                if not cand_sym:
                    continue
                cache_pos = resolve_kraken_trade_cache_position(cand_sym, self.cache_dir)
                if (
                    isinstance(cache_pos, dict)
                    and (safe_nonneg_float(cache_pos.get("net_fills_qty"), default=0.0) or 0.0) > 1e-6
                    and (cache_px := safe_pos_float(cache_pos.get("avg_cost_px"), default=None)) is not None
                ):
                    return cache_px
        for cand_sym in cand_syms:
            if not cand_sym:
                continue
            st_path = os.path.join(venue_dir, cycle_state_filename(cand_sym, preserve_case=True))
            if cachedb_exists(st_path, include_previous=True):
                st_data = read_json_dict(st_path, default={}, recover_previous=True)
                qty_f = safe_nonneg_float(st_data.get("qty"), default=0.0) or 0.0
                cost_f = safe_nonneg_float(st_data.get("cost_usd") if st_data.get("cost_usd") is not None else st_data.get("cost"), default=0.0) or 0.0
                if qty_f > 1e-6:
                    px_v = first_pos_float(
                        (cost_f / qty_f) if cost_f > 0.0 else None,
                        st_data.get("entry_price"),
                        st_data.get("last_buy_price"),
                    )
                    if px_v is not None:
                        return px_v
        cfg_map = load_key_value_conf(venue_dir, root_dir=venue_dir)
        for raw_spec in split_csv(cfg_map.get("xstock_active_pairs") or ""):
            raw_p, _, raw_c = raw_spec.partition(":")
            norm_p = resolve_kraken_tokenized_pair(raw_p.strip())[0] if raw_p.strip() else ""
            if norm_p and norm_p.upper() == wire_sym.upper():
                if (cfg_px := safe_pos_float(raw_c, default=None)) is not None:
                    return cfg_px
        return None

    def _dispatch_notify(self, title: str, body: str, symbol: str) -> bool:
        """Dispatches an alert notification via notify_engine."""
        if not self.notify_enabled:
            return False

        try:
            _alertnotifiers.notify(
                title=title,
                body=body,
                source="llm_trader",
                symbol=symbol,
            )
            return True
        except Exception as e:
            logger.debug("Failed notifying via notify_engine: %s", e)
            return False

    @staticmethod
    def _rollback_canceled_orders(
        inst: Any,
        intent: Dict[str, Any],
        canceled_order_snapshots: List[Dict[str, Any]],
        *,
        reason: str,
    ) -> None:
        if not canceled_order_snapshots:
            return
        restored_out: List[Dict[str, Any]] = []
        restored = restore_cancelled_instrument_orders(
            inst,
            canceled_order_snapshots,
            bypass_guards=AUTONOMOUS_BYPASS_GUARDS,
            log_fn=logger.warning,
            reason=reason,
            restored_orders_out=restored_out,
        )
        if restored > 0:
            intent["restored_prior_open_orders"] = restored
            ai_ids: List[str] = list(intent.get("restored_ai_order_ids") or [])
            for item in restored_out:
                if isinstance(item, dict) and item.get("actor") == "AI":
                    for k in ("order_id", "client_order_id"):
                        if (cid := clean_order_id(item.get(k))) and cid not in ai_ids:
                            ai_ids.append(cid)
            if ai_ids:
                intent["restored_ai_order_ids"] = ai_ids

    @staticmethod
    def _abort_on_cancel_failure(
        inst: Any,
        intent: Dict[str, Any],
        *,
        decision: str,
        failed_order_ids: List[str],
        scan_errors: List[str],
        now_ts: float,
    ) -> Dict[str, Any]:
        """Stop order replacement when canceling any targeted prior open order fails or cannot be verified."""
        sym_lbl = str(getattr(inst, "symbol", "") or intent.get("symbol") or "")
        if failed_order_ids:
            intent["failed_cancel_order_ids"] = list(failed_order_ids)
            reason = f"cancel_prior_open_order_failed:{','.join(failed_order_ids)}"
        else:
            reason = f"open_orders_scan_failed:{scan_errors[0] if scan_errors else 'unknown'}"
        logger.warning(
            "[LLM TRADER \u00b7 ENFORCE] Aborting replacement %s %s: %s",
            decision, sym_lbl, reason,
        )
        intent["status"] = "FAILED"
        hist_entry: Dict[str, Any] = {
            "ts": now_ts,
            "action": "CANCEL_PRIOR_ORDER_FAILED",
            "reason": reason,
        }
        if failed_order_ids:
            hist_entry["failed_order_ids"] = list(failed_order_ids)
        if scan_errors:
            hist_entry["scan_errors"] = list(scan_errors)
        append_intent_history(intent, hist_entry)
        return intent

    def _cancel_prior_orders_or_abort(
        self,
        inst: Any,
        intent: Optional[Dict[str, Any]] = None,
        *,
        intent_id: str = "",
        decision: str = "",
        now_ts: float = 0.0,
        canceled_order_snapshots: Optional[List[Dict[str, Any]]] = None,
        rollback_reason: str = "",
        include_bot_orders: bool = False,
        only_bot_orders: bool = False,
        any_side: bool = False,
        exclude_order_ids: Optional[set[str]] = None,
        target_order_ids: Optional[List[str]] = None,
        verify_not_filled: bool = False,
        extra_intent_key: Optional[str] = None,
        intents: Optional[List[Dict[str, Any]]] = None,
        failed_ids_out: Optional[List[str]] = None,
        scan_errors_out: Optional[List[str]] = None,
    ) -> Tuple[bool, List[str]]:
        if intent is not None:
            cache_ok, cache_reason = ensure_submit_account_cache_ready(inst, side=decision)
            if not cache_ok:
                sym_lbl = str(getattr(inst, "symbol", "") or intent.get("symbol") or "")
                if canceled_order_snapshots:
                    self._rollback_canceled_orders(
                        inst,
                        intent,
                        list(canceled_order_snapshots),
                        reason=f"intent {intent_id} {cache_reason}",
                    )
                    canceled_order_snapshots.clear()
                logger.warning(
                    "[LLM TRADER \u00b7 ENFORCE] Aborting pre-submit cancel for %s %s: %s (keeping prior open orders active)",
                    decision,
                    sym_lbl,
                    cache_reason,
                )
                intent["status"] = "FAILED"
                append_intent_history(intent, {
                    "ts": now_ts,
                    "action": "FAILED",
                    "reason": cache_reason,
                    "category": "refused_pre_submit",
                })
                return False, []
        failed_ids: List[str] = []
        scan_errors: List[str] = []
        canceled_ids = cancel_existing_ai_open_orders(
            inst,
            decision,
            log=logger,
            intents=intents if intents is not None else self.load_all_intents(),
            include_bot_orders=include_bot_orders,
            only_bot_orders=only_bot_orders,
            any_side=any_side,
            exclude_order_ids=exclude_order_ids or None,
            target_order_ids=target_order_ids or None,
            verify_not_filled=verify_not_filled,
            failed_order_ids=failed_ids,
            scan_errors=scan_errors,
            canceled_orders_out=canceled_order_snapshots,
        )
        if failed_ids_out is not None and failed_ids:
            failed_ids_out.extend(failed_ids)
        if scan_errors_out is not None and scan_errors:
            scan_errors_out.extend(scan_errors)
        if intent is None:
            return not (failed_ids or scan_errors), canceled_ids
        if canceled_ids:
            prior_c = list(intent.get("canceled_prior_open_orders") or [])
            for cid in canceled_ids:
                if cid not in prior_c:
                    prior_c.append(cid)
            intent["canceled_prior_open_orders"] = prior_c
            if extra_intent_key:
                new_snaps = (canceled_order_snapshots or [])[-len(canceled_ids):]
                bot_cids = (
                    [
                        cid
                        for cid, snap in zip(canceled_ids, new_snaps)
                        if isinstance(snap, dict) and snap.get("actor") != "AI"
                    ]
                    if new_snaps
                    else list(canceled_ids)
                )
                if bot_cids:
                    intent[extra_intent_key] = bot_cids
        if failed_ids or scan_errors:
            self._rollback_canceled_orders(
                inst,
                intent,
                canceled_order_snapshots or [],
                reason=f"intent {intent_id} {rollback_reason}",
            )
            self._abort_on_cancel_failure(
                inst,
                intent,
                decision=decision,
                failed_order_ids=failed_ids,
                scan_errors=scan_errors,
                now_ts=now_ts,
            )
            return False, canceled_ids
        return True, canceled_ids

    @staticmethod
    def _record_duplicate_open_order_skip(
        inst: Any,
        intent: Dict[str, Any],
        *,
        decision: str,
        price: float,
        effective_usd: float,
        dup_oid: str,
        now_ts: float,
    ) -> Dict[str, Any]:
        logger.info(
            "[LLM TRADER \u00b7 ENFORCE] Skipping duplicate %s LIMIT %s @ $%.4f ($%.2f); identical AI order %s is already open.",
            decision, inst.symbol, price, effective_usd, dup_oid,
        )
        intent["status"] = "SKIPPED_DUPLICATE_OPEN_ORDER"
        intent["duplicate_of_order_id"] = dup_oid
        append_intent_history(intent, {
            "ts": now_ts,
            "action": "SKIPPED_DUPLICATE_OPEN_ORDER",
            "duplicate_of_order_id": dup_oid,
            "price": price,
            "notional_usd": effective_usd,
        })
        return intent

    def _block_by_circuit_breaker(
        self,
        inst: Any,
        intent: Dict[str, Any],
        *,
        mode: str,
        decision: str,
        price: float,
        effective_usd: float,
        confidence: float,
        thesis: str,
        trade_tag: str,
        now_ts: float,
        is_limit: bool,
        reason_str: str,
        metrics_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        cb_qty = effective_usd / price if (effective_usd > 0.0 and price > 0.0) else 0.0
        if mode == "shadow":
            return record_shadow_intent_observation(
                inst,
                intent,
                decision=decision,
                target_qty=cb_qty,
                price=price,
                effective_usd=effective_usd,
                confidence=confidence,
                guard_allowed=False,
                guard_reason=reason_str,
                thesis=thesis,
                trade_tag=trade_tag,
                now_ts=now_ts,
                is_limit=is_limit,
                dispatch_notify_fn=self._dispatch_notify,
                log=logger,
            )
        logger.warning(
            "[LLM TRADER \u00b7 ENFORCE] AI circuit-breaker blocked %s %s: %s",
            decision, inst.symbol, reason_str,
        )
        _alertnotifiers.emit_trade_notification(
            self._dispatch_notify,
            "BLOCKED",
            decision,
            inst.symbol,
            format_notify_price(price, comma=False),
            cb_qty,
            effective_usd,
            venue=inst.provider_name,
            owner=trade_tag,
            detail=("LIMIT" if is_limit else "MARKET", f"Reason: {reason_str}"),
            thesis=thesis,
            notify_kwargs={"symbol": inst.symbol},
        )
        intent["status"] = "REJECTED_AI_CIRCUIT_BREAKER"
        intent["circuit_breaker_reason"] = reason_str
        intent["circuit_breaker_metrics"] = metrics_dict
        append_intent_history(intent, {
            "ts": now_ts,
            "action": "REJECTED_AI_CIRCUIT_BREAKER",
            "reason": reason_str,
            **metrics_dict,
        })
        return intent

    def _resolve_live_round_trip_fee_bps(
        self,
        venue: str,
        symbol: str,
        *,
        is_limit: bool,
        now_ts: float,
    ) -> Optional[float]:
        """Resolve live persisted round-trip (2x leg) fee bps when fresh fee state exists; else None."""
        cache_dir = getattr(self, "cache_dir", "")
        if not cache_dir:
            return None
        spec = read_persisted_venue_fee_spec(venue, symbol, cache_dir=cache_dir, now_ts=now_ts)
        if not isinstance(spec, dict) or not spec:
            return None
        leg_key = "maker_fee_bps" if is_limit else "taker_fee_bps"
        leg_bps = safe_nonneg_float(spec.get(leg_key), default=None)
        return round(2.0 * leg_bps, 4) if leg_bps is not None else None

    def _evaluate_ai_circuit_breakers_and_survival(
        self,
        inst: Any,
        intent: Dict[str, Any],
        all_intents: List[Dict[str, Any]],
        *,
        mode: str,
        decision: str,
        symbol: str,
        venue: str,
        price: float,
        effective_usd: float,
        confidence: float,
        thesis: str,
        trade_tag: str,
        now_ts: float,
        is_limit: bool,
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        """Run AI circuit-breaker, anti-ping-pong guard, and 48h scorecard; return (short_circuit_intent, updated_usd)."""
        cb_allowed, cb_usd, cb_reason, cb_metrics = evaluate_ai_intent_circuit_breaker(
            intent,
            all_intents,
            decision=decision,
            symbol=symbol,
            venue=venue,
            price=price,
            effective_usd=effective_usd,
            min_notional_usd=self.min_notional_usd,
            now_ts=now_ts,
            max_daily_buys_per_symbol=getattr(self, "max_daily_buys_per_symbol", DEFAULT_MAX_DAILY_BUYS_PER_SYMBOL),
            max_hourly_replaces_per_symbol=getattr(self, "max_hourly_replaces_per_symbol", DEFAULT_MAX_HOURLY_REPLACES_PER_SYMBOL),
            replace_min_price_delta_pct=getattr(self, "replace_min_price_delta_pct", DEFAULT_REPLACE_MIN_PRICE_DELTA_PCT),
            max_daily_buy_notional_usd=getattr(self, "max_daily_buy_notional_usd", DEFAULT_MAX_DAILY_BUY_NOTIONAL_USD),
        )
        if cb_allowed and cb_reason and cb_usd < effective_usd:
            logger.info(
                "[LLM TRADER] Clamped %s %s notional from $%.2f to $%.2f (%s).",
                decision, inst.symbol, effective_usd, cb_usd, cb_reason,
            )
            effective_usd = cb_usd
            intent["circuit_breaker_clamped"] = cb_reason
        elif not cb_allowed:
            return (
                self._block_by_circuit_breaker(
                    inst, intent, mode=mode, decision=decision, price=price, effective_usd=effective_usd,
                    confidence=confidence, thesis=thesis, trade_tag=trade_tag, now_ts=now_ts,
                    is_limit=is_limit, reason_str=cb_reason, metrics_dict=cb_metrics,
                ),
                effective_usd,
            )

        live_rt_fee_bps = self._resolve_live_round_trip_fee_bps(
            venue, symbol, is_limit=is_limit, now_ts=now_ts
        )
        ev_allowed, ev_reason, ev_metrics = evaluate_ai_target_ev_guard(
            intent,
            decision=decision,
            symbol=symbol,
            venue=venue,
            price=price,
            target_ev_fee_mult=getattr(self, "target_ev_fee_mult", DEFAULT_TARGET_EV_FEE_MULT),
            min_target_rr=getattr(self, "min_target_rr", DEFAULT_MIN_TARGET_RR),
            round_trip_fee_bps=live_rt_fee_bps,
        )
        if not ev_allowed:
            return (
                self._block_by_circuit_breaker(
                    inst, intent, mode=mode, decision=decision, price=price, effective_usd=effective_usd,
                    confidence=confidence, thesis=thesis, trade_tag=trade_tag, now_ts=now_ts,
                    is_limit=is_limit, reason_str=ev_reason, metrics_dict=ev_metrics,
                ),
                effective_usd,
            )

        if (
            decision == "SELL"
            and ProviderVenueMetadataMixin.supports_tokenized_assets(venue)
            and resolve_symbol_identity(symbol, venue=venue).is_tokenized_equity
        ):
            tok_cost_px = self._resolve_tokenized_equity_cost_basis(venue, symbol)
            xs_allowed, xs_reason, xs_metrics = evaluate_ai_tokenized_equity_sell_guard(
                intent,
                all_intents,
                decision=decision,
                symbol=symbol,
                venue=venue,
                price=price,
                confidence=confidence,
                now_ts=now_ts,
                cost_basis_px=tok_cost_px,
                min_profit_pct=getattr(self, "xstock_sell_min_profit_pct", DEFAULT_XSTOCK_SELL_MIN_PROFIT_PCT),
                min_target_drop_pct=getattr(self, "xstock_sell_min_target_drop_pct", DEFAULT_XSTOCK_SELL_MIN_TARGET_DROP_PCT),
                min_confidence=getattr(self, "xstock_sell_min_confidence", DEFAULT_XSTOCK_SELL_MIN_CONFIDENCE),
                cooldown_hours=getattr(self, "xstock_sell_cooldown_hours", DEFAULT_XSTOCK_SELL_COOLDOWN_HOURS),
                crash_min_target_drop_pct=getattr(
                    self,
                    "xstock_sell_crash_min_target_drop_pct",
                    DEFAULT_XSTOCK_SELL_CRASH_MIN_TARGET_DROP_PCT,
                ),
                crash_min_confidence=getattr(
                    self,
                    "xstock_sell_crash_min_confidence",
                    DEFAULT_XSTOCK_SELL_CRASH_MIN_CONFIDENCE,
                ),
            )
            if not xs_allowed:
                return (
                    self._block_by_circuit_breaker(
                        inst, intent, mode=mode, decision=decision, price=price, effective_usd=effective_usd,
                        confidence=confidence, thesis=thesis, trade_tag=trade_tag, now_ts=now_ts,
                        is_limit=is_limit, reason_str=xs_reason, metrics_dict=xs_metrics,
                    ),
                    effective_usd,
                )

        live_spot_px = price
        if is_limit:
            try:
                fetched_spot = safe_pos_float(self._get_live_price(inst), default=None)
                if fetched_spot is not None:
                    live_spot_px = fetched_spot
            except Exception:
                pass

        recent_bot_fill = self._resolve_recent_bot_opposite_fill(inst, decision, now_ts, all_intents)
        pp_allowed, pp_reason, pp_metrics = evaluate_ai_ping_pong_guard(
            intent,
            all_intents,
            decision=decision,
            symbol=symbol,
            venue=venue,
            price=price,
            now_ts=now_ts,
            flip_cooldown_min=getattr(self, "flip_cooldown_min", DEFAULT_FLIP_COOLDOWN_MIN),
            flip_min_edge_pct=getattr(self, "flip_min_edge_pct", DEFAULT_FLIP_MIN_EDGE_PCT),
            max_flips_6h=getattr(self, "max_flips_6h", DEFAULT_MAX_FLIPS_6H),
            spot_price=live_spot_px,
            recent_bot_fill=recent_bot_fill,
            round_trip_fee_bps=live_rt_fee_bps,
        )
        if not pp_allowed:
            return (
                self._block_by_circuit_breaker(
                    inst, intent, mode=mode, decision=decision, price=price, effective_usd=effective_usd,
                    confidence=confidence, thesis=thesis, trade_tag=trade_tag, now_ts=now_ts,
                    is_limit=is_limit, reason_str=pp_reason, metrics_dict=pp_metrics,
                ),
                effective_usd,
            )

        sc_win = getattr(self, "scorecard_window_trades", DEFAULT_SCORECARD_WINDOW_TRADES)
        sc_streak = getattr(self, "scorecard_max_loss_streak", DEFAULT_SCORECARD_MAX_LOSS_STREAK)
        sc_cd_h = getattr(self, "scorecard_cooldown_hours", DEFAULT_SCORECARD_COOLDOWN_HOURS)
        if decision == "BUY" and sc_win > 0 and sc_streak > 0 and sc_cd_h > 0.0:
            sc_prices = self._load_live_prices_for_scorecard(symbol, venue, live_spot_px)
            sc_scope = "equities" if is_equity_venue_or_suffix(symbol, venue) else "crypto"
            sc_fee_resolver = build_persisted_round_trip_fee_resolver(
                getattr(self, "cache_dir", None), now_ts=now_ts
            )
            scorecard = evaluate_ai_decision_scorecard(
                all_intents,
                now_ts=now_ts,
                live_prices=sc_prices,
                window_trades=sc_win,
                max_loss_streak=sc_streak,
                cooldown_hours=sc_cd_h,
                scope=sc_scope,
                fee_resolver=sc_fee_resolver,
            )
            if scorecard.get("buy_shadow_active"):
                sc_reason = str(scorecard.get("reason") or "ai_scorecard_buy_shadow_cooldown")
                intent["scorecard_downgraded_to_shadow"] = sc_reason
                intent["ai_decision_scorecard"] = scorecard
                sc_qty = effective_usd / price if (effective_usd > 0.0 and price > 0.0) else 0.0
                logger.warning(
                    "[LLM TRADER] Downgrading BUY %s to shadow via AI decision scorecard: %s",
                    inst.symbol, sc_reason,
                )
                return (
                    record_shadow_intent_observation(
                        inst,
                        intent,
                        decision=decision,
                        target_qty=sc_qty,
                        price=price,
                        effective_usd=effective_usd,
                        confidence=confidence,
                        guard_allowed=False,
                        guard_reason=sc_reason,
                        thesis=thesis,
                        trade_tag=trade_tag,
                        now_ts=now_ts,
                        is_limit=is_limit,
                        dispatch_notify_fn=self._dispatch_notify,
                        log=logger,
                    ),
                    effective_usd,
                )

        nav_dd_mode = getattr(self, "daily_nav_dd_mode", DEFAULT_DAILY_NAV_DD_MODE)
        nav_dd_limit = getattr(self, "daily_nav_dd_limit_pct", DEFAULT_DAILY_NAV_DD_LIMIT_PCT)
        if decision == "BUY" and nav_dd_mode in ("shadow", "enforce") and nav_dd_limit > 0.0:
            nav_metrics = load_and_evaluate_nav_drawdown(self.workspace_dir, now_ts=now_ts, include_live_point=True)
            nav_ok, nav_reason, nav_g_metrics = evaluate_daily_nav_drawdown_guard(
                nav_metrics,
                decision=decision,
                mode=nav_dd_mode,
                limit_pct=nav_dd_limit,
            )
            if not nav_ok:
                intent["nav_drawdown_metrics"] = nav_g_metrics
                if nav_dd_mode == "shadow":
                    intent["daily_nav_dd_shadow_warning"] = nav_reason
                    logger.warning(
                        "[LLM TRADER] Observational daily NAV drawdown warning on BUY %s (shadow mode, order allowed): %s",
                        inst.symbol, nav_reason,
                    )
                else:
                    intent["daily_nav_dd_downgraded_to_shadow"] = nav_reason
                    nav_qty = effective_usd / price if (effective_usd > 0.0 and price > 0.0) else 0.0
                    logger.warning(
                        "[LLM TRADER] Downgrading BUY %s to shadow via daily NAV drawdown guard: %s",
                        inst.symbol, nav_reason,
                    )
                    return (
                        record_shadow_intent_observation(
                            inst,
                            intent,
                            decision=decision,
                            target_qty=nav_qty,
                            price=price,
                            effective_usd=effective_usd,
                            confidence=confidence,
                            guard_allowed=False,
                            guard_reason=nav_reason,
                            thesis=thesis,
                            trade_tag=trade_tag,
                            now_ts=now_ts,
                            is_limit=is_limit,
                            dispatch_notify_fn=self._dispatch_notify,
                            log=logger,
                        ),
                        effective_usd,
                    )
        return None, effective_usd

    def _unlock_and_clamp_wallet_notional(
        self,
        inst: Any,
        intent: Dict[str, Any],
        all_intents: List[Dict[str, Any]],
        *,
        mode: str,
        intent_id: str,
        decision: str,
        price: float,
        effective_usd: float,
        now_ts: float,
        replace_open: bool,
        cancel_bot_open: bool,
        target_oids: List[str],
        thesis_cancels_bot_order: bool,
        canceled_order_snapshots: List[Dict[str, Any]],
    ) -> Tuple[bool, bool, float, str]:
        """Clamp notional to live wallet balance, canceling bot-owned locking orders on enforce when needed."""
        requested_effective_usd = effective_usd
        self._last_cycle_clamp_balance = None
        prev_freed = getattr(self, "_active_intent_freed_buy_quote", 0.0)
        self._active_intent_freed_buy_quote = (
            self._sum_canceled_buy_quote(canceled_order_snapshots)
            if str(decision or "").strip().upper() == "BUY"
            else 0.0
        )
        try:
            guard_allowed, effective_usd, guard_reason = self._clamp_notional_to_wallet_balance(
                inst, decision, price, requested_effective_usd
            )
            is_balance_constrained = (
                (not guard_allowed and "insufficient free" in str(guard_reason or "").lower())
                or (guard_allowed and effective_usd < requested_effective_usd * 0.95)
            )
            # Only same-side resting orders lock the asset this intent spends (quote for BUY, base for SELL);
            # opposite-side orders (trailing stops, take-profits, SL) free nothing and must stay active.
            if (
                mode == "enforce"
                and replace_open
                and not (cancel_bot_open or target_oids or thesis_cancels_bot_order)
                and is_balance_constrained
            ):
                ok_unlock, unlocked_ids = self._cancel_prior_orders_or_abort(
                    inst,
                    intent,
                    intent_id=intent_id,
                    decision=decision,
                    now_ts=now_ts,
                    canceled_order_snapshots=canceled_order_snapshots,
                    rollback_reason="unlock_cancel_failed",
                    include_bot_orders=True,
                    only_bot_orders=True,
                    extra_intent_key="canceled_bot_open_orders",
                    intents=all_intents,
                )
                if not ok_unlock:
                    return False, guard_allowed, effective_usd, guard_reason
                if unlocked_ids:
                    if str(decision or "").strip().upper() == "BUY":
                        self._active_intent_freed_buy_quote = self._sum_canceled_buy_quote(canceled_order_snapshots)
                    guard_allowed, effective_usd, guard_reason = self._clamp_notional_to_wallet_balance(
                        inst, decision, price, requested_effective_usd
                    )
            return True, guard_allowed, effective_usd, guard_reason
        finally:
            self._active_intent_freed_buy_quote = prev_freed

    def _record_advisory_guard_overrides(
        self,
        intent: Dict[str, Any],
        *,
        symbol: str,
        decision: str,
        price: float,
        venue: str,
        now_ts: float,
    ) -> None:
        try:
            use_disk = os.path.abspath(self.workspace_dir) == os.path.abspath(ROOT_DIR)
            probe_res = probe_deterministic_guards(
                symbol,
                decision,
                price,
                venue=venue,
                now_ts=now_ts,
                allow_disk_cache=use_disk,
            )
            if probe_flags := probe_res.get("flags"):
                intent["advisory_guards_overridden"] = [
                    f"{f['guard']}:{f['action']}({f['reason']})" for f in probe_flags
                ]
        except Exception:
            pass

    def process_intent(
        self,
        intent: Dict[str, Any],
        instruments_map: Optional[Dict[str, Any]] = None,
        mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Validates and processes a single intent according to ``mode`` (default: the gateway mode)."""
        if not isinstance(intent, dict):
            return {}
        resume_reconcile_cid = bool(intent.pop("_resume_reconcile_cid", False))
        raw_cap = intent.get("max_mode")
        if isinstance(mode, str) and mode.strip():
            mode = mode.strip().lower()
        elif mode is None or mode == "":
            mode = self.mode
        else:
            mode = str(mode)
        if raw_cap is not None and mode in MODE_RANK:
            mode = min(mode, _normalize_max_mode(raw_cap), key=lambda m: MODE_RANK.get(m, 0))
        now = time.time()
        (
            ok,
            intent_id,
            decision,
            symbol,
            venue,
            confidence,
            inst,
            effective_usd,
            price,
            is_limit,
        ) = preflight_intent_and_resolve_price(
            intent,
            min_confidence=self.min_confidence,
            min_notional_usd=self.min_notional_usd,
            max_notional_usd=self.max_notional_usd,
            resolve_instrument_fn=self.resolve_instrument,
            get_live_price_fn=self._get_live_price,
            instruments_map=instruments_map,
            now_ts=now,
            log=logger,
        )
        if not ok:
            return intent
        if resume_reconcile_cid and mode == "enforce" and reconcile_open_intent_by_client_id(inst, intent, now, log=logger):
            return intent

        replace_open = _parse_bool_flag(intent.get("replace_open_orders"), default=False)
        cancel_bot_open = _parse_bool_flag(intent.get("cancel_bot_open_orders"), default=False)
        target_oids = extract_intent_cancel_order_ids(intent)
        thesis_low = str(intent.get("thesis") or "").lower()
        thesis_cancels_bot_order = bool(
            replace_open
            and any(b in thesis_low for b in ("bot", "tradeall", "assetguardian", "by=bot"))
            and any(a in thesis_low for a in ("cancel", "override", "unwanted", "premature", "unlock"))
        )
        all_intents = self.load_all_intents()
        dup_oid: Optional[str] = None
        if mode == "enforce" and is_limit and not (cancel_bot_open or thesis_cancels_bot_order or target_oids):
            dup_oid = find_duplicate_ai_open_limit_order(
                inst,
                decision,
                price,
                effective_usd,
                intents=all_intents,
                require_notional_match=replace_open,
            )
        if dup_oid and not (replace_open or cancel_bot_open or bool(target_oids)):
            return self._record_duplicate_open_order_skip(
                inst, intent, decision=decision, price=price, effective_usd=effective_usd, dup_oid=dup_oid, now_ts=now
            )

        raw_thesis = intent.get("thesis")
        thesis = (
            "Autonomous strategy thesis"
            if raw_thesis is None or isinstance(raw_thesis, bool) or not str(raw_thesis).strip()
            else str(raw_thesis).strip()
        )
        trade_tag = resolve_intent_topic_tag(intent, venue)

        if not dup_oid and mode in ("enforce", "shadow"):
            short_circuit, effective_usd = self._evaluate_ai_circuit_breakers_and_survival(
                inst,
                intent,
                all_intents,
                mode=mode,
                decision=decision,
                symbol=symbol,
                venue=venue,
                price=price,
                effective_usd=effective_usd,
                confidence=confidence,
                thesis=thesis,
                trade_tag=trade_tag,
                now_ts=now,
                is_limit=is_limit,
            )
            if short_circuit is not None:
                return short_circuit

        intent_ref: List[Dict[str, Any]] = [intent]
        with OrderPreemptionTransaction(
            restore_fn=lambda orders, reason: self._rollback_canceled_orders(
                inst, intent_ref[0], orders, reason=reason
            ),
            default_reason=f"intent {intent_id} aborted",
        ) as preempt_tx:
            if mode == "enforce" and (replace_open or cancel_bot_open or bool(target_oids)):
                ok_cancel, _ = self._cancel_prior_orders_or_abort(
                    inst,
                    intent,
                    intent_id=intent_id,
                    decision=decision,
                    now_ts=now,
                    canceled_order_snapshots=preempt_tx.canceled_orders,
                    rollback_reason="unlock_cancel_failed" if thesis_cancels_bot_order else "cancel_prior_order_failed",
                    include_bot_orders=cancel_bot_open or bool(target_oids) or thesis_cancels_bot_order,
                    any_side=bool(target_oids),
                    exclude_order_ids={dup_oid} if dup_oid else None,
                    target_order_ids=target_oids or None,
                    verify_not_filled=replace_open or bool(target_oids),
                    extra_intent_key="canceled_bot_open_orders" if thesis_cancels_bot_order else None,
                    intents=all_intents,
                )
                if not ok_cancel:
                    # _cancel_prior_orders_or_abort already rolled back canceled_order_snapshots.
                    preempt_tx.commit()
                    return intent
            if dup_oid:
                preempt_tx.commit()
                return self._record_duplicate_open_order_skip(
                    inst, intent, decision=decision, price=price, effective_usd=effective_usd, dup_oid=dup_oid, now_ts=now
                )

            ok_wallet, guard_allowed, effective_usd, guard_reason = self._unlock_and_clamp_wallet_notional(
                inst,
                intent,
                all_intents,
                mode=mode,
                intent_id=intent_id,
                decision=decision,
                price=price,
                effective_usd=effective_usd,
                now_ts=now,
                replace_open=replace_open,
                cancel_bot_open=cancel_bot_open,
                target_oids=target_oids,
                thesis_cancels_bot_order=thesis_cancels_bot_order,
                canceled_order_snapshots=preempt_tx.canceled_orders,
            )
            if not ok_wallet:
                # _unlock_and_clamp_wallet_notional already rolled back on cancel failure.
                preempt_tx.commit()
                return intent
            if guard_allowed and effective_usd > 0:
                intent["clamped_notional_usd"] = round(effective_usd, 2)
            target_qty = effective_usd / price if effective_usd > 0 else 0.0

            if mode == "shadow":
                shadow_res = record_shadow_intent_observation(
                    inst,
                    intent,
                    decision=decision,
                    target_qty=target_qty,
                    price=price,
                    effective_usd=effective_usd,
                    confidence=confidence,
                    guard_allowed=guard_allowed,
                    guard_reason=guard_reason,
                    thesis=thesis,
                    trade_tag=trade_tag,
                    now_ts=now,
                    is_limit=is_limit,
                    dispatch_notify_fn=self._dispatch_notify,
                    log=logger,
                )
                if guard_allowed:
                    self._commit_cycle_quote_spend(decision, effective_usd)
                return shadow_res

            if mode != "enforce":
                logger.info("[LLM TRADER] Gateway mode is %r \u2014 skipping execution for intent %s", mode, intent_id)
                intent["status"] = "REJECTED_GATEWAY_DISABLED"
                append_intent_history(intent, {
                    "ts": now,
                    "action": "REJECTED_GATEWAY_DISABLED",
                    "mode": mode,
                })
                return intent

            if guard_allowed:
                self._record_advisory_guard_overrides(
                    intent, symbol=symbol, decision=decision, price=price, venue=venue, now_ts=now
                )

            res_intent = execute_enforce_intent_order(
                inst,
                intent,
                intent_id=intent_id,
                decision=decision,
                target_qty=target_qty,
                price=price,
                effective_usd=effective_usd,
                guard_allowed=guard_allowed,
                guard_reason=guard_reason,
                thesis=thesis,
                trade_tag=trade_tag,
                now_ts=now,
                bypass_guards=AUTONOMOUS_BYPASS_GUARDS,
                is_limit=is_limit,
                dispatch_notify_fn=self._dispatch_notify,
                log=logger,
            )
            intent_ref[0] = res_intent
            last_hist = (res_intent.get("history") or [{}])[-1] if isinstance(res_intent.get("history"), list) else {}
            if res_intent.get("execution_payload"):
                preempt_tx.commit()
                self._commit_cycle_quote_spend(decision, effective_usd)
            elif last_hist.get("category") == "deferred":
                preempt_tx.mark_deferred()
                self._commit_cycle_quote_spend(decision, effective_usd)
            elif last_hist.get("submission_state") == "unknown" or last_hist.get("reason") == "submit_ambiguous":
                preempt_tx.mark_ambiguous()
                self._commit_cycle_quote_spend(decision, effective_usd)
            elif (
                (not guard_allowed and res_intent.get("status") == "BLOCKED_BY_GUARD")
                or (guard_allowed and res_intent.get("status") == "FAILED")
            ):
                preempt_tx.set_reason(f"intent {intent_id} {res_intent.get('status')}")
            else:
                preempt_tx.set_reason(f"intent {intent_id} {res_intent.get('status') or 'uncommitted'}")
            return res_intent

    def submit(
        self,
        decision: Dict[str, Any],
        *,
        source: str,
        max_mode: str,
        instruments_map: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Single linear entry point for autonomous LLM trade decisions."""
        now = time.time()
        intents = self.load_all_intents()
        norm_max_mode = _normalize_max_mode(max_mode)

        def _resolve_submit_spec(sym: str, ven: str) -> Any:
            if "resolve_instrument" in self.__dict__:
                return self.resolve_instrument(sym, venue=ven, instruments_map=instruments_map)
            spec_map = (
                instruments_map
                if instruments_map is not None
                else discover_tradable_specs(self.workspace_dir, include_closed_rth=True, now_ts=now)
            )
            return resolve_instrument_spec(sym, venue=ven, instruments_map=spec_map, workspace_dir=self.workspace_dir)

        intent = validate_and_build_intent_dict(
            decision,
            source=source,
            min_confidence=self.min_confidence,
            min_notional_usd=self.min_notional_usd,
            max_notional_usd=self.max_notional_usd,
            existing_intents=intents,
            now_ts=now,
            resolve_instrument_fn=_resolve_submit_spec,
            max_mode=norm_max_mode,
            log=logger,
        )
        if intent is None:
            mode = min(self.mode, norm_max_mode, key=lambda m: MODE_RANK.get(m, 0))
            inst, target_oids, only_bots = resolve_hold_cancel_target(
                decision,
                min_confidence=self.min_confidence,
                resolve_instrument_fn=lambda sym, ven: self.resolve_instrument(
                    sym, venue=ven, instruments_map=instruments_map
                ),
            )
            if inst is None:
                return None
            raw_dec = clean_upper_scalar(decision.get("decision") or decision.get("action") or decision.get("side")) or "HOLD"
            cancel_rec: Dict[str, Any] = {
                "decision": "CANCEL" if raw_dec == "CANCEL" else "HOLD_CANCEL",
                "symbol": str(getattr(inst, "symbol", "") or decision.get("symbol") or ""),
                "venue": normalize_venue_name(getattr(inst, "provider_name", None) or decision.get("venue")) or "",
                "requested_order_ids": list(target_oids),
                "canceled_order_ids": [],
                "failed_order_ids": [],
                "timestamp": now,
                "created_by": clean_scalar_str(source) or "",
                "max_mode": norm_max_mode,
            }
            if mode == "enforce":
                failed_ids: List[str] = []
                scan_errors: List[str] = []
                cancel_snaps: List[Dict[str, Any]] = []
                _, canceled_ids = self._cancel_prior_orders_or_abort(
                    inst,
                    canceled_order_snapshots=cancel_snaps,
                    include_bot_orders=True,
                    only_bot_orders=only_bots,
                    any_side=True,
                    target_order_ids=target_oids,
                    verify_not_filled=True,
                    intents=intents,
                    failed_ids_out=failed_ids,
                    scan_errors_out=scan_errors,
                )
                cycle_caps = getattr(self, "_cycle_quote_caps", None)
                if isinstance(cycle_caps, dict) and cancel_snaps:
                    if (q_key := self._resolve_cycle_quote_key(inst)) is not None and q_key in cycle_caps:
                        freed_c = self._sum_canceled_buy_quote(cancel_snaps)
                        if freed_c > 0.0:
                            cycle_caps[q_key] += freed_c
                cancel_rec["canceled_order_ids"] = list(canceled_ids)
                cancel_rec["failed_order_ids"] = list(failed_ids)
                if canceled_ids:
                    self._mark_canceled_ai_intents(canceled_ids, now)
                if scan_errors:
                    cancel_rec["scan_errors"] = list(scan_errors)
                if canceled_ids and not failed_ids and not scan_errors and len(canceled_ids) >= len(target_oids):
                    cancel_rec["status"] = "CANCELED"
                elif canceled_ids:
                    cancel_rec["status"] = "CANCEL_PARTIAL"
                elif failed_ids or scan_errors:
                    cancel_rec["status"] = "CANCEL_FAILED"
                else:
                    cancel_rec["status"] = "CANCEL_NOOP"
            elif mode == "shadow":
                cancel_rec["status"] = "OBSERVED_SHADOW"
            else:
                cancel_rec["status"] = "REJECTED_GATEWAY_DISABLED"
            return cancel_rec
        intent["status"] = "PENDING_EXECUTION"
        if not upsert_durable_intent(self.intents_file, intent, reconcile_fn=self.reconcile_intents, log=logger):
            return None
        mode = min(self.mode, norm_max_mode, key=lambda m: MODE_RANK.get(m, 0))
        processed = self.process_intent(intent, instruments_map=instruments_map, mode=mode)
        upsert_durable_intent(self.intents_file, processed, reconcile_fn=self.reconcile_intents, log=logger)
        return processed

    def process_all_pending(self) -> List[Dict[str, Any]]:
        """Processes all pending trade intents from the durable queue."""
        intents = self.load_all_intents()
        pending_indices = [
            idx for idx, item in enumerate(intents)
            if str(item.get("status") or "").strip().upper() in ("PENDING", "PENDING_EXECUTION")
        ]
        if not pending_indices:
            logger.info("No intents found in queue." if not intents else "No pending trade intents to process.")
            return []

        logger.info("Processing %d pending autonomous trade intent(s) (mode=%s)...",
                    len(pending_indices), self.mode)

        instruments_map = self.load_tradable_instruments() or None

        processed: List[Dict[str, Any]] = []
        with self.cycle_shared_cash_scope():
            for idx in pending_indices:
                item = intents[idx]
                prev_st = str(item.get("status") or "").strip().upper()
                item["status"] = "PENDING_EXECUTION"
                if not upsert_durable_intent(self.intents_file, item, reconcile_fn=self.reconcile_intents, log=logger):
                    continue
                if prev_st == "PENDING_EXECUTION":
                    item["_resume_reconcile_cid"] = True
                raw_cap = item.get("max_mode")
                item_mode = (
                    min(self.mode, _normalize_max_mode(raw_cap), key=lambda m: MODE_RANK.get(m, 0))
                    if raw_cap is not None
                    else self.mode
                )
                updated = self.process_intent(item, instruments_map=instruments_map, mode=item_mode)
                intents[idx] = updated
                processed.append(updated)
                upsert_durable_intent(self.intents_file, updated, reconcile_fn=self.reconcile_intents, log=logger)

        return processed


def main() -> None:
    disable_auto_start_websockets_default()
    parser = argparse.ArgumentParser(description="MPTrade Autonomous Intent Gateway")
    parser.add_argument("--mode", choices=["shadow", "enforce"], default=None, help="Override operational mode")
    add_store_true_cli_args(
        parser,
        ("--process-pending", "Process all pending trade intents"),
        ("--status", "Display status of queued intents"),
        ("--clear-pending", "Cancel/clear all pending intents"),
    )
    args = parser.parse_args()

    gateway = AutonomousIntentGateway(mode=args.mode)

    if args.status:
        intents = gateway.load_all_intents()
        print(f"Total Intents in Queue: {len(intents)}")
        for i in intents[-10:]:
            notional_val = i.get("suggested_notional_usd")
            print(f"- [{i.get('status')}] {i.get('decision')} {i.get('symbol')} "
                  f"(venue={i.get('venue')}, conf={i.get('confidence')}, notional=${notional_val}): {i.get('intent_id')}")
        return

    if args.clear_pending:
        intents = gateway.load_all_intents()
        cleared_cnt = 0
        for item in intents:
            if str(item.get("status") or "").strip().upper() in ("PENDING", "PENDING_EXECUTION"):
                item["status"] = "CANCELLED"
                cleared_cnt += 1
        gateway.save_all_intents(intents)
        print(f"Cleared {cleared_cnt} pending intent(s).")
        return

    # Default action: process pending
    processed = gateway.process_all_pending()
    print(f"Processed {len(processed)} intent(s).")


if __name__ == "__main__":
    main()

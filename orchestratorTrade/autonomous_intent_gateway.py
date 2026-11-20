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
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("orchestratorTrade.autonomous_intent_gateway")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from botcore import (
    _parse_bool_flag,
    add_store_true_cli_args,
    clean_order_id,
    disable_auto_start_websockets_default,
    is_us_rth_open,
    load_key_value_conf,
)
from cachepaths import AUTONOMOUS_TRADE_INTENTS_STATE_FILENAME
from instrument import (
    load_tradable_instruments,
    resolve_instrument_live_price,
    resolve_tradable_instrument,
)
from instrument_registry import discover_tradable_specs, resolve_instrument_spec
import notify_engine.alertnotifiers as _alertnotifiers
from order_lifecycle import restore_cancelled_instrument_orders
from providers.intent_reconciliation import (
    reconcile_intents_from_execution_audit,
)
from providers.intent_execution import (
    cancel_existing_ai_open_orders,
    execute_enforce_intent_order,
    find_duplicate_ai_open_limit_order,
    preflight_intent_and_resolve_price,
    reconcile_open_intent_by_client_id,
    record_shadow_intent_observation,
    resolve_intent_topic_tag,
)
from providers.intent_state import (
    append_intent_history,
    extract_intent_cancel_order_ids,
    load_durable_intents_state,
    parse_intent_gateway_config,
    resolve_hold_cancel_target,
    save_durable_intents_state,
    upsert_durable_intent,
    validate_and_build_intent_dict,
)
from providers.quantity import clamp_notional_to_wallet_balance

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
        self.intents_file = os.path.join(self.cache_dir, AUTONOMOUS_TRADE_INTENTS_STATE_FILENAME)

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

    def load_all_intents(self) -> List[Dict[str, Any]]:
        """Loads all intents from the durable JSON queue with `.previous` crash recovery."""
        return load_durable_intents_state(self.intents_file, reconcile_fn=self.reconcile_intents)

    def reconcile_intents(self, intents: List[Dict[str, Any]]) -> bool:
        """Reconciles FAILED/PENDING/PENDING_EXECUTION/SUBMIT_UNKNOWN and incomplete EXECUTED intents against execution_audit_*.jsonl."""
        audit_dir = os.path.join(self.workspace_dir, "logger", "execution_audit")
        updated = reconcile_intents_from_execution_audit(intents, audit_dir)
        if updated:
            self.save_all_intents(intents)
        return updated

    def save_all_intents(self, intents: List[Dict[str, Any]]) -> bool:
        """Transactionally persists all intents to disk using `.previous` staging and atomic replace."""
        return save_durable_intents_state(self.intents_file, intents, log=logger)

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

    def _clamp_notional_to_wallet_balance(
        self,
        inst: Any,
        decision: str,
        price: float,
        desired_usd: float,
    ) -> Tuple[bool, float, str]:
        """Clamp desired USD notional to real spot wallet liquidity via unified SSOT in providers.quantity."""
        return clamp_notional_to_wallet_balance(
            inst,
            decision,
            price,
            desired_usd,
            min_notional_usd=self.min_notional_usd,
        )

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
    ) -> Tuple[bool, List[str]]:
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
        dup_oid: Optional[str] = None
        if mode == "enforce" and is_limit and not (cancel_bot_open or thesis_cancels_bot_order or target_oids):
            dup_oid = find_duplicate_ai_open_limit_order(
                inst,
                decision,
                price,
                effective_usd,
                intents=self.load_all_intents(),
                require_notional_match=replace_open,
            )
        canceled_order_snapshots: list[dict[str, Any]] = []
        if mode == "enforce" and (replace_open or cancel_bot_open or bool(target_oids)):
            ok_cancel, _ = self._cancel_prior_orders_or_abort(
                inst,
                intent,
                intent_id=intent_id,
                decision=decision,
                now_ts=now,
                canceled_order_snapshots=canceled_order_snapshots,
                rollback_reason="unlock_cancel_failed" if thesis_cancels_bot_order else "cancel_prior_order_failed",
                include_bot_orders=cancel_bot_open or bool(target_oids) or thesis_cancels_bot_order,
                any_side=bool(target_oids),
                exclude_order_ids={dup_oid} if dup_oid else None,
                target_order_ids=target_oids or None,
                verify_not_filled=replace_open or bool(target_oids),
                extra_intent_key="canceled_bot_open_orders" if thesis_cancels_bot_order else None,
            )
            if not ok_cancel:
                return intent
        if dup_oid:
            logger.info(
                "[LLM TRADER \u00b7 ENFORCE] Skipping duplicate %s LIMIT %s @ $%.4f ($%.2f); identical AI order %s is already open.",
                decision, inst.symbol, price, effective_usd, dup_oid,
            )
            intent["status"] = "SKIPPED_DUPLICATE_OPEN_ORDER"
            intent["duplicate_of_order_id"] = dup_oid
            append_intent_history(intent, {
                "ts": now,
                "action": "SKIPPED_DUPLICATE_OPEN_ORDER",
                "duplicate_of_order_id": dup_oid,
                "price": price,
                "notional_usd": effective_usd,
            })
            return intent
        requested_effective_usd = effective_usd
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
                now_ts=now,
                canceled_order_snapshots=canceled_order_snapshots,
                rollback_reason="unlock_cancel_failed",
                include_bot_orders=True,
                only_bot_orders=True,
                extra_intent_key="canceled_bot_open_orders",
            )
            if not ok_unlock:
                return intent
            if unlocked_ids:
                guard_allowed, effective_usd, guard_reason = self._clamp_notional_to_wallet_balance(
                    inst, decision, price, requested_effective_usd
                )
        if guard_allowed and effective_usd > 0:
            intent["clamped_notional_usd"] = round(effective_usd, 2)
        target_qty = effective_usd / price if effective_usd > 0 else 0.0

        raw_thesis = intent.get("thesis")
        thesis = (
            "Autonomous strategy thesis"
            if raw_thesis is None or isinstance(raw_thesis, bool) or not str(raw_thesis).strip()
            else str(raw_thesis).strip()
        )
        trade_tag = resolve_intent_topic_tag(intent, venue)

        # Execution or Observation Branch
        if mode == "shadow":
            return record_shadow_intent_observation(
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

        # Gate: only "enforce" mode is permitted to execute live orders
        if mode != "enforce":
            logger.info("[LLM TRADER] Gateway mode is %r \u2014 skipping execution for intent %s", mode, intent_id)
            intent["status"] = "REJECTED_GATEWAY_DISABLED"
            append_intent_history(intent, {
                "ts": now,
                "action": "REJECTED_GATEWAY_DISABLED",
                "mode": mode,
            })
            return intent

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
        last_hist = (res_intent.get("history") or [{}])[-1] if isinstance(res_intent.get("history"), list) else {}
        if (
            canceled_order_snapshots
            and guard_allowed
            and res_intent.get("status") == "FAILED"
            and not res_intent.get("execution_payload")
            and last_hist.get("category") != "deferred"
            and last_hist.get("submission_state") != "unknown"
            and last_hist.get("reason") != "submit_ambiguous"
        ):
            self._rollback_canceled_orders(
                inst,
                res_intent,
                canceled_order_snapshots,
                reason=f"intent {intent_id} {res_intent.get('status')}",
            )
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
            if mode == "enforce":
                inst, target_oids, only_bots = resolve_hold_cancel_target(
                    decision,
                    min_confidence=self.min_confidence,
                    resolve_instrument_fn=lambda sym, ven: self.resolve_instrument(
                        sym, venue=ven, instruments_map=instruments_map
                    ),
                )
                if inst is not None:
                    self._cancel_prior_orders_or_abort(
                        inst,
                        include_bot_orders=True,
                        only_bot_orders=only_bots,
                        any_side=True,
                        target_order_ids=target_oids,
                        verify_not_filled=True,
                        intents=intents,
                    )
            return None
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

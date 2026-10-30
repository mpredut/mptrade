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
from dataclasses import dataclass, field
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
    extract_current_price_item,
    is_us_rth_open as _is_us_rth_open,
    load_key_value_conf,
    parse_notify_enabled_flag as _parse_notify_enabled_flag,
    safe_float,
)
from instruments_config import (
    discover_kraken_cache_pairs as _discover_kraken_cache_pairs,
    discover_t212_equities as _discover_t212_equities,
    load_tradable_instruments as _load_tradable_instruments,
    resolve_instrument_live_price,
    resolve_tradable_instrument,
)
from providers.execution_audit import (
    append_intent_history as _append_intent_history,
    cancel_existing_ai_open_orders,
    execute_enforce_intent_order,
    extract_intent_notional as _extract_intent_notional,
    load_durable_intents_state,
    parse_intent_confidence as _parse_intent_confidence,
    parse_intent_gateway_config,
    preflight_intent_and_resolve_price,
    reconcile_intents_from_execution_audit,
    record_shadow_intent_observation,
    resolve_intent_topic_tag,
    save_durable_intents_state,
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


@dataclass
class TradeIntentRecord:
    """Canonical structure for an autonomous trade intent."""
    intent_id: str
    timestamp: float
    decision: str                          # "BUY" | "SELL"
    symbol: str                            # e.g. "BTCUSDC", "TAOUSDC", "HYPEUSD"
    venue: str                             # "binance" | "kraken" | "hyperliquid" | "t212"
    confidence: float
    suggested_notional_usd: float
    order_type: str = "MARKET"             # "MARKET" | "LIMIT"
    limit_price: Optional[float] = None
    replace_open_orders: bool = False
    urgency: str = "NORMAL"
    thesis: str = ""
    status: str = "PENDING"                # "PENDING" | "OBSERVED_SHADOW" | "EXECUTED" | "BLOCKED_BY_GUARD" | "REJECTED_LOW_CONFIDENCE" | "REJECTED_UNKNOWN_INSTRUMENT" | "FAILED"
    execution_payload: Optional[Dict[str, Any]] = None
    created_by: str = "autonomous_ai_reconciler"
    history: List[Dict[str, Any]] = field(default_factory=list)


MODE_RANK = {"off": 0, "shadow": 1, "enforce": 2}
_parse_finite_float = safe_float


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
        self.intents_file = os.path.join(self.cache_dir, "autonomous_trade_intents_state.json")

        conf = self._load_conf()
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

    def _load_conf(self) -> Dict[str, str]:
        """Loads configuration from order_guard.conf / config.env, overlaying workspace-specific overrides."""
        return load_key_value_conf(self.workspace_dir, ROOT_DIR)

    def load_all_intents(self) -> List[Dict[str, Any]]:
        """Loads all intents from the durable JSON queue with `.previous` crash recovery."""
        return load_durable_intents_state(self.intents_file, reconcile_fn=self.reconcile_intents)

    def reconcile_intents(self, intents: List[Dict[str, Any]]) -> bool:
        """Reconciles FAILED or PENDING_EXECUTION intents against execution_audit_*.jsonl if later filled via retry worker."""
        audit_dir = os.path.join(self.workspace_dir, "logger", "execution_audit")
        updated = reconcile_intents_from_execution_audit(intents, audit_dir)
        if updated:
            self.save_all_intents(intents)
        return updated

    def save_all_intents(self, intents: List[Dict[str, Any]]) -> bool:
        """Transactionally persists all intents to disk using `.previous` staging and atomic replace."""
        return save_durable_intents_state(self.intents_file, intents, log=logger)

    def get_pending_intents(self) -> List[Dict[str, Any]]:
        """Returns all intents with PENDING status."""
        return [
            i for i in self.load_all_intents()
            if str(i.get("status") or "").strip().upper() == "PENDING"
        ]

    def load_tradable_instruments(
        self,
        *,
        only_workspace_conf: bool = False,
        include_closed_rth: bool = False,
        now_ts: Optional[float] = None,
        asset_scope: str = "all",
    ) -> Dict[str, Any]:
        """Load enabled instruments filtered by asset_scope ('crypto' | 'equities' | 'all')."""
        return _load_tradable_instruments(
            self.workspace_dir,
            only_workspace_conf=only_workspace_conf,
            include_closed_rth=include_closed_rth,
            now_ts=now_ts,
            asset_scope=asset_scope,
            is_rth_open_fn=_is_us_rth_open,
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
            from notify_engine.alertnotifiers import notify
            notify(
                title=title,
                body=body,
                source="llm_trader",
                symbol=symbol,
            )
            return True
        except Exception as e:
            logger.debug("Failed notifying via notify_engine: %s", e)
            return False

    def process_intent(
        self,
        intent: Dict[str, Any],
        instruments_map: Optional[Dict[str, Any]] = None,
        mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Validates and processes a single intent according to ``mode`` (default: the gateway mode)."""
        if not isinstance(intent, dict):
            return {}
        if isinstance(mode, str) and mode.strip():
            mode = mode.strip().lower()
        elif mode is None or mode == "":
            mode = self.mode
        else:
            mode = str(mode)
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

        replace_open = _parse_bool_flag(intent.get("replace_open_orders"), default=False)
        cancel_bot_open = _parse_bool_flag(intent.get("cancel_bot_open_orders"), default=False)
        if mode == "enforce" and (replace_open or cancel_bot_open):
            canceled_ids = self._cancel_existing_ai_open_orders(
                inst,
                decision,
                include_bot_orders=cancel_bot_open,
                any_side=cancel_bot_open,
            )
            if canceled_ids:
                intent["canceled_prior_open_orders"] = canceled_ids

        # Minimal Operational Preflight (Symmetric BUY / SELL, Zero Strategy or Daily-Cap Guards):
        # Clamp desired notional strictly to real spot wallet free liquidity.
        requested_effective_usd = effective_usd
        guard_allowed, effective_usd, guard_reason = self._clamp_notional_to_wallet_balance(
            inst, decision, price, requested_effective_usd
        )
        if mode == "enforce" and replace_open and not cancel_bot_open and not guard_allowed:
            unlocked_ids = self._cancel_existing_ai_open_orders(
                inst,
                decision,
                include_bot_orders=True,
                any_side=True,
            )
            if unlocked_ids:
                prior_c = list(intent.get("canceled_prior_open_orders") or [])
                for uid in unlocked_ids:
                    if uid not in prior_c:
                        prior_c.append(uid)
                intent["canceled_prior_open_orders"] = prior_c
                intent["canceled_bot_open_orders"] = unlocked_ids
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
            logger.info("[LLM TRADER] Gateway mode is %r — skipping execution for intent %s", mode, intent_id)
            intent["status"] = "REJECTED_GATEWAY_DISABLED"
            _append_intent_history(intent, {
                "ts": now,
                "action": "REJECTED_GATEWAY_DISABLED",
                "mode": mode,
            })
            return intent

        return execute_enforce_intent_order(
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

    def _cancel_existing_ai_open_orders(
        self,
        inst: Any,
        side: str,
        *,
        include_bot_orders: bool = False,
        any_side: bool = False,
    ) -> List[str]:
        """Cancel any prior unfilled AI-placed (or bot-placed when requested) open orders on the symbol."""
        return cancel_existing_ai_open_orders(
            inst,
            side,
            log=logger,
            intents=self.load_all_intents(),
            include_bot_orders=include_bot_orders,
            any_side=any_side,
        )

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
        intent = validate_and_build_intent_dict(
            decision,
            source=source,
            min_confidence=self.min_confidence,
            min_notional_usd=self.min_notional_usd,
            max_notional_usd=self.max_notional_usd,
            existing_intents=intents,
            now_ts=now,
            log=logger,
        )
        if intent is None:
            return None
        intents.append(intent)
        self.save_all_intents(intents)
        norm_max_mode = (
            "off"
            if isinstance(max_mode, bool)
            else str(max_mode or "shadow").strip().lower()
        )
        if norm_max_mode not in MODE_RANK:
            norm_max_mode = "off"
        mode = min(self.mode, norm_max_mode, key=lambda m: MODE_RANK.get(m, 0))
        processed = self.process_intent(intent, instruments_map=instruments_map, mode=mode)
        intents[-1] = processed
        self.save_all_intents(intents)
        return processed

    def process_all_pending(self) -> List[Dict[str, Any]]:
        """Processes all pending trade intents from the durable queue."""
        intents = self.load_all_intents()
        if not intents:
            logger.info("No intents found in queue.")
            return []

        pending_indices = [
            idx for idx, item in enumerate(intents)
            if str(item.get("status") or "").strip().upper() == "PENDING"
        ]
        if not pending_indices:
            logger.info("No pending trade intents to process.")
            return []

        logger.info("Processing %d pending autonomous trade intent(s) (mode=%s)...",
                    len(pending_indices), self.mode)

        instruments_map = self.load_tradable_instruments() or None

        processed: List[Dict[str, Any]] = []
        for idx in pending_indices:
            updated = self.process_intent(intents[idx], instruments_map=instruments_map)
            intents[idx] = updated
            processed.append(updated)
            self.save_all_intents(intents)

        return processed


def main() -> None:
    os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")
    parser = argparse.ArgumentParser(description="MPTrade Autonomous Intent Gateway")
    parser.add_argument("--process-pending", action="store_true", help="Process all pending trade intents")
    parser.add_argument("--mode", choices=["shadow", "enforce"], default=None, help="Override operational mode")
    parser.add_argument("--status", action="store_true", help="Display status of queued intents")
    parser.add_argument("--clear-pending", action="store_true", help="Cancel/clear all pending intents")
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
            if str(item.get("status") or "").strip().upper() == "PENDING":
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

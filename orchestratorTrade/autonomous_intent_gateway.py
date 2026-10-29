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
from dataclasses import asdict, dataclass, field
import glob
import json
import logging
import math
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from botcore import (
    _parse_bool_flag,
    parse_dotenv,
    safe_float,
    safe_nonneg_float,
    safe_pos_float,
)
from providers.base import normalize_side

logger = logging.getLogger("orchestratorTrade.autonomous_intent_gateway")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from cachepaths import (
    cache_symbol_variants,
    canonical_base_asset,
    normalize_venue_name,
    resolve_cachedb_filepath,
    split_symbol_assets,
)


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
    urgency: str = "NORMAL"
    thesis: str = ""
    status: str = "PENDING"                # "PENDING" | "OBSERVED_SHADOW" | "EXECUTED" | "BLOCKED_BY_GUARD" | "REJECTED_LOW_CONFIDENCE" | "REJECTED_UNKNOWN_INSTRUMENT" | "FAILED"
    execution_payload: Optional[Dict[str, Any]] = None
    created_by: str = "autonomous_ai_reconciler"
    history: List[Dict[str, Any]] = field(default_factory=list)


MODE_RANK = {"off": 0, "shadow": 1, "enforce": 2}


_parse_finite_float = safe_float

from botcore import (
    conf_or_override as _conf_or_override,
    extract_current_price_item,
    is_us_rth_open as _is_us_rth_open,
    load_env_file_defaults as _load_env_file_defaults,
    load_key_value_conf,
    parse_env_file as _parse_env_file,
    parse_notify_enabled_flag as _parse_notify_enabled_flag,
)
from instruments_config import (
    discover_hyperliquid_spot as _discover_hyperliquid_spot,
    discover_kraken_cache_pairs as _discover_kraken_cache_pairs,
    discover_t212_equities as _discover_t212_equities,
    load_tradable_instruments as _load_tradable_instruments,
    resolve_instrument_live_price,
    resolve_tradable_instrument,
)
from intelligence.macro_sentiment_ssot import clean_scale, clean_symbol_key
from providers.execution_audit import (
    append_intent_history as _append_intent_history,
    extract_intent_notional as _extract_intent_notional,
    extract_intent_side as _extract_intent_side,
    parse_intent_confidence as _parse_intent_confidence,
    reconcile_intents_from_execution_audit,
)

_norm_sym = clean_symbol_key
_split_symbol_assets = split_symbol_assets


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

        raw_mode = _conf_or_override(conf, "autonomous_trading_mode", mode)
        self.mode = "" if isinstance(raw_mode, bool) else str(raw_mode).strip().lower()
        if self.mode not in ("off", "shadow", "enforce"):
            raise ValueError(
                f"Invalid autonomous_trading_mode {raw_mode!r}; must be one of ['enforce', 'off', 'shadow']"
            )

        raw_min = _conf_or_override(conf, "autonomous_trading_min_notional_usd")
        min_usd = safe_pos_float(raw_min, default=None)
        if min_usd is None:
            raise ValueError(f"Invalid autonomous_trading_min_notional_usd {raw_min!r}")
        self.min_notional_usd = min_usd

        raw_max = _conf_or_override(conf, "autonomous_trading_max_notional_usd", max_notional_usd)
        max_usd = safe_pos_float(raw_max, default=None)
        if max_usd is None or max_usd < self.min_notional_usd:
            raise ValueError(
                f"Invalid autonomous_trading_max_notional_usd {raw_max!r}; "
                f"must be >= autonomous_trading_min_notional_usd ({self.min_notional_usd})"
            )
        self.max_notional_usd = max_usd

        raw_conf = _conf_or_override(conf, "autonomous_trading_min_confidence", min_confidence)
        conf_val = safe_nonneg_float(raw_conf, default=None)
        if conf_val is None or conf_val > 1.0:
            raise ValueError(
                f"Invalid autonomous_trading_min_confidence {raw_conf!r}; must be in [0.0, 1.0]"
            )
        self.min_confidence = conf_val

        self.notify_enabled = _parse_notify_enabled_flag(
            notify_enabled, conf, "autonomous_trading_notify"
        )

    def _load_conf(self) -> Dict[str, str]:
        """Loads configuration from order_guard.conf / config.env, overlaying workspace-specific overrides."""
        return load_key_value_conf(self.workspace_dir, ROOT_DIR)

    def load_all_intents(self) -> List[Dict[str, Any]]:
        """Loads all intents from the durable JSON queue with `.previous` crash recovery."""
        from state_io import load_json_state
        data = load_json_state(
            self.intents_file,
            default_factory=list,
            fail_closed=False,
            label="autonomous_trade_intents",
            root_type=list,
        )
        items = [item for item in data if isinstance(item, dict)]
        self.reconcile_intents(items)
        return items

    def reconcile_intents(self, intents: List[Dict[str, Any]]) -> bool:
        """Reconciles FAILED or PENDING_EXECUTION intents against execution_audit_*.jsonl if later filled via retry worker."""
        audit_dir = os.path.join(self.workspace_dir, "logger", "execution_audit")
        updated = reconcile_intents_from_execution_audit(intents, audit_dir)
        if updated:
            self.save_all_intents(intents)
        return updated

    def save_all_intents(self, intents: List[Dict[str, Any]]) -> bool:
        """Transactionally persists all intents to disk using `.previous` staging and atomic replace."""
        from orchestratorOS.lib.technical_repair_ssot import clean_json_value
        from state_io import save_json_state
        try:
            cleaned = clean_json_value(intents if isinstance(intents, list) else [])
            save_json_state(self.intents_file, cleaned, indent=2)
            return True
        except Exception as e:
            logger.error("Failed saving intents to %s: %s", self.intents_file, e)
            return False

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
        from providers.quantity import clamp_notional_to_wallet_balance
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
        raw_iid = intent.get("intent_id")
        if raw_iid is None or isinstance(raw_iid, bool) or not str(raw_iid).strip():
            raw_iid = intent.get("id")
        if raw_iid is None or isinstance(raw_iid, bool) or not str(raw_iid).strip():
            raw_iid = f"intent_{int(time.time())}"
        intent_id = str(raw_iid).strip()
        decision = _extract_intent_side(intent)
        raw_sym = intent.get("symbol", "")
        symbol = "" if isinstance(raw_sym, bool) or raw_sym is None else str(raw_sym).strip().upper()
        raw_venue = intent.get("venue", "binance")
        if isinstance(raw_venue, bool):
            venue = "__invalid__"
        elif raw_venue is None or not str(raw_venue).strip():
            venue = "binance"
        else:
            venue = normalize_venue_name(str(raw_venue)) or str(raw_venue).strip().lower()
        now = time.time()

        if decision not in ("BUY", "SELL"):
            intent["status"] = "REJECTED_INVALID_DECISION"
            _append_intent_history(intent, {
                "ts": now, "action": "REJECTED_INVALID_DECISION", "decision": decision
            })
            return intent

        confidence = _parse_intent_confidence(intent.get("confidence"), 0.0)
        intent["confidence"] = confidence

        if confidence < self.min_confidence:
            logger.info("[INTENT GATEWAY] Intent %s rejected: confidence %.2f < %.2f",
                        intent_id, confidence, self.min_confidence)
            intent["status"] = "REJECTED_LOW_CONFIDENCE"
            _append_intent_history(intent, {
                "ts": now, "action": "REJECTED_LOW_CONFIDENCE",
                "confidence": confidence, "min_confidence": self.min_confidence
            })
            return intent

        inst = self.resolve_instrument(symbol, venue=venue, instruments_map=instruments_map)
        if not inst or (
            normalize_venue_name(str(getattr(inst, "provider_name", ""))) != "t212"
            and str(getattr(inst, "symbol", "")).upper().endswith("BTC")
        ):
            logger.warning("[INTENT GATEWAY] Intent %s rejected: unknown or non-fiat instrument %s on %s",
                           intent_id, symbol, venue)
            intent["status"] = "REJECTED_UNKNOWN_INSTRUMENT"
            _append_intent_history(intent, {
                "ts": now, "action": "REJECTED_UNKNOWN_INSTRUMENT", "symbol": symbol, "venue": venue
            })
            return intent

        suggested_usd = _extract_intent_notional(intent, self.max_notional_usd)
        effective_usd = max(self.min_notional_usd, min(suggested_usd, self.max_notional_usd))

        price = safe_pos_float(self._get_live_price(inst), default=None)
        if price is None:
            logger.warning("[INTENT GATEWAY] Intent %s rejected: invalid price for %s", intent_id, symbol)
            intent["status"] = "REJECTED_PRICE_UNAVAILABLE"
            _append_intent_history(intent, {
                "ts": now, "action": "REJECTED_PRICE_UNAVAILABLE", "symbol": symbol
            })
            return intent

        # Minimal Operational Preflight (Symmetric BUY / SELL, Zero Strategy or Daily-Cap Guards):
        # Clamp desired notional strictly to real spot wallet free liquidity.
        guard_allowed, effective_usd, guard_reason = self._clamp_notional_to_wallet_balance(
            inst, decision, price, effective_usd
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
        from orchestratorOS.agy.topics import TOPIC_ALIASES, TOPIC_SPECS, topic_spec
        raw_src = intent.get("created_by")
        src_topic = "" if raw_src is None or isinstance(raw_src, bool) else str(raw_src).strip()
        if src_topic in TOPIC_SPECS or src_topic in TOPIC_ALIASES:
            trade_tag = topic_spec(src_topic).bracket_tag
        elif venue == "t212":
            trade_tag = topic_spec("auto_sharetrade_executor").bracket_tag
        else:
            trade_tag = topic_spec("auto_cryptotrade_executor").bracket_tag

        # Execution or Observation Branch
        if mode == "shadow":
            logger.info("[LLM TRADER · SHADOW] Would execute %s %s (qty=%g, price=$%.2f, notional=$%.2f) on %s. Thesis: %s",
                        decision, inst.symbol, target_qty, price, effective_usd, inst.provider_name, thesis)

            title = f"🟢 [{trade_tag.strip('[]')} · SHADOW] {decision} {inst.symbol} ${effective_usd:.0f}"
            body = (
                f"Venue: {inst.provider_name} · Qty: {target_qty:g} @ ${price:,.2f}\n"
                f"Confidence: {confidence:.2f} · Notional: ${effective_usd:.2f}\n"
                f"Operational Preflight: {'PASSED' if guard_allowed else guard_reason}\n"
                f"Thesis: {thesis}"
            )
            self._dispatch_notify(title=title, body=body, symbol=inst.symbol)

            intent["status"] = "OBSERVED_SHADOW"
            _append_intent_history(intent, {
                "ts": now,
                "action": "OBSERVED_SHADOW",
                "guard_allowed": guard_allowed,
                "guard_reason": guard_reason,
                "price": price,
                "qty": target_qty,
                "notional_usd": effective_usd,
            })
            return intent

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

        # Mode == "enforce"
        if not guard_allowed:
            logger.warning("[LLM TRADER · ENFORCE] Operational guard blocked %s %s: %s",
                           decision, inst.symbol, guard_reason)
            title = f"🛡 {trade_tag} BLOCKED {decision} {inst.symbol}"
            body = (
                f"Venue: {inst.provider_name} · Price: ${price:,.2f} · Notional: ${effective_usd:.0f}\n"
                f"Reason: {guard_reason}\n"
                f"Thesis: {thesis}"
            )
            self._dispatch_notify(title=title, body=body, symbol=inst.symbol)

            intent["status"] = "BLOCKED_BY_GUARD"
            _append_intent_history(intent, {
                "ts": now,
                "action": "BLOCKED_BY_GUARD",
                "guard_reason": guard_reason,
                "price": price,
                "qty": target_qty,
                "notional_usd": effective_usd,
            })
            return intent

        logger.warning("[LLM TRADER · ENFORCE] Executing %s %s (qty=%g, notional=$%.2f) on %s...",
                       decision, inst.symbol, target_qty, effective_usd, inst.provider_name)

        try:
            intent_cid = f"AI_{intent_id[:28]}" if intent_id else None
            outcome_ctx: Dict[str, Any] = {}
            order_res = None
            for attempt in range(2):
                outcome_ctx.clear()
                order_res = inst.place(
                    side=decision,
                    qty=target_qty,
                    price=price,
                    market=True,
                    enforce_business_minimum=False,
                    wait_for_trend=False,
                    intent_id=intent_id,
                    client_order_id=intent_cid,
                    regime_context=None,
                    bypass_guards=AUTONOMOUS_BYPASS_GUARDS,
                    _outcome_context=outcome_ctx,
                )
                if (
                    order_res
                    or attempt > 0
                    or (
                        outcome_ctx.get("state") != "refused"
                        and outcome_ctx.get("category") != "refused_pre_submit"
                    )
                    or outcome_ctx.get("reason") not in ("account_cache_not_fresh", "account_cache_snapshot_changed")
                ):
                    break
                time.sleep(0.25)
            if order_res:
                from providers.base import extract_order_id
                logger.info("[LLM TRADER · ENFORCE] Successfully placed %s %s: %s",
                            decision, inst.symbol, order_res)
                title = f"🚀 {trade_tag} EXECUTED {decision} {inst.symbol} ${effective_usd:.0f}"
                order_id_str = extract_order_id(order_res, default="N/A")
                body = (
                    f"Venue: {inst.provider_name} · Qty: {target_qty:g} @ ${price:,.2f} (${effective_usd:,.0f})\n"
                    f"Order ID: {order_id_str}\n"
                    f"Thesis: {thesis}"
                )
                self._dispatch_notify(title=title, body=body, symbol=inst.symbol)

                intent["status"] = "EXECUTED"
                intent["execution_payload"] = order_res
                _append_intent_history(intent, {
                    "ts": now,
                    "action": "EXECUTED",
                    "notional_usd": effective_usd,
                    "payload": order_res,
                })
            else:
                refusal_reason = str(outcome_ctx.get("reason") or "Instrument.place returned None")
                refusal_cat = str(outcome_ctx.get("category") or "")
                logger.warning("[LLM TRADER · ENFORCE] Order placement refused or failed for %s %s: %s",
                               decision, inst.symbol, refusal_reason)
                title = f"⚠️ {trade_tag} REFUSED {decision} {inst.symbol}"
                body = f"Venue {inst.provider_name} refused placement ({refusal_reason}).\nThesis: {thesis}"
                self._dispatch_notify(title=title, body=body, symbol=inst.symbol)

                intent["status"] = "FAILED"
                hist_entry: Dict[str, Any] = {
                    "ts": now,
                    "action": "FAILED",
                    "reason": refusal_reason,
                }
                if refusal_cat:
                    hist_entry["category"] = refusal_cat
                _append_intent_history(intent, hist_entry)
        except Exception as e:
            logger.error("[LLM TRADER · ENFORCE] Exception executing %s %s: %s", decision, inst.symbol, e)
            title = f"🚨 {trade_tag} ERROR {decision} {inst.symbol}"
            body = f"Execution exception: {e}\nThesis: {thesis}"
            self._dispatch_notify(title=title, body=body, symbol=inst.symbol)

            intent["status"] = "FAILED"
            _append_intent_history(intent, {
                "ts": now,
                "action": "ERROR",
                "error": str(e),
            })

        return intent

    def submit(
        self,
        decision: Dict[str, Any],
        *,
        source: str,
        max_mode: str,
        instruments_map: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Single linear entry point for autonomous LLM trade decisions.

        Validates confidence and symbol/venue, clamps the USD notional, persists the intent, and
        processes it immediately in the more restrictive of the gateway mode and ``max_mode``
        (the producer's own mode), so a shadow producer can never trigger a live order.
        """
        if not isinstance(decision, dict):
            return None
        side = _extract_intent_side(decision)
        if side not in ("BUY", "SELL"):
            raw_dec = str(decision.get("decision") or decision.get("action") or decision.get("side") or "HOLD").strip().upper() or "HOLD"
            logger.info("[INTENT GATEWAY] %s decision is %s (no intent).", source, raw_dec)
            return None
        raw_sym = decision.get("symbol", "")
        symbol = "" if isinstance(raw_sym, bool) or raw_sym is None else str(raw_sym).strip().upper()
        raw_venue = decision.get("venue", "")
        venue = "" if isinstance(raw_venue, bool) or raw_venue is None else normalize_venue_name(str(raw_venue))
        if not symbol or not venue:
            logger.warning("[INTENT GATEWAY] %s %s decision missing symbol/venue (no intent).", source, side)
            return None
        confidence = _parse_intent_confidence(decision.get("confidence"), 0.0)
        if confidence < self.min_confidence:
            logger.info("[INTENT GATEWAY] %s %s %s confidence %.2f < %.2f (no intent).",
                        source, side, symbol, confidence, self.min_confidence)
            return None
        notional = _extract_intent_notional(decision, self.max_notional_usd)
        notional = max(self.min_notional_usd, min(notional, self.max_notional_usd))

        now = time.time()
        intents = self.load_all_intents()
        base_id = f"intent_{int(now)}_{side.lower()}_{symbol.lower()}"
        existing_ids = {str(i.get("intent_id") or i.get("id")) for i in intents if isinstance(i, dict)}
        intent_id = base_id
        suffix = len(intents) + 1
        while intent_id in existing_ids:
            intent_id = f"{base_id}_{suffix}"
            suffix += 1
        raw_urg = decision.get("urgency", "NORMAL")
        urgency = (
            "NORMAL"
            if isinstance(raw_urg, bool) or raw_urg is None or not str(raw_urg).strip()
            else str(raw_urg).strip().upper()
        )
        raw_thesis = decision.get("thesis", "")
        thesis = "" if isinstance(raw_thesis, bool) or raw_thesis is None else str(raw_thesis).strip()
        intent = asdict(TradeIntentRecord(
            intent_id=intent_id,
            timestamp=now,
            decision=side,
            symbol=symbol,
            venue=venue,
            confidence=confidence,
            suggested_notional_usd=notional,
            urgency=urgency,
            thesis=thesis,
            created_by=str(source or "").strip(),
        ))
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

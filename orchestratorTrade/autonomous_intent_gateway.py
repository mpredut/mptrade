"""Autonomous AI Trading Intent Gateway.

Linear choke point and bridge between decoupled AI intent producers and the
unified MPTrade execution pipeline (Instrument.place & order_guard).

Operational Architecture:
1. Decoupled Ingestion:
   Reads high-conviction trade intents emitted into `cachedb/autonomous_trade_intents.json`.
2. Guard Validation & Safety Caps:
   - Confidence threshold verification (default >= 0.85).
   - Maximum notional EUR position cap (default <= 500 EUR).
   - Deterministic profit guard vetting (order_guard.profit_guard).
3. Modes of Operation:
   - "shadow" (default): Simulates order, validates guards, logs simulated action, dispatches phone alerts via ntfy.
   - "enforce": Preflights guards, sizes lot units, and submits order via canonical `Instrument.place()`.
4. Durable Intent Tracking:
   Updates intent lifecycle state: PENDING -> OBSERVED_SHADOW | EXECUTED | BLOCKED_BY_GUARD | REJECTED.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import logging
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("orchestratorTrade.autonomous_intent_gateway")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


@dataclass
class TradeIntentRecord:
    """Canonical structure for an autonomous trade intent."""
    intent_id: str
    timestamp: float
    decision: str                          # "BUY" | "SELL"
    symbol: str                            # e.g. "BTCUSDC"
    venue: str                             # "binance" | "kraken" | "hyperliquid" | "t212"
    confidence: float
    suggested_notional_eur: float
    urgency: str = "NORMAL"
    thesis: str = ""
    status: str = "PENDING"                # "PENDING" | "OBSERVED_SHADOW" | "EXECUTED" | "BLOCKED_BY_GUARD" | "REJECTED_LOW_CONFIDENCE" | "REJECTED_UNKNOWN_INSTRUMENT" | "FAILED"
    execution_payload: Optional[Dict[str, Any]] = None
    created_by: str = "autonomous_ai_reconciler"
    history: List[Dict[str, Any]] = field(default_factory=list)


class AutonomousIntentGateway:
    """Gateway bridge consuming decoupled trade intents and executing through Instrument.place."""

    def __init__(
        self,
        workspace_dir: Optional[str] = None,
        cache_dir: str = "cachedb",
        mode: Optional[str] = None,
        max_notional_eur: Optional[float] = None,
        min_confidence: Optional[float] = None,
        notify_enabled: Optional[bool] = None,
    ) -> None:
        self.workspace_dir = workspace_dir or ROOT_DIR
        self.cache_dir = os.path.join(self.workspace_dir, cache_dir)
        self.intents_file = os.path.join(self.cache_dir, "autonomous_trade_intents.json")

        conf = self._load_conf()

        raw_mode = mode or conf.get("autonomous_trading_mode", "shadow")
        self.mode = str(raw_mode).strip().lower()

        if max_notional_eur is not None:
            self.max_notional_eur = float(max_notional_eur)
        else:
            try:
                self.max_notional_eur = float(conf.get("autonomous_trading_max_notional_eur", "500.0"))
            except (ValueError, TypeError):
                self.max_notional_eur = 500.0

        if min_confidence is not None:
            self.min_confidence = float(min_confidence)
        else:
            try:
                self.min_confidence = float(conf.get("autonomous_trading_min_confidence", "0.85"))
            except (ValueError, TypeError):
                self.min_confidence = 0.85

        if notify_enabled is not None:
            self.notify_enabled = bool(notify_enabled)
        else:
            self.notify_enabled = bool(int(float(conf.get("autonomous_trading_notify", conf.get("shadow_notify", "1")))))

    def _load_conf(self) -> Dict[str, str]:
        """Loads configuration from order_guard.conf without vendor dependencies."""
        conf: Dict[str, str] = {}
        for fname in ("order_guard.conf", "config.env", ".env"):
            fpath = os.path.join(self.workspace_dir, fname)
            if not os.path.exists(fpath):
                continue
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "=" in line:
                            k, v = line.split("=", 1)
                            conf[k.strip().lower()] = v.strip().strip('"').strip("'")
            except Exception:
                pass
        return conf

    def load_all_intents(self) -> List[Dict[str, Any]]:
        """Loads all intents from the durable JSON queue."""
        if not os.path.exists(self.intents_file):
            return []
        try:
            with open(self.intents_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except Exception as e:
            logger.warning("Could not load intents from %s: %s", self.intents_file, e)
            return []

    def save_all_intents(self, intents: List[Dict[str, Any]]) -> bool:
        """Atomically persists all intents to disk."""
        try:
            from state_io import atomic_text_writer
            with atomic_text_writer(self.intents_file) as f:
                json.dump(intents, f, indent=2)
            return True
        except Exception:
            try:
                os.makedirs(self.cache_dir, exist_ok=True)
                with open(self.intents_file, "w", encoding="utf-8") as f:
                    json.dump(intents, f, indent=2)
                return True
            except Exception as e:
                logger.error("Failed saving intents to %s: %s", self.intents_file, e)
                return False

    def get_pending_intents(self) -> List[Dict[str, Any]]:
        """Returns all intents with PENDING status."""
        return [i for i in self.load_all_intents() if i.get("status") == "PENDING"]

    def resolve_instrument(
        self,
        symbol: str,
        venue: Optional[str] = None,
        instruments_map: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """Resolves Instrument instance for the given symbol and venue."""
        if instruments_map is None:
            try:
                from instruments_config import load_instruments
                instruments_map = load_instruments()
            except Exception as e:
                logger.error("Failed loading instrument registry: %s", e)
                return None

        sym_norm = symbol.strip().upper()
        venue_norm = venue.strip().lower() if venue else None

        # 1. Match by exact symbol and venue
        if venue_norm:
            for inst in instruments_map.values():
                if inst.symbol.upper() == sym_norm and inst.provider_name.lower() == venue_norm:
                    return inst

        # 2. Match by exact symbol
        for inst in instruments_map.values():
            if inst.symbol.upper() == sym_norm:
                return inst

        # 3. Match by base asset prefix (e.g. BTC in BTCUSDC)
        for inst in instruments_map.values():
            if inst.symbol.upper().startswith(sym_norm) or sym_norm.startswith(inst.symbol.upper()):
                return inst

        return None

    def _get_live_price(self, inst: Any) -> float:
        """Fetches live market price from provider or cache."""
        try:
            if hasattr(inst._provider, "get_current_price"):
                px = inst._provider.get_current_price(inst.symbol)
                px_f = float(px)
                if math.isfinite(px_f) and px_f > 0:
                    return px_f
        except Exception:
            pass

        # Fallback to cache_currentprice.json
        cache_path = os.path.join(self.cache_dir, "cache_currentprice.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cdata = json.load(f)
                    items = cdata.get("items", {})
                    if inst.symbol in items and items[inst.symbol]:
                        return float(items[inst.symbol][0][1])
            except Exception:
                pass

        return float("nan")

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
    ) -> Dict[str, Any]:
        """Validates and processes a single intent according to operational mode."""
        intent_id = intent.get("intent_id", f"intent_{int(time.time())}")
        decision = str(intent.get("decision", "HOLD")).strip().upper()
        symbol = str(intent.get("symbol", "")).strip().upper()
        venue = str(intent.get("venue", "binance")).strip().lower()
        now = time.time()

        if decision not in ("BUY", "SELL"):
            intent["status"] = "REJECTED_INVALID_DECISION"
            intent.setdefault("history", []).append({
                "ts": now, "action": "REJECTED_INVALID_DECISION", "decision": decision
            })
            return intent

        try:
            confidence = float(intent.get("confidence", 0.0))
        except (ValueError, TypeError):
            confidence = 0.0

        if confidence < self.min_confidence:
            logger.info("[INTENT GATEWAY] Intent %s rejected: confidence %.2f < %.2f",
                        intent_id, confidence, self.min_confidence)
            intent["status"] = "REJECTED_LOW_CONFIDENCE"
            intent.setdefault("history", []).append({
                "ts": now, "action": "REJECTED_LOW_CONFIDENCE",
                "confidence": confidence, "min_confidence": self.min_confidence
            })
            return intent

        inst = self.resolve_instrument(symbol, venue=venue, instruments_map=instruments_map)
        if not inst:
            logger.warning("[INTENT GATEWAY] Intent %s rejected: unknown instrument %s on %s",
                           intent_id, symbol, venue)
            intent["status"] = "REJECTED_UNKNOWN_INSTRUMENT"
            intent.setdefault("history", []).append({
                "ts": now, "action": "REJECTED_UNKNOWN_INSTRUMENT", "symbol": symbol, "venue": venue
            })
            return intent

        try:
            suggested_eur = float(intent.get("suggested_notional_eur", 250.0))
        except (ValueError, TypeError):
            suggested_eur = 250.0

        effective_eur = min(suggested_eur, self.max_notional_eur)
        effective_eur = max(10.0, effective_eur)

        price = self._get_live_price(inst)
        if not math.isfinite(price) or price <= 0:
            logger.warning("[INTENT GATEWAY] Intent %s rejected: invalid price for %s", intent_id, symbol)
            intent["status"] = "REJECTED_PRICE_UNAVAILABLE"
            intent.setdefault("history", []).append({
                "ts": now, "action": "REJECTED_PRICE_UNAVAILABLE", "symbol": symbol
            })
            return intent

        target_qty = effective_eur / price

        # Preflight Guard Vetting
        guard_allowed = True
        guard_reason = "Preflight checks passed"
        try:
            import order_guard
            profit_margin = order_guard.margin_for(inst.provider_name)
            window_ref = inst._call_profit_guard_window_ref(inst.symbol, decision, None)
            guard_allowed = order_guard.profit_guard(
                inst._provider, inst.symbol, decision, price, profit_margin,
                window_ref=window_ref, qty=target_qty,
            )
            if not guard_allowed:
                guard_reason = f"Profit guard vetoed {decision} {inst.symbol} @ {price}"
        except Exception as g_err:
            logger.warning("[INTENT GATEWAY] Guard evaluation exception: %s", g_err)
            guard_allowed = False
            guard_reason = f"Guard check failed: {g_err}"

        thesis = intent.get("thesis", "Autonomous strategy thesis")

        # Execution or Observation Branch
        if self.mode == "shadow":
            logger.info("[LLM TRADER · SHADOW] Would execute %s %s (qty=%g, price=$%.2f, notional=€%.2f) on %s. Thesis: %s",
                        decision, inst.symbol, target_qty, price, effective_eur, inst.provider_name, thesis)

            title = f"🟢 [LLM TRADER · SHADOW] {decision} {inst.symbol} €{effective_eur:.0f}"
            body = (
                f"Venue: {inst.provider_name} · Qty: {target_qty:g} @ ${price:,.2f}\n"
                f"Confidence: {confidence:.2f} · Notional: €{effective_eur:.2f}\n"
                f"Guards Preflight: {'PASSED' if guard_allowed else 'BLOCKED_BY_PROFIT_GUARD'}\n"
                f"Thesis: {thesis}"
            )
            self._dispatch_notify(title=title, body=body, symbol=inst.symbol)

            intent["status"] = "OBSERVED_SHADOW"
            intent.setdefault("history", []).append({
                "ts": now,
                "action": "OBSERVED_SHADOW",
                "guard_allowed": guard_allowed,
                "guard_reason": guard_reason,
                "price": price,
                "qty": target_qty,
                "notional_eur": effective_eur,
            })
            return intent

        # Mode == "enforce"
        if not guard_allowed:
            logger.warning("[LLM TRADER · ENFORCE] Guard blocked %s %s: %s", decision, inst.symbol, guard_reason)
            title = f"🛡 [LLM TRADER · ENFORCE] BLOCKED {decision} {inst.symbol}"
            body = (
                f"Venue: {inst.provider_name} · Price: ${price:,.2f} · Notional: €{effective_eur:.0f}\n"
                f"Reason: {guard_reason}\n"
                f"Thesis: {thesis}"
            )
            self._dispatch_notify(title=title, body=body, symbol=inst.symbol)

            intent["status"] = "BLOCKED_BY_GUARD"
            intent.setdefault("history", []).append({
                "ts": now,
                "action": "BLOCKED_BY_GUARD",
                "guard_reason": guard_reason,
                "price": price,
                "qty": target_qty,
            })
            return intent

        logger.warning("[LLM TRADER · ENFORCE] Executing %s %s (qty=%g, notional=€%.2f) on %s...",
                       decision, inst.symbol, target_qty, effective_eur, inst.provider_name)

        try:
            order_res = inst.place(
                side=decision,
                qty=target_qty,
                price=price,
                is_market=True,
            )
            if order_res:
                logger.info("[LLM TRADER · ENFORCE] Successfully placed %s %s: %s",
                            decision, inst.symbol, order_res)
                title = f"🚀 [LLM TRADER · ENFORCE] EXECUTED {decision} {inst.symbol} €{effective_eur:.0f}"
                body = (
                    f"Venue: {inst.provider_name} · Qty: {target_qty:g} @ ${price:,.2f}\n"
                    f"Order ID: {order_res.get('orderId', order_res.get('id', 'N/A'))}\n"
                    f"Thesis: {thesis}"
                )
                self._dispatch_notify(title=title, body=body, symbol=inst.symbol)

                intent["status"] = "EXECUTED"
                intent["execution_payload"] = order_res
                intent.setdefault("history", []).append({
                    "ts": now,
                    "action": "EXECUTED",
                    "payload": order_res,
                })
            else:
                logger.warning("[LLM TRADER · ENFORCE] Order placement refused or failed for %s %s",
                               decision, inst.symbol)
                title = f"⚠️ [LLM TRADER · ENFORCE] REFUSED {decision} {inst.symbol}"
                body = f"Venue {inst.provider_name} refused placement.\nThesis: {thesis}"
                self._dispatch_notify(title=title, body=body, symbol=inst.symbol)

                intent["status"] = "FAILED"
                intent.setdefault("history", []).append({
                    "ts": now,
                    "action": "FAILED",
                    "reason": "Instrument.place returned None",
                })
        except Exception as e:
            logger.error("[LLM TRADER · ENFORCE] Exception executing %s %s: %s", decision, inst.symbol, e)
            title = f"🚨 [LLM TRADER · ENFORCE] ERROR {decision} {inst.symbol}"
            body = f"Execution exception: {e}\nThesis: {thesis}"
            self._dispatch_notify(title=title, body=body, symbol=inst.symbol)

            intent["status"] = "FAILED"
            intent.setdefault("history", []).append({
                "ts": now,
                "action": "ERROR",
                "error": str(e),
            })

        return intent

    def process_all_pending(self) -> List[Dict[str, Any]]:
        """Processes all pending trade intents from the durable queue."""
        intents = self.load_all_intents()
        if not intents:
            logger.info("No intents found in queue.")
            return []

        pending_indices = [idx for idx, item in enumerate(intents) if item.get("status") == "PENDING"]
        if not pending_indices:
            logger.info("No pending trade intents to process.")
            return []

        logger.info("Processing %d pending autonomous trade intent(s) (mode=%s)...",
                    len(pending_indices), self.mode)

        instruments_map = None
        try:
            from instruments_config import load_instruments
            instruments_map = load_instruments()
        except Exception as e:
            logger.warning("Could not pre-load instruments: %s", e)

        processed: List[Dict[str, Any]] = []
        for idx in pending_indices:
            updated = self.process_intent(intents[idx], instruments_map=instruments_map)
            intents[idx] = updated
            processed.append(updated)

        self.save_all_intents(intents)
        return processed


def main() -> None:
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
            print(f"- [{i.get('status')}] {i.get('decision')} {i.get('symbol')} "
                  f"(conf={i.get('confidence')}, notional=€{i.get('suggested_notional_eur')}): {i.get('intent_id')}")
        return

    if args.clear_pending:
        intents = gateway.load_all_intents()
        cleared_cnt = 0
        for item in intents:
            if item.get("status") == "PENDING":
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

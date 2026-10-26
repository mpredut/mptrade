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
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("orchestratorTrade.autonomous_intent_gateway")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


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

_QUOTE_SUFFIXES: Tuple[str, ...] = ("USDC", "USDT", "FDUSD", "EUR", "RON", "GBP", "BTC", "ETH", "USD")


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


def _parse_finite_float(val: Any, default: float) -> float:
    """Parse ``val`` into a finite float, rejecting booleans, None, and NaN/Inf."""
    if val is None or isinstance(val, bool):
        return default
    try:
        parsed = float(val)
        return parsed if math.isfinite(parsed) else default
    except (ValueError, TypeError):
        return default


def _norm_sym(symbol: str) -> str:
    """Normalize a symbol for comparison by stripping separators and uppercasing."""
    return (symbol or "").strip().upper().replace("/", "").replace("-", "").replace("_", "")


def _split_symbol_assets(symbol: str) -> Optional[Tuple[str, str]]:
    """Split a standard spot pair symbol (e.g. TAOUSD, BTCUSDC, TAO/USD) into (base, quote)."""
    sym_u = _norm_sym(symbol)
    if sym_u.endswith("USEQ") and len(sym_u) > 4:
        return sym_u[:-4], "USD"
    for quote in _QUOTE_SUFFIXES:
        if sym_u.endswith(quote) and len(sym_u) > len(quote):
            return sym_u[: -len(quote)], quote
    return None


def _is_us_rth_open(now_ts: Optional[float] = None) -> bool:
    """Return True if US equities regular trading hours (09:30-16:00 America/New_York, Mon-Fri) are open."""
    from datetime import datetime, time as dt_time
    from zoneinfo import ZoneInfo

    ts = float(now_ts) if now_ts is not None else time.time()
    ny_dt = datetime.fromtimestamp(ts, tz=ZoneInfo("America/New_York"))
    if ny_dt.weekday() >= 5:
        return False
    return dt_time(9, 30) <= ny_dt.time() < dt_time(16, 0)


def _load_env_file_defaults(env_path: str) -> Dict[str, str]:
    """Parse an env file and set non-STRAT_* keys in os.environ via setdefault."""
    parsed: Dict[str, str] = {}
    if not os.path.exists(env_path):
        return parsed
    try:
        with open(env_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                cleaned = line.split("#", 1)[0].strip()
                if not cleaned or "=" not in cleaned:
                    continue
                k, v = cleaned.split("=", 1)
                key = k.strip()
                val = v.strip().strip('"').strip("'")
                if not key:
                    continue
                parsed[key.upper()] = val
                if not key.upper().startswith("STRAT_"):
                    os.environ.setdefault(key, val)
    except OSError:
        pass
    return parsed


def _discover_kraken_cache_pairs(workspace_dir: str) -> List[Tuple[str, str, str]]:
    """Read KRAKEN_CACHE_PAIRS from kraken/config.env to include all live Kraken spot pairs."""
    cfg_path = os.path.join(workspace_dir, "kraken", "config.env")
    if not os.path.exists(cfg_path):
        return []
    raw_pairs = ""
    try:
        with open(cfg_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                cleaned = line.split("#", 1)[0].strip()
                if not cleaned or "=" not in cleaned:
                    continue
                k, v = cleaned.split("=", 1)
                if k.strip().upper() == "KRAKEN_CACHE_PAIRS":
                    raw_pairs = v.strip().strip('"').strip("'")
                    break
    except OSError:
        return []

    out: List[Tuple[str, str, str]] = []
    for item in raw_pairs.split(","):
        pair = item.strip().upper()
        if not pair:
            continue
        split = _split_symbol_assets(pair)
        if split is not None:
            out.append((pair, split[0], split[1]))
    return out


def _discover_hyperliquid_spot(workspace_dir: str) -> List[Tuple[str, str, str]]:
    """Read hyperliquid/config.env to include live Hyperliquid spot pairs (e.g. HYPEUSDC)."""
    cfg_path = os.path.join(workspace_dir, "hyperliquid", "config.env")
    if not os.path.exists(cfg_path):
        return []
    kv = _load_env_file_defaults(cfg_path)
    if kv.get("HL_LIVE_ORDERS", "").lower() not in ("1", "true", "yes", "on"):
        return []
    token = (kv.get("HL_SPOT_TOKEN") or "HYPE").strip().upper()
    if not token:
        return []
    return [(f"{token}USDC", token, "USDC")]


def _discover_t212_equities(workspace_dir: str) -> List[Tuple[str, str, str, str]]:
    """Read 212trading/config.*.env to discover configured Trading212 equity tickers."""
    t212_dir = os.path.join(workspace_dir, "212trading")
    if not os.path.isdir(t212_dir):
        return []
    out: List[Tuple[str, str, str, str]] = []
    seen: set[str] = set()
    for cfg_path in sorted(glob.glob(os.path.join(t212_dir, "config.*.env"))):
        try:
            with open(cfg_path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    cleaned = line.split("#", 1)[0].strip()
                    if not cleaned or "=" not in cleaned:
                        continue
                    k, v = cleaned.split("=", 1)
                    if k.strip().upper() == "STRAT_PAIR":
                        ticker = v.strip().strip('"').strip("'").upper()
                        if ticker and ticker not in seen:
                            seen.add(ticker)
                            short = ticker.split("_", 1)[0]
                            out.append((f"T212_{short}", ticker, ticker, "USD"))
                        break
        except OSError:
            continue
    return out


def load_key_value_conf(workspace_dir: str, root_dir: str = ROOT_DIR) -> Dict[str, str]:
    """Load key=value settings from order_guard.conf / config.env / .env (root_dir overlaid by workspace_dir)."""
    conf: Dict[str, str] = {}
    search_dirs = [root_dir]
    if os.path.abspath(workspace_dir) != os.path.abspath(root_dir):
        search_dirs.append(workspace_dir)
    for base_dir in search_dirs:
        for fname in ("order_guard.conf", "config.env", ".env"):
            fpath = os.path.join(base_dir, fname)
            if not os.path.exists(fpath):
                continue
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.split("#", 1)[0].strip()
                    if not line or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    conf[k.strip().lower()] = v.strip().strip('"').strip("'")
    return conf


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
        self.intents_file = os.path.join(self.cache_dir, "autonomous_trade_intents.json")

        conf = self._load_conf()

        if mode is not None:
            raw_mode = mode
        else:
            if "autonomous_trading_mode" not in conf:
                raise KeyError("Missing required 'autonomous_trading_mode' in order_guard.conf")
            raw_mode = conf["autonomous_trading_mode"]
        self.mode = str(raw_mode).strip().lower()
        if self.mode not in ("off", "shadow", "enforce"):
            raise ValueError(
                f"Invalid autonomous_trading_mode {raw_mode!r}; must be one of ['enforce', 'off', 'shadow']"
            )

        if "autonomous_trading_min_notional_usd" not in conf:
            raise KeyError("Missing required 'autonomous_trading_min_notional_usd' in order_guard.conf")
        self.min_notional_usd = float(conf["autonomous_trading_min_notional_usd"])
        if not (math.isfinite(self.min_notional_usd) and self.min_notional_usd > 0):
            raise ValueError(f"Invalid autonomous_trading_min_notional_usd {self.min_notional_usd!r}")

        if max_notional_usd is not None:
            if isinstance(max_notional_usd, bool):
                raise ValueError(f"Invalid autonomous_trading_max_notional_usd {max_notional_usd!r}")
            self.max_notional_usd = float(max_notional_usd)
        else:
            if "autonomous_trading_max_notional_usd" not in conf:
                raise KeyError("Missing required 'autonomous_trading_max_notional_usd' in order_guard.conf")
            self.max_notional_usd = float(conf["autonomous_trading_max_notional_usd"])
        if not (math.isfinite(self.max_notional_usd) and self.max_notional_usd >= self.min_notional_usd):
            raise ValueError(
                f"Invalid autonomous_trading_max_notional_usd {self.max_notional_usd!r}; "
                f"must be >= autonomous_trading_min_notional_usd ({self.min_notional_usd})"
            )

        if min_confidence is not None:
            if isinstance(min_confidence, bool):
                raise ValueError(f"Invalid autonomous_trading_min_confidence {min_confidence!r}")
            self.min_confidence = float(min_confidence)
        else:
            if "autonomous_trading_min_confidence" not in conf:
                raise KeyError("Missing required 'autonomous_trading_min_confidence' in order_guard.conf")
            self.min_confidence = float(conf["autonomous_trading_min_confidence"])
        if not (math.isfinite(self.min_confidence) and 0.0 <= self.min_confidence <= 1.0):
            raise ValueError(
                f"Invalid autonomous_trading_min_confidence {self.min_confidence!r}; must be in [0.0, 1.0]"
            )

        if notify_enabled is not None:
            self.notify_enabled = bool(notify_enabled)
        else:
            if "autonomous_trading_notify" not in conf:
                raise KeyError("Missing required 'autonomous_trading_notify' in order_guard.conf")
            raw_notify = str(conf["autonomous_trading_notify"]).strip().lower()
            if raw_notify in ("1", "true", "yes", "on"):
                self.notify_enabled = True
            elif raw_notify in ("0", "false", "no", "off"):
                self.notify_enabled = False
            else:
                self.notify_enabled = bool(int(float(raw_notify)))

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
        """Reconciles FAILED intents against execution_audit_*.jsonl if later filled via retry worker."""
        failed_by_id: Dict[str, Dict[str, Any]] = {
            str(item.get("intent_id") or item.get("id")): item
            for item in intents
            if isinstance(item, dict)
            and item.get("status") == "FAILED"
            and (item.get("intent_id") or item.get("id"))
        }
        if not failed_by_id:
            return False

        audit_dir = os.path.join(self.workspace_dir, "logger", "execution_audit")
        if not os.path.isdir(audit_dir):
            return False

        matched_records: Dict[str, Dict[str, Any]] = {}
        for audit_path in sorted(glob.glob(os.path.join(audit_dir, "execution_audit_*.jsonl"))):
            try:
                with open(audit_path, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except ValueError:
                            continue
                        if not isinstance(rec, dict):
                            continue
                        iid = str(rec.get("intent_id") or "")
                        if iid not in failed_by_id:
                            continue
                        ev = str(rec.get("event") or "")
                        if ev == "submit_accepted":
                            matched_records.setdefault(iid, rec)
                        elif ev == "order_terminal":
                            v_stat = str(rec.get("venue_status") or "").upper()
                            l_stat = str(rec.get("status") or "").lower()
                            if v_stat in ("FILLED", "PARTIALLY_FILLED") or l_stat in ("closed", "filled", "partially_filled"):
                                matched_records[iid] = rec
            except OSError:
                continue

        if not matched_records:
            return False

        updated = False
        for iid, rec in matched_records.items():
            intent = failed_by_id[iid]
            filled_qty = rec.get("filled_qty") if rec.get("filled_qty") is not None else rec.get("qty")
            exec_payload: Dict[str, Any] = {
                "orderId": rec.get("order_id"),
                "symbol": rec.get("symbol") or intent.get("symbol"),
                "side": str(rec.get("side") or intent.get("decision") or "").upper(),
                "executedQty": str(filled_qty) if filled_qty is not None else "",
                "cummulativeQuoteQty": str(rec.get("cost")) if rec.get("cost") is not None else "",
                "status": str(rec.get("venue_status") or "FILLED").upper(),
                "reconciled_from_audit": True,
            }
            intent["status"] = "EXECUTED"
            intent["execution_payload"] = exec_payload
            try:
                rec_ts = float(rec.get("ts") or time.time())
                if not math.isfinite(rec_ts) or rec_ts <= 0:
                    rec_ts = time.time()
            except (TypeError, ValueError):
                rec_ts = time.time()
            intent.setdefault("history", []).append({
                "ts": rec_ts,
                "action": "RECONCILED_EXECUTED",
                "order_id": rec.get("order_id"),
                "payload": exec_payload,
            })
            updated = True

        if updated:
            self.save_all_intents(intents)
        return updated

    def save_all_intents(self, intents: List[Dict[str, Any]]) -> bool:
        """Transactionally persists all intents to disk using `.previous` staging and atomic replace."""
        from state_io import atomic_write_json, discard_generation_file, stage_previous_generation
        backup_path: Optional[str] = None
        try:
            if os.path.exists(self.intents_file):
                backup_path = stage_previous_generation(self.intents_file)
            atomic_write_json(self.intents_file, intents, indent=2)
            if backup_path:
                discard_generation_file(backup_path)
            return True
        except Exception as e:
            logger.error("Failed saving intents to %s: %s", self.intents_file, e)
            return False

    def get_pending_intents(self) -> List[Dict[str, Any]]:
        """Returns all intents with PENDING status."""
        return [i for i in self.load_all_intents() if i.get("status") == "PENDING"]

    def load_tradable_instruments(
        self,
        *,
        only_workspace_conf: bool = False,
        include_closed_rth: bool = False,
        now_ts: Optional[float] = None,
        asset_scope: str = "all",
    ) -> Dict[str, Any]:
        """Load enabled instruments filtered by asset_scope ('crypto' | 'equities' | 'all')."""
        scope = (asset_scope or "all").strip().lower()
        ws_inst_conf = os.path.join(self.workspace_dir, "instruments.conf")
        if only_workspace_conf and not os.path.exists(ws_inst_conf):
            return {}

        try:
            from instruments_config import load_instruments
            raw_map = load_instruments(ws_inst_conf) if os.path.exists(ws_inst_conf) else load_instruments()
        except Exception as e:
            logger.error("Failed loading instrument registry: %s", e)
            return {}

        active_map: Dict[str, Any] = {}
        if scope in ("all", "crypto"):
            for k, inst in raw_map.items():
                if getattr(inst, "enabled", True) is False:
                    continue
                if scope == "crypto" and str(getattr(inst, "provider_name", "")).lower() == "t212":
                    continue
                active_map[k] = inst

            existing_kraken_syms = {
                str(getattr(inst, "symbol", "")).upper()
                for inst in active_map.values()
                if str(getattr(inst, "provider_name", "")).lower() == "kraken"
            }
            for pair, base, quote in _discover_kraken_cache_pairs(self.workspace_dir):
                if pair in existing_kraken_syms:
                    continue
                try:
                    from instrument import Instrument
                    inst = Instrument(
                        name=f"KRAKEN_{base}",
                        provider="kraken",
                        symbol=pair,
                        base=base,
                        quote=quote,
                        enabled=True,
                    )
                    active_map[f"KRAKEN_{base}"] = inst
                    existing_kraken_syms.add(pair)
                except Exception as exc:
                    logger.debug("Could not bind Kraken pair %s: %s", pair, exc)

            existing_hl_syms = {
                str(getattr(inst, "symbol", "")).upper()
                for inst in active_map.values()
                if str(getattr(inst, "provider_name", "")).lower() == "hyperliquid"
            }
            for pair, base, quote in _discover_hyperliquid_spot(self.workspace_dir):
                if pair in existing_hl_syms or base in existing_hl_syms:
                    continue
                try:
                    from instrument import Instrument
                    inst = Instrument(
                        name=f"HYPERLIQUID_{base}",
                        provider="hyperliquid",
                        symbol=pair,
                        base=base,
                        quote=quote,
                        enabled=True,
                    )
                    if hasattr(inst, "_provider"):
                        inst._provider._live_enabled = True
                    active_map[f"HYPERLIQUID_{base}"] = inst
                    existing_hl_syms.add(pair)
                except Exception as exc:
                    logger.debug("Could not bind Hyperliquid pair %s: %s", pair, exc)

        if scope in ("all", "equities") and (include_closed_rth or _is_us_rth_open(now_ts)):
            _load_env_file_defaults(os.path.join(self.workspace_dir, "212trading", "runtime.env"))
            existing_t212_syms = {
                str(getattr(inst, "symbol", "")).upper()
                for inst in active_map.values()
                if str(getattr(inst, "provider_name", "")).lower() == "t212"
            }
            for inst_name, ticker, base, quote in _discover_t212_equities(self.workspace_dir):
                if ticker in existing_t212_syms:
                    continue
                try:
                    from instrument import Instrument
                    inst = Instrument(
                        name=inst_name,
                        provider="t212",
                        symbol=ticker,
                        base=base,
                        quote=quote,
                        enabled=True,
                        market_hours="rth",
                    )
                    if hasattr(inst, "_provider"):
                        inst._provider._live_enabled = True
                        inst._provider._order_validity = "DAY"
                    active_map[inst_name] = inst
                    existing_t212_syms.add(ticker)
                except Exception as exc:
                    logger.debug("Could not bind Trading212 ticker %s: %s", ticker, exc)

        return active_map

    def resolve_instrument(
        self,
        symbol: str,
        venue: Optional[str] = None,
        instruments_map: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """Resolves an enabled Instrument instance for the given symbol and venue."""
        if instruments_map is None:
            instruments_map = self.load_tradable_instruments()

        active_values = [
            inst for inst in instruments_map.values()
            if getattr(inst, "enabled", True) is not False
        ]

        sym_norm = _norm_sym(symbol)
        venue_norm = venue.strip().lower() if venue else None
        req_base = (_split_symbol_assets(sym_norm) or (sym_norm, ""))[0]

        def _inst_base_norm(inst: Any) -> str:
            raw_b = getattr(inst, "base", None) or (_split_symbol_assets(inst.symbol) or (inst.symbol, ""))[0]
            b_norm = _norm_sym(str(raw_b))
            if b_norm.endswith("USEQ") and len(b_norm) > 4:
                return b_norm[:-4]
            return b_norm

        # 1. Match strictly within the specified venue (never cross-venue fallback)
        if venue_norm:
            for inst in active_values:
                if _norm_sym(inst.symbol) == sym_norm and inst.provider_name.lower() == venue_norm:
                    return inst
            for inst in active_values:
                inst_sym_n = _norm_sym(inst.symbol)
                if inst.provider_name.lower() == venue_norm and (
                    inst_sym_n.startswith(sym_norm) or sym_norm.startswith(inst_sym_n)
                ):
                    return inst
            if req_base:
                for inst in active_values:
                    if inst.provider_name.lower() == venue_norm and _inst_base_norm(inst) == req_base:
                        return inst
            return None

        # 2. Match by exact symbol
        for inst in active_values:
            if _norm_sym(inst.symbol) == sym_norm:
                return inst

        # 3. Match by base asset prefix (e.g. BTC in BTCUSDC) or shared base asset
        for inst in active_values:
            inst_sym_n = _norm_sym(inst.symbol)
            if inst_sym_n.startswith(sym_norm) or sym_norm.startswith(inst_sym_n):
                return inst
        if req_base:
            for inst in active_values:
                if _inst_base_norm(inst) == req_base:
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

        # Fallback to cache_current_price.json (supporting cross-venue quote variants, e.g. TAOUSD -> TAOUSDC)
        cache_path = os.path.join(self.cache_dir, "cache_current_price.json")
        try:
            from state_io import read_recovering_previous
            cdata = read_recovering_previous(cache_path, json.loads, default=None)
            if isinstance(cdata, dict):
                items = cdata.get("items", {})
                sym_u = str(inst.symbol or "").strip().upper()
                candidates = [inst.symbol, sym_u]
                inst_base = getattr(inst, "base", None) or (_split_symbol_assets(sym_u) or (sym_u, ""))[0]
                if inst_base:
                    for suffix in ("USDC", "USDT", "USD"):
                        cand = f"{str(inst_base).upper()}{suffix}"
                        if cand not in candidates:
                            candidates.append(cand)
                for cand in candidates:
                    if cand in items and items[cand]:
                        try:
                            px_f = float(items[cand][0][1])
                            if math.isfinite(px_f) and px_f > 0:
                                return px_f
                        except (TypeError, ValueError, IndexError):
                            continue
                if inst_base:
                    base_u = str(inst_base).upper()
                    for k, v in items.items():
                        k_base = (_split_symbol_assets(str(k)) or (str(k).upper(), ""))[0]
                        if k_base == base_u and v:
                            try:
                                px_f = float(v[0][1])
                                if math.isfinite(px_f) and px_f > 0:
                                    return px_f
                            except (TypeError, ValueError, IndexError):
                                continue
        except Exception:
            pass

        return float("nan")

    def _clamp_notional_to_wallet_balance(
        self,
        inst: Any,
        decision: str,
        price: float,
        desired_usd: float,
    ) -> Tuple[bool, float, str]:
        """Clamp desired USD notional to real spot wallet liquidity when finite balance is exposed.

        Returns ``(allowed, clamped_usd, reason)`` symmetrically across BUY (quote asset) and SELL (base asset).
        """
        provider = getattr(inst, "_provider", None)
        free_fn = getattr(provider, "free_balance", None)
        if not callable(free_fn):
            return True, desired_usd, "Wallet balance check skipped (no provider hook)"

        from providers.quantity import resolve_assets
        raw_base = getattr(inst, "base", None)
        raw_quote = getattr(inst, "quote", None)
        base_asset, quote_asset = resolve_assets(
            str(inst.symbol),
            raw_base if isinstance(raw_base, str) else None,
            raw_quote if isinstance(raw_quote, str) else None,
        )
        target_asset = quote_asset if decision == "BUY" else base_asset
        if not target_asset:
            return True, desired_usd, "Wallet balance check skipped (unresolved asset)"

        try:
            raw_bal = free_fn(target_asset)
        except Exception as exc:
            logger.debug("Wallet free_balance(%s) check skipped: %s", target_asset, exc)
            return True, desired_usd, "Wallet balance check unavailable"

        if raw_bal is None or isinstance(raw_bal, bool) or not isinstance(raw_bal, (int, float)):
            return True, desired_usd, "Wallet balance check skipped (non-numeric)"
        bal = float(raw_bal)
        if not math.isfinite(bal):
            return True, desired_usd, "Wallet balance check skipped (non-finite)"

        venue_l = str(getattr(inst, "provider_name", "")).lower()
        buy_buf = 0.95 if venue_l == "hyperliquid" else 0.998
        avail_usd = bal * buy_buf if decision == "BUY" else bal * price
        if avail_usd < self.min_notional_usd:
            return (
                False,
                0.0,
                f"Insufficient free {target_asset} balance (${avail_usd:.2f} < ${self.min_notional_usd:.2f})",
            )
        if avail_usd < desired_usd:
            return True, avail_usd, f"Clamped notional from ${desired_usd:.2f} to available ${avail_usd:.2f}"
        return True, desired_usd, "Sufficient wallet balance"

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
        mode = mode or self.mode
        intent_id = intent.get("intent_id") or intent.get("id") or f"intent_{int(time.time())}"
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

        confidence = _parse_finite_float(intent.get("confidence"), 0.0)

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
        if not inst or (
            str(getattr(inst, "provider_name", "")).lower() != "t212"
            and str(getattr(inst, "symbol", "")).upper().endswith("BTC")
        ):
            logger.warning("[INTENT GATEWAY] Intent %s rejected: unknown or non-fiat instrument %s on %s",
                           intent_id, symbol, venue)
            intent["status"] = "REJECTED_UNKNOWN_INSTRUMENT"
            intent.setdefault("history", []).append({
                "ts": now, "action": "REJECTED_UNKNOWN_INSTRUMENT", "symbol": symbol, "venue": venue
            })
            return intent

        suggested_usd = _parse_finite_float(intent.get("suggested_notional_usd"), self.max_notional_usd)
        effective_usd = max(self.min_notional_usd, min(suggested_usd, self.max_notional_usd))

        price = self._get_live_price(inst)
        if not math.isfinite(price) or price <= 0:
            logger.warning("[INTENT GATEWAY] Intent %s rejected: invalid price for %s", intent_id, symbol)
            intent["status"] = "REJECTED_PRICE_UNAVAILABLE"
            intent.setdefault("history", []).append({
                "ts": now, "action": "REJECTED_PRICE_UNAVAILABLE", "symbol": symbol
            })
            return intent

        # Minimal Operational Preflight (Symmetric BUY / SELL, Zero Strategy or Daily-Cap Guards):
        # Clamp desired notional strictly to real spot wallet free liquidity.
        guard_allowed, effective_usd, guard_reason = self._clamp_notional_to_wallet_balance(
            inst, decision, price, effective_usd
        )
        target_qty = effective_usd / price if effective_usd > 0 else 0.0

        thesis = intent.get("thesis", "Autonomous strategy thesis")
        from orchestratorOS.agy.topics import TOPIC_SPECS, topic_spec
        src_topic = str(intent.get("created_by") or "").strip()
        if src_topic in TOPIC_SPECS:
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
            intent.setdefault("history", []).append({
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
            intent.setdefault("history", []).append({
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
            intent.setdefault("history", []).append({
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
                intent.setdefault("history", []).append({
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
                intent.setdefault("history", []).append(hist_entry)
        except Exception as e:
            logger.error("[LLM TRADER · ENFORCE] Exception executing %s %s: %s", decision, inst.symbol, e)
            title = f"🚨 {trade_tag} ERROR {decision} {inst.symbol}"
            body = f"Execution exception: {e}\nThesis: {thesis}"
            self._dispatch_notify(title=title, body=body, symbol=inst.symbol)

            intent["status"] = "FAILED"
            intent.setdefault("history", []).append({
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
        side = str(decision.get("decision", "HOLD")).strip().upper()
        if side not in ("BUY", "SELL"):
            logger.info("[INTENT GATEWAY] %s decision is %s (no intent).", source, side)
            return None
        symbol = str(decision.get("symbol", "")).strip().upper()
        venue = str(decision.get("venue", "")).strip().lower()
        if not symbol or not venue:
            logger.warning("[INTENT GATEWAY] %s %s decision missing symbol/venue (no intent).", source, side)
            return None
        confidence = _parse_finite_float(decision.get("confidence"), 0.0)
        if confidence < self.min_confidence:
            logger.info("[INTENT GATEWAY] %s %s %s confidence %.2f < %.2f (no intent).",
                        source, side, symbol, confidence, self.min_confidence)
            return None
        notional = _parse_finite_float(decision.get("suggested_notional_usd"), self.max_notional_usd)
        notional = max(self.min_notional_usd, min(notional, self.max_notional_usd))

        now = time.time()
        intents = self.load_all_intents()
        base_id = f"intent_{int(now)}_{side.lower()}_{symbol.lower()}"
        existing_ids = {i.get("intent_id") for i in intents if isinstance(i, dict)}
        intent_id = base_id if base_id not in existing_ids else f"{base_id}_{len(intents) + 1}"
        intent = asdict(TradeIntentRecord(
            intent_id=intent_id,
            timestamp=now,
            decision=side,
            symbol=symbol,
            venue=venue,
            confidence=confidence,
            suggested_notional_usd=notional,
            urgency=str(decision.get("urgency", "NORMAL")).strip().upper(),
            thesis=str(decision.get("thesis", "")).strip(),
            created_by=source,
        ))
        intents.append(intent)
        self.save_all_intents(intents)
        norm_max_mode = str(max_mode or "shadow").strip().lower()
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

        pending_indices = [idx for idx, item in enumerate(intents) if item.get("status") == "PENDING"]
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

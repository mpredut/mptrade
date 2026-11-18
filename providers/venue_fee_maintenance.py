"""3-tier polymorphic venue fee optimization and reserve maintenance (BNB fee reserve, live fee tiers)."""
from __future__ import annotations

from abc import ABC, abstractmethod
import importlib
import math
import os
from typing import Any, Callable, Dict, List, Optional, Type

import botcore
from botcore import (
    FileLock,
    _parse_bool_flag,
    clean_scalar_str,
    clean_upper_scalar,
    exact_pos_int,
    extract_order_id,
    extract_timestamp_s,
    is_cache_fresh,
    is_interval_due,
    load_env_stack,
    normalize_side,
    require_pos_float,
    required_bool_env,
    required_float_env,
    required_int_env,
    resolve_now_ts,
    safe_nonneg_float,
    safe_pos_float,
)
from cachepaths import (
    VENUE_FEE_MAINTENANCE_STATE_FILENAME,
    cache_path,
    has_equity_suffix,
    normalize_venue_name,
    resolve_cachedb_filepath,
)
from instrument_registry import casefold_get, symbols_for
from state_io import read_json_dict, save_json_state

from .base import MarketDataProvider
from .execution_audit import ExecutionAudit, new_intent_id
from .hyperliquid_provider import HyperliquidProvider, _call_hl_info_read
from .kraken_provider import KrakenProvider
from .quantity import BinanceOrderRules
from .strategy_executor import SubmissionRefused


VENUE_FEE_MAINTAINER_REGISTRY: Dict[str, Type["VenueFeeMaintainerBase"]] = {}

_REQUIRED_CFG_SPEC: tuple[tuple[str, str], ...] = (
    ("FEE_MAINTENANCE_ENABLED", "bool"),
    ("FEE_MAINTENANCE_INTERVAL_SEC", "int"),
    ("BINANCE_BNB_RESERVE_MIN_USDC", "float"),
    ("BINANCE_BNB_RESERVE_TARGET_USDC", "float"),
    ("BINANCE_BNB_MIN_FREE_USDC", "float"),
    ("BINANCE_BNB_ORDER_STALE_SEC", "int"),
    ("KRAKEN_IDLE_USDC_ALERT_MIN", "float"),
    ("KRAKEN_IDLE_USDC_ALERT_COOLDOWN_SEC", "int"),
)


def load_fee_maintenance_config(cfg_map: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load and validate mandatory venue fee maintenance policy without static fallbacks."""
    if cfg_map is None:
        load_env_stack()
        enabled = required_bool_env("FEE_MAINTENANCE_ENABLED")
        interval_sec = require_pos_float(required_int_env("FEE_MAINTENANCE_INTERVAL_SEC"), "FEE_MAINTENANCE_INTERVAL_SEC")
        reserve_min = require_pos_float(required_float_env("BINANCE_BNB_RESERVE_MIN_USDC"), "BINANCE_BNB_RESERVE_MIN_USDC")
        reserve_target = require_pos_float(required_float_env("BINANCE_BNB_RESERVE_TARGET_USDC"), "BINANCE_BNB_RESERVE_TARGET_USDC")
        min_free_usdc = require_pos_float(required_float_env("BINANCE_BNB_MIN_FREE_USDC"), "BINANCE_BNB_MIN_FREE_USDC")
        stale_sec = require_pos_float(required_int_env("BINANCE_BNB_ORDER_STALE_SEC"), "BINANCE_BNB_ORDER_STALE_SEC")
        kr_idle_min = require_pos_float(required_float_env("KRAKEN_IDLE_USDC_ALERT_MIN"), "KRAKEN_IDLE_USDC_ALERT_MIN")
        kr_cooldown_sec = require_pos_float(required_int_env("KRAKEN_IDLE_USDC_ALERT_COOLDOWN_SEC"), "KRAKEN_IDLE_USDC_ALERT_COOLDOWN_SEC")
    else:
        for key, _kind in _REQUIRED_CFG_SPEC:
            if key not in cfg_map:
                raise KeyError(f"Missing mandatory fee maintenance setting: {key}")
        raw_enabled = cfg_map["FEE_MAINTENANCE_ENABLED"]
        if isinstance(raw_enabled, bool):
            enabled = raw_enabled
        else:
            s_en = (clean_scalar_str(raw_enabled) or "").lower()
            if s_en in {"1", "true", "yes", "on"}:
                enabled = True
            elif s_en in {"0", "false", "no", "off"}:
                enabled = False
            else:
                raise ValueError(f"Invalid boolean for FEE_MAINTENANCE_ENABLED: {raw_enabled!r}")
        interval_sec = require_pos_float(cfg_map["FEE_MAINTENANCE_INTERVAL_SEC"], "FEE_MAINTENANCE_INTERVAL_SEC")
        reserve_min = require_pos_float(cfg_map["BINANCE_BNB_RESERVE_MIN_USDC"], "BINANCE_BNB_RESERVE_MIN_USDC")
        reserve_target = require_pos_float(cfg_map["BINANCE_BNB_RESERVE_TARGET_USDC"], "BINANCE_BNB_RESERVE_TARGET_USDC")
        min_free_usdc = require_pos_float(cfg_map["BINANCE_BNB_MIN_FREE_USDC"], "BINANCE_BNB_MIN_FREE_USDC")
        stale_sec = require_pos_float(cfg_map["BINANCE_BNB_ORDER_STALE_SEC"], "BINANCE_BNB_ORDER_STALE_SEC")
        kr_idle_min = require_pos_float(cfg_map["KRAKEN_IDLE_USDC_ALERT_MIN"], "KRAKEN_IDLE_USDC_ALERT_MIN")
        kr_cooldown_sec = require_pos_float(cfg_map["KRAKEN_IDLE_USDC_ALERT_COOLDOWN_SEC"], "KRAKEN_IDLE_USDC_ALERT_COOLDOWN_SEC")

    if reserve_min >= reserve_target:
        raise ValueError(
            f"BINANCE_BNB_RESERVE_MIN_USDC ({reserve_min}) must be strictly below BINANCE_BNB_RESERVE_TARGET_USDC ({reserve_target})"
        )
    if min_free_usdc < reserve_target:
        raise ValueError(
            f"BINANCE_BNB_MIN_FREE_USDC ({min_free_usdc}) must be >= BINANCE_BNB_RESERVE_TARGET_USDC ({reserve_target})"
        )
    return {
        "enabled": enabled,
        "interval_sec": interval_sec,
        "state_max_age_sec": interval_sec * 6.0,
        "binance_bnb_reserve_min_usdc": reserve_min,
        "binance_bnb_reserve_target_usdc": reserve_target,
        "binance_bnb_min_free_usdc": min_free_usdc,
        "binance_bnb_order_stale_sec": stale_sec,
        "kraken_idle_usdc_alert_min": kr_idle_min,
        "kraken_idle_usdc_alert_cooldown_sec": kr_cooldown_sec,
    }


def resolve_fee_maintenance_state_path(cache_dir: Optional[str] = None) -> str:
    """Resolve the canonical cachedb/state/venue_fee_maintenance_state.json path."""
    if cache_dir and (clean_dir := clean_scalar_str(cache_dir)):
        resolved = str(resolve_cachedb_filepath(os.path.join(clean_dir, VENUE_FEE_MAINTENANCE_STATE_FILENAME)))
        if os.path.exists(resolved):
            return resolved
        return os.path.join(clean_dir, "state", VENUE_FEE_MAINTENANCE_STATE_FILENAME)
    return str(cache_path(VENUE_FEE_MAINTENANCE_STATE_FILENAME))


class VenueFeeMaintainerInterface(ABC):
    """Level 1 pure abstract contract for per-venue fee hygiene and fee-schedule harvesting."""

    venue: str = ""

    @abstractmethod
    def maintain(
        self,
        now_ts: float,
        prev_venue_state: Dict[str, Any],
        cfg: Dict[str, Any],
        *,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Execute idempotent fee maintenance for this venue and return its persisted state."""


class VenueFeeMaintainerBase(VenueFeeMaintainerInterface, ABC):
    """Level 2 abstract base registering concrete venue maintainers and shared fee-spec builders."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        v_key = normalize_venue_name(getattr(cls, "venue", ""))
        if v_key and not getattr(cls, "__abstractmethods__", None):
            VENUE_FEE_MAINTAINER_REGISTRY[v_key] = cls

    @staticmethod
    def build_fee_spec(fee_tier: str, maker_fee_bps: float, taker_fee_bps: float) -> Dict[str, Any]:
        m_bps = safe_nonneg_float(maker_fee_bps, default=0.0) or 0.0
        t_bps = safe_nonneg_float(taker_fee_bps, default=0.0) or 0.0
        return {
            "fee_tier": clean_scalar_str(fee_tier) or "DEFAULT",
            "maker_fee_bps": round(m_bps, 2),
            "taker_fee_bps": round(t_bps, 2),
        }


class BinanceFeeMaintainer(VenueFeeMaintainerBase):
    """Level 3 maintainer ensuring Binance spotBNBBurn=True, BNB reserve hysteresis (<100 -> 200 USDC), and live fee tier."""

    venue = "binance"
    BNB_SYMBOL = "BNBUSDC"
    CLIENT_ORDER_PREFIX = "FEE_BNB_"
    SPOT_BNB_DISCOUNT_FACTOR = 0.75

    def __init__(
        self,
        *,
        client: Any = None,
        bapi_mod: Any = None,
        placeorder_mod: Any = None,
        audit: Optional[ExecutionAudit] = None,
    ) -> None:
        self._client = client
        self._bapi_mod = bapi_mod
        self._placeorder_mod = placeorder_mod
        self._audit = audit

    def _resolve_deps(self) -> tuple[Any, Any, Any, ExecutionAudit]:
        bapi_mod = self._bapi_mod or importlib.import_module("binance_api.bapi")
        placeorder_mod = self._placeorder_mod or importlib.import_module("binance_api.bapi_placeorder")
        cli = self._client if self._client is not None else bapi_mod.client
        audit = self._audit if self._audit is not None else ExecutionAudit()
        return cli, bapi_mod, placeorder_mod, audit

    @staticmethod
    def _extract_bnb_and_usdc(account_payload: Any) -> tuple[float, float, float]:
        bnb_free = 0.0
        bnb_locked = 0.0
        free_usdc = 0.0
        rows = account_payload.get("balances") if isinstance(account_payload, dict) else None
        for row in rows or ():
            if not isinstance(row, dict):
                continue
            asset = clean_upper_scalar(row.get("asset"))
            if asset == "BNB":
                bnb_free = safe_nonneg_float(row.get("free"), default=0.0) or 0.0
                bnb_locked = safe_nonneg_float(row.get("locked"), default=0.0) or 0.0
            elif asset == "USDC":
                free_usdc = safe_nonneg_float(row.get("free"), default=0.0) or 0.0
        return bnb_free, bnb_locked, free_usdc

    @classmethod
    def _resolve_passive_bnb_bid(cls, cli: Any) -> float:
        bid_px: Optional[float] = None
        ob_fn = getattr(cli, "get_orderbook_ticker", None)
        if callable(ob_fn):
            try:
                ob = ob_fn(symbol=cls.BNB_SYMBOL)
                if isinstance(ob, dict):
                    bid_px = safe_pos_float(ob.get("bidPrice"), default=None)
            except Exception:
                bid_px = None
        if bid_px is None:
            tk_fn = getattr(cli, "get_symbol_ticker", None)
            if callable(tk_fn):
                tk = tk_fn(symbol=cls.BNB_SYMBOL)
                if isinstance(tk, dict):
                    bid_px = safe_pos_float(tk.get("price"), default=None)
        if bid_px is None:
            raise RuntimeError("BNBUSDC passive bid price unavailable from Binance")
        return bid_px

    @staticmethod
    def _extract_base_fee_bps(cli: Any) -> tuple[float, float]:
        default_tier = MarketDataProvider.resolve_venue_default_fee_and_precision("binance", 0.0)
        maker_bps = safe_nonneg_float(default_tier[1], default=10.0) or 10.0
        taker_bps = safe_nonneg_float(default_tier[2], default=10.0) or 10.0
        tf_fn = getattr(cli, "get_trade_fee", None)
        if not callable(tf_fn):
            return maker_bps, taker_bps
        try:
            raw_tf = tf_fn(symbol="BTCUSDC")
            row = raw_tf[0] if isinstance(raw_tf, list) and raw_tf else (raw_tf if isinstance(raw_tf, dict) else None)
            if isinstance(row, dict):
                m_rate = safe_nonneg_float(row.get("makerCommission"), default=None)
                t_rate = safe_nonneg_float(row.get("takerCommission"), default=None)
                if m_rate is not None:
                    maker_bps = m_rate * 10000.0
                if t_rate is not None:
                    taker_bps = t_rate * 10000.0
        except Exception:
            pass
        return maker_bps, taker_bps

    def maintain(
        self,
        now_ts: float,
        prev_venue_state: Dict[str, Any],
        cfg: Dict[str, Any],
        *,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        cli, bapi_mod, placeorder_mod, audit = self._resolve_deps()
        events: List[Dict[str, Any]] = []
        live_allowed = (not _parse_bool_flag(dry_run, False)) and bool(botcore.is_trade_enabled())

        spot_bnb_burn: Optional[bool] = None
        bnb_burn_error: Optional[str] = None
        get_burn_fn = getattr(cli, "get_bnb_burn_spot_margin", None)
        if callable(get_burn_fn):
            try:
                burn_resp = get_burn_fn()
                if isinstance(burn_resp, dict):
                    spot_bnb_burn = _parse_bool_flag(burn_resp.get("spotBNBBurn"), False)
                    if not spot_bnb_burn and live_allowed:
                        toggle_fn = getattr(cli, "toggle_bnb_burn_spot_margin", None)
                        if callable(toggle_fn):
                            toggled = toggle_fn(spotBNBBurn=True)
                            spot_bnb_burn = _parse_bool_flag(
                                toggled.get("spotBNBBurn") if isinstance(toggled, dict) else True,
                                True,
                            )
                            events.append(
                                {
                                    "kind": "ops_notice",
                                    "venue": "binance",
                                    "symbol": self.BNB_SYMBOL,
                                    "title": "[BINANCE:FEE_MAINTAINER] Enabled Spot BNB Burn (25% Fee Discount)",
                                    "body": "spotBNBBurn was off on Binance; enabled spotBNBBurn=True automatically via SAPI.",
                                }
                            )
            except Exception as exc:
                bnb_burn_error = " ".join(str(exc).split())
                prev_burn_err_ts = safe_nonneg_float(prev_venue_state.get("last_bnb_burn_error_ts"), default=0.0) or 0.0
                if is_interval_due(now_ts, prev_burn_err_ts, cfg["kraken_idle_usdc_alert_cooldown_sec"]):
                    events.append(
                        {
                            "kind": "ops_notice",
                            "venue": "binance",
                            "symbol": self.BNB_SYMBOL,
                            "title": "[BINANCE:FEE_MAINTAINER] Check Spot BNB Burn Toggle",
                            "body": f"Could not query/toggle spotBNBBurn via API ({bnb_burn_error}). Verify 'Use BNB to pay fees' in Binance UI.",
                        }
                    )

        bnb_free, bnb_locked, free_usdc = self._extract_bnb_and_usdc(cli.get_account())
        total_bnb = bnb_free + bnb_locked
        best_bid = self._resolve_passive_bnb_bid(cli)

        stale_sec = cfg["binance_bnb_order_stale_sec"]
        fresh_open_bnb_qty = 0.0
        has_pending_bnb_order = False
        canceled_stale_ids: List[str] = []
        open_orders = cli.get_open_orders(symbol=self.BNB_SYMBOL) or []
        for oo in open_orders:
            if not isinstance(oo, dict):
                continue
            coid = clean_scalar_str(oo.get("clientOrderId")) or ""
            if not coid.startswith(self.CLIENT_ORDER_PREFIX) or normalize_side(oo.get("side")) != "BUY":
                continue
            orig_q = safe_nonneg_float(oo.get("origQty"), default=0.0) or 0.0
            exec_q = safe_nonneg_float(oo.get("executedQty"), default=0.0) or 0.0
            rem_q = max(0.0, orig_q - exec_q)
            if rem_q <= 0.0:
                continue
            ord_ts = extract_timestamp_s(oo, default=None)
            age_sec = max(0.0, now_ts - ord_ts) if ord_ts is not None else 0.0
            oid_str = extract_order_id(oo, default="") or ""
            if age_sec >= stale_sec and live_allowed and oid_str:
                if bapi_mod._cancel_and_resolve(self.BNB_SYMBOL, oid_str):
                    canceled_stale_ids.append(oid_str)
                    audit.record(
                        "cancel_stale_fee_order",
                        intent_id=coid,
                        venue="binance",
                        symbol=self.BNB_SYMBOL,
                        order_id=oid_str,
                        age_sec=round(age_sec, 1),
                    )
                    continue
            fresh_open_bnb_qty += rem_q
            has_pending_bnb_order = True

        held_bnb_usdc = total_bnb * best_bid
        effective_bnb_usdc = (total_bnb + fresh_open_bnb_qty) * best_bid
        reserve_min_usdc = cfg["binance_bnb_reserve_min_usdc"]
        reserve_target_usdc = cfg["binance_bnb_reserve_target_usdc"]
        min_free_usdc = cfg["binance_bnb_min_free_usdc"]

        topup_action = "none"
        placed_order_id: Optional[str] = None
        retry_soon = False

        if effective_bnb_usdc < reserve_min_usdc and not has_pending_bnb_order:
            deficit_usdc = reserve_target_usdc - effective_bnb_usdc
            required_free_usdc = max(min_free_usdc, deficit_usdc * 1.01)
            if free_usdc < required_free_usdc:
                topup_action = "skipped_low_free_usdc"
            else:
                rules = BinanceOrderRules.from_symbol_info(cli.get_symbol_info(self.BNB_SYMBOL))
                norm_qty_dec, norm_px_dec = rules.normalize(
                    quantity=deficit_usdc / best_bid,
                    price=best_bid,
                    market=False,
                )
                qty_f = safe_pos_float(str(norm_qty_dec), default=None)
                px_f = safe_pos_float(str(norm_px_dec), default=None)
                if qty_f is None or px_f is None:
                    raise RuntimeError("Normalized BNBUSDC reserve top-up quantity or price is non-positive")
                ts_tag = exact_pos_int(math.trunc(now_ts), default=1) or 1
                coid = f"{self.CLIENT_ORDER_PREFIX}{ts_tag}"
                if _parse_bool_flag(dry_run, False):
                    topup_action = "dry_run_would_buy"
                elif not botcore.is_trade_enabled():
                    topup_action = "trading_disabled"
                else:
                    intent_id = new_intent_id("binance", self.BNB_SYMBOL, "fee_reserve")
                    audit.record(
                        "submit_requested",
                        intent_id=intent_id,
                        venue="binance",
                        symbol=self.BNB_SYMBOL,
                        side="buy",
                        qty=qty_f,
                        price=px_f,
                        market=False,
                        kind="fee_reserve",
                        client_order_id=coid,
                    )
                    try:
                        order = placeorder_mod._submit_binance_order(
                            "BUY",
                            self.BNB_SYMBOL,
                            qty_f,
                            price=px_f,
                            market=False,
                            client_order_id=coid,
                            api_client=cli,
                            kind="fee_reserve",
                            post_only=True,
                        )
                    except SubmissionRefused as ref_exc:
                        audit.record(
                            "submit_refused",
                            intent_id=intent_id,
                            venue="binance",
                            symbol=self.BNB_SYMBOL,
                            submission_state="refused",
                            reason=str(ref_exc),
                            side="buy",
                            qty=qty_f,
                            price=px_f,
                            market=False,
                            kind="fee_reserve",
                            client_order_id=coid,
                        )
                        topup_action = f"refused:{ref_exc}"
                        retry_soon = True
                        order = None

                    if isinstance(order, dict) and (oid := extract_order_id(order, default=None)):
                        placed_order_id = str(oid)
                        sub_px = safe_pos_float(order.get("price"), default=px_f) or px_f
                        topup_action = "placed_limit_maker"
                        has_pending_bnb_order = True
                        audit.record(
                            "submit_accepted",
                            intent_id=intent_id,
                            venue="binance",
                            symbol=self.BNB_SYMBOL,
                            order_id=placed_order_id,
                            submission_state="accepted",
                            side="buy",
                            qty=qty_f,
                            price=sub_px,
                            market=False,
                            kind="fee_reserve",
                            client_order_id=coid,
                        )
                        events.append(
                            {
                                "kind": "order_placed",
                                "venue": "binance",
                                "symbol": self.BNB_SYMBOL,
                                "side": "BUY",
                                "order_id": placed_order_id,
                                "price": sub_px,
                                "qty": qty_f,
                                "notional_usd": round(sub_px * qty_f, 2),
                                "order_type": "LIMIT_MAKER",
                                "owner": "FEE_MAINTAINER",
                                "thesis": (
                                    f"BNB reserve {held_bnb_usdc:.2f} USDC < {reserve_min_usdc:.0f} USDC; "
                                    f"passive LIMIT_MAKER top-up to {reserve_target_usdc:.0f} USDC target."
                                ),
                            }
                        )
                    elif not retry_soon:
                        topup_action = "submit_failed"
                        retry_soon = True

        base_maker_bps, base_taker_bps = self._extract_base_fee_bps(cli)
        discount_active = bool(spot_bnb_burn) and held_bnb_usdc >= 1.0
        eff_maker_bps = base_maker_bps * self.SPOT_BNB_DISCOUNT_FACTOR if discount_active else base_maker_bps
        eff_taker_bps = base_taker_bps * self.SPOT_BNB_DISCOUNT_FACTOR if discount_active else base_taker_bps
        fee_tier = "VIP0_BNB" if discount_active else "VIP0"

        out: Dict[str, Any] = {
            "ts": round(now_ts, 3),
            "fee_spec": self.build_fee_spec(fee_tier, eff_maker_bps, eff_taker_bps),
            "spot_bnb_burn": spot_bnb_burn,
            "bnb_free": round(bnb_free, 6),
            "bnb_locked": round(bnb_locked, 6),
            "bnb_price_usdc": round(best_bid, 4),
            "held_bnb_usdc": round(held_bnb_usdc, 2),
            "effective_bnb_usdc": round(effective_bnb_usdc, 2),
            "free_usdc": round(free_usdc, 2),
            "has_pending_bnb_order": has_pending_bnb_order,
            "retry_soon": retry_soon,
            "topup_action": topup_action,
            "events": events,
        }
        if placed_order_id:
            out["placed_order_id"] = placed_order_id
        if canceled_stale_ids:
            out["canceled_stale_ids"] = canceled_stale_ids
        if bnb_burn_error:
            out["bnb_burn_error"] = bnb_burn_error
            out["last_bnb_burn_error_ts"] = round(now_ts, 3)
        elif prev_venue_state.get("last_bnb_burn_error_ts"):
            out["last_bnb_burn_error_ts"] = prev_venue_state["last_bnb_burn_error_ts"]
        return out


class KrakenFeeMaintainer(VenueFeeMaintainerBase):
    """Level 3 maintainer querying live Kraken 30d TradeVolume fee tiers and monitoring idle USDC."""

    venue = "kraken"

    def __init__(self, *, provider: Optional[KrakenProvider] = None) -> None:
        self._provider = provider

    def maintain(
        self,
        now_ts: float,
        prev_venue_state: Dict[str, Any],
        cfg: Dict[str, Any],
        *,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        kp = self._provider if self._provider is not None else KrakenProvider()
        events: List[Dict[str, Any]] = []
        default_tier = MarketDataProvider.resolve_venue_default_fee_and_precision("kraken", 0.0)
        maker_bps = safe_nonneg_float(default_tier[1], default=16.0) or 16.0
        taker_bps = safe_nonneg_float(default_tier[2], default=26.0) or 26.0
        fee_tier = str(default_tier[0])
        volume_30d_usd: Optional[float] = None
        by_symbol: Dict[str, Dict[str, Any]] = {}

        spot_pairs = [
            s
            for raw_s in symbols_for("kraken")
            if (s := clean_upper_scalar(raw_s)) is not None
            and not s.endswith("X.T")
            and not str(raw_s).strip().endswith("x.T")
            and not has_equity_suffix(s)
        ]
        cli = kp._client()
        priv_fn = getattr(cli, "_private", None)
        if callable(priv_fn) and spot_pairs:
            vol_resp = priv_fn("TradeVolume", {"pair": ",".join(spot_pairs)})
            if isinstance(vol_resp, dict):
                volume_30d_usd = safe_nonneg_float(vol_resp.get("volume"), default=None)
                fees_taker = vol_resp.get("fees") if isinstance(vol_resp.get("fees"), dict) else {}
                fees_maker = vol_resp.get("fees_maker") if isinstance(vol_resp.get("fees_maker"), dict) else {}
                pair_keys = list(dict.fromkeys([*fees_taker.keys(), *fees_maker.keys()]))
                for p_key in pair_keys:
                    t_row = fees_taker.get(p_key) if isinstance(fees_taker.get(p_key), dict) else {}
                    m_row = fees_maker.get(p_key) if isinstance(fees_maker.get(p_key), dict) else {}
                    t_pct = safe_nonneg_float(t_row.get("fee"), default=None)
                    m_pct = safe_nonneg_float(m_row.get("fee"), default=t_pct)
                    if t_pct is not None and m_pct is not None:
                        spec = self.build_fee_spec("PRO_LIVE", m_pct * 100.0, t_pct * 100.0)
                        by_symbol[str(p_key).strip().upper()] = spec
                if by_symbol:
                    first_spec = next(iter(by_symbol.values()))
                    fee_tier = first_spec["fee_tier"]
                    maker_bps = first_spec["maker_fee_bps"]
                    taker_bps = first_spec["taker_fee_bps"]

        idle_usdc = 0.0
        bd_fn = getattr(kp, "usd_cash_breakdown", None)
        if callable(bd_fn):
            raw_bd = bd_fn()
            if isinstance(raw_bd, dict):
                idle_usdc = safe_nonneg_float(raw_bd.get("USDC"), default=0.0) or 0.0

        last_alert_ts = safe_nonneg_float(prev_venue_state.get("last_idle_usdc_alert_ts"), default=0.0) or 0.0
        if (
            idle_usdc >= cfg["kraken_idle_usdc_alert_min"]
            and is_interval_due(now_ts, last_alert_ts, cfg["kraken_idle_usdc_alert_cooldown_sec"])
        ):
            events.append(
                {
                    "kind": "ops_notice",
                    "venue": "kraken",
                    "symbol": "USDCUSD",
                    "title": "[KRAKEN:FEE_MAINTAINER] Idle USDC Detected (Convert Free 1:1 to USD)",
                    "body": (
                        f"Kraken holds {idle_usdc:.2f} USDC idle. Convert USDC -> USD 1:1 with 0% fee in Kraken Convert "
                        f"to avoid the 20 bps USDCUSD orderbook fee."
                    ),
                }
            )
            if not _parse_bool_flag(dry_run, False):
                last_alert_ts = round(now_ts, 3)

        out: Dict[str, Any] = {
            "ts": round(now_ts, 3),
            "fee_spec": self.build_fee_spec(fee_tier, maker_bps, taker_bps),
            "by_symbol": by_symbol,
            "idle_usdc": round(idle_usdc, 2),
            "events": events,
        }
        if volume_30d_usd is not None:
            out["volume_30d_usd"] = round(volume_30d_usd, 2)
        if last_alert_ts > 0.0:
            out["last_idle_usdc_alert_ts"] = last_alert_ts
        return out


class HyperliquidFeeMaintainer(VenueFeeMaintainerBase):
    """Level 3 maintainer querying live Hyperliquid user_fees schedule and referral/staking discounts."""

    venue = "hyperliquid"

    def __init__(self, *, provider: Optional[HyperliquidProvider] = None) -> None:
        self._provider = provider

    def maintain(
        self,
        now_ts: float,
        prev_venue_state: Dict[str, Any],
        cfg: Dict[str, Any],
        *,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        del prev_venue_state, cfg, dry_run
        hp = self._provider if self._provider is not None else HyperliquidProvider()
        default_tier = MarketDataProvider.resolve_venue_default_fee_and_precision("hyperliquid", 0.0)
        maker_bps = safe_nonneg_float(default_tier[1], default=4.0) or 4.0
        taker_bps = safe_nonneg_float(default_tier[2], default=7.0) or 7.0
        fee_tier = str(default_tier[0])
        ref_disc = 0.0
        stake_disc = 0.0

        cli = hp._hl()
        addr = hp._env_account_address(cli)
        info_obj = getattr(cli, "info", None)
        uf_fn = getattr(info_obj, "user_fees", None)
        if callable(uf_fn) and addr:
            uf = _call_hl_info_read(uf_fn, addr)
            if isinstance(uf, dict):
                add_rate = safe_nonneg_float(uf.get("userSpotAddRate"), default=None)
                cross_rate = safe_nonneg_float(uf.get("userSpotCrossRate"), default=None)
                if add_rate is not None:
                    maker_bps = add_rate * 10000.0
                if cross_rate is not None:
                    taker_bps = cross_rate * 10000.0
                ref_disc = safe_nonneg_float(uf.get("activeReferralDiscount"), default=0.0) or 0.0
                stake_disc = safe_nonneg_float(uf.get("activeStakingDiscount"), default=0.0) or 0.0
                fee_tier = "SPOT_LIVE_DISC" if (ref_disc > 0.0 or stake_disc > 0.0) else "SPOT_TIER0"

        return {
            "ts": round(now_ts, 3),
            "fee_spec": self.build_fee_spec(fee_tier, maker_bps, taker_bps),
            "active_referral_discount": round(ref_disc, 6),
            "active_staking_discount": round(stake_disc, 6),
            "events": [],
        }


class T212FeeMaintainer(VenueFeeMaintainerBase):
    """Level 3 maintainer publishing canonical Trading 212 FX conversion fee metadata without API rate-limit usage."""

    venue = "t212"

    def maintain(
        self,
        now_ts: float,
        prev_venue_state: Dict[str, Any],
        cfg: Dict[str, Any],
        *,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        del prev_venue_state, cfg, dry_run
        default_tier = MarketDataProvider.resolve_venue_default_fee_and_precision("t212", 0.0)
        return {
            "ts": round(now_ts, 3),
            "fee_spec": self.build_fee_spec(str(default_tier[0]), default_tier[1], default_tier[2]),
            "events": [],
        }


def read_persisted_venue_fee_spec(
    venue: str,
    symbol: str = "",
    *,
    cache_dir: Optional[str] = None,
    now_ts: Optional[float] = None,
) -> Dict[str, Any]:
    """Read live persisted fee tier and maker/taker bps when an explicit cache_dir is provided."""
    if not cache_dir or not clean_scalar_str(cache_dir):
        return {}
    v_norm = normalize_venue_name(venue)
    if not v_norm:
        return {}
    state_path = resolve_fee_maintenance_state_path(cache_dir)
    payload = read_json_dict(state_path, default={})
    if not isinstance(payload, dict) or not payload:
        return {}
    venues_map = payload.get("venues")
    if not isinstance(venues_map, dict):
        return {}
    v_entry = venues_map.get(v_norm)
    if not isinstance(v_entry, dict):
        return {}
    max_age_sec = safe_pos_float(payload.get("state_max_age_sec"), default=None)
    if max_age_sec is None:
        iv_sec = safe_pos_float(payload.get("interval_sec"), default=14400.0) or 14400.0
        max_age_sec = iv_sec * 6.0
    eff_now = resolve_now_ts(now_ts)
    entry_ts = safe_pos_float(v_entry.get("ts"), default=safe_pos_float(payload.get("ts"), default=None))
    if not is_cache_fresh(entry_ts, eff_now, max_age_sec):
        return {}

    spec: Dict[str, Any] = {}
    base_spec = v_entry.get("fee_spec")
    if isinstance(base_spec, dict):
        spec.update(base_spec)
    by_sym = v_entry.get("by_symbol")
    if symbol and isinstance(by_sym, dict):
        sym_spec = casefold_get(by_sym, symbol)
        if isinstance(sym_spec, dict):
            spec.update(sym_spec)

    out: Dict[str, Any] = {}
    if clean_scalar_str(spec.get("fee_tier")):
        out["fee_tier"] = str(spec["fee_tier"]).strip()
    if (m_bps := safe_nonneg_float(spec.get("maker_fee_bps"), default=None)) is not None:
        out["maker_fee_bps"] = round(m_bps, 2)
    if (t_bps := safe_nonneg_float(spec.get("taker_fee_bps"), default=None)) is not None:
        out["taker_fee_bps"] = round(t_bps, 2)
    return out


def _effective_due_interval_sec(prev_state: Dict[str, Any], cfg: Dict[str, Any]) -> float:
    venues_map = prev_state.get("venues") if isinstance(prev_state, dict) else None
    bn_state = venues_map.get("binance") if isinstance(venues_map, dict) else None
    if isinstance(bn_state, dict) and (
        _parse_bool_flag(bn_state.get("has_pending_bnb_order"), False)
        or _parse_bool_flag(bn_state.get("retry_soon"), False)
    ):
        return min(cfg["interval_sec"], cfg["binance_bnb_order_stale_sec"])
    return cfg["interval_sec"]


def run_venue_fee_maintenance(
    *,
    venues: Optional[List[str]] = None,
    force: bool = False,
    dry_run: bool = False,
    cache_dir: Optional[str] = None,
    cfg_map: Optional[Dict[str, Any]] = None,
    maintainers: Optional[Dict[str, VenueFeeMaintainerInterface]] = None,
    event_handler: Optional[Callable[[Dict[str, Any]], None]] = None,
    now_ts: Optional[float] = None,
) -> Dict[str, Any]:
    """Run serialized cross-venue fee hygiene and persist state atomically under FileLock."""
    cfg = load_fee_maintenance_config(cfg_map)
    eff_now = resolve_now_ts(now_ts)
    if not cfg["enabled"] and not _parse_bool_flag(force, False):
        return {"status": "disabled", "ts": round(eff_now, 3), "venues": {}, "events": []}

    state_path = resolve_fee_maintenance_state_path(cache_dir)
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    lock_path = f"{state_path}.lock"

    with FileLock(lock_path):
        prev_state = read_json_dict(state_path, default={}, repair_files=True)
        if not isinstance(prev_state, dict):
            prev_state = {}
        last_run_ts = safe_nonneg_float(prev_state.get("last_run_ts"), default=0.0) or 0.0
        due_interval = _effective_due_interval_sec(prev_state, cfg)
        if last_run_ts > 0.0 and not is_interval_due(eff_now, last_run_ts, due_interval, force=force):
            return {
                "status": "not_due",
                "ts": round(eff_now, 3),
                "last_run_ts": last_run_ts,
                "due_in_sec": round(max(0.0, (last_run_ts + due_interval) - eff_now), 1),
                "venues": prev_state.get("venues") if isinstance(prev_state.get("venues"), dict) else {},
                "events": [],
            }

        target_venues = [
            normalize_venue_name(v)
            for v in (venues if venues else list(VENUE_FEE_MAINTAINER_REGISTRY.keys()))
            if normalize_venue_name(v)
        ]
        prev_venues = prev_state.get("venues") if isinstance(prev_state.get("venues"), dict) else {}
        updated_venues: Dict[str, Any] = dict(prev_venues)
        all_events: List[Dict[str, Any]] = []
        errors: Dict[str, str] = {}

        for v_key in target_venues:
            inst = (maintainers or {}).get(v_key)
            if inst is None:
                cls = VENUE_FEE_MAINTAINER_REGISTRY.get(v_key)
                if cls is None:
                    errors[v_key] = f"Unknown venue fee maintainer: {v_key}"
                    continue
                inst = cls()
            prev_v = prev_venues.get(v_key) if isinstance(prev_venues.get(v_key), dict) else {}
            try:
                res = dict(inst.maintain(eff_now, prev_v, cfg, dry_run=_parse_bool_flag(dry_run, False)))
                v_events = list(res.pop("events", None) or ())
                updated_venues[v_key] = res
                for ev in v_events:
                    all_events.append(ev)
                    if callable(event_handler):
                        event_handler(ev)
            except Exception as exc:
                err_msg = " ".join(str(exc).split())
                errors[v_key] = err_msg
                preserved = dict(prev_v)
                preserved["last_error"] = err_msg
                preserved["last_error_ts"] = round(eff_now, 3)
                updated_venues[v_key] = preserved

        new_state: Dict[str, Any] = {
            "schema_version": 1,
            "ts": round(eff_now, 3),
            "last_run_ts": round(eff_now, 3) if not errors or len(errors) < len(target_venues) else last_run_ts,
            "interval_sec": cfg["interval_sec"],
            "state_max_age_sec": cfg["state_max_age_sec"],
            "venues": updated_venues,
        }
        if not _parse_bool_flag(dry_run, False):
            save_json_state(state_path, new_state, indent=2)

        return {
            "status": "partial_error" if errors else "ok",
            "ts": round(eff_now, 3),
            "last_run_ts": new_state["last_run_ts"],
            "venues": updated_venues,
            "events": all_events,
            "errors": errors,
        }


__all__ = [
    "BinanceFeeMaintainer",
    "HyperliquidFeeMaintainer",
    "KrakenFeeMaintainer",
    "T212FeeMaintainer",
    "VENUE_FEE_MAINTAINER_REGISTRY",
    "VenueFeeMaintainerBase",
    "VenueFeeMaintainerInterface",
    "load_fee_maintenance_config",
    "read_persisted_venue_fee_spec",
    "resolve_fee_maintenance_state_path",
    "run_venue_fee_maintenance",
]

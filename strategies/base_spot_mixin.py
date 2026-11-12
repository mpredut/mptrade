"""Shared spot strategy mixin for reentry gate, position snapshot, volatility tuple, and exit manager wiring.

Unifies state persistence, open-order lookup, fill accounting, 1-hour volatility resolution,
adaptive pullback/drop calculation, PositionSnapshot construction, MHDSL stop-loss
coordination, MDHTP take-profit configuration, and ReentrySnapshot evaluation across
``strategies/spot_engine.py`` and ``212trading/strategy.py``.
"""

from __future__ import annotations

from collections import OrderedDict, deque
import math
import os
import signal
import threading
import time
from typing import Any, Callable

from botcore import (
    _parse_bool_flag, are_close, exact_int, exact_nonneg_int, exact_pos_int, log,
    lru_dict_put, normalize_epoch_sec, optional_env,
    required_bool_env, required_env, required_float_env, required_int_env,
    resolve_now_ts, safe_float, safe_int, safe_nonneg_float, safe_nonneg_int, safe_pos_float, safe_pos_int,
    split_csv, touch_module_heartbeat,
)
from intelligence.internal.state.volatility import (
    adaptive_thresholds, resolve_cached_volatility_1h,
    resolve_cached_volatility_1h_pct, vol_pct_to_ratio,
)
from providers.base import MarketDataProvider, normalize_side
from providers.order_ohlc_utils import clean_order_id
from .position_exit_manager import PositionExitManager, PositionSnapshot, ReentrySnapshot, is_stop_exit_kind
from .spot_rules import _iter_valid_shadow_prices
from .state_store import (
    JsonStateStore, build_dataclass_from_dict, build_strategy_exit_manager,
    compute_adaptive_reentry_pullback_pct, find_open_order, update_post_sell_anchors,
)


class SpotParamsMixin:
    """Shared from_dict constructor and percentage-sizing helpers for StratParams dataclasses."""

    @classmethod
    def from_dict(cls, values: dict, *, defaults: dict | None = None, warn_dropped: bool = False):
        """Build StratParams from a dict (e.g. a stored baseline report), ignoring removed fields."""
        return build_dataclass_from_dict(cls, values, defaults=defaults, warn_dropped=warn_dropped)

    def pct_sizing_on(self) -> bool:
        return (safe_pos_float(getattr(self, "total_budget", 0.0), default=0.0) or 0.0) > 0.0

    def allocated_budget(self) -> float:
        if self.pct_sizing_on():
            tot = safe_pos_float(getattr(self, "total_budget", 0.0), default=0.0) or 0.0
            alloc = safe_nonneg_float(getattr(self, "alloc_pct", 0.0), default=0.0) or 0.0
            return tot * alloc / 100.0
        return safe_nonneg_float(getattr(self, "max_budget", 0.0), default=0.0) or 0.0

    def effective_entry_amount(self) -> float:
        if self.pct_sizing_on():
            epct = safe_nonneg_float(getattr(self, "entry_pct", 0.0), default=0.0) or 0.0
            return self.allocated_budget() * epct / 100.0
        return safe_nonneg_float(getattr(self, "entry_amount", 0.0), default=0.0) or 0.0

    def effective_dca_amount_base(self) -> float:
        if self.pct_sizing_on():
            dpct = safe_nonneg_float(getattr(self, "dca_pct", 0.0), default=0.0) or 0.0
            return self.allocated_budget() * dpct / 100.0
        return safe_nonneg_float(getattr(self, "dca_amount", 0.0), default=0.0) or 0.0


def generate_tp_ladder(
    min_pct: float, max_pct: float, steps: int = 10, curve: str = "linear"
) -> list[tuple[float, float]]:
    """Generate a scale-out take-profit ladder from min_pct to max_pct across `steps` rungs."""
    if isinstance(min_pct, bool) or isinstance(max_pct, bool) or isinstance(steps, bool):
        raise ValueError("generate_tp_ladder parameters cannot be bool")
    min_f = safe_float(min_pct, default=None)
    max_f = safe_float(max_pct, default=None)
    steps_i = exact_int(steps, default=None)
    if min_f is None or max_f is None or steps_i is None:
        raise ValueError("generate_tp_ladder requires finite min_pct, max_pct, and integer steps")
    if steps_i < 1:
        return []
    if steps_i > 1 and max_f < min_f:
        raise ValueError("generate_tp_ladder requires max_pct >= min_pct")
    if steps_i == 1:
        return [(round(min_f, 1), 1.0)]
    steps = steps_i
    curve_lower = str(curve or "linear").strip().lower()
    if curve_lower not in (
        "linear", "uniform", "bell", "gaussian", "normal", "pyramid", "front_heavy", "back_heavy", "geometric",
    ):
        raise ValueError(f"unsupported generate_tp_ladder curve: {curve!r}")
    if curve_lower == "geometric":
        r_start, r_end = 1.0 + min_f / 100.0, 1.0 + max_f / 100.0
        if r_start <= 0.0 or r_end <= 0.0:
            raise ValueError("geometric generate_tp_ladder requires min_pct > -100 and max_pct > -100")
        ratio = (r_end / r_start) ** (1.0 / (steps - 1)) if steps > 1 else 1.0
        levels = [round(((r_start * (ratio ** i)) - 1.0) * 100.0, 1) for i in range(steps)]
    else:
        step_size = (max_f - min_f) / (steps - 1)
        levels = [round(min_f + i * step_size, 1) for i in range(steps)]
    if curve_lower in ("bell", "gaussian", "normal", "pyramid"):
        center = (steps - 1) / 2.0
        sigma = max(0.5, (steps - 1) / 2.8) if steps > 1 else 1.0
        weights = [math.exp(-0.5 * ((i - center) / sigma) ** 2) for i in range(steps)]
    elif curve_lower == "front_heavy":
        weights = [(steps - i) * 1.0 for i in range(steps)]
    elif curve_lower == "back_heavy":
        weights = [(i + 1) * 1.0 for i in range(steps)]
    else:
        weights = [1.0 for _ in range(steps)]
    total_w = sum(weights)
    return [(lvl, round(w / total_w, 4)) for lvl, w in zip(levels, weights)]


def adapt_ladder_rungs(
    remaining: list[tuple[float, float]], held: float, min_tranche_qty: float = 0.01
) -> list[tuple[float, float]]:
    """Dynamically clamp remaining rungs so each order has viable trading quantity."""
    held_f = safe_pos_float(held, None)
    if not remaining or held_f is None or held_f <= 1e-9:
        return []
    min_q = safe_pos_float(min_tranche_qty, 0.01) or 0.01
    max_viable_rungs = max(1, safe_nonneg_int(math.floor(held_f / max(1e-6, min_q)), default=1) or 1)
    if len(remaining) <= max_viable_rungs:
        return list(remaining)
    if max_viable_rungs == 1:
        return [remaining[-1]]
    indices = [
        safe_nonneg_int(round(i * (len(remaining) - 1) / (max_viable_rungs - 1)), default=0) or 0
        for i in range(max_viable_rungs)
    ]
    return [remaining[idx] for idx in sorted(set(indices))]


def _parse_tranches(spec: str) -> list:
    """Parse ``3:50,6:50`` into percentage/share pairs whose shares total 100."""
    if isinstance(spec, bool):
        raise ValueError(f"Invalid STRAT_TP_TRANCHES: {spec!r}")
    if spec is None or not str(spec).strip():
        return []
    spec = str(spec)
    out = []
    try:
        for part in spec.split(","):
            p_str, s_str = part.split(":")
            pct = safe_nonneg_float(p_str, default=None)
            share = safe_pos_float(s_str, default=None)
            if pct is None or share is None:
                raise ValueError("Invalid tranche values")
            out.append((pct, share))
        if abs(sum(share for _, share in out) - 100) >= 1e-6:
            raise ValueError("Tranche shares must total 100")
    except ValueError as exc:
        raise ValueError(f"Invalid STRAT_TP_TRANCHES: {spec!r}") from exc
    return out


def load_common_core_strat_env(env: dict | None, *, include_entry_dca: bool = True) -> dict[str, Any]:
    """Load shared core strategy settings across spot, T212, and Hyperliquid engines."""
    mode = required_env("STRATEGY_MODE", env).strip().lower()
    if mode not in {"avg_tp", "dca_only"}:
        raise ValueError(f"Invalid STRATEGY_MODE: {mode!r}")
    f = required_float_env
    cfg: dict[str, Any] = {
        "currency": required_env("STRAT_CURRENCY", env).strip().upper(),
        "entry_discount_pct": f("STRAT_ENTRY_DISCOUNT_PCT", env),
        "dca_drop_pct": f("STRAT_DCA_DROP_PCT", env),
        "check_minutes": f("STRAT_CHECK_MINUTES", env),
        "takeprofit_pct": f("STRAT_TAKEPROFIT_PCT", env),
        "max_budget": f("STRAT_MAX_BUDGET", env),
        "enable_takeprofit": (mode != "dca_only"),
        "order_ttl_min": f("STRAT_ORDER_TTL_MIN", env),
        "stop_loss_pct": f("STRAT_STOP_LOSS_PCT", env),
        "reentry_tolerance_pct": f("STRAT_REENTRY_TOLERANCE_PCT", env),
    }
    if include_entry_dca:
        cfg.update(
            entry_amount=f("STRAT_ENTRY", env),
            dca_amount=f("STRAT_DCA", env),
            max_dca_buys=required_int_env("STRAT_MAX_DCA_BUYS", env),
        )
    return cfg


def parse_tp_ladder_env(env: dict | None, generate_ladder_fn: Callable) -> list[tuple[float, float]]:
    """Parse ``STRAT_TP_LADDER`` or ``STRAT_TP_LADDER_MIN/MAX/STEPS`` via shared botcore env API."""
    if (mode := required_env("STRATEGY_MODE", env).strip().lower()) not in {"avg_tp", "dca_only"}:
        raise ValueError(f"Invalid STRATEGY_MODE: {mode!r}")
    raw_ladder = optional_env("STRAT_TP_LADDER", env=env)
    ladder_curve = optional_env("STRAT_TP_LADDER_CURVE", "linear", env=env).lower()
    if raw_ladder:
        clean_raw = raw_ladder.removeprefix("range:")
        if clean_raw.count(":") == 2 and "," not in clean_raw:
            try:
                p_min, p_max, p_steps = clean_raw.split(":")
                steps_i = exact_pos_int(p_steps, default=None)
                min_f = safe_pos_float(p_min, default=None)
                max_f = safe_pos_float(p_max, default=None)
                if steps_i is None or min_f is None or max_f is None:
                    raise ValueError("non-integer steps or non-positive min/max")
                return generate_ladder_fn(min_f, max_f, steps_i, curve=ladder_curve)
            except (ValueError, OverflowError) as exc:
                raise ValueError(f"Invalid STRAT_TP_LADDER: {raw_ladder!r}") from exc
        ladder: list[tuple[float, float]] = []
        for part in split_csv(raw_ladder):
            if ":" not in part:
                raise ValueError(f"Invalid STRAT_TP_LADDER: {raw_ladder!r}")
            lvl_s, frac_s = part.split(":", 1)
            lvl_f = safe_pos_float(lvl_s, default=None)
            frac_raw = safe_pos_float(frac_s, default=None)
            if lvl_f is None or frac_raw is None:
                raise ValueError(f"Invalid STRAT_TP_LADDER: {raw_ladder!r}")
            ladder.append((lvl_f, frac_raw / 100.0))
        if not ladder:
            raise ValueError(f"Invalid STRAT_TP_LADDER: {raw_ladder!r}")
        if sum(f for _, f in ladder) > 1.0 + 1e-6:
            raise ValueError(f"Invalid STRAT_TP_LADDER (fractions exceed 100%): {raw_ladder!r}")
        return ladder
    has_min = bool(optional_env("STRAT_TP_LADDER_MIN", env=env))
    has_max = bool(optional_env("STRAT_TP_LADDER_MAX", env=env))
    has_steps = bool(optional_env("STRAT_TP_LADDER_STEPS", env=env))
    if has_min or has_max or has_steps:
        if not (has_min and has_max and has_steps):
            raise ValueError(
                "STRAT_TP_LADDER_MIN, STRAT_TP_LADDER_MAX, and STRAT_TP_LADDER_STEPS must all be set together"
            )
        min_l = safe_pos_float(required_float_env("STRAT_TP_LADDER_MIN", env), default=None)
        max_l = safe_pos_float(required_float_env("STRAT_TP_LADDER_MAX", env), default=None)
        steps_l = required_int_env("STRAT_TP_LADDER_STEPS", env)
        if min_l is None or max_l is None or steps_l < 1:
            raise ValueError("STRAT_TP_LADDER_MIN/MAX must be > 0 and STRAT_TP_LADDER_STEPS >= 1")
        return generate_ladder_fn(min_l, max_l, steps_l, curve=ladder_curve)
    return []


def parse_max_dca_buys_env(
    env: dict | None, budget: float, entry_amt: float, dca_amt: float
) -> tuple[int, float]:
    """Parse ``STRAT_MAX_DCA_BUYS`` (supporting ``auto``/``budget`` or explicit non-negative integer)."""
    budget_f = safe_nonneg_float(budget, default=None)
    entry_f = safe_nonneg_float(entry_amt, default=None)
    dca_f = safe_nonneg_float(dca_amt, default=None)
    if budget_f is None or entry_f is None or dca_f is None:
        raise ValueError("parse_max_dca_buys_env requires finite non-negative budget, entry_amt, and dca_amt")
    mdb = optional_env("STRAT_MAX_DCA_BUYS", env=env).lower()
    if mdb in ("auto", "budget"):
        if budget_f < entry_f:
            raise ValueError(f"STRAT_MAX_BUDGET ({budget_f}) cannot be smaller than STRAT_ENTRY ({entry_f})")
        if dca_f <= 0.0:
            return 0, entry_f
        max_dca = safe_nonneg_int(math.floor((budget_f - entry_f) / dca_f), default=0) or 0
        return max_dca, budget_f - max_dca * dca_f
    if mdb:
        max_dca_i = exact_nonneg_int(mdb, default=None)
        if max_dca_i is None:
            raise ValueError(f"Invalid STRAT_MAX_DCA_BUYS: {mdb!r}")
        return max_dca_i, entry_f
    raise ValueError("Missing or empty mandatory setting: STRAT_MAX_DCA_BUYS")


def load_common_mhdpa_env(env: dict | None, *, include_spot_sl: bool = True) -> dict[str, Any]:
    """Load shared Hybrid Re-entry, MHDSL, and MDHTP configuration keys from ``env``."""
    b, f = required_bool_env, required_float_env
    cfg: dict[str, Any] = {
        "reentry_hybrid_enabled": b("STRAT_REENTRY_HYBRID_ENABLED", env),
        "reentry_peak_relative": b("STRAT_REENTRY_PEAK_RELATIVE", env),
        "reentry_pullback_pct": f("STRAT_REENTRY_PULLBACK_PCT", env),
        "reentry_pullback_adaptive": b("STRAT_REENTRY_PULLBACK_ADAPTIVE", env),
        "reentry_pullback_k": f("STRAT_REENTRY_PULLBACK_K", env),
        "reentry_pullback_min": f("STRAT_REENTRY_PULLBACK_MIN", env),
        "reentry_pullback_max": f("STRAT_REENTRY_PULLBACK_MAX", env),
        "reentry_ttl_hours": f("STRAT_REENTRY_TTL_HOURS", env),
        "sl_mhdsl_enabled": b("STRAT_SL_MHDSL_ENABLED", env),
        "sl_disaster_buffer_ratio": f("STRAT_SL_DISASTER_BUFFER_RATIO", env),
        "sl_disaster_buffer_max_pct": f("STRAT_SL_DISASTER_BUFFER_MAX_PCT", env),
        "sl_trend_defer_max_sec": f("STRAT_SL_TREND_DEFER_MAX_SEC", env),
        "sl_bounce_min_recovery_pct": f("STRAT_SL_BOUNCE_MIN_RECOVERY_PCT", env),
        "sl_bounce_pullback_pct": f("STRAT_SL_BOUNCE_PULLBACK_PCT", env),
        "sl_structural_break_enabled": b("STRAT_SL_STRUCTURAL_BREAK_ENABLED", env),
        "tp_mdhtp_enabled": b("STRAT_TP_MDHTP_ENABLED", env),
        "tp_stagnation_max_sec": f("STRAT_TP_STAGNATION_MAX_SEC", env),
    }
    if include_spot_sl:
        cfg.update(
            reentry_bear_bounce_pct=f("STRAT_REENTRY_BEAR_BOUNCE_PCT", env),
            tp_dynamic_flat=b("STRAT_TP_DYNAMIC_FLAT", env),
            tp_min_pct=f("STRAT_TP_MIN_PCT", env),
            tp_max_pct=f("STRAT_TP_MAX_PCT", env),
            trend_trail_dynamic=b("STRAT_TREND_TRAIL_DYNAMIC", env),
            trend_trail_base_pct=f("STRAT_TREND_TRAIL_BASE_PCT", env),
            trend_trail_min_pct=f("STRAT_TREND_TRAIL_MIN_PCT", env),
            trend_trail_ratchet_k=f("STRAT_TREND_TRAIL_RATCHET_K", env),
            trend_trail_gain_threshold=f("STRAT_TREND_TRAIL_GAIN_THRESH", env),
            fast_profit_guard=b("STRAT_FAST_PROFIT_GUARD", env),
            fast_profit_mult=f("STRAT_FAST_PROFIT_MULT", env),
            fast_profit_window_min=f("STRAT_FAST_PROFIT_WINDOW_MIN", env),
            fast_profit_drop_pct=f("STRAT_FAST_PROFIT_DROP_PCT", env),
            surge_guard=b("STRAT_SURGE_GUARD", env),
            surge_gain_pct=f("STRAT_SURGE_GAIN_PCT", env),
            surge_window_hours=f("STRAT_SURGE_WINDOW_HOURS", env),
            surge_move_pct=f("STRAT_SURGE_MOVE_PCT", env),
            surge_exit_pullback_pct=f("STRAT_SURGE_EXIT_PULLBACK_PCT", env),
            surge_dynamic=b("STRAT_SURGE_DYNAMIC", env),
            surge_min_gain_pct=f("STRAT_SURGE_MIN_GAIN_PCT", env),
            surge_max_gain_pct=f("STRAT_SURGE_MAX_GAIN_PCT", env),
            surge_vol_multiplier=f("STRAT_SURGE_VOL_MULTIPLIER", env),
            slow_grind_guard=b("STRAT_SLOW_GRIND_GUARD", env),
            slow_grind_days=f("STRAT_SLOW_GRIND_DAYS", env),
            slow_grind_min_gain_pct=f("STRAT_SLOW_GRIND_MIN_GAIN_PCT", env),
            slow_grind_flash_window_min=f("STRAT_SLOW_GRIND_FLASH_WINDOW_MIN", env),
            slow_grind_flash_drop_pct=f("STRAT_SLOW_GRIND_FLASH_DROP_PCT", env),
            slow_grind_hourly_drop_pct=f("STRAT_SLOW_GRIND_HOURLY_DROP_PCT", env),
            sl_disaster_vol_multiplier=f("STRAT_SL_DISASTER_VOL_MULTIPLIER", env),
            sl_disaster_buffer_min_pct=f("STRAT_SL_DISASTER_BUFFER_MIN_PCT", env),
            sl_bounce_ratchet_enabled=b("STRAT_SL_BOUNCE_RATCHET_ENABLED", env),
            sl_bounce_ratchet_k=f("STRAT_SL_BOUNCE_RATCHET_K", env),
            sl_bounce_min_pullback_pct=f("STRAT_SL_BOUNCE_MIN_PULLBACK_PCT", env),
        )
    return cfg


def sanitize_common_spot_state(
    state: dict | None,
    new_state_fn: Callable[[], dict],
    *,
    nonneg_float_keys: tuple[str, ...],
    signed_float_keys: tuple[str, ...],
    nonneg_int_keys: tuple[str, ...],
    pos_float_keys: tuple[str, ...],
    bool_keys: tuple[str, ...],
    dict_keys: tuple[str, ...],
    nullable_dict_keys: tuple[str, ...] = (),
    pos_float_list_keys: tuple[str, ...] = (),
    epoch_keys: tuple[str, ...] = (),
    epoch_dict_keys: tuple[str, ...] = (),
    sanitize_orders: bool = True,
    order_id_key: str = "txid",
    lowercase_order_side: bool = True,
    order_pos_float_keys: tuple[str, ...] = ("vol", "price"),
    order_nonneg_float_keys: tuple[str, ...] = ("amount", "ts"),
    order_bool_keys: tuple[str, ...] = ("market", "cancel_requested"),
) -> dict:
    """Sanitize common persisted strategy state keys in-place across spot and T212 engines."""
    defaults = new_state_fn()
    if not isinstance(state, dict):
        state = dict(defaults)
    for k, v in defaults.items():
        state.setdefault(k, v)
    cyc_i = safe_pos_int(state.get("cycle"), default=None)
    if cyc_i is None:
        cyc_f = safe_float(state.get("cycle"), default=None)
        cyc_i = max(1, safe_int(math.trunc(cyc_f), default=1) or 1) if cyc_f is not None else 1
    state["cycle"] = cyc_i
    for nkey in nonneg_float_keys:
        state[nkey] = safe_nonneg_float(state.get(nkey), 0.0) or 0.0
    for fkey in signed_float_keys:
        state[fkey] = safe_float(state.get(fkey), default=0.0) or 0.0
    for ikey in nonneg_int_keys:
        i = safe_int(state.get(ikey), default=None)
        if i is None:
            if_val = safe_float(state.get(ikey), default=None)
            i = safe_int(math.trunc(if_val), default=0) if if_val is not None else 0
        state[ikey] = i if (i is not None and i >= 0) else 0
    for pkey in pos_float_keys:
        if pkey in state and state.get(pkey) is not None:
            state[pkey] = (
                normalize_epoch_sec(state.get(pkey), None)
                if pkey in ("entry_ts", "last_sell_ts")
                else safe_pos_float(state.get(pkey), None)
            )
    for bkey in bool_keys:
        if bkey in state:
            state[bkey] = _parse_bool_flag(state.get(bkey), False)
    for dkey in dict_keys:
        if not isinstance(state.get(dkey), dict):
            state[dkey] = {}
    for ndkey in nullable_dict_keys:
        if not isinstance(state.get(ndkey), dict):
            state[ndkey] = None
    for lkey in pos_float_list_keys:
        raw_lst = state.get(lkey)
        clean_lst: list[float] = []
        if isinstance(raw_lst, list):
            for item in raw_lst:
                val_f = safe_pos_float(item, None)
                if val_f is not None and val_f not in clean_lst:
                    clean_lst.append(val_f)
        state[lkey] = clean_lst
    for ekey in epoch_keys:
        if ekey in state:
            cleaned = normalize_epoch_sec(state.get(ekey), default=None, allow_zero=True)
            if cleaned is not None:
                state[ekey] = cleaned
            else:
                state.pop(ekey, None)
    for edkey in epoch_dict_keys:
        if edkey in state:
            raw_ed = state.get(edkey)
            if isinstance(raw_ed, dict):
                state[edkey] = {
                    str(fk): cv
                    for fk, fv in raw_ed.items()
                    if (cv := normalize_epoch_sec(fv, default=None)) is not None
                }
            else:
                state.pop(edkey, None)
    if sanitize_orders:
        if isinstance(state.get("orders"), list):
            clean_orders = []
            for o in state["orders"]:
                if not isinstance(o, dict):
                    continue
                if isinstance(o.get("side"), bool):
                    continue
                s_norm = normalize_side(o.get("side"))
                if s_norm not in ("BUY", "SELL"):
                    continue
                o["side"] = s_norm.lower() if lowercase_order_side else s_norm
                if order_id_key in o and o.get(order_id_key) is not None:
                    o[order_id_key] = clean_order_id(o.get(order_id_key))
                if "intent_id" in o:
                    o["intent_id"] = clean_order_id(o.get("intent_id"))
                for pk in order_pos_float_keys:
                    if pk in o:
                        o[pk] = safe_pos_float(o.get(pk), 0.0)
                for nk in order_nonneg_float_keys:
                    if nk in o:
                        o[nk] = (
                            normalize_epoch_sec(o.get(nk), 0.0, allow_zero=True)
                            if nk == "ts"
                            else safe_nonneg_float(o.get(nk), 0.0)
                        )
                for bk in order_bool_keys:
                    if bk in o:
                        o[bk] = _parse_bool_flag(o.get(bk), False)
                clean_orders.append(o)
            state["orders"] = clean_orders
        else:
            state["orders"] = []
    pi = state.get("pending_intent")
    if isinstance(pi, dict):
        for id_k in ("order_id", "intent_id"):
            if id_k in pi and pi.get(id_k) is not None:
                pi[id_k] = clean_order_id(pi.get(id_k))
        if "market" in pi:
            pi["market"] = _parse_bool_flag(pi.get("market"), False)
        meta = pi.get("metadata")
        if isinstance(meta, dict) and "market" in meta:
            meta["market"] = _parse_bool_flag(meta.get("market"), False)
    return state


class BaseSpotMixin:
    """Shared venue-agnostic mixin for Spot and Equity DCA + TP/SL engines."""

    venue_label: str = "Strategy"
    _cost_state_key: str = "cost"
    _spent_state_key: str = "spent"
    _pending_submit_state_key: str | None = None
    _qty_eps: float = 1e-12
    _vol_default_source: str = "default"
    _tp_floor_param_attr: str = "tp_trail_profit_floor_pct"
    _supports_fast_profit_guard: bool = True
    _track_stop_sl_low: bool = True
    _include_shadow_in_snapshot: bool = True
    _use_shadow_tick_time_default: bool = True
    _exit_config_ratio_fn: Callable[[float], float] | None = None

    _SHADOW_VOL_CACHE: OrderedDict = OrderedDict()
    _TRAIL_VOL_CACHE: OrderedDict = OrderedDict()
    _HOURLY_VOL_CACHE: OrderedDict = OrderedDict()
    _VOL_CACHE_LOCK: threading.Lock = threading.Lock()
    _REPLAY_CACHE_MAX: int = 16384
    _CACHE_MISS = object()

    @staticmethod
    def _cache_get(cache: OrderedDict, key: tuple) -> Any:
        with BaseSpotMixin._VOL_CACHE_LOCK:
            if key not in cache:
                return BaseSpotMixin._CACHE_MISS
            cache.move_to_end(key)
            return cache[key]

    @staticmethod
    def _cache_store(cache: OrderedDict, key: tuple, val: Any) -> Any:
        with BaseSpotMixin._VOL_CACHE_LOCK:
            lru_dict_put(cache, key, val, max_entries=BaseSpotMixin._REPLAY_CACHE_MAX)
        return val

    def _mod_global(self, name: str, default: Any = None) -> Any:
        """Resolve a module-level symbol from the concrete Strategy's defining module."""
        globs = getattr(getattr(type(self), "__init__", None), "__globals__", None)
        return globs[name] if isinstance(globs, dict) and name in globs else default

    def _log_msg(self, msg: str) -> None:
        self._mod_global("log", log)(msg)

    @property
    def _symbol_id(self) -> str:
        return getattr(self, "pair", None) or getattr(self, "ticker", None) or getattr(self, "coin", "")

    def _now(self) -> float:
        """Return the current timestamp honoring custom clock or simulated time in replay mode."""
        replay = bool(getattr(self, "replay_mode", False))
        if replay:
            sim_ts = getattr(self, "_simulated_ts", None)
            if sim_ts is not None:
                return sim_ts
        clock = getattr(self, "_clock", None)
        if callable(clock) and not (replay and clock is time.time):
            raw_val = clock()
            norm_val = normalize_epoch_sec(raw_val, default=None, allow_zero=True)
            if norm_val is None:
                raise ValueError(f"Invalid Strategy clock timestamp: {raw_val!r}")
            return norm_val
        if replay:
            sp = getattr(self, "_shadow_prices", None)
            if sp:
                return normalize_epoch_sec(sp[-1][0], 0.0, allow_zero=True) or 0.0
            return 0.0
        raw_now = self._mod_global("time", time).time()
        norm_now = normalize_epoch_sec(raw_now, default=None, allow_zero=True)
        return norm_now if norm_now is not None else resolve_now_ts()

    def set_simulated_time(self, ts: float | None) -> None:
        """Explicitly set simulated timestamp for replay and backtesting drivers."""
        self._simulated_ts = normalize_epoch_sec(ts, None, allow_zero=True)

    def _init_common_spot_runtime_state(
        self,
        params: Any,
        *,
        state_file: str,
        initial_state: dict | None = None,
        dry_run: bool = True,
        desktop: bool = False,
        replay_mode: bool = False,
        observe_only: bool = False,
        require_dict_initial_state: bool = True,
        init_exit_manager: bool = True,
        init_shadow_prices: bool = True,
    ) -> None:
        self.p = params
        self.ccy = params.currency
        self.dry_run = _parse_bool_flag(dry_run, True)
        self.observe_only = _parse_bool_flag(observe_only, False)
        self.replay_mode = _parse_bool_flag(replay_mode, False)
        self.desktop = _parse_bool_flag(desktop, False)
        self.state_file = state_file
        self._state_write_failed = False
        self._state_file_mtime: float = 0.0
        has_init_state = (
            isinstance(initial_state, dict)
            if require_dict_initial_state
            else (initial_state is not None)
        )
        if has_init_state:
            self.s = self._sanitize_state_dict(initial_state)
        elif self.replay_mode and not init_exit_manager:
            self.s = self._mod_global("_new_state")()
        else:
            self.s = self._load()
        if init_exit_manager:
            self.exit_manager: PositionExitManager | None = (
                self._init_exit_manager()
                if (getattr(self.p, "sl_mhdsl_enabled", False) or getattr(self.p, "tp_mdhtp_enabled", False))
                else None
            )
        self._paper_seq = 0
        if init_shadow_prices:
            self._shadow_prices = deque(maxlen=90)

    def _next_paper_order_id(self) -> str:
        self._paper_seq += 1
        return f"PAPER-{self._paper_seq}"

    def _reset_cycle_state(self, preserved_keys: tuple[str, ...], **extra_fields: Any) -> int:
        """Reset ``self.s`` via module ``_new_state()``, incrementing ``cycle`` and preserving cumulative keys."""
        preserved = {k: self.s.get(k, 0) for k in preserved_keys}
        next_cycle = (safe_pos_int(self.s.get("cycle"), default=1) or 1) + 1
        self.s = self._mod_global("_new_state")()
        self.s.update(preserved, cycle=next_cycle, **extra_fields)
        return next_cycle

    # -- Persistence -----------------------------------------------------------
    def _store(self) -> JsonStateStore:
        store_cls = self._mod_global("JsonStateStore", JsonStateStore)
        return store_cls(
            self.state_file,
            self._mod_global("_new_state"),
            label=self.venue_label,
            logger=self._mod_global("log", log),
            fail_closed=not self.dry_run,
        )

    def _record_state_file_mtime(self) -> None:
        if getattr(self, "replay_mode", False) or not isinstance(getattr(self, "state_file", None), str):
            return
        try:
            if os.path.exists(self.state_file):
                self._state_file_mtime = safe_nonneg_float(os.path.getmtime(self.state_file), default=0.0) or 0.0
        except OSError:
            pass

    def _load(self) -> dict:
        loaded = self._sanitize_state_dict(self._store().load())
        self._record_state_file_mtime()
        return loaded

    def _save(self) -> None:
        self._state_write_failed = True
        if self._store().save(self.s):
            self._state_write_failed = False
            self._record_state_file_mtime()

    def _recover_state_write_or_skip(self) -> bool:
        """Retry saving after an earlier disk write failure; return False if still failing."""
        if not getattr(self, "_state_write_failed", False):
            return True
        self._save()
        return not self._state_write_failed

    def _step_and_save(self, price: float) -> None:
        self.step(price)
        self._save()

    # -- Position, Fill Accounting & Order Helpers -----------------------------
    def _state_flag(self, key: str, default: bool = False) -> bool:
        """Read a boolean or nullable-dict state slot from ``self.s`` safely (treating ``'false'``/``'0'`` as False)."""
        raw = self.s.get(key) if isinstance(getattr(self, "s", None), dict) else None
        if isinstance(raw, dict):
            return bool(raw)
        return default if raw is None else bool(_parse_bool_flag(raw, default))

    def _avg(self) -> float | None:
        qty = safe_pos_float(self.s.get("qty"), None)
        cost = safe_pos_float(self.s.get(self._cost_state_key), None)
        if qty is None or cost is None or qty <= self._qty_eps:
            return None
        return safe_pos_float(cost / qty, default=None)

    def _qty_for(self, amount: float, price: float) -> float:
        amt_f, px_f = safe_pos_float(amount, default=None), safe_pos_float(price, default=None)
        dec = getattr(self, "vol_dec", getattr(self, "sz_dec", 8))
        return round(amt_f / px_f, dec) if (amt_f is not None and px_f is not None) else 0.0

    def _apply_buy_fill_accounting(
        self, order: dict, qty: float, price: float, *, fallback_to_notional: bool = False
    ) -> None:
        """Update position quantity, cost basis, entry/last buy prices, spent cap, and DCA count."""
        self.s["qty"] += qty
        self.s[self._cost_state_key] += qty * price
        self.s["last_buy_price"] = price
        if self.s["entry_price"] is None:
            self.s["entry_price"] = price
        raw_amt = order.get("amount")
        amt_val = safe_nonneg_float(raw_amt, None)
        self.s[self._spent_state_key] += (
            amt_val
            if (amt_val is not None and raw_amt is not None)
            else ((qty * price) if fallback_to_notional else 0.0)
        )
        if order.get("kind") == "DCA":
            self.s["dca_buys"] += 1

    def _release_sell_cost_basis(self, qty: float, avg: float) -> None:
        """Reduce position quantity and release cost basis proportionally on full or partial SELL."""
        self.s["qty"] -= qty
        self.s[self._cost_state_key] = max(0.0, self.s[self._cost_state_key] - avg * qty)

    def _zero_position_exposure(self, *, clear_spent: bool = False) -> None:
        """Zero out open quantity, cost basis, and optionally spent budget when flat."""
        self.s["qty"] = 0.0
        self.s[self._cost_state_key] = 0.0
        if clear_spent:
            self.s[self._spent_state_key] = 0.0

    def _find_open(self, side: str) -> dict | None:
        return find_open_order(self.s, side)

    def _has_open(self, side: str) -> bool:
        if isinstance(side, bool):
            return False
        if self._pending_submit_state_key:
            side_u = str(side or "").strip().upper()
            if not side_u:
                return False
            raw_pending = self.s.get(self._pending_submit_state_key)
            pending = raw_pending if isinstance(raw_pending, dict) else {}
            return str(pending.get("side") or "").strip().upper() == side_u or self._find_open(side_u) is not None
        return self._find_open(side) is not None

    def _remove(self, o: dict) -> None:
        if o in self.s["orders"]:
            self.s["orders"].remove(o)

    def _mark_order_cancel_requested(self, o: dict, id_key: str) -> bool:
        o["cancel_requested"] = True
        o["cancel_ts"] = self._now()
        self._save()
        self._log_msg(f"  [STRAT] cancel requested {o.get('side', '?')} {o[id_key]} -- waiting for a terminal status")
        return True

    def _cancel_tracked_order(
        self, o: dict, id_key: str, cancel_fn: Callable[[dict], None], *, log_prefix: str = "cancelled"
    ) -> bool:
        if self.dry_run or str(o[id_key]).startswith("PAPER"):
            self._remove(o)
            self._log_msg(f"  [STRAT] {log_prefix} {o.get('side', '?')} {o[id_key]}")
            return True
        if _parse_bool_flag(o.get("cancel_requested"), False):
            return True
        try:
            cancel_fn(o)
        except Exception as exc:
            if not isinstance(exc, self._mod_global("ProviderError", Exception)):
                raise
            self._log_msg(f"  ! [STRAT] cancel failed for {o[id_key]}: {exc} -- the order stays tracked")
            return False
        return self._mark_order_cancel_requested(o, id_key)

    def _cancel_open(self, side: str) -> bool:
        o = self._find_open(side)
        if not o:
            return True
        return self._cancel_order(o)

    def _log_and_cancel_stale_buy(
        self, o: dict, id_key: str, side_label: str, dash: str = "--"
    ) -> None:
        self._log_msg(
            f"  [STRAT] {side_label} {o[id_key]} unfilled, the price rose {dash} cancelling and re-placing"
        )
        self._cancel_order(o)

    def _seed_post_exit_anchors(
        self, price: float, exit_kind: str | None, exit_ts: float, *, include_sl_low: bool = True
    ) -> None:
        """Initialize post-sale price and timestamp anchors after a cycle closes."""
        self.s["last_sell_price"] = price
        self.s["last_exit_kind"] = exit_kind
        if include_sl_low:
            self.s["sl_low"] = price
        self.s["post_sell_peak"] = price
        self.s["post_sell_trough"] = price
        self.s["last_sell_ts"] = exit_ts
        if self.exit_manager is not None:
            self.exit_manager.clear_state(self._symbol_id)
            self.exit_manager.bind_state(self.s)

    def _restore_reentry_state(self) -> None:
        if self.exit_manager is not None:
            self.exit_manager.restore_reentry_state(self._symbol_id)

    def _record_tp_sold_level(self, level: object) -> None:
        lvl_f = safe_pos_float(level, None)
        if lvl_f is not None:
            sold = self.s.setdefault("tp_sold_levels", [])
            if lvl_f not in sold:
                sold.append(lvl_f)

    # -- Percentage & Volatility-Scaled Sizing ---------------------------------
    def _effective_dca_amount(self) -> float:
        """Scale base DCA size by volatility when enabled, falling back during warm-up."""
        base = self.p.effective_dca_amount_base()
        k_val = safe_float(getattr(self.p, "dca_vol_scale_k", 0.0), default=0.0)
        ref_val = safe_pos_float(getattr(self.p, "dca_vol_ref", None), default=None)
        if not k_val or ref_val is None:
            return base
        try:
            vol = safe_pos_float(self._dca_vol_1h(), default=None)
            if vol is None:
                return base
            scale = safe_pos_float((ref_val / vol) ** k_val, default=None)
            if scale is None:
                return base
        except (ArithmeticError, TypeError, ValueError):
            return base
        return base * max(0.3, min(3.0, scale))

    # -- Volatility Tuple & Adaptive Thresholds --------------------------------
    @staticmethod
    def _scaled_log_return_vol_pct(returns: list[float], steps_per_hour: float) -> float | None:
        """Return sample standard deviation of log returns scaled to 1 hour (in %), or None if < 19 samples."""
        n_r = len(returns)
        if n_r < 19 or steps_per_hour <= 0:
            return None
        mean_r = sum(returns) / n_r
        var_r = sum((r - mean_r) ** 2 for r in returns) / (n_r - 1)
        return math.sqrt(var_r) * math.sqrt(steps_per_hour) * 100.0

    @staticmethod
    def _hourly_vol_from_closes(closes: list[float], interval_minutes: float) -> float | None:
        """Return log-return deviation normalized to one hour for a fixed cadence."""
        itv_f = safe_pos_float(interval_minutes, default=None)
        if not closes or itv_f is None:
            return None
        closes_slice = tuple(cf for c in closes[-90:] if (cf := safe_pos_float(c, default=None)) is not None)
        if len(closes_slice) < 20:
            return None
        cache_key = (closes_slice, itv_f)
        cached_val = BaseSpotMixin._cache_get(BaseSpotMixin._HOURLY_VOL_CACHE, cache_key)
        if cached_val is not BaseSpotMixin._CACHE_MISS:
            return cached_val
        returns = [math.log(cur / prev) for prev, cur in zip(closes_slice, closes_slice[1:]) if prev > 0 and cur > 0]
        val = BaseSpotMixin._scaled_log_return_vol_pct(returns, 60.0 / itv_f)
        return BaseSpotMixin._cache_store(BaseSpotMixin._HOURLY_VOL_CACHE, cache_key, val)

    def _shadow_vol_1h(self) -> float | None:
        """Return one-hour volatility from local tick history, or None during warm-up."""
        sp = getattr(self, "_shadow_prices", None)
        if not sp or len(sp) < 20:
            return None
        clean_pts = _iter_valid_shadow_prices(sp)
        n_clean = len(clean_pts)
        if n_clean < 20:
            return None
        mean_dt = (clean_pts[-1][0] - clean_pts[0][0]) / (n_clean - 1)
        if mean_dt <= 0:
            return None
        if getattr(self, "replay_mode", False):
            cache_key = (
                round(clean_pts[-1][0] - clean_pts[0][0], 3),
                tuple(pt[1] for pt in clean_pts),
            )
            if getattr(self, "_shadow_vol_cache_key", None) == cache_key:
                return self._shadow_vol_cache_val
            cached_val = BaseSpotMixin._cache_get(BaseSpotMixin._SHADOW_VOL_CACHE, cache_key)
            if cached_val is not BaseSpotMixin._CACHE_MISS:
                self._shadow_vol_cache_key = cache_key
                self._shadow_vol_cache_val = cached_val
                return cached_val
            rets = [math.log(clean_pts[i][1] / clean_pts[i - 1][1]) for i in range(1, n_clean)]
            val = self._scaled_log_return_vol_pct(rets, 3600.0 / mean_dt)
            if val is not None:
                self._shadow_vol_cache_key = cache_key
                self._shadow_vol_cache_val = val
                self._cache_store(BaseSpotMixin._SHADOW_VOL_CACHE, cache_key, val)
            return val
        rets = [math.log(clean_pts[i][1] / clean_pts[i - 1][1]) for i in range(1, n_clean)]
        return self._scaled_log_return_vol_pct(rets, 3600.0 / mean_dt)

    def _trail_vol_1h(self) -> float | None:
        """Return one-hour-normalized volatility at one OHLC cadence in live and replay."""
        trail_itv = getattr(self.p, "tp_trail_vol_interval", None)
        if getattr(self, "replay_mode", False):
            sp = getattr(self, "_shadow_prices", None)
            if not sp or len(sp) < 20:
                return None
            cache_key = (
                tuple(item[1] for item in sp),
                trail_itv,
            )
            if getattr(self, "_trail_vol_cache_key", None) == cache_key:
                return self._trail_vol_cache_val
            cached_val = BaseSpotMixin._cache_get(BaseSpotMixin._TRAIL_VOL_CACHE, cache_key)
            if cached_val is not BaseSpotMixin._CACHE_MISS:
                self._trail_vol_cache_key = cache_key
                self._trail_vol_cache_val = cached_val
                return cached_val
            val = self._hourly_vol_from_closes([price for _, price in sp], trail_itv)
            self._trail_vol_cache_key = cache_key
            self._trail_vol_cache_val = val
            return self._cache_store(BaseSpotMixin._TRAIL_VOL_CACHE, cache_key, val)
        if not hasattr(self.client, "ohlc_closes") or trail_itv is None:
            return None
        closes = self.client.ohlc_closes(self._symbol_id, trail_itv)
        return self._hourly_vol_from_closes(closes, trail_itv)

    def _dca_vol_1h(self) -> float | None:
        """Return DCA volatility from OHLC at the same cadence in live and replay."""
        dca_itv = getattr(self.p, "dca_vol_interval", None)
        if getattr(self, "replay_mode", False):
            closes = [price for _, price in getattr(self, "_shadow_prices", ())]
        elif hasattr(self.client, "ohlc_closes") and dca_itv is not None:
            closes = self.client.ohlc_closes(self._symbol_id, dca_itv)
        else:
            return None
        return self._hourly_vol_from_closes(closes, dca_itv)

    def _resolve_shadow_vol_1h_pct(self) -> float | None:
        if hasattr(self, "_shadow_vol_1h"):
            try:
                return safe_pos_float(self._shadow_vol_1h(), default=None)
            except Exception:
                return None
        return None

    def _raw_reentry_shadow_vol_pct(self) -> float | None:
        return self._shadow_vol_1h()

    def _local_vol_1h_sources(self) -> tuple[tuple[Callable[[], float | None], str], ...]:
        return ((self._trail_vol_1h, "trail_ohlc"), (self._shadow_vol_1h, "shadow_ticks"))

    def _reentry_pullback_labels(self, sp_len: int) -> tuple[str, str]:
        return f"fixed (fallback, warm-up {sp_len}/20)", "vol_1h"

    def _vol_1h_ratio(self) -> tuple[float, str]:
        """Return 1h volatility as a decimal ratio [e.g. 0.025 for 2.5%] and its source."""
        for fn, src in self._local_vol_1h_sources():
            try:
                vol = safe_pos_float(fn(), default=None)
                if vol is not None:
                    return vol_pct_to_ratio(vol), src
            except Exception:
                pass
        if not self.replay_mode:
            cached = self._mod_global("resolve_cached_volatility_1h", resolve_cached_volatility_1h)(self._symbol_id)
            if cached is not None:
                return cached, "cache"
        return 0.0, self._vol_default_source

    def _effective_reentry_drop_pct(self) -> tuple[float, str]:
        """Return the effective adaptive or fixed reentry threshold."""
        if not getattr(self.p, "reentry_adaptive", False):
            return self.p.reentry_drop_pct, "fix"
        k_re = self.p.shadow_k_reentry
        try:
            vol = self._raw_reentry_shadow_vol_pct()
            if vol is None and not self.replay_mode:
                vol = self._mod_global("resolve_cached_volatility_1h_pct", resolve_cached_volatility_1h_pct)(
                    self._symbol_id
                )
        except Exception as e:  # noqa: BLE001 -- observational failure must not stop trading.
            self._log_msg(f"  [REENTRY-ADAPTIVE] calculation error ({e}) -- fallback to fixed")
            return self.p.reentry_drop_pct, "fix (fallback, error)"
        if vol is None:
            return self.p.reentry_drop_pct, f"fix (fallback, warm-up {len(self._shadow_prices)}/20)"
        adapt_drop, _ = adaptive_thresholds(vol, k_reentry=k_re)
        return (adapt_drop if adapt_drop is not None else k_re * vol), f"adaptive (vol_1h {vol:.2f}% x k={k_re})"

    def _effective_reentry_pullback_pct(self) -> tuple[float, str]:
        """Return the dynamic volatility-scaled or fixed pullback threshold from peak."""
        if not self.p.reentry_pullback_adaptive:
            return self.p.reentry_pullback_pct, "fixed"
        try:
            if (vol := self._raw_reentry_shadow_vol_pct()) is None and not self.replay_mode:
                vol = self._mod_global("resolve_cached_volatility_1h_pct", resolve_cached_volatility_1h_pct)(
                    self._symbol_id
                )
        except Exception as e:  # noqa: BLE001 -- observational failure must not stop trading.
            self._log_msg(f"  [REENTRY-PULLBACK-ADAPTIVE] calculation error ({e}) -- fallback to fixed")
            vol = None
        sp_len = len(getattr(self, "_shadow_prices", None) or ())
        fallback_reason, vol_label = self._reentry_pullback_labels(sp_len)
        return compute_adaptive_reentry_pullback_pct(
            self.p,
            vol,
            fallback_reason=fallback_reason,
            vol_label=vol_label,
        )

    def _effective_trail_pct(self) -> float:
        """Return the clamped volatility-adaptive or fixed trailing pullback."""
        if not getattr(self.p, "tp_trail_adaptive", False):
            return self.p.tp_trail_pct
        try:
            vol = self._trail_vol_1h()
        except Exception as e:  # noqa: BLE001 -- observational failure must not stop trading.
            self._log_msg(f"  [STRAT] adaptive trailing: OHLC unavailable ({e}) -- fallback to fixed")
            return self.p.tp_trail_pct
        if vol is None:
            return self.p.tp_trail_pct
        return max(self.p.tp_trail_min, min(self.p.tp_trail_max, self.p.tp_trail_k * vol))

    # -- Exit Manager, Snapshot & Re-entry Gate --------------------------------
    def _exit_config_overrides(self) -> dict[str, Any]:
        p = self.p
        return dict(
            sl_disaster_vol_multiplier=p.sl_disaster_vol_multiplier,
            sl_disaster_buffer_min_pct=p.sl_disaster_buffer_min_pct / 100.0,
            sl_bounce_ratchet_enabled=p.sl_bounce_ratchet_enabled,
            sl_bounce_ratchet_k=p.sl_bounce_ratchet_k,
            sl_bounce_min_pullback_pct=p.sl_bounce_min_pullback_pct / 100.0,
            tp_trend_ride_enabled=p.tp_trend_hold,
            tp_profit_floor_min_pct=(p.tp_trail_profit_floor_pct or 0.0) / 100.0,
            tp_pullback_pct=p.tp_trail_pct / 100.0,
            fast_profit_guard_enabled=p.fast_profit_guard,
            fast_profit_mult=p.fast_profit_mult,
            fast_profit_window_sec=p.fast_profit_window_min * 60.0,
            fast_profit_drop_pct=p.fast_profit_drop_pct,
            reentry_sl_bounce_pct=p.reentry_sl_bounce_pct / 100.0,
            reentry_bear_bounce_pct=p.reentry_bear_bounce_pct / 100.0,
        )

    def _init_exit_manager(self) -> PositionExitManager:
        return build_strategy_exit_manager(
            self.p,
            self.s,
            lambda _symbol: self._vol_1h_ratio(),
            allow_external_cache=not self.replay_mode,
            ratio_fn=self._exit_config_ratio_fn,
            extra_config=self._exit_config_overrides(),
        )

    def _sellable_qty(self, held: float, ref_price: float | None = None) -> float:
        return held

    def _snapshot_provider_name(self) -> str:
        return self.venue_label

    def _snapshot_trend_up(self, regime: Any) -> bool:
        return self._is_trend_up(regime)

    def _create_position_snapshot(
        self,
        price: float,
        avg: float,
        held: float,
        regime: Any = None,
        include_vol: bool = True,
        include_shadow: bool = True,
        include_trend: bool = True,
        tick_time: float | None = None,
    ) -> PositionSnapshot:
        """Construct a standardized PositionSnapshot using the strategy's current context."""
        sp = getattr(self, "_shadow_prices", None)
        if tick_time is None:
            tick_time = sp[-1][0] if (self._use_shadow_tick_time_default and sp) else self._now()
        vol_1h, vol_source = self._vol_1h_ratio() if include_vol else (0.0, "default")
        trend_up = self._snapshot_trend_up(regime) if include_trend else False
        sell_qty = self._sellable_qty(held, price)
        provider = self._snapshot_provider_name()
        shadow = list(sp) if (self._include_shadow_in_snapshot and include_shadow and sp is not None) else []
        return PositionSnapshot(
            symbol=self._symbol_id,
            current_price=price,
            avg_buy_price=avg,
            net_qty=held,
            sellable_qty=sell_qty,
            current_time_s=tick_time,
            trend_up=trend_up,
            vol_1h=vol_1h,
            vol_source=vol_source,
            provider=provider,
            shadow_prices=shadow,
        )

    def _in_mhdsl_territory(self, loss_pct: float, *, reset_episode_on_exit: bool = True) -> bool:
        """Return True if the position is in MHDSL stop-loss or bounce-watch territory."""
        in_sl = loss_pct > 0.0 and (
            loss_pct >= self.p.stop_loss_pct
            or are_close(loss_pct, self.p.stop_loss_pct, self.p.reentry_tolerance_pct)
        )
        if not in_sl and "sl_pending_exit" in self.s:
            self.s["sl_pending_exit"] = None
        mgr, sym = self.exit_manager, self._symbol_id
        if not in_sl and not (mgr is not None and sym in mgr.sl_bounce_state):
            if reset_episode_on_exit and mgr is not None and mgr.sl_episode_state:
                mgr.reset_sl_episode(sym)
            return False
        return True

    def _handle_mhdsl_hold_decision(self, decision: Any) -> None:
        """Log MHDSL bounce deferral when holding inside stop-loss territory."""
        if decision.metadata.get("priority") in (1, 2) or (
            self.exit_manager is not None and self._symbol_id in self.exit_manager.sl_bounce_state
        ):
            self._log_msg(f"  [SHIELD] [STRAT] MHDSL BOUNCE DEFERRAL: {decision.reason}")

    def _sync_mdhtp_config(self, eff_trail_pct: float, *, trend_ride_enabled: bool | None = None) -> None:
        """Synchronize MDHTP parameters onto ``self.exit_manager.config``."""
        if self.exit_manager is None:
            self.exit_manager = self._init_exit_manager()
        cfg = self.exit_manager.config
        cfg.tp_pullback_pct = eff_trail_pct / 100.0
        cfg.tp_trend_ride_enabled = (
            bool(getattr(self.p, "tp_trend_hold", True)) if trend_ride_enabled is None else trend_ride_enabled
        )
        cfg.tp_profit_floor_min_pct = (getattr(self.p, self._tp_floor_param_attr, None) or 0.0) / 100.0
        cfg.tp_stagnation_max_sec = self.p.tp_stagnation_max_sec
        if self._supports_fast_profit_guard:
            cfg.fast_profit_guard_enabled = self.p.fast_profit_guard
            cfg.fast_profit_mult = self.p.fast_profit_mult
            cfg.fast_profit_window_sec = self.p.fast_profit_window_min * 60.0
            cfg.fast_profit_drop_pct = self.p.fast_profit_drop_pct

    def _pre_reentry_gate_hook(self, price: float) -> tuple[bool, str] | None:
        return None

    def _resolve_reentry_context(
        self,
        mgr: PositionExitManager,
        norm_exit_kind: str,
        is_stop_exit: bool,
        regime: Any,
    ) -> tuple[str | None, bool, bool]:
        last_kind = "STOP" if is_stop_exit else (norm_exit_kind or None)
        mgr.config.reentry_bear_bounce_pct = vol_pct_to_ratio(self.p.reentry_bear_bounce_pct)
        mgr.config.reentry_sl_bounce_pct = vol_pct_to_ratio(self.p.reentry_sl_bounce_pct)
        needs_reentry_regime = bool(
            self.p.trend_overlay
            or self.p.reentry_hybrid_enabled
            or (last_kind == "STOP" and self.p.sl_mhdsl_enabled)
        )
        if regime is None and needs_reentry_regime:
            try:
                regime, _ = self._regime_context()
            except Exception:
                regime = None
        trend_up = self._is_trend_up(regime) if (needs_reentry_regime or self.p.reentry_bear_bounce_pct > 0) else False
        trend_down = self._is_trend_direction(regime, "bear", ("down", "bear", "downtrend")) if needs_reentry_regime else False
        return last_kind, trend_up, trend_down

    def _resolve_reentry_custom_drop(self, is_bull_reentry: bool, last_kind: str | None) -> float | None:
        drop_pct = (
            self._effective_reentry_drop_pct()[0]
            if ((not is_bull_reentry and last_kind != "STOP") or not self.replay_mode)
            else self.p.reentry_drop_pct
        )
        return vol_pct_to_ratio(drop_pct)

    def _evaluate_reentry_gate(
        self, price: float, regime: Any = None, tick_time: float | None = None
    ) -> tuple[bool, str]:
        """Evaluate re-entry gate conditions symmetrically across spot and equity engines."""
        lsp = update_post_sell_anchors(self.s, price, track_stop_sl_low=self._track_stop_sl_low)
        if not lsp:
            return True, "no_last_sell"

        early = self._pre_reentry_gate_hook(price)
        if early is not None:
            return early

        if self.exit_manager is None:
            self.exit_manager = self._init_exit_manager()
        mgr = self.exit_manager
        mgr.config.reentry_drop_pct = vol_pct_to_ratio(self.p.reentry_drop_pct)
        mgr.config.reentry_ttl_hours = self.p.reentry_ttl_hours
        mgr.config.reentry_tolerance_pct = self.p.reentry_tolerance_pct
        mgr.config.reentry_hybrid_enabled = self.p.reentry_hybrid_enabled
        mgr.config.reentry_peak_relative = self.p.reentry_peak_relative

        raw_exit_kind = self.s.get("last_exit_kind")
        norm_exit_kind = "" if isinstance(raw_exit_kind, bool) else str(raw_exit_kind or "").strip().upper()
        is_stop_exit = is_stop_exit_kind(norm_exit_kind)
        last_kind, trend_up, trend_down = self._resolve_reentry_context(
            mgr, norm_exit_kind, is_stop_exit, regime
        )

        is_bull_reentry = bool(
            last_kind != "STOP" and self.p.reentry_hybrid_enabled and (trend_up or self.p.reentry_peak_relative)
        )
        pullback_pct = (
            self._effective_reentry_pullback_pct()[0]
            if (is_bull_reentry or not self.replay_mode)
            else self.p.reentry_pullback_pct
        )
        mgr.config.reentry_pullback_pct = vol_pct_to_ratio(pullback_pct)
        custom_drop = self._resolve_reentry_custom_drop(is_bull_reentry, last_kind)

        vol_1h, vol_source = (0.0, "default") if self.replay_mode else self._vol_1h_ratio()
        reentry_snap = ReentrySnapshot(
            symbol=self._symbol_id,
            current_price=price,
            last_sell_price=lsp,
            last_exit_kind=last_kind,
            post_sell_peak=self.s.get("post_sell_peak"),
            post_sell_trough=self.s.get("post_sell_trough"),
            current_time_s=tick_time if tick_time is not None else self._now(),
            last_sell_time_s=self.s.get("last_sell_ts"),
            trend_up=trend_up,
            trend_down=trend_down,
            vol_1h=vol_1h,
            vol_source=vol_source,
        )
        reentry_dec = (
            mgr.evaluate_reentry(reentry_snap, custom_drop_pct=custom_drop)
            if custom_drop is not None
            else mgr.evaluate_reentry(reentry_snap)
        )
        return reentry_dec.should_enter, reentry_dec.reason

    def _place_entry_order(self, entry_amt: float, entry_px: float) -> bool:
        return self._place("buy", self._qty_for(entry_amt, entry_px), entry_px, kind="ENTRY", amount=entry_amt)

    def _execute_flat_reentry(
        self,
        price: float,
        entry_px: float,
        entry_amt: float,
        budget: float,
        *,
        regime: Any = None,
        tick_time: float | None = None,
        cap_label: str = "cap",
    ) -> bool:
        """Evaluate flat-position re-entry gate and place initial ENTRY buy order."""
        blocked_tick = getattr(self, "_reentry_blocked_tick", None)
        if tick_time is not None and blocked_tick is not None and blocked_tick[:2] == (tick_time, price):
            allowed, reason = False, blocked_tick[2]
        else:
            allowed, reason = self._evaluate_reentry_gate(price, regime, tick_time)
        if not allowed:
            self._log_msg(f"  [STRAT] re-entry blocked: {reason}")
            return False
        if (safe_pos_float(self.s.get("last_sell_price"), 0.0) or 0.0) > 0.0:
            self._log_msg(f"  [STRAT] re-entry approved: {reason} -- placing ENTRY")
        if self.s[self._spent_state_key] + entry_amt > budget:
            self._restore_reentry_state()
            self._log_msg(f"  [STRAT] {cap_label} {budget:.0f} {self.ccy} reached -- not entering")
            return False
        placed = self._place_entry_order(entry_amt, entry_px)
        if not placed:
            self._restore_reentry_state()
        return placed

    def _touch_strategy_heartbeat(self, price: float | None, sleep_sec: float, *, force: bool = False) -> None:
        if getattr(self, "replay_mode", False):
            return
        sym = str(self._symbol_id or "").strip().upper()
        hb_name = MarketDataProvider.resolve_venue_strategy_heartbeat(
            getattr(self, "venue_label", ""), sym
        )
        touch_module_heartbeat(
            hb_name,
            interval_sec=30.0,
            force=force,
            status={"price": price, "symbol": sym},
            symbols=[sym] if sym else None,
            intervals={"check_sec": sleep_sec},
        )

    def _resolve_stop_loss_avg_and_loss_pct(
        self, price: float, *, sign: int = 1
    ) -> tuple[float, float] | None:
        """Return ``(avg_entry_price, unrealized_loss_pct)`` when stop-loss is enabled and position/price are valid."""
        sl_pct = safe_pos_float(getattr(self.p, "stop_loss_pct", None), default=None)
        px_f = safe_pos_float(price, default=None)
        avg_f = safe_pos_float(self._avg(), default=None)
        if sl_pct is None or px_f is None or avg_f is None:
            return None
        eff_sign = -1 if sign == -1 else 1
        return avg_f, (-eff_sign * (px_f - avg_f) / avg_f * 100.0)

    def _clear_state_slots(self, *keys: str) -> None:
        """Reset each state key in ``keys`` to ``None`` in ``self.s``."""
        for k in keys:
            self.s[k] = None

    def _persist_pending_state_slot(
        self,
        key: str,
        pending: dict | None,
        *,
        require_write: bool = False,
        error_cls: type[Exception] = RuntimeError,
    ) -> None:
        """Store ``pending`` (or ``None``) in ``self.s[key]`` and persist strategy state."""
        self.s[key] = None if pending is None else dict(pending)
        self._save()
        if _parse_bool_flag(require_write, False) and _parse_bool_flag(
            getattr(self, "_state_write_failed", False), False
        ):
            raise error_cls("the intent state could not be persisted")

    def _resolve_run_mode_and_banner(
        self,
        *,
        real_banner: str = "[WARN] REAL -- MONEY",
        ai_banner: str = "[OBSERVE ONLY -- AI MANAGED]",
    ) -> tuple[str, str]:
        """Return ``(mode_label, exec_banner)`` for strategy startup logging."""
        mode = "avg_tp" if _parse_bool_flag(getattr(self.p, "enable_takeprofit", False), False) else "dca_only"
        exec_banner = (
            ai_banner
            if _parse_bool_flag(getattr(self, "observe_only", False), False)
            else ("[PAPER]" if _parse_bool_flag(getattr(self, "dry_run", True), True) else real_banner)
        )
        return mode, exec_banner

    def _run_strategy_loop(
        self,
        *,
        price_fn: Callable[[], float | None],
        status_log_fn: Callable[[float], None],
        stop: Any = None,
        reload_external: bool = False,
    ) -> None:
        """Shared self-healing execution loop for spot and T212 strategies."""
        check_min = safe_pos_float(self.p.check_minutes, default=None)
        sleep_sec = max(1.0, check_min * 60.0) if check_min is not None else 60.0
        sleep_fn = self._mod_global("time", time).sleep
        last_price: list[float | None] = [None]
        self._touch_strategy_heartbeat(None, sleep_sec, force=True)
        sigterm_ev = threading.Event()
        prev_sigterm = None
        if stop is None and threading.current_thread() is threading.main_thread():
            try:
                prev_sigterm = signal.getsignal(signal.SIGTERM)
                signal.signal(
                    signal.SIGTERM,
                    lambda signum, frame: (
                        sigterm_ev.set(),
                        prev_sigterm(signum, frame) if callable(prev_sigterm) else None,
                    ),
                )
            except (ValueError, OSError):
                prev_sigterm = None

        def _delay() -> None:
            self._touch_strategy_heartbeat(last_price[0], sleep_sec)
            if stop is not None:
                stop.wait(sleep_sec)
            elif sleep_fn is time.sleep:
                sigterm_ev.wait(sleep_sec)
            else:
                sleep_fn(sleep_sec)

        try:
            while not ((stop is not None and stop.is_set()) or sigterm_ev.is_set()):
                price = safe_pos_float(price_fn(), default=None)
                last_price[0] = price
                if price is None:
                    self._log_msg("  [STRAT] price unavailable -- retrying")
                    _delay()
                    continue
                try:
                    if not self._recover_state_write_or_skip():
                        _delay()
                        continue
                    if (
                        reload_external
                        and not getattr(self, "replay_mode", False)
                        and not getattr(self, "_state_write_failed", False)
                        and isinstance(getattr(self, "state_file", None), str)
                    ):
                        try:
                            if os.path.exists(self.state_file):
                                mtime_val = safe_nonneg_float(os.path.getmtime(self.state_file), default=0.0) or 0.0
                                if mtime_val > getattr(self, "_state_file_mtime", 0.0) + 1e-3:
                                    self.s = self._load()
                        except OSError:
                            pass
                    if self.reconcile(price) is False:
                        _delay()
                        continue
                    self._step_and_save(price)
                except Exception as e:  # noqa: BLE001 -- RESILIENCE: network/API down -> retry
                    self._log_msg(f"  ! [STRAT] error ({e.__class__.__name__}: {e}) -- retrying")
                    _delay()
                    continue
                status_log_fn(price)
                _delay()
            if getattr(self, "_state_write_failed", False):
                self._save()
        except KeyboardInterrupt:
            self._log_msg("  [STRAT] stopped manually.")
            self._save()
        finally:
            if prev_sigterm is not None:
                try:
                    signal.signal(signal.SIGTERM, prev_sigterm)
                except (ValueError, OSError):
                    pass

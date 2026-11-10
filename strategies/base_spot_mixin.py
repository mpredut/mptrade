"""Shared spot strategy mixin for reentry gate, position snapshot, volatility tuple, and exit manager wiring.

Unifies state persistence, open-order lookup, fill accounting, 1-hour volatility resolution,
adaptive pullback/drop calculation, PositionSnapshot construction, MHDSL stop-loss
coordination, MDHTP take-profit configuration, and ReentrySnapshot evaluation across
``strategies/spot_engine.py`` and ``212trading/strategy.py``.
"""

from __future__ import annotations

import math
import os
import time
from typing import Any, Callable

from botcore import (
    _parse_bool_flag, are_close, exact_int, exact_nonneg_int, exact_pos_int, log,
    normalize_epoch_sec, optional_env,
    required_bool_env, required_env, required_float_env, required_int_env,
    safe_float, safe_int, safe_nonneg_float, safe_pos_float, safe_pos_int,
    touch_module_heartbeat,
)
from intelligence.internal.state.volatility import (
    adaptive_thresholds, resolve_cached_volatility_1h,
    resolve_cached_volatility_1h_pct, vol_pct_to_ratio,
)
from providers.base import normalize_side
from providers.order_ohlc_utils import clean_order_id
from .position_exit_manager import PositionExitManager, PositionSnapshot, ReentrySnapshot
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
        weights = [float(steps - i) for i in range(steps)]
    elif curve_lower == "back_heavy":
        weights = [float(i + 1) for i in range(steps)]
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
    min_q = safe_pos_float(min_tranche_qty, 0.01)
    max_viable_rungs = max(1, int(held_f / max(1e-6, min_q)))
    if len(remaining) <= max_viable_rungs:
        return list(remaining)
    if max_viable_rungs == 1:
        return [remaining[-1]]
    indices = [int(round(i * (len(remaining) - 1) / (max_viable_rungs - 1))) for i in range(max_viable_rungs)]
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
    mode = required_env("STRATEGY_MODE", env).strip().lower()
    if mode not in {"avg_tp", "dca_only"}:
        raise ValueError(f"Invalid STRATEGY_MODE: {mode!r}")
    raw_ladder = optional_env("STRAT_TP_LADDER", env=env)
    ladder_curve = optional_env("STRAT_TP_LADDER_CURVE", "linear", env=env).lower()
    if raw_ladder:
        clean_raw = raw_ladder.removeprefix("range:")
        if clean_raw.count(":") == 2 and "," not in clean_raw:
            try:
                p_min, p_max, p_steps = clean_raw.split(":")
                steps_i = exact_pos_int(p_steps, default=None)
                min_f = safe_float(p_min, default=None)
                max_f = safe_float(p_max, default=None)
                if steps_i is None or min_f is None or max_f is None:
                    raise ValueError("non-integer or non-finite steps")
                return generate_ladder_fn(min_f, max_f, steps_i, curve=ladder_curve)
            except (ValueError, OverflowError) as exc:
                raise ValueError(f"Invalid STRAT_TP_LADDER: {raw_ladder!r}") from exc
        ladder: list[tuple[float, float]] = []
        for part in raw_ladder.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" not in part:
                raise ValueError(f"Invalid STRAT_TP_LADDER: {raw_ladder!r}")
            lvl_s, frac_s = part.split(":", 1)
            lvl_f = safe_pos_float(lvl_s, default=None)
            frac_raw = safe_pos_float(frac_s, default=None)
            if lvl_f is None or frac_raw is None:
                raise ValueError(f"Invalid STRAT_TP_LADDER: {raw_ladder!r}")
            ladder.append((lvl_f, frac_raw / 100.0))
        if sum(f for _, f in ladder) > 1.0 + 1e-6:
            raise ValueError(f"Invalid STRAT_TP_LADDER (fractions exceed 100%): {raw_ladder!r}")
        return ladder
    if optional_env("STRAT_TP_LADDER_MIN", env=env) and optional_env("STRAT_TP_LADDER_MAX", env=env):
        return generate_ladder_fn(
            required_float_env("STRAT_TP_LADDER_MIN", env),
            required_float_env("STRAT_TP_LADDER_MAX", env),
            required_int_env("STRAT_TP_LADDER_STEPS", env),
            curve=ladder_curve,
        )
    return []


def parse_max_dca_buys_env(
    env: dict | None, budget: float, entry_amt: float, dca_amt: float
) -> tuple[int, float]:
    """Parse ``STRAT_MAX_DCA_BUYS`` (supporting ``auto``/``budget`` or explicit non-negative integer)."""
    mdb = optional_env("STRAT_MAX_DCA_BUYS", env=env).lower()
    if mdb in ("auto", "buget", "budget"):
        max_dca = max(0, int((budget - entry_amt) // dca_amt)) if dca_amt > 0 else 0
        return max_dca, budget - max_dca * dca_amt
    if mdb:
        max_dca_i = exact_nonneg_int(mdb, default=None)
        if max_dca_i is None:
            raise ValueError(f"Invalid STRAT_MAX_DCA_BUYS: {mdb!r}")
        return max_dca_i, entry_amt
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
        cyc_i = max(1, int(cyc_f)) if cyc_f is not None else 1
    state["cycle"] = cyc_i
    for nkey in nonneg_float_keys:
        state[nkey] = safe_nonneg_float(state.get(nkey), 0.0) or 0.0
    for fkey in signed_float_keys:
        state[fkey] = safe_float(state.get(fkey), default=0.0) or 0.0
    for ikey in nonneg_int_keys:
        i = safe_int(state.get(ikey), default=None)
        if i is None:
            if_val = safe_float(state.get(ikey), default=None)
            i = int(if_val) if if_val is not None else 0
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
                if order_id_key in o and o.get(order_id_key) is not None:
                    o[order_id_key] = clean_order_id(o.get(order_id_key))
                if "side" in o and not isinstance(o.get("side"), bool):
                    s_norm = normalize_side(o.get("side"))
                    if s_norm in ("BUY", "SELL"):
                        o["side"] = s_norm.lower() if lowercase_order_side else s_norm
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
    """Shared mixin for Spot (Kraken/Binance/HL) and Equity (T212) DCA + TP/SL engines."""

    _exit_manager_mode: str = "spot"
    _cost_state_key: str = "cost"
    _spent_state_key: str = "spent"
    _qty_eps: float = 1e-12
    _vol_default_source: str = "default"

    _SHADOW_VOL_CACHE: dict = {}
    _TRAIL_VOL_CACHE: dict = {}
    _HOURLY_VOL_CACHE: dict = {}
    _REPLAY_CACHE_MAX: int = 262144

    @staticmethod
    def _cache_store(cache: dict, key: tuple, val: Any) -> Any:
        if len(cache) >= BaseSpotMixin._REPLAY_CACHE_MAX:
            cache.clear()
        cache[key] = val
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
        clock = getattr(self, "_clock", None)
        if callable(clock):
            val = float(clock())
            return normalize_epoch_sec(val, val, allow_zero=True) or val
        if getattr(self, "replay_mode", False):
            sim_ts = getattr(self, "_simulated_ts", None)
            if sim_ts is not None:
                return sim_ts
            sp = getattr(self, "_shadow_prices", None)
            if sp:
                return normalize_epoch_sec(sp[-1][0], 0.0, allow_zero=True) or 0.0
            return 0.0
        val = float(self._mod_global("time", time).time())
        return normalize_epoch_sec(val, val, allow_zero=True) or val

    def set_simulated_time(self, ts: float | None) -> None:
        """Explicitly set simulated timestamp for replay and backtesting drivers."""
        self._simulated_ts = normalize_epoch_sec(ts, None, allow_zero=True)

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
            label=getattr(self, "venue_label", "T212"),
            logger=self._mod_global("log", log),
            fail_closed=not self.dry_run,
        )

    def _record_state_file_mtime(self) -> None:
        if getattr(self, "replay_mode", False) or not isinstance(getattr(self, "state_file", None), str):
            return
        try:
            if os.path.exists(self.state_file):
                self._state_file_mtime = float(os.path.getmtime(self.state_file))
        except OSError:
            pass

    def _reload_if_externally_modified(self) -> None:
        if (
            getattr(self, "replay_mode", False)
            or getattr(self, "_state_write_failed", False)
            or not isinstance(getattr(self, "state_file", None), str)
        ):
            return
        try:
            if os.path.exists(self.state_file):
                if float(os.path.getmtime(self.state_file)) > getattr(self, "_state_file_mtime", 0.0) + 1e-3:
                    self.s = self._load()
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

    # -- Position, Fill Accounting & Order Helpers -----------------------------
    def _avg(self) -> float | None:
        qty = safe_pos_float(self.s.get("qty"), None)
        cost = safe_pos_float(self.s.get(self._cost_state_key), None)
        if qty is None or cost is None or qty <= self._qty_eps:
            return None
        return safe_pos_float(cost / qty, default=None) if self._exit_manager_mode == "t212" else cost / qty

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

    def _find_open(self, side: str) -> dict | None:
        return find_open_order(self.s, side)

    def _has_open(self, side: str) -> bool:
        if self._exit_manager_mode == "t212":
            if isinstance(side, bool):
                return False
            side_u = str(side or "").strip().upper()
            if not side_u:
                return False
            raw_pending = self.s.get("pending_submit")
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
        self._log_msg(f"  [STRAT] cancel requested {o.get('side', '?')} {o[id_key]} — waiting for a terminal status")
        return True

    def _cancel_tracked_order(
        self, o: dict, id_key: str, cancel_fn: Callable[[dict], None], *, log_prefix: str = "cancelled"
    ) -> bool:
        if self.dry_run or str(o[id_key]).startswith("PAPER"):
            self._remove(o)
            self._log_msg(f"  [STRAT] {log_prefix} {o.get('side', '?')} {o[id_key]}")
            return True
        if o.get("cancel_requested"):
            return True
        try:
            cancel_fn(o)
        except Exception as exc:
            if not isinstance(exc, self._mod_global("ProviderError", Exception)):
                raise
            self._log_msg(f"  ! [STRAT] cancel failed for {o[id_key]}: {exc} — the order stays tracked")
            return False
        return self._mark_order_cancel_requested(o, id_key)

    def _cancel_open(self, side: str) -> bool:
        o = self._find_open(side)
        if not o:
            return True
        return self._cancel_specific(o) if self._exit_manager_mode == "t212" else self._cancel_order(o)

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

    def _all_tp_ladder_levels_sold(self) -> bool:
        if not getattr(self.p, "tp_ladder", None):
            return False
        ladder_levels = {lvl for (lvl, _) in self.p.tp_ladder}
        return bool(ladder_levels and ladder_levels.issubset(set(self.s.get("tp_sold_levels", []))))

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
        if cache_key in BaseSpotMixin._HOURLY_VOL_CACHE:
            return BaseSpotMixin._HOURLY_VOL_CACHE[cache_key]
        returns = [math.log(cur / prev) for prev, cur in zip(closes_slice, closes_slice[1:]) if prev > 0 and cur > 0]
        val = BaseSpotMixin._scaled_log_return_vol_pct(returns, 60.0 / itv_f)
        return BaseSpotMixin._cache_store(BaseSpotMixin._HOURLY_VOL_CACHE, cache_key, val)

    def _shadow_vol_1h(self) -> float | None:
        """Return one-hour volatility from local tick history, or None during warm-up."""
        sp = getattr(self, "_shadow_prices", None)
        if not sp or (n_sp := len(sp)) < 20:
            return None
        t0_raw, tn_raw = sp[0][0], sp[-1][0]
        if (
            self.replay_mode
            and isinstance(t0_raw, (int, float))
            and isinstance(tn_raw, (int, float))
            and -1e11 < t0_raw <= 1e11
            and -1e11 < tn_raw <= 1e11
            and tn_raw > t0_raw
        ):
            cache_key = (
                n_sp, round(tn_raw - t0_raw, 3),
                sp[0][1], sp[1][1], sp[n_sp >> 1][1], sp[-2][1], sp[-1][1],
            )
            if getattr(self, "_shadow_vol_cache_key", None) == cache_key:
                return self._shadow_vol_cache_val
            if cache_key in BaseSpotMixin._SHADOW_VOL_CACHE:
                self._shadow_vol_cache_key = cache_key
                self._shadow_vol_cache_val = val = BaseSpotMixin._SHADOW_VOL_CACHE[cache_key]
                return val
            mean_dt = (tn_raw - t0_raw) / (n_sp - 1)
            pts = list(sp)
            rets = [
                math.log(pts[i][1] / pts[i - 1][1])
                for i in range(1, n_sp)
                if 0.0 < pts[i - 1][1] < float("inf") and 0.0 < pts[i][1] < float("inf")
            ]
            val = self._scaled_log_return_vol_pct(rets, 3600.0 / mean_dt)
            if val is not None:
                self._shadow_vol_cache_key = cache_key
                self._shadow_vol_cache_val = val
                self._cache_store(BaseSpotMixin._SHADOW_VOL_CACHE, cache_key, val)
            return val
        clean_pts = []
        for item in sp:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                tf = safe_float(item[0], default=None)
                pf = safe_pos_float(item[1], default=None)
                if tf is not None and pf is not None:
                    if tf >= 1e11:
                        tf /= 1000.0
                    clean_pts.append((tf, pf))
        n_clean = len(clean_pts)
        if n_clean < 20:
            return None
        if any(clean_pts[i][0] > clean_pts[i + 1][0] for i in range(n_clean - 1)):
            clean_pts.sort(key=lambda x: x[0])
        mean_dt = (clean_pts[-1][0] - clean_pts[0][0]) / (n_clean - 1)
        if mean_dt <= 0:
            return None
        rets = [math.log(clean_pts[i][1] / clean_pts[i - 1][1]) for i in range(1, n_clean)]
        return self._scaled_log_return_vol_pct(rets, 3600.0 / mean_dt)

    def _trail_vol_1h(self) -> float | None:
        """Return one-hour-normalized volatility at one OHLC cadence in live and replay."""
        trail_itv = getattr(self.p, "tp_trail_vol_interval", None)
        if getattr(self, "replay_mode", False):
            sp = getattr(self, "_shadow_prices", None)
            if not sp or (n_sp := len(sp)) < 20:
                return None
            cache_key = (
                n_sp, sp[0][1], sp[1][1], sp[n_sp >> 1][1], sp[-2][1], sp[-1][1], trail_itv,
            )
            if getattr(self, "_trail_vol_cache_key", None) == cache_key:
                return self._trail_vol_cache_val
            if cache_key in BaseSpotMixin._TRAIL_VOL_CACHE:
                self._trail_vol_cache_key = cache_key
                self._trail_vol_cache_val = val = BaseSpotMixin._TRAIL_VOL_CACHE[cache_key]
                return val
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

    def _resolve_reentry_vol_1h_pct(self) -> float | None:
        """Resolve 1-hour volatility (%) from shadow ticks with non-replay cache fallback."""
        vol = self._shadow_vol_1h() if self._exit_manager_mode == "spot" else self._resolve_shadow_vol_1h_pct()
        if vol is None and not self.replay_mode:
            vol = self._mod_global("resolve_cached_volatility_1h_pct", resolve_cached_volatility_1h_pct)(self._symbol_id)
        return vol

    def _vol_1h_ratio(self) -> tuple[float, str]:
        """Return 1h volatility as a decimal ratio [e.g. 0.025 for 2.5%] and its source."""
        if self._exit_manager_mode == "spot":
            for fn, src in ((self._trail_vol_1h, "trail_ohlc"), (self._shadow_vol_1h, "shadow_ticks")):
                try:
                    vol = safe_pos_float(fn(), default=None)
                    if vol is not None:
                        return vol_pct_to_ratio(vol), src
                except Exception:
                    pass
        else:
            vol = self._resolve_shadow_vol_1h_pct()
            if vol is not None:
                return vol_pct_to_ratio(vol), "shadow"
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
            vol = self._resolve_reentry_vol_1h_pct()
        except Exception as e:  # noqa: BLE001 — observational failure must not stop trading.
            self._log_msg(f"  [REENTRY-ADAPTIVE] calculation error ({e}) — fallback to fixed")
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
            vol = self._resolve_reentry_vol_1h_pct()
        except Exception as e:
            self._log_msg(f"  [REENTRY-PULLBACK-ADAPTIVE] calculation error ({e}) — fallback to fixed")
            vol = None
        is_spot = self._exit_manager_mode == "spot"
        sp_len = len(getattr(self, "_shadow_prices", None) or ())
        return compute_adaptive_reentry_pullback_pct(
            self.p,
            vol,
            fallback_reason=f"fixed (fallback, warm-up {sp_len}/20)" if is_spot else "fixed (fallback)",
            vol_label="vol_1h" if is_spot else "vol",
        )

    def _effective_trail_pct(self) -> float:
        """Return the clamped volatility-adaptive or fixed trailing pullback."""
        if not getattr(self.p, "tp_trail_adaptive", False):
            return self.p.tp_trail_pct
        try:
            vol = self._trail_vol_1h()
        except Exception as e:  # noqa: BLE001 — observational failure must not stop trading.
            self._log_msg(f"  [STRAT] adaptive trailing: OHLC unavailable ({e}) — fallback to fixed")
            vol = None
        if vol is None:
            return self.p.tp_trail_pct
        return max(self.p.tp_trail_min, min(self.p.tp_trail_max, self.p.tp_trail_k * vol))

    # -- Exit Manager, Snapshot & Re-entry Gate --------------------------------
    def _init_exit_manager(self) -> PositionExitManager:
        return build_strategy_exit_manager(
            self.p, self.s, lambda _symbol: self._vol_1h_ratio(),
            allow_external_cache=not self.replay_mode, mode=self._exit_manager_mode,
        )

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
        is_spot = self._exit_manager_mode == "spot"
        sp = getattr(self, "_shadow_prices", None)
        if tick_time is None:
            tick_time = sp[-1][0] if (is_spot and sp) else self._now()
        vol_1h, vol_source = self._vol_1h_ratio() if include_vol else (0.0, "default")
        if is_spot:
            trend_up = self._is_trend_up(regime) if include_trend else False
            sell_qty, provider = held, getattr(self, "venue_label", "Kraken")
            shadow = list(sp) if (include_shadow and sp is not None) else []
        else:
            trend_up = False
            slope_fn = getattr(self, "_trend_slope_provider", None)
            if include_trend and slope_fn:
                try:
                    if safe_nonneg_float(slope_fn(self.yahoo_sym), default=None) is not None:
                        trend_up = True
                except Exception:
                    pass
            sell_qty, provider, shadow = self._sellable_qty(held, price), "t212", []
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
            self._log_msg(f"  🛡️ [STRAT] MHDSL BOUNCE DEFERRAL: {decision.reason}")

    def _sync_mdhtp_config(self, eff_trail_pct: float, *, trend_ride_enabled: bool | None = None) -> None:
        """Synchronize MDHTP parameters onto ``self.exit_manager.config``."""
        if self.exit_manager is None:
            self.exit_manager = self._init_exit_manager()
        cfg = self.exit_manager.config
        cfg.tp_pullback_pct = eff_trail_pct / 100.0
        cfg.tp_trend_ride_enabled = (
            bool(getattr(self.p, "tp_trend_hold", True)) if trend_ride_enabled is None else trend_ride_enabled
        )
        floor_attr = "tp_trail_profit_floor_pct" if self._exit_manager_mode == "spot" else "trail_min_profit_pct"
        cfg.tp_profit_floor_min_pct = (getattr(self.p, floor_attr, None) or 0.0) / 100.0
        cfg.tp_stagnation_max_sec = self.p.tp_stagnation_max_sec
        if self._exit_manager_mode == "spot":
            cfg.fast_profit_guard_enabled = self.p.fast_profit_guard
            cfg.fast_profit_mult = self.p.fast_profit_mult
            cfg.fast_profit_window_sec = self.p.fast_profit_window_min * 60.0
            cfg.fast_profit_drop_pct = self.p.fast_profit_drop_pct

    def _evaluate_reentry_gate(
        self, price: float, regime: Any = None, tick_time: float | None = None
    ) -> tuple[bool, str]:
        """Evaluate re-entry gate conditions symmetrically across spot and equity engines."""
        is_spot = self._exit_manager_mode == "spot"
        lsp = update_post_sell_anchors(self.s, price, track_stop_sl_low=is_spot)
        if not lsp:
            return True, "no_last_sell"

        if not is_spot:
            last_sell_ts = normalize_epoch_sec(self.s.get("last_sell_ts"), None)
            if self.p.reentry_ttl_hours > 0 and last_sell_ts:
                if (self._now() - last_sell_ts) >= self.p.reentry_ttl_hours * 3600.0:
                    self.s.pop("sl_rebuy", None)
            if self.s.get("sl_rebuy"):
                self._handle_sl_rebuy(price)
                return False, "handling_sl_rebuy"

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
        is_stop_exit = norm_exit_kind in {"STOP", "SL", "STOP_LOSS", "AI_STOP_LOSS"}
        if is_spot:
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
        else:
            last_kind = "STOP" if is_stop_exit else "TP"
            mgr.config.reentry_sl_bounce_pct = (
                vol_pct_to_ratio(self.p.sl_rebuy_bounce_pct)
                if self.p.sl_rebuy_enabled
                else (0.015 if self.p.sl_mhdsl_enabled else 0.0)
            )
            mgr.config.reentry_bear_gate_enabled = False
            mgr.config.reentry_bear_bounce_pct = 0.0
            trend_up, trend_down = bool(self.p.reentry_peak_relative), False
            slope_fn = getattr(self, "_trend_slope_provider", None)
            if not trend_up and slope_fn:
                try:
                    slope_f = safe_float(slope_fn(self.yahoo_sym), default=None)
                    if slope_f is not None:
                        if slope_f >= 0.0:
                            trend_up = True
                        elif slope_f < (-self.p.dca_trend_gate_pct if self.p.dca_trend_gate_pct > 0 else -0.1):
                            trend_down = True
                except Exception:
                    pass

        is_bull_reentry = bool(
            last_kind != "STOP" and self.p.reentry_hybrid_enabled and (trend_up or self.p.reentry_peak_relative)
        )
        pullback_pct = (
            self._effective_reentry_pullback_pct()[0]
            if (is_bull_reentry or not self.replay_mode)
            else self.p.reentry_pullback_pct
        )
        mgr.config.reentry_pullback_pct = vol_pct_to_ratio(pullback_pct)

        custom_drop = None
        if is_spot:
            drop_pct = (
                self._effective_reentry_drop_pct()[0]
                if ((not is_bull_reentry and last_kind != "STOP") or not self.replay_mode)
                else self.p.reentry_drop_pct
            )
            custom_drop = vol_pct_to_ratio(drop_pct)

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
        is_spot = self._exit_manager_mode == "spot"
        blocked_tick = getattr(self, "_reentry_blocked_tick", None)
        if blocked_tick is not None and blocked_tick[:2] == (tick_time, price):
            allowed, reason = False, blocked_tick[2]
        else:
            allowed, reason = self._evaluate_reentry_gate(price, regime, tick_time)
        if not allowed:
            self._log_msg(f"  [STRAT] re-entry blocked: {reason}")
            return False
        if self.s.get("last_sell_price"):
            self._log_msg(f"  [STRAT] re-entry approved: {reason} — placing ENTRY")
        if self.s[self._spent_state_key] + entry_amt > budget:
            self._restore_reentry_state()
            self._log_msg(f"  [STRAT] {cap_label} {budget:.0f} {self.ccy} reached — not entering")
            return False
        placed = (
            self._place("buy", self._qty_for(entry_amt, entry_px), entry_px, kind="ENTRY", amount=entry_amt)
            if is_spot
            else self._place_buy(entry_amt, entry_px, kind="ENTRY")
        )
        if not placed:
            self._restore_reentry_state()
        return placed

    def _touch_strategy_heartbeat(self, price: float | None, sleep_sec: float, *, force: bool = False) -> None:
        if getattr(self, "replay_mode", False):
            return
        venue = str(getattr(self, "venue_label", "") or "").strip().lower()
        sym = str(self._symbol_id or "").strip().upper()
        if venue == "hyperliquid":
            hb_name = "HL-bot"
        elif venue == "kraken":
            hb_name = "Kraken-HYPE" if ("HYPE" in sym or not sym) else f"Kraken-{sym}"
        else:
            hb_name = "T212-bot"
        touch_module_heartbeat(
            hb_name,
            interval_sec=30.0,
            force=force,
            status={"price": price, "symbol": sym},
            symbols=[sym] if sym else None,
            intervals={"check_sec": sleep_sec},
        )

    def _run_strategy_loop(
        self,
        *,
        price_fn: Callable[[], float | None],
        status_log_fn: Callable[[float], None],
        stop: Any = None,
        reload_external: bool = False,
    ) -> None:
        """Shared self-healing execution loop for spot and T212 strategies."""
        import signal
        import threading

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

                def _on_sigterm(signum, frame):
                    sigterm_ev.set()
                    if callable(prev_sigterm):
                        prev_sigterm(signum, frame)

                signal.signal(signal.SIGTERM, _on_sigterm)
            except (ValueError, OSError):
                prev_sigterm = None

        def _is_stopped() -> bool:
            return bool((stop is not None and stop.is_set()) or sigterm_ev.is_set())

        def _delay() -> None:
            self._touch_strategy_heartbeat(last_price[0], sleep_sec)
            if stop is not None:
                stop.wait(sleep_sec)
            elif sleep_fn is time.sleep:
                sigterm_ev.wait(sleep_sec)
            else:
                sleep_fn(sleep_sec)

        try:
            while not _is_stopped():
                price = safe_pos_float(price_fn(), default=None)
                last_price[0] = price
                if price is None:
                    self._log_msg("  [STRAT] price unavailable — retrying")
                    _delay()
                    continue
                try:
                    if not self._recover_state_write_or_skip():
                        _delay()
                        continue
                    if reload_external:
                        self._reload_if_externally_modified()
                    if self.reconcile(price) is False:
                        _delay()
                        continue
                    self.step(price)
                    self._save()
                except Exception as e:  # noqa: BLE001 — RESILIENCE: network/API down -> retry
                    self._log_msg(f"  ! [STRAT] error ({e.__class__.__name__}: {e}) — retrying")
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

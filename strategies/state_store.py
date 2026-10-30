"""Shared JSON persistence for financial-engine state.

Writes are atomic within the same directory. Live runtimes may fail closed so
corrupt or unsavable state is not mistaken for empty state that could permit a
duplicate entry.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import fields as _dc_fields
from botcore import _parse_bool_flag, safe_nonneg_float, safe_pos_float
from intelligence.internal.state.volatility import vol_pct_to_ratio
from state_io import StateReadError, load_json_state, save_json_state
from .position_exit_manager import PositionExitConfig, PositionExitManager


def clean_pos_float(val: object, default: float | None = None) -> float | None:
    return safe_pos_float(val, default=default)


def clean_nonneg_float(val: object, default: float | None = None) -> float | None:
    return safe_nonneg_float(val, default=default)


def find_open_order(state: dict, side: str) -> dict | None:
    """Return the first open order in ``state['orders']`` matching ``side`` (case-insensitive)."""
    if not isinstance(state, dict) or isinstance(side, bool):
        return None
    target = str(side or "").strip().lower()
    if not target:
        return None
    orders = state.get("orders")
    if not isinstance(orders, list):
        return None
    return next(
        (o for o in orders if isinstance(o, dict) and str(o.get("side") or "").strip().lower() == target),
        None,
    )


def compute_adaptive_reentry_pullback_pct(
    params,
    vol: float | None,
    *,
    fallback_reason: str = "fixed (fallback)",
    vol_label: str = "vol",
) -> tuple[float, str]:
    """Compute volatility-scaled or fixed re-entry pullback threshold from peak."""
    if not getattr(params, "reentry_pullback_adaptive", False):
        return params.reentry_pullback_pct, "fixed"
    if vol is None:
        return params.reentry_pullback_pct, fallback_reason
    val = max(
        params.reentry_pullback_min,
        min(params.reentry_pullback_max, params.reentry_pullback_k * vol),
    )
    return val, f"adaptive ({vol_label} {vol:.2f}% x k={params.reentry_pullback_k} -> {val:.2f}%)"


def build_strategy_exit_manager(
    params,
    state: dict,
    vol_resolver: Callable[[str], tuple[float, str]],
    *,
    allow_external_cache: bool,
    mode: str = "spot",
) -> PositionExitManager:
    """Construct and bind a ``PositionExitManager`` for spot or T212 strategy engines."""
    p = params
    if mode == "t212":
        exit_cfg = PositionExitConfig(
            lost_threshold=vol_pct_to_ratio(p.stop_loss_pct),
            sl_disaster_buffer_ratio=p.sl_disaster_buffer_ratio,
            sl_disaster_buffer_max_pct=vol_pct_to_ratio(p.sl_disaster_buffer_max_pct),
            sl_trend_defer_max_sec=p.sl_trend_defer_max_sec,
            sl_bounce_min_recovery_pct=vol_pct_to_ratio(p.sl_bounce_min_recovery_pct),
            sl_bounce_pullback_pct=vol_pct_to_ratio(p.sl_bounce_pullback_pct),
            sl_structural_break_enabled=p.sl_structural_break_enabled,
            smart_sl_enabled=False,
            reentry_enabled=True,
            reentry_sl_bounce_pct=(
                vol_pct_to_ratio(p.sl_rebuy_bounce_pct)
                if p.sl_rebuy_enabled
                else (0.015 if p.sl_mhdsl_enabled else 0.0)
            ),
            reentry_drop_pct=vol_pct_to_ratio(p.reentry_drop_pct),
            reentry_pullback_pct=vol_pct_to_ratio(p.reentry_pullback_pct),
            reentry_ttl_hours=p.reentry_ttl_hours,
            reentry_tolerance_pct=p.reentry_tolerance_pct,
            reentry_bear_gate_enabled=False,
            reentry_bear_bounce_pct=0.0,
            gain_threshold=vol_pct_to_ratio(p.takeprofit_pct),
            tp_trend_ride_enabled=p.tp_mdhtp_enabled,
            tp_profit_floor_min_pct=vol_pct_to_ratio(p.trail_min_profit_pct) if p.trail_min_profit_pct > 0 else 0.0,
            tp_pullback_pct=vol_pct_to_ratio(p.trail_pct) if p.trail_pct > 0 else 0.015,
            tp_stagnation_max_sec=p.tp_stagnation_max_sec,
            allow_external_cache=allow_external_cache,
        )
    else:
        exit_cfg = PositionExitConfig(
            lost_threshold=p.stop_loss_pct / 100.0,
            sl_disaster_buffer_ratio=p.sl_disaster_buffer_ratio,
            sl_disaster_vol_multiplier=p.sl_disaster_vol_multiplier,
            sl_disaster_buffer_min_pct=p.sl_disaster_buffer_min_pct / 100.0,
            sl_disaster_buffer_max_pct=p.sl_disaster_buffer_max_pct / 100.0,
            sl_trend_defer_max_sec=p.sl_trend_defer_max_sec,
            sl_bounce_min_recovery_pct=p.sl_bounce_min_recovery_pct / 100.0,
            sl_bounce_pullback_pct=p.sl_bounce_pullback_pct / 100.0,
            sl_bounce_ratchet_enabled=p.sl_bounce_ratchet_enabled,
            sl_bounce_ratchet_k=p.sl_bounce_ratchet_k,
            sl_bounce_min_pullback_pct=p.sl_bounce_min_pullback_pct / 100.0,
            sl_structural_break_enabled=p.sl_structural_break_enabled,
            smart_sl_enabled=False,
            gain_threshold=p.takeprofit_pct / 100.0,
            tp_trend_ride_enabled=p.tp_trend_hold,
            tp_profit_floor_min_pct=(p.tp_trail_profit_floor_pct or 0.0) / 100.0,
            tp_pullback_pct=p.tp_trail_pct / 100.0,
            tp_stagnation_max_sec=p.tp_stagnation_max_sec,
            fast_profit_guard_enabled=p.fast_profit_guard,
            fast_profit_mult=p.fast_profit_mult,
            fast_profit_window_sec=p.fast_profit_window_min * 60.0,
            fast_profit_drop_pct=p.fast_profit_drop_pct,
            reentry_enabled=True,
            reentry_sl_bounce_pct=p.reentry_sl_bounce_pct / 100.0,
            reentry_drop_pct=p.reentry_drop_pct / 100.0,
            reentry_pullback_pct=p.reentry_pullback_pct / 100.0,
            reentry_bear_bounce_pct=p.reentry_bear_bounce_pct / 100.0,
            reentry_ttl_hours=p.reentry_ttl_hours,
            reentry_tolerance_pct=p.reentry_tolerance_pct,
            allow_external_cache=allow_external_cache,
        )
    mgr = PositionExitManager(config=exit_cfg, vol_resolver=vol_resolver)
    mgr.bind_state(state)
    return mgr


def build_dataclass_from_dict(
    cls: type,
    values: dict | None,
    *,
    defaults: dict | None = None,
    warn_dropped: bool = False,
):
    """Build a dataclass instance from a dict, merging class ``_DEFAULT_DICT_FIELDS`` and ignoring removed fields."""
    base_defaults = dict(getattr(cls, "_DEFAULT_DICT_FIELDS", {}) or {})
    if isinstance(defaults, dict):
        base_defaults.update(defaults)
    base_values = dict(values) if isinstance(values, dict) else {}
    merged = {**base_defaults, **base_values}
    valid = {f.name for f in _dc_fields(cls) if f.init}
    if _parse_bool_flag(warn_dropped, False):
        dropped = sorted(set(merged) - valid)
        if dropped:
            print(f"[params] ignoring parameters removed from {cls.__name__}: {dropped}")
    return cls(**{k: v for k, v in merged.items() if k in valid})


def update_post_sell_anchors(
    state: dict,
    price: float,
    *,
    include_last_sell_alias: bool = True,
    track_stop_sl_low: bool = False,
) -> float | None:
    """Resolve ``last_sell_price`` and update ``post_sell_peak`` / ``post_sell_trough`` (and optional ``sl_low``)."""
    lsp = safe_pos_float(state.get("last_sell_price"), None)
    if not lsp and include_last_sell_alias:
        lsp = safe_pos_float(state.get("last_sell"), None)
    if not lsp:
        return None
    cur_peak = safe_pos_float(state.get("post_sell_peak"), lsp)
    state["post_sell_peak"] = max(cur_peak, price)
    cur_trough = safe_pos_float(state.get("post_sell_trough"), lsp)
    state["post_sell_trough"] = min(cur_trough, price)
    if track_stop_sl_low and state.get("last_exit_kind") == "STOP":
        cur_sl_low = safe_pos_float(state.get("sl_low"), price)
        state["sl_low"] = min(cur_sl_low, price)
    return lsp


def safe_state_token(val: object, default: str = "unknown", *, lower: bool = False) -> str:
    """Sanitize a symbol/ticker/direction token for use in a ``.state_<token>.json`` filename."""
    if isinstance(val, bool) or not isinstance(val, str):
        return default
    s = val.strip().lower() if lower else val
    return "".join(c for c in s if c.isalnum() or c in "._-").strip("._-") or default


def state_path_for(
    symbol: str,
    state_dir: str | os.PathLike | None = None,
    *,
    default_dir: str = "",
) -> str:
    """Return ``<dir>/.state_<safe_symbol>.json`` using ``default_dir`` when ``state_dir`` is blank."""
    safe = safe_state_token(symbol, "unknown")
    if isinstance(state_dir, os.PathLike):
        directory = os.fspath(state_dir) or default_dir
    elif isinstance(state_dir, str) and state_dir.strip():
        directory = state_dir.strip()
    else:
        directory = default_dir
    return os.path.join(directory, f".state_{safe}.json")


class StatePersistenceError(RuntimeError):
    """Financial state cannot be read or persisted safely."""


class JsonStateStore:
    def __init__(self, path: str, default_factory: Callable[[], dict], *,
                 label: str, logger: Callable[[str], None], fail_closed: bool):
        self.path = path
        self.default_factory = default_factory
        self.label = label
        self.logger = logger
        self.fail_closed = False if fail_closed is None else _parse_bool_flag(fail_closed, True)

    def load(self) -> dict:
        try:
            loaded = load_json_state(
                self.path, default_factory=dict, fail_closed=True,
                label=self.label,
            )
            if not isinstance(loaded, dict):
                raise ValueError("loaded state is not a dict")
            merged = self.default_factory()
            if not isinstance(merged, dict):
                raise TypeError("default_factory did not return a dict")
            merged.update(loaded)
            self.logger(
                f"  [STRAT] state loaded (cycle {merged.get('cycle')}, "
                f"qty {merged.get('qty')})"
            )
            return merged
        except (OSError, StateReadError, TypeError, ValueError) as exc:
            message = f"invalid {self.label} state in {self.path}: {exc}"
            if self.fail_closed:
                raise StatePersistenceError(message) from exc
            self.logger(f"  ! [STRAT] {message}; reset allowed only in PAPER")
            return self.default_factory()

    def save(self, state: dict) -> bool:
        try:
            if not isinstance(state, dict):
                raise TypeError("state must be a dict")
            save_json_state(self.path, state, indent=2)
            return True
        except (OSError, TypeError, ValueError) as exc:
            self.logger(f"  ! [STRAT] cannot save the state: {exc}")
            if self.fail_closed:
                raise StatePersistenceError(
                    f"persisting the {self.label} state failed: {exc}"
                ) from exc
            return False

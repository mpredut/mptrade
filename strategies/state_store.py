"""Shared JSON persistence for financial-engine state.

Writes are atomic within the same directory. Live runtimes may fail closed so
corrupt or unsavable state is not mistaken for empty state that could permit a
duplicate entry.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import fields
from botcore import _parse_bool_flag, safe_pos_float
from cachepaths import resolve_venue_runtime_dir
from state_io import StateReadError, load_json_state, save_json_state
from .position_exit_manager import PositionExitConfig, PositionExitManager, is_stop_exit_kind


def find_open_order(state: dict, side: str) -> dict | None:
    """Return the first open order in ``state['orders']`` matching ``side`` (case-insensitive)."""
    if not isinstance(state, dict) or isinstance(side, bool):
        return None
    target = str(side or "").strip().lower()
    orders = state.get("orders") if target else None
    if not isinstance(orders, list):
        return None
    return next((o for o in orders if isinstance(o, dict) and str(o.get("side") or "").strip().lower() == target), None)


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
    vol_f = safe_pos_float(vol, default=None)
    if vol_f is None:
        return params.reentry_pullback_pct, fallback_reason
    lo = min(params.reentry_pullback_min, params.reentry_pullback_max)
    hi = max(params.reentry_pullback_min, params.reentry_pullback_max)
    val = max(lo, min(hi, params.reentry_pullback_k * vol_f))
    return val, f"adaptive ({vol_label} {vol_f:.2f}% x k={params.reentry_pullback_k} -> {val:.2f}%)"


def build_strategy_exit_manager(
    params,
    state: dict,
    vol_resolver: Callable[[str], tuple[float, str]],
    *,
    allow_external_cache: bool,
    ratio_fn: Callable[[float], float] | None = None,
    extra_config: dict | None = None,
) -> PositionExitManager:
    """Construct and bind a ``PositionExitManager`` using declarative strategy overrides."""
    p = params
    to_ratio = ratio_fn if callable(ratio_fn) else (lambda v: v / 100.0)
    common = dict(
        lost_threshold=to_ratio(p.stop_loss_pct),
        sl_disaster_buffer_ratio=p.sl_disaster_buffer_ratio,
        sl_disaster_buffer_max_pct=to_ratio(p.sl_disaster_buffer_max_pct),
        sl_trend_defer_max_sec=p.sl_trend_defer_max_sec,
        sl_bounce_min_recovery_pct=to_ratio(p.sl_bounce_min_recovery_pct),
        sl_bounce_pullback_pct=to_ratio(p.sl_bounce_pullback_pct),
        sl_structural_break_enabled=p.sl_structural_break_enabled,
        smart_sl_enabled=False,
        reentry_enabled=True,
        reentry_drop_pct=to_ratio(p.reentry_drop_pct),
        reentry_pullback_pct=to_ratio(p.reentry_pullback_pct),
        reentry_ttl_hours=p.reentry_ttl_hours,
        reentry_tolerance_pct=p.reentry_tolerance_pct,
        gain_threshold=to_ratio(p.takeprofit_pct),
        tp_stagnation_max_sec=p.tp_stagnation_max_sec,
        allow_external_cache=allow_external_cache,
    )
    if isinstance(extra_config, dict):
        common.update(extra_config)
    exit_cfg = PositionExitConfig(**common)
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
    merged = {**base_defaults, **(dict(values) if isinstance(values, dict) else {})}
    valid = {f.name for f in fields(cls) if f.init}
    if _parse_bool_flag(warn_dropped, False):
        dropped = sorted(set(merged) - valid)
        if dropped:
            print(f"[params] ignoring parameters removed from {cls.__name__}: {dropped}")
    return cls(**{k: v for k, v in merged.items() if k in valid})


def update_post_sell_anchors(
    state: dict,
    price: float,
    *,
    track_stop_sl_low: bool = False,
) -> float | None:
    """Resolve ``last_sell_price`` and update ``post_sell_peak`` / ``post_sell_trough`` (and optional ``sl_low``)."""
    if not isinstance(state, dict):
        return None
    lsp = safe_pos_float(state.get("last_sell_price"), None)
    if not lsp:
        return None
    px_f = safe_pos_float(price, None)
    if px_f is None:
        return lsp
    state["post_sell_peak"] = max(safe_pos_float(state.get("post_sell_peak"), lsp), px_f)
    state["post_sell_trough"] = min(safe_pos_float(state.get("post_sell_trough"), lsp), px_f)
    if track_stop_sl_low and is_stop_exit_kind(state.get("last_exit_kind")):
        state["sl_low"] = min(safe_pos_float(state.get("sl_low"), lsp), px_f)
    return lsp


def safe_state_token(
    val: object,
    default: str = "unknown",
    *,
    lower: bool = False,
    allow_equals: bool = False,
) -> str:
    """Sanitize a symbol/ticker/direction token for use in a ``.state_<token>.json`` filename."""
    if isinstance(val, bool) or not isinstance(val, str):
        return default
    s = val.strip().lower() if lower else val
    allowed = "._-=" if allow_equals else "._-"
    return "".join(c for c in s if (c.isascii() and c.isalnum()) or c in allowed).strip(allowed) or default


def state_path_for(
    symbol: str,
    state_dir: str | os.PathLike | None = None,
    *,
    default_dir: str = "",
    venue: str | None = None,
) -> str:
    """Return ``<dir>/.state_<safe_symbol>.json`` using ``default_dir`` or ``resolve_venue_runtime_dir(venue)`` when ``state_dir`` is blank."""
    safe = safe_state_token(symbol, "unknown")
    eff_default = default_dir or (resolve_venue_runtime_dir(venue) if venue else "")
    if isinstance(state_dir, os.PathLike):
        directory = os.fspath(state_dir) or eff_default
    elif isinstance(state_dir, str) and state_dir.strip():
        directory = state_dir.strip()
    else:
        directory = eff_default
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
            loaded = load_json_state(self.path, default_factory=dict, fail_closed=True, label=self.label)
            if not isinstance(loaded, dict):
                raise ValueError("loaded state is not a dict")
            merged = self.default_factory()
            if not isinstance(merged, dict):
                raise TypeError("default_factory did not return a dict")
            merged.update(loaded)
            self.logger(f"  [STRAT] state loaded (cycle {merged.get('cycle')}, qty {merged.get('qty')})")
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
                raise StatePersistenceError(f"persisting the {self.label} state failed: {exc}") from exc
            return False

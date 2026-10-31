"""Shared spot strategy mixin for reentry gate, position snapshot, volatility tuple, and exit manager wiring.

Unifies the common state persistence, open-order lookup, 1-hour volatility resolution,
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
    _parse_bool_flag,
    are_close,
    log,
    required_bool_env,
    required_float_env,
    safe_float,
    safe_int,
    safe_nonneg_float,
    safe_pos_float,
    safe_pos_int,
)
from intelligence.internal.state.volatility import (
    adaptive_thresholds,
    resolve_cached_volatility_1h,
    resolve_cached_volatility_1h_pct,
    vol_pct_to_ratio,
)
from providers.base import _clean_single_order_id as _clean_order_id, normalize_side
from .position_exit_manager import (
    ExitAction,
    PositionExitManager,
    PositionSnapshot,
    ReentrySnapshot,
)
from .state_store import (
    JsonStateStore,
    build_strategy_exit_manager,
    clean_nonneg_float as _clean_nonneg_float,
    clean_pos_float as _clean_pos_float,
    compute_adaptive_reentry_pullback_pct,
    find_open_order,
    update_post_sell_anchors,
)


def load_common_mhdpa_env(env: dict | None, *, include_spot_sl: bool = True) -> dict[str, Any]:
    """Load shared Hybrid Re-entry, MHDSL, and MDHTP configuration keys from ``env``."""
    cfg: dict[str, Any] = {
        "reentry_hybrid_enabled": required_bool_env("STRAT_REENTRY_HYBRID_ENABLED", env),
        "reentry_peak_relative": required_bool_env("STRAT_REENTRY_PEAK_RELATIVE", env),
        "reentry_pullback_pct": required_float_env("STRAT_REENTRY_PULLBACK_PCT", env),
        "reentry_pullback_adaptive": required_bool_env("STRAT_REENTRY_PULLBACK_ADAPTIVE", env),
        "reentry_pullback_k": required_float_env("STRAT_REENTRY_PULLBACK_K", env),
        "reentry_pullback_min": required_float_env("STRAT_REENTRY_PULLBACK_MIN", env),
        "reentry_pullback_max": required_float_env("STRAT_REENTRY_PULLBACK_MAX", env),
        "reentry_ttl_hours": required_float_env("STRAT_REENTRY_TTL_HOURS", env),
        "sl_mhdsl_enabled": required_bool_env("STRAT_SL_MHDSL_ENABLED", env),
        "sl_disaster_buffer_ratio": required_float_env("STRAT_SL_DISASTER_BUFFER_RATIO", env),
        "sl_disaster_buffer_max_pct": required_float_env("STRAT_SL_DISASTER_BUFFER_MAX_PCT", env),
        "sl_trend_defer_max_sec": required_float_env("STRAT_SL_TREND_DEFER_MAX_SEC", env),
        "sl_bounce_min_recovery_pct": required_float_env("STRAT_SL_BOUNCE_MIN_RECOVERY_PCT", env),
        "sl_bounce_pullback_pct": required_float_env("STRAT_SL_BOUNCE_PULLBACK_PCT", env),
        "sl_structural_break_enabled": required_bool_env("STRAT_SL_STRUCTURAL_BREAK_ENABLED", env),
        "tp_mdhtp_enabled": required_bool_env("STRAT_TP_MDHTP_ENABLED", env),
        "tp_stagnation_max_sec": required_float_env("STRAT_TP_STAGNATION_MAX_SEC", env),
    }
    if include_spot_sl:
        cfg.update(
            reentry_bear_bounce_pct=required_float_env("STRAT_REENTRY_BEAR_BOUNCE_PCT", env),
            tp_dynamic_flat=required_bool_env("STRAT_TP_DYNAMIC_FLAT", env),
            tp_min_pct=required_float_env("STRAT_TP_MIN_PCT", env),
            tp_max_pct=required_float_env("STRAT_TP_MAX_PCT", env),
            trend_trail_dynamic=required_bool_env("STRAT_TREND_TRAIL_DYNAMIC", env),
            trend_trail_base_pct=required_float_env("STRAT_TREND_TRAIL_BASE_PCT", env),
            trend_trail_min_pct=required_float_env("STRAT_TREND_TRAIL_MIN_PCT", env),
            trend_trail_ratchet_k=required_float_env("STRAT_TREND_TRAIL_RATCHET_K", env),
            trend_trail_gain_threshold=required_float_env("STRAT_TREND_TRAIL_GAIN_THRESH", env),
            fast_profit_guard=required_bool_env("STRAT_FAST_PROFIT_GUARD", env),
            fast_profit_mult=required_float_env("STRAT_FAST_PROFIT_MULT", env),
            fast_profit_window_min=required_float_env("STRAT_FAST_PROFIT_WINDOW_MIN", env),
            fast_profit_drop_pct=required_float_env("STRAT_FAST_PROFIT_DROP_PCT", env),
            surge_guard=required_bool_env("STRAT_SURGE_GUARD", env),
            surge_gain_pct=required_float_env("STRAT_SURGE_GAIN_PCT", env),
            surge_window_hours=required_float_env("STRAT_SURGE_WINDOW_HOURS", env),
            surge_move_pct=required_float_env("STRAT_SURGE_MOVE_PCT", env),
            surge_exit_pullback_pct=required_float_env("STRAT_SURGE_EXIT_PULLBACK_PCT", env),
            surge_dynamic=required_bool_env("STRAT_SURGE_DYNAMIC", env),
            surge_min_gain_pct=required_float_env("STRAT_SURGE_MIN_GAIN_PCT", env),
            surge_max_gain_pct=required_float_env("STRAT_SURGE_MAX_GAIN_PCT", env),
            surge_vol_multiplier=required_float_env("STRAT_SURGE_VOL_MULTIPLIER", env),
            slow_grind_guard=required_bool_env("STRAT_SLOW_GRIND_GUARD", env),
            slow_grind_days=required_float_env("STRAT_SLOW_GRIND_DAYS", env),
            slow_grind_min_gain_pct=required_float_env("STRAT_SLOW_GRIND_MIN_GAIN_PCT", env),
            slow_grind_flash_window_min=required_float_env("STRAT_SLOW_GRIND_FLASH_WINDOW_MIN", env),
            slow_grind_flash_drop_pct=required_float_env("STRAT_SLOW_GRIND_FLASH_DROP_PCT", env),
            slow_grind_hourly_drop_pct=required_float_env("STRAT_SLOW_GRIND_HOURLY_DROP_PCT", env),
            sl_disaster_vol_multiplier=required_float_env("STRAT_SL_DISASTER_VOL_MULTIPLIER", env),
            sl_disaster_buffer_min_pct=required_float_env("STRAT_SL_DISASTER_BUFFER_MIN_PCT", env),
            sl_bounce_ratchet_enabled=required_bool_env("STRAT_SL_BOUNCE_RATCHET_ENABLED", env),
            sl_bounce_ratchet_k=required_float_env("STRAT_SL_BOUNCE_RATCHET_K", env),
            sl_bounce_min_pullback_pct=required_float_env("STRAT_SL_BOUNCE_MIN_PULLBACK_PCT", env),
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
        state[nkey] = _clean_nonneg_float(state.get(nkey), 0.0) or 0.0
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
            state[pkey] = _clean_pos_float(state.get(pkey))
    for bkey in bool_keys:
        if bkey in state:
            state[bkey] = _parse_bool_flag(state.get(bkey), False)
    for dkey in dict_keys:
        if not isinstance(state.get(dkey), dict):
            state[dkey] = {}
    for ndkey in nullable_dict_keys:
        if not isinstance(state.get(ndkey), dict):
            state[ndkey] = None
    if sanitize_orders:
        if isinstance(state.get("orders"), list):
            clean_orders = []
            for o in state["orders"]:
                if not isinstance(o, dict):
                    continue
                if order_id_key in o and o.get(order_id_key) is not None:
                    o[order_id_key] = _clean_order_id(o.get(order_id_key))
                if "side" in o and not isinstance(o.get("side"), bool):
                    s_norm = normalize_side(o.get("side"))
                    if s_norm in ("BUY", "SELL"):
                        o["side"] = s_norm.lower() if lowercase_order_side else s_norm
                if "intent_id" in o:
                    o["intent_id"] = _clean_order_id(o.get("intent_id"))
                for pk in order_pos_float_keys:
                    if pk in o:
                        o[pk] = safe_pos_float(o.get(pk), 0.0)
                for nk in order_nonneg_float_keys:
                    if nk in o:
                        o[nk] = safe_nonneg_float(o.get(nk), 0.0)
                for bk in order_bool_keys:
                    if bk in o:
                        o[bk] = _parse_bool_flag(o.get(bk), False)
                clean_orders.append(o)
            state["orders"] = clean_orders
        else:
            state["orders"] = []
    pi = state.get("pending_intent")
    if isinstance(pi, dict):
        if "order_id" in pi and pi.get("order_id") is not None:
            pi["order_id"] = _clean_order_id(pi.get("order_id"))
        if "intent_id" in pi and pi.get("intent_id") is not None:
            pi["intent_id"] = _clean_order_id(pi.get("intent_id"))
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

    def _mod_global(self, name: str, default: Any = None) -> Any:
        """Resolve a module-level symbol from the concrete Strategy's defining module."""
        init_fn = getattr(type(self), "__init__", None)
        globs = getattr(init_fn, "__globals__", None)
        if isinstance(globs, dict) and name in globs:
            return globs[name]
        return default

    def _log_msg(self, msg: str) -> None:
        self._mod_global("log", log)(msg)

    @property
    def _symbol_id(self) -> str:
        return getattr(self, "pair", None) or getattr(self, "ticker", "")

    def _now(self) -> float:
        """Return the current timestamp honoring custom clock or simulated time in replay mode."""
        clock = getattr(self, "_clock", None)
        if callable(clock):
            return float(clock())
        if getattr(self, "replay_mode", False):
            sim_ts = getattr(self, "_simulated_ts", None)
            if sim_ts is not None:
                return sim_ts
            sp = getattr(self, "_shadow_prices", None)
            if sp:
                return sp[-1][0]
            return 0.0
        time_mod = self._mod_global("time", time)
        return float(time_mod.time())

    def set_simulated_time(self, ts: float | None) -> None:
        """Explicitly set simulated timestamp for replay and backtesting drivers."""
        self._simulated_ts = _clean_nonneg_float(ts)

    # -- Persistence -----------------------------------------------------------
    def _store(self) -> JsonStateStore:
        store_cls = self._mod_global("JsonStateStore", JsonStateStore)
        new_state_fn = self._mod_global("_new_state")
        log_fn = self._mod_global("log", log)
        label = getattr(self, "venue_label", "T212")
        return store_cls(
            self.state_file,
            new_state_fn,
            label=label,
            logger=log_fn,
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
                disk_mtime = float(os.path.getmtime(self.state_file))
                if disk_mtime > getattr(self, "_state_file_mtime", 0.0) + 1e-3:
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

    # -- Position & Order Helpers ----------------------------------------------
    def _avg(self) -> float | None:
        qty = _clean_pos_float(self.s.get("qty"))
        cost = _clean_pos_float(self.s.get(self._cost_state_key))
        if qty is None or cost is None or qty <= self._qty_eps:
            return None
        return safe_pos_float(cost / qty, default=None) if self._exit_manager_mode == "t212" else cost / qty

    _avg_cost = _avg

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
            return (
                str(pending.get("side") or "").strip().upper() == side_u
                or self._find_open(side_u) is not None
            )
        return self._find_open(side) is not None

    def _remove(self, o: dict) -> None:
        if o in self.s["orders"]:
            self.s["orders"].remove(o)

    _remove_order = _remove

    def _cancel_open(self, side: str) -> bool:
        o = self._find_open(side)
        if not o:
            return True
        if self._exit_manager_mode == "t212":
            return self._cancel_specific(o)
        return self._cancel_order(o)

    def _seed_post_exit_anchors(
        self,
        price: float,
        exit_kind: str | None,
        exit_ts: float,
        *,
        include_sl_low: bool = True,
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
    def _hourly_vol_from_closes(
        closes: list[float], interval_minutes: float,
    ) -> float | None:
        """Return log-return deviation normalized to one hour for a fixed cadence."""
        itv_f = safe_pos_float(interval_minutes, default=None)
        if not closes or itv_f is None:
            return None
        clean_closes = [
            cf for c in closes[-90:]
            if (cf := safe_pos_float(c, default=None)) is not None
        ]
        closes_slice = tuple(clean_closes)
        if len(closes_slice) < 20:
            return None
        cache_key = (closes_slice, itv_f)
        if cache_key in BaseSpotMixin._HOURLY_VOL_CACHE:
            return BaseSpotMixin._HOURLY_VOL_CACHE[cache_key]
        returns = [
            math.log(current / previous)
            for previous, current in zip(closes_slice, closes_slice[1:])
            if previous > 0 and current > 0
        ]
        val = BaseSpotMixin._scaled_log_return_vol_pct(returns, 60.0 / itv_f)
        if len(BaseSpotMixin._HOURLY_VOL_CACHE) >= BaseSpotMixin._REPLAY_CACHE_MAX:
            BaseSpotMixin._HOURLY_VOL_CACHE.clear()
        BaseSpotMixin._HOURLY_VOL_CACHE[cache_key] = val
        return val

    def _shadow_vol_1h(self) -> float | None:
        """Return one-hour volatility from local tick history, or None during warm-up."""
        sp = getattr(self, "_shadow_prices", None)
        if not sp:
            return None
        n_sp = len(sp)
        if n_sp < 20:
            return None
        if self.replay_mode:
            cache_key = (
                n_sp,
                round(sp[-1][0] - sp[0][0], 3),
                sp[0][1],
                sp[1][1],
                sp[n_sp >> 1][1],
                sp[-2][1],
                sp[-1][1],
            )
            if getattr(self, "_shadow_vol_cache_key", None) == cache_key:
                return self._shadow_vol_cache_val
            if cache_key in BaseSpotMixin._SHADOW_VOL_CACHE:
                val = BaseSpotMixin._SHADOW_VOL_CACHE[cache_key]
                self._shadow_vol_cache_key = cache_key
                self._shadow_vol_cache_val = val
                return val
        mean_dt = (sp[-1][0] - sp[0][0]) / (n_sp - 1)
        if mean_dt <= 0:
            return None
        pts = list(sp)
        rets = [
            math.log(pts[i][1] / pts[i - 1][1])
            for i in range(1, n_sp)
            if pts[i - 1][1] > 0 and pts[i][1] > 0
        ]
        val = self._scaled_log_return_vol_pct(rets, 3600.0 / mean_dt)
        if val is None:
            return None
        if self.replay_mode:
            self._shadow_vol_cache_key = cache_key
            self._shadow_vol_cache_val = val
            if len(BaseSpotMixin._SHADOW_VOL_CACHE) >= BaseSpotMixin._REPLAY_CACHE_MAX:
                BaseSpotMixin._SHADOW_VOL_CACHE.clear()
            BaseSpotMixin._SHADOW_VOL_CACHE[cache_key] = val
        return val

    def _trail_vol_1h(self) -> float | None:
        """Return one-hour-normalized volatility at one OHLC cadence in live and replay."""
        if self.replay_mode:
            sp = getattr(self, "_shadow_prices", None)
            if not sp:
                return None
            n_sp = len(sp)
            if n_sp < 20:
                return None
            cache_key = (
                n_sp,
                sp[0][1],
                sp[1][1],
                sp[n_sp >> 1][1],
                sp[-2][1],
                sp[-1][1],
                self.p.tp_trail_vol_interval,
            )
            if getattr(self, "_trail_vol_cache_key", None) == cache_key:
                return self._trail_vol_cache_val
            if cache_key in BaseSpotMixin._TRAIL_VOL_CACHE:
                val = BaseSpotMixin._TRAIL_VOL_CACHE[cache_key]
                self._trail_vol_cache_key = cache_key
                self._trail_vol_cache_val = val
                return val
            closes = [price for _, price in sp]
            val = self._hourly_vol_from_closes(closes, self.p.tp_trail_vol_interval)
            self._trail_vol_cache_key = cache_key
            self._trail_vol_cache_val = val
            if len(BaseSpotMixin._TRAIL_VOL_CACHE) >= BaseSpotMixin._REPLAY_CACHE_MAX:
                BaseSpotMixin._TRAIL_VOL_CACHE.clear()
            BaseSpotMixin._TRAIL_VOL_CACHE[cache_key] = val
            return val
        closes = self.client.ohlc_closes(self._symbol_id, self.p.tp_trail_vol_interval)
        return self._hourly_vol_from_closes(closes, self.p.tp_trail_vol_interval)

    def _dca_vol_1h(self) -> float | None:
        """Return DCA volatility from OHLC at the same cadence in live and replay."""
        if self.replay_mode:
            closes = [price for _, price in getattr(self, "_shadow_prices", ())]
        else:
            closes = self.client.ohlc_closes(self._symbol_id, self.p.dca_vol_interval)
        return self._hourly_vol_from_closes(closes, self.p.dca_vol_interval)

    def _resolve_shadow_vol_1h_pct(self) -> float | None:
        if hasattr(self, "_shadow_vol_1h"):
            try:
                return safe_pos_float(self._shadow_vol_1h(), default=None)
            except Exception:
                return None
        return None

    def _resolve_reentry_vol_1h_pct(self) -> float | None:
        """Resolve 1-hour volatility (%) from shadow ticks with non-replay cache fallback."""
        vol = (
            self._shadow_vol_1h()
            if self._exit_manager_mode == "spot"
            else self._resolve_shadow_vol_1h_pct()
        )
        if vol is None and not self.replay_mode:
            resolver = self._mod_global(
                "resolve_cached_volatility_1h_pct", resolve_cached_volatility_1h_pct,
            )
            vol = resolver(self._symbol_id)
        return vol

    def _vol_1h_ratio(self) -> tuple[float, str]:
        """Return 1h volatility as a decimal ratio [e.g. 0.025 for 2.5%] and its source."""
        if self._exit_manager_mode == "spot":
            try:
                vol = safe_pos_float(self._trail_vol_1h(), default=None)
                if vol is not None:
                    return vol_pct_to_ratio(vol), "trail_ohlc"
            except Exception:
                pass
            try:
                vol = safe_pos_float(self._shadow_vol_1h(), default=None)
                if vol is not None:
                    return vol_pct_to_ratio(vol), "shadow_ticks"
            except Exception:
                pass
        else:
            vol = self._resolve_shadow_vol_1h_pct()
            if vol is not None:
                return vol_pct_to_ratio(vol), "shadow"
        if not self.replay_mode:
            resolver = self._mod_global(
                "resolve_cached_volatility_1h", resolve_cached_volatility_1h,
            )
            cached = resolver(self._symbol_id)
            if cached is not None:
                return cached, "cache"
        return 0.0, self._vol_default_source

    def _resolve_volatility_tuple(self, symbol: str) -> tuple[float, str]:
        """Callable adapter for PositionExitManager vol_resolver."""
        return self._vol_1h_ratio()

    def _shadow_reentry_line(self, price: float, lsp: float, prag_fix: float) -> None:
        try:
            k_re = self.p.shadow_k_reentry
            vol = self._resolve_reentry_vol_1h_pct()
            if vol is None:
                self._log_msg(f"  [SHADOW] adaptive threshold: warm-up ({len(self._shadow_prices)}/20 points)")
                return
            adapt_pct, _ = adaptive_thresholds(vol, k_reentry=k_re)
            if adapt_pct is None:
                adapt_pct = k_re * vol
            prag_adapt = lsp * (1 - adapt_pct / 100)
            verdict = "WOULD HAVE ENTERED" if price <= prag_adapt else "would not have entered either"
            self._log_msg(
                f"  [SHADOW] fixed threshold {prag_fix:.2f} vs adaptive {prag_adapt:.2f} "
                f"(vol_1h {vol:.2f}% x k={k_re}) → {verdict}"
            )
        except Exception as e:  # noqa: BLE001 — observational
            self._log_msg(f"  [SHADOW] calculation error ({e}) — ignored")

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
        eff_drop = adapt_drop if adapt_drop is not None else (k_re * vol)
        return eff_drop, f"adaptiv (vol_1h {vol:.2f}% x k={k_re})"

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
            self._log_msg(f"  [STRAT] trailing adaptiv: OHLC indisponibil ({e}) — fallback pe fix")
            vol = None
        if vol is None:
            return self.p.tp_trail_pct
        return max(self.p.tp_trail_min, min(self.p.tp_trail_max, self.p.tp_trail_k * vol))

    # -- Exit Manager, Snapshot & Re-entry Gate --------------------------------
    def _init_exit_manager(self) -> PositionExitManager:
        return build_strategy_exit_manager(
            self.p,
            self.s,
            self._resolve_volatility_tuple,
            allow_external_cache=not self.replay_mode,
            mode=self._exit_manager_mode,
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
    ) -> PositionSnapshot:
        """Construct a standardized PositionSnapshot using the strategy's current context."""
        is_spot = self._exit_manager_mode == "spot"
        sp = getattr(self, "_shadow_prices", None)
        tick_time = sp[-1][0] if (is_spot and sp) else self._now()
        if include_vol:
            vol_1h, vol_source = self._vol_1h_ratio()
        else:
            vol_1h, vol_source = 0.0, "default"
        if is_spot:
            trend_up = self._is_trend_up(regime) if include_trend else False
            sell_qty = held
            provider = getattr(self, "venue_label", "Kraken")
            shadow = list(sp) if (include_shadow and sp is not None) else []
        else:
            trend_up = False
            slope_fn = getattr(self, "_trend_slope_provider", None)
            if include_trend and slope_fn:
                try:
                    slope = safe_nonneg_float(slope_fn(self.yahoo_sym), default=None)
                    if slope is not None:
                        trend_up = True
                except Exception:
                    pass
            sell_qty = self._sellable_qty(held, price)
            provider = "t212"
            shadow = []
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

    def _in_mhdsl_territory(
        self, loss_pct: float, *, reset_episode_on_exit: bool = True
    ) -> bool:
        """Return True if the position is in MHDSL stop-loss or bounce-watch territory."""
        in_sl = loss_pct > 0.0 and (
            loss_pct >= self.p.stop_loss_pct
            or are_close(loss_pct, self.p.stop_loss_pct, self.p.reentry_tolerance_pct)
        )
        if not in_sl and "sl_pending_exit" in self.s:
            self.s["sl_pending_exit"] = None
        mgr = self.exit_manager
        sym = self._symbol_id
        in_bounce_watch = bool(mgr is not None and sym in mgr.sl_bounce_state)
        if not in_sl and not in_bounce_watch:
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

    def _sync_mdhtp_config(
        self,
        eff_trail_pct: float,
        *,
        trend_ride_enabled: bool | None = None,
    ) -> None:
        """Synchronize MDHTP parameters onto ``self.exit_manager.config``."""
        if self.exit_manager is None:
            self.exit_manager = self._init_exit_manager()
        cfg = self.exit_manager.config
        cfg.tp_pullback_pct = eff_trail_pct / 100.0
        cfg.tp_trend_ride_enabled = (
            bool(getattr(self.p, "tp_trend_hold", True))
            if trend_ride_enabled is None
            else trend_ride_enabled
        )
        floor_pct = (
            getattr(self.p, "tp_trail_profit_floor_pct", None)
            if self._exit_manager_mode == "spot"
            else getattr(self.p, "trail_min_profit_pct", None)
        )
        cfg.tp_profit_floor_min_pct = (floor_pct or 0.0) / 100.0
        cfg.tp_stagnation_max_sec = self.p.tp_stagnation_max_sec
        if self._exit_manager_mode == "spot":
            cfg.fast_profit_guard_enabled = self.p.fast_profit_guard
            cfg.fast_profit_mult = self.p.fast_profit_mult
            cfg.fast_profit_window_sec = self.p.fast_profit_window_min * 60.0
            cfg.fast_profit_drop_pct = self.p.fast_profit_drop_pct

    def _evaluate_reentry_gate(
        self,
        price: float,
        regime: Any = None,
        tick_time: float | None = None,
    ) -> tuple[bool, str]:
        """Evaluate re-entry gate conditions symmetrically across spot and equity engines."""
        is_spot = self._exit_manager_mode == "spot"
        lsp = update_post_sell_anchors(
            self.s,
            price,
            include_last_sell_alias=is_spot,
            track_stop_sl_low=is_spot,
        )
        if not lsp:
            return True, "no_last_sell"

        if not is_spot:
            last_sell_ts = _clean_pos_float(self.s.get("last_sell_ts"))
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

        if is_spot:
            last_kind = self.s.get("last_exit_kind")
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
            trend_up = (
                self._is_trend_up(regime)
                if (needs_reentry_regime or self.p.reentry_bear_bounce_pct > 0)
                else False
            )
            trend_down = self._is_trend_down(regime) if needs_reentry_regime else False
        else:
            last_kind = "STOP" if self.s.get("last_exit_kind") == "STOP" else "TP"
            mgr.config.reentry_sl_bounce_pct = (
                vol_pct_to_ratio(self.p.sl_rebuy_bounce_pct)
                if self.p.sl_rebuy_enabled
                else (0.015 if self.p.sl_mhdsl_enabled else 0.0)
            )
            mgr.config.reentry_bear_gate_enabled = False
            mgr.config.reentry_bear_bounce_pct = 0.0
            trend_up = bool(self.p.reentry_peak_relative)
            trend_down = False
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
            last_kind != "STOP"
            and self.p.reentry_hybrid_enabled
            and (trend_up or self.p.reentry_peak_relative)
        )
        if is_bull_reentry or not self.replay_mode:
            pullback_pct, _ = self._effective_reentry_pullback_pct()
        else:
            pullback_pct = self.p.reentry_pullback_pct
        mgr.config.reentry_pullback_pct = vol_pct_to_ratio(pullback_pct)

        custom_drop = None
        if is_spot:
            if (not is_bull_reentry and last_kind != "STOP") or not self.replay_mode:
                drop_pct, _ = self._effective_reentry_drop_pct()
            else:
                drop_pct = self.p.reentry_drop_pct
            custom_drop = vol_pct_to_ratio(drop_pct)

        if self.replay_mode:
            vol_1h, vol_source = 0.0, "default"
        else:
            vol_1h, vol_source = self._vol_1h_ratio()

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
            allowed, reason = (
                self._evaluate_reentry_gate(price, regime, tick_time)
                if is_spot
                else self._evaluate_reentry_gate(price)
            )
        if not allowed:
            self._log_msg(f"  [STRAT] re-entry blocked: {reason}")
            return False
        lsp = self.s.get("last_sell_price") or (self.s.get("last_sell") if is_spot else None)
        if lsp:
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

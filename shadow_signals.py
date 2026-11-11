#!/usr/bin/env python3
"""Kalman trend and adaptive-volatility signals consumed by tradeall.

Re-exports the 1D constant-velocity ``KalmanTrend`` and realized-volatility
primitives from ``pricewindow`` and the stateful ``KalmanStateTracker`` /
``KalmanJournal`` persistence layer from ``cachetrend``.
"""
from __future__ import annotations

from botcore import (
    normalize_epoch_sec,
    pos_float_env,
    safe_float,
    safe_nonneg_float,
    safe_nonneg_ts,
    safe_pos_float,
    sanitize_pipe_field,
)
from cachepaths import cache_symbol_variants
from cachetrend import (
    KALMAN_SAMPLE_SEC,
    KalmanJournal,
    KalmanStateTracker,
    _DEFAULT_LOG_DIR,
    _ENVELOPE_KEYS,
    _KALMAN_KEYS,
    _has_seq_min_len,
    _unwrap_state_items,
)
from instrument_registry import casefold_get
from pricewindow import (
    CONF_ENTER,
    CONF_EXIT,
    DT_MIN,
    GAP_RESET_SEC,
    KALMAN_QR,
    K_DCA_DEFAULT as K_DCA,
    K_REENTRY_DEFAULT as K_REENTRY,
    MIN_VEL_PCT_MIN,
    KalmanTrend,
    adaptive_thresholds,
    calculate_volatility_1h,
)
from state_io import append_text_line, durable_replace_file, read_json_dict, save_json_state

__all__ = [
    "CONF_ENTER",
    "CONF_EXIT",
    "DT_MIN",
    "GAP_RESET_SEC",
    "KALMAN_QR",
    "KALMAN_SAMPLE_SEC",
    "K_DCA",
    "K_REENTRY",
    "KalmanJournal",
    "KalmanStateTracker",
    "KalmanTrend",
    "MIN_VEL_PCT_MIN",
    "adaptive_thresholds",
    "calculate_volatility_1h",
    "pos_float_env",
]

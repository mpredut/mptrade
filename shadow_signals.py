#!/usr/bin/env python3
"""Kalman trend and adaptive-volatility signals consumed by tradeall.

Re-exports the 1D constant-velocity ``KalmanTrend`` and realized-volatility
primitives from ``pricewindow`` and the stateful ``KalmanStateTracker`` /
``KalmanJournal`` persistence layer from ``cachetrend``.
"""
from __future__ import annotations

from botcore import pos_float_env
from cachetrend import (
    KALMAN_SAMPLE_SEC,
    KalmanJournal,
    KalmanStateTracker,
)
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

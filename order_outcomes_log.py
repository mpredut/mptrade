# order_outcomes_log.py
"""Fleet-wide log of order-submission attempts, one pipe-delimited row per call.

The writer is observational and suppresses its own I/O errors. New rows use
``accepted`` when the provider returns a truthy submission payload. Historical rows
may contain the misleading legacy value ``executed``; neither label proves a fill.
Fill truth belongs to venue status and trade reconciliation, which this log does not
perform.

The format is consumed by ``tradeall_observe.py``:
    ts|symbol|side|price|qty|outcome|refuse_reason|caller|motivation
"""
import math
import os
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
ORDER_OUTCOMES_LOG_DIR = os.path.join(ROOT, "logger")


_INVALID_LOG_STRINGS = {"nan", "null", "none", "inf", "-inf", "+inf"}


def _sanitize_outcome_field(value):
    """Remove characters that would break the pipe-delimited record format."""
    if value is None or isinstance(value, (bool, dict, list, tuple, set)):
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    text = str(value).replace("|", "/").replace("\r", " ").replace("\n", " ").strip()
    return "" if text.lower() in _INVALID_LOG_STRINGS else text


def _clean_numeric_field(value):
    if value is None or isinstance(value, (bool, dict, list, tuple, set)):
        return ""
    if isinstance(value, float):
        return value if math.isfinite(value) else ""
    if isinstance(value, int):
        return value
    s = str(value).strip()
    if not s or s.lower() in _INVALID_LOG_STRINGS:
        return ""
    return s


def log_order_outcome(
    symbol,
    side,
    price,
    qty,
    outcome,
    refuse_reason,
    motivation,
    caller=None,
    *,
    regime_context=None,
):
    """Append one submission-attempt record with optional market regime telemetry.

    ``caller`` is computed by each call site because stack depth differs between
    the legacy Binance adapter and ``Instrument.place``.
    """
    try:
        os.makedirs(ORDER_OUTCOMES_LOG_DIR, exist_ok=True)
        path = os.path.join(ORDER_OUTCOMES_LOG_DIR,
                             f"order_outcomes_{datetime.now().strftime('%Y-%m-%d')}.log")
        sym_val = _sanitize_outcome_field(symbol)
        side_val = _sanitize_outcome_field(side)
        price_val = _clean_numeric_field(price)
        qty_val = _clean_numeric_field(qty)
        out_val = _sanitize_outcome_field(outcome)
        ref_val = _sanitize_outcome_field(refuse_reason)
        cal_val = _sanitize_outcome_field(caller)
        mot_val = _sanitize_outcome_field(motivation)
        cols = [time.time(), sym_val, side_val, price_val, qty_val, out_val,
                ref_val, cal_val, mot_val]
        if regime_context is not None:
            strength_val = getattr(regime_context, "strength", None)
            strength_str = ""
            if strength_val is not None and not isinstance(strength_val, bool):
                try:
                    sf = float(strength_val)
                    if math.isfinite(sf):
                        strength_str = f"{sf:.4f}"
                except (TypeError, ValueError, OverflowError):
                    strength_str = ""
            reg_raw = getattr(regime_context, "regime", "")
            src_raw = getattr(regime_context, "source", "")
            rsn_raw = getattr(regime_context, "reason", "")
            fresh_raw = getattr(regime_context, "fresh", None)
            fb_raw = getattr(regime_context, "fallback_used", None)
            cols.extend([
                reg_raw if isinstance(reg_raw, str) else "",
                strength_str,
                src_raw if isinstance(src_raw, str) else "",
                "" if fresh_raw is None else str(fresh_raw),
                "" if fb_raw is None else str(fb_raw),
                rsn_raw if isinstance(rsn_raw, str) else "",
            ])
        line = "|".join(
            str(c).replace("|", "/").replace("\r", " ").replace("\n", " ")
            if c is not None
            else ""
            for c in cols
        )
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:
        print(f"[log_order_outcome] error writing outcomes log: {e}")

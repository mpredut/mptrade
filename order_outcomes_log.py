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

from botcore import _parse_bool_flag, safe_float
from intelligence.macro_sentiment_ssot import obj_attr as _regime_attr

ROOT = os.path.dirname(os.path.abspath(__file__))
ORDER_OUTCOMES_LOG_DIR = os.path.join(ROOT, "logger")


_INVALID_LOG_STRINGS = {"nan", "null", "none", "inf", "-inf", "+inf"}


def _sanitize_outcome_field(value):
    """Remove characters that would break the pipe-delimited record format."""
    if value is None or isinstance(value, (bool, dict, list, tuple, set, frozenset)):
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    text = str(value).replace("|", "/").replace("\r", " ").replace("\n", " ").strip()
    return "" if text.lower() in _INVALID_LOG_STRINGS else text


def _clean_numeric_field(value):
    if isinstance(value, (dict, list, tuple, set, frozenset)):
        return ""
    f = safe_float(value, default=None)
    if f is None:
        return ""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    s = str(value).strip()
    return s if s.lower() not in _INVALID_LOG_STRINGS else ""


def _clean_bool_field(value) -> str:
    if value is None or isinstance(value, (dict, list, tuple, set, frozenset)):
        return ""
    parsed = _parse_bool_flag(value, default=None)
    return str(parsed) if parsed is not None else ""


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
        if regime_context is not None and not isinstance(
            regime_context, (bool, int, float, str, bytes, list, tuple, set, frozenset)
        ):
            strength_val = _regime_attr(regime_context, "strength", None)
            sf = safe_float(strength_val, default=None)
            strength_str = f"{sf:.4f}" if sf is not None else ""
            reg_raw = _regime_attr(regime_context, "regime", "")
            src_raw = _regime_attr(regime_context, "source", "")
            rsn_raw = _regime_attr(regime_context, "reason", "")
            fresh_raw = _regime_attr(regime_context, "fresh", None)
            fb_raw = _regime_attr(regime_context, "fallback_used", None)

            def _clean_regime_str(val) -> str:
                if isinstance(val, str):
                    return _sanitize_outcome_field(val)
                if val is not None and not isinstance(
                    val, (bool, int, float, bytes, dict, list, tuple, set, frozenset)
                ):
                    inner = getattr(val, "value", getattr(val, "name", None))
                    if isinstance(inner, str):
                        return _sanitize_outcome_field(inner)
                return ""

            cols.extend([
                _clean_regime_str(reg_raw),
                strength_str,
                _clean_regime_str(src_raw),
                _clean_bool_field(fresh_raw),
                _clean_bool_field(fb_raw),
                _clean_regime_str(rsn_raw),
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

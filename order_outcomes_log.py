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


def _sanitize_outcome_field(value):
    """Remove characters that would break the pipe-delimited record format."""
    return (
        str(value).replace("|", "/").replace("\r", " ").replace("\n", " ")
        if value is not None
        else ""
    )


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
        sym_val = "" if symbol is None or isinstance(symbol, bool) else symbol
        side_val = "" if side is None or isinstance(side, bool) else side
        price_val = "" if price is None or isinstance(price, bool) else price
        qty_val = "" if qty is None or isinstance(qty, bool) else qty
        out_val = "" if outcome is None or isinstance(outcome, bool) else outcome
        ref_val = refuse_reason if isinstance(refuse_reason, str) else ("" if refuse_reason is None or isinstance(refuse_reason, bool) else str(refuse_reason))
        cal_val = caller if isinstance(caller, str) else ("" if caller is None or isinstance(caller, bool) else str(caller))
        mot_val = motivation if isinstance(motivation, str) else ("" if motivation is None or isinstance(motivation, bool) else str(motivation))
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
            cols.extend([
                reg_raw if isinstance(reg_raw, str) else "",
                strength_str,
                src_raw if isinstance(src_raw, str) else "",
                str(getattr(regime_context, "fresh", "")),
                str(getattr(regime_context, "fallback_used", "")),
                rsn_raw if isinstance(rsn_raw, str) else "",
            ])
        line = "|".join(_sanitize_outcome_field(c) for c in cols)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:
        print(f"[log_order_outcome] error writing outcomes log: {e}")

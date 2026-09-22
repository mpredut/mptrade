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
import os
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
ORDER_OUTCOMES_LOG_DIR = os.path.join(ROOT, "logger")


def _sanitize_outcome_field(value):
    """Remove characters that would break the pipe-delimited record format."""
    return str(value).replace("|", "/").replace("\n", " ") if value is not None else ""


def log_order_outcome(symbol, side, price, qty, outcome, refuse_reason, motivation, caller=None):
    """Append one submission-attempt record.

    ``caller`` is computed by each call site because stack depth differs between
    the legacy Binance adapter and ``Instrument.place``.
    """
    try:
        os.makedirs(ORDER_OUTCOMES_LOG_DIR, exist_ok=True)
        path = os.path.join(ORDER_OUTCOMES_LOG_DIR,
                             f"order_outcomes_{datetime.now().strftime('%Y-%m-%d')}.log")
        cols = [time.time(), symbol, side, price, qty, outcome,
                refuse_reason or "", caller or "", motivation or ""]
        line = "|".join(_sanitize_outcome_field(c) for c in cols)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:
        print(f"[log_order_outcome] error writing outcomes log: {e}")

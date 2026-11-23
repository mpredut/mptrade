#!/usr/bin/env python3
"""order_ledger_worker.py -- single-writer, network-free fleet daemon for the global Order Ledger SSoT.

Continuously reconciles ``order_events_YYYY-MM-DD.jsonl`` against ``CacheOrderOpenManager`` and
``CacheOrderFilledManager`` (read-only cache API), publishing ``orders_active.json`` and
archiving terminal rows into ``orders_closed_YYYY-MM.jsonl``.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from botcore import (
    add_store_true_cli_args,
    load_dotenv,
    required_float_env,
    single_instance,
    touch_module_heartbeat,
)
from order_ledger_reconciler import OrderLedgerReconciler

load_dotenv()

WORKER_POLL_SEC = required_float_env("ORDER_LEDGER_POLL_SEC")
if WORKER_POLL_SEC <= 0:
    raise ValueError("ORDER_LEDGER_POLL_SEC must be finite and > 0")


def _touch_ledger_hb(status: dict, *, force: bool = False) -> None:
    touch_module_heartbeat(
        "order_ledger",
        interval_sec=30.0,
        force=force,
        status=status,
        intervals={"poll": WORKER_POLL_SEC},
    )


def main() -> int:
    import log  # noqa: F401 -- ensure stdout timestamps and daily logger file from startup

    parser = add_store_true_cli_args(
        argparse.ArgumentParser(description=__doc__),
        ("--once", "Run a single reconciliation pass and exit."),
    )
    args = parser.parse_args()
    reconciler = OrderLedgerReconciler()

    if args.once:
        stats = reconciler.reconcile_once()
        print(f"[order_ledger] single pass: {stats}")
        return 0

    single_instance("order_ledger_worker")
    print(
        f"[order_ledger] start (poll={WORKER_POLL_SEC:g}s, "
        f"grace={reconciler.reconcile_grace_sec:g}s, intent_stale={reconciler.intent_stale_sec:g}s)"
    )
    _touch_ledger_hb({"active_count": 0}, force=True)
    while True:
        last_stats = None
        try:
            stats = reconciler.reconcile_once()
            last_stats = stats
            if (
                stats["events_applied"]
                or stats["reconciled_open"]
                or stats["reconciled_filled"]
                or stats["reconciled_closed"]
                or stats["orphans_adopted"]
            ):
                print(f"[order_ledger] {stats}")
        except Exception as exc:  # noqa: BLE001
            print(f"[order_ledger] pass error (continuing): {exc}")
        _touch_ledger_hb(last_stats if isinstance(last_stats, dict) else {"status": "ok"})
        time.sleep(WORKER_POLL_SEC)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

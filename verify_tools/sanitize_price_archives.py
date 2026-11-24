#!/usr/bin/env python3
"""Streaming sanitizer for dense/sparse JSONL price archives (cache_365_price_*.jsonl).

Detects and removes isolated 1-tick spikes (and 2-tick extreme bursts) in O(1) RAM
using a per-symbol sliding window, atomically rewrites affected archives when ``--apply``
is passed, and invalidates stale sibling ``.npz`` caches so offline backtests rebuild cleanly.

Usage:
    python verify_tools/sanitize_price_archives.py --dry-run
    python verify_tools/sanitize_price_archives.py --apply --paths cachedb/prices/cache_365_price_ARBUSDC.jsonl
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
import glob
import os
import sys
from typing import Any

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from botcore import (
    add_store_true_cli_args,
    clean_scalar_str,
    parse_pos_cli_float,
    safe_pos_float,
    safe_pos_int,
)
from cachepaths import get_cachedb_dir, resolve_cachedb_filepath
from state_io import atomic_text_writer, iter_jsonl_retention_lines, iter_text_lines


def _is_isolated_spike(
    p_prev: float,
    p_cur: float,
    p_next: float,
    *,
    neighbor_agree_ratio: float,
    spike_ratio: float,
) -> bool:
    lo_nb = p_prev if p_prev <= p_next else p_next
    hi_nb = p_next if p_prev <= p_next else p_prev
    if lo_nb <= 0.0 or (hi_nb / lo_nb) > neighbor_agree_ratio:
        return False
    return p_cur > hi_nb * spike_ratio or p_cur < lo_nb / spike_ratio


def _is_two_tick_burst(
    p_left: float,
    p1: float,
    p2: float,
    p_right: float,
    *,
    neighbor_agree_ratio: float,
    edge_ratio: float,
) -> bool:
    lo_nb = p_left if p_left <= p_right else p_right
    hi_nb = p_right if p_left <= p_right else p_left
    if lo_nb <= 0.0 or (hi_nb / lo_nb) > neighbor_agree_ratio:
        return False
    return (p1 > hi_nb * edge_ratio or p1 < lo_nb / edge_ratio) and (
        p2 > hi_nb * edge_ratio or p2 < lo_nb / edge_ratio
    )


def find_isolated_outlier_lines(
    path: str,
    *,
    neighbor_agree_ratio: float = 1.20,
    spike_ratio: float = 1.5,
    edge_ratio: float = 2.5,
) -> tuple[int, list[dict[str, Any]]]:
    """Stream ``path`` in O(1) memory (single JSON parse per line) and return ``(total_lines, outlier_descriptors)``."""
    windows: dict[str, deque[tuple[int, int, float]]] = defaultdict(lambda: deque(maxlen=4))
    flagged: dict[int, dict[str, Any]] = {}
    state = {"line_no": 0}

    def _inspect_record(rec: dict[str, Any]) -> bool:
        line_no = state["line_no"] + 1
        state["line_no"] = line_no
        sym = clean_scalar_str(rec.get("s"))
        item = rec.get("i")
        if not sym or not isinstance(item, (list, tuple)) or len(item) < 2:
            return True
        ts_ms = safe_pos_int(item[0], default=None)
        px = safe_pos_float(item[1], default=None)
        if ts_ms is None or px is None:
            return True

        win = windows[sym]
        win.append((line_no, ts_ms, px))
        if len(win) >= 3:
            _l0, _t0, p0 = win[-3]
            l1, t1, p1 = win[-2]
            _l2, _t2, p2 = win[-1]
            if l1 not in flagged and _is_isolated_spike(
                p0,
                p1,
                p2,
                neighbor_agree_ratio=neighbor_agree_ratio,
                spike_ratio=spike_ratio,
            ):
                flagged[l1] = {
                    "line_no": l1,
                    "symbol": sym,
                    "ts_ms": t1,
                    "price": p1,
                    "prev_price": p0,
                    "next_price": p2,
                    "kind": "1tick_spike",
                }
        if len(win) == 4:
            _l0, _t0, p0 = win[0]
            l1, t1, p1 = win[1]
            l2, t2, p2 = win[2]
            _l3, _t3, p3 = win[3]
            if (
                l1 not in flagged
                and l2 not in flagged
                and _is_two_tick_burst(
                    p0,
                    p1,
                    p2,
                    p3,
                    neighbor_agree_ratio=neighbor_agree_ratio,
                    edge_ratio=edge_ratio,
                )
            ):
                flagged[l1] = {
                    "line_no": l1,
                    "symbol": sym,
                    "ts_ms": t1,
                    "price": p1,
                    "prev_price": p0,
                    "next_price": p3,
                    "kind": "2tick_burst",
                }
                flagged[l2] = {
                    "line_no": l2,
                    "symbol": sym,
                    "ts_ms": t2,
                    "price": p2,
                    "prev_price": p0,
                    "next_price": p3,
                    "kind": "2tick_burst",
                }
        return True

    total_lines = 0
    for total_lines, (_raw_line, _keep) in enumerate(iter_jsonl_retention_lines(path, _inspect_record), 1):
        if state["line_no"] < total_lines:
            state["line_no"] = total_lines

    outliers = [flagged[k] for k in sorted(flagged)]
    return total_lines, outliers


def sanitize_jsonl_price_archive(
    path: str,
    *,
    dry_run: bool = True,
    neighbor_agree_ratio: float = 1.20,
    spike_ratio: float = 1.5,
    edge_ratio: float = 2.5,
) -> dict[str, Any]:
    """Scan ``path`` and (when ``dry_run=False``) atomically remove flagged outlier lines + stale ``.npz``."""
    target = str(resolve_cachedb_filepath(path)) if not os.path.exists(path) else path
    if not os.path.isfile(target):
        return {"path": target, "exists": False, "total_lines": 0, "outliers": [], "rewritten": False}

    total_lines, outliers = find_isolated_outlier_lines(
        target,
        neighbor_agree_ratio=neighbor_agree_ratio,
        spike_ratio=spike_ratio,
        edge_ratio=edge_ratio,
    )
    removed_npz = False
    rewritten = False
    if outliers and not dry_run:
        drop_lines = {entry["line_no"] for entry in outliers}
        with atomic_text_writer(target, encoding="utf-8") as handle:
            for line_no, raw_line in enumerate(iter_text_lines(target, encoding="utf-8"), 1):
                if line_no in drop_lines:
                    continue
                handle.write(raw_line)
        rewritten = True
        if target.endswith(".jsonl"):
            npz_sibling = target[:-6] + ".npz"
            if os.path.isfile(npz_sibling):
                try:
                    os.remove(npz_sibling)
                    removed_npz = True
                except OSError:
                    pass
    return {
        "path": target,
        "exists": True,
        "total_lines": total_lines,
        "outliers": outliers,
        "rewritten": rewritten,
        "removed_npz": removed_npz,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sanitize isolated cross-symbol outlier ticks from JSONL price archives.")
    parser.add_argument("--paths", nargs="*", default=None, help="Explicit JSONL archive paths (defaults to cachedb/prices/cache_365_price_*.jsonl)")
    parser.add_argument("--spike-ratio", type=parse_pos_cli_float, default=1.5, help="Single-tick isolated outlier ratio threshold (default: 1.5)")
    parser.add_argument("--edge-ratio", type=parse_pos_cli_float, default=2.5, help="2-tick burst ratio threshold (default: 2.5)")
    add_store_true_cli_args(
        parser,
        ("--apply", "Rewrite affected archives atomically (omit for dry-run)"),
    )
    args = parser.parse_args(argv)

    if args.paths:
        paths = list(args.paths)
    else:
        prices_dir = os.path.join(get_cachedb_dir(), "prices")
        paths = sorted(glob.glob(os.path.join(prices_dir, "cache_365_price_*.jsonl")))

    total_flagged = 0
    for p in paths:
        res = sanitize_jsonl_price_archive(
            p,
            dry_run=not args.apply,
            spike_ratio=args.spike_ratio,
            edge_ratio=args.edge_ratio,
        )
        n_out = len(res["outliers"])
        total_flagged += n_out
        if n_out:
            action = "REWRITTEN" if res["rewritten"] else "DRY-RUN"
            print(f"[{action}] {res['path']}: lines={res['total_lines']} outliers={n_out} removed_npz={res.get('removed_npz', False)}")
            for item in res["outliers"][:10]:
                print(
                    f"  L{item['line_no']} {item['symbol']} ts={item['ts_ms']} "
                    f"px={item['price']} (prev={item['prev_price']}, next={item['next_price']}, {item['kind']})"
                )
        else:
            print(f"[CLEAN] {res['path']}: lines={res['total_lines']}")
    print(f"Done: scanned={len(paths)} files, total_outliers={total_flagged}, mode={'apply' if args.apply else 'dry-run'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

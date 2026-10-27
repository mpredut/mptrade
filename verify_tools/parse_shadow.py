#!/usr/bin/env python3
"""verify_tools/parse_shadow.py — Display latest forward/window shadow evaluation table."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _f(val, default: float = 0.0) -> float:
    if val is None or isinstance(val, bool):
        return default
    try:
        res = float(val)
    except (TypeError, ValueError):
        return default
    return res if math.isfinite(res) else default


def _i(val, default: int = 0) -> int:
    if val is None or isinstance(val, bool):
        return default
    try:
        fval = float(val)
    except (TypeError, ValueError):
        return default
    return int(fval) if math.isfinite(fval) else default


def _last_valid_record(lines: list[str]) -> dict | None:
    for raw in reversed(lines):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse and display latest shadow_live results")
    parser.add_argument(
        "--file",
        type=Path,
        default=ROOT / "logs" / "shadow_live" / "HYPEUSD_240m.jsonl",
        help="Path to shadow .jsonl log",
    )
    args = parser.parse_args()

    if not args.file.exists():
        print(f"Shadow log not found: {args.file}")
        return

    with open(args.file, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        print(f"Shadow log is empty: {args.file}")
        return

    last = _last_valid_record(lines)
    if last is None:
        print(f"Shadow log has no valid JSON records: {args.file}")
        return

    print("=== SHADOW 240m (4-Hour Bars) ===")
    print(f"Timestamp: {last.get('ts')}, Pair: {last.get('pair')}, Last Close: {last.get('last_close')}")
    fwd = last.get("forward")
    if isinstance(fwd, dict) and fwd:
        bars = _i(fwd.get("bars"), 0)
        print(f"FORWARD: {bars} bars ({bars * 4 / 24:.1f} days), Buy&Hold: {fwd.get('buyhold_pct')}%\n")
        print(f"  {'Configuration':<26} {'Net %':>8} {'Total %':>9} {'MaxDD %':>9} {'Cycles':>7}")
        print("  " + "-" * 63)
        fwd_cfgs = fwd.get("configs") if isinstance(fwd.get("configs"), dict) else {}
        for k, v in fwd_cfgs.items():
            if not isinstance(v, dict):
                continue
            print(f"  {k:<26} {_f(v.get('net_pct')):>8.2f} {_f(v.get('total_pct')):>9.2f} {_f(v.get('maxdd_pct')):>9.2f} {_i(v.get('cycles')):>7}")

    print("\n" + "=" * 65 + "\n")
    win = last.get("window")
    if isinstance(win, dict) and win:
        bars = _i(win.get("bars"), 0)
        print(f"FULL WINDOW (Kraken OHLC): {bars} bars ({bars * 4 / 24:.1f} days), Buy&Hold: {win.get('buyhold_pct')}%\n")
        print(f"  {'Configuration':<26} {'Net %':>8} {'Total %':>9} {'MaxDD %':>9} {'Cycles':>7}")
        print("  " + "-" * 63)
        win_cfgs = win.get("configs") if isinstance(win.get("configs"), dict) else {}
        for k, v in win_cfgs.items():
            if not isinstance(v, dict):
                continue
            print(f"  {k:<26} {_f(v.get('net_pct')):>8.2f} {_f(v.get('total_pct')):>9.2f} {_f(v.get('maxdd_pct')):>9.2f} {_i(v.get('cycles')):>7}")

    path_60m = ROOT / "logs" / "shadow_live" / "HYPEUSD_60m.jsonl"
    if path_60m.exists():
        try:
            with open(path_60m, encoding="utf-8") as f:
                lines60 = [line.strip() for line in f if line.strip()]
            last60 = _last_valid_record(lines60)
            if last60 is not None:
                print("\n" + "=" * 65 + "\n")
                print("=== SHADOW 60m (1-Hour Bars) ===")
                print(f"Timestamp: {last60.get('ts')}, Pair: {last60.get('pair')}, Last Close: {last60.get('last_close')}")
                fwd60 = last60.get("forward")
                if isinstance(fwd60, dict) and fwd60:
                    bars60 = _i(fwd60.get("bars"), 0)
                    print(f"FORWARD: {bars60} bars ({bars60 / 24:.1f} days), Buy&Hold: {fwd60.get('buyhold_pct')}%\n")
                    print(f"  {'Configuration':<26} {'Net %':>8} {'Total %':>9} {'MaxDD %':>9} {'Cycles':>7}")
                    print("  " + "-" * 63)
                    fwd60_cfgs = fwd60.get("configs") if isinstance(fwd60.get("configs"), dict) else {}
                    for k, v in fwd60_cfgs.items():
                        if not isinstance(v, dict):
                            continue
                        print(f"  {k:<26} {_f(v.get('net_pct')):>8.2f} {_f(v.get('total_pct')):>9.2f} {_f(v.get('maxdd_pct')):>9.2f} {_i(v.get('cycles')):>7}")
        except Exception as e:
            print("Error loading 60m:", e)


if __name__ == "__main__":
    main()

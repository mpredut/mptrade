#!/usr/bin/env python3
"""verify_tools/parse_shadow.py — Display latest forward/window shadow evaluation table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


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

    last = json.loads(lines[-1])

    print("=== SHADOW 240m (4-Hour Bars) ===")
    print(f"Timestamp: {last.get('ts')}, Pair: {last.get('pair')}, Last Close: {last.get('last_close')}")
    fwd = last.get("forward", {})
    if fwd:
        bars = fwd.get("bars", 0)
        print(f"FORWARD: {bars} bars ({bars * 4 / 24:.1f} days), Buy&Hold: {fwd.get('buyhold_pct')}%\n")
        print(f"  {'Configuration':<26} {'Net %':>8} {'Total %':>9} {'MaxDD %':>9} {'Cycles':>7}")
        print("  " + "-" * 63)
        for k, v in fwd.get("configs", {}).items():
            print(f"  {k:<26} {v.get('net_pct', 0.0):>8.2f} {v.get('total_pct', 0.0):>9.2f} {v.get('maxdd_pct', 0.0):>9.2f} {v.get('cycles', 0):>7}")

    print("\n" + "=" * 65 + "\n")
    win = last.get("window", {})
    if win:
        bars = win.get("bars", 0)
        print(f"FULL WINDOW (Kraken OHLC): {bars} bars ({bars * 4 / 24:.1f} days), Buy&Hold: {win.get('buyhold_pct')}%\n")
        print(f"  {'Configuration':<26} {'Net %':>8} {'Total %':>9} {'MaxDD %':>9} {'Cycles':>7}")
        print("  " + "-" * 63)
        for k, v in win.get("configs", {}).items():
            print(f"  {k:<26} {v.get('net_pct', 0.0):>8.2f} {v.get('total_pct', 0.0):>9.2f} {v.get('maxdd_pct', 0.0):>9.2f} {v.get('cycles', 0):>7}")

    path_60m = ROOT / "logs" / "shadow_live" / "HYPEUSD_60m.jsonl"
    if path_60m.exists():
        try:
            with open(path_60m, encoding="utf-8") as f:
                lines60 = [line.strip() for line in f if line.strip()]
            if lines60:
                last60 = json.loads(lines60[-1])
                print("\n" + "=" * 65 + "\n")
                print("=== SHADOW 60m (1-Hour Bars) ===")
                print(f"Timestamp: {last60.get('ts')}, Pair: {last60.get('pair')}, Last Close: {last60.get('last_close')}")
                fwd60 = last60.get("forward", {})
                if fwd60:
                    bars60 = fwd60.get("bars", 0)
                    print(f"FORWARD: {bars60} bars ({bars60 / 24:.1f} days), Buy&Hold: {fwd60.get('buyhold_pct')}%\n")
                    print(f"  {'Configuration':<26} {'Net %':>8} {'Total %':>9} {'MaxDD %':>9} {'Cycles':>7}")
                    print("  " + "-" * 63)
                    for k, v in fwd60.get("configs", {}).items():
                        print(f"  {k:<26} {v.get('net_pct', 0.0):>8.2f} {v.get('total_pct', 0.0):>9.2f} {v.get('maxdd_pct', 0.0):>9.2f} {v.get('cycles', 0):>7}")
        except Exception as e:
            print("Error loading 60m:", e)


if __name__ == "__main__":
    main()

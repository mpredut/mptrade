#!/usr/bin/env python3
"""verify_tools/parse_shadow.py — Display latest forward/window shadow evaluation table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from botcore import safe_float as _f, safe_int as _i


def _last_valid_record(lines: list[str]) -> dict | None:
    for raw in reversed(lines):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _print_section(label: str, section: object, hours_per_bar: float) -> None:
    if not isinstance(section, dict) or not section:
        return
    bars = _i(section.get("bars"), 0)
    print(f"{label}: {bars} bars ({bars * hours_per_bar / 24:.1f} days), Buy&Hold: {section.get('buyhold_pct')}%\n")
    print(f"  {'Configuration':<26} {'Net %':>8} {'Total %':>9} {'MaxDD %':>9} {'Cycles':>7}")
    print("  " + "-" * 63)
    cfgs = section.get("configs") if isinstance(section.get("configs"), dict) else {}
    for k, v in cfgs.items():
        if not isinstance(v, dict):
            continue
        print(f"  {k:<26} {_f(v.get('net_pct')):>8.2f} {_f(v.get('total_pct')):>9.2f} {_f(v.get('maxdd_pct')):>9.2f} {_i(v.get('cycles')):>7}")


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
    _print_section("FORWARD", last.get("forward"), 4.0)

    print("\n" + "=" * 65 + "\n")
    _print_section("FULL WINDOW (Kraken OHLC)", last.get("window"), 4.0)

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
                _print_section("FORWARD", last60.get("forward"), 1.0)
        except Exception as e:
            print("Error loading 60m:", e)


if __name__ == "__main__":
    main()

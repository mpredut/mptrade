#!/usr/bin/env python3
"""Canonical unified walk-forward runner dispatcher.

Provides a single CLI entry point for walk-forward baseline evaluation and
candidate comparison across all supported venues:
  - Kraken (spot crypto)
  - Binance (spot crypto)
  - Hyperliquid (perpetual crypto)
  - Trading 212 / t212 (equity/stock CFDs)

Usage:
  # Baseline mode (default):
  python offline/runners/walk_forward.py --venue kraken [kraken_baseline_options...]
  python offline/runners/walk_forward.py --venue binance [binance_baseline_options...]
  python offline/runners/walk_forward.py --venue hyperliquid [hl_baseline_options...]
  python offline/runners/walk_forward.py --venue t212 [t212_baseline_options...]

  # Compare mode:
  python offline/runners/walk_forward.py --mode compare <baseline_report.json> [compare_options...]
  python offline/runners/walk_forward.py --venue kraken --mode compare <baseline_report.json> [compare_options...]
"""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys

RUNNER_DIR = Path(__file__).resolve().parent
ROOT = RUNNER_DIR.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(RUNNER_DIR) not in sys.path:
    sys.path.append(str(RUNNER_DIR))

SUPPORTED_VENUES = ("kraken", "binance", "hyperliquid", "trading212", "t212")
SUPPORTED_MODES = ("baseline", "compare")

VENUE_RUNNER_MODULES = {
    "kraken": "offline.runners.kraken_walk_forward_baseline",
    "binance": "offline.runners.binance_walk_forward_baseline",
    "hyperliquid": "offline.runners.hyperliquid_walk_forward_baseline",
    "t212": "offline.runners.t212_walk_forward_baseline",
    "trading212": "offline.runners.t212_walk_forward_baseline",
}


def canonical_venue(venue: str | None) -> str | None:
    if venue is None or isinstance(venue, bool):
        return None
    normalized = str(getattr(venue, "name", venue) or "").strip().lower()
    if not normalized:
        return None
    if normalized == "trading212":
        return "t212"
    return normalized


def run_baseline(venue: str, argv: list[str] | None = None) -> int:
    """Dispatch to the canonical baseline runner for the given venue."""
    norm_venue = canonical_venue(venue)
    if not norm_venue or norm_venue not in VENUE_RUNNER_MODULES:
        raise ValueError(
            f"Unsupported baseline venue {venue!r}. Supported venues: {', '.join(sorted(VENUE_RUNNER_MODULES))}"
        )
    mod_name = VENUE_RUNNER_MODULES[norm_venue]
    runner_mod = importlib.import_module(mod_name)
    runner_main = getattr(runner_mod, "main")
    return runner_main(argv or [])


def run_compare(argv: list[str] | None = None, venue: str | None = None) -> int:
    """Dispatch to the canonical multi-venue compare runner."""
    from offline.runners import walk_forward_compare

    forward_args = list(argv or [])
    norm_venue = canonical_venue(venue)
    if norm_venue and not any(a == "--venue" or a.startswith("--venue=") for a in forward_args):
        forward_args.extend(["--venue", norm_venue])
    return walk_forward_compare.main(forward_args)


def dispatch(argv: list[str] | None = None) -> int:
    """Parse top-level dispatcher options and route to the corresponding runner."""
    raw_args = list(sys.argv[1:] if argv is None else argv)

    # Pre-parse --venue and --mode while leaving all other flags intact
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--venue",
        type=lambda s: str(s).strip().lower(),
        choices=SUPPORTED_VENUES,
        default=None,
    )
    pre_parser.add_argument(
        "--mode",
        type=lambda s: str(s).strip().lower(),
        choices=SUPPORTED_MODES,
        default="baseline",
    )

    # If top-level help is requested without venue/mode, display dispatcher help
    if any(h in raw_args for h in ("-h", "--help")):
        # Check if user specifically passed --venue or --mode to see venue-specific help
        known, _ = pre_parser.parse_known_args(raw_args)
        if known.venue is None and known.mode == "baseline":
            doc_help = (
                "usage: walk_forward.py [--venue {kraken,binance,hyperliquid,t212}] "
                "[--mode {baseline,compare}] ...\n\n"
                "Canonical unified walk-forward runner dispatcher across all venues.\n\n"
                "options:\n"
                "  --venue {kraken,binance,hyperliquid,trading212,t212}\n"
                "                        target trading venue (required for baseline mode)\n"
                "  --mode {baseline,compare}\n"
                "                        execution mode (defaults to 'baseline')\n\n"
                "Examples:\n"
                "  python offline/runners/walk_forward.py --venue kraken --intervals 240\n"
                "  python offline/runners/walk_forward.py --venue binance --symbol BTCUSDT\n"
                "  python offline/runners/walk_forward.py --venue hyperliquid --coin HYPE\n"
                "  python offline/runners/walk_forward.py --venue t212 --profile nvda\n"
                "  python offline/runners/walk_forward.py --mode compare path/to/report.json\n"
            )
            print(doc_help)
            return 0

    known, remaining_args = pre_parser.parse_known_args(raw_args)

    if known.mode == "baseline":
        if not known.venue:
            pre_parser.error(
                "--venue is required in baseline mode. Choose from: "
                f"{', '.join(sorted(set(VENUE_RUNNER_MODULES.keys())))}"
            )
        return run_baseline(known.venue, remaining_args)

    if known.mode == "compare":
        return run_compare(remaining_args, venue=known.venue)

    pre_parser.error(f"Unknown mode: {known.mode}")
    return 1


def main(argv: list[str] | None = None) -> int:
    return dispatch(argv)


if __name__ == "__main__":
    raise SystemExit(main())

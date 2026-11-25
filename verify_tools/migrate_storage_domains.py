#!/usr/bin/env python3
"""CLI cutover tool migrating workspace storage into Variant A ``cachedb/`` (regenerable) + ``statedb/`` (durable).

Safe by default: runs in ``--dry-run`` preview mode unless ``--apply`` is explicitly passed.
Run Pasul 2B cutover ONLY after all daemons are stopped (``orchestrator`` paused) and updated to
dual-layout Pasul 2A code.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Sequence

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from botcore import add_store_true_cli_args
from cachepaths import organize_storage_domains
from state_io import encode_json_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Organize workspace storage into pure Variant A domains: "
            "cachedb/ (regenerable caches & exchange mirrors) vs statedb/ (irreplaceable durable state)."
        )
    )
    parser.add_argument(
        "--workspace-dir",
        default=_ROOT_DIR,
        help="Workspace root containing cachedb/ and statedb/ (default: repo root).",
    )
    parser.add_argument(
        "--cachedb-dir",
        default=None,
        help="Optional explicit cachedb directory override.",
    )
    parser.add_argument(
        "--statedb-dir",
        default=None,
        help="Optional explicit statedb directory override.",
    )
    add_store_true_cli_args(
        parser,
        ("--apply", "Execute moves/renames on disk (default is dry-run preview)."),
        ("--keep-stale-staging", "Do not clean stale .tmp/.previous.* staging files during cutover."),
        ("--json", "Emit machine-readable JSON summary."),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    dry_run = not bool(args.apply)
    moved = organize_storage_domains(
        workspace_dir=args.workspace_dir,
        cachedb_dir=args.cachedb_dir,
        statedb_dir=args.statedb_dir,
        remove_stale_staging=not bool(args.keep_stale_staging),
        dry_run=dry_run,
    )
    total_moved = sum(len(items) for items in moved.values())
    summary = {
        "mode": "dry_run" if dry_run else "applied",
        "workspace_dir": os.path.abspath(args.workspace_dir),
        "total_files": total_moved,
        "by_domain": {k: sorted(v) for k, v in moved.items() if v},
    }
    if args.json:
        print(encode_json_payload(summary, indent=2, sort_keys=True))
    else:
        header = "DRY-RUN PREVIEW (pass --apply to execute)" if dry_run else "APPLIED VARIANT A STORAGE CUTOVER"
        print(f"=== {header} ===")
        print(f"Workspace: {summary['workspace_dir']}")
        print(f"Total files to move/rename: {total_moved}" if dry_run else f"Total files moved/renamed: {total_moved}")
        for domain, files in sorted(summary["by_domain"].items()):
            print(f"  [{domain}] ({len(files)}): {', '.join(files[:12])}{' ...' if len(files) > 12 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

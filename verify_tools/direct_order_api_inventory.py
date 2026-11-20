#!/usr/bin/env python3
"""Static gate for direct venue submit/cancel calls outside the common lifecycle.

Approved entries are narrow mechanics or venue adapters. A new call site fails the
gate until its ownership, persistence and recovery behavior are reviewed explicitly.
"""
from __future__ import annotations

import ast
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from state_io import encode_json_payload, read_text_file
from verify_tools.ownership_inventory import create_root_cli_parser

SENSITIVE_METHODS = {
    "create_order", "order_limit_buy", "order_limit_sell",
    "order_market_buy", "order_market_sell", "add_order", "spot_order",
    "place_limit_order", "place_market_order", "cancel_order",
}
EXCLUDED_PARTS = {"archive", "offline", "tests", "verify_tools", ".venv", "myenv"}


@dataclass(frozen=True, order=True)
class DirectCall:
    path: str
    function: str
    method: str


# Reviewed low-level boundaries. Strategy modules are present only for venue-specific
# mechanics whose financial state is persisted by that same owner before the call.
APPROVED_BOUNDARIES = {
    "212trading/order_manager.py": {"place_limit_order"},
    "binance_api/bapi.py": {"cancel_order"},
    "binance_api/bapi_placeorder.py": {
        "order_limit_buy", "order_limit_sell",
        "order_market_buy", "order_market_sell",
    },
    "monitororder.py": {"cancel_order"},
    "orchestratorOS/lib/technical_repair_ssot.py": {"cancel_order"},
    "providers/binance_provider.py": {"cancel_order"},
    "providers/hyperliquid_provider.py": {"spot_order"},
    "providers/kraken_provider.py": {"add_order", "cancel_order"},
    "providers/market_api.py": {"cancel_order"},
    "providers/t212_provider.py": {
        "place_limit_order", "place_market_order", "cancel_order",
    },
}


class _Visitor(ast.NodeVisitor):
    def __init__(self, relative: str):
        self.relative = relative
        self.functions: list[str] = []
        self.calls: set[DirectCall] = set()

    def visit_FunctionDef(self, node):
        self.functions.append(node.name)
        self.generic_visit(node)
        self.functions.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Attribute(self, node):
        method = node.attr
        if method == "cancel_order":
            receiver = ast.unparse(node.value)
            if (receiver in {"self.client", "mkt", "market_api"}
                    or "executor" in receiver or "market_api" in receiver):
                self.generic_visit(node)
                return
        if method in SENSITIVE_METHODS:
            self.calls.add(DirectCall(
                self.relative, self.functions[-1] if self.functions else "<module>", method,
            ))
        self.generic_visit(node)


def _is_direct_order_scan_target(relative_path: Path) -> bool:
    return not (
        set(relative_path.parts) & EXCLUDED_PARTS
        or any(part.startswith(".") for part in relative_path.parts)
        or relative_path.name.startswith("test_")
        or relative_path.name.endswith("_test.py")
    )


_WALK_SKIP_DIRS = EXCLUDED_PARTS | {"__pycache__", "venv", "site-packages", "cachedb", "logs", "media", "models"}


def scan_direct_calls(
    root: Path = ROOT,
    rel_paths: list[str] | tuple[str, ...] | None = None,
    *,
    _parsed_cache: dict | None = None,
) -> set[DirectCall]:
    result: set[DirectCall] = set()
    if rel_paths is None:
        candidates: list[Path] = []
        for cur_dir, dirnames, filenames in os.walk(str(root)):
            dirnames[:] = sorted(
                d for d in dirnames
                if not d.startswith(".") and d not in _WALK_SKIP_DIRS
            )
            for fname in sorted(filenames):
                if fname.endswith(".py"):
                    candidates.append(Path(cur_dir, fname).relative_to(root))
    else:
        candidates = [Path(r) for r in rel_paths if r.endswith(".py") and (root / r).is_file()]
    for relative_path in candidates:
        if not _is_direct_order_scan_target(relative_path):
            continue
        rel_s = relative_path.as_posix()
        visitor = _Visitor(rel_s)
        tree = None
        if _parsed_cache is not None and rel_s in _parsed_cache:
            tree = _parsed_cache[rel_s][1]
        if tree is None:
            try:
                tree = ast.parse(read_text_file(root / relative_path))
            except (OSError, SyntaxError, UnicodeError):
                continue
        visitor.visit(tree)
        result.update(visitor.calls)
    return result


def audit_direct_order_boundaries(
    root: Path = ROOT,
    rel_paths: list[str] | tuple[str, ...] | None = None,
    *,
    _parsed_cache: dict | None = None,
) -> list[str]:
    """Return human-readable violations for unapproved direct order calls (and stale allowlist on full scan)."""
    found = scan_direct_calls(root, rel_paths=rel_paths, _parsed_cache=_parsed_cache)
    violations: list[str] = []
    for call in sorted(found):
        if call.method not in APPROVED_BOUNDARIES.get(call.path, set()):
            violations.append(
                f"{call.path}: {call.function}() calls unapproved direct venue order API .{call.method}()"
            )
    if rel_paths is None:
        found_boundaries = {(call.path, call.method) for call in found}
        for path, methods in sorted(APPROVED_BOUNDARIES.items()):
            for method in sorted(methods):
                if (path, method) not in found_boundaries:
                    violations.append(f"{path}: stale direct-order allowlist entry .{method}()")
    return violations


def inventory(root: Path = ROOT) -> dict:
    found = scan_direct_calls(root)
    approved = {
        call for call in found
        if call.method in APPROVED_BOUNDARIES.get(call.path, set())
    }
    found_boundaries = {(call.path, call.method) for call in found}
    configured_boundaries = {
        (path, method) for path, methods in APPROVED_BOUNDARIES.items()
        for method in methods
    }
    return {
        "approved": [asdict(call) for call in sorted(approved)],
        "unapproved": [asdict(call) for call in sorted(found - approved)],
        "stale_allowlist": [
            {"path": path, "method": method}
            for path, method in sorted(configured_boundaries - found_boundaries)
        ],
    }


def main() -> int:
    parser = create_root_cli_parser(__doc__, ROOT)
    args = parser.parse_args()
    report = inventory(args.root.resolve())
    print(encode_json_payload(report, indent=2, sort_keys=True))
    return 1 if report["unapproved"] or report["stale_allowlist"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

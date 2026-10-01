#!/usr/bin/env python3
"""Static gate for direct venue submit/cancel calls outside the common lifecycle.

Approved entries are narrow mechanics or venue adapters. A new call site fails the
gate until its ownership, persistence and recovery behavior are reviewed explicitly.
"""
from __future__ import annotations

import argparse
import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
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
        "cancel_order", "order_limit_buy", "order_limit_sell",
        "order_market_buy", "order_market_sell",
    },
    "monitororder.py": {"cancel_order"},
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


def scan_direct_calls(root: Path = ROOT) -> set[DirectCall]:
    result: set[DirectCall] = set()
    for path in root.rglob("*.py"):
        relative_path = path.relative_to(root)
        if (set(relative_path.parts) & EXCLUDED_PARTS
                or any(part.startswith(".") for part in relative_path.parts)
                or path.name.startswith("test_") or path.name.endswith("_test.py")):
            continue
        visitor = _Visitor(relative_path.as_posix())
        try:
            visitor.visit(ast.parse(path.read_text(encoding="utf-8")))
        except (OSError, SyntaxError, UnicodeError):
            continue
        result.update(visitor.calls)
    return result


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    report = inventory(args.root.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report["unapproved"] or report["stale_allowlist"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

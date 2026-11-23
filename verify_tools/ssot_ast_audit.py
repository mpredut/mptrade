#!/usr/bin/env python3
"""Canonical SSOT AST purity, DAG, layering, and policy auditor.

Enforces repository-wide static AST invariants for both pytest policy suites
and the unified git pre-commit gate (`verify_tools/pre_commit_gate.py`).

Subcommands
  pre-commit [files...]  Audit staged Python files (or explicit paths).
  scan                   Audit the entire repository.
"""
from __future__ import annotations

import argparse
import ast
import builtins
from collections import defaultdict
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from state_io import read_text_file
from verify_tools.translation_audit import PROD_PYTHON_SKIP_TOP, find_first_non_ascii_line

ParsedCache = dict[str, tuple[str | None, ast.Module | None, str | None]]

JSON_OWNERS = frozenset({"state_io.py", "botcore.py", "cacheintegrity.py"})
TEXT_OWNERS = frozenset({"state_io.py", "botcore.py"})
ENV_OWNERS = frozenset({
    "botcore.py",
    "cachepaths.py",
    "credentials.py",
    "core/config.py",
    "core/heartbeat_runtime.py",
    "core/llm_topics.py",
})
OPEN_OWNERS = frozenset({"state_io.py", "botcore.py", "core/lock.py", "log.py"})
COERCION_OWNERS = frozenset({"botcore.py", "cachepaths.py", "orchestratorOS/agy/protobuf.py"})
CONFIGPARSER_OWNERS = frozenset({"botcore.py", "core/config.py"})
HTTP_OWNERS = frozenset({"botcore.py", "core/http.py"})
INVALID_SCALAR_OWNERS = frozenset({"botcore.py", "cachepaths.py", "core/math_coercion.py"})
QUOTE_ENDSWITH_OWNERS = frozenset({
    "cachepaths.py",
    "instrument_registry.py",
    "intelligence/sentiment/collectors/market_breadth_collector.py",
})
QUOTE_SUFFIX_TOKENS = frozenset({"USDC", "USDT", "USD", "EUR", "FDUSD", "TUSD", "BUSD"})
EQUITY_SUFFIX_TOKENS = frozenset({"_EQ", "_US_EQ"})
_US_EQ_SUFFIX = "_US_EQ"
_OHLC_24H_PREFIX = "cache_ohlc_24h_"
CACHEPRICE_POLLER_ALIASES = frozenset({
    "NonBinanceTrendPoller",
    "VenuePriceStreamBase",
    "fetch_price_with_deadline",
    "start_nonbinance_trend_poller",
})
VENUE_LEAF_DIRS = ("binance_api/", "kraken/", "hyperliquid/", "212trading/", "market_monitor/")
VENUE_TOKENS = frozenset({"binance", "kraken", "hyperliquid", "t212", "trading212"})
VENUE_LITERAL_SUBSTRINGS = ("binance", "kraken", "hyperliquid", "t212", "212trading")
VENUE_BRANCH_EXCLUDED_TOP_DIRS = frozenset({
    "binance_api",
    "kraken",
    "hyperliquid",
    "212trading",
    "tests",
    "offline",
    "verify_tools",
})
VENUE_LEAF_ADAPTERS = frozenset({
    "providers/binance_provider.py",
    "providers/kraken_provider.py",
    "providers/hyperliquid_provider.py",
    "providers/t212_provider.py",
    "providers/replay_provider.py",
})
VENUE_BRANCH_EXTRA_GUARDED = frozenset({"verify_tools/check_cache_coherence.py"})
EQUITY_ENDSWITH_EXEMPT_PREFIXES = ("tests/", "offline/", ".venv/", "myenv/")
REEXPORT_EXTRA_DIRS = ("offline", "verify_tools")
_BUILTIN_NAMES = frozenset(dir(builtins)) | frozenset({
    "__file__", "__name__", "__package__", "__doc__", "__spec__", "__annotations__",
})

# -- Layer DAG & isolation constants -------------------------------------------

LAYER2_MODULES = frozenset({"cachepaths.py", "state_io.py", "cacheintegrity.py", "pricewindow.py"})
FORBIDDEN_LAYER2_IMPORTS = frozenset({
    "account_cache_health",
    "binance_cache_health",
    "cacheInterface",
    "cacheaccount",
    "cacheprice",
    "cachetrend",
    "cacheManager",
    "providers",
    "instrument",
    "order_guard",
    "strategies",
    "intelligence",
    "shadow_signals",
    "binance_api",
    "kraken",
    "hyperliquid",
    "212trading",
})
CACHE_MODULES = frozenset({
    "cacheInterface.py",
    "cacheaccount.py",
    "cacheprice.py",
    "cachetrend.py",
    "cacheManager.py",
})
CACHE_SIBLING_ROOTS = frozenset({"cacheaccount", "cacheprice", "cachetrend"})
FORBIDDEN_CACHE_UPWARD = frozenset({
    "binance_cache_health",
    "instrument",
    "order_guard",
    "guard_hierarchy",
    "order_lifecycle",
    "order_retry",
    "strategies",
    "trailing_core",
    "intelligence",
    "shadow_signals",
    "orchestratorOS",
    "orchestratorTrade",
    "binance_api",
    "kraken",
    "hyperliquid",
    "212trading",
})
FORBIDDEN_CACHE_EXACT_PROVIDERS = frozenset({
    "providers.binance_provider",
    "providers.kraken_provider",
    "providers.hyperliquid_provider",
    "providers.t212_provider",
})
PROVIDER_BASE_MODULES = frozenset({
    "providers/base.py",
    "providers/ws_stream.py",
    "providers/ssot_bridge.py",
    "providers/venue_metadata.py",
    "providers/ohlc_utils.py",
    "providers/price_poller.py",
})
FORBIDDEN_PROVIDER_BASE_IMPORTS = frozenset({
    "order_guard",
    "guard_hierarchy",
    "llm_order_guard",
    "order_lifecycle",
    "order_retry",
    "order_retry_worker",
    "instrument",
    "strategies",
    "market_regime",
    "intelligence",
    "orchestratorTrade",
    "orchestratorOS",
    "cacheManager",
    "binance_api",
    "kraken",
    "hyperliquid",
    "212trading",
})
PROVIDER_NON_FACADE_MODULES = frozenset({
    "providers/base.py",
    "providers/ws_stream.py",
    "providers/ssot_bridge.py",
    "providers/venue_metadata.py",
    "providers/venue_ws_protocols.py",
    "providers/broker_history.py",
    "providers/price_poller.py",
    "providers/binance_provider.py",
    "providers/hyperliquid_provider.py",
    "providers/kraken_provider.py",
    "providers/t212_provider.py",
    "providers/replay_provider.py",
    "providers/order_id_context.py",
    "providers/ohlc_utils.py",
    "providers/order_ohlc_utils.py",
    "providers/quantity.py",
    "providers/strategy_executor.py",
    "providers/execution_audit.py",
    "providers/intent_state.py",
    "providers/intent_execution.py",
    "providers/intent_reconciliation.py",
    "providers/intent_telemetry.py",
})
FORBIDDEN_PROVIDER_UPWARD = frozenset({
    "cacheInterface",
    "cacheaccount",
    "cacheprice",
    "cachetrend",
    "cacheManager",
    "market_regime",
    "order_guard",
    "guard_hierarchy",
    "llm_order_guard",
    "order_lifecycle",
    "order_retry",
    "order_retry_worker",
    "instrument",
    "strategies",
    "intelligence",
    "orchestratorTrade",
    "orchestratorOS",
})
VENUE_TRANSPORT_MODULES = frozenset({
    "binance_api/bapi.py",
    "binance_api/bapi_allorders.py",
    "binance_api/bapi_client.py",
    "binance_api/bapi_placeorder.py",
    "binance_api/bapi_trades.py",
    "binance_api/bapi_ws.py",
    "binance_api/bapi_ws_bridge.py",
    "binance_api/bapi_ws_normalize.py",
    "binance_api/market_data.py",
    "212trading/t212_client.py",
    "212trading/market_data.py",
    "kraken/kraken_client.py",
    "kraken/market_data.py",
    "hyperliquid/hl_client.py",
    "hyperliquid/market_data.py",
})
FORBIDDEN_TRANSPORT_PROVIDER_TARGETS = frozenset({
    "providers.market_api",
    "providers.binance_provider",
    "providers.kraken_provider",
    "providers.hyperliquid_provider",
    "providers.t212_provider",
    "providers.replay_provider",
})
ORDER_GATEWAY_MODULES = frozenset({
    "instrument.py",
    "order_guard.py",
    "order_lifecycle.py",
    "order_retry.py",
    "order_retry_worker.py",
})
FORBIDDEN_GATEWAY_VENUE_SDKS = frozenset({
    "binance_api", "binance", "kraken", "hyperliquid", "212trading", "hl_client",
})
FORBIDDEN_VENUE_SDK_ROOTS = frozenset({"binance", "binance_api", "kraken", "hyperliquid", "212trading"})
VENUE_SDK_EXEMPT_PREFIXES = (
    "binance_api/",
    "kraken/",
    "hyperliquid/",
    "212trading/",
    "providers/",
    "tests/",
    "archive/",
    ".venv/",
    "myenv/",
)
VENUE_SDK_EXEMPT_FILES = frozenset({"verify_tools/verify_account_facade.py"})
STRATEGIES_FORBIDDEN_IMPORTS = frozenset({
    "binance_api", "binance", "kraken", "hyperliquid", "212trading",
    "hl_client", "orchestratorOS", "orchestratorTrade",
    "monitortrades", "assetguardian", "rtrade", "tradeall",
})
STRATEGIES_PURE_VENUE_FILES = frozenset({
    "strategies/base_spot_mixin.py",
    "strategies/mt_position_accounting.py",
    "strategies/position_exit_manager.py",
    "strategies/rtrade_pair.py",
    "strategies/spot_replay.py",
    "strategies/spot_rules.py",
    "strategies/state_store.py",
})
CACHE_PRODUCER_GETTERS = frozenset({
    "get_cache_manager",
    "get_current_price_manager",
    "get_24_price_manager",
    "get_sparse_price_manager",
    "get_short_trend_manager",
    "get_long_trend_manager",
    "get_asset_value_manager",
    "get_order_filled_manager",
    "get_order_open_manager",
    "get_trade_manager",
    "attach_default_price_ws_stream",
})
CACHE_LEGACY_KWARGS = frozenset({"start_sync", "sync_to_file", "mode", "writer"})
CACHE_WS_SYNC_OWNERS = frozenset({"cacheaccount.py", "binance_api/bapi_ws_bridge.py"})
CACHE_PRODUCER_SKIP_DIRS = frozenset({
    "myenv", "venv", "__pycache__", "tests", "cachedb", "logs", "logger", "offline", "archive", "media", "models",
})
_TMP_WRITER_RE = re.compile(
    r"\b(?:tmp|tmp_file|tmp_path|tmp_bin|temporary)\s*=.*(?:\.tmp|mkstemp)"
)
_LOCAL_ATOMIC_DEF_RE = re.compile(
    r"^\s*def\s+(?:_?atomic_write(?:_json)?|_write_json_atomic)\b",
    re.MULTILINE,
)


def _norm_rel_path(path: Path, root: Path) -> str | None:
    if not path.is_absolute():
        return path.as_posix().lstrip("./")
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return None


def _is_prod_python_rel(rel_s: str) -> bool:
    if not rel_s.endswith(".py"):
        return False
    parts = Path(rel_s).parts
    return not any(
        part in PROD_PYTHON_SKIP_TOP or part.startswith(".") or part == "site-packages"
        for part in parts
    )


def _rel_to_mod(rel_path: str) -> str:
    return (
        rel_path[:-12].replace("/", ".")
        if rel_path.endswith("/__init__.py")
        else rel_path[:-3].replace("/", ".")
    )


def _resolve_import_target(rel_path: str, level: int, module: str | None) -> str:
    base_dir = "/".join(rel_path.split("/")[:-1])
    base_pkg = base_dir.replace("/", ".")
    if level == 0 or not base_pkg:
        return module or ""
    parts = base_pkg.split(".")
    keep = parts[: len(parts) - (level - 1)] if level > 1 else parts
    prefix = ".".join(keep)
    return f"{prefix}.{module}" if (module and prefix) else (module or prefix)


def _extract_reexport_defs(tree: ast.Module) -> set[str]:
    def_set: set[str] = set()
    imp_set: set[str] = set()
    for top_n in tree.body:
        if isinstance(top_n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            def_set.add(top_n.name)
        elif isinstance(top_n, ast.Assign):
            for tgt in top_n.targets:
                if isinstance(tgt, ast.Name):
                    def_set.add(tgt.id)
                    if tgt.id == "_LAZY_COMPAT_EXPORTS" and isinstance(top_n.value, ast.Dict):
                        for k_node in top_n.value.keys:
                            if isinstance(k_node, ast.Constant) and isinstance(k_node.value, str):
                                imp_set.add(k_node.value)
        elif isinstance(top_n, ast.AnnAssign) and isinstance(top_n.target, ast.Name):
            def_set.add(top_n.target.id)
            if top_n.target.id == "_LAZY_COMPAT_EXPORTS" and isinstance(top_n.value, ast.Dict):
                for k_node in top_n.value.keys:
                    if isinstance(k_node, ast.Constant) and isinstance(k_node.value, str):
                        imp_set.add(k_node.value)
        elif isinstance(top_n, ast.ImportFrom) and top_n.module:
            for al in top_n.names:
                if al.name != "*":
                    imp_set.add(al.asname or al.name)
    return imp_set - def_set


def _detect_venv_roots(root: Path) -> set[str]:
    if not root.is_dir():
        return set()
    return {
        d.name
        for d in root.iterdir()
        if d.is_dir() and (d.name in ("venv", ".venv", "myenv") or (d / "pyvenv.cfg").exists())
    }


def _is_venue_branch_guarded(rel_s: str, excluded_top_dirs: set[str]) -> bool:
    if not rel_s.endswith(".py"):
        return False
    if rel_s in VENUE_BRANCH_EXTRA_GUARDED:
        return True
    parts = Path(rel_s).parts
    if not parts:
        return False
    if (
        any(part.startswith(".") or part in ("__pycache__", "site-packages") for part in parts)
        or parts[0] in excluded_top_dirs
        or rel_s in VENUE_LEAF_ADAPTERS
    ):
        return False
    return True


def _is_equity_endswith_guarded(rel_s: str) -> bool:
    if not rel_s.endswith(".py") or rel_s == "cachepaths.py":
        return False
    if rel_s.startswith(EQUITY_ENDSWITH_EXEMPT_PREFIXES):
        return False
    parts = Path(rel_s).parts
    return not any(
        part.startswith(".") or part in ("__pycache__", "site-packages")
        for part in parts
    )


_WALK_PRUNE_DIRS = frozenset({
    "__pycache__", "venv", "myenv", "site-packages", "cachedb", "logs", "media", "models",
})


def _iter_repo_python_rels(root: Path) -> list[str]:
    rels: list[str] = []
    for cur_dir, dirnames, filenames in os.walk(str(root)):
        dirnames[:] = sorted(
            d for d in dirnames
            if not d.startswith(".") and d not in _WALK_PRUNE_DIRS
        )
        for fname in sorted(filenames):
            if fname.endswith(".py"):
                rels.append(Path(cur_dir, fname).relative_to(root).as_posix())
    return rels


def _normalize_target_rels(root: Path, rel_paths: Sequence[str] | None) -> list[str]:
    if rel_paths is None:
        return _iter_repo_python_rels(root)
    seen: set[str] = set()
    result: list[str] = []
    for raw in rel_paths:
        rel_s = _norm_rel_path(Path(raw), root)
        if not rel_s or rel_s in seen:
            continue
        full = root / rel_s
        if full.is_file():
            seen.add(rel_s)
            result.append(rel_s)
    return result


def parse_module_cached(
    path: Path,
    rel_s: str,
    cache: ParsedCache | None = None,
) -> tuple[str | None, ast.Module | None, str | None]:
    if cache is not None and rel_s in cache:
        return cache[rel_s]
    src_text = read_text_file(str(path), default=None)
    if src_text is None:
        res: tuple[str | None, ast.Module | None, str | None] = (None, None, None)
    else:
        try:
            res = (src_text, ast.parse(src_text, filename=rel_s), None)
        except SyntaxError as exc:
            res = (src_text, None, f"{rel_s}:{exc.lineno or 1}: syntax error: {exc.msg}")
    if cache is not None:
        cache[rel_s] = res
    return res


def _docstring_nodes(tree: ast.Module) -> set[ast.AST]:
    nodes: set[ast.AST] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                nodes.add(node.body[0].value)
    return nodes


def _imported_roots_and_full(tree: ast.Module, *, include_dynamic: bool = False) -> tuple[set[str], set[str]]:
    roots: set[str] = set()
    fulls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
                fulls.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
            fulls.add(node.module)
            for alias in node.names:
                fulls.add(f"{node.module}.{alias.name}")
        elif (
            include_dynamic
            and isinstance(node, ast.Call)
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            if (
                isinstance(node.func, ast.Attribute) and node.func.attr == "import_module"
            ) or (
                isinstance(node.func, ast.Name) and node.func.id in ("import_module", "__import__")
            ):
                val = node.args[0].value
                roots.add(val.split(".")[0])
                fulls.add(val)
    return roots, fulls


# -- Audit 1: Repo-wide SSOT primitives, ASCII, undefined names, re-exports ----

def audit_repo_ssot_ast(
    root: Path = _ROOT,
    rel_paths: Sequence[str] | None = None,
    *,
    _parsed_cache: ParsedCache | None = None,
) -> list[str]:
    """Audit production Python modules (and extra dirs for transitive re-exports) for SSOT AST purity."""
    root = Path(root)
    target_rels = _normalize_target_rels(root, rel_paths)
    prod_rels = [r for r in target_rels if _is_prod_python_rel(r)]
    reexport_extra_rels = [
        r for r in target_rels
        if not _is_prod_python_rel(r)
        and r.endswith(".py")
        and "__pycache__" not in r
        and any(r.startswith(f"{d}/") for d in REEXPORT_EXTRA_DIRS)
    ]

    parsed_trees: dict[str, ast.Module] = {}
    reexport_cache: dict[str, set[str] | None] = {}
    ssot_violations: list[str] = []

    for rel_s in prod_rels:
        src_text, tree, syntax_err = parse_module_cached(root / rel_s, rel_s, _parsed_cache)
        if src_text is not None:
            bad_ln = find_first_non_ascii_line(src_text)
            if bad_ln is not None:
                ssot_violations.append(
                    f"{rel_s}:{bad_ln}: non-ASCII source character (use \\uXXXX/\\UXXXXXXXX escapes in runtime strings)"
                )
        if syntax_err is not None:
            ssot_violations.append(syntax_err)
            continue
        if tree is None:
            continue
        parsed_trees[rel_s] = tree
        mod_name = _rel_to_mod(rel_s)
        if rel_s.endswith("/__init__.py"):
            reexport_cache[mod_name] = None
        else:
            reexport_cache[mod_name] = _extract_reexport_defs(tree)

    def _target_reexport_defs(target_mod: str) -> set[str] | None:
        if not target_mod or target_mod == "botcore":
            return None
        if target_mod in reexport_cache:
            return reexport_cache[target_mod]
        mod_parts = target_mod.split(".")
        if root.joinpath(*mod_parts, "__init__.py").is_file():
            reexport_cache[target_mod] = None
            return None
        mod_path = root.joinpath(*mod_parts).with_suffix(".py")
        if not mod_path.is_file():
            reexport_cache[target_mod] = None
            return None
        mod_rel = mod_path.relative_to(root).as_posix()
        if not _is_prod_python_rel(mod_rel):
            reexport_cache[target_mod] = None
            return None
        _, mod_tree, _ = parse_module_cached(mod_path, mod_rel, _parsed_cache)
        reexports = _extract_reexport_defs(mod_tree) if mod_tree is not None else None
        reexport_cache[target_mod] = reexports
        return reexports

    def _check_reexport_imports(rel_s: str, node: ast.ImportFrom) -> None:
        lineno = getattr(node, "lineno", 0)
        if node.module == "providers.base" or (
            rel_s.startswith("providers/") and node.level == 1 and node.module == "base"
        ):
            if rel_s.startswith(VENUE_LEAF_DIRS):
                ssot_violations.append(f"{rel_s}:{lineno}: upward layer import from providers.base")
            if bad_names := [a.name for a in node.names if a.name != "MarketDataProvider"]:
                ssot_violations.append(f"{rel_s}:{lineno}: legacy re-export import from providers.base: {bad_names}")
        elif node.module == "cacheprice" and (
            bad_cp := [a.name for a in node.names if a.name in CACHEPRICE_POLLER_ALIASES]
        ):
            ssot_violations.append(f"{rel_s}:{lineno}: non-canonical poller re-export import from cacheprice: {bad_cp}")
        if node.module:
            target_mod = _resolve_import_target(rel_s, node.level, node.module)
            target_reexports = _target_reexport_defs(target_mod)
            if target_reexports and (bad_reexp := [a.name for a in node.names if a.name in target_reexports]):
                ssot_violations.append(
                    f"{rel_s}:{lineno}: non-canonical transitive re-export import from {target_mod}: {bad_reexp}"
                )

    for rel_s in prod_rels:
        tree = parsed_trees.get(rel_s)
        if tree is None:
            continue
        bound_names = set(_BUILTIN_NAMES)
        ann_nodes: set[ast.AST] = set()
        has_wildcard_imp = False
        for w_node in ast.walk(tree):
            if isinstance(w_node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bound_names.add(w_node.name)
            if isinstance(w_node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                for arg_n in (
                    *w_node.args.posonlyargs,
                    *w_node.args.args,
                    *w_node.args.kwonlyargs,
                    w_node.args.vararg,
                    w_node.args.kwarg,
                ):
                    if arg_n is not None:
                        bound_names.add(arg_n.arg)
                        if getattr(arg_n, "annotation", None) is not None:
                            ann_nodes.update(ast.walk(arg_n.annotation))
                if getattr(w_node, "returns", None) is not None:
                    ann_nodes.update(ast.walk(w_node.returns))
            elif isinstance(w_node, (ast.Import, ast.ImportFrom)):
                for al in w_node.names:
                    if al.name == "*":
                        has_wildcard_imp = True
                    else:
                        bound_names.add((al.asname or al.name).split(".")[0])
            elif isinstance(w_node, ast.Name) and isinstance(w_node.ctx, (ast.Store, ast.Del)):
                bound_names.add(w_node.id)
            elif isinstance(w_node, ast.NamedExpr) and isinstance(w_node.target, ast.Name):
                bound_names.add(w_node.target.id)
            elif isinstance(w_node, ast.ExceptHandler) and w_node.name:
                bound_names.add(w_node.name)
            elif isinstance(w_node, ast.MatchAs) and w_node.name:
                bound_names.add(w_node.name)
            elif isinstance(w_node, ast.AnnAssign) and w_node.annotation is not None:
                ann_nodes.update(ast.walk(w_node.annotation))
        if not has_wildcard_imp:
            for w_node in ast.walk(tree):
                if (
                    isinstance(w_node, ast.Name)
                    and isinstance(w_node.ctx, ast.Load)
                    and w_node not in ann_nodes
                    and w_node.id not in bound_names
                ):
                    ssot_violations.append(f"{rel_s}:{w_node.lineno}: undefined name {w_node.id!r}")

        seen_top_import_from: set[tuple[int, str]] = set()
        for top_node in tree.body:
            if isinstance(top_node, ast.ImportFrom) and top_node.module:
                imp_key = (top_node.level, top_node.module)
                if imp_key in seen_top_import_from:
                    ssot_violations.append(f"{rel_s}:{top_node.lineno}: duplicate top-level ImportFrom {top_node.module}")
                seen_top_import_from.add(imp_key)

        for node in ast.walk(tree):
            lineno = getattr(node, "lineno", 0)
            if isinstance(node, ast.Call):
                fn = node.func
                fn_name = fn.id if isinstance(fn, ast.Name) else (fn.attr if isinstance(fn, ast.Attribute) else "")
                owner_mod = fn.value.id if (isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name)) else ""
                if isinstance(fn, ast.Name):
                    if fn_name == "open" and rel_s not in OPEN_OWNERS:
                        ssot_violations.append(f"{rel_s}:{lineno}: raw open()")
                    elif fn_name in ("float", "int") and rel_s not in COERCION_OWNERS:
                        ssot_violations.append(f"{rel_s}:{lineno}: raw {fn_name}()")
                if rel_s not in JSON_OWNERS and owner_mod == "json" and fn_name in ("load", "loads", "dump", "dumps"):
                    ssot_violations.append(f"{rel_s}:{lineno}: raw json.{fn_name}")
                elif rel_s not in TEXT_OWNERS and isinstance(fn, ast.Attribute) and fn_name in ("read_text", "write_text"):
                    ssot_violations.append(f"{rel_s}:{lineno}: Path.{fn_name}")
                elif rel_s not in CONFIGPARSER_OWNERS and fn_name == "ConfigParser":
                    ssot_violations.append(f"{rel_s}:{lineno}: ConfigParser")
                elif rel_s not in HTTP_OWNERS and fn_name in ("urlopen", "Request"):
                    ssot_violations.append(f"{rel_s}:{lineno}: urllib {fn_name}")
                elif rel_s not in HTTP_OWNERS and owner_mod == "requests" and fn_name in ("get", "post", "put", "delete", "request"):
                    ssot_violations.append(f"{rel_s}:{lineno}: raw requests.{fn_name}")
                elif rel_s not in QUOTE_ENDSWITH_OWNERS and isinstance(fn, ast.Attribute) and fn_name == "endswith" and node.args:
                    ew_arg = node.args[0]
                    ew_nodes = ew_arg.elts if isinstance(ew_arg, (ast.Tuple, ast.List)) else (ew_arg,)
                    ew_vals = [e.value for e in ew_nodes if isinstance(e, ast.Constant) and isinstance(e.value, str)]
                    if any(v in QUOTE_SUFFIX_TOKENS for v in ew_vals):
                        ssot_violations.append(f"{rel_s}:{lineno}: manual quote .endswith({ew_vals})")
            elif (
                rel_s not in ENV_OWNERS
                and isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "os"
                and node.attr in ("environ", "getenv")
            ):
                ssot_violations.append(f"{rel_s}:{lineno}: os.{node.attr}")
            elif isinstance(node, ast.ImportFrom):
                _check_reexport_imports(rel_s, node)
            elif isinstance(node, ast.Import) and rel_s.startswith(VENUE_LEAF_DIRS):
                for alias in node.names:
                    if alias.name == "providers.base":
                        ssot_violations.append(f"{rel_s}:{lineno}: upward layer import providers.base")
            elif (
                rel_s != "cachepaths.py"
                and isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value.startswith(_OHLC_24H_PREFIX)
            ):
                ssot_violations.append(f"{rel_s}:{lineno}: hardcoded cache_ohlc_24h_ filename prefix")
            elif rel_s not in INVALID_SCALAR_OWNERS and (
                (isinstance(node, ast.Name) and node.id == "INVALID_SCALAR_STRINGS")
                or (isinstance(node, ast.Attribute) and node.attr == "INVALID_SCALAR_STRINGS")
            ):
                ssot_violations.append(
                    f"{rel_s}:{lineno}: direct INVALID_SCALAR_STRINGS use (call clean_scalar_str / clean_upper_scalar)"
                )
            elif (
                rel_s != "cachepaths.py"
                and isinstance(node, ast.JoinedStr)
                and node.values
                and isinstance(node.values[-1], ast.Constant)
                and isinstance(node.values[-1].value, str)
                and node.values[-1].value.endswith(_US_EQ_SUFFIX)
            ):
                ssot_violations.append(
                    f"{rel_s}:{lineno}: manual f-string _US_EQ concatenation (call ensure_equity_suffix)"
                )

    for rel_s in reexport_extra_rels:
        _, tree, syntax_err = parse_module_cached(root / rel_s, rel_s, _parsed_cache)
        if syntax_err is not None:
            ssot_violations.append(syntax_err)
            continue
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                _check_reexport_imports(rel_s, node)

    return ssot_violations


# -- Audit 2: Venue string branches --------------------------------------------

def audit_venue_string_branches_ast(
    root: Path = _ROOT,
    rel_paths: Sequence[str] | None = None,
    *,
    _parsed_cache: ParsedCache | None = None,
) -> list[str]:
    """Audit guarded non-venue-adapter modules for hardcoded venue string comparisons."""
    root = Path(root)
    excluded_top_dirs = set(VENUE_BRANCH_EXCLUDED_TOP_DIRS) | _detect_venv_roots(root)
    if rel_paths is None:
        guarded = [r for r in _iter_repo_python_rels(root) if _is_venue_branch_guarded(r, excluded_top_dirs)]
        for extra in sorted(VENUE_BRANCH_EXTRA_GUARDED):
            if (root / extra).is_file() and extra not in guarded:
                guarded.insert(0, extra)
    else:
        guarded = [
            r for r in _normalize_target_rels(root, rel_paths)
            if _is_venue_branch_guarded(r, excluded_top_dirs)
        ]

    violations: list[str] = []
    for rel in guarded:
        _, tree, syntax_err = parse_module_cached(root / rel, rel, _parsed_cache)
        if syntax_err is not None:
            violations.append(syntax_err)
            continue
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare) and any(
                isinstance(op, (ast.Eq, ast.NotEq, ast.In, ast.NotIn)) for op in node.ops
            ):
                for sub in (node.left, *node.comparators):
                    if isinstance(sub, ast.Constant) and isinstance(sub.value, str) and sub.value.strip().lower() in VENUE_TOKENS:
                        violations.append(f"{rel}:{node.lineno}: hardcoded venue branch {sub.value!r}")
                    elif isinstance(sub, (ast.Tuple, ast.List, ast.Set)):
                        for elt in sub.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str) and elt.value.strip().lower() in VENUE_TOKENS:
                                violations.append(f"{rel}:{node.lineno}: hardcoded venue branch {elt.value!r}")
    return violations


# -- Audit 3: Manual equity suffix .endswith() ---------------------------------

def audit_equity_suffix_endswith_ast(
    root: Path = _ROOT,
    rel_paths: Sequence[str] | None = None,
    *,
    _parsed_cache: ParsedCache | None = None,
) -> list[str]:
    """Audit Python modules outside cachepaths.py for manual _EQ/_US_EQ .endswith() calls."""
    root = Path(root)
    target_rels = [
        r for r in _normalize_target_rels(root, rel_paths)
        if _is_equity_endswith_guarded(r)
    ]
    violations: list[str] = []
    for rel in target_rels:
        _, tree, syntax_err = parse_module_cached(root / rel, rel, _parsed_cache)
        if syntax_err is not None:
            violations.append(syntax_err)
            continue
        if tree is None:
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "endswith"
                and node.args
            ):
                arg = node.args[0]
                matched = [
                    sub.value
                    for sub in ast.walk(arg)
                    if isinstance(sub, ast.Constant)
                    and isinstance(sub.value, str)
                    and sub.value in EQUITY_SUFFIX_TOKENS
                ]
                if matched:
                    violations.append(f"{rel}:{node.lineno}: manual equity .endswith({ast.unparse(arg)})")
    return violations


# -- Audit 4: Strict acyclic layer DAG, venue SDK & shim isolation -------------

def audit_venue_sdk_imports_ast(
    root: Path = _ROOT,
    rel_paths: Sequence[str] | None = None,
    *,
    _parsed_cache: ParsedCache | None = None,
) -> list[str]:
    """Audit zero direct or dynamic venue SDK imports outside venue adapters and providers/."""
    root = Path(root)
    target_rels = _normalize_target_rels(root, rel_paths)
    dynamic_exempt = VENUE_SDK_EXEMPT_PREFIXES + ("offline/",)
    violations: list[str] = []
    for rel in target_rels:
        if not rel.endswith(".py"):
            continue
        if any(part.startswith(".") for part in Path(rel).parts):
            continue
        if rel.startswith(VENUE_SDK_EXEMPT_PREFIXES) or rel in VENUE_SDK_EXEMPT_FILES:
            continue
        _, tree, syntax_err = parse_module_cached(root / rel, rel, _parsed_cache)
        if syntax_err is not None:
            violations.append(syntax_err)
            continue
        if tree is None:
            continue
        check_dynamic = not rel.startswith(dynamic_exempt)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".", 1)[0] in FORBIDDEN_VENUE_SDK_ROOTS:
                        violations.append(f"{rel}:{node.lineno}: direct venue SDK import {alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module.split(".", 1)[0] in FORBIDDEN_VENUE_SDK_ROOTS:
                    violations.append(f"{rel}:{node.lineno}: direct venue SDK import from {node.module}")
            elif check_dynamic and isinstance(node, ast.Call):
                fn_name = (
                    node.func.id
                    if isinstance(node.func, ast.Name)
                    else (node.func.attr if isinstance(node.func, ast.Attribute) else "")
                )
                if fn_name in ("load_local_module", "load_venue_module", "import_module"):
                    for sub in ast.walk(node):
                        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                            parts = set(sub.value.replace("/", ".").replace("\\", ".").split("."))
                            if parts & FORBIDDEN_VENUE_SDK_ROOTS:
                                violations.append(
                                    f"{rel}:{node.lineno}: dynamic venue SDK {fn_name}(... {sub.value!r} ...)"
                                )
    return violations


def audit_persistence_and_cache_dag_ast(
    root: Path = _ROOT,
    rel_paths: Sequence[str] | None = None,
    *,
    _parsed_cache: ParsedCache | None = None,
) -> list[str]:
    """Audit Layer 2 persistence, shadow_signals, and Layer 3 cache DAG, sibling purity, and WS delegation."""
    root = Path(root)
    watched = LAYER2_MODULES | {"account_cache_health.py", "shadow_signals.py"} | CACHE_MODULES
    if rel_paths is None:
        target_rels = [r for r in sorted(watched) if (root / r).is_file()]
    else:
        target_rels = [r for r in _normalize_target_rels(root, rel_paths) if r in watched]
    violations: list[str] = []
    for rel in target_rels:
        src, tree, syntax_err = parse_module_cached(root / rel, rel, _parsed_cache)
        if syntax_err is not None:
            violations.append(syntax_err)
            continue
        if tree is None or src is None:
            continue
        roots, fulls = _imported_roots_and_full(tree, include_dynamic=True)
        if rel in LAYER2_MODULES:
            if leak := sorted(roots & FORBIDDEN_LAYER2_IMPORTS):
                violations.append(f"{rel}: upward layer leak {leak}")
        elif rel == "account_cache_health.py":
            if leak := sorted(roots & (FORBIDDEN_LAYER2_IMPORTS - {"account_cache_health"})):
                violations.append(f"{rel}: upward layer leak {leak}")
        elif rel == "shadow_signals.py":
            if "intelligence" in roots:
                violations.append("shadow_signals.py: upward leak into intelligence")
        elif rel in CACHE_MODULES:
            if "wss://" in src:
                violations.append(f"{rel}: hardcoded venue wss:// URL (delegate to providers/)")
            if leak := sorted(roots & FORBIDDEN_CACHE_UPWARD):
                violations.append(f"{rel}: upward layer/venue leak {leak}")
            if prov_leak := sorted(fulls & FORBIDDEN_CACHE_EXACT_PROVIDERS):
                violations.append(f"{rel}: concrete venue provider import {prov_leak}")
            stem = rel[:-3]
            if stem in CACHE_SIBLING_ROOTS:
                if sib_leak := sorted(roots & (CACHE_SIBLING_ROOTS - {stem})):
                    violations.append(f"{rel}: sibling cache domain import {sib_leak}")
                if "cacheManager" in roots:
                    violations.append(f"{rel}: sibling cache must not import cacheManager")
    return violations


def audit_layer_dag_ast(
    root: Path = _ROOT,
    rel_paths: Sequence[str] | None = None,
    *,
    _parsed_cache: ParsedCache | None = None,
) -> list[str]:
    """Audit all acyclic architecture layer boundaries, venue SDK isolation, strategies/trailing purity, and removed shims."""
    root = Path(root)
    target_rels = _normalize_target_rels(root, rel_paths)
    violations: list[str] = []
    violations.extend(audit_persistence_and_cache_dag_ast(root, rel_paths=target_rels, _parsed_cache=_parsed_cache))
    violations.extend(audit_venue_sdk_imports_ast(root, rel_paths=target_rels, _parsed_cache=_parsed_cache))

    for rel in target_rels:
        if not rel.endswith(".py"):
            continue
        parts = Path(rel).parts
        if any(part.startswith(".") or part in ("__pycache__", "site-packages") for part in parts):
            continue

        needs_check = (
            rel in PROVIDER_BASE_MODULES
            or rel in PROVIDER_NON_FACADE_MODULES
            or rel == "providers/market_api.py"
            or rel in VENUE_TRANSPORT_MODULES
            or rel in ORDER_GATEWAY_MODULES
            or rel == "intelligence/globaltelemetry_collector.py"
            or rel == "trailing_core.py"
            or rel.startswith("strategies/")
            or _is_prod_python_rel(rel)
            or (parts[0] not in ("archive", "tests"))
        )
        if not needs_check:
            continue

        src, tree, syntax_err = parse_module_cached(root / rel, rel, _parsed_cache)
        if syntax_err is not None or tree is None or src is None:
            continue
        roots, fulls = _imported_roots_and_full(tree, include_dynamic=False)

        if rel in PROVIDER_BASE_MODULES:
            if leak := sorted(roots & FORBIDDEN_PROVIDER_BASE_IMPORTS):
                violations.append(f"{rel}: provider base DAG leak {leak}")
        if rel in PROVIDER_NON_FACADE_MODULES:
            if leak := sorted(roots & FORBIDDEN_PROVIDER_UPWARD):
                violations.append(f"{rel}: provider upward DAG leak {leak}")
        if rel == "providers/market_api.py":
            if leak := sorted(roots & (FORBIDDEN_PROVIDER_UPWARD - {"market_regime"})):
                violations.append(f"{rel}: MarketApi upward DAG leak {leak}")
        if rel in VENUE_TRANSPORT_MODULES:
            if leak := sorted(roots & (FORBIDDEN_PROVIDER_UPWARD | {"offline"})):
                violations.append(f"{rel}: transport upward DAG leak {leak}")
            if t_leak := sorted(fulls & FORBIDDEN_TRANSPORT_PROVIDER_TARGETS):
                violations.append(f"{rel}: transport-to-adapter DAG leak {t_leak}")
        if rel in ORDER_GATEWAY_MODULES:
            if leak := sorted(roots & FORBIDDEN_GATEWAY_VENUE_SDKS):
                violations.append(f"{rel}: order gateway venue SDK leak {leak}")
        if rel == "intelligence/globaltelemetry_collector.py":
            if leak := sorted(roots & {"orchestratorTrade", "orchestratorOS"}):
                violations.append(f"{rel}: upward orchestrator import {leak}")
        if rel.startswith("strategies/"):
            if "_exit_manager_mode" in src:
                violations.append(f"{rel}: forbidden legacy _exit_manager_mode reference")
            if leak := sorted(roots & STRATEGIES_FORBIDDEN_IMPORTS):
                violations.append(f"{rel}: strategies DAG leak {leak}")
            if rel in STRATEGIES_PURE_VENUE_FILES:
                docs = _docstring_nodes(tree)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Constant) and isinstance(node.value, str) and node not in docs:
                        low = node.value.lower()
                        if any(tok in low for tok in VENUE_LITERAL_SUBSTRINGS):
                            violations.append(f"{rel}:{node.lineno}: hardcoded venue literal {node.value!r}")
        if rel == "trailing_core.py":
            docs = _docstring_nodes(tree)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        r_low = alias.name.split(".")[0].lower()
                        if any(v in r_low for v in VENUE_LITERAL_SUBSTRINGS):
                            violations.append(f"trailing_core.py:{node.lineno}: forbidden venue import {alias.name}")
                elif isinstance(node, ast.ImportFrom) and node.module:
                    r_low = node.module.split(".")[0].lower()
                    if any(v in r_low for v in VENUE_LITERAL_SUBSTRINGS):
                        violations.append(f"trailing_core.py:{node.lineno}: forbidden venue import from {node.module}")
                elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node not in docs:
                    low = node.value.lower()
                    if any(v in low for v in VENUE_LITERAL_SUBSTRINGS):
                        violations.append(f"trailing_core.py:{node.lineno}: hardcoded venue literal {node.value!r}")
        if _is_prod_python_rel(rel) and "offline" in roots:
            violations.append(f"{rel}: production module imports offline.*")
        if parts[0] not in ("archive", "tests") and "providers.tracked_order" in fulls:
            violations.append(f"{rel}: removed compatibility shim providers.tracked_order")

    return violations


def audit_removed_shim_imports_ast(
    root: Path = _ROOT,
    rel_paths: Sequence[str] | None = None,
    *,
    _parsed_cache: ParsedCache | None = None,
) -> list[str]:
    """Audit zero active AST imports of removed compatibility shims (e.g., providers.tracked_order)."""
    root = Path(root)
    violations: list[str] = []
    for rel in _normalize_target_rels(root, rel_paths):
        parts = Path(rel).parts
        if not rel.endswith(".py") or not parts or parts[0] in ("archive", "tests"):
            continue
        _, tree, _ = parse_module_cached(root / rel, rel, _parsed_cache)
        if tree is None:
            continue
        _, fulls = _imported_roots_and_full(tree, include_dynamic=True)
        if "providers.tracked_order" in fulls:
            violations.append(f"{rel}: removed compatibility shim providers.tracked_order")
    return violations


# -- Audit 5: Import cycles (lazy module DFS in pre-commit, Tarjan SCC in scan) -

def _extract_local_import_targets(
    rel_s: str,
    tree: ast.Module,
    mod_exists,
) -> list[str]:
    pkg_parts = Path(rel_s).parts[:-1]
    targets: list[str] = []
    for node in ast.walk(tree):
        raw_tgts: list[str] = []
        if isinstance(node, ast.Import):
            for a in node.names:
                raw_tgts.append(a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                raw_tgts.append(node.module)
                for a in node.names:
                    sub = f"{node.module}.{a.name}"
                    if mod_exists(sub):
                        raw_tgts.append(sub)
            elif node.level > 0:
                base_parts = list(pkg_parts[: len(pkg_parts) - (node.level - 1)])
                if node.module:
                    base_parts.extend(node.module.split("."))
                rel_mod = ".".join(base_parts)
                if node.module and rel_mod:
                    raw_tgts.append(rel_mod)
                for a in node.names:
                    sub = f"{rel_mod}.{a.name}" if rel_mod else a.name
                    if mod_exists(sub):
                        raw_tgts.append(sub)
        for tgt in raw_tgts:
            if tgt.startswith(("hyperliquid.info", "hyperliquid.exchange", "hyperliquid.utils")):
                continue
            targets.append(tgt)
    return targets


def _find_sccs(graph: dict[str, set[str]]) -> list[list[str]]:
    index = 0
    indices: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    sccs: list[list[str]] = []

    def _strongconnect(v: str) -> None:
        nonlocal index
        indices[v] = index
        lowlink[v] = index
        index += 1
        stack.append(v)
        on_stack.add(v)
        for w in sorted(graph.get(v, ())):
            if w not in indices:
                _strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], indices[w])
        if lowlink[v] == indices[v]:
            scc: list[str] = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                scc.append(w)
                if w == v:
                    break
            if len(scc) > 1:
                sccs.append(sorted(scc))

    for v in sorted(graph):
        if v not in indices:
            _strongconnect(v)
    return sccs


def audit_import_cycles_ast(
    root: Path = _ROOT,
    rel_paths: Sequence[str] | None = None,
    *,
    _parsed_cache: ParsedCache | None = None,
) -> list[str]:
    """Detect circular imports: lazy DFS from staged modules in pre-commit, full Tarjan SCC in repo scan."""
    root = Path(root)
    excluded_top_dirs = {"tests", "offline", "verify_tools"} | _detect_venv_roots(root)

    def _subpkg_id(mod: str) -> str:
        parts = mod.split(".")
        return ".".join(parts[:2]) if len(parts) >= 2 else parts[0]

    def _is_scc_member(rel_s: str) -> bool:
        if not rel_s.endswith(".py"):
            return False
        parts = Path(rel_s).parts
        return not (
            not parts
            or parts[0] in excluded_top_dirs
            or any(part.startswith(".") or part in ("__pycache__", "site-packages") for part in parts)
        )

    if rel_paths is not None:
        # Fast incremental module-level cycle detection lazily expanding only reachable local imports.
        staged_rels = [r for r in _normalize_target_rels(root, rel_paths) if _is_scc_member(r)]
        if not staged_rels:
            return []

        mod_path_cache: dict[str, str | None] = {}

        def _resolve_mod_rel(mod: str) -> str | None:
            if mod in mod_path_cache:
                return mod_path_cache[mod]
            if not mod or mod.split(".")[0] in excluded_top_dirs:
                mod_path_cache[mod] = None
                return None
            parts = mod.split(".")
            cand_py = root.joinpath(*parts).with_suffix(".py")
            if cand_py.is_file():
                rel_c = cand_py.relative_to(root).as_posix()
                mod_path_cache[mod] = rel_c if _is_scc_member(rel_c) else None
                return mod_path_cache[mod]
            cand_init = root.joinpath(*parts, "__init__.py")
            if cand_init.is_file():
                rel_c = cand_init.relative_to(root).as_posix()
                mod_path_cache[mod] = rel_c if _is_scc_member(rel_c) else None
                return mod_path_cache[mod]
            mod_path_cache[mod] = None
            return None

        edges_cache: dict[str, set[str]] = {}

        def _neighbors(mod: str) -> set[str]:
            if mod in edges_cache:
                return edges_cache[mod]
            rel_m = _resolve_mod_rel(mod)
            if rel_m is None:
                edges_cache[mod] = set()
                return set()
            _, tree, _ = parse_module_cached(root / rel_m, rel_m, _parsed_cache)
            if tree is None:
                edges_cache[mod] = set()
                return set()
            nbrs: set[str] = set()
            for tgt in _extract_local_import_targets(rel_m, tree, lambda m: _resolve_mod_rel(m) is not None):
                parts = tgt.split(".")
                for i in range(len(parts), 0, -1):
                    cand = ".".join(parts[:i])
                    if _resolve_mod_rel(cand) is not None:
                        if cand != mod:
                            nbrs.add(cand)
                        break
            edges_cache[mod] = nbrs
            return nbrs

        violations: list[str] = []
        seen_cycles: set[tuple[str, ...]] = set()
        state: dict[str, int] = {}
        path_stack: list[str] = []

        def _dfs(u: str) -> None:
            state[u] = 1
            path_stack.append(u)
            for v in sorted(_neighbors(u)):
                st = state.get(v, 0)
                if st == 1:
                    cyc = path_stack[path_stack.index(v):]
                    key = tuple(sorted(cyc))
                    if key not in seen_cycles:
                        seen_cycles.add(key)
                        violations.append(f"circular module import cycle: {' -> '.join([*cyc, v])}")
                elif st == 0:
                    _dfs(v)
            path_stack.pop()
            state[u] = 2

        for rel_s in staged_rels:
            mod = _rel_to_mod(rel_s)
            if state.get(mod, 0) == 0:
                _dfs(mod)
        return violations

    # Full-repository Tarjan SCC across module, subpackage, and top-level package graphs.
    all_mods: dict[str, str] = {}
    for rel_s in _iter_repo_python_rels(root):
        if _is_scc_member(rel_s):
            all_mods[_rel_to_mod(rel_s)] = rel_s

    pkg_edges: dict[str, set[str]] = defaultdict(set)
    subpkg_edges: dict[str, set[str]] = defaultdict(set)
    mod_edges: dict[str, set[str]] = defaultdict(set)
    for mod_name, rel_s in sorted(all_mods.items()):
        src_top = Path(rel_s).parts[0].removesuffix(".py")
        src_subpkg = _subpkg_id(mod_name)
        _, tree, _ = parse_module_cached(root / rel_s, rel_s, _parsed_cache)
        if tree is None:
            continue
        for tgt in _extract_local_import_targets(rel_s, tree, lambda m: m in all_mods):
            t_top = tgt.split(".")[0]
            if (
                t_top != src_top
                and t_top not in excluded_top_dirs
                and ((root / f"{t_top}.py").exists() or (root / t_top).is_dir())
            ):
                pkg_edges[src_top].add(t_top)
            parts = tgt.split(".")
            for i in range(len(parts), 0, -1):
                cand = ".".join(parts[:i])
                if cand in all_mods:
                    if cand != mod_name:
                        mod_edges[mod_name].add(cand)
                    t_subpkg = _subpkg_id(cand)
                    if t_subpkg != src_subpkg:
                        subpkg_edges[src_subpkg].add(t_subpkg)
                    break

    violations = []
    for cyc in _find_sccs(pkg_edges):
        violations.append(f"top-level package SCC cycle: {cyc}")
    for cyc in _find_sccs(subpkg_edges):
        violations.append(f"subpackage SCC cycle: {cyc}")
    for cyc in _find_sccs(mod_edges):
        violations.append(f"module SCC cycle: {cyc}")
    return violations


# -- Audit 6: Producer-only cache ownership ------------------------------------

def audit_producer_only_caches_ast(
    root: Path = _ROOT,
    rel_paths: Sequence[str] | None = None,
    *,
    _parsed_cache: ParsedCache | None = None,
) -> list[str]:
    """Audit that only cacheManager.py enables producer cache mode and zero legacy cache kwargs remain."""
    root = Path(root)
    target_rels = _normalize_target_rels(root, rel_paths)
    violations: list[str] = []
    for rel_s in target_rels:
        if not rel_s.endswith(".py") or rel_s == "cacheManager.py":
            continue
        parts = Path(rel_s).parts
        if any(part.startswith(".") or part in CACHE_PRODUCER_SKIP_DIRS for part in parts):
            continue
        _, tree, syntax_err = parse_module_cached(root / rel_s, rel_s, _parsed_cache)
        if syntax_err is not None or tree is None:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func_name: str | None = None
            is_cache_factory_get = False
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
                if func_name == "get" and (
                    (isinstance(node.func.value, ast.Name) and node.func.value.id == "CacheFactory")
                    or (isinstance(node.func.value, ast.Attribute) and node.func.value.attr == "CacheFactory")
                ):
                    is_cache_factory_get = True
            if func_name in ("enable_real_ws_event_sync", "start_account_cache_daemon") and rel_s not in CACHE_WS_SYNC_OWNERS:
                violations.append(f"{rel_s}:{node.lineno}: forbidden consumer cache WS/daemon call {func_name}()")
            if is_cache_factory_get or func_name in CACHE_PRODUCER_GETTERS:
                for kw in node.keywords:
                    if kw.arg in CACHE_LEGACY_KWARGS:
                        violations.append(f"{rel_s}:{node.lineno}: {func_name} legacy kwarg {kw.arg}")
                    elif (
                        kw.arg == "data_sync_mode"
                        and isinstance(kw.value, ast.Constant)
                        and str(kw.value.value).strip().lower() == "producer"
                    ):
                        violations.append(f"{rel_s}:{node.lineno}: {func_name} data_sync_mode='producer'")
    return violations


# -- Audit 7: Zero bespoke atomic writers outside state_io.py ------------------

def audit_bespoke_atomic_writers(
    root: Path = _ROOT,
    rel_paths: Sequence[str] | None = None,
    *,
    _parsed_cache: ParsedCache | None = None,
) -> list[str]:
    """Audit that zero bespoke mkstemp/tmp+os.replace or _atomic_write helpers exist outside state_io.py."""
    root = Path(root)
    target_rels = _normalize_target_rels(root, rel_paths)
    violations: list[str] = []
    for rel_s in target_rels:
        if (
            not rel_s.endswith(".py")
            or rel_s == "state_io.py"
            or rel_s.startswith(("tests/", "offline/", "archive/"))
            or "/test_" in rel_s
        ):
            continue
        if _parsed_cache is not None and rel_s in _parsed_cache:
            text = _parsed_cache[rel_s][0]
        else:
            text = read_text_file(str(root / rel_s), default=None)
        if not text:
            continue
        owns_tmp = bool(_TMP_WRITER_RE.search(text))
        defines_local = bool(_LOCAL_ATOMIC_DEF_RE.search(text))
        if (owns_tmp and ("os.replace" in text or "mkstemp" in text)) or defines_local:
            violations.append(f"{rel_s}: bespoke atomic file writer (delegate to state_io.py)")
    return violations


# -- Combined SSOT AST audit runner --------------------------------------------

def audit_all_ssot_ast(
    root: Path = _ROOT,
    rel_paths: Sequence[str] | None = None,
    *,
    _parsed_cache: ParsedCache | None = None,
) -> list[str]:
    """Run all SSOT AST audits (primitives, venue branches, equity suffixes, DAG, cycles, producers, atomic writers)."""
    root = Path(root)
    target_rels = _normalize_target_rels(root, rel_paths) if rel_paths is not None else None
    parsed_cache: ParsedCache = _parsed_cache if _parsed_cache is not None else {}
    combined: list[str] = []
    seen: set[str] = set()
    for err in (
        *audit_repo_ssot_ast(root, rel_paths=target_rels, _parsed_cache=parsed_cache),
        *audit_venue_string_branches_ast(root, rel_paths=target_rels, _parsed_cache=parsed_cache),
        *audit_equity_suffix_endswith_ast(root, rel_paths=target_rels, _parsed_cache=parsed_cache),
        *audit_layer_dag_ast(root, rel_paths=target_rels, _parsed_cache=parsed_cache),
        *audit_import_cycles_ast(root, rel_paths=target_rels, _parsed_cache=parsed_cache),
        *audit_producer_only_caches_ast(root, rel_paths=target_rels, _parsed_cache=parsed_cache),
        *audit_bespoke_atomic_writers(root, rel_paths=target_rels, _parsed_cache=parsed_cache),
    ):
        if err not in seen:
            seen.add(err)
            combined.append(err)
    return combined


def _staged_python_rels(root: Path, explicit_files: Iterable[str] | None = None) -> list[str]:
    if explicit_files is not None:
        candidates = list(explicit_files)
    else:
        proc = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
        )
        candidates = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    rels: list[str] = []
    for raw in candidates:
        if not raw.endswith(".py"):
            continue
        rel_s = _norm_rel_path(Path(raw), root)
        if rel_s and (root / rel_s).is_file():
            rels.append(rel_s)
    return rels


def cmd_pre_commit(args) -> int:
    root = Path(getattr(args, "root", None) or _ROOT)
    explicit = args.files if getattr(args, "files", None) else None
    rels = _staged_python_rels(root, explicit)
    if not rels:
        return 0
    violations = audit_all_ssot_ast(root, rel_paths=rels)
    if violations:
        print("\n[pre-commit] Blocked: SSOT AST violation(s) detected:")
        for err in violations:
            print(f"  {err}")
        print(
            "Delegate parsing/IO/env/venue checks to canonical SSOT APIs "
            "(cachepaths.py, instrument_registry.py, core/* via botcore.py, state_io.py, ProviderVenueMetadataMixin)."
        )
        return 1
    return 0


def cmd_scan(args) -> int:
    root = Path(getattr(args, "root", None) or _ROOT)
    violations = audit_all_ssot_ast(root)
    if violations:
        print(f"FAIL: {len(violations)} SSOT AST violation(s) detected:")
        for err in violations:
            print(f"  {err}")
        return 1
    print("OK: zero SSOT AST violations detected.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    pre = sub.add_parser("pre-commit", help="audit staged Python files for SSOT AST purity")
    pre.add_argument("files", nargs="*", help="optional explicit file paths")
    pre.add_argument("--root", default=None, help="optional repository root override")
    pre.set_defaults(func=cmd_pre_commit)

    scan = sub.add_parser("scan", help="audit all repository Python files for SSOT AST purity")
    scan.add_argument("--root", default=None, help="optional repository root override")
    scan.set_defaults(func=cmd_scan)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

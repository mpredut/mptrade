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

import cachepaths as _cachepaths
import instrument_registry as _instrument_registry
from state_io import read_text_file
from state_stores.registry import CRITICAL_STATE_SPECS, expected_store_class_name
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
_XSTOCK_ENDSWITH_TOKENS = frozenset({".T", "x.T", "X.T", "xUSD", "XUSD", "x/USD", "X/USD"})
_US_EQ_SUFFIX = "_US_EQ"
_XSTOCK_PAIR_FSTRING_PREFIXES = ("xUSD", "XUSD", "x/USD", "X/USD")
_FORBIDDEN_MANUAL_CONCAT_SUFFIXES = frozenset({
    "_EQ", "_US_EQ", "xUSD", "XUSD", "x/USD", "X/USD",
    "USD", "USDC", "USDT", "EUR", "FDUSD", "TUSD", "BUSD",
})
_PRICE_CACHE_FILE_PREFIXES = (
    "cache_ohlc_24h_",
    "cache_24_price_",
    "cache_365_price_",
    "price_24h_",
)
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
FORBIDDEN_STATE_STORE_IMPORTS = frozenset({
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
    "guard_hierarchy",
    "llm_order_guard",
    "order_lifecycle",
    "order_retry",
    "order_retry_worker",
    "order_ledger_reconciler_worker",
    "strategies",
    "trailing_core",
    "intelligence",
    "shadow_signals",
    "orchestratorTrade",
    "orchestratorOS",
    "binance_api",
    "kraken",
    "hyperliquid",
    "212trading",
})
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
    "order_ledger_reconciler_worker",
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
    "providers/order_ledger.py",
    "providers/ohlc_utils.py",
    "providers/order_ohlc_utils.py",
    "providers/quantity.py",
    "providers/strategy_executor.py",
    "providers/execution_audit.py",
    "providers/intent_state.py",
    "providers/intent_state_store.py",
    "providers/intent_execution.py",
    "providers/intent_reconciliation.py",
    "providers/intent_telemetry.py",
    "providers/ai_attribution.py",
    "providers/ai_attribution_store.py",
    "providers/nav_drawdown.py",
    "providers/nav_history_store.py",
    "providers/portfolio_state_store.py",
    "providers/portfolio_valuation.py",
})
PURE_VALUATION_MODULES = frozenset({
    "providers/portfolio_valuation.py",
    "providers/nav_drawdown.py",
    "providers/ai_attribution.py",
    "providers/quantity.py",
    "providers/intent_state.py",
    "intelligence/sentiment/evaluators/price_series_evaluator.py",
})
FORBIDDEN_PURE_VALUATION_IMPORTS = frozenset({
    "os",
    "glob",
    "pathlib",
    "io",
    "shutil",
    "tempfile",
    "state_io",
    "state_stores",
})
EVALUATOR_FORBIDDEN_WRITE_FNS = frozenset({
    "save_json_state",
    "save_json_file",
    "write_json_file",
    "atomic_write_json",
    "atomic_write_jsonl",
    "append_jsonl_record",
    "write_text_file",
    "write_bytes_file",
    "atomic_text_writer",
    "generation_transaction",
    "generation_file_lock",
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
    "order_ledger_reconciler_worker",
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
    "order_ledger_reconciler_worker.py",
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
    "get_balance_manager",
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
EVALUATOR_CLOCK_GUARDED_PREFIX = "intelligence/sentiment/evaluators/"
EVALUATOR_CLOCK_EXTRA_FILES = frozenset({"orchestratorOS/supervisor/prompt_sections.py"})
EQUITY_QUOTE_PROVENANCE_KEYS = frozenset({"quote_source", "quote_age_min"})
EQUITY_QUOTE_PROVENANCE_OWNER_FILE = "intelligence/sentiment/collectors/microstructure_collector.py"
EQUITY_QUOTE_PROVENANCE_OWNER_FN = "_apply_resolved_equity_quote"
SIGNED_ZERO_GUARDED_PREFIXES = ("intelligence/sentiment/", "orchestratorOS/supervisor/")
SIGNED_ZERO_EXTRA_FILES = frozenset({
    "providers/ai_survival_guard.py",
    "providers/nav_drawdown.py",
    "providers/ai_attribution.py",
})
_RAW_SIGNED_FLOAT_SPEC_RE = re.compile(r"\+[\d,]*\.\d+f")
VENUE_BALANCES_STATE_OWNER_FILES = frozenset({
    "state_stores/venue_balances_state.py",
    "state_stores/portfolio_positions_state.py",
})
VENUE_BALANCES_STATE_OWNER_FILE = "state_stores/venue_balances_state.py"
VENUE_BALANCES_STATE_OWNER_FNS = frozenset({
    "upsert_venue_balances_snapshot",
    "upsert_portfolio_positions_snapshot",
    "upsert_locked",
})
VENUE_BALANCE_CACHE_OWNER_FILE = "cacheaccount.py"
VENUE_BALANCE_CACHE_OWNER_FN = "replace_venue_balances_snapshot"
VENUE_BALANCES_STATE_PRODUCER_FILES = frozenset({
    "state_stores/venue_balances_state.py",
    "state_stores/portfolio_positions_state.py",
    "providers/portfolio_state_store.py",
    "intelligence/sentiment/collectors/portfolio_collector.py",
})
_VENUE_BALANCES_WRITE_PATH_RE = re.compile(
    r"(?:venue_balances_state|VENUE_BALANCES_STATE_FILENAME|"
    r"portfolio_positions_state|PORTFOLIO_POSITIONS_STATE_FILENAME|"
    r"cache_balance_|CACHE_[A-Z0-9_]+_BALANCE_FILENAME|\bbcache_path\b|\bpos_path\b)"
)
_VENUE_BALANCES_WRITER_FNS = frozenset({
    "save_json_state",
    "save_json_file",
    "write_json_file",
    "atomic_write_json",
    "write_text_file",
    "write_bytes_file",
})
STATE_FILE_OWNER_MAP: dict[str, frozenset[str]] = {
    **{spec.const_name: frozenset({spec.owner_module}) for spec in CRITICAL_STATE_SPECS},
    "AUTONOMOUS_ORDER_INTENTS_STATE_FILENAME": frozenset({"state_stores/auto_order_intents_state.py"}),
    "AUTONOMOUS_TRADE_INTENTS_STATE_FILENAME": frozenset({"state_stores/auto_order_intents_state.py"}),
    "CACHE_FEE_SCHEDULE_FILENAME": frozenset({"providers/venue_fee_maintenance.py"}),
    "CACHE_BINANCE_BALANCE_FILENAME": frozenset({"cacheaccount.py", "providers/portfolio_state_store.py"}),
    "CACHE_KRAKEN_BALANCE_FILENAME": frozenset({"cacheaccount.py", "providers/portfolio_state_store.py"}),
    "CACHE_HYPERLIQUID_BALANCE_FILENAME": frozenset({"cacheaccount.py", "providers/portfolio_state_store.py"}),
    "CACHE_T212_BALANCE_FILENAME": frozenset({"cacheaccount.py", "providers/portfolio_state_store.py"}),
}
STATE_FILE_LITERAL_TO_CONST: dict[str, str] = {
    **{spec.filename: spec.const_name for spec in CRITICAL_STATE_SPECS},
    "autonomous_order_intents_state.json": "AUTO_ORDER_INTENTS_STATE_FILENAME",
    "autonomous_trade_intents_state.json": "AUTO_ORDER_INTENTS_STATE_FILENAME",
    "autonomous_trade_intents.json": "AUTO_ORDER_INTENTS_STATE_FILENAME",
    "cache_fee_schedule.json": "CACHE_FEE_SCHEDULE_FILENAME",
    "cache_balance_binance.json": "CACHE_BINANCE_BALANCE_FILENAME",
    "cache_balance_kraken.json": "CACHE_KRAKEN_BALANCE_FILENAME",
    "cache_balance_hyperliquid.json": "CACHE_HYPERLIQUID_BALANCE_FILENAME",
    "cache_balance_t212.json": "CACHE_T212_BALANCE_FILENAME",
}
STATE_WRITE_ONLY_OWNER_MAP: dict[str, frozenset[str]] = {
    "CACHE_OHLC_24H_PREFIX": frozenset({"cacheprice.py"}),
    "cache_ohlc_24h_filename": frozenset({"cacheprice.py"}),
}
STATE_WRITE_ONLY_LITERAL_TO_CONST: dict[str, str] = {
    "cache_ohlc_24h_": "CACHE_OHLC_24H_PREFIX",
}
STATE_RAW_IO_READERS: frozenset[str] = frozenset({
    "read_json_dict",
    "read_json_dict_cached",
    "read_json_dict_recovering",
    "read_json_file",
    "read_json_recovering",
    "load_json_state",
    "read_jsonl_dicts",
    "read_tail_jsonl_dicts",
    "read_jsonl_dicts_from_offset",
})
STATE_RAW_IO_WRITERS: frozenset[str] = frozenset({
    "save_json_state",
    "save_json_file",
    "write_json_file",
    "atomic_write_json",
    "atomic_write_jsonl",
    "append_jsonl_record",
})


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

def _load_forbidden_upper_xstock_re(root: Path) -> tuple[re.Pattern[str] | None, dict[str, str]]:
    """Derive case-sensitive corrupted uppercase xStock tokens dynamically from registered mixed-case instruments."""
    conf_candidates = [root / "instruments.conf"]
    root_conf = _ROOT / "instruments.conf"
    if root_conf not in conf_candidates:
        conf_candidates.append(root_conf)
    tokens: dict[str, str] = {}
    for conf_p in conf_candidates:
        if not conf_p.is_file():
            continue
        try:
            reg = _instrument_registry.load_registry(conf_p)
        except Exception:
            continue
        for spec in reg.values():
            for canon in (spec.symbol, spec.base):
                if canon and any(ch.islower() for ch in canon) and _cachepaths.is_readonly_xstock_symbol(canon):
                    bad_u = canon.upper()
                    if bad_u != canon:
                        tokens[bad_u] = canon
                        if not bad_u.endswith("USD") and not bad_u.endswith(".T"):
                            tokens[f"{bad_u}.T"] = f"{canon}.T"
        if tokens:
            break
    if not tokens:
        return None, {}
    pat = re.compile(
        r"\b(?:" + "|".join(re.escape(tok) for tok in sorted(tokens, key=len, reverse=True)) + r")\b"
    )
    return pat, tokens


def audit_repo_ssot_ast(
    root: Path = _ROOT,
    rel_paths: Sequence[str] | None = None,
    *,
    _parsed_cache: ParsedCache | None = None,
) -> list[str]:
    """Audit production Python modules (and extra dirs for transitive re-exports) for SSOT AST purity."""
    root = Path(root)
    bad_xstock_re, bad_xstock_map = _load_forbidden_upper_xstock_re(root)
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
                elif isinstance(fn, ast.Attribute) and fn_name == "endswith" and node.args:
                    ew_arg = node.args[0]
                    ew_nodes = ew_arg.elts if isinstance(ew_arg, (ast.Tuple, ast.List)) else (ew_arg,)
                    ew_vals = [e.value for e in ew_nodes if isinstance(e, ast.Constant) and isinstance(e.value, str)]
                    if rel_s not in QUOTE_ENDSWITH_OWNERS and any(v in QUOTE_SUFFIX_TOKENS for v in ew_vals):
                        ssot_violations.append(f"{rel_s}:{lineno}: manual quote .endswith({ew_vals})")
                    if rel_s != "cachepaths.py" and any(v in _XSTOCK_ENDSWITH_TOKENS for v in ew_vals):
                        ssot_violations.append(
                            f"{rel_s}:{lineno}: manual xStock .endswith({ew_vals}) (call is_readonly_xstock_symbol)"
                        )
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
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if rel_s != "cachepaths.py":
                    for pfx in _PRICE_CACHE_FILE_PREFIXES:
                        if node.value.startswith(pfx):
                            ssot_violations.append(
                                f"{rel_s}:{lineno}: hardcoded {pfx} filename prefix (use cachepaths filename builders)"
                            )
                            break
                if bad_xstock_re is not None and (bad_m := bad_xstock_re.search(node.value)) is not None:
                    bad_tok = bad_m.group(0)
                    ssot_violations.append(
                        f"{rel_s}:{lineno}: corrupted uppercase xStock literal {bad_tok!r} "
                        f"(use canonical {bad_xstock_map.get(bad_tok, '')!r} or SymbolIdentity API)"
                    )
            elif rel_s not in INVALID_SCALAR_OWNERS and (
                (isinstance(node, ast.Name) and node.id == "INVALID_SCALAR_STRINGS")
                or (isinstance(node, ast.Attribute) and node.attr == "INVALID_SCALAR_STRINGS")
            ):
                ssot_violations.append(
                    f"{rel_s}:{lineno}: direct INVALID_SCALAR_STRINGS use (call clean_scalar_str / clean_upper_scalar)"
                )
            elif rel_s != "cachepaths.py" and isinstance(node, ast.JoinedStr) and node.values:
                if (
                    isinstance(node.values[-1], ast.Constant)
                    and isinstance(node.values[-1].value, str)
                    and node.values[-1].value.endswith(_US_EQ_SUFFIX)
                ):
                    ssot_violations.append(
                        f"{rel_s}:{lineno}: manual f-string _US_EQ concatenation (call ensure_equity_suffix)"
                    )
                if any(
                    isinstance(node.values[idx - 1], ast.FormattedValue)
                    and isinstance(node.values[idx], ast.Constant)
                    and isinstance(node.values[idx].value, str)
                    and node.values[idx].value.startswith(_XSTOCK_PAIR_FSTRING_PREFIXES)
                    for idx in range(1, len(node.values))
                ):
                    ssot_violations.append(
                        f"{rel_s}:{lineno}: manual f-string xStock pair concatenation "
                        f"(call ensure_xstock_pair / SymbolIdentity.quote_pair)"
                    )
                if (
                    len(node.values) == 2
                    and isinstance(node.values[0], ast.FormattedValue)
                    and isinstance(node.values[1], ast.Constant)
                    and isinstance(node.values[1].value, str)
                ):
                    tail_s = node.values[1].value
                    if tail_s in QUOTE_SUFFIX_TOKENS:
                        ssot_violations.append(
                            f"{rel_s}:{lineno}: manual f-string {tail_s} pair concatenation "
                            f"(call ensure_quote_suffix / SymbolIdentity.quote_pair)"
                        )
                    elif tail_s == "x" and node.values[0].format_spec is None:
                        ssot_violations.append(
                            f"{rel_s}:{lineno}: manual f-string xStock asset concatenation "
                            f"(call ensure_xstock_asset / SymbolIdentity.balance_asset_key)"
                        )
            elif (
                rel_s != "cachepaths.py"
                and isinstance(node, ast.BinOp)
                and isinstance(node.op, ast.Add)
                and isinstance(node.right, ast.Constant)
                and isinstance(node.right.value, str)
                and node.right.value in _FORBIDDEN_MANUAL_CONCAT_SUFFIXES
            ):
                ssot_violations.append(
                    f"{rel_s}:{lineno}: manual string concatenation {node.right.value!r} "
                    f"(use cachepaths / SymbolIdentity canonical builders)"
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
        state_store_rels = [
            r for r in _iter_repo_python_rels(root) if r.startswith("state_stores/")
        ]
        target_rels = [r for r in sorted(watched) if (root / r).is_file()] + sorted(state_store_rels)
    else:
        target_rels = [
            r
            for r in _normalize_target_rels(root, rel_paths)
            if r in watched or r.startswith("state_stores/")
        ]
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
            if leak := sorted(roots & (FORBIDDEN_LAYER2_IMPORTS | {"state_stores"})):
                violations.append(f"{rel}: upward layer leak {leak}")
        elif rel.startswith("state_stores/"):
            if leak := sorted(roots & FORBIDDEN_STATE_STORE_IMPORTS):
                violations.append(f"{rel}: state_stores Layer 2.5 upward leak {leak}")
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
        if rel in PURE_VALUATION_MODULES:
            if io_leak := sorted(roots & FORBIDDEN_PURE_VALUATION_IMPORTS):
                violations.append(
                    f"{rel}: pure valuation module must not import disk/state I/O modules {io_leak} "
                    "(delegate persistence to providers/portfolio_state_store.py, cacheprice.py, or state_stores/)"
                )
        if rel.startswith(EVALUATOR_CLOCK_GUARDED_PREFIX):
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    fn_nm = (
                        node.func.id
                        if isinstance(node.func, ast.Name)
                        else (node.func.attr if isinstance(node.func, ast.Attribute) else "")
                    )
                    if fn_nm in EVALUATOR_FORBIDDEN_WRITE_FNS:
                        violations.append(
                            f"{rel}:{node.lineno}: stateless evaluator must not call disk writer {fn_nm}() "
                            "(delegate persistence to Layer 3 cache manager or state_stores/)"
                        )
        if rel.startswith("intelligence/"):
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and node.attr == "_client":
                    violations.append(
                        f"{rel}:{node.lineno}: private provider ._client access in intelligence/ "
                        "(delegate through a public provider facade method)"
                    )
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


# -- Audit 8: Evaluator replay clock purity (no bare wall-clock calls) ---------

def audit_evaluator_clock_purity_ast(
    root: Path = _ROOT,
    rel_paths: Sequence[str] | None = None,
    *,
    _parsed_cache: ParsedCache | None = None,
) -> list[str]:
    """Enforce zero direct time.time()/datetime.now()/datetime.utcnow() calls inside stateless evaluators."""
    root = Path(root)
    target_rels = [
        r
        for r in _normalize_target_rels(root, rel_paths)
        if (r.startswith(EVALUATOR_CLOCK_GUARDED_PREFIX) or r in EVALUATOR_CLOCK_EXTRA_FILES)
        and r.endswith(".py")
    ]
    violations: list[str] = []
    for rel_s in target_rels:
        _, tree, syntax_err = parse_module_cached(root / rel_s, rel_s, _parsed_cache)
        if syntax_err is not None or tree is None:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            attr = node.func.attr
            val = node.func.value
            if isinstance(val, ast.Name):
                if val.id in ("time", "_time") and attr == "time":
                    violations.append(
                        f"{rel_s}:{node.lineno}: direct wall-clock {val.id}.time() in evaluator (use resolve_now_ts(now_ts))"
                    )
                elif val.id in ("datetime", "_datetime") and attr in ("now", "utcnow"):
                    violations.append(
                        f"{rel_s}:{node.lineno}: direct wall-clock {val.id}.{attr}() in evaluator (use resolve_now_ts(now_ts))"
                    )
    return violations


# -- Audit 9: Single atomic writer for equity quote provenance -----------------

def audit_equity_quote_provenance_ast(
    root: Path = _ROOT,
    rel_paths: Sequence[str] | None = None,
    *,
    _parsed_cache: ParsedCache | None = None,
) -> list[str]:
    """Enforce that quote_source and quote_age_min are written ONLY via Fleet24hAndMicrostructureEnricher._apply_resolved_equity_quote."""
    root = Path(root)
    target_rels = [r for r in _normalize_target_rels(root, rel_paths) if _is_prod_python_rel(r)]
    violations: list[str] = []

    def _const_str(node: ast.AST | None) -> str | None:
        return node.value if (isinstance(node, ast.Constant) and isinstance(node.value, str)) else None

    for rel_s in target_rels:
        _, tree, syntax_err = parse_module_cached(root / rel_s, rel_s, _parsed_cache)
        if syntax_err is not None or tree is None:
            continue

        def _visit(node: ast.AST, fn_name: str | None) -> None:
            cur_fn = fn_name
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                cur_fn = node.name
            is_owner = (
                rel_s == EQUITY_QUOTE_PROVENANCE_OWNER_FILE
                and cur_fn == EQUITY_QUOTE_PROVENANCE_OWNER_FN
            )
            if not is_owner:
                if (
                    isinstance(node, ast.Subscript)
                    and isinstance(node.ctx, ast.Store)
                    and (key_s := _const_str(node.slice)) in EQUITY_QUOTE_PROVENANCE_KEYS
                ):
                    violations.append(
                        f"{rel_s}:{node.lineno}: direct write to {key_s!r} outside {EQUITY_QUOTE_PROVENANCE_OWNER_FN}"
                    )
                elif (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "setdefault"
                    and node.args
                    and (key_s := _const_str(node.args[0])) in EQUITY_QUOTE_PROVENANCE_KEYS
                ):
                    violations.append(
                        f"{rel_s}:{node.lineno}: direct setdefault({key_s!r}, ...) outside {EQUITY_QUOTE_PROVENANCE_OWNER_FN}"
                    )
                elif isinstance(node, ast.Dict):
                    for k_node in node.keys:
                        if (key_s := _const_str(k_node)) in EQUITY_QUOTE_PROVENANCE_KEYS:
                            violations.append(
                                f"{rel_s}:{getattr(k_node, 'lineno', node.lineno)}: dict literal {key_s!r} outside {EQUITY_QUOTE_PROVENANCE_OWNER_FN}"
                            )
            for child in ast.iter_child_nodes(node):
                _visit(child, cur_fn)

        _visit(tree, None)
    return violations


# -- Audit 10: Signed-zero safe float formatting in prompts & telemetry --------

def audit_signed_zero_format_ast(
    root: Path = _ROOT,
    rel_paths: Sequence[str] | None = None,
    *,
    _parsed_cache: ParsedCache | None = None,
) -> list[str]:
    """Enforce botcore.fmt_signed instead of raw :+.Nf format specs in prompt/telemetry code (prevents -0.00%)."""
    root = Path(root)
    target_rels = [
        r
        for r in _normalize_target_rels(root, rel_paths)
        if (r.startswith(SIGNED_ZERO_GUARDED_PREFIXES) or r in SIGNED_ZERO_EXTRA_FILES)
        and r.endswith(".py")
    ]
    violations: list[str] = []
    for rel_s in target_rels:
        _, tree, syntax_err = parse_module_cached(root / rel_s, rel_s, _parsed_cache)
        if syntax_err is not None or tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FormattedValue) and isinstance(node.format_spec, ast.JoinedStr):
                spec_parts = [
                    v.value for v in node.format_spec.values if isinstance(v, ast.Constant) and isinstance(v.value, str)
                ]
                spec_s = "".join(spec_parts)
                if spec_s and _RAW_SIGNED_FLOAT_SPEC_RE.search(spec_s):
                    violations.append(
                        f"{rel_s}:{node.lineno}: raw signed float format spec {spec_s!r} (use botcore.fmt_signed)"
                    )
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "format"
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
                and _RAW_SIGNED_FLOAT_SPEC_RE.search(node.args[1].value)
            ):
                violations.append(
                    f"{rel_s}:{node.lineno}: raw signed format(..., {node.args[1].value!r}) (use botcore.fmt_signed)"
                )
    return violations


# -- Audit 11: Cross-module local import & call-signature integrity ------------

_SIGNATURE_PRESERVING_DECORATORS = frozenset({
    "contextmanager", "lru_cache", "cache", "singledispatch",
})


def _collect_module_exports_and_funcs(
    tree: ast.Module,
) -> tuple[set[str], bool, dict[str, ast.arguments]]:
    exports: set[str] = set()
    funcs: dict[str, ast.arguments] = {}
    is_dynamic = False
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            exports.add(node.name)
            if node.name == "__getattr__":
                is_dynamic = True
            decs_ok = all(
                (isinstance(d, ast.Name) and d.id in _SIGNATURE_PRESERVING_DECORATORS)
                or (isinstance(d, ast.Attribute) and d.attr in _SIGNATURE_PRESERVING_DECORATORS)
                or (
                    isinstance(d, ast.Call)
                    and (
                        (isinstance(d.func, ast.Name) and d.func.id in _SIGNATURE_PRESERVING_DECORATORS)
                        or (isinstance(d.func, ast.Attribute) and d.func.attr in _SIGNATURE_PRESERVING_DECORATORS)
                    )
                )
                for d in node.decorator_list
            )
            if decs_ok:
                funcs[node.name] = node.args
        elif isinstance(node, ast.ClassDef):
            exports.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                for sub in ast.walk(t):
                    if isinstance(sub, ast.Name):
                        exports.add(sub.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            exports.add(node.target.id)
        elif isinstance(node, ast.Import):
            for a in node.names:
                exports.add(a.asname or a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                if a.name == "*":
                    is_dynamic = True
                else:
                    exports.add(a.asname or a.name)
        elif isinstance(node, (ast.If, ast.Try)):
            for sub in ast.walk(node):
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    exports.add(sub.name)
                elif isinstance(sub, ast.Assign):
                    for t in sub.targets:
                        if isinstance(t, ast.Name):
                            exports.add(t.id)
                elif isinstance(sub, ast.ImportFrom):
                    for a in sub.names:
                        if a.name == "*":
                            is_dynamic = True
                        else:
                            exports.add(a.asname or a.name)
                elif isinstance(sub, ast.Import):
                    for a in sub.names:
                        exports.add(a.asname or a.name.split(".")[0])
    return exports, is_dynamic, funcs


def _check_call_against_signature(call: ast.Call, fargs: ast.arguments) -> str | None:
    if any(isinstance(a, ast.Starred) for a in call.args):
        return None
    if any(kw.arg is None for kw in call.keywords):
        return None
    all_pos = [a.arg for a in fargs.posonlyargs] + [a.arg for a in fargs.args]
    pos_or_kw = [a.arg for a in fargs.args]
    kw_only = [a.arg for a in fargs.kwonlyargs]
    req_pos = all_pos[: len(all_pos) - len(fargs.defaults)]
    req_kw = [a.arg for a, d in zip(fargs.kwonlyargs, fargs.kw_defaults) if d is None]
    n_pos = len(call.args)
    if fargs.vararg is None and n_pos > len(all_pos):
        return f"too many positional args ({n_pos} > {len(all_pos)})"
    kw_names = [kw.arg for kw in call.keywords if kw.arg is not None]
    if fargs.kwarg is None:
        allowed_kw = set(pos_or_kw) | set(kw_only)
        bad_kw = [k for k in kw_names if k not in allowed_kw]
        if bad_kw:
            return f"unexpected kwarg(s) {bad_kw}"
    passed_pos = set(all_pos[:n_pos])
    dup = passed_pos & set(kw_names)
    if dup:
        return f"multiple values for arg(s) {sorted(dup)}"
    missing_pos = [a for a in req_pos if a not in passed_pos and a not in kw_names]
    if missing_pos:
        return f"missing required positional arg(s) {missing_pos}"
    missing_kw = [a for a in req_kw if a not in kw_names]
    if missing_kw:
        return f"missing required kw-only arg(s) {missing_kw}"
    return None


def audit_cross_module_calls_and_imports_ast(
    root: Path = _ROOT,
    rel_paths: Sequence[str] | None = None,
    *,
    _parsed_cache: ParsedCache | None = None,
) -> list[str]:
    """Verify local ``from mod import sym`` existence and direct top-level call signatures across prod modules."""
    root = Path(root)
    parsed_cache: ParsedCache = _parsed_cache if _parsed_cache is not None else {}
    all_rels = _iter_repo_python_rels(root)
    mod_to_rel: dict[str, str] = {}
    for rel_s in all_rels:
        if rel_s.startswith(("tests/", "offline/", "archive/")):
            continue
        mod_to_rel[_rel_to_mod(rel_s)] = rel_s

    mod_meta_cache: dict[str, tuple[set[str], bool, dict[str, ast.arguments]]] = {}

    def _get_mod_meta(mod_name: str) -> tuple[set[str], bool, dict[str, ast.arguments]] | None:
        if mod_name in mod_meta_cache:
            return mod_meta_cache[mod_name]
        rel_m = mod_to_rel.get(mod_name)
        if not rel_m:
            return None
        _, m_tree, m_err = parse_module_cached(root / rel_m, rel_m, parsed_cache)
        if m_err is not None or m_tree is None:
            return None
        meta = _collect_module_exports_and_funcs(m_tree)
        mod_meta_cache[mod_name] = meta
        return meta

    target_rels = [r for r in _normalize_target_rels(root, rel_paths) if _is_prod_python_rel(r)]
    violations: list[str] = []
    for rel_s in target_rels:
        _, tree, syntax_err = parse_module_cached(root / rel_s, rel_s, parsed_cache)
        if syntax_err is not None or tree is None:
            continue
        # Collect locally redefined top-level names so we don't flag shadowed calls
        local_defs = {
            n.name
            for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
        sym_map: dict[str, tuple[str, str]] = {}
        mod_alias: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name in mod_to_rel:
                        mod_alias[a.asname or a.name] = a.name
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module in mod_to_rel:
                src_mod = node.module
                meta = _get_mod_meta(src_mod)
                if meta is None:
                    continue
                exports, is_dynamic, _ = meta
                for a in node.names:
                    if a.name == "*":
                        continue
                    local_name = a.asname or a.name
                    if local_name not in local_defs:
                        sym_map[local_name] = (src_mod, a.name)
                    if not is_dynamic and a.name not in exports and f"{src_mod}.{a.name}" not in mod_to_rel:
                        violations.append(
                            f"{rel_s}:{node.lineno}: broken local import `from {src_mod} import {a.name}`"
                        )
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            target_info: tuple[str, ast.arguments] | None = None
            if isinstance(node.func, ast.Name) and node.func.id in sym_map:
                src_mod, orig = sym_map[node.func.id]
                meta = _get_mod_meta(src_mod)
                if meta is not None and orig in meta[2]:
                    target_info = (f"{src_mod}.{orig}", meta[2][orig])
            elif (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in mod_alias
            ):
                src_mod = mod_alias[node.func.value.id]
                orig = node.func.attr
                meta = _get_mod_meta(src_mod)
                if meta is not None and orig in meta[2]:
                    target_info = (f"{src_mod}.{orig}", meta[2][orig])
            if target_info is not None:
                if err := _check_call_against_signature(node, target_info[1]):
                    violations.append(f"{rel_s}:{node.lineno}: call to {target_info[0]}() {err}")
    return violations


# -- Audit 12: Canonical cachedb subdomain path & JSONL reader purity ----------

_CACHEDB_SUBDOMAIN_TOKENS = frozenset({
    "prices", "trends", "macro", "orders", "account", "portfolio", "bots", "state", "telemetry", "llm", "system", "heartbeat",
})
_RAW_CACHEDIR_VAR_RE = re.compile(r"(?:^|_)(?:cache_dir|cachedb_dir|cdir|eff_dir)$")
_JSONL_CACHE_PATH_HINT_RE = re.compile(
    r"(?:\.jsonl\b|cache_(?:24|365|sparse)_price_|cache_asset_value|cache_trade_binance|order_retry_queue)",
    re.I,
)
_JSON_DICT_READER_NAMES = frozenset({
    "read_json_dict", "read_json_dict_cached", "read_json_recovering", "read_json_dict_recovering",
})
_CACHEDB_PATH_PURITY_EXEMPT = frozenset({"cachepaths.py", "state_io.py"})


def audit_cachedb_path_and_jsonl_purity_ast(
    root: Path = _ROOT,
    rel_paths: Sequence[str] | None = None,
    *,
    _parsed_cache: ParsedCache | None = None,
) -> list[str]:
    """Enforce resolved cachedb subdomain checks and prevent JSON dict readers on JSONL cache files."""
    root = Path(root)
    target_rels = [
        r
        for r in _normalize_target_rels(root, rel_paths)
        if _is_prod_python_rel(r) and r not in _CACHEDB_PATH_PURITY_EXEMPT
    ]
    violations: list[str] = []
    for rel_s in target_rels:
        _, tree, syntax_err = parse_module_cached(root / rel_s, rel_s, _parsed_cache)
        if syntax_err is not None or tree is None:
            continue
        for fn_node in ast.walk(tree):
            if not isinstance(fn_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            raw_joined_vars: dict[str, int] = {}
            for node in ast.walk(fn_node):
                # Track `var = os.path.join(cache_dir, ...)` without resolve_cachedb_filepath / explicit subdomain
                if (
                    isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Attribute)
                    and node.value.func.attr == "join"
                    and isinstance(node.value.func.value, ast.Attribute)
                    and node.value.func.value.attr == "path"
                    and len(node.value.args) == 2
                ):
                    arg0 = node.value.args[0]
                    if isinstance(arg0, ast.Name) and _RAW_CACHEDIR_VAR_RE.search(arg0.id):
                        has_subdomain_arg = any(
                            isinstance(a, ast.Constant)
                            and isinstance(a.value, str)
                            and a.value.strip("/") in _CACHEDB_SUBDOMAIN_TOKENS
                            for a in node.value.args[1:]
                        )
                        if not has_subdomain_arg:
                            raw_joined_vars[node.targets[0].id] = node.lineno
                # Check direct FS calls on unresolved raw_joined_vars
                if isinstance(node, ast.Call) and node.args:
                    call_s = ast.unparse(node.func)
                    if call_s in (
                        "os.path.exists", "os.path.isfile", "os.path.getmtime", "os.stat",
                    ) and isinstance(node.args[0], ast.Name) and node.args[0].id in raw_joined_vars:
                        var_name = node.args[0].id
                        violations.append(
                            f"{rel_s}:{node.lineno}: {call_s}({var_name}) on unresolved os.path.join(cache_dir, ...) "
                            f"from line {raw_joined_vars[var_name]} (use resolve_cachedb_filepath or state_io.cachedb_exists)"
                        )
                    # Check JSON dict readers called directly on known JSONL cache path literals
                    fn_attr = (
                        node.func.id
                        if isinstance(node.func, ast.Name)
                        else (node.func.attr if isinstance(node.func, ast.Attribute) else "")
                    )
                    if fn_attr in _JSON_DICT_READER_NAMES:
                        arg0_s = ast.unparse(node.args[0])
                        if _JSONL_CACHE_PATH_HINT_RE.search(arg0_s) and ".meta" not in arg0_s and ".json\"" not in arg0_s and ".json'" not in arg0_s:
                            violations.append(
                                f"{rel_s}:{node.lineno}: {fn_attr}({arg0_s[:80]}) called on JSONL cache path "
                                "(use read_jsonl_bucket_map / read_jsonl_or_json_dict_cached)"
                            )
    return violations


# -- Audit 13: DurabilityTier financial fsync & cachedb dirname purity ---------

_STATE_WRITER_FN_NAMES = frozenset({
    "atomic_write_json",
    "atomic_write_text",
    "atomic_text_writer",
    "save_json_state",
    "write_meta_json",
    "commit_file_and_meta",
    "stage_previous_generation",
})
_FINANCIAL_CRITICAL_PATH_RE = re.compile(
    r"(?:ORDER_LEDGER_|orders_active|order_events_|orders_closed_|"
    r"RTRADE_PAIRS_|rtrade_pairs|ASSETGUARDIAN_STATE_|assetguardian_state|"
    r"AUTO_ORDER_INTENTS_|auto_order_intents|"
    r"AUTONOMOUS_ORDER_INTENTS_|autonomous_order_intents|"
    r"AUTONOMOUS_TRADE_INTENTS_|autonomous_trade_intents|"
    r"CACHE_[A-Z0-9_]*ORDER_|cache_order_|CACHE_[A-Z0-9_]*BALANCE_|cache_balance_|CACHE_[A-Z0-9_]*TRADE_|cache_trade_|"
    r"CACHE_FEE_SCHEDULE_|cache_fee_schedule|VENUE_FEE_MAINTENANCE_STATE_|venue_fee_maintenance_state|"
    r"ORDER_RETRY_QUEUE_|order_retry_queue|VENUE_BALANCES_STATE_|venue_balances_state|"
    r"PORTFOLIO_POSITIONS_STATE_|portfolio_positions_state|"
    r"NAV_HISTORY_STATE_|nav_history_state|SMART_STOP_LOSS_STATE_|smart_stop_loss_state|"
    r"TRAILING_STATE_|trailing_state|MT_STATE_|mt_state|"
    r"LLM_TOPIC_THREADS_|llm_topic_threads|NOTIFY_QUEUE_|notify_queue)",
    re.I,
)
_STATE_FILE_DIRNAME_ARG_RE = re.compile(r"(?:^|\.)(?:STATE\.path|state_path|state_file)$")


def audit_durability_tier_and_cachedb_dirname_ast(
    root: Path = _ROOT,
    rel_paths: Sequence[str] | None = None,
    *,
    _parsed_cache: ParsedCache | None = None,
) -> list[str]:
    """Block ``fsync=False`` on ``FINANCIAL_CRITICAL`` state paths and ``os.path.dirname(state_path)`` as a cachedb root."""
    root = Path(root)
    target_rels = [
        r
        for r in _normalize_target_rels(root, rel_paths)
        if _is_prod_python_rel(r) and r not in _CACHEDB_PATH_PURITY_EXEMPT
    ]
    violations: list[str] = []
    for rel_s in target_rels:
        _, tree, syntax_err = parse_module_cached(root / rel_s, rel_s, _parsed_cache)
        if syntax_err is not None or tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and node.args:
                fn_attr = (
                    node.func.id
                    if isinstance(node.func, ast.Name)
                    else (node.func.attr if isinstance(node.func, ast.Attribute) else "")
                )
                if fn_attr in _STATE_WRITER_FN_NAMES:
                    has_fsync_false = any(
                        kw.arg == "fsync"
                        and isinstance(kw.value, ast.Constant)
                        and kw.value.value is False
                        for kw in node.keywords
                    )
                    if has_fsync_false:
                        arg0_s = ast.unparse(node.args[0])
                        if _FINANCIAL_CRITICAL_PATH_RE.search(arg0_s):
                            violations.append(
                                f"{rel_s}:{node.lineno}: forbidden fsync=False on FINANCIAL_CRITICAL path "
                                f"{arg0_s[:80]} in {fn_attr}() (use default DurabilityPolicy / fsync=True)"
                            )
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and _RAW_CACHEDIR_VAR_RE.search(node.targets[0].id)
                and isinstance(node.value, ast.Call)
                and ast.unparse(node.value.func) == "os.path.dirname"
                and len(node.value.args) == 1
            ):
                arg0_s = ast.unparse(node.value.args[0])
                if _STATE_FILE_DIRNAME_ARG_RE.search(arg0_s):
                    violations.append(
                        f"{rel_s}:{node.lineno}: {node.targets[0].id} = os.path.dirname({arg0_s}) "
                        "may return a cachedb/<subdomain>/ leaf (use CachedbRoot.from_path or cachedb_root_for_file)"
                    )
    return violations


# -- Audit 14: Single canonical locked writer for venue_balances_state.json ----

def audit_venue_balances_single_writer_ast(
    root: Path = _ROOT,
    rel_paths: Sequence[str] | None = None,
    *,
    _parsed_cache: ParsedCache | None = None,
) -> list[str]:
    """Enforce single writer & producer (portfolio_collector -> providers.portfolio_state_store / state_stores)."""
    root = Path(root)
    target_rels = [r for r in _normalize_target_rels(root, rel_paths) if _is_prod_python_rel(r)]
    violations: list[str] = []

    def _writer_call_name(call: ast.Call) -> str:
        fn = call.func
        if isinstance(fn, ast.Name):
            return fn.id
        if isinstance(fn, ast.Attribute):
            return fn.attr
        if isinstance(fn, ast.Call):
            inner = fn.func
            inner_name = (
                inner.id
                if isinstance(inner, ast.Name)
                else (inner.attr if isinstance(inner, ast.Attribute) else "")
            )
            if inner_name == "resolve_save_json_state":
                return "save_json_state"
        return ""

    for rel_s in target_rels:
        _, tree, syntax_err = parse_module_cached(root / rel_s, rel_s, _parsed_cache)
        if syntax_err is not None or tree is None:
            continue

        def _visit(node: ast.AST, fn_name: str | None) -> None:
            cur_fn = fn_name
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                cur_fn = node.name
            is_owner = (
                (rel_s in VENUE_BALANCES_STATE_OWNER_FILES and cur_fn in VENUE_BALANCES_STATE_OWNER_FNS)
                or (rel_s == VENUE_BALANCE_CACHE_OWNER_FILE and cur_fn == VENUE_BALANCE_CACHE_OWNER_FN)
            )
            if isinstance(node, ast.Call) and node.args:
                wname = _writer_call_name(node)
                if not is_owner and wname in _VENUE_BALANCES_WRITER_FNS:
                    arg0_s = ast.unparse(node.args[0])
                    if _VENUE_BALANCES_WRITE_PATH_RE.search(arg0_s):
                        violations.append(
                            f"{rel_s}:{node.lineno}: direct write to balance/position state via "
                            f"{wname}({arg0_s[:80]}, ...) outside canonical owners"
                        )
                elif (
                    wname in VENUE_BALANCES_STATE_OWNER_FNS
                    and rel_s not in VENUE_BALANCES_STATE_PRODUCER_FILES
                ):
                    violations.append(
                        f"{rel_s}:{node.lineno}: forbidden producer call {wname}() "
                        f"outside {sorted(VENUE_BALANCES_STATE_PRODUCER_FILES)}"
                    )
            for child in ast.iter_child_nodes(node):
                _visit(child, cur_fn)

        _visit(tree, None)
    return violations


# -- Audit 15: Exclusive typed state file owners -------------------------------

def _audit_state_store_registry_symmetry(
    root: Path,
    *,
    _parsed_cache: ParsedCache | None = None,
) -> list[str]:
    """Verify strict 1:1 filename <-> state_stores/<stem>.py <-> <PascalCase>Store symmetry."""
    violations: list[str] = []
    seen_files: set[str] = set()
    seen_owners: set[str] = set()
    for spec in CRITICAL_STATE_SPECS:
        if spec.filename in seen_files:
            violations.append(f"state_stores/registry.py: duplicate filename registration {spec.filename!r}")
        seen_files.add(spec.filename)
        if spec.owner_module in seen_owners:
            violations.append(f"state_stores/registry.py: duplicate owner module {spec.owner_module!r}")
        seen_owners.add(spec.owner_module)
        expected_mod = f"state_stores/{Path(spec.filename).stem}.py"
        if spec.owner_module != expected_mod:
            violations.append(
                f"state_stores/registry.py: asymmetric owner {spec.owner_module!r} for {spec.filename!r} "
                f"(expected {expected_mod!r})"
            )
        expected_cls = expected_store_class_name(spec.filename)
        if spec.store_class != expected_cls:
            violations.append(
                f"state_stores/registry.py: asymmetric store class {spec.store_class!r} for {spec.filename!r} "
                f"(expected {expected_cls!r})"
            )
        cp_val = getattr(_cachepaths, spec.const_name, None)
        if cp_val != spec.filename:
            violations.append(
                f"state_stores/registry.py: cachepaths.{spec.const_name}={cp_val!r} != {spec.filename!r}"
            )
        mod_path = root / spec.owner_module
        if not mod_path.is_file():
            violations.append(f"state_stores/registry.py: missing 1:1 owner module {spec.owner_module}")
            continue
        _, tree, syntax_err = parse_module_cached(mod_path, spec.owner_module, _parsed_cache)
        if syntax_err is not None:
            violations.append(syntax_err)
            continue
        if tree is not None:
            defined_classes = {n.name for n in tree.body if isinstance(n, ast.ClassDef)}
            if spec.store_class not in defined_classes:
                violations.append(
                    f"{spec.owner_module}: missing 1:1 store class {spec.store_class} for {spec.filename}"
                )
    return violations


def audit_state_file_owners_ast(
    root: Path = _ROOT,
    rel_paths: Sequence[str] | None = None,
    *,
    _parsed_cache: ParsedCache | None = None,
) -> list[str]:
    """Enforce 1:1 state_stores symmetry and forbid raw state_io reads/writes outside each file's exclusive owner."""
    root = Path(root)
    norm_rels = _normalize_target_rels(root, rel_paths)
    target_rels = [
        r
        for r in norm_rels
        if _is_prod_python_rel(r) and r != "cachepaths.py" and not r.startswith("verify_tools/")
    ]
    violations: list[str] = []
    if rel_paths is None or any(r.startswith("state_stores/") or r == "cachepaths.py" for r in norm_rels):
        violations.extend(_audit_state_store_registry_symmetry(root, _parsed_cache=_parsed_cache))

    def _expr_state_consts(expr: ast.AST | None, tainted: dict[str, set[str]]) -> set[str]:
        found: set[str] = set()
        if expr is None:
            return found
        for sub in ast.walk(expr):
            if isinstance(sub, ast.Name):
                if sub.id in STATE_FILE_OWNER_MAP or sub.id in STATE_WRITE_ONLY_OWNER_MAP:
                    found.add(sub.id)
                elif sub.id in tainted:
                    found.update(tainted[sub.id])
            elif isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                low = sub.value.strip().lower()
                for lit_fname, const_name in STATE_FILE_LITERAL_TO_CONST.items():
                    if lit_fname in low:
                        found.add(const_name)
                for lit_pref, const_name in STATE_WRITE_ONLY_LITERAL_TO_CONST.items():
                    if lit_pref in low:
                        found.add(const_name)
        return found

    def _audit_scope(rel_s: str, stmts: list[ast.stmt], inherited: dict[str, set[str]]) -> None:
        tainted: dict[str, set[str]] = {k: set(v) for k, v in inherited.items()}
        nested_scopes: list[list[ast.stmt]] = []

        def _collect_local_stmts(block: list[ast.stmt], out: list[ast.stmt]) -> None:
            for stmt in block:
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    nested_scopes.append(list(stmt.body))
                    continue
                out.append(stmt)
                for attr in ("body", "orelse", "finalbody"):
                    if isinstance(sub_b := getattr(stmt, attr, None), list):
                        _collect_local_stmts(sub_b, out)
                if isinstance(handlers := getattr(stmt, "handlers", None), list):
                    for h in handlers:
                        if isinstance(h, ast.ExceptHandler):
                            _collect_local_stmts(h.body, out)

        local_stmts: list[ast.stmt] = []
        _collect_local_stmts(stmts, local_stmts)

        # Two O(N) passes propagate local path variables (raw_path -> resolved_path -> reader).
        for _ in range(2):
            for stmt in local_stmts:
                if isinstance(stmt, ast.Assign):
                    consts = _expr_state_consts(stmt.value, tainted)
                    if consts:
                        for tgt in stmt.targets:
                            if isinstance(tgt, ast.Name):
                                tainted.setdefault(tgt.id, set()).update(consts)
                elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    consts = _expr_state_consts(stmt.value, tainted)
                    if consts:
                        tainted.setdefault(stmt.target.id, set()).update(consts)
                elif isinstance(stmt, ast.For) and isinstance(stmt.target, ast.Name):
                    consts = _expr_state_consts(stmt.iter, tainted)
                    if consts:
                        tainted.setdefault(stmt.target.id, set()).update(consts)

        def _scan_expr_calls(node: ast.AST | None) -> None:
            if node is None or isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                return
            if isinstance(node, ast.Call):
                fn_name: str | None = None
                if isinstance(node.func, ast.Name):
                    fn_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    fn_name = node.func.attr
                if fn_name in STATE_RAW_IO_READERS or fn_name in STATE_RAW_IO_WRITERS:
                    hit_consts: set[str] = set()
                    for arg in node.args:
                        hit_consts.update(_expr_state_consts(arg, tainted))
                    for kw in node.keywords:
                        hit_consts.update(_expr_state_consts(kw.value, tainted))
                    io_kind = "write" if fn_name in STATE_RAW_IO_WRITERS else "read"
                    for const_name in sorted(hit_consts):
                        if const_name in STATE_FILE_OWNER_MAP:
                            owners = STATE_FILE_OWNER_MAP[const_name]
                        elif io_kind == "write" and const_name in STATE_WRITE_ONLY_OWNER_MAP:
                            owners = STATE_WRITE_ONLY_OWNER_MAP[const_name]
                        else:
                            continue
                        if rel_s not in owners:
                            violations.append(
                                f"{rel_s}:{node.lineno}: direct raw {io_kind} of {const_name} via {fn_name}() "
                                f"outside SSOT owner {sorted(owners)}"
                            )
            for child in ast.iter_child_nodes(node):
                if not isinstance(child, ast.stmt):
                    _scan_expr_calls(child)

        for stmt in local_stmts:
            for child in ast.iter_child_nodes(stmt):
                if not isinstance(child, ast.stmt):
                    _scan_expr_calls(child)
        for nested_body in nested_scopes:
            _audit_scope(rel_s, nested_body, tainted)

    for rel_s in target_rels:
        _, tree, syntax_err = parse_module_cached(root / rel_s, rel_s, _parsed_cache)
        if syntax_err is not None or tree is None:
            continue
        _audit_scope(rel_s, list(tree.body), {})
    return violations


# -- Audit 16: Dataclass slots=True assignment safety --------------------------

def _is_slotted_dataclass_def(cls_node: ast.ClassDef) -> bool:
    for dec in cls_node.decorator_list:
        if not isinstance(dec, ast.Call):
            continue
        fn = dec.func
        is_dc = (isinstance(fn, ast.Name) and fn.id == "dataclass") or (
            isinstance(fn, ast.Attribute) and fn.attr == "dataclass"
        )
        if not is_dc:
            continue
        for kw in dec.keywords:
            if kw.arg == "slots" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                return True
    return False


def _is_classvar_annotation(ann: ast.AST | None) -> bool:
    if ann is None:
        return False
    if isinstance(ann, ast.Name) and ann.id == "ClassVar":
        return True
    if isinstance(ann, ast.Attribute) and ann.attr == "ClassVar":
        return True
    if isinstance(ann, ast.Subscript):
        return _is_classvar_annotation(ann.value)
    return False


def _annotation_matches_cls(ann: ast.AST | None, slotted_names: set[str]) -> str | None:
    if ann is None:
        return None
    if isinstance(ann, ast.Name) and ann.id in slotted_names:
        return ann.id
    if isinstance(ann, ast.Constant) and isinstance(ann.value, str):
        raw = ann.value.strip()
        if raw in slotted_names:
            return raw
        for name in slotted_names:
            if raw == f"Optional[{name}]" or raw == f"{name} | None" or raw == f"None | {name}":
                return name
        return None
    if isinstance(ann, ast.Subscript):
        base_id = (
            ann.value.id
            if isinstance(ann.value, ast.Name)
            else (ann.value.attr if isinstance(ann.value, ast.Attribute) else "")
        )
        if base_id in ("Optional", "Union"):
            elts = ann.slice.elts if isinstance(ann.slice, ast.Tuple) else [ann.slice]
            matches = {m for e in elts if (m := _annotation_matches_cls(e, slotted_names))}
            if len(matches) == 1:
                return next(iter(matches))
    if isinstance(ann, ast.BinOp) and isinstance(ann.op, ast.BitOr):
        m_left = _annotation_matches_cls(ann.left, slotted_names)
        m_right = _annotation_matches_cls(ann.right, slotted_names)
        if m_left and not m_right:
            return m_left
        if m_right and not m_left:
            return m_right
    return None


def _tuple_return_slots(ann: ast.AST | None, slotted_names: set[str]) -> list[str | None] | None:
    if isinstance(ann, ast.Subscript):
        base_id = (
            ann.value.id
            if isinstance(ann.value, ast.Name)
            else (ann.value.attr if isinstance(ann.value, ast.Attribute) else "")
        )
        if base_id.lower() == "tuple" and isinstance(ann.slice, ast.Tuple):
            return [_annotation_matches_cls(e, slotted_names) for e in ann.slice.elts]
    return None


def audit_dataclass_slots_assignments_ast(
    root: Path = _ROOT,
    rel_paths: Sequence[str] | None = None,
    *,
    _parsed_cache: ParsedCache | None = None,
) -> list[str]:
    """Verify every attribute write to a ``@dataclass(..., slots=True)`` targets a declared slot field."""
    root = Path(root)
    target_rels = [r for r in _normalize_target_rels(root, rel_paths) if _is_prod_python_rel(r)]
    violations: list[str] = []

    for rel_s in target_rels:
        _, tree, syntax_err = parse_module_cached(root / rel_s, rel_s, _parsed_cache)
        if syntax_err is not None or tree is None:
            continue

        raw_fields_by_cls: dict[str, set[str]] = {}
        cls_bases: dict[str, list[str]] = {}
        slotted_names: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            fields: set[str] = set()
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    if not _is_classvar_annotation(stmt.annotation):
                        fields.add(stmt.target.id)
                elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for dec in stmt.decorator_list:
                        if isinstance(dec, ast.Attribute) and dec.attr == "setter":
                            fields.add(stmt.name)
            raw_fields_by_cls[node.name] = fields
            cls_bases[node.name] = [b.id for b in node.bases if isinstance(b, ast.Name)]
            if _is_slotted_dataclass_def(node):
                slotted_names.add(node.name)

        if not slotted_names:
            continue

        slotted_fields: dict[str, set[str]] = {}
        for cname in slotted_names:
            acc: set[str] = set()
            stack = [cname]
            seen_c: set[str] = set()
            while stack:
                cur = stack.pop()
                if cur in seen_c:
                    continue
                seen_c.add(cur)
                acc.update(raw_fields_by_cls.get(cur, ()))
                stack.extend(cls_bases.get(cur, ()))
            slotted_fields[cname] = acc

        fn_return_cls: dict[str, str] = {}
        fn_return_tuple: dict[str, list[str | None]] = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if ret_c := _annotation_matches_cls(node.returns, slotted_names):
                    fn_return_cls[node.name] = ret_c
                elif ret_t := _tuple_return_slots(node.returns, slotted_names):
                    fn_return_tuple[node.name] = ret_t

        def _call_name(call_node: ast.Call) -> str:
            if isinstance(call_node.func, ast.Name):
                return call_node.func.id
            if isinstance(call_node.func, ast.Attribute):
                return call_node.func.attr
            return ""

        def _audit_fn(fn_node: ast.FunctionDef | ast.AsyncFunctionDef, enclosing_cls: str | None) -> None:
            var_types: dict[str, str] = {}
            all_args = [*fn_node.args.posonlyargs, *fn_node.args.args, *fn_node.args.kwonlyargs]
            if enclosing_cls in slotted_fields and all_args:
                var_types[all_args[0].arg] = enclosing_cls
            for arg in all_args:
                if arg_cls := _annotation_matches_cls(arg.annotation, slotted_names):
                    var_types[arg.arg] = arg_cls

            def _scan_stmts(stmts: list[ast.stmt]) -> None:
                for stmt in stmts:
                    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        _audit_fn(stmt, None)
                        continue
                    if isinstance(stmt, ast.ClassDef):
                        for c_stmt in stmt.body:
                            if isinstance(c_stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                _audit_fn(c_stmt, stmt.name)
                        continue
                    # Record local variable bindings first for assignments.
                    if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
                        cname_call = _call_name(stmt.value)
                        for tgt in stmt.targets:
                            if isinstance(tgt, ast.Name):
                                if cname_call in slotted_fields:
                                    var_types[tgt.id] = cname_call
                                elif cname_call in fn_return_cls:
                                    var_types[tgt.id] = fn_return_cls[cname_call]
                                else:
                                    var_types.pop(tgt.id, None)
                            elif isinstance(tgt, (ast.Tuple, ast.List)) and cname_call in fn_return_tuple:
                                ret_slots = fn_return_tuple[cname_call]
                                if len(tgt.elts) == len(ret_slots):
                                    for elt, slot_cls in zip(tgt.elts, ret_slots):
                                        if isinstance(elt, ast.Name):
                                            if slot_cls:
                                                var_types[elt.id] = slot_cls
                                            else:
                                                var_types.pop(elt.id, None)
                    elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                        if ann_c := _annotation_matches_cls(stmt.annotation, slotted_names):
                            var_types[stmt.target.id] = ann_c
                        elif isinstance(stmt.value, ast.Call):
                            cname_call = _call_name(stmt.value)
                            if cname_call in slotted_fields:
                                var_types[stmt.target.id] = cname_call

                    def _check_expr_writes(node: ast.AST) -> None:
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                            return
                        if (
                            isinstance(node, ast.Attribute)
                            and isinstance(node.ctx, ast.Store)
                            and isinstance(node.value, ast.Name)
                            and (cls_name := var_types.get(node.value.id)) is not None
                        ):
                            if node.attr not in slotted_fields[cls_name]:
                                violations.append(
                                    f"{rel_s}:{node.lineno}: assignment to undeclared slot {cls_name}.{node.attr} "
                                    f"(declared: {sorted(slotted_fields[cls_name])})"
                                )
                        elif (
                            isinstance(node, ast.Call)
                            and isinstance(node.func, ast.Name)
                            and node.func.id == "setattr"
                            and len(node.args) >= 2
                            and isinstance(node.args[0], ast.Name)
                            and isinstance(node.args[1], ast.Constant)
                            and isinstance(node.args[1].value, str)
                            and (cls_name := var_types.get(node.args[0].id)) is not None
                        ):
                            attr_s = node.args[1].value
                            if attr_s not in slotted_fields[cls_name]:
                                violations.append(
                                    f"{rel_s}:{node.lineno}: setattr to undeclared slot {cls_name}.{attr_s} "
                                    f"(declared: {sorted(slotted_fields[cls_name])})"
                                )
                        for child in ast.iter_child_nodes(node):
                            if not isinstance(child, ast.stmt):
                                _check_expr_writes(child)

                    for child in ast.iter_child_nodes(stmt):
                        if not isinstance(child, ast.stmt):
                            _check_expr_writes(child)

                    for child_list_attr in ("body", "orelse", "finalbody"):
                        if isinstance(sub_stmts := getattr(stmt, child_list_attr, None), list):
                            _scan_stmts(sub_stmts)
                    if isinstance(handlers := getattr(stmt, "handlers", None), list):
                        for h in handlers:
                            if isinstance(h, ast.ExceptHandler):
                                _scan_stmts(h.body)

            _scan_stmts(fn_node.body)

        for top_stmt in tree.body:
            if isinstance(top_stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                _audit_fn(top_stmt, None)
            elif isinstance(top_stmt, ast.ClassDef):
                for c_stmt in top_stmt.body:
                    if isinstance(c_stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        _audit_fn(c_stmt, top_stmt.name)
    return violations


# -- Combined SSOT AST audit runner --------------------------------------------

def audit_all_ssot_ast(
    root: Path = _ROOT,
    rel_paths: Sequence[str] | None = None,
    *,
    _parsed_cache: ParsedCache | None = None,
) -> list[str]:
    """Run all SSOT AST audits (primitives, venue branches, equity suffixes, DAG, cycles, producers, atomic writers, structural guards)."""
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
        *audit_evaluator_clock_purity_ast(root, rel_paths=target_rels, _parsed_cache=parsed_cache),
        *audit_equity_quote_provenance_ast(root, rel_paths=target_rels, _parsed_cache=parsed_cache),
        *audit_signed_zero_format_ast(root, rel_paths=target_rels, _parsed_cache=parsed_cache),
        *audit_cross_module_calls_and_imports_ast(root, rel_paths=target_rels, _parsed_cache=parsed_cache),
        *audit_cachedb_path_and_jsonl_purity_ast(root, rel_paths=target_rels, _parsed_cache=parsed_cache),
        *audit_durability_tier_and_cachedb_dirname_ast(root, rel_paths=target_rels, _parsed_cache=parsed_cache),
        *audit_venue_balances_single_writer_ast(root, rel_paths=target_rels, _parsed_cache=parsed_cache),
        *audit_state_file_owners_ast(root, rel_paths=target_rels, _parsed_cache=parsed_cache),
        *audit_dataclass_slots_assignments_ast(root, rel_paths=target_rels, _parsed_cache=parsed_cache),
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

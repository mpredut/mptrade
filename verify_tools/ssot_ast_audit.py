#!/usr/bin/env python3
"""Canonical SSOT AST purity auditor for tests and git pre-commit hooks.

Enforces repository-wide SSOT boundaries so manual symbol/quote/equity parsing,
venue string branching, raw stdlib I/O/HTTP/env/coercion calls, and transitive
re-export imports are blocked immediately at commit time rather than surfacing
later in full DEV regressions.

Subcommands
  pre-commit [files...]  Audit staged Python files (or explicit paths).
  scan                   Audit the entire repository.
"""
from __future__ import annotations

import argparse
import ast
import builtins
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from state_io import read_text_file
from verify_tools.translation_audit import PROD_PYTHON_SKIP_TOP, find_first_non_ascii_line

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


def _iter_repo_python_rels(root: Path) -> list[str]:
    rels: list[str] = []
    for p in sorted(root.rglob("*.py")):
        rel = p.relative_to(root)
        if any(part.startswith(".") or part in ("__pycache__", "site-packages") for part in rel.parts):
            continue
        rels.append(rel.as_posix())
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


def _parse_module(
    path: Path,
    rel_s: str,
    cache: dict[str, tuple[str | None, ast.Module | None, str | None]] | None = None,
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


def audit_repo_ssot_ast(
    root: Path = _ROOT,
    rel_paths: Sequence[str] | None = None,
    *,
    _parsed_cache: dict[str, tuple[str | None, ast.Module | None, str | None]] | None = None,
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
        src_text, tree, syntax_err = _parse_module(root / rel_s, rel_s, _parsed_cache)
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
        _, mod_tree, _ = _parse_module(mod_path, mod_rel, _parsed_cache)
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
        _, tree, syntax_err = _parse_module(root / rel_s, rel_s, _parsed_cache)
        if syntax_err is not None:
            ssot_violations.append(syntax_err)
            continue
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                _check_reexport_imports(rel_s, node)

    return ssot_violations


def audit_venue_string_branches_ast(
    root: Path = _ROOT,
    rel_paths: Sequence[str] | None = None,
    *,
    _parsed_cache: dict[str, tuple[str | None, ast.Module | None, str | None]] | None = None,
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
        _, tree, syntax_err = _parse_module(root / rel, rel, _parsed_cache)
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


def audit_equity_suffix_endswith_ast(
    root: Path = _ROOT,
    rel_paths: Sequence[str] | None = None,
    *,
    _parsed_cache: dict[str, tuple[str | None, ast.Module | None, str | None]] | None = None,
) -> list[str]:
    """Audit Python modules outside cachepaths.py for manual _EQ/_US_EQ .endswith() calls."""
    root = Path(root)
    target_rels = [
        r for r in _normalize_target_rels(root, rel_paths)
        if _is_equity_endswith_guarded(r)
    ]
    violations: list[str] = []
    for rel in target_rels:
        _, tree, syntax_err = _parse_module(root / rel, rel, _parsed_cache)
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


def audit_all_ssot_ast(
    root: Path = _ROOT,
    rel_paths: Sequence[str] | None = None,
) -> list[str]:
    """Run all SSOT AST audits (primitives/imports, venue branches, equity suffixes) and deduplicate."""
    root = Path(root)
    target_rels = _normalize_target_rels(root, rel_paths)
    parsed_cache: dict[str, tuple[str | None, ast.Module | None, str | None]] = {}
    combined: list[str] = []
    seen: set[str] = set()
    for err in (
        *audit_repo_ssot_ast(root, rel_paths=target_rels, _parsed_cache=parsed_cache),
        *audit_venue_string_branches_ast(root, rel_paths=target_rels, _parsed_cache=parsed_cache),
        *audit_equity_suffix_endswith_ast(root, rel_paths=target_rels, _parsed_cache=parsed_cache),
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

#!/usr/bin/env python3
"""Unified deterministic pre-commit and repository hygiene gate.

Invoked by `.githooks/pre-commit` inside a single Python process to orchestrate
every fast, deterministic hygiene, syntax, config, SSOT AST, DAG, ownership,
and order-boundary check in the repository.

Subcommands
  pre-commit [files...]  Run all deterministic gates on staged files (or explicit paths).
  scan                   Run all deterministic gates across the entire repository.
  list                   Display the registered deterministic gates.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Iterable, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from botcore import parse_nonneg_cli_int
from core.config import load_key_value_conf
from instrument_registry import load_registry
from state_io import parse_json_payload, read_bytes_file, read_text_file
from verify_tools.direct_order_api_inventory import audit_direct_order_boundaries
from verify_tools.ssot_ast_audit import (
    ParsedCache,
    audit_bespoke_atomic_writers,
    audit_cachedb_path_and_jsonl_purity_ast,
    audit_cross_module_calls_and_imports_ast,
    audit_dataclass_slots_assignments_ast,
    audit_durability_tier_and_cachedb_dirname_ast,
    audit_equity_quote_provenance_ast,
    audit_equity_suffix_endswith_ast,
    audit_evaluator_clock_purity_ast,
    audit_import_cycles_ast,
    audit_layer_dag_ast,
    audit_producer_only_caches_ast,
    audit_repo_ssot_ast,
    audit_signed_zero_format_ast,
    audit_state_file_owners_ast,
    audit_venue_balances_single_writer_ast,
    audit_venue_string_branches_ast,
)
from verify_tools.translation_audit import (
    TEXT_FILENAMES,
    TEXT_SUFFIXES,
    cmd_pre_commit as _translation_pre_commit,
)

REQUIRED_OPERATIONAL_CONFIG_KEYS = frozenset({
    "CM_CURRENT_PRICE_POLL_SEC",
    "CM_PRICE_FRESHNESS_MAX_AGE_SEC",
    "WATCHDOG_STALE_MINUTES",
    "WATCHDOG_COOLDOWN_MINUTES",
    "WATCHDOG_FAST_PRICE_MINUTES",
    "ANOMALY_WINDOW_FILES_MIN",
    "ANOMALY_COOLDOWN_MINUTES",
    "ANOMALY_MAX_READ_BYTES",
    "ANOMALY_THRESH_RATE_LIMIT",
    "ANOMALY_THRESH_AUTH",
    "ANOMALY_THRESH_BLIND",
    "ANOMALY_THRESH_TRACEBACK",
    "MONITOR_PORT",
})
OPERATIONAL_THRESHOLD_RUNTIME_FILES = frozenset({
    "cacheprice.py",
    "cacheManager.py",
    "orchestratorOS/lib/technical_repair_ssot.py",
    "verify_tools/check_cache_coherence.py",
    "verify_tools/monitor_night.py",
})


@dataclass
class GateContext:
    root: Path
    mode: str
    rel_paths: tuple[str, ...] | None
    python_rels: tuple[str, ...] | None
    explicit_files: bool = False
    min_ro_score: int = 2
    parsed_cache: ParsedCache = field(default_factory=dict)


@dataclass(frozen=True)
class DeterministicGate:
    name: str
    description: str
    run: Callable[[GateContext], list[str]]


def _norm_rel(raw: str, root: Path) -> str | None:
    p = Path(raw)
    if not p.is_absolute():
        return p.as_posix().lstrip("./")
    try:
        return p.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return None


def _git_ls_files(root: Path) -> list[str] | None:
    if not (root / ".git").exists():
        return None
    proc = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=str(root),
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return [r for r in proc.stdout.decode("utf-8", errors="replace").split("\0") if r]


def _staged_rels(root: Path, explicit_files: Iterable[str] | None = None) -> list[str]:
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
    seen: set[str] = set()
    for raw in candidates:
        rel_s = _norm_rel(raw, root)
        if not rel_s or rel_s in seen:
            continue
        if (root / rel_s).is_file():
            seen.add(rel_s)
            rels.append(rel_s)
    return rels


def _is_shell_script_rel(rel_s: str) -> bool:
    return rel_s.endswith(".sh") or rel_s.startswith(".githooks/")


def _gate_text_hygiene(ctx: GateContext) -> list[str]:
    if ctx.mode == "pre-commit":
        # When explicit files are passed (e.g., tests or manual file list), pass absolute paths so
        # translation_audit does not invoke `git add` on synthetic files.
        files_arg = (
            [str(ctx.root / r) for r in (ctx.rel_paths or ())]
            if ctx.explicit_files
            else []
        )
        rc = _translation_pre_commit(SimpleNamespace(files=files_arg, min_score=ctx.min_ro_score))
        return ["text hygiene / ASCII / Romanian prose check failed (see details above)"] if rc != 0 else []

    # Full-repo scan mode: verify zero CR/CRLF line endings across tracked text files (non-mutating).
    tracked = _git_ls_files(ctx.root)
    if tracked is None:
        tracked = [p.relative_to(ctx.root).as_posix() for p in sorted(ctx.root.rglob("*")) if p.is_file()]
    violations: list[str] = []
    for rel_s in tracked:
        p = ctx.root / rel_s
        if not p.is_file():
            continue
        if p.suffix.lower() not in TEXT_SUFFIXES and p.name not in TEXT_FILENAMES:
            continue
        raw = read_bytes_file(str(p), default=b"")
        if raw and b"\r" in raw:
            violations.append(f"{rel_s}: contains CR/CRLF line endings (use Unix LF)")
    return violations


def audit_shell_syntax(root: Path = _ROOT, rel_paths: Sequence[str] | None = None) -> list[str]:
    """Validate shell syntax with `bash -n` on `.sh` scripts and `.githooks/*`."""
    root = Path(root)
    if rel_paths is not None:
        targets = [r for r in rel_paths if _is_shell_script_rel(r) and (root / r).is_file()]
    else:
        tracked = _git_ls_files(root)
        if tracked is not None:
            targets = [r for r in tracked if _is_shell_script_rel(r) and (root / r).is_file()]
        else:
            targets = [
                p.relative_to(root).as_posix()
                for p in sorted(root.rglob("*"))
                if p.is_file() and _is_shell_script_rel(p.relative_to(root).as_posix())
            ]
    violations: list[str] = []
    for rel_s in targets:
        proc = subprocess.run(
            ["bash", "-n", str(root / rel_s)],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout).strip().splitlines()
            detail = err[0] if err else "syntax error"
            violations.append(f"{rel_s}: bash -n failed ({detail})")
    return violations


def audit_static_config_policy(root: Path = _ROOT, rel_paths: Sequence[str] | None = None) -> list[str]:
    """Validate JSON/TOML/INI syntax, config.env shell safety & SSOT keys, and instruments.conf schema."""
    root = Path(root)
    violations: list[str] = []
    if rel_paths is not None:
        rel_set = set(rel_paths)
        json_targets = [r for r in rel_paths if r.endswith(".json") and (root / r).is_file()]
        toml_targets = [r for r in rel_paths if r.endswith(".toml") and (root / r).is_file()]
        check_config_env = "config.env" in rel_set
        check_runtime_fallbacks = sorted(rel_set & OPERATIONAL_THRESHOLD_RUNTIME_FILES)
        check_instruments = "instruments.conf" in rel_set
        check_order_guard_conf = "order_guard.conf" in rel_set
    else:
        tracked = _git_ls_files(root)
        all_rels = tracked if tracked is not None else [
            p.relative_to(root).as_posix() for p in sorted(root.rglob("*")) if p.is_file()
        ]
        json_targets = [r for r in all_rels if r.endswith(".json") and (root / r).is_file()]
        toml_targets = [r for r in all_rels if r.endswith(".toml") and (root / r).is_file()]
        check_config_env = (root / "config.env").is_file()
        check_runtime_fallbacks = sorted(
            r for r in OPERATIONAL_THRESHOLD_RUNTIME_FILES if (root / r).is_file()
        )
        check_instruments = (root / "instruments.conf").is_file()
        check_order_guard_conf = (root / "order_guard.conf").is_file()

    for rel_s in json_targets:
        raw = read_text_file(str(root / rel_s), default=None)
        if raw is None:
            continue
        try:
            parse_json_payload(raw)
        except Exception as exc:
            violations.append(f"{rel_s}: invalid JSON ({exc})")

    for rel_s in toml_targets:
        raw = read_text_file(str(root / rel_s), default=None)
        if raw is None:
            continue
        try:
            tomllib.loads(raw)
        except Exception as exc:
            violations.append(f"{rel_s}: invalid TOML ({exc})")

    if check_config_env and (root / "config.env").is_file():
        proc = subprocess.run(
            ["/bin/bash", "-c", "set -u; set -a; . ./config.env && echo SOURCE_OK"],
            cwd=str(root),
            env={},
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        if proc.returncode != 0 or "SOURCE_OK" not in proc.stdout:
            detail = (proc.stderr or proc.stdout).strip().splitlines()
            violations.append(
                f"config.env: failed `set -u` empty-env shell sourcing ({detail[0] if detail else 'error'})"
            )
        cfg_text = read_text_file(str(root / "config.env"), default="") or ""
        configured = {
            ln.split("=", 1)[0].strip()
            for ln in cfg_text.splitlines()
            if "=" in ln and not ln.lstrip().startswith("#")
        }
        missing_keys = sorted(REQUIRED_OPERATIONAL_CONFIG_KEYS - configured)
        if missing_keys:
            violations.append(f"config.env: missing required SSOT operational key(s): {missing_keys}")

    for rel_s in check_runtime_fallbacks:
        text = read_text_file(str(root / rel_s), default="") or ""
        for key in sorted(REQUIRED_OPERATIONAL_CONFIG_KEYS):
            if f'os.environ.get("{key}",' in text:
                violations.append(f"{rel_s}: hardcoded fallback default for SSOT key {key}")

    if check_instruments and (root / "instruments.conf").is_file():
        try:
            load_registry(root / "instruments.conf")
        except Exception as exc:
            violations.append(f"instruments.conf: registry validation failed ({exc})")

    if check_order_guard_conf and (root / "order_guard.conf").is_file():
        raw_og = read_text_file(str(root / "order_guard.conf"), default="") or ""
        for idx, ln in enumerate(raw_og.splitlines(), 1):
            stripped = ln.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "=" not in stripped:
                violations.append(f"order_guard.conf:{idx}: malformed key=value line")
        try:
            kv = load_key_value_conf(root, root, filenames=("order_guard.conf",))
            if not kv or "default" not in kv:
                violations.append("order_guard.conf: missing required key 'default'")
        except Exception as exc:
            violations.append(f"order_guard.conf: failed to parse key=value conf ({exc})")

    return violations


DETERMINISTIC_GATES: tuple[DeterministicGate, ...] = (
    DeterministicGate(
        "text_hygiene",
        "Auto-fix/verify Unix LF, 7-bit ASCII, and English-only prose (translation_audit)",
        _gate_text_hygiene,
    ),
    DeterministicGate(
        "shell_syntax",
        "Validate `.sh` scripts and `.githooks/*` with `bash -n`",
        lambda ctx: audit_shell_syntax(ctx.root, ctx.rel_paths),
    ),
    DeterministicGate(
        "config_policy",
        "Validate JSON/TOML/INI syntax, `config.env` shell-safety/SSOT keys, and `instruments.conf`",
        lambda ctx: audit_static_config_policy(ctx.root, ctx.rel_paths),
    ),
    DeterministicGate(
        "ssot_primitives_ast",
        "Enforce zero raw open/float/int/json/Path/env/HTTP/quote-endswith/re-export/undefined-name AST rules",
        lambda ctx: audit_repo_ssot_ast(ctx.root, rel_paths=ctx.python_rels, _parsed_cache=ctx.parsed_cache),
    ),
    DeterministicGate(
        "venue_branches_ast",
        "Enforce zero hardcoded venue string comparisons outside leaf adapters",
        lambda ctx: audit_venue_string_branches_ast(
            ctx.root, rel_paths=ctx.python_rels, _parsed_cache=ctx.parsed_cache
        ),
    ),
    DeterministicGate(
        "equity_suffix_ast",
        "Enforce zero manual `_EQ`/`_US_EQ` `.endswith()` checks outside `cachepaths.py`",
        lambda ctx: audit_equity_suffix_endswith_ast(
            ctx.root, rel_paths=ctx.python_rels, _parsed_cache=ctx.parsed_cache
        ),
    ),
    DeterministicGate(
        "layer_dag_ast",
        "Enforce strict acyclic layer DAG, cache sibling/WS isolation, venue SDK bans, strategies/trailing purity",
        lambda ctx: audit_layer_dag_ast(ctx.root, rel_paths=ctx.python_rels, _parsed_cache=ctx.parsed_cache),
    ),
    DeterministicGate(
        "import_cycles_ast",
        "Detect circular imports (lazy DFS on staged modules in pre-commit; full Tarjan SCC in scan)",
        lambda ctx: audit_import_cycles_ast(ctx.root, rel_paths=ctx.python_rels, _parsed_cache=ctx.parsed_cache),
    ),
    DeterministicGate(
        "producer_only_caches_ast",
        "Enforce single-writer producer cache ownership and ban legacy cache kwargs",
        lambda ctx: audit_producer_only_caches_ast(
            ctx.root, rel_paths=ctx.python_rels, _parsed_cache=ctx.parsed_cache
        ),
    ),
    DeterministicGate(
        "atomic_writers_ast",
        "Enforce zero bespoke mkstemp/tmp+os.replace atomic writers outside `state_io.py`",
        lambda ctx: audit_bespoke_atomic_writers(
            ctx.root, rel_paths=ctx.python_rels, _parsed_cache=ctx.parsed_cache
        ),
    ),
    DeterministicGate(
        "direct_order_boundaries_ast",
        "Enforce reviewed allowlist for direct venue order submit/cancel APIs",
        lambda ctx: audit_direct_order_boundaries(
            ctx.root, rel_paths=ctx.python_rels, _parsed_cache=ctx.parsed_cache
        ),
    ),
    DeterministicGate(
        "evaluator_clock_purity_ast",
        "Enforce zero bare time.time()/datetime.now() wall-clock calls inside stateless evaluators",
        lambda ctx: audit_evaluator_clock_purity_ast(
            ctx.root, rel_paths=ctx.python_rels, _parsed_cache=ctx.parsed_cache
        ),
    ),
    DeterministicGate(
        "equity_quote_provenance_ast",
        "Enforce single atomic writer (_apply_resolved_equity_quote) for quote_source/quote_age_min",
        lambda ctx: audit_equity_quote_provenance_ast(
            ctx.root, rel_paths=ctx.python_rels, _parsed_cache=ctx.parsed_cache
        ),
    ),
    DeterministicGate(
        "signed_zero_format_ast",
        "Enforce botcore.fmt_signed instead of raw :+.Nf specs in prompt/telemetry code (prevents -0.00%)",
        lambda ctx: audit_signed_zero_format_ast(
            ctx.root, rel_paths=ctx.python_rels, _parsed_cache=ctx.parsed_cache
        ),
    ),
    DeterministicGate(
        "venue_balances_single_writer_ast",
        "Enforce canonical single writers & producers for cache_balance_<venue>.json and portfolio_positions_state.json",
        lambda ctx: audit_venue_balances_single_writer_ast(
            ctx.root, rel_paths=ctx.python_rels, _parsed_cache=ctx.parsed_cache
        ),
    ),
    DeterministicGate(
        "state_file_owners_ast",
        "Enforce 1:1 state_stores/<file_stem>.py symmetry and exclusive read/write ownership of critical state files",
        lambda ctx: audit_state_file_owners_ast(
            ctx.root, rel_paths=ctx.python_rels, _parsed_cache=ctx.parsed_cache
        ),
    ),
    DeterministicGate(
        "cross_module_calls_and_imports_ast",
        "Verify local from-import symbol existence and direct top-level call signatures across prod modules",
        lambda ctx: audit_cross_module_calls_and_imports_ast(
            ctx.root, rel_paths=ctx.python_rels, _parsed_cache=ctx.parsed_cache
        ),
    ),
    DeterministicGate(
        "cachedb_path_and_jsonl_purity_ast",
        "Enforce resolved cachedb subdomain checks and prevent JSON dict readers on JSONL cache files",
        lambda ctx: audit_cachedb_path_and_jsonl_purity_ast(
            ctx.root, rel_paths=ctx.python_rels, _parsed_cache=ctx.parsed_cache
        ),
    ),
    DeterministicGate(
        "durability_tier_and_cachedb_dirname_ast",
        "Block fsync=False on FINANCIAL_CRITICAL state paths and os.path.dirname(state_path) as a cachedb root",
        lambda ctx: audit_durability_tier_and_cachedb_dirname_ast(
            ctx.root, rel_paths=ctx.python_rels, _parsed_cache=ctx.parsed_cache
        ),
    ),
    DeterministicGate(
        "dataclass_slots_assignments_ast",
        "Verify every attribute write to a @dataclass(slots=True) targets a declared slot field",
        lambda ctx: audit_dataclass_slots_assignments_ast(
            ctx.root, rel_paths=ctx.python_rels, _parsed_cache=ctx.parsed_cache
        ),
    ),
)


def run_gates(ctx: GateContext, gates: Sequence[DeterministicGate] = DETERMINISTIC_GATES) -> dict[str, list[str]]:
    """Run all registered deterministic gates and return a map of gate_name -> violations."""
    results: dict[str, list[str]] = {}
    for gate in gates:
        violations = gate.run(ctx)
        if violations:
            results[gate.name] = violations
    return results


def _print_failures(prefix: str, failures: dict[str, list[str]]) -> None:
    total = sum(len(v) for v in failures.values())
    print(f"\n[{prefix}] Blocked: {total} deterministic gate violation(s) across {len(failures)} gate(s):")
    for gate_name, items in failures.items():
        print(f"  -- {gate_name} ({len(items)}) --")
        for err in items:
            print(f"    {err}")


def cmd_pre_commit(args) -> int:
    root = Path(getattr(args, "root", None) or _ROOT)
    explicit = list(args.files) if getattr(args, "files", None) else None
    rels = _staged_rels(root, explicit)
    if not rels:
        return 0
    py_rels = tuple(r for r in rels if r.endswith(".py"))
    ctx = GateContext(
        root=root,
        mode="pre-commit",
        rel_paths=tuple(rels),
        python_rels=py_rels,
        explicit_files=explicit is not None,
        min_ro_score=getattr(args, "min_score", 2),
    )
    failures = run_gates(ctx)
    if failures:
        _print_failures("pre-commit", failures)
        return 1
    return 0


def cmd_scan(args) -> int:
    root = Path(getattr(args, "root", None) or _ROOT)
    ctx = GateContext(
        root=root,
        mode="scan",
        rel_paths=None,
        python_rels=None,
        explicit_files=False,
        min_ro_score=getattr(args, "min_score", 2),
    )
    failures = run_gates(ctx)
    if failures:
        _print_failures("scan", failures)
        return 1
    print(f"OK: all {len(DETERMINISTIC_GATES)} deterministic gates passed with zero violations.")
    return 0


def cmd_list(_args) -> int:
    for idx, gate in enumerate(DETERMINISTIC_GATES, 1):
        print(f"{idx:>2}. {gate.name:<28} {gate.description}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    pre = sub.add_parser("pre-commit", help="run all deterministic gates on staged files")
    pre.add_argument("files", nargs="*", help="optional explicit file paths")
    pre.add_argument("--root", default=None, help="optional repository root override")
    pre.add_argument("--min-score", type=parse_nonneg_cli_int, default=2, help="Romanian prose score threshold (default: 2)")
    pre.set_defaults(func=cmd_pre_commit)

    scan = sub.add_parser("scan", help="run all deterministic gates across the entire repository")
    scan.add_argument("--root", default=None, help="optional repository root override")
    scan.add_argument("--min-score", type=parse_nonneg_cli_int, default=2, help="Romanian prose score threshold (default: 2)")
    scan.set_defaults(func=cmd_scan)

    lst = sub.add_parser("list", help="list all registered deterministic gates")
    lst.set_defaults(func=cmd_list)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

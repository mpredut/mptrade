"""A generic walk-forward evaluator for strategy adapters.

The strategy stays venue-specific. The shared contract is a pure function
``replay(ohlc, warmup_ohlc, context) -> metrics``, so Kraken and Trading212 can
use their own live engine without copying the window, buy-and-hold and
aggregation logic. The context carries the timestamps the historical FX needs.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from collections import Counter
from dataclasses import dataclass
import datetime as dt
import importlib.util
import os
from pathlib import Path
from statistics import fmean
import sys
from typing import Any

from offline.backtests.execution import ExecutionModel
from offline.backtests.metrics import validate_replay_metrics
from offline.backtests.walk_forward import summarize_test_windows, walk_forward_splits
from botcore import exact_pos_int, float_env, parse_dotenv, safe_float, safe_int


Ohlc = list[tuple[float, float, float, float]]


@dataclass(frozen=True)
class ReplayContext:
    """The temporal identity, kept separate from the strategy's OHLC tuples."""

    timestamps: tuple[int, ...]
    warmup_timestamps: tuple[int, ...]


ReplayFunction = Callable[[Ohlc, Ohlc, ReplayContext], dict[str, Any]]


def assess_decision_evidence(
    records: Sequence[dict], aggregate: dict[str, Any], *,
    minimum_folds: int = 20, minimum_history_days: float = 90.0,
    minimum_cycles: int = 3,
) -> dict[str, Any]:
    """Separate a lack of statistical evidence from unfavourable financial signals.

    A negative worst fold is not automatically a reason to optimise: it is a risk flag.
    Too little history, too few folds or too few cycles makes the result mere
    characterisation, even if the mean is positive.
    """
    if not records:
        raise ValueError("evaluating the evidence requires at least one bar")
    raw_first_ts = records[0].get("timestamp") if isinstance(records[0], dict) else None
    raw_last_ts = records[-1].get("timestamp") if isinstance(records[-1], dict) else None
    first_ts_f = safe_float(raw_first_ts, default=None)
    last_ts_f = safe_float(raw_last_ts, default=None)
    if first_ts_f is None or last_ts_f is None:
        raise ValueError("invalid record timestamps")
    history_days = max(0.0, (int(last_ts_f) - int(first_ts_f)) / 86_400.0)
    agg = aggregate if isinstance(aggregate, dict) else {}
    fold_count = safe_int(agg.get("fold_count"), 0)
    cycles = safe_int(agg.get("total_cycles"), 0)
    worst_return = safe_float(agg.get("worst_return_pct"), 0.0)
    data_issues = []
    if fold_count < minimum_folds:
        data_issues.append(f"folds {fold_count} < {minimum_folds}")
    if history_days < minimum_history_days:
        data_issues.append(
            f"history_days {history_days:.1f} < {minimum_history_days:.1f}"
        )
    if cycles < minimum_cycles:
        data_issues.append(f"completed_cycles {cycles} < {minimum_cycles}")
    risk_flags = []
    if worst_return < 0:
        risk_flags.append(f"negative_worst_fold {worst_return:+.3f}%")
    if data_issues and risk_flags:
        verdict = "characterization_only_with_risk_flags"
    elif data_issues:
        verdict = "characterization_only"
    elif risk_flags:
        verdict = "decision_grade_with_risk_flags"
    else:
        verdict = "decision_grade"
    return {
        "verdict": verdict,
        "decision_grade_data": not data_issues,
        "history_days": history_days,
        "minimums": {
            "folds": minimum_folds,
            "history_days": minimum_history_days,
            "completed_cycles": minimum_cycles,
        },
        "data_issues": data_issues,
        "risk_flags": risk_flags,
    }


def automatic_window_sizes(sample_count: int) -> tuple[int, int, int, int]:
    """Trei fold-uri aproximative: 45% train, 10% validation, 15% test."""
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 20:
        raise ValueError("at least 20 bars are required for automatic windows")
    train = max(1, int(sample_count * 0.45))
    validation = max(1, int(sample_count * 0.10))
    test = max(1, int(sample_count * 0.15))
    return train, validation, test, test


def iso_utc(timestamp: int) -> str:
    return dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc).isoformat()


def to_ohlc(records: Sequence[dict]) -> list[tuple[float, float, float, float]]:
    return [
        (
            safe_float(row["open"], 0.0),
            safe_float(row["high"], 0.0),
            safe_float(row["low"], 0.0),
            safe_float(row["close"], 0.0),
        )
        for row in records
    ]


def evaluate_segment(
    records: Sequence[dict],
    replay: ReplayFunction,
    *,
    warmup_records: Sequence[dict] = (),
    validate_contract: bool = False,
) -> dict:
    if not records:
        raise ValueError("segmentul de evaluare cannot be empty")
    metrics = replay(
        to_ohlc(records),
        to_ohlc(warmup_records),
        ReplayContext(
            timestamps=tuple(int(row["timestamp"]) for row in records),
            warmup_timestamps=tuple(int(row["timestamp"]) for row in warmup_records),
        ),
    )
    if not isinstance(metrics, dict):
        raise ValueError("the replay adapter must return a metrics dictionary")
    required = {"return_pct", "max_drawdown_pct", "cycles", "fills"}
    missing = required.difference(metrics)
    if missing:
        raise ValueError(f"the replay adapter did not supply the metrics: {sorted(missing)}")
    if validate_contract:
        validate_replay_metrics(metrics)
    first_close = safe_float(records[0]["close"], 0.0)
    last_close = safe_float(records[-1]["close"], 0.0)
    buy_hold_pct = (last_close / first_close - 1.0) * 100.0 if first_close > 0 else None
    return {
        "start_utc": iso_utc(int(records[0]["timestamp"])),
        "end_utc": iso_utc(int(records[-1]["timestamp"])),
        "bars": len(records),
        "buy_hold_return_pct": buy_hold_pct,
        "metrics": metrics,
    }


def evaluate_walk_forward(
    records: Sequence[dict],
    replay: ReplayFunction,
    *,
    train_size: int,
    validation_size: int,
    test_size: int,
    step_size: int,
    anchored_train: bool = False,
    warmup_bars: int = 0,
    validate_contract: bool = False,
) -> dict:
    if isinstance(warmup_bars, bool) or not isinstance(warmup_bars, int) or warmup_bars < 0:
        raise ValueError("warmup_bars cannot be negative")
    folds = walk_forward_splits(
        len(records),
        train_size=train_size,
        validation_size=validation_size,
        test_size=test_size,
        step_size=step_size,
        anchored_train=anchored_train,
    )
    if not folds:
        raise ValueError(
            f"not enough data: {len(records)} bars for train={train_size}, "
            f"validation={validation_size}, test={test_size}"
        )

    fold_results = []
    test_windows = []
    for index, fold in enumerate(folds, start=1):
        def warmup_before(segment: slice) -> Sequence[dict]:
            return records[max(0, segment.start - warmup_bars):segment.start]

        train = evaluate_segment(
            records[fold.train], replay, warmup_records=warmup_before(fold.train),
            validate_contract=validate_contract,
        )
        validation = evaluate_segment(
            records[fold.validation], replay, warmup_records=warmup_before(fold.validation),
            validate_contract=validate_contract,
        )
        test = evaluate_segment(
            records[fold.test], replay, warmup_records=warmup_before(fold.test),
            validate_contract=validate_contract,
        )
        fold_results.append({
            "fold": index,
            "train": train,
            "validation": validation,
            "test": test,
        })
        test_windows.append({
            "return_pct": test["metrics"]["return_pct"],
            "max_drawdown_pct": test["metrics"]["max_drawdown_pct"],
            "buy_hold_return_pct": test["buy_hold_return_pct"],
            "cycles": test["metrics"]["cycles"],
            "fills": test["metrics"]["fills"],
            "ambiguous_bars": test["metrics"].get("ambiguous_bars", 0),
            "intrabar_policy_selected": test["metrics"].get(
                "intrabar_policy_selected"
            ),
        })

    summary = summarize_test_windows(test_windows)
    compounded = 100.0
    for window in test_windows:
        compounded *= 1.0 + safe_float(window.get("return_pct"), 0.0) / 100.0
    bh_values = [
        bh_f
        for window in test_windows
        if (bh_f := safe_float(window.get("buy_hold_return_pct"), default=None)) is not None
    ]

    return {
        "folds": fold_results,
        "aggregate_test": {
            "fold_count": summary["window_count"],
            "mean_return_pct": summary["mean_return_pct"],
            "median_return_pct": summary["median_return_pct"],
            "worst_return_pct": summary["worst_return_pct"],
            "compounded_reset_return_pct": compounded - 100.0,
            "worst_max_drawdown_pct": summary["worst_max_drawdown_pct"],
            "worst_sortino": _optional_min(
                [fold["test"]["metrics"].get("sortino") for fold in fold_results]
            ),
            "mean_buy_hold_return_pct": fmean(bh_values) if bh_values else None,
            "total_cycles": summary["total_cycles"],
            "total_fills": summary["total_fills"],
            "total_ambiguous_bars": sum(
                max(0, safe_int(window.get("ambiguous_bars"), 0)) for window in test_windows
            ),
            "intrabar_policy_selected_counts": dict(Counter(
                window["intrabar_policy_selected"] for window in test_windows
                if window["intrabar_policy_selected"] is not None
            )),
        },
    }


def _optional_min(values: list[float | None]) -> float | None:
    present = [vf for value in values if (vf := safe_float(value, default=None)) is not None]
    return min(present) if present else None


def parse_intervals(value: str) -> list[int]:
    """Parse comma-separated positive integer interval minutes."""
    if not isinstance(value, str):
        raise argparse.ArgumentTypeError("intervals must be positive minutes")
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("intervals must be positive minutes")
    intervals = []
    for part in parts:
        iv = exact_pos_int(part, default=None)
        if iv is None:
            raise argparse.ArgumentTypeError("intervals must be positive minutes")
        intervals.append(iv)
    return intervals


def extract_cfg_float(cfg: Any, key: str, default: float) -> float:
    """Extract a finite float parameter from configuration mapping using botcore.float_env."""
    if not isinstance(cfg, (dict, Mapping)):
        return default
    val = float_env(key, cfg)
    return default if val is None else val


def extract_cfg_int(cfg: Any, key: str, default: int) -> int:
    """Extract an integer parameter from configuration mapping."""
    val = extract_cfg_float(cfg, key, float(default))
    return int(val)


def load_venue_module(
    subdir: str,
    filename: str = "replay.py",
    module_name: str | None = None,
    *,
    reload: bool = False,
):
    """Load a module from a venue directory (e.g. 'hyperliquid', '212trading') while preserving sys.path."""
    root = Path(__file__).resolve().parents[2]
    venue_dir = (root / subdir).resolve()
    file_path = venue_dir / (filename if filename.endswith(".py") else f"{filename}.py")
    mod_name = module_name or f"{subdir}_{file_path.stem}"
    if not reload and mod_name in sys.modules and sys.modules[mod_name] is not None:
        cached_mod = sys.modules[mod_name]
        cached_file = getattr(cached_mod, "__file__", None)
        if cached_file:
            try:
                if Path(cached_file).resolve().is_relative_to(venue_dir):
                    return cached_mod
            except (ValueError, TypeError):
                pass
    orig_sys_path = list(sys.path)

    _AMBIGUOUS = ("strategy", "market_data", "replay", "common", "notify", "hl_client", "t212_client")
    saved_ambiguous: dict[str, Any] = {}
    for name in _AMBIGUOUS:
        if name in sys.modules:
            mod_file = getattr(sys.modules[name], "__file__", "") or ""
            if mod_file:
                try:
                    is_rel = Path(mod_file).resolve().is_relative_to(venue_dir)
                except (ValueError, TypeError):
                    is_rel = False
                if not is_rel:
                    saved_ambiguous[name] = sys.modules.pop(name)

    try:
        venue_str = str(venue_dir)
        root_str = str(root)
        while venue_str in sys.path:
            sys.path.remove(venue_str)
        while root_str in sys.path:
            sys.path.remove(root_str)
        sys.path.insert(0, root_str)
        sys.path.insert(0, venue_str)

        spec = importlib.util.spec_from_file_location(mod_name, str(file_path))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        try:
            spec.loader.exec_module(mod)
        except Exception:
            sys.modules.pop(mod_name, None)
            raise
        return mod
    finally:
        sys.path[:] = orig_sys_path
        for name, saved_mod in saved_ambiguous.items():
            sys.modules[name] = saved_mod


def load_layered_env_config(
    config_file: str | Path,
    env_file: str | Path,
    *,
    override_prefixes: Sequence[str] = (),
    fallback_prefixes: Sequence[str] = ("STRAT_",),
) -> dict[str, str]:
    """Load config_file then env_file, applying os.environ overrides and fallbacks."""
    cfg: dict[str, str] = {}
    for path_str in (config_file, env_file):
        p = Path(path_str)
        if p.exists():
            cfg.update(dict(parse_dotenv(str(p))))
    ov_tuple = tuple(override_prefixes)
    fb_tuple = tuple(fallback_prefixes)
    for k, v in os.environ.items():
        if (ov_tuple and k.startswith(ov_tuple)) or (fb_tuple and k.startswith(fb_tuple) and k not in cfg):
            cfg[k] = v
    return cfg


def parse_dataset_specs(specs: list[str]) -> dict[int, Path]:
    """Parse repeatable INTERVAL=PATH CLI specifications into a resolved Path mapping."""
    result: dict[int, Path] = {}
    for spec in specs:
        if not isinstance(spec, str):
            raise ValueError(f"invalid dataset spec {spec!r}; format is INTERVAL=PATH")
        try:
            interval, path = spec.split("=", 1)
            interval_i = exact_pos_int(interval.strip(), default=None)
            p_str = path.strip()
            if interval_i is None or not p_str:
                raise ValueError("interval must be positive and path non-empty")
            result[interval_i] = Path(p_str).expanduser().resolve()
        except (ValueError, OSError) as exc:
            raise ValueError(f"invalid dataset spec {spec!r}; format is INTERVAL=PATH") from exc
    return result


def add_walk_forward_cli_args(
    parser: argparse.ArgumentParser,
    *,
    include_warmup: bool = True,
    default_warmup: int | None = 0,
) -> None:
    """Register standard execution model and walk-forward window arguments on a CLI parser."""
    parser.add_argument("--spread-bps", type=float, default=0.0)
    parser.add_argument("--market-slippage-bps", type=float, default=0.0)
    parser.add_argument("--partial-fill-ratio", type=float, default=1.0)
    parser.add_argument(
        "--intrabar-policy",
        choices=("buy_first", "sell_first", "worst_case"),
        default="buy_first",
    )
    parser.add_argument("--train", type=int, help="training bars; automatic by default")
    parser.add_argument("--validation", type=int, help="validation bars; automatic by default")
    parser.add_argument("--test", type=int, help="test bars; automatic by default")
    parser.add_argument("--step", type=int, help="step size in bars; defaults to test size")
    if include_warmup:
        parser.add_argument(
            "--warmup",
            type=int,
            default=default_warmup,
            help="preceding signal bars; they carry no position or P&L into the segment",
        )


def build_execution_model_from_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> ExecutionModel:
    """Validate walk-forward window CLI flags and construct ExecutionModel."""
    explicit_windows = (args.train, args.validation, args.test)
    if any(isinstance(value, bool) for value in (*explicit_windows, args.step)):
        parser.error("the walk-forward sizes must be positive")
    if any(value is not None for value in explicit_windows) and not all(
        value is not None for value in explicit_windows
    ):
        parser.error("--train, --validation and --test must be supplied together")
    if any(value is not None and value <= 0 for value in (*explicit_windows, args.step)):
        parser.error("the walk-forward sizes must be positive")
    warmup = getattr(args, "warmup", None)
    if isinstance(warmup, bool) or (warmup is not None and warmup < 0):
        parser.error("--warmup cannot be negative")
    try:
        return ExecutionModel(
            spread_bps=args.spread_bps,
            market_slippage_bps=args.market_slippage_bps,
            partial_fill_ratio=args.partial_fill_ratio,
            intrabar_policy=args.intrabar_policy,
        )
    except ValueError as exc:
        parser.error(str(exc))


def resolve_window_sizes(
    args: argparse.Namespace, record_count: int
) -> tuple[int, int, int, int]:
    """Return (train, validation, test, step) sizes from CLI args or automatic sizing."""
    if args.train is None:
        return automatic_window_sizes(record_count)
    return args.train, args.validation, args.test, (args.step or args.test)

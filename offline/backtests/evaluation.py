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
import dataclasses
from dataclasses import dataclass
import datetime as dt
from pathlib import Path
from statistics import fmean
from typing import Any

from cachepaths import normalize_venue_name
from offline.backtests.datasets import dataset_metadata
from offline.backtests.walk_forward import summarize_test_windows, walk_forward_splits
from state_io import atomic_write_json
from strategies.replay_execution import ExecutionModel
from strategies.replay_metrics import validate_replay_metrics
from strategies.spot_replay import align_spot_interval_params, make_spot_replay_adapter
from botcore import (
    exact_int,
    exact_pos_int,
    float_env,
    load_layered_env_config,
    load_local_module,
    normalize_epoch_sec,
    require_pos_float,
    safe_float,
    safe_int,
    split_csv,
)
from providers.order_ohlc_utils import normalize_record


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
    raw_first_ts = records[0].get("timestamp") if isinstance(records[0], Mapping) else None
    raw_last_ts = records[-1].get("timestamp") if isinstance(records[-1], Mapping) else None
    first_ts_f = normalize_epoch_sec(raw_first_ts, default=None, allow_zero=True)
    last_ts_f = normalize_epoch_sec(raw_last_ts, default=None, allow_zero=True)
    if first_ts_f is None or last_ts_f is None:
        raise ValueError("invalid record timestamps")
    history_days = max(0.0, (last_ts_f - first_ts_f) / 86_400.0)
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
    """Approximate three folds: 45% train, 10% validation, 15% test."""
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 20:
        raise ValueError("at least 20 bars are required for automatic windows")
    train = max(1, safe_int(sample_count * 0.45, 1))
    validation = max(1, safe_int(sample_count * 0.10, 1))
    test = max(1, safe_int(sample_count * 0.15, 1))
    return train, validation, test, test


def iso_utc(timestamp: int) -> str:
    return dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc).isoformat()


def _normalize_segment_row(row: Mapping) -> dict[str, Any]:
    norm = normalize_record(row)
    for field in ("open", "high", "low", "close"):
        require_pos_float(norm[field], name=field)
    return norm


def to_ohlc(records: Sequence[Mapping]) -> list[tuple[float, float, float, float]]:
    norm_rows = [_normalize_segment_row(row) for row in records]
    return [
        (row["open"], row["high"], row["low"], row["close"])
        for row in norm_rows
    ]


def evaluate_segment(
    records: Sequence[Mapping],
    replay: ReplayFunction,
    *,
    warmup_records: Sequence[Mapping] = (),
    validate_contract: bool = False,
) -> dict:
    if not records:
        raise ValueError("evaluation segment cannot be empty")
    norm_records = [_normalize_segment_row(row) for row in records]
    norm_warmup = [_normalize_segment_row(row) for row in warmup_records]
    metrics = replay(
        [(row["open"], row["high"], row["low"], row["close"]) for row in norm_records],
        [(row["open"], row["high"], row["low"], row["close"]) for row in norm_warmup],
        ReplayContext(
            timestamps=tuple(row["timestamp"] for row in norm_records),
            warmup_timestamps=tuple(row["timestamp"] for row in norm_warmup),
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
    first_close = norm_records[0]["close"]
    last_close = norm_records[-1]["close"]
    buy_hold_pct = (last_close / first_close - 1.0) * 100.0 if first_close > 0 else None
    return {
        "start_utc": iso_utc(norm_records[0]["timestamp"]),
        "end_utc": iso_utc(norm_records[-1]["timestamp"]),
        "bars": len(norm_records),
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
    has_overlap = any(
        folds[i].test.stop > folds[i + 1].test.start
        for i in range(len(folds) - 1)
    )
    compounded = 100.0
    for window in test_windows:
        compounded *= max(0.0, 1.0 + safe_float(window.get("return_pct"), 0.0) / 100.0)
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
            "compounded_reset_return_pct": None if has_overlap else (compounded - 100.0),
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
    parts = split_csv(value)
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
    """Extract an exact integer parameter from configuration mapping."""
    if not isinstance(cfg, (dict, Mapping)):
        return default
    raw = cfg.get(key)
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return default
    val = exact_int(raw, default=None)
    if val is None:
        raise ValueError(f"Invalid integer config value for {key}: {raw!r}")
    return val


VENUE_REPLAY_DIRS: dict[str, str] = {
    "hyperliquid": "hyperliquid",
    "t212": "212trading",
}


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
    stem = filename[:-3] if filename.endswith(".py") else filename
    return load_local_module(
        str(venue_dir),
        stem,
        module_key=module_name or f"{subdir}_{stem}",
        reload=reload,
    )


def load_venue_replay_module(
    venue: str,
    module_name: str | None = None,
    *,
    reload: bool = False,
):
    """Load the registered ``replay.py`` adapter module for ``venue`` via ``VENUE_REPLAY_DIRS``."""
    v_norm = normalize_venue_name(venue) or str(venue or "").strip().lower()
    subdir = VENUE_REPLAY_DIRS.get(v_norm)
    if not subdir:
        raise ValueError(f"Unsupported replay venue: {venue!r}")
    return load_venue_module(
        subdir,
        "replay.py",
        module_name or f"{v_norm}_wf_replay",
        reload=reload,
    )


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
    include_dataset_list: bool = False,
    default_output_dir: Path | str | None = None,
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
    if include_dataset_list:
        parser.add_argument(
            "--dataset",
            action="append",
            default=[],
            metavar="INTERVAL=PATH",
            help="frozen dataset CSV path for an interval (e.g. 60=path/to/candles.csv)",
        )
    if default_output_dir is not None:
        parser.add_argument(
            "--output-dir",
            type=Path,
            default=Path(default_output_dir),
        )


def parse_optional_lookback_days(
    parser: argparse.ArgumentParser, raw_val: Any
) -> int | None:
    """Validate optional ``--lookback-days`` CLI value as a positive integer."""
    if raw_val is None:
        return None
    lookback_days = exact_pos_int(raw_val, default=None)
    if lookback_days is None:
        parser.error("--lookback-days must be positive")
    return lookback_days


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


def evaluate_spot_segment(
    records: Sequence[dict],
    params: Any,
    fee_pct: float,
    interval: int,
    warmup_records: Sequence[dict] = (),
    execution: ExecutionModel | None = None,
    *,
    validate_contract: bool = True,
) -> dict:
    """Evaluate an OHLC segment using aligned spot parameters and the canonical spot replay adapter."""
    adapter = make_spot_replay_adapter(
        align_spot_interval_params(params, interval),
        fee_pct=fee_pct,
        bar_minutes=interval,
        execution=execution,
    )
    return evaluate_segment(
        records,
        adapter,
        warmup_records=warmup_records,
        validate_contract=validate_contract,
    )


def build_and_save_interval_baseline_report(
    records: Sequence[dict],
    adapter: ReplayFunction,
    *,
    args: argparse.Namespace,
    interval: int,
    symbol_or_coin: str,
    purpose: str,
    runtime: dict[str, Any],
    strategy_params: dict[str, Any],
    fee_pct: float,
    execution: ExecutionModel,
    source_label: str,
    frozen_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Run walk-forward evaluation for a single interval, assess evidence, and write the baseline JSON report."""
    train_size, val_size, test_size, step_size = resolve_window_sizes(args, len(records))
    warmup_bars = max(0, safe_int(getattr(args, "warmup", None), 0))
    wf_result = evaluate_walk_forward(
        records,
        adapter,
        train_size=train_size,
        validation_size=val_size,
        test_size=test_size,
        step_size=step_size,
        warmup_bars=warmup_bars,
        validate_contract=True,
    )
    evidence = assess_decision_evidence(records, wf_result["aggregate_test"])
    generated_at = dt.datetime.now(tz=dt.timezone.utc)
    meta = dataset_metadata(records, interval_minutes=interval)
    report = {
        "schema_version": 1,
        "generated_at_utc": generated_at.isoformat(),
        "purpose": purpose,
        "runtime": runtime,
        "strategy_params": strategy_params,
        "fee_pct_per_leg": fee_pct,
        "execution_model": dataclasses.asdict(execution),
        "dataset": {
            "source": source_label,
            "frozen_file": str(frozen_path),
            **meta,
        },
        "windows": {
            "train": train_size,
            "validation": val_size,
            "test": test_size,
            "step": step_size,
            "signal_warmup_bars": warmup_bars,
        },
        "evidence": evidence,
        "aggregate_test": wf_result["aggregate_test"],
        "folds": wf_result["folds"],
    }
    report_path = output_dir / f"{symbol_or_coin}_{interval}m_baseline.json"
    atomic_write_json(report_path, report, indent=2, ensure_ascii=True)
    print(
        f"[{symbol_or_coin} {interval}m] Verdict: {evidence['verdict']} | "
        f"Worst Return: {(wf_result['aggregate_test'].get('worst_return_pct') or 0.0):+.2f}% | "
        f"Folds: {wf_result['aggregate_test']['fold_count']} -> {report_path.name}"
    )
    return wf_result["aggregate_test"]

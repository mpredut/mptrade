"""Count real refusal signals without counting their log mirrors as blind trading."""
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "verify_tools"))
import watchdogfor_anomaly as watchdog


def write_log(tmp_path, name, lines):
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def test_three_streams_of_59_refusals_count_as_59_not_177(tmp_path):
    outcomes = [f"{1788830400 + i}|TAOUSDC|SELL|262.0|1.5827|refused|"
                "weight_policy_unavailable|market_api.py|" for i in range(59)]
    worker_lines = [f"04:20:{i} [TAOUSDC] SELL refused weight_policy_unavailable"
                    for i in range(59)]
    paths = [write_log(tmp_path, "order_outcomes_2026-09-08.log", outcomes),
             write_log(tmp_path, "order_retry_worker.log", worker_lines),
             write_log(tmp_path, "order_retry_worker_2026-09-08.log", worker_lines)]
    offsets = dict.fromkeys(paths, 0)
    counts, _, files, _ = watchdog._count_events(paths, offsets)
    assert counts["execution_blocked"] == 59
    assert counts["blind"] == 0
    assert files["execution_blocked"] == {"order_outcomes_2026-09-08.log"}
    assert sum(watchdog._count_events(paths, offsets)[0].values()) == 0


def test_worker_fallback_preserves_repeated_events_not_mirror_copies(tmp_path):
    lines = ["04:20:00 [TAOUSDC] refused weight_policy_unavailable"] * 3
    paths = [write_log(tmp_path, "order_retry_worker.log", lines),
             write_log(tmp_path, "order_retry_worker_2026-09-08.log", lines)]
    counts, _, _, _ = watchdog._count_events(paths, dict.fromkeys(paths, 0))
    assert counts["execution_blocked"] == 3
    assert counts["blind"] == 0


def test_new_or_unreadable_outcome_stream_keeps_worker_fallback(tmp_path):
    lines = ["04:20:00 refused weight_policy_unavailable"]
    worker = write_log(tmp_path, "order_retry_worker.log", lines)
    outcomes = write_log(tmp_path, "order_outcomes_2026-09-08.log", lines)
    paths = [worker, outcomes]
    assert watchdog._count_events(paths, {worker: 0})[0]["execution_blocked"] == 1
    with patch.object(watchdog, "_new_lines", side_effect=lambda p, _: lines if p == worker else None):
        assert watchdog._count_events(paths, dict.fromkeys(paths, 0))[0]["execution_blocked"] == 1


def test_old_daily_outcomes_do_not_hide_new_daily_fallback(tmp_path):
    lines = ["00:00:01 refused weight_policy_unavailable"] * 3
    old = write_log(tmp_path, "order_outcomes_2026-09-07.log", lines)
    new = write_log(tmp_path, "order_outcomes_2026-09-08.log", lines)
    worker = write_log(tmp_path, "order_retry_worker.log", lines)
    paths = [old, new, worker]
    counts, _, files, _ = watchdog._count_events(paths, {old: 0, worker: 0})
    assert counts["execution_blocked"] == 3
    assert files["execution_blocked"] == {"order_retry_worker.log"}


def test_missing_outcome_file_does_not_abort_scan(tmp_path):
    missing = str(tmp_path / "order_outcomes_2026-09-08.log")
    worker = write_log(tmp_path, "order_retry_worker.log", ["weight_policy_unavailable"])
    counts, _, _, _ = watchdog._count_events([missing, worker], {missing: 0, worker: 0})
    assert counts["execution_blocked"] == 1


def test_other_blind_failures_and_distinct_bots_remain_visible(tmp_path):
    lines = ["04:20:00 account cache unavailable", "Traceback (most recent call last):"] * 2
    paths = [write_log(tmp_path, "tradeall.log", lines),
             write_log(tmp_path, "tradeall_2026-09-08.log", lines),
             write_log(tmp_path, "rtrade_2026-09-08.log", lines)]
    counts, _, _, _ = watchdog._count_events(paths, dict.fromkeys(paths, 0))
    assert counts["blind"] == counts["traceback"] == 4
    assert counts["execution_blocked"] == 0


def test_first_scan_baselines_and_truncation_remains_supported(tmp_path):
    path = write_log(tmp_path, "rtrade.log", ["account cache unavailable"] * 10)
    offsets = {}
    assert sum(watchdog._count_events([path], offsets)[0].values()) == 0
    Path(path).write_text("account cache unavailable\n")
    assert watchdog._count_events([path], offsets)[0]["blind"] == 1


def test_watchdog_does_not_recount_its_own_quoted_examples():
    for path in ("logs/anomaly_watchdog.log", "logs/watchdogfor_anomaly.log",
                 "logger/watchdogfor_anomaly_2026-09-08.log"):
        assert watchdog._is_dev_log(path)


def test_alert_reports_execution_blocked_and_retains_cooldown(tmp_path):
    path = write_log(tmp_path, "order_outcomes_2026-09-08.log",
                     ["refused|weight_policy_unavailable"] * 30)
    state = {"offsets": {path: 0}, "cooldowns": {}}
    with patch.object(watchdog, "_active_logs", return_value=[path]), \
            patch.object(watchdog.wc, "load_state", return_value=state), \
            patch.object(watchdog.wc, "save_state"), \
            patch.object(watchdog.wc, "alert") as alert:
        assert watchdog.check_once(now=10000)
        assert "execution blocked (pre-submit" in alert.call_args.args[1]
        assert "blind:" not in alert.call_args.args[1]
        state["offsets"][path] = 0
        assert not watchdog.check_once(now=10001)
        assert alert.call_count == 1

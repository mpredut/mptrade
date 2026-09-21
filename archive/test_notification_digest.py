"""Notification-only aggregation: restart, quiet periods and independent incidents."""
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from notification_digest import IncidentDigest
from state_io import StateReadError


def event(symbol="TAOUSDC", qty=1.0, sample="intent-1"):
    return {"labels": {"symbol": symbol, "reason": "weight_policy_unavailable"},
            "quantity": qty, "sample_id": sample}


def test_immediate_batch_then_persistent_summary_even_without_new_events(tmp_path):
    path = tmp_path / "digest.json"
    digest = IncidentDigest(path, 900)
    first = digest.collect([event(), event(qty=2.0)], now=1000)
    assert len(first) == 1
    assert first[0]["count"] == 2 and first[0]["quantity"] == 3
    assert not first[0]["reported"]
    assert digest.collect([event(sample="second")], now=1060) == []
    restarted = IncidentDigest(path, 900)
    assert restarted.collect([event(qty=3.0)], now=1800) == []
    before = path.read_bytes()
    assert restarted.collect([], now=1899) == []
    assert path.read_bytes() == before  # Idle polls do not rewrite disk state.
    summary = restarted.collect([], now=1900)
    assert len(summary) == 1
    assert summary[0]["count"] == 2 and summary[0]["quantity"] == 4
    assert summary[0]["reported"] and summary[0]["sample_id"] == "second"
    assert IncidentDigest(path, 900).collect([], now=1901) == []


def test_independent_incident_is_immediate_and_quiet_groups_are_pruned(tmp_path):
    path = tmp_path / "digest.json"
    digest = IncidentDigest(path, 900)
    digest.collect([event()], now=1000)
    assert len(digest.collect([event("BTCUSDC")], now=1001)) == 1
    assert digest.collect([], now=1901) == []
    assert json.loads(path.read_text())["groups"] == {}
    assert not digest.collect([event()], now=2000)[0]["reported"]


def test_missing_empty_state_does_not_create_files(tmp_path):
    assert IncidentDigest(tmp_path / "absent" / "digest.json", 900).collect([], now=1000) == []
    assert list(tmp_path.iterdir()) == []


def test_corruption_does_not_reset_cooldown_or_overwrite_evidence(tmp_path):
    path = tmp_path / "digest.json"
    path.write_text("broken json")
    with pytest.raises(StateReadError):
        IncidentDigest(path, 900).collect([event()], now=1000)
    assert path.read_text() == "broken json"


@pytest.mark.parametrize("interval", [0, -1, float("nan"), float("inf")])
def test_interval_must_be_explicit_and_valid(tmp_path, interval):
    with pytest.raises(ValueError):
        IncidentDigest(tmp_path / "digest.json", interval)


def test_concurrent_instances_reserve_only_one_first_alert(tmp_path):
    path = tmp_path / "digest.json"
    def collect(_):
        return IncidentDigest(path, 900).collect([event()], now=1000)
    with ThreadPoolExecutor(max_workers=4) as pool:
        reports = list(pool.map(collect, range(20)))
    assert sum(len(r) for r in reports) == 1
    assert IncidentDigest(path, 900).collect([], now=1900)[0]["count"] == 19


def test_500_expiries_never_exceed_one_report_per_interval(tmp_path):
    path = tmp_path / "digest.json"
    reports = []
    # Recreate the helper on each pass, as after a restart; no process-local gate.
    for i in range(500):
        reports.extend(IncidentDigest(path, 900).collect(
            [event(sample=f"intent-{i}")], now=1000 + 30 * i))
    reports.extend(IncidentDigest(path, 900).collect([], now=1000 + 30 * 500 + 900))
    assert len(reports) == 18
    assert sum(report["count"] for report in reports) == 500

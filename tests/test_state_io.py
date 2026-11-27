import csv
import datetime as dt
import gzip
import json
import math
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

import cacheprice
import cachetrend
import order_guard
import state_io
from intelligence.sentiment.guards.high_stake_guard import collect_pretrade_telemetry
from providers.kraken_provider import KrakenProvider
from state_io import (
    StateReadError,
    atomic_file_path,
    atomic_gzip_file,
    atomic_snapshot_file,
    atomic_text_writer,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    commit_file_and_meta,
    discard_generation_file,
    load_json_state,
    meta_file_path,
    previous_generation_path,
    read_bytes_file,
    read_csv_dicts,
    read_json_dict,
    read_json_dict_cached,
    read_json_dict_recovering,
    read_json_file,
    read_jsonl_dicts,
    read_jsonl_records,
    read_recovering_previous,
    read_tail_jsonl_dicts,
    read_tail_jsonl_records,
    read_tail_lines,
    read_text_file,
    read_text_lines,
    save_json_state,
    stage_previous_generation,
    write_meta_json,
)
from strategies.state_store import JsonStateStore, StatePersistenceError


def test_atomic_write_json_replaces_complete_document(tmp_path):
    path = tmp_path / "nested" / "state.json"
    atomic_write_json(path, {"cycle": 1}, indent=2)
    atomic_write_json(path, {"cycle": 2}, indent=2)
    assert json.loads(path.read_text(encoding="utf-8")) == {"cycle": 2}
    assert list(path.parent.glob("*.tmp")) == []


def test_atomic_writer_fsyncs_directory_after_replace(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    events = []
    real_replace = os.replace

    def observed_replace(source, destination):
        real_replace(source, destination)
        if destination == str(path):
            events.append(("replace", destination))

    def observed_directory_fsync(destination):
        if destination == str(path):
            assert path.read_text(encoding="utf-8") == '{"safe": true}'
            events.append(("directory_fsync", destination))

    monkeypatch.setattr("state_io.os.replace", observed_replace)
    monkeypatch.setattr(
        "state_io._fsync_parent_directory", observed_directory_fsync)

    with atomic_text_writer(path) as handle:
        handle.write('{"safe": true}')

    assert [event[0] for event in events] == [
        "replace", "directory_fsync",
    ]


def test_atomic_writer_preserves_previous_file_on_failure(tmp_path):
    path = tmp_path / "state.json"
    path.write_text('{"safe": true}', encoding="utf-8")
    with pytest.raises(RuntimeError):
        with atomic_text_writer(path) as handle:
            handle.write('{"safe": false}')
            raise RuntimeError("interrupted")
    assert path.read_text(encoding="utf-8") == '{"safe": true}'
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_snapshot_preserves_replaced_source_generation(tmp_path):
    source = tmp_path / "current.jsonl"
    snapshot = tmp_path / "current.jsonl.previous"
    source.write_text("old-generation\n", encoding="utf-8")

    atomic_snapshot_file(source, snapshot)
    with atomic_text_writer(source) as handle:
        handle.write("new-generation\n")

    assert source.read_text(encoding="utf-8") == "new-generation\n"
    assert snapshot.read_text(encoding="utf-8") == "old-generation\n"


def test_atomic_snapshot_falls_back_to_copy(tmp_path, monkeypatch):
    source = tmp_path / "current.jsonl"
    snapshot = tmp_path / "current.jsonl.previous"
    source.write_text("certified\n", encoding="utf-8")

    def unavailable_hard_link(*_args, **_kwargs):
        raise OSError("hard links unavailable")

    monkeypatch.setattr("state_io.os.link", unavailable_hard_link)
    atomic_snapshot_file(source, snapshot)

    assert snapshot.read_text(encoding="utf-8") == "certified\n"
    assert [path for path in tmp_path.iterdir() if path.is_dir()] == []


def test_missing_state_is_a_clean_first_start(tmp_path):
    assert load_json_state(
        tmp_path / "missing.json", default_factory=lambda: {"cycle": 0},
        fail_closed=True, label="strategy",
    ) == {"cycle": 0}


def test_invalid_live_state_fails_closed(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(StateReadError, match="invalid strategy state"):
        load_json_state(
            path, default_factory=dict, fail_closed=True, label="strategy",
        )


def test_invalid_observational_state_can_reset(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("[]", encoding="utf-8")
    assert load_json_state(
        path, default_factory=lambda: {"paper": True},
        fail_closed=False, label="paper", root_type=dict,
    ) == {"paper": True}


def test_json_file_readers_preserve_constant_policy_and_patched_loader(tmp_path, monkeypatch):
    path = tmp_path / "nonstandard.json"
    path.write_text('{"value": NaN}', encoding="utf-8")
    assert math.isnan(read_json_file(path)["value"])
    assert math.isnan(load_json_state(path, default_factory=dict, fail_closed=False, label="paper")["value"])
    with pytest.raises(StateReadError):
        load_json_state(path, default_factory=dict, fail_closed=True, label="live")

    original_load = json.load
    observed = []

    def patched_load(handle, **kwargs):
        if handle.name == str(path):
            observed.append(kwargs)
        return original_load(handle, **kwargs)

    monkeypatch.setattr(state_io.json, "load", patched_load)
    assert math.isnan(read_json_file(path)["value"])
    assert math.isnan(load_json_state(path, default_factory=dict, fail_closed=False, label="paper")["value"])
    with pytest.raises(StateReadError):
        load_json_state(path, default_factory=dict, fail_closed=True, label="live")
    assert observed == [
        {},
        {"parse_constant": None},
        {"parse_constant": state_io._reject_json_constant},
    ]


def test_json_file_readers_reject_invalid_utf8(tmp_path):
    path = tmp_path / "invalid_utf8.json"
    path.write_bytes(b'{"value": "\xff"}')
    with pytest.raises(UnicodeDecodeError):
        read_json_file(path)
    with pytest.raises(StateReadError):
        load_json_state(path, default_factory=dict, fail_closed=True, label="live")


def test_trailing_once_commands_share_daemon_lock(monkeypatch):
    import sys
    import trailing_core

    root = Path(__file__).resolve().parents[1]
    core_text = (root / "trailing_core.py").read_text(encoding="utf-8")
    assert (
        "if not args.status:" in core_text
        or "if not _parse_bool_flag(args.status" in core_text
    )
    assert "single_instance(lock_name)" in core_text
    acquired_locks = []
    monkeypatch.setattr(trailing_core, "single_instance", lambda name: acquired_locks.append(name))
    monkeypatch.setattr(sys, "argv", ["trailing_stop.py", "--once"])
    args_once = trailing_core.parse_trailing_cli_args("desc", "test_trailing_lock")
    assert args_once.once is True and args_once.status is False
    assert acquired_locks == ["test_trailing_lock"]
    monkeypatch.setattr(sys, "argv", ["trailing_stop.py", "--status"])
    args_status = trailing_core.parse_trailing_cli_args("desc", "test_trailing_lock")
    assert args_status.status is True
    assert acquired_locks == ["test_trailing_lock"]
    for relative, lock_name in (
        ("binance_api/trailing_stop.py", "binance_trailing"),
        ("kraken/trailing_stop.py", "kraken_trailing"),
        ("hyperliquid/trailing_stop.py", "hyperliquid_trailing"),
    ):
        text = (root / relative).read_text(encoding="utf-8")
        assert "parse_trailing_cli_args(" in text
        assert f'"{lock_name}"' in text


def test_xstock_once_shares_lock_but_isolated_trial_does_not():
    root = Path(__file__).resolve().parents[1]
    text = (root / "kraken/kraken_xstock_watch.py").read_text(encoding="utf-8")
    assert "if not args.status and not args.trial:" in text
    assert 'single_instance("kraken_xstock_watch")' in text


def test_strategy_test_commands_use_instrument_daemon_locks():
    root = Path(__file__).resolve().parents[1]
    kraken = (root / "kraken/kraken_bot.py").read_text(encoding="utf-8")
    hyperliquid = (root / "hyperliquid/hl_bot.py").read_text(encoding="utf-8")
    assert "args.test_strategy = _norm_pair(args.test_strategy)" in kraken
    assert 'single_instance(f"kraken_bot_{args.test_strategy}")' in kraken
    assert 'single_instance(f"hl_bot_{token}")' in hyperliquid


def test_atomic_binary_and_gzip_helpers(tmp_path):
    bin_path = tmp_path / "payload.bin"
    atomic_write_bytes(bin_path, b"\x00\x01\x02")
    assert bin_path.read_bytes() == b"\x00\x01\x02"

    staged_path = tmp_path / "rendered.bin"
    with atomic_file_path(staged_path, suffix=".part") as tmp_file:
        Path(tmp_file).write_bytes(b"rendered-ok")
    assert staged_path.read_bytes() == b"rendered-ok"

    archive_src = tmp_path / "events.jsonl"
    archive_src.write_text("line1\nline2\n", encoding="utf-8")
    gz_path = Path(atomic_gzip_file(archive_src, remove_source=True))
    assert not archive_src.exists()
    assert gz_path.exists()
    with gzip.open(gz_path, "rt", encoding="utf-8") as handle:
        assert handle.read() == "line1\nline2\n"


def test_commit_file_and_meta_rolls_back_and_recovers_previous(tmp_path):
    data_path = str(tmp_path / "cache.json")

    # Initial commit succeeds and leaves no .previous file behind.
    assert commit_file_and_meta(
        data_path,
        lambda: atomic_write_json(data_path, {"version": 1}),
        lambda: (write_meta_json(data_path, {"version": 1}) or True),
    )
    assert not Path(previous_generation_path(data_path)).exists()
    assert json.loads(Path(data_path).read_text(encoding="utf-8")) == {"version": 1}

    # Failed meta commit rolls data_path back to version 1.
    assert not commit_file_and_meta(
        data_path,
        lambda: atomic_write_json(data_path, {"version": 2}),
        lambda: False,
    )
    assert not Path(previous_generation_path(data_path)).exists()
    assert json.loads(Path(data_path).read_text(encoding="utf-8")) == {"version": 1}

    # Simulated crash between data write and meta write is recovered by read_recovering_previous.
    stage_previous_generation(data_path)
    atomic_write_json(data_path, {"version": 2})

    def _strict_reader(candidate_path: str):
        meta = json.loads(Path(meta_file_path(data_path)).read_text(encoding="utf-8"))
        payload = json.loads(Path(candidate_path).read_text(encoding="utf-8"))
        if payload.get("version") != meta.get("version"):
            raise ValueError("version mismatch")
        return payload

    recovered = read_recovering_previous(data_path, _strict_reader)
    assert recovered == {"version": 1}
    assert json.loads(Path(data_path).read_text(encoding="utf-8")) == {"version": 1}
    assert not Path(previous_generation_path(data_path)).exists()


def test_load_json_state_recovers_from_previous_generation(tmp_path):
    path = tmp_path / "bot_state.json"
    backup = Path(previous_generation_path(str(path)))
    path.write_text("{corrupted", encoding="utf-8")
    backup.write_text('{"cycle": 42, "status": "ok"}', encoding="utf-8")

    loaded = load_json_state(
        path, default_factory=dict, fail_closed=True, label="bot_state"
    )
    assert loaded == {"cycle": 42, "status": "ok"}
    assert json.loads(path.read_text(encoding="utf-8")) == {"cycle": 42, "status": "ok"}
    assert not backup.exists()


def test_save_json_state_restores_previous_on_write_failure(tmp_path, monkeypatch):
    path = tmp_path / "bot_state_save.json"
    save_json_state(path, {"cycle": 10}, indent=2)
    assert json.loads(path.read_text(encoding="utf-8")) == {"cycle": 10}
    assert not Path(previous_generation_path(str(path))).exists()

    def broken_atomic_write(*_args, **_kwargs):
        path.unlink()
        path.write_text("{partial", encoding="utf-8")
        raise OSError("simulated disk error")

    monkeypatch.setattr("state_io.atomic_write_json", broken_atomic_write)
    with pytest.raises(OSError, match="simulated disk error"):
        save_json_state(path, {"cycle": 11}, indent=2)

    assert json.loads(path.read_text(encoding="utf-8")) == {"cycle": 10}


def test_audit46_transactional_recovery_and_cross_venue_fallbacks(tmp_path, monkeypatch):
    now_s = time.time()
    now_ms = int(now_s * 1000)

    # 2. KrakenProvider._fills_from_cache recovers from cache_trade_kraken.json.previous
    kr_cache = tmp_path / "cache_trade_kraken.json"
    Path(previous_generation_path(str(kr_cache))).write_text(
        json.dumps({
            "items": {
                "TAOUSD": [
                    {
                        "symbol": "TAOUSD",
                        "id": "T1",
                        "orderId": "O1",
                        "price": 250.0,
                        "qty": 1.5,
                        "quantity": 1.5,
                        "time": now_ms - 5000,
                        "timestamp": now_ms - 5000,
                        "isBuyer": True,
                        "side": "BUY",
                    }
                ]
            },
            "fetchtime": {"TAOUSD": now_ms},
        }),
        encoding="utf-8",
    )
    kr_cache.write_text("{corrupt", encoding="utf-8")
    monkeypatch.setattr("providers.kraken_provider._KRAKEN_CACHE_FILE", str(kr_cache))
    kp = KrakenProvider()
    fills = kp._fills_from_cache("TAOUSD")
    assert fills is not None and len(fills) == 1
    assert fills[0]["price"] == pytest.approx(250.0)

    # 3. order_guard._read_cached_price_history skips empty cache_24_price_TAOUSD.json
    # and recovers from cache_24_price_TAOUSDC.json.previous
    monkeypatch.setenv("MPTRADE_CACHEDB_DIR", str(tmp_path))
    (tmp_path / "cache_24_price_TAOUSD.json").write_text(
        json.dumps({"items": {}}), encoding="utf-8"
    )
    p24_usdc = tmp_path / "cache_24_price_TAOUSDC.json"
    Path(previous_generation_path(str(p24_usdc))).write_text(
        json.dumps({
            "items": {
                "TAOUSDC": [
                    [now_ms - 1800_000, 240.0],
                    [now_ms - 900_000, 245.0],
                    [now_ms, 250.0],
                ]
            }
        }),
        encoding="utf-8",
    )
    hist = order_guard._read_cached_price_history("TAOUSD", window_seconds=3600.0)
    assert hist is not None and len(hist) == 3

    # 4. collect_pretrade_telemetry recovers cache_short_trend.json.previous (with "items" envelope)
    # and reads recent fills from cache_trade_kraken.json
    st_path = tmp_path / "cache_short_trend.json"
    Path(previous_generation_path(str(st_path))).write_text(
        json.dumps({
            "items": {
                "TAOUSDC": {
                    "dynamic": {
                        "222": {"slope_full": 0.0011},
                        "900": {"slope_full": 0.0022},
                        "9000": {"slope_full": 0.0033},
                    }
                }
            }
        }),
        encoding="utf-8",
    )
    st_path.write_text("{corrupt", encoding="utf-8")
    telem = collect_pretrade_telemetry("TAO/USD", cachedb_dir=str(tmp_path))
    assert "24h Multi-Horizon Price Evolution" in telem
    assert "Dynamic Multi-Window Slopes" in telem
    assert "Recent Venue Fills" in telem


def test_atomic_write_json_allow_nan(tmp_path):
    p = tmp_path / "test_nan.json"
    with pytest.raises(ValueError):
        atomic_write_json(p, {"nan_val": float("nan")}, allow_nan=False)
    atomic_write_json(p, {"nan_val": float("nan")}, allow_nan=True)
    assert p.exists()


def test_atomic_text_writer_newline(tmp_path):
    p = tmp_path / "test.csv"
    with atomic_text_writer(p, newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["a", "b"])
        writer.writerow(["1", "2"])
    content = p.read_bytes()
    assert b"\r\r\n" not in content
    assert b"a,b" in content


def test_atomic_write_json_default(tmp_path):
    p = tmp_path / "test_default.json"
    data = {"path": Path("/some/path"), "time": dt.datetime(2026, 1, 1, 12, 0, 0)}
    atomic_write_json(p, data, default=str)
    assert p.exists()
    loaded = json.loads(p.read_text(encoding="utf-8"))
    assert "/some/path" in loaded["path"]
    assert "2026-01-01" in loaded["time"]


def test_read_json_dict(tmp_path):
    p_ok = tmp_path / "ok.json"
    atomic_write_json(p_ok, {"k": "v"})
    assert read_json_dict(str(p_ok)) == {"k": "v"}

    p_list = tmp_path / "list.json"
    atomic_write_json(p_list, [1, 2, 3])
    with pytest.raises(ValueError, match="JSON root must be a dict"):
        read_json_dict(str(p_list))
    assert read_json_dict(str(p_list), default={}) == {}
    assert read_json_file(str(p_list)) == [1, 2, 3]
    assert read_json_file(str(tmp_path / "missing.json"), default=None) is None

    # .previous recovery with read_json_dict(recover_previous=True)
    p_rec = tmp_path / "rec.json"
    p_rec_prev = tmp_path / "rec.json.previous"
    p_rec.write_text("[1, 2]", encoding="utf-8")
    p_rec_prev.write_text('{"healed": 1}', encoding="utf-8")
    assert read_json_dict(str(p_rec), recover_previous=True) == {"healed": 1}

    # JSONL helpers
    p_jsonl = tmp_path / "items.jsonl"
    p_jsonl.write_text('{"a": 1}\n\ncorrupt\n[1, 2]\n{"b": 2}\n', encoding="utf-8")
    assert read_jsonl_records(str(p_jsonl)) == [{"a": 1}, [1, 2], {"b": 2}]
    assert read_jsonl_dicts(str(p_jsonl)) == [{"a": 1}, {"b": 2}]
    assert read_jsonl_dicts(str(tmp_path / "missing.jsonl")) == []
    with pytest.raises(ValueError):
        read_jsonl_dicts(str(p_jsonl), strict=True)

    p_out_jsonl = tmp_path / "written.jsonl"
    atomic_write_jsonl(str(p_out_jsonl), [{"x": 10}, {"y": 20}])
    assert read_jsonl_dicts(str(p_out_jsonl)) == [{"x": 10}, {"y": 20}]
    assert p_out_jsonl.read_text(encoding="utf-8") == '{"x": 10}\n{"y": 20}\n'


def test_read_text_bytes_csv_and_tail_max_bytes(tmp_path):
    p_txt = tmp_path / "sample.txt"
    atomic_write_text(str(p_txt), " first \n\nsecond\n")
    assert read_text_file(str(p_txt)) == " first \n\nsecond\n"
    assert read_text_file(str(tmp_path / "missing.txt"), default="fallback") == "fallback"
    assert read_text_lines(str(p_txt)) == [" first ", "", "second"]
    assert read_text_lines(str(p_txt), strip=True, skip_empty=True) == ["first", "second"]
    assert read_text_lines(str(p_txt), keepends=True) == [" first \n", "\n", "second\n"]
    assert read_text_lines(str(tmp_path / "missing.txt")) == []

    p_bin = tmp_path / "sample.bin"
    atomic_write_bytes(str(p_bin), b"\x00\x01\x02")
    assert read_bytes_file(str(p_bin)) == b"\x00\x01\x02"
    assert read_bytes_file(str(tmp_path / "missing.bin"), default=None) is None

    p_csv = tmp_path / "ohlc.csv"
    atomic_write_text(str(p_csv), "timestamp,open,close\n100,1.0,2.0\n")
    assert read_csv_dicts(str(p_csv)) == [{"timestamp": "100", "open": "1.0", "close": "2.0"}]
    assert read_csv_dicts(str(tmp_path / "missing.csv")) == []
    assert read_csv_dicts(str(tmp_path / "missing.csv"), default=[{"fallback": True}]) == [{"fallback": True}]
    with pytest.raises(FileNotFoundError):
        read_csv_dicts(str(tmp_path / "missing.csv"), strict=True)

    p_tail = tmp_path / "tail.jsonl"
    lines = [json.dumps({"idx": i, "pad": "x" * 20}) for i in range(10)]
    atomic_write_text(str(p_tail), "\n".join(lines) + "\n")
    tail_dicts = read_tail_jsonl_dicts(str(p_tail), limit=None, max_bytes=120)
    assert len(tail_dicts) >= 2
    assert tail_dicts[-1]["idx"] == 9
    tail_recs = read_tail_jsonl_records(str(p_tail), limit=3, max_bytes=300)
    assert [r["idx"] for r in tail_recs] == [7, 8, 9]

    assert read_recovering_previous(None, default={"ok": 1}) == {"ok": 1}
    assert read_json_dict(None, default={"ok": 2}, recover_previous=True) == {"ok": 2}
    assert read_json_dict(123, default={"ok": 3}, recover_previous=True) == {"ok": 3}
    assert read_jsonl_records(object(), strict=False) == []
    assert read_text_lines(object()) == []
    discard_generation_file(1.5, include_previous=True, include_meta=True)
    discard_generation_file(object(), include_previous=True, include_meta=True)

    p_cached = tmp_path / "cached_rec.json"
    p_cached_prev = tmp_path / "cached_rec.json.previous"
    atomic_write_text(str(p_cached), "{corrupt")
    atomic_write_text(str(p_cached_prev), '{"recovered": 42}')
    memo = {}
    assert read_json_dict_cached(p_cached, memo) == {"recovered": 42}
    assert p_cached_prev.exists()
    assert p_cached.read_text(encoding="utf-8") == "{corrupt"
    assert read_json_dict_cached(p_cached, memo) == {"recovered": 42}


def test_stage_previous_generation_and_concurrent_reader_race(tmp_path, monkeypatch):
    from intelligence.sentiment.collectors import pretrade_telemetry_collector as ptc

    data_path = str(tmp_path / "cache_current_price.json")
    atomic_write_json(data_path, {"items": {"BTCUSDC": 70000.0}})

    # 1. Passive snapshot readers (read_json_dict_recovering, pretrade_telemetry_collector)
    # never unlink .previous when primary is valid (even on a new inode) and never overwrite
    # primary when primary is invalid; only repair_files=True (producer) mutates disk.
    backup_path = stage_previous_generation(data_path)
    assert os.path.exists(backup_path)
    assert os.path.samefile(data_path, backup_path)
    assert read_json_dict_recovering(data_path) == {"items": {"BTCUSDC": 70000.0}}
    assert os.path.exists(backup_path)

    atomic_write_json(data_path, {"items": {"BTCUSDC": 70100.0}})
    assert not os.path.samefile(data_path, backup_path)
    assert read_json_dict_recovering(data_path) == {"items": {"BTCUSDC": 70100.0}}
    assert ptc.read_json_cached(data_path) == {"items": {"BTCUSDC": 70100.0}}
    assert os.path.exists(backup_path)

    atomic_write_text(data_path, "{corrupt_mid_write")
    assert ptc.read_json_cached(data_path) == {"items": {"BTCUSDC": 70000.0}}
    assert Path(data_path).read_text(encoding="utf-8") == "{corrupt_mid_write"
    assert os.path.exists(backup_path)

    # Authoritative producer recovery (repair_files=True) restores primary and cleans .previous.
    assert read_json_dict_recovering(data_path, repair_files=True) == {"items": {"BTCUSDC": 70000.0}}
    assert json.loads(Path(data_path).read_text(encoding="utf-8")) == {"items": {"BTCUSDC": 70000.0}}
    assert not os.path.exists(backup_path)

    # 2. If a concurrent reader unlinks destination immediately after os.replace inside
    # durable_replace_file, stage_previous_generation must not raise FileNotFoundError.
    real_replace = os.replace

    def replace_then_concurrent_unlink(src, dst):
        real_replace(src, dst)
        if dst.endswith(".previous") and os.path.exists(dst):
            os.unlink(dst)

    monkeypatch.setattr("state_io.os.replace", replace_then_concurrent_unlink)
    stage_previous_generation(data_path)

# ============================================================================
# Domain Regressions & Input Hygiene (Targeted Validation)
# ============================================================================


def test_state_io_read_tail_lines_truncate_file_bytes_and_prune_rotated_archives_bool_zero_protection_iter46_s4(tmp_path):
    """[ITER46 #4] state_io: read_tail_lines, truncate_file_bytes, and prune_rotated_archives bool/zero protection."""
    # 4. state_io: read_tail_lines, truncate_file_bytes, and prune_rotated_archives bool/zero protection
    tail_file = tmp_path / "tail_test.txt"
    tail_file.write_text("line1\nline2\nline3\n", encoding="utf-8")
    assert state_io.read_tail_lines(str(tail_file), 2, chunk_size=0) == [b"line2", b"line3"]
    assert state_io.read_tail_lines(str(tail_file), 2, chunk_size=True) == [b"line2", b"line3"]
    with pytest.raises(ValueError):
        state_io.truncate_file_bytes(str(tail_file), True)
    with pytest.raises(ValueError):
        state_io.truncate_file_bytes(str(tail_file), -1)

def test_jsonstatestore_save_rejects_non_dict_state_and_state_io_iter64_s2(tmp_path):
    """[ITER64 #2] JsonStateStore.save rejects non-dict state and state_io.read_tail_lines guards directory/None limit."""
    # 2. JsonStateStore.save rejects non-dict state and state_io.read_tail_lines guards directory/None limit
    store = JsonStateStore(
        str(tmp_path / "strat_state.json"),
        default_factory=lambda: {"cycle": 0, "qty": 0.0},
        label="test_strat",
        logger=lambda _msg: None,
        fail_closed=True,
    )
    with pytest.raises(StatePersistenceError):
        store.save(["not", "a", "dict"])
    assert read_tail_lines(str(tmp_path), 10) == []
    assert read_tail_lines(str(tmp_path / "strat_state.json"), None) == []

def test_state_io_truncate_file_bytes_rejects_fractional_non_finite_sizes_and_iter89_s4(tmp_path):
    """[ITER89 #4] state_io.truncate_file_bytes rejects fractional/non-finite sizes and _positive_chunk_size falls back safely."""
    # 4. state_io.truncate_file_bytes rejects fractional/non-finite sizes and _positive_chunk_size falls back safely
    f_path = tmp_path / "data.bin"
    f_path.write_bytes(b"0123456789")
    with pytest.raises(ValueError, match="non-negative integer"):
        state_io.truncate_file_bytes(str(f_path), 4.5)
    with pytest.raises(ValueError, match="non-negative integer"):
        state_io.truncate_file_bytes(str(f_path), float("inf"))
    assert state_io.exact_pos_int("invalid", default=4096) == 4096
    assert state_io.exact_pos_int(True, default=4096) == 4096

def test_state_io_read_tail_lines__positive_chunk_size_prune_rotated_archives_iter99_s1(tmp_path):
    """[ITER99 #1] state_io.read_tail_lines / _positive_chunk_size / prune_rotated_archives."""
    # 1. state_io.read_tail_lines / _positive_chunk_size / prune_rotated_archives
    # Passing bool or int as path must not treat file descriptors (e.g. stdout=1) as paths!
    assert state_io.read_tail_lines(True, 10) == []
    assert state_io.read_tail_lines(1, 10) == []
    assert state_io.read_tail_lines(None, 10) == []

    sample_file = tmp_path / "sample.jsonl"
    sample_file.write_text("line1\nline2\nline3\n", encoding="utf-8")
    assert state_io.read_tail_lines(str(sample_file), 2) == [b"line2", b"line3"]
    assert state_io.read_tail_lines(str(sample_file), 2.5) == []
    assert state_io.read_tail_lines(str(sample_file), True) == []
    assert state_io.exact_pos_int(4096.5, default=65536) == 65536
    assert state_io.exact_pos_int(True, default=65536) == 65536
    assert state_io.prune_rotated_archives(str(sample_file), keep_count=2.5) == []
    assert state_io.prune_rotated_archives(True, keep_count=2) == []

def test_state_io_load_json_state_with_fail_closed_false_truncate_file_bytes_with_atomic_gzip_file_iter161_s1(tmp_path):
    """[ITER161 #1] state_io: load_json_state with fail_closed='false', truncate_file_bytes with '4.0', atomic_gzip_file with remove_source='false'."""
    # 1. state_io: load_json_state with fail_closed="false", truncate_file_bytes with "4.0", atomic_gzip_file with remove_source="false"
    corrupt_json = tmp_path / "corrupt.json"
    corrupt_json.write_text("not-json", encoding="utf-8")
    assert state_io.load_json_state(str(corrupt_json), default_factory=dict, fail_closed="false", label="test") == {}

    bin_file = tmp_path / "data.bin"
    bin_file.write_bytes(b"0123456789")
    state_io.truncate_file_bytes(str(bin_file), "4.0")
    assert bin_file.read_bytes() == b"0123"

    _ = state_io.atomic_gzip_file(str(bin_file), remove_source="false")
    assert bin_file.exists()
    assert (tmp_path / "data.bin.gz").exists()

def test_read_json_recovering_read_json_dict_recovering_handle_missing_corrupt_previous_and_non_iter184_s2(tmp_path):
    """[ITER184 #2] read_json_recovering & read_json_dict_recovering handle missing, corrupt+.previous, and non-dict files."""
    # 2. read_json_recovering & read_json_dict_recovering handle missing, corrupt+.previous, and non-dict files
    missing_path = str(tmp_path / "missing.json")
    assert state_io.read_json_recovering(missing_path) is None
    assert state_io.read_json_dict_recovering(missing_path) is None

    list_path = tmp_path / "list_state.json"
    list_path.write_text("[1, 2, 3]", encoding="utf-8")
    assert state_io.read_json_recovering(str(list_path)) == [1, 2, 3]
    assert state_io.read_json_dict_recovering(str(list_path)) is None

def test_read_json_dict_cached_memoizes_by_stat_signature_and_recovers_from_iter184_s3(tmp_path):
    """[ITER184 #3] read_json_dict_cached memoizes by stat signature and recovers from .previous."""
    # 3. read_json_dict_cached memoizes by stat signature and recovers from .previous
    memo = {}
    dict_path = tmp_path / "cached_dict.json"
    prev_path = tmp_path / "cached_dict.json.previous"
    dict_path.write_text("{corrupt", encoding="utf-8")
    prev_path.write_text(json.dumps({"BTCUSDC": {"T": 12.0}}), encoding="utf-8")
    loaded = state_io.read_json_dict_cached(str(dict_path), memo, max_entries=8)
    assert loaded == {"BTCUSDC": {"T": 12.0}}
    assert len(memo) == 1

def test_state_io_atomic_write_jsonl_writes_iterable_of_dicts_atomically_and_iter185_s1(tmp_path):
    """[ITER185 #1] state_io.atomic_write_jsonl writes iterable of dicts atomically and matches read_jsonl_dicts."""
    # 1. state_io.atomic_write_jsonl writes iterable of dicts atomically and matches read_jsonl_dicts
    p_jsonl = tmp_path / "orders.jsonl"
    data = [{"order_id": 1, "sym": "BTCUSDC"}, {"order_id": 2, "sym": "ETHUSDC"}]
    state_io.atomic_write_jsonl(str(p_jsonl), data)
    assert state_io.read_jsonl_dicts(str(p_jsonl)) == data


def test_read_json_dict_cached_inode_invalidation_and_memoized_ssot_price_trend_readers(tmp_path):
    """read_json_dict_cached invalidates on atomic inode swap even with identical mtime/size, and memoizes SSOT price/trend lookups."""
    memo = {}
    same_size_file = str(tmp_path / "same_size.json")
    atomic_write_json(same_size_file, {"v": 1})
    st1 = os.stat(same_size_file)
    assert read_json_dict_cached(same_size_file, memo) == {"v": 1}

    # Atomic replace with identical byte size and forced identical mtime_ns -> new inode invalidates memo
    atomic_write_json(same_size_file, {"v": 2})
    os.utime(same_size_file, ns=(st1.st_atime_ns, st1.st_mtime_ns))
    assert read_json_dict_cached(same_size_file, memo) == {"v": 2}

    # resolve_cached_symbol_price & read_ssot_short_trend_candidates stat-memoize unchanged files across multi-symbol calls
    cp_file = tmp_path / "cache_current_price.json"
    atomic_write_json(
        str(cp_file),
        {"items": {"BTCUSDC": [[1_790_000_000_000, 91234.5]], "ETHUSDC": [[1_790_000_000_000, 3456.7]]}},
    )
    trend_file = tmp_path / "cache_short_trend.json"
    now_s = 1_790_000_005.0
    atomic_write_json(
        str(trend_file),
        {"items": {"BTCUSDC": {"ts": now_s - 2.0, "slope": 0.42}}},
    )
    with patch("state_io.json.load", wraps=json.load) as spy_load:
        assert cacheprice.resolve_cached_symbol_price("BTCUSDC", cache_dir=str(tmp_path)) == pytest.approx(91234.5)
        assert spy_load.call_count == 1
        assert cacheprice.resolve_cached_symbol_price("ETHUSDC", cache_dir=str(tmp_path)) == pytest.approx(3456.7)
        assert spy_load.call_count == 1

        s1 = cachetrend.CacheShortTrendManager.read_ssot_short_trend_file(
            str(trend_file), ["BTCUSDC"], 60.0, now_s=now_s
        )
        assert s1 is not None and s1["slope"] == pytest.approx(0.42)
        assert spy_load.call_count == 2
        s2 = cachetrend.CacheShortTrendManager.read_ssot_short_trend_file(
            str(trend_file), ["BTCUSDC"], 60.0, now_s=now_s
        )
        assert s2 is not None and s2["slope"] == pytest.approx(0.42)
        assert spy_load.call_count == 2
    assert cachetrend.CacheShortTrendManager._c24_entry_ts_ms([float("nan"), 100.0]) == 0.0

    # importlib.reload(state_io) (triggered by orchestrator hot-reload) preserves stat-memoization
    # in Cache24PriceManager and CacheShortTrendManager while honoring state_io.read_json_dict monkeypatches
    import importlib

    importlib.reload(state_io)
    assert state_io.is_state_io_dict_reader_unpatched() is True
    c24_file = tmp_path / cacheprice.Cache24PriceManager.disk_filename_for_symbol("BTCUSDC")
    now_ms = int(now_s * 1000)
    atomic_write_json(
        str(c24_file),
        {"items": {"BTCUSDC": [[now_ms - k * 60_000, 85_000.0 + k] for k in range(80, 0, -1)]}},
    )
    cacheprice._SSOT_24PRICE_DISK_MEMO.clear()
    cacheprice._SSOT_24PRICE_PARSED_MEMO.clear()
    with patch.object(cacheprice, "_now_time", return_value=now_s), patch(
        "state_io.json.load", wraps=json.load
    ) as spy_after_reload:
        h_a = cacheprice.Cache24PriceManager.read_ssot_price_history(("BTCUSDC",), 1.0, custom_dir=str(tmp_path))
        h_b = cacheprice.Cache24PriceManager.read_ssot_price_history(("BTCUSDC",), 0.5, custom_dir=str(tmp_path))
        assert h_a is not None and h_b is not None
        assert spy_after_reload.call_count == 1
    with patch.object(state_io, "read_json_dict", return_value={"items": {"BTCUSDC": [[now_ms - 60_000, 1.0], [now_ms - 10_000, 2.0]]}}):
        assert state_io.is_state_io_dict_reader_unpatched() is False
        with patch.object(cacheprice, "_now_time", return_value=now_s):
            h_p = cacheprice.Cache24PriceManager.read_ssot_price_history(("BTCUSDC",), 1.0, custom_dir=str(tmp_path))
            assert h_p is not None and h_p[-1]["price"] == pytest.approx(2.0)


def test_cache_freshness_xstock_aliases_passive_read_jsonl_bounds_and_record_cid_ssot(tmp_path):
    """Verify scoped fetchtime refresh, xStock alias dedup/merge, passive path resolution, JSONL span bounds, and strict record CID SSOT."""
    import cacheaccount
    import cachepaths
    import order_lifecycle
    import order_retry
    from providers.order_id_context import record_client_order_id

    now_s = time.time()
    now_ms = int(now_s * 1000)
    stale_48h_ms = now_ms - 48 * 3600 * 1000

    # 1. Execution cache freshness: updating HYPEUSD with create_missing=True must NOT refresh 48h-old TAOUSD.
    kr_file = tmp_path / "cache_trade_kraken.json"
    save_json_state(
        str(kr_file),
        {
            "items": {
                "TAOUSD": [{"id": "old-tao", "symbol": "TAOUSD", "time": stale_48h_ms, "price": 300.0, "qty": 1.0}],
                "HYPEUSD": [],
            },
            "fetchtime": {"TAOUSD": stale_48h_ms, "HYPEUSD": stale_48h_ms},
        },
    )
    assert cacheaccount.CacheTradeManager.merge_venue_fills_snapshot(
        str(kr_file),
        {"HYPEUSD": [{"id": "new-hype", "symbol": "HYPEUSD", "time": now_ms, "price": 50.0, "qty": 2.0}]},
        now_ms=now_ms,
        is_snapshot=False,
        touch_symbols=["HYPEUSD"],
        touch_create_missing=True,
    ) is True
    after_hype = read_json_dict(str(kr_file))
    assert after_hype["fetchtime"]["HYPEUSD"] == now_ms
    assert after_hype["fetchtime"]["TAOUSD"] == stale_48h_ms
    assert cacheaccount.CacheTradeManager.read_venue_fills_snapshot(
        "TAOUSD", str(kr_file), max_stale_sec=90.0, candidate_keys=("TAOUSD",)
    ) is None
    hype_fills = cacheaccount.CacheTradeManager.read_venue_fills_snapshot(
        "HYPEUSD", str(kr_file), max_stale_sec=90.0, candidate_keys=("HYPEUSD",)
    )
    assert hype_fills is not None and [r["id"] for r in hype_fills] == ["new-hype"]

    # Explicit wildcard '*' still covers all existing keys when create_missing=True,
    # and respects invalidated keys when create_missing=False.
    wc_doc = {"items": {"TAOUSD": [], "HYPEUSD": []}, "fetchtime": {"HYPEUSD": stale_48h_ms}}
    assert cacheaccount.CacheTradeManager.touch_venue_ws_fetchtimes_in_dict(
        wc_doc, now_ms, symbols=["*"], create_missing=False
    ) is True
    assert wc_doc["fetchtime"] == {"HYPEUSD": now_ms}
    assert cacheaccount.CacheTradeManager.touch_venue_ws_fetchtimes_in_dict(
        wc_doc, now_ms + 1000, symbols=["*"], create_missing=True
    ) is True
    assert wc_doc["fetchtime"] == {"HYPEUSD": now_ms + 1000, "TAOUSD": now_ms + 1000}

    # 2. Kraken xStock aliases: incremental merge preserves all alias fills, read dedups aliases, snapshot replaces.
    f1 = {"id": "1", "symbol": "SPCXxUSD", "time": now_ms - 3000, "price": 140.0, "qty": 1.0}
    f2 = {"id": "2", "symbol": "SPCXXUSD", "time": now_ms - 2000, "price": 141.0, "qty": 1.0}
    f3 = {"id": "3", "symbol": "SPCXxUSD", "time": now_ms - 1000, "price": 142.0, "qty": 1.0}
    inc_doc = {
        "items": {"SPCXxUSD": [dict(f1)], "SPCXXUSD": [dict(f2)]},
        "fetchtime": {"SPCXxUSD": now_ms - 3000, "SPCXXUSD": now_ms - 2000},
    }
    assert cacheaccount.CacheTradeManager.merge_venue_fills_into_dict(
        inc_doc, {"SPCXxUSD": [dict(f3)]}, now_ms=now_ms, is_snapshot=False
    ) is True
    assert list(inc_doc["items"].keys()) == ["SPCXxUSD"]
    assert [r["id"] for r in inc_doc["items"]["SPCXxUSD"]] == ["1", "2", "3"]
    assert all(r["symbol"] == "SPCXxUSD" for r in inc_doc["items"]["SPCXxUSD"])

    snap_doc = {
        "items": {"SPCXxUSD": [dict(f1)], "SPCXXUSD": [dict(f2)]},
        "fetchtime": {"SPCXxUSD": now_ms - 3000, "SPCXXUSD": now_ms - 2000},
    }
    assert cacheaccount.CacheTradeManager.merge_venue_fills_into_dict(
        snap_doc, {"SPCXxUSD": [dict(f3)]}, now_ms=now_ms, is_snapshot=True
    ) is True
    assert list(snap_doc["items"].keys()) == ["SPCXxUSD"]
    assert [r["id"] for r in snap_doc["items"]["SPCXxUSD"]] == ["3"]

    alias_disk = tmp_path / "alias_kraken.json"
    alias_disk.write_text(
        json.dumps(
            {
                "items": {"SPCXxUSD": [dict(f1)], "SPCXXUSD": [dict(f1), dict(f2)]},
                "fetchtime": {"SPCXxUSD": now_ms, "SPCXXUSD": now_ms},
            }
        ),
        encoding="utf-8",
    )
    san = cacheaccount.CacheTradeManager.load_sanitized_trade_snapshot(str(alias_disk), venue="kraken")
    assert list(san["items"].keys()) == ["SPCXxUSD"]
    assert [r["id"] for r in san["items"]["SPCXxUSD"]] == ["1", "2"]
    read_alias = cacheaccount.CacheTradeManager.read_venue_fills_snapshot(
        "SPCXxUSD", str(alias_disk), max_stale_sec=90.0, candidate_keys=("SPCXxUSD", "SPCXXUSD")
    )
    assert read_alias is not None and [r["id"] for r in read_alias] == ["1", "2"]

    # 3. Passive read_json_dict(..., repair_files=False) must NOT physically move flat files into orders/.
    cdb = tmp_path / "cachedb_passive"
    (cdb / "orders").mkdir(parents=True)
    flat_kr = cdb / "cache_trade_kraken.json"
    flat_kr.write_text(json.dumps({"items": {"HYPEUSD": []}, "fetchtime": {"HYPEUSD": now_ms}}), encoding="utf-8")
    loaded_passive = read_json_dict(str(flat_kr), repair_files=False)
    assert loaded_passive == {"items": {"HYPEUSD": []}, "fetchtime": {"HYPEUSD": now_ms}}
    assert flat_kr.is_file()
    assert not (cdb / "orders" / "cache_trade_kraken.json").exists()
    assert read_json_dict_cached(str(flat_kr), memo={}) == loaded_passive
    assert flat_kr.is_file()
    assert not (cdb / "orders" / "cache_trade_kraken.json").exists()
    # Explicit producer save or explicit migrate=True moves the file into orders/.
    migrated_path = cachepaths.resolve_cachedb_filepath(str(flat_kr), migrate=True)
    assert Path(str(migrated_path)) == cdb / "orders" / "cache_trade_kraken.json"
    assert (cdb / "orders" / "cache_trade_kraken.json").is_file()
    assert not flat_kr.exists()

    # 4. read_jsonl_records_since_span enforces exact cutoff window and strict <= max_records cap keeping last row.
    j_6000 = tmp_path / "dense_6000.jsonl"
    base_ms = 1_795_000_000_000
    j_6000.write_text(
        "\n".join(json.dumps({"s": "BTCUSDC", "i": [base_ms + i * 1000, 90000.0 + i]}) for i in range(6000)) + "\n",
        encoding="utf-8",
    )
    recs_cap = state_io.read_jsonl_records_since_span(str(j_6000), span_ms=10_000_000, max_records=4000)
    assert len(recs_cap) <= 4000
    assert recs_cap[-1]["i"][0] == base_ms + 5999 * 1000

    recs_one = state_io.read_jsonl_records_since_span(str(j_6000), span_ms=10_000_000, max_records=1)
    assert len(recs_one) == 1
    assert recs_one[0]["i"][0] == base_ms + 5999 * 1000

    j_1500 = tmp_path / "span_1500s.jsonl"
    j_1500.write_text(
        "\n".join(json.dumps({"s": "ETHUSDC", "i": [base_ms + i * 1000, 3000.0 + i * 0.1]}) for i in range(1500)) + "\n",
        encoding="utf-8",
    )
    recs_10s = state_io.read_jsonl_records_since_span(str(j_1500), span_ms=10_000, max_records=4000)
    assert len(recs_10s) == 11
    assert recs_10s[0]["i"][0] == base_ms + 1489 * 1000
    assert recs_10s[-1]["i"][0] == base_ms + 1499 * 1000
    assert (recs_10s[-1]["i"][0] - recs_10s[0]["i"][0]) <= 10_000

    # 5. Single SSOT record_client_order_id keeps strict record['place_kwargs']['client_order_id'] contract.
    assert order_retry.record_client_order_id is record_client_order_id
    assert not hasattr(order_lifecycle, "_claim_client_order_id")
    assert record_client_order_id({"place_kwargs": {"client_order_id": " TA_123 "}}) == "TA_123"
    assert record_client_order_id({"client_order_id": "LEGACY_TOP_LEVEL"}) is None
    assert record_client_order_id({"clientOrderId": "LEGACY_CAMEL"}) is None
    assert record_client_order_id({"place_kwargs": {"clientOrderId": "WRONG_KEY"}}) is None
    assert record_client_order_id({"place_kwargs": "not-a-dict"}) is None
    assert record_client_order_id({"place_kwargs": {"client_order_id": True}}) is None
    assert record_client_order_id({"place_kwargs": {"client_order_id": "nan"}}) is None
    assert record_client_order_id("not-a-dict") is None


def test_phase2_passive_readers_lock_alignment_and_trade_alias_coalescing(tmp_path):
    """Verify repo-wide passive read-only guarantees, lock-vs-migration alignment, and non-append Trade alias coalescing."""
    import cacheaccount
    import cacheprice
    import order_retry_worker
    import rtrade_pair_store
    import trailing_core
    from providers import intent_state, venue_fee_maintenance

    # 1. load_json_state(..., repair_files=False) recovers .previous passively without moving flat files or mutating disk.
    cdb = tmp_path / "cachedb_ro"
    (cdb / "state").mkdir(parents=True)
    flat_state = cdb / "trailing_state.json"
    flat_prev = Path(previous_generation_path(str(flat_state)))
    flat_state.write_text("{corrupt", encoding="utf-8")
    flat_prev.write_text(json.dumps({"BTC": {"peak": 123.0}}), encoding="utf-8")

    recovered = state_io.load_json_state(
        str(flat_state),
        default_factory=dict,
        fail_closed=True,
        label="trailing",
        repair_files=False,
    )
    assert recovered == {"BTC": {"peak": 123.0}}
    assert flat_state.read_text(encoding="utf-8") == "{corrupt"
    assert flat_prev.is_file()
    assert not (cdb / "state" / "trailing_state.json").exists()

    # TrailingCore.load(read_only=True) stays completely passive.
    core = trailing_core.TrailingCore(
        None,
        log=lambda *_: None,
        enabled=True,
        state_file=str(flat_state),
        min_notional=10.0,
        rebuy_enabled=False,
        rebuy_bounce_pct=0.0,
        rebuy_skip_if_trend_down=False,
    )
    assert core.load(read_only=True) == {"BTC": {"peak": 123.0}}
    assert flat_state.read_text(encoding="utf-8") == "{corrupt"
    assert flat_prev.is_file()
    assert not (cdb / "state" / "trailing_state.json").exists()

    # 2. load_durable_intents_state(..., read_only=True) does not lock, migrate, or repair disk.
    from providers import intent_state_store

    flat_intents = cdb / "autonomous_trade_intents.json"
    flat_intents_prev = Path(previous_generation_path(str(flat_intents)))
    flat_intents.write_text("{corrupt", encoding="utf-8")
    flat_intents_prev.write_text(json.dumps([{"intent_id": "i1"}]), encoding="utf-8")
    ro_intents = intent_state_store.load_durable_intents_state(str(flat_intents), read_only=True)
    assert ro_intents == [{"intent_id": "i1"}]
    assert flat_intents.read_text(encoding="utf-8") == "{corrupt"
    assert flat_intents_prev.is_file()
    assert not (cdb / "state" / "autonomous_trade_intents.json").exists()
    assert not Path(state_io.generation_lock_path(str(flat_intents))).exists()

    # Writable durable_intents_lock migrates flat file into state/ BEFORE locking so lock and file share state/.
    with intent_state_store.durable_intents_lock(str(flat_intents)):
        assert (cdb / "state" / "autonomous_trade_intents.json").is_file()
        assert not flat_intents.exists()
        assert Path(state_io.generation_lock_path(str(cdb / "state" / "autonomous_trade_intents.json"))).is_file()
        assert not Path(state_io.generation_lock_path(str(flat_intents))).exists()

    # 3. RTradePairStore.mutate migrates flat file before locking; dry_run fee maintenance stays passive.
    flat_rtrade = cdb / "rtrade_pairs.json"
    flat_rtrade.write_text(json.dumps({"version": 1, "pairs": {}}), encoding="utf-8")
    store = rtrade_pair_store.RTradePairStore(str(flat_rtrade))
    assert store.active("BTCUSDC") == []
    assert flat_rtrade.is_file()
    assert not (cdb / "state" / "rtrade_pairs.json").exists()
    store.mutate(lambda pairs: True)
    assert (cdb / "state" / "rtrade_pairs.json").is_file()
    assert not flat_rtrade.exists()
    assert (cdb / "state" / "rtrade_pairs.json.lock").is_file()

    flat_fee = cdb / "venue_fee_maintenance_state.json"
    flat_fee_prev = Path(previous_generation_path(str(flat_fee)))
    flat_fee.write_text("{corrupt", encoding="utf-8")
    flat_fee_prev.write_text(json.dumps({"last_run_ts": 100.0, "venues": {}}), encoding="utf-8")
    fee_res = venue_fee_maintenance.run_venue_fee_maintenance(
        venues=[],
        force=False,
        dry_run=True,
        cache_dir=str(cdb),
        cfg_map={
            "FEE_MAINTENANCE_ENABLED": "true",
            "FEE_MAINTENANCE_INTERVAL_SEC": "3600",
            "BINANCE_BNB_RESERVE_MIN_USDC": "5.0",
            "BINANCE_BNB_RESERVE_TARGET_USDC": "15.0",
            "BINANCE_BNB_MIN_FREE_USDC": "25.0",
            "BINANCE_BNB_ORDER_STALE_SEC": "600",
            "KRAKEN_IDLE_USDC_ALERT_MIN": "50.0",
            "KRAKEN_IDLE_USDC_ALERT_COOLDOWN_SEC": "3600",
        },
        now_ts=120.0,
    )
    assert fee_res["status"] == "not_due"
    assert flat_fee.read_text(encoding="utf-8") == "{corrupt"
    assert flat_fee_prev.is_file()
    assert not (cdb / "state" / "venue_fee_maintenance_state.json").exists()

    # IncidentDigest.collect migrates flat orders/ file before acquiring its lock.
    (cdb / "orders").mkdir(parents=True)
    flat_digest = cdb / "order_retry_digest.json"
    flat_digest.write_text(json.dumps({"version": 1, "groups": {}}), encoding="utf-8")
    digest = order_retry_worker.IncidentDigest(str(flat_digest), 300.0)
    due = digest.collect(
        [{"labels": {"symbol": "BTCUSDC"}, "quantity": 1.0, "sample_id": "s1"}],
        now=1000.0,
    )
    assert len(due) == 1
    assert (cdb / "orders" / "order_retry_digest.json").is_file()
    assert (cdb / "orders" / "order_retry_digest.json.lock").is_file()
    assert not flat_digest.exists()

    # 4. Cache365PriceManager(read_only=True) recovers .previous in memory without restoring on disk.
    p365 = tmp_path / "cache_365_price_BTCUSDC.jsonl"
    p365_prev = Path(previous_generation_path(str(p365)))
    p365_prev.write_text(json.dumps({"s": "BTCUSDC", "i": [1_795_000_000_000, 95000.0]}) + "\n", encoding="utf-8")
    mgr365 = cacheprice.Cache365PriceManager(
        3600, ["BTCUSDC"], str(p365), api_client=None, read_only=True
    )
    assert mgr365.cache.get("BTCUSDC") == [[1_795_000_000_000, 95000.0]]
    assert not p365.exists()
    assert p365_prev.is_file()

    # 5. CacheTradeManager(read_only=True) must NOT migrate legacy .json -> .jsonl on disk.
    legacy_trade_jsonl = tmp_path / "cache_trade.jsonl"
    legacy_trade_json = tmp_path / "cache_trade.json"
    legacy_trade_json.write_text(
        json.dumps({"items": {"BTCUSDC": []}, "fetchtime": {"BTCUSDC": 1000}}),
        encoding="utf-8",
    )
    ro_trade = cacheaccount.CacheTradeManager(
        3600, ["BTCUSDC"], str(legacy_trade_jsonl), api_client=None, read_only=True
    )
    assert ro_trade.read_only is True
    assert not legacy_trade_jsonl.exists()
    assert not Path(state_io.meta_file_path(str(legacy_trade_jsonl))).exists()

    # 6. CacheTradeManager._persist_items (append_persist=False) coalesces xStock case aliases.
    now_ms = int(time.time() * 1000)
    kr_snap = tmp_path / "cache_trade_kraken_mgr.json"
    kr_mgr = cacheaccount.CacheTradeManager(
        3600, ["SPCXxUSD"], str(kr_snap), api_client=None, read_only=True, venue="kraken", append_persist=False
    )
    f1 = {"symbol": "SPCXxUSD", "id": "1", "orderId": "O1", "price": 10.0, "qty": 1.0, "time": now_ms - 3000, "isBuyer": True}
    f2 = {"symbol": "SPCXXUSD", "id": "2", "orderId": "O2", "price": 11.0, "qty": 1.0, "time": now_ms - 2000, "isBuyer": True}
    f3 = {"symbol": "SPCXXUSD", "id": "3", "orderId": "O3", "price": 12.0, "qty": 1.0, "time": now_ms - 1000, "isBuyer": True}
    kr_mgr.cache = {"SPCXxUSD": [dict(f1)], "SPCXXUSD": [dict(f2)]}
    kr_mgr.fetchtime_per_symbol = {"SPCXxUSD": now_ms - 5000, "SPCXXUSD": now_ms - 1000}
    kr_mgr._persist_items("SPCXXUSD", [dict(f3)])
    assert list(kr_mgr.cache.keys()) == ["SPCXxUSD"]
    assert [r["id"] for r in kr_mgr.cache["SPCXxUSD"]] == ["1", "2", "3"]
    assert all(r["symbol"] == "SPCXxUSD" for r in kr_mgr.cache["SPCXxUSD"])
    assert "SPCXXUSD" not in kr_mgr.fetchtime_per_symbol

    # 7. order_retry_worker._format_record_context uses strict record_client_order_id fallback.
    fmt = order_retry_worker._format_record_context(
        {
            "symbol": "BTCUSDC",
            "side": "BUY",
            "qty": 0.1,
            "requested_price": 50000.0,
            "id": "rec_fallback_1",
            "place_kwargs": {"client_order_id": "   "},
        }
    )
    assert "ID: rec_fallback_1" in fmt


def test_atomic_write_json_fsync_flag_and_ephemeral_telemetry_collector(tmp_path, monkeypatch):
    """Verify atomic_write_json(fsync=False) skips os.fsync while preserving POSIX atomicity,
    and ephemeral telemetry collectors default to fsync=False while honoring monkeypatched save_json_state."""
    from intelligence.external.collectors import base_collector as bc_mod
    from intelligence.external.collectors.orderbook_depth_collector import OrderbookDepthCollector, OrderbookSnapshot

    fsync_calls: list[int] = []
    monkeypatch.setattr(os, "fsync", lambda fd: fsync_calls.append(fd))

    # 1. Default fsync=True calls os.fsync
    durable_path = tmp_path / "durable.json"
    state_io.atomic_write_json(str(durable_path), {"ok": True}, indent=2)
    assert json.loads(durable_path.read_text(encoding="utf-8")) == {"ok": True}
    assert len(fsync_calls) >= 1

    # 2. fsync=False writes atomically with zero os.fsync calls
    fsync_calls.clear()
    fast_path = tmp_path / "ephemeral.json"
    state_io.atomic_write_json(str(fast_path), {"fast": 1}, indent=2, fsync=False)
    assert json.loads(fast_path.read_text(encoding="utf-8")) == {"fast": 1}
    assert fsync_calls == []

    # 3. _DiskBackedSymbolCollector._save_to_disk uses fsync=False and creates no .previous staging
    col = OrderbookDepthCollector(cache_dir=str(tmp_path), start_sync=False)
    snap = OrderbookSnapshot(
        symbol="BTCUSDC",
        mid_price=70000.0,
        bid_depth_usd=100000.0,
        ask_depth_usd=100000.0,
        imbalance_ratio=0.5,
        largest_bid_wall_usd=25000.0,
        largest_bid_wall_price=69990.0,
        largest_ask_wall_usd=25000.0,
        largest_ask_wall_price=70010.0,
        ts=1_760_000_000.0,
        best_bid=69999.0,
        best_ask=70001.0,
        spread_bps=0.29,
    )
    col._save_to_disk(snap)
    assert fsync_calls == []
    assert not list(tmp_path.rglob("*.previous"))
    saved_file = tmp_path / "orderbook_depth_BTCUSDC.json"
    assert saved_file.exists()
    assert json.loads(saved_file.read_text(encoding="utf-8"))["symbol"] == "BTCUSDC"

    # 4. Monkeypatching save_json_state on base_collector still intercepts collector saves
    patched_calls: list[str] = []
    monkeypatch.setattr(bc_mod, "save_json_state", lambda p, d, **_kw: patched_calls.append(p))
    col._save_to_disk(snap)
    assert len(patched_calls) == 1


def test_cache24_and_current_price_write_compact_json_without_fsync(tmp_path, monkeypatch):
    """Verify Cache24PriceManager and CacheCurrentPriceManager persist compact JSON with zero fsync calls,
    while default commit_file_and_meta(fsync=True) remains fully fsync-durable."""
    import cacheprice

    fsync_calls: list[int] = []
    monkeypatch.setattr(os, "fsync", lambda fd: fsync_calls.append(fd))

    # 1. Default commit_file_and_meta(fsync=True) still calls os.fsync
    durable_target = tmp_path / "durable_account.json"
    assert state_io.commit_file_and_meta(
        str(durable_target),
        lambda: state_io.atomic_write_json(str(durable_target), {"orders": [1]}),
        lambda: (state_io.write_meta_json(str(durable_target), {"max_ts": 1}), True)[1],
    )
    assert len(fsync_calls) >= 2

    # 2. Cache24PriceManager saves compact JSON (no indent newlines) with 0 fsync calls
    fsync_calls.clear()
    now_ms = int(time.time() * 1000)
    c24_path = tmp_path / "cache_24_price_BTCUSDC.json"
    mgr24 = cacheprice.Cache24PriceManager(30, ["BTCUSDC"], str(c24_path), api_client=None, read_only=False)
    mgr24.cache = {"BTCUSDC": [[now_ms - 1000, 60000.0], [now_ms, 60010.0]]}
    mgr24.fetchtime_per_symbol = {"BTCUSDC": now_ms}
    assert mgr24.save_state_to_file() is True
    assert fsync_calls == []
    raw_24 = c24_path.read_text(encoding="utf-8")
    assert "\n" not in raw_24
    assert json.loads(raw_24)["items"]["BTCUSDC"][-1][1] == 60010.0

    # Second save (with existing file -> triggers stage_previous_generation(fsync=False)) ALSO uses 0 fsync
    mgr24.cache["BTCUSDC"].append([now_ms + 1000, 60020.0])
    mgr24.fetchtime_per_symbol["BTCUSDC"] = now_ms + 1000
    assert mgr24.save_state_to_file() is True
    assert fsync_calls == []

    # 3. CacheCurrentPriceManager saves compact JSON with 0 fsync calls
    cp_path = tmp_path / "cache_current_price.json"
    cpm = cacheprice.CacheCurrentPriceManager(5, ["BTCUSDC"], str(cp_path), api_client=None, read_only=False)
    cpm.on_items_update("BTCUSDC", [[now_ms + 2000, 60030.0]])
    assert cpm.save_state_to_file() is True
    assert fsync_calls == []
    raw_cp = cp_path.read_text(encoding="utf-8")
    assert "\n" not in raw_cp
    assert json.loads(raw_cp)["items"]["BTCUSDC"] == [[now_ms + 2000, 60030.0]]
    cpm.shutdown(timeout=1.0)


def test_cache24_jsonl_append_migration_compaction_and_staggered_cron(tmp_path, monkeypatch):
    """Verify Cache24PriceManager .jsonl legacy migration, append-only delta writes (0 fsync),
    periodic rolling-24h compaction, dual readers, low-priority rsync scripts, and staggered crontab."""
    import subprocess
    import cacheprice
    import cachetrend
    from intelligence.sentiment.collectors.microstructure_collector import load_symbol_price_series

    fsync_calls: list[int] = []
    monkeypatch.setattr(os, "fsync", lambda fd: fsync_calls.append(fd))

    prices_dir = tmp_path / "prices"
    prices_dir.mkdir(parents=True, exist_ok=True)
    now_s = time.time()
    now_ms = int(now_s * 1000)

    # 1. Seed legacy cache_24_price_BTCUSDC.json -> Cache24PriceManager(.jsonl) auto-migrates on load_state()
    legacy_json = prices_dir / "cache_24_price_BTCUSDC.json"
    jsonl_path = prices_dir / "cache_24_price_BTCUSDC.jsonl"
    seed_pts = [[now_ms - (30 - i) * 60_000, 60000.0 + i] for i in range(30)]
    legacy_json.write_text(
        json.dumps({"items": {"BTCUSDC": seed_pts}, "fetchtime": {"BTCUSDC": seed_pts[-1][0]}}),
        encoding="utf-8",
    )

    mgr = cacheprice.Cache24PriceManager(30, ["BTCUSDC"], str(jsonl_path), api_client=None, read_only=False)
    assert mgr.append_persist is True
    mgr.load_state()
    assert jsonl_path.exists()
    assert fsync_calls == []
    lines_after_migrate = jsonl_path.read_text(encoding="utf-8").splitlines()
    assert len(lines_after_migrate) == 30

    # 2. New ticks append only delta lines without rewriting the whole file (compact_jsonl not called!)
    compact_calls: list[int] = []
    orig_compact = mgr.compact_jsonl

    def _spy_compact():
        compact_calls.append(1)
        return orig_compact()

    monkeypatch.setattr(mgr, "compact_jsonl", _spy_compact)
    size_before = jsonl_path.stat().st_size
    mgr.on_price_update("BTCUSDC", now_ms - 10_000, 60100.0)
    mgr.on_price_update("BTCUSDC", now_ms - 5_000, 60105.0)
    assert mgr.save_state_to_file() is True
    assert compact_calls == []
    assert fsync_calls == []
    assert jsonl_path.stat().st_size > size_before
    assert len(jsonl_path.read_text(encoding="utf-8").splitlines()) == 32

    # 3. Trimming expired >24h rows from memory adjusts _persisted_counts without forcing immediate rewrite,
    #    then triggers compaction once COMPACT_INTERVAL_SEC elapses.
    old_ts = now_ms - (25 * 3600 * 1000)
    mgr.cache["BTCUSDC"].insert(0, [old_ts, 59000.0])
    mgr._persisted_counts["BTCUSDC"] += 1
    mgr._trim_old_data("BTCUSDC", min_batch=1)
    assert mgr._jsonl_trimmed_disk_rows == 1
    assert mgr._should_compact_before_append_locked() is False
    mgr._last_24h_compact_mono -= (mgr.COMPACT_INTERVAL_SEC + 10.0)
    assert mgr._should_compact_before_append_locked() is True
    assert mgr.save_state_to_file() is True
    assert len(compact_calls) == 1
    assert mgr._jsonl_trimmed_disk_rows == 0
    assert fsync_calls == []

    # 4. All major consumers read cache_24_price_BTCUSDC.jsonl seamlessly (even if legacy .json is absent)
    legacy_json.unlink()
    cacheprice._SSOT_24PRICE_DISK_MEMO.clear()
    cacheprice._SSOT_24PRICE_PARSED_MEMO.clear()
    hist = cacheprice.Cache24PriceManager.read_ssot_price_history(("BTCUSDC",), 0.25, custom_dir=str(tmp_path))
    assert hist is not None and hist[-1]["price"] == pytest.approx(60105.0)
    stm = cachetrend.CacheShortTrendManager(symbols=["BTCUSDC"], filename=str(tmp_path / "cache_short_trend.json"), read_only=True)
    disk_entries = stm._load_disk_cache24_entries("BTCUSDC")
    assert disk_entries and disk_entries[-1][1] == pytest.approx(60105.0)
    ser_24h, _ = load_symbol_price_series(str(tmp_path), "BTCUSDC", include_7d=False)
    assert ser_24h and ser_24h[-1][1] == pytest.approx(60105.0)

    # 5. Low-priority rsync scripts pass bash -n and crontab.prod.txt runs sync_cache_to at 60m staggered minute 11
    repo_root = Path(__file__).resolve().parents[1]
    for sh_rel in ("orchestratorOS/lib/backup_sync_common.sh", "offline/runners/sync_cache_to.sh"):
        sh_path = repo_root / sh_rel
        assert subprocess.run(["bash", "-n", str(sh_path)], capture_output=True, text=True, check=False).returncode == 0
    assert "_low_prio_cmd" in (repo_root / "orchestratorOS/lib/backup_sync_common.sh").read_text(encoding="utf-8")
    cron_lines = [
        ln.strip()
        for ln in (repo_root / "systemd/crontab.prod.txt").read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#") and "=" not in ln.split()[0]
    ]
    sync_lines = [ln for ln in cron_lines if "sync_cache_to.sh dev" in ln]
    assert len(sync_lines) == 1 and sync_lines[0].startswith("11 * * * * ")
    assert not any(ln.startswith(("*/15 ", "*/20 ", "*/30 ", "*/5 ", "0 ")) for ln in cron_lines)
    # Shadow forward-tests run on DEV (crontab.dev.txt), not PROD, and sync_cache_to.sh does not overwrite DEV shadow logs
    assert not any("shadow_live.py" in ln or "shadow_longterm.py" in ln for ln in cron_lines)
    dev_cron_lines = [
        ln.strip()
        for ln in (repo_root / "systemd/crontab.dev.txt").read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#") and "=" not in ln.split()[0]
    ]
    assert any("kraken/shadow_live.py" in ln for ln in dev_cron_lines)
    assert any("hyperliquid/shadow_longterm.py" in ln for ln in dev_cron_lines)
    sync_sh_txt = (repo_root / "offline/runners/sync_cache_to.sh").read_text(encoding="utf-8")
    assert "for sub in shadow_live hyperliquid_shadow" not in sync_sh_txt


def test_price_analysis_cpu_throttling_and_low_priority_background_jobs(monkeypatch):
    """Verify priceAnalysis disables PNG writes by default (PRICEANALYSIS_DRAW_PLOTS=false), skips unchanged
    SparsePrice series, optionally throttles PNG redraws when enabled, idempotently raises niceness,
    and background crons/procs.conf run with nice/ionice."""
    import priceAnalysis as pa

    # 0. PNG plot drawing is opt-in and disabled by default in config.env and module flag
    repo_root = Path(__file__).resolve().parents[1]
    assert pa.PRICEANALYSIS_DRAW_PLOTS is False
    assert "PRICEANALYSIS_DRAW_PLOTS=false" in (repo_root / "config.env").read_text(encoding="utf-8")

    # 1. _ensure_min_niceness is idempotent (computes delta, never double-adds)
    nice_state = [0]

    def _fake_nice(inc: int) -> int:
        nice_state[0] += inc
        return nice_state[0]

    monkeypatch.setattr(pa.os, "nice", _fake_nice)
    assert pa._ensure_min_niceness(10) == 10
    assert pa._ensure_min_niceness(10) == 10
    assert nice_state[0] == 10

    # 2a. Default mode (PRICEANALYSIS_DRAW_PLOTS=False): NEVER draws PNGs, even on first cycle, flip, or 15m refresh
    sig_cur = [(100, 1_760_000_000_000, 60000.0)]
    dir_cur = ["up"]
    calls: list[tuple[str, bool]] = []

    monkeypatch.setattr(pa, "_series_signature_for", lambda _s: sig_cur[0])

    def _fake_trend(sym, **kw):
        calls.append((sym, bool(kw.get("draw")), kw.get("draw_on_direction_change_from")))
        return {
            "timestamp": 1000,
            "direction": dir_cur[0],
            "start_timestamp": 900.0,
            "duration_seconds": 100.0,
            "estimated_future_hours": 10.0,
        }

    monkeypatch.setattr(pa, "getTrendLongTerm_fixed", _fake_trend)
    state_off: dict = {}

    r1 = pa.run_price_analysis_cycle(["BTCUSDC"], state_off, now_ts=1000.0, inter_symbol_yield_sec=0.0)
    assert calls == [("BTCUSDC", False, None)] and r1["BTCUSDC"]["direction"] == "up"

    r2 = pa.run_price_analysis_cycle(["BTCUSDC"], state_off, now_ts=1060.0, inter_symbol_yield_sec=0.0)
    assert calls == [("BTCUSDC", False, None)] and r2["BTCUSDC"]["timestamp"] == 1060

    sig_cur[0] = (101, 1_760_000_180_000, 59500.0)
    dir_cur[0] = "down"
    r3 = pa.run_price_analysis_cycle(["BTCUSDC"], state_off, now_ts=1180.0, inter_symbol_yield_sec=0.0)
    assert calls == [("BTCUSDC", False, None), ("BTCUSDC", False, None)] and r3["BTCUSDC"]["direction"] == "down"

    pa.run_price_analysis_cycle(["BTCUSDC"], state_off, now_ts=2300.0, inter_symbol_yield_sec=0.0)
    assert calls == [("BTCUSDC", False, None), ("BTCUSDC", False, None), ("BTCUSDC", False, None)]

    # 2b. Opt-in PNG drawing (draw_plots=True): throttles PNG redraws to 15m and redraws in a SINGLE call on direction flip
    calls.clear()
    sig_cur[0] = (100, 1_760_000_000_000, 60000.0)
    dir_cur[0] = "up"
    state_on: dict = {}

    pa.run_price_analysis_cycle(["BTCUSDC"], state_on, now_ts=1000.0, draw_plots=True, inter_symbol_yield_sec=0.0)
    assert calls == [("BTCUSDC", True, None)]

    pa.run_price_analysis_cycle(["BTCUSDC"], state_on, now_ts=1060.0, draw_plots=True, inter_symbol_yield_sec=0.0)
    assert calls == [("BTCUSDC", True, None)]

    sig_cur[0] = (101, 1_760_000_180_000, 60050.0)
    pa.run_price_analysis_cycle(["BTCUSDC"], state_on, now_ts=1180.0, draw_plots=True, inter_symbol_yield_sec=0.0)
    assert calls == [("BTCUSDC", True, None), ("BTCUSDC", False, "up")]

    sig_cur[0] = (102, 1_760_000_360_000, 59500.0)
    dir_cur[0] = "down"
    pa.run_price_analysis_cycle(["BTCUSDC"], state_on, now_ts=1360.0, draw_plots=True, inter_symbol_yield_sec=0.0)
    assert calls == [("BTCUSDC", True, None), ("BTCUSDC", False, "up"), ("BTCUSDC", False, "up")]
    assert state_on["last_draw_ts"]["BTCUSDC"] == 1360.0

    pa.run_price_analysis_cycle(["BTCUSDC"], state_on, now_ts=2300.0, draw_plots=True, inter_symbol_yield_sec=0.0)
    assert calls[-1] == ("BTCUSDC", True, None)

    # 3. procs.conf & crontab.prod.txt apply nice/ionice to heavy background jobs while keeping trade executors at normal priority
    procs_txt = (repo_root / "procs.conf").read_text(encoding="utf-8")
    pa_rows = [ln for ln in procs_txt.splitlines() if ln.startswith("priceAnalysis.py|")]
    assert len(pa_rows) == 1 and "nice -n 10 ionice -c3 python priceAnalysis.py" in pa_rows[0]
    cron_txt = (repo_root / "systemd/crontab.prod.txt").read_text(encoding="utf-8")
    for bg_tok in ("manage_logs.sh", "manage_backups.sh", "sync_cache_to.sh dev", "check_cache_coherence.py"):
        line = next(ln for ln in cron_txt.splitlines() if bg_tok in ln and not ln.strip().startswith("#"))
        assert "nice -n 15" in line
    for exec_tok in ("--topic-key auto_cryptotrade_executor", "--topic-key auto_sharetrade_executor"):
        line = next(ln for ln in cron_txt.splitlines() if exec_tok in ln and not ln.strip().startswith("#"))
        assert "nice -n" not in line


def test_incremental_jsonl_tailing_and_assetguardian_vectorized_extrema(tmp_path, monkeypatch):
    """Verify read-only JSONL caches tail appended lines incrementally (falling back cleanly on compaction/rewrite/partial EOF)
    and AssetGuardian's vectorized window extrema/volatility matches legacy dict helpers."""
    import assetguardian as ag
    import cacheInterface as ci
    import cacheprice

    # 1. Incremental JSONL tailing: append -> tail only new bytes; partial EOF -> pause safely;
    #    compaction (new inode) & in-place rewrite (same inode, anchor mismatch) -> clean full reload.
    prices_dir = tmp_path / "prices"
    prices_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = prices_dir / "cache_24_price_BTCUSDC.jsonl"
    now_ms = int(time.time() * 1000)

    prod = cacheprice.Cache24PriceManager(
        30, ["BTCUSDC"], str(jsonl_path), api_client=None, read_only=False,
        current_price_getter=lambda _s: None,
    )
    for idx in range(20):
        prod.on_price_update("BTCUSDC", now_ms - (30 - idx) * 1000, 60000.0 + idx)
    assert prod.save_state_to_file() is True

    reader = cacheprice.Cache24PriceManager(30, ["BTCUSDC"], str(jsonl_path), api_client=None, read_only=True)
    assert len(reader.cache["BTCUSDC"]) == 20
    assert reader._jsonl_tail_state is not None
    offset_after_init = reader._jsonl_tail_state[2]
    assert offset_after_init > 0

    # Spy on read_jsonl_bucket_map_from_offset to verify incremental_used=True on append
    tail_calls: list[tuple[int, bool]] = []
    orig_tail = state_io.read_jsonl_bucket_map_from_offset

    def _spy_tail(*args, **kwargs):
        res = orig_tail(*args, **kwargs)
        tail_calls.append((kwargs.get("start_offset", 0), res[5]))
        return res

    monkeypatch.setattr(ci, "read_jsonl_bucket_map_from_offset", _spy_tail)

    # Producer appends 3 new ticks on same inode -> reader MUST use incremental tail from offset_after_init
    prod.on_price_update("BTCUSDC", now_ms - 5_000, 60100.0)
    prod.on_price_update("BTCUSDC", now_ms - 4_000, 60101.0)
    prod.on_price_update("BTCUSDC", now_ms - 3_000, 60102.0)
    assert prod.save_state_to_file() is True
    reader.resync_mem_file()
    assert len(reader.cache["BTCUSDC"]) == 23
    assert reader.cache["BTCUSDC"][-1][1] == pytest.approx(60102.0)
    assert tail_calls[-1] == (offset_after_init, True)

    # In-flight partial trailing line without '\n': reader preserves cursor and picks up line once completed
    prev_off = reader._jsonl_tail_state[2]
    partial_line = json.dumps({"s": "BTCUSDC", "i": [now_ms - 2_000, 60103.0]}, separators=(",", ":"))
    with open(jsonl_path, "ab") as fh:
        fh.write(partial_line[:14].encode("utf-8"))
    reader._load_jsonl()
    assert len(reader.cache["BTCUSDC"]) == 23
    assert reader._jsonl_tail_state[2] == prev_off
    with open(jsonl_path, "ab") as fh:
        fh.write((partial_line[14:] + "\n").encode("utf-8"))
    reader._load_jsonl()
    assert len(reader.cache["BTCUSDC"]) == 24
    assert reader.cache["BTCUSDC"][-1][1] == pytest.approx(60103.0)

    # Producer compacts (atomic replace -> new inode) -> reader detects inode change and performs full reload
    prod.cache["BTCUSDC"] = prod.cache["BTCUSDC"][-5:]
    assert prod.compact_jsonl() is True
    reader.cache["BTCUSDC"].append([now_ms - 99_999, 1.0])  # ensure full reload clears stale rows
    reader._load_jsonl()
    assert tail_calls[-1][1] is False
    assert len(reader.cache["BTCUSDC"]) == 5

    # In-place overwrite on same inode with larger payload -> anchor mismatch forces full reload
    big_rows = [
        json.dumps({"s": "BTCUSDC", "i": [now_ms - (50 - k) * 1000, 61000.0 + k]}, separators=(",", ":"))
        for k in range(40)
    ]
    with open(jsonl_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(big_rows) + "\n")
    reader._load_jsonl()
    assert tail_calls[-1][1] is False
    assert len(reader.cache["BTCUSDC"]) == 40
    assert reader.cache["BTCUSDC"][0][1] == pytest.approx(61000.0)

    # 2. AssetGuardian vectorized outlier filter, volatility multiplier, and _get_symbol_window_extrema
    #    match legacy dict helpers exactly (including duplicate timestamps, spikes, and extreme outliers).
    base_ts = 1_795_000_000.0
    dict_rows = []
    raw_rows = []
    for k in range(180):
        ts_s = base_ts + k * 60.0
        px = 100.0 + (k % 11) * 0.4 + (0.15 if k % 3 == 0 else -0.1)
        if k == 42:
            px = 250.0  # isolated single-tick spike
        elif k == 120:
            px = 450.0  # uncorroborated extreme outlier
        dict_rows.append({"timestamp": ts_s, "price": px})
        raw_rows.append([int(ts_s * 1000), px])
    # Add an exact duplicate and a same-ts different-px row + malformed rows
    raw_rows.append(list(raw_rows[10]))
    raw_rows.append(["bad", 100.0])
    raw_rows.append([int((base_ts + 20 * 60.0) * 1000), float("nan")])

    legacy_clean, legacy_removed = ag._filter_isolated_price_outliers(dict_rows)
    legacy_vol = ag._compute_symbol_volatility_multiplier(legacy_clean)

    monkeypatch.setattr(ag, "_read_symbol_price_rows", lambda _s: raw_rows)
    monkeypatch.setattr(ag.time, "time", lambda: base_ts + 180 * 60.0)
    ag._local_price_samples["BTCUSDC"] = []
    extrema = ag._get_symbol_window_extrema("BTCUSDC", 101.5, minutes_back=240.0)
    assert extrema is not None
    lo, hi = extrema
    # Legacy clean + current_price at base_ts + 180*60 should match min/max & vol_mult closely
    dict_plus = dict_rows + [{"timestamp": base_ts + 180 * 60.0, "price": 101.5}]
    exp_clean, exp_removed = ag._filter_isolated_price_outliers(dict_plus)
    assert exp_removed == 2 and legacy_removed == 2
    assert lo == min(exp_clean, key=lambda r: r["price"])
    assert hi == max(exp_clean, key=lambda r: r["price"])
    assert ag._last_evaluation["BTCUSDC"]["volatility_mult"] == ag._compute_symbol_volatility_multiplier(exp_clean)
    assert abs(ag._last_evaluation["BTCUSDC"]["volatility_mult"] - legacy_vol) <= 0.05

    # Outlier current_price is still rejected cleanly
    assert ag._get_symbol_window_extrema("BTCUSDC", 9999.0, minutes_back=240.0) is None


def test_fleet_wide_ephemeral_no_fsync_and_incremental_jsonl_cached_readers(tmp_path, monkeypatch):
    """Verify ephemeral trends/heartbeats skip fsync (while default save_json_state keeps fsync=True),
    read_jsonl_or_json_dict_cached & Cache365PriceManager tail incrementally, and macro_analyzer/DEV backtests are niced."""
    import botcore
    import cacheprice
    import cachetrend
    import priceAnalysis as pa

    fsync_calls: list[int] = []
    monkeypatch.setattr(os, "fsync", lambda fd: fsync_calls.append(fd))

    # 1. Default save_json_state(fsync=True) keeps full fsync durability; fsync=False writes atomically with 0 fsync
    durable_p = tmp_path / "durable_orders.json"
    state_io.save_json_state(str(durable_p), {"orders": [1]})
    assert len(fsync_calls) >= 1
    fsync_calls.clear()

    eph_p = tmp_path / "ephemeral_state.json"
    state_io.save_json_state(str(eph_p), {"v": 1}, separators=(",", ":"), fsync=False)
    state_io.save_json_state(str(eph_p), {"v": 2}, separators=(",", ":"), fsync=False)
    assert fsync_calls == []
    assert eph_p.read_text(encoding="utf-8") == '{"v":2}'

    # 2. Heartbeat, CacheShortTrendManager, 24trend/live_*, KalmanStateTracker, and priceAnalysis all skip fsync
    hb_p = tmp_path / "daemon.heartbeat"
    botcore.touch_heartbeat_file(hb_p, force=True, now=100.0, status={"ok": True})
    assert fsync_calls == [] and "\n" not in hb_p.read_text(encoding="utf-8")

    st_file = tmp_path / "cache_short_trend.json"
    stm = cachetrend.CacheShortTrendManager(symbols=["BTCUSDC"], filename=str(st_file), read_only=False)
    stm._mem["BTCUSDC"] = {"ts": time.time(), "slope": 0.12, "current_price": 60000.0}
    stm._live_windows["BTCUSDC"] = {"222": {"ts": time.time(), "slope": 0.12}}
    stm._dirty_live_symbols.add("BTCUSDC")
    assert stm.save_state_to_file() is True
    assert fsync_calls == []
    assert "\n" not in st_file.read_text(encoding="utf-8")

    kal_file = tmp_path / "cache_kalman_trend.json"
    kt = cachetrend.KalmanStateTracker(state_path=str(kal_file), read_only=False, state_min_interval=0.0)
    kt._state["BTCUSDC"] = {"ts": time.time(), "kalman_trend": 1.0}
    kt._write_state(time.time())
    assert fsync_calls == [] and kal_file.exists()

    pa_file = tmp_path / "priceanalysis.json"
    pa.write_all_trends({"BTCUSDC": {"direction": "up", "start_timestamp": 1000, "duration_seconds": 60, "estimated_future_hours": 10}}, filename=str(pa_file))
    assert fsync_calls == [] and json.loads(pa_file.read_text(encoding="utf-8"))["BTCUSDC"]["direction"] == "up"

    # 3. read_jsonl_or_json_dict_cached tails appended JSONL lines incrementally without mutating prior snapshots
    c24_p = tmp_path / "cache_24_price_BTCUSDC.jsonl"
    now_ms = int(time.time() * 1000)
    lines0 = [json.dumps({"s": "BTCUSDC", "i": [now_ms - (20 - k) * 1000, 60000.0 + k]}, separators=(",", ":")) for k in range(10)]
    c24_p.write_text("\n".join(lines0) + "\n", encoding="utf-8")
    memo: dict = {}
    tail_events: list[bool] = []
    orig_off = state_io.read_jsonl_bucket_map_from_offset

    def _spy_off(*args, **kwargs):
        res = orig_off(*args, **kwargs)
        tail_events.append(res[5])
        return res

    monkeypatch.setattr(state_io, "read_jsonl_bucket_map_from_offset", _spy_off)
    d1 = state_io.read_jsonl_or_json_dict_cached(str(c24_p), memo)
    assert d1 is not None and len(d1["items"]["BTCUSDC"]) == 10 and tail_events == [False]
    snap1_list = d1["items"]["BTCUSDC"]

    # Append 2 lines on same inode -> incremental tail (True) and prior list reference stays unmutated
    with open(c24_p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"s": "BTCUSDC", "i": [now_ms - 5000, 60100.0]}, separators=(",", ":")) + "\n")
        fh.write(json.dumps({"s": "BTCUSDC", "i": [now_ms - 4000, 60101.0]}, separators=(",", ":")) + "\n")
    d2 = state_io.read_jsonl_or_json_dict_cached(str(c24_p), memo)
    assert d2 is not None and len(d2["items"]["BTCUSDC"]) == 12 and tail_events == [False, True]
    assert len(snap1_list) == 10

    # Atomic replace (new inode, e.g. compaction) -> clean full reload (False)
    state_io.atomic_write_text(
        str(c24_p),
        json.dumps({"s": "BTCUSDC", "i": [now_ms - 1000, 60200.0]}, separators=(",", ":")) + "\n",
    )
    d3 = state_io.read_jsonl_or_json_dict_cached(str(c24_p), memo)
    assert d3 is not None and len(d3["items"]["BTCUSDC"]) == 1 and tail_events[-1] is False

    # 4. Cache365PriceManager (read_only=True) tails newly appended lines incrementally & falls back on inode swap
    sp_p = tmp_path / "cache_sparse_price_BTCUSDC.jsonl"
    sp_p.write_text("\n".join(lines0) + "\n", encoding="utf-8")
    monkeypatch.setattr(cacheprice, "read_jsonl_bucket_map_from_offset", _spy_off)
    sp_ro = cacheprice.Cache365PriceManager(180, ["BTCUSDC"], str(sp_p), api_client=None, read_only=True)
    assert len(sp_ro.cache["BTCUSDC"]) == 10
    with open(sp_p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"s": "BTCUSDC", "i": [now_ms - 2000, 60300.0]}, separators=(",", ":")) + "\n")
    tail_events.clear()
    sp_ro._load_jsonl()
    assert tail_events == [True] and len(sp_ro.cache["BTCUSDC"]) == 11 and sp_ro.cache["BTCUSDC"][-1][1] == pytest.approx(60300.0)

    # Atomic replace (rotation/new inode) -> falls back to read_tail_lines_with_cursor cleanly
    state_io.atomic_write_text(
        str(sp_p),
        json.dumps({"s": "BTCUSDC", "i": [now_ms - 500, 60400.0]}, separators=(",", ":")) + "\n",
    )
    tail_events.clear()
    sp_ro._load_jsonl()
    assert tail_events == [False] and len(sp_ro.cache["BTCUSDC"]) == 1 and sp_ro.cache["BTCUSDC"][0][1] == pytest.approx(60400.0)

    # 5. macro_analyzer.py & DEV offline backtests run with nice/ionice while DEV CI watcher stays normal priority
    repo_root = Path(__file__).resolve().parents[1]
    macro_row = next(ln for ln in (repo_root / "procs.conf").read_text(encoding="utf-8").splitlines() if ln.startswith("macro_analyzer.py|"))
    assert "nice -n 5 ionice -c3 python intelligence/macro_analyzer.py" in macro_row
    dev_cron = (repo_root / "systemd/crontab.dev.txt").read_text(encoding="utf-8")
    for tok in ("run_backtest_cycle.sh", "run_overnight_master_suite.sh", "run_dev_observability.sh"):
        ln = next(line for line in dev_cron.splitlines() if tok in line and not line.strip().startswith("#"))
        assert "nice -n 15 ionice -c3" in ln
    ci_ln = next(line for line in dev_cron.splitlines() if "run_dev_regression.sh watch" in line and not line.strip().startswith("#"))
    assert "nice -n" not in ci_ln


def test_order_ledger_and_rtrade_pair_store_skip_redundant_noop_fsync_writes(tmp_path, monkeypatch):
    """Verify OrderLedgerReconciler and RTradePairStore skip redundant disk/fsync writes when state is unchanged
    while preserving immediate durable fsync writes on real state transitions."""
    import order_ledger_reconciler_worker as olw
    import rtrade_pair_store

    # 1. OrderLedgerReconciler writes on initial creation and on real changes, skipping no-op polls
    cdb = tmp_path / "cachedb"
    (cdb / "orders").mkdir(parents=True, exist_ok=True)
    rec = olw.OrderLedgerReconciler(cachedb_dir=str(cdb), use_singletons=False)
    write_calls: list[dict] = []
    orig_write = rec.write_active_snapshot

    def _spy_write(snap):
        write_calls.append(dict(snap))
        return orig_write(snap)

    monkeypatch.setattr(rec, "write_active_snapshot", _spy_write)
    rec.reconcile_once(now_ts=1_795_000_000.0)
    assert len(write_calls) == 1

    # Second identical no-op pass (different wall clock now_ts, zero events/venue changes) -> 0 writes!
    rec.reconcile_once(now_ts=1_795_000_002.0)
    assert len(write_calls) == 1

    # Real event appended -> triggers immediate durable write!
    rec.append_event(
        olw.build_order_event(
            olw.OrderEventType.INTENT.value,
            ts=1_795_000_004.0,
            src="test",
            venue="binance",
            symbol="BTCUSDC",
            owner="tradeall",
            side="BUY",
            order_type="LIMIT",
            qty=0.01,
            px=60000.0,
            iid="iid_dirty_test_1",
        )
    )
    rec.reconcile_once(now_ts=1_795_000_005.0)
    assert len(write_calls) == 2

    # Subsequent identical pass -> 0 writes again!
    rec.reconcile_once(now_ts=1_795_000_007.0)
    assert len(write_calls) == 2

    # 2. RTradePairStore.checkpoint_many skips fsync write when state/phase/terminal is unchanged
    store_path = tmp_path / "rtrade_pairs.json"
    store = rtrade_pair_store.RTradePairStore(str(store_path))
    store.intent("p1", "BUY", 100.0, 1.0, "CID_P1", symbol="TAOUSDC", start_side="BUY")
    st_quoting = {"pair_id": "p1", "qty": 1.0, "start_side": "BUY", "phase": "quoting", "tickets": []}

    fsync_calls: list[int] = []
    monkeypatch.setattr(os, "fsync", lambda fd: fsync_calls.append(fd))

    store.checkpoint_many([("p1", st_quoting, False)])
    assert len(fsync_calls) >= 1

    fsync_calls.clear()
    store.checkpoint_many([("p1", dict(st_quoting), False)])
    assert fsync_calls == []

    # Phase transition -> immediate durable fsync write
    store.checkpoint_many([("p1", {"pair_id": "p1", "phase": "complete"}, True)])
    assert len(fsync_calls) >= 1


def test_adversarial_audit_bugfixes_regression(tmp_path, monkeypatch):
    """Verify all 5 edge-case bugs caught during full-thread audit:
    1) read_jsonl_or_json_dict_cached cursor object-identity & thread-safety;
    2) read_jsonl_bucket_map_from_offset partial EOF line on start_offset=0;
    3) WS stale watchdog triggers even when server heartbeats prevent TimeoutError;
    4) OrderLedgerReconciler heals corrupt orders_active.json without .previous;
    5) priceAnalysis caches None trend outcomes when SparsePrice signature is unchanged."""
    import asyncio
    import threading
    import binance_api.bapi_ws as bapi_ws
    import order_ledger_reconciler_worker as olw
    import priceAnalysis as pa
    import providers.price_poller as pp

    # 1. Cursor object-id mismatch cannot drop rows + concurrent readers/writer stay consistent
    j_path = tmp_path / "race_24h.jsonl"
    now_ms = int(time.time() * 1000)
    j_path.write_text(
        "\n".join(json.dumps({"s": "BTCUSDC", "i": [now_ms - (20 - k) * 1000, 60000.0 + k]}, separators=(",", ":")) for k in range(5)) + "\n",
        encoding="utf-8",
    )
    memo: dict = {}
    first = state_io.read_jsonl_or_json_dict_cached(str(j_path), memo)
    assert first is not None and len(first["items"]["BTCUSDC"]) == 5
    with open(j_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"s": "BTCUSDC", "i": [now_ms - 5000, 60100.0]}, separators=(",", ":")) + "\n")
    # Simulate stale memo dict object paired against newer cursor object id
    resolved = os.path.abspath(str(j_path))
    ck = (id(memo), resolved)
    old_sig = memo[resolved][0]
    memo[resolved] = (old_sig, {"items": {"BTCUSDC": [[now_ms - 99_000, 1.0]]}})
    second = state_io.read_jsonl_or_json_dict_cached(str(j_path), memo)
    assert second is not None and len(second["items"]["BTCUSDC"]) == 6
    assert second["items"]["BTCUSDC"][-1][1] == pytest.approx(60100.0)
    assert state_io._JSONL_MEMO_ENTRIES[ck].doc is second

    # Multi-threaded concurrent read + append check
    errs: list[Exception] = []
    shared_memo: dict = {}

    def _reader():
        try:
            for _ in range(25):
                doc = state_io.read_jsonl_or_json_dict_cached(str(j_path), shared_memo)
                assert doc is not None and len(doc["items"]["BTCUSDC"]) >= 6
        except Exception as exc:
            errs.append(exc)

    threads = [threading.Thread(target=_reader) for _ in range(4)]
    for t in threads:
        t.start()
    for step in range(5):
        with open(j_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"s": "BTCUSDC", "i": [now_ms - 4000 + step * 100, 60200.0 + step]}, separators=(",", ":")) + "\n")
    for t in threads:
        t.join()
    assert errs == []
    final_doc = state_io.read_jsonl_or_json_dict_cached(str(j_path), shared_memo)
    assert final_doc is not None and len(final_doc["items"]["BTCUSDC"]) == 11

    # 2. Partial unterminated EOF line on initial load (start_offset=0) keeps cursor BEFORE partial line
    part_p = tmp_path / "partial_init.jsonl"
    line1 = json.dumps({"s": "BTCUSDC", "i": [now_ms - 3000, 61000.0]}, separators=(",", ":")) + "\n"
    line2 = json.dumps({"s": "BTCUSDC", "i": [now_ms - 2000, 61001.0]}, separators=(",", ":")) + "\n"
    line3 = json.dumps({"s": "BTCUSDC", "i": [now_ms - 1000, 61002.0]}, separators=(",", ":"))
    part_p.write_bytes((line1 + line2 + line3[:12]).encode("utf-8"))
    bmap0, bad0, off0, dev_ino0, anc0, inc0 = state_io.read_jsonl_bucket_map_from_offset(str(part_p), start_offset=0)
    assert bad0 is True and inc0 is False and len(bmap0["BTCUSDC"]) == 2
    assert off0 == len((line1 + line2).encode("utf-8"))
    with open(part_p, "ab") as fh:
        fh.write((line3[12:] + "\n").encode("utf-8"))
    bmap1, bad1, _off1, _di1, _anc1, inc1 = state_io.read_jsonl_bucket_map_from_offset(
        str(part_p), start_offset=off0, expected_dev_ino=dev_ino0, expected_anchor=anc0
    )
    assert bad1 is False and inc1 is True and bmap1["BTCUSDC"] == [[now_ms - 1000, 61002.0]]

    # 3. WS stale watchdog breaks out even when heartbeats keep ws.recv() returning without TimeoutError
    mono_clock = [1000.0]
    monkeypatch.setattr(bapi_ws.time, "monotonic", lambda: mono_clock[0])
    monkeypatch.setattr(pp.time, "monotonic", lambda: mono_clock[0])
    import providers.ws_stream as ws_stream
    monkeypatch.setattr(ws_stream.time, "monotonic", lambda: mono_clock[0])

    class _HeartbeatOnlyWs:
        def __init__(self):
            self.recv_count = 0

        async def recv(self):
            self.recv_count += 1
            mono_clock[0] += 40.0
            return '{"channel":"heartbeat"}'

    bstream = bapi_ws.BinancePriceStream(symbols=["BTCUSDC"], stale_reconnect_sec=100.0)
    bws = _HeartbeatOnlyWs()
    asyncio.run(bstream._recv_loop(bws))
    assert bws.recv_count == 3  # 1000 -> 1040 -> 1080 -> 1120 (>= 100s stale!) breaks without TimeoutError

    # 4. OrderLedgerReconciler heals corrupt orders_active.json with no .previous
    cdb = tmp_path / "cachedb_corrupt"
    (cdb / "orders").mkdir(parents=True, exist_ok=True)
    corrupt_active = cdb / "orders" / "orders_active.json"
    corrupt_active.write_text("{corrupt_json", encoding="utf-8")
    rec = olw.OrderLedgerReconciler(cachedb_dir=str(cdb), use_singletons=False)
    rec.reconcile_once(now_ts=1_795_000_500.0)
    healed = json.loads(corrupt_active.read_text(encoding="utf-8"))
    assert healed["updated_ts"] == pytest.approx(1_795_000_500.0)

    # 5. priceAnalysis caches None trend outcomes (e.g. off-hours equities) when SparsePrice signature is unchanged
    sig_val = [(50, 1_760_000_000_000, 10.0)]
    none_calls: list[str] = []
    monkeypatch.setattr(pa, "_series_signature_for", lambda _s: sig_val[0])
    monkeypatch.setattr(pa, "getTrendLongTerm_fixed", lambda s, **_kw: (none_calls.append(s), None)[1])
    st: dict = {}
    out1 = pa.run_price_analysis_cycle(["RGNT_US_EQ"], st, now_ts=1000.0, inter_symbol_yield_sec=0.0)
    out2 = pa.run_price_analysis_cycle(["RGNT_US_EQ"], st, now_ts=1060.0, inter_symbol_yield_sec=0.0)
    assert out1 == {"RGNT_US_EQ": None} and out2 == {"RGNT_US_EQ": None}
    assert none_calls == ["RGNT_US_EQ"]  # skipped recompute on cycle 2!
    pa.run_price_analysis_cycle(["RGNT_US_EQ"], st, now_ts=2000.0, inter_symbol_yield_sec=0.0)
    assert len(none_calls) == 2  # safety refresh after >=900s!


def test_harvest_hotpath_optimizations_parity(tmp_path, monkeypatch):
    """Verify exact parity & edge cases for autonomous_ai_reconciler harvest hotpath optimizations:
    1) state_io read_jsonl_bucket_map / read_jsonl_bucket_map_from_offset (orjson bytes + symbol cache + NaN + partial EOF);
    2) pricewindow.filter_isolated_price_points NumPy (>=256) vs pure-Python bit-exact parity;
    3) price_series_evaluator VWAP/RVOL/OHLC/sort_ts_rows/normalize_and_dedup_24h_series;
    4) portfolio_evaluator split-fill AI intent attribution without O(M*N) scans."""
    import math
    import pricewindow as pw
    import intelligence.macro_sentiment_ssot as mss
    import intelligence.sentiment.evaluators.price_series_evaluator as pse
    import intelligence.sentiment.evaluators.portfolio_evaluator as pe

    # 1. Fast JSONL bucket map reader: repeated symbol, whitespace alias, symbol switch, blank/leading-ws, NaN, bad & partial EOF
    jf = tmp_path / "fast_bucket.jsonl"
    now_ms = 1_795_000_000_000
    lines = [
        json.dumps({"s": "BTCUSDC", "i": [now_ms, 90000.0]}, separators=(",", ":")) + "\n",
        json.dumps({"s": "BTCUSDC", "i": [now_ms + 1000, 90001.0]}, separators=(",", ":")) + "\n",
        "   \n",
        "  " + json.dumps({"s": " BTCUSDC ", "i": [now_ms + 2000, 90002.0]}, separators=(",", ":")) + "\n",
        json.dumps({"s": "ETHUSDC", "i": [now_ms + 3000, 3000.0]}, separators=(",", ":")) + "\n",
        '{"s":"ETHUSDC","i":[1795000004000,NaN]}\n',
        "{not_valid_json\n",
    ]
    tail_full = json.dumps({"s": "BTCUSDC", "i": [now_ms + 5000, 90005.0]}, separators=(",", ":"))
    jf.write_bytes(("".join(lines) + tail_full[:10]).encode("utf-8"))

    bmap, had_bad, off, dev_ino, anc, inc = state_io.read_jsonl_bucket_map_from_offset(str(jf), start_offset=0)
    assert had_bad is True and inc is False
    assert len(bmap["BTCUSDC"]) == 3 and bmap["BTCUSDC"][-1] == [now_ms + 2000, 90002.0]
    assert len(bmap["ETHUSDC"]) == 2 and math.isnan(bmap["ETHUSDC"][1][1])
    assert off == len("".join(lines).encode("utf-8"))

    with open(jf, "ab") as fh:
        fh.write((tail_full[10:] + "\n").encode("utf-8"))
    bmap_inc, bad_inc, _off2, _di2, _anc2, inc2 = state_io.read_jsonl_bucket_map_from_offset(
        str(jf), start_offset=off, expected_dev_ino=dev_ino, expected_anchor=anc
    )
    assert bad_inc is False and inc2 is True
    assert bmap_inc == {"BTCUSDC": [[now_ms + 5000, 90005.0]]}

    full_map, full_bad = state_io.read_jsonl_bucket_map(str(jf))
    assert full_bad is True and len(full_map["BTCUSDC"]) == 4 and len(full_map["ETHUSDC"]) == 2

    # 2. filter_isolated_price_points NumPy path (>=256) matches pure-Python path bit-for-bit
    pts_big = [[now_ms + k * 1000, 100.0 + (k % 5) * 0.1] for k in range(320)]
    pts_big[0][1] = 500.0          # head outlier
    pts_big[20][1] = 500.0         # 1-tick high spike
    pts_big[40][1] = 10.0          # 1-tick low spike
    pts_big[80][1] = 400.0         # 2-tick burst
    pts_big[81][1] = 400.0
    pts_big[120][1] = 400.0        # 3-tick overlapping burst check
    pts_big[121][1] = 400.0
    pts_big[122][1] = 400.0
    pts_big[160][1] = 400.0        # 1-tick spike followed by moderate value
    pts_big[161][1] = 101.0
    pts_big[-1][1] = 10.0          # tail outlier
    res_np = pw.filter_isolated_price_points(pts_big)
    monkeypatch.setattr(pw, "np", None)
    res_py = pw.filter_isolated_price_points(pts_big)
    monkeypatch.undo()
    assert res_np == res_py
    kept_ts = {p[0] for p in res_np}
    for bad_idx in (0, 20, 40, 80, 81, 160, 319):
        assert pts_big[bad_idx][0] not in kept_ts

    # 3. sort_ts_rows + normalize_and_dedup_24h_series + VWAP/RVOL/OHLC 2-col fast paths & mixed tie-breaker
    raw_unsorted = [
        [now_ms + 1000, 100.0, 5.0],
        [now_ms + 1000, 101.0],
        [now_ms, 99.0],
        [now_ms + 2000, 102.0],
    ]
    norm_u = pse.normalize_and_dedup_24h_series(raw_unsorted, "BTCUSDC")
    assert norm_u == [[float(now_ms), 99.0], [float(now_ms + 1000), 101.0, 5.0], [float(now_ms + 2000), 102.0]]

    ser_2col = [[now_ms + k * 60_000, 100.0 + k * 0.1] for k in range(30)]
    norm_2col = pse.normalize_and_dedup_24h_series(ser_2col, "BTCUSDC")
    clean_pts = mss._extract_sorted_ts_px(norm_2col)
    assert mss._TWO_COL_TS_PX_IDS.get(id(clean_pts)) is clean_pts
    assert pse.compute_session_vwap_metrics(norm_2col, precleaned_ts_px=clean_pts) == {}
    assert pse.compute_volume_rvol_metrics(norm_2col, precleaned_ts_px=clean_pts) == {}
    bars_fast = pse._resample_ohlc_bars(norm_2col, 900_000.0, precleaned_ts_px=clean_pts)
    bars_slow = pse._resample_ohlc_bars(norm_2col, 900_000.0)
    assert bars_fast == bars_slow and len(bars_fast) >= 2

    # Mixed 2-col and 3-col VWAP: later 2-col row at same max timestamp still determines last_any_px
    day0_ms = (now_ms // 86_400_000) * 86_400_000 + 3_600_000
    mixed_vwap = [
        [day0_ms, 100.0, 10.0],
        [day0_ms + 60_000, 110.0, 10.0],
        [day0_ms + 60_000, 120.0],  # tied max ts, 2-col -> eff_px must be 120.0 when cur_px=None
    ]
    vw = pse.compute_session_vwap_metrics(mixed_vwap, is_equity=False, cur_px=None)
    assert vw["session_vwap_px"] == pytest.approx(105.0)
    assert vw["dist_vs_session_vwap_pct"] == pytest.approx(round(((120.0 / 105.0) - 1.0) * 100.0, 2))

    # 4. portfolio_evaluator split-fill attribution still tags unattributed child fills as AI
    t_base_s = 1_795_000_100.0
    orders = [
        {"symbol": "BTCUSDC", "orderId": "F1", "side": "BUY", "price": 90000.0, "executedQty": 0.4, "timestamp": int((t_base_s + 5) * 1000), "status": "FILLED"},
        {"symbol": "BTCUSDC", "orderId": "F2", "side": "BUY", "price": 90010.0, "executedQty": 0.6, "timestamp": int((t_base_s + 10) * 1000), "status": "FILLED"},
    ]
    intents = [
        {
            "intent_id": "int_split_1",
            "status": "EXECUTED",
            "symbol": "BTCUSDC",
            "decision": "BUY",
            "ts": t_base_s,
            "execution_payload": {"origQty": 1.0, "status": "NEW"},
        }
    ]
    fills_res = pe.extract_symbol_recent_fills(orders, ai_intents=intents, symbol="BTCUSDC")
    assert fills_res is not None
    assert fills_res["last_action_actor"] == "AI"
    assert all(b["actor"] == "AI" for b in fills_res["recent_buys"])


def test_assetguardian_p3b_subdomain_and_jsonl_generation_lock_ohlc_merge(tmp_path, monkeypatch):
    """Verify 3 live bugfixes:
    1) AssetGuardian._read_p3b_sentiment resolves cachedb/macro/sentiment_eval.json when STATE.path is under cachedb/state/;
    2) CacheManagerInterface._save_jsonl_append & compact_jsonl serialize under _generation_lock();
    3) Cache24PriceManager (.jsonl) preserves externally persisted 3-col OHLCV bars when live ticks also advance."""
    import contextlib
    import assetguardian as ag
    import cacheprice

    # 1. AssetGuardian P3B reads cachedb/macro/sentiment_eval.json even when STATE.path lives in cachedb/state/
    cdb = tmp_path / "cachedb"
    (cdb / "state").mkdir(parents=True)
    (cdb / "macro").mkdir(parents=True)
    now_s = time.time()
    (cdb / "macro" / "sentiment_eval.json").write_text(
        json.dumps(
            {
                "market_bias": "BEARISH",
                "risk_level": "HIGH",
                "recommended_action": "DEFENSIVE",
                "confidence": 0.85,
                "ts": now_s - 60.0,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(ag, "STATE", ag.AssetGuardianState(path=str(cdb / "state" / "assetguardian_state.json")))
    p3b = ag._read_p3b_sentiment(now_ts=now_s)
    assert p3b is not None
    assert p3b["recommended_action"] == "DEFENSIVE"
    assert p3b["market_bias"] == "BEARISH"
    assert p3b["confidence"] == pytest.approx(0.85)
    sell_mult, buy_mult, ass = ag._p3b_threshold_multipliers(assessment=p3b, bypass_guards=set())
    assert ass is not None and sell_mult < 1.0 and buy_mult > 1.0

    # 2 & 3. Cache24PriceManager (.jsonl) uses _generation_lock on append+compact and merges external 3-col OHLCV bars
    (cdb / "prices").mkdir(parents=True)
    p24 = cdb / "prices" / "cache_24_price_NVDA_US_EQ.jsonl"
    now_ms = int(now_s * 1000)
    mgr = cacheprice.Cache24PriceManager(
        30,
        ["NVDA_US_EQ"],
        str(p24),
        api_client=None,
        read_only=False,
        append_persist=True,
    )
    mgr.cache["NVDA_US_EQ"] = [[now_ms - 600_000, 140.0]]
    mgr.fetchtime_per_symbol["NVDA_US_EQ"] = now_ms - 600_000
    mgr._rest_sync_cursor_per_symbol["NVDA_US_EQ"] = now_ms - 600_000

    lock_events: list[str] = []
    orig_gen_lock = mgr._generation_lock

    @contextlib.contextmanager
    def _tracking_lock():
        lock_events.append("enter")
        with orig_gen_lock():
            yield
        lock_events.append("exit")

    monkeypatch.setattr(mgr, "_generation_lock", _tracking_lock)
    assert mgr.save_state_to_file() is True
    assert lock_events.count("enter") >= 1 and lock_events.count("enter") == lock_events.count("exit")

    # External writer persists 3-col OHLCV bars with newer timestamp than last_disk_sync_max_ts
    ohlc_pts = [
        [now_ms - 600_000, 140.0, 1500.0, 141.0, 139.5],
        [now_ms - 300_000, 141.5, 2200.0, 142.0, 140.5],
    ]
    cacheprice.Cache24PriceManager.persist_external_ohlc_series(
        str(p24), "NVDA_US_EQ", ohlc_pts, now_ts=(now_ms - 100_000) / 1000.0
    )

    # Concurrent live tick arrives in memory (newer than external write)
    mgr.cache["NVDA_US_EQ"].append([now_ms, 142.2])
    mgr.fetchtime_per_symbol["NVDA_US_EQ"] = now_ms
    lock_events.clear()
    assert mgr.save_state_to_file() is True
    assert lock_events.count("enter") >= 2 and lock_events.count("enter") == lock_events.count("exit")

    # Verify disk JSONL retained BOTH the externally written 3-col OHLCV bars AND the new live tick!
    reloaded, had_bad = state_io.read_jsonl_bucket_map(str(p24))
    assert had_bad is False
    rows = reloaded["NVDA_US_EQ"]
    assert any(len(r) >= 3 and r[2] == pytest.approx(2200.0) for r in rows), f"lost OHLCV bars: {rows}"
    assert rows[-1] == [now_ms, 142.2]


def test_cacheprice_subdomain_anchor_sparse_fallback_and_seed_guard(tmp_path, monkeypatch):
    """Verify 3 cacheprice subdomain-resolution bug fixes:
    1) CacheCurrentPriceManager.__init__ resolves filename before checking file_existed_before_init,
       seeding _price_anchor_confirmations=2 on restart so the very first outlier tick is rejected;
    2) resolve_cached_symbol_price resolves cachedb/prices/ for sparse & 365d JSONL fallbacks;
    3) Cache365PriceManager.seed_initial_archive_if_missing resolves target_path before exists check
       so an unresolved cachedb/<file> path never clobbers an existing cachedb/prices/ archive."""
    import cacheprice

    cdb = tmp_path / "cachedb"
    prices_dir = cdb / "prices"
    prices_dir.mkdir(parents=True)
    monkeypatch.setenv("MPTRADE_CACHEDB_DIR", str(cdb))

    # 1. Restart outlier-anchor seeding with bare DEFAULT_FILENAME ("cache_current_price.json")
    now_ms = int(time.time() * 1000)
    cp_file = prices_dir / "cache_current_price.json"
    cp_file.write_text(
        json.dumps({"items": {"BTCUSDC": [[now_ms - 2000, 60000.0]]}, "fetchtime": {"BTCUSDC": now_ms - 2000}}),
        encoding="utf-8",
    )
    cpm = cacheprice.CacheCurrentPriceManager(
        sync_ts=60,
        symbols=["BTCUSDC"],
        filename=cacheprice.CacheCurrentPriceManager.DEFAULT_FILENAME,
        read_only=True,
        data_sync_mode="live_reader",
    )
    assert cpm._price_anchor_confirmations.get("BTCUSDC", 0) >= 2
    # First corrupt cross-symbol outlier tick after restart must be rejected immediately!
    cpm.push_price("BTCUSDC", 0.15, ts_ms=now_ms)
    assert cpm.extract_price_and_ts(cpm.cache.get("BTCUSDC"))[0] == pytest.approx(60000.0)

    # 2. resolve_cached_symbol_price falls back to cachedb/prices/cache_sparse_price_<SYM>.jsonl
    sparse_file = prices_dir / "cache_sparse_price_NVDA_US_EQ.jsonl"
    sparse_file.write_text(
        json.dumps({"s": "NVDA_US_EQ", "i": [now_ms - 3_600_000, 145.75]}) + "\n",
        encoding="utf-8",
    )
    px_sparse = cacheprice.resolve_cached_symbol_price("NVDA_US_EQ", cache_dir=str(cdb))
    assert px_sparse == pytest.approx(145.75)

    # And falls back to cachedb/prices/cache_365_price_<SYM>.jsonl when sparse is absent
    p365_file = prices_dir / "cache_365_price_AMD_US_EQ.jsonl"
    p365_file.write_text(
        json.dumps({"s": "AMD_US_EQ", "i": [now_ms - 7_200_000, 118.25]}) + "\n",
        encoding="utf-8",
    )
    px_365 = cacheprice.resolve_cached_symbol_price("AMD_US_EQ", cache_dir=str(cdb))
    assert px_365 == pytest.approx(118.25)

    # 3. seed_initial_archive_if_missing with unresolved cachedb/ path must NOT overwrite existing archive
    unresolved_p365 = str(cdb / "cache_365_price_AMD_US_EQ.jsonl")
    assert cacheprice.Cache365PriceManager.seed_initial_archive_if_missing(
        unresolved_p365, "AMD_US_EQ", [[now_ms, 999.0]]
    ) is False
    reloaded_365, bad = state_io.read_jsonl_bucket_map(str(p365_file))
    assert bad is False
    assert reloaded_365["AMD_US_EQ"] == [[now_ms - 7_200_000, 118.25]]


def test_compact_price_series_and_bounded_sparse_cache_parity(tmp_path, monkeypatch):
    """Verify A+B design for CacheSparsePriceManager:
    1) CompactPriceSeries stores int64+float64 (16 B/pt raw), supports list-like API + equality,
       slices compactly, and releases frombuffer views before returning so subsequent appends never BufferError;
    2) CacheSparsePriceManager loads only the bounded tail into CompactPriceSeries while preserving
       100% of the full JSONL archive on disk across appends, trims, compact_jsonl, and maintain_append_persist;
    3) Read-only reader tails incrementally into CompactPriceSeries and serves get_series_arrays() cleanly;
    4) priceAnalysis.getTrendLongTerm_fixed() fast-path over get_series_arrays() matches legacy tuple path."""
    import sys
    import numpy as np
    import cacheprice
    import priceAnalysis as pa

    # 1. CompactPriceSeries contract, ~16 B/pt raw storage, and zero BufferError after to_numpy()
    cps = cacheprice.CompactPriceSeries()
    base_ms = int((time.time() - 20 * 86400) * 1000)
    for k in range(2000):
        cps.append([base_ms + k * 60_000, 50000.0 + k * 0.5])
    assert len(cps) == 2000
    assert cps[0] == [base_ms, 50000.0]
    assert cps[-1] == [base_ms + 1999 * 60_000, 50000.0 + 1999 * 0.5]
    assert cps.last_pair() == (base_ms + 1999 * 60_000, 50000.0 + 1999 * 0.5)
    assert cps.max_timestamp_ms() == base_ms + 1999 * 60_000
    raw_buf_bytes = sys.getsizeof(cps._ts) + sys.getsizeof(cps._px)
    assert raw_buf_bytes / len(cps) < 28.0  # ~16 B/pt + small C array growth headroom (vs ~153 B/pt Python list)
    sliced = cps[-5:]
    assert isinstance(sliced, cacheprice.CompactPriceSeries)
    assert sliced == [[base_ms + k * 60_000, 50000.0 + k * 0.5] for k in range(1995, 2000)]

    ts_arr, px_arr = cps.to_numpy(since_ms=base_ms + 1990 * 60_000)
    assert ts_arr.dtype == np.int64 and px_arr.dtype == np.float64
    assert len(ts_arr) == 10 and px_arr[0] == pytest.approx(50000.0 + 1990 * 0.5)
    # Subsequent append/resize must NOT raise BufferError
    cps.append([base_ms + 2000 * 60_000, 51000.0])
    assert len(cps) == 2001 and len(ts_arr) == 10  # returned array is independent

    # Non-monotonic to_numpy fallback
    cps.append([base_ms + 1000 * 60_000, 50500.0])
    assert cps._monotonic is False
    ts_nm, px_nm = cps.to_numpy(since_ms=base_ms + 1999 * 60_000)
    assert len(ts_nm) == 2 and float(px_nm[-1]) == pytest.approx(51000.0)

    # 2. CacheSparsePriceManager bounded memory tail + full disk preservation
    cdb = tmp_path / "cachedb"
    prices_dir = cdb / "prices"
    prices_dir.mkdir(parents=True)
    sp_path = prices_dir / "cache_sparse_price_BTCUSDC.jsonl"
    now_ms = int(time.time() * 1000)
    # Write 40 rows spanning last 20 days (12h step) on disk; cap memory at 12 rows
    total_disk_rows = 40
    step_ms = 12 * 3600 * 1000
    start_ms = now_ms - (total_disk_rows + 2) * step_ms
    with open(sp_path, "w", encoding="utf-8") as fh:
        for i in range(total_disk_rows):
            fh.write(json.dumps({"s": "BTCUSDC", "i": [start_ms + i * step_ms, 50000.0 + i * 250.0]}) + "\n")

    monkeypatch.setattr(cacheprice.CacheSparsePriceManager, "SPARSE_MEMORY_ROWS", 12)
    writer = cacheprice.CacheSparsePriceManager(
        180, ["BTCUSDC"], str(sp_path), api_client=None, read_only=False, data_sync_mode="producer"
    )
    assert isinstance(writer.cache["BTCUSDC"], cacheprice.CompactPriceSeries)
    assert len(writer.cache["BTCUSDC"]) == 12
    assert writer.cache["BTCUSDC"][0][0] == start_ms + (total_disk_rows - 12) * step_ms

    # Append 2 new live ticks, save, and verify disk keeps ALL 42 rows while memory stays capped at 12
    t_new1 = start_ms + total_disk_rows * step_ms
    t_new2 = start_ms + (total_disk_rows + 1) * step_ms
    writer.update_cache_per_symbol("BTCUSDC", [[t_new1, 60000.0]])
    writer.update_cache_per_symbol("BTCUSDC", [[t_new2, 60500.0]])
    assert len(writer.cache["BTCUSDC"]) == 12
    assert writer.save_state_to_file() is True

    # compact_jsonl on an existing archive must preserve ALL 42 disk rows (never truncate to 12!)
    assert writer.compact_jsonl() is True
    full_disk, had_bad = state_io.read_jsonl_bucket_map(str(sp_path))
    assert had_bad is False
    assert len(full_disk["BTCUSDC"]) == total_disk_rows + 2
    assert full_disk["BTCUSDC"][0][0] == start_ms
    assert full_disk["BTCUSDC"][-1] == [t_new2, 60500.0]

    # 3. Read-only reader tails incrementally into CompactPriceSeries & serves get_series_arrays
    reader = cacheprice.CacheSparsePriceManager(
        180, ["BTCUSDC"], str(sp_path), api_client=None, read_only=True, data_sync_mode="disk_reader"
    )
    assert isinstance(reader.cache["BTCUSDC"], cacheprice.CompactPriceSeries)
    assert len(reader.cache["BTCUSDC"]) == 12
    assert reader._jsonl_tail_state is not None

    t_new3 = start_ms + (total_disk_rows + 2) * step_ms
    writer.update_cache_per_symbol("BTCUSDC", [[t_new3, 61000.0]])
    assert writer.save_state_to_file() is True
    ts_r, px_r = reader.get_series_arrays("BTCUSDC", resync=True)
    assert len(ts_r) == 12 and int(ts_r[-1]) == t_new3 and float(px_r[-1]) == pytest.approx(61000.0)

    # 4. priceAnalysis _series_signature_for + getTrendLongTerm_fixed fast-path parity vs legacy priceLstFor
    monkeypatch.setattr(cacheprice.CacheSparsePriceManager, "SPARSE_MEMORY_ROWS", 35_000)
    sp_trend_path = prices_dir / "cache_sparse_price_ETHUSDC.jsonl"
    # Build 20 days of 1-hour rising points (480 pts)
    n_pts = 480
    t0_ms = now_ms - n_pts * 3600 * 1000
    with open(sp_trend_path, "w", encoding="utf-8") as fh:
        for i in range(n_pts):
            fh.write(json.dumps({"s": "ETHUSDC", "i": [t0_ms + i * 3600 * 1000, 2000.0 + i * 2.5]}) + "\n")
    pa_mgr = cacheprice.CacheSparsePriceManager(
        180, ["ETHUSDC"], str(sp_trend_path), api_client=None, read_only=True, data_sync_mode="disk_reader"
    )
    monkeypatch.setattr(pa, "price_cache_manager", {"ETHUSDC": pa_mgr})
    sig = pa._series_signature_for("ETHUSDC")
    assert sig == (n_pts, t0_ms + (n_pts - 1) * 3600 * 1000, 2000.0 + (n_pts - 1) * 2.5)

    monkeypatch.setattr(pa, "estimate_remaining_trend_hours", lambda *a, **kw: 12.0)
    res_fast = pa.getTrendLongTerm_fixed("ETHUSDC", window_hours=16, step_hours=8, lookback_days=30, draw=False)
    # Compare against legacy tuple path by monkeypatching priceLstFor to a wrapper (disables fastpath)
    orig_plf = pa.priceLstFor
    monkeypatch.setattr(pa, "priceLstFor", lambda s: orig_plf(s))
    res_legacy = pa.getTrendLongTerm_fixed("ETHUSDC", window_hours=16, step_hours=8, lookback_days=30, draw=False)
    assert res_fast is not None and res_legacy is not None
    assert res_fast["direction"] == res_legacy["direction"]
    assert res_fast["start_timestamp"] == pytest.approx(res_legacy["start_timestamp"], rel=1e-9)
    assert res_fast["duration_seconds"] == pytest.approx(res_legacy["duration_seconds"], rel=1e-9)


def test_jsonl_unterminated_eof_parity_and_corrupt_utf8_previous_recovery(tmp_path):
    """Audit #4 + Cycle 2/3/5: Both JSONL offset readers withhold valid unterminated EOF lines until '\\n',
    and JSONL cached/read-only/producer loaders recover cleanly from corrupt UTF-8 primary files and .previous."""
    import cacheprice

    now_ms = int(time.time() * 1000)

    # 1. Valid JSON line without trailing '\n': both bucket-map and dict offset readers keep cursor before it
    j_eof = tmp_path / "eof_parity.jsonl"
    complete_line = json.dumps({"s": "BTCUSDC", "i": [now_ms - 2000, 60000.0]}, separators=(",", ":")) + "\n"
    unterminated_valid = json.dumps({"s": "BTCUSDC", "i": [now_ms - 1000, 60050.0]}, separators=(",", ":"))
    j_eof.write_bytes((complete_line + unterminated_valid).encode("utf-8"))

    bmap0, bad0, off0, di0, anc0, inc0 = state_io.read_jsonl_bucket_map_from_offset(str(j_eof), start_offset=0)
    dicts0, doff0 = state_io.read_jsonl_dicts_from_offset(str(j_eof), 0)
    expected_off = len(complete_line.encode("utf-8"))
    assert off0 == expected_off and doff0 == expected_off
    assert bad0 is True and inc0 is False
    assert bmap0["BTCUSDC"] == [[now_ms - 2000, 60000.0]]
    assert len(dicts0) == 1

    bmap_wait, bad_wait, off_wait, _di_w, _anc_w, inc_wait = state_io.read_jsonl_bucket_map_from_offset(
        str(j_eof), start_offset=off0, expected_dev_ino=di0, expected_anchor=anc0
    )
    dicts_wait, doff_wait = state_io.read_jsonl_dicts_from_offset(str(j_eof), doff0)
    assert inc_wait is True and bad_wait is False
    assert bmap_wait == {} and off_wait == expected_off
    assert dicts_wait == [] and doff_wait == expected_off

    with open(j_eof, "ab") as fh:
        fh.write(b"\n")
    bmap1, bad1, off1, _di1, _anc1, inc1 = state_io.read_jsonl_bucket_map_from_offset(
        str(j_eof), start_offset=off0, expected_dev_ino=di0, expected_anchor=anc0
    )
    dicts1, doff1 = state_io.read_jsonl_dicts_from_offset(str(j_eof), doff0)
    assert inc1 is True and bad1 is False
    assert bmap1["BTCUSDC"] == [[now_ms - 1000, 60050.0]]
    assert len(dicts1) == 1 and dicts1[0]["i"] == [now_ms - 1000, 60050.0]
    assert off1 == doff1 == j_eof.stat().st_size

    # 2. Invalid UTF-8 primary .jsonl + valid .jsonl.previous -> recovers cleanly in BOTH state_io and Cache24PriceManager(read_only=True/False)
    corrupt_p = tmp_path / "cache_24_price_ETHUSDC.jsonl"
    prev_p = Path(previous_generation_path(str(corrupt_p)))
    corrupt_p.write_bytes(b"\xff\xfe\x80\x81\n")
    prev_p.write_text(
        json.dumps({"s": "ETHUSDC", "i": [now_ms - 5000, 3100.0]}, separators=(",", ":")) + "\n"
        + json.dumps({"s": "ETHUSDC", "i": [now_ms - 4000, 3105.0]}, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    memo: dict = {}
    recovered = state_io.read_jsonl_or_json_dict_cached(str(corrupt_p), memo)
    assert recovered is not None
    assert recovered["items"]["ETHUSDC"] == [[now_ms - 5000, 3100.0], [now_ms - 4000, 3105.0]]

    ro_mgr = cacheprice.Cache24PriceManager(30, ["ETHUSDC"], str(corrupt_p), api_client=None, read_only=True)
    assert ro_mgr.cache["ETHUSDC"] == [[now_ms - 5000, 3100.0], [now_ms - 4000, 3105.0]]

    rw_mgr = cacheprice.Cache24PriceManager(
        30, ["ETHUSDC"], str(corrupt_p), api_client=None, read_only=False, append_persist=True
    )
    assert rw_mgr.cache["ETHUSDC"] == [[now_ms - 5000, 3100.0], [now_ms - 4000, 3105.0]]
    assert rw_mgr._legacy_jsonl_needs_rewrite is True
    assert prev_p.exists()

    # 3. Corrupt UTF-8 line 1 followed by valid JSONL lines (no .previous) recovers valid rows cleanly
    corrupt_head = tmp_path / "cache_24_price_SOLUSDC.jsonl"
    corrupt_head.write_bytes(
        b"\xff\xfe\x80garbage\n"
        + (json.dumps({"s": "SOLUSDC", "i": [now_ms - 3000, 150.0]}, separators=(",", ":")) + "\n").encode("utf-8")
    )
    bmap_direct, had_bad_direct = state_io.read_jsonl_bucket_map(str(corrupt_head))
    assert had_bad_direct is True and bmap_direct == {"SOLUSDC": [[now_ms - 3000, 150.0]]}
    rec_head = state_io.read_jsonl_or_json_dict_cached(str(corrupt_head), {})
    assert rec_head is not None and rec_head["items"] == {"SOLUSDC": [[now_ms - 3000, 150.0]]}


def test_ssot_p1_p2_reentrant_filelock_cachedb_exists_365_compact_and_ast_gates(tmp_path, monkeypatch):
    """Verify P1+P2 structural SSOT invariants:
    1) FileLock is re-entrant per thread (nested/recursive acquisition never deadlocks) while still
       strictly excluding other threads via flock;
    2) state_io.cachedb_exists resolves cachedb/<subdomain>/ identically to atomic_write_*;
    3) Cache365PriceManager shares CacheBoundedJsonlArchiveMixin + CompactPriceSeries (16 B/pt)
       while preserving issubclass(Cache365PriceManager, Cache24PriceManager) and get_series_arrays();
    4) New SSOT AST gates (Audits 11 & 12) catch broken local imports, bad call signatures,
       unresolved os.path.exists(os.path.join(cache_dir, ...)), and JSON readers on JSONL paths."""
    import threading
    from pathlib import Path
    import numpy as np
    import cacheprice
    from core.lock import FileLock
    from verify_tools import ssot_ast_audit as ast_audit

    # 1. Re-entrant FileLock (nested instances + recursive same instance) + cross-thread exclusion
    lock_p = tmp_path / "reentrant.lock"
    other_thread_acquired: list[bool] = []
    other_thread_after_release: list[bool] = []

    def _contend(out_list: list[bool]):
        with FileLock(lock_p, timeout=0.05, poll_interval=0.01, raise_on_timeout=False, reentrant=True) as lk:
            out_list.append(bool(lk.acquired))

    re_lock = FileLock(lock_p, reentrant=True)
    with re_lock as outer:
        assert outer.acquired is True
        # Same instance recursive re-entry
        with re_lock as inner_same:
            assert inner_same.acquired is True
            # Distinct instance nested re-entry on same thread
            with FileLock(lock_p, timeout=0.05, reentrant=True) as inner_other:
                assert inner_other.acquired is True
                # Another thread MUST still be blocked while outer lock is held
                t1 = threading.Thread(target=_contend, args=(other_thread_acquired,))
                t1.start()
                t1.join(timeout=2.0)
                assert other_thread_acquired == [False]
        # Still held after inner exits!
        assert outer.acquired is True
    # Released after outermost exit -> other thread can now acquire immediately
    t2 = threading.Thread(target=_contend, args=(other_thread_after_release,))
    t2.start()
    t2.join(timeout=2.0)
    assert other_thread_after_release == [True]

    # Unified CacheManagerInterface._generation_lock() shares re-entrancy with state_io.generation_file_lock()
    gen_target = tmp_path / "gen_nested.jsonl"
    with state_io.generation_file_lock(str(gen_target)):
        with state_io.generation_file_lock(str(gen_target)):
            pass

    # 2. state_io.cachedb_exists resolves subdomain paths + .previous
    cdb = tmp_path / "cachedb"
    prices_dir = cdb / "prices"
    prices_dir.mkdir(parents=True)
    monkeypatch.setenv("MPTRADE_CACHEDB_DIR", str(cdb))
    p365_real = prices_dir / "cache_365_price_BTCUSDC.jsonl"
    p365_unresolved = str(cdb / "cache_365_price_BTCUSDC.jsonl")
    assert state_io.cachedb_exists(p365_unresolved) is False
    p365_real.write_text(json.dumps({"s": "BTCUSDC", "i": [1_795_000_000_000, 60000.0]}) + "\n", encoding="utf-8")
    assert state_io.cachedb_exists(p365_unresolved) is True
    p365_real.rename(prices_dir / "cache_365_price_BTCUSDC.jsonl.previous")
    assert state_io.cachedb_exists(p365_unresolved, include_previous=False) is False
    assert state_io.cachedb_exists(p365_unresolved, include_previous=True) is True

    # 3. Cache365PriceManager shares CacheBoundedJsonlArchiveMixin + CompactPriceSeries
    assert issubclass(cacheprice.Cache365PriceManager, cacheprice.Cache24PriceManager)
    assert issubclass(cacheprice.Cache365PriceManager, cacheprice.CacheBoundedJsonlArchiveMixin)
    assert issubclass(cacheprice.CacheSparsePriceManager, cacheprice.CacheBoundedJsonlArchiveMixin)
    p365_live = prices_dir / "cache_365_price_ETHUSDC.jsonl"
    now_ms = int(time.time() * 1000)
    p365_live.write_text(
        "\n".join(json.dumps({"s": "ETHUSDC", "i": [now_ms - (5 - k) * 1000, 3000.0 + k]}) for k in range(5)) + "\n",
        encoding="utf-8",
    )
    mgr365 = cacheprice.Cache365PriceManager(60, ["ETHUSDC"], str(p365_live), read_only=True)
    assert isinstance(mgr365.cache["ETHUSDC"], cacheprice.CompactPriceSeries)
    ts365, px365 = mgr365.get_series_arrays("ETHUSDC", since_ms=now_ms - 3000)
    assert ts365.dtype == np.int64 and px365.dtype == np.float64
    assert len(ts365) == 3 and float(px365[-1]) == pytest.approx(3004.0)

    # 4. Audits 11 & 12 catch synthetic violations cleanly on an isolated mini-repo
    mini = tmp_path / "mini_repo"
    mini.mkdir()
    (mini / "callee_mod.py").write_text("def need_two(a, b):\n    return a + b\n", encoding="utf-8")
    (mini / "bad_caller.py").write_text(
        "import os\n"
        "from callee_mod import need_two, missing_sym\n"
        "def f(cache_dir):\n"
        "    need_two(1)\n"
        "    p = os.path.join(cache_dir, 'cache_365_price_BTCUSDC.jsonl')\n"
        "    if os.path.exists(p):\n"
        "        return read_json_dict('cache_24_price_BTCUSDC.jsonl')\n",
        encoding="utf-8",
    )
    v11 = ast_audit.audit_cross_module_calls_and_imports_ast(mini, rel_paths=["bad_caller.py"])
    assert any("missing_sym" in v for v in v11), v11
    assert any("missing required positional arg" in v for v in v11), v11
    v12 = ast_audit.audit_cachedb_path_and_jsonl_purity_ast(mini, rel_paths=["bad_caller.py"])
    assert any("unresolved os.path.join(cache_dir" in v for v in v12), v12
    assert any("JSONL cache path" in v for v in v12), v12


def test_ssot_p3_durability_tier_cachedb_root_generation_tx_ws_watchdog_and_ast13(tmp_path, monkeypatch):
    """Verify SSOT P3 structural refactors:
    1) CachedbRoot.from_path normalizes any file/subdomain/nested dir to the root cachedb/;
    2) DurabilityTier / resolve_durability_policy drives default fsync, .previous, compact JSON, and skip_if_unchanged;
    3) generation_transaction resolves subdomain + nests re-entrantly with generation_file_lock;
    4) Unified PublicPriceStreamBase._check_data_staleness_or_reconnect works for Binance & Venue WS streams;
    5) AST Audit 13 blocks fsync=False on FINANCIAL_CRITICAL paths and os.path.dirname(STATE.path) as cache_dir."""
    import asyncio
    import binance_api.bapi_ws as bapi_ws
    import cachepaths
    import providers.price_poller as pp
    import providers.ws_stream as ws_stream
    from verify_tools import ssot_ast_audit as ast_audit

    # 1. CachedbRoot value-object never leaks a subdomain leaf
    cdb = tmp_path / "cachedb"
    for sub in ("state", "prices", "orders", "macro", "trends/24trend", "system/heartbeat"):
        (cdb / sub).mkdir(parents=True, exist_ok=True)
    for sample in (
        cdb,
        cdb / "state",
        cdb / "state" / "assetguardian_state.json",
        cdb / "trends" / "24trend",
        cdb / "trends" / "24trend" / "live_BTCUSDC.json",
        cdb / "system" / "heartbeat",
        cdb / "system" / "heartbeat" / "cacheManager.heartbeat",
    ):
        root_obj = cachepaths.CachedbRoot.from_path(sample)
        assert str(root_obj) == str(cdb)
        assert os.fspath(root_obj) == str(cdb)
    assert cachepaths.CachedbRoot.from_path(cdb / "state" / "assetguardian_state.json").resolve(
        cachepaths.CACHE_SENTIMENT_EVAL_FILENAME
    ) == str(cdb / "macro" / cachepaths.CACHE_SENTIMENT_EVAL_FILENAME)

    # 2. Declarative DurabilityTier classification + state_io defaults + skip_if_unchanged dirty-check
    assert cachepaths.resolve_durability_policy("orders_active.json").tier is cachepaths.DurabilityTier.FINANCIAL_CRITICAL
    assert cachepaths.is_financial_critical_path("rtrade_pairs_state.json") is True
    assert cachepaths.resolve_durability_policy("cache_sparse_price_BTCUSDC.jsonl").tier is cachepaths.DurabilityTier.ARCHIVE_DURABLE
    assert cachepaths.resolve_durability_policy("cache_365_price_BTCUSDC.jsonl").tier is cachepaths.DurabilityTier.ARCHIVE_DURABLE
    assert cachepaths.resolve_durability_policy("cache_ohlc_bars_1d_BTCUSDC.json").tier is cachepaths.DurabilityTier.ARCHIVE_DURABLE
    assert cachepaths.resolve_durability_policy("cache_ohlc_bars_4h_SPCXxUSD.json").tier is cachepaths.DurabilityTier.ARCHIVE_DURABLE
    assert cachepaths.resolve_durability_policy("cacheManager.heartbeat").tier is cachepaths.DurabilityTier.EPHEMERAL_TELEMETRY
    assert cachepaths.resolve_durability_policy("orderbook_depth_BTCUSDC.json").tier is cachepaths.DurabilityTier.EPHEMERAL_TELEMETRY
    assert cachepaths.resolve_durability_policy("priceanalysis.json").tier is cachepaths.DurabilityTier.EPHEMERAL_TELEMETRY
    assert cachepaths.resolve_durability_policy("cache_24_price_BTCUSDC.jsonl").tier is cachepaths.DurabilityTier.EPHEMERAL_TELEMETRY
    assert cachepaths.resolve_durability_policy("cache_ohlc_24h_NVDA_US_EQ.json").tier is cachepaths.DurabilityTier.EPHEMERAL_TELEMETRY
    assert cachepaths.resolve_durability_policy("sentiment_eval.json").tier is cachepaths.DurabilityTier.FINANCIAL_CRITICAL

    fsync_calls: list[int] = []
    stage_prev_calls: list[str] = []
    orig_stage_prev = state_io.stage_previous_generation
    monkeypatch.setattr(os, "fsync", lambda fd: fsync_calls.append(fd))
    monkeypatch.setattr(
        state_io,
        "stage_previous_generation",
        lambda p, **kw: (stage_prev_calls.append(str(p)), orig_stage_prev(p, **kw))[1],
    )

    # Ephemeral telemetry path: default save_json_state skips fsync, skips .previous, and writes compact JSON
    pa_path = cdb / "priceanalysis.json"
    state_io.save_json_state(str(pa_path), {"BTCUSDC": {"slope": 1.2}})
    state_io.save_json_state(str(pa_path), {"BTCUSDC": {"slope": 1.3}})
    assert fsync_calls == []
    assert stage_prev_calls == []
    raw_pa = (cdb / "trends" / "priceanalysis.json").read_text(encoding="utf-8")
    assert "\n" not in raw_pa and ": " not in raw_pa

    # Financial critical path: default save_json_state enforces fsync=True + .previous, and skip_if_unchanged avoids no-op writes
    ord_path = cdb / "orders_active.json"
    state_io.save_json_state(str(ord_path), {"orders": {"1": "NEW"}}, sort_keys=True, skip_if_unchanged=True)
    assert len(fsync_calls) >= 1
    fsync_calls.clear()
    stage_prev_calls.clear()
    # Identical payload with skip_if_unchanged=True -> 0 fsync, 0 .previous staging
    state_io.save_json_state(str(ord_path), {"orders": {"1": "NEW"}}, sort_keys=True, skip_if_unchanged=True)
    assert fsync_calls == []
    assert stage_prev_calls == []
    # Changed payload -> writes durably with fsync + .previous staging
    state_io.save_json_state(str(ord_path), {"orders": {"1": "FILLED"}}, sort_keys=True, skip_if_unchanged=True)
    assert len(fsync_calls) >= 1
    assert len(stage_prev_calls) == 1

    # 3. generation_transaction resolves subdomain and nests re-entrantly with generation_file_lock
    with state_io.generation_transaction(str(cdb / "cache_24_price_BTCUSDC.jsonl")) as tx_path:
        assert tx_path == str(cdb / "prices" / "cache_24_price_BTCUSDC.jsonl")
        with state_io.generation_file_lock(tx_path):
            pass

    # 4. Unified WS staleness watchdog on PublicPriceStreamBase (Binance + VenuePriceStreamBase)
    mono = [500.0]
    monkeypatch.setattr(ws_stream.time, "monotonic", lambda: mono[0])
    monkeypatch.setattr(bapi_ws.time, "monotonic", lambda: mono[0])
    monkeypatch.setattr(pp.time, "monotonic", lambda: mono[0])

    class _DummyVenueStream(pp.VenuePriceStreamBase):
        def _build_subscribe_payload(self):
            return None

        def _process_message(self, raw, *, now_ms=None):
            _ = (raw, now_ms)

    hl_like = _DummyVenueStream(symbols=["BTCUSDC"], ws_url="wss://example.invalid", stale_reconnect_sec=60.0)
    assert hl_like._check_data_staleness_or_reconnect(500.0) is False
    mono[0] = 561.0
    assert hl_like._check_data_staleness_or_reconnect(500.0) is True

    class _SilentWs:
        def __init__(self):
            self.n = 0

        async def recv(self):
            self.n += 1
            mono[0] += 35.0
            return '{"channel":"pong"}'

    mono[0] = 1000.0
    bstream = bapi_ws.BinancePriceStream(symbols=["BTCUSDC"], stale_reconnect_sec=60.0)
    sws = _SilentWs()
    asyncio.run(bstream._recv_loop(sws))
    assert sws.n == 2

    # 5. AST Audit 13 catches forbidden fsync=False on financial files & os.path.dirname(STATE.path)
    mini = tmp_path / "mini_p3"
    mini.mkdir()
    (mini / "bad_p3.py").write_text(
        "import os\n"
        "def bad(STATE, ORDER_LEDGER_ACTIVE_FILENAME, rtrade_pairs_path):\n"
        "    cache_dir = os.path.dirname(STATE.path)\n"
        "    atomic_write_json(ORDER_LEDGER_ACTIVE_FILENAME, {}, fsync=False)\n"
        "    save_json_state(rtrade_pairs_path, {}, fsync=False)\n"
        "    return cache_dir\n",
        encoding="utf-8",
    )
    v13 = ast_audit.audit_durability_tier_and_cachedb_dirname_ast(mini, rel_paths=["bad_p3.py"])
    assert any("ORDER_LEDGER_ACTIVE_FILENAME" in v for v in v13), v13
    assert any("rtrade_pairs_path" in v for v in v13), v13
    assert any("os.path.dirname(STATE.path)" in v for v in v13), v13


def test_orders_state_skip_if_unchanged_and_filled_touch_no_resurrection(tmp_path, monkeypatch):
    """Verify:
    1) CacheOrderFilledManager.touch_venue_filled_orders_snapshot & merge_venue_filled_orders_snapshot
       never resurrect an invalidated symbol's fetchtime when create_missing=False (including via unresolved flat path);
    2) Financial orders/ & state/ snapshot writers skip redundant fsync/.previous rewrites on unchanged payloads
       while writing durably on any real mutation."""
    import assetguardian
    import cacheaccount
    import rtrade_pair_store

    cdb = tmp_path / "cachedb"
    orders_dir = cdb / "orders"
    state_dir = cdb / "state"
    orders_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    # 1. Invalidated symbol in cache_order_filled_kraken.json must NOT be resurrected by WS keepalive touch
    #    or by a fill for a different symbol (even when called via unresolved flat cachedb/ path).
    flat_filled = str(cdb / "cache_order_filled_kraken.json")
    real_filled = orders_dir / "cache_order_filled_kraken.json"
    now_ms = 1_795_600_000_000
    f_hype = {"orderId": "O-H1", "tid": "T-H1", "side": "BUY", "price": 50.0, "quantity": 1.0, "time": now_ms - 5000}
    f_spcx = {"orderId": "O-S1", "tid": "T-S1", "side": "BUY", "price": 100.0, "quantity": 1.0, "time": now_ms - 4000}
    cacheaccount.CacheOrderFilledManager.merge_venue_filled_orders_snapshot(
        flat_filled,
        {"HYPEUSD": [f_hype], "SPCXxUSD": [f_spcx]},
        venue="kraken",
        now_ms=now_ms,
        is_snapshot=True,
    )
    assert real_filled.is_file()
    cacheaccount.CacheOrderFilledManager.invalidate_venue_fills_snapshot(
        "HYPEUSD", flat_filled, candidate_keys=("HYPEUSD",)
    )
    doc_after_inv = json.loads(real_filled.read_text(encoding="utf-8"))
    assert "HYPEUSD" not in doc_after_inv["fetchtime"]
    assert "SPCXxUSD" in doc_after_inv["fetchtime"]

    # WS keepalive touch (create_missing=False) must update SPCXxUSD but NOT resurrect HYPEUSD
    touched = cacheaccount.CacheOrderFilledManager.touch_venue_filled_orders_snapshot(
        flat_filled, now_ms=now_ms + 15_000, symbols=["HYPEUSD", "SPCXxUSD"], create_missing=False
    )
    assert touched is True
    doc_after_touch = json.loads(real_filled.read_text(encoding="utf-8"))
    assert "HYPEUSD" not in doc_after_touch["fetchtime"]
    assert doc_after_touch["fetchtime"]["SPCXxUSD"] == now_ms + 15_000

    # A fill arriving for SPCXxUSD with touch_symbols=[HYPEUSD, SPCXxUSD] and touch_create_missing=False
    # must ALSO not resurrect HYPEUSD!
    f_spcx2 = {"orderId": "O-S2", "tid": "T-S2", "side": "SELL", "price": 105.0, "quantity": 1.0, "time": now_ms + 20_000}
    cacheaccount.CacheOrderFilledManager.merge_venue_filled_orders_snapshot(
        flat_filled,
        {"SPCXxUSD": [f_spcx2]},
        venue="kraken",
        now_ms=now_ms + 20_000,
        is_snapshot=False,
        touch_symbols=["HYPEUSD", "SPCXxUSD"],
        touch_create_missing=False,
    )
    doc_after_merge = json.loads(real_filled.read_text(encoding="utf-8"))
    assert "HYPEUSD" not in doc_after_merge["fetchtime"]
    assert len(doc_after_merge["items"]["SPCXxUSD"]) == 2

    # 2. skip_if_unchanged=True prevents redundant fsync & .previous staging on identical orders/ and state/ snapshots
    fsync_calls: list[int] = []
    stage_calls: list[str] = []
    orig_stage = state_io.stage_previous_generation
    monkeypatch.setattr(os, "fsync", lambda fd: fsync_calls.append(fd))
    monkeypatch.setattr(
        state_io,
        "stage_previous_generation",
        lambda p, **kw: (stage_calls.append(str(p)), orig_stage(p, **kw))[1],
    )

    # 2a. CacheOrderOpenManager.replace_venue_open_orders_snapshot (same open orders + same timestamp -> 0 fsync)
    open_p = str(cdb / "cache_order_open_kraken.json")
    cacheaccount.CacheOrderOpenManager.replace_venue_open_orders_snapshot(
        open_p, {"HYPEUSD": []}, now_ms=now_ms, verified_ts=now_ms / 1000.0
    )
    assert len(fsync_calls) >= 1
    fsync_calls.clear()
    stage_calls.clear()
    cacheaccount.CacheOrderOpenManager.replace_venue_open_orders_snapshot(
        open_p, {"HYPEUSD": []}, now_ms=now_ms, verified_ts=now_ms / 1000.0
    )
    assert fsync_calls == [] and stage_calls == []

    # 2b. CacheOrderFilledManager.merge_venue_filled_orders_snapshot (same snapshot + force_save=True -> 0 fsync)
    cacheaccount.CacheOrderFilledManager.merge_venue_filled_orders_snapshot(
        flat_filled,
        {"SPCXxUSD": [f_spcx, f_spcx2]},
        venue="kraken",
        now_ms=now_ms + 20_000,
        is_snapshot=True,
        force_save=True,
    )
    assert fsync_calls == [] and stage_calls == []

    # 2c. AssetGuardianState.save (unchanged state -> 0 fsync; mutated state -> durable fsync)
    ag_st = assetguardian.AssetGuardianState(
        path=str(state_dir / "assetguardian_state.json"),
    )
    ag_st.save({"BTCUSDC": {"tier": 1}})
    assert len(fsync_calls) >= 1
    fsync_calls.clear()
    stage_calls.clear()
    ag_st.save({"BTCUSDC": {"tier": 1}})
    assert fsync_calls == [] and stage_calls == []
    ag_st.save({"BTCUSDC": {"tier": 2}})
    assert len(fsync_calls) >= 1 and len(stage_calls) == 1

    # 2d. RTradePairStore._write (unchanged payload -> 0 fsync)
    rt_store = rtrade_pair_store.RTradePairStore(path=str(state_dir / "rtrade_pairs_state.json"))
    fsync_calls.clear()
    rt_store._write({"version": 1, "pairs": {}})
    assert len(fsync_calls) >= 1
    fsync_calls.clear()
    rt_store._write({"version": 1, "pairs": {}})
    assert fsync_calls == []


def test_compact_price_series_3col_5col_cache24_short_trend_and_assetguardian_fastpath(tmp_path, monkeypatch):
    """Verify CompactPriceSeries lazy 3-col/5-col OHLCV support, Cache24PriceManager compact storage
    (.jsonl append + compact + reload + legacy .json), and AssetGuardian/CacheShortTrend zero-copy fast-paths."""
    import numpy as np
    import assetguardian
    import cacheprice
    import cachetrend

    # 1. CompactPriceSeries: 0 bytes _vol/_extra for 2-col crypto, lossless lazy 3-col & 5-col OHLCV for equities
    s2 = cacheprice.CompactPriceSeries([[1_795_000_000_000, 60000.0], [1_795_000_001_000, 60010.0]])
    assert s2._vol is None and s2._extra is None
    assert s2.has_positive_volume() is False
    assert s2.bisect_left_ts(1_795_000_000_500) == 1
    assert list(s2) == [[1_795_000_000_000, 60000.0], [1_795_000_001_000, 60010.0]]

    s_mix = cacheprice.CompactPriceSeries([
        [1_795_000_000_000, 100.0],
        [1_795_000_060_000, 101.0, 250.5],
        [1_795_000_120_000, 102.0, 0.0],
        [1_795_000_180_000, 103.0, 500.0, 104.0, 99.5],
        [1_795_000_240_000, 103.5],
    ])
    assert s_mix._vol is not None and s_mix._extra == {3: (104.0, 99.5)}
    assert s_mix.has_positive_volume() is True
    assert list(s_mix) == [
        [1_795_000_000_000, 100.0],
        [1_795_000_060_000, 101.0, 250.5],
        [1_795_000_120_000, 102.0, 0.0],
        [1_795_000_180_000, 103.0, 500.0, 104.0, 99.5],
        [1_795_000_240_000, 103.5],
    ]
    sliced = s_mix[1:4]
    assert isinstance(sliced, cacheprice.CompactPriceSeries)
    assert list(sliced) == [
        [1_795_000_060_000, 101.0, 250.5],
        [1_795_000_120_000, 102.0, 0.0],
        [1_795_000_180_000, 103.0, 500.0, 104.0, 99.5],
    ]
    s_mix.trim_head(2)
    assert list(s_mix) == [
        [1_795_000_120_000, 102.0, 0.0],
        [1_795_000_180_000, 103.0, 500.0, 104.0, 99.5],
        [1_795_000_240_000, 103.5],
    ]
    ts_np, px_np = s_mix.to_numpy(since_ms=1_795_000_180_000)
    assert ts_np.dtype == np.int64 and px_np.dtype == np.float64
    assert ts_np.tolist() == [1_795_000_180_000, 1_795_000_240_000]
    assert px_np.tolist() == [103.0, 103.5]

    # 2. Cache24PriceManager stores CompactPriceSeries across append, compact_jsonl, reload, and OHLC merge
    now_ms = int(time.time() * 1000)
    c24_p = tmp_path / "cache_24_price_BTCUSDC.jsonl"
    mgr = cacheprice.Cache24PriceManager(
        30, ["BTCUSDC"], str(c24_p), api_client=None, data_sync_mode="producer", current_price_getter=lambda: None
    )
    mgr.update_cache_per_symbol("BTCUSDC", [[now_ms - 10_000 + i * 1000, 60000.0 + i] for i in range(10)])
    assert isinstance(mgr.cache["BTCUSDC"], cacheprice.CompactPriceSeries)
    assert mgr.cache["BTCUSDC"]._vol is None
    assert mgr.save_state_to_file() is True
    assert mgr.compact_jsonl() is True
    assert isinstance(mgr.cache["BTCUSDC"], cacheprice.CompactPriceSeries)

    reader = cacheprice.Cache24PriceManager(
        30, ["BTCUSDC"], str(c24_p), api_client=None, read_only=True
    )
    assert isinstance(reader.cache["BTCUSDC"], cacheprice.CompactPriceSeries)
    ts_r, px_r = reader.get_series_arrays("BTCUSDC", since_ms=now_ms - 5000)
    assert len(ts_r) == 5 and float(px_r[-1]) == pytest.approx(60009.0)

    # Equity 3-col OHLC merge stays CompactPriceSeries and round-trips through JSONL
    eq_p = tmp_path / "cache_24_price_NVDA_US_EQ.jsonl"
    eq_mgr = cacheprice.Cache24PriceManager(
        30, ["NVDA_US_EQ"], str(eq_p), api_client=None, data_sync_mode="producer", current_price_getter=lambda: None
    )
    eq_mgr._merge_ohlc_rows_locked(
        "NVDA_US_EQ",
        [[now_ms - 600_000, 130.0, 1234.5], [now_ms - 300_000, 131.0, 2345.0, 131.5, 129.8]],
    )
    assert isinstance(eq_mgr.cache["NVDA_US_EQ"], cacheprice.CompactPriceSeries)
    assert eq_mgr.cache["NVDA_US_EQ"].has_positive_volume() is True
    eq_mgr.fetchtime_per_symbol["NVDA_US_EQ"] = now_ms + 1000
    eq_mgr._jsonl_needs_compact = True
    assert eq_mgr.save_state_to_file() is True
    eq_reader = cacheprice.Cache24PriceManager(
        30, ["NVDA_US_EQ"], str(eq_p), api_client=None, read_only=True
    )
    assert list(eq_reader.cache["NVDA_US_EQ"]) == [
        [now_ms - 600_000, 130.0, 1234.5],
        [now_ms - 300_000, 131.0, 2345.0, 131.5, 129.8],
    ]

    # 3. AssetGuardian fast-path reads directly from CompactPriceSeries without _read_symbol_price_rows list copy
    monkeypatch.setattr(assetguardian, "get_24_price_manager", lambda symbols=None: {"BTCUSDC": reader})
    ts_ag, px_ag = assetguardian._read_symbol_window_arrays(
        "BTCUSDC", (now_ms - 5500) / 1000.0, (now_ms + 5000) / 1000.0
    )
    assert len(ts_ag) == 5 and float(px_ag[-1]) == pytest.approx(60009.0)

    # 4. CacheShortTrendManager evaluates CompactPriceSeries directly via bisect_left_ts & minimal slice
    st_p = tmp_path / "cache_short_trend.json"
    stm = cachetrend.CacheShortTrendManager(symbols=["BTCUSDC"], filename=str(st_p), read_only=True)
    stm._cache24_managers = {"BTCUSDC": reader}
    out_w = stm.get_short_trend_for_window("BTCUSDC", 300.0, now=(now_ms + 1000) / 1000.0)
    assert isinstance(out_w, dict) and out_w["n_samples"] == 10


def test_jsonl_nul_corruption_auto_heals_on_next_flush_and_restores_incremental_tailing(tmp_path):
    """Regression for reboot-induced NUL bytes in ephemeral fsync=False Cache24 JSONL files:
    1) Startup sets _legacy_jsonl_needs_rewrite=True while preserving all valid rows;
    2) Next save_state_to_file() auto-compacts away the corrupt line and clears _legacy_jsonl_needs_rewrite;
    3) read_only consumers immediately regain O(1) incremental byte-offset tailing (_jsonl_tail_state != None);
    4) CacheBoundedJsonlArchiveMixin (Sparse/365d) never triggers memory-to-disk compaction on append."""
    import cacheprice

    now_ms = int(time.time() * 1000)
    eth_p = tmp_path / "cache_24_price_ETHUSDC.jsonl"
    row1 = json.dumps({"s": "ETHUSDC", "i": [now_ms - 5000, 3100.0]}, separators=(",", ":")) + "\n"
    row2 = json.dumps({"s": "ETHUSDC", "i": [now_ms - 4000, 3105.0]}, separators=(",", ":")) + "\n"
    row3 = json.dumps({"s": "ETHUSDC", "i": [now_ms - 3000, 3110.0]}, separators=(",", ":")) + "\n"
    # Exact reproduction of an unclean kernel reboot leaving a NUL-byte sector mid-file
    eth_p.write_bytes(
        row1.encode("utf-8")
        + row2.encode("utf-8")
        + (b"\x00" * 1024 + b"\n")
        + row3.encode("utf-8")
    )
    meta_p = Path(state_io.meta_file_path(str(eth_p)))
    meta_p.write_text(
        json.dumps({"fetchtime": {"ETHUSDC": now_ms - 3000}, "counts": {"ETHUSDC": 3}, "max_ts": now_ms - 3000}),
        encoding="utf-8",
    )

    # Producer loads valid rows and marks rewrite needed
    prod = cacheprice.Cache24PriceManager(
        30, ["ETHUSDC"], str(eth_p), api_client=None, data_sync_mode="producer", current_price_getter=lambda: None
    )
    assert len(prod.cache["ETHUSDC"]) == 3
    assert prod._legacy_jsonl_needs_rewrite is True

    # Before flush, read_only consumer sees corrupt line and cannot arm incremental tailing
    reader = cacheprice.Cache24PriceManager(
        30, ["ETHUSDC"], str(eth_p), api_client=None, read_only=True
    )
    assert len(reader.cache["ETHUSDC"]) == 3
    assert reader._jsonl_tail_state is None

    # Next flush auto-heals the file (even with just 1 new tick, well below 4000-row / 30m compact thresholds)
    prod.update_cache_per_symbol("ETHUSDC", [[now_ms - 2000, 3115.0]])
    assert prod.save_state_to_file() is True
    assert prod._legacy_jsonl_needs_rewrite is False
    disk_map, had_bad = state_io.read_jsonl_bucket_map(str(eth_p))
    assert had_bad is False
    assert disk_map["ETHUSDC"] == [
        [now_ms - 5000, 3100.0],
        [now_ms - 4000, 3105.0],
        [now_ms - 3000, 3110.0],
        [now_ms - 2000, 3115.0],
    ]

    # Reader resyncs cleanly onto the healed file and arms incremental tailing
    reader.resync_mem_file()
    assert len(reader.cache["ETHUSDC"]) == 4
    assert reader._jsonl_tail_state is not None
    prev_tail = reader._jsonl_tail_state

    # Subsequent append uses fast O(1) incremental byte-offset tailing (same inode, larger offset)
    prod.update_cache_per_symbol("ETHUSDC", [[now_ms - 1000, 3120.0]])
    assert prod.save_state_to_file() is True
    reader.resync_mem_file()
    assert len(reader.cache["ETHUSDC"]) == 5
    assert reader._jsonl_tail_state is not None
    assert reader._jsonl_tail_state[1] == prev_tail[1]  # same (dev, ino)
    assert reader._jsonl_tail_state[2] > prev_tail[2]   # advanced byte offset

    # Multi-year bounded archives (Sparse/365d) must NEVER compact from bounded memory on append
    sp_p = tmp_path / "cache_sparse_price_BTCUSDC.jsonl"
    sp_mgr = cacheprice.CacheSparsePriceManager(
        300, ["BTCUSDC"], str(sp_p), api_client=None, data_sync_mode="producer", current_price_getter=lambda: None
    )
    sp_mgr._legacy_jsonl_needs_rewrite = True
    assert sp_mgr._should_compact_before_append_locked() is False


def test_previous_recovery_no_short_circuit_repair_implies_recover_and_nav_bnb_pricing(tmp_path):
    """Verify three structural fixes:
    1) Missing primary + valid .previous recovers cleanly (no os.path.isfile short-circuit) across
       OrderLedgerReader.read_active_snapshot, CacheOrderFilledManager, and CacheOrderOpenManager;
    2) state_io.read_json_dict(..., repair_files=True) structurally implies recover_previous=True;
    3) nav_history_store prices Binance fee-reserve BNB from venue_fee_maintenance_state.json when
       BNBUSDC is absent from cache_current_price.json (eliminating fake BNB top-up drawdown)."""
    import cacheaccount
    from providers.nav_history_store import _load_workspace_balances_and_prices, record_nav_history_snapshot
    from providers.order_ledger import OrderLedgerReader

    now_s = time.time()
    now_ms = int(now_s * 1000)
    orders_dir = tmp_path / "cachedb" / "orders"
    state_dir = tmp_path / "cachedb" / "state"
    prices_dir = tmp_path / "cachedb" / "prices"
    orders_dir.mkdir(parents=True)
    state_dir.mkdir(parents=True)
    prices_dir.mkdir(parents=True)

    # 1a. OrderLedgerReader.read_active_snapshot recovers orders_active.json.previous when primary is missing
    active_p = orders_dir / "orders_active.json"
    active_prev = Path(state_io.previous_generation_path(str(active_p)))
    active_prev.write_text(
        json.dumps({
            "schema_version": 1,
            "updated_ts": now_s,
            "journal_cursor": {"file": "order_events_2026-11-25.jsonl", "offset": 128, "offsets": {"order_events_2026-11-25.jsonl": 128}},
            "venue_verified_ts": {"binance": now_s},
            "symbol_verified_ts": {"binance:btcusdc": now_s},
            "recently_closed": {},
            "orders": {"o1": {"venue": "binance", "symbol": "BTCUSDC", "state": "OPEN"}},
        }),
        encoding="utf-8",
    )
    snap_ro = OrderLedgerReader(cachedb_dir=str(tmp_path / "cachedb")).read_active_snapshot(repair_files=False)
    assert "o1" in snap_ro["orders"] and snap_ro["journal_cursor"]["offset"] == 128
    assert not active_p.exists() and active_prev.exists()  # passive reader did not mutate disk

    # 1b. CacheOrderFilledManager read & touch recover .previous when primary is missing
    filled_p = orders_dir / "cache_order_filled_kraken.json"
    filled_prev = Path(state_io.previous_generation_path(str(filled_p)))
    filled_prev.write_text(
        json.dumps({
            "items": {"HYPEUSD": [{"orderId": "kr1", "side": "BUY", "price": 55.0, "quantity": 2.0, "timestamp": now_ms - 5000}]},
            "fetchtime": {"HYPEUSD": now_ms - 5000},
        }),
        encoding="utf-8",
    )
    f_items, f_ft = cacheaccount.CacheOrderFilledManager.read_venue_filled_orders_snapshot(str(filled_p), venue="kraken")
    assert len(f_items.get("HYPEUSD", [])) == 1 and f_items["HYPEUSD"][0]["orderId"] == "kr1"
    assert f_ft.get("HYPEUSD") == now_ms - 5000
    assert cacheaccount.CacheOrderFilledManager.touch_venue_filled_orders_snapshot(
        str(filled_p), now_ms=now_ms, symbols=["HYPEUSD"], create_missing=False
    ) is True
    assert filled_p.exists()

    # 1c. CacheOrderOpenManager.read_venue_open_orders_snapshot_detailed recovers .previous when primary is missing
    open_p = orders_dir / "cache_order_open_hyperliquid.json"
    open_prev = Path(state_io.previous_generation_path(str(open_p)))
    open_prev.write_text(
        json.dumps({
            "items": {"HYPEUSDC": [{"orderId": "hl99", "side": "BUY", "price": 50.0, "origQty": 3.0, "status": "OPEN", "timestamp": now_ms}]},
            "fetchtime": {"HYPEUSDC": now_ms},
            "verified_ts": now_s,
            "symbol_verified_ts": {"HYPEUSDC": now_s},
        }),
        encoding="utf-8",
    )
    o_items, _, o_ver, o_sym_ver = cacheaccount.CacheOrderOpenManager.read_venue_open_orders_snapshot_detailed(str(open_p))
    assert len(o_items.get("HYPEUSDC", [])) == 1 and o_items["HYPEUSDC"][0]["orderId"] == "hl99"
    assert o_ver == pytest.approx(now_s, rel=1e-3) and "HYPEUSDC" in o_sym_ver

    # 2. read_json_dict(..., repair_files=True) structurally implies recover_previous=True
    rep_p = state_dir / "repair_implies_recover.json"
    rep_prev = Path(state_io.previous_generation_path(str(rep_p)))
    rep_prev.write_text(json.dumps({"ok": True, "v": 42}), encoding="utf-8")
    recovered_doc = state_io.read_json_dict(str(rep_p), default=None, repair_files=True)
    assert recovered_doc == {"ok": True, "v": 42}
    assert rep_p.exists() and not rep_prev.exists()

    # 3. nav_history_store seeds BNB fee-reserve price from venue_fee_maintenance_state.json
    cacheaccount.CacheBalanceManager.replace_venue_balances_snapshot(
        str(orders_dir / "cache_balance_binance.json"),
        {"USDC": (1000.0, 0.0), "BNB": (0.2, 0.0)},
        venue="binance",
        now_s=now_s,
        verified_ts=now_s,
    )
    (prices_dir / "cache_current_price.json").write_text(
        json.dumps({"items": {"BTCUSDC": [[now_ms, 60000.0]]}}),
        encoding="utf-8",
    )
    (state_dir / "venue_fee_maintenance_state.json").write_text(
        json.dumps({"ts": now_s, "venues": {"binance": {"ts": now_s, "bnb_price_usdc": 550.0}}}),
        encoding="utf-8",
    )
    _, _, cp_seeded = _load_workspace_balances_and_prices(str(tmp_path))
    assert "BNBUSDC" in cp_seeded["items"] and cp_seeded["items"]["BNBUSDC"][0][1] == pytest.approx(550.0)
    dd = record_nav_history_snapshot(str(tmp_path), now_ts=now_s)
    # 1000 USDC + 0.2 BNB * 550 = 1110.0 USD (BNB reserve included!)
    assert dd["current_nav_usd"] == pytest.approx(1110.0, abs=0.05)


def test_read_jsonl_cached_never_pins_backup_fallback_under_primary_stat_sig(tmp_path, monkeypatch):
    """Invariant & cross-check: a transient primary JSONL read error falling back to .previous never pins stale backup under primary's stat sig."""
    now_ms = int(time.time() * 1000)
    primary = tmp_path / "cache_24_price_BTCUSDC.jsonl"
    backup = Path(state_io.previous_generation_path(str(primary)))
    primary.write_text(
        json.dumps({"s": "BTCUSDC", "i": [now_ms, 65000.0]}, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    backup.write_text(
        json.dumps({"s": "BTCUSDC", "i": [now_ms - 5000, 60000.0]}, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    real_from_offset = state_io.read_jsonl_bucket_map_from_offset
    calls = {"n": 0}

    def _flaky_from_offset(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("transient primary read failure")
        return real_from_offset(*args, **kwargs)

    monkeypatch.setattr(state_io, "read_jsonl_bucket_map_from_offset", _flaky_from_offset)
    memo: dict = {}

    # Call 1: transient failure falls back to .previous (60000.0) without poisoning memo under primary's stat sig.
    doc1 = state_io.read_jsonl_or_json_dict_cached(str(primary), memo)
    assert doc1 == {"items": {"BTCUSDC": [[now_ms - 5000, 60000.0]]}}
    assert str(primary) not in memo

    # Call 2: primary stat sig is unchanged, yet fresh primary (65000.0) is read cleanly (not stuck on 60000.0).
    doc2 = state_io.read_jsonl_or_json_dict_cached(str(primary), memo)
    assert doc2 == {"items": {"BTCUSDC": [[now_ms, 65000.0]]}}
    assert str(primary) in memo


def test_pass14_ephemeral_staging_cleanup_throttle_and_append_aware_rsync_invariants(tmp_path, monkeypatch):
    """Pass #14 invariants & cross-checks: ephemeral fsync=False staging sweeps throttle redundant listdir calls
    (while durable fsync=True sweeps stay immediate), and rsync_cachedb_append_aware handles grow, shrink, and
    same-size inode-replacement rewrites cleanly."""
    import subprocess
    from cachepaths import STALE_STAGING_MAX_AGE_SEC

    # 1. Pure invariants: ephemeral fsync=False writes sweep on first call, throttle within 30s, then sweep again;
    #    durable fsync=True writes always sweep immediately (preserving orphan-cleanup guarantees).
    state_io._STAGING_CLEANUP_LAST_MONO.clear()
    mono_clock = [1000.0]
    monkeypatch.setattr(state_io.time, "perf_counter", lambda: mono_clock[0])

    sweep_calls: list[tuple[str, str, bool]] = []
    real_cleanup = state_io.cleanup_stale_staging_siblings

    def _spy_cleanup(directory, prefix, **kwargs):
        sweep_calls.append((directory, prefix, bool(kwargs.get("dirs_only", False))))
        return real_cleanup(directory, prefix, **kwargs)

    monkeypatch.setattr(state_io, "cleanup_stale_staging_siblings", _spy_cleanup)

    eph_p = tmp_path / "cache_current_price.json"
    stale_eph = tmp_path / ".cache_current_price.json.orphan1.tmp"
    stale_eph.write_text("{}", encoding="utf-8")
    old_ts = time.time() - (STALE_STAGING_MAX_AGE_SEC + 60.0)
    os.utime(stale_eph, (old_ts, old_ts))

    # First ephemeral write sweeps immediately and cleans the stale orphan.
    state_io.atomic_write_json(str(eph_p), {"v": 1}, separators=(",", ":"), fsync=False)
    assert not stale_eph.exists()
    assert len(sweep_calls) == 1

    # Next 4 rapid ephemeral writes within 30s must NOT trigger redundant os.listdir sweeps.
    for k in range(2, 6):
        mono_clock[0] += 1.0
        state_io.atomic_write_json(str(eph_p), {"v": k}, separators=(",", ":"), fsync=False)
    assert len(sweep_calls) == 1

    # Once 30s elapses, the next ephemeral write sweeps again and cleans newly aged debris.
    stale_eph2 = tmp_path / ".cache_current_price.json.orphan2.tmp"
    stale_eph2.write_text("{}", encoding="utf-8")
    os.utime(stale_eph2, (old_ts, old_ts))
    mono_clock[0] += state_io._STAGING_CLEANUP_MIN_INTERVAL_SEC + 1.0
    state_io.atomic_write_json(str(eph_p), {"v": 6}, separators=(",", ":"), fsync=False)
    assert not stale_eph2.exists()
    assert len(sweep_calls) == 2

    # Durable fsync=True writes ALWAYS sweep immediately even on rapid consecutive writes.
    dur_p = tmp_path / "shared_durable.json"
    state_io.atomic_write_json(str(dur_p), {"v": 1}, fsync=True)
    stale_dur = tmp_path / ".shared_durable.json.orphan3.tmp"
    stale_dur.write_text("{}", encoding="utf-8")
    os.utime(stale_dur, (old_ts, old_ts))
    state_io.atomic_write_json(str(dur_p), {"v": 2}, fsync=True)
    assert not stale_dur.exists()
    assert len(sweep_calls) == 4

    # 2. Dependent end-to-end cross-check: real bash + rsync execution of rsync_cachedb_append_aware
    repo_root = Path(__file__).resolve().parents[1]
    common_sh = repo_root / "orchestratorOS" / "lib" / "backup_sync_common.sh"
    src = tmp_path / "src_cachedb"
    dst = tmp_path / "dst_cachedb"
    (src / "prices").mkdir(parents=True)
    dst.mkdir(parents=True)
    state_tsv = tmp_path / "archive_inodes.tsv"

    arc365 = src / "prices" / "cache_365_price_BTCUSDC.jsonl"
    arc_sp = src / "prices" / "cache_sparse_price_BTCUSDC.jsonl"
    hot_json = src / "prices" / "cache_current_price.json"
    arc365.write_text('{"s":"BTCUSDC","i":[1,60000.0]}\n', encoding="utf-8")
    arc_sp.write_text('{"s":"BTCUSDC","i":[1,60000.0]}\n', encoding="utf-8")
    hot_json.write_text('{"items":{"BTCUSDC":[[1,60000.0]]}}', encoding="utf-8")

    def _run_append_sync() -> subprocess.CompletedProcess:
        cmd = (
            f'set -euo pipefail; source "{common_sh}"; '
            f'rsync_cachedb_append_aware "{src}/" "{dst}/" "{state_tsv}"'
        )
        res = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, check=False)
        assert res.returncode == 0, f"rsync_cachedb_append_aware failed: {res.stdout}\n{res.stderr}"
        return res

    # Initial sync populates dst and records archive inodes.
    _run_append_sync()
    assert (dst / "prices" / "cache_365_price_BTCUSDC.jsonl").read_text(encoding="utf-8") == arc365.read_text(encoding="utf-8")
    assert (dst / "prices" / "cache_sparse_price_BTCUSDC.jsonl").read_text(encoding="utf-8") == arc_sp.read_text(encoding="utf-8")
    assert state_tsv.is_file() and "prices/cache_365_price_BTCUSDC.jsonl" in state_tsv.read_text(encoding="utf-8")

    # Case A: pure append (same inode, src_size > dst_size) transfers tail via --append cleanly.
    with open(arc365, "a", encoding="utf-8") as fh:
        fh.write('{"s":"BTCUSDC","i":[2,60100.0]}\n')
    _run_append_sync()
    assert (dst / "prices" / "cache_365_price_BTCUSDC.jsonl").read_text(encoding="utf-8") == arc365.read_text(encoding="utf-8")

    # Case B: compaction/shrink (src_size < dst_size) MUST NOT be skipped by --append.
    arc365.write_text('{"s":"BTCUSDC","i":[2,60100.0]}\n', encoding="utf-8")
    _run_append_sync()
    assert (dst / "prices" / "cache_365_price_BTCUSDC.jsonl").read_text(encoding="utf-8") == '{"s":"BTCUSDC","i":[2,60100.0]}\n'

    # Case C: same-second, exact-same-size atomic inode replacement MUST trigger full sync via inode tracker + --ignore-times.
    repl_tmp = src / "prices" / "._replace_sparse.tmp"
    rewritten_sp = '{"s":"BTCUSDC","i":[9,69999.0]}\n'
    assert len(rewritten_sp) == len(arc_sp.read_text(encoding="utf-8"))
    repl_tmp.write_text(rewritten_sp, encoding="utf-8")
    os.replace(str(repl_tmp), str(arc_sp))
    hot_json.write_text('{"items":{"BTCUSDC":[[9,69999.0]]},"seq":2}', encoding="utf-8")
    _run_append_sync()
    assert (dst / "prices" / "cache_sparse_price_BTCUSDC.jsonl").read_text(encoding="utf-8") == rewritten_sp
    assert (dst / "prices" / "cache_current_price.json").read_text(encoding="utf-8") == '{"items":{"BTCUSDC":[[9,69999.0]]},"seq":2}'


def test_pass19_staging_cleanup_jsonl_ts_and_balance_snapshot_invariants(tmp_path):
    """Pass 19 pure invariants: staging cleanup reaps far-future .tmp, JSONL ts_ms='0' does not shadow ts, and balance/asset-value snapshots resist '0'/ms/future ts."""
    import cacheaccount as ca
    import cachepaths as cp

    now_s = time.time()
    future_s = now_s + 7200.0

    # 1. state_io.cleanup_stale_staging_siblings and cachepaths._cleanup_stale_staging_entries reap corrupt far-future .tmp while keeping fresh .tmp
    st_dir = tmp_path / "staging"
    st_dir.mkdir()
    fresh_tmp = st_dir / ".cache_test.json.1.tmp"
    future_tmp = st_dir / ".cache_test.json.2.tmp"
    fresh_tmp.write_text("fresh", encoding="utf-8")
    future_tmp.write_text("corrupt_future", encoding="utf-8")
    os.utime(fresh_tmp, (now_s - 5.0, now_s - 5.0))
    os.utime(future_tmp, (future_s, future_s))

    state_io.cleanup_stale_staging_siblings(str(st_dir), ".cache_test.json.", suffix=".tmp", max_age_sec=300.0)
    assert fresh_tmp.exists()
    assert not future_tmp.exists()

    future_tmp_cp = st_dir / ".legacy.tmp"
    fresh_tmp_cp = st_dir / ".active.tmp"
    future_tmp_cp.write_text("corrupt_future", encoding="utf-8")
    fresh_tmp_cp.write_text("fresh", encoding="utf-8")
    os.utime(future_tmp_cp, (future_s, future_s))
    os.utime(fresh_tmp_cp, (now_s - 5.0, now_s - 5.0))
    cp._clean_stale_staging_in_dir(str(st_dir), min_file_age_sec=300.0)
    assert fresh_tmp_cp.exists()
    assert not future_tmp_cp.exists()

    # 2. _extract_jsonl_record_ts_ms skips truthy "0" in ts_ms and falls back to ts / timestamp
    t_sec = 1_795_000_000.0
    assert state_io._extract_jsonl_record_ts_ms({"ts_ms": "0", "ts": t_sec}) == pytest.approx(t_sec * 1000.0)
    assert state_io._extract_jsonl_record_ts_ms({"ts_ms": "bad", "ts": "0", "timestamp": t_sec * 1000.0}) == pytest.approx(t_sec * 1000.0)
    jsonl_p = tmp_path / "span_test.jsonl"
    jsonl_p.write_text(
        json.dumps({"ts_ms": "0", "ts": t_sec - 120.0, "v": 1}) + "\n"
        + json.dumps({"ts_ms": "0", "ts": t_sec, "v": 2}) + "\n",
        encoding="utf-8",
    )
    recs = state_io.read_jsonl_records_since_span(str(jsonl_p), span_ms=600_000.0)
    assert len(recs) == 2 and recs[-1]["v"] == 2

    # 3. CacheBalanceManager snapshot loading avoids verified_ts="0" shadowing valid ts and normalizes epoch-ms verified_ts
    bal_p = tmp_path / "cache_balance_binance.json"
    bal_p.write_text(
        json.dumps({
            "venue": "binance",
            "verified_ts": "0",
            "ts": t_sec,
            "items": {"BTC": {"free": 1.5, "locked": 0.0}},
        }),
        encoding="utf-8",
    )
    items_out, ver_out, _ = ca.CacheBalanceManager.read_venue_balances_snapshot(str(bal_p))
    assert items_out["BTC"]["ts"] == pytest.approx(t_sec)
    assert ver_out == 0.0

    mgr = ca.CacheBalanceManager(60, ["BTC"], str(bal_p), read_only=True, venue="binance")
    assert mgr.cache["BTC"][0]["ts"] == pytest.approx(t_sec)

    bal_ms_p = tmp_path / "cache_balance_kraken.json"
    bal_ms_p.write_text(
        json.dumps({
            "venue": "kraken",
            "verified_ts": int(t_sec * 1000),
            "items": {"BTC": {"free": 0.5, "locked": 0.0}},
        }),
        encoding="utf-8",
    )
    items_ms, ver_ms, _ = ca.CacheBalanceManager.read_venue_balances_snapshot(str(bal_ms_p))
    assert ver_ms == pytest.approx(t_sec)
    assert items_ms["BTC"]["ts"] == pytest.approx(t_sec)

    # 4. CacheAssetValueManager.extract_asset_value_summary omits age for corrupt far-future records
    av_future = ca.CacheAssetValueManager.extract_asset_value_summary(
        {"t": t_sec + 7200.0, "i": {"total_value_usdc": 20000.0}},
        now_ts=t_sec,
    )
    assert "total_value_usdc_age_min" not in av_future
    av_valid = ca.CacheAssetValueManager.extract_asset_value_summary(
        {"t": t_sec - 120.0, "i": {"total_value_usdc": 20000.0}},
        now_ts=t_sec,
    )
    assert av_valid["total_value_usdc_age_min"] == pytest.approx(2.0)


def test_pass19_shell_scripts_syntax_and_future_timestamp_guard_crosscheck():
    """Pass 19 dependent cross-check: autodeploy, vpn_watchdog, and pia_supervisor shell timestamp guards pass bash -n and reject corrupt future/rollback timestamps."""
    import subprocess
    repo_root = Path(__file__).resolve().parents[1]
    scripts = (
        repo_root / "orchestratorOS" / "admin" / "manage_gitautodeploy.sh",
        repo_root / "orchestratorOS" / "livecheck" / "vpn_watchdog.sh",
        repo_root / "orchestratorOS" / "livecheck" / "pia_supervisor.sh",
    )
    for sc in scripts:
        res = subprocess.run(["bash", "-n", str(sc)], capture_output=True, text=True, check=False)
        assert res.returncode == 0, f"bash -n failed for {sc}: {res.stderr}"

    # Verify extracted guard semantics under corrupt future timestamps, non-numeric values, and clock rollback
    deploy_txt = scripts[0].read_text(encoding="utf-8")
    vpn_txt = scripts[1].read_text(encoding="utf-8")
    pia_txt = scripts[2].read_text(encoding="utf-8")
    assert '[ "$deploy_age" -ge -60 ] && [ "$deploy_age" -lt "$COOLDOWN" ]' in deploy_txt
    assert '[ "$age" -ge -60 ] && [ "$age" -lt "$REINSTALL_COOLDOWN" ]' in vpn_txt
    assert '[ "$NOW" -lt "$NEXT_RETRY" ] && [ $(( NEXT_RETRY - NOW )) -le 7260 ]' in vpn_txt
    assert '[ "$now" -lt "$last_prewarm" ] || [ $((now - last_prewarm)) -ge "$PREWARM_INTERVAL" ]' in pia_txt
    assert '[ "$now" -lt "$last_restore_attempt" ] || [ $((now - last_restore_attempt)) -ge "$current_restore_interval" ]' in pia_txt

    probe = subprocess.run(
        [
            "bash",
            "-c",
            "set -e; "
            "NOW=1795000000; COOLDOWN=900; "
            "last_ts=2000000000; deploy_age=$(( NOW - last_ts )); "
            "! { [ \"$deploy_age\" -ge -60 ] && [ \"$deploy_age\" -lt \"$COOLDOWN\" ]; }; "
            "last_ts=bad_val; [[ \"${last_ts:-}\" =~ ^[0-9]+$ ]] || last_ts=0; deploy_age=$(( NOW - last_ts )); "
            "! { [ \"$deploy_age\" -ge -60 ] && [ \"$deploy_age\" -lt \"$COOLDOWN\" ]; }; "
            "NEXT_RETRY=$(( NOW + 999999 )); "
            "! { [ \"$NOW\" -lt \"$NEXT_RETRY\" ] && [ $(( NEXT_RETRY - NOW )) -le 7260 ]; }; "
            "NEXT_RETRY=$(( NOW + 120 )); "
            "{ [ \"$NOW\" -lt \"$NEXT_RETRY\" ] && [ $(( NEXT_RETRY - NOW )) -le 7260 ]; }; "
            "last_restore_attempt=$(( NOW + 500 )); current_restore_interval=600; "
            "{ [ \"$NOW\" -lt \"$last_restore_attempt\" ] || [ $((NOW - last_restore_attempt)) -ge \"$current_restore_interval\" ]; }",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe.returncode == 0, f"shell guard cross-check failed: {probe.stderr}"





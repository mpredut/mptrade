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
    assert 'single_instance(f"kraken_bot_{args.test_strategy.strip().upper()}")' in kraken
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
    flat_intents = cdb / "autonomous_trade_intents.json"
    flat_intents_prev = Path(previous_generation_path(str(flat_intents)))
    flat_intents.write_text("{corrupt", encoding="utf-8")
    flat_intents_prev.write_text(json.dumps([{"intent_id": "i1"}]), encoding="utf-8")
    ro_intents = intent_state.load_durable_intents_state(str(flat_intents), read_only=True)
    assert ro_intents == [{"intent_id": "i1"}]
    assert flat_intents.read_text(encoding="utf-8") == "{corrupt"
    assert flat_intents_prev.is_file()
    assert not (cdb / "state" / "autonomous_trade_intents.json").exists()
    assert not Path(state_io.generation_lock_path(str(flat_intents))).exists()

    # Writable durable_intents_lock migrates flat file into state/ BEFORE locking so lock and file share state/.
    with intent_state.durable_intents_lock(str(flat_intents)):
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




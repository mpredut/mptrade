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
    for bg_tok in ("manage_logs.sh", "manage_backups.sh", "sync_cache_to.sh dev", "check_cache_coherence.py", "portfolio_snapshot.py"):
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












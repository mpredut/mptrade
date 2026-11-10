import gzip
import json
from pathlib import Path

import pytest

from state_io import (
    StateReadError,
    atomic_file_path,
    atomic_gzip_file,
    atomic_snapshot_file,
    atomic_text_writer,
    atomic_write_bytes,
    atomic_write_json,
    commit_file_and_meta,
    load_json_state,
    meta_file_path,
    previous_generation_path,
    read_recovering_previous,
    save_json_state,
    stage_previous_generation,
    write_meta_json,
)


def test_atomic_write_json_replaces_complete_document(tmp_path):
    path = tmp_path / "nested" / "state.json"
    atomic_write_json(path, {"cycle": 1}, indent=2)
    atomic_write_json(path, {"cycle": 2}, indent=2)
    assert json.loads(path.read_text(encoding="utf-8")) == {"cycle": 2}
    assert list(path.parent.glob("*.tmp")) == []


def test_atomic_writer_fsyncs_directory_after_replace(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    events = []
    real_replace = __import__("os").replace

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


def test_trailing_once_commands_share_daemon_lock():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    core_text = (root / "trailing_core.py").read_text(encoding="utf-8")
    assert "if not args.status:" in core_text
    assert "single_instance(lock_name)" in core_text
    for relative, lock_name in (
        ("binance_api/trailing_stop.py", "binance_trailing"),
        ("kraken/trailing_stop.py", "kraken_trailing"),
    ):
        text = (root / relative).read_text(encoding="utf-8")
        assert "parse_trailing_cli_args(" in text
        assert f'"{lock_name}"' in text


def test_xstock_once_shares_lock_but_isolated_trial_does_not():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    text = (root / "kraken/kraken_xstock_watch.py").read_text(encoding="utf-8")
    assert "if not args.status and not args.trial:" in text
    assert 'single_instance("kraken_xstock_watch")' in text


def test_strategy_test_commands_use_instrument_daemon_locks():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    kraken = (root / "kraken/kraken_bot.py").read_text(encoding="utf-8")
    hyperliquid = (root / "hyperliquid/hl_bot.py").read_text(encoding="utf-8")
    assert 'single_instance(f"kraken_bot_{args.test_strategy.strip()}")' in kraken
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
    import time
    import order_guard
    from providers.kraken_provider import KrakenProvider
    from intelligence.sentiment.guards.high_stake_guard import collect_pretrade_telemetry

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
    import pytest
    with pytest.raises(ValueError):
        atomic_write_json(p, {"nan_val": float("nan")}, allow_nan=False)
    atomic_write_json(p, {"nan_val": float("nan")}, allow_nan=True)
    assert p.exists()


def test_atomic_text_writer_newline(tmp_path):
    import csv
    p = tmp_path / "test.csv"
    with atomic_text_writer(p, newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["a", "b"])
        writer.writerow(["1", "2"])
    content = p.read_bytes()
    assert b"\r\r\n" not in content
    assert b"a,b" in content


def test_atomic_write_json_default(tmp_path):
    import datetime as dt
    from pathlib import Path
    p = tmp_path / "test_default.json"
    data = {"path": Path("/some/path"), "time": dt.datetime(2026, 1, 1, 12, 0, 0)}
    atomic_write_json(p, data, default=str)
    assert p.exists()
    loaded = json.loads(p.read_text(encoding="utf-8"))
    assert "/some/path" in loaded["path"]
    assert "2026-01-01" in loaded["time"]


def test_read_json_dict(tmp_path):
    from state_io import atomic_write_jsonl, read_json_dict, read_json_file, read_jsonl_dicts, read_jsonl_records
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
    from state_io import (
        atomic_write_bytes,
        atomic_write_text,
        read_bytes_file,
        read_csv_dicts,
        read_tail_jsonl_dicts,
        read_tail_jsonl_records,
        read_text_file,
        read_text_lines,
    )

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

    from state_io import (
        discard_generation_file,
        read_json_dict,
        read_json_dict_cached,
        read_jsonl_records,
        read_recovering_previous,
    )
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
    assert not p_cached_prev.exists()
    assert read_json_dict_cached(p_cached, memo) == {"recovered": 42}


def test_stage_previous_generation_and_concurrent_reader_race(tmp_path, monkeypatch):
    import os
    from state_io import read_json_dict_recovering

    data_path = str(tmp_path / "cache_current_price.json")
    atomic_write_json(data_path, {"items": {"BTCUSDC": 70000.0}})

    # 1. While writer has staged .previous via hardlink (before replacing data_path),
    # a concurrent reader calling read_json_dict_recovering must not unlink .previous.
    backup_path = stage_previous_generation(data_path)
    assert os.path.exists(backup_path)
    assert os.path.samefile(data_path, backup_path)
    assert read_json_dict_recovering(data_path) == {"items": {"BTCUSDC": 70000.0}}
    assert os.path.exists(backup_path)

    # Once data_path is replaced with a new inode and left with a stale .previous,
    # a subsequent read cleans up the stale .previous.
    atomic_write_json(data_path, {"items": {"BTCUSDC": 70100.0}})
    assert not os.path.samefile(data_path, backup_path)
    assert read_json_dict_recovering(data_path) == {"items": {"BTCUSDC": 70100.0}}
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
    import pytest
    import state_io as sio
    # 4. state_io: read_tail_lines, truncate_file_bytes, and prune_rotated_archives bool/zero protection
    tail_file = tmp_path / "tail_test.txt"
    tail_file.write_text("line1\nline2\nline3\n", encoding="utf-8")
    assert sio.read_tail_lines(str(tail_file), 2, chunk_size=0) == [b"line2", b"line3"]
    assert sio.read_tail_lines(str(tail_file), 2, chunk_size=True) == [b"line2", b"line3"]
    with pytest.raises(ValueError):
        sio.truncate_file_bytes(str(tail_file), True)
    with pytest.raises(ValueError):
        sio.truncate_file_bytes(str(tail_file), -1)

def test_jsonstatestore_save_rejects_non_dict_state_and_state_io_iter64_s2(tmp_path):
    """[ITER64 #2] JsonStateStore.save rejects non-dict state and state_io.read_tail_lines guards directory/None limit."""
    import pytest
    from state_io import read_tail_lines
    # 2. JsonStateStore.save rejects non-dict state and state_io.read_tail_lines guards directory/None limit
    from strategies.state_store import JsonStateStore, StatePersistenceError
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
    import pytest
    import state_io
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
    import state_io
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
    import state_io
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
    import state_io
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
    import json
    import state_io
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
    import state_io
    # 1. state_io.atomic_write_jsonl writes iterable of dicts atomically and matches read_jsonl_dicts
    p_jsonl = tmp_path / "orders.jsonl"
    data = [{"order_id": 1, "sym": "BTCUSDC"}, {"order_id": 2, "sym": "ETHUSDC"}]
    state_io.atomic_write_jsonl(str(p_jsonl), data)
    assert state_io.read_jsonl_dicts(str(p_jsonl)) == data

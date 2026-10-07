import json

import pytest

from state_io import (
    StateReadError, atomic_snapshot_file, atomic_text_writer,
    atomic_write_json, load_json_state,
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
    for relative, lock_name in (
        ("binance_api/trailing_stop.py", "binance_trailing"),
        ("kraken/trailing_stop.py", "kraken_trailing"),
    ):
        text = (root / relative).read_text(encoding="utf-8")
        assert "if not args.status:" in text
        assert f'single_instance("{lock_name}")' in text


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
    assert 'single_instance(f"hl_bot_{args.test_strategy.strip()}")' in hyperliquid
    assert 'single_instance(f"hl_bot_{token}")' in hyperliquid


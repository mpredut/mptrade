import cacheManager as cm
import datetime
import json
import os
import pytest
import tradeall
from matplotlib import pyplot as plt
from pathlib import Path
from state_io import read_tail_lines
from tests.conftest import TempDirTestCase
from types import SimpleNamespace
from unittest.mock import Mock, patch
from visuals import tradeall_observe as observe


@pytest.fixture(autouse=True)
def _reset_observer_process_caches():
    observe._PIPE_LOG_CACHE.clear()
    observe._PRICE_FILE_CACHE.clear()
    observe._LAST_SAMPLED_TS.clear()
    yield
    observe._PIPE_LOG_CACHE.clear()
    observe._PRICE_FILE_CACHE.clear()
    observe._LAST_SAMPLED_TS.clear()


def test_pipe_log_cache_is_bounded_and_forgets_removed_files(tmp_path, monkeypatch):
    monkeypatch.setattr(observe, "_PIPE_LOG_CACHE_MAX_FILES", 2)
    paths = []
    for index in range(3):
        path = tmp_path / f"events-{index}.log"
        path.write_text(f"{index}|value\n", encoding="utf-8")
        paths.append(path)
        assert observe._read_pipe_log(path, 2) == [[str(index), "value"]]

    assert len(observe._PIPE_LOG_CACHE) == 2
    assert paths[0] not in observe._PIPE_LOG_CACHE

    paths[1].unlink()
    assert observe._read_pipe_log(paths[1], 2) == []
    assert paths[1] not in observe._PIPE_LOG_CACHE


def test_pipe_log_cache_detects_same_size_rewrite_and_rotation(tmp_path):
    path = tmp_path / "events.log"
    path.write_text("a|b\n", encoding="utf-8")
    assert observe._read_pipe_log(path, 2) == [["a", "b"]]

    previous = path.stat()
    path.write_text("c|d\n", encoding="utf-8")
    os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns + 1_000_000))
    assert observe._read_pipe_log(path, 2) == [["c", "d"]]

    replacement = tmp_path / "replacement.log"
    replacement.write_text("e|f\n", encoding="utf-8")
    os.replace(replacement, path)
    assert observe._read_pipe_log(path, 2) == [["e", "f"]]


def test_price_sampling_deduplicates_an_unchanged_snapshot(tmp_path, monkeypatch):
    cache_path = tmp_path / "cache_short_trend.json"
    logger_dir = tmp_path / "logger"
    monkeypatch.setattr(observe, "CACHE_TREND_PATH", str(cache_path))
    monkeypatch.setattr(observe, "LOGGER_DIR", str(logger_dir))

    cache_path.write_text(
        json.dumps({"BTCUSDC": {"ts": 123.0, "current_price": 42.5}}),
        encoding="utf-8",
    )
    observe.sample_current_prices(["BTCUSDC"])
    observe.sample_current_prices(["BTCUSDC"])

    log_path = next(logger_dir.glob("tradeall_price_samples_*.log"))
    assert log_path.read_text(encoding="utf-8").splitlines() == ["123.0|BTCUSDC|42.5"]

    cache_path.write_text(
        json.dumps({"BTCUSDC": {"ts": 124.0, "current_price": 43.0}}),
        encoding="utf-8",
    )
    observe.sample_current_prices(["BTCUSDC"])
    assert len(log_path.read_text(encoding="utf-8").splitlines()) == 2


def test_price_sampling_uses_fresher_ts_fast_when_ts_is_unchanged(tmp_path, monkeypatch):
    cache_path = tmp_path / "cache_short_trend.json"
    logger_dir = tmp_path / "logger"
    monkeypatch.setattr(observe, "CACHE_TREND_PATH", str(cache_path))
    monkeypatch.setattr(observe, "LOGGER_DIR", str(logger_dir))

    cache_path.write_text(
        json.dumps({"BTCUSDC": {"ts": 123.0, "ts_fast": 123.5, "current_price": 42.5}}),
        encoding="utf-8",
    )
    observe.sample_current_prices(["BTCUSDC"])

    cache_path.write_text(
        json.dumps({"BTCUSDC": {"ts": 123.0, "ts_fast": 125.0, "current_price": 43.2}}),
        encoding="utf-8",
    )
    observe.sample_current_prices(["BTCUSDC"])

    log_path = next(logger_dir.glob("tradeall_price_samples_*.log"))
    assert log_path.read_text(encoding="utf-8").splitlines() == [
        "123.5|BTCUSDC|42.5",
        "125.0|BTCUSDC|43.2",
    ]


def test_price_sampling_skips_corrupt_timestamp_instead_of_fabricating_now(tmp_path, monkeypatch):
    cache_path = tmp_path / "cache_short_trend.json"
    logger_dir = tmp_path / "logger"
    monkeypatch.setattr(observe, "CACHE_TREND_PATH", str(cache_path))
    monkeypatch.setattr(observe, "LOGGER_DIR", str(logger_dir))

    cache_path.write_text(
        json.dumps({"BTCUSDC": {"ts": "corrupt", "current_price": 42.5}}),
        encoding="utf-8",
    )
    observe.sample_current_prices(["BTCUSDC"])
    assert not logger_dir.exists()


def test_history_tail_keeps_first_record_when_file_is_smaller_than_tail(tmp_path, monkeypatch):
    cachedb = tmp_path / "cachedb"
    cachedb.mkdir(parents=True, exist_ok=True)
    history = cachedb / "cache_sparse_price_BTCUSDC.jsonl"
    history.write_text(
        '\n'.join(
            [
                json.dumps({"s": "BTCUSDC", "i": [1_000, 10.0]}),
                json.dumps({"s": "BTCUSDC", "i": [2_000, 11.0]}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(observe, "ROOT", str(tmp_path))

    assert observe._load_cachedb_price_entries(observe.cache_sparse_price_filename("BTCUSDC"), "BTCUSDC", max_bytes=4096) == [
        [1_000, 10.0],
        [2_000, 11.0],
    ]


def test_live_dense_price_has_priority_over_archived_sources(monkeypatch):
    monkeypatch.setattr(observe, "load_price_samples", lambda _symbol, _days: ([1.0], [3.0]))

    def cached_entries(filename, _symbol):
        if filename.startswith("cache_sparse_price_"):
            return [[1_000, 1.0]]
        if filename.startswith("cache_365_price_"):
            return [[1_000, 2.0]]
        if filename.startswith("cache_24_price_") and filename.endswith(".jsonl"):
            return [[1_000, 4.0]]
        return []

    monkeypatch.setattr(observe, "_load_cachedb_price_entries", cached_entries)

    assert observe.load_price_series_live("BTCUSDC", 9, include_history=True) == ([1], [4.0])


def test_live_dense_price_falls_back_to_legacy_cache24_json(monkeypatch):
    monkeypatch.setattr(observe, "load_price_samples", lambda _symbol, _days: ([], []))

    def cached_entries(filename, _symbol):
        if filename == "cache_24_price_BTCUSDC.json":
            return [[2_000, 5.5]]
        return []

    monkeypatch.setattr(observe, "_load_cachedb_price_entries", cached_entries)

    assert observe.load_price_series_live("BTCUSDC", 2, include_history=False) == ([2], [5.5])


def test_cache24_jsonl_uses_larger_tail_budget_than_long_archives(tmp_path, monkeypatch):
    cachedb = tmp_path / "cachedb"
    cachedb.mkdir(parents=True, exist_ok=True)
    for fname in ("cache_24_price_BTCUSDC.jsonl", "cache_365_price_BTCUSDC.jsonl"):
        (cachedb / fname).write_text(json.dumps({"s": "BTCUSDC", "i": [1_000, 10.0]}) + "\n", encoding="utf-8")

    observed_budgets = {}

    def fake_tail(path, limit=None, max_bytes=None):
        observed_budgets[os.path.basename(path)] = max_bytes
        return [{"s": "BTCUSDC", "i": [1_000, 10.0]}]

    monkeypatch.setattr(observe, "read_tail_jsonl_dicts", fake_tail)

    assert observe._load_cachedb_price_entries("cache_24_price_BTCUSDC.jsonl", "BTCUSDC") == [[1_000, 10.0]]
    assert observe._load_cachedb_price_entries("cache_365_price_BTCUSDC.jsonl", "BTCUSDC") == [[1_000, 10.0]]
    assert observed_budgets["cache_24_price_BTCUSDC.jsonl"] == observe.CACHE24_TAIL_MAX_BYTES
    assert observed_budgets["cache_365_price_BTCUSDC.jsonl"] == observe.DEFAULT_HISTORY_TAIL_MAX_BYTES
    assert observe.CACHE24_TAIL_MAX_BYTES > observe.DEFAULT_HISTORY_TAIL_MAX_BYTES


def test_default_symbols_include_multivenue_tradeall_fire_instruments():
    default_syms = observe._resolve_default_symbols()
    assert "BTCUSDC" in default_syms
    assert "HYPEUSDC" in default_syms


def test_price_file_cache_tails_jsonl_incrementally_and_reloads_on_replace(tmp_path, monkeypatch):
    cachedb = tmp_path / "cachedb"
    cachedb.mkdir(parents=True, exist_ok=True)
    jsonl = cachedb / "cache_24_price_BTCUSDC.jsonl"
    jsonl.write_text(
        "\n".join([
            json.dumps({"s": "BTCUSDC", "i": [1_000, 10.0]}),
            json.dumps({"s": "BTCUSDC", "i": [2_000, 11.0]}),
        ]) + "\n",
        encoding="utf-8",
    )

    tail_calls = []
    orig_offset = observe.read_jsonl_bucket_map_from_offset

    def _spy_offset(*args, **kwargs):
        res = orig_offset(*args, **kwargs)
        tail_calls.append((kwargs.get("start_offset", 0), res[5]))
        return res

    monkeypatch.setattr(observe, "read_jsonl_bucket_map_from_offset", _spy_offset)

    # Cold read populates cache + cursor.
    assert observe._load_cachedb_price_entries("cache_24_price_BTCUSDC.jsonl", "BTCUSDC") == [
        [1_000, 10.0],
        [2_000, 11.0],
    ]
    assert tail_calls == []

    # Unchanged file hits stat cache with zero incremental or full reads.
    assert observe._load_cachedb_price_entries("cache_24_price_BTCUSDC.jsonl", "BTCUSDC") == [
        [1_000, 10.0],
        [2_000, 11.0],
    ]
    assert tail_calls == []

    # Appended line on same inode uses incremental byte-offset tailing.
    with open(jsonl, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"s": "BTCUSDC", "i": [3_000, 12.5]}) + "\n")
    assert observe._load_cachedb_price_entries("cache_24_price_BTCUSDC.jsonl", "BTCUSDC") == [
        [1_000, 10.0],
        [2_000, 11.0],
        [3_000, 12.5],
    ]
    assert len(tail_calls) == 1 and tail_calls[0][1] is True

    # Atomic compaction (new inode) triggers clean reload.
    replacement = cachedb / "replacement.jsonl"
    replacement.write_text(json.dumps({"s": "BTCUSDC", "i": [4_000, 15.0]}) + "\n", encoding="utf-8")
    os.replace(replacement, jsonl)
    assert observe._load_cachedb_price_entries("cache_24_price_BTCUSDC.jsonl", "BTCUSDC") == [[4_000, 15.0]]


def test_ingest_cachedb_points_rejects_bool_nan_and_nonpositive_values():
    pts = {}
    observe._ingest_cachedb_points(
        pts,
        [
            [True, 10.0],
            [1_000, False],
            [1_000, float("nan")],
            [float("inf"), 10.0],
            [-1_000, 10.0],
            [2_000, 20],
            ["3000", "30.5"],
        ],
    )
    assert pts == {2: 20.0, 3: 30.5}


def test_order_markers_are_batched_by_side():
    ax = Mock()
    events = [
        {
            "ts": float(index),
            "side": "BUY" if index % 2 == 0 else "SELL",
            "price": 100.0 + index,
            "reason": "guard",
        }
        for index in range(10_000)
    ]

    observe._plot_order_markers(ax, events, filled=False)

    assert ax.scatter.call_count == 2
    assert sum(len(call.args[0]) for call in ax.scatter.call_args_list) == len(events)
    ax.annotate.assert_not_called()


def test_live_windows_share_one_loaded_snapshot(monkeypatch, tmp_path):
    load_prices = Mock(return_value=([1.0], [100.0]))
    load_trends = Mock(return_value=[])
    load_orders = Mock(return_value=[])
    load_shadow = Mock(return_value=[])
    render = Mock()
    monkeypatch.setattr(observe, "load_price_series_live", load_prices)
    monkeypatch.setattr(observe, "load_trend_starts", load_trends)
    monkeypatch.setattr(observe, "load_order_events", load_orders)
    monkeypatch.setattr(observe, "load_shadow_events", load_shadow)
    monkeypatch.setattr(observe, "render_chart", render)
    specs = [
        ("live", 3_600, tmp_path / "live.png"),
        ("day", observe.DAY_SECONDS, tmp_path / "day.png"),
        ("week", observe.WEEK_SECONDS, tmp_path / "week.png"),
    ]

    observe.render_symbol_charts_live("BTCUSDC", specs, window_end=1_000_000.0)

    load_prices.assert_called_once_with("BTCUSDC", 9, include_history=True)
    load_trends.assert_called_once_with("BTCUSDC", 9)
    load_orders.assert_called_once_with("BTCUSDC", 9)
    load_shadow.assert_called_once_with("BTCUSDC", 9)
    assert render.call_count == 3


def test_atomic_figure_publish_replaces_complete_file_and_cleans_temp(tmp_path):
    target = tmp_path / "chart.png"
    target.write_bytes(b"old")
    fig = SimpleNamespace(savefig=lambda path, **_kwargs: Path(path).write_bytes(b"complete-png"))
    observe._save_figure_atomic(fig, target, dpi=110)

    assert target.read_bytes() == b"complete-png"
    assert list(tmp_path.glob(".tradeall_observe_*.png")) == []


def test_atomic_figure_publish_preserves_previous_file_on_failure(tmp_path):
    target = tmp_path / "chart.png"
    target.write_bytes(b"old")

    def _broken_savefig(path, **_kwargs):
        Path(path).write_bytes(b"partial")
        raise RuntimeError("render failed")

    with pytest.raises(RuntimeError, match="render failed"):
        observe._save_figure_atomic(SimpleNamespace(savefig=_broken_savefig), target)

    assert target.read_bytes() == b"old"
    assert list(tmp_path.glob(".tradeall_observe_*.png")) == []


def test_render_failure_does_not_leak_matplotlib_figures(tmp_path, monkeypatch):
    before = set(plt.get_fignums())
    monkeypatch.setattr(
        observe,
        "_plot_order_markers",
        Mock(side_effect=RuntimeError("bad event")),
    )

    with pytest.raises(RuntimeError, match="bad event"):
        observe.render_chart(
            "BTCUSDC",
            "test",
            1.0,
            2.0,
            [],
            [],
            [],
            [],
            tmp_path / "chart.png",
        )

    assert set(plt.get_fignums()) == before


def test_load_order_events_recognizes_tradeall_via_market_api_and_filters_other_callers(tmp_path, monkeypatch):
    today = datetime.datetime.now().date()
    log_file = tmp_path / f"order_outcomes_{today.isoformat()}.log"
    monkeypatch.setattr(observe, "LOGGER_DIR", str(tmp_path))

    rows = [
        # ts|symbol|side|price|qty|outcome|refuse_reason|caller|motivation[|regime|...]
        "100.0|BTCUSDC|BUY|85000.0|0.1|accepted||tradeall.py|trend_confirmed_up",
        "200.0|BTCUSDC|BUY|85100.0|0.1|refused|deferred|market_api.py|kalman_primary_up",
        "300.0|BTCUSDC|SELL|85500.0|0.1|accepted||market_api.py|consistent_or_old_down|BEAR|0.6200|live|True|False|ok",
        "350.0|BTCUSDC|SELL|85400.0|0.1|accepted||market_api.py|small_window_plunge_down",
        "360.0|BTCUSDC|SELL|85350.0|0.1|accepted||market_api.py|accel_plunge_down",
        "370.0|BTCUSDC|BUY|84900.0|0.1|accepted||market_api.py|slope<=-5.1_and_old_down",
        "380.0|BTCUSDC|BUY|84950.0||refused|kalman_gate_strict(trend=0)||trend_confirmed_up",
        "390.0|BTCUSDC|SELL|85450.0|0.1|unknown|submit_timeout|order_retry_worker.py|trend_confirmed_down",
        "400.0|BTCUSDC|BUY|85200.0|0.1|refused|error|market_api.py|rtrade_pair_quote",
        "500.0|BTCUSDC|SELL|85300.0|0.1|accepted||monitortrades.py|monitortrades_exit",
        "600.0|TAOUSDC|BUY|300.0|1.0|accepted||tradeall.py|trend_confirmed_up",
        "700.0|SPCXXUSD|BUY|125.0|1.0|accepted||tradeall.py|trend_confirmed_up",
    ]
    log_file.write_text("\n".join(rows) + "\n", encoding="utf-8")

    btc_events = observe.load_order_events("BTCUSDC", 1)
    assert len(btc_events) == 8
    assert [e["ts"] for e in btc_events] == [100.0, 200.0, 300.0, 350.0, 360.0, 370.0, 380.0, 390.0]
    assert [e["reason"] for e in btc_events] == [
        "trend_confirmed_up",
        "kalman_primary_up",
        "consistent_or_old_down",
        "small_window_plunge_down",
        "accel_plunge_down",
        "slope<=-5.1_and_old_down",
        "trend_confirmed_up",
        "trend_confirmed_down",
    ]
    assert btc_events[2]["regime"] == "BEAR"
    assert btc_events[6]["refuse_reason"] == "kalman_gate_strict(trend=0)"
    assert btc_events[7]["outcome"] == "unknown"

    tao_events = observe.load_order_events("TAOUSDC", 1)
    assert len(tao_events) == 1
    assert tao_events[0]["ts"] == 600.0

    spcx_events = observe.load_order_events("SPCXxUSD", 1)
    assert len(spcx_events) == 1
    assert spcx_events[0]["ts"] == 700.0


def test_format_state_text_with_fast_trend_pending_and_exposure():
    entry = {
        "current_price": 312.6,
        "final_trend": -1,
        "gradient_recent": -0.005,
        "slope_small": 0.0,
        "slope_big": 0.0,
        "epsilon": 0.038,
        "trend_fast": 1,
        "gradient_recent_fast": 0.0025,
        "kalman_trend": 0,
        "kalman_vel": 0.015,
        "kalman_vel_std": 0.12,
        "vel_90s_pct_min": 0.042,
        "accel_90s_pct_min2": 0.055,
        "vel_5m_pct_min": -0.011,
        "accel_5m_pct_min2": -0.002,
    }
    text = observe.format_state_text(
        entry, "HEADER NOW",
        pending="BUY $1482.45",
        exposure="2.126 ($650.00)",
    )
    assert "HEADER NOW" in text
    assert "trend:     DOWN" in text
    assert "fast:      UP (grad: +0.0025)" in text
    assert "kin 90s:   v=+0.042%/m a=+0.055%/m2" in text
    assert "kin 5m:    v=-0.011%/m a=-0.002%/m2" in text
    assert "slope short:+0.000" in text
    assert "slope long: +0.000" in text
    assert "pending:   BUY $1482.45" in text
    assert "position:  2.126 ($650.00)" in text


def test_write_html_generates_summary_bar(tmp_path, monkeypatch):
    monkeypatch.setattr(observe, "LIVE_OUT_DIR", str(tmp_path))
    active_summary = {
        "pending_notional": {"total_buy": 1500.0, "total_sell": 2500.0},
        "net_exposures": {"total_cost": 3000.0},
    }
    observe.write_html(["BTCUSDC"], live_minutes=60, active_summary=active_summary, refresh_ms=5000)
    html_file = tmp_path / "tradeall_live.html"
    assert html_file.exists()
    content = html_file.read_text(encoding="utf-8")
    assert "Fleet Pending BUY:</strong> $1500.00" in content
    assert "Fleet Pending SELL:</strong> $2500.00" in content
    assert "Open Cost Basis:</strong> $3000.00" in content
    assert "setInterval(bust, 5000);" in content
    assert "Day" in content
    assert "Week" in content


def test_render_chart_handles_unknown_order_outcomes(tmp_path):
    out = tmp_path / "chart.png"
    observe.render_chart(
        "BTCUSDC",
        "LIVE",
        1000.0,
        2000.0,
        [1100.0, 1500.0, 1900.0],
        [60000.0, 60100.0, 60050.0],
        [{"ts": 1200.0, "state": "UP", "old_state": "HOLD", "price": 60020.0, "prev_confirm_count": "2"}],
        [
            {"ts": 1300.0, "side": "BUY", "price": 60040.0, "outcome": "refused",
             "reason": "trend_confirmed_up", "refuse_reason": "kalman_gate_strict(trend=0)"},
            {"ts": 1600.0, "side": "SELL", "price": 60080.0, "outcome": "unknown",
             "reason": "trend_confirmed_down", "refuse_reason": "submit_timeout"},
        ],
        str(out),
    )
    assert out.exists() and out.stat().st_size > 0


def test_kalman_gate_blocks_handles_unmapped_symbols():
    blocked, mode, trend = tradeall._kalman_gate_blocks("NON_EXISTENT_COIN", "BUY")
    assert blocked is False
    assert mode == "off"
    assert trend is None


# ---------------------------------------------------------------------------
# Cache365PriceManager long-archive persistence & maintenance tests
# ---------------------------------------------------------------------------


class TestArchiverLifecycle(TempDirTestCase):
    def test_jsonl_restart_loads_existing_history_and_appends_only_new_tick(self):
        path = os.path.join(self.tmp, "restart.jsonl")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"s": "BTCUSDC", "i": [1000, 10.0]}) + "\n")
        manager = cm.Cache365PriceManager(60, ["BTCUSDC"], path)
        manager.KEEP_HOURS = 10**9
        manager.on_price_update("BTCUSDC", 2000, 11.0)
        manager.save_state_to_file()
        with open(path, encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        self.assertEqual(rows, [
            {"s": "BTCUSDC", "i": [1000, 10.0]},
            {"s": "BTCUSDC", "i": [2000, 11.0]},
        ])

    def test_long_archive_trimming_never_compacts_from_the_memory_tail(self):
        path = os.path.join(self.tmp, "trim.jsonl")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"s": "BTCUSDC", "i": [1, 10.0]}) + "\n")
        manager = cm.Cache365PriceManager(60, ["BTCUSDC"], path)
        manager.KEEP_HOURS = 1
        manager.cache = {"BTCUSDC": [[1, 10.0]]}
        manager._last_long_trim = 100.0
        with patch("cacheManager.time.monotonic", side_effect=[101.0, 100.0 + 86401]), \
                patch.object(manager, "compact_jsonl") as compact:
            manager._trim_old_data("BTCUSDC")
            compact.assert_not_called()
            manager._trim_old_data("BTCUSDC")
            compact.assert_not_called()

    def test_long_archive_memory_is_bounded_without_truncating_disk_history(self):
        path = os.path.join(self.tmp, "bounded.jsonl")
        total = cm.CM_LONG_ARCHIVE_MEMORY_ROWS + 25
        with open(path, "w", encoding="utf-8") as handle:
            for index in range(total):
                handle.write(json.dumps({"s": "BTCUSDC", "i": [index + 1, 10.0]}) + "\n")
        manager = cm.Cache365PriceManager(60, ["BTCUSDC"], path)
        manager.KEEP_HOURS = 10**9
        self.assertEqual(len(manager.cache["BTCUSDC"]), cm.CM_LONG_ARCHIVE_MEMORY_ROWS)
        manager.on_price_update("BTCUSDC", total + 1, 11.0)
        manager.save_state_to_file()
        with open(path, encoding="utf-8") as handle:
            self.assertEqual(sum(1 for line in handle if line.strip()), total + 1)

    def test_legacy_json_migration_succeeds_for_cache365_price_manager(self):
        legacy_json = os.path.join(self.tmp, "cache_365_price_BTCUSDC.json")
        target_jsonl = os.path.join(self.tmp, "cache_365_price_BTCUSDC.jsonl")
        with open(legacy_json, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "items": {"BTCUSDC": [[1000, 10.0], [2000, 11.0]]},
                    "fetchtime": {"BTCUSDC": 2000},
                },
                handle,
            )
        manager = cm.Cache365PriceManager(60, ["BTCUSDC"], target_jsonl)
        self.assertTrue(os.path.exists(target_jsonl))
        self.assertEqual(manager.cache.get("BTCUSDC"), [[1000, 10.0], [2000, 11.0]])
        with open(target_jsonl, encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        self.assertEqual(
            rows,
            [
                {"s": "BTCUSDC", "i": [1000, 10.0]},
                {"s": "BTCUSDC", "i": [2000, 11.0]},
            ],
        )

    def test_cache365_rotation_failure_restores_active_file_and_memory(self):
        target_jsonl = os.path.join(self.tmp, "cache_365_price_BTCUSDC.jsonl")
        now_ms = int(cm.time.time() * 1000)
        with open(target_jsonl, "w", encoding="utf-8") as handle:
            for i in range(20):
                handle.write(
                    json.dumps({"s": "BTCUSDC", "i": [now_ms + i, 10.0 + i]}) + "\n"
                )
        manager = cm.Cache365PriceManager(60, ["BTCUSDC"], target_jsonl)
        manager.save_state = True
        manager.MAX_FILE_BYTES = 1
        manager.ROTATE_KEEP_FRACTION = 0.25
        with patch.object(manager, "_write_meta", return_value=False):
            self.assertFalse(manager.maintain_append_persist())
        self.assertTrue(os.path.exists(target_jsonl))
        with open(target_jsonl, encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        self.assertEqual(len(rows), 20)
        self.assertEqual(len(manager.cache["BTCUSDC"]), 20)
        self.assertEqual(manager._persisted_counts.get("BTCUSDC"), 20)
        self.assertEqual(
            [f for f in os.listdir(self.tmp) if ".archive" in f],
            [],
        )

    def test_cache365_maintenance_preserves_unpersisted_memory_tail_and_fetchtime(self):
        target_jsonl = os.path.join(self.tmp, "cache_365_price_BTCUSDC.jsonl")
        now_ms = int(cm.time.time() * 1000)
        with open(target_jsonl, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps({"s": "BTCUSDC", "i": [now_ms - 2000, 10.0]}) + "\n"
            )
        manager = cm.Cache365PriceManager(60, ["BTCUSDC"], target_jsonl)
        manager.save_state = True
        manager.KEEP_HOURS = 10**9
        manager.on_price_update("BTCUSDC", now_ms, 12.5)
        self.assertTrue(manager.maintain_append_persist())
        self.assertEqual(
            manager.cache.get("BTCUSDC"),
            [[now_ms - 2000, 10.0], [now_ms, 12.5]],
        )
        self.assertEqual(manager._persisted_counts.get("BTCUSDC"), 2)
        self.assertGreaterEqual(manager.fetchtime_per_symbol.get("BTCUSDC", 0), now_ms)
        with open(target_jsonl, encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        self.assertEqual(
            rows,
            [
                {"s": "BTCUSDC", "i": [now_ms - 2000, 10.0]},
                {"s": "BTCUSDC", "i": [now_ms, 12.5]},
            ],
        )
        manager.on_price_update("BTCUSDC", now_ms + 1000, 15.0)
        self.assertTrue(manager._rotate_disk_archive())
        self.assertEqual(manager.cache.get("BTCUSDC")[-1], [now_ms + 1000, 15.0])
        self.assertEqual(manager._persisted_counts.get("BTCUSDC"), len(manager.cache.get("BTCUSDC", [])))
        with open(target_jsonl + ".meta", "w", encoding="utf-8") as handle:
            handle.write("[]")
        manager._load_jsonl()
        self.assertGreaterEqual(manager.fetchtime_per_symbol.get("BTCUSDC", 0), now_ms + 1000)
        self.assertEqual(read_tail_lines(target_jsonl, True), [])
        with open(target_jsonl, "a", encoding="utf-8") as handle:
            handle.write("\n\n")
        tail_rows = read_tail_lines(target_jsonl, 1, chunk_size=7)
        self.assertEqual(len(tail_rows), 1)
        self.assertEqual(json.loads(tail_rows[0])["i"], [now_ms + 1000, 15.0])

    def test_cache365_maintenance_streams_unlocked_and_merges_concurrent_ws_ticks(self):
        target_jsonl = os.path.join(self.tmp, "cache_365_price_BTCUSDC.jsonl")
        self.assertEqual(
            cm.Cache365PriceManager.KEEP_HOURS,
            cm.CM_LONG_ARCHIVE_MONTHS * 30 * 24,
        )
        self.assertEqual(
            cm.Cache365PriceManager.RETENTION_DAYS,
            cm.CM_LONG_ARCHIVE_MONTHS * 30,
        )
        now_ms = int(cm.time.time() * 1000)
        with open(target_jsonl, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps({"s": "BTCUSDC", "i": [now_ms - 2000, 10.0]}) + "\n"
            )
        manager = cm.Cache365PriceManager(60, ["BTCUSDC"], target_jsonl)
        manager.save_state = True
        real_snapshot = cm.atomic_snapshot_file
        ws_lock_acquired = []

        def _snapshot_with_concurrent_ws(src, dst):
            real_snapshot(src, dst)
            # Simulate the WebSocket thread delivering a tick while disk I/O runs.
            # Because maintain_append_persist releases self.lock during disk I/O,
            # another thread can acquire manager.lock immediately without blocking.
            def _ws_tick():
                acquired = manager.lock.acquire(blocking=False)
                ws_lock_acquired.append(acquired)
                if acquired:
                    manager.lock.release()
                manager.on_price_update("BTCUSDC", now_ms + 500, 14.0)

            t = cm.threading.Thread(target=_ws_tick)
            t.start()
            t.join(timeout=2.0)

        with patch("cacheManager.atomic_snapshot_file", side_effect=_snapshot_with_concurrent_ws):
            self.assertTrue(manager.maintain_append_persist())

        self.assertEqual(ws_lock_acquired, [True])
        self.assertEqual(
            manager.cache.get("BTCUSDC"),
            [[now_ms - 2000, 10.0], [now_ms + 500, 14.0]],
        )
        with open(target_jsonl, encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        self.assertEqual(
            rows,
            [
                {"s": "BTCUSDC", "i": [now_ms - 2000, 10.0]},
                {"s": "BTCUSDC", "i": [now_ms + 500, 14.0]},
            ],
        )


def test_trend_coordinator_run_reports_degraded_error_and_recovery_heartbeats(monkeypatch):
    """Invariant & cross-check: TrendCoordinator.run exposes per-symbol eval failures across idle waits and recovers cleanly."""
    dummy_instant = SimpleNamespace(
        register_price_listener=lambda _cb: None,
        add_listener=lambda _cb: None,
    )
    coord = tradeall.TrendCoordinator(
        symbols=["BTCUSDC", "ETHUSDC"],
        instant_mgr=dummy_instant,
        current_price_mgr=SimpleNamespace(register_price_listener=lambda _cb: None),
        cache24_managers={},
        min_interval=0.01,
        max_interval=0.05,
    )
    phase = {"n": 0}
    heartbeats: list[dict] = []

    def _fake_eval(symbol):
        p = phase["n"]
        coord._mark_evaluated(symbol)
        if p == 0 and symbol == "BTCUSDC":
            raise RuntimeError("btc calc boom")
        if p == 1:
            raise RuntimeError(f"{symbol} down")

    def _fake_hb(name, **kw):
        heartbeats.append({"name": name, **kw})

    wait_calls = {"n": 0}

    def _fake_wait(timeout=None):
        wait_calls["n"] += 1
        phase["n"] += 1
        if phase["n"] >= 3:
            coord.stop()
            return True
        # Force both symbols due again on next cycle
        coord._last_eval.clear()
        return True

    coord._last_eval.clear()
    monkeypatch.setattr(coord, "evaluate", _fake_eval)
    monkeypatch.setattr(tradeall, "touch_module_heartbeat", _fake_hb)
    monkeypatch.setattr(coord._event, "wait", _fake_wait)
    monkeypatch.setattr(coord, "_is_due", lambda sym, now: sym not in coord._last_eval)

    coord.run()

    # Heartbeat sequence: startup ok -> cycle 0 degraded (BTCUSDC failed) -> idle wait still degraded
    # -> cycle 1 error (all failed) -> idle wait still error -> cycle 2 ok (recovered).
    statuses = [(h.get("status", {}).get("status"), h.get("force", False), h.get("status", {}).get("failed_symbols")) for h in heartbeats]
    assert statuses[0] == ("ok", True, None)
    assert statuses[1] == ("degraded", True, ["BTCUSDC"])
    assert statuses[2][0] == "degraded" and statuses[2][2] == ["BTCUSDC"]
    assert statuses[3] == ("error", True, ["BTCUSDC", "ETHUSDC"])
    assert statuses[4][0] == "error" and statuses[4][2] == ["BTCUSDC", "ETHUSDC"]
    assert statuses[5] == ("ok", True, None)


def test_pass22_entry_latest_ts_and_sampling_reject_future_poison_invariant_and_crosscheck(tmp_path, monkeypatch):
    """Pass 22 pure invariant & cross-check: _entry_latest_ts normalizes ms and rejects far-future poison so sample_current_prices never freezes _LAST_SAMPLED_TS."""
    t_now = 1_795_000_000.0
    monkeypatch.setattr(observe.time, "time", lambda: t_now)

    # 1. Pure invariant: far-future ts_fast is ignored in favor of valid ts; epoch-ms normalizes cleanly
    assert observe._entry_latest_ts({"ts": t_now - 5.0, "ts_fast": t_now + 86400.0}, now_ts=t_now) == pytest.approx(t_now - 5.0)
    assert observe._entry_latest_ts({"ts": (t_now - 3.0) * 1000.0}, now_ts=t_now) == pytest.approx(t_now - 3.0)
    assert observe._entry_latest_ts({"ts": t_now + 3600.0, "ts_fast": t_now + 7200.0}, default=None, now_ts=t_now) is None

    # 2. Dependent cross-check: poisoned far-future snapshot must NOT write or wedge _LAST_SAMPLED_TS for subsequent valid ticks
    cache_path = tmp_path / "cache_short_trend.json"
    logger_dir = tmp_path / "logger"
    monkeypatch.setattr(observe, "CACHE_TREND_PATH", str(cache_path))
    monkeypatch.setattr(observe, "LOGGER_DIR", str(logger_dir))

    cache_path.write_text(
        json.dumps({"BTCUSDC": {"ts": t_now + 86400.0, "current_price": 99999.0}}),
        encoding="utf-8",
    )
    observe.sample_current_prices(["BTCUSDC"])
    assert not logger_dir.exists()
    assert "BTCUSDC" not in observe._LAST_SAMPLED_TS

    cache_path.write_text(
        json.dumps({"BTCUSDC": {"ts": t_now - 2.0, "current_price": 61250.5}}),
        encoding="utf-8",
    )
    observe.sample_current_prices(["BTCUSDC"])
    log_path = next(logger_dir.glob("tradeall_price_samples_*.log"))
    assert log_path.read_text(encoding="utf-8").splitlines() == [f"{t_now - 2.0}|BTCUSDC|61250.5"]




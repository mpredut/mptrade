import json
import os
from pathlib import Path
from unittest.mock import Mock

import pytest
from matplotlib import pyplot as plt

import tradeall_observe as observe


@pytest.fixture(autouse=True)
def _reset_observer_process_caches():
    observe._PIPE_LOG_CACHE.clear()
    observe._LAST_SAMPLED_TS.clear()
    yield
    observe._PIPE_LOG_CACHE.clear()
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
    cache_path = tmp_path / "cache_instant_trend.json"
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


def test_history_tail_keeps_first_record_when_file_is_smaller_than_tail(tmp_path, monkeypatch):
    cachedb = tmp_path / "cachedb"
    cachedb.mkdir()
    history = cachedb / "cache_price_BTCUSDC.jsonl"
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

    assert observe._load_history_jsonl_tail("BTCUSDC", max_bytes=4096) == [
        [1_000, 10.0],
        [2_000, 11.0],
    ]


def test_live_dense_price_has_priority_over_archived_sources(monkeypatch):
    monkeypatch.setattr(observe, "_load_history_jsonl_tail", lambda _symbol: [[1_000, 1.0]])
    monkeypatch.setattr(observe, "load_price_samples", lambda _symbol, _days: ([1.0], [3.0]))

    def cached_entries(filename, _symbol):
        return [[1_000, 4.0]] if filename.endswith(".json") else [[1_000, 2.0]]

    monkeypatch.setattr(observe, "_load_cachedb_price_entries", cached_entries)

    assert observe.load_price_series_live("BTCUSDC", 9, include_history=True) == ([1], [4.0])


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

    class Figure:
        def savefig(self, path, **_kwargs):
            Path(path).write_bytes(b"complete-png")

    observe._save_figure_atomic(Figure(), target, dpi=110)

    assert target.read_bytes() == b"complete-png"
    assert list(tmp_path.glob(".tradeall_observe_*.png")) == []


def test_atomic_figure_publish_preserves_previous_file_on_failure(tmp_path):
    target = tmp_path / "chart.png"
    target.write_bytes(b"old")

    class BrokenFigure:
        def savefig(self, path, **_kwargs):
            Path(path).write_bytes(b"partial")
            raise RuntimeError("render failed")

    with pytest.raises(RuntimeError, match="render failed"):
        observe._save_figure_atomic(BrokenFigure(), target)

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
    import datetime
    today = datetime.datetime.now().date()
    log_file = tmp_path / f"order_outcomes_{today.isoformat()}.log"
    monkeypatch.setattr(observe, "LOGGER_DIR", str(tmp_path))

    rows = [
        # ts|symbol|side|price|qty|outcome|refuse_reason|caller|motivation
        "100.0|BTCUSDC|BUY|85000.0|0.1|accepted||tradeall.py|trend_confirmed_up",
        "200.0|BTCUSDC|BUY|85100.0|0.1|refused|deferred|market_api.py|kalman_primary_up",
        "300.0|BTCUSDC|SELL|85500.0|0.1|accepted||market_api.py|consistent_or_old_down",
        "400.0|BTCUSDC|BUY|85200.0|0.1|refused|error|market_api.py|rtrade_pair_quote",
        "500.0|BTCUSDC|SELL|85300.0|0.1|accepted||monitortrades.py|monitortrades_exit",
        "600.0|TAOUSDC|BUY|300.0|1.0|accepted||tradeall.py|trend_confirmed_up",
    ]
    log_file.write_text("\n".join(rows) + "\n", encoding="utf-8")

    btc_events = observe.load_order_events("BTCUSDC", 1)
    assert len(btc_events) == 3
    assert [e["ts"] for e in btc_events] == [100.0, 200.0, 300.0]
    assert [e["reason"] for e in btc_events] == ["trend_confirmed_up", "kalman_primary_up", "consistent_or_old_down"]

    tao_events = observe.load_order_events("TAOUSDC", 1)
    assert len(tao_events) == 1
    assert tao_events[0]["ts"] == 600.0


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
    }
    text = observe.format_state_text(
        entry, "HEADER NOW",
        pending="BUY $1482.45",
        exposure="2.126 ($650.00)",
    )
    assert "HEADER NOW" in text
    assert "trend:     DOWN" in text
    assert "fast:      UP (grad: +0.0025)" in text
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
    observe.write_html(["BTCUSDC"], live_minutes=60, active_summary=active_summary)
    html_file = tmp_path / "tradeall_live.html"
    assert html_file.exists()
    content = html_file.read_text(encoding="utf-8")
    assert "Fleet Pending BUY:</strong> $1500.00" in content
    assert "Fleet Pending SELL:</strong> $2500.00" in content
    assert "Open Cost Basis:</strong> $3000.00" in content
    assert "Day" in content
    assert "Week" in content


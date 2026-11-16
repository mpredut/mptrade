"""Regression tests for Wave 16 replay provider, continuous grid, T212 fetch, and sweep fixes."""

import ast
import dataclasses
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

import pytest

from offline.backtests.datasets import save_dataset, should_skip_ohlc_csv_refresh
from offline.research.comprehensive_surge_sweep import run_deep_surge_sweep as rdss
from offline.research.kraken_continuous_grid import (
    fetch_data as kcg_fetch,
    grid as kcg_grid,
    report2 as kcg_report2,
    report3 as kcg_report3,
)
from offline.research.t212_continuous_grid import fetch as t212_fetch
from providers.replay_provider import ReplayMarketDataProvider
from state_io import atomic_write_jsonl, atomic_write_text, read_text_file
from strategies import spot_engine as strat

ROOT = Path(__file__).resolve().parents[1]

_WAVE16_PY_FILES = (
    "providers/replay_provider.py",
    "offline/backtests/datasets.py",
    "offline/research/comprehensive_surge_sweep/run_deep_surge_sweep.py",
    "offline/research/kraken_continuous_grid/fetch_data.py",
    "offline/research/kraken_continuous_grid/grid2.py",
    "offline/research/kraken_continuous_grid/grid3.py",
    "offline/research/kraken_continuous_grid/report2.py",
    "offline/research/kraken_continuous_grid/report3.py",
    "offline/research/t212_continuous_grid/fetch.py",
    "offline/legacy_tools/clean_order_cache.py",
    "offline/legacy_tools/migrate_price_cache.py",
    "offline/legacy_tools/test.py",
    "offline/legacy_tools/test2.py",
    "tests/test_wave16_backtest_fixes.py",
)


def test_wave16_ascii_and_ast_hygiene() -> None:
    """All Wave 16 files and tracked offline/ files obey strict ASCII and SSOT AST rules."""
    forbidden_calls = {"float", "int", "open"}
    for rel in _WAVE16_PY_FILES:
        src = read_text_file(str(ROOT / rel))
        assert all(ord(ch) < 128 for ch in src), f"Non-ASCII character in {rel}"
        tree = ast.parse(src, filename=rel)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    assert node.func.id not in forbidden_calls, (
                        f"Forbidden raw call {node.func.id}() in {rel}:{node.lineno}"
                    )
                elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                    assert not (
                        node.func.value.id == "json" and node.func.attr in {"load", "loads", "dump", "dumps"}
                    ), f"Forbidden json.{node.func.attr}() in {rel}:{node.lineno}"
                    assert not (
                        node.func.value.id == "requests" and node.func.attr in {"get", "post"}
                    ), f"Forbidden requests.{node.func.attr}() in {rel}:{node.lineno}"

    for md_path in sorted((ROOT / "offline").rglob("*.md")):
        if "results" in md_path.parts:
            continue
        md_src = read_text_file(str(md_path))
        assert all(ord(ch) < 128 for ch in md_src), f"Non-ASCII character in {md_path.relative_to(ROOT)}"


def test_replay_provider_multisymbol_base_asset_and_orderlist_isolation() -> None:
    """ReplayMarketDataProvider sums multi-quote base inventory and isolates cached _OrderList."""
    prov = ReplayMarketDataProvider(
        {
            "BTCUSDC": [(1000.0, 50000.0), (1060.0, 51000.0)],
            "BTCUSDT": [(1000.0, 50000.0), (1060.0, 51000.0)],
            "XXBTZUSD": [(1000.0, 50000.0), (1060.0, 51000.0)],
        }
    )
    prov.advance("BTCUSDC")
    prov.advance("BTCUSDT")
    prov.advance("XXBTZUSD")

    assert prov.place_order("BTCUSDC", "BUY", 50000.0, 1.25) is not None
    assert prov.free_balance("BTC") == pytest.approx(1.25)
    assert prov.free_balance("XXBT") == pytest.approx(1.25)

    assert prov.place_order("XXBTZUSD", "BUY", 50000.0, 0.5) is not None
    assert prov.free_balance("BTC") == pytest.approx(1.75)
    assert prov.free_balance("XXBT") == pytest.approx(1.75)

    # _OrderList cache isolation + shared O(1) _summary memo across unmutated calls
    orders_1 = prov.get_orders("BTCUSDC", "BUY", 3600.0)
    assert len(orders_1) == 1
    orders_1._summary = ("cached_summary", 1.25)

    orders_2 = prov.get_orders("BTCUSDC", "BUY", 3600.0)
    assert orders_2 is not orders_1
    assert orders_2._summary == ("cached_summary", 1.25)

    # Mutating caller list detaches its summary and leaves internal provider cache intact
    orders_1.clear()
    assert orders_1 == []
    assert orders_1._summary is None

    orders_3 = prov.get_orders("BTCUSDC", "BUY", 3600.0)
    assert len(orders_3) == 1
    assert orders_3[0]["qty"] == pytest.approx(1.25)
    assert orders_3._summary == ("cached_summary", 1.25)

    orders_2.sort(key=lambda item: item["timestamp"], reverse=True)
    assert orders_2._summary is None
    assert prov.get_orders("BTCUSDC", "BUY", 3600.0)._summary == ("cached_summary", 1.25)

    # Selling BTCUSDC position works cleanly despite BTCUSDT sharing base asset BTC
    prov.advance("BTCUSDC")
    assert prov.place_order("BTCUSDC", "SELL", 51000.0, 1.25) is not None
    assert prov.position("BTCUSDC")[0] == pytest.approx(0.0)
    assert prov.free_balance("BTC") == pytest.approx(0.5)


def test_kraken_grid2_grid3_report2_report3_fail_closed(tmp_path: Path) -> None:
    """Kraken continuous grid Waves 2 & 3 fail closed on missing/empty data or empty results."""
    empty_out = tmp_path / "empty_grid"
    env = {**os.environ, "GRID_OUT": str(empty_out)}

    # Missing DATA dir -> exit 1 without creating results2.jsonl / results3.jsonl
    for script_name, jsonl_name in (("grid2.py", "results2.jsonl"), ("grid3.py", "results3.jsonl")):
        proc = subprocess.run(
            [sys.executable, str(ROOT / "offline/research/kraken_continuous_grid" / script_name)],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 1
        assert not (empty_out / jsonl_name).exists()

    # Empty DATA dir -> exit 1 without truncating pre-existing results2.jsonl / results3.jsonl
    (empty_out / "data").mkdir(parents=True)
    atomic_write_text(str(empty_out / "results2.jsonl"), '{"sentinel": 2}\n')
    atomic_write_text(str(empty_out / "results3.jsonl"), '{"sentinel": 3}\n')
    for script_name, jsonl_name, sentinel in (
        ("grid2.py", "results2.jsonl", '{"sentinel": 2}\n'),
        ("grid3.py", "results3.jsonl", '{"sentinel": 3}\n'),
    ):
        proc = subprocess.run(
            [sys.executable, str(ROOT / "offline/research/kraken_continuous_grid" / script_name)],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 1
        assert read_text_file(str(empty_out / jsonl_name)) == sentinel

    # report2.py & report3.py preserve pre-existing REPORT2.md/REPORT3.md and exit 1 when wave rows are empty
    rep_dir = tmp_path / "rep_exp"
    rep_dir.mkdir(parents=True)
    live_p, live_pk = kcg_grid.CONFIGS["live"]
    atomic_write_jsonl(
        str(rep_dir / "results.jsonl"),
        [
            {
                "cfg": "live",
                "asset": "BTC",
                "params_repr": repr(live_p),
                "peak": bool(live_pk),
                "full": {"ret": 12.0, "dd": 4.0},
                "w90": [[2.0, 1.0]],
            }
        ],
    )
    atomic_write_jsonl(str(rep_dir / "results2.jsonl"), [{"kind": "error", "error": "boom"}])
    atomic_write_jsonl(str(rep_dir / "results3.jsonl"), [{"asset": "BTC", "cfg": "live", "error": "boom"}])
    atomic_write_text(str(rep_dir / "REPORT2.md"), "SENTINEL_REPORT_2\n")
    atomic_write_text(str(rep_dir / "REPORT3.md"), "SENTINEL_REPORT_3\n")

    kcg_report2.generate_report(str(rep_dir))
    kcg_report3.generate_report(str(rep_dir))
    assert read_text_file(str(rep_dir / "REPORT2.md")) == "SENTINEL_REPORT_2\n"
    assert read_text_file(str(rep_dir / "REPORT3.md")) == "SENTINEL_REPORT_3\n"

    rep_env = {**os.environ, "GRID_OUT": str(rep_dir)}
    for rep_script in ("report2.py", "report3.py"):
        proc = subprocess.run(
            [sys.executable, str(ROOT / "offline/research/kraken_continuous_grid" / rep_script)],
            env=rep_env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 1


def test_t212_fetch_missing_only_and_cached_fallback_usable_total(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """t212 fetch.py exits 0 when valid cached CSVs exist under --missing-only or transient Yahoo failure."""
    grid_out = tmp_path / "t212_grid"
    data_dir = grid_out / "data"
    data_dir.mkdir(parents=True)

    monkeypatch.setenv("GRID_OUT", str(grid_out))
    monkeypatch.setattr(t212_fetch.time, "sleep", lambda _s: None)
    monkeypatch.setattr(t212_fetch.mkt, "fetch_ohlc_candles", lambda *a, **kw: [])

    # 1. Zero cached CSVs and all live fetches fail -> exit 1
    monkeypatch.setattr("sys.argv", ["fetch.py", "--missing-only"])
    with pytest.raises(SystemExit) as exc_empty:
        runpy.run_path(str(ROOT / "offline/research/t212_continuous_grid/fetch.py"), run_name="__main__")
    assert exc_empty.value.code == 1

    # 2. Seed one valid >32-bar CSV -> --missing-only and transient outage both preserve usable_total >= 1 (exit 0)
    records = [
        {
            "timestamp": 1700000000 + i * 3600,
            "open": 100.0 + (i % 4),
            "high": 102.0 + (i % 4),
            "low": 99.0 + (i % 4),
            "close": 101.0 + (i % 4),
            "volume": 1000.0,
        }
        for i in range(40)
    ]
    save_dataset(records, data_dir / "NVDA_1h.csv")

    for argv in (["fetch.py", "--missing-only"], ["fetch.py"]):
        monkeypatch.setattr("sys.argv", argv)
        runpy.run_path(str(ROOT / "offline/research/t212_continuous_grid/fetch.py"), run_name="__main__")


def test_deep_surge_sweep_unscaled_profile_load_and_scaled_slice_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_deep_surge_sweep keeps load_base_params unscaled while scaling 240m sub-bar timeouts per slice."""
    hl_params = rdss.load_base_params(ROOT / "hyperliquid/config.env")
    assert hl_params.check_minutes == pytest.approx(1.0)

    seen_params: list[strat.StratParams] = []

    def _fake_replay(**kwargs):
        seen_params.append(kwargs["params"])
        return {"total": 1.0, "max_drawdown_pct": 1.0, "cycles": 1}

    monkeypatch.setattr(rdss, "run_replay", _fake_replay)
    bars = [(100.0, 101.0, 99.0, 100.0)] * 80
    cand = {"id": "CURRENT_CONFIG", "category": "Current", "params": hl_params, "desc": "test"}
    out = rdss.evaluate_single_candidate(cand, {"HYPE": bars}, 1050.0, 0.4)
    assert len(seen_params) == 3
    assert all(p.check_minutes == pytest.approx(240.0) for p in seen_params)
    assert out["strategy_params"] == dataclasses.asdict(hl_params)


def test_ohlc_csv_daily_staleness_refresh_and_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fresh OHLC CSVs (<18h) skip re-download; stale CSVs (>18h) auto-refresh and fall back safely."""
    data_dir = tmp_path / "kcg_data"
    data_dir.mkdir(parents=True)
    monkeypatch.setattr(kcg_fetch, "DATA", str(data_dir))

    initial_records = [
        {
            "timestamp": 1700000000 + i * 14400,
            "open": 50.0 + (i % 3),
            "high": 52.0 + (i % 3),
            "low": 49.0 + (i % 3),
            "close": 51.0 + (i % 3),
            "volume": 100.0,
        }
        for i in range(50)
    ]
    btc_csv = data_dir / "BTC_240m.csv"
    save_dataset(initial_records, btc_csv)

    calls: list[str] = []
    updated_records = initial_records + [
        {
            "timestamp": 1700000000 + 50 * 14400,
            "open": 60.0,
            "high": 62.0,
            "low": 59.0,
            "close": 61.0,
            "volume": 200.0,
        }
    ]

    monkeypatch.setattr(
        kcg_fetch.mkt,
        "fetch_ohlc_candles",
        lambda sym, **kw: calls.append(sym) or list(updated_records),
    )

    # 1. Fresh CSV (<18h) skips network call by default
    assert kcg_fetch.fetch("BTCUSDT", "BTC") is True
    assert calls == []

    # 2. Age CSV past 18h -> --missing-only still skips, default refreshes to 51 bars
    stale_ts = time.time() - 90000.0
    os.utime(btc_csv, (stale_ts, stale_ts))
    assert should_skip_ohlc_csv_refresh(btc_csv, min_bars=43, missing_only=True) == (True, True)
    assert should_skip_ohlc_csv_refresh(btc_csv, min_bars=43, missing_only=False) == (False, True)

    assert kcg_fetch.fetch("BTCUSDT", "BTC", missing_only=True) is True
    assert calls == []

    assert kcg_fetch.fetch("BTCUSDT", "BTC") is True
    assert calls == ["BTCUSDT"]
    assert len(read_text_file(str(btc_csv)).strip().splitlines()) == 52  # header + 51 bars

    # 3. Stale CSV + transient empty API response preserves existing CSV and returns True
    os.utime(btc_csv, (stale_ts, stale_ts))
    monkeypatch.setattr(kcg_fetch.mkt, "fetch_ohlc_candles", lambda sym, **kw: [])
    assert kcg_fetch.fetch("BTCUSDT", "BTC") is True
    assert len(read_text_file(str(btc_csv)).strip().splitlines()) == 52


def test_ohlc_refresh_preserves_history_via_merge_and_reports_effective_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Truncated (>min_bars) API responses merge into existing CSVs without shrinking history, and reports show coverage."""
    from offline.backtests.datasets import load_dataset
    from offline.research.kraken_continuous_grid import report as kcg_rep, report2 as kcg_rep2, report3 as kcg_rep3
    from offline.research.t212_continuous_grid import grid as t212_grid, report as t212_rep

    # A. Kraken fetch_data merges truncated (>43-bar) response with multi-year CSV and trims < START_MS
    k_dir = tmp_path / "k_exp"
    k_data = k_dir / "data"
    k_data.mkdir(parents=True)
    monkeypatch.setattr(kcg_fetch, "DATA", str(k_data))

    start_sec = kcg_fetch.START_MS // 1000
    existing_k = [
        {
            "timestamp": start_sec - 14400,
            "open": 9.0,
            "high": 9.5,
            "low": 8.5,
            "close": 9.2,
            "volume": 50.0,
        },
        *[
            {
                "timestamp": start_sec + i * 14400,
                "open": 100.0 + (i % 5),
                "high": 103.0 + (i % 5),
                "low": 99.0 + (i % 5),
                "close": 101.0 + (i % 5),
                "volume": 100.0,
            }
            for i in range(200)
        ],
    ]
    btc_csv = k_data / "BTC_240m.csv"
    save_dataset(existing_k, btc_csv)
    stale_ts = time.time() - 90000.0
    os.utime(btc_csv, (stale_ts, stale_ts))

    partial_tail_k = [
        {
            "timestamp": start_sec + i * 14400,
            "open": 110.0 + (i % 5),
            "high": 114.0 + (i % 5),
            "low": 109.0 + (i % 5),
            "close": 112.0 + (i % 5),
            "volume": 150.0,
        }
        for i in range(152, 202)  # 50 bars >= 43, overlapping 152..199 and appending 200..201
    ]
    monkeypatch.setattr(kcg_fetch.mkt, "fetch_ohlc_candles", lambda sym, **kw: list(partial_tail_k))
    assert kcg_fetch.fetch("BTCUSDT", "BTC") is True
    merged_k = load_dataset(btc_csv)
    assert len(merged_k) == 202
    assert merged_k[0]["timestamp"] == start_sec
    assert merged_k[-1]["timestamp"] == start_sec + 201 * 14400
    assert merged_k[160]["close"] == pytest.approx(112.0 + (160 % 5))

    # B. T212 fetch.py merges truncated (100-bar > 32) response into 400-bar 1d CSV and enforces 730d window on 1h
    t_out = tmp_path / "t212_exp"
    t_data = t_out / "data"
    t_data.mkdir(parents=True)
    monkeypatch.setenv("GRID_OUT", str(t_out))
    monkeypatch.setattr(t212_fetch, "ASSETS", ["NVDA"])
    monkeypatch.setattr(t212_fetch.time, "sleep", lambda _s: None)

    base_1d_ts = 1_700_006_400  # 2023-11-15 00:00:00 UTC
    existing_1d = [
        {
            "timestamp": base_1d_ts + i * 86400,
            "open": 200.0 + (i % 7),
            "high": 205.0 + (i % 7),
            "low": 198.0 + (i % 7),
            "close": 202.0 + (i % 7),
        }
        for i in range(400)
    ]
    nvda_1d_csv = t_data / "NVDA_1d.csv"
    save_dataset(existing_1d, nvda_1d_csv)
    os.utime(nvda_1d_csv, (stale_ts, stale_ts))

    ancient_1h_ts = base_1d_ts - 800 * 86400
    existing_1h = [
        {
            "timestamp": ancient_1h_ts,
            "open": 50.0,
            "high": 51.0,
            "low": 49.0,
            "close": 50.5,
        },
        *[
            {
                "timestamp": base_1d_ts + i * 3600,
                "open": 200.0 + (i % 4),
                "high": 203.0 + (i % 4),
                "low": 199.0 + (i % 4),
                "close": 201.0 + (i % 4),
            }
            for i in range(300)
        ],
    ]
    nvda_1h_csv = t_data / "NVDA_1h.csv"
    save_dataset(existing_1h, nvda_1h_csv)
    os.utime(nvda_1h_csv, (stale_ts, stale_ts))

    def _fake_yahoo(sym, *, interval="1h", range_="730d", drop_incomplete=True, provider_name="t212"):
        if sym != "NVDA":
            return []
        if interval == "1d":
            return [
                {
                    "timestamp": base_1d_ts + i * 86400,
                    "open": 210.0,
                    "high": 215.0,
                    "low": 208.0,
                    "close": 212.0,
                }
                for i in range(302, 402)  # 100 bars (> 32), must NOT wipe 0..301
            ]
        return [
            {
                "timestamp": base_1d_ts + i * 3600,
                "open": 210.0,
                "high": 214.0,
                "low": 209.0,
                "close": 211.0,
            }
            for i in range(262, 302)  # 40 bars (> 32)
        ]

    monkeypatch.setattr(t212_fetch.mkt, "fetch_ohlc_candles", _fake_yahoo)
    monkeypatch.setattr("sys.argv", ["fetch.py"])
    runpy.run_path(str(ROOT / "offline/research/t212_continuous_grid/fetch.py"), run_name="__main__")

    merged_1d = load_dataset(nvda_1d_csv)
    assert len(merged_1d) == 402
    assert merged_1d[0]["timestamp"] == base_1d_ts
    assert merged_1d[-1]["timestamp"] == base_1d_ts + 401 * 86400

    merged_1h = load_dataset(nvda_1h_csv)
    assert len(merged_1h) == 302
    assert merged_1h[0]["timestamp"] == base_1d_ts  # ancient bar (>730d before end) trimmed

    # C. Grid & reports render last bar UTC date and effective interval coverage
    monkeypatch.setattr(t212_grid, "DATA", str(t_data))
    t212_grid._SPLIT_CACHE.clear()
    t212_grid._TASK_CACHE.clear()
    row_t = t212_grid.task(("nvda", "live", "NVDA"))
    assert row_t["1d"]["bars"] == 402
    assert row_t["1d"]["start_ts"] == base_1d_ts
    assert row_t["1d"]["end_ts"] == base_1d_ts + 401 * 86400
    from state_io import atomic_write_jsonl

    atomic_write_jsonl(str(t_out / "results.jsonl"), [row_t])
    rep_t = t212_rep.generate_report(str(t_out))
    assert "Effective coverage: 1h: 2023-11-15 -> 2023-11-27 (last bar: 2023-11-27 13:00 UTC); 1d: 2023-11-15 -> 2024-12-20 (last bar: 2024-12-20)." in rep_t

    live_p, live_pk = kcg_rep.load_local_module(
        str(ROOT / "offline/research/kraken_continuous_grid"), "grid", module_key="offline.research.kraken_continuous_grid.grid"
    ).CONFIGS["live"]
    atomic_write_jsonl(
        str(k_dir / "results.jsonl"),
        [
            {
                "asset": "BTC",
                "cfg": "live",
                "bars": 202,
                "start_ts": start_sec,
                "end_ts": start_sec + 201 * 14400,
                "params_repr": repr(live_p),
                "peak": bool(live_pk),
                "full": {"ret": 15.0, "dd": 6.0, "expo": 40.0, "bh": 10.0},
                "year": {"2018": {"ret": 15.0}},
                "w30": [[1.0, 2.0, 0.5]],
                "w90": [[3.0, 4.0, 1.5]],
            }
        ],
    )
    rep_k = "\n".join(kcg_rep.generate_report(str(k_dir)))
    assert "Effective coverage: 2018-01-01 -> 2018-02-03 (last bar: 2018-02-03 12:00 UTC)." in rep_k
    rep_k2 = "\n".join(kcg_rep2.generate_report(str(k_dir)))
    assert "Effective coverage: 2018-01-01 -> 2018-02-03 (last bar: 2018-02-03 12:00 UTC)." in rep_k2
    atomic_write_jsonl(
        str(k_dir / "results3.jsonl"),
        [
            {
                "asset": "BTC",
                "cfg": "live",
                "start_ts": start_sec,
                "end_ts": start_sec + 201 * 14400,
                "params_repr": repr(live_p),
                "peak": bool(live_pk),
                "0.26": {"ret": 12.0, "dd": 5.0},
                "0.4": {"ret": 10.0, "dd": 6.0},
            }
        ],
    )
    rep_k3 = "\n".join(kcg_rep3.generate_report(str(k_dir)))
    assert "Effective coverage: 2018-01-01 -> 2018-02-03 (last bar: 2018-02-03 12:00 UTC)." in rep_k3

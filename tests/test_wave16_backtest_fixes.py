"""Regression tests for Wave 16 replay provider, continuous grid, T212 fetch, and sweep fixes."""

from __future__ import annotations

import ast
import dataclasses
import os
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

from offline.backtests.datasets import save_dataset
from offline.research.comprehensive_surge_sweep import run_deep_surge_sweep as rdss
from offline.research.kraken_continuous_grid import (
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
    "offline/research/comprehensive_surge_sweep/run_deep_surge_sweep.py",
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

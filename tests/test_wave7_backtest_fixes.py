"""Targeted regression tests for Wave 7 backtest, replay, strategy, runner, and research fixes."""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives.serialization import load_pem_public_key

from market_regime import MarketRegimeContext, MarketRegimeDecision
from offline.backtests.comparison import relative_to_baseline
from offline.backtests.execution_calibration import calibrate_execution_events
from offline.backtests.financial_benchmark import aggregate_financial_windows
from offline.backtests.promotion import _extract_scenario_windows, _validate_promotion_reports
from offline.backtests.walk_forward import summarize_test_windows
from offline.legacy_tools.convertor import convert_price_cache
from offline.legacy_tools.gen_cryptography import generate_ed25519_keypair
from offline.research.backtest_ranges import scan_all, scan_backtest_ranges
from offline.research.kraken_continuous_grid.grid2 import _risk_adjusted_score
from offline.research.kraken_continuous_grid.report3 import _ret_dd_ratio
from offline.research.monitortrades_backtest import compare_sl_variants
from offline.runners.apply_proposals import _read_proposals
from offline.runners.fetch_hyperliquid_candles import resolve_spot_pair
from offline.runners.kraken_financial_benchmark import markdown_report
from offline.runners.spot_profile_review import revision_params
from offline.runners.t212_walk_forward_baseline import fx_quote
from offline.simulations.sim import PriceWindow
from providers.replay_provider import ReplayMarketDataProvider, SimClock
from state_io import atomic_write_json, atomic_write_text, read_json_file, read_text_file
from strategies.base_spot_mixin import (
    BaseSpotMixin,
    generate_tp_ladder,
    parse_max_dca_buys_env,
    parse_tp_ladder_env,
)
from strategies.mt_position_accounting import _summarize_orders
from strategies.position_exit_manager import is_reentry_state_stale
from strategies.replay_metrics import calculate_performance_metrics
from strategies.rtrade_pair import OrderSnapshot, OrderTicket, PairCoordinator, PairPolicy
from trailing_core import TrailingCore

ROOT = Path(__file__).resolve().parents[1]

WAVE7_PURE_ASCII_FILES = (
    # Wave 7A
    "hyperliquid/strategy.py",
    "hyperliquid/replay.py",
    "hyperliquid/backtest.py",
    "hyperliquid/signals.py",
    "kraken/backtest_trail.py",
    "kraken/trailing_stop.py",
    "212trading/analyze_volatility.py",
    "strategies/replay_execution.py",
    "strategies/replay_metrics.py",
    "providers/replay_provider.py",
    # Wave 7B
    "strategies/base_spot_mixin.py",
    "strategies/position_exit_manager.py",
    "strategies/mt_position_accounting.py",
    "strategies/rtrade_pair.py",
    "trailing_core.py",
    "market_regime.py",
    # Wave 7C
    "offline/backtests/tradeall.py",
    "offline/backtests/financial_benchmark.py",
    "offline/backtests/promotion.py",
    "offline/backtests/comparison.py",
    "offline/backtests/execution_calibration.py",
    "offline/backtests/walk_forward.py",
    "offline/runners/binance_walk_forward_baseline.py",
    "offline/runners/hyperliquid_walk_forward_baseline.py",
    "offline/runners/t212_walk_forward_baseline.py",
    "offline/runners/spot_profile_review.py",
    "offline/runners/apply_proposals.py",
    "offline/runners/kraken_financial_benchmark.py",
    "offline/runners/kraken_financial_compare.py",
    "offline/runners/fetch_hyperliquid_candles.py",
    # Wave 7D
    "offline/simulations/sim.py",
    "offline/research/intelligence_backtest.py",
    "offline/research/chop_weight_backtest.py",
    "offline/research/short_trend_estimator_benchmark.py",
    "offline/research/verify_optimizations_parity.py",
    "offline/research/compare_mhdsl_reentry_replay.py",
    "offline/research/backtest_ranges.py",
    "offline/research/kraken_continuous_grid/fetch_data.py",
    "offline/research/kraken_continuous_grid/grid.py",
    "offline/research/kraken_continuous_grid/grid2.py",
    "offline/research/kraken_continuous_grid/grid3.py",
    "offline/research/kraken_continuous_grid/report3.py",
    "offline/research/t212_continuous_grid/fetch.py",
    "offline/legacy_tools/convertor.py",
    "offline/legacy_tools/gen_cryptography.py",
    "offline/research/monitortrades_backtest/compare_sl_variants.py",
    "offline/research/monitortrades_backtest/sweep_mt_models.py",
    "tests/test_wave7_backtest_fixes.py",
)


def test_wave7_ascii_and_ast_ssot_hygiene():
    for rel in WAVE7_PURE_ASCII_FILES:
        path = ROOT / rel
        assert path.is_file(), f"Missing file: {rel}"
        text = read_text_file(path)
        bad_ascii = [
            (idx + 1, ch)
            for idx, line in enumerate(text.splitlines())
            for ch in line
            if ord(ch) >= 128
        ]
        assert not bad_ascii, f"Non-ASCII characters in {rel}: {bad_ascii[:5]}"

        tree = ast.parse(text, filename=rel)
        raw_coercions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in ("float", "int", "open"):
                    raw_coercions.append((node.lineno, node.func.id))
        assert not raw_coercions, f"Forbidden raw {raw_coercions[:5]} in {rel}"


def test_wave7a_replay_metrics_and_replay_provider_guards():
    # Break-even 0.0 PnL trades must not inflate wins or force 0-loss profit_factor to 0
    summary_all_flat_or_win = calculate_performance_metrics(
        [1000.0, 1010.0, 1010.0, 1015.0],
        initial_capital=1000.0,
        trade_pnls=[10.0, 0.0, 5.0],
    )
    assert summary_all_flat_or_win["trade_count"] == 3
    assert summary_all_flat_or_win["gross_profit"] == pytest.approx(15.0)
    assert summary_all_flat_or_win["gross_loss"] == pytest.approx(0.0)
    assert summary_all_flat_or_win["profit_factor"] is None
    assert summary_all_flat_or_win["win_rate_pct"] == pytest.approx(200.0 / 3.0)

    summary_mixed = calculate_performance_metrics(
        [1000.0, 1010.0, 1010.0, 1005.0],
        initial_capital=1000.0,
        trade_pnls=[10.0, 0.0, -5.0],
    )
    assert summary_mixed["profit_factor"] == pytest.approx(2.0)

    # SimClock rejects bools; ReplayMarketDataProvider sorts out-of-order series chronologically
    with pytest.raises(ValueError, match="finite number"):
        SimClock(True)
    rp = ReplayMarketDataProvider({"BTCUSDC": [(1700000100.0, 105.0), (1700000000.0, 100.0)]})
    assert [pt[0] for pt in rp._series["BTCUSDC"]] == [1700000000.0, 1700000100.0]


def test_wave7b_base_spot_mixin_and_exit_manager_contracts():
    with pytest.raises(ValueError, match="cannot be bool"):
        generate_tp_ladder(True, 10.0, 3)
    with pytest.raises(ValueError, match="max_pct >= min_pct"):
        generate_tp_ladder(10.0, 5.0, 3)

    with pytest.raises(ValueError, match="must all be set together"):
        parse_tp_ladder_env(
            {"STRATEGY_MODE": "avg_tp", "STRAT_TP_LADDER_MIN": "2.0"},
            generate_tp_ladder,
        )
    with pytest.raises(ValueError, match="Invalid STRAT_TP_LADDER"):
        parse_tp_ladder_env(
            {"STRATEGY_MODE": "avg_tp", "STRAT_TP_LADDER": "-2:50,5:50"},
            generate_tp_ladder,
        )

    with pytest.raises(ValueError, match="cannot be smaller than STRAT_ENTRY"):
        parse_max_dca_buys_env({"STRAT_MAX_DCA_BUYS": "auto"}, budget=50.0, entry_amt=100.0, dca_amt=25.0)

    class _DummyReplayBot(BaseSpotMixin):
        pass

    bot = _DummyReplayBot()
    bot.replay_mode = True
    bot._simulated_ts = 1700001234.0
    bot._clock = lambda: 9999999999.0
    assert bot._now() == pytest.approx(1700001234.0)

    # Sub-microsecond float round-trip jitter must not mark reentry state stale
    st = {"sell_time": 1700000000.0}
    assert is_reentry_state_stale(st, 1700000000.0 + 1e-10) is False
    assert is_reentry_state_stale(st, 1700000001.0) is True


def test_wave7b_mt_accounting_rtrade_trailing_and_regime():
    # Locked non-dict contract + order-independent tie-break on identical timestamps
    assert _summarize_orders(["not_a_dict"]) == (0.0, 0.0, (1, 0.0, 0.0, 0, 0))
    o1 = {"id": "ord-a", "price": 100.0, "origQty": 1.0, "executedQty": 1.0, "time": 1700000000000}
    o2 = {"id": "ord-b", "price": 102.0, "origQty": 2.0, "executedQty": 2.0, "time": 1700000000000}
    assert _summarize_orders([o1, o2]) == _summarize_orders([o2, o1])

    # PairCoordinator.outcome() must compute realized gross_pnl only on matched qty
    class _StubVenue:
        def current_price(self):
            return 100.0

        def place_limit(self, side, price, qty, pair_id, *, revision=0):
            return OrderTicket(order_id=f"{side}-1", side=side, price=price, qty=qty, pair_id=pair_id, revision=revision)

        def place_market(self, side, qty, pair_id, *, reason, revision=0):
            return None

        def market_exit_allowed(self, exposure_side, loss_fraction, reason):
            return False

        def order_status(self, order_id):
            return OrderSnapshot(status="open", filled_qty=0.0, cost=0.0, fee=0.0)

        def cancel(self, order_id):
            return True

    coord = PairCoordinator(
        _StubVenue(), 2.0, PairPolicy(adjustment_fraction=0.01), clock=lambda: 10.0, sleeper=lambda _: None
    )
    coord.start(mid=100.0, pair_id="p1")
    coord.snapshots["BUY-1"] = OrderSnapshot(status="closed", filled_qty=2.0, cost=199.0, fee=0.1)
    coord.snapshots["SELL-1"] = OrderSnapshot(status="open", filled_qty=0.0, cost=0.0, fee=0.0)
    out = coord.outcome()
    assert out.buy_qty == pytest.approx(2.0)
    assert out.sell_qty == pytest.approx(0.0)
    assert out.gross_pnl == pytest.approx(0.0)

    # TrailingCore._finish_pending preserves and merges prior rebuy qty on multi-step exits
    class _TrailAdapter:
        name = "TEST"
        is_live = False

        def assets(self):
            return [("BTC", "BTC", "BTCUSD", 5.0)]

        def free_qty(self, asset):
            return 0.0

        def total_qty(self, asset):
            return 0.0

        def trend(self, pair):
            return 0.0

    eng = TrailingCore(
        _TrailAdapter(),
        log=lambda _m: None,
        enabled=False,
        state_file="/tmp/unused_trail_state.json",
        min_notional=10.0,
        rebuy_enabled=True,
        rebuy_bounce_pct=1.0,
        rebuy_skip_if_trend_down=False,
    )
    st_trail = {
        "peak": 110.0,
        "target_exit_qty": 1.0,
        "rebuy": {"qty": 1.5, "sell_price": 100.0, "low": 98.0},
    }
    res_obj = SimpleNamespace(
        status=SimpleNamespace(filled_qty=1.0, cost=106.0),
        intent={"action": "SELL", "asset": "BTC"},
    )
    eng._finish_pending(st_trail, res_obj, 106.0, "BTCUSD")
    assert st_trail["rebuy"]["qty"] == pytest.approx(2.5)
    assert st_trail["rebuy"]["sell_price"] == pytest.approx((1.5 * 100.0 + 1.0 * 106.0) / 2.5)
    assert st_trail["rebuy"]["low"] == pytest.approx(98.0)

    # MarketRegimeContext.is_valid_for rejects non-positive max_age/horizon and bool provider
    dec = MarketRegimeDecision(
        regime="bull", gradient=0.01, epsilon=0.001, strength=2.0, fresh=True, reason="ok"
    )
    ctx = MarketRegimeContext.from_decision(dec, evaluated_at=1700000000.0, symbol="BTCUSD", provider="kraken")
    assert ctx.is_valid_for(now=1700000010.0, max_age_seconds=60.0) is True
    assert ctx.is_valid_for(now=1700000010.0, max_age_seconds=0.0) is False
    assert ctx.is_valid_for(now=1700000010.0, max_horizon_seconds=0.0) is False
    assert ctx.is_valid_for(now=1700000010.0, provider=True) is False


def test_wave7c_backtests_and_runners_contracts(tmp_path):
    # Break-even window (buy_hold=0.0 -> sideways, expectancy=0.0 -> excluded from PF)
    agg = aggregate_financial_windows(
        [
            {
                "key": "w1",
                "return_pct": 0.0,
                "buy_hold_return_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "cycles": 1,
                "fills": 2,
                "metrics": {"return_pct": 0.0, "max_drawdown_pct": 0.0, "trade_count": 1, "expectancy": 0.0},
            },
            {
                "key": "w2",
                "return_pct": 5.0,
                "buy_hold_return_pct": 4.0,
                "max_drawdown_pct": 1.0,
                "cycles": 1,
                "fills": 2,
                "metrics": {"return_pct": 5.0, "max_drawdown_pct": 1.0, "trade_count": 0},
            },
        ],
        initial_capital=1000.0,
    )
    assert agg["positive_windows"] == 1
    assert agg["regimes"]["sideways"]["windows"] == 1
    assert agg["median_profit_factor"] is None

    # Promotion: bool vs int identity mismatch rejected, mixed int/str window keys sorted safely
    base_rep = {
        "venue": "kraken",
        "initial_capital_usd": 1000,
        "dataset": {"sha256": "abc", "bars": 100},
        "walk_forward": {"interval_minutes": 240, "train": 50, "validation": 10, "test": 10, "step": 10, "warmup": 1},
        "scenarios": {"central": {"assumptions": {"fee": 0.1}, "windows": [{"key": 1}, {"key": "w2"}]}},
    }
    cand_bool = {
        **base_rep,
        "walk_forward": {**base_rep["walk_forward"], "warmup": True},
    }
    with pytest.raises(ValueError, match="different datasets or windows"):
        _validate_promotion_reports(base_rep, cand_bool)
    bw, cw, _, _ = _extract_scenario_windows(
        "central", base_rep["scenarios"]["central"], base_rep["scenarios"]["central"]
    )
    assert set(bw) == {1, "w2"} and set(cw) == {1, "w2"}

    # Comparison: duplicate window keys keep first-seen consistently
    rel = relative_to_baseline(
        [{"key": "w1", "return_pct": 5.0}, {"key": "w1", "return_pct": -10.0}],
        [{"key": "w1", "return_pct": 2.0}, {"key": "w1", "return_pct": 20.0}],
    )
    assert rel["wins_vs_live"] == 1
    assert rel["losses_vs_live"] == 0

    # Execution calibration: bool client_order_id invalid, pre-acceptance status merged without double-counting,
    # and missing fee -> fee_bps count == 0
    cal = calibrate_execution_events(
        [
            {
                "intent_id": "i1",
                "event": "submit_requested",
                "venue": "binance",
                "symbol": "BTCUSDC",
                "side": "BUY",
                "qty": 2.0,
                "price": 100.0,
                "market": False,
                "client_order_id": True,
                "ts": 1000.0,
            },
            {
                "intent_id": "i1",
                "event": "order_status",
                "filled_qty": 1.0,
                "cost": 100.0,
                "fee": None,
                "ts": 1000.5,
            },
            {
                "intent_id": "i1",
                "event": "submit_accepted",
                "order_id": "OID-1",
                "ts": 1001.0,
            },
            {
                "intent_id": "i1",
                "event": "order_status",
                "order_id": "OID-1",
                "filled_qty": 2.0,
                "cost": 200.0,
                "fee": None,
                "ts": 1002.0,
            },
        ]
    )
    assert len(cal["orders"]) == 1
    assert cal["orders"][0]["filled_qty"] == pytest.approx(2.0)
    assert cal["summary"]["limit"]["filled"] == 1
    assert cal["summary"]["limit"]["final_fill_ratio"]["max"] == pytest.approx(1.0)
    assert cal["summary"]["limit"]["fee_bps"]["count"] == 0
    assert cal["client_order_id_validation"]["invalid_client_order_ids"] == 1

    # Walk-forward summarize_test_windows accepts generators and rejects Mappings
    gen_summary = summarize_test_windows(
        w
        for w in [
            {
                "return_pct": 4.0,
                "max_drawdown_pct": 1.0,
                "buy_hold_return_pct": 2.0,
                "cycles": 1,
                "fills": 2,
            }
        ]
    )
    assert gen_summary["window_count"] == 1
    with pytest.raises(ValueError, match="TEST window"):
        summarize_test_windows({"return_pct": 1.0})  # type: ignore[arg-type]

    # T212 fx_quote guards
    with pytest.raises(ValueError, match="USD"):
        fx_quote("USD")
    with pytest.raises(ValueError, match="GBX"):
        fx_quote("GBX")
    with pytest.raises(ValueError, match="GBX"):
        fx_quote("GBp")
    assert fx_quote("RON") == ("USDRON=X", True)
    assert fx_quote("EUR") == ("EURUSD=X", False)

    # Hyperliquid resolve_spot_pair prefers isCanonical
    meta = {
        "tokens": [
            {"name": "HYPE", "index": 1, "isCanonical": False},
            {"name": "HYPE", "index": 2, "isCanonical": True},
            {"name": "USDC", "index": 0, "isCanonical": True},
        ],
        "universe": [
            {"name": "@999", "tokens": [1, 0], "isCanonical": False},
            {"name": "@107", "tokens": [2, 0], "isCanonical": True},
        ],
    }
    assert resolve_spot_pair("HYPE", "USDC", meta) == "@107"

    # spot_profile_review.revision_params supports local .env path
    env_file = tmp_path / "local_spot.env"
    cfg_src = read_text_file(ROOT / "kraken" / "config.env")
    atomic_write_text(env_file, cfg_src)
    loaded_params = revision_params(str(env_file))
    assert loaded_params.entry_amount >= 0.0

    # kraken_financial_benchmark overlap note
    overlap_md = markdown_report(
        {
            "candidate_name": "test_bench",
            "dataset": {"symbol": "HYPEUSD", "interval_minutes": 240, "bars": 1000, "sha256": "deadbeef"},
            "walk_forward": {"train": 500, "validation": 100, "test": 90, "step": 30, "windows": 3},
            "initial_capital_usd": 1000.0,
            "scenarios": {},
        }
    )
    assert "with overlapping windows" in overlap_md

    # apply_proposals branch guard
    with pytest.raises(RuntimeError, match="Invalid BACKTEST_PROPOSALS_BRANCH"):
        _read_proposals("--upload-pack=evil")


def test_wave7d_simulations_research_and_legacy_tools(tmp_path):
    # PriceWindow gradual < epsilon descent followed by expiry preserves exact window minimum
    pw = PriceWindow(window_size=3, epsilon=1e-5)
    seq = [1.0, 1.0 - 4e-6, 1.0 - 8e-6, 2.0, 2.0]
    for px in seq:
        pw.process_price(px)
        expected_min = min(pw.prices)
        got_min, _ = pw.get_min()
        assert got_min == pytest.approx(expected_min, abs=1e-12)

    # Monotone risk-adjusted negative-return ranking (smaller DD ranks strictly higher than larger DD)
    assert _risk_adjusted_score(-10.0, 10.0) > _risk_adjusted_score(-10.0, 50.0)
    assert _ret_dd_ratio(-10.0, 10.0) > _ret_dd_ratio(-10.0, 50.0)

    # scan_backtest_ranges & scan_all accept Path objects directly
    conf_path = tmp_path / "sample_instruments.conf"
    atomic_write_text(
        conf_path,
        "[BINANCE_BTC]\n# BACKTEST: 5.0, 6.0, 7.0\nmt.gain = 6.0\n",
    )
    ranges = scan_backtest_ranges(conf_path)
    assert ranges == {"BINANCE_BTC.mt.gain": ["5.0", "6.0", "7.0"]}
    all_ranges = scan_all([conf_path])
    assert all_ranges == {str(conf_path): {"BINANCE_BTC.mt.gain": ["5.0", "6.0", "7.0"]}}

    # convert_price_cache idempotency on already-converted dict items
    in_json = tmp_path / "in_cache.json"
    out_json = tmp_path / "out_cache.json"
    atomic_write_json(in_json, {"items": {"BTCUSDC": [[1700000000, 35000.0]]}, "fetchtime": {"BTCUSDC": 1700000005}})
    convert_price_cache(str(in_json), str(out_json), "BTCUSDC")
    converted = read_json_file(out_json)
    assert converted["items"]["BTCUSDC"] == [[1700000000.0, 35000.0]]

    # generate_ed25519_keypair writes valid PEM public key loadable by cryptography
    priv_pem = tmp_path / "priv.pem"
    pub_pem = tmp_path / "pub.pem"
    pub_b64 = generate_ed25519_keypair(str(priv_pem), str(pub_pem))
    assert pub_b64
    pem_text = read_text_file(pub_pem)
    assert pem_text.startswith("-----BEGIN PUBLIC KEY-----")
    loaded_pub = load_pem_public_key(pem_text.encode("ascii"))
    assert loaded_pub is not None

    # compare_sl_variants uses bounded per-symbol CoW worker model without pickling multi-GB series_map
    assert callable(compare_sl_variants._run_variant_symbol)

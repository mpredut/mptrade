"""
CONSOLIDATED tests for extracting the constants into config files (28 Jul —
unified from 4 nearly identical files: test_{tradeall,monitortrades,rtrade,
assetguardian}_config_extraction.py, ~56 sparse tests -> table-driven here).

The intent is PRESERVED in full for the legacy modules and strict for AssetGuardian:
  - DEFAULTS: the values loaded from config are the expected versioned ones;
    AssetGuardian no longer has hidden fallbacks in the code.
  - OVERRIDE: setting the environment variable + reload actually changes the value
    (the override mechanism really works, not just apparently).
  - Per-module specifics: import safety (rtrade/assetguardian do NOT start
    the live loop on import), dead code removed (monitortrades SYMBOL_PARAMS),
    BUY/SELL asymmetry preserved (rtrade).
"""
import importlib
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

import utils as u
import tradeall as ta
import monitortrades as mt
import rtrade as rt
import assetguardian as ag

# Computed values (identical to what the original files did).
_PC_SMALL = u.calculate_difference_percent(60000, 60000 - 310)
_PC_BIG = u.calculate_difference_percent(97000, 95000 - 377)
_RT_ADJ = round(u.calculate_difference_percent(60000, 60000 - 380) / 100, 4)

# (module, attribute, expected_default_value = the old hardcoded value)
DEFAULTS = [
    (ta, "TREND_TO_BE_OLD_SECONDS", 60 * 60 * 1.9),
    (ta, "PRICE_CHANGE_THRESHOLD_EUR", _PC_SMALL),
    (ta, "PRICE_CHANGE_THRESHOLD_BIG_EUR", _PC_BIG),
    (ta, "TREND_MIN_VALIDATED_SECONDS", 30),
    (ta, "TREND_MIN_VALIDATED_CONFIRMS", 3),
    (ta, "TREND_CONSISTENT_CONFIRMS", 8 * 3),
    (ta, "TREND_UNIFORM_RATE_THRESHOLD", 0.08),
    (ta, "SLOPE_EXTREME_THRESHOLD", 5.1),
    (ta, "FIRE_MIN_RETRY_INTERVAL_SEC", 6 * 60),
    (ta, "FIRE_MAX_PER_TREND", 3),
    (mt, "MT_ARE_CLOSE_TOLERANCE_PCT", 1.0),
    (mt, "MT_MAIN_LOOP_SLEEP_SEC", 60 * 0.8),
    (mt, "MT_BUY_PRICE_OFFSET", 0.5),
    (mt, "MT_SELL_SAFEBACK_HOURS", 2),
    (mt, "MT_BUY_SAFEBACK_HOURS", 48),
    (mt, "MT_NOTIFY_ONLY", True),
    (mt, "HARD_TP_PCT", 0.17),
    (mt, "HARD_TP_FRACTION", 0.5),
    (mt, "HARD_TP_COOLDOWN_S", 6 * 3600),
    (mt, "TP_REFERENCE", "last"),
    (rt, "WAIT_FOR_ORDER", 32),
    (rt, "MIN_adjustment_percent", 0.01),
    (rt, "RTRADE_NOTIONAL_USDC", 500),
    (rt, "DEFAULT_ADJUSTMENT_PERCENT", _RT_ADJ),
    (rt, "RTRADE_INITIAL_SPREAD_PCT", 0.1),
    (rt, "RTRADE_BUY_DECAY_PCT", 0.005),
    (rt, "RTRADE_SELL_DECAY_PCT", 0.01),
    (rt, "RTRADE_BUY_DESPERATE_HOURS_BASE", 0.3),
    (rt, "RTRADE_SELL_DESPERATE_HOURS_BASE", 0.23),
    (rt, "RTRADE_DESPERATE_SAFEBACK_SEC", 1 * 3600 + 60),
    (rt, "RTRADE_BUY_NORMAL_HOURS", 16),
    (rt, "RTRADE_SELL_NORMAL_HOURS", 12),
    (rt, "RTRADE_FOLLOWUP_OFFSET_PCT", 0.01),
    (rt, "RTRADE_FOLLOWUP_HOURS", 2.7),
    (rt, "RTRADE_BAD_DAY_TOLERANCE_PCT", 0.1),
    (rt, "RTRADE_BAD_DAY_MULTIPLIER", 1.7),
    (rt, "RTRADE_ZERO_EPSILON", 0.0001),
    (rt, "RTRADE_MAX_FAILURES", 10),
    (rt, "RTRADE_PAIR_COORDINATOR_ENABLED", False),
    (rt, "RTRADE_PAIR_POLL_SEC", 1),
    (rt, "RTRADE_PAIR_MAX_ACTIVE_ROUNDS", 4),
    (rt, "RTRADE_PAIR_START_INTERVAL_SEC", 8),
    (rt, "RTRADE_PAIR_DIRECTIONS", ("BUY", "SELL")),
    (rt, "RTRADE_INSUFFICIENT_FUNDS_BACKOFF_SEC", 180),
    (rt, "RTRADE_PLACE_FAILURE_BACKOFF_SEC", 180),
    (rt, "RTRADE_FAST_FILL_RATIO", 0.25),
    (rt, "RTRADE_MIN_EDGE_PCT", 0.0115),
    (rt, "RTRADE_SHOCK_HARD_STOP_PCT", 0.04),
    (rt, "RTRADE_HARD_STOP_PCT", 0.08),
    (ag, "CHECK_INTERVAL_SECONDS", 0.9 * 60),
    (ag, "ASSET_REFERENCE_MINUTES_BACK_DEFAULT", 24 * 60),
    (ag, "BUY_USE_CASH_RATIO", 0.995),
    (ag, "SELL_REARM_GROWTH_PERCENT", 2.0),
    (ag, "ORDER_MAX_AGE_SECONDS", 15 * 60),
    (ag, "ORDER_MISSING_CONFIRMATIONS", 1),
    (ag, "TREND_DEFER_MAX_SECONDS", 3 * 60),
    (ag, "PORTFOLIO_GROWTH_PERCENT", 2.2),
    (ag, "PORTFOLIO_ASSET_MIN_GROWTH_PERCENT", 4.1),
    (ag, "PORTFOLIO_SELL_ALLOCATION", 0.20),
    (ag, "RTRADE_BLOCK_CANCEL_SECONDS", 120.0),
]

# (module, env_var, value_str, attribute, expected_value_after_the_override)
OVERRIDES = [
    (ta, "TRADEALL_SLOPE_EXTREME_THRESHOLD", "9.9", "SLOPE_EXTREME_THRESHOLD", 9.9),
    (ta, "TRADEALL_FIRE_MAX_PER_TREND", "7", "FIRE_MAX_PER_TREND", 7),
    (ta, "TRADEALL_FIRE_MIN_RETRY_MINUTES", "1", "FIRE_MIN_RETRY_INTERVAL_SEC", 60),
    (ta, "TRADEALL_TREND_CONSISTENT_CONFIRMS", "48", "TREND_CONSISTENT_CONFIRMS", 48),
    (ta, "TRADEALL_TREND_OLD_HOURS", "3.0", "TREND_TO_BE_OLD_SECONDS", 3.0 * 3600),
    (mt, "MT_ARE_CLOSE_TOLERANCE_PCT", "2.5", "MT_ARE_CLOSE_TOLERANCE_PCT", 2.5),
    (mt, "MT_MAIN_LOOP_SLEEP_SEC", "30", "MT_MAIN_LOOP_SLEEP_SEC", 30),
    (mt, "MT_BUY_PRICE_OFFSET", "1.0", "MT_BUY_PRICE_OFFSET", 1.0),
    (mt, "MT_SELL_SAFEBACK_HOURS", "4", "MT_SELL_SAFEBACK_HOURS", 4),
    (mt, "MT_BUY_SAFEBACK_HOURS", "24", "MT_BUY_SAFEBACK_HOURS", 24),
    (rt, "RTRADE_NOTIONAL_USDC", "250", "RTRADE_NOTIONAL_USDC", 250),
    (rt, "RTRADE_BUY_DECAY_PCT", "0.02", "RTRADE_BUY_DECAY_PCT", 0.02),
    (rt, "RTRADE_BAD_DAY_MULTIPLIER", "2.0", "RTRADE_BAD_DAY_MULTIPLIER", 2.0),
    (rt, "RTRADE_MAX_FAILURES", "3", "RTRADE_MAX_FAILURES", 3),
    (rt, "RTRADE_DEFAULT_ADJUSTMENT_PCT", "0.02", "DEFAULT_ADJUSTMENT_PERCENT", 0.02),
    (rt, "RTRADE_PAIR_POLL_SEC", "2.5", "RTRADE_PAIR_POLL_SEC", 2.5),
    (rt, "RTRADE_PAIR_MAX_ACTIVE_ROUNDS", "7", "RTRADE_PAIR_MAX_ACTIVE_ROUNDS", 7),
    (rt, "RTRADE_PAIR_START_INTERVAL_SEC", "3.5", "RTRADE_PAIR_START_INTERVAL_SEC", 3.5),
    (rt, "RTRADE_FAST_FILL_RATIO", "0.5", "RTRADE_FAST_FILL_RATIO", 0.5),
    (ag, "AG_CHECK_INTERVAL_SEC", "30", "CHECK_INTERVAL_SECONDS", 30),
    (ag, "AG_SELL_REARM_GROWTH_PCT", "4", "SELL_REARM_GROWTH_PERCENT", 4),
    (ag, "AG_ORDER_MAX_AGE_SEC", "120", "ORDER_MAX_AGE_SECONDS", 120),
    (ag, "AG_TREND_DEFER_MAX_SEC", "60", "TREND_DEFER_MAX_SECONDS", 60),
    (ag, "AG_REFERENCE_MINUTES_BACK", "60", "ASSET_REFERENCE_MINUTES_BACK_DEFAULT", 60),
    (ag, "AG_BUY_USE_CASH_RATIO", "0.5", "BUY_USE_CASH_RATIO", 0.5),
]


class TestConfigDefaults(unittest.TestCase):
    """Every extracted constant has as its DEFAULT exactly the old hardcoded value."""

    def test_defaults_match_old_hardcoded_values(self):
        for module, attr, expected in DEFAULTS:
            with self.subTest(module=module.__name__, attr=attr):
                actual = getattr(module, attr)
                if isinstance(expected, str):
                    self.assertEqual(actual, expected)
                else:
                    self.assertAlmostEqual(actual, expected, places=6)


class TestConfigOverrides(unittest.TestCase):
    """Setting the environment variable + reload actually changes the constant value."""

    def test_env_overrides_take_effect(self):
        for module, env, val, attr, expected in OVERRIDES:
            with self.subTest(module=module.__name__, env=env):
                os.environ[env] = val
                try:
                    importlib.reload(module)
                    self.assertAlmostEqual(getattr(module, attr), expected, places=6)
                finally:
                    os.environ.pop(env, None)
                    importlib.reload(module)   # restore module to default for remaining tests


class TestModuleSpecifics(unittest.TestCase):
    """Per-module checks (they do not fit the table)."""

    def test_rtrade_import_does_not_start_live_bot(self):
        # rtrade used to run bot.run() (the LIVE loop) unconditionally on import;
        # moved under __main__ -> the import must be safe.
        self.assertFalse(hasattr(rt, "bot"),
                         "importing rtrade must not instantiate or start the bot")

    def test_assetguardian_import_does_not_run_forever(self):
        self.assertTrue(hasattr(ag, "run_forever"))   # It exists, but the __main__ guard did not run it.

    def test_assetguardian_has_no_redundant_drop_parameter(self):
        self.assertFalse(hasattr(ag, "TARGET_DROP_PERCENT"))
        self.assertFalse(hasattr(ag, "TARGET_GROWTH_PERCENT"))
        self.assertEqual(
            ag.BUY_TIERS,
            ((7.0, 0.20), (10.0, 0.30), (14.0, 0.50)),
        )
        self.assertEqual(
            ag.SELL_TIERS,
            ((9.0, 0.12), (12.0, 0.16), (15.0, 0.20), (25.0, 0.24), (35.0, 0.28)),
        )

    def test_assetguardian_requires_every_config_key(self):
        required = (
            "AG_CHECK_INTERVAL_SEC",
            "AG_REFERENCE_MINUTES_BACK",
            "AG_BUY_USE_CASH_RATIO",
            "AG_SELL_REARM_GROWTH_PCT",
            "AG_ORDER_MAX_AGE_SEC",
            "AG_ORDER_MISSING_CONFIRMATIONS",
            "AG_TREND_DEFER_MAX_SEC",
            "AG_RECOVERY_RESET_PCT",
            "AG_NEAR_TRIGGER_SEC",
            "AG_ACTIVE_TRIGGER_SEC",
            "AG_NEAR_TRIGGER_DISTANCE_PCT",
            "AG_PORTFOLIO_GROWTH_PCT",
            "AG_PORTFOLIO_ASSET_MIN_GROWTH_PCT",
            "AG_PORTFOLIO_SELL_ALLOCATION",
            "AG_PORTFOLIO_SELL_MAX_DISCOUNT",
            "AG_RTRADE_BLOCK_CANCEL_SEC",
            "AG_P3B_MAX_AGE_SEC",
            "AG_P3B_SELL_DEFENSIVE_MAX_DISCOUNT",
            "AG_P3B_SELL_TRIM_MAX_DISCOUNT",
            "AG_P3B_BUY_ACCUMULATE_MAX_DISCOUNT",
            "AG_P3B_BUY_DEFENSIVE_MAX_PENALTY",
        )
        for key in required:
            with self.subTest(key=key):
                previous = os.environ.get(key)
                os.environ[key] = ""
                try:
                    with self.assertRaisesRegex(ValueError, key):
                        importlib.reload(ag)
                finally:
                    if previous is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = previous
                    importlib.reload(ag)

    def test_assetguardian_rejects_invalid_numeric_config_without_fallback(self):
        key = "AG_SELL_REARM_GROWTH_PCT"
        previous = os.environ.get(key)
        os.environ[key] = "not-a-number"
        try:
            with self.assertRaisesRegex(ValueError, key):
                importlib.reload(ag)
        finally:
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous
            importlib.reload(ag)

    def test_assetguardian_rejects_invalid_progressive_initial_allocation(self):
        with self.assertRaisesRegex(ValueError, "initial_allocation"):
            ag._progressive_tier_allocations(3, 0.40)

    def test_assetguardian_rearm_must_be_below_first_sell_tier(self):
        key = "AG_SELL_REARM_GROWTH_PCT"
        previous = os.environ.get(key)
        os.environ[key] = "15"
        try:
            with self.assertRaisesRegex(ValueError, "below the first SELL threshold"):
                importlib.reload(ag)
        finally:
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous
            importlib.reload(ag)

    def test_monitortrades_dead_symbol_params_removed(self):
        # SYMBOL_PARAMS (the per-symbol lines from monitortrades.conf) was parsed but
        # never read — the real source is instruments.conf. Removed.
        self.assertFalse(hasattr(mt, "SYMBOL_PARAMS"))

    def test_monitortrades_global_hardtp_fallbacks_loaded(self):
        self.assertFalse(mt.HARD_TP_ENABLED)
        self.assertEqual(mt.MT_REENTRY_POST_STOP_COOLDOWN_SEC, 0.0)
        self.assertEqual(mt.TP_REFERENCE, "last")

    def test_rtrade_buy_sell_asymmetry_preserved(self):
        self.assertNotEqual(rt.RTRADE_BUY_DECAY_PCT, rt.RTRADE_SELL_DECAY_PCT)
        self.assertNotEqual(rt.RTRADE_BUY_DESPERATE_HOURS_BASE, rt.RTRADE_SELL_DESPERATE_HOURS_BASE)
        self.assertNotEqual(rt.RTRADE_BUY_NORMAL_HOURS, rt.RTRADE_SELL_NORMAL_HOURS)


from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import pytest

from botcore import parse_dotenv
from providers.strategy_executor import PairPrecision
from strategies import spot_dca, spot_engine
from strategies.spot_engine import StratParams

ROOT = Path(__file__).resolve().parents[1]
PROFILES = {venue: parse_dotenv(str(ROOT / venue / "config.env")) for venue in ("kraken", "hyperliquid")}
REQUIRED_KEYS = sorted(
    key for key in PROFILES["kraken"] if key.startswith("STRAT_") and key != "STRAT_EXECUTE"
) + ["STRATEGY_MODE", "SHADOW_K_REENTRY"]


class _Executor:
    def pair_precision(self, _symbol):
        return PairPrecision(2, 4, 0.01, "ASSET")


def _params():
    return spot_engine.StratParams(
        currency="USD", entry_amount=100.0, entry_discount_pct=0.2,
        dca_amount=50.0, dca_drop_pct=2.0, check_minutes=2.0,
        takeprofit_pct=5.0, max_budget=1000.0, max_dca_buys=10,
        enable_takeprofit=True, order_ttl_min=10.0, stop_loss_pct=12.5,
        adopt_cost=0.0, adopt_qty=0.0, reentry_drop_pct=2.2,
        reentry_tolerance_pct=0.0, reentry_adaptive=False,
        reentry_sl_bounce_pct=1.5, tp_tranches=[],
    )


class SpotModuleBoundaryTest(unittest.TestCase):
    def test_spot_dca_shim_matches_spot_engine_canonical(self):
        self.assertIs(spot_dca.Strategy, spot_engine.Strategy)
        self.assertIs(spot_dca.StratParams, spot_engine.StratParams)
        self.assertEqual(spot_dca._new_state(), spot_engine._new_state())
        self.assertEqual(spot_dca.state_path_for("HYPEUSD"), spot_engine.state_path_for("HYPEUSD"))

    def test_missing_venue_precision_fails_before_strategy_start(self):
        executor = _Executor()
        executor.pair_precision = lambda _symbol: None
        with self.assertRaisesRegex(RuntimeError, "no pair metadata"):
            spot_engine.Strategy(
                executor, "ASSETUSD", _params(), dry_run=False, initial_state=spot_engine._new_state(),
            )

    def test_default_state_path_stays_in_legacy_kraken_directory(self):
        expected = os.path.join(ROOT, "kraken", ".state_HYPEUSD.json")
        self.assertEqual(spot_engine.state_path_for("HYPE/USD"), expected)

    def test_state_directory_is_injectable_for_another_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = spot_engine.Strategy(
                _Executor(), "ASSET/USD", _params(), dry_run=True,
                initial_state=spot_engine._new_state(), state_dir=directory,
            )
            self.assertEqual(engine.state_file, os.path.join(directory, ".state_ASSETUSD.json"))

    def test_notification_sink_and_source_are_injectable(self):
        events = []
        engine = spot_engine.Strategy(
            _Executor(), "ASSETUSD", _params(), dry_run=True,
            initial_state=spot_engine._new_state(), notifier=lambda **event: events.append(event),
            notification_source="paper-venue",
        )
        order = {"side": "buy", "kind": "ENTRY", "amount": 100.0, "vol": 1.0, "price": 100.0}
        engine._apply_fill(order, 1.0, 100.0, 0.0)
        self.assertEqual(events[0]["source"], "paper-venue")
        self.assertEqual(events[0]["price"], 100.0)


def test_every_live_strategy_setting_must_exist(monkeypatch):
    for venue in PROFILES:
        for key in REQUIRED_KEYS:
            env = PROFILES[venue].copy()
            monkeypatch.setenv(key, env.pop(key))
            with pytest.raises(ValueError, match=key):
                StratParams.from_env(env)


def test_invalid_settings_never_disable_or_default_policy():
    for venue in PROFILES:
        for key in REQUIRED_KEYS:
            env = {**PROFILES[venue], key: "not-a-setting"}
            if key == "STRAT_CURRENCY":
                env[key] = ""
            with pytest.raises(ValueError, match=key):
                StratParams.from_env(env)


def test_versioned_profiles_preserve_effective_values_and_environment():
    for venue in PROFILES:
        env = PROFILES[venue]
        before = dict(os.environ)
        params = StratParams.from_env(env)
        assert dict(os.environ) == before
        with patch.dict(os.environ, env, clear=True):
            assert StratParams.from_env() == params
        assert params.stop_loss_pct == float(env["STRAT_STOP_LOSS_PCT"])
        assert params.reentry_peak_relative is False
        assert params.reentry_pullback_pct == float(env["STRAT_REENTRY_PULLBACK_PCT"])
        assert params.reentry_hybrid_enabled is True
        assert params.surge_gain_pct == float(env["STRAT_SURGE_GAIN_PCT"])
        assert params.trend_sma_n == 30
        assert params.tp_trend_min_pct == 0.5
        assert params.tp_trail_pct == (3 if venue == "kraken" else 2)


def test_versioned_profile_is_isolated_from_the_calling_environment():
    from offline.runners.spot_profile_review import revision_params
    config = (ROOT / "hyperliquid/config.env").read_text()
    with patch.dict(os.environ, STRAT_ENTRY="999999"), patch(
            "offline.runners.spot_profile_review.subprocess.run",
            return_value=SimpleNamespace(stdout=config)):
        params = revision_params("test:hyperliquid/config.env")
        assert params.entry_amount == 350
        assert os.environ["STRAT_ENTRY"] == "999999"


def test_nonfinite_or_empty_new_policy_is_rejected():
    for value in ["nan", "inf", "-inf", ""]:
        with pytest.raises(ValueError, match="STRAT_SURGE_GAIN_PCT"):
            StratParams.from_env({**PROFILES["kraken"], "STRAT_SURGE_GAIN_PCT": value})


def test_malformed_tranches_cannot_silently_become_single_tp():
    for value in ["garbage", "3:40", "3:100,junk", "nan:100", "3:-50,6:150"]:
        with pytest.raises(ValueError, match="STRAT_TP_TRANCHES"):
            StratParams.from_env({**PROFILES["kraken"], "STRAT_TP_TRANCHES": value})


def test_empty_and_valid_tranches_are_supported():
    assert StratParams.from_env(PROFILES["kraken"]).tp_tranches == []
    params = StratParams.from_env({**PROFILES["kraken"], "STRAT_TP_TRANCHES": "3:50,6:50"})
    assert params.tp_tranches == [(3, 50), (6, 50)]


def test_sweep_uses_selected_profile_not_cwd_or_shell(tmp_path, monkeypatch):
    from offline.research.comprehensive_surge_sweep import run_deep_surge_sweep as sweep

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("STRAT_ENTRY", "999999")
    params = sweep.load_base_params(ROOT / "hyperliquid/config.env")
    assert Path(sweep.ROOT) == ROOT
    assert params == StratParams.from_env(PROFILES["hyperliquid"])
    assert os.environ["STRAT_ENTRY"] == "999999"
    candidates = sweep.build_candidate_grid(params)
    assert len({candidate["id"] for candidate in candidates}) == len(candidates)
    assert next(c["params"] for c in candidates if c["id"] == "CURRENT_CONFIG") == params
    control = next(c["params"] for c in candidates if c["id"] == "BASELINE_SURGE_OFF")
    assert control == replace(params, surge_guard=False, surge_dynamic=False)
    assert all(c["params"].effective_max_budget() == 1050 for c in candidates)


def test_sweep_threads_cash_and_explicit_fees_through_all_replays(monkeypatch):
    from offline.research.comprehensive_surge_sweep import run_deep_surge_sweep as sweep

    params = StratParams.from_env(PROFILES["hyperliquid"])
    candidate = sweep.build_candidate_grid(params)[0]
    calls = []

    def replay(**kwargs):
        calls.append(kwargs)
        return {"total": 5, "max_drawdown_pct": 2, "cycles": 1}

    monkeypatch.setattr(sweep, "run_replay", replay)
    result = sweep.evaluate_single_candidate(candidate, {"HYPE": [(100, 101, 99, 100)] * 80}, 1050, 0.4)
    assert len(calls) == 3
    assert all(call["initial_cash"] == 1050 and call["fee_pct"] == 0.4 for call in calls)
    assert result["total_profit_usd"] == 5
    assert "calmar_ratio" not in result


def test_sweep_refuses_missing_datasets(tmp_path, monkeypatch):
    from offline.research.comprehensive_surge_sweep import run_deep_surge_sweep as sweep

    monkeypatch.setattr("sys.argv", ["sweep", "--fee-pct", "0.4", "--data-dir", str(tmp_path),
                                     "--out-dir", str(tmp_path / "output")])
    with pytest.raises(SystemExit) as error:
        sweep.main()
    assert error.value.code == 2
    assert not (tmp_path / "output/sweep_results.json").exists()


@pytest.mark.parametrize("fail_control", [False, True])
def test_sweep_report_is_complete_and_records_effective_policy(tmp_path, monkeypatch, fail_control):
    from concurrent.futures import ThreadPoolExecutor
    from offline.research.comprehensive_surge_sweep import run_deep_surge_sweep as sweep

    profile = ROOT / "hyperliquid/config.env"
    params = sweep.load_base_params(profile)
    candidates = sweep.build_candidate_grid(params)[:2]
    (tmp_path / "TEST_240m.csv").touch()
    monkeypatch.setattr(sweep, "ASSETS", ["TEST"])
    monkeypatch.setattr(sweep, "load_ohlc_csv", lambda _: [(100, 101, 99, 100)] * 80)
    monkeypatch.setattr(sweep, "build_candidate_grid", lambda _: candidates)
    monkeypatch.setattr(sweep.concurrent.futures, "ProcessPoolExecutor", ThreadPoolExecutor)

    def evaluate(candidate, asset_data, initial_cash, fee_pct):
        if fail_control and candidate["category"] == "Baseline":
            raise RuntimeError("Synthetic baseline failure")
        assert list(asset_data) == ["TEST"]
        assert initial_cash == params.effective_max_budget()
        assert fee_pct == 0.4
        return {
            "id": candidate["id"], "category": candidate["category"], "desc": candidate["desc"],
            "total_profit_usd": 5, "worst_maxdd_pct": 2, "pnl_per_drawdown_point": 2.5,
            "h1_profit_usd": 2, "h2_profit_usd": 3, "asset_profits": {"TEST": 5},
            "asset_maxdds": {"TEST": 2},
        }

    monkeypatch.setattr(sweep, "evaluate_single_candidate", evaluate)
    monkeypatch.setattr("sys.argv", [
        "sweep", "--config", str(profile), "--fee-pct", "0.4",
        "--data-dir", str(tmp_path), "--out-dir", str(tmp_path / "output"),
    ])
    report_path = tmp_path / "output/sweep_results.json"
    if fail_control:
        with pytest.raises(RuntimeError, match="Incomplete sweep"):
            sweep.main()
        assert not report_path.exists()
    else:
        sweep.main()
        report = json.loads(report_path.read_text())
        assert report["schema_version"] == 2
        assert report["strategy_params"]["surge_gain_pct"] == params.surge_gain_pct
        assert report["initial_cash_per_asset"] == params.effective_max_budget()
        assert report["fee_pct_per_leg"] == 0.4
        assert report["total_candidates"] == 2
        assert len(report["limitations"]) == 4
        text = (tmp_path / "output/DEEP_SURGE_SWEEP_REPORT.md").read_text()
        assert "no live promotion is justified" in text
        assert "Calmar" not in text


if __name__ == "__main__":
    unittest.main()

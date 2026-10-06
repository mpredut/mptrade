"""
CONSOLIDATED tests for extracting the constants into config files (28 Jul —
unified from 4 nearly identical files: test_{tradeall,monitortrades,rtrade,
assetguardian}_config_extraction.py, ~56 teste sparse -> table-driven aici).

The intent is PRESERVED in full for the legacy modules and strict for AssetGuardian:
  - DEFAULTS: the values loaded from config are the expected versioned ones;
    AssetGuardian no longer has hidden fallbacks in the code.
  - OVERRIDE: setarea variabilei de mediu + reload chiar schimba valoarea
    (the override mechanism really works, not just apparently).
  - Per-module specifics: import safety (rtrade/assetguardian do NOT start
    the live loop on import), dead code removed (monitortrades SYMBOL_PARAMS),
    asimetria BUY/SELL pastrata (rtrade).
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
    (mt, "MT_RECENT_TRADE_BLOCK_SEC", 3 * 60 * 60),
    (mt, "MT_ALL_TRADES_BLOCK_SEC", 1 * 60 * 60),
    (mt, "MT_MAIN_LOOP_SLEEP_SEC", 60 * 0.8),
    (mt, "MT_BUY_PRICE_OFFSET", 0.5),
    (mt, "MT_SELL_SAFEBACK_HOURS", 2),
    (mt, "MT_BUY_SAFEBACK_HOURS", 48),
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
    (ag, "SELL_REARM_GROWTH_PERCENT", 5.0),
    (ag, "ORDER_MAX_AGE_SECONDS", 15 * 60),
    (ag, "ORDER_MISSING_CONFIRMATIONS", 1),
    (ag, "TREND_DEFER_MAX_SECONDS", 3 * 60),
]

# (module, env_var, value_str, attribute, expected_value_after_the_override)
OVERRIDES = [
    (ta, "TRADEALL_SLOPE_EXTREME_THRESHOLD", "9.9", "SLOPE_EXTREME_THRESHOLD", 9.9),
    (ta, "TRADEALL_FIRE_MAX_PER_TREND", "7", "FIRE_MAX_PER_TREND", 7),
    (ta, "TRADEALL_FIRE_MIN_RETRY_MINUTES", "1", "FIRE_MIN_RETRY_INTERVAL_SEC", 60),
    (ta, "TRADEALL_TREND_CONSISTENT_CONFIRMS", "48", "TREND_CONSISTENT_CONFIRMS", 48),
    (ta, "TRADEALL_TREND_OLD_HOURS", "3.0", "TREND_TO_BE_OLD_SECONDS", 3.0 * 3600),
    (mt, "MT_ARE_CLOSE_TOLERANCE_PCT", "2.5", "MT_ARE_CLOSE_TOLERANCE_PCT", 2.5),
    (mt, "MT_RECENT_TRADE_BLOCK_HOURS", "5", "MT_RECENT_TRADE_BLOCK_SEC", 5 * 3600),
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
    """Setarea variabilei de mediu + reload chiar schimba valoarea constantei."""

    def test_env_overrides_take_effect(self):
        for module, env, val, attr, expected in OVERRIDES:
            with self.subTest(module=module.__name__, env=env):
                os.environ[env] = val
                try:
                    importlib.reload(module)
                    self.assertAlmostEqual(getattr(module, attr), expected, places=6)
                finally:
                    os.environ.pop(env, None)
                    importlib.reload(module)   # readuce modulul la default pt restul


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
        self.assertEqual(ag.SELL_TIERS, ((15.0, 0.30), (25.0, 0.30), (35.0, 0.40)))

    def test_assetguardian_requires_every_config_key(self):
        required = (
            "AG_CHECK_INTERVAL_SEC",
            "AG_REFERENCE_MINUTES_BACK",
            "AG_BUY_USE_CASH_RATIO",
            "AG_BUY_TIERS",
            "AG_SELL_TIERS",
            "AG_SELL_REARM_GROWTH_PCT",
            "AG_ORDER_MAX_AGE_SEC",
            "AG_ORDER_MISSING_CONFIRMATIONS",
            "AG_TREND_DEFER_MAX_SEC",
            "AG_RECOVERY_RESET_PCT",
            "AG_NEAR_TRIGGER_SEC",
            "AG_ACTIVE_TRIGGER_SEC",
            "AG_NEAR_TRIGGER_DISTANCE_PCT",
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

    def test_assetguardian_rejects_incomplete_sell_allocations(self):
        key = "AG_SELL_TIERS"
        previous = os.environ.get(key)
        os.environ[key] = "15:0.30,25:0.30"
        try:
            with self.assertRaisesRegex(ValueError, "exactly 1"):
                importlib.reload(ag)
        finally:
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous
            importlib.reload(ag)

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
        self.assertTrue(mt.HARD_TP_ENABLED)
        self.assertEqual(mt.TP_REFERENCE, "last")

    def test_rtrade_buy_sell_asymmetry_preserved(self):
        self.assertNotEqual(rt.RTRADE_BUY_DECAY_PCT, rt.RTRADE_SELL_DECAY_PCT)
        self.assertNotEqual(rt.RTRADE_BUY_DESPERATE_HOURS_BASE, rt.RTRADE_SELL_DESPERATE_HOURS_BASE)
        self.assertNotEqual(rt.RTRADE_BUY_NORMAL_HOURS, rt.RTRADE_SELL_NORMAL_HOURS)


if __name__ == "__main__":
    unittest.main()

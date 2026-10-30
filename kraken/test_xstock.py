#!/usr/bin/env python3
"""
The test suite for the xStock watcher plus the adoption mode in the strategy.
NO real API, NO money: a fake client, captured notifications, state in temp files.

  python3 test_xstock.py -v
"""

from __future__ import annotations

import os
import importlib.util
import signal
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace

KRAKEN_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(KRAKEN_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)
from botcore import load_local_module
from kraken_common import parse_dotenv  # noqa: E402
import kraken_xstock_watch as xw  # noqa: E402

strategy = load_local_module(KRAKEN_DIR, "strategy", module_key="kraken_xstock_strategy_under_test")

Strategy = strategy.Strategy
StratParams = strategy.StratParams
state_path_for = strategy.state_path_for


class FakeKraken:
    def __init__(self, bal=None, pairs=None, price=200.0):
        self.bal = bal or {}
        self.pairs = pairs or {}
        self.price = price

    def balance(self):
        return self.bal

    def asset_pairs(self):
        return self.pairs

    def last_price(self, pair):
        return self.price

    def pair_info(self, pair):
        return {"base": "TSTX", "pair_decimals": 2, "lot_decimals": 6, "ordermin": "0.01"}

    # A minimal StrategyExecutor adapter for the generic engine's tests.
    def pair_precision(self, pair):
        info = self.pair_info(pair)
        return SimpleNamespace(
            price_decimals=info["pair_decimals"],
            volume_decimals=info["lot_decimals"],
            order_min=float(info["ordermin"]),
            base_asset=info["base"],
        )

    def free_balance(self, asset):
        return float(self.bal.get(asset, 0.0) or 0.0)


class Base(unittest.TestCase):
    """Complete isolation: restored env, state in temp, captured notify."""

    def setUp(self):
        self._env = dict(os.environ)
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.remove(path)
        self.state_path = path
        os.environ["XSTOCK_STATE_FILE"] = path
        os.environ.update(parse_dotenv(os.path.join(KRAKEN_DIR, "config.env")))
        self.alerts: list[str] = []
        self._xw_notify = xw.notify
        self._st_notify = strategy.notify
        xw.notify = lambda **kw: self.alerts.append(kw.get("title", ""))
        strategy.notify = lambda **kw: self.alerts.append(kw.get("title", ""))

    def tearDown(self):
        xw.notify = self._xw_notify
        strategy.notify = self._st_notify
        if os.path.exists(self.state_path):
            os.remove(self.state_path)
        os.environ.clear()
        os.environ.update(self._env)


# ---------------------------------------------------------------------------
# Watcher: persistence and no duplicates on restart
# ---------------------------------------------------------------------------
class TestWatcherPersistenta(Base):
    def test_allocation_alerts_once_across_a_restart(self):
        c = FakeKraken(bal={"ZUSD": "100"})
        st = xw._load_state()
        xw.check_balance(c, st, "SPCX", False)              # The first run = a snapshot.
        self.assertEqual(len(self.alerts), 0)
        c.bal = {"ZUSD": "100", "SPCXX": "37.5"}            # soseste alocarea
        xw.check_balance(c, st, "SPCX", False)
        self.assertEqual(len(self.alerts), 1)
        self.assertEqual(st["allocated"]["asset"], "SPCXX")
        xw._save_state(st)
        st = xw._load_state()                               # *** RESTART ***
        xw.check_balance(c, st, "SPCX", False)
        self.assertEqual(len(self.alerts), 1, "a restart must not duplicate the alert")
        self.assertEqual(st["allocated"]["qty"], 37.5, "the allocation must be known after a restart")

    def test_a_new_unmatched_asset_gives_an_informational_alert(self):
        c = FakeKraken(bal={"ZUSD": "100"})
        st = xw._load_state()
        xw.check_balance(c, st, "SPCX", False)
        c.bal = {"ZUSD": "100", "RANDOMCOIN": "5"}
        xw.check_balance(c, st, "SPCX", False)
        self.assertEqual(len(self.alerts), 1)
        self.assertIsNone(st["allocated"], "a non-matching asset is not an allocation")

    def test_the_pair_prefers_the_quote_currency_and_does_not_duplicate(self):
        pairs = {"SPCXXEUR": {"wsname": "SPCXx/EUR", "base": "SPCXX"},
                 "SPCXXUSD": {"wsname": "SPCXx/USD", "base": "SPCXX"}}
        c = FakeKraken(pairs=pairs)
        st = xw._load_state()
        xw.check_pairs(c, st, "SPCX", False, quote="USD")
        self.assertEqual(st["pair"], "SPCXXUSD", "the USD quote must be preferred")
        self.assertEqual(len(self.alerts), 1)
        xw._save_state(st)
        st = xw._load_state()                               # restart
        xw.check_pairs(c, st, "SPCX", False, quote="USD")
        self.assertEqual(len(self.alerts), 1, "the listing alert is not duplicated")

    def test_the_tp_and_sl_level_alerts_fire_once(self):
        c = FakeKraken(price=250.0)
        st = xw._load_state()
        self.assertIn("alerted_tp2", st)
        self.assertFalse(st["alerted_tp2"])
        st["allocated"] = {"asset": "SPCXX", "qty": 37.5}
        st["pair"] = "SPCXXUSD"
        xw.check_levels(c, st, 200.0, 20.0, 15.0, "", False)   # +25% > +20%
        xw.check_levels(c, st, 200.0, 20.0, 15.0, "", False)   # repetat
        self.assertEqual(len(self.alerts), 1, "alerta TP o singura data")
        c.price = 160.0                                        # -20% < -15%
        xw.check_levels(c, st, 200.0, 20.0, 15.0, "", False)
        self.assertEqual(len(self.alerts), 2, "and the SL alert fires only once")

    def test_load_state_sanitizes_corrupt_allocated_and_bot_pid(self):
        xw._save_state({
            "known_assets": {"ZUSD": 100.0},
            "allocated": {"asset": True, "qty": "corrupt"},
            "bot_pid": True,
        })
        st = xw._load_state()
        self.assertIsNone(st["allocated"])
        self.assertIsNone(st["bot_pid"])

        xw._save_state({
            "known_assets": {"ZUSD": 100.0},
            "allocated": {"asset": " SPCXX ", "qty": "37.5", "ts": -5},
            "bot_pid": 4321.0,
        })
        st2 = xw._load_state()
        self.assertEqual(st2["allocated"]["asset"], "SPCXX")
        self.assertEqual(st2["allocated"]["qty"], 37.5)
        self.assertEqual(st2["allocated"]["ts"], 0.0)
        self.assertEqual(st2["bot_pid"], 4321)


# ---------------------------------------------------------------------------
# Auto-start: idempotent, watchdog, conditii
# ---------------------------------------------------------------------------
class TestAutoStart(Base):
    def setUp(self):
        super().setUp()
        self._script, self._log = xw.BOT_SCRIPT, xw.BOT_LOG
        fd, self.stub = tempfile.mkstemp(suffix=".py")
        with os.fdopen(fd, "w") as f:
            f.write("import time\ntime.sleep(120)\n")
        xw.BOT_SCRIPT = self.stub
        xw.BOT_LOG = self.stub + ".log"
        os.environ["XSTOCK_AUTOSTART"] = "true"
        self.st = {"allocated": {"asset": "SPCXX", "qty": 37.5}, "pair": "SPCXXUSD",
                   "bot_pid": None, "alerted_need_price": False}

    def tearDown(self):
        if self.st.get("bot_pid"):
            try:
                os.kill(int(self.st["bot_pid"]), signal.SIGKILL)
                time.sleep(0.2)
                xw._bot_alive(self.st["bot_pid"])
            except (OSError, ValueError, TypeError):
                pass
        for p in (self.stub, xw.BOT_LOG):
            if os.path.exists(p):
                os.remove(p)
        xw.BOT_SCRIPT, xw.BOT_LOG = self._script, self._log
        super().tearDown()

    def test_it_starts_and_is_idempotent(self):
        xw.maybe_start_bot(self.st, 200.0, False)
        pid = self.st["bot_pid"]
        self.assertTrue(xw._bot_alive(pid))
        self.assertEqual(len(self.alerts), 1)
        xw.maybe_start_bot(self.st, 200.0, False)           # a doua chemare
        self.assertEqual(self.st["bot_pid"], pid, "it does not start a second bot")
        self.assertEqual(len(self.alerts), 1)

    def test_the_watchdog_restarts_a_dead_or_zombie_bot(self):
        xw.maybe_start_bot(self.st, 200.0, False)
        pid = self.st["bot_pid"]
        os.kill(pid, signal.SIGKILL)                        # devine zombie (copilul nostru)
        time.sleep(0.3)
        self.assertFalse(xw._bot_alive(pid), "the zombie must be reaped and declared dead")
        xw.maybe_start_bot(self.st, 200.0, False)
        self.assertNotEqual(self.st["bot_pid"], pid)
        self.assertTrue(xw._bot_alive(self.st["bot_pid"]))
        self.assertIn("RESTARTED", self.alerts[-1])

    def test_it_does_not_start_without_a_price_but_asks_for_it_once(self):
        xw.maybe_start_bot(self.st, 0.0, False)
        xw.maybe_start_bot(self.st, 0.0, False)
        self.assertIsNone(self.st["bot_pid"])
        self.assertEqual(len(self.alerts), 1)
        self.assertIn("XSTOCK_ALLOC_PRICE", self.alerts[0])

    def test_autostart_disabled_does_not_start(self):
        os.environ["XSTOCK_AUTOSTART"] = "false"
        xw.maybe_start_bot(self.st, 200.0, False)
        self.assertIsNone(self.st["bot_pid"])
        self.assertEqual(len(self.alerts), 0)

    def test_it_does_not_start_without_an_allocation_or_a_pair(self):
        xw.maybe_start_bot({"allocated": None, "pair": "X", "bot_pid": None,
                            "alerted_need_price": False}, 200.0, False)
        xw.maybe_start_bot({"allocated": {"asset": "A", "qty": 1}, "pair": None,
                            "bot_pid": None, "alerted_need_price": False}, 200.0, False)
        self.assertEqual(len(self.alerts), 0)

    def test_the_paper_flag_is_passed_to_the_bot(self):
        os.environ["XSTOCK_BOT_PAPER"] = "true"
        with open(self.stub, "w") as f:                     # a stub that writes its own argv
            f.write("import sys,time\nopen(sys.argv[0]+'.argv','w').write(' '.join(sys.argv[1:]))\ntime.sleep(120)\n")
        xw.maybe_start_bot(self.st, 200.0, False)
        time.sleep(0.6)
        argv = open(self.stub + ".argv").read()
        os.remove(self.stub + ".argv")
        self.assertIn("--paper", argv)
        self.assertIn("--pair SPCXXUSD", argv)


# ---------------------------------------------------------------------------
# Strategie: adoptarea pozitiei
# ---------------------------------------------------------------------------
class TestAdoptare(Base):
    PAIR = "TSTXUSD"

    def setUp(self):
        super().setUp()
        os.environ.update(parse_dotenv(os.path.join(KRAKEN_DIR, "config.env")))
        os.environ["STRAT_ADOPT_COST"] = "200"
        sp = state_path_for(self.PAIR)
        if os.path.exists(sp):
            os.remove(sp)

    def tearDown(self):
        sp = state_path_for(self.PAIR)
        if os.path.exists(sp):
            os.remove(sp)
        super().tearDown()

    def _strategy(self, client):
        return Strategy(client, self.PAIR, StratParams.from_env(), dry_run=True, desktop=False)

    def test_adopts_from_the_balance_and_places_a_tp(self):
        s = self._strategy(FakeKraken(bal={"TSTX": "37.5"}))
        s.step(200.0)
        self.assertTrue(s.s["adopted"])
        # Adoption from the balance leaves a dust amount (the Kraken balance is reported rounded).
        self.assertAlmostEqual(s.s["qty"], 37.5, delta=0.001)
        self.assertLess(s.s["qty"], 37.5 + 1e-12, "it never sells more than the ledger")
        self.assertAlmostEqual(s.s["cost"] / s.s["qty"], 200.0)
        sells = [o for o in s.s["orders"] if o["side"] == "sell" and o["kind"] == "TP"]
        self.assertEqual(
            len(sells), 0,
            "configured trend-hold waits for the TP threshold before trailing",
        )

    def test_waits_for_the_allocation_without_buying(self):
        c = FakeKraken(bal={"ZUSD": "100"})                 # Allocation is still unavailable.
        s = self._strategy(c)
        s.step(100.0)
        self.assertEqual(s.s["orders"], [], "it does NOT buy a new entry while waiting")
        c.bal = {"ZUSD": "100", "TSTX": "37.5"}             # soseste
        s.step(200.0)
        self.assertTrue(s.s["adopted"])

    def test_it_does_not_adopt_over_a_cycle_in_progress(self):
        import json
        with open(state_path_for(self.PAIR), "w") as f:
            json.dump({"cycle": 3, "qty": 2.46, "cost": 150.0, "spent": 150.0,
                       "dca_buys": 1, "orders": []}, f)
        s = self._strategy(FakeKraken(bal={"TSTX": "99"}))
        s._maybe_adopt()
        self.assertFalse(s.s.get("adopted", False))
        self.assertEqual(s.s["qty"], 2.46, "pozitia existenta ramane neatinsa")

    def test_it_does_not_re_adopt_after_a_restart(self):
        s = self._strategy(FakeKraken(bal={"TSTX": "37.5"}))
        s._maybe_adopt()
        q1 = s.s["qty"]
        s._save()
        s2 = self._strategy(FakeKraken(bal={"TSTX": "99999"}))   # The balance grew in the meantime.
        s2._maybe_adopt()
        self.assertEqual(s2.s["qty"], q1, "a restart does not re-adopt or duplicate")

    def test_the_allocation_does_not_consume_the_dca_cap(self):
        s = self._strategy(FakeKraken(bal={"TSTX": "37.5"}))
        s._maybe_adopt()
        self.assertEqual(s.s["spent"], 0.0, "plafonul ramane pt bani aditionali")


if __name__ == "__main__":
    unittest.main(verbosity=2)

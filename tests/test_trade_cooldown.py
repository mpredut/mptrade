import os, sys, threading, tempfile, unittest
import multiprocessing as mp
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lock import trade_cooldown as tc


def _proc_attempt(state_file, lock_file, q):
    # Runs in a spawned child and uses the same state/lock files.
    tc.STATE_FILE = state_file
    tc.LOCK_FILE = lock_file
    ok, _ = tc.reserve_trade("BUY", "BTCUSDC", cooldown_sec=180)
    q.put(ok)


class TestTradeCooldown(unittest.TestCase):
    def setUp(self):
        # isolated state/lock per test
        self.tmp = tempfile.mkdtemp()
        tc.STATE_FILE = os.path.join(self.tmp, "trade_cooldown.json")
        tc.LOCK_FILE = os.path.join(self.tmp, "trade_cooldown.lock")

    def test_first_allowed_second_blocked(self):
        ok, _ = tc.reserve_trade("BUY", "BTCUSDC", cooldown_sec=180)
        self.assertTrue(ok)
        ok2, last = tc.reserve_trade("SELL", "BTCUSDC", cooldown_sec=180)
        self.assertFalse(ok2)                       # < 3 min -> blocked
        self.assertEqual(last["side"], "BUY")

    def test_per_symbol_independent(self):
        self.assertTrue(tc.reserve_trade("BUY", "BTCUSDC", cooldown_sec=180)[0])
        # another symbol is NOT blocked by BTC's cooldown
        self.assertTrue(tc.reserve_trade("BUY", "TAOUSDC", cooldown_sec=180)[0])

    def test_allowed_after_cooldown(self):
        # A controlled clock, not a real sleep: avoids flakiness on wall-clock/NTP
        # adjustments and tests the threshold semantics exactly (allowed after expiry).
        with mock.patch("lock.cooldown.time.time") as now:
            now.return_value = 1_000.0
            self.assertTrue(tc.reserve_trade("BUY", "BTCUSDC", cooldown_sec=1)[0])
            self.assertFalse(tc.reserve_trade("SELL", "BTCUSDC", cooldown_sec=1)[0])
            now.return_value = 1_001.1
            self.assertTrue(tc.reserve_trade("SELL", "BTCUSDC", cooldown_sec=1)[0])

    def test_release_unblocks(self):
        self.assertTrue(tc.reserve_trade("BUY", "BTCUSDC", cooldown_sec=180)[0])
        tc.release_trade("BTCUSDC")                 # failed order -> released
        self.assertTrue(tc.reserve_trade("BUY", "BTCUSDC", cooldown_sec=180)[0])  # allowed again

    def test_update_binance_order_id(self):
        tc.reserve_trade("BUY", "BTCUSDC", cooldown_sec=180)
        tc.update_binance_order_id("BTCUSDC", 12345)
        self.assertIn("12345", tc.describe_last_trade("BTCUSDC"))

    def test_concurrent_threads_single_winner(self):
        # 20 threads fire at once on the SAME symbol -> exactly ONE gets through
        results = []
        barrier = threading.Barrier(20)

        def attempt():
            barrier.wait()                          # they all start together
            ok, _ = tc.reserve_trade("BUY", "BTCUSDC", cooldown_sec=180)
            results.append(ok)

        threads = [threading.Thread(target=attempt) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(sum(1 for r in results if r), 1)   # a single winner

    def test_concurrent_processes_single_winner(self):
        # Spawn avoids forking the suite's background threads on Python 3.14 while
        # preserving the cross-process flock contract under test.
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        procs = [ctx.Process(target=_proc_attempt, args=(tc.STATE_FILE, tc.LOCK_FILE, q))
                 for _ in range(10)]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=10)
        results = [q.get() for _ in range(10)]
        q.close()
        q.join_thread()
        self.assertEqual(sum(1 for r in results if r), 1)   # a single winner

    # ─── RAII / trade_slot ────────────────────────────────────────────────────
    def test_slot_commit_keeps_reservation(self):
        with tc.trade_slot("BUY", "BTCUSDC", cooldown_sec=180) as slot:
            self.assertTrue(slot.allowed)
            slot.commit(999)                         # the order was placed
        # commit -> the reservation STAYS -> the second one is blocked
        self.assertFalse(tc.reserve_trade("SELL", "BTCUSDC", cooldown_sec=180)[0])

    def test_slot_no_commit_auto_releases(self):
        with tc.trade_slot("BUY", "BTCUSDC", cooldown_sec=180) as slot:
            self.assertTrue(slot.allowed)
            # we do NOT commit (as if the order failed, or we forgot)
        # auto-release on exit -> allowed again
        self.assertTrue(tc.reserve_trade("BUY", "BTCUSDC", cooldown_sec=180)[0])

    def test_slot_exception_auto_releases(self):
        with self.assertRaises(RuntimeError):
            with tc.trade_slot("BUY", "BTCUSDC", cooldown_sec=180) as slot:
                self.assertTrue(slot.allowed)
                raise RuntimeError("placement crashed")   # exception -> automatic rollback
        self.assertTrue(tc.reserve_trade("BUY", "BTCUSDC", cooldown_sec=180)[0])

    def test_slot_blocked_does_not_release_existing(self):
        # the first reserves; the second slot is blocked -> on exit it must NOT delete
        # rezervarea primului
        self.assertTrue(tc.reserve_trade("BUY", "BTCUSDC", cooldown_sec=180)[0])
        with tc.trade_slot("SELL", "BTCUSDC", cooldown_sec=180) as slot:
            self.assertFalse(slot.allowed)
        # the initial reservation is still active
        self.assertFalse(tc.reserve_trade("BUY", "BTCUSDC", cooldown_sec=180)[0])

    def test_get_last_trade_age(self):
        self.assertIsNone(tc.get_last_trade_age("BTCUSDC"))
        tc.reserve_trade("BUY", "BTCUSDC", cooldown_sec=180)
        age = tc.get_last_trade_age("BTCUSDC")
        self.assertIsNotNone(age)
        self.assertLess(age, 5)

    # ─── pereche atomica BUY + SELL ──────────────────────────────────────────
    def test_pair_allows_one_buy_and_one_sell_inside_same_cooldown(self):
        with tc.trade_slot("BUY", "TAOUSDC", cooldown_sec=180,
                           pair_id="pair-1") as buy:
            self.assertTrue(buy.allowed)
            buy.commit(101)
        with tc.trade_slot("SELL", "TAOUSDC", cooldown_sec=180,
                           pair_id="pair-1") as sell:
            self.assertTrue(sell.allowed)
            sell.commit(102)

        state = tc._cooldown().get("TAOUSDC")
        self.assertCountEqual(state["group_committed"], ["BUY", "SELL"])
        self.assertEqual(state["group_results"]["BUY"]["binance_order_id"], 101)
        self.assertEqual(state["group_results"]["SELL"]["binance_order_id"], 102)

    def test_pair_blocks_duplicate_side_and_unrelated_group(self):
        with tc.trade_slot("BUY", "TAOUSDC", cooldown_sec=180,
                           pair_id="pair-1") as buy:
            self.assertTrue(buy.allowed)
            buy.commit(101)

        with tc.trade_slot("BUY", "TAOUSDC", cooldown_sec=180,
                           pair_id="pair-1") as duplicate:
            self.assertFalse(duplicate.allowed)
        with tc.trade_slot("SELL", "TAOUSDC", cooldown_sec=180,
                           pair_id="pair-2") as outsider:
            self.assertFalse(outsider.allowed)
        self.assertFalse(
            tc.reserve_trade("SELL", "TAOUSDC", cooldown_sec=180)[0],
            "an order without a pair_id must not bypass the active pair")

    def test_failed_second_leg_rolls_back_only_that_leg(self):
        with tc.trade_slot("BUY", "TAOUSDC", cooldown_sec=180,
                           pair_id="pair-1") as buy:
            buy.commit(101)
        with tc.trade_slot("SELL", "TAOUSDC", cooldown_sec=180,
                           pair_id="pair-1") as sell:
            self.assertTrue(sell.allowed)
            # without a commit -> only the SELL is rolled back

        state = tc._cooldown().get("TAOUSDC")
        self.assertEqual(state["group_members"], ["BUY"])
        self.assertEqual(state["group_committed"], ["BUY"])
        with tc.trade_slot("SELL", "TAOUSDC", cooldown_sec=180,
                           pair_id="pair-1") as retry:
            self.assertTrue(retry.allowed)

    def test_legacy_reservation_still_blocks_pair(self):
        self.assertTrue(tc.reserve_trade("BUY", "TAOUSDC", cooldown_sec=180)[0])
        with tc.trade_slot("SELL", "TAOUSDC", cooldown_sec=180,
                           pair_id="pair-1") as leg:
            self.assertFalse(leg.allowed)

    def test_canceled_pair_leg_can_be_replaced_without_opening_other_groups(self):
        with tc.trade_slot("SELL", "TAOUSDC", cooldown_sec=180,
                           pair_id="pair-1") as sell:
            sell.commit(102)
        self.assertTrue(tc.release_pair_leg("TAOUSDC", "pair-1", "SELL"))

        with tc.trade_slot("SELL", "TAOUSDC", cooldown_sec=180,
                           pair_id="pair-2") as outsider:
            self.assertFalse(outsider.allowed)
        with tc.trade_slot("SELL", "TAOUSDC", cooldown_sec=180,
                           pair_id="pair-1") as replacement:
            self.assertTrue(replacement.allowed)
            replacement.commit(103)


if __name__ == "__main__":
    unittest.main()

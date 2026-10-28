import json
import os
import tempfile
import unittest

from rtrade_pair_store import RTradePairStore, rtrade_client_order_id


class RTradePairStoreTest(unittest.TestCase):
    def test_the_client_order_id_is_stable_and_rtrade_specific(self):
        first = rtrade_client_order_id("pair-1", "buy")
        self.assertEqual(first, rtrade_client_order_id("pair-1", "BUY"))
        self.assertTrue(first.startswith("RT_"))
        self.assertEqual(len(first), 35)
        self.assertNotEqual(first, rtrade_client_order_id("pair-1", "SELL"))
        self.assertNotEqual(
            first, rtrade_client_order_id("pair-1", "BUY", "hard_stop"))

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = RTradePairStore(os.path.join(self.tmp.name, "pairs.json"))

    def tearDown(self):
        self.tmp.cleanup()

    def _assert_existing_invalid_store_is_preserved(self, payload):
        original = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        with open(self.store.path, "wb") as handle:
            handle.write(original)

        with self.assertRaises(ValueError):
            self.store.active("TAOUSDC")
        self.assertEqual(open(self.store.path, "rb").read(), original)

        with self.assertRaises(ValueError):
            self.store.begin("TAOUSDC", "new-pair", "BUY", 1.0)
        self.assertEqual(open(self.store.path, "rb").read(), original)

    def test_existing_invalid_roots_fail_closed_without_rewrite(self):
        for payload in (
                [], {}, {"version": 1, "pairs": []}, {"pairs": []},
                {"version": 2, "pairs": {}}):
            with self.subTest(payload=payload):
                self._assert_existing_invalid_store_is_preserved(payload)

    def test_existing_malformed_active_pair_fails_closed_without_rewrite(self):
        self._assert_existing_invalid_store_is_preserved({
            "version": 1,
            "pairs": {
                "pair-1": {
                    "symbol": "TAOUSDC",
                    "pair_id": "another-pair",
                    "start_side": "BUY",
                    "qty": 1.0,
                    "phase": "reserved",
                    "terminal": False,
                    "intents": {},
                    "state": None,
                    "created_ts": 100.0,
                    "updated_ts": 100.0,
                },
            },
        })

    def test_existing_malformed_intent_identity_fails_closed_without_rewrite(self):
        self._assert_existing_invalid_store_is_preserved({
            "version": 1,
            "pairs": {
                "pair-1": {
                    "symbol": "TAOUSDC",
                    "pair_id": "pair-1",
                    "start_side": "BUY",
                    "qty": 1.0,
                    "phase": "reserved",
                    "terminal": False,
                    "intents": {
                        "limit:BUY": {
                            "side": "SELL",
                            "kind": "limit",
                            "qty": 1.0,
                            "price": 100.0,
                            "client_order_id": "RT_invalid",
                            "order_id": None,
                        },
                    },
                    "state": None,
                    "created_ts": 100.0,
                    "updated_ts": 100.0,
                },
            },
        })

    def test_valid_legacy_v1_record_loads_and_gains_only_requested_checkpoint(self):
        legacy = {
            "version": 1,
            "pairs": {
                "pair-v1": {
                    "symbol": "TAOUSDC",
                    "pair_id": "pair-v1",
                    "start_side": "BUY",
                    "qty": 1.0,
                    "phase": "reserved",
                    "terminal": False,
                    "intents": {
                        "limit:BUY": {
                            "side": "BUY",
                            "kind": "limit",
                            "qty": 1.0,
                            "price": 99.0,
                            "client_order_id": "RT_legacy",
                            "order_id": None,
                        },
                    },
                    "state": None,
                    "created_ts": 100.0,
                    "updated_ts": 100.0,
                },
            },
        }
        with open(self.store.path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(legacy, handle, separators=(",", ":"))

        loaded = self.store.active("TAOUSDC")
        self.assertEqual(loaded[0]["intents"]["limit:BUY"]["qty"], 1.0)

        state = {
            "pair_id": "pair-v1",
            "qty": 1.0,
            "start_side": "BUY",
            "phase": "quoting",
            "tickets": [],
        }
        self.store.checkpoint("pair-v1", state, terminal=False)
        updated = self.store.active("TAOUSDC")[0]
        self.assertEqual(updated["state"], state)
        self.assertNotIn("limit_revisions", updated["state"])

    def test_intent_is_durable_before_acceptance_and_checkpoint(self):
        self.store.begin("TAOUSDC", "pair-1", "BUY", 2.0)
        self.store.intent(
            "pair-1", "BUY", 100.0, 2.0, "SD_client", kind="limit")
        pending = self.store.active("TAOUSDC")[0]
        self.assertIsNone(pending["intents"]["limit:BUY"]["order_id"])

        self.store.accepted("pair-1", "BUY", "77", kind="limit")
        state = {
            "pair_id": "pair-1", "qty": 2.0, "start_side": "BUY",
            "phase": "quoting", "tickets": [],
        }
        self.store.checkpoint("pair-1", state, terminal=False)
        adopted = self.store.active("TAOUSDC")[0]
        self.assertEqual(adopted["intents"]["limit:BUY"]["order_id"], "77")
        self.assertEqual(adopted["state"]["phase"], "quoting")

    def test_terminal_pair_is_not_recovered(self):
        self.store.begin("TAOUSDC", "pair-2", "SELL", 1.0)
        self.store.checkpoint(
            "pair-2", {"pair_id": "pair-2", "phase": "complete", "tickets": []},
            terminal=True)
        self.assertEqual(self.store.active("TAOUSDC"), [])

    def test_intent_can_atomically_create_pair_without_separate_begin(self):
        self.store.intent(
            "pair-3", "BUY", 100.0, 1.0, "SD_atomic", kind="limit",
            symbol="TAOUSDC", start_side="BUY")
        rec = self.store.active("TAOUSDC")[0]
        self.assertEqual(rec["pair_id"], "pair-3")
        self.assertEqual(rec["intents"]["limit:BUY"]["client_order_id"],
                         "SD_atomic")

    def test_canonical_tracked_intent_replaces_and_removes_exact_leg(self):
        pending = {
            "intent_id": "rtrade:pair-4:limit:buy",
            "client_order_id": "RT_canonical",
            "symbol": "TAOUSDC", "side": "BUY", "kind": "limit",
            "requested_qty": 1.25, "requested_price": 99.5,
            "attempt": 1, "created_at": 100.0, "lookup_misses": 0,
        }
        self.store.persist_intent(
            "pair-4", "BUY", "limit", pending,
            symbol="TAOUSDC", start_side="BUY")
        rec = self.store.active("TAOUSDC")[0]
        stored = rec["intents"]["limit:BUY"]
        self.assertEqual(stored["requested_qty"], 1.25)
        self.assertEqual(stored["requested_price"], 99.5)
        self.assertEqual(stored["qty"], 1.25)
        self.assertEqual(stored["price"], 99.5)

        accepted = dict(pending, order_id="77", submitted_qty=1.2)
        self.store.persist_intent(
            "pair-4", "BUY", "limit", accepted, symbol="TAOUSDC")
        rec = self.store.active("TAOUSDC")[0]
        self.assertEqual(rec["intents"]["limit:BUY"]["order_id"], "77")

        self.store.persist_intent(
            "pair-4", "BUY", "limit", None, symbol="TAOUSDC")
        rec = self.store.active("TAOUSDC")[0]
        self.assertNotIn("limit:BUY", rec["intents"])

    def test_checkpoint_many_updates_rounds_in_one_transaction(self):
        for pair_id in ("p1", "p2"):
            self.store.begin("TAOUSDC", pair_id, "BUY", 1.0)
        self.store.checkpoint_many([
            ("p1", {
                "pair_id": "p1", "qty": 1.0, "start_side": "BUY",
                "phase": "quoting", "tickets": [],
            }, False),
            ("p2", {"phase": "complete"}, True),
        ])
        active = self.store.active("TAOUSDC")
        self.assertEqual([rec["pair_id"] for rec in active], ["p1"])

    def test_terminal_history_is_bounded(self):
        store = RTradePairStore(
            os.path.join(self.tmp.name, "bounded.json"), terminal_retention=2)
        for i in range(4):
            pair_id = f"t{i}"
            store.begin("TAOUSDC", pair_id, "BUY", 1.0)
            store.checkpoint(pair_id, {"phase": "complete"}, terminal=True)
        import json
        with open(store.path, encoding="utf-8") as handle:
            pairs = json.load(handle)["pairs"]
        self.assertEqual(len(pairs), 2)

    def test_default_in_memory_store_creates_no_disk_file(self):
        store = RTradePairStore()
        self.assertIsNone(store.path)
        self.assertIsNone(store.lock_path)
        store.begin("TAOUSDC", "mem-1", "BUY", 1.0)
        self.assertEqual(len(store.active("TAOUSDC")), 1)
        store.checkpoint("mem-1", {"phase": "complete"}, terminal=True)
        self.assertEqual(store.active("TAOUSDC"), [])


if __name__ == "__main__":
    unittest.main()

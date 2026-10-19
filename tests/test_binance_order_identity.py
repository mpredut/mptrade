import unittest
from unittest.mock import patch, MagicMock

from binance_api import order_id_context as rc
from binance_api import bapi as api
import monitororder


class TestBinanceOrderIdContext(unittest.TestCase):
    def test_resolve_order_owner_all_bots_and_manual(self):
        cases = [
            ("RT_1234567890abcdef", "rtrade"),
            ("TA_MainThread_20261008", "tradeall"),
            ("MO_MainThread_12345678", "monitororder"),
            ("MT_MainThread_99999999", "monitortrades"),
            ("SD_abcdef0123456789", "spot_dca"),
            ("AG_MainThread_88888888", "assetguardian"),
            ("SRV_worker_77777777", "server"),
            ("CW_watchdog_66666666", "watchdogfor_cacheandconfig"),
            ("MA_alerts_55555555", "market_alerts"),
            ("OR_retry_12345678", "order_retry"),
            ("TA_OR_85bea316b24b4a10ab3f_0", "tradeall"),
            ("MO_TA_OR_85bea316b24b4a10ab_0", "tradeall"),
            ("MO_MN_OR_85bea316b24b4a10ab_0", "manual"),
            ("MN_OR_85bea316b24b4a10ab3f_0", "manual"),
            ("MO_OR_85bea316b24b4a10ab3f_0", "monitororder"),
            ("RT_OR_85bea316b24b4a10ab3f_0", "rtrade"),
            ("MT_OR_85bea316b24b4a10ab3f_0", "monitortrades"),
            ("SD_OR_85bea316b24b4a10ab3f_0", "spot_dca"),
            ("and_123456789", "manual"),
            ("ios_987654321", "manual"),
            ("web_abcdef123", "manual"),
            ("x-R4D3XYZ", "manual"),
            ("", "unspecified"),
            (None, "unspecified"),
            ("XYZ_unknown_order", "alt:XYZ_"),
        ]
        for cid, expected in cases:
            with self.subTest(cid=cid, expected=expected):
                self.assertEqual(rc.resolve_order_owner(cid), expected)

    def test_is_order_repriceable_by_monitororder(self):
        # Repriceable origins: tradeall, monitororder, manual (app/web), unspecified (for test mocks)
        self.assertTrue(rc.is_order_repriceable_by_monitororder("TA_MainThread_123"))
        self.assertTrue(rc.is_order_repriceable_by_monitororder("TA_OR_85bea316b24b4a10ab3f_0"))
        self.assertTrue(rc.is_order_repriceable_by_monitororder("MO_TA_OR_85bea316b24b4a10ab_0"))
        self.assertTrue(rc.is_order_repriceable_by_monitororder("MO_MN_OR_85bea316b24b4a10ab_0"))
        self.assertTrue(rc.is_order_repriceable_by_monitororder("MN_OR_85bea316b24b4a10ab3f_0"))
        self.assertTrue(rc.is_order_repriceable_by_monitororder("MO_MainThread_456"))
        self.assertTrue(rc.is_order_repriceable_by_monitororder("MO_OR_12345"))
        self.assertTrue(rc.is_order_repriceable_by_monitororder("and_mobile_app_123"))
        self.assertTrue(rc.is_order_repriceable_by_monitororder("web_browser_123"))
        self.assertTrue(rc.is_order_repriceable_by_monitororder(""))
        self.assertTrue(rc.is_order_repriceable_by_monitororder(None))

        # Strictly PROTECTED bots & origins: must NEVER be repriced
        self.assertFalse(rc.is_order_repriceable_by_monitororder("OR_outbox_retry_123"))
        self.assertFalse(rc.is_order_repriceable_by_monitororder("RT_grid_pair_123"))
        self.assertFalse(rc.is_order_repriceable_by_monitororder("RT_OR_12345"))
        self.assertFalse(rc.is_order_repriceable_by_monitororder("SD_trailing_stop_123"))
        self.assertFalse(rc.is_order_repriceable_by_monitororder("SD_OR_12345"))
        self.assertFalse(rc.is_order_repriceable_by_monitororder("MT_stop_loss_123"))
        self.assertFalse(rc.is_order_repriceable_by_monitororder("MT_OR_12345"))
        self.assertFalse(rc.is_order_repriceable_by_monitororder("AG_risk_guard_123"))
        self.assertFalse(rc.is_order_repriceable_by_monitororder("SRV_server_123"))
        self.assertFalse(rc.is_order_repriceable_by_monitororder("UNKNOWN_12345"))

    def test_order_matches_owner(self):
        self.assertTrue(rc.order_matches_owner("TA_123", "tradeall"))
        self.assertTrue(rc.order_matches_owner("TA_OR_123", "tradeall"))
        self.assertTrue(rc.order_matches_owner("MO_TA_OR_123", "tradeall"))
        self.assertTrue(rc.order_matches_owner("MO_TA_OR_123", "monitororder"))
        self.assertTrue(rc.order_matches_owner("TA_123", {"tradeall", "monitororder"}))
        self.assertTrue(rc.order_matches_owner("web_123", "manual"))
        self.assertTrue(rc.order_matches_owner("MO_MN_OR_123", "manual"))
        self.assertFalse(rc.order_matches_owner("RT_123", "tradeall"))
        self.assertFalse(rc.order_matches_owner("RT_OR_123", "tradeall"))
        self.assertTrue(rc.order_matches_owner("RT_123", None))


class TestBapiOrderIdentityFiltering(unittest.TestCase):
    def test_get_open_orders_populates_client_order_id_and_filters_owner(self):
        raw_venue_orders = [
            {"orderId": 101, "side": "BUY", "origQty": "1.0", "executedQty": "0.0", "price": "100.0", "time": 1700000000000, "clientOrderId": "TA_buy_1"},
            {"orderId": 102, "side": "BUY", "origQty": "2.0", "executedQty": "0.0", "price": "95.0", "time": 1700000000000, "clientOrderId": "RT_grid_1"},
            {"orderId": 103, "side": "BUY", "origQty": "0.5", "executedQty": "0.0", "price": "90.0", "time": 1700000000000, "clientOrderId": "web_manual_1"},
        ]

        mock_client = MagicMock()
        mock_client.get_open_orders.return_value = raw_venue_orders

        with patch.object(api, "client", mock_client):
            # 1. No owner filter -> all BUY returned, clientOrderId populated
            all_orders = api.get_open_orders("BUY", "BTCUSDC")
            self.assertEqual(len(all_orders), 3)
            self.assertEqual(all_orders[101]["clientOrderId"], "TA_buy_1")
            self.assertEqual(all_orders[102]["clientOrderId"], "RT_grid_1")
            self.assertEqual(all_orders[103]["clientOrderId"], "web_manual_1")

            # 2. Filter allowed_owners={"tradeall"} -> only TA order returned
            ta_orders = api.get_open_orders("BUY", "BTCUSDC", allowed_owners={"tradeall"})
            self.assertEqual(list(ta_orders.keys()), [101])

            # 3. Filter allowed_owners={"rtrade"} -> only RT order returned
            rt_orders = api.get_open_orders("BUY", "BTCUSDC", allowed_owners={"rtrade"})
            self.assertEqual(list(rt_orders.keys()), [102])

    def test_cancel_expired_orders_respects_allowed_owners(self):
        now = 1700001000
        open_orders = {
            201: {"price": 100.0, "timestamp": now - 3600, "clientOrderId": "TA_order_1"},
            202: {"price": 95.0, "timestamp": now - 3600, "clientOrderId": "RT_order_2"},
        }
        with patch.object(api, "get_open_orders", side_effect=lambda side, sym, allowed_owners=None: {
            oid: data for oid, data in open_orders.items()
            if allowed_owners is None or rc.order_matches_owner(data["clientOrderId"], allowed_owners)
        }), patch.object(api, "cancel_order", return_value=True) as mock_cancel, \
             patch("time.time", return_value=now):
            # Calling cancel_expired_orders scoped to tradeall
            api.cancel_expired_orders("BUY", "BTCUSDC", expire_time=600, allowed_owners={"tradeall"})
            mock_cancel.assert_called_once_with("BTCUSDC", 201)


class TestMonitororderProtectsOtherBots(unittest.TestCase):
    def setUp(self):
        monitororder.reset_monitororder_state_for_test()
        self._cooldown_patcher = patch.object(
            monitororder.trade_cooldown, "get_last_trade_age", return_value=999.0,
        )
        self._cooldown_patcher.start()

    def tearDown(self):
        self._cooldown_patcher.stop()
        monitororder.reset_monitororder_state_for_test()

    def test_monitororder_skips_repricing_for_rtrade_and_circuit_breakers(self):
        # Open orders include:
        # - an rtrade grid order close to market (100.1 vs 100.0) -> MUST NOT BE REPRICED
        # - a spot_dca trailing stop order close to market -> MUST NOT BE REPRICED
        # - a monitortrades stop-loss order close to market -> MUST NOT BE REPRICED
        # - a tradeall order close to market -> REPRICED
        open_orders = {
            501: {"price": 100.0, "quantity": 1.0, "remainingQty": 1.0, "clientOrderId": "RT_grid_pair_1"},
            502: {"price": 100.0, "quantity": 1.0, "remainingQty": 1.0, "clientOrderId": "SD_trailing_stop"},
            503: {"price": 100.0, "quantity": 1.0, "remainingQty": 1.0, "clientOrderId": "MT_stop_loss"},
            504: {"price": 100.0, "quantity": 1.0, "remainingQty": 1.0, "clientOrderId": "TA_tradeall_order"},
        }

        with patch.object(monitororder.api, "get_open_orders", return_value=open_orders), \
             patch.object(monitororder.api, "get_current_price", return_value=100.1), \
             patch.object(monitororder.order_retry, "enqueue", return_value="rep-1") as mock_enqueue, \
             patch.object(monitororder.order_retry, "claim", return_value=[{"place_kwargs": {"client_order_id": "MO_cid"}}]), \
             patch.object(monitororder.order_retry, "activate_claimed_replacement", return_value="activated"), \
             patch.object(monitororder.order_retry, "begin_claimed_submit", return_value={"place_kwargs": {"client_order_id": "MO_cid"}}), \
             patch.object(monitororder.accepted_order_persistence, "complete_accepted_claim", return_value=True), \
             patch.object(monitororder.order_retry, "complete_claim", return_value=True), \
             patch.object(monitororder.mkt, "order_filter_refusal", return_value=None), \
             patch.object(monitororder.mkt, "preflight_order", return_value=object()), \
             patch.object(monitororder.api, "cancel_order", return_value=True) as mock_cancel, \
             patch.object(monitororder.mkt, "order_status", return_value=MagicMock(venue_status="CANCELED", terminal=False)), \
             patch.object(monitororder.mkt, "place", return_value={"orderId": 9999}) as mock_place, \
             patch("alertnotifiers.notify"):
            monitororder.monitor_open_orders_by_type("TAOUSDC", "BUY")

            # Only order 504 (TA_) was canceled and replaced!
            # Orders 501 (RT_), 502 (SD_), and 503 (MT_) were completely protected!
            mock_cancel.assert_called_once_with("TAOUSDC", 504)
            mock_place.assert_called_once()
            self.assertEqual(mock_enqueue.call_args[0][3].get("prefix"), "MO_TA_OR")

    def test_monitororder_reprices_manual_orders(self):
        # A manual limit order placed by the user via web app
        open_orders = {
            601: {"price": 100.0, "quantity": 1.0, "remainingQty": 1.0, "clientOrderId": "web_manual_order"},
        }

        with patch.object(monitororder.api, "get_open_orders", return_value=open_orders), \
             patch.object(monitororder.api, "get_current_price", return_value=100.1), \
             patch.object(monitororder.order_retry, "enqueue", return_value="rep-2") as mock_enqueue, \
             patch.object(monitororder.order_retry, "claim", return_value=[{"place_kwargs": {"client_order_id": "MO_cid"}}]), \
             patch.object(monitororder.order_retry, "activate_claimed_replacement", return_value="activated"), \
             patch.object(monitororder.order_retry, "begin_claimed_submit", return_value={"place_kwargs": {"client_order_id": "MO_cid"}}), \
             patch.object(monitororder.accepted_order_persistence, "complete_accepted_claim", return_value=True), \
             patch.object(monitororder.order_retry, "complete_claim", return_value=True), \
             patch.object(monitororder.mkt, "order_filter_refusal", return_value=None), \
             patch.object(monitororder.mkt, "preflight_order", return_value=object()), \
             patch.object(monitororder.api, "cancel_order", return_value=True) as mock_cancel, \
             patch.object(monitororder.mkt, "order_status", return_value=MagicMock(venue_status="CANCELED", terminal=False)), \
             patch.object(monitororder.mkt, "place", return_value={"orderId": 8888}) as mock_place, \
             patch("alertnotifiers.notify"):
            monitororder.monitor_open_orders_by_type("TAOUSDC", "BUY")

            mock_cancel.assert_called_once_with("TAOUSDC", 601)
            mock_place.assert_called_once()
            self.assertEqual(mock_enqueue.call_args[0][3].get("prefix"), "MO_MN_OR")

    def test_instrument_apply_outbox_lineage_prefix_preserves_bot_lineage(self):
        from instrument import Instrument
        target = {}
        with patch("binance_api.order_id_context.get_process_name", return_value="TA"):
            Instrument._apply_outbox_lineage_prefix(target, {})
        self.assertEqual(target.get("prefix"), "TA_OR")

        runner_target = {}
        with patch("binance_api.order_id_context.get_process_name", return_value="PYTEST"):
            Instrument._apply_outbox_lineage_prefix(runner_target, {})
        self.assertNotIn("prefix", runner_target)

    def test_order_retry_infers_owner_and_cleans_up_accepted_trackers(self):
        import os
        import tempfile
        import order_retry as oq

        with tempfile.TemporaryDirectory() as tmpdir:
            qfile = os.path.join(tmpdir, "order_retry_queue.jsonl")
            lfile = os.path.join(tmpdir, "order_retry.lock")
            with patch.object(oq, "QUEUE_FILE", qfile), patch.object(oq, "LOCK_FILE", lfile):
                rid = oq.enqueue(
                    "BTCUSDC", "BUY", 0.00066, {},
                    requested_price=81733.6, ref_price=81815.42,
                    kind="kalman_primary_up", provider_name="binance", now=1000.0,
                )
                rec = oq.get(rid)
                self.assertEqual(rec["prefix"], "TA_OR")
                self.assertTrue(rec["place_kwargs"]["client_order_id"].startswith("TA_OR_"))
                self.assertEqual(oq.infer_record_owner(rec), "tradeall")

                # Simulate legacy OR_ client_order_id with kalman_primary_up kind
                oq.mark_accepted(rid, {"orderId": 10574289378, "status": "NEW"}, now=1010.0)
                owner_map = oq.get_accepted_order_owner_map()
                self.assertEqual(owner_map.get("10574289378"), "tradeall")
                self.assertEqual(
                    oq.get_accepted_order_ids_for_owner("BTCUSDC", "tradeall"),
                    ["10574289378"],
                )

                # Replacement activation retires the replaced accepted tracker
                rep_id = oq.enqueue(
                    "BTCUSDC", "BUY", 0.00066, {"parent_owner": "tradeall"},
                    requested_price=81800.0, ref_price=81800.0,
                    provider_name="binance", now=1020.0,
                    lifecycle="awaiting_cancel",
                    replaces_order_id="10574289378",
                    replaces_original_qty=0.00066,
                )
                claimed_rep = oq.claim([rep_id], now=1021.0)[0]
                self.assertEqual(
                    oq.activate_claimed_replacement(claimed_rep, 0.00066, now=1022.0),
                    "activated",
                )
                self.assertEqual(
                    oq.get_accepted_order_ids_for_owner("BTCUSDC", "tradeall"),
                    [],
                )

                # Intentional external cancellation resolves accepted tracker
                rid2 = oq.enqueue(
                    "BTCUSDC", "BUY", 0.00064, {},
                    requested_price=82125.73, ref_price=82207.94,
                    kind="kalman_primary_up", provider_name="binance", now=1030.0,
                )
                oq.mark_accepted(rid2, {"orderId": 10574512957, "status": "NEW"}, now=1031.0)
                self.assertTrue(oq.resolve_accepted_order("BTCUSDC", 10574512957))
                self.assertEqual(oq.load_all(), [oq.get(rep_id)])

    def test_same_deferred_stream_consolidates_across_deferral_reasons(self):
        import os
        import tempfile
        import order_retry as oq

        with tempfile.TemporaryDirectory() as tmpdir:
            qfile = os.path.join(tmpdir, "order_retry_queue.jsonl")
            lfile = os.path.join(tmpdir, "order_retry.lock")
            with patch.object(oq, "QUEUE_FILE", qfile), \
                 patch.object(oq, "LOCK_FILE", lfile), \
                 patch.object(oq, "RETRY_DEDUP", False):
                first = oq.enqueue(
                    "BTCUSDC", "BUY", 0.00066, {},
                    requested_price=81733.6, ref_price=81815.42,
                    failure_reason="trend_deferred",
                    kind="kalman_primary_up", provider_name="binance", now=1000.0,
                )
                oq.mark_failure(
                    first, "profit_guard", now=1010.0, submission_state="refused"
                )
                second = oq.enqueue(
                    "BTCUSDC", "BUY", 0.00064, {},
                    requested_price=82125.73, ref_price=82207.94,
                    failure_reason="trend_deferred",
                    kind="kalman_primary_up", provider_name="binance", now=1020.0,
                )
                rows = oq.load_all()
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["id"], second)

    def test_monitororder_reprices_legacy_or_tradeall_order(self):
        open_orders = {
            701: {
                "price": 100.0,
                "quantity": 1.0,
                "remainingQty": 1.0,
                "clientOrderId": "OR_a5e8e1cb99d8450882ede051_0",
            },
        }
        with patch.object(monitororder.api, "get_open_orders", return_value=open_orders), \
             patch.object(monitororder.api, "get_current_price", return_value=100.1), \
             patch.object(monitororder.order_retry, "get_accepted_order_owner_map", return_value={"701": "tradeall"}), \
             patch.object(monitororder.order_retry, "enqueue", return_value="rep-701") as mock_enqueue, \
             patch.object(monitororder.order_retry, "claim", return_value=[{"place_kwargs": {"client_order_id": "MO_TA_cid"}}]), \
             patch.object(monitororder.order_retry, "activate_claimed_replacement", return_value="activated"), \
             patch.object(monitororder.order_retry, "begin_claimed_submit", return_value={"place_kwargs": {"client_order_id": "MO_TA_cid"}}), \
             patch.object(monitororder.accepted_order_persistence, "complete_accepted_claim", return_value=True), \
             patch.object(monitororder.order_retry, "complete_claim", return_value=True), \
             patch.object(monitororder.mkt, "order_filter_refusal", return_value=None), \
             patch.object(monitororder.mkt, "preflight_order", return_value=object()), \
             patch.object(monitororder.api, "cancel_order", return_value=True) as mock_cancel, \
             patch.object(monitororder.mkt, "order_status", return_value=MagicMock(venue_status="CANCELED", terminal=False)), \
             patch.object(monitororder.mkt, "place", return_value={"orderId": 7777}) as mock_place, \
             patch("alertnotifiers.notify"):
            monitororder.monitor_open_orders_by_type("BTCUSDC", "BUY")
            mock_cancel.assert_called_once_with("BTCUSDC", 701)
            mock_place.assert_called_once()
            self.assertEqual(mock_enqueue.call_args[0][3].get("prefix"), "MO_TA_OR")

    def test_tradeall_cancels_outbox_accepted_orders_and_resolves_tracker(self):
        import tradeall as ta
        import order_retry as oq

        ta._symbol_order_ids.clear()
        with patch.object(ta.api, "cancel_expired_orders"), \
             patch.object(ta.mkt, "cancel_order", return_value=True) as mock_cancel, \
             patch.object(oq, "get_accepted_order_ids_for_owner", return_value=["10574289378"]), \
             patch.object(oq, "resolve_accepted_order", return_value=True) as mock_resolve, \
             patch.object(ta, "_fire_order", return_value={"orderId": 10574512957}):
            order_ids = ta.track_and_place_order("BUY", "BTCUSDC", 1, 82125.73, 82207.94)
            mock_cancel.assert_called_once_with("BTCUSDC", "10574289378")
            mock_resolve.assert_called_once_with("BTCUSDC", "10574289378")
            self.assertEqual(order_ids, [10574512957])

    def test_monitortrades_hard_tp_preserves_8_decimals_and_passes_tp_kind(self):
        import monitortrades as mt

        snapshot = MagicMock()
        snapshot.symbol = "BTCUSDC"
        snapshot.sellable_qty = 0.00066
        snapshot.current_time_s = 2000.0
        snapshot.gain_pct = 18.5
        snapshot.current_price = 95000.0
        inst = MagicMock()

        with patch.dict(mt._hard_tp_last, {}, clear=True), \
             patch.object(mt, "_place_guarded", return_value=True) as mock_place, \
             patch.object(mt, "_resolve_exit") as mock_resolve:
            sold = mt._evaluate_hard_take_profit(
                inst, snapshot, hard_tp_pct=0.17, hard_tp_frac=0.5,
                hard_tp_cd=3600.0, min_qty=0.00001, sbs=3600.0,
            )
            self.assertTrue(sold)
            self.assertEqual(mock_place.call_args[0][3], 0.00033)
            self.assertEqual(mock_place.call_args[1].get("kind"), "TP")
            mock_resolve.assert_called_once_with("BTCUSDC", inst, kind="TP")

    def test_cancel_open_recent_and_opposite_orders_resolve_accepted_retry_tracker(self):
        import time
        from binance_api import bapi
        from binance_api import bapi_placeorder as bp
        import order_retry

        now_s = int(time.time())
        open_map = {"9001": {"price": 80000.0, "timestamp": now_s - 5}}
        with patch.object(bapi.sym, "validate_params"), \
             patch.object(bapi, "get_open_orders", return_value=open_map), \
             patch.object(bapi, "cancel_order", return_value=True), \
             patch.object(order_retry, "resolve_accepted_order") as mock_resolve:
            bapi.cancel_open_orders("BUY", "BTCUSDC")
            mock_resolve.assert_called_once_with("BTCUSDC", "9001")

            mock_resolve.reset_mock()
            bapi.cancel_recent_orders("BUY", "BTCUSDC", max_age_seconds=60)
            mock_resolve.assert_called_once_with("BTCUSDC", "9001")

            mock_resolve.reset_mock()
            bp.cancel_opposite_orders("BUY", "BTCUSDC", requested_price=85000.0)
            mock_resolve.assert_called_once_with("BTCUSDC", "9001")




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
        # Repriceable bots: tradeall, monitororder, unspecified (for test mocks)
        self.assertTrue(rc.is_order_repriceable_by_monitororder("TA_MainThread_123"))
        self.assertTrue(rc.is_order_repriceable_by_monitororder("MO_MainThread_456"))
        self.assertTrue(rc.is_order_repriceable_by_monitororder(""))
        self.assertTrue(rc.is_order_repriceable_by_monitororder(None))

        # Strictly PROTECTED bots & origins: must NEVER be repriced
        self.assertFalse(rc.is_order_repriceable_by_monitororder("RT_grid_pair_123"))
        self.assertFalse(rc.is_order_repriceable_by_monitororder("SD_trailing_stop_123"))
        self.assertFalse(rc.is_order_repriceable_by_monitororder("MT_stop_loss_123"))
        self.assertFalse(rc.is_order_repriceable_by_monitororder("AG_risk_guard_123"))
        self.assertFalse(rc.is_order_repriceable_by_monitororder("SRV_server_123"))
        self.assertFalse(rc.is_order_repriceable_by_monitororder("and_mobile_app_123"))
        self.assertFalse(rc.is_order_repriceable_by_monitororder("web_browser_123"))
        self.assertFalse(rc.is_order_repriceable_by_monitororder("UNKNOWN_12345"))

    def test_order_matches_owner(self):
        self.assertTrue(rc.order_matches_owner("TA_123", "tradeall"))
        self.assertTrue(rc.order_matches_owner("TA_123", {"tradeall", "monitororder"}))
        self.assertFalse(rc.order_matches_owner("RT_123", "tradeall"))
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

    def tearDown(self):
        monitororder.reset_monitororder_state_for_test()

    def test_monitororder_skips_repricing_for_rtrade_and_manual_orders(self):
        # Open orders include:
        # - an rtrade grid order close to market (100.1 vs 100.0) -> MUST NOT BE REPRICED
        # - a manual web order close to market -> MUST NOT BE REPRICED
        # - a tradeall order close to market -> REPRICED
        open_orders = {
            501: {"price": 100.0, "quantity": 1.0, "remainingQty": 1.0, "clientOrderId": "RT_grid_pair_1"},
            502: {"price": 100.0, "quantity": 1.0, "remainingQty": 1.0, "clientOrderId": "web_manual_order"},
            503: {"price": 100.0, "quantity": 1.0, "remainingQty": 1.0, "clientOrderId": "TA_tradeall_order"},
        }

        with patch.object(monitororder.api, "get_open_orders", return_value=open_orders), \
             patch.object(monitororder.api, "get_current_price", return_value=100.1), \
             patch.object(monitororder.order_retry, "enqueue", return_value="rep-1"), \
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

            # Only order 503 (TA_) was canceled and replaced!
            # Orders 501 (RT_) and 502 (web_) were completely protected!
            mock_cancel.assert_called_once_with("TAOUSDC", 503)
            mock_place.assert_called_once()

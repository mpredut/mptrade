"""Characterisation: runtime imports do not start network or caches in the background."""

import os
import subprocess
import sys
import textwrap
import unittest


class RuntimeImportSafetyTest(unittest.TestCase):
    def test_provider_layer_imports_without_binance_key_module(self):
        code = textwrap.dedent(
            """
            import sys
            import types
            import importlib

            # Simulate a minimal installation where the optional keys shim is absent.
            keys = types.ModuleType("keys")
            keys.__path__ = []
            sys.modules["keys"] = keys
            sys.modules.pop("keys.apikeys", None)

            from providers.t212_provider import T212Provider
            from providers import market_api
            importlib.import_module("212trading.t212_client")

            if market_api._bapi is not None:
                raise AssertionError("Binance bapi a fost importat eager")
            T212Provider()  # a lazy construction, without T212/Binance keys
            if market_api.api.provider_by_name("t212") is None:
                raise AssertionError("T212 is missing from the registry after a direct import")

            from binance_api import bapi_ws
            if bapi_ws.api_key_ws:
                raise AssertionError("the test loaded a private key")
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            env=dict(os.environ, BINANCE_AUTO_START_WEBSOCKETS="0", BINANCE_API_KEY_WS=""),
            capture_output=True,
            text=True,
            timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_cache_manager_import_is_network_and_worker_free(self):
        code = textwrap.dedent(
            """
            import threading
            import binance.client

            class ForbiddenClient:
                def __init__(self, *args, **kwargs):
                    raise AssertionError("Binance Client construit la import")

            binance.client.Client = ForbiddenClient
            import cacheManager
            from binance_api import bapi_trades

            forbidden = {
                "BinanceTimeResync", "CacheTrade", "CacheTradeManager",
                "CacheOrder", "CacheOrderManager",
                "CacheCurrentPrice", "CacheCurrentPriceManager", "NonBinanceTrendPoller",
                "InstantTrendFullEval", "InstantTrendFlush",
            }
            active = {thread.name for thread in threading.enumerate()}
            leaked = sorted(active & forbidden)
            if leaked:
                raise AssertionError(f"thread-uri pornite la import: {leaked}")
            """
        )
        env = dict(os.environ, BINANCE_AUTO_START_WEBSOCKETS="0",
                   BINANCE_API_KEY="", BINANCE_API_SECRET="", BINANCE_API_KEY_WS="")
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            env=env,
            capture_output=True,
            text=True,
            timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_missing_rest_credentials_fail_before_client_construction(self):
        code = textwrap.dedent('''
            from unittest.mock import patch
            from binance_api import bapi_client
            from keys import apikeys
            assert apikeys.api_key_ws == "", "REST key must not substitute for the Ed25519 key"
            with patch.object(bapi_client, "Client") as client:
                try:
                    bapi_client.getClient()
                except ValueError as exc:
                    assert "BINANCE_API_SECRET" in str(exc)
                else:
                    raise AssertionError("Missing credentials were accepted")
                client.assert_not_called()
        ''')
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            env=dict(os.environ, BINANCE_API_KEY="test-only", BINANCE_API_SECRET="",
                     BINANCE_API_KEY_WS="", BINANCE_AUTO_START_WEBSOCKETS="0"),
            text=True, capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_bapi_client_recvwindow_and_minus_1021_resync_retry(self):
        from unittest.mock import MagicMock, patch
        from binance_api import bapi_client

        class _RecvWindowError(Exception):
            code = -1021

        calls = []

        class _DummyClient:
            def __init__(self):
                self.session = MagicMock()
                self.REQUEST_RECVWINDOW = None
                self.timestamp_offset = 0

            def get_server_time(self):
                return {"serverTime": 1_700_000_005_000}

            def _request(self, method, uri, signed, force_params=False, **kwargs):
                data = kwargs.get("data")
                calls.append(dict(data) if isinstance(data, dict) else data)
                if len(calls) == 1:
                    if isinstance(data, dict):
                        data["timestamp"] = 111
                        data["signature"] = "stale_sig"
                    raise _RecvWindowError("Timestamp outside recvWindow")
                return {"ok": True, "data": data}

        dummy = _DummyClient()
        bapi_client._install_recvwindow_retry(dummy)
        with patch.object(bapi_client, "_client", dummy):
            res = dummy._request("get", "https://api.binance.com/api/v3/account", True, data={"symbol": "BTCUSDC"})
        self.assertEqual(res["ok"], True)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1], {"symbol": "BTCUSDC"})
        self.assertEqual(bapi_client.REQUEST_RECVWINDOW_MS, 30000)

        class _Html500Error(Exception):
            code = 0
            status_code = 500
            def __init__(self):
                self.message = "Invalid JSON error message from Binance: <html>\r\n<head><title>500 Internal Server Error</title></head>\r\n</html>"
            def __str__(self):
                return f"APIError(code={self.code}): {self.message}"

        class _Failing500Client:
            def __init__(self):
                self.calls = 0
            def _request(self, method, uri, signed, force_params=False, **kwargs):
                self.calls += 1
                raise _Html500Error()

        fail500 = _Failing500Client()
        bapi_client._install_recvwindow_retry(fail500)
        with self.assertRaises(_Html500Error) as ctx:
            fail500._request("get", "https://api.binance.com/api/v3/account", True, data={"symbol": "BTCUSDC"})
        self.assertEqual(fail500.calls, 2)
        self.assertNotIn("\n", str(ctx.exception))
        self.assertNotIn("\r", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

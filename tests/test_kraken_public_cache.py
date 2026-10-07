"""Kraken public reads never cache an empty result.

An empty AssetPairs or Ticker response represents a transient failed fetch. Caching
it for the one-hour AssetPairs TTL would hide order minimums and permit rejected
dust-order churn until the cache expired.
"""
import base64
import os
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "kraken"))
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

import kraken_client as kc


EMPTY = (200, b'{"error":[],"result":{}}')
FULL = (200, b'{"error":[],"result":{"HYPEUSD":{"ordermin":"0.1"}}}')


class PublicCacheTest(unittest.TestCase):
    def setUp(self):
        kc._CACHE.clear()

    def test_empty_not_cached_refetches_then_caches_full(self):
        with mock.patch.object(kc, "http_get") as mget:
            mget.side_effect = [EMPTY, FULL]
            client = kc.KrakenClient()
            r1 = client._public("AssetPairs", {"pair": "HYPEUSD"})
            self.assertEqual(r1, {})                       # Empty response.
            r2 = client._public("AssetPairs", {"pair": "HYPEUSD"})
            self.assertIn("HYPEUSD", r2)                   # Refetched after empty.
            self.assertEqual(mget.call_count, 2)
            # The complete result is cached, so no new fetch is needed.
            r3 = client._public("AssetPairs", {"pair": "HYPEUSD"})
            self.assertIn("HYPEUSD", r3)
            self.assertEqual(mget.call_count, 2)

    def test_pair_info_none_on_empty_then_recovers(self):
        with mock.patch.object(kc, "http_get") as mget:
            mget.side_effect = [EMPTY, FULL]
            client = kc.KrakenClient()
            self.assertIsNone(client.pair_info("HYPEUSD"))       # Empty response becomes None.
            info = client.pair_info("HYPEUSD")  # Refetch returns a complete result.
            self.assertEqual(info.get("ordermin"), "0.1")

    def test_ohlc_closes_excludes_potentially_incomplete_last_candle(self):
        response = {
            "HYPEUSD": [
                [1, "9", "11", "8", "10", "10", "1", 1],
                [14401, "10", "12", "9", "11", "11", "1", 1],
                [28801, "11", "99", "1", "42", "42", "1", 1],
            ],
            "last": 28801,
        }
        client = kc.KrakenClient()
        with mock.patch.object(client, "_public", return_value=response):
            closes, last_closed_at = client.ohlc_closes_with_timestamp(
                "HYPEUSD", 240)
            all_closes, timestamps = client.ohlc_closes_with_timestamps(
                "HYPEUSD", 240)

        self.assertEqual(closes, [10.0, 11.0])
        self.assertEqual(last_closed_at, 14401 + 240 * 60)
        self.assertEqual(all_closes, closes)
        self.assertEqual(timestamps, (14401.0, 28801.0))

    def test_ohlc_cache_ttl_is_interval_aware(self):
        self.assertEqual(kc._read_ttl("OHLC", {"interval": 1}), 30.0)
        self.assertEqual(kc._read_ttl("OHLC", {"interval": 5}), 150.0)
        self.assertEqual(kc._read_ttl("OHLC", {"interval": 240}), 900.0)

    def test_cache_prunes_expired_and_enforces_hard_cap(self):
        with mock.patch.object(kc, "_CACHE_MAX", 2), \
             mock.patch.object(kc.time, "time", return_value=100.0):
            kc._CACHE[("Ticker", (("pair", "OLD"),))] = (99.0, {"old": 1})
            kc._cache_put("Ticker", {"pair": "A"}, 3.0, {"a": 1})
            kc._cache_put("Ticker", {"pair": "B"}, 4.0, {"b": 1})
            kc._cache_put("Ticker", {"pair": "C"}, 5.0, {"c": 1})
        self.assertEqual(len(kc._CACHE), 2)
        self.assertNotIn(("Ticker", (("pair", "OLD"),)), kc._CACHE)
    def test_private_read_retries_on_invalid_nonce_and_succeeds(self):
        client = kc.KrakenClient(api_key="key", api_secret=base64.b64encode(b"secret").decode())
        fail_nonce = (200, b'{"error":["EAPI:Invalid nonce"],"result":{}}')
        success_bal = (200, b'{"error":[],"result":{"HYPE":"7.54"}}')
        with mock.patch.object(kc, "http_post_form") as mpost, \
             mock.patch.object(kc.time, "sleep") as msleep:
            mpost.side_effect = [fail_nonce, success_bal]
            res = client._private("Balance")
            self.assertEqual(res, {"HYPE": "7.54"})
            self.assertEqual(mpost.call_count, 2)
            self.assertEqual(msleep.call_count, 1)

    def test_private_write_does_not_retry_on_invalid_nonce(self):
        client = kc.KrakenClient(api_key="key", api_secret=base64.b64encode(b"secret").decode())
        fail_nonce = (200, b'{"error":["EAPI:Invalid nonce"],"result":{}}')
        with mock.patch.object(kc, "http_post_form") as mpost:
            mpost.return_value = fail_nonce
            with self.assertRaises(kc.KrakenError) as ctx:
                client._private("AddOrder", {"pair": "HYPEUSD"})
            self.assertIn("Invalid nonce", str(ctx.exception))
            self.assertEqual(mpost.call_count, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

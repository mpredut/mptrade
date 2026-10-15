"""Unit tests for cachepaths SSOT module."""
import os
import unittest
from unittest.mock import patch

from cachepaths import cache_path, get_cachedb_dir
import utils


class TestCachepaths(unittest.TestCase):
    def test_mptrade_cachedb_dir_override(self):
        with patch.dict(os.environ, {"MPTRADE_CACHEDB_DIR": "/custom/mptrade/cache"}, clear=False):
            self.assertEqual(get_cachedb_dir(), "/custom/mptrade/cache")
            path = cache_path("test.json")
            self.assertEqual(path.replace("\\", "/"), "/custom/mptrade/cache/test.json")

    def test_binance_cache_dir_fallback(self):
        env = dict(os.environ)
        env.pop("MPTRADE_CACHEDB_DIR", None)
        env["BINANCE_CACHE_DIR"] = "/custom/binance/cache"
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(get_cachedb_dir(), "/custom/binance/cache")
            path = cache_path("test.json")
            self.assertEqual(path.replace("\\", "/"), "/custom/binance/cache/test.json")

    def test_default_dir_ends_with_cachedb(self):
        env = dict(os.environ)
        env.pop("MPTRADE_CACHEDB_DIR", None)
        env.pop("BINANCE_CACHE_DIR", None)
        with patch.dict(os.environ, env, clear=True):
            d = get_cachedb_dir()
            self.assertTrue(d.replace("\\", "/").endswith("cachedb"))
            self.assertTrue(cache_path("test.json").replace("\\", "/").endswith("cachedb/test.json"))

    def test_preserves_absolute_paths(self):
        abs_path = os.path.abspath("already/absolute/file.json")
        self.assertEqual(cache_path(abs_path), abs_path)

    def test_empty_or_none(self):
        self.assertEqual(cache_path(""), "")
        self.assertIsNone(cache_path(None))

    def test_utils_cache_path_integration(self):
        with patch.dict(os.environ, {"MPTRADE_CACHEDB_DIR": "/tmp/custom_cache"}, clear=False):
            p = utils.cache_path("my_cache.json")
            self.assertEqual(p.replace("\\", "/"), "/tmp/custom_cache/my_cache.json")


if __name__ == "__main__":
    unittest.main()
